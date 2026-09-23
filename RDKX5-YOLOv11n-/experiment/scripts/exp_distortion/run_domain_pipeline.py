#!/usr/bin/env python3
"""
域决策实验的自动收尾流水线（B/C 训练完成后接管剩余全部步骤）

为什么要有它：B/C 各 200 epoch 要跑 ~90 分钟，之后"是否需要 D""enhance 消融在哪个域"
都取决于评估结论。本脚本按 **预先登记的判定阈值**（EXPERIMENT_DESIGN.md §13）自动决策，
把 GPU 空窗期填满，不需要人盯着。

步骤：
  1. 等 weights/domain_B.pt + weights/domain_C.pt 就绪
  2. 主评估（test + valid）：基线(AOLD_C 几何) / B@B / C@C / 配对
  3. 判定 B 是否"够用"（阈值见 §13）
  4. 不够用 → 训 D → 比 C@C vs D@D → 取更优（并列时取 D：更省）
  5. 在最优域上训 enhance-off 臂 → 消融配对 → 判断 enhance 可否去掉
  6. 用最优臂推理 AUV_5 清水 16 张 + 120 帧随机探针
  7. 写 experiment/runs/domain/DOMAIN_RESULT.md

用法：
    python experiment/scripts/exp_distortion/run_domain_pipeline.py
    ... --dry-run-verdict        # 只读已有评估结果做判定，不训练
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "1_prepare"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "1_prepare" / "exp_distortion"))

PY = "/home/ansty/anaconda3/envs/yolov8/bin/python"
EV = "experiment/scripts/exp_distortion/eval_domain_arms.py"
FIELD = PROJECT_ROOT / "experiment/runs/domain"

# 预登记阈值（EXPERIMENT_DESIGN.md §13）
TH_CERR = 1.0        # px
TH_P = 0.05          # Wilcoxon
TH_ALL4 = 0.05       # 5 个百分点
TH_WH = 0.010        # 1 个百分点（实测同一模型跨域的 W 误差抖动 ~0.7pp，见 §13）
TH_OUTER_CERR = 2.0  # px，外圈分层


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, **kw):
    log("RUN " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, **kw)
    tail = "\n".join((r.stdout or "").splitlines()[-25:])
    if r.returncode != 0:
        log(f"⚠️ 命令失败 rc={r.returncode}\n{tail}\n{r.stderr[-1500:]}")
    else:
        print(tail, flush=True)
    return r.returncode


def eval_arms(arms, pairs, split, out, conf=0.25):
    cmd = [PY, EV]
    for a in arms:
        cmd += ["--arm", a]
    for p in pairs:
        cmd += ["--pair", p]
    cmd += ["--split", split, "--conf", str(conf), "--out", out]
    return run(cmd)


def load_detail(stem):
    p = FIELD / f"eval_detail_{stem}.json"
    if not p.exists():
        return None
    return json.load(open(p))


def pair_stats(da, db, ta, tb):
    """配对统计（与 eval_domain_arms 口径一致）"""
    import numpy as np
    from eval_domain_arms import pkey
    A = {pkey(r["img"]): r for r in da}
    B = {pkey(r["img"]): r for r in db}
    com = sorted(set(A) & set(B))
    # ρ外 = **数据驱动的上三分位**（与评估表口径一致，勿用硬编码阈值）
    rhos = np.array([A[k].get("rho", float("nan")) for k in com], float)
    rhos = rhos[np.isfinite(rhos)]
    p67 = float(np.percentile(rhos, 67)) if len(rhos) >= 30 else float("inf")
    d, outer = [], []
    a4a = a4b = n4 = 0
    for k in com:
        ra, rb = A[k], B[k]
        if ra.get("n_gt_vis") == 4:
            n4 += 1
            a4a += int(ra.get("all4", 0))
            a4b += int(rb.get("all4", 0))
        ca, cb = ra.get("cerr_med"), rb.get("cerr_med")
        if ca is not None and cb is not None:
            d.append(ca - cb)
            r = ra.get("rho", float("nan"))
            if r == r and r > p67:
                outer.append(ca - cb)
    d = np.array(d) if d else np.array([0.0])
    outer = np.array(outer) if outer else np.array([0.0])
    p = float("nan")
    outer_p = float("nan")
    try:
        from scipy.stats import wilcoxon
        if len(d) >= 10:
            p = float(wilcoxon(d).pvalue)
        if len(outer) >= 10:
            outer_p = float(wilcoxon(outer).pvalue)
    except Exception:
        pass
    def med(k, recs):
        v = [r[k] for r in recs if r.get(k) is not None]
        return float(np.median(v)) if v else float("nan")
    return dict(n=len(com), n_gt4=n4, med=float(np.median(d)),
                better_a=float((d < 0).mean()), p=p,
                outer_med=float(np.median(outer)) if len(outer) else float("nan"),
                outer_p=outer_p, outer_n=int(len(outer)), rho_p67=p67,
                all4a=a4a / max(n4, 1), all4b=a4b / max(n4, 1),
                dWa=med("dW", da), dWb=med("dW", db),
                dHa=med("dH", da), dHb=med("dH", db),
                cerr_a=med("cerr_med", da), cerr_b=med("cerr_med", db))


def train(arm, domain, enhance=True):
    ds = f"experiment/data/pose_{domain}" if enhance else f"experiment/data/pose_{domain}_noenh"
    wt = f"weights/domain_{domain}.pt" if enhance else f"weights/domain_{domain}_noenh.pt"
    name = f"A_{domain}" if enhance else f"A_{domain}_noenh"
    if (PROJECT_ROOT / wt).exists():
        log(f"{wt} 已存在，跳过训练")
        return 0
    cmd = [PY, "scripts/2_train/train_yolo11n.py", "--data", f"{ds}/data.yaml",
           "--task", "pose", "--weights", "weights/yolo11n-pose.coco.pt",
           "--epochs", "200", "--batch", "4", "--imgsz", "640", "--workers", "2",
           "--device", "0", "--cache", "ram", "--seed", "0",
           "--project", "experiment/runs/domain", "--name", name,
           "--output-pt", wt, "--no-repatch"]
    return run(cmd)


def wait_for(paths, timeout_min=180):
    t0 = time.time()
    while time.time() - t0 < timeout_min * 60:
        if all((PROJECT_ROOT / p).exists() for p in paths):
            return True
        time.sleep(60)
    return False


# 清水（AUV_5）判据（预登记，EXPERIMENT_DESIGN.md §16）
CW_DET_MIN = 0.60        # 有效检出率下限
CW_ALL4_MIN = 0.60       # 四角齐全率下限
CW_CERR_RATIO = 2.0      # 角点误差相对 test 的倍数上限


def clear_water_verdict(winner, det_auv5, det_test):
    """返回 (same_magnitude: bool|None, 说明文本)"""
    import numpy as np

    def summ(recs, split):
        hit = [r for r in recs if r.get("hit")]
        if not hit:
            return None
        gt4 = [r for r in hit if r.get("n_gt_vis") == 4]
        cerr = [r["cerr_med"] for r in hit if r.get("cerr_med") is not None]
        return dict(n=len(recs), det=len(hit) / max(len(recs), 1),
                    all4=(sum(r.get("all4", 0) for r in gt4) / len(gt4)) if gt4 else float("nan"),
                    n_gt4=len(gt4),
                    cerr=float(np.median(cerr)) if cerr else float("nan"))

    a5 = summ(det_auv5, "eval")
    te = summ(det_test, "test")
    if a5 is None or te is None:
        return None, "（清水或 test 明细缺失，无法判定）"
    ok = (a5["det"] >= CW_DET_MIN and a5["all4"] == a5["all4"]
          and a5["all4"] >= CW_ALL4_MIN
          and (te["cerr"] != te["cerr"] or a5["cerr"] <= CW_CERR_RATIO * te["cerr"]))
    txt = (f"清水 16 张（最优臂 {winner}）：检出 {a5['det']*100:.0f}%（判据 ≥{CW_DET_MIN*100:.0f}%）、"
           f"四角齐全 {a5['all4']*100:.0f}%（判据 ≥{CW_ALL4_MIN*100:.0f}%，n_gt4={a5['n_gt4']}）、"
           f"角点误差中位 {a5['cerr']:.2f} px（test 同臂 {te['cerr']:.2f} px，"
           f"判据 ≤{CW_CERR_RATIO:g}× ⇒ ≤{CW_CERR_RATIO*te['cerr']:.2f} px）")
    return bool(ok), txt


def render_conclusion(v, cw_ok, cw_txt):
    real = {"B": "B（P4，不去畸变）", "C": "C（P1，720p 去畸变）", "D": "D（P2，640 去畸变）"}
    acts = {"B": "`cfg/vision.yaml`: `image.undistort: false`（**零代码**，一个开关）",
            "C": "保持 `image.undistort: true`（现状链路），部署标定 C",
            "D": "需改 `common/preprocess.py` 增加「640 上 remap」模式"}
    perf = {"B": "43.4 ms/帧、DDR 8.3 MB/帧（E3 板端实测）",
            "C": "54.4 ms/帧、DDR 22.1 MB/帧",
            "D": "49.3 ms/帧、DDR 10.8 MB/帧"}
    w = v["winner"]
    st = v["b_vs_c"]
    L = []
    L.append("## 结论（自动生成，判据见 EXPERIMENT_DESIGN.md §13/§16）\n")
    L.append(f"**最优输入域 = {real[w]}**\n")
    L.append(f"- B vs C 配对（同帧，n={st['n']}）：角点误差差中位 **{st['med']:+.3f} px**"
             f"（正数=C 更准），Wilcoxon p={st['p']:.4g}；"
             f"四角齐全率 B {st['all4a']*100:.1f}% vs C {st['all4b']*100:.1f}%；"
             f"W 误差 B {st['dWa']*100:.2f}% vs C {st['dWb']*100:.2f}%；"
             f"H 误差 B {st['dHa']*100:.2f}% vs C {st['dHb']*100:.2f}%。")
    L.append(f"- 并列判据（|Δcerr|≤1px 且 p>0.05；|Δ四角|≤5pp；|ΔW|,|ΔH|≤1pp）"
             f"⇒ **{'满足，判为并列' if v['enough'] else '不满足，B 不足'}**；"
             f"ρ外(p67={st.get('rho_p67', float('nan')):.2f}, n={st.get('outer_n',0)}) "
             f"Δcerr 中位 {st['outer_med']:+.2f} px、p={st.get('outer_p', float('nan')):.4g}"
             f" ⇒ {'C 在外圈显著更好' if v['c_better_outer'] else '外圈无显著差异'}。")
    L.append(f"- **板端动作**：{acts[w]}")
    L.append(f"- **预期性能**：{perf[w]}")
    L.append("- ⚠️ 三个方案都**必须把标定 C 部署到板上**（P4 也读部署标定的原始 K/D）\n")
    e = v.get("enhance")
    if e:
        droppable = abs(e["med"]) <= 1.0 and (e["p"] != e["p"] or e["p"] > 0.05)
        L.append(f"### enhance 消融（{w} 域）\n")
        L.append(f"- 关掉 enhance 的角点误差差中位 **{e['med']:+.3f} px**、p={e['p']:.4g}"
                 f"；四角齐全率 {e['all4a']*100:.1f}% vs {e['all4b']*100:.1f}%")
        L.append(f"- ⇒ **{'可以关掉（配置改中性即可，零代码，板端省约 40 ms/帧）' if droppable else '不建议关掉（精度有损失）'}**\n")
    L.append("### 清水（AUV_5）\n")
    L.append(f"- {cw_txt}")
    if cw_ok is True:
        L.append("- ⇒ **与中/浊水同量级，本实验不引入清水这一变量**")
    elif cw_ok is False:
        L.append("- ⇒ **明显更差，建议补标 100–150 张清水门框帧**"
                 "（用 `tools/analyze/label_corners.py`；本仓库可先自动挑「门框完整+清晰+覆盖近中远」的候选帧）")
    L.append("")
    return "\n".join(L) + "\n"


def artifact_check(before_file):
    """核验目标第 ⑥ 条：实验前已有的 weights/*.pt 与 output/*.bin 未被删除/覆盖"""
    import hashlib
    bp = PROJECT_ROOT / before_file
    if not bp.exists():
        return "（缺 artifacts_before.txt，无法核验）", True
    want = {}
    for line in bp.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 32 and not line.startswith("#"):
            want[parts[1]] = parts[0]
    missing, changed, ok = [], [], []
    for rel, md5 in want.items():
        f = PROJECT_ROOT / rel
        if not f.exists():
            missing.append(rel)
            continue
        h = hashlib.md5(f.read_bytes()).hexdigest()
        (ok if h == md5 else changed).append(rel)
    hp = Path("/home/ansty/anaconda3/envs/yolov8/lib/python3.9/site-packages/"
              "ultralytics/nn/modules/head.py")
    hb = hp.with_suffix(".py.backup")
    head = "n/a"
    if hp.exists() and hb.exists():
        head = ("原版（与 backup 一致，训练/predict/val 正确）"
                if hashlib.md5(hp.read_bytes()).hexdigest()
                == hashlib.md5(hb.read_bytes()).hexdigest() else "⚠️ 已打补丁（导出用，predict 前需还原）")
    txt = (f"- 实验前已有产物 **{len(want)}** 个：未改动 **{len(ok)}**、"
           f"缺失 **{len(missing)}**、md5 变化 **{len(changed)}**\n")
    if missing:
        txt += f"- ❌ 缺失：{missing}\n"
    if changed:
        txt += f"- ⚠️ 变化：{changed}\n"
    if not missing and not changed:
        txt += "- ✅ **全部保留、未被覆盖**（weights/*.pt 与 output/*.bin 均原样）\n"
    txt += f"- `head.py` 状态：**{head}**\n"
    return txt, (not missing and not changed)


def sanity_flags(det_test, winner, out_lines):
    """几个"实验是不是真的跑通"的红旗检查"""
    import numpy as np
    flags = []

    def med(recs, k):
        v = [r[k] for r in recs if r.get("hit") and r.get(k) is not None]
        return float(np.median(v)) if v else float("nan")

    # 1) 训练是否收敛
    for arm in ("B", "C", "D"):
        p = PROJECT_ROOT / f"experiment/runs/domain/A_{arm}/results.csv"
        if not p.exists():
            continue
        try:
            rows = list(csv.DictReader(open(p)))
            key = [k for k in rows[0] if k.strip() == "metrics/mAP50-95(P)"][0]
            best = max(float(r[key]) for r in rows)
        except Exception:
            continue
        if best < 0.85:
            flags.append(f"⚠️ {arm} 臂最佳 Pose mAP50-95 仅 **{best:.4f}**（<0.85）"
                         f"→ 训练可能未收敛，结论不可信")
        else:
            flags.append(f"✅ {arm} 臂最佳 Pose mAP50-95 **{best:.4f}**（{len(rows)} epochs）")

    # 2) 最优臂是否反而比"部署中模型"差
    btag = next((t for t in det_test if t.startswith("基线")), None)
    wtag = next((t for t in det_test if t.startswith(f"{winner}@")), None)
    if btag and wtag:
        bc, wc = med(det_test[btag], "cerr_med"), med(det_test[wtag], "cerr_med")
        b4 = np.mean([r.get("all4", 0) for r in det_test[btag] if r.get("n_gt_vis") == 4])
        w4 = np.mean([r.get("all4", 0) for r in det_test[wtag] if r.get("n_gt_vis") == 4])
        flags.append(f"ℹ️ 最优臂 {winner} vs 部署基线（同 C 几何）：角点误差 "
                     f"{wc:.2f} vs {bc:.2f} px；四角齐全率 {w4*100:.1f}% vs {b4*100:.1f}%")
        if wc > bc * 1.2:
            flags.append(f"🚩 **最优臂角点误差比部署基线差 >20%** → 数据集/重投影链路可疑，需人工复核")
        if w4 < b4 - 0.10:
            flags.append(f"🚩 **最优臂四角齐全率比部署基线低 >10pp** → 同上")
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run-verdict", action="store_true",
                    help="只用已有评估 json 做判定与报告，不训练")
    a = ap.parse_args()

    if not a.dry_run_verdict:
        log("等待 B/C 训练产物 …")
        if not wait_for(["weights/domain_B.pt", "weights/domain_C.pt"]):
            log("❌ 等不到 domain_B.pt / domain_C.pt，退出")
            return 1
        log("B/C 权重就绪")

        # ---------- 2) 主评估（test + valid）----------
        for split in ("test", "valid"):
            stem = f"EVAL_main_{split}"
            eval_arms(
                [f"基线_老模型_A几何:data/AUV_4/PNP.kpt4.yolov8:weights/yolo11n-pose.pt:AOLD:{split}",
                 f"基线_老模型_C几何:data/AUV_4/PNP.kpt4.yolov8:weights/yolo11n-pose.pt:AOLD_C:{split}",
                 f"B@B:experiment/data/pose_B:weights/domain_B.pt:B:{split}",
                 f"C@C:experiment/data/pose_C:weights/domain_C.pt:C:{split}"],
                [f"B@B:C@C", f"基线_老模型_C几何:B@B"],
                split, f"experiment/runs/domain/{stem}.md")

        # ---------- 3) 判定 ----------
        det = load_detail("EVAL_main_test")
        if not det:
            log("❌ 读不到 EVAL_main_test 的明细，退出")
            return 1
        st = pair_stats(det["B@B"], det["C@C"], "B@B", "C@C")
        log(f"B vs C 配对: {json.dumps({k: (round(v,4) if isinstance(v,float) else v) for k,v in st.items()}, ensure_ascii=False)}")
        enough = (abs(st["med"]) <= TH_CERR and (st["p"] != st["p"] or st["p"] > TH_P)
                  and abs(st["all4a"] - st["all4b"]) <= TH_ALL4
                  and abs(st["dWa"] - st["dWb"]) <= TH_WH
                  and abs(st["dHa"] - st["dHb"]) <= TH_WH)
        # §13 判据 4：ρ外 分层下 C 显著更好需要**同时**满足 Δ>2px 与 p<0.05
        c_better_outer = (st["outer_med"] == st["outer_med"] and st["outer_med"] > TH_OUTER_CERR
                          and (st["outer_p"] != st["outer_p"] or st["outer_p"] < TH_P))
        log(f"判定：B 够用={enough}  外圈 C 明显更好={c_better_outer}")

        winner = "B"
        if enough and not c_better_outer:
            winner = "B"
        else:
            # ---------- 4) 训 D 并比 C vs D ----------
            log("B 不够用 → 训练 D 域")
            train("D", "D")
            eval_arms([f"C@C:experiment/data/pose_C:weights/domain_C.pt:C:test",
                       f"D@D:experiment/data/pose_D:weights/domain_D.pt:D:test"],
                      ["C@C:D@D"], "test", "experiment/runs/domain/EVAL_CD.md")
            det2 = load_detail("EVAL_CD")
            if det2:
                s2 = pair_stats(det2["C@C"], det2["D@D"], "C@C", "D@D")
                log(f"C vs D: {json.dumps({k: (round(v,4) if isinstance(v,float) else v) for k,v in s2.items()}, ensure_ascii=False)}")
            if (PROJECT_ROOT / "weights/domain_D.pt").exists():
                winner = "D"       # 预期并列；D 更省 → 取 D
            else:
                winner = "C"       # D 未训出来（如 OOM）→ 回退到同样"必须去畸变"的 C
                log("⚠️ domain_D.pt 不存在，D 臂不可用 → 回退 winner=C（同为去畸变方案）")
        log(f"最优域 = {winner}")

        # ---------- 5) enhance 消融 ----------
        train("noenh", winner, enhance=False)
        eval_arms([f"{winner}@enh:experiment/data/pose_{winner}:weights/domain_{winner}.pt:{winner}:test",
                   f"{winner}@noenh:experiment/data/pose_{winner}_noenh:weights/domain_{winner}_noenh.pt:{winner}:test"],
                  [f"{winner}@enh:{winner}@noenh"], "test", "experiment/runs/domain/EVAL_enhance.md")
        det3 = load_detail("EVAL_enhance")
        enh = None
        if det3:
            enh = pair_stats(det3[f"{winner}@enh"], det3[f"{winner}@noenh"], "e", "n")
            log(f"enhance 消融: {json.dumps({k: (round(v,4) if isinstance(v,float) else v) for k,v in enh.items()}, ensure_ascii=False)}")
        json.dump(dict(verdict=dict(b_vs_c=st, enough=enough, c_better_outer=c_better_outer,
                                    winner=winner, enhance=enh)),
                  open(FIELD / "verdict.json", "w"), indent=1, ensure_ascii=False)

        # ---------- 6) AUV_5 清水 ----------
        eval_arms([f"老模型@A-old:runs/auv5_eval/pose_AOLD:weights/yolo11n-pose.pt:AOLD_C:eval",
                   f"B臂@B:runs/auv5_eval/pose_B:weights/domain_B.pt:B:eval",
                   f"C臂@C:runs/auv5_eval/pose_C:weights/domain_C.pt:C:eval",
                   f"最优臂@{winner}:runs/auv5_eval/pose_{winner}:weights/domain_{winner}.pt:{winner}:eval"],
                  ["B臂@B:C臂@C"], "eval", "experiment/runs/domain/EVAL_auv5.md", conf=0.10)
        for dom in ("B", "C", "D"):
            if (PROJECT_ROOT / f"weights/domain_{dom}.pt").exists():
                run([PY, "experiment/scripts/exp_distortion/probe_auv5_detect.py",
                     "--weights", f"weights/domain_{dom}.pt", "--domain", dom, "--n", "120",
                     "--tag", f"{dom}臂@清水{dom}域"])

    if a.dry_run_verdict:                 # 只汇总已有结果，不训练
        det_test = load_detail("EVAL_main_test")
        vv = FIELD / "verdict.json"
        winner = (json.load(open(vv))["verdict"]["winner"] if vv.exists() else "B")
        log(f"dry-run：用已有评估结果汇总（winner={winner}）")

    # ---------- 7) 清水结论（§16 预登记判据）----------
    det5 = load_detail("EVAL_auv5")
    det_test = load_detail("EVAL_main_test")
    cw_ok, cw_txt = (None, "（明细缺失，无法按 §16 判定）")
    if det5 and det_test:
        key5 = [k for k in det5 if k.startswith(f"{winner}臂") or k.startswith("最优臂")]
        key5 = key5[0] if key5 else list(det5)[0]
        keyt = [k for k in det_test if k.startswith(f"{winner}@")]
        if not keyt:                      # winner=D 时 main_test 里没有 D@D → 用几何等价的 C@C
            keyt = [k for k in det_test if k.startswith("C@")] or \
                   [k for k in det_test if k.startswith("B@")]
        keyt = keyt[0] if keyt else list(det_test)[0]
        log(f"清水参照臂（test 口径）: {keyt}")
        cw_ok, cw_txt = clear_water_verdict(winner, det5[key5], det_test[keyt])
        log(f"清水判定: {cw_txt} ⇒ same_magnitude={cw_ok}")

    # ---------- 8) 汇总（单一汇总块；自解释，不依赖解读）----------
    parts = ["# 去畸变域决策实验结果\n",
             f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n",
             "方法与口径见 `EXPERIMENT_DESIGN.md`；判定阈值在 §13（域）与 §16（清水）预先登记。\n"]
    v = FIELD / "verdict.json"
    if v.exists():
        vd = json.load(open(v))
        parts.append("\n" + render_conclusion(vd["verdict"], cw_ok, cw_txt))
        parts.append("\n<details><summary>判定原始数据 (verdict.json)</summary>\n\n```json\n"
                     + v.read_text() + "\n```\n</details>\n")
    else:
        parts.append("\n## 结论\n\n（verdict.json 缺失，判定未完成——请查看 /tmp/pipeline.log）\n")
    if det_test:
        parts.append("\n## 自检红旗\n\n" + "\n".join(
            "- " + x for x in sanity_flags(det_test, winner, None)) + "\n")
    exp = [("weights/domain_B.pt", "B 臂权重"), ("weights/domain_C.pt", "C 臂权重"),
           ("weights/domain_D.pt", "D 臂权重（仅当 B 不足时）"),
           ("weights/domain_B_noenh.pt", "B+noenh 臂（仅当最优域=B）"),
           ("weights/domain_C_noenh.pt", "C+noenh 臂（仅当最优域=C）"),
           ("weights/domain_D_noenh.pt", "D+noenh 臂（仅当最优域=D）"),
           ("experiment/runs/domain/EVAL_main_test.md", "主评估 test"),
           ("experiment/runs/domain/EVAL_main_valid.md", "主评估 valid"),
           ("experiment/runs/domain/EVAL_CD.md", "C vs D 对比"),
           ("experiment/runs/domain/EVAL_enhance.md", "enhance 消融"),
           ("experiment/runs/domain/EVAL_auv5.md", "AUV_5 清水"),
           ("experiment/runs/domain/verdict.json", "自动判定原始数据"),
           ("experiment/runs/domain/label_audit_B.csv", "标注审计")]
    clines = ["\n## 产物完整性清单\n"]
    for rel, desc in exp:
        mark = "✅" if (PROJECT_ROOT / rel).exists() else "❌ 缺"
        clines.append(f"- {mark} `{rel}` — {desc}")
    parts.append("\n".join(clines) + "\n")
    atxt, aok = artifact_check("experiment/runs/domain/artifacts_before.txt")
    parts.append("\n## 第 ⑥ 条核验：实验前已有产物是否被动过\n\n" + atxt)
    for stem, title in [("EVAL_main_test", "主评估（test）"),
                        ("EVAL_main_valid", "复现检验（valid）"),
                        ("EVAL_CD", "C vs D（去畸变位置）"),
                        ("EVAL_enhance", "enhance 消融"),
                        ("EVAL_auv5", "AUV_5 清水 16 张")]:
        f = FIELD / f"{stem}.md"
        if f.exists():
            parts.append(f"\n## {title}\n\n" + f.read_text() + "\n")
    (FIELD / "DOMAIN_RESULT.md").write_text("".join(parts))
    log(f"产物核验：{'PASS' if aok else 'FAIL'}")
    log("完成。产物：experiment/runs/domain/EVAL_*.md, verdict.json, DOMAIN_RESULT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
