#!/usr/bin/env python3
"""
畸变实验 · 相邻关键点间距一致性评估（"各组相邻特征点之间的距离是否一致"）

## 判据怎么来的

门框在物理上是一个**矩形**（板端 `gate.geometry`：0.70 m × 0.50 m）。四个角点
TL,TR,BR,BL 的**相邻边**就是门框的四条边：

    s0 = |TL-TR|（上边）   s1 = |TR-BR|（右边）
    s2 = |BR-BL|（下边）   s3 = |BL-TL|（左边）

关键：**不要直接跨组比像素长度**。哪一组更"对"跟绝对像素数无关，而且
`alpha=0` 的整流会让 A/C 相对 B 整体缩放，跨组比绝对长度是错的。要看的是
**同一张图内部四条边的自洽性**（尺度无关）：

| 指标 | 定义 | 期望 |
|---|---|---|
| `e_opp` | `max(\|s0-s2\|/(s0+s2), \|s1-s3\|/(s1+s3))` | 越接近 0 越好：正对门框时对边应等长 |
| `r_tb` | `s0/s2` | 随俯仰缓慢变化，**不应随"门在画面里的偏心程度"漂移** |
| `r_lr` | `s1/s3` | 同上 |
| `rho` | 四边形中心到**本组主点**的距离（按画面对角线归一化） | 畸变误差是径向的 → 关键判据是 `e_opp` 与 `rho` 的相关性 |

**最有诊断价值的一条**：`e_opp` 对 `rho` 的皮尔逊相关 `corr`。畸变没被正确
校正时，画面外圈的门框四边形会被径向拉伸/压缩，`e_opp` 随 `rho` 单调变差；
正确整流后这条相关性应当消失（≈0）。脚本会逐组给出 `corr` 与分环统计。

> 门框还有宽高比 0.70/0.50 = 1.4，但 `r_aspect` 同时受真实姿态影响，
> 所以只作为参考量打印，**不作为判据**。

## 用法

    # 三组预测都跑一遍（用当前模型，无需重新训练）
    python experiment/scripts/exp_distortion/compare_kpt_spacing.py \
        --experiment data/exp_distortion/processed \
        --group A --group B --group "Cprime=B@rect" \
        --model weights/yolo11n-pose.pt --device 0 \
        --out-dir experiment/runs/exp_distortion/eval

    # 用人工标注（标签目录）当输入，测的是"几何本身"而不是模型
    python ... --group A --group B --group C --labels-root data/exp_distortion/labels

    # 只看 A 组里那些"标定可信 + 有标注"的帧
    python ... --valid-only --min-kpt 4

组名写法：`A` / `B` / `C`（短名）或全名；`B@rect` = 用 B 的图检测、再把坐标
换算到去畸变域（= C′ 组）；用 `--experiment` 时会自动补全目录。

⚠️ 用 `--model` 预测需要**原版 head**（脚本会自己检查并报错，附恢复命令）。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from distortion_geometry import (Geometry, GROUP_A, GROUP_B, GROUP_C,  # noqa: E402
                                 GROUP_CPRIME, GROUP_DESC, resolve_group)
from transform_pose_labels import normalize_stem  # noqa: E402  （RoboFlow 文件名归一化）

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ------------------------------------------------------------------ 输入

def parse_group_spec(spec: str, exp: dict | None):
    """组说明 → (tag, group, dir, rect)

    写法：
      `A` / `B` / `C`        ：短名，目录从 --experiment 推断
      `C'`                   ：= B 的图 + 坐标后映射（畸变域检测 → a640 域）
      `NAME@rect`            ：显式对 B 加坐标后映射
      `NAME=DIR`             ：显式目录（NAME 仍须是 A/B/C/C'）
    """
    rect = False
    if spec.endswith("@rect"):
        rect, spec = True, spec[:-5]
    if "=" in spec:
        tag, d = spec.split("=", 1)
    else:
        tag, d = spec, None
    tag = tag.strip()
    resolved = resolve_group(tag)
    if resolved == GROUP_CPRIME:      # C′ = B 的图 + 坐标后映射
        rect, group = True, GROUP_B
    else:
        group = resolved
    if d is None:
        if exp is None:
            sys.exit(f"❌ --group {tag} 未给目录，且没有 --experiment 可推断")
        d = exp["groups"][group]["dir"]
    return tag, group, Path(d), rect


def collect_instances(images_dir: Path, labels_dir: Path | None,
                      model, size: int, conf: float, device, max_side: int = 0):
    """产出一致结构: [(fname, kpts(N,4,2) 像素, box(N,4), vis(N,4))]

    `vis` 是关键点可信度：标注路径用 `v`，模型路径用 `keypoints.conf`。

    ⚠️ YOLO 对**不确定的关键点会输出 (0,0)**，标注里没标的点也是 (0,0)。
    把 (0,0) 当坐标用，相邻边长就变成垃圾（"两点重合 → 边长 0"），
    所以下游必须按 `vis` 过滤，**不能直接拿坐标算**。
    """
    out = []
    files = sorted(f for f in images_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
    if max_side:
        files = files[:max_side]
    if labels_dir is not None:
        # RoboFlow 导出的标签名是 `xxx_jpg.rf.<hash>.txt`，先建规范化索引兜底
        idx = {}
        for txt in labels_dir.glob("*.txt"):
            idx.setdefault(normalize_stem(txt.stem), txt)
        for f in files:
            lab = labels_dir / f"{f.stem}.txt"
            if not lab.exists():
                lab = idx.get(normalize_stem(f.stem))
            if lab is None or not Path(lab).exists():
                continue
            kp, bx, vs = [], [], []
            for line in lab.read_text(encoding="utf-8").splitlines():
                v = line.split()
                if len(v) < 5 + 3 * 4:
                    continue
                k = np.array([[float(v[5 + 3 * i]) * size,
                               float(v[6 + 3 * i]) * size,
                               float(v[7 + 3 * i])] for i in range(4)])
                kp.append(k[:, :2])
                vs.append(k[:, 2])
                bx.append(np.array([float(v[1]) * size, float(v[2]) * size,
                                    float(v[3]) * size, float(v[4]) * size]))
            if kp:
                out.append((f.name, np.stack(kp), np.stack(bx), np.stack(vs)))
        return out
    if model is None:
        return out
    for r in model.predict(source=str(images_dir), imgsz=size, conf=conf,
                           device=device, verbose=False, stream=True):
        if r.keypoints is None or len(r.keypoints) == 0:
            continue
        k = r.keypoints.xy.cpu().numpy()                       # (N,4,2)
        vs = (r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None
              else np.ones(k.shape[:2]))
        b = r.boxes.xywh.cpu().numpy() if r.boxes is not None else np.zeros((len(k), 4))
        if len(b) != len(k):
            # ultralytics 在个别帧上会给出「boxes 为空但 keypoints 非空」的组合
            # （AUV_5 上实测遇到过），直接索引会 IndexError。
            # instance_metrics 只用 kpts、不用 box，这里对齐长度即可，语义不变。
            b = np.zeros((len(k), 4), np.float32)
        out.append((Path(r.path).name, k, b, vs))
    return out


# ------------------------------------------------------------------ 指标

def instance_metrics(kpts: np.ndarray, box: np.ndarray, size: int) -> dict | None:
    if kpts.shape[0] != 4:
        return None
    s = [float(np.linalg.norm(kpts[i] - kpts[(i + 1) % 4])) for i in range(4)]
    if min(s) <= 0:
        return None
    s0, s1, s2, s3 = s
    e_opp = max(abs(s0 - s2) / (s0 + s2), abs(s1 - s3) / (s1 + s3))
    return {"s_top": s0, "s_right": s1, "s_bottom": s2, "s_left": s3,
            "s_mean": float(np.mean(s)), "e_opp": float(e_opp),
            "r_tb": s0 / s2, "r_lr": s1 / s3,
            "r_aspect": (0.5 * (s0 + s2)) / max(1e-9, 0.5 * (s1 + s3)),
            "q_diag": float(np.linalg.norm(kpts[0] - kpts[2])),
            "center_x": float(kpts[:, 0].mean()), "center_y": float(kpts[:, 1].mean())}


def summarize(rows: list[dict], principal: tuple[float, float], size: int) -> dict:
    if not rows:
        return {}
    e = np.array([r["e_opp"] for r in rows])
    rho = np.array([np.hypot(r["center_x"] - principal[0],
                             r["center_y"] - principal[1]) for r in rows])
    diag = size * np.sqrt(2)
    rn = rho / diag
    corr = float(np.corrcoef(rn, e)[0, 1]) if len(rows) > 2 and e.std() > 1e-9 else float("nan")
    out = {"n_instances": len(rows),
           "e_opp_mean": float(e.mean()), "e_opp_median": float(np.median(e)),
           "e_opp_p90": float(np.percentile(e, 90)),
           "e_opp_frac_lt_0.02": float((e < 0.02).mean()),
           "e_opp_frac_lt_0.05": float((e < 0.05).mean()),
           "r_tb_mean": float(np.mean([r["r_tb"] for r in rows])),
           "r_tb_std": float(np.std([r["r_tb"] for r in rows])),
           "r_lr_mean": float(np.mean([r["r_lr"] for r in rows])),
           "r_lr_std": float(np.std([r["r_lr"] for r in rows])),
           "s_mean_median_px": float(np.median([r["s_mean"] for r in rows])),
           "rho_norm_median": float(np.median(rn)),
           "corr_e_opp_vs_rho": corr}
    # 分环：中心 / 中间 / 外圈（按归一化半径三分位）
    q1, q2 = np.percentile(rn, [33, 66]) if len(rows) >= 3 else (0, 0)
    for name, m in (("inner", rn <= q1), ("mid", (rn > q1) & (rn <= q2)),
                    ("outer", rn > q2)):
        if m.sum():
            out[f"e_opp_{name}"] = float(e[m].mean())
    return out


# ------------------------------------------------------------------ head 守卫

def check_head_patched() -> str | None:
    """预测需要原版 head；返回错误信息（None = 可以跑）"""
    try:
        p = Path(importlib.util.find_spec("ultralytics").origin).parent / "nn/modules/head.py"
    except Exception:  # noqa: BLE001
        return None
    if not p.exists():
        return None
    if "permute(0, 2, 3, 1)" in p.read_text(encoding="utf-8"):
        return (f"❌ site-packages 的 head.py 是**补丁版**（分裂输出头），"
                f"predict 会报张量形状错。先恢复原版：\n"
                f"   /home/ansty/anaconda3/envs/yolov8/bin/python "
                f"scripts/3_export/modify_ultralytics.py --restore")
    return None


# ------------------------------------------------------------------ 主流程

def main() -> None:
    ap = argparse.ArgumentParser(
        description="相邻关键点间距一致性评估（畸变实验三组对比）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--experiment", type=Path, default=None,
                    help="prepare_distortion_groups.py 的 --out-root（读 experiment.json）")
    ap.add_argument("--group", action="append", default=[],
                    help="组说明，可重复：NAME[=DIR][@rect]（如 A / B / B@rect）")
    ap.add_argument("--labels-root", type=Path, default=None,
                    help="若有则用 <root>/<组名>/ 下的人工标注代替模型预测")
    ap.add_argument("--model", type=Path, default=None, help="pose 权重（原版 head 时可用）")
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25, help="检测框置信度阈值")
    ap.add_argument("--kpt-conf", type=float, default=0.5,
                    help="关键点可信度阈值：四点全过才参与边长统计（模型看 keypoints.conf，"
                         "标注看 v>0）。低于阈值的点会被 YOLO 输出成 (0,0)，不能当坐标用")
    ap.add_argument("--max-images", type=int, default=0, help="每组最多评估多少张（0=全部）")
    ap.add_argument("--restrict-mappable", action="store_true",
                    help="同口径比较：剔除所有「在 a640 域无定义」的实例"
                         "（A/C 整流图只覆盖原始画面中央，不剔除会让 B 组白占便宜）")
    ap.add_argument("--valid-only", action="store_true",
                    help="只用「标定可信 + 全部关键点落在有效区域」的实例（推荐）")
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT_ROOT / "runs" / "exp_distortion" / "eval")
    a = ap.parse_args()

    exp = None
    if a.experiment:
        p = a.experiment / "experiment.json"
        if not p.exists():
            sys.exit(f"❌ 找不到 {p}")
        exp = json.loads(p.read_text(encoding="utf-8"))

    specs = a.group or (["A", "B", "C"] if exp else [])
    if not specs:
        sys.exit("❌ 至少给一个 --group（或 --experiment）")
    parsed = [parse_group_spec(s, exp) for s in specs]

    # 几何（换算 + 有效区域判定）
    geo = None
    if exp:
        geo = Geometry(exp["calibration"], exp["size"], exp["alpha"],
                       tuple(exp["raw_size"]))
        g = geo.for_frame(*exp["raw_size"])
    need_rect = any(r for _, _, _, r in parsed)
    if need_rect and geo is None:
        sys.exit("❌ 用了 @rect 就必须给 --experiment（需要标定与几何参数）")

    model = None
    if a.labels_root is None:
        if a.model is None:
            sys.exit("❌ 没有 --labels-root 就必须给 --model（否则没有输入）")
        err = check_head_patched()
        if err:
            sys.exit(err)
        from ultralytics import YOLO
        model = YOLO(str(a.model))
        print(f"🤖 模型 {a.model}  device={a.device} conf={a.conf}")
    else:
        if not a.labels_root.is_dir():
            sys.exit(f"❌ --labels-root 不存在: {a.labels_root}")
        print(f"📄 使用人工标注作为输入: {a.labels_root}/<组名>/（不跑模型）")

    size = int(exp["size"]) if exp else 640
    a.out_dir.mkdir(parents=True, exist_ok=True)
    summary_all, per_frame = {}, {}

    for tag, group, images_dir, rect in parsed:
        if not images_dir.is_dir():
            sys.exit(f"❌ 图片目录不存在: {images_dir}")
        # 给了 --labels-root 就用标注（即便 @rect：先取畸变域标注，再整体换算）
        labels_dir = None
        if a.labels_root is not None:
            short = {GROUP_A: "A", GROUP_B: "B", GROUP_C: "C"}.get(group, tag)
            for cand in (a.labels_root / group, a.labels_root / short,
                         a.labels_root / f"labels_{short}", a.labels_root / tag):
                if cand.is_dir():
                    labels_dir = cand
                    break
            if labels_dir is None:
                print(f"⚠️  {tag}: {a.labels_root} 下找不到该组标注目录"
                      f"（试过 {group}/、{short}/、labels_{short}/、{tag}/），跳过")
                continue
        inst = collect_instances(images_dir, labels_dir, model, size, a.conf,
                                 a.device, a.max_images)

        # ---- 同口径约束：只保留「在 a640 域里有定义」的实例 ----
        # 为什么必须做：A/C 的整流图只覆盖原始画面中央（本标定实测
        # raw x∈[147,1137]、y∈[34,689]），所以 B 组能检到的门有一部分
        # **根本不在 A/C 的画面里**。不筛的话 A/C 会白背"检出率低"的锅，
        # 那不是模型不行，是画面没覆盖到。
        unreachable = 0
        if a.restrict_mappable and geo is not None:
            kept = []
            for name, k, b, vs in inst:
                _, ok = g.transform(k.reshape(-1, 2), group, GROUP_A)
                reach = ok.reshape(-1, 4).all(axis=1)
                unreachable += int((~reach).sum())
                if reach.any():
                    kept.append((name, k[reach], b[reach], vs[reach]))
            inst = kept

        # 换算到去畸变域（C′：检测在畸变域做、坐标最后换算）
        if rect and group != GROUP_B:
            print(f"⚠️  {tag}: @rect 只对 B（畸变域）有意义，已忽略")
            rect = False
        if rect:
            out_inst = []
            for name, k, b, vs in inst:
                moved, ok = g.transform(k.reshape(-1, 2), group, GROUP_A)
                keep = ok.reshape(-1, 4).all(axis=1)
                if keep.any():
                    out_inst.append((name, moved.reshape(-1, 4, 2)[keep],
                                     b[keep], vs[keep]))
            inst = out_inst
        if a.restrict_mappable and geo is not None and unreachable:
            print(f"🔎 {tag}: 排除 {unreachable} 个「在 a640 域无定义」的实例"
                  f"（A/C 画面覆盖不到，同口径比较必须剔除）")

        # ---- 关键点可信度过滤 ----
        # 只保留「四点全部可信」的实例：算的是四条边的长度，缺一个点就没有边。
        # 模型对不确定的点会输出 (0,0)，标注里没标的点也是 (0,0)，直接当坐标用
        # 会得到"边长 0"这种垃圾值。丢了多少本身就是兼容性信号，所以要报出来。
        n_inst_total = int(sum(len(k) for _, k, _, _ in inst))
        rows = []
        n_lowconf = n_degen = 0
        for name, k, b, vs in inst:
            for i in range(len(k)):
                if not np.all(vs[i] >= a.kpt_conf):
                    n_lowconf += 1
                    continue
                if a.valid_only and geo is not None and not np.all(
                        g.inside(k[i].reshape(-1, 2))):
                    continue
                m = instance_metrics(k[i], b[i], size)
                if m is None:
                    n_degen += 1
                    continue
                m["file"] = name
                m["kpt_conf_min"] = float(vs[i].min())
                rows.append(m)

        # 主点（按各组自身坐标域）
        if exp:
            if group == GROUP_B:
                principal = (g.K[0, 2] * g.sx, g.K[1, 2] * g.sy)
            elif group == GROUP_C and not rect:
                principal = (g.newK640[0, 2], g.newK640[1, 2])
            else:
                principal = (g.newK720[0, 2] * g.sx, g.newK720[1, 2] * g.sy)
        else:
            principal = (size / 2, size / 2)

        summ = summarize(rows, principal, size)
        summ["images"] = len({r["file"] for r in rows})
        summ["instances_raw"] = n_inst_total
        summ["instances_lowconf"] = n_lowconf
        summ["instances_degenerate"] = n_degen
        summ["dir"] = str(images_dir)
        summ["source"] = "labels" if labels_dir else "model"
        summ["domain"] = ("去畸变域(a640)" if (rect or group in (GROUP_A, GROUP_C))
                          else "畸变域(b640)")
        summ["principal_point"] = list(principal)
        summary_all[tag] = summ
        per_frame[tag] = {r["file"]: r for r in rows}

        csvp = a.out_dir / f"metrics_{tag.replace(chr(39), 'p')}.csv"
        with csvp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["file"])
            w.writeheader()
            w.writerows(rows)

    # ---- 打印 ----
    hdr = (f"{'组':10s} {'域':14s} {'图':>4s} {'检出':>5s} {'四点可信':>8s} "
           f"{'低可信丢':>8s} {'e_opp中位':>9s} {'e_opp均值':>9s} {'p90':>7s} "
           f"{'<0.05占比':>9s} {'r_tb±sd':>14s} {'corr(ρ)':>8s} {'内圈':>7s} {'外圈':>7s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for tag, s in summary_all.items():
        if not s:
            print(f"{tag:10s} （无有效实例）")
            continue
        print(f"{tag:10s} {s['domain']:14s} {s['images']:4d} "
              f"{s.get('instances_raw', 0):5d} {s['n_instances']:8d} "
              f"{s.get('instances_lowconf', 0):8d} "
              f"{s['e_opp_median']:9.4f} {s['e_opp_mean']:9.4f} "
              f"{s['e_opp_p90']:7.4f} {s['e_opp_frac_lt_0.05'] * 100:8.1f}% "
              f"{s['r_tb_mean']:6.3f}±{s['r_tb_std']:.3f} "
              f"{s['corr_e_opp_vs_rho']:8.3f} "
              f"{s.get('e_opp_inner', float('nan')):7.4f} "
              f"{s.get('e_opp_outer', float('nan')):7.4f}")

    # ---- 配对比较（同一帧，尺度无关的比值指标）----
    tags = list(per_frame)
    pair_rows = []
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            x, y = tags[i], tags[j]
            common = sorted(set(per_frame[x]) & set(per_frame[y]))
            if not common:
                continue
            de, drtb, drlr, sratio = [], [], [], []
            for f in common:
                rx, ry = per_frame[x][f], per_frame[y][f]
                de.append(rx["e_opp"] - ry["e_opp"])
                drtb.append(rx["r_tb"] - ry["r_tb"])
                drlr.append(rx["r_lr"] - ry["r_lr"])
                sratio.append(rx["s_mean"] / max(1e-9, ry["s_mean"]))
            de = np.array(de)
            pair_rows.append({"pair": f"{x} - {y}", "n_common": len(common),
                              "d_e_opp_mean": float(de.mean()),
                              "d_e_opp_median": float(np.median(de)),
                              "frac_x_better": float((de < 0).mean()),
                              "d_r_tb_std": float(np.std(drtb)),
                              "d_r_lr_std": float(np.std(drlr)),
                              "scale_ratio_median": float(np.median(sratio))})
    if pair_rows:
        print(f"\n{'配对（左-右，负=左边更规整）':28s} {'共同帧':>6s} {'Δe_opp均值':>11s} "
              f"{'Δe_opp中位':>11s} {'左边更好占比':>12s} {'尺度比中位':>10s}")
        for r in pair_rows:
            print(f"{r['pair']:28s} {r['n_common']:6d} {r['d_e_opp_mean']:11.4f} "
                  f"{r['d_e_opp_median']:11.4f} {r['frac_x_better'] * 100:11.1f}% "
                  f"{r['scale_ratio_median']:10.3f}")
        with (a.out_dir / "pairwise.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
            w.writeheader()
            w.writerows(pair_rows)

    (a.out_dir / "summary.json").write_text(
        json.dumps({"summary": summary_all, "pairwise": pair_rows},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n💾 {a.out_dir}/summary.json + pairwise.csv + metrics_*.csv")
    print("   读法：e_opp 越小越好；corr(ρ) 是 e_opp 与『门框中心到主点距离』的相关性，")
    print("         畸变没校正干净时会显著为正（外圈更差）；正确整流后应接近 0。")
    print("   ⚠️ 跨组不要比 s_mean 绝对值：A/C 经 alpha=0 整流后整体缩放，与 B 不同。")
    if exp and exp["mappable_stats"]["b640_mappable_frac"] < 0.9:
        print(f"   ⚠️ 本标定只有 {exp['mappable_stats']['b640_mappable_frac'] * 100:.0f}% "
              f"的画面存在去畸变解，外圈结论不可用（见 calib_diagnose.py）")


if __name__ == "__main__":
    main()
