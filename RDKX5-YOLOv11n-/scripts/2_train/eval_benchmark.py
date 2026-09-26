#!/usr/bin/env python3
"""固定基准评测器：用同一套板端判据量「权重到底能不能用」，输出 A–D 四节 + 门槛判定。

A. 清水人眼基准   data/derived/AUV_5_selected_3000_Dwb 的 1190 张（带肉眼分桶）
                  判定取 judgment/rejudge_log.csv，其次 triage_log.csv；可用 --verdict 覆盖
B. 分层保留集     tables/pool/eval_reserved.csv（无标注，只报代理指标盯遗忘）
C. 负样本集       tables/pool/negatives_verified.csv（不存在则跳过并标注未测）
D. 板端视角基准   judgment/boardview_*_log.csv + tables/boardview/boardview_list_*.csv
                  可用 --boardview-log / --boardview-list / --boardview-subset 指定；
                  报严格口径（每个画出的门都必须对）与宽松口径两个不可用率
                  ⚠️ D 与 A 不是同一件事、不可对比：A 问「识别结果是否理想」，D 问「画出来的门对不对」

板端判据（本仓工程约束，不是正确性标准）：
  1) 逐角点 v ≥ V_MIN(0.8)；拦掉 (0,0) 占位点造成的假乱飞
  2) 四边形不自交、顺序合法
  3) 去重复框：小框 conf 更低 且 ≥MIN_EDGES(2) 条边重合（tol=EDGE_TOL(0.10)×小框短边）→ 丢小框；不看「包含」
  4) 多实例排序：按干净度（末位面积）——本脚本自己的规则，不代表板端选门

验收门槛：good 保留率 ≥ 0.90（good+so-so）且 ≥ 0.72（good）；坏率 + 负样本检出率 ≤ 0.10

用法：
    /home/ansty/anaconda3/envs/yolov8/bin/python scripts/2_train/eval_benchmark.py \
        --weights weights/yolo11n-pose.pt --device 0 \
        --json output/preview/auv5_pose/benchmark_<name>.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PREVIEW = REPO / "output" / "preview" / "auv5_pose"
POOL = REPO / "data" / "derived" / "AUV_5_selected_3000_Dwb"
BAD = {"1", "2"}

V_MIN = 0.8          # 逐角点置信度下限
EDGE_TOL = 0.10      # 重复框：边重合容忍 = 该比例 × 小框短边
MIN_EDGES = 2        # 重复框：至少几条边重合
# 验收门槛：
#   (good+so-so) 保留 >= 0.90 且 good 保留 >= 0.72（宽档 0.63）且 good 占多数
#   坏率 + 负样本误检率 <= 0.10
TARGET_COMBINED_KEEP = 0.90
TARGET_GOOD_KEEP = 0.72          # 严档；宽档 = 0.63
TARGET_GOOD_KEEP_LOOSE = 0.63
TARGET_BAD_TOTAL = 0.10


# ---------------- 几何判据 ----------------

def _cross2(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def self_crossed(xy) -> bool:
    def hit(a, b, c, d):
        d1, d2 = _cross2(a, b, c), _cross2(a, b, d)
        d3, d4 = _cross2(c, d, a), _cross2(c, d, b)
        return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))
    return hit(xy[0], xy[1], xy[2], xy[3]) or hit(xy[1], xy[2], xy[3], xy[0])


def order_ok(xy) -> bool:
    x, y = xy[:, 0], xy[:, 1]
    return (x[0] < x[1]) and (x[3] < x[2]) and (y[0] < y[3]) and (y[1] < y[2])


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def n_edge_coincident(a, b) -> int:
    wa, ha, wb, hb = a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]
    tol = EDGE_TOL * min(wa, ha, wb, hb)
    return sum(1 for i in range(4) if abs(a[i] - b[i]) <= tol)


def instance_quality(xy: np.ndarray, v: np.ndarray, box: np.ndarray) -> dict:
    """给一个实例打分：能否当板端输出"""
    n_hi = int((v >= V_MIN).sum())
    crossed = self_crossed(xy)
    ok = bool(n_hi == 4 and not crossed and order_ok(xy))
    return dict(n_hi=n_hi, crossed=int(crossed), usable=ok,
                score=(n_hi, -int(crossed), int(order_ok(xy)), float(v.min()),
                       float(area(box))), box=box)


def decode_instances(insts: list[dict], return_dropped: bool = False):
    """板端解码：先去重复框，再按干净度排序（不按 conf）

    return_dropped=True 时返回 (keep, dropped_insts)，供可视化使用。
    """
    dropped: set[int] = set()
    n = len(insts)
    for i in range(n):
        for j in range(i + 1, n):
            big_i, small_i = (i, j) if area(insts[i]["box"]) >= area(insts[j]["box"]) else (j, i)
            big, small = insts[big_i]["box"], insts[small_i]["box"]
            if area(small) <= 0:
                continue
            # 不看「包含」：远处第二个门可能恰好落在近门的框内，用包含判重复会误删真门。
            # 只用「小框 conf 更低 + ≥2 条边重合」。
            if insts[small_i]["conf"] >= insts[big_i]["conf"]:
                continue                       # 小的那个必须置信度更低
            if n_edge_coincident(small, big) < 2:
                continue                       # 至少两条边重合才算同一个门的重复框
            dropped.add(small_i)               # 丢小框，留大框
    keep = [r for k, r in enumerate(insts) if k not in dropped]
    keep.sort(key=lambda r: r["score"], reverse=True)
    if return_dropped:
        return keep, [r for k, r in enumerate(insts) if k in dropped]
    return keep


def run_model(weights: str, names: list[str], imgs_root: Path, args) -> dict:
    """跑推理，返回 name -> [实例质量]"""
    from ultralytics import YOLO
    lst = PREVIEW / "filelists" / "sets" / "_bench_filelist.txt"
    lst.write_text("\n".join(str((imgs_root / n).absolute()) for n in names), encoding="utf-8")
    model = YOLO(weights)
    out = defaultdict(list)
    res_iter = model.predict(source=str(lst), imgsz=args.imgsz, conf=args.conf,
                            device=args.device, batch=args.batch, max_det=10,
                            stream=True, verbose=False)
    for res in res_iter:
        n = Path(res.path).name
        k = 0 if res.boxes is None else len(res.boxes)
        if k and res.keypoints is not None and len(res.keypoints) == k:
            kd = res.keypoints.data.cpu().numpy()
            bx = res.boxes.xyxy.cpu().numpy()
            cf = res.boxes.conf.cpu().numpy()
            for i in range(k):
                q = instance_quality(kd[i, :, :2], kd[i, :, 2], bx[i])
                q["conf"] = float(cf[i])
                out[n].append(q)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--weights", default="weights/yolo11n-pose.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.6, help="检出阈值")
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--json", default=None, help="把结果写成 JSON（供选型/对比）")
    ap.add_argument("--boardview-log", default=None,
                    help="D 节的板端视角判定日志（默认 judgment/boardview_log.csv）")
    ap.add_argument("--boardview-subset", default=None,
                    help="D 节只统计 subset 列命中的帧（如 core,both → 282 张核心集）")
    ap.add_argument("--boardview-list", default=None,
                    help="D 节的板端视角清单（默认 tables/boardview/boardview_list.csv）")
    ap.add_argument("--verdict", default=None,
                    help="人眼判定日志（默认优先 rejudge_log.csv，其次 triage_log.csv）")
    a = ap.parse_args()

    report, summary = [], {}

    report.append(
        "> **口径与来源（先读这段）**\n"
        ">\n"
        "> * **误差**只能由 Roboflow 的 labels 算（已标 844 张）；没有 labels 的地方**算不出误差**。\n"
        "> * **好坏**只能由人眼判定（A 节的 1190 张分桶、C 节的无门确认、D 节的板端选中实例判定）。\n"
        "> * 本脚本里的 `v≥0.8`、`非自交`、`顺序合法`、`干净实例`、`重复框规则` 都是\n"
        ">   **工程约束**（目的是「板端别输出崩掉的门」），**不是正确性标准**；\n"
        ">   它们与上一条人眼判定相乘得到的数字（如坏率、good 保留）**不是纯 ground truth**，\n"
        ">   判据一改数字就变，比较时必须在同一判据下比。\n"
        "> * B 节是**模型自评的代理指标**（保留集还没有标注/判定），只用于同口径比较，不能当结论。\n"
        "> * **A 节与 D 节不是同一件事，不可对比**：A 问「这张图的识别结果是否理想」（严口径人眼），"
        "D 问「画出来的门对不对」（板端可用性）。实测同一批 339 帧在 A 的判定里 184 张被判坏，"
        "在 D 里只有 27 张非 k。\n"
    )

    # ---------- A. 清水人眼基准 ----------
    # 判定来源优先级：重判结果 > 第一批判定；可用 --verdict 覆盖。
    if a.verdict:
        vpath = Path(a.verdict)
    else:
        vpath = next((p for p in (PREVIEW / "judgment" / "rejudge_log.csv", PREVIEW / "judgment" / "triage_log.csv")
                      if p.exists()), None)
    if vpath is None:
        raise SystemExit("❌ 找不到判定日志（rejudge_log.csv / triage_log.csv）")
    verdict = {r["name"]: r["bucket"] for r in csv.DictReader(open(vpath, encoding="utf-8"))}
    namesA = sorted(verdict)
    detA = run_model(a.weights, namesA, POOL, a)
    n_good = sum(1 for b in verdict.values() if b == "g")
    n_soso = sum(1 for b in verdict.values() if b == "s")
    acc, bad, good_keep, soso_keep = 0, 0, 0, 0
    miss_reason = defaultdict(int)          # 未放行的 good/so-so 帧，卡在哪一步
    layers = defaultdict(lambda: [0, 0, 0])          # water -> [n, accepted, bad]
    water = {r["name"]: r["water"] for r in
             csv.DictReader(open(POOL / "manifest.csv", encoding="utf-8"))}
    for n in namesA:
        raw = detA.get(n, [])
        insts = decode_instances(raw)
        usable = [r for r in insts if r["usable"]]
        if usable:
            acc += 1
            if verdict[n] in BAD:
                bad += 1
            elif verdict[n] == "g":
                good_keep += 1
            else:
                soso_keep += 1
        elif verdict[n] in ("g", "s"):
            # 人眼说可用但板端拿不出实例 —— 卡在哪一步？
            if not raw:
                miss_reason["a 无任何检出（conf<阈值）"] += 1
            elif not any(r["n_hi"] == 4 for r in raw):
                miss_reason["b 有检出但角点不足 4 个 v>=V_MIN（部分可见）"] += 1
            # c=「有四角齐的实例，但没一个可用（几何/顺序不合格）」；d=被去重复框丢掉
            elif not any(r["usable"] for r in raw):
                miss_reason["c 四角齐但几何/顺序不合格"] += 1
            else:
                miss_reason["d 其它（被去重复框或内部逻辑丢掉）"] += 1
        w = water.get(n, "?")
        layers[w][0] += 1
        if usable:
            layers[w][1] += 1
            if verdict[n] in BAD:
                layers[w][2] += 1
    combined_keep = (good_keep + soso_keep) / max(n_good + n_soso, 1)
    A_res = dict(n=len(namesA), accepted=acc,
                 bad_rate=bad / max(acc, 1),
                 good_keep=good_keep / max(n_good, 1),
                 soso_keep=soso_keep / max(n_soso, 1),
                 combined_keep=combined_keep,
                 good_share_of_kept=good_keep / max(good_keep + soso_keep, 1))
    summary["A_清水人眼基准"] = A_res
    report.append(f"## A. 清水人眼基准（{len(verdict)} 张，带肉眼判定）\n")
    report.append(f"> 判定来源：`{vpath.name}`（重判结果优先于第一批判定）\n")
    report.append(f"- 放行 **{acc}/{len(namesA)}**；放行里人眼判不可用 **{bad} = {100*A_res['bad_rate']:.1f}%**")
    report.append(f"- good 保留 **{good_keep}/{n_good} = {100*A_res['good_keep']:.1f}%**"
                  f"（门槛 ≥72% → {'✅' if A_res['good_keep'] >= 0.72 else '❌'}；"
                  f"≥63% → {'✅' if A_res['good_keep'] >= 0.63 else '❌'}）")
    report.append(f"- so-so 保留 {soso_keep}/{n_soso} = {100*A_res['soso_keep']:.1f}%")
    report.append(f"- **(good+so-so) 保留 {good_keep + soso_keep}/{n_good + n_soso} = "
                  f"{100*combined_keep:.1f}%**（门槛 ≥90% → "
                  f"{'✅' if combined_keep >= 0.90 else '❌'}）")
    report.append(f"- 放行中 good 占比 {100*A_res['good_share_of_kept']:.1f}%（good 占大多数 → "
                  f"{'✅' if A_res['good_share_of_kept'] > 0.5 else '❌'}）")
    report.append(f"- 坏率 {100*A_res['bad_rate']:.1f}% + 误检率 → 见 C 节")
    if miss_reason:
        report.append("\n**人眼判 good/so-so 但板端拿不出实例的帧卡在哪：**\n")
        report.append("| 原因 | 张数 |")
        report.append("|---|---|")
        for k_, v_ in sorted(miss_reason.items()):
            report.append(f"| {k_} | {v_} |")
        report.append("")
    report.append("\n| 水质 | 张数 | 放行 | 放行中坏 | 坏率 |")
    report.append("|---|---|---|---|---|")
    for w, (n, ac, bd) in sorted(layers.items()):
        report.append(f"| {w} | {n} | {ac} | {bd} | {100*bd/max(ac,1):.1f}% |")

    # ---------- B. 分层保留集（无标注，代理指标） ----------
    if (PREVIEW / "tables" / "pool" / "eval_reserved.csv").exists():
        rowsB = list(csv.DictReader(open(PREVIEW / "tables" / "pool" / "eval_reserved.csv", encoding="utf-8")))
        namesB = [r["name"] for r in rowsB]
        detB = run_model(a.weights, namesB, POOL, a)
        per = defaultdict(lambda: [0, 0])
        for r in rowsB:
            st = [x for x in decode_instances(detB.get(r["name"], [])) if x["usable"]]
            per[r["water"]][0] += 1
            per[r["water"]][1] += 1 if st else 0
        report.append("\n## B. 分层保留集（仅盯遗忘，代理指标）\n")
        report.append("| 水质 | 张数 | 有可用实例 | 可用率 |")
        report.append("|---|---|---|---|")
        for w, (n, k) in sorted(per.items()):
            report.append(f"| {w} | {n} | {k} | {100*k/max(n,1):.1f}% |")
            summary[f"B_{w}_usable"] = k / max(n, 1)

    # ---------- C. 负样本集 ----------
    negf = PREVIEW / "tables" / "pool" / "negatives_verified.csv"
    if not negf.exists():
        negf = PREVIEW / "judgment" / "negatives_log.csv"          # 用 triage_review 判的直接也能读
    if negf.exists():
        rows_neg = list(csv.DictReader(open(negf, encoding="utf-8")))
        if negf.name == "negatives_log.csv":          # triage 日志：n=无门
            negs = [r["name"] for r in rows_neg if r.get("bucket") == "n"]
        else:                                         # 手填表：human_no_gate=1
            negs = [r["name"] for r in rows_neg
                    if str(r.get("human_no_gate", "")).lower() in ("1", "1.0", "yes", "y", "true")]
        negdir = PREVIEW / "renders" / "negatives"
        root = negdir if negs and (negdir / negs[0]).exists() else POOL
        if not negs:
            report.append("\n## C. 负样本集\n\n- ⚠️ 判定文件存在但没有任何「无门」标记，误检率未测\n")
            negs = None
    if negf.exists() and negs:
        detC = run_model(a.weights, negs, root, a)
        fired = sum(1 for n in negs if [x for x in decode_instances(detC.get(n, [])) if x["usable"]])
        anydet = sum(1 for n in negs if detC.get(n))
        fp = fired / max(len(negs), 1)
        report.append(f"\n## C. 负样本集（人眼确认无门，{len(negs)} 张）\n")
        report.append(f"- 有任何检出（conf≥{a.conf}）：{anydet} 张 = {100*anydet/len(negs):.1f}%")
        report.append(f"- 通过解码判据（会被板端当门用）：**{fired} 张 = {100*fp:.1f}%** ← 误检率")
        summary["C_negative_fp"] = fp
    else:
        report.append("\n## C. 负样本集\n\n- ⚠️ 尚无 `negatives_verified.csv`，**误检率未测**；"
                      "先用 raw-data 未入选帧建立候选集再人眼确认（见 MATERIAL_LABELING_PLAN.md）\n")

    # ---------- D. 板端视角基准（图里所有实例都画出后的人眼判定）----------
    # 桶：k/w/b/p/x，只有 k 算"可用"。
    bv_log = Path(a.boardview_log) if a.boardview_log else PREVIEW / "judgment" / "boardview_log.csv"
    bv_list = Path(a.boardview_list) if a.boardview_list else PREVIEW / "tables" / "boardview" / "boardview_list.csv"
    if bv_log.exists() and bv_list.exists():
        bl = list(csv.DictReader(open(bv_list, encoding="utf-8")))
        info = {r["name"]: r for r in bl}
        vd = {r["name"]: r["bucket"] for r in csv.DictReader(open(bv_log, encoding="utf-8"))}
        judged = [n for n in vd if n in info]
        if a.boardview_subset:
            want = {s.strip() for s in a.boardview_subset.split(",") if s.strip()}
            # subset 列 >1 个值（如 "both"）时拆开匹配；兼容没有 subset 列的老清单
            judged = [n for n in judged
                      if want & set(str(info[n].get("subset", "all")).split("+"))]
        report.append(f"\n## D. 板端视角基准（{len(judged)} 张判过）\n")
        report.append(f"> 判定来源：`{bv_log.name}` + 清单 `{bv_list.name}`"
                      + (f"，subset 过滤 `{a.boardview_subset}`" if a.boardview_subset else "") + "\n")
        n_k = sum(1 for n in judged if vd[n] == "k")
        n_y = sum(1 for n in judged if vd[n] == "y")   # 多门：至少一个门可用、另有错门
        nbad_strict = len(judged) - n_k                # 严格：图上每个门都必须对，y 也算不可用
        nbad_lenient = len(judged) - n_k - n_y         # 宽松：多门里只要有一个好门就算可用
        tot = max(len(judged), 1)
        report.append(f"- **严格口径**（图上画出的每个门都必须对）：不可用率 "
                      f"**{nbad_strict}/{len(judged)} = {100*nbad_strict/tot:.1f}%**\n"
                      f"- **宽松口径**（多门里至少有一个门好 → 板端仍能拿到门）：不可用率 "
                      f"**{nbad_lenient}/{len(judged)} = {100*nbad_lenient/tot:.1f}%**"
                      f"（差额 `y` = {n_y} 张：一个对一个错的帧）\n"
                      "- 桶定义：k=都对且没漏门 / y=多门中一对多错 / w=有角点错 / b=有框错或多余框 / "
                      "p=有门只露一部分 / x=有门没被画出来（漏门）\n"
                      "> **板端选门规则（2026-09-26 用户给定）**：**近距离优先**；"
                      "**无法测距时**改为选**置信度高**或**画面大**的门。"
                      "所以严格口径更贴近板端真实行为（板端大概率拿最近的那个门 → 它必须是对的），"
                      "宽松口径只作参考。\n")
        summary["D_boardview_bad_rate_strict"] = nbad_strict / tot
        summary["D_boardview_bad_rate_lenient"] = nbad_lenient / tot
        summary["D_boardview_one_ok_one_bad"] = n_y / tot
        if any("subset" in r for r in bl):
            report.append("\n| subset | 判过 | 不可用 | 不可用率 |")
            report.append("|---|---|---|---|")
            sub = defaultdict(lambda: [0, 0])
            for n in judged:
                sub[info[n].get("subset", "?")][0] += 1
                if vd[n] not in ("k", "y"):
                    sub[info[n].get("subset", "?")][1] += 1
            for k, (n_, b_) in sorted(sub.items()):
                report.append(f"| {k} | {n_} | {b_} | {100*b_/max(n_,1):.1f}% |")
        report.append("\n| 桶 | 张数 |")
        report.append("|---|---|")
        for k, c in sorted(Counter(vd[n] for n in judged).items()):
            report.append(f"| {k} | {c} |")
        by_hb = defaultdict(lambda: [0, 0])
        for n in judged:
            by_hb[info[n]["human_bucket"]][0] += 1
            if vd[n] not in ("k", "y"):
                by_hb[info[n]["human_bucket"]][1] += 1
        report.append("\n| 原图级判定 | 判过 | 板端不可用 | 不可用率 |")
        report.append("|---|---|---|---|")
        for k, (n_, b_) in sorted(by_hb.items()):
            report.append(f"| {k} | {n_} | {b_} | {100*b_/max(n_,1):.1f}% |")
        if "n_pass" in (bl[0] if bl else {}):
            by_pass = defaultdict(lambda: [0, 0])
            for n in judged:
                key = f"过约束实例={info[n]['n_pass']}"
                by_pass[key][0] += 1
                if vd[n] not in ("k", "y"):
                    by_pass[key][1] += 1
            report.append("\n| 该图过约束实例数 | 判过 | 不可用 | 不可用率 |")
            report.append("|---|---|---|---|")
            for k, (n_, b_) in sorted(by_pass.items()):
                report.append(f"| {k} | {n_} | {b_} | {100*b_/max(n_,1):.1f}% |")
        else:
            by_multi = defaultdict(lambda: [0, 0])
            for n in judged:
                key = f"n_det={info[n]['n_all']}"
                by_multi[key][0] += 1
                if vd[n] not in ("k", "y"):
                    by_multi[key][1] += 1
            report.append("\n| 该图实例总数 | 判过 | 选中实例不可用 | 不可用率 |")
            report.append("|---|---|---|---|")
            for k, (n_, b_) in sorted(by_multi.items()):
                report.append(f"| {k} | {n_} | {b_} | {100*b_/max(n_,1):.1f}% |")
        report.append("\n> 用途：图级坏率回答「模型干净不干净」；D 节回答「板端真正拿到的那一个对不对」。"
                      "两者差额就是「被多余的框连累」的部分。\n")
    else:
        report.append("\n## D. 板端视角基准\n\n- ⚠️ 尚无 boardview 判定；先生成并判定："
                      "`make_boardview_set.py` → `triage_review.py --buckets "
                      "\"k=ok,w=wrong_kpt,b=wrong_box,p=partial,x=missed\"`\n")

    # ---------- 门槛判定 ----------
    total_bad = A_res["bad_rate"] + summary.get("C_negative_fp", 0.0)
    ok = (A_res["combined_keep"] >= TARGET_COMBINED_KEEP
          and A_res["good_keep"] >= TARGET_GOOD_KEEP
          and total_bad <= TARGET_BAD_TOTAL)
    report.append(f"\n## 门槛判定（2026-09-25 新标准）\n\n"
                  f"- (good+so-so) 保留 {100*A_res['combined_keep']:.1f}%（目标 ≥90%）\n"
                  f"- good 保留 {100*A_res['good_keep']:.1f}%（目标 ≥72%）\n"
                  f"- 坏率 {100*A_res['bad_rate']:.1f}% + 负样本检出率 "
                  f"{100*summary.get('C_negative_fp', 0):.1f}% = **{100*total_bad:.1f}%**"
                  f"（目标 ≤{TARGET_BAD_TOTAL:.0%}）\n"
                  f"- **总体：{'✅ 达标' if ok else '❌ 未达标'}**（权重 {a.weights}）\n")
    text = "\n".join(report)
    print(text)
    if a.json:
        Path(a.json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n💾 {a.json}")


if __name__ == "__main__":
    main()
