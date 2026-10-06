#!/usr/bin/env python3
"""夹取小球 · 方案 A 的参数扫描（坐标下降），**用可量化的参照物而不是手感**。

参照口径（诚实说明：这不是人工标注 GT）：
  * `ref` 集 = YOLO `red_ball` conf ≥ `--ref-conf`(默认 0.5) 的图 —— 中/大球上 YOLO 很可靠，
    当作"球确实在那里、位置可信"的参照，用来量 **CV 的召回与定位精度**（匹配：圆心距 ≤ 0.6×r_ref
    且半径落在 [0.5, 1.8]×r_ref）。
  * `FP 代理` = CV 有检出、而 YOLO 连 conf ≥ 0.02 的 red_ball 输出都没有的图 → 疑似误检，
    数量报出来，最后**靠 sheet 肉眼复核**（脚本不替你做这个判断）。

扫描策略：单维坐标下降两轮（每维候选由 `--grid` 决定），目标
`J = R_ref + 0.5×IoU_med − 0.002×N_multi − 0.02×(FP_proxy 比例)`，并在结果表里把各项分开列，
方便你自己权衡（例如要"宁可多检不可漏"就把 N_hit 拉高、接受 FP 上升）。

用法：
    /home/ansty/anaconda3/envs/yolov8/bin/python experiment/scripts/small_ball/tune_cv_ball.py \
        --src data/derived/small_ball_selected_700 --sample 200 \
        --yolo-json output/preview/small_ball/tables/yolo_ball.json \
        --out output/preview/small_ball/tables/cv_tune.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_red_ball import Params, detect  # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
FIELDS = ["dom_min", "rel_min", "s_min", "v_min", "close_k", "open_k",
          "min_area", "circ_min", "cov_min", "min_arc_deg", "fill_hole_frac"]
GRID = {
    "dom_min": [30, 40, 55, 70, 90],
    "rel_min": [0.15, 0.25, 0.30, 0.40],
    "s_min": [40, 60, 100, 140],
    "v_min": [25, 40, 60],
    "close_k": [3, 5, 9, 13],
    "open_k": [0, 3, 5],
    "min_area": [20, 50, 100, 200],
    "circ_min": [0.20, 0.30, 0.40, 0.50],
    "cov_min": [0.30, 0.42, 0.55, 0.70],
    "min_arc_deg": [100.0, 150.0, 200.0, 260.0],
    "fill_hole_frac": [0.35, 0.65, 0.85],
}


def build_ref(yolo_json: Path, names: set, ref_conf: float):
    """返回 {file: (cx, cy, r, conf)}（取该图最大 red_ball 框），以及每图的 max conf。"""
    d = json.loads(yolo_json.read_text(encoding="utf-8"))
    ref, maxconf = {}, {}
    for rec in d["records"]:
        if rec["file"] not in names:
            continue
        reds = [x for x in rec["det"] if x["name"] == "red_ball"]
        maxconf[rec["file"]] = max([x["conf"] for x in reds], default=0.0)
        if not reds:
            continue
        b = max(reds, key=lambda x: (x["xyxy"][2] - x["xyxy"][0]) * (x["xyxy"][3] - x["xyxy"][1]))
        x1, y1, x2, y2 = b["xyxy"]
        ref[rec["file"]] = ((x1 + x2) / 2, (y1 + y2) / 2,
                            (x2 - x1 + y2 - y1) / 4, b["conf"])
    return ref, maxconf


def evaluate(imgs, p: Params, ref, maxconf, ref_conf: float):
    """返回指标 dict + 逐图摘要（给后续 sheet / 复核用）"""
    n_ref = n_hit_ref = n_hit = n_multi = 0
    ious, times, per = [], [], []
    fp_frames = []
    for name, img in imgs:
        t0 = time.time()
        dets = detect(img, p)
        times.append((time.time() - t0) * 1000)
        per.append((name, dets))
        if dets:
            n_hit += 1
            if len(dets) > 1:
                n_multi += 1
        if name in ref and ref[name][3] >= ref_conf:
            n_ref += 1
            cx, cy, r, _ = ref[name]
            ok = False
            for d in dets:
                if np.hypot(d.cx - cx, d.cy - cy) <= 0.6 * r and 0.5 * r <= d.r <= 1.8 * r:
                    ok = True
            if ok:
                n_hit_ref += 1
                best = min(dets, key=lambda d: np.hypot(d.cx - cx, d.cy - cy))
                # 圆(按外接方框) 与 YOLO 框的 IoU
                bx1, by1, bx2, by2 = best.cx - best.r, best.cy - best.r, best.cx + best.r, best.cy + best.r
                rx1, ry1, rx2, ry2 = cx - r, cy - r, cx + r, cy + r
                ix = max(0, min(bx2, rx2) - max(bx1, rx1))
                iy = max(0, min(by2, ry2) - max(by1, ry1))
                inter = ix * iy
                uni = (bx2 - bx1) * (by2 - by1) + (rx2 - rx1) * (ry2 - ry1) - inter
                if uni > 0:
                    ious.append(inter / uni)
        if dets and maxconf.get(name, 0.0) <= 0.02:
            fp_frames.append(name)
    r_ref = n_hit_ref / n_ref if n_ref else 0.0
    iou_med = float(np.median(ious)) if ious else 0.0
    n = len(imgs)
    fp_rate = len(fp_frames) / n
    J = r_ref + 0.5 * iou_med - 0.002 * n_multi - 0.02 * fp_rate
    return {"R_ref": r_ref, "n_ref": n_ref, "IoU_med": iou_med, "N_hit": n_hit,
            "hit_rate": n_hit / n, "N_multi": n_multi, "FP_proxy": len(fp_frames),
            "FP_rate": fp_rate, "ms": float(np.median(times)), "J": J,
            "fp_frames": fp_frames, "per": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--yolo-json", required=True)
    ap.add_argument("--sample", type=int, default=200, help="调参用子集大小（等间隔抽，保证覆盖全序列）")
    ap.add_argument("--ref-conf", type=float, default=0.5)
    ap.add_argument("--out", default="output/preview/small_ball/tables/cv_tune.csv")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--fields", default=",".join(FIELDS))
    args = ap.parse_args()

    src = Path(args.src)
    files = sorted(p.name for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
    if args.sample and args.sample < len(files):
        idx = np.linspace(0, len(files) - 1, args.sample).round().astype(int)
        files = [files[i] for i in idx]
    print(f"调参子集: {len(files)} 张（等间隔覆盖全序列）")
    imgs = [(f, cv2.imread(str(src / f))) for f in files]
    ref, maxconf = build_ref(Path(args.yolo_json), set(files), args.ref_conf)
    print(f"参照集（YOLO red conf≥{args.ref_conf}）: {len(ref)} 张")

    p = Params()
    fields = [f for f in args.fields.split(",") if f]
    rows = []
    best = evaluate(imgs, p, ref, maxconf, args.ref_conf)
    rows.append((dict(p.__dict__), best))
    print(f"[init] J={best['J']:.4f} R_ref={best['R_ref']:.3f} IoU={best['IoU_med']:.3f} "
          f"hit={best['N_hit']}/{len(files)} multi={best['N_multi']} FP={best['FP_proxy']} "
          f"{best['ms']:.0f}ms")

    for rnd in range(args.rounds):
        improved = False
        for f in fields:
            cur = getattr(p, f)
            base_J = best["J"]
            best_val, best_res = cur, best
            for cand in GRID.get(f, [cur]):
                if cand == cur:
                    continue
                trial = replace(p, **{f: cand})
                res = evaluate(imgs, trial, ref, maxconf, args.ref_conf)
                rows.append((dict(trial.__dict__), res))
                if res["J"] > best_res["J"]:
                    best_res, best_val = res, cand
            if best_res["J"] > base_J + 1e-9:
                setattr(p, f, best_val)
                best = best_res
                improved = True
                print(f"[r{rnd+1}] {f}: {cur} → {best_val}   J={best['J']:.4f} "
                      f"R_ref={best['R_ref']:.3f} IoU={best['IoU_med']:.3f} "
                      f"hit={best['N_hit']}/{len(files)} FP={best['FP_proxy']}")
        if not improved:
            print(f"[r{rnd+1}] 无改进，停止")
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["set"] + list(FIELDS) + ["J", "R_ref", "n_ref", "IoU_med", "N_hit",
                                             "hit_rate", "N_multi", "FP_proxy", "FP_rate", "ms"])
        for cfgv, res in rows:
            w.writerow(["cfg"] + [cfgv[f] for f in FIELDS] +
                       [f"{res['J']:.4f}", f"{res['R_ref']:.3f}", res["n_ref"],
                        f"{res['IoU_med']:.3f}", res["N_hit"], f"{res['hit_rate']:.3f}",
                        res["N_multi"], res["FP_proxy"], f"{res['FP_rate']:.3f}", f"{res['ms']:.1f}"])
        bestcfg = dict(p.__dict__)
        w.writerow(["BEST"] + [bestcfg.get(f, "") for f in FIELDS] +
                   [f"{best['J']:.4f}", f"{best['R_ref']:.3f}", best["n_ref"],
                    f"{best['IoU_med']:.3f}", best["N_hit"], f"{best['hit_rate']:.3f}",
                    best["N_multi"], best["FP_proxy"], f"{best['FP_rate']:.3f}", f"{best['ms']:.1f}"])
    (out.with_suffix(".json")).write_text(json.dumps(
        {"best_params": {k: v for k, v in p.__dict__.items()},
         "best_metrics": {k: v for k, v in best.items() if k not in ("per", "fp_frames")},
         "fp_frames": best["fp_frames"], "sample": len(files)}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\n✅ 最优参数: {p}")
    print(f"✅ 结果表: {out}  (含每一组扫描的 R_ref/IoU/hit/FP)")




if __name__ == "__main__":
    main()
