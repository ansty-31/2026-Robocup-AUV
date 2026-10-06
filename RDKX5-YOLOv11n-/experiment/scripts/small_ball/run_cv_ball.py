#!/usr/bin/env python3
"""夹取小球 · 方案 A：把调好的传统 CV 检测器跑满整个目录，出 JSON/CSV + 联系表。

含两件调参阶段没有的东西：
  1. **跨帧轨迹一致性**（`--temporal link`）：真实小球的检出在时间上是连续的（相邻帧圆心位移
     ≲ 1.2×r），水面焦散的误检不会形成轨迹。所以先按帧号顺序把检出串成 track，丢掉长度
     < `--min-track` 的孤立检出 —— 这让我们能在颜色/几何上放松阈值（保住被遮挡的小球），
     同时把误检按下去。这是**唯一**用到"视频是连续的"这条信息的步骤。
  2. **排序出表**：按帧序（看轨迹连续性）/ 按半径（看尺度覆盖）/ 按分数 三种 sheet。

用法：
    /home/ansty/anaconda3/envs/yolov8/bin/python experiment/scripts/small_ball/run_cv_ball.py \
        --src data/derived/small_ball_selected_700 \
        --params-json output/preview/small_ball/cv_tune.json \
        --set cov_min=0.42 --temporal link --min-track 3 \
        --out-json output/preview/small_ball/cv_ball.json \
        --sheet-dir output/preview/small_ball/sheets_cv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace, fields as dc_fields
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import make_sheet  # noqa: E402
from cv_red_ball import Detection, Params, detect  # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def load_params(params_json: str | None, overrides: list[str]) -> Params:
    p = Params()
    if params_json and Path(params_json).exists():
        cfg = json.loads(Path(params_json).read_text(encoding="utf-8"))
        for k, v in cfg.get("best_params", {}).items():
            if k in {f.name for f in dc_fields(Params)}:
                setattr(p, k, v)
    for kv in overrides:
        k, v = kv.split("=", 1)
        cur = getattr(p, k)
        if isinstance(cur, bool):
            v2 = v.lower() in ("1", "true", "yes")
        elif isinstance(cur, int):
            v2 = int(v)
        elif isinstance(cur, float):
            v2 = float(v)
        else:
            v2 = v
        setattr(p, k, v2)
    return p


def link_tracks(per_frame: list[tuple[int, list[Detection]]], max_gap: int, min_len: int,
                factor: float = 1.2, pad: float = 0.0):
    """把逐帧检出串成轨迹：相邻帧（帧号间隔 ≤ max_gap）圆心距 ≤ 1.2×max(r) 视为同一球。
    返回 {frame_idx: [保留的 Detection]} 与每条轨迹的长度。"""
    tracks: list[dict] = []          # {"pts": [(fi, det)], "last": fi}
    for fi, dets in per_frame:
        for d in dets:
            hit = None
            for t in tracks:
                if fi - t["last"] > max_gap:
                    continue
                _, pd = t["pts"][-1]
                if np.hypot(d.cx - pd.cx, d.cy - pd.cy) <= factor * max(d.r, pd.r) + pad:
                    hit = t
                    break
            if hit is None:
                tracks.append({"pts": [(fi, d)], "last": fi})
            else:
                hit["pts"].append((fi, d))
                hit["last"] = fi
    kept: dict[int, list[Detection]] = {}
    lens = []
    for t in tracks:
        lens.append(len(t["pts"]))
        if len(t["pts"]) >= min_len:
            for fi, d in t["pts"]:
                kept.setdefault(fi, []).append(d)
    return kept, lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--params-json", default=None)
    ap.add_argument("--set", dest="sets", action="append", default=[],
                    help="覆盖单个参数，如 --set cov_min=0.42（可多次）")
    ap.add_argument("--temporal", choices=["none", "link"], default="link")
    ap.add_argument("--min-track", type=int, default=2)
    ap.add_argument("--link-factor", type=float, default=2.5,
                    help="跨帧链接的圆心距上限 = factor×max(r) + pad（球被夹爪带着走得快，1.2 会断链）")
    ap.add_argument("--link-pad", type=float, default=20.0)
    ap.add_argument("--max-gap", type=int, default=6, help="相邻帧的帧号差上限（本数据集帧号连续）")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--sheet-dir", default=None)
    ap.add_argument("--sheet-per", type=int, default=20)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--sheet-sort", default="time", choices=["time", "radius", "score", "miss"])
    ap.add_argument("--max-sheets", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.src)
    files = sorted(p.name for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
    p = load_params(args.params_json, args.sets)
    print(f"参数: {p}")

    per_frame, times = [], []
    for i, f in enumerate(files):
        img = cv2.imread(str(src / f))
        if img is None:
            print(f"⚠️ 读不出 {f}")
            continue
        t0 = time.time()
        dets = detect(img, p)
        times.append((time.time() - t0) * 1000)
        per_frame.append((i, f, dets))
        if (i + 1) % 200 == 0:
            print(f"  … {i+1}/{len(files)}")

    kept = {}
    track_lens = []
    if args.temporal == "link":
        kept, track_lens = link_tracks([(i, d) for i, _, d in per_frame], args.max_gap,
                                      args.min_track, args.link_factor, args.link_pad)

    records = []
    n_raw = n_kept = 0
    for i, f, dets in per_frame:
        keep = kept.get(i, []) if args.temporal == "link" else dets
        if dets:
            n_raw += 1
        if keep:
            n_kept += 1
        records.append({"file": f, "frame": int(f[6:12]), "n_raw": len(dets),
                        "n_keep": len(keep),
                        "det": [d.as_dict() for d in dets],
                        "keep": [[d.cx, d.cy, d.r] for d in keep]})

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"src": str(src), "params": p.__dict__,
                               "temporal": args.temporal, "min_track": args.min_track,
                               "link_factor": args.link_factor, "link_pad": args.link_pad,
                               "n_raw_frames": n_raw, "n_keep_frames": n_kept,
                               "track_lens": track_lens,
                               "ms_median": float(np.median(times)),
                               "fps_cpu": float(1000.0 / np.median(times)),
                               "records": records},
                              ensure_ascii=False, indent=1), encoding="utf-8")

    csvp = out.with_suffix(".csv")
    with open(csvp, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["file", "frame", "n_raw", "n_keep", "cx", "cy", "r", "score",
                    "area", "circ", "cov", "arc_deg", "border_frac"])
        for rec, (_, _, dets) in zip(records, per_frame):
            keep = rec["keep"]
            if not keep:
                w.writerow([rec["file"], rec["frame"], rec["n_raw"], 0] + [""] * 9)
                continue
            for k, (cx, cy, r) in enumerate(keep):
                d = min(dets, key=lambda d: abs(d.r - r))
                w.writerow([rec["file"], rec["frame"], rec["n_raw"], rec["n_keep"],
                            f"{cx:.1f}", f"{cy:.1f}", f"{r:.1f}", f"{d.score:.3f}",
                            d.area, f"{d.circ:.3f}", f"{d.cov:.3f}", f"{d.arc_deg:.0f}",
                            f"{d.border_frac:.3f}"])

    rs = np.array([d.r for _, _, dets in per_frame for d in dets] + [0])
    print(f"✅ JSON: {out}\n✅ CSV : {csvp}")
    print(f"   有检出(raw): {n_raw}/{len(per_frame)} = {n_raw/len(per_frame):.1%}"
          f"  轨迹过滤后: {n_kept}/{len(per_frame)} = {n_kept/len(per_frame):.1%}")
    if track_lens:
        print(f"   轨迹数={len(track_lens)}  长度≥{args.min_track} 的轨迹="
              f"{sum(1 for x in track_lens if x >= args.min_track)}")
    print(f"   半径分位 p5/p50/p95 = {np.percentile(rs[rs>0],[5,50,95]).round(1) if (rs>0).any() else 'n/a'}")
    print(f"   单帧耗时中位 {np.median(times):.1f} ms → {1000/np.median(times):.1f} FPS (CPU, 1280x720)")

    # ---------------- sheets ----------------
    if args.sheet_dir:
        sdir = Path(args.sheet_dir)
        sdir.mkdir(parents=True, exist_ok=True)
        items = list(zip(per_frame, records))
        if args.sheet_sort == "radius":
            items.sort(key=lambda ir: max([d.r for d in ir[1]["keep"]] or [0]) or 1e9)
        elif args.sheet_sort == "score":
            items.sort(key=lambda ir: -max([d["score"] for d in ir[0][2]] or [0]))
        elif args.sheet_sort == "miss":
            items.sort(key=lambda ir: (1 if ir[1]["n_keep"] else 0,
                                       max([d.r for d in ir[1]["keep"]] or [0])))
        per, cols = args.sheet_per, args.cols
        n_sheets = min((len(items) + per - 1) // per, args.max_sheets or 10 ** 9)
        for s in range(n_sheets):
            entries = []
            for k, ((i, f, dets), rec) in enumerate(items[s * per:(s + 1) * per]):
                img = cv2.imread(str(src / f))
                keep = rec["keep"]
                for d in dets:
                    isk = any(abs(d.cx - c) < 1 and abs(d.cy - y) < 1 for c, y, _ in keep)
                    cv2.circle(img, (int(d.cx), int(d.cy)), int(d.r),
                               (0, 255, 0) if isk else (120, 120, 120), 2 if isk else 1)
                    cv2.drawMarker(img, (int(d.cx), int(d.cy)), (0, 255, 255),
                                   cv2.MARKER_CROSS, 12, 2)
                caps = [f"{s*per+k+1}. f{f[6:12]}  n={len(keep)}"]
                if keep:
                    c0 = max(keep, key=lambda c: c[2])
                    caps.append(f"r={c0[2]:.0f}px c=({c0[0]:.0f},{c0[1]:.0f})")
                else:
                    caps.append(f"NO DET (raw={rec['n_raw']})")
                entries.append((img, caps, bool(keep)))
            make_sheet(entries, sdir / f"sheet_{s+1:02d}.jpg", cols=cols,
                       title=f"Plan A - CV red-ball (dom>={p.dom_min} rel>={p.rel_min} "
                             f"cov>={p.cov_min} area>={p.min_area}) sort={args.sheet_sort} "
                             f"temporal={args.temporal} - {s+1}/{n_sheets}")
        print(f"✅ sheets: {n_sheets} 张 → {sdir}/ (每张 {per} 图)")


if __name__ == "__main__":
    main()
