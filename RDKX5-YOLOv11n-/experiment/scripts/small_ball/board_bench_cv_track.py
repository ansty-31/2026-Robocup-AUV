#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""板端：CV 流式（ROI 跟踪）vs 每帧全图 —— 速度与精度对照。

按**帧号顺序**喂 700 帧（与实机视频同序），统计：
  * 逐帧耗时 p50/p90（含读盘 / 仅算法）
  * 三种模式计数：全图搜索(full) / ROI 跟踪(roi) / 回退(fallback)
  * ROI 模式的稳态单帧耗时（跟踪时的真实帧率）
  * 命中帧数与每帧全图基线的差异
用法：python3 board_bench_cv_track.py --src raw --out board_cv_track.json --threads 4
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_red_ball import Params, detect
from cv_ball_tracker import BallTracker

PC_PARAMS = dict(dom_min=30, use_rel=True, rel_min=0.30, s_min=40, v_min=60,
                 close_k=9, open_k=5, fill_hole_frac=0.65, min_area=30,
                 circ_min=0.30, hull_fill_min=0.55, cov_min=0.35,
                 min_arc_deg=150.0, r_min=25.0, score_min=0.0, fast_mask=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--margin", type=float, default=64.0, help="ROI 在球外每边多留的像素")
    ap.add_argument("--proc-side", type=int, default=0, help=">0: ROI 缩放到该边长再检测（尺度归一化）")
    args = ap.parse_args()
    cv2.setNumThreads(args.threads)
    p = Params(**PC_PARAMS)
    # 帧号顺序（文件名 frame_00XXXX.jpg）—— 必须与视频同序，否则跟踪链断了没有意义
    files = sorted(Path(args.src).glob("*.jpg"), key=lambda f: int(f.stem[6:]))
    tr = BallTracker(p, margin=args.margin, proc_side=args.proc_side)
    per, per_det, modes, recs = [], [], {}, []
    for f in files:
        t0 = time.perf_counter()
        im = cv2.imread(str(f))
        t1 = time.perf_counter()
        d, info = tr.step(im)
        t2 = time.perf_counter()
        per.append((t2 - t0) * 1e3); per_det.append((t2 - t1) * 1e3)
        modes.setdefault(info.mode, []).append((t2 - t1) * 1e3)
        recs.append({"file": f.name, "mode": info.mode, "ms": round((t2 - t1) * 1e3, 2),
                     "roi_side": tr.last_side,
                     "det": [round(v, 1) for v in (d.cx, d.cy, d.r)] if d else None})
        if len(per) % 100 == 0:
            print(f"  … {len(per)}/{len(files)}  p50={np.median(per_det[-100:]):.1f} ms")
    a = np.array(per_det); ra = np.array(per)
    hit = sum(1 for r in recs if r["det"])
    out = {"params": p.__dict__, "threads": args.threads,
           "roi_margin": args.margin, "proc_side": args.proc_side,
           "n_frames": len(recs), "n_hit": hit,
           "stats": {"det_p50": float(np.median(a)), "det_p90": float(np.percentile(a, 90)),
                     "det_fps": float(1000 / np.median(a)),
                     "withread_p50": float(np.median(ra)),
                     "avg_fps": float(1000 / a.mean())},
           "modes": {k: {"n": len(v), "p50": float(np.median(v)),
                         "fps": float(1000 / np.median(v))} for k, v in modes.items()},
           "records": recs}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[track-bench] 帧数 {len(recs)}  命中 {hit}")
    print(f"[track-bench] 逐帧 仅算法 p50={np.median(a):.1f} ms p90={np.percentile(a,90):.1f} ms "
          f"平均帧率={1000/a.mean():.1f} FPS（含读盘 p50={np.median(ra):.1f}）")
    for k, v in sorted(modes.items()):
        print(f"    mode={k:8s} n={len(v):3d}  p50={np.median(v):6.1f} ms  → {1000/np.median(v):6.1f} FPS")
    print(f"[track-bench] JSON → {args.out}")


if __name__ == "__main__":
    main()
