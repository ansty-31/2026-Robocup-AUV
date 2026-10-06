#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""板端基准：传统 CV 红球检测（方案 A）在 RDK X5 CPU 上的速度与结果。

与 PC 侧**同一份算法**（`cv_red_ball.py` 原样拷过来，参数用 PC 调好的那套），
输入同样是 1280×720 原图、**不做任何预处理**（CV 靠颜色，不需要板端的 enhance/去畸变）。

用法（板端）：
    python3 /home/sunrise/small_ball_test/board_bench_cv.py \
        --src /home/sunrise/small_ball_test/raw \
        --out /home/sunrise/small_ball_test/board_cv_t1.json --threads 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_red_ball import Params, detect          # noqa: E402

# PC 侧扫出的最终参数（见 output/preview/small_ball/README.md §2）
PC_PARAMS = dict(dom_min=30, use_rel=True, rel_min=0.30, s_min=40, v_min=60,
                 close_k=9, open_k=5, fill_hole_frac=0.65, min_area=30,
                 circ_min=0.30, hull_fill_min=0.55, cov_min=0.35,
                 min_arc_deg=150.0, r_min=25.0, score_min=0.0, fast_mask=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=1, help="整段重复次数（取稳定值）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ref-mask", action="store_true",
                    help="用 numpy 参考实现（慢路径）做 A/B")
    args = ap.parse_args()

    cv2.setNumThreads(args.threads)
    kw = dict(PC_PARAMS)
    if args.ref_mask:
        kw["fast_mask"] = False
    p = Params(**kw)
    print(f"[cv-bench] cv2_threads={cv2.getNumThreads()} params={PC_PARAMS}")

    src = Path(args.src)
    files = sorted(x for x in src.iterdir() if x.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if args.limit:
        files = files[:args.limit]

    recs, t_all, t_det = [], [], []
    for rep in range(args.repeat):
        for i, f in enumerate(files):
            t0 = time.perf_counter()
            img = cv2.imread(str(f))
            t1 = time.perf_counter()
            dets = detect(img, p)
            t2 = time.perf_counter()
            if rep == args.repeat - 1:
                t_all.append((t2 - t0) * 1e3)      # 含读盘
                t_det.append((t2 - t1) * 1e3)      # **只算算法**（实机帧来自相机缓冲，无读盘）
                recs.append({"file": f.name,
                             "det": [{"cx": round(d.cx, 1), "cy": round(d.cy, 1),
                                      "r": round(d.r, 1), "score": round(d.score, 3),
                                      "cov": round(d.cov, 2)} for d in dets]})
        print(f"  … rep {rep+1}/{args.repeat}  done ({len(files)} frames)")

    a = np.array(t_all[10:])
    d = np.array(t_det[10:])
    stats = {"p50": float(np.median(a)), "p90": float(np.percentile(a, 90)),
             "mean": float(a.mean()), "fps_p50": float(1000.0 / np.median(a)),
             "det_only_p50": float(np.median(d)),
             "det_only_p90": float(np.percentile(d, 90)),
             "fps_det_only": float(1000.0 / np.median(d))}
    hit = sum(1 for r in recs if r["det"])
    print(f"[cv-bench] 含读盘 p50={stats['p50']:.2f} ms → {stats['fps_p50']:.1f} FPS")
    print(f"[cv-bench] 仅算法 p50={stats['det_only_p50']:.2f} ms "
          f"p90={stats['det_only_p90']:.2f} ms → {stats['fps_det_only']:.1f} FPS")
    print(f"[cv-bench] 有检出的帧: {hit}/{len(recs)}")
    Path(args.out).write_text(json.dumps(
        {"params": p.__dict__, "cv2_threads": cv2.getNumThreads(),
         "stats_ms": stats, "n_hit": hit, "n_frames": len(recs),
         "records": recs}, ensure_ascii=False), encoding="utf-8")
    print(f"[cv-bench] JSON → {args.out}")


if __name__ == "__main__":
    main()
