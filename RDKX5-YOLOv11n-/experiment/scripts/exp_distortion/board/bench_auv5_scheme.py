#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_auv5_scheme.py — 板端：新方案（D 链序 + auv5 门权重）的速度与角点实测

只读帧 + 只跑推理，**不打开串口、不驱动船**（不构造 UartController，也不 import main）。

度量口径（与 2026-09-23 那批板端测量保持一致）：
  · preprocess = `ModelPreprocessor.process(frame)`（新序：resize640 → enhance → remap@640）
  · detect     = `GateKeypointBackend.detect(frame)`（含 预处理的再次调用 + NV12 + BPU + decode）
    ⚠️ backend.detect 内部会自己再预处理一次，所以 e2e ≈ preprocess + (NV12+BPU+decode)
  · 每帧重复 --reps 次取**中位**；先跑 --warmup 帧热身（首帧含权重加载/XNN 编译）

角点落在**去畸变 720p 空间**（kpt 回缩放后的域，见 common/preprocess.py 的换序注释），
因此与 GT 比较时把 640 的标签按 (frame_w/640, frame_h/640) 放大即可。
⚠️ 每帧最多 2 个门实例（一图两门），PC 侧比对按框中心最近配对。

用法（板端，工程根下）：
    python3 /tmp/auv5_bench/bench_auv5_scheme.py \
        --manifest /tmp/auv5_bench/manifest.csv --out /tmp/auv5_bench/bench_out.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.getcwd())

import base.settings as S                      # noqa: E402
from common.preprocess import ModelPreprocessor  # noqa: E402
from gate.gate_detector import build_gate_backend  # noqa: E402


def measure(pp, backend, frame, reps):
    tp, td = [], []
    dets = []
    for _ in range(reps):
        t0 = time.perf_counter()
        pp.process(frame)
        t1 = time.perf_counter()
        dets = backend.detect(frame)
        t2 = time.perf_counter()
        tp.append((t1 - t0) * 1e3)
        td.append((t2 - t1) * 1e3)
    return st.median(tp), st.median(td), dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV: name,raw,label")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=3)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.manifest)))
    if not rows:
        print("空 manifest")
        return 1

    conf_thr = float(S.get("vision.gate.keypoint.conf_thr", 0.5))
    pp = ModelPreprocessor(calib_path=S.vision.camera.front.calibration)
    backend = build_gate_backend()
    if backend is None:
        print("[bench] ✗ gate 后端不可用（权重缺失？）")
        return 1

    print("[bench] cfg: undistort=%s clip=%s gains=%s size=%s" %
          (S.vision.image.undistort, S.vision.image.clahe_clip,
           S.vision.image.white_balance_bgr, S.vision.model.input_size))
    print("[bench] gate 权重: %s | fast_nv12=%s | conf_thr=%.2f" %
          (S.get("vision.model.task_models.gate.path"),
           S.get("vision.model.fast_nv12", False), conf_thr))
    print("[bench] 帧数 %d，每帧重复 %d 次，热身 %d 次" % (len(rows), a.reps, a.warmup))

    # --- 热身 ---
    warm = cv2.imread(rows[0]["raw"])
    for _ in range(max(1, a.warmup)):
        backend.detect(warm)

    per_frame, tp_all, td_all = [], [], []
    for i, r in enumerate(rows):
        frame = cv2.imread(r["raw"])
        if frame is None:
            print("  ✗ 读不到 %s" % r["raw"])
            continue
        h, w = frame.shape[:2]
        tp, td, dets = measure(pp, backend, frame, a.reps)
        tp_all.append(tp)
        td_all.append(td)
        best = max(dets, key=lambda d: d.score) if dets else None
        rec = dict(name=r["name"], raw=r["raw"], frame_wh=[w, h],
                   pre_ms=round(tp, 3), detect_ms=round(td, 3),
                   e2e_ms=round(tp + td, 3), n_det=len(dets))
        if best is not None:
            kp = best.kpts if best.kpts is not None else []
            kc = best.kpt_conf if best.kpt_conf is not None else []
            rec.update(score=round(float(best.score), 4),
                       bbox=[best.x, best.y, best.w, best.h],
                       kpts=[[round(float(p[0]), 2), round(float(p[1]), 2)] for p in kp],
                       kpt_conf=[round(float(c), 4) for c in kc])
            rec["n_kpt_vis"] = int(sum(1 for c in kc if float(c) >= conf_thr))
        per_frame.append(rec)
        print("  [%2d/%d] %-52s pre %6.2f  det %6.2f ms  dets=%d  角点达标=%s"
              % (i + 1, len(rows), r["name"][:52], tp, td, len(dets),
                 rec.get("n_kpt_vis", "-")))

    e2e = [f["e2e_ms"] for f in per_frame]
    summary = dict(
        n_frame=len(per_frame), reps=a.reps,
        pre_ms_median=round(st.median(tp_all), 3),
        detect_ms_median=round(st.median(td_all), 3),
        e2e_ms_median=round(st.median(e2e), 3),
        e2e_ms_mean=round(st.mean(e2e), 3),
        fps_median=round(1000.0 / st.median(e2e), 2),
        four_corner_frames=int(sum(1 for f in per_frame if f.get("n_kpt_vis") == 4)),
        conf_thr=conf_thr,
    )
    out = dict(cfg=dict(undistort=bool(S.vision.image.undistort),
                        clahe_clip=float(S.vision.image.clahe_clip),
                        gains=list(S.vision.image.white_balance_bgr),
                        gamma=float(S.vision.image.gamma),
                        input_size=int(S.vision.model.input_size),
                        fast_nv12=bool(S.get("vision.model.fast_nv12", False)),
                        gate_model=str(S.get("vision.model.task_models.gate.path")),
                        calibration=str(S.vision.camera.front.calibration)),
               summary=summary, per_frame=per_frame)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("\n[bench] 汇总: pre 中位 %.2f ms | detect 中位 %.2f ms | e2e 中位 %.2f ms (%.1f FPS) | 四角齐全 %d/%d"
          % (summary["pre_ms_median"], summary["detect_ms_median"],
             summary["e2e_ms_median"], summary["fps_median"],
             summary["four_corner_frames"], summary["n_frame"]))
    print("[bench] 写出 %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
