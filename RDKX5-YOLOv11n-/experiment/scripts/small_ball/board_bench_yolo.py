#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""板端基准：撞球权重（yolo11n_detect_bayese_640x640_nv12.bin）走**板端实机 A 域链路**跑满 700 帧。

**自包含**（2026-10-06 改）：早先它 import 板端工程的 `base.settings` / `common.preprocess` /
`common.detector`，但那些模块所在的**旧树 `/home/sunrise/AUV` 已于 2026-10-06 归档删除**
（见 cleanup_record/reports/BOARD_Adomain_retire_20261006.md），所以在用副本的模块路径也改成了
`common/vision/preprocess.py`。为了让脚本在"板端工程怎么挪都能跑"，现在把解码函数原样搬进来，
`--chain none-stretch|none-letterbox` 时**完全不依赖板端工程**；只有 `--chain A`（已废弃）才 import
在用副本的 `ModelPreprocessor`。复刻的是 `HbmRuntimeDetector._infer` 的每一步，并在每步之间打点：

    A 域：remap@720p(老内参) → resize 640（拉伸）→ enhance(WB + CLAHE(0.5) + gamma 0.85)
    → bgr_to_packed_nv12_fast → hbm_runtime model.run → decode_yolo11_split

用法（板端）：
    cd /home/sunrise/AUV && python3 /home/sunrise/small_ball_test/board_bench_yolo.py \
        --src /home/sunrise/small_ball_test/raw \
        --out /home/sunrise/small_ball_test/board_yolo.json \
        --save640 /home/sunrise/small_ball_test/adomain640 --threads 0 --conf 0.05
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

BOARD = "/home/sunrise/Desktop/AUV_New"   # 在用副本（旧树 /home/sunrise/AUV 已删）

import cv2                            # noqa: E402
import numpy as np                    # noqa: E402


# --- 板端 nv12 转换（与在用副本 common/vision/preprocess.py 的 cv2 版逐字一致） ---
def bgr_to_packed_nv12_fast(bgr, out_w, out_h):
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420).reshape(-1)
    n = out_w * out_h
    uv = np.empty(n // 2, dtype=np.uint8)
    uv[0::2] = yuv[n:n + n // 4]
    uv[1::2] = yuv[n + n // 4:n + n // 2]
    return np.concatenate([yuv[:n], uv])


def bgr_to_packed_nv12(bgr, out_w, out_h):
    f = bgr.astype(np.float32)
    y = np.clip(0.299 * f[..., 2] + 0.587 * f[..., 1] + 0.114 * f[..., 0], 0, 255).astype(np.uint8)
    q = f[0::2, 0::2] + f[0::2, 1::2] + f[1::2, 0::2] + f[1::2, 1::2]
    u = np.clip(-0.169 * q[..., 2] - 0.331 * q[..., 1] + 0.5 * q[..., 0] + 128, 0, 255).astype(np.uint8)
    v = np.clip(0.5 * q[..., 2] - 0.419 * q[..., 1] - 0.081 * q[..., 0] + 128, 0, 255).astype(np.uint8)
    uv = np.empty((out_h // 2, out_w), dtype=np.uint8)
    uv[:, 0::2] = u
    uv[:, 1::2] = v
    return np.concatenate([y.reshape(-1), uv.reshape(-1)])


# --- 解码（原样来自在用副本 common/vision/detector.py 的 decode_yolo11_split） ---
def _nms(boxes, scores, iou_th):
    keep = []
    order = scores.argsort()[::-1]
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        x1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        y1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        x2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        y2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        a1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a2 = ((boxes[order[1:], 2] - boxes[order[1:], 0]) *
              (boxes[order[1:], 3] - boxes[order[1:], 1]))
        iou = inter / (a1 + a2 - inter + 1e-9)
        order = order[1:][iou <= iou_th]
    return keep


def decode_yolo11_split(outputs, labels, frame_w, frame_h, conf=0.25, iou=0.45,
                        input_w=640, input_h=640, reg_max=16):
    """X5 split-head 解码：每尺度 reg 64ch(DFL logits) + cls nc ch(logits)。"""
    nc = len(labels)
    bins = np.arange(reg_max, dtype=np.float32)
    grids = {}
    for arr in outputs.values():
        a = np.asarray(arr)
        if a.ndim != 4 or a.shape[0] != 1:
            continue
        grids.setdefault(a.shape[1], {})[a.shape[3]] = a[0]
    allb, alls, allc = [], [], []
    for g, tens in grids.items():
        if 64 not in tens or nc not in tens:
            continue
        stride = float(input_w) / g
        reg = tens[64].reshape(g * g, 4, reg_max)
        cls = tens[nc].reshape(g * g, nc)
        pr = 1.0 / (1.0 + np.exp(-cls.astype(np.float64)))
        score = pr.max(axis=1)
        idx = np.where(score >= conf)[0]
        if not idx.size:
            continue
        r = reg[idx].astype(np.float64)
        e = np.exp(r - r.max(axis=2, keepdims=True))
        e /= e.sum(axis=2, keepdims=True)
        dist = (e * bins).sum(axis=2)
        ys, xs = np.divmod(idx, g)
        cx = (xs.astype(np.float32) + 0.5) * stride
        cy = (ys.astype(np.float32) + 0.5) * stride
        allb.append(np.stack([cx - dist[:, 0] * stride, cy - dist[:, 1] * stride,
                              cx + dist[:, 2] * stride, cy + dist[:, 3] * stride], axis=1))
        alls.append(score[idx]); allc.append(pr[idx].argmax(axis=1))
    if not allb:
        return []
    boxes = np.concatenate(allb); scores = np.concatenate(alls); clss = np.concatenate(allc)
    sx, sy = frame_w / input_w, frame_h / input_h
    boxes = np.stack([np.clip(boxes[:, 0] * sx, 0, frame_w), np.clip(boxes[:, 1] * sy, 0, frame_h),
                      np.clip(boxes[:, 2] * sx, 0, frame_w), np.clip(boxes[:, 3] * sy, 0, frame_h)], axis=1)
    keep = []
    for c in range(nc):
        b = np.where(clss == c)[0]
        if b.size:
            keep.extend(b[_nms(boxes[b], scores[b], iou)])


    class _D:
        __slots__ = ("kind", "score", "x", "y", "w", "h")

        def __init__(self, kind, score, x, y, w, h):
            self.kind, self.score = kind, float(score)
            self.x, self.y, self.w, self.h = int(x), int(y), int(w), int(h)

    out = [_D(labels[int(clss[i])], scores[i], boxes[i, 0], boxes[i, 1],
              boxes[i, 2] - boxes[i, 0], boxes[i, 3] - boxes[i, 1]) for i in keep]
    out.sort(key=lambda d: d.score, reverse=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save640", default=None)
    ap.add_argument("--model", default="/home/sunrise/Desktop/AUV_New/models/"
                                       "yolo11n_detect_bayese_640x640_nv12.bin")
    ap.add_argument("--labels", default="blue_ball,gate,red_ball")
    ap.add_argument("--threads", type=int, default=0, help="0=cv2 默认；>0 = setNumThreads")
    ap.add_argument("--conf", type=float, default=0.05, help="记录下限（板端工作点是 0.5）")
    ap.add_argument("--chain", default="A", choices=["A", "none-stretch", "none-letterbox"],
                    help="A=板端原链路(remap@720p→resize→enhance)；"
                         "none-stretch=只 cv2.resize 到 640x640（后续训练口径）；"
                         "none-letterbox=640 letterbox 补边 114（PC 侧那份对照）")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.threads > 0:
        cv2.setNumThreads(args.threads)
    size = 640
    labels = [s for s in args.labels.split(",") if s]
    if args.chain == "A":
        sys.path.insert(0, BOARD)
        os.chdir(BOARD)
    model_path = args.model
    print(f"[bench] model={model_path}")
    print(f"[bench] labels={labels} size={size} model={model_path}")
    print(f"[bench] cv2_threads={cv2.getNumThreads()} chain={args.chain}")

    pre = None
    if args.chain == "A":
        from common.vision.preprocess import ModelPreprocessor   # 在用副本的新路径
        import base.settings as S
        pre = ModelPreprocessor(calib_path=S.vision.camera.front.calibration)
    elif args.chain == "none-letterbox":
        import cv2 as _cv2

        def _lb(frame, size=size):
            h, w = frame.shape[:2]
            r = min(size / h, size / w)
            nw, nh = int(round(w * r)), int(round(h * r))
            interp = _cv2.INTER_LINEAR if r > 1 else _cv2.INTER_AREA
            rz = _cv2.resize(frame, (nw, nh), interpolation=interp)
            dw, dh = (size - nw) / 2, (size - nh) / 2
            top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
            bot, right = int(round(dh + 0.1)), int(round(dw + 0.1))
            out = _cv2.copyMakeBorder(rz, top, bot, left, right,
                                      _cv2.BORDER_CONSTANT, value=(114,) * 3)
            return out, r, left, top
    print(f"[bench] 实际输入链路: {args.chain}")
    import hbm_runtime
    model = hbm_runtime.HB_HBMRuntime(model_path)
    key = "images"
    for attr in ("input_names", "input_name"):
        v = getattr(model, attr, None)
        if isinstance(v, dict):
            for val in v.values():
                key = str(val[0]) if isinstance(val, (list, tuple)) else str(val)
                break
        elif isinstance(v, (list, tuple)) and v:
            key = str(v[0])
        elif isinstance(v, str) and v:
            key = v

    src = Path(args.src)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if args.limit:
        files = files[:args.limit]
    if args.save640:
        Path(args.save640).mkdir(parents=True, exist_ok=True)

    recs, t_read, t_pre, t_nv12, t_inf, t_dec, t_all = [], [], [], [], [], [], []
    for i, f in enumerate(files):
        t0 = time.perf_counter()
        frame = cv2.imread(str(f))
        t1 = time.perf_counter()
        if args.chain == "A":
            square = pre.process(frame)                  # A 域：remap→resize→enhance
            lb = (1.0, 0, 0)
        elif args.chain == "none-stretch":
            square = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
            lb = (1.0, 0, 0)                             # 拉伸：x640 = x*(640/w)
        else:
            square, lb_r, lb_px, lb_py = _lb(frame, size)
        t2 = time.perf_counter()
        nv12 = bgr_to_packed_nv12_fast(square, size, size)
        t3 = time.perf_counter()
        out = model.run({key: nv12})
        if isinstance(out, dict) and len(out) == 1:
            out = next(iter(out.values()))
        t4 = time.perf_counter()
        h, w = frame.shape[:2]
        if args.chain == "none-letterbox":
            dets = decode_yolo11_split(out, labels, size, size,   # 先在 640 空间解码
                                       conf=args.conf, input_w=size, input_h=size)
            r_, px_, py_ = lb_r, lb_px, lb_py
            for d in dets:                                        # 再映射回原图坐标
                d.x = int(round((d.x - px_) / r_)); d.y = int(round((d.y - py_) / r_))
                d.w = int(round(d.w / r_));         d.h = int(round(d.h / r_))
        else:
            dets = decode_yolo11_split(out, labels, w, h,
                                       conf=args.conf, input_w=size, input_h=size)
        t5 = time.perf_counter()
        if args.save640 and i < 700:
            cv2.imwrite(str(Path(args.save640) / f.name), square,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        recs.append({"file": f.name,
                     "det": [{"kind": d.kind, "score": round(d.score, 4),
                              "xyxy": [d.x, d.y, d.x + d.w, d.y + d.h]} for d in dets]})
        t_read.append((t1 - t0) * 1e3); t_pre.append((t2 - t1) * 1e3)
        t_nv12.append((t3 - t2) * 1e3); t_inf.append((t4 - t3) * 1e3)
        t_dec.append((t5 - t4) * 1e3); t_all.append((t5 - t0) * 1e3)
        if (i + 1) % 100 == 0:
            print(f"  … {i+1}/{len(files)}  e2e {np.median(t_all[-100:]):.1f} ms/frame")

    def st(x, skip=10):
        a = np.array(x[skip:])
        return {"p50": float(np.median(a)), "p90": float(np.percentile(a, 90)),
                "mean": float(a.mean())}

    stats = {"read": st(t_read), "preprocess_A": st(t_pre), "nv12": st(t_nv12),
             "bpu_run": st(t_inf), "decode": st(t_dec), "end2end": st(t_all)}
    stats["fps_e2e_p50"] = 1000.0 / stats["end2end"]["p50"]
    print("[bench] 各阶段 p50 ms:", {k: round(v["p50"], 2) for k, v in stats.items()
                                     if isinstance(v, dict)})
    print(f"[bench] 端到端 p50 = {stats['end2end']['p50']:.2f} ms → "
          f"{stats['fps_e2e_p50']:.1f} FPS")
    hit = sum(1 for r in recs if any(d["kind"] == "red_ball" and d["score"] >= 0.5
                                     for d in r["det"]))
    hit25 = sum(1 for r in recs if any(d["kind"] == "red_ball" and d["score"] >= 0.25
                                       for d in r["det"]))
    print(f"[bench] red_ball 命中: conf>=0.5 → {hit}/{len(recs)}  "
          f"conf>=0.25 → {hit25}/{len(recs)}")

    Path(args.out).write_text(json.dumps(
        {"model": model_path, "labels": labels, "chain": args.chain,
         "cv2_threads": cv2.getNumThreads(), "stats_ms": stats,
         "records": recs}, ensure_ascii=False), encoding="utf-8")
    print(f"[bench] JSON → {args.out}")


if __name__ == "__main__":
    main()
