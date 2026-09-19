#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/check_kpt_decode.py — 门 keypoint 解码约定自检（离线，单图，不动相机/船）

用途：把"角点乱飞"定性到底是 **模型** 还是 **解码约定**。
同一张图跑一次模型，把同一个 cell 的原始 kpt 输出用几种候选公式解出来，
画在同一张图上（不同颜色）并打印数值 —— 哪个公式把点钉在门框角上，就是对的。

    # 板端（有权重 + hbm_runtime）
    python3 tools/check_kpt_decode.py --image log/frames/pv_000040.jpg --out /tmp/decode_check.jpg

候选公式（anchor = 选择到的网格 cell 下标；stride = input_w / g）：
    V0 现在实现 : (raw)                    * stride
    V1 ultralytics: (raw*2 + anchor - 0.5) * stride
    V2 anchor   : (raw   + anchor)         * stride
    V3 anchor-.5: (raw   + anchor - 0.5)   * stride
    V4 已是像素 : (raw)                    * 1        （有些导出把解码烘进模型）
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_ROOT, os.path.join(_ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import base.settings as S                                       # noqa: E402
from common.preprocess import ModelPreprocessor                 # noqa: E402
from common.detector import bgr_to_packed_nv12_fast, bgr_to_packed_nv12  # noqa: E402
from gate.vision.gate_decode import find_model_input, _KPT_PER_PT  # noqa: E402

VARIANTS = ("V0", "V1", "V2", "V3", "V4")
COLORS = {"V0": (0, 0, 255), "V1": (0, 255, 0), "V2": (255, 0, 0),
          "V3": (0, 255, 255), "V4": (255, 0, 255)}


def decode_variant(raw, xs, ys, stride, tag):
    """raw: (kpt_dim, 3) 原始输出；返回 (kpt_dim,2) 输入像素坐标。"""
    xy = raw[:, :2].astype(np.float64)
    ax, ay = float(xs), float(ys)
    if tag == "V0":
        return xy * stride
    if tag == "V1":
        return (xy * 2.0 + np.array([ax - 0.5, ay - 0.5])) * stride
    if tag == "V2":
        return (xy + np.array([ax, ay])) * stride
    if tag == "V3":
        return (xy + np.array([ax - 0.5, ay - 0.5])) * stride
    if tag == "V4":
        return xy
    raise ValueError(tag)


def _find_cell_for_box(grids, nc, kch, pre_size, box_in, score):
    """由真实解码出的框反推 (g, cell)：逐个尺度扫 score>=conf 的 cell，
    用该 cell 自己解出的框与目标框比"中心距离 + 尺寸"，最接近者即来源。"""
    bx = 0.5 * (box_in[0] + box_in[2])
    by = 0.5 * (box_in[1] + box_in[3])
    bw = box_in[2] - box_in[0]
    bh = box_in[3] - box_in[1]
    best = None
    for g, tens in grids.items():
        if 64 not in tens or nc not in tens or kch not in tens:
            continue
        stride = float(pre_size) / g
        cls = tens[nc].reshape(g * g, nc)
        pr = 1.0 / (1.0 + np.exp(-cls.astype(np.float64)))
        sc = pr.max(axis=1)
        cand = np.where(sc >= max(0.25, score - 0.05))[0]
        if not cand.size:
            continue
        reg = tens[64].reshape(g * g, 4, 16).astype(np.float64)
        bins = np.arange(16, dtype=np.float64)
        for i in cand:
            r = reg[i]
            e = np.exp(r - r.max(axis=1, keepdims=True))
            e /= e.sum(axis=1, keepdims=True)
            dist = (e * bins).sum(axis=1)
            ys, xs = divmod(int(i), g)
            cx, cy = (xs + 0.5) * stride, (ys + 0.5) * stride
            x1, y1 = cx - dist[0] * stride, cy - dist[1] * stride
            x2, y2 = cx + dist[2] * stride, cy + dist[3] * stride
            err = (abs(0.5 * (x1 + x2) - bx) + abs(0.5 * (y1 + y2) - by)
                   + abs((x2 - x1) - bw) + abs((y2 - y1) - bh))
            if best is None or err < best[0]:
                best = (err, g, xs, ys, stride, float(sc[i]))
    if best is None or best[0] > 12.0:      # 中心+尺寸合计误差 >12px 认为没匹配上
        return None
    return best[1], best[2], best[3], best[4], best[5]


def _scores(box, pts):
    """(框内占比, 到最近框角的平均距离 px) —— 正确的解码应"4 点落在框的四角"。"""
    x1, y1, x2, y2 = box
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], float)
    inside = np.mean([(x1 - 8 <= px <= x2 + 8) and (y1 - 8 <= py <= y2 + 8)
                      for px, py in pts])
    err = float(np.mean([np.min(np.linalg.norm(corners - p, axis=1)) for p in pts]))
    return float(inside), err


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="/tmp/decode_check.jpg")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--per-cell", action="store_true",
                    help="逐 cell 全局判定（推荐：不依赖 NMS 选了哪个 cell）")
    a = ap.parse_args()

    import cv2
    frame = cv2.imread(a.image)
    if frame is None:
        print("读不到图片:", a.image); sys.exit(2)
    h, w = frame.shape[:2]
    gc = S.vision.model.task_models.gate
    labels = list(gc.get("labels", ["gate"]))
    nc = len(labels)
    kch = _KPT_PER_PT * 4
    print("图片 %dx%d | 权重 %s" % (w, h, os.path.basename(str(gc.path))))

    import hbm_runtime
    from gate.vision.gate_decode import decode_yolo11_kpt
    model = hbm_runtime.HB_HBMRuntime(str(gc.path))
    key = find_model_input(model)
    pre = ModelPreprocessor(calib_path=S.vision.camera.front.calibration)
    square = pre.process(frame)
    nv12 = (bgr_to_packed_nv12_fast if S.get("vision.model.fast_nv12", False)
            else bgr_to_packed_nv12)(square, pre.size, pre.size)
    outs = model.run({key: nv12})
    if isinstance(outs, dict) and len(outs) == 1:
        outs = next(iter(outs.values()))
    if not isinstance(outs, dict):
        outs = {("out%d" % i): o for i, o in enumerate(outs)}

    conf = a.conf if a.conf is not None else S.vision.model.score_threshold
    if a.per_cell:
        per_cell_report(outs, labels, pre.size, conf=max(0.4, conf - 0.1))
        return
    # ① 用**真实解码器**得到"框"（你已确认框是对的）
    dets = decode_yolo11_kpt(outs, labels, w, h, conf=conf,
                             input_w=pre.size, input_h=pre.size)
    if not dets:
        print("真实解码器没给出检测（conf=%.2f）" % conf); sys.exit(1)

    grids = {}
    for arr in outs.values():
        arr = np.asarray(arr)
        if arr.ndim == 4 and arr.shape[0] == 1:
            grids.setdefault(arr.shape[1], {})[arr.shape[3]] = arr[0]

    sx = w / float(pre.size)
    vis = frame.copy()
    d = dets[0]
    box_in = [d.x / sx, d.y / sx, (d.x + d.w) / sx, (d.y + d.h) / sx]
    print("真实解码: score=%.3f  框(帧域)=[%d,%d,%d,%d]  当前 kpts:"
          % (d.score, d.x, d.y, d.x + d.w, d.y + d.h))
    for k in range(4):
        print("    #%d (%.0f,%.0f) conf=%.2f" % (k, d.kpts[k][0], d.kpts[k][1],
                                                 d.kpt_conf[k]))
    hit = _find_cell_for_box(grids, nc, kch, pre.size, box_in, d.score)
    if hit is None:
        print("!! 反推不到产生该框的 cell（可能被 NMS/多尺度合并）"); sys.exit(1)
    g, xs, ys, stride, sc = hit
    tens = grids[g]
    raw = tens[kch].reshape(g * g, 4, _KPT_PER_PT)[ys * g + xs]
    print("-" * 70)
    print("该框来自 scale g=%d stride=%.0f cell=(%d,%d) score=%.3f" % (g, stride, xs, ys, sc))
    for k in range(4):
        print("    raw#%d=(%8.3f,%8.3f) vis_raw=%7.3f sigmoid=%.4f"
              % (k, raw[k, 0], raw[k, 1], raw[k, 2],
                 1.0 / (1.0 + np.exp(-float(raw[k, 2])))))
    print("-" * 70)
    print("候选解码 vs 该框（正确解应: 框内占比=1.0、到框角平均距离≈0）:")
    best = None
    for tag in VARIANTS:
        pts = decode_variant(raw, xs, ys, stride, tag) * sx
        ins, err = _scores([d.x, d.y, d.x + d.w, d.y + d.h], pts)
        print("  %s: 框内=%.2f  角距=%7.1f px   pts=%s"
              % (tag, ins, err, " ".join("(%.0f,%.0f)" % (p[0], p[1]) for p in pts)))
        if best is None or (ins, -err) > (best[1], -best[2]):
            best = (tag, ins, err)
        for p in pts:
            cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), 4, COLORS[tag], -1)
    cv2.rectangle(vis, (d.x, d.y), (d.x + d.w, d.y + d.h), (60, 255, 60), 2)
    print("-" * 70)
    print("结论: 最像正确约定的是 %s（框内 %.2f，角距 %.1f px）" % best)
    print("颜色: " + "  ".join("%s=%s" % (t, COLORS[t]) for t in VARIANTS) + "  (BGR)")
    cv2.imwrite(a.out, vis)
    print("标注图 -> %s" % a.out)




# ------------------------------------------------------------------ 逐 cell 全局判定
def per_cell_report(outs, labels, pre_size, conf=0.4, limit=200):
    """对每个"有输出的 cell"比较：该 cell 解出的**框** vs 各候选公式解出的**四角包围盒**。

    YOLO-pose 的性质：目标的框 ≈ 其关键点的包围盒（门的框就是四角的外接框）。
    所以**正确约定**应让 |bbox(kpts) - box| ≈ 几个 px；错的会差 ≈ stride 量级。
    这个判据不依赖肉眼，也不依赖 NMS 选了哪个 cell。
    """
    nc = len(labels)
    kch = _KPT_PER_PT * 4
    grids = {}
    for arr in outs.values():
        arr = np.asarray(arr)
        if arr.ndim == 4 and arr.shape[0] == 1:
            grids.setdefault(arr.shape[1], {})[arr.shape[3]] = arr[0]
    rows = {t: [] for t in VARIANTS}
    n_cell = 0
    for g, tens in sorted(grids.items()):
        if 64 not in tens or nc not in tens or kch not in tens:
            continue
        stride = float(pre_size) / g
        cls = tens[nc].reshape(g * g, nc)
        sc = (1.0 / (1.0 + np.exp(-cls.astype(np.float64)))).max(axis=1)
        for i in np.where(sc >= conf)[0][:limit]:
            reg = tens[64].reshape(g * g, 4, 16).astype(np.float64)[i]
            e = np.exp(reg - reg.max(axis=1, keepdims=True))
            e /= e.sum(axis=1, keepdims=True)
            dist = (e * np.arange(16, dtype=np.float64)).sum(axis=1)
            ys, xs = divmod(int(i), g)
            cx, cy = (xs + 0.5) * stride, (ys + 0.5) * stride
            box = np.array([cx - dist[0] * stride, cy - dist[1] * stride,
                            cx + dist[2] * stride, cy + dist[3] * stride])
            raw = tens[kch].reshape(g * g, 4, _KPT_PER_PT)[i]
            n_cell += 1
            for t in VARIANTS:
                pts = decode_variant(raw, xs, ys, stride, t)
                qb = np.array([pts[:, 0].min(), pts[:, 1].min(),
                               pts[:, 0].max(), pts[:, 1].max()])
                rows[t].append(float(np.mean(np.abs(qb - box))))
    print("=" * 70)
    print("逐 cell 判定（conf>=%.2f，共 %d 个 cell）: |四角包围盒 - 框| 平均误差 px"
          % (conf, n_cell))
    print("%-6s %10s %10s %10s" % ("约定", "均值", "中位", "p90"))
    for t in VARIANTS:
        v = np.asarray(rows[t], float)
        if not v.size:
            print("%-6s %10s" % (t, "n/a")); continue
        print("%-6s %10.1f %10.1f %10.1f" % (t, v.mean(), np.median(v),
                                             np.percentile(v, 90)))
    best = min((t for t in VARIANTS if rows[t]),
               key=lambda t: float(np.mean(rows[t])))
    print("→ 误差最小（最可能正确）: %s" % best)
    print("  参考：stride 量级 = 输入域 16/32 px（帧域 ×2 = 32/64 px）")
    return best


if __name__ == "__main__":
    main()
