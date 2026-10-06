#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""夹取小球 · CV 的**流式/ROI 跟踪模式**（板端省算力的正解）。

## 为什么需要它（板端实测倒逼出来的设计）

RDK X5 的 A55 上，**每一次全图（1280×720）uint8 pass 约 2.6 ms**（内存带宽受限，不是算力受限）。
于是"每帧全图"的固定成本就是"pass 数 × 2.6 ms"：v2 快掩膜里 split/max/subtract/compare 5 个 pass
(5.9 ms) + rel LUT/compare/or 3 个 pass (5.5 ms) + HSV cvtColor 与 5 个 inRange/and pass (14.7 ms)
≈ 26 ms，再加连通域/几何 ≈ 40-60 ms/帧。**微优化 pass 已经没有多少油水了，必须减少像素数。**

而夹取小球是**跟踪**任务：球在两帧之间的位移远小于它的半径（本数据集实测球被夹爪带走时
4 帧位移 166 px，即 ≈40 px/帧，而球半径 25–270 px）。所以只有**第一次（或跟丢后）**需要全图搜索，
之后每帧只要在"上一帧球心 ± k·r"的 ROI 里做同一套检测即可。

ROI(256×256) 相比全图(1280×720) 像素少 **14 倍** ⇒ 掩膜成本从 ~26 ms 掉到 ~2 ms。

## 正确性口径（必须写清楚，否则就是悄悄掉精度）

* ROI 内含**完整的球**时，结果与全图检测**完全一致**（同一套算子，只是作用域变小）。
* ROI 边界会引入人工的 0 邻域，所以**贴着 ROI 边界**的检出不可信 —— 这时（以及 ROI 内没检出时）
  立刻回退一次全图检测（`fallback`）。回退率是本模块的核心指标，bench 会报出来。
* ROI 大小 = `r × roi_gain + roi_pad`，默认 gain=2.0 / pad=40：
  默认 assume 帧间位移 < r（本数据集 ≈0.2r，留 10 倍余量）。

用法（流式，按帧号顺序喂）：
    tracker = BallTracker(p)              # p = cv_red_ball.Params
    for frame in frames:
        det, info = tracker.step(frame)   # info: mode('full'|'roi'|'fallback') / ms
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from cv_red_ball import Params, Detection, detect


@dataclass
class TrackInfo:
    mode: str            # 'full' | 'roi' | 'fallback'
    roi: tuple = ()      # (x0, y0, x1, y1) 或空
    lost: int = 0        # 连续跟丢帧数


class BallTracker:
    """单目标（红球）ROI 跟踪器：只在 ROI 里跑 `detect()`，跟丢/贴边时回退全图。

    参数：
        roi_gain / roi_pad  ROI 半径 = r*gain + pad
        max_lost            连续多少帧没检出就放弃 ROI（回到全图搜索）
        edge_margin         检出圆离 ROI 边界近于该值(px) 就判定"不可信" → 回退
    """

    def __init__(self, params: Params, margin: float = 64.0, max_lost: int = 3,
                 edge_margin: float = 6.0, min_roi: int = 128, proc_side: int = 0):
        """margin: ROI 在球外多留的像素（每边）。
        ⚠️ 一开始我写成 `side = 4r + 80`（把 gain 乘在半径上又乘了 2），球半径 188 时
        ROI 变成 832×832 = 69% 全图 —— 板端实测 "ROI 模式反而比全图慢"。改成"球 + 固定余量"后
        ROI 面积随球大小线性增长，小远球（r≈40，ROI 208²）只要 ~3 ms。
        默认 64 px：本数据集球被夹爪带走时最大 ≈40 px/帧，留 1.6 倍余量。"""
        self.p = params
        self.margin = margin
        self.max_lost = max_lost
        self.edge_margin = edge_margin
        self.min_roi = min_roi
        # proc_side>0：尺度归一化 —— ROI 一律缩放到 proc_side² 再检测（成本与球大小无关），
        # 检出再乘回缩放比。代价：近距时中心精度按比例下降（proc 256 / ROI 664 时 1px≈2.6px 原图）。
        self.proc_side = int(proc_side)
        self.prev: Detection | None = None
        self.lost = 0
        self.n_full = self.n_roi = self.n_fallback = 0
        self.last_side = 0

    # ---- 内部：把 ROI 内的检出搬回全图坐标 ----
    @staticmethod
    def _shift(dets: list[Detection], x0: int, y0: int) -> list[Detection]:
        for d in dets:
            d.cx += x0
            d.cy += y0
            d.bbox = tuple(b + o for b, o in zip(d.bbox, (x0, y0, x0, y0)))
        return dets

    def _roi_box(self, img) -> tuple[int, int, int, int]:
        h, w = img.shape[:2]
        d = self.prev
        side = int(max(self.min_roi, 2 * (d.r + self.margin)))
        side = min(side, max(w, h))
        x0 = int(np.clip(d.cx - side / 2, 0, max(0, w - side)))
        y0 = int(np.clip(d.cy - side / 2, 0, max(0, h - side)))
        x1 = min(w, x0 + side)
        y1 = min(h, y0 + side)
        self.last_side = max(x1 - x0, y1 - y0)
        return x0, y0, x1, y1

    def step(self, img: np.ndarray):
        """返回 (Detection | None, TrackInfo)。"""
        need_full = self.prev is None or self.lost >= self.max_lost
        if not need_full:
            x0, y0, x1, y1 = self._roi_box(img)
            sub = img[y0:y1, x0:x1]
            sc = 1.0
            p_use = self.p
            if self.proc_side and max(sub.shape[:2]) > self.proc_side:
                sc = self.proc_side / float(max(sub.shape[:2]))
                sub = cv2.resize(sub, (self.proc_side, self.proc_side),
                                 interpolation=cv2.INTER_AREA)
                p_use = replace(self.p, min_area=max(6, int(round(self.p.min_area * sc * sc))),
                                r_min=max(3.0, self.p.r_min * sc))
            dets = detect(sub, p_use)
            if sc != 1.0:                      # 检出的圆心/半径按比例搬回原尺寸
                for d in dets:
                    d.cx /= sc; d.cy /= sc; d.r /= sc
            dets = self._shift(dets, x0, y0)
            ok = self._pick(dets, (x0, y0, x1, y1), x0, y0, x1, y1, img.shape, sc)
            if ok is not None:
                self.n_roi += 1
                self.prev = ok
                self.lost = 0
                return ok, TrackInfo("roi", (x0, y0, x1, y1), 0)
            # ROI 里没有可信结果 → 回退全图
            self.n_fallback += 1
            dets = detect(img, self.p)
            best = self._pick(dets, None, 0, 0, img.shape[1], img.shape[0], img.shape, 1.0)
            self._update(best)
            return best, TrackInfo("fallback", (x0, y0, x1, y1), self.lost)

        self.n_full += 1
        dets = detect(img, self.p)
        best = self._pick(dets, None, 0, 0, img.shape[1], img.shape[0], img.shape, 1.0)
        self._update(best)
        return best, TrackInfo("full", (), self.lost)

    def _update(self, best):
        if best is None:
            self.lost += 1
            if self.lost >= self.max_lost:
                self.prev = None
        else:
            self.prev = best
            self.lost = 0

    def _pick(self, dets, roi, x0, y0, x1, y1, shape, sc=1.0):
        """选最可信的一个：优先靠近上一帧球心、且不贴 ROI 边的。
        sc: 尺度归一化的缩放比（贴边阈值也要跟着缩）。"""
        if not dets:
            return None
        if self.prev is None:
            return max(dets, key=lambda d: d.score)
        cand = []
        for d in dets:
            edge = min(d.cx - x0, d.cy - y0, x1 - d.cx, y1 - d.cy)
            if edge < max(self.edge_margin, 6.0 / max(sc, 1e-6)):      # 贴着裁剪边 → 可能被切断，不可信
                continue
            jump = np.hypot(d.cx - self.prev.cx, d.cy - self.prev.cy) / max(self.prev.r, 1.0)
            cand.append((jump, d))
        if not cand:
            return None
        inside = [c for c in cand if c[0] <= 2.0]
        pool = inside or cand
        return min(pool, key=lambda c: c[0])[1]
