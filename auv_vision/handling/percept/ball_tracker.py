# -*- coding: utf-8 -*-
"""ball_tracker.py — 夹取小球的**流式 ROI 跟踪**（板端省算力的关键，见 `grab/README.md`）。
（详细用法、判据与实测见 doc/注释历史.md）"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from handling.percept.cv_ball import Detection, Params, detect


@dataclass
class TrackInfo:
    """一帧的跟踪状态（给日志/调试用，别拿它做控制判据）。"""

    mode: str            # 'roi' | 'full' | 'fallback'
    roi: tuple = ()      # (x0, y0, x1, y1)
    lost: int = 0        # 连续跟丢帧数
    scale: float = 1.0   # 尺度归一化的缩放比（1.0 = 未缩放）


class BallTracker:
    """单目标（红球）ROI 跟踪器。"""

    def __init__(self, params: Params, margin: float = 64.0, proc_side: int = 256,
                 max_lost: int = 3, edge_margin: float = 6.0, min_roi: int = 128):
        self.p = params
        self.margin = float(margin)
        self.proc_side = int(proc_side)
        self.max_lost = int(max_lost)
        self.edge_margin = float(edge_margin)
        self.min_roi = int(min_roi)
        self.prev: Detection | None = None
        self.lost = 0
        self.n_roi = self.n_full = self.n_fallback = 0
        self.last_side = 0

    # ------------------------------------------------------------------ #
    def reset(self):
        """丢掉轨迹（下一帧从全图搜索开始）。"""
        self.prev, self.lost = None, 0

    @staticmethod
    def _shift(dets: list, x0: int, y0: int) -> list:
        """把 ROI 内检出的坐标搬回全图。"""
        for d in dets:
            d.cx += x0
            d.cy += y0
            d.bbox = tuple(b + o for b, o in zip(d.bbox, (x0, y0, x0, y0)))
        return dets

    def _roi_box(self, img):
        h, w = img.shape[:2]
        d = self.prev
        # side = 2(r + margin)：⚠️ 别写成 4r+80（球半径 188 时 ROI = 832² = 69% 全图， 实测"ROI 反而比全图慢"——踩过这个坑）
        side = int(max(self.min_roi, 2 * (d.r + self.margin)))
        side = min(side, max(w, h))
        x0 = int(np.clip(d.cx - side / 2, 0, max(0, w - side)))
        y0 = int(np.clip(d.cy - side / 2, 0, max(0, h - side)))
        x1 = min(w, x0 + side)
        y1 = min(h, y0 + side)
        self.last_side = max(x1 - x0, y1 - y0)
        return x0, y0, x1, y1

    def _pick(self, dets, x0, y0, x1, y1, sc=1.0):
        """选最可信的一个：不贴 ROI 边、且离上一帧球心最近的。"""
        if not dets:
            return None
        if self.prev is None:
            return max(dets, key=lambda d: d.score)
        cand = []
        for d in dets:
            edge = min(d.cx - x0, d.cy - y0, x1 - d.cx, y1 - d.cy)
            if edge < self.edge_margin + 1.0 / max(sc, 1e-6):
                continue
            jump = np.hypot(d.cx - self.prev.cx, d.cy - self.prev.cy) / max(self.prev.r, 1.0)
            cand.append((jump, d))
        if not cand:
            return None
        inside = [c for c in cand if c[0] <= 2.0]
        return min(inside or cand, key=lambda c: c[0])[1]

    def _update(self, best):
        if best is None:
            self.lost += 1
            if self.lost >= self.max_lost:
                self.prev = None
        else:
            self.prev, self.lost = best, 0

    # ------------------------------------------------------------------ #
    def step(self, img: np.ndarray):
        """喂一帧，返回 `(Detection|None, TrackInfo)`。"""
        if self.prev is not None and self.lost < self.max_lost:
            x0, y0, x1, y1 = self._roi_box(img)
            sub = img[y0:y1, x0:x1]
            sc, p_use = 1.0, self.p
            if self.proc_side and max(sub.shape[:2]) > self.proc_side:
                sc = self.proc_side / float(max(sub.shape[:2]))
                # ⚠️ 必须**按同一比例缩两轴**（fx=fy=sc），不能 resize 到 (proc, proc)：
                sub = cv2.resize(sub, None, fx=sc, fy=sc,
                                 interpolation=cv2.INTER_AREA)
                p_use = replace(self.p,
                                min_area=max(6, int(round(self.p.min_area * sc * sc))),
                                r_min=max(3.0, self.p.r_min * sc))
            dets = detect(sub, p_use)
            if sc != 1.0:                       # 圆心/半径搬回原尺寸
                for d in dets:
                    d.cx /= sc
                    d.cy /= sc
                    d.r /= sc
            dets = self._shift(dets, x0, y0)
            ok = self._pick(dets, x0, y0, x1, y1, sc)
            if ok is not None:
                self.n_roi += 1
                self.prev = ok
                self.lost = 0
                return ok, TrackInfo("roi", (x0, y0, x1, y1), 0, sc)
            self.n_fallback += 1
            best = self._pick(detect(img, self.p), 0, 0, img.shape[1], img.shape[0])
            self._update(best)
            return best, TrackInfo("fallback", (x0, y0, x1, y1), self.lost)

        self.n_full += 1
        best = self._pick(detect(img, self.p), 0, 0, img.shape[1], img.shape[0])
        self._update(best)
        return best, TrackInfo("full", (), self.lost)

    def stats(self):
        """回退率等计数（接日志用）。"""
        n = self.n_roi + self.n_full + self.n_fallback
        return {"n": n, "roi": self.n_roi, "full": self.n_full,
                "fallback": self.n_fallback,
                "fallback_rate": (self.n_fallback / n) if n else 0.0}
