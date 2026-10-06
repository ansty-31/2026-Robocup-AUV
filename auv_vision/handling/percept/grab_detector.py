# -*- coding: utf-8 -*-
"""grab_detector.py — 把 CV 红球检测接进工程的统一接口（`Det` + `DetectorHub`）。
（详细用法、判据与实测见 doc/注释历史.md）"""

from __future__ import annotations

import numpy as np

from common.vision.detector import Det, DetectorBase
from handling.percept.ball_tracker import BallTracker
from handling.percept.cv_ball import (Detection, Params, best, detect)

try:
    import base.cfg.settings as S
except Exception:                                     # pragma: no cover
    S = None


def _cfg(path, default):
    """读 `vision.<path>`；漏 `vision.` 前缀会静默取默认值（2026-10-06 修，见 cv_ball._cfg）。"""
    if S is None:
        return default
    p = path if path.startswith(("vision.", "comm.")) else "vision." + path
    v = S.get(p, None)
    return default if v is None else v


class GrabBallDetector(DetectorBase):
    """夹取小球的 CV 后端。"""

    def __init__(self, params: Params | None = None):
        self.p = params or Params.from_cfg()
        self.kind = str(_cfg("grab.kind", "red_ball"))
        self.track_enable = bool(_cfg("grab.track.enable", True))
        self._tracker = BallTracker(
            self.p,
            margin=float(_cfg("grab.track.margin", 64.0)),
            proc_side=int(_cfg("grab.track.proc_side", 256)),
            max_lost=int(_cfg("grab.track.max_lost", 3))) if self.track_enable else None
        self._last: Detection | None = None

    # ---------------------------------------------------------------- 接口
    @staticmethod
    def _to_det(d: Detection, kind: str) -> Det:
        x1, y1, x2, y2 = d.bbox
        if x2 <= x1 or y2 <= y1:          # 兜底：用圆的外接方框
            x1, y1, x2, y2 = (d.cx - d.r, d.cy - d.r, d.cx + d.r, d.cy + d.r)
        return Det(kind, d.score, x1, y1, x2 - x1, y2 - y1)

    def detect(self, frame) -> list:
        """整帧检出 `[Det]`（跟踪开启时只会用一帧的 ROI，等价结果）。"""
        circles = self.circles(frame)
        return [self._to_det(d, self.kind) for d in circles]

    # ---------------------------------------------------------------- 夹取专用
    def circles(self, frame) -> list:
        """整帧检出 `[Detection]`（带圆心/半径/质量指标）。"""
        if self._tracker is None:
            dets = detect(frame, self.p)
            self._last = best(dets)
            return dets
        d, _info = self._tracker.step(frame)
        self._last = d
        return [] if d is None else [d]

    def step(self, frame):
        """流式：返回 `(Detection|None, TrackInfo)`；未开跟踪时 mode 恒为 'full'。"""
        if self._tracker is None:
            d = best(detect(frame, self.p))
            self._last = d
            from handling.percept.ball_tracker import TrackInfo
            return d, TrackInfo("full", (), 0)
        d, info = self._tracker.step(frame)
        self._last = d
        return d, info

    @property
    def last(self):
        """上一帧的球（没跟到就是 None）。"""
        return self._last

    def reset(self):
        """换任务/丢目标时清轨迹。"""
        if self._tracker is not None:
            self._tracker.reset()
        self._last = None

    def stats(self):
        base = {"kind": self.kind, "track": self.track_enable}
        if self._tracker is not None:
            base.update(self._tracker.stats())
        return base


def build_grab_backend():
    """构造夹取小球后端；`grab.detect.mode: mock` 时返回 None（交回 mock/legacy）。"""
    if str(_cfg("grab.detect.mode", "cv")).strip().lower() == "mock":
        return None
    return GrabBallDetector()


def circle_to_ratio(d: Detection, fw: int, fh: int) -> float:
    """球在画面里的"面积占比"（等价 `Det.ratio`，但按圆算 —— 撞球的判据量纲一致）。"""
    return float(np.pi * d.r * d.r) / float(max(1, fw) * max(1, fh))
