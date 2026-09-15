# -*- coding: utf-8 -*-
"""common/frame_stamp.py — 唯一帧 / 帧有效性（共用件：本地/板端对齐、任务自检）

* 唯一帧  ：`frame_id` 严格递增 —— 同一张图重复推理仍是同一帧，不能重复计数；
* 单调时间：控制时刻不回退；采集时间严格递增；
* 新鲜度  ：`age = now - captured_at`；过期帧**仍可用于估计**，但不计入确认（两分法）；
* 不连续  ：相邻采集间隔 > max_gap → 标记断点（调用方只重置正在累计的那段确认）；
* 容差    ：`now_ms` 取整/时钟粒度会造成最多几毫秒的"微未来"，`future_tol_s` 内按新鲜处理。

`StampCfg` 默认值可用 `cfg/comm.yaml → gate.stamp:` 覆盖（`from_settings(section=...)` 可换区段）。
后向兼容：调用方不传帧标识时，自动用"帧对象身份 + 计数"生成
（同一帧对象重复调用不会重复计数）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class StampCfg:
    max_age_s: float = 0.9
    max_gap_s: float = 0.35
    adaptive_gap: bool = True
    future_tol_s: float = 0.005

    @classmethod
    def from_settings(cls, S, section="comm.gate.stamp"):
        """从配置读取覆盖值；`section` 可按使用方改写（共用件不绑定具体任务）。"""
        cfg = cls()
        data = S.get(section) or {}
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


@dataclass(frozen=True)
class FrameStamp:
    frame_id: int
    captured_at: float        # 单调秒（建议 time.monotonic()）


@dataclass(frozen=True)
class Verdict:
    ok: bool
    kind: str                 # ok | stale | duplicate | out_of_order | future | bad | time_back
    usable: bool              # 可否用于估计
    countable: bool           # 可否计入确认
    discontinuity: bool
    age: float = 0.0
    gap: float = 0.0
    reason: str = ""


class FrameValidator(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self._last_id = None
        self._last_cap = None
        self._last_now = None
        self._dts = []
        self.bad_frames = 0

    def _gap_limit(self):
        base = float(self.cfg.max_gap_s)
        if self.cfg.adaptive_gap and len(self._dts) >= 3:
            recent = self._dts[-9:]
            med = sorted(recent)[len(recent) // 2]
            base = max(base, 3.0 * med)
        return base

    def validate(self, stamp, now):
        if not isinstance(stamp, FrameStamp):
            self.bad_frames += 1
            return Verdict(False, "bad", False, False, False,
                           reason="缺少帧标识")
        if isinstance(stamp.frame_id, bool) or not isinstance(stamp.frame_id, int) \
                or stamp.frame_id < 0:
            self.bad_frames += 1
            return Verdict(False, "bad", False, False, False, reason="frame_id 非法")
        if not isinstance(stamp.captured_at, (int, float)) or \
                not math.isfinite(stamp.captured_at):
            self.bad_frames += 1
            return Verdict(False, "bad", False, False, False, reason="captured_at 非法")
        now = float(now)
        if not math.isfinite(now):
            self.bad_frames += 1
            return Verdict(False, "bad", False, False, False, reason="now 非法")
        if self._last_now is not None and now < self._last_now - 1e-9:
            self.bad_frames += 1
            return Verdict(False, "time_back", False, False, False, reason="控制时刻回退")

        age = now - float(stamp.captured_at)
        tol = float(getattr(self.cfg, "future_tol_s", 0.005))
        if age < -tol:
            self.bad_frames += 1
            return Verdict(False, "future", False, False, False, age=age,
                           reason="采集时间在未来")
        if age < 0.0:
            age = 0.0        # 取整/时钟粒度造成的微负值：按新鲜处理

        if self._last_id is not None and stamp.frame_id <= self._last_id:
            self.bad_frames += 1
            kind = "duplicate" if stamp.frame_id == self._last_id else "out_of_order"
            return Verdict(False, kind, False, False, False, age=age,
                           reason="重复帧或帧号回退")
        if self._last_cap is not None and float(stamp.captured_at) <= self._last_cap:
            self.bad_frames += 1
            return Verdict(False, "duplicate", False, False, False, age=age,
                           reason="采集时间重复或回退")

        gap = 0.0 if self._last_cap is None else float(stamp.captured_at) - self._last_cap
        disc = self._last_cap is not None and gap > self._gap_limit()

        if age > float(self.cfg.max_age_s):
            self.bad_frames += 1
            self._accept(stamp, now, gap)
            return Verdict(False, "stale", True, False, disc, age=age, gap=gap,
                           reason="帧龄超限(仍可用于估计)")

        self._accept(stamp, now, gap)
        if disc:
            return Verdict(True, "ok", True, True, True, age=age, gap=gap,
                           reason="采集断点")
        return Verdict(True, "ok", True, True, False, age=age, gap=gap)

    def _accept(self, stamp, now, gap):
        self._last_id = int(stamp.frame_id)
        self._last_cap = float(stamp.captured_at)
        self._last_now = float(now)
        if gap > 0:
            self._dts.append(gap)
            if len(self._dts) > 32:
                self._dts.pop(0)
