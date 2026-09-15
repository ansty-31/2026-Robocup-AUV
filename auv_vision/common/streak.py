# -*- coding: utf-8 -*-
"""common/streak.py — 确认机制：时间 + 帧数双条件（共用件：本地/板端对齐、任务自检）

为什么加：现在确认只看"帧数"（如 6 帧 / 20 帧），而帧率会变（板端 ~11fps、PC 上更高），
同一份配置在不同机器上的**物理时间**不同 → 行为漂移。改成
    ready = (count >= frames) AND (t - start >= seconds)
后，两种条件同时满足才确认：*重复/过期帧不能凑数*，且时长可控。

`ConfirmCfg` 的默认值按"当前帧数在 11fps 下的等效时间"取值，
因此在板端行为与升级前**基本一致**，但不再随帧率漂移。
全部可用 `cfg/comm.yaml → gate.confirm:` 覆盖（不必改代码）。
"""
from __future__ import annotations

from dataclasses import dataclass


class Streak(object):
    __slots__ = ("key", "start", "count")

    def __init__(self):
        self.reset()

    def reset(self):
        self.key = None
        self.start = None
        self.count = 0

    def add(self, key, t):
        if key != self.key:
            self.key = key
            self.start = t
            self.count = 0
        self.count += 1
        return self.count

    def ready(self, t, seconds, frames=1):
        if self.start is None or self.count < int(frames):
            return False
        return (t - self.start) + 1e-9 >= float(seconds)

    def progress(self, t):
        if self.start is None:
            return 0.0
        return max(0.0, float(t) - float(self.start))


@dataclass
class ConfirmCfg:
    # 语义：确认 = (帧数 >= frames) 且 (时长 >= seconds)。
    # 默认 seconds 取"升级前帧数在 30fps 下的等效时间"→ 常规帧率(≤30fps)由**帧数主导**，
    # 行为与升级前一致；只有帧率更高时秒数下限才生效（防止确认被过快满足）。
    align_s: float = 0.20
    align_frames: int = 6          # = 升级前 G.align.confirm_frames
    hold_s: float = 0.66
    hold_frames: int = 20          # = 升级前 G.hold.max_frames
    cross_s: float = 0.07
    cross_frames: int = 2          # = 升级前 cross_confirm_frames
    through_s: float = 0.27
    through_frames: int = 8        # = 升级前 through.confirm_frames
    pose_hold_s: float = 0.33
    pose_hold_frames: int = 10     # = 升级前 pose_hold_frames
    cue_s: float = 0.10
    cue_frames: int = 3
    cue_after_frames: int = 15     # 位姿连续不可用多少帧后，恢复线索才接管

    @classmethod
    def from_settings(cls, S, section="comm.gate.confirm"):
        """从配置读取覆盖值；`section` 可按使用方改写（共用件不绑定具体任务）。"""
        cfg = cls()
        data = S.get(section) or {}
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg
