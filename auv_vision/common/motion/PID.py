# -*- coding: utf-8 -*-
"""PID.py — 任务公用小件（PID 等）
（详细用法、判据与实测见 doc/注释历史.md）"""
from __future__ import annotations

import math

class PID(object):
    """位置式 PID（视觉居中用：死区清零积分、限幅防 windup、**可选执行器死区补偿**）。

    `bias`（默认 0 = 不补偿）：只要出了死区，就先把输出**抬到执行器死区之上**再按 kp 线性升。
    为什么需要它：推进器在 `|DOF| < 0.138` 时**一点推力都没有**（见 cfg 注释），
    所以"按 kp 算出来 0.05"这种小输出等于没推力 —— 控制器会表现成"要么顶满、要么不动"。
    `bias` 取执行器死区值（0.138）即：误差刚出死区就给最小有效推力，之后线性升到 out_max。
    """
    def __init__(self, kp=1.0, ki=0.0, kd=0.0,
                 out_min=-1.0, out_max=1.0, deadzone=0.0, bias=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        self.deadzone = deadzone
        self.bias = float(bias or 0.0)
        self._last_err = 0.0
        self._integral = 0.0
        self._last_t = None
        self.last_out = 0.0

    def reset(self):
        self._last_err = 0.0
        self._integral = 0.0
        self._last_t = None
        self.last_out = 0.0

    def update(self, error, now_ms=None):
        if now_ms is None:
            dt = 0.05
        else:
            if self._last_t is None:
                dt = 0.05
            else:
                dt = max(0.001, (now_ms - self._last_t) / 1000.0)
            self._last_t = now_ms
        if abs(error) <= self.deadzone:
            self._integral = 0.0
            self._last_err = error
            self.last_out = 0.0
            return 0.0
        self._integral += error * dt
        lim = max(0.5 * (self.out_max - self.out_min), 0.5)
        self._integral = max(-lim, min(lim, self._integral))
        deriv = (error - self._last_err) / dt if dt > 0 else 0.0
        self._last_err = error
        out = self.kp * error + self.ki * self._integral + self.kd * deriv
        if self.bias > 0.0:
            # ★ 死区补偿：抬到执行器死区之上（对负误差对称），上限仍是 out_max
            out = math.copysign(min(abs(out) + self.bias, self.out_max), out)
        self.last_out = max(self.out_min, min(self.out_max, out))
        return self.last_out
