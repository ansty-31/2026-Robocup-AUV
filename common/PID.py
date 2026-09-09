# -*- coding: utf-8 -*-
"""PID.py — 任务公用小件（PID 等）

供任务一(ball)/gate 共用（早期由 root tasks.py 迁移至此）。
逻辑与迁移前逐字一致，仅调整位置。
"""
from __future__ import annotations


class PID(object):
    """位置式 PID（视觉居中用：死区清零积分、限幅防 windup）。"""

    def __init__(self, kp=1.0, ki=0.0, kd=0.0,
                 out_min=-1.0, out_max=1.0, deadzone=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        self.deadzone = deadzone
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
        self.last_out = max(self.out_min, min(self.out_max, out))
        return self.last_out
