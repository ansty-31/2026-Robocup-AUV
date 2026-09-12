# -*- coding: utf-8 -*-
"""ramp.py — 线性斜坡 / 变化率限幅（slew-rate limiter），全项目通用

目的：把“期望值”按**每秒最大变化率**线性逼近，而不是瞬间跳到目标，
模拟人手动遥控推杆时速度的连续变化（不会直接激增到目标转速）。

被谁用：`base/uart.py` 在把上层 DOF 目标转成串口帧之前做 DOF 级斜坡
（配置 `cfg/comm.yaml` → `ramp.dof_per_s`，单位：归一化 DOF/秒；<=0 表示直通不限制）。

用法：
    from common.ramp import LinearRamp, DofRamp

    r = LinearRamp(rate=1.0)          # 1.0 /秒
    v = r.update(0.8, dt=0.05)        # 每帧调用，逐步逼近目标（线性）

    dr = DofRamp(rate=1.0)            # 四通道 (surge,sway,heave,yaw)
    dof = dr.update((0.5, 0.0, 0.0, 0.2), dt)
"""


def _clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))


class LinearRamp(object):
    """单通道线性斜坡。

    rate : 每秒最大变化量（单位/秒）；<=0 表示“直通”（不限制，一步到位）
    value: 当前值；lo/hi 为其限幅
    """

    def __init__(self, rate=1.0, value=0.0, lo=-1.0, hi=1.0):
        self.rate = float(rate)
        self.lo, self.hi = float(lo), float(hi)
        self.value = _clamp(value, self.lo, self.hi)

    def reset(self, value=0.0):
        """直接置位（不走斜坡），返回当前值。"""
        self.value = _clamp(value, self.lo, self.hi)
        return self.value

    def set_rate(self, rate):
        self.rate = float(rate)

    def update(self, target, dt):
        """按 dt 秒线性逼近 target，返回新的当前值。"""
        target = _clamp(target, self.lo, self.hi)
        if self.rate <= 0 or dt <= 0:
            self.value = target
            return self.value
        step = self.rate * dt
        d = target - self.value
        if abs(d) <= step:
            self.value = target
        else:
            self.value += step if d > 0 else -step
        return self.value


class DofRamp(object):
    """多通道线性斜坡（默认 surge / sway / heave / yaw）。"""

    NAMES = ("surge", "sway", "heave", "yaw")

    def __init__(self, rate=1.0, names=None, lo=-1.0, hi=1.0):
        self.names = tuple(names or self.NAMES)
        self.rate = float(rate)
        self.lo, self.hi = float(lo), float(hi)
        self._ch = {n: LinearRamp(rate, 0.0, lo, hi) for n in self.names}
        self.values = tuple(0.0 for _ in self.names)

    def reset(self, values=None):
        """直接置位（不走斜坡），values 顺序同 names；省略=全 0。"""
        vals = list(values) if values is not None else [0.0] * len(self.names)
        for n, v in zip(self.names, vals):
            self._ch[n].reset(v)
        self.values = tuple(self._ch[n].value for n in self.names)
        return self.values

    def set_rate(self, rate):
        self.rate = float(rate)
        for c in self._ch.values():
            c.set_rate(rate)

    def update(self, targets, dt):
        """targets 顺序同 names；返回平滑后的元组。"""
        for n, t in zip(self.names, targets):
            self._ch[n].update(t, dt)
        self.values = tuple(self._ch[n].value for n in self.names)
        return self.values
