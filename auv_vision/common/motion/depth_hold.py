# -*- coding: utf-8 -*-
"""common/motion/depth_hold.py — **定深**（把深度管在一个区间里，超出就朝目标推）。

用户 2026-10-06 定：翘头时深度要控制在 **0.5~0.6 m**，"不准太深也不准太浅"。

先把物理事实写清（否则这条要求会被误解）：
* 深度**数值越大 = 越深**；`heave > 0` = 上浮、`heave < 0` = 下潜（见 `comm.dof_map`）。
* **可达到的最浅深度 = `comm.depth_guard.min_depth_m`**：限深保护在 `深度 ≤ 该值` 时把上浮清零
  ⇒ 船**没法停在比它更浅的地方**。所以区间下沿的实际生效值是 `max(band_lo, 全局下限)`；
  两者不一致时本模块会在启动时**打印一次**（不静默）。
* 现状（2026-10-06）：全局已由用户下调为 **0.50**，与 `band_lo` 同值 ⇒ **0.5~0.6 整段可用**，
  工作点在 `target_m`=0.55（区间中点）。若哪天把全局抬回去（>0.50），生效下沿会跟着抬。

策略（比例 + 死区滞环，无积分 ⇒ 不会和定深 PID 打架/自激）：
* 太深（`depth > hi + deadzone`）→ 朝目标上浮；
* 太浅（`depth < eff_lo - deadzone`）→ 朝目标下潜；
* 区间内 / 无遥测 → **0**（不瞎推）。
"""
from __future__ import annotations

import base.cfg.settings as S
from common.cfg.cfgnode import merge, sub, req, req_flag, req_node, MissingCfg

# 兜底表（= 当前 cfg/comm.yaml 的 grab.depth_hold；删键/删段都不崩）
_K_HOLD = (
    "enable",
    "target_m"
)


def hold_cfg(prefix="grab"):
    """`comm.<prefix>.depth_hold` + 代码兜底。"""
    return req_node(S.get("comm.%s" % prefix, None) or {}, "depth_hold")


class DepthHold(object):
    """区间定深（每帧 `step(now_ms, uart)` → heave DOF；`0.0` = 本帧不动）。"""

    def __init__(self, prefix="grab", log=None):
        self.prefix = prefix
        self.log = log or (lambda *a: None)
        c = hold_cfg(prefix)
        self.enable = bool(c["enable"])
        self.target = float(c["target_m"])
        self.band_lo = float(c["band_lo_m"])
        self.band_hi = float(c["band_hi_m"])
        self.kp = float(c["kp"])
        self.out_max = abs(float(c["out_max"]))
        self.deadzone = abs(float(c["deadzone_m"]))
        self._warned_lo = False
        self._warned_tel = False
        self.last_depth = None
        self.last_heave = 0.0

    def _clip(self, v):
        return max(-self.out_max, min(self.out_max, v))

    def step(self, now_ms, uart=None):
        """返回本帧 heave（`+` = 上浮）。区间内/无遥测/未启用 ⇒ 0。"""
        if not self.enable:
            return 0.0
        floor = 0.0
        getter = getattr(uart, "effective_min_depth_m", None) if uart is not None else None
        if getter is not None:
            try:
                floor = float(getter)
            except (TypeError, ValueError):
                floor = 0.0
        lo = max(self.band_lo, floor)          # ★ 浅端被全局下限顶住（0.50 做不到的根源）
        if lo > self.band_lo + 1e-9 and not self._warned_lo:
            self._warned_lo = True
            self.log("[HOLD] ⚠️ 期望下沿 %.2fm 比全局限深下限 %.2fm 浅 ⇒ **实际生效下沿 %.2fm**"
                     "（要真的到 %.2fm 得下调 comm.depth_guard.min_depth_m）"
                     % (self.band_lo, floor, lo, self.band_lo))
        d = getattr(uart, "depth_m", None) if uart is not None else None
        if d is None:
            if not self._warned_tel:
                self._warned_tel = True
                self.log("[HOLD] 没有深度遥测 ⇒ 定深不动作（信息不足不瞎推）")
            self.last_heave = 0.0
            return 0.0
        self.last_depth = float(d)
        if self.last_depth > self.band_hi + self.deadzone:
            out = self._clip(self.kp * (self.last_depth - self.target))     # 太深 → 上浮
        elif self.last_depth < lo - self.deadzone:
            out = self._clip(-self.kp * (self.target - self.last_depth))    # 太浅 → 下潜
        else:
            out = 0.0                                                      # 在区间内 → 不动
        self.last_heave = out
        return out
