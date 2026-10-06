# -*- coding: utf-8 -*-
"""handling/motion/actions.py — 两个流程共用的**相位常量与兜底值表**。
（方法体一字未改，只搬位置。）"""
from __future__ import annotations

import math

import base.cfg.settings as S
from common.cfg.cfgnode import flag, merge, pid_kw, sub
from common.motion.PID import PID
from common.motion.axis import AxisMove, EnsureDepth, TimedDof, _D_DIP, _D_RISE, action_cfg, axis_deg, grab_cfg, grab_node, level_relative_deg, pitch_up_deg
from common.motion.axis import AxisMove, TimedDof, ZERO_DOF, axis_deg, level_relative_deg, task_node
from handling.motion.params import _D_CENTER, _D_APPROACH, _D_ALIGN


def _cvnode(key, defaults):
    """`comm.grab.<key>` + 代码兜底（缺段/缺键都不崩）。"""
    return merge(sub(grab_node(), key), defaults)


def _ratio_radius(r, fw, fh):
    """半径 → 面积占比（与 `grab_detector.circle_to_ratio` 同一公式：πr²/(fw·fh)）。"""
    return float(math.pi) * float(r) * float(r) / float(max(1, fw) * max(1, fh))


def _dx_norm(cx, w):
    return (float(cx) - w / 2.0) / (w / 2.0)


def _dy_norm(cy, h):
    return (float(cy) - h / 2.0) / (h / 2.0)


def _clip(v):
    return float(max(-1.0, min(1.0, v)))


class HandlingActions(object):
    """关键动作与小件：PID 构造/校验、执行原语、横倾放球序列、下发口 `_set_info`。"""

    def _frame(self, frame):
        return self._down_frame if self._down_frame is not None else frame

    def _circles(self, f):
        """下视帧里所有球的圆（圆心+半径）。真实后端走 `circles()`，否则退化为 bbox 外接圆。"""
        if f is None:
            return []
        be = None
        try:
            be = self.hub.extra(self.name)
        except Exception:
            be = None
        if be is not None and hasattr(be, "circles"):
            try:
                return list(be.circles(f))
            except Exception as e:
                self.log("[GRAB] ⚠️ circles() 异常：%s" % e)
                return []
        out = []
        for d in self.hub.detect_list(self.name, f):
            try:
                cx, cy = d.center
                out.append(_Circle(cx, cy, max(float(d.w), float(d.h)) / 2.0))
            except Exception:
                continue
        return out

    def _best(self, f):
        """本帧最大占比的球（其余球是干扰/更远的球）。返回 `(circle, ratio)`。"""
        best, br = None, 0.0
        for c in self._circles(f):
            r = _ratio_radius(c.r, self.w, self.h)
            if r > br:
                best, br = c, r
        return best, br

    def _ema_update(self, r, alpha):
        if self._ema is None:
            self._ema = r
            return 0.0
        prev = self._ema
        self._ema = (1.0 - alpha) * prev + alpha * r
        return self._ema - prev

    def _new_pid(self, node, defaults):
        return PID(**pid_kw(node, defaults))

    def _ensure_pids(self):
        if self._pid_yaw is None:
            c = _cvnode("center_yaw", _D_CENTER)
            self._pid_yaw = self._new_pid(sub(c, "pid"), _D_CENTER["pid"])
            a = _cvnode("approach", _D_APPROACH)
            self._pid_sway = self._new_pid(sub(a, "sway"), _D_APPROACH["sway"])
            l = _cvnode("align_level", _D_ALIGN)
            self._pid_asw = self._new_pid(sub(l, "sway"), _D_ALIGN["sway"])
            self._pid_asu = self._new_pid(sub(l, "surge"), _D_ALIGN["surge"])

    def _set_info(self, action, ratio, dx, dy, sway, surge, heave, yaw, percent):
        self.last_info.update({
            "action": action, "phase": self._phase,
            "ratio": round(ratio, 4), "dx": round(dx, 4), "dy": round(dy, 4),
            "sway": round(sway, 4), "surge": round(surge, 4),
            "heave": round(heave, 4), "yaw": round(yaw, 4),
            "percent": -1.0 if percent is None else round(float(percent), 4),
            "attempts": self._attempts,
            "holds_ball": bool(self._ok),
            "status": S.STATUS_DONE if self._finished else S.STATUS_RUNNING,
            "reason": self._reason})

    def _accept(self, reason):
        self._finish(reason, ok=True)
        return 0.0, 0.0, 0.0, 0.0

    def _restore_floor(self):
        setter = getattr(self.uart, "set_extra_min_depth", None)
        if callable(setter) and self._floor_raised:
            setter(0.0)                    # 撤回本任务的收紧（全局 min_depth_m 没动过）
            self._floor_raised = False

    def _roll_err(self):
        """当前 roll 离目标角差多少（度）；拿不到遥测 → None（**不许瞎发**）。"""
        now = axis_deg(self.uart, "roll")
        return None if now is None else level_relative_deg(self._roll_target, now)

    def _start_relevel(self):
        err = self._roll_err()
        if err is None:
            self.log("[PLACE] ⚠️ 拿不到 roll 遥测 ⇒ 回不了正（球会滚出来）⇒ 停手")
            return False
        self._move = AxisMove("roll", err, name="roll", log=self.log, prefix="place")
        self.log("[PLACE] 回正 roll：相对角 %+.2f°（目标 %.1f°）" % (err, self._roll_target))
        return True
