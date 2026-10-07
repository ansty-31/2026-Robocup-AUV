# -*- coding: utf-8 -*-
"""handling/motion/actions.py — 两个流程共用的**相位常量与兜底值表**。
（方法体一字未改，只搬位置。）"""
from __future__ import annotations

import math

import base.cfg.settings as S
from common.cfg.cfgnode import flag, merge, motion_pid, num, pid_kw, sub, req, req_flag, req_node, MissingCfg
from common.motion.PID import PID
from common.motion.search_sweep import duty_gate
from common.motion.axis import AxisMove, EnsureDepth, TimedDof, _K_DIP, _K_RISE, action_cfg, axis_deg, grab_cfg, grab_node, level_relative_deg, pitch_up_deg
from common.motion.axis import AxisMove, TimedDof, ZERO_DOF, axis_deg, level_relative_deg, task_node
from handling.motion.params import _K_CENTER, _K_APPROACH, _K_ALIGN


def _cvnode(key):
    """`comm.grab.<key>`：**必填**（★ 2026-10-07 删掉兜底；缺则报名字）。"""
    return req_node(grab_node(), key)


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

    def _new_pid(self, node, where=""):
        """★ pid_kw 现在**必填**（无兜底）；`where` 只用于缺键时报错指路。"""
        return PID(**pid_kw(node, where))

    #: ★ `surge`（dy 通道）默认取**共用 sway 的一半**：用户 2026-10-07 定「surch 的 pid 可以小一点」。
    _D_CENTER_SURGE_RATIO = 0.5

    def _lost_predict_dof(self, dxp, dyp, scale):
        """丢目标时按**历史推演位置**给的小幅修正 → `(surge, sway, 0, 0)`。

        口径（用户 2026-10-07：对比前几帧位置推演，防止球丢失找不到）：
        * 纯**比例 + 硬上限**，不引入积分 —— 不确定的信息上挂积分会攒出大输出；
        * 幅值再乘 `scale`（`TrackMemory` 给的时效折减，越久没看到越小）；
        * `heave`/`yaw` 恒 0（深度归定深、夹取不旋转）；**不推进任何相位判据**
          （喂合成值进 `_step_center/_approach` 会误判"已居中"→ 盲冲）。
        """
        l = _cvnode("lost")
        k = num(l, "predict_kp", 2.0)
        sw_max = abs(num(l, "predict_sway_max", 0.25))
        su_max = abs(num(l, "predict_surge_max", 0.12))
        s = max(0.0, min(1.0, float(scale)))
        sway = max(-sw_max, min(sw_max, k * float(dxp))) * s
        surge = max(-su_max, min(su_max, k * float(dyp))) * s
        return surge, sway, 0.0, 0.0

    def _center_pid_kw(self, which):
        """居中段两个通道的 PID 参数。

        * 基准 = **本任务自己那套** `comm.grab.pid_sway`（用户 2026-10-07：
          「夹取的 sway 单独设置一个，小一点的」；缺键才退回共用 `comm.motion.pid_sway`）；
        * sway 直接用基准；surge 再乘 `center.surge_ratio`（默认 0.5 —— 用户："surch 的 pid 小一点"），
          **死区不缩**（死区是执行器物理量 ~0.138，缩了会推不动）；
        * 要整体换掉某路：写 `comm.grab.center.sway` / `.surge`（整块覆盖，含增益）。
        """
        c = _cvnode("center")
        ov = sub(c, which) if isinstance(c.get(which), dict) else None
        if ov:
            return pid_kw(ov, "comm.grab.center.%s" % which), "comm.grab.center.%s" % which
        base, where = self._grab_sway_kw()
        if which == "sway":
            return dict(base), where
        ratio = num(c, "surge_ratio", self._D_CENTER_SURGE_RATIO)
        kw = dict(base)
        for k in ("kp", "ki", "kd"):
            if kw.get(k) is not None:
                kw[k] = float(kw[k]) * ratio
        if kw.get("out_max") is not None:
            kw["out_max"] = float(kw["out_max"]) * ratio
        return kw, "（%s）× surge_ratio %.2f（用户：surge 的 pid 小一点）" % (where, ratio)

    #: 各相位**前进**占空比的缺省（横向一律用 `duty`）：
    #:   居中要**极小**（小球很小 + 该机前进推力大，用户 2026-10-07）；进近**不额外削**（要能接近）。
    _D_DUTY_SURGE_PHASE = {"center": 0.25, "align": 0.35, "approach": 1.0}

    def _grab_duty(self, surge, sway, now_ms, phase=None):
        """按 `comm.grab.duty` 给**平移**做脉冲闸门（用户 2026-10-07：速度还是太快）。

        * 只作用于 sway/surge；`heave` 归定深、`yaw` 恒 0 ⇒ 不碰；
        * `enable: false` 或 `duty: 1.0` ⇒ 原样通过（一行退回旧行为）；
        * 幅值**不变** ⇒ 不会掉到执行器死区之下（这是"能真的更慢"的唯一办法）。
        """
        d = _cvnode("duty")
        if not flag(d, "enable", True):
            return surge, sway
        w = num(d, "duty_window_ms", 250.0)
        k = num(d, "duty", 1.0)
        # ★ 前进按**相位**再削（用户 2026-10-07：「前进的动力再小还是太大」/「居中的时候前进速度要很小」）：
        #   幅值已贴着执行器死区（再减就完全不动）⇒ 只能靠占空比降平均速度；
        #   居中(center)最稀、对准(align)次之、进近(approach)默认不额外削（否则永远接近不了）。
        key = {"CENTER": "center", "ALIGN": "align", "APPROACH": "approach"}.get(str(phase), None)
        ks = num(d, "duty_surge", k)
        if key:
            ks = num(d, "duty_surge_%s" % key,
                     self._D_DUTY_SURGE_PHASE.get(key, ks))
        # 前进还可以用**更短的脉冲窗口** ⇒ 每一下只推动一小步（该机前进推力大，用户 2026-10-07：
        #   「前进的动力再小，还是太大了」/「前进的死区不局限于此」—— 别拿 0.138 当"够慢"的理由）。
        w_surge = num(d, "duty_window_ms_surge", w)
        return (duty_gate(surge, now_ms, ks, w_surge), duty_gate(sway, now_ms, k, w))

    def _grab_sway_kw(self):
        """**夹取任务自己的横向 PID**：`comm.grab.pid_sway` → 否则共用 `comm.motion.pid_sway`。

        与 gate 的 `comm.gate.pid_sway | comm.motion.pid_sway` 同一条路子（任务段只在
        "确实要单独一套"时才写覆盖键）。
        """
        ov = grab_node().get("pid_sway")
        if isinstance(ov, dict):
            return dict(pid_kw(ov, "comm.grab.pid_sway")), "comm.grab.pid_sway（本任务自己那套）"
        return dict(motion_pid("pid_sway")), "comm.motion.pid_sway（缺任务覆盖时的共用那份）"

    def _ensure_pids(self):
        if self._pid_yaw is None:
            # ★ 名字仍叫 `_pid_yaw`（历史），但**通道是 sway**：用户 2026-10-06 定
            #   「夹取任务不需要旋转」⇒ 居中用平移，不再发 yaw。
            #   2026-10-07：居中两路 = dx→sway（与 gate/ball 同源）+ dy→surge（小一点）。
            kw, where = self._center_pid_kw("sway")
            self._pid_yaw = self._new_pid(kw, where)
            kw2, where2 = self._center_pid_kw("surge")
            self._pid_csu = self._new_pid(kw2, where2)
            # ★ 横向（sway）**三个平移相位共用本任务自己那套** `comm.grab.pid_sway`
            kw_s, where_s = self._grab_sway_kw()
            self._pid_sway = self._new_pid(dict(kw_s), where_s)      # APPROACH 用
            self._pid_asw = self._new_pid(dict(kw_s), where_s)       # ALIGN 用
            l = _cvnode("align_level")
            self._pid_asu = self._new_pid(req_node(l, "surge"), "handling.align.surge")

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
