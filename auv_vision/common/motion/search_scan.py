# -*- coding: utf-8 -*-
"""common/motion/search_scan.py — 公共搜索扫描（撞球 / 过门 / 未来夹球共用）
⚠️ **推力下限（别踩）**：执行器死区 **0.138**。`pid.out_max` 取 0.15 会
⚠️ **下发频率 = 主任务帧率**（"睁眼"）：`step()` 就是主循环每帧调一次（门/球 ~10Hz），
（详细用法、判据与实测见 doc/注释历史.md）"""
from __future__ import annotations

import math

import base.cfg.settings as S
from common.cfg.cfgnode import merge, num, pid_kw, sub, req, req_node, MissingCfg
from common.motion.PID import PID
from common.motion.turn_deg import yaw_sign

# 兜底表（取值 = 当前 cfg/comm.yaml 的 motion.search_scan 段）
_K_SCAN_PID = ("kp", "ki", "kd", "out_max", "deadzone")
_K_SCAN = (
    "span_deg", "pause_ms", "tol_deg", "norm_deg", "seg_timeout_ms",
    "runaway_slack_deg", "abs_slack_deg", "duty", "duty_window_ms",
    "step_deg", "step_pause_ms", "pid",
)


def telemetry_yaw(uart):
    """本帧遥测航向（闭环量）。拿不到 → None（`Scan` 因此**不转**，不乱扫）。"""
    try:
        return uart.telemetry.yaw_deg
    except Exception:
        return None


class Scan(object):
    """非阻塞慢扫状态机。每帧 `step(now_ms, yaw_telemetry)` → 返回 yaw DOF。"""

    def __init__(self, node=None, log=None):
        self._node = node
        self.log = log or (lambda *a: None)
        self.sigma = float(yaw_sign()[0])       # σ：psi = σ × 遥测 yaw
        self.reset()

    # ------------------------------------------------------------------ 配置
    def _cfg(self):
        # ⚠️ `sub(node, key)` **只吃单个 key，不吃点号路径** —— 写成
        base = self._node if self._node is not None else \
            sub(sub(S.comm, "motion"), "search_scan")
        return base          # ★ 无兜底：下面一律 req()（缺键即报名字）

    # ------------------------------------------------------------------ 状态
    def reset(self, yaw_telemetry=None):
        """复位（换目标/重新搜索时调）。`yaw_telemetry` 给了就用它当入场朝向。"""
        c = self._cfg()
        self.span = float(req(c, "span_deg"))
        self.pause_ms = float(req(c, "pause_ms"))
        self.tol = float(req(c, "tol_deg"))
        self.norm = max(1e-6, float(req(c, "norm_deg")))
        self.seg_timeout_ms = float(req(c, "seg_timeout_ms"))
        self.slack = float(req(c, "runaway_slack_deg"))
        self.abs_slack = float(req(c, "abs_slack_deg"))
        self.duty = max(0.0, min(1.0, float(req(c, "duty"))))
        self.duty_window_ms = max(1.0, float(req(c, "duty_window_ms")))
        self.step_deg = max(0.0, float(req(c, "step_deg")))
        self.step_pause_ms = max(0.0, float(req(c, "step_pause_ms")))
        self._sub_idx = 0        # 本段已走到第几格（step_deg>0 时用）
        self._step_pause_until = None
        self._pid = PID(**pid_kw(req_node(c, "pid"), "comm.motion.search_scan.pid"))
        self.psi0 = None if yaw_telemetry is None else self.sigma * float(yaw_telemetry)
        self._leg = 0             # 已完成/正在走的段号（0 起）
        self._target = None       # 当前段目标 psi（相对入场朝向）
        self._seg_t0 = None
        self._leg_psi0 = None      # 本段起始 psi（防转圈预算的基准）
        self._pause_until = None
        self.legs_done = 0        # 走过的段数（诊断用）

    @property
    def target_psi(self):
        """当前段的目标 psi（None = 还没定）。"""
        return self._target

    @property
    def span_total_deg(self):
        """总扫幅（度）= 2×span_deg。"""
        return 2.0 * self.span

    # ------------------------------------------------------------------ 每帧
    def step(self, now_ms, yaw_telemetry):
        """推进一帧，返回 yaw DOF（**+ = 右转**）。"""
        if yaw_telemetry is None:
            return 0.0
        psi = self.sigma * float(yaw_telemetry)
        if self.psi0 is None:
            self.psi0 = psi                     # 入场朝向 = 第一次拿到的朝向
            self.log("[SCAN] 入场朝向 psi0=%.1f°（span=±%.0f° ⇒ 扫 %.0f° 视角）"
                     % (self.psi0, self.span, self.span_total_deg))
        # ---- 段间停顿 ----
        if self._pause_until is not None:
            if now_ms < self._pause_until:
                return 0.0
            self._pause_until = None
        # ---- 开新段：-span, +span, -span, … ----
        if self._target is None:
            sign = -1.0 if (self._leg % 2 == 0) else 1.0
            self._target = self.psi0 + sign * self.span
            self._seg_t0 = now_ms
            self._leg_psi0 = psi
            self._sub_idx = 0
            self._step_pause_until = None
            self._pid.reset()
            self.log("[SCAN] 第 %d 段 → psi=%.1f°（相对入场 %+.0f°）"
                     % (self._leg + 1, self._target, sign * self.span))
        # ---- 本帧**有效目标**：分段逼近时按 step_deg 一格一格走 ----
        eff = self._target
        if self.step_deg > 0:
            d = self._target - self.psi0
            total = max(1, int(math.ceil(abs(d) / self.step_deg - 1e-9)))
            k = min(self._sub_idx + 1, total)
            eff = self.psi0 + math.copysign(min(abs(d), self.step_deg * k), d)
        self._eff_target = eff
        # 格间停顿
        if self._step_pause_until is not None:
            if now_ms < self._step_pause_until:
                return 0.0
            self._step_pause_until = None
        # ---- PID 收敛到（有效）目标 ----
        err = eff - psi
        if abs(err) <= self.tol and eff is not self._target and abs(eff - self._target) > 1e-9:
            self.log("[SCAN] 第 %d 段·第 %d 格到位（psi=%.1f° → %.1f°）"
                     % (self._leg + 1, self._sub_idx + 1, eff, self._target))
            self._sub_idx += 1
            if self.step_pause_ms > 0:
                self._step_pause_until = now_ms + self.step_pause_ms
            return 0.0
        if abs(err) <= self.tol:
            self.log("[SCAN] 第 %d 段到位（残差 %+.1f°）→ 停 %.0fms"
                     % (self._leg + 1, err, self.pause_ms))
            self._leg += 1
            self.legs_done = self._leg
            self._target = None
            self._pause_until = now_ms + self.pause_ms
            return 0.0
        abs_budget = self.span + self.tol + self.abs_slack
        if self.psi0 is not None and abs(psi - self.psi0) > abs_budget:
            self.log("[SCAN] ⚠ psi 离入场朝向 %.0f° 超绝对预算 %.0f° ⇒ 停手（防走圈）"
                     % (abs(psi - self.psi0), abs_budget))
            self._leg += 1
            self.legs_done = self._leg
            self._target = None
            self._pause_until = now_ms + self.pause_ms
            return 0.0
        budget = 2.0 * self.span + self.slack
        if self._leg_psi0 is not None and abs(psi - self._leg_psi0) > budget:
            self.log("[SCAN] ⚠ 本段已转 %.0f° 仍未到位（预算 %.0f°）⇒ 判为极性/控制异常，停手"
                     % (abs(psi - self._leg_psi0), budget))
            self._leg += 1
            self.legs_done = self._leg
            self._target = None
            self._pause_until = now_ms + self.pause_ms
            return 0.0
        if (now_ms - self._seg_t0) > self.seg_timeout_ms:
            self.log("[SCAN] 第 %d 段超时（残差 %+.1f°）→ 停 %.0fms 后换向"
                     % (self._leg + 1, err, self.pause_ms))
            self._leg += 1
            self.legs_done = self._leg
            self._target = None
            self._pause_until = now_ms + self.pause_ms
            return 0.0
        out = float(self._pid.update(err / self.norm, now_ms))
        if self.duty < 1.0:
            # ★ 慢速占空比：本窗口内只推 duty 那一部分时间，其余发 0（幅值不动 ⇒ 不掉死区）
            ph = (now_ms % self.duty_window_ms) / self.duty_window_ms
            if ph >= self.duty:
                out = 0.0
        return out
