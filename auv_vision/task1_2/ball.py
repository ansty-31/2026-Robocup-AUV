# -*- coding: utf-8 -*-
"""ball.py — 任务一 撞球（BallTask）· 运动逻辑融合版

运动链（三段式运动移植自早期独立实验，**该实验目录已删除**；**视觉识别仍用本项目**）：

  SEARCH   无目标：原地**脉冲旋转**（我们的 spin_s/pause_s）+ 周期性慢速前进探测
           （转 spin_s → 停 pause_s，停的间隙让检测有静止帧）
  CENTER   看到球：**surge=0、sway=0**，仅 yaw(edge_yaw PID) + heave(ball.pid)，
           把球在**水平与竖直**同时居中（沿用我们的 yaw 控制方式）
  APPROACH 稳定居中后：surge **分级**前进（远 fast / 近 slow）+ **仅 sway** 做水平修正
           （approach_pid，移植自独立实验）；**yaw=0、heave=0**
  DASH     面积(EMA) ≥ dash_ratio 连续 dash_confirm_frames 帧 → **surge_fast**（分级里
           最高那档）冲刺 dash_dur_s，**不检测有没有撞到**
  STOP     冲刺结束 → 全 0 保持 stop_hold_s（**稳定停住**）→ DONE("hit")

丢目标（沿用原思路的阶梯）：短时丢失（帧窗口）→ 保持/惯性 → hold（原地全 0，
engage_hold_s）→ 回 SEARCH。**CENTER 阶段丢失绝不前进**（居中不允许带前进）。
时限：`comm.ball.timeout_ms`（当前 cfg 30s；DASH/STOP 期间不打断，保证命中与停稳都能走完）。

参数一律在 cfg/comm.yaml 的 `ball:` 段（settings 读取）。
"""
from __future__ import annotations

import base.settings as S
from common.PID import PID

PH_SEARCH = "SEARCH"
PH_CENTER = "CENTER"
PH_APPROACH = "APPROACH"
PH_DASH = "DASH"
PH_STOP = "STOP"


def _dx_norm(cx, w):
    """目标中心 X 相对画面中心的归一化偏差（±1 = 半屏）。"""
    return (cx - w / 2.0) / (w / 2.0)


def _dy_norm(cy, h):
    return (cy - h / 2.0) / (h / 2.0)


def _clip(v):
    return float(max(-1.0, min(1.0, v)))


class BallTask(object):
    name = "ball"

    def __init__(self, uart, hub, frame_w, frame_h):
        self.uart = uart
        self.hub = hub
        self.w = frame_w
        self.h = frame_h
        self.frames = 0
        self._start_ms = None
        self._ema = None
        self._seen = False
        self._lost_cnt = 0
        self._finished = False

        # 相位机状态
        self._phase = PH_SEARCH
        self._center_cnt = 0             # 连续居中帧数
        self._center_ready = False       # 居中已确认，下一帧才起步进近
        self._dash_cnt = 0               # 连续面积达标帧数
        self._dash_until_ms = 0          # 冲刺截止时刻
        self._stop_until_ms = 0          # 停稳保持截止时刻
        # SEARCH 计时（脉冲 + 周期性前进探测）
        self._search_start_ms = None     # 本轮"连续无目标"起点
        self._advance_until_ms = None    # 前进探测截止时刻
        self._spin_start_ms = None       # 旋转脉冲起点
        # 丢目标后的 hold 窗口
        self._hold_until_ms = None

        # 三套 PID（量纲都是"归一化偏差 ±1"）
        hk = dict(kp=S.comm.ball.pid.kp, ki=S.comm.ball.pid.ki,
                  kd=S.comm.ball.pid.kd,
                  out_min=-S.comm.ball.pid.out_max,
                  out_max=S.comm.ball.pid.out_max,
                  deadzone=S.comm.ball.pid.deadzone)
        # ① 居中 yaw（我们的方式：带阻尼 PID + 限幅，靠近自动减速）
        self._pid_yaw = PID(kp=S.comm.ball.edge_yaw_kp, ki=0.0,
                            kd=S.comm.ball.edge_yaw_kd,
                            out_min=-S.comm.ball.edge_yaw_max,
                            out_max=S.comm.ball.edge_yaw_max,
                            deadzone=0.0)
        # ② 居中 heave（垂直）
        self._pid_heave = PID(**hk)
        # ③ 接近 sway（水平；移植独立实验的调参）
        ak = dict(kp=S.comm.ball.approach_pid.kp, ki=S.comm.ball.approach_pid.ki,
                  kd=S.comm.ball.approach_pid.kd,
                  out_min=-S.comm.ball.approach_pid.out_max,
                  out_max=S.comm.ball.approach_pid.out_max,
                  deadzone=S.comm.ball.approach_pid.deadzone)
        self._pid_sway = PID(**ak)

        self.last_info = {"action": "stop", "phase": PH_SEARCH, "ratio": 0.0,
                          "growth": 0.0, "dx": 0.0, "dy": 0.0, "sway": 0.0,
                          "heave": 0.0, "surge": 0.0, "yaw": 0.0,
                          "status": S.STATUS_RUNNING, "reason": ""}
        self.last_dets = []          # 本帧检测结果（供 main._draw 画框，不再二次推理）

    @property
    def ready(self):
        return self.hub.ready(self.name)

    # ------------------------------------------------------------------ 辅助
    def _reset_pids(self):
        self._pid_yaw.reset()
        self._pid_heave.reset()
        self._pid_sway.reset()

    def _ema_update(self, r):
        """面积占比 EMA（抗单帧抖动）；返回 growth（本次增量）。"""
        if self._ema is None:
            self._ema = r
            return 0.0
        prev = self._ema
        self._ema = (1 - S.comm.ball.ema_alpha) * self._ema + \
                    S.comm.ball.ema_alpha * r
        return self._ema - prev

    def _surge_plan(self, r, g):
        """面积占比分级 → (前进速度, 标签)。接近段只用两档（fast/slow）。"""
        fast = S.comm.ball.surge_fast
        slow = S.comm.ball.surge_slow
        if r >= S.comm.ball.r_slow:
            if g > S.comm.ball.growth_eps:
                return fast, "approach_fast"
            return slow, "approach_slow"
        return fast, "approach_fast"

    def _search_pulse_yaw(self, now_ms):
        """搜索旋转脉冲：转 search_spin_s → 停 search_pause_s（停的间隙检测更稳）。"""
        if self._spin_start_ms is None:
            self._spin_start_ms = now_ms
        spin_ms = S.comm.ball.search_spin_s * 1000
        pause_ms = S.comm.ball.search_pause_s * 1000
        ph = (now_ms - self._spin_start_ms) % (spin_ms + pause_ms)
        return S.comm.motion.search_yaw if ph <= spin_ms else 0.0

    def _search_advance(self, now_ms):
        """连续无目标 search_advance_after_s → 慢速前进 search_advance_dur_s 探测新区域。"""
        if self._search_start_ms is None:
            self._search_start_ms = now_ms
        after_ms = S.comm.ball.search_advance_after_s * 1000
        dur_ms = S.comm.ball.search_advance_dur_s * 1000
        if self._advance_until_ms is not None:
            if now_ms < self._advance_until_ms:
                return "search_advance", S.comm.ball.search_advance_surge, 0.0
            self._advance_until_ms = None
            self._search_start_ms = now_ms
            self._spin_start_ms = now_ms        # 前进完，旋转脉冲重新计
        if now_ms - self._search_start_ms >= after_ms:
            self._advance_until_ms = now_ms + dur_ms
            self._search_start_ms = now_ms
            return "search_advance", S.comm.ball.search_advance_surge, 0.0
        return "search", 0.0, self._search_pulse_yaw(now_ms)

    # ------------------------------------------------------------------ 各相位
    def _step_search(self, now_ms):
        """SEARCH：脉冲旋转 + 周期性前进探测。"""
        return self._search_advance(now_ms)

    def _step_center(self, dx, dy, now_ms):
        """CENTER：只转 + 只升降（**不前进、不横移**），水平与竖直同时居中。"""
        yaw = _clip(self._pid_yaw.update(dx, now_ms))
        heave = _clip(-self._pid_heave.update(dy, now_ms))
        centered = abs(dx) <= S.comm.ball.center_eps and \
            abs(dy) <= S.comm.ball.center_eps
        self._center_cnt = self._center_cnt + 1 if centered else 0
        if self._center_cnt >= S.comm.ball.center_confirm_frames:
            # 居中确认：本帧仍属于 CENTER（输出与相位标签一致），下一帧才起步进近
            self._center_ready = True
            self._center_cnt = 0
            self._dash_cnt = 0
            return "center_done", 0.0, 0.0, heave, yaw
        return "center", 0.0, 0.0, heave, yaw

    def _step_approach(self, dx, now_ms, r, g):
        """APPROACH：分级前进 + **仅 sway** 修水平（yaw=0、heave=0）。"""
        surge, label = self._surge_plan(r, g)
        sway = _clip(self._pid_sway.update(dx, now_ms))
        if r >= S.comm.ball.dash_ratio:
            self._dash_cnt += 1
            if self._dash_cnt >= S.comm.ball.dash_confirm_frames:
                self._phase = PH_DASH
                self._dash_until_ms = now_ms + S.comm.ball.dash_dur_s * 1000
                self._dash_cnt = 0
                self._pid_sway.reset()
                # 冲刺帧就是纯前进（不带上一帧的横移修正）
                return "dash", S.comm.ball.surge_fast, 0.0, 0.0, 0.0
        else:
            self._dash_cnt = 0
        return label, surge, sway, 0.0, 0.0

    def _step_dash(self, now_ms):
        """DASH：纯前进冲刺（不检测）；到时 → STOP 保持。"""
        if now_ms < self._dash_until_ms:
            return "dash", S.comm.ball.surge_fast
        self._phase = PH_STOP
        self._stop_until_ms = now_ms + S.comm.ball.stop_hold_s * 1000
        self._reset_pids()
        return "dash_done", 0.0

    def _step_stop(self, now_ms):
        """STOP：全 0 保持（稳定停住），保持够久 → DONE。"""
        if now_ms < self._stop_until_ms:
            return "stop", 0.0
        self._finished = True
        return "stop", 0.0

    def _step_lost(self, now_ms):
        """丢目标阶梯（沿用原思路）：短时保持/惯性 → hold(原地全 0) → 回 SEARCH。

        SEARCH 相位下（含从未见过球、或阶梯走完回到搜索）→ 继续搜索，不再套 hold。
        """
        if self._phase == PH_SEARCH:
            return self._search_advance(now_ms)
        grace = S.comm.ball.lost_grace_frames
        if self._lost_cnt <= grace:
            if self._phase == PH_APPROACH:
                # 接近中短时丢失：低速惯性保持（还在前进段，允许前进）
                return "approach_inertia", S.comm.ball.lost_inertia_surge, 0.0
            # SEARCH / CENTER：原地保持（**居中段绝不前进**）
            return "hold", 0.0, 0.0
        if self._hold_until_ms is None:
            self._hold_until_ms = now_ms + S.comm.ball.engage_hold_s * 1000
        if now_ms < self._hold_until_ms:
            return "hold", 0.0, 0.0
        # 保持窗口也过了：回搜索重新找
        self._hold_until_ms = None
        self._spin_start_ms = now_ms
        self._search_start_ms = now_ms
        self._phase = PH_SEARCH
        self._center_ready = False
        return self._search_advance(now_ms)

    # ------------------------------------------------------------------ 主入口
    def process(self, frame, now_ms):
        self.frames += 1
        if self._start_ms is None:
            self._start_ms = now_ms
        if not self.ready:
            return S.STATUS_DONE

        det = self.hub.detect(self.name, frame)
        self.last_dets = self.hub.detect_all(frame)   # 同帧缓存命中, 零额外开销
        ratio = det.ratio(self.w, self.h) if det is not None else 0.0
        growth = self._ema_update(ratio)

        dx = dy = 0.0
        if det is not None:
            self._seen = True
            self._lost_cnt = 0
            self._hold_until_ms = None
            cx, cy = det.center
            dx = _dx_norm(cx, self.w)
            dy = _dy_norm(cy, self.h)
        else:
            self._lost_cnt += 1

        label = "hold"
        surge = yaw = sway = heave = 0.0

        if self._phase == PH_DASH:
            label, surge = self._step_dash(now_ms)
        elif self._phase == PH_STOP:
            label, surge = self._step_stop(now_ms)
        elif det is None:
            label, surge, yaw = self._step_lost(now_ms)
        else:
            if self._phase == PH_SEARCH:
                # 看到球 → 进入居中（先复位 PID，本帧即开始对准）
                self._phase = PH_CENTER
                self._center_cnt = 0
                self._center_ready = False
                self._reset_pids()
            elif self._phase == PH_CENTER and self._center_ready:
                # 上一帧居中已确认 → 本帧开始进近（sway 环路从零开始）
                self._phase = PH_APPROACH
                self._center_ready = False
                self._pid_sway.reset()
            if self._phase == PH_CENTER:
                label, surge, sway, heave, yaw = \
                    self._step_center(dx, dy, now_ms)
            else:
                label, surge, sway, heave, yaw = \
                    self._step_approach(dx, now_ms, self._ema or 0.0, growth)

        # 下发 DOF：搜索用脉冲旋转；其余（含 CENTER 的 yaw/heave、APPROACH 的 sway）直发
        if label in ("search", "search_advance"):
            self.uart.send_dof(surge, 0.0, 0.0, yaw)
        else:
            self.uart.send_dof(surge, sway, heave, yaw)

        status = S.STATUS_RUNNING
        reason = ""
        if self._finished:
            status, reason = S.STATUS_DONE, "hit"
            self.uart.neutral()                 # 命中：明确回中位（STOP 已保持过）
        elif self._phase not in (PH_DASH, PH_STOP) and self._start_ms is not None \
                and now_ms - self._start_ms >= S.comm.ball.timeout_ms:
            # 总时限（DASH/STOP 期间不打断：命中与停稳必须走完）
            status, reason = S.STATUS_DONE, "timeout"
            self.uart.neutral()

        self.last_info.update({
            "action": label, "phase": self._phase,
            "ratio": round(self._ema or 0.0, 4), "growth": round(growth, 4),
            "dx": round(dx, 4), "dy": round(dy, 4), "sway": round(sway, 4),
            "heave": round(heave, 4), "surge": round(surge, 3),
            "yaw": round(yaw, 4), "status": status, "reason": reason})
        if S.DEBUG and status == S.STATUS_DONE:
            print("[BALL] DONE(%s) phase=%s r=%.3f frames=%d"
                  % (reason, self._phase, self._ema or 0.0, self.frames))
        return status
