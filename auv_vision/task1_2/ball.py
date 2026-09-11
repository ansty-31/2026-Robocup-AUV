# -*- coding: utf-8 -*-
"""ball.py — 任务一 撞球（BallTask）

撞球完成后按记忆返回（task1_2/return_by_memory.py），由 run_ball_return.sh 编排。
- PID 由 common/PID.py 提供（import 保持兼容）
- BallTask  任务一 撞球：面积占比分级调速 + 视觉居中 PID + 智能搜索
参数一律在 cfg/comm.yaml（settings 读取）
"""


import numpy as np

import base.settings as S
from common.PID import PID     # 兼容别名：tasks.PID = common.PID

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def _dx_norm(cx, w):
    """球心/目标中心 X 相对画面中心的归一化偏差。"""
    return (cx - w / 2.0) / (w / 2.0)


def _dy_norm(cy, h):
    return (cy - h / 2.0) / (h / 2.0)


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
        self._hit_cnt = 0
        self._search_start_ms = None   # 连续无目标搜索窗口起点
        self._advance_until_ms = None  # 慢速前进探测截止时刻
        self._search_exhausted = False # 总计 180s 未搜到 → 完全停止
        self._centered = False         # 球心已居中，允许前进
        self._center_cnt = 0           # 连续居中帧数
        self._spin_start_ms = None     # 搜索旋转脉冲起点
        self._engage_until_ms = 0      # 看到球后“切入保持”截止时刻(期间不打搜索脉冲)
        self._dashing = False          # 最后一次冲刺中(命中确认后居中+前进)
        self._dash_until_ms = 0        # 冲刺截止时刻
        self._hit_done = False         # 冲刺完成 = 命中(DONE)
        pid_kw = dict(kp=S.comm.ball.pid.kp, ki=S.comm.ball.pid.ki,
                      kd=S.comm.ball.pid.kd,
                      out_min=-S.comm.ball.pid.out_max,
                      out_max=S.comm.ball.pid.out_max,
                      deadzone=S.comm.ball.pid.deadzone)
        # 垂直居中(heave) 用 ball.pid；水平居中不再横移，只用下面的 yaw PID
        self._pid_y = PID(**pid_kw)
        # 居中 yaw：带阻尼(kd)的 PID，靠近时自动减速/刹车，避免转过头
        self._pid_yaw = PID(kp=S.comm.ball.edge_yaw_kp, ki=0.0,
                            kd=S.comm.ball.edge_yaw_kd,
                            out_min=-S.comm.ball.edge_yaw_max,
                            out_max=S.comm.ball.edge_yaw_max,
                            deadzone=0.0)
        self.last_info = {"action": "stop", "ratio": 0.0, "growth": 0.0,
                          "dx": 0.0, "dy": 0.0, "sway": 0.0, "heave": 0.0,
                          "surge": 0.0,
                          "status": S.STATUS_RUNNING, "reason": ""}

    @property
    def ready(self):
        return self.hub.ready(self.name)

    # ------------------------------------------------------------------ 辅助
    def _ema_update(self, r):
        if self._ema is None:
            self._ema = r
            return 0.0
        prev = self._ema
        self._ema = (1 - S.comm.ball.ema_alpha) * self._ema + \
                    S.comm.ball.ema_alpha * r
        return self._ema - prev

    def _surge_plan(self, r, g):
        """面积占比分级 → (前进目标, 阶段标签)。

        远/需加速：surge_fast；接近(r>=r_near)：surge_slow 减速；到 r_hit 停。
        """
        fast = S.comm.ball.surge_fast
        slow = S.comm.ball.surge_slow
        if r >= S.comm.ball.r_hit:
            return 0.0, "stop"
        if r >= S.comm.ball.r_near:
            return slow, "forward_slow"
        if r >= S.comm.ball.r_slow:
            if g > S.comm.ball.growth_eps:
                return fast, "forward_fast"
            return slow, "forward_slow"
        return fast, "forward_fast"

    def _center_ctl(self, cx, cy, now_ms):
        """球心偏差 → (dx, dy, sway, heave, yaw)。

        撞球居中**只用 yaw 旋转**（不需要左右平移，sway 恒为 0）；
        heave（升降）按 align_y 可选。
        """
        dx = (cx - self.w / 2.0) / (self.w / 2.0)
        dy = (cy - self.h / 2.0) / (self.h / 2.0)
        yaw = self._pid_yaw.update(dx, now_ms) if S.comm.ball.align_x else 0.0
        py = self._pid_y.update(dy, now_ms) if S.comm.ball.align_y else 0.0
        # 偏差在下 → 需向上修正：DOF 输出取反
        return dx, dy, 0.0, -py, yaw

    def _search_logic(self, now_ms):
        """无目标智能搜索：连续 60s 无目标→慢速前进 2s 探测新区域；
        总计 180s 仍未搜到→完全停止(neutral+DONE)。返回 (label, surge)。"""
        if self._search_start_ms is None:
            self._search_start_ms = now_ms
        if now_ms - self._start_ms >= S.comm.ball.search_total_stop_s * 1000:
            self._search_exhausted = True
            self.uart.neutral()
            return "stop", 0.0
        after_ms = S.comm.ball.search_advance_after_s * 1000
        dur_ms = S.comm.ball.search_advance_dur_s * 1000
        if self._advance_until_ms is not None:
            if now_ms < self._advance_until_ms:
                return "forward_slow", S.comm.ball.search_advance_surge
            self._advance_until_ms = None
            self._search_start_ms = now_ms
        if now_ms - self._search_start_ms >= after_ms:
            self._advance_until_ms = now_ms + dur_ms
            self._search_start_ms = now_ms
            return "forward_slow", S.comm.ball.search_advance_surge
        return "search", 0.0

    def _search_pulse_yaw(self, now_ms):
        """搜索旋转脉冲：转 search_spin_s，再停 search_pause_s。返回 yaw(0=停)。"""
        if self._spin_start_ms is None:
            self._spin_start_ms = now_ms
        spin_ms = S.comm.ball.search_spin_s * 1000
        pause_ms = S.comm.ball.search_pause_s * 1000
        ph = (now_ms - self._spin_start_ms) % (spin_ms + pause_ms)
        return S.comm.motion.search_yaw if ph <= spin_ms else 0.0

    # ------------------------------------------------------------------ 主入口
    def process(self, frame, now_ms):
        self.frames += 1
        if self._start_ms is None:
            self._start_ms = now_ms
        if not self.ready:
            return S.STATUS_DONE

        det = self.hub.detect(self.name, frame)
        ratio = det.ratio(self.w, self.h) if det is not None else 0.0
        growth = self._ema_update(ratio)

        surge = 0.0
        yaw = 0.0
        label = "search"
        dx = dy = sway = heave = 0.0
        if det is not None:
            self._seen = True
            self._lost_cnt = 0
            self._search_start_ms = now_ms
            self._advance_until_ms = None
            self._spin_start_ms = None     # 重新看到目标：搜索脉冲从头开始
            self._engage_until_ms = now_ms + S.comm.ball.engage_hold_s * 1000
            cx, cy = det.center
            dx = (cx - self.w / 2.0) / (self.w / 2.0)
            dy = (cy - self.h / 2.0) / (self.h / 2.0)
            if self._dashing:
                # 最后一次冲刺：居中(yaw) + 前进
                dx, dy, sway, heave, yaw = self._center_ctl(cx, cy, now_ms)
                if now_ms < self._dash_until_ms:
                    surge, label = S.comm.ball.dash_surge, "dash"
                else:
                    self._dashing = False
                    self._hit_done = True       # 冲刺完成 = 命中
                    surge, label = 0.0, "stop"
            elif not self._centered:
                # 看到球：先停(不前进)，只用 yaw 旋转把球心转到画面中间
                if abs(dx) <= S.comm.ball.center_eps and \
                        abs(dy) <= S.comm.ball.center_eps:
                    self._center_cnt += 1
                    if self._center_cnt >= S.comm.ball.center_confirm_frames:
                        self._centered = True
                else:
                    self._center_cnt = 0
                surge = 0.0
                label = "center"
                # 撞球居中：仅 yaw(旋转)，不做左右平移(sway 恒 0)
                yaw = self._pid_yaw.update(dx, now_ms) if S.comm.ball.align_x \
                    else 0.0
                sway = 0.0
                heave = -self._pid_y.update(dy, now_ms) if S.comm.ball.align_y \
                    else 0.0
            else:
                # 已居中：按面积分级前进 + 持续居中修正
                surge, label = self._surge_plan(self._ema or 0.0, growth)
                if label == "stop":
                    # 只是“贴近并停住”不算撞到：短时连续达阈值 → 最后一次冲刺
                    self._hit_cnt += 1
                    if self._hit_cnt >= S.comm.ball.hit_confirm_frames:
                        self._dashing = True
                        self._dash_until_ms = now_ms + \
                            S.comm.ball.dash_dur_s * 1000
                        surge, label = S.comm.ball.dash_surge, "dash"
                else:
                    self._hit_cnt = 0          # 非连续近距帧 → 重新计数
                if surge > 0:
                    dx, dy, sway, heave, yaw = self._center_ctl(cx, cy, now_ms)
                else:
                    self._pid_y.reset()
                    self._pid_yaw.reset()
        else:
            if self._dashing:
                # 冲刺中丢目标：盲冲到底(仅前进)
                if now_ms < self._dash_until_ms:
                    surge, label = S.comm.ball.dash_surge, "dash"
                else:
                    self._dashing = False
                    self._hit_done = True
                    surge, label = 0.0, "stop"
            elif self._seen and self._ema is not None and \
                    self._ema >= S.comm.ball.r_near:
                # 近距消失：不能算命中 → 低速后退
                self._hit_cnt = 0
                surge = S.comm.ball.backward_surge
                label = "backward"
            elif self._seen and self._lost_cnt < S.comm.ball.lost_grace_frames:
                self._lost_cnt += 1
                surge = 0.25                   # 短时丢失：低速保持惯性
                label = "forward_slow"
            elif now_ms < self._engage_until_ms:
                # 刚看到过球但短暂丢失：暂停搜索脉冲，原地保持(中性)等重新锁定
                self._lost_cnt += 1
                label = "hold"
            elif self._start_ms is not None and \
                    now_ms - self._start_ms < S.comm.ball.entry_look_s * 1000:
                # 入场“看一眼”：先原地保持、只识别不动作；看到球就按识别结果进 center
                label = "look"
            else:
                self._lost_cnt += 1            # 未见/长丢：原地旋转搜索
                self._centered = False         # 丢球：重新等待居中对准
                self._hit_cnt = 0
                label, surge = self._search_logic(now_ms)

        # 直接下发 DOF：前进+居中修正；搜索用脉冲旋转(转一下停一会儿)
        if label == "search":
            yaw = self._search_pulse_yaw(now_ms)
            self.uart.send_dof(0.0, 0.0, 0.0, yaw)
        else:
            # center 阶段可能带 yaw(球在画面边缘时转向球)，其余阶段 yaw=0
            self.uart.send_dof(surge, sway, heave, yaw)

        status = S.STATUS_RUNNING
        reason = ""
        if self._hit_done:
            status, reason = S.STATUS_DONE, "hit"
        elif self._search_exhausted:
            status, reason = S.STATUS_DONE, "search_stop"
        elif self._start_ms is not None and \
                now_ms - self._start_ms >= S.comm.ball.timeout_ms:
            status, reason = S.STATUS_DONE, "timeout"

        self.last_info.update({
            "action": label, "ratio": round(self._ema or 0.0, 4),
            "growth": round(growth, 4), "dx": round(dx, 4),
            "dy": round(dy, 4), "sway": round(sway, 4),
            "heave": round(heave, 4), "surge": round(surge, 3),
            "yaw": round(yaw, 4),
            "status": status, "reason": reason})
        if S.DEBUG and status == S.STATUS_DONE:
            print("[BALL] DONE(%s) %s r=%.3f sway=%.2f frames=%d"
                  % (reason, label, self._ema or 0, sway, self.frames))
        return status
