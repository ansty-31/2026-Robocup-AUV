# -*- coding: utf-8 -*-
"""ball_forward.py — 撞球“简化版”：不做命中识别，只 搜索 → 居中 → 恒速前进

与 ball.py（BallTask）的区别：
  - **不做任何撞到/命中判定**（无 hit_cnt / dash / 近距消失判定）；
  - 看到球并居中对准后，以**恒定速度**前进（`ball_forward.forward_surge`，保持不变）；
  - **累计前进时间**达到 `ball_forward.forward_total_s`（默认 10s）→ 任务停止(DONE)。
  - 搜索/居中/PID/短时丢失惯性 复用 BallTask 的实现与 cfg/comm.yaml `ball:` 段参数；
    前进/计时参数在 `ball_forward:` 段。

任务名(name)保持 "ball"，以便 DetectorHub 复用撞球的类别选择（mission.target_color）。

用法：  python3 main.py --task ball_fwd
配置：  cfg/comm.yaml `ball_forward:`（forward_surge / forward_total_s / timeout_ms …）
"""
import base.settings as S
from task1_2.ball import BallTask


def CFG(name, default=None):
    """读 comm.ball_forward.<name>。"""
    return S.get("comm.ball_forward." + name, default)


class BallForwardTask(BallTask):
    name = "ball"          # 复用 DetectorHub 的撞球类别选择与后端就绪判定

    def __init__(self, uart, hub, frame_w, frame_h):
        super().__init__(uart, hub, frame_w, frame_h)
        self._last_ms = None       # 上一帧时刻（累计前进时间用）
        self._fwd_ms = 0           # 累计前进时间(ms)
        self.last_info["fwd_s"] = 0.0

    def process(self, frame, now_ms):
        self.frames += 1
        if self._start_ms is None:
            self._start_ms = now_ms
            self._last_ms = now_ms
        if not self.ready:
            return S.STATUS_DONE

        dt_ms = max(0, now_ms - self._last_ms)
        self._last_ms = now_ms

        det = self.hub.detect(self.name, frame)
        ratio = det.ratio(self.w, self.h) if det is not None else 0.0
        self._ema_update(ratio)            # 仅用于日志/画框

        surge = 0.0
        yaw = 0.0
        label = "search"
        dx = dy = sway = heave = 0.0
        forward = False                    # 本帧是否“向球前进”(计入前进时间)

        if det is not None:
            self._seen = True
            self._lost_cnt = 0
            self._search_start_ms = now_ms
            self._advance_until_ms = None
            self._spin_start_ms = None      # 重新看到目标：搜索脉冲从头开始
            cx, cy = det.center
            # 持续居中(yaw 旋转，不做左右平移)；前进时也保持居中
            dx, dy, sway, heave, yaw = self._center_ctl(cx, cy, now_ms)
            if not self._centered:
                eps = CFG("center_eps", S.comm.ball.center_eps)
                if abs(dx) <= eps and abs(dy) <= eps:
                    self._center_cnt += 1
                    if self._center_cnt >= CFG("center_confirm_frames",
                                               S.comm.ball.center_confirm_frames):
                        self._centered = True
                else:
                    self._center_cnt = 0
                surge = 0.0
                label = "center"            # 先居中，不前进
            else:
                surge = CFG("forward_surge", 0.5)   # 恒定速度前进
                label = "forward"
                forward = True
        else:
            if self._seen and self._lost_cnt < CFG("lost_grace_frames",
                                                   S.comm.ball.lost_grace_frames):
                self._lost_cnt += 1
                surge = CFG("forward_surge", 0.5)   # 短时丢失：恒速保持前进
                label = "forward"
                forward = True
            else:
                self._lost_cnt += 1
                self._centered = False      # 丢球：重新等待居中对准
                label, surge = self._search_logic(now_ms)

        if forward:
            self._fwd_ms += dt_ms

        # 下发：搜索用脉冲旋转；其余用当前 surge/yaw(居中) 等
        if label == "search":
            self.uart.send_dof(0.0, 0.0, 0.0, self._search_pulse_yaw(now_ms))
        else:
            self.uart.send_dof(surge, sway, heave, yaw)

        total_ms = CFG("forward_total_s", 10.0) * 1000
        status = S.STATUS_RUNNING
        reason = ""
        if self._fwd_ms >= total_ms:
            status, reason = S.STATUS_DONE, "forward_time"
            self.uart.neutral()             # 到时停车
        elif self._start_ms is not None and \
                now_ms - self._start_ms >= CFG("total_stop_s", 120.0) * 1000:
            # 总时限兜底：不管有没有撞到，至多跑这么久就停(防一直搜索像“卡死”)
            status, reason = S.STATUS_DONE, "total_stop"
            self.uart.neutral()
        elif self._search_exhausted:
            status, reason = S.STATUS_DONE, "search_stop"
        elif self._start_ms is not None and \
                now_ms - self._start_ms >= CFG("timeout_ms", 300000):
            status, reason = S.STATUS_DONE, "timeout"

        self.last_info.update({
            "action": label, "ratio": round(self._ema or 0.0, 4),
            "growth": 0.0, "dx": round(dx, 4), "dy": round(dy, 4),
            "sway": round(sway, 4), "heave": round(heave, 4),
            "surge": round(surge, 3), "yaw": round(yaw, 4),
            "fwd_s": round(self._fwd_ms / 1000.0, 2),
            "status": status, "reason": reason})
        if S.DEBUG and status == S.STATUS_DONE:
            print("[BALLF] DONE(%s) 前进累计=%.1fs frames=%d"
                  % (reason, self._fwd_ms / 1000.0, self.frames))
        return status
