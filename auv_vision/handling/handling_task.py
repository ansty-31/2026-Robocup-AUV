# -*- coding: utf-8 -*-
"""handling/handling_task.py — 「夹取 + 放置」总调度（原 `grab/grab_task.py` + `place/place_task.py`）。"""
from __future__ import annotations

import base.cfg.settings as S
from common.cfg.cfgnode import flag, merge, pid_kw, sub
from common.motion.search_scan import Scan, telemetry_yaw
from common.motion.axis import AxisMove, EnsureDepth, TimedDof, _D_DIP, _D_RISE, action_cfg, axis_deg, grab_cfg, grab_node, level_relative_deg, pitch_up_deg
from handling.percept.cage_color import red_percent
from common.motion.axis import AxisMove, TimedDof, ZERO_DOF, axis_deg, level_relative_deg, task_node
from handling.motion.params import (PH_GRAB_INIT,
                                  PH_GRAB_PITCH_UP,
                                  PH_GRAB_SEARCH,
                                  PH_GRAB_CENTER,
                                  PH_GRAB_APPROACH,
                                  PH_GRAB_LEVEL,
                                  PH_GRAB_ALIGN,
                                  PH_GRAB_DIP,
                                  PH_GRAB_RISE,
                                  PH_GRAB_VERIFY,
                                  PH_GRAB_DUMP,
                                  PH_GRAB_EXIT,
                                  PH_GRAB_DONE,
                                  _D_VERIFY,
                                  _D_LOST,
                                  PH_PLACE_INIT,
                                  PH_PLACE_TRANSPORT,
                                  PH_PLACE_RELEASE,
                                  PH_PLACE_STOP,
                                  PH_PLACE_DONE)
from handling.motion.actions import HandlingActions, _cvnode, _dx_norm, _dy_norm
from handling.motion.phases import GrabPhases, PlacePhases
from handling.motion.params import place_cfg


# 兼容出口：相位常量与兜底表在 `motion/params.py`
from handling.motion.params import *                    # noqa: F401,F403
from handling.motion.params import (                    # noqa: F401
    _D_CENTER, _D_APPROACH, _D_ALIGN, _D_VERIFY, _D_RETRY, _D_LOST, _D_PLACE, place_cfg)
from handling.motion.actions import (_cvnode, _ratio_radius, _dx_norm, _dy_norm, _clip)  # noqa: F401


class HandlingTask(HandlingActions, GrabPhases, PlacePhases):
    """任务「夹取 + 放置」总调度：`mode="grab"|"place"|"full"`，同一实例可交接。

    相位/关键动作/兜底值分别拆在 `motion/{phases,actions,params}.py`；感知在 `percept/`。
    方法体一字未改，只有 `__init__` / `process` 分发 / 接口属性是三处必须合写的。
    """

    name = "handling"

    def __init__(self, uart, hub=None, frame_w=0, frame_h=0, log=None, mode="grab"):
        """`mode="grab"` 只跑夹取；`"place"` 只跑放置；`"full"` = 先夹取后放置（同一实例交接）。"""
        if mode not in ("grab", "place", "full"):
            raise ValueError('mode 只能是 grab|place|full（当前 %r）' % mode)
        self.mode = mode
        self._stage = "place" if mode == "place" else "grab"
        # ── 装配面（两个流程共用）──
        self.uart = uart
        self.hub = hub
        self.w, self.h = int(frame_w), int(frame_h)
        self.log = log or (lambda *a: None)
        self.frames = 0
        # ── 共用相位骨架 ──
        self._phase = PH_PLACE_INIT if self._stage == "place" else PH_GRAB_INIT
        self._sub = ""
        self._start_ms = None
        self._now_ms = 0
        self._finished = False
        self._reason = ""
        self._ok = False                       # 夹取侧：笼里（自认）有球
        self._floor_raised = False             # 本任务是否抬过限深下限
        self._move = None
        self._timed = None
        self._drop = None                      # 横倾放球序列（两个流程共用一个组件）
        # ── 夹取侧专有 ──
        self._down_frame = None                # 下视帧（装配层每帧注入）
        self._attempts = 0
        self._dumps = 0
        self._pitched = False                  # 机头是否还仰着（决定退出前要不要放平）
        self._exit_reason, self._exit_ok = "", False
        self._exit_deadline = 0.0
        self._ref_pitch = None                 # 抬头前的 pitch 遥测 = 回水平基准
        self._ensure = None
        self._scan = Scan(log=self.log)
        self._ema = None
        self._lost_cnt = 0
        self._hold_until_ms = None
        self._hit_cnt = 0
        self._align_cnt = 0
        self._verify_cnt = 0
        self._pid_yaw = None                   # 判据 PID（进相位时 reset，不重建）
        self._pid_sway = None
        self._pid_asw = None
        self._pid_asu = None
        self._warned = False
        self.last_dets = []
        # ── 放置侧专有 ──
        self._arrived = False                  # 外部判「到点」（正式入口 `arrived()`）
        self._roll_target = 0.0
        self._last_relevel_ms = None
        self._holds_ball = False               # 交接：笼里有没有球（`full` 模式由夹取段自动置位）
        self.last_info = {"action": "init", "phase": self._phase, "ratio": 0.0,
                          "dx": 0.0, "dy": 0.0, "sway": 0.0, "surge": 0.0,
                          "heave": 0.0, "yaw": 0.0, "percent": -1.0,
                          "status": S.STATUS_RUNNING, "reason": "", "mode": mode}

    # ────────────────────────────────────────────────────────── 模式与分发
    def _enter_place(self):
        """夹取成功 → 同一实例切到放置流程（限深/收尾都由本任务继续持有）。"""
        self._stage = "place"
        self._holds_ball = bool(self._ok)
        self._phase = PH_PLACE_INIT
        self._sub = ""
        self._finished = False
        self._reason = ""
        self._start_ms = None                  # 放置流程自己的超时基准
        self._move = None
        self._timed = None
        self._arrived = False
        self._roll_target = 0.0
        self.last_info["mode"] = self.mode
        self.log("[HANDLING] 夹取完成 → 进入放置流程")

    def process(self, frame, now_ms):
        """每帧一次：按 `mode` 分发（`grab` / `place` / `full`）。"""
        if self._stage == "place":
            return self._process_place(frame, now_ms)
        st = self._process_grab(frame, now_ms)
        if self.mode == "full" and st == S.STATUS_DONE and self._ok:
            self._enter_place()
            return S.STATUS_RUNNING
        return st

    # ────────────────────────────────────────────────────────── 装配面（模式感知）
    @property
    def calibrated(self):
        """`calibrated: false ⇒ 不 ready ⇒ 装配层跳过`。`full` 要求两边都标定过。"""
        if self.mode == "place":
            return bool(place_cfg()["calibrated"])
        g = bool(flag(grab_node(), "calibrated", False))
        return g if self.mode == "grab" else (g and bool(place_cfg()["calibrated"]))

    @property
    def ready(self):
        """未标定 ⇒ 不 ready（机械保障：没设计完的任务不占状态机）。"""
        if not self.calibrated and not self._warned:
            self._warned = True
            self.log("[HANDLING] ⚠️ calibrated=false ⇒ 本任务不运行（mode=%s）" % self.mode)
        return self.calibrated

    @property
    def holds_ball(self):
        """笼里（自认）有球吗。夹取侧 = 本任务夹取成功；放置侧 = 交接方设置（`set_holds_ball`）。"""
        return bool(self._holds_ball) if self._stage == "place" else bool(self._ok)

    def set_holds_ball(self, yes):
        """交接：告诉放置流程笼里有没有球（原 `PlaceTask.holds_ball(yes)`，改名以免与只读属性冲突）。"""
        self._holds_ball = bool(yes)
        return self._holds_ball

    @property
    def placed(self):
        """放置完成（`_ok`）—— 装配层/上位机据此收尾。"""
        return bool(self._ok)

    def arrived(self, yes=True):
        """外部判「到点」（装配层/上位机）。`transport_s` 只是占位判据。"""
        self._arrived = bool(yes)
        if self._arrived:
            self.log("[PLACE] 收到「到点」信号 → 准备放球")
        return self._arrived

    def set_down_frame(self, frame):
        """装配层每帧注入下视帧（`DownCamFeeder`）。"""
        self._down_frame = frame

    def wants_down(self):
        """要不要下视帧：夹取/全程要；只放置时不需要感知。"""
        return self.mode != "place"

    # ────────────────────────────────────────────────────────── 收尾（两流程合并）
    def _bail(self, reason, ok=False):
        """**立即**收尾（不尝试放平）：夹取侧=还没抬头/急停没法驱动；放置侧=任何异常退出。"""
        if self._stage == "place":
            self._finished = True
            self._reason = reason
            self._phase = PH_PLACE_DONE
            self._restore_floor()
            try:
                self.uart.neutral()
            except Exception:
                pass
            self.last_info.update({"action": "abort", "phase": PH_PLACE_DONE, "reason": reason,
                                   "status": S.STATUS_DONE})
            self.log("[PLACE] 结束：%s" % reason)
            return S.STATUS_DONE
        return self._finalize(reason, ok)

    def _finalize(self, reason, ok=False):
        """**真正**收尾：撤回任务级限深下限 + 停手 + 落状态。幂等。"""
        if self._finished:
            return S.STATUS_DONE
        self._finished = True
        self._reason = reason
        self._ok = bool(ok)
        self._phase = PH_GRAB_DONE
        setter = getattr(self.uart, "set_extra_min_depth", None)
        if callable(setter) and self._floor_raised:
            setter(0.0)                    # 撤回：本任务的收紧只在本任务有效
            self._floor_raised = False
        try:
            self.uart.neutral()
        except Exception:
            pass
        self.log("[GRAB] 结束：%s（笼里%s目标球；尝试 %d 次；机头%s）"
                 % (reason, "有" if ok else "没有", self._attempts,
                    "已放平" if not self._pitched else "仍仰着 ⚠️"))
        self.last_info.update({"action": "abort", "phase": PH_GRAB_DONE,
                               "status": S.STATUS_DONE, "reason": reason,
                               "holds_ball": bool(ok)})
        return S.STATUS_DONE

    def _finish(self, reason, ok=False):
        """收尾（幂等）：**若机头还仰着，先走 PH_GRAB_EXIT 放平再结束**。

        为什么：抬头 30° 是任务的**前置姿态**，直接结束会把 30° 俯仰和"机头贴着水面"的状态
        留给下一个任务（可能把机头顶出水面 ⇒ 比赛立即停止）。放平带超时兜底，不会卡住。
        """
        if self._finished:
            return S.STATUS_DONE
        if self._phase == PH_GRAB_EXIT:
            return S.STATUS_RUNNING
        if self._pitched:
            rel = level_relative_deg(self._ref_pitch, axis_deg(self.uart, "pitch"))
            if rel is not None:
                self._exit_reason, self._exit_ok = reason, bool(ok)
                self._phase = PH_GRAB_EXIT
                self._move = AxisMove("pitch", rel, name="level", log=self.log)
                self._exit_deadline = self._now_ms + float(
                    S.get("comm.grab.level.timeout_s", 8.0) or 8.0) * 1000.0
                self.log("[GRAB] 结束前先把机头放平（原因：%s，相对角 %+.2f°）" % (reason, rel))
                return S.STATUS_RUNNING
            self.log("[GRAB] ⚠️ 机头仰着但算不出放平角（无 pitch 遥测）——按原样结束")
        return self._finalize(reason, ok)

    def _process_grab(self, frame, now_ms):
        self.frames += 1
        self._now_ms = now_ms
        if self._start_ms is None:
            self._start_ms = now_ms
        if self._finished:
            return S.STATUS_DONE
        if not self.calibrated:
            self._bail("uncalibrated")
            return S.STATUS_DONE
        if getattr(self.uart, "estop_active", False):
            self._bail("estop")
            return S.STATUS_DONE
        if self._phase != PH_GRAB_EXIT and \
                now_ms - self._start_ms >= float(grab_cfg()["timeout_ms"]):
            self._finish("timeout")
            if self._finished:
                return S.STATUS_DONE

        f = self._frame(frame)
        if f is None and self._phase not in (PH_GRAB_INIT, PH_GRAB_PITCH_UP, PH_GRAB_DUMP):
            # 没有下视帧 = 没有感知 ⇒ 本帧停手（绝不盲动）
            self.uart.send_dof(0.0, 0.0, 0.0, 0.0)
            self._set_info("no_frame", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None)
            return S.STATUS_RUNNING

        lost = _cvnode("lost", _D_LOST)
        alpha = float(lost["ema_alpha"])
        surge = sway = heave = yaw = 0.0
        dx = dy = 0.0
        ratio = 0.0
        percent = None
        ball = None
        if self._phase in (PH_GRAB_SEARCH, PH_GRAB_CENTER, PH_GRAB_APPROACH, PH_GRAB_ALIGN) and f is not None:
            ball, r = self._best(f)
            if ball is not None:
                self._lost_cnt = 0
                self._hold_until_ms = None
                dx = _dx_norm(ball.cx, self.w)
                dy = _dy_norm(ball.cy, self.h)
            else:
                self._lost_cnt += 1
            growth = self._ema_update(r, alpha)
            ratio = self._ema or 0.0
        else:
            growth = 0.0
        if self._phase == PH_GRAB_VERIFY and f is not None:
            v = _cvnode("verify", _D_VERIFY)
            be = self.hub.extra(self.name)
            percent = red_percent(f, v["roi"], getattr(be, "p", None))

        action = "hold"
        if self._phase == PH_GRAB_EXIT:
            surge, sway, heave, yaw = self._step_exit(now_ms)
            action = "exit_level"
        elif self._phase == PH_GRAB_INIT:
            surge, sway, heave, yaw = self._step_grab_init(now_ms)
            action = "init_depth_floor"
        elif self._phase == PH_GRAB_PITCH_UP:
            surge, sway, heave, yaw = self._step_pitch_up(now_ms)
            action = "pitch_up"
        elif self._phase == PH_GRAB_DUMP:
            surge, sway, heave, yaw = self._step_dump(now_ms)
            action = "dump_%s" % self._sub
        elif self._phase == PH_GRAB_LEVEL:
            surge, sway, heave, yaw = self._step_level(now_ms)
            action = "level"
        elif self._phase == PH_GRAB_DIP:
            surge, sway, heave, yaw = self._step_dip(now_ms)
            action = "dip"
        elif self._phase == PH_GRAB_RISE:
            surge, sway, heave, yaw = self._step_rise(now_ms)
            action = "rise"
        elif self._phase == PH_GRAB_VERIFY:
            surge, sway, heave, yaw = self._step_verify(now_ms, percent)
            action = "verify"
        elif self._phase in (PH_GRAB_CENTER, PH_GRAB_APPROACH, PH_GRAB_ALIGN) and ball is None:
            # 丢目标阶梯：宽限期内保持（**居中/对准段绝不前进**）→ 之后回搜索
            if self._lost_cnt <= int(lost["grace_frames"]):
                action = "hold"
            else:
                if self._hold_until_ms is None:
                    self._hold_until_ms = now_ms + float(lost["hold_s"]) * 1000.0
                if now_ms < self._hold_until_ms:
                    action = "hold"
                else:
                    self._hold_until_ms = None
                    self._phase = PH_GRAB_SEARCH
                    self._ema = None
                    action = "search"
        elif self._phase == PH_GRAB_SEARCH:
            surge, sway, heave, yaw = self._step_search(now_ms, ball)
            action = "search" if ball is None else "search_hit"
        elif self._phase == PH_GRAB_CENTER:
            surge, sway, heave, yaw = self._step_center(now_ms, dx)
            action = "center"
        elif self._phase == PH_GRAB_APPROACH:
            surge, sway, heave, yaw = self._step_approach(now_ms, dx, ratio, growth)
            action = "approach"
        elif self._phase == PH_GRAB_ALIGN:
            surge, sway, heave, yaw = self._step_align(now_ms, dx, dy)
            action = "align"

        if not self._finished:
            self.uart.send_dof(surge, sway, heave, yaw)
        self.last_dets = self.hub.detect_list(self.name, f) if f is not None else []
        self._set_info(action, ratio, dx, dy, sway, surge, heave, yaw, percent)
        return S.STATUS_DONE if self._finished else S.STATUS_RUNNING

    def _process_place(self, frame, now_ms):
        """每帧一次。`frame` 本任务**不用**（无感知）—— 保留签名以便将来加投放点识别。"""
        self.frames += 1
        self._now_ms = now_ms
        if self._start_ms is None:
            self._start_ms = now_ms
        if self._finished:
            return S.STATUS_DONE
        if not self.calibrated:
            self._bail("uncalibrated")
            return S.STATUS_DONE
        if getattr(self.uart, "estop_active", False):
            self._bail("estop")
            return S.STATUS_DONE
        if now_ms - self._start_ms >= float(place_cfg()["timeout_ms"]):
            self._bail("timeout")
            return S.STATUS_DONE

        if self._phase == PH_PLACE_INIT:
            dof, action = self._step_place_init(now_ms), "init_floor_roll"
        elif self._phase == PH_PLACE_TRANSPORT:
            dof = self._step_transport(now_ms)
            action = "relevel_roll" if self._move is not None else "transport"
        elif self._phase == PH_PLACE_RELEASE:
            dof, action = self._step_release(now_ms), "release"
        elif self._phase == PH_PLACE_STOP:
            dof, action = self._step_stop(now_ms), "stop_power"
        else:
            dof, action = ZERO_DOF, "done"

        if not self._finished:
            self.uart.send_dof(*dof)
        err = self._roll_err()
        self.last_info.update({
            "action": action, "phase": self._phase,
            "surge": round(dof[0], 4), "sway": round(dof[1], 4),
            "heave": round(dof[2], 4), "yaw": round(dof[3], 4),
            "roll_err": -999.0 if err is None else round(float(err), 3),
            "holds_ball": bool(self._holds_ball), "placed": bool(self._ok),
            "status": S.STATUS_DONE if self._finished else S.STATUS_RUNNING,
            "reason": self._reason})
        return S.STATUS_DONE if self._finished else S.STATUS_RUNNING


class GrabTask(HandlingTask):
    """兼容壳：`mode="grab"`（原 `grab.grab_task.GrabTask`）。"""

    name = "grab"

    def __init__(self, uart, hub, frame_w, frame_h, log=None):
        HandlingTask.__init__(self, uart, hub, frame_w, frame_h, log=log, mode="grab")


class PlaceTask(HandlingTask):
    """兼容壳：`mode="place"`（原 `place.place_task.PlaceTask`）。"""

    name = "place"

    def __init__(self, uart, hub=None, frame_w=0, frame_h=0, log=None):
        HandlingTask.__init__(self, uart, hub, frame_w, frame_h, log=log, mode="place")
