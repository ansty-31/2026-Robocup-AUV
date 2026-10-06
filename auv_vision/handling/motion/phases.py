# -*- coding: utf-8 -*-
"""handling/motion/phases.py — 两个流程的相位处理（方法体一字未改）。"""
from __future__ import annotations

from handling.motion.params import place_cfg

from handling.motion.actions import _cvnode, _clip, _dx_norm, _dy_norm, _ratio_radius

# ★ 任务三/四的 search = **纯左右平移扫视**（用户 2026-10-06 定；不用 search_scan 的 yaw 慢扫）
from common.motion.search_sweep import Sweep
from common.motion.axis import AxisMove, EnsureDepth, TimedDof, _K_DIP, _K_RISE, action_cfg, axis_deg, grab_cfg, grab_node, level_relative_deg, pitch_up_deg
from common.motion.drop import BallDropSequence
from common.motion.axis import AxisMove, TimedDof, ZERO_DOF, axis_deg, level_relative_deg, task_node
from handling.motion.params import (PH_GRAB_PITCH_UP,
                                  PH_GRAB_SEARCH,
                                  PH_GRAB_CENTER,
                                  PH_GRAB_APPROACH,
                                  PH_GRAB_LEVEL,
                                  PH_GRAB_ALIGN,
                                  PH_GRAB_DIP,
                                  PH_GRAB_RISE,
                                  PH_GRAB_VERIFY,
                                  PH_GRAB_DUMP,
                                  _K_CENTER,
                                  _K_APPROACH,
                                  _K_ALIGN,
                                  _K_VERIFY,
                                  _K_RETRY,
                                  PH_PLACE_TRANSPORT,
                                  PH_PLACE_RELEASE,
                                  PH_PLACE_STOP,
                                  PH_PLACE_DONE)


class GrabPhases(object):
    """夹取流程的相位处理（原 `GrabTask._step_*`）。"""

    def _step_grab_init(self, now_ms):
        setter = getattr(self.uart, "set_extra_min_depth", None)
        if not callable(setter):
            # 拿不到"收紧限深"的能力 ⇒ 抬头就可能露出水面（比赛立即停止）⇒ 直接不干
            self.log("[GRAB] ⚠️ 串口对象不支持任务级限深下限 ⇒ 拒绝执行夹取（安全优先）")
            self._bail("no_depth_floor")
            return 0.0, 0.0, 0.0, 0.0
        c = grab_cfg()
        setter(c["depth_floor_m"])         # 只抬不降；全局 min_depth_m 一个字没动
        self._floor_raised = True
        self._ensure = EnsureDepth(log=self.log)
        self._phase = PH_GRAB_PITCH_UP
        self._sub = "ensure"
        self.log("[GRAB] 任务级限深下限抬到 %.2fm（全程有效），先下潜到位再抬头"
                 % float(c["depth_floor_m"]))
        return 0.0, 0.0, 0.0, 0.0

    def _step_pitch_up(self, now_ms):
        if self._sub == "ensure":
            st, dof = self._ensure.step(now_ms, self.uart)
            if st == EnsureDepth.FAILED:
                self._bail("depth_%s" % self._ensure.why)
                return 0.0, 0.0, 0.0, 0.0
            if st != EnsureDepth.DONE:
                return dof
            self._move = AxisMove("pitch", pitch_up_deg(), name="pitch", log=self.log)
            self._sub = "raise"
        st = self._move.step(now_ms, self.uart)
        if st == AxisMove.DONE:
            self._ref_pitch = self._move.ref_deg      # ★ 回水平的基准（动作前的遥测）
            self._pitched = True
            self._phase = PH_GRAB_SEARCH
            self._sweep.reset(now_ms)
            self.log("[GRAB] 已抬头；水平基准 pitch=%s，开始扫描"
                     % ("n/a" if self._ref_pitch is None else "%.2f°" % self._ref_pitch))
        elif st == AxisMove.FAILED:
            self._bail("pitch_failed")
        return 0.0, 0.0, 0.0, 0.0

    def _step_search(self, now_ms, ball):
        if ball is not None:
            self._phase = PH_GRAB_CENTER
            self._hit_cnt = 0
            self._ensure_pids()
            self._pid_yaw.reset()
            return 0.0, 0.0, 0.0, 0.0
        # ★ 只左右平移扫视：**不转 yaw**（抬头时转 yaw 会让目标绕圈跑；且不依赖 yaw 遥测）
        sway = _clip(self._sweep.step(now_ms))
        return 0.0, sway, 0.0, 0.0

    def _step_center(self, now_ms, dx):
        c = _cvnode("center_yaw")
        yaw = _clip(self._pid_yaw.update(dx, now_ms))
        if abs(dx) <= float(c["eps"]):
            self._hit_cnt += 1
            if self._hit_cnt >= int(c["confirm_frames"]):
                self._phase = PH_GRAB_APPROACH
                self._hit_cnt = 0
                self._pid_sway.reset()
        else:
            self._hit_cnt = 0
        return 0.0, 0.0, 0.0, yaw

    def _step_approach(self, now_ms, dx, ratio, growth):
        a = _cvnode("approach")
        if ratio >= float(a["slow_ratio"]) and growth <= float(a["growth_eps"]):
            surge = float(a["surge_slow"])
        else:
            surge = float(a["surge_fast"])
        sway = _clip(self._pid_sway.update(dx, now_ms))
        if ratio >= float(a["dip_ratio"]):
            self._hit_cnt += 1
            if self._hit_cnt >= int(a["confirm_frames"]):
                self._hit_cnt = 0
                self._phase = PH_GRAB_LEVEL
                now_deg = axis_deg(self.uart, "pitch")
                rel = level_relative_deg(self._ref_pitch, now_deg)
                if rel is None:
                    # 回水平算不出相对角（没姿态遥测）⇒ 不敢按着角度下压
                    self.log("[GRAB] ⚠️ 拿不到 pitch 遥测，回不了水平 ⇒ 停手")
                    self._finish("no_pitch_telemetry")      # 仍仰着 ⇒ 走放平再结束
                    return 0.0, 0.0, 0.0, 0.0
                self._move = AxisMove("pitch", rel, name="level", log=self.log)
                self.log("[GRAB] 面积达标（%.3f ≥ %.3f）→ 回水平（相对角 %+.2f°）"
                         % (ratio, float(a["dip_ratio"]), rel))
                return 0.0, 0.0, 0.0, 0.0
        else:
            self._hit_cnt = 0
        return surge, sway, 0.0, 0.0

    def _step_level(self, now_ms):
        st = self._move.step(now_ms, self.uart)
        if st == AxisMove.DONE:
            self._pitched = False                     # 机头已放平
            self._phase = PH_GRAB_ALIGN
            self._align_cnt = 0
            self._ensure_pids()
            self._pid_asw.reset()
            self._pid_asu.reset()
        elif st == AxisMove.FAILED:
            self._finish("level_failed")              # 仍仰着 ⇒ 再试着放平
        return 0.0, 0.0, 0.0, 0.0

    def _step_align(self, now_ms, dx, dy):
        """**dx→sway、dy→surge，两通道都轻微（抗水波），不用 yaw**（用户 2026-10-06 定）。"""
        l = _cvnode("align_level")
        sway = _clip(self._pid_asw.update(dx, now_ms))
        surge = _clip(self._pid_asu.update(dy, now_ms))
        if abs(dx) <= float(l["eps_x"]) and abs(dy) <= float(l["eps_y"]):
            self._align_cnt += 1
            if self._align_cnt >= int(l["confirm_frames"]):
                self._align_cnt = 0
                self._phase = PH_GRAB_DIP
                c = action_cfg("dip")
                self._timed = TimedDof(dur_s=c["dur_s"], name="下压", heave=c["heave"],
                                       log=self.log)
                return self._timed.dof
        else:
            self._align_cnt = 0
        return surge, sway, 0.0, 0.0

    def _step_dip(self, now_ms):
        if self._timed.step(now_ms) == TimedDof.DONE:
            self._phase = PH_GRAB_RISE
            c = action_cfg("rise")
            self._timed = TimedDof(dur_s=c["dur_s"], name="上升", heave=c["heave"],
                                   log=self.log)
        return self._timed.dof \
            if self._timed.state == TimedDof.RUN else (0.0, 0.0, 0.0, 0.0)

    def _step_rise(self, now_ms):
        if self._timed.step(now_ms) == TimedDof.DONE:
            self._attempts += 1
            if not bool(_cvnode("verify")["enable"]):
                return self._accept("verify_off")     # 关掉验色 = 收下（标定前的台架档）
            self._phase = PH_GRAB_VERIFY
            self._verify_cnt = 0
        return self._timed.dof \
            if self._timed.state == TimedDof.RUN else (0.0, 0.0, 0.0, 0.0)

    def _step_verify(self, now_ms, percent):
        """读 percent（笼区目标色占比），决定"收下"还是"倒掉重来"。

        用户口径：比赛只有两个球 ⇒ **第 `assume_ok_after`(2) 次尝试起默认是对的、直接过**。
        `max_dumps` 是"倒球重来"的次数上限（安全阀：绝不无限循环）。
        `percent is None`（帧/ROI 非法）**不等于"不是目标色"** ⇒ 没有证据就按"没夹到"倒掉重来。
        """
        r = _cvnode("retry")
        v = _cvnode("verify")
        if bool(r["assume_ok"]) and self._attempts >= int(r["assume_ok_after"]):
            return self._accept("assumed_ok")
        if percent is not None and float(percent) >= float(v["red_percent_min"]):
            self._verify_cnt += 1
            if self._verify_cnt >= int(v["confirm_frames"]):
                return self._accept("grab_ok")
            return 0.0, 0.0, 0.0, 0.0
        self._verify_cnt = 0
        if percent is None:
            self.log("[GRAB] ⚠️ 验色拿不到占比（ROI/帧非法）⇒ 没有证据，按没夹到处理")
        else:
            self.log("[GRAB] 笼内目标色占比 %.3f < %.3f ⇒ 不是目标球"
                     % (float(percent), float(v["red_percent_min"])))
        if self._dumps >= max(0, int(r["max_dumps"])):
            self._finish("wrong_ball")          # 重来次数用完：安全退出，不空转
            return 0.0, 0.0, 0.0, 0.0
        self._dumps += 1
        return self._start_dump()

    def _step_dump(self, now_ms):
        st, dof = self._drop.step(now_ms, self.uart)
        if st == BallDropSequence.FAILED:
            self.log("[GRAB] ⚠️ 倒球序列失败（%s）⇒ 停手" % self._drop.why)
            self._bail("dump_failed")
            return 0.0, 0.0, 0.0, 0.0
        if st == BallDropSequence.DONE:
            # 重来：球在别处，且机身已回水平 ⇒ 从"抬头"重新进任务
            self._phase = PH_GRAB_PITCH_UP
            self._sub = "ensure"
            self._ensure = EnsureDepth(log=self.log)
            self._ema = None
            self._hit_cnt = 0
            self._sweep.reset(now_ms)
            self.log("[GRAB] 回到抬头，重新进夹取")
            return 0.0, 0.0, 0.0, 0.0
        return dof

    def _step_exit(self, now_ms):
        """收尾放平：放平完成/失败/超时都要**真的结束**（绝不卡在这一步）。"""
        st = AxisMove.FAILED if self._move is None else self._move.step(now_ms, self.uart)
        if st == AxisMove.DONE:
            self._pitched = False
            self.log("[GRAB] 机头已放平")
            self._finalize(self._exit_reason, self._exit_ok)
        elif st == AxisMove.FAILED or now_ms >= self._exit_deadline:
            self.log("[GRAB] ⚠️ 放平失败/超时（%s）——机头姿态交给人工确认"
                     % (getattr(self._move, "why", "") or "deadline"))
            self._finalize(self._exit_reason, self._exit_ok)
        return 0.0, 0.0, 0.0, 0.0

    def _start_dump(self):
        """倒掉"错球"：交给**共享件** `common/motion/drop.py`（任务四放球用同一套）。"""
        self._sub = ""
        self._drop = BallDropSequence("dump", log=self.log)
        self._phase = PH_GRAB_DUMP
        self.log("[GRAB] 笼里不是目标球 → 倒掉重来（第 %d 次尝试）" % self._attempts)
        return 0.0, 0.0, 0.0, 0.0


class PlacePhases(object):
    """放置流程的相位处理（原 `PlaceTask._step_*`）。"""

    def _step_place_init(self, now_ms):
        """首次：抬限深下限（只抬不降）+ 回正 roll；回正完成后进运输。"""
        if self._sub == "":
            setter = getattr(self.uart, "set_extra_min_depth", None)
            if not callable(setter):
                self.log("[PLACE] ⚠️ 串口对象不支持任务级限深下限 ⇒ 拒绝执行（安全优先）")
                self._bail("no_depth_floor")
                return ZERO_DOF
            c = place_cfg()
            setter(c["depth_floor_m"])
            self._floor_raised = True
            self._roll_target = float(c["roll_target_deg"])
            if not self._start_relevel():
                self._bail("no_roll_telemetry")
                return ZERO_DOF
            self._sub = "relevel"
            return ZERO_DOF
        st = self._move.step(now_ms, self.uart)
        if st == AxisMove.DONE:
            self._move = None
            self._sub = ""
            self._phase = PH_PLACE_TRANSPORT
            self._start_ms = now_ms              # 运输计时从"真正开始运输"起算
            self.log("[PLACE] roll 已回正 → 开始运输（保持 roll 平衡，避免球滚出）")
        elif st == AxisMove.FAILED:
            self._bail("relevel_failed")
        return ZERO_DOF

    def _step_transport(self, now_ms):
        """运输：前进；roll 偏差超容差就重新回正（受 `relevel_min_s` 节流）。"""
        c = place_cfg()
        if self._move is not None:
            st = self._move.step(now_ms, self.uart)
            if st == AxisMove.FAILED:
                self.log("[PLACE] ⚠️ 回正失败（%s）⇒ 停手" % self._move.why)
                self._bail("relevel_failed")
                return ZERO_DOF
            if st != AxisMove.DONE:
                return ZERO_DOF                    # 轴动作期间不发推力
            self._move = None
        err = self._roll_err()
        if err is None:
            self._bail("no_roll_telemetry")
            return ZERO_DOF
        if abs(err) > float(c["roll_tol_deg"]):
            cd = float(c["relevel_min_s"]) * 1000.0
            if self._last_relevel_ms is None or now_ms - self._last_relevel_ms >= cd:
                self._last_relevel_ms = now_ms
                if not self._start_relevel():
                    self._bail("no_roll_telemetry")
                return ZERO_DOF
        if self._arrived or (now_ms - self._start_ms) >= float(c["transport_s"]) * 1000.0:
            self.log("[PLACE] 到点（%s）→ 放球"
                     % ("外部信号" if self._arrived
                        else "计时 %.0fs 到" % float(c["transport_s"])))
            self._phase = PH_PLACE_RELEASE
            self._drop = BallDropSequence("release", log=self.log, prefix="place")
            return ZERO_DOF
        return (float(c["surge"]), 0.0, 0.0, 0.0)

    def _step_release(self, now_ms):
        st, dof = self._drop.step(now_ms, self.uart)
        if st == BallDropSequence.FAILED:
            self.log("[PLACE] ⚠️ 放球序列失败（%s）" % self._drop.why)
            self._bail("release_failed")
            return ZERO_DOF
        if st == BallDropSequence.DONE:
            self._phase = PH_PLACE_STOP
            self._timed = TimedDof(dur_s=float(place_cfg()["stop_s"]),
                                   name="停动力", log=self.log)
            self._ok = True
            return ZERO_DOF
        return dof

    def _step_stop(self, now_ms):
        if self._timed.step(now_ms) == TimedDof.DONE:
            self._finished = True
            self._reason = "placed"
            self._phase = PH_PLACE_DONE
            self._restore_floor()
            self.log("[PLACE] 放置完成（停动力 %.1fs）" % float(place_cfg()["stop_s"]))
        return ZERO_DOF
