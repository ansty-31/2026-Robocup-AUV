# -*- coding: utf-8 -*-
"""gate/motion/hdg.py — ALIGN.HDG 正航向对齐（**算法核心 + 任务侧胶水**；2026-09-30 由
（详细用法、判据与实测见 doc/注释历史.md）"""
from __future__ import annotations

from common.cfg.cfgnode import flag, num, req, req_flag, req_node, MissingCfg
from common.motion.turn_deg import TurnCore
import base.cfg.settings as S
from base.log.turn_log import turn_log
from common.cfg.cfgnode import flag, motion_node, num, sub
from gate.motion.params import PH_ALIGN, SUB_HDG, SUB_GOLDEN, _K_HDG, _BOOL_KEYS
from gate.motion.channels import _dof_clip

# -*- coding: utf-8 -*-


# 状态
IDLE = "idle"          # 未启用 / 本门已结束
TURN = "turn"          # 正在按冻结的目标角转
DONE = "done"          # 已与光轴平行（到位）
GIVEUP = "giveup"      # 没得转（没测到 psi / 转向超时）→ 带残余航向继续走
ABORTED = "aborted"    # 外部中止（如转向中整门丢失）

_TERMINAL = (DONE, GIVEUP, ABORTED)



def hdg_cfg(node=None):
    """读 `comm.gate.hdg`（缺键 → 代码默认，不抛异常）。"""
    if node is None:
        try:
            import base.cfg.settings as S
            node = (S.comm.get("gate", None) or {}).get("hdg", None)
        except Exception:
            node = None
    # ★ 2026-10-07：**无兜底** —— 逐键必填，缺哪个报哪个（`_K_HDG` 见 params.py）
    if not isinstance(node, dict):
        raise MissingCfg("cfg 缺 comm.gate.hdg（代码已无兜底）")
    out = {}
    for k in _K_HDG:
        out[k] = req_flag(node, k) if k in _BOOL_KEYS else req(node, k)
    return out


class HeadingAligner(object):
    """一次到位的正航向状态机（每帧推进一次，不阻塞主循环）。"""

    def __init__(self, cfg=None, log=None, turn_kwargs=None):
        self.cfg = hdg_cfg(cfg)
        self.log = log or (lambda *a: None)
        self.turn_kwargs = dict(turn_kwargs or {})
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self.state = IDLE
        self.iters = 0
        self.psi_meas = None           # 冻结的目标角（起转那一刻的 psi）
        self.last_dir = ""
        self.last_d = 0.0              # 本次转向的方向（+1 右 / -1 左；0=没转）→ 转完的反向补偿要用
        self.last_target_deg = 0.0
        self._core = None
        self._t_stage = None

    @property
    def enabled(self):
        return bool(self.cfg.get("enable", True))

    def finished(self):
        """本门是否已有结论（可以放行到下一步）。"""
        return (not self.enabled) or self.state in _TERMINAL

    @property
    def turning(self):
        """是否正在执行转向。"""
        return self.state == TURN

    def summary(self):
        return "hdg=%s psi=%s last=%s%g°" % (
            self.state, "n/a" if self.psi_meas is None else "%.1f" % self.psi_meas,
            self.last_dir, self.last_target_deg)

    # ------------------------------------------------------------------
    def start(self, now_ms, psi=None):
        """进正航向（`gate_task` 在**居中达标**后调用一次）。"""
        if not self.enabled or self.state in _TERMINAL:
            return self.state
        self._t_stage = now_ms
        if psi is None:
            return self._giveup("没有可用的 PnP 航向测量（本门 full 帧一次都没测到）")
        psi = float(psi)
        self.psi_meas = psi
        left = bool(psi < 0)
        self.last_d = -1.0 if left else 1.0        # +1=右转 / -1=左转（与 turn_deg 的 d 同义）
        self.last_dir = "左转" if left else "右转"
        raw = abs(psi)
        if raw <= float(self.cfg["tol_deg"]):
            self.state = DONE
            self.log("[HDG] ✅ 已与光轴平行（|psi|=%.1f° ≤ %.1f°）→ 不转" % (raw, self.cfg["tol_deg"]))
            return self.state
        #   顺序要紧：先缩放再钳位（钳位是安全上限，不是收敛手段）。1.0 / 0 分别等于"不缩/不限"。
        scale = float(self.cfg.get("turn_scale", 1.0) or 1.0)
        deg = raw * scale
        cap = float(self.cfg.get("max_step_deg", 0.0) or 0.0)
        if abs(deg - raw) > 1e-9 or (cap > 0 and deg > cap):
            self.log("[HDG] 下发角：|psi|=%.1f° × turn_scale %.2f = %.1f°%s"
                     % (raw, scale, deg, ("；max_step_deg 钳到 %.1f°" % cap) if (cap > 0 and deg > cap) else ""))
        if cap > 0 and deg > cap:
            deg = cap
        self.last_target_deg = deg
        # ★ 2026-10-07：**实际下发量进日志**（这一行就是答案，别再靠 cfg 反推了）
        if hasattr(self, "last_info"):
            self.last_info["turn_deg"] = round(float(deg), 2)
            self.last_info["turn_dir"] = str(self.last_dir)
            self.last_info["turn_raw_psi"] = round(float(raw), 2)
        kw = dict(self.turn_kwargs)
        kw.setdefault("timeout", float(self.cfg.get("turn_timeout_s", 8.0)))
        self._core = TurnCore(deg=deg, left=left, log=self.log, **kw)
        self._core.start(now_ms)
        self.state = TURN
        self.log("[HDG] 起转：%s %.1f°（相对目标角交给下位机；等待完成反馈）"
                 % (self.last_dir, deg))
        return self.state

    def rearm(self):
        """**允许本门再转一次**（逐小步逼近，用户 2026-09-28 定）。"""
        if self.enabled and self.state in _TERMINAL:
            self.state = IDLE
            self._core = None
            self._t_stage = None
        return self.state

    def abort(self, why="外部中止"):
        """外部要求中止 → 立刻停转并给结论。只用于**整门丢失**/进冲刺这类真该停的情况。"""
        if self._core is not None:
            self._core.abort("external")
            self._core = None
        if self.state not in _TERMINAL:
            self.state = ABORTED
            self.log("[HDG] ⛔ %s → 中止正航向（本门不再尝试，带当前航向继续）" % why)
        return self.state

    def step(self, now_ms, yaw_telemetry=None, gate_lost=False, uart=None):
        """推进一帧，返回 (state, yaw_cmd)。"""
        if not self.enabled or self.state in _TERMINAL:
            return self.state, 0.0
        if gate_lost:
            return self.abort("整门丢失"), 0.0
        if self.state == IDLE:
            return self._giveup("没拿到目标角就进了正航向"), 0.0
        t_stage = self._t_stage if self._t_stage is not None else now_ms
        if now_ms - t_stage > self.cfg["timeout_ms"]:
            return self._giveup("正航向总超时 %.0fs" % (self.cfg["timeout_ms"] / 1000.0))
        if self._core is None:
            return self._giveup("转向没建立起来"), 0.0

        st, out = self._core.step(now_ms, uart)
        if st == TurnCore.DONE:
            self.iters += 1
            self.state = DONE
            self._core = None
            self.log("[HDG] ✅ 转向完成（%s %.1f°）→ 正航向结束" % (self.last_dir, self.last_target_deg))
            return self.state, 0.0
        if st == TurnCore.TIMEOUT:
            self.iters += 1
            return self._giveup("转向超时（目标 %.1f°）" % self.last_target_deg)
        if st == TurnCore.ABORTED:
            return self.abort("转向中止（%s）" % (self._core.why or "?")), 0.0
        return TURN, out

    # ------------------------------------------------------------------
    def _giveup(self, why):
        self.state = GIVEUP
        self._core = None
        self.log("[HDG] ❌ %s → **带残余航向继续走**（不卡死）" % why)
        return GIVEUP, 0.0


# -*- coding: utf-8 -*-


class GateHdg(object):
    def _hdg_new_psi_since_turn(self):
        """这次起转能不能用**当前这个 ψ**：必须是"上一次转向结束之后"测到的。"""
        if self._hdg_deg is None or self._hdg_ms is None:
            return False
        if self._hdg_turns_last_ms is None:
            return True                      # 本门还没转过 ⇒ 任何 ψ 都算"新"
        return self._hdg_ms > self._hdg_turns_last_ms
    def _hdg_skip_reason(self, mode, now_ms):
        """居中达标却没进正航向的**原因**（水里复盘看 `hdg_skip` 这个字段）。"""
        if not flag(self._hdg_cfg, "enable", True) or not self._hdg.enabled:
            return "off(未启用)"
        if self._hdg_done:
            if getattr(self, "_hdg_ok", False):
                return "done(本门航向已确认)"
            return "done_without_confirmation(转向未以 DONE 收口)"
        if self._hdg_deg is None or self._hdg_ms is None:
            return "no_psi(还没测到过 full 帧的 psi)"
        if self._hdg_turns_full():
            return "turns_full(本门已达转向上限 %d 次)" % self._hdg_turns
        if not self._hdg_new_psi_since_turn():
            return "wait_psi(等转向之后的新 ψ 测量)"
        return "psi_ok(其实可以起转 —— 若出现说明是居中没达标)"
    def _note_hdg_skip(self, why):
        """居中达标却跳过正航向 → 写进日志字段 + 终端报一次（每门一次）。"""
        self.last_info["hdg_skip"] = why
        if self._hdg_skip_logged:
            return
        self._hdg_skip_logged = True
        print("[GATE] ⚠️ 居中达标但跳过正航向：%s ⇒ 带残余航向直接进近（这一门不会再转）" % why)
    def _hdg_abort(self, why):
        """中止正在进行的正航向转向（只有**整门丢失 / 进冲刺**才调）。"""
        if self.substate == SUB_HDG and not self._hdg.finished():
            self._hdg.abort(why)
            self._hdg_done = True
            self.substate = SUB_GOLDEN
            self._center_cnt = 0
            return True
        return False
    def _turn_blocking(self, now_ms, target_deg=None, left=None):
        """★ **唯一的阻塞路径**（`turn` 专用）：**输入目标角度，输出"阶段结束"标志**。"""
        import time as _time
        _t0 = _time.monotonic()
        if target_deg is not None and self._hdg.state == "idle":
            self._hdg.start(now_ms, psi=float(target_deg))
        # ② 闭环推进：本帧一步 +（真实时钟域下）阻塞到终态
        yaw_cmd = 0.0
        if self.substate == SUB_HDG:
            _st, yaw_cmd = self._hdg.step(now_ms, yaw_telemetry=self._uart_yaw(),
                                          gate_lost=False, uart=self.uart)
            yaw_cmd = self._turn_inner_loop(yaw_cmd, now_ms)
        ended = bool(self._hdg.finished())
        # ③ 转向调用日志（每帧一行；ms 很大 = 这一帧被这次转向占住了）
        turn_log("gate_loop", src="gate", frame=self.frames, t_ms=now_ms,
                 state=self._hdg.state, yaw=yaw_cmd, tyaw=self._uart_yaw(),
                 tgt=self._hdg.last_target_deg, dir=self._hdg.last_dir,
                 iters=self._hdg.iters, done=ended, ended=ended,
                 ms=(_time.monotonic() - _t0) * 1000.0)
        # ④ 收尾（只做一次）
        if ended and self.substate == SUB_HDG:
            #   然后等"转向之后的新 ψ 测量"；真正 DONE 将锁存本门航向确认标志。
            self._hdg_turns += 1
            self._hdg_done = True
            if self._hdg.state == "done":
                self._hdg_ok = True
            self.substate = SUB_GOLDEN
            print("[GATE] 本门第 %d 次转向结束（下位机执行阶段已结束）"
                  % self._hdg_turns)
            self._center_cnt = 0          # 转向会动到门在画面里的位置 → 之后复核居中
            _now_stop, _stop_ok = self._stop_hard_framed(now_ms, quiet=not S.DEBUG)
            if _stop_ok is False:
                print("[GATE] ⚠️ 转向收尾硬停**未确认**（遥测显示仍在转）—— 平移窗口照开，注意船姿态")
            _now_true = int(now_ms + (_time.monotonic() - _t0) * 1000.0)
            self._now_ms = _now_true      # 本帧后续（`_set_info` 的窗口判据 / z 新鲜度）都用真实时刻
            self._hdg_turns_last_ms = _now_true   # 判"有没有新的 ψ"的基准（必须晚于它）
            self._start_post_sway(_now_true)
        self.last_info["hdg_i"] = self._hdg.iters
        self.last_info["turn_end"] = ended        # ★ 阶段结束标志（消费方/日志都看得到）
        self._set_info("hdg" if not ended else "center",
                       mode=self.mode or "", z=self._z_last, dx=0.0, dy=0.0,
                       sway=0.0, heave=0.0, surge=0.0, yaw=yaw_cmd,
                       kpt=self._dbg_kpt, hdg=self._hdg_deg,
                       hdg_state=self._hdg.state)
        return ended
    def _stop_hard_framed(self, now_ms, quiet=False):
        """转向收尾**硬停** + **帧时钟域显式折算**（与 `_turn_inner_loop` 同一套做法）。"""
        import time as _time
        if not flag(sub(self._G, "hdg"), "stop_hard", True):
            return now_ms, None
        t0 = _time.monotonic()
        ok = True
        try:
            _sh = getattr(self.uart, "stop_hard", None)
            if callable(_sh):
                ok = _sh(quiet=quiet)
            else:
                for _ in range(14):              # 旧对象：自己连发中性帧走完 ramp
                    self.uart.neutral()
        except Exception as e:
            ok = False
            print("[GATE] 转向收尾硬停失败：%s" % e)
        dt_ms = (_time.monotonic() - t0) * 1000.0
        turn_log("gate_stop_hard", src="gate", frame=self.frames, t_ms=now_ms,
                 ms=dt_ms, now_after=int(now_ms + dt_ms), ok=ok)
        return int(now_ms + dt_ms), ok
    def _start_post_sway(self, now_ms):
        """转向结束 → **进入「转完反向平移」状态**（用户 2026-09-27 定；主循环里的一个状态）。"""
        if self._post_sway_until_ms is not None or not self._hdg.last_d:
            return
        win = float(self._hdg_cfg.get("post_sway_ms", 600.0) or 0.0)
        mag = float(self._hdg_cfg.get("post_sway", 0.20) or 0.0)
        if win <= 0 or mag <= 0:
            return
        back = float(self._hdg_cfg.get("post_sway_back", 0.0) or 0.0)
        self._post_sway = _dof_clip(-self._hdg.last_d * mag)     # 反向平移
        self._post_sway_back = _dof_clip(-abs(back))             # ★ 同时**缓慢后退**（2026-09-28 定）
        self._post_sway_until_ms = now_ms + win
        self._post_sway_frame0 = self.frames
        #    （这期间不测位姿）⇒ 加了新鲜度必然为 None、z 闸形同虚设。新鲜度只对**当帧**的 z 有意义。
    def _turn_inner_loop(self, yaw_cmd, now_ms=0):
        """转向期间按 `comm.gate.hdg.turn_period`（默认 0.05s=20Hz）轮询"下位机完成反馈"。"""
        import time as _time
        # ★ 2026-10-07：无兜底 —— 缺键即报名字
        period = req(sub(self._G, "hdg"), "turn_period")
        period = max(0.01, min(0.2, period))
        t_wall0 = _time.monotonic()
        budget = float(self._hdg_cfg.get("turn_timeout_s", 8.0)) + 1.0
        _wall = abs(now_ms - _time.time() * 1000.0)
        _mono = abs(now_ms - _time.monotonic() * 1000.0)
        use_wall = min(_wall, _mono) < 60000.0
        if not use_wall:
            return yaw_cmd
        n = 0
        while (not self._hdg.finished()) and (_time.monotonic() - t_wall0) < budget:
            n += 1
            #   没有它，一次不收敛的转向会把主循环占满 `turn_timeout_s+1`≈9s，操作员插不进来。
            if getattr(self.uart, "estop_active", False):
                self._hdg.abort("estop(Ctrl-C)")
                break
            if use_wall:
                _time.sleep(period)
                now2 = int(now_ms + (_time.monotonic() - t_wall0) * 1000.0)
            else:
                now2 = int(now_ms + n * period * 1000.0)
            st_h, yaw_cmd = self._hdg.step(
                now2, yaw_telemetry=self._uart_yaw(), gate_lost=False, uart=self.uart)
            self.last_info["hdg_i"] = self._hdg.iters
            self._set_info("hdg" if not self._hdg.finished() else "center",
                           mode=self.mode, z=self._z_last, dx=0.0, dy=0.0,
                           sway=0.0, heave=0.0, surge=0.0, yaw=yaw_cmd,
                           kpt=self._dbg_kpt, hdg=self._hdg_deg,
                           hdg_state=self._hdg.state)
        return yaw_cmd
    def _hdg_turns_full(self):
        """**本门转向次数是否已达上限** `hdg.max_turns`（0/负 = 不限）。"""
        cap = int(req(sub(self._G, "hdg"), "max_turns") or 0)
        return cap > 0 and self._hdg_turns >= cap

    def _hdg_ready(self, mode, now_ms):
        """是否可以（或必须）进正航向：启用 + 本门还没做 + **没到转向上限** + **测到过 full 帧的 psi**。"""
        # _hdg_done 表示本门不再启动新的转向；_hdg_ok 才表示可以通过冲刺前航向闸门。
        if self._hdg_done or self._hdg_ok or not self._hdg.enabled:
            return False
        if self._hdg_turns_full():
            return False
        # 每步幅度由 max_step_deg/turn_scale 限；够正时 start() 自己给 DONE（不转）。
        return self._hdg_new_psi_since_turn()
    def _hdg_start(self, now_ms, reason):
        """起转（两条入口共用）：置子状态 + 把**当前 psi 冻结成目标角**交给 aligner。"""
        self.substate = SUB_HDG
        self.last_info["hdg_skip"] = None
        self.last_info["hdg_entry"] = reason   # 只有 golden=居中达标（「航向优先」2026-09-30 已删）
        turn_log("gate_start", src="gate", frame=self.frames, t_ms=now_ms, reason=reason,
                 psi=self._hdg_deg, psi_ms=self._hdg_ms, n_turn=self._hdg_turns + 1)
        if self._hdg.finished():
            # 逐小步逼近的下一小步：aligner 还在终态 ⇒ 先退回 IDLE 才能再 start 一次
            self._hdg.rearm()
        self._hdg.start(now_ms, psi=self._hdg_deg)
        turn_log("gate_start_done", src="gate", frame=self.frames,
                 state=self._hdg.state, dir=self._hdg.last_dir,
                 tgt=self._hdg.last_target_deg, iters=self._hdg.iters)
        if not self._hdg.finished():
            #   阻塞发生在本帧内（"自包含动作"）：主循环/相机/日志在这段时间里不进新帧。
            self._turn_blocking(now_ms, target_deg=self._hdg.psi_meas,
                                left=(self._hdg.last_d < 0))
        if self._hdg.finished():
            self._hdg_done = True
            if self._hdg.state == "done":
                self._hdg_ok = True
            self.substate = SUB_GOLDEN
        self.last_info["hdg_i"] = self._hdg.iters
        return self._hdg.state
    def _tick_hdg_degraded(self, now_ms):
        """档位退化（width/coarse）但 HDG 正在跑 → **只推进 HDG**，别让 CREEP/HOLD 冲掉子状态。"""
        if not (self.phase == PH_ALIGN and self.substate == SUB_HDG
                and not self._hdg.finished()):
            return False
        _st, yaw_cmd = self._hdg.step(now_ms, yaw_telemetry=self._uart_yaw(),
                                      gate_lost=False, uart=self.uart)
        if self._hdg.finished():
            self._hdg_done = True
            if self._hdg.state == "done":
                self._hdg_ok = True
            self.substate = SUB_GOLDEN
            self._center_cnt = 0
        self.last_info["hdg_i"] = self._hdg.iters
        self._set_info("hdg" if not self._hdg.finished() else "center",
                       mode=self.mode or "", z=self._z_last,
                       dx=self.last_info.get("dx", 0.0),
                       dy=self.last_info.get("dy", 0.0),
                       sway=0.0, heave=0.0, surge=0.0, yaw=yaw_cmd,
                       kpt=self._dbg_kpt, hdg=self._hdg_deg,
                       hdg_state=self._hdg.state)
        return True
