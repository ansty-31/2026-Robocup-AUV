# -*- coding: utf-8 -*-
"""gate/motion/hdg.py — ALIGN.HDG 正航向对齐（**算法核心 + 任务侧胶水**；2026-09-30 由
`heading_align.py` 与 `hdg.py` 合并而来）。

流程（用户定，center → hdg → postsway；**没有"航向优先"入口**）：
  ① **先居中**：居中达标（`align.*` 判据，见 `modes.py`）才允许起转；
  ② **再转向**：`HeadingAligner` 冻结本帧 ψ → 按遥测 yaw 闭环离散转一次 → 硬停收尾；
  ③ **然后反向平移 + 缓慢后退**（`_start_post_sway`；每帧 action=`sway_back`）：
     朝转走方向的反向平移（`hdg.post_sway`）+ 同时小幅后退（`hdg.post_sway_back`），
     直到**当前门**（本帧选中的那扇）出现 ≥ `hdg.post_sway_kpt_min` 个
     conf ≥ `vision.gate.keypoint.conf_thr` 的角点 ⇒ 退出，把控制交回视觉
     （"对齐光轴"的目的达成）；`hdg.post_sway_ms` 到期是**安全兜底**（绝不永久停摆）；
     转向/冲刺进行中一律让位（`sway_exit=yield(...)`）—— 平移优先级最低。

参数：`comm.gate.hdg.*`（缺键 → `_D_HDG` 兜底 = 当前 cfg 的值）。
"""
from __future__ import annotations

from common.cfg.cfgnode import flag, num
from common.motion.turn_deg import TurnCore
import base.cfg.settings as S
from base.log.turn_log import turn_log
from common.cfg.cfgnode import flag, motion_node, num, sub
from gate.motion.params import PH_ALIGN, SUB_HDG, SUB_GOLDEN, _D_Z, _D_HDG, _BOOL_KEYS
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
    if not isinstance(node, dict):
        return dict(_D_HDG)
    out = dict(_D_HDG)
    for k in _D_HDG:
        if k in _BOOL_KEYS:
            out[k] = flag(node, k, _D_HDG[k])
        elif node.get(k) is not None:
            out[k] = num(node, k, _D_HDG[k])
    return out


class HeadingAligner(object):
    """一次到位的正航向状态机（每帧推进一次，不阻塞主循环）。
    终态（DONE/GIVEUP/ABORTED）一直保持到 `reset()`，而 reset 只在"新的一门"时调用"""

    def __init__(self, cfg=None, imag_sign=None, log=None, turn_kwargs=None):
        self.cfg = hdg_cfg(cfg)
        self.imag_sign = imag_sign          # 仅用例注入（生产 None：极性在 turn_deg 里定死）
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
        """是否正在执行转向。
        用途：`gate_task._step` 在这个状态下**绕开整条视觉链路**（用户定：先居中再转向，
        或位姿被拒，转向就被打断了。"""
        return self.state == TURN

    def summary(self):
        return "hdg=%s psi=%s last=%s%g°" % (
            self.state, "n/a" if self.psi_meas is None else "%.1f" % self.psi_meas,
            self.last_dir, self.last_target_deg)

    # ------------------------------------------------------------------
    def start(self, now_ms, psi=None):
        """进正航向（`gate_task` 在**居中达标**后调用一次）。
        Args:"""
        if not self.enabled or self.state in _TERMINAL:
            return self.state
        self._t_stage = now_ms
        if psi is None:
            return self._giveup("没有可用的 PnP 航向测量（本门 full 帧一次都没测到）")
        psi = float(psi)
        self.psi_meas = psi
        #     psi < 0 ⇒ 左转 ／ psi > 0 ⇒ 右转
        #     `gate_start{psi=+36.6}` → 按**旧映射**（psi>0⇒左转）执行左转，实船反馈**转反了**；
        #     同一趟日志里遥测也可以对上：σ=-1 ⇒ ψ=−tyaw，tyaw 7.5→40.5 意味着 ψ 从 −7.5 变到 −40.5，
        #     即机身朝"让 ψ 更负"的方向转，而 ψ>0 的偏差需要的是**减小 ψ** ⇒ 方向确实反了。
        #     映射取反它自动跟着反，不需要单独改。
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
        kw = dict(self.turn_kwargs)
        if self.imag_sign not in (None, 0):
            kw.setdefault("imag_sign", float(self.imag_sign))
        kw.setdefault("timeout", float(self.cfg.get("turn_timeout_s", 8.0)))
        self._core = TurnCore(deg=deg, left=left, log=self.log, **kw)
        self._core.start(now_ms)
        self.state = TURN
        self.log("[HDG] 起转：%s %.1f°（冻结的 PnP 目标角；转完即结束，不重测）"
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

    def step(self, now_ms, yaw_telemetry=None, gate_lost=False):
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

        st, out = self._core.step(now_ms, yaw_telemetry)
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
        """这次起转能不能用**当前这个 ψ**：必须是"上一次转向结束之后"测到的。
        之后防自转的闸门：没有它，转向一结束就会拿"起转前那个旧 ψ"立刻再转一次，无限循环。
        够不够正不在这里判（`|ψ| ≤ tol` 时 `HeadingAligner.start()` 自己给 DONE、不转）。
        基准是 `_hdg_turns_last_ms`（本门上一次转向结束时刻），与"多久没测量"无关。"""
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
            return "done(本门正航向已收口：够正/中止/未启用)"
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
        """中止正在进行的正航向转向（只有**整门丢失 / 进冲刺**才调）。

        档位退化（width/coarse）不触发中止：转向期间视觉链路整个被绕开（见 `_step`），
        那才是"不被画面干扰"的正确做法。原地旋转看不到门就不该继续盲转 ——
        `TurnCore.abort()` 会立刻把本帧 yaw 归零，`stop_hard` 由调用方在收尾时负责。
        """
        if self.substate == SUB_HDG and not self._hdg.finished():
            self._hdg.abort(why)
            self._hdg_done = True
            self.substate = SUB_GOLDEN
            self._center_cnt = 0
            return True
        return False
    def _turn_blocking(self, now_ms, target_deg=None, left=None):
        """★ **唯一的阻塞路径**（`turn` 专用）：**输入目标角度，输出"阶段结束"标志**。
        不传（每帧续跑）就沿用 aligner 里已冻结的目标角。
        · **输出**：`True = 这次转向阶段已结束`（本门不会再转，调用方可继续往下走）；
        `False = 尚未结束`（本帧先回主循环，下一帧从同一个冻结目标角续跑）。
        · 全程只看「冻结的目标角 + 遥测 yaw」：画面丢失、门转出视野、档位退化、位姿被拒
        **都不打断**；唯一中止来自 `TurnCore`（缺遥测 / 方向自证 / 超时）。
        · 收尾（`_hdg_done` / 回 GOLDEN / 挂"转完反向平移"）**只做一次**。"""
        import time as _time
        _t0 = _time.monotonic()
        if target_deg is not None and self._hdg.state == "idle":
            self._hdg.start(now_ms, psi=float(target_deg))
        # ② 闭环推进：本帧一步 +（真实时钟域下）阻塞到终态
        yaw_cmd = 0.0
        if self.substate == SUB_HDG:
            _st, yaw_cmd = self._hdg.step(now_ms, yaw_telemetry=self._uart_yaw(),
                                          gate_lost=False)
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
            #   然后等"转向之后的新 ψ 测量"；若残余仍 > tol 就自动再走一小步（见 `_hdg_ready`）。
            self._hdg_turns += 1
            self.substate = SUB_GOLDEN
            print("[GATE] 本门第 %d 次转向结束（等新的 ψ 测量决定要不要下一小步）"
                  % self._hdg_turns)
            self._center_cnt = 0          # 转向会动到门在画面里的位置 → 之后复核居中
            #   （84→128 只到 ~95 = 仍在转）；这时一旦断流/关串口，船就**锁在那个值上一直转**。
            #   开关：`comm.gate.hdg.stop_hard`（默认 true；嫌慢可设 false，但风险自负）。
            # 硬停完成后拿到**折算过的帧时基**，平移窗口从那一刻起算（见 `_stop_hard_framed`）。
            _now_stop, _stop_ok = self._stop_hard_framed(now_ms, quiet=not S.DEBUG)
            if _stop_ok is False:
                print("[GATE] ⚠️ 转向收尾硬停**未确认**（遥测显示仍在转）—— 平移窗口照开，注意船姿态")
            #   两者在同一帧里先后阻塞、中间**没有任何一帧**（`_t0` → `_turn_inner_loop` 20Hz
            #   阻塞闭环 1~3s → `_stop_hard_framed` 再阻塞 0.7~2.5s → 本帧才收尾）。
            #   只折算硬停 ⇒ 窗口起点仍**落后"这次转向的耗时"**，而窗口只有 `post_sway_ms`(600ms)
            #   ≪ 转向耗时 ⇒ **窗口一出世就已过期**，生产里照旧只发得出 1 帧平移。
            #   （用例 `test_post_sway_starts_after_the_turn_end_hard_stop` 同时模拟"转向阻塞"与
            #     "硬停阻塞"，只折算一半就会红。）
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
        """转向收尾**硬停** + **帧时钟域显式折算**（与 `_turn_inner_loop` 同一套做法）。
        `base/hw/uart.py::stop_hard` 是**帧内阻塞**的，它自己按 `dt=0.05`（20Hz）连发中性帧、
        （用例 `test_post_sway_starts_after_the_turn_end_hard_stop` 咬的就是这个次序）；
        Returns:
        "平移窗口从整块阻塞结束起算"由 `_turn_blocking` 用整帧流逝时间再折算一次（见那里）。"""
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
        """转向结束 → **进入「转完反向平移」状态**（用户 2026-09-27 定；主循环里的一个状态）。
        反方向**平移一小段把它拉回视野；一旦重新检出到"刚丢的那个门"就立刻交回视觉。
        检测、PnP、ψ 更新、居中确认、丢门处理一律照跑。"""
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
        #    （这期间不测位姿）⇒ 加了新鲜度必然为 None、z 闸形同虚设。新鲜度只对**当帧**的 z 有意义。
    def _turn_inner_loop(self, yaw_cmd, now_ms=0):
        """转向期间按 `comm.motion.turn_pid.period`（默认 0.05s=20Hz）推进 HDG 到终态。
        同「进带 3 帧收舵」。返回最后一次 yaw 指令。只在**真实时钟域**阻塞推进（用例的假时钟"""
        import time as _time
        # 缺配置也要能跑（用例里 motion 段可能没有 turn_pid）
        try:
            period = float(num(motion_node("turn_pid"), "period", 0.05) or 0.05)
        except Exception:
            period = 0.05
        period = max(0.01, min(0.2, period))
        # 兜底：单个转向最多推进 turn_timeout_s + 1s（TurnCore 自身也有超时）
        #   基准取传入的 now_ms —— 绝不能把 time.monotonic() 的绝对值喂进 HDG，否则时间跳变
        t_wall0 = _time.monotonic()
        budget = float(self._hdg_cfg.get("turn_timeout_s", 8.0)) + 1.0
        # 时钟域自适应：now_ms 与真实时钟同域（运行期）→ 真按 20Hz 睡；
        #   不同域（用例的假时钟）→ **不真睡**，改用合成周期推进，测试才不会被拖成真时间。
        # 时钟域判据：`now_ms` 来自**真实时钟**就算真域 —— 生产是 `main.py` 的
        # `int(time.time()*1000)`（epoch 毫秒，板端 ~1.79e12），用例里也可能是
        # `time.monotonic()*1000`（开机毫秒）。**两个都要认**；假时钟（用例 1000+100i）两者都远。
        #    ⇒ 内层闭环**整个被跳过**（转向退化成"每相机帧一步"、`gate_loop` 日志永不触发、
        _wall = abs(now_ms - _time.time() * 1000.0)
        _mono = abs(now_ms - _time.monotonic() * 1000.0)
        use_wall = min(_wall, _mono) < 60000.0
        if not use_wall:
            # 假时钟（无硬件用例：假船按"每帧一步"积分航向）⇒ 保持旧的每帧一步，
            # 高频内层循环是**运行期**行为，只在真实时钟域生效（不引入任何新逻辑）。
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
                now2, yaw_telemetry=self._uart_yaw(), gate_lost=False)
            self.last_info["hdg_i"] = self._hdg.iters
            self._set_info("hdg" if not self._hdg.finished() else "center",
                           mode=self.mode, z=self._z_last, dx=0.0, dy=0.0,
                           sway=0.0, heave=0.0, surge=0.0, yaw=yaw_cmd,
                           kpt=self._dbg_kpt, hdg=self._hdg_deg,
                           hdg_state=self._hdg.state)
        return yaw_cmd
    def _hdg_turns_full(self):
        """**本门转向次数是否已达上限** `hdg.max_turns`（0/负 = 不限）。

        这是"转向无效时自转"（2026-09-27 板的疯狂旋转 bug）的兜底闸：到顶就不再起转，
        带残余航向进近（`hdg_skip=turns_full`）——宁可不正、也不能无限转。
        """
        cap = int(num(sub(self._G, "hdg"), "max_turns", _D_HDG["max_turns"]) or 0)
        return cap > 0 and self._hdg_turns >= cap

    def _hdg_ready(self, mode, now_ms):
        """是否可以（或必须）进正航向：启用 + 本门还没做 + **没到转向上限** + **测到过 full 帧的 psi**。
        新模型下 full 只占约 27%，死等 full 那一刻等于永远不转。"""
        if self._hdg_done or not self._hdg.enabled:
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
            # 已经够正（|psi| ≤ tol）或没得转 ⇒ 本门不再正航向。
            #    不同步就会在 ALIGN 里反复起转（死循环）。
            self._hdg_done = True
            self.substate = SUB_GOLDEN
        self.last_info["hdg_i"] = self._hdg.iters
        return self._hdg.state
    def _tick_hdg_degraded(self, now_ms):
        """档位退化（width/coarse）但 HDG 正在跑 → **只推进 HDG**，别让 CREEP/HOLD 冲掉子状态。
        Returns: True = 本帧已被 HDG 接管。"""
        if not (self.phase == PH_ALIGN and self.substate == SUB_HDG
                and not self._hdg.finished()):
            return False
        _st, yaw_cmd = self._hdg.step(now_ms, yaw_telemetry=self._uart_yaw(),
                                      gate_lost=False)
        if self._hdg.finished():
            self._hdg_done = True
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
