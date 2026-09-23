# -*- coding: utf-8 -*-
"""heading_align.py — ALIGN 阶段的「正航向」：**测 → 转 → 停稳 → 再测**（离散迭代）。

用户 2026-09-18 定的设计：
  · 看全门（4 角/full）时用 PnP 测出**航向误差** psi（门法向 vs 光轴）；
  · **不在转动中测量**：先停稳 → 测 → 按测得的量转（转的目标是**冻结**的）→ 停稳 → 再测；
  · 迭代到 |psi| ≤ 阈值（cfg `gate.hdg.tol_deg`，用户定 8~10°）→ 锁定，之后不再调 yaw；
  · 迭代用完仍不达标 → **带残余航向继续走**（不卡死，日志记下来）；
  · 顺序上"先常规居中，居中后再正航向"，转完**回 GOLDEN 复核居中**（转向会改变方位角，
    从而改变门在画面里的横向位置）。

转动本身交给 `common/turn_deg.py::TurnCore`（同一套探向/增益/到位判据/硬停），
所以"转多少度"是**精确**的：闭环量是下位机回传的绝对航向，而不是时间。

**为什么 p3p 不能用**：3 点解 6-DoF 是欠定的（实测最优解重投影 RMS 恒 0.0px、深度偏 −28%、
航向 std 64~69°；full 干净场景只有 5.4°）→ 只有 `mode == full` 的 psi 才喂进来。

本模块是**纯状态机**（不碰串口、不读相机）：每帧喂 `step()`，返回本帧该发的 yaw，
由 `gate_task` 负责实际下发与相位流转 —— 这样离线可单测（tests/tasks/test_motion.py）。
"""
from __future__ import annotations

import statistics as st

from common.cfgnode import flag, num
from common.turn_deg import TurnCore, wrap180

# 状态
IDLE = "idle"          # 未启用 / 本门已结束
SETTLE = "settle"      # 静止窗（转向后停稳；也是测量前的等待）
MEASURE = "measure"    # 攒够 k 帧 full 的 psi
TURN = "turn"          # 正在按冻结的目标角转
DONE = "done"          # 已与光轴平行（锁定）
GIVEUP = "giveup"      # 迭代用完/测不到 → 带残余航向继续走
ABORTED = "aborted"    # 外部中止（如转向中整门丢失）

_TERMINAL = (DONE, GIVEUP, ABORTED)

_D_HDG = dict(enable=True, tol_deg=8.0, measure_frames=5, max_iters=3,
              max_step_deg=0.0, settle_ms=400, settle_tol_deg=2.0,
              measure_timeout_ms=2000, timeout_ms=30000,
              # 一直没有遥测 yaw 时的等待上限（超过就放弃正航向，别白等总超时）
              wait_tel_ms=3000.0, turn_timeout_s=8.0, fresh_ms=800.0,
              # 居中达标但"那一刻不是 full 帧"（thus 没有可用 psi）时，**原地等**多久再放弃正航向。
              # 默认 0 = 旧行为（立刻进 APPROACH，带着残余航向）；
              # 斜门/远距容易只在部分帧拿到 4 角，这时把它设成 1500~2000 更划算 ——
              # 船本来就在 GOLDEN 停着（不下发 surge），等一帧新鲜 full 的代价只是时间。
              wait_fresh_ms=0.0)

# 开关键：其余键都是数值，只有 enable 是布尔 → 用 flag() 解析（别用 bool()）
_BOOL_KEYS = ("enable",)


def hdg_cfg(node=None):
    """读 `comm.gate.hdg`（缺键 → 代码默认，不抛异常）。"""
    if node is None:
        try:
            import base.settings as S
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
    """离散正航向状态机（每帧推进一次，不阻塞主循环）。

    "收敛后锁定 yaw" 的实现方式 = **终态（DONE/GIVEUP/ABORTED）直到 `reset()`**：
    reset 只在"新的一门"（`_start_search`）时调用，所以本门内一旦有结论就不会再动 yaw。
    """

    def __init__(self, cfg=None, imag_sign=None, log=None, turn_kwargs=None):
        self.cfg = hdg_cfg(cfg)
        self.imag_sign = imag_sign          # 遥测 yaw 正方向（None=首次转向时探向）
        self.log = log or (lambda *a: None)
        self.turn_kwargs = dict(turn_kwargs or {})
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self.state = IDLE
        self.iters = 0
        self.psi_meas = None
        self.last_dir = ""
        self.last_target_deg = 0.0
        self.last_err_deg = None
        self.aborts = 0
        self._t_stage = None
        self._t_settle = None
        self._yaw_ref = None
        self._samples = []
        self._t_meas = None
        self._core = None
        self._no_tel_logged = False

    @property
    def enabled(self):
        return bool(self.cfg.get("enable", True))

    def finished(self):
        """本门是否已有结论（可以放行到下一步）。"""
        return (not self.enabled) or self.state in _TERMINAL

    @property
    def turning(self):
        """是否正在执行一次转向（含探向/等遥测）。

        用途：`gate_task._step` 在这个状态下**绕开整条视觉链路**（用户定：
        先居中再转向，转的时候不要被画面干扰，信任测算的航向角）——
        否则档位一退化（width/coarse）或位姿被拒，转向就被打断了。
        """
        return self.state == TURN

    def summary(self):
        return ("hdg=%s iters=%d psi=%s last=%s%g°" % (
            self.state, self.iters,
            "n/a" if self.psi_meas is None else "%.1f" % self.psi_meas,
            self.last_dir, self.last_target_deg))

    # ------------------------------------------------------------------
    def start(self, now_ms):
        """进入正航向（由 gate_task 在"常规居中已达标"后调用一次）。"""
        if not self.enabled or self.state in _TERMINAL:
            return self.state
        self.state = SETTLE
        self._t_stage = now_ms
        self._t_settle = now_ms
        self._yaw_ref = None
        self._samples = []
        cap = float(self.cfg.get("max_step_deg", 0.0) or 0.0)
        self.log("[HDG] 开始正航向：阈值 %.1f° 最多 %d 次，单次%s"
                 % (self.cfg["tol_deg"], self.cfg["max_iters"],
                    ("不设限" if cap <= 0 else "≤%.0f°" % cap)))
        return self.state

    def abort(self, why="外部中止"):
        """外部要求中止 → 立刻停转并给结论。

        只用于**整门丢失**（或总超时/缺遥测这类真正没得转的情况）：
        档位退化（width/coarse）、位姿被拒、单帧角点闪烁**都不中止** —— 那是画面的事，
        转向只信「已经测好的航向角 + 下位机回传的 yaw」（用户定）。
        """
        if self._core is not None:
            self._core.abort()
            self._core = None
        if self.state not in _TERMINAL:
            self.aborts += 1
            self.state = ABORTED
            self.log("[HDG] ⛔ %s → 中止正航向（本门不再尝试，带当前航向继续）" % why)
        return self.state

    # ------------------------------------------------------------------
    def step(self, now_ms, psi_deg=None, psi_fresh=False, yaw_telemetry=None,
             gate_lost=False):
        """推进一帧。

        Args:
            psi_deg:     本帧从 **full** 帧测到的航向误差（度；None=本帧没有）
            psi_fresh:   该测量是否新鲜（gate_task 用时间戳判；旧值不算）
            yaw_telemetry: 下位机回传的绝对航向（度；None=没有遥测）
            gate_lost:   本帧整门丢失
        Returns:
            (state, yaw_cmd)
        """
        if not self.enabled or self.state in _TERMINAL:
            return self.state, 0.0
        if self.state == IDLE:
            self.start(now_ms)

        if gate_lost:
            return self.abort("整门丢失"), 0.0
        # 注意：不能用 `or` 兜底 —— 时间戳可能正好是 0（虚拟时钟/开机第一帧），
        #       那样会被当成「未设置」，总超时永远不触发。
        t_stage = self._t_stage if self._t_stage is not None else now_ms
        if now_ms - t_stage > self.cfg["timeout_ms"]:
            return self._giveup("正航向总超时 %.0fs" % (self.cfg["timeout_ms"] / 1000.0))

        # ---------------- 静止窗：等到"船真的不转了"再测 ----------------
        if self.state == SETTLE:
            if yaw_telemetry is None:
                if not self._no_tel_logged:
                    self._no_tel_logged = True
                    self.log("[HDG] ⚠️ 没有遥测 yaw：无法确认停稳、也无法闭环转向")
                wait_ms = float(self.cfg.get("wait_tel_ms", 3000.0))
                if now_ms - (self._t_settle if self._t_settle is not None
                             else now_ms) > wait_ms:
                    return self._giveup("%.0fs 内一直没有遥测 yaw"
                                        % (wait_ms / 1000.0))
                return SETTLE, 0.0
            if self._yaw_ref is None:
                self._yaw_ref = yaw_telemetry
                self._t_settle = now_ms
                return SETTLE, 0.0
            if abs(wrap180(yaw_telemetry - self._yaw_ref)) > self.cfg["settle_tol_deg"]:
                # 还在转 → 窗口重开
                self._yaw_ref = yaw_telemetry
                self._t_settle = now_ms
                return SETTLE, 0.0
            if now_ms - self._t_settle >= self.cfg["settle_ms"]:
                self.state = MEASURE
                self._samples = []
                self._t_meas = now_ms
                return MEASURE, 0.0
            return SETTLE, 0.0

        # ---------------- 测量：攒 k 帧 full 的 psi，取中位数 ----------------
        if self.state == MEASURE:
            if psi_deg is not None and psi_fresh:
                self._samples.append(float(psi_deg))
            need = max(1, int(self.cfg["measure_frames"]))
            if len(self._samples) >= need:
                self.psi_meas = float(st.median(self._samples))
                self.log("[HDG] 测量 %d 帧 → psi=%+.1f°（中位数）" % (len(self._samples), self.psi_meas))
                if abs(self.psi_meas) <= self.cfg["tol_deg"]:
                    self.state = DONE
                    self.log("[HDG] ✅ 已与光轴平行（|psi|=%.1f° ≤ %.1f°）→ 锁定 yaw"
                             % (abs(self.psi_meas), self.cfg["tol_deg"]))
                    return DONE, 0.0
                if self.iters >= int(self.cfg["max_iters"]):
                    return self._giveup("迭代已用满 %d 次（残余 psi=%+.1f°）"
                                        % (self.iters, self.psi_meas))
                return self._start_turn(now_ms)
            t_meas = self._t_meas if self._t_meas is not None else now_ms
            if now_ms - t_meas > self.cfg["measure_timeout_ms"]:
                return self._giveup("%.1fs 内没攒够 %d 帧 full 测量"
                                    % (self.cfg["measure_timeout_ms"] / 1000.0, need))
            return MEASURE, 0.0

        # ---------------- 转（目标冻结；不在转动中重测）----------------
        if self.state == TURN and self._core is not None:
            st_, out = self._core.step(now_ms, yaw_telemetry)
            if self._core.imag_sign is not None:
                self.imag_sign = self._core.imag_sign        # 探向结果复用给后续迭代
            if st_ == TurnCore.DONE:
                self.iters += 1
                self.log("[HDG] 第 %d 次转向完成（目标 %.1f°）→ 回静止窗重测"
                         % (self.iters, self.last_target_deg))
                return self._to_settle(now_ms)
            if st_ == TurnCore.TIMEOUT:
                self.iters += 1
                self.last_err_deg = self._core.done_deg
                self.log("[HDG] ⚠️ 第 %d 次转向超时（实测转了 %.1f°）→ 回静止窗重测"
                         % (self.iters, self._core.done_deg))
                return self._to_settle(now_ms)
            if st_ == TurnCore.ABORTED:
                return self.abort("转向中止（缺遥测）"), 0.0
            return TURN, out
        return self.state, 0.0

    # ------------------------------------------------------------------
    def _to_settle(self, now_ms):
        self._core = None
        self._samples = []
        self._yaw_ref = None
        self._t_settle = now_ms
        self.state = SETTLE
        return SETTLE, 0.0

    def _start_turn(self, now_ms):
        deg = abs(self.psi_meas)
        cap = float(self.cfg.get("max_step_deg", 0.0) or 0.0)
        # 用户 2026-09-18 定：**单次转角不设限**（0/负 = 不限制）→ 测多少就一次转多少。
        #   理由：测的是纯姿态量，且转之前已经居中；转动本身在遥测 yaw 闭环下精确执行。
        #   分多次转只会多出几轮「停稳-重测」的等待。安全阀仍保留：设成 >0 就恢复限幅。
        if cap > 0 and deg > cap:
            self.log("[HDG] 单次幅度限制：%.1f° → %.0f°（配置了 max_step_deg>0）"
                     % (deg, cap))
            deg = cap
        # psi>0 = 门法向偏画面右 = 机身相对门左偏 → 需要**右转**（+yaw）；反之左转
        left = bool(self.psi_meas < 0)
        self.last_dir = "左转" if left else "右转"
        self.last_target_deg = deg
        kw = dict(self.turn_kwargs)
        kw.setdefault("imag_sign", self.imag_sign)
        # 单次转向的超时（默认 8s：25° 在 10~20°/s 下只需 1.2~2.5s，8s 很宽裕）。
        # 预算关系：max_iters × (turn_timeout + settle + measure) 应 < timeout_ms，
        # 否则总超时会先触发（那时 iterations 用不满 —— 也是安全的，只是日志会显示放弃）。
        kw.setdefault("timeout", float(self.cfg.get("turn_timeout_s", 8.0)))
        self._core = TurnCore(deg=deg, left=left, log=self.log, **kw)
        self._core.start(now_ms)
        self.log("[HDG] 第 %d 次转向：%s %.1f°（按冻结的测量值；转完停稳再重测）"
                 % (self.iters + 1, self.last_dir, deg))
        self.state = TURN
        return TURN, 0.0

    def _giveup(self, why):
        self.state = GIVEUP
        self._core = None
        self.log("[HDG] ❌ %s → **带残余航向继续走**（不卡死；转完仍会回 GOLDEN 复核居中）" % why)
        return GIVEUP, 0.0
