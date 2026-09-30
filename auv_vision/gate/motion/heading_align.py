# -*- coding: utf-8 -*-
"""heading_align.py — ALIGN 的「正航向」：**PnP 测出目标角 → 交给 turn_deg 转一次 → 结束**。
· 目标角来自 `gate_task` 的 full 帧 psi（`_hdg_deg` = 位姿法向 vs 光轴，EMA 平滑后）；
只认 `mode == full` 的 psi：p3p 的三点解 6-DoF 欠定（重投影残差恒 0、深度与航向都不可信），
本模块是**纯状态机**（不碰串口、不读相机）：每帧喂 `step()`，返回本帧该发的 yaw，
由 `gate_task` 负责实际下发与相位流转 —— 这样离线可单测（tests/tasks/test_motion.py）。"""
from __future__ import annotations

from common.cfg.cfgnode import flag, num
from common.motion.turn_deg import TurnCore

# 状态
IDLE = "idle"          # 未启用 / 本门已结束
TURN = "turn"          # 正在按冻结的目标角转
DONE = "done"          # 已与光轴平行（到位）
GIVEUP = "giveup"      # 没得转（没测到 psi / 转向超时）→ 带残余航向继续走
ABORTED = "aborted"    # 外部中止（如转向中整门丢失）

_TERMINAL = (DONE, GIVEUP, ABORTED)

_D_HDG = dict(enable=True, tol_deg=8.0,
              # max_step_deg：单次转角的上限（**0 = 不设限**，就是按测到的 psi 转）。
              # 它只是防"垃圾 psi 把船甩出去"的安全钳位，不是收敛手段（本设计只转一次）。
              #   先按 turn_scale 缩小测到的 ψ，再用它钳位。被钳掉的残余**由下一小步补转**
              #   （逐小步逼近，无次数上限）⇒ 它只是"每步别太猛"，不再等于"只转一次"。
              max_step_deg=15.0,
              #   偏大，先按 0.8 缩一档试；1.0 = 不缩。它只改"下发的 deg"，不改 ψ 的测量与判据。
              turn_scale=0.8,
              timeout_ms=30000.0, turn_timeout_s=8.0,
              #   就超出 32° 半视场），门又看不见时视觉没法纠 ⇒ 在**丢门的帧**上朝**转向的反方向**
              #   平移一小段，把门拉回视野。post_sway 必须 > 执行器死区 0.138；窗口 = post_sway_ms。
              post_sway_ms=600.0, post_sway=0.20,
              #   post_sway_back = 后退幅度（正值，代码取负；0 = 不后退）；
              #   post_sway_kpt_min = 退出门限：画面里出现 ≥ 这么多**原始角点**就交回视觉
              #   （0 = 不看角点数，只靠到期/同门判据）。
              post_sway_back=0.15, post_sway_kpt_min=2,
              # post_sway_far_ratio / post_sway_same_z_m：窗口内判"这算不算刚丢的那个门"的两道闸
              #   （实现 `gate_task._post_sway_same_gate`）：z 优先（|Δz| ≤ same_z_m），
              #   z 拿不到才退框占比（≥ far_ratio × 转走那一刻的占比）；两个都设 0 = 关掉判据。
              post_sway_far_ratio=0.7, post_sway_same_z_m=0.5,
              # stop_hard：转向收尾**硬停**（阻塞：连发中性帧走完 ramp + 遥测 yaw 验证）。
              #   必须 true —— 下位机没有"无帧超时停车"，只发一帧中性时 yaw 轴还在 ramp 半路，
              stop_hard=True,
              # entry_psi_first：**航向优先** —— 居中还没达标也能起转（解"航向歪 ⇒ 门偏一侧 ⇒
              #   居中确认不了 ⇒ 不许转 yaw"的死锁）。false = 先居中再转。
              entry_psi_first=False,
              # 航向优先的距离门（满足任一）：新鲜 z ≤ psi_first_z_max 或 框占比 ≥ psi_first_ratio
              psi_first_z_max=2.5, psi_first_ratio=0.30)

_BOOL_KEYS = ("enable", "entry_psi_first", "stop_hard")


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
        """进正航向（`gate_task` 在"居中达标"或"航向优先"时调用一次）。
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
