# -*- coding: utf-8 -*-
"""tests/tasks/test_motion.py — 运动原语：额定转角 `TurnCore` + 正航向 `HeadingAligner`。

**2026-10-02 起模型变了**（板端先行的版本，本地对齐）：旋转**交给下位机执行**——
上位机只发"相对角度 + 编号"（`uart.request_turn`）、由 UART 层按同一编号重发，
然后等 15B 遥测里的完成标志（`uart.poll_turn_complete`）。**上位机不再跑 yaw PID 闭环**。

所以这一半测的是新契约：
  · 符号：`+` = 右转、`−` = 左转（`TurnCore.d` 左 −1 / 右 +1）；
  · 执行期间上位机的 yaw 恒 0（手动 yaw 永远回中，转头这件事不归它）；
  · 完成/超时/下发失败/急停四条收场路径，以及超时要**取消**下位机那次转动；
  · 正航向 `HeadingAligner`：起转前冻结目标角 → 交给下位机 → 等完成。
"""
import os
import sys

import pytest

# 工程根 = 向上第一个含 `cfg/` 的目录（**别写死层级**：脚本搬过位置，写死会静默指错）
_ROOT = os.path.dirname(os.path.abspath(__file__))
while _ROOT != os.path.dirname(_ROOT) and not os.path.isdir(os.path.join(_ROOT, "cfg")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)

import base.cfg.settings as S                                          # noqa: E402
from common.motion.turn_deg import TurnCore, turn, wrap180, yaw_sign   # noqa: E402
from gate.motion.hdg import (ABORTED, DONE, GIVEUP, TURN,             # noqa: E402
                             HeadingAligner, hdg_cfg)

# 真机极性：+yaw 命令 → 遥测 yaw 减小（σ = -1）
BOARD_SIGN = -1.0


# ======================================================================
# 假下位机：**旋转的执行者**（2026-10-02 起）
#   上位机 `request_turn(相对角)` → 下位机转动 → 遥测回 `turn_done`
#   ⇒ 测试里用 `request_turn` / `poll_turn_complete` / `cancel_turn` 三个钩子模拟它。
# ======================================================================
class _World(object):
    """只记航向的假船：`H` 正 = 右转，门的航向误差 **psi = psi0 − H**。"""

    def __init__(self, psi0=0.0):
        self.psi0 = float(psi0)
        self.H = 0.0

    @property
    def psi(self):
        return self.psi0 - self.H

    def turn_by(self, angle_deg):
        """下位机执行相对角：正 = 右转 ⇒ H 增大 ⇒ psi 变小。"""
        self.H += float(angle_deg)


class _Lower(object):
    """假下位机。

    Args:
        world:   给了就"真的"把假世界转过去（等价于下位机执行）。
        replies: poll 几次之后才回报完成（1 = 当帧完成；很大 = 永不回报 → 触发上位机超时）。
        accept:  False = 模拟下发失败（`request_turn` 返回 False）。
        estop:   True = 下位机处于急停（上位机必须立刻收手）。
    """

    def __init__(self, world=None, replies=1, accept=True, estop=False):
        self.world = world
        self.replies = int(replies)
        self.accept = accept
        self.estop_active = estop
        self.reqs = []            # [(编号, 相对角)] —— 断言"发了多少度"看这里
        self.cancels = []
        self.stop_hard_calls = 0
        self._id = 0
        self._pending = 0

    def request_turn(self, angle_deg, turn_id=None):
        if self.estop_active or not self.accept:
            return False
        self._id = int(turn_id) if turn_id is not None else self._id + 1
        self.reqs.append((self._id, float(angle_deg)))
        if self.world is not None:
            self.world.turn_by(float(angle_deg))
        self._pending = self.replies
        return True

    def poll_turn_complete(self):
        if self._pending <= 0:
            return False
        self._pending -= 1
        return self._pending <= 0

    def cancel_turn(self):
        if self.estop_active:
            return
        self.cancels.append(self._id)
        self._pending = 0

    def stop_hard(self, *a, **kw):
        self.stop_hard_calls += 1
        self.cancel_turn()
        return True

    def send_dof(self, surge=0.0, sway=0.0, heave=0.0, yaw=0.0):
        """脚本入口每帧会回中；假下位机不关心（转头由 request_turn 表现）。"""
        self.dof_calls = getattr(self, "dof_calls", 0) + 1


def _run_core(core, lower, dt=50, frames=200):
    """按帧推进 TurnCore，返回 (终态, 每帧 yaw 指令)。"""
    now = 0
    yaws = []
    for _ in range(frames):
        st, yaw = core.step(now, lower)
        yaws.append(yaw)
        if core.finished():
            break
        now += dt
    return core.state, yaws


# ======================================================================
# 额定转角（common/motion/turn_deg.py）
# ======================================================================
def test_yaw_sign_is_a_fixed_derivation_not_a_measurement():
    """**极性 σ 是算出来的常量**（固件 × dof_map × telemetry），不配置、不探向、不现场测。"""
    sig, src = yaw_sign()
    assert sig == pytest.approx(-1.0), "出厂极性应为 -1（%s），实测 %s" % (src, sig)
    assert "dof_map" in src and "telemetry" in src
    # 两个旋钮任一翻转，σ 自动跟着翻 —— 不允许"配置与代码各记一套符号"
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setitem(S.comm.dof_map, "yaw", {"axis": 0, "sign": -1})
        assert yaw_sign()[0] == pytest.approx(+1.0)
        monkeypatch.setitem(S.comm.dof_map, "yaw", {"axis": 0, "sign": 1})
        monkeypatch.setitem(S.comm, "telemetry", S.Y(dict(S.comm.telemetry, yaw_sign=-1)))
        assert yaw_sign()[0] == pytest.approx(+1.0)
    finally:
        monkeypatch.undo()
    assert yaw_sign()[0] == pytest.approx(-1.0)


def test_left_turn_is_a_negative_relative_angle():
    """左转 90° ⇒ 发给下位机的相对角是 **−90°**（`d` 左 −1 / 右 +1）。"""
    low = _Lower(replies=3)                       # 别当帧就完成，好断言"执行中"
    core = TurnCore(deg=90.0, left=True, timeout=5.0, log=lambda *a: None)
    core.start(0)
    st, yaw = core.step(0, low)
    assert low.reqs == [(1, -90.0)], "左转应下发 −90°，实际 %s" % low.reqs
    assert st == TurnCore.RUN and abs(yaw) < 1e-9, "执行期间上位机 yaw 必须恒 0"


def test_right_turn_is_a_positive_relative_angle():
    """右转 90° ⇒ **+90°**（真机极性下 +yaw=右转，见 `yaw_sign`）。"""
    low = _Lower()
    core = TurnCore(deg=90.0, left=False, timeout=5.0, log=lambda *a: None)
    core.start(0)
    core.step(0, low)
    assert low.reqs == [(1, 90.0)], low.reqs


def test_upper_pc_never_outputs_yaw_during_the_turn():
    """**上位机不再驱动 yaw**：整段转向里 `step()` 回给主循环的 yaw 永远是 0。"""
    low = _Lower(replies=3)
    core = TurnCore(deg=45.0, left=False, timeout=5.0, log=lambda *a: None)
    st, yaws = _run_core(core, low)
    assert st == TurnCore.DONE
    assert all(abs(y) < 1e-9 for y in yaws), "转向期间不许再有上位机 yaw：%s" % set(yaws)


def test_turn_done_when_the_lower_reports_completion():
    """下位机回报完成 ⇒ DONE；且**只下发一次**（重发是 UART 层按同一编号做的事）。"""
    low = _Lower(replies=2)
    core = TurnCore(deg=30.0, left=True, timeout=5.0, log=lambda *a: None)
    st, _ = _run_core(core, low)
    assert st == TurnCore.DONE and core.state == "done", core.state
    assert len(low.reqs) == 1 and low.reqs[0][1] == -30.0, low.reqs


def test_turn_times_out_then_cancels_the_lower():
    """下位机一直不回报 ⇒ 超时收场，并且**必须取消**那一次转动（别让它继续转）。"""
    low = _Lower(replies=10 ** 6)
    core = TurnCore(deg=90.0, left=True, timeout=1.0, log=lambda *a: None)
    st, _ = _run_core(core, low, frames=500)
    assert st == TurnCore.TIMEOUT and core.why == "completion_timeout", (st, core.why)
    assert low.cancels == [1], "超时应取消编号 1，实际 %s" % low.cancels


def test_turn_aborts_when_the_lower_rejects_the_request():
    """下发失败 ⇒ ABORTED(send_failed)，**不许**当成"转过了"继续往下走。"""
    low = _Lower(accept=False)
    core = TurnCore(deg=90.0, left=True, timeout=5.0, log=lambda *a: None)
    st, _ = _run_core(core, low, frames=5)
    assert st == TurnCore.ABORTED and core.why == "send_failed", (st, core.why)
    assert low.reqs == [], "下发都失败了，不该有成功的转向请求：%s" % low.reqs


def test_estop_aborts_the_turn_immediately():
    """下位机急停中 ⇒ 立刻 ABORTED(external)，一根指令都不发。"""
    low = _Lower(estop=True)
    core = TurnCore(deg=90.0, left=True, timeout=5.0, log=lambda *a: None)
    st, yaw = core.step(0, low)
    assert st == TurnCore.ABORTED and core.why == "external", (st, core.why)
    assert low.reqs == [] and abs(yaw) < 1e-9


def test_script_rc_mapping_matches_the_new_model():
    """脚本入口 `turn()` 的退出码：0=完成 / 3=超时 / 5=下发失败或中止。"""
    ok = _Lower(replies=1)
    assert turn(ok, deg=90.0, left=True, timeout=2.0, log=lambda *a: None) == 0
    assert ok.stop_hard_calls >= 1, "脚本收尾必须硬停"
    slow = _Lower(replies=10 ** 6)
    assert turn(slow, deg=90.0, left=True, timeout=0.5, log=lambda *a: None) == 3
    bad = _Lower(accept=False)
    assert turn(bad, deg=90.0, left=True, timeout=2.0, log=lambda *a: None) == 5


def test_old_upper_pc_pid_is_gone_not_just_disabled():
    """**旧机关不许复活**（源码级）：上位机 PID / 方向自证 / 等待遥测那套已经不在转向路径里。"""
    import inspect
    from common.motion import turn_deg as TD
    src = inspect.getsource(TD)
    for dead in ("自我证明", "方向自证失败", "_pid_out", "wait_tel_s=3.0, at_least"):
        assert dead not in src, "旧的 %s 还在 turn_deg 里" % dead
    assert "request_turn" in inspect.getsource(TD.TurnCore.step), "新模型必须走 request_turn"


# ======================================================================
# 正航向（gate/motion/hdg.py）：冻结 PnP 目标角 → 交下位机转一次 → 结束
# ======================================================================
class _Hull(object):
    """假船 + 假门：h=机身转角(正=右)，**psi=psi0−h**（右转使 psi 变小），遥测 yaw=h×imag_sign。"""

    def __init__(self, psi0=20.0, imag_sign=BOARD_SIGN, stuck=False):
        self.psi0 = float(psi0)
        self.imag_sign = float(imag_sign)
        self.stuck = bool(stuck)
        self.H = 0.0

    @property
    def psi(self):
        return self.psi0 - self.H

    @property
    def yaw_tel(self):
        return self.H * self.imag_sign

    def turn_by(self, angle_deg):
        if not self.stuck:
            self.H += float(angle_deg)


def _run_hd(aligner, hull, frames=600, lost_after=None, cfg_over=None, on_frame=None,
            lower=None):
    """按帧推进 aligner：下位机执行转动，aligner 只等完成标志。"""
    now = 0
    logs = []
    if cfg_over:
        aligner.cfg.update(cfg_over)
    aligner.log = logs.append
    low = lower if lower is not None else _Lower(hull)
    if not aligner.finished():
        aligner.start(now, psi=hull.psi)         # gate_task 的 `_hdg_start` 就这一句
    for i in range(frames):
        if on_frame is not None:
            on_frame(i)
        lost = lost_after is not None and i >= lost_after
        st, _yaw = aligner.step(now, yaw_telemetry=hull.yaw_tel, gate_lost=lost, uart=low)
        if aligner.finished():
            break
        now += 50
    return low, logs, now


def test_converges_with_one_turn():
    """psi=+20° ⇒ 向（右转 +20°）；转完 DONE、**只转一次**、残余落到容差内。"""
    hull = _Hull(psi0=20.0)
    al = HeadingAligner(cfg=dict(hdg_cfg(), max_step_deg=0.0, turn_scale=1.0),
                        log=lambda *a: None)
    low, _, _ = _run_hd(al, hull)
    assert al.state == DONE, "应收敛，实际 %s（%s）" % (al.state, al.summary())
    assert al.iters == 1, "只该转一次（实际 %d 次）" % al.iters
    assert al.last_dir == "右转", "psi>0 应**右转**，实际 %s" % al.last_dir
    assert al.last_target_deg == pytest.approx(20.0), al.summary()
    assert len(low.reqs) == 1 and low.reqs[0][1] == pytest.approx(20.0), low.reqs
    assert abs(hull.psi) <= float(hdg_cfg()["tol_deg"]) + 1e-6, "转完残余 %.1f°" % hull.psi


def test_left_turn_when_psi_is_negative():
    """psi=−25° ⇒ 向（左转 −25°）。"""
    hull = _Hull(psi0=-25.0)
    al = HeadingAligner(cfg=dict(hdg_cfg(), max_step_deg=0.0, turn_scale=1.0),
                        log=lambda *a: None)
    low, _, _ = _run_hd(al, hull)
    assert al.state == DONE and al.last_dir == "左转", al.summary()
    assert len(low.reqs) == 1 and low.reqs[0][1] == pytest.approx(-25.0), low.reqs
    assert abs(hull.psi) <= float(hdg_cfg()["tol_deg"]) + 1e-6


def test_target_angle_is_frozen_while_turning():
    """**转动中不重测**：目标角在起转那一刻冻结成死数（入参里根本没有新的 psi）。"""
    import inspect
    params = list(inspect.signature(HeadingAligner.step).parameters)
    for dead in ("psi_deg", "psi_fresh", "psi_stale"):
        assert dead not in params, "转向中不该再接收 %s（冻结目标角的结构性保证）：%s" % (dead, params)
    hull = _Hull(psi0=20.0)
    al = HeadingAligner(cfg=dict(hdg_cfg(), max_step_deg=0.0, turn_scale=1.0),
                        log=lambda *a: None)
    seen = {"n": 0}

    def on_frame(i):
        if al.state == TURN:
            seen["n"] += 1

    low, _, _ = _run_hd(al, hull, on_frame=on_frame)
    assert al.state == DONE and al.iters == 1, al.summary()
    assert len(low.reqs) == 1, "冻结目标角 ⇒ 只该有一次下发：%s" % low.reqs
    assert abs(hull.psi) <= float(hdg_cfg()["tol_deg"]) + 1e-6, "转完残余 %.1f°" % hull.psi


def test_stuck_turn_times_out_then_gives_up():
    """下位机不回报（卡住）→ 超时后 GIVEUP（带残余航向继续走），不反复重试。"""
    hull = _Hull(psi0=60.0, stuck=True)
    al = HeadingAligner(log=lambda *a: None, turn_kwargs=dict(timeout=1.0))
    low, logs, _ = _run_hd(al, hull, frames=800, lower=_Lower(hull, replies=10 ** 6))
    assert al.state == GIVEUP, "应放弃而不是死循环，实际 %s" % al.state
    assert al.iters == 1, "一次到位 ⇒ 只该有一次转向（实际 %d）" % al.iters
    assert any("带残余航向继续走" in s or "放弃" in s for s in logs), logs[-3:]


def test_step_cap_clamps_the_requested_angle():
    """`max_step_deg` 是安全钳位：**0 = 不设限**（按测到的 psi 转）；>0 才限幅。"""
    h0 = _Hull(psi0=60.0, stuck=True)
    al0 = HeadingAligner(cfg=dict(hdg_cfg(), max_step_deg=0.0, turn_scale=1.0),
                         log=lambda *a: None)
    low0, _, _ = _run_hd(al0, h0, frames=3)
    assert al0.last_target_deg == pytest.approx(60.0), \
        "max_step_deg=0 应不设限（实际 %.1f°）" % al0.last_target_deg
    h2 = _Hull(psi0=60.0, stuck=True)
    al2 = HeadingAligner(cfg=dict(hdg_cfg(), max_step_deg=25.0, turn_scale=1.0),
                         log=lambda *a: None)
    low2, _, _ = _run_hd(al2, h2, frames=3)
    assert al2.last_target_deg == pytest.approx(25.0), \
        "配了 max_step_deg=25 应限幅到 25°（实际 %.1f°）" % al2.last_target_deg
    assert low2.reqs and abs(low2.reqs[0][1]) == pytest.approx(25.0), low2.reqs


def test_small_psi_is_left_alone():
    """|psi| ≤ tol_deg ⇒ 直接 DONE，**一次下发都没有**（不为了 2~3° 去推一下）。"""
    hull = _Hull(psi0=3.0)
    al = HeadingAligner(log=lambda *a: None)
    low, logs, _ = _run_hd(al, hull, frames=50)
    assert al.state == DONE and al.iters == 0, al.summary()
    assert low.reqs == [], "不该有任何转向下发：%s" % low.reqs
    assert any("不转" in s for s in logs)


def test_no_psi_no_turn():
    """没有 PnP 目标角 ⇒ GIVEUP，不转（没有"开环盲转"这条路）。"""
    al = HeadingAligner(log=lambda *a: None)
    st = al.start(0, psi=None)
    if isinstance(st, tuple):                    # start() 有的分支回 (state, yaw)
        st = st[0]
    assert st == GIVEUP, st
    assert al.finished(), "GIVEUP 之后本门不再转，应视为已结束"


def test_gate_lost_aborts_immediately():
    """转向中整门丢失 → 立刻中止，并**取消下位机那一次转动**，本门不再尝试。"""
    hull = _Hull(psi0=45.0)
    al = HeadingAligner(log=lambda *a: None)
    low = _Lower(hull, replies=10 ** 6)          # 让它一直"转着"，好在途中丢门
    al.start(0, psi=hull.psi)
    _st, _ = al.step(0, yaw_telemetry=hull.yaw_tel, gate_lost=False, uart=low)
    assert al.state == TURN, al.summary()
    _st, _ = al.step(50, yaw_telemetry=hull.yaw_tel, gate_lost=True, uart=low)
    assert al.state == ABORTED, al.summary()
    assert low.cancels == [1], "中止时应取消下位机那次转动：%s" % low.cancels
    assert al.finished(), "中止后应视为已结束（放行到下一步）"


def test_disabled_switches_off_completely():
    """`enable: false` → 永远 finished、不发任何指令（一键关掉正航向）。"""
    al = HeadingAligner(cfg=dict(hdg_cfg(), enable=False), log=lambda *a: None)
    assert al.finished()
    low = _Lower()
    st, yaw = al.step(0, yaw_telemetry=0.0, uart=low)
    assert st == al.state and abs(yaw) < 1e-9 and low.reqs == []


def test_p3p_frames_do_not_feed_the_filter():
    """**p3p 的 psi 不许进滤波器**（3 点解欠定，航向 std 极大）。

    做法：同一个 GateTask 先喂两帧 full（建立 +20° 的测量），再喂几帧 p3p（旋转过的、会测出别的值）
    → `_hdg_deg` / `_hdg_ms` 必须保持 full 那帧的值不变。
    """
    import numpy as np
    import cv2
    from common.vision.detector import Det
    from gate.percept.gate_detector import board_camera
    from gate.gate_task import GateTask
    from gate.percept.geometry import object_points
    from tests.tasks.gate.test_gate_flow import _Hub, _Uart

    cam = board_camera()
    obj = object_points()

    def det(n_angle, conf):
        R = cv2.Rodrigues(np.array([0.0, np.radians(n_angle), 0.0]))[0]
        uv = cam.project(obj, cv2.Rodrigues(R)[0].ravel(), np.array([0.0, 0.0, 1.5]))
        return [Det("gate", 0.95, 100.0, 100.0, 400.0, 300.0, kpts=uv,
                    kpt_conf=np.asarray(conf, np.float32))]

    frame = np.zeros((cam.height, cam.width, 3), np.uint8)
    seq = {"i": 0}

    def fn():
        i = seq["i"]
        seq["i"] += 1
        if i < 2:
            return det(20.0, (0.95, 0.95, 0.95, 0.95))     # full：psi=+20
        return det(-25.0, (0.95, 0.95, 0.95, 0.0))         # p3p：不该被采信

    task = GateTask(_Uart(), _Hub(fn), cam.width, cam.height)
    task.process(frame, 1000)
    task.process(frame, 1100)
    assert task._hdg_deg == pytest.approx(20.0, abs=6.0), \
        "full 帧应给出约 +20° 的测量，实际 %s" % task._hdg_deg
    kept, kept_ms = task._hdg_deg, task._hdg_ms
    for k in range(3):
        task.process(frame, 1200 + 100 * k)                # 全是 p3p
    assert task._hdg_deg == kept and task._hdg_ms == kept_ms, \
        "p3p 帧污染了航向测量（%.1f → %.1f）" % (kept, task._hdg_deg)


def test_wrap180_wraps_into_half_open_range():
    """回绕归一化：(-180, 180]。"""
    assert wrap180(0.0) == pytest.approx(0.0)
    assert wrap180(180.0) == pytest.approx(180.0)
    assert wrap180(190.0) == pytest.approx(-170.0)
    assert wrap180(-190.0) == pytest.approx(170.0)


# ======================================================================
# 手动入口（CLI）：**给一个目标角度就转**（与 gate 自动接受同一条执行链）
# ======================================================================
class _FakeUartCtl(object):
    """冒充 `UartController`：把 request_turn / poll / cancel 转发给假下位机。"""

    sim = False

    def __init__(self, low):
        self._low = low
        self._turn_counter = 0

    def close(self):
        pass

    def _write(self, *a, **kw):
        pass

    def set_motion(self, *a, **kw):
        pass

    def __getattr__(self, name):
        return getattr(self._low, name)


def test_manual_cli_takes_a_target_angle(monkeypatch):
    """**手动规定目标角度**：`turn_deg.py --deg 45 --dir right` ⇒ 下发的正是 +45°（右转）。

    与自动那条对照：两者都落到 `uart.request_turn()`（同一个 `TurnCore`）。
    """
    import sys as _sys
    from common.motion import turn_deg as TD

    low = _Lower(replies=1)
    monkeypatch.setattr("base.hw.uart.UartController", lambda *a, **kw: _FakeUartCtl(low))
    monkeypatch.setattr(_sys, "argv", ["turn_deg.py", "--deg", "45", "--dir", "right"])
    rc = TD.main()
    assert rc == 0, "手动转向应报完成（rc=0），实际 %s" % rc
    assert low.reqs == [(1, 45.0)], "应把 +45°（右转）下发给下位机，实际 %s" % low.reqs
    assert low.stop_hard_calls >= 1, "收尾必须硬停"


def test_manual_cli_left_is_a_negative_angle(monkeypatch):
    """左转 = −角（`--dir left`），与 gate 那条的符号约定完全一致。"""
    import sys as _sys
    from common.motion import turn_deg as TD

    low = _Lower(replies=1)
    monkeypatch.setattr("base.hw.uart.UartController", lambda *a, **kw: _FakeUartCtl(low))
    monkeypatch.setattr(_sys, "argv", ["turn_deg.py", "--deg", "30", "--dir", "left"])
    assert TD.main() == 0
    assert low.reqs and low.reqs[0][1] == -30.0, low.reqs


def test_manual_cli_only_takes_deg_dir_timeout(monkeypatch):
    """**只接受 `--deg/--dir/--timeout`**（用户 2026-10-02 定）：限幅/PID/归一化等运动参数都归下位机，
    老写法（`--out-max/--kp/--kd/--norm-deg/--imag-sign`）现在必须**报错**而不是被静默忽略 ——
    静默忽略会让"以为设了限幅其实没设"，宁可让老脚本当场失败。"""
    import sys as _sys
    from common.motion import turn_deg as TD

    low = _Lower(replies=1)
    monkeypatch.setattr("base.hw.uart.UartController", lambda *a, **kw: _FakeUartCtl(low))
    for dead in (["--out-max", "0.3"], ["--kp", "0.2"], ["--kd", "0.05"],
                 ["--norm-deg", "15"], ["--imag-sign", "-1"]):
        monkeypatch.setattr(_sys, "argv", ["turn_deg.py", "--deg", "10"] + dead)
        with pytest.raises(SystemExit) as e:
            TD.main()
        assert e.value.code == 2, "旧旋钮 %s 应被拒（实际退出码 %s）" % (dead[0], e.value.code)
    assert low.reqs == [], "参数不合法时不该下发任何转向"


# ======================================================================
# 2026-10-02：运行时**可信边界** `z.relock_away_m = 1.2 m`（超出即不采纳该帧位姿）。
# 本模块绝大多数用例合成的门放在 1.5–3 m，考的是**位姿之后的逻辑**（起转/出口/恢复/SWAY_BACK…），
# 与"多远才算可信"正交 ⇒ 这里统一把边界放宽到 99（= 关闭），只有专门考这条的用例用真值
# （用例名里带 jump/relock/far 的自动跳过，不覆盖）。
# ======================================================================
@pytest.fixture(autouse=True)
def _relax_z_trust_boundary(request, monkeypatch):
    name = request.node.name
    if any(k in name for k in ("jump", "relock", "far", "stale", "history", "hist")):
        return
    if "relock_away_m" in S.comm.gate.get("z", {}):
        monkeypatch.setitem(S.comm.gate["z"], "relock_away_m", 99.0)
