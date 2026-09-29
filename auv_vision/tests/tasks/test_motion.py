# -*- coding: utf-8 -*-
"""tests/tasks/test_motion.py — 运动原语：额定转角 `TurnCore` + 正航向 `HeadingAligner`。

两半都是"遥测 yaw 闭环"状态机（转角是原语；正航向是 gate ALIGN 的子状态），
所以离线用**假船**测：遥测 yaw 按 `角速率 × 下发的 yaw DOF × dt` 积分，
时钟与 sleep 全部注入 → 确定性、毫秒级跑完。

假船的默认极性 = **真机极性**（命令 +yaw 时遥测 yaw **减小**）—— 谁把 `dof_map.yaw.sign` /
`telemetry.yaw_sign` / 接线改了，这里会先红。（板端实测依据见
doc/_注释历史_fragments/base_tests.md）
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base.settings as S                                          # noqa: E402
from common.turn_deg import TurnCore, turn, turn_cfg, wrap180, yaw_sign             # noqa: E402
from gate.heading_align import (ABORTED, DONE, GIVEUP, TURN,             # noqa: E402
                                HeadingAligner, hdg_cfg)

# 真机极性：+yaw 命令 → 遥测 yaw 减小（σ = -1）
BOARD_SIGN = -1.0


# ======================================================================
# 额定转角（common/turn_deg.py）
# ======================================================================
class _Tel(object):
    def __init__(self):
        self.yaw_deg = 170.0        # 故意放在回绕边界附近
class _FakeBoat(object):
    """假船：下发的 yaw DOF → 遥测 yaw 按 imag_sign 方向积分（纯积分器，无延迟）。"""

    def __init__(self, imag_sign=BOARD_SIGN, gain=60.0, start_yaw=170.0, dt=0.05):
        self.telemetry = _Tel()
        self.telemetry.yaw_deg = start_yaw
        self.imag_sign = imag_sign
        self.gain = gain            # 度/秒 每 1.0 DOF
        self.dt = dt
        self.cmd = []
        self.neutral_calls = 0
        self.closed = False

    def send_dof(self, surge=0.0, sway=0.0, heave=0.0, yaw=0.0):
        self.cmd.append((surge, sway, heave, yaw))
        if self.telemetry.yaw_deg is None:      # 无遥测的台架：yaw 不可用
            return
        self.telemetry.yaw_deg = wrap180(
            self.telemetry.yaw_deg + yaw * self.gain * self.dt * self.imag_sign + 360.0)

    def neutral(self):
        self.neutral_calls += 1
        self.cmd.append((0.0, 0.0, 0.0, 0.0))

    def close(self):
        self.closed = True
class _Clock(object):
    def __init__(self, dt=0.05):
        self.t = 0.0
        self.dt = dt

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += max(self.dt, s)
def _run_turn(boat, **kw):
    clk = _Clock(dt=boat.dt)
    logs = []
    rc = turn(boat, log=logs.append, now=clk.now, sleep=clk.sleep, **kw)
    return rc, logs
def _yaw_cmds(boat):
    return [c[3] for c in boat.cmd if abs(c[3]) > 1e-9]

def test_yaw_sign_is_a_fixed_derivation_not_a_measurement():
    """**极性 σ 是算出来的常量**（固件 × dof_map × telemetry），不配置、不探向、不现场测。

    （当初把 σ 当成 +1 曾闭合成正反馈，板端实测依据见 doc/_注释历史_fragments/base_tests.md。）
    """
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

def test_turn_left_reaches_rated_angle():
    """左转 90°：遥测 yaw 应转出 +90°（真机极性下 +yaw 命令 → yaw 减小 ⇒ 左转 = yaw 增大）。

    起点 170°（回绕边界附近）→ 左转 90° 会跨过 ±180，所以这条同时验回绕。
    """
    b = _FakeBoat()
    y0 = b.telemetry.yaw_deg
    rc, logs = _run_turn(b, deg=90.0, left=True, timeout=20.0)
    assert rc == 0, "应到达目标角，logs=%s" % logs[-3:]
    got = wrap180(b.telemetry.yaw_deg - y0)
    assert abs(got - 90.0) <= 8.0, "实测 %.1f° 应≈目标 +90°" % got
    assert b.neutral_calls >= 1, "结束时必须回中位"
    ys = _yaw_cmds(b)
    assert ys and all(y < 0 for y in ys), "左转不该出现正（右转）舵：%s" % set(ys)

def test_turn_right_uses_positive_yaw():
    b = _FakeBoat(start_yaw=0.0, gain=30.0)
    rc, _ = _run_turn(b, deg=90.0, left=False, timeout=20.0)
    assert rc == 0
    assert abs(wrap180(b.telemetry.yaw_deg - 0.0) + 90.0) <= 8.0, "右转应转出 -90°（真机极性）"
    ys = _yaw_cmds(b)
    assert ys and all(y > 0 for y in ys), "右转应是正舵（+yaw=右转）"

def test_pid_eases_off_near_the_target_and_respects_out_max():
    """**用 PID 的判据**：① 命令不过 out_max；② 近目标时自动收力（不是一路满舵到点）。"""
    cfg, _src = turn_cfg()
    om = float(cfg["out_max"])
    b = _FakeBoat(start_yaw=0.0, gain=40.0)
    rc, _ = _run_turn(b, deg=90.0, left=True, timeout=20.0)
    assert rc == 0
    ys = [abs(y) for y in _yaw_cmds(b)]
    assert max(ys) <= om + 1e-9, "命令不该超过限幅 %.2f（实际 %.3f）" % (om, max(ys))
    assert min(ys[-3:]) < 0.5 * max(ys), \
        "近目标没收力（峰值 %.3f，末段 %s）→ 说明不是 PID 闭环" % (max(ys), ys[-3:])

def test_loop_converges_without_limit_cycle():
    """**必须"逐步收敛"，不许原地极限环**（2026-09-27 仿真抓到的真问题）。

    现行 cfg 是 `kd=0 / deadzone_deg=6`：一次转到位、出力不反号（导致极限环的旧参数
    `kd=0.10 / deadzone_deg=3.0` 及其失效过程见 doc/_注释历史_fragments/base_tests.md）。
    """
    cfg, _ = turn_cfg()
    assert float(cfg["kd"]) == pytest.approx(0.0), "kd 必须为 0（见 cfg/comm.yaml 的 turn_pid 注释）"
    b = _FakeBoat(start_yaw=75.07, gain=200.0)      # 增益按板端实测量级取
    rc, _ = _run_turn(b, deg=52.6, left=False, timeout=10.0)
    assert rc == 0, "应一次转到位（实际退出码 %s）" % rc
    res = wrap180(b.telemetry.yaw_deg - (75.07 - 52.6))     # 真机极性：右转 → yaw 减小
    assert abs(res) <= float(cfg["deadzone_deg"]) + 1.0, "残余 %.2f° 应落在到达窗内" % res
    outs = _yaw_cmds(b)
    flips = sum(1 for i in range(1, len(outs)) if outs[i] * outs[i - 1] < 0)
    assert flips <= 2, "出力来回反号 %d 次 ⇒ 极限环（不是收敛）" % flips


def test_turn_gains_are_independent_from_ball(monkeypatch):
    """**独立不变量**（2026-09-18 用户定）：转角这套 PID 不跟小球那套联动。"""
    cfg0, src0 = turn_cfg()
    assert src0 == "turn_pid(独立)", "默认来源应为独立那套，实际=%s" % src0
    kp0 = float(cfg0["kp"])
    monkeypatch.setitem(S.comm.ball, "edge_yaw_kp", 1.25)
    cfg1, src1 = turn_cfg()
    assert float(cfg1["kp"]) == pytest.approx(kp0), \
        "改了 ball.edge_yaw_kp 却把转角 PID 带跑了 → 又变成联动（用户要独立）"
    assert src1 == "turn_pid(独立)"
    monkeypatch.setitem(S.comm.motion, "turn_pid",
                        dict(S.comm.motion.get("turn_pid", {}), kp=0.7))
    cfg2, _ = turn_cfg()
    assert float(cfg2["kp"]) == pytest.approx(0.7)

def test_timeout_reports_how_far_it_actually_turned():
    """舵效不足（几乎不转）→ 退出码 4；转了一部分但没到位 → 3。都带实测角度。"""
    b = _FakeBoat(gain=0.2)                       # 极弱
    # sat_max_s=0：本用例测的是**超时**那条保护，先把"满舵+无进展即停"这条新保护关掉，
    #   否则新保护会先中止（两条保护的职责不同，用例要各自隔离）。
    rc, logs = _run_turn(b, deg=90.0, left=True, timeout=3.0, sat_max_s=0.0)
    assert rc == 4, "几乎没转应报 4，实际 %s" % rc
    assert any("超时" in s for s in logs)
    assert any("实测只转了" in s for s in logs)
    assert b.neutral_calls >= 1

    b2 = _FakeBoat(gain=8.0)                      # 能转但很慢
    rc2, _ = _run_turn(b2, deg=90.0, left=True, timeout=3.0, sat_max_s=0.0)
    assert rc2 == 3, "转了一部分应报 3，实际 %s" % rc2

def test_refuses_to_turn_without_telemetry():
    """**默认行为**：没有遥测 → **拒转**（rc=5），一根转向指令都不许发
    （不能"以为转了 90° 其实瞎转"），但必须回中位、不能挂死。**没有盲转备案。**"""
    b = _FakeBoat()
    b.telemetry.yaw_deg = None
    rc, logs = _run_turn(b, deg=90.0, left=True, timeout=5.0, wait_tel_s=0.5)
    assert rc == 5, "缺闭环量应拒转（rc=5），实际 %s" % rc
    assert not _yaw_cmds(b), "拒转时不该发出任何非零舵：%s" % _yaw_cmds(b)
    assert b.neutral_calls >= 1, "拒转也要回中位"
    assert any("拒绝转向" in s for s in logs)

def test_divergence_guard_stops_the_turn():
    """**方向自证**（唯一的兜底，不是测量）：命令朝一边、船朝另一边 ⇒ 停转（rc=6）。

    构造：假船极性与真机相反（+yaw 命令使 yaw **增大**）⇒ 闭环变正反馈。
    必须"转一点点就停"，而不是像板端那样一路转到 120°+ 把门甩出画面。
    """
    b = _FakeBoat(imag_sign=+1.0, start_yaw=0.0, gain=60.0)      # 反极性
    # sat_max_s=0：本用例测**方向自证**，新保护（满舵无进展即停）会抢在前面 ⇒ 隔离掉
    rc, logs = _run_turn(b, deg=30.0, left=True, timeout=20.0, sat_max_s=0.0)
    assert rc == 6, "反极性应被方向自证拦下（rc=6），实际 %s（logs=%s）" % (rc, logs[-2:])
    turned = abs(wrap180(b.telemetry.yaw_deg - 0.0))
    limit = 30.0 + float(turn_cfg()[0]["div_deg"]) + 3.0
    assert turned <= limit, "反极性下实测转了 %.1f° > 上限 %.1f°（兜底没生效）" % (turned, limit)
    assert any("方向自证失败" in s for s in logs)
    assert b.neutral_calls >= 1
    # 关掉兜底（div_deg=0）→ 只能靠超时收场（旧行为：一路转到底，转出 limit 之外）
    b2 = _FakeBoat(imag_sign=+1.0, start_yaw=0.0, gain=60.0)
    rc2, _ = _run_turn(b2, deg=30.0, left=True, timeout=6.0, div_deg=0.0, sat_max_s=0.0)
    turned2 = abs(wrap180(b2.telemetry.yaw_deg - 0.0))
    assert rc2 == 3 and turned2 > limit, \
        "div_deg=0 时应退回旧行为（一路转到 %.1f° 才超时）—— 这就是兜底存在的理由" % turned2


# ======================================================================
# 正航向（gate/heading_align.py）：PnP 目标角 → 转一次 → 结束
# ======================================================================
class _World(object):
    """假船 + 假门：h=机身转角(正=右)，**psi=psi0−h**（2026-09-28 实船改正：右转使 psi **变小**），
    遥测 yaw=h×imag_sign。"""

    def __init__(self, psi0=20.0, imag_sign=BOARD_SIGN, gain=60.0, dt=0.05, stuck=False):
        self.psi0 = float(psi0)
        self.imag_sign = float(imag_sign)
        self.gain = float(gain)
        self.dt = float(dt)
        self.stuck = bool(stuck)
        self.h = 0.0
        self.cmds = []

    @property
    def psi(self):
        return self.psi0 - self.h

    @property
    def yaw_tel(self):
        return self.h * self.imag_sign

    def send(self, yaw_cmd):
        self.cmds.append(float(yaw_cmd))
        if not self.stuck:
            self.h += float(yaw_cmd) * self.gain * self.dt
def _run_hd(aligner, world, frames=1200, lost_after=None, telemetry=True, cfg_over=None,
            on_frame=None):
    """按帧推进 aligner；`on_frame(i)` 是每帧回调（用来数帧/断言状态）。"""
    now = 0
    logs = []
    if cfg_over:
        aligner.cfg.update(cfg_over)
    aligner.log = logs.append
    if not aligner.finished():
        aligner.start(now, psi=world.psi)      # gate_task 的 `_hdg_start` 就这一句
    states = []
    for i in range(frames):
        if on_frame is not None:
            on_frame(i)
        lost = lost_after is not None and i >= lost_after
        st, yaw = aligner.step(now, yaw_telemetry=(world.yaw_tel if telemetry else None),
                               gate_lost=lost)
        world.send(yaw)
        states.append(st)
        if aligner.finished():
            break
        now += int(world.dt * 1000)
    return states, logs, now

def test_converges_with_one_turn():
    """psi=+20° → **右转 20°** → 结束（2026-09-28 实船改正；见 heading_align.start 的注释）。

    一次到位：目标角在起转那一刻冻结，转完就是 DONE，**只转一次**。
    """
    w = _World(psi0=20.0)
    al = HeadingAligner(cfg=dict(hdg_cfg(), max_step_deg=0.0, turn_scale=1.0), log=lambda *a: None)
    _run_hd(al, w)
    assert al.state == DONE, "应收敛，实际 %s（%s）" % (al.state, al.summary())
    assert al.iters == 1, "只该转一次（实际 %d 次）" % al.iters
    assert al.last_dir == "右转", "psi>0 应**右转**（2026-09-28 实船改正），实际 %s" % al.last_dir
    assert al.last_target_deg == pytest.approx(20.0), "目标角就是测到的 psi（%s）" % al.summary()
    assert abs(w.psi) <= float(hdg_cfg()["tol_deg"]) + 1e-6, "转完残余 psi=%.1f°" % w.psi
    nz = [c for c in w.cmds if abs(c) > 1e-9]
    assert nz and all(c > 0 for c in nz), "右转不该出现负舵：%s" % set(nz)
    assert max(abs(c) for c in nz) <= turn_cfg()[0]["out_max"] + 1e-9

def test_left_turn_when_psi_is_negative():
    """psi=-25° → **左转 25°**，舵全为负（2026-09-28 实船改正）。"""
    w = _World(psi0=-25.0)
    al = HeadingAligner(cfg=dict(hdg_cfg(), max_step_deg=0.0, turn_scale=1.0), log=lambda *a: None)
    _run_hd(al, w)
    assert al.state == DONE and al.last_dir == "左转", al.summary()
    nz = [c for c in w.cmds if abs(c) > 1e-9]
    assert nz and all(c < 0 for c in nz), "左转不该出现正舵：%s" % set(nz)
    assert abs(w.psi) <= float(hdg_cfg()["tol_deg"]) + 1e-6

def test_target_angle_is_frozen_while_turning():
    """**转动中不重测**：目标角在起转那一刻冻结成死数。

    结构保证：转向期间 `HeadingAligner.step()` **根本收不到 psi**（入参只有
    now_ms / yaw_telemetry / gate_lost）⇒ 转动中测到的"假已平行"不可能提前结束转向。
    """
    import inspect
    params = list(inspect.signature(HeadingAligner.step).parameters)
    for dead in ("psi_deg", "psi_fresh", "psi_stale"):
        assert dead not in params, "转向中不该再接收 %s（一次到位的结构性保证）：%s" % (dead, params)
    w = _World(psi0=20.0)
    al = HeadingAligner(cfg=dict(hdg_cfg(), max_step_deg=0.0, turn_scale=1.0), log=lambda *a: None)
    seen = {"turn": 0}

    def on_frame(i):
        if al.state == TURN:
            seen["turn"] += 1

    _run_hd(al, w, on_frame=on_frame)
    assert seen["turn"] >= 3, "转向应持续多帧（实际 %d）" % seen["turn"]
    assert al.state == DONE and al.iters == 1, al.summary()
    assert abs(w.psi) <= float(hdg_cfg()["tol_deg"]) + 1e-6, "转完残余 %.1f°" % w.psi


def test_stuck_turn_times_out_then_gives_up():
    """转不动（卡住）→ **只转一次**、超时后 GIVEUP（带残余航向继续走），不反复重试。"""
    w = _World(psi0=60.0, stuck=True)
    al = HeadingAligner(log=lambda *a: None, turn_kwargs=dict(timeout=1.0))
    _, logs, _ = _run_hd(al, w, frames=5000)
    assert al.state == GIVEUP, "应放弃而不是死循环，实际 %s" % al.state
    assert al.iters == 1, "一次到位 ⇒ 只该有一次转向（实际 %d）" % al.iters
    assert any("带残余航向继续走" in s for s in logs)

def test_step_cap_clamps_single_turn_and_zero_means_unlimited():
    """`max_step_deg` 只是安全钳位：**0 = 不设限**（按测到的 psi 转）；>0 才限幅。

    ⚠️ 出厂默认 2026-09-28 起是 **10.0°**（不是 0）⇒ "不设限"要显式配（见 `test_turn_scale_and_max_step_shape_the_issued_angle`）。
    """
    w = _World(psi0=60.0, stuck=True)
    al0 = HeadingAligner(cfg=dict(hdg_cfg(), max_step_deg=0.0, turn_scale=1.0), log=lambda *a: None)
    _run_hd(al0, w, frames=4)
    assert al0.last_target_deg == pytest.approx(60.0), \
        "max_step_deg=0 应不设限（实际 %.1f°）" % al0.last_target_deg
    w2 = _World(psi0=60.0, stuck=True)
    al2 = HeadingAligner(cfg=dict(hdg_cfg(), max_step_deg=25.0, turn_scale=1.0), log=lambda *a: None)
    _run_hd(al2, w2, frames=4)
    assert al2.last_target_deg == pytest.approx(25.0), \
        "配了 max_step_deg=25 应限幅到 25°（实际 %.1f°）" % al2.last_target_deg

def test_small_psi_is_left_alone():
    """|psi| ≤ tol_deg ⇒ 直接 DONE，一根舵都不发（不为了 2~3° 去推一下）。"""
    w = _World(psi0=3.0)
    al = HeadingAligner(log=lambda *a: None)
    _, logs, _ = _run_hd(al, w, frames=200)
    assert al.state == DONE and al.iters == 0, al.summary()
    assert all(abs(c) < 1e-9 for c in w.cmds), "不该有任何转向指令"
    assert any("不转" in s for s in logs)

def test_no_psi_no_turn():
    """没有 PnP 目标角 ⇒ GIVEUP，不转（没有"开环盲转"这条路）。"""
    w = _World(psi0=20.0)
    al = HeadingAligner(log=lambda *a: None)
    st, yaw = al.start(0, psi=None)
    assert st == GIVEUP and abs(yaw) < 1e-9
    assert all(abs(c) < 1e-9 for c in w.cmds)

def test_gate_lost_aborts_immediately():
    """转向中整门丢失 → 立刻中止（不再发任何转向指令），本门不再尝试。"""
    w = _World(psi0=45.0)
    al = HeadingAligner(log=lambda *a: None)
    al.start(0, psi=w.psi)
    now = 0
    for i in range(600):
        st, yaw = al.step(now, yaw_telemetry=w.yaw_tel, gate_lost=False)
        if al.state == TURN and i > 3:                  # 转几帧后丢门
            st, yaw = al.step(now, yaw_telemetry=w.yaw_tel, gate_lost=True)
            assert al.state == ABORTED
            assert abs(yaw) < 1e-9, "丢门时本帧不许再给转向舵"
            break
        w.send(yaw)
        now += 50
    assert al.state == ABORTED and al.finished(), "中止后应视为已结束（放行到下一步）"

def test_disabled_switches_off_completely():
    """`enable: false` → 永远 finished、不发任何指令（一键关掉正航向）。"""
    al = HeadingAligner(cfg=dict(hdg_cfg(), enable=False), log=lambda *a: None)
    assert al.finished()
    st, yaw = al.step(0, yaw_telemetry=0.0)
    assert st == al.state and abs(yaw) < 1e-9

def test_no_telemetry_never_turns():
    """**没有遥测 yaw ⇒ 不转**（ABORTED，且全程零舵）。

    这条守的是用户定的原则：转向只允许闭环，**不留开环盲转备案**
    （`blind_enable` 那种"角度÷假设角速率"定时盲转已删除，不许复活）。
    """
    w = _World(psi0=20.0)
    al = HeadingAligner(log=lambda *a: None, turn_kwargs=dict(wait_tel_s=0.5))
    _, logs, _ = _run_hd(al, w, frames=400, telemetry=False)
    assert al.state == ABORTED, "无遥测应中止，实际 %s" % al.state
    assert all(abs(c) < 1e-9 for c in w.cmds), "无遥测时一根舵都不许发：%s" % set(w.cmds)
    assert any("拒绝转向" in s for s in logs)

def test_p3p_frames_do_not_feed_the_filter():
    """**p3p 的 psi 不许进滤波器**（3 点解欠定，航向 std 极大）。

    做法：同一个 GateTask 先喂一帧 full（建立测量），再喂一帧 p3p（旋转过的、会测出别的值）
    → `_hdg_deg` / `_hdg_ms` 必须保持 full 那帧的值不变。
    """
    import numpy as np
    import cv2
    from common.detector import Det
    from gate.gate_detector import board_camera
    from gate.gate_task import GateTask
    from gate.geometry import object_points
    from tests.tasks.test_gate_flow import _Hub, _Uart

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


def test_overshoot_tolerance_is_measured_against_the_issued_deg():
    """**超转容差 = 10°**（用户 2026-09-28 定："可以超转，但超转不得超过 10 度"）。

    `0 ≤ done_deg − 下发的 deg ≤ overshoot_tol_deg` 即算这一小步到位，且在这窗里**不再触发满舵保护**：
      · 关掉容差 ⇒ 超转后 |err| 与"朝下发角推进"都不再改善 ⇒ 2s 后被掐（现场"被限死"那种）；
      · 超转 ≤ 10° ⇒ 认到位、正常 DONE；**超过 10° 不认**（由下一个新鲜 ψ 带回来）。
    """
    def run(tol, cap=45.0):
        c = TurnCore(deg=20.0, left=True, timeout=30.0, log=lambda *a: None,
                     # out_max 调小只为让"满舵"这条判据在合成航迹上真的生效
                     #   （默认 0.30 需要 |err| ≥ 20.25° 才饱和，测起来不直观）
                     cfg=dict(overshoot_tol_deg=tol, sat_max_s=2.0, sat_progress_deg=5.0,
                              out_max=0.10))
        t, i = 0, 0
        while not c.finished() and t < 15000:
            i += 1
            # 0 → 45°：**每帧 8° 冲过去**（快得跳过 3 帧死区确认），冲到 45° 停住 ⇒ 下发 20°、超转 25°。
            #   必须"跳过死区"才复现现场那种"转过头 ⇒ 回不来 ⇒ 被满舵保护掐死"。
            c.step(t, min(cap, i * 8.0))
            t += 100
        return c, t

    a, _ = run(0.0)                       # 关掉容差：超转后不再改善 ⇒ 被满舵保护掐死
    assert a.state == TurnCore.ABORTED and a.why == "sat", (a.state, a.why)
    b, _ = run(10.0, cap=28.0)             # 冲到 28°（下发 20° ⇒ 超转 8° ≤ 10°）⇒ 认到位
    assert b.state == TurnCore.DONE and 0.0 <= b.overshoot_deg() <= 10.0 + 1e-9, \
        (b.state, b.overshoot_deg())
    c_big, _ = run(10.0, cap=80.0)         # 冲到 80°（超转 60° ≫ 10°）⇒ **不许**判到位
    assert c_big.state != TurnCore.DONE, (c_big.state, c_big.overshoot_deg())

    # 反面：容差**不许**把"只转了一点点"判成到位（只放宽超转一侧）
    c = TurnCore(deg=20.0, left=True, timeout=2.0, log=lambda *a: None,
                 cfg=dict(overshoot_tol_deg=10.0, sat_max_s=0.0, deadzone_deg=6.0))
    t = 0
    while not c.finished() and t < 3000:
        c.step(t, 5.0)                           # 只转了 5°（下发 20°，差 15° > 死区）
        t += 50
    assert c.state != TurnCore.DONE, "未转够 15° 不该判到位（容差只管超转）：%s" % c.state
