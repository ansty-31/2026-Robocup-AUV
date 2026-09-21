# -*- coding: utf-8 -*-
"""tests/tasks/test_motion.py — 运动原语：额定转角 `TurnCore` + 离散正航向 `HeadingAligner`。

两半都是"遥测 yaw 闭环"状态机（转角是原语；正航向是 gate ALIGN 的子状态），
所以离线用**假船**测：遥测 yaw 按 `角速率 × 下发的 yaw DOF × dt` 积分，
时钟与 sleep 全部注入 → 确定性、毫秒级跑完。
  · 转角（前半）：`common/turn_deg.py`；**缺遥测拒转**是默认行为（要盲转必须显式）。
  · 正航向（后半）：`gate/heading_align.py`；「测→转→停稳→再测」，只吃 full 帧的 psi。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base.settings as S                                          # noqa: E402
from common.turn_deg import turn, turn_cfg, wrap180                # noqa: E402
from gate.heading_align import (ABORTED, DONE, GIVEUP, MEASURE, SETTLE, TURN,
                                HeadingAligner, hdg_cfg)            # noqa: E402

# ======================================================================
# 额定转角（common/turn_deg.py）
# ======================================================================
class _Tel(object):
    def __init__(self):
        self.yaw_deg = 170.0        # 故意放在回绕边界附近
class _FakeBoat(object):
    """假船：下发的 yaw DOF → 遥测 yaw 按 imag_sign 方向积分（纯积分器，无延迟）。"""

    def __init__(self, imag_sign=1.0, gain=60.0, start_yaw=170.0, dt=0.05):
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

@pytest.mark.parametrize("imag_sign", (1.0, -1.0))
def test_turn_left_reaches_rated_angle(imag_sign):
    """左转 90°：无论遥测正方向如何，遥测转角都应达到 -90°×imag_sign。

    起点 170°（回绕边界附近）→ 左转 90° 会跨过 ±180，所以这条同时验回绕。
    """
    b = _FakeBoat(imag_sign=imag_sign)
    y0 = b.telemetry.yaw_deg
    rc, logs = _run_turn(b, deg=90.0, left=True, timeout=20.0)
    assert rc == 0, "应到达目标角，logs=%s" % logs[-3:]
    got = wrap180(b.telemetry.yaw_deg - y0)
    want = -90.0 * imag_sign
    assert abs(got - want) <= 8.0, "实测 %.1f° 应≈目标 %.1f°" % (got, want)
    assert b.neutral_calls >= 1, "结束时必须回中位"
    ys = _yaw_cmds(b)
    assert ys and all(y < 0 for y in ys), "左转不该出现正（右转）舵：%s" % set(ys)

def test_turn_right_uses_positive_yaw():
    b = _FakeBoat(imag_sign=1.0, start_yaw=0.0, gain=30.0)
    rc, _ = _run_turn(b, deg=90.0, left=False, timeout=20.0)
    assert rc == 0
    assert abs(wrap180(b.telemetry.yaw_deg - 0.0) - 90.0) <= 8.0
    ys = _yaw_cmds(b)
    assert ys and all(y > 0 for y in ys), "右转应是正舵（+yaw=右转）"

def test_pid_eases_off_near_the_target_and_respects_out_max():
    """**用 PID 的判据**：① 命令不过 out_max；② 近目标时自动收力（不是一路满舵到点）。

    这一条正是"定点转角"比"固定舵量 + 容差"强的地方：远了满舵、近了减速。
    """
    cfg, _src = turn_cfg()
    om = float(cfg["out_max"])
    b = _FakeBoat(imag_sign=1.0, start_yaw=0.0, gain=40.0)
    rc, _ = _run_turn(b, deg=90.0, left=True, timeout=20.0)
    assert rc == 0
    ys = [abs(y) for y in _yaw_cmds(b)]
    assert max(ys) <= om + 1e-9, "命令不该超过限幅 %.2f（实际 %.3f）" % (om, max(ys))
    # 最后 3 条非零命令应明显小于峰值（收力）
    assert min(ys[-3:]) < 0.5 * max(ys), \
        "近目标没收力（峰值 %.3f，末段 %s）→ 说明不是 PID 闭环" % (max(ys), ys[-3:])

def test_turn_gains_are_independent_from_ball(monkeypatch):
    """**独立不变量**（2026-09-18 用户定）：转角这套 PID 不跟小球那套联动。

    小球是"像素误差居中"的调参，转角是"额定角度机动"，量纲/工况都不同；
    用户明确要独立（"0.3 有点大"）。所以：改 `comm.ball.edge_yaw_kp` **不得**影响转角，
    反之亦然；来源必须报 `turn_pid(独立)`。
    """
    cfg0, src0 = turn_cfg()
    assert src0 == "turn_pid(独立)", "默认来源应为独立那套，实际=%s" % src0
    kp0 = float(cfg0["kp"])
    monkeypatch.setitem(S.comm.ball, "edge_yaw_kp", 1.25)
    cfg1, src1 = turn_cfg()
    assert float(cfg1["kp"]) == pytest.approx(kp0), \
        "改了 ball.edge_yaw_kp 却把转角 PID 带跑了 → 又变成联动（用户要独立）"
    assert src1 == "turn_pid(独立)"
    # 显式覆盖仍然生效（cfg 是唯一来源）
    monkeypatch.setitem(S.comm.motion, "turn_pid",
                        dict(S.comm.motion.get("turn_pid", {}), kp=0.7))
    cfg2, _ = turn_cfg()
    assert float(cfg2["kp"]) == pytest.approx(0.7)

def test_timeout_reports_how_far_it_actually_turned():
    """舵效不足（几乎不转）→ 退出码 4；转了一部分但没到位 → 3。都带实测角度。"""
    b = _FakeBoat(gain=0.2)                       # 极弱
    rc, logs = _run_turn(b, deg=90.0, left=True, timeout=3.0)
    assert rc == 4, "几乎没转应报 4，实际 %s" % rc
    assert any("超时" in s for s in logs)
    assert any("实测只转了" in s for s in logs)
    assert b.neutral_calls >= 1

    b2 = _FakeBoat(gain=8.0)                      # 能转但很慢
    rc2, _ = _run_turn(b2, deg=90.0, left=True, timeout=3.0)
    assert rc2 == 3, "转了一部分应报 3，实际 %s" % rc2

def test_refuses_to_turn_without_telemetry():
    """**默认行为**（2026-09-18 用户定：接口只有角度）：没有遥测 → **拒转**（rc=5），
    一根转向指令都不许发（不能"以为转了 90° 其实瞎转"），但必须回中位、不能挂死。"""
    b = _FakeBoat()
    b.telemetry.yaw_deg = None
    rc, logs = _run_turn(b, deg=90.0, left=True, timeout=5.0, wait_tel_s=0.5)
    assert rc == 5, "缺闭环量应拒转（rc=5），实际 %s" % rc
    assert not _yaw_cmds(b), "拒转时不该发出任何非零舵：%s" % _yaw_cmds(b)
    assert b.neutral_calls >= 1, "拒转也要回中位"
    assert any("拒绝转向" in s for s in logs)

def test_blind_turn_is_angle_over_rate():
    """显式 `--blind`（mode=timed）才盲转：时长必须是 **角度 ÷ 角速率假设**，不是拍脑袋的秒数。"""
    b = _FakeBoat()
    b.telemetry.yaw_deg = None
    rc, logs = _run_turn(b, deg=90.0, left=True, timeout=5.0, wait_tel_s=0.5,
                    mode="timed", blind_rate_dps=30.0)      # 90/30 = 3.0s → 60 帧@0.05s
    assert rc == 2
    ys = _yaw_cmds(b)
    assert ys and all(y < 0 for y in ys), "盲转也要按方向给舵"
    # 60 帧 ±2 帧；发送循环是"先发后 sleep"，所以帧数≈时长/周期
    assert 58 <= len(ys) <= 62, "盲转时长应为 90/30=3s（≈60 帧），实际 %d 帧" % len(ys)
    assert len(ys) == pytest.approx(60, abs=2)
    assert any("盲转" in s for s in logs)


# ======================================================================
# 离散正航向（gate/heading_align.py）
# ======================================================================
class _World(object):
    """假船 + 假门：h=机身转角(正=右)，psi=psi0−h，遥测 yaw=h×imag_sign。"""

    def __init__(self, psi0=20.0, imag_sign=1.0, gain=60.0, dt=0.05, stuck=False):
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
def _run_hd(aligner, world, frames=1200, psi_fresh=True, lost_after=None,
         telemetry=True, cfg_over=None):
    now = 0
    logs = []
    if cfg_over:
        aligner.cfg.update(cfg_over)
    aligner.log = logs.append
    states = []
    for i in range(frames):
        lost = lost_after is not None and i >= lost_after
        st, yaw = aligner.step(now, psi_deg=world.psi, psi_fresh=psi_fresh,
                               yaw_telemetry=(world.yaw_tel if telemetry else None),
                               gate_lost=lost)
        aligner.psi_meas = aligner.psi_meas
        world.send(yaw)
        states.append(st)
        if aligner.finished() or st in (DONE, GIVEUP, ABORTED):
            break
        now += int(world.dt * 1000)
    return states, logs, now

@pytest.mark.parametrize("imag_sign", (1.0, -1.0))
def test_converges_with_one_turn(imag_sign):
    """psi=+20°（门法向偏画面右 = 机身左偏）→ 应**右转** ~20° → 重测后 |psi|≤阈值 → DONE。"""
    w = _World(psi0=20.0, imag_sign=imag_sign)
    al = HeadingAligner(log=lambda *a: None)
    states, logs, _now = _run_hd(al, w, frames=1200, cfg_over=dict(tol_deg=8.0))
    assert al.state == DONE, "应收敛，实际 %s（logs=%s）" % (al.state, logs[-3:])
    assert al.iters == 1, "一次转向应够（实际 %d 次）" % al.iters
    assert al.last_dir == "右转", "psi>0 应右转，实际 %s" % al.last_dir
    assert abs(w.psi) <= 8.0 + 1e-6, "转完残余 psi=%.1f° 应 ≤ 阈值" % w.psi
    # 转向命令是正（右转）且不超过限幅
    nz = [c for c in w.cmds if abs(c) > 1e-9]
    assert nz and all(c > 0 for c in nz), "右转不该出现负舵：%s" % set(nz)
    assert max(abs(c) for c in nz) <= turn_cfg()[0]["out_max"] + 1e-9, \
        "转向舵不该超过 motion.turn_pid.out_max"
    assert any("已与光轴平行" in s for s in logs)

def test_no_remeasure_while_turning():
    """**不在转动中重测**：转向期间即使喂进"已经平行"的 psi，也不能因此提前收敛。

    每个 TURN 帧之间我们故意把 psi 改成 0（模拟"转动中测量不可信/会骗人"），
    aligner 必须仍然把冻结的那次转向走完，然后才重新测量。
    """
    w = _World(psi0=20.0)
    al = HeadingAligner(log=lambda *a: None)
    now = 0
    turn_frames = 0
    saw_zero_psi_during_turn = False
    for i in range(2000):
        if al.state == TURN:
            turn_frames += 1
            psi = 0.0                      # ← 转动中喂"假的已平行"
            saw_zero_psi_during_turn = True
        else:
            psi = w.psi
        st, yaw = al.step(now, psi_deg=psi, psi_fresh=True,
                          yaw_telemetry=w.yaw_tel)
        w.send(yaw)
        now += int(w.dt * 1000)
        if al.state in (DONE, GIVEUP, ABORTED):
            break
    assert saw_zero_psi_during_turn, "用例前提：应确实经历过 TURN"
    assert turn_frames >= 3, "转向应持续多帧（实际 %d 帧）" % turn_frames
    assert al.iters >= 1, "应至少完成一次转向（实际 %d）" % al.iters
    # 转向期间那 5 帧"假 0"没有被当成测量样本 → 收敛靠的是转动之后的真实测量
    assert al.state in (DONE, GIVEUP), "应给出结论，实际 %s" % al.state

def test_gives_up_after_max_iters_and_continues():
    """迭代用完仍不达标 → GIVEUP（**带残余航向继续走**），且次数严格受限。"""
    w = _World(psi0=60.0, stuck=True)      # 卡住：怎么转 psi 都不变
    # 单次转向超时压到 1s，好让 3 次都装进总超时里（真实默认：转向 8s + 总 30s）
    al = HeadingAligner(log=lambda *a: None, turn_kwargs=dict(timeout=1.0))
    states, logs, _ = _run_hd(al, w, frames=5000)
    assert al.state == GIVEUP, "应放弃而不是死循环，实际 %s" % al.state
    assert al.iters == int(al.cfg["max_iters"]), \
        "转向次数应正好等于 max_iters=%d，实际 %d" % (al.cfg["max_iters"], al.iters)
    assert any("带残余航向继续走" in s for s in logs)

def test_no_cap_by_default_and_cap_when_configured():
    """**单次转角不设限是默认**（用户定：测多少就一次转多少）；只有显式配 >0 才限幅。"""
    # **出厂配置必须是「不限制」**（用户定）：只测代码默认不够，配置被改回去也要报红
    from base.settings import comm as _C
    assert float(_C.gate.hdg.max_step_deg) == 0.0, \
        "出厂 cfg 的 gate.hdg.max_step_deg 应为 0（=不限制）"
    # 默认（配置里 max_step_deg=0）→ 一帧就按测得的 60° 转，不分次
    w = _World(psi0=60.0, stuck=True)
    al = HeadingAligner(cfg=dict(hdg_cfg(), max_step_deg=0.0), log=lambda *a: None)
    _run_hd(al, w, frames=200)
    assert al.last_target_deg == pytest.approx(60.0), \
        "默认不设限：首次目标应等于测得的 60°，实际 %.1f°" % al.last_target_deg
    # 显式配成 25° → 恢复限幅（安全阀还在）
    w2 = _World(psi0=60.0, stuck=True)
    al2 = HeadingAligner(cfg=dict(hdg_cfg(), max_step_deg=25.0), log=lambda *a: None)
    _run_hd(al2, w2, frames=200)
    assert al2.last_target_deg == pytest.approx(25.0), \
        "配了 max_step_deg=25 应限幅到 25°，实际 %.1f°" % al2.last_target_deg

def test_settle_waits_until_rotation_stops():
    """静止窗：遥测 yaw 还在变时**不许**进入测量（转完还没停稳就测会得到坏角度）。"""
    al = HeadingAligner(log=lambda *a: None)
    al.cfg.update(dict(settle_ms=400, settle_tol_deg=2.0))
    now = 0
    moving = [30.0]                       # 一直以 30°/帧在动
    reached_measure_at = None
    for i in range(40):
        moving[0] += 30.0                 # 遥测 yaw 持续变化
        st, _y = al.step(now, psi_deg=20.0, psi_fresh=True, yaw_telemetry=moving[0])
        if st == MEASURE and reached_measure_at is None:
            reached_measure_at = i
        now += 50
    assert reached_measure_at is None, "还在转就进了测量（第 %s 帧）" % reached_measure_at
    assert al.state == SETTLE, "应停在静止窗，实际 %s" % al.state

def test_gate_lost_aborts_immediately():
    """转向中整门丢失 → 立刻中止（不再发任何转向指令），本门不再尝试。"""
    w = _World(psi0=45.0)
    al = HeadingAligner(log=lambda *a: None)
    states, logs, _ = _run_hd(al, w, frames=600, lost_after=None)
    # 重新来一次：在第一次转向中途丢门
    al.reset()
    w2 = _World(psi0=45.0)
    now = 0
    turns = 0
    for i in range(600):
        st, yaw = al.step(now, psi_deg=w2.psi, psi_fresh=True,
                          yaw_telemetry=w2.yaw_tel, gate_lost=False)
        if al.state == TURN:
            turns += 1
            if turns > 3:                       # 转了几帧后丢门
                st, yaw = al.step(now, psi_deg=w2.psi, psi_fresh=True,
                                  yaw_telemetry=w2.yaw_tel, gate_lost=True)
                assert al.state == ABORTED
                assert abs(yaw) < 1e-9, "丢门时本帧不许再给转向舵"
                break
        w2.send(yaw)
        now += 50
    assert al.state == ABORTED and al.finished(), "中止后应视为已结束（放行到下一步）"

def test_disabled_switches_off_completely():
    """`enable: false` → 永远 finished、不发任何指令（可一键回退到旧行为）。"""
    al = HeadingAligner(cfg=dict(hdg_cfg(), enable=False), log=lambda *a: None)
    assert al.finished()
    st, yaw = al.step(0, psi_deg=30.0, psi_fresh=True, yaw_telemetry=0.0)
    assert st == al.state and abs(yaw) < 1e-9


def test_stale_measurement_is_not_used():
    """**不新鲜的 psi 不算测量**（p3p / 旧帧 / 非 full）→ 攒不够就 GIVEUP，绝不乱转。"""
    w = _World(psi0=30.0)
    al = HeadingAligner(log=lambda *a: None)
    states, logs, _ = _run_hd(al, w, frames=3000, psi_fresh=False)
    assert al.state == GIVEUP, "应放弃，实际 %s" % al.state
    assert al.iters == 0, "没有可靠测量就不该转，实际转了 %d 次" % al.iters
    assert all(abs(c) < 1e-9 for c in w.cmds), "不该有任何转向指令"
    assert any("没攒够" in s for s in logs)

    # 没有遥测（既无法确认停稳、也无法闭环）同样只能放弃，且一根舵都不发
    w2 = _World(psi0=20.0)
    al2 = HeadingAligner(log=lambda *a: None)
    _run_hd(al2, w2, frames=5000, telemetry=False)
    assert al2.state == GIVEUP
    assert all(abs(c) < 1e-9 for c in w2.cmds)

def test_p3p_frames_do_not_feed_the_filter():
    """**p3p 的 psi 不许进滤波器**（3 点解欠定，实测航向 std 64~69°）。

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

    # 同一份新鲜度判据：刚测到算新鲜、超过 fresh_ms 算过期、p3p 任何时候都不算样本
    fresh_ms = float(task._hdg_cfg.get("fresh_ms", 800.0))
    assert task._hdg_fresh("full", kept_ms) is True
    assert task._hdg_fresh("full", kept_ms + fresh_ms + 1) is False, "超过 fresh_ms 必须判过期"
    assert task._hdg_fresh("p3p", kept_ms) is False, "p3p 任何时候都不算可用测量"
