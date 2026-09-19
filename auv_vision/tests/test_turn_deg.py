# -*- coding: utf-8 -*-
"""tests/test_turn_deg.py — 额定转角（task1_2/turn_deg.py）的离线闭环测试。

被控对象 = 假船：遥测 yaw 按 `yaw_rate_gain × 下发的 yaw DOF × dt` 积分（纯积分器），
模拟两种遥测符号（+DOF 使遥测增大 / 减小）并在 ±180 回绕；时钟与 sleep 都注入
→ 确定性、毫秒级跑完。控制器用的是**项目既有的 common/PID.py**。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base.settings as S                                          # noqa: E402
from task1_2.turn_deg import turn, turn_pid_cfg, wrap180           # noqa: E402


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


def _run(boat, **kw):
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
    rc, logs = _run(b, deg=90.0, left=True, timeout=20.0)
    assert rc == 0, "应到达目标角，logs=%s" % logs[-3:]
    got = wrap180(b.telemetry.yaw_deg - y0)
    want = -90.0 * imag_sign
    assert abs(got - want) <= 8.0, "实测 %.1f° 应≈目标 %.1f°" % (got, want)
    assert b.neutral_calls >= 1, "结束时必须回中位"
    ys = _yaw_cmds(b)
    assert ys and all(y < 0 for y in ys), "左转不该出现正（右转）舵：%s" % set(ys)


def test_turn_right_uses_positive_yaw():
    b = _FakeBoat(imag_sign=1.0, start_yaw=0.0, gain=30.0)
    rc, _ = _run(b, deg=90.0, left=False, timeout=20.0)
    assert rc == 0
    assert abs(wrap180(b.telemetry.yaw_deg - 0.0) - 90.0) <= 8.0
    ys = _yaw_cmds(b)
    assert ys and all(y > 0 for y in ys), "右转应是正舵（+yaw=右转）"


def test_pid_eases_off_near_the_target_and_respects_out_max():
    """**用 PID 的判据**：① 命令不过 out_max；② 近目标时自动收力（不是一路满舵到点）。

    这一条正是"定点转角"比"固定舵量 + 容差"强的地方：远了满舵、近了减速。
    """
    cfg, _src = turn_pid_cfg()
    om = float(cfg["out_max"])
    b = _FakeBoat(imag_sign=1.0, start_yaw=0.0, gain=40.0)
    rc, _ = _run(b, deg=90.0, left=True, timeout=20.0)
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
    cfg0, src0 = turn_pid_cfg()
    assert src0 == "turn_pid(独立)", "默认来源应为独立那套，实际=%s" % src0
    kp0 = float(cfg0["kp"])
    monkeypatch.setitem(S.comm.ball, "edge_yaw_kp", 1.25)
    cfg1, src1 = turn_pid_cfg()
    assert float(cfg1["kp"]) == pytest.approx(kp0), \
        "改了 ball.edge_yaw_kp 却把转角 PID 带跑了 → 又变成联动（用户要独立）"
    assert src1 == "turn_pid(独立)"
    # 显式覆盖仍然生效（cfg 是唯一来源）
    monkeypatch.setitem(S.comm.motion, "turn_pid",
                        dict(S.comm.motion.get("turn_pid", {}), kp=0.7))
    cfg2, _ = turn_pid_cfg()
    assert float(cfg2["kp"]) == pytest.approx(0.7)


def test_timeout_reports_how_far_it_actually_turned():
    """舵效不足（几乎不转）→ 退出码 4；转了一部分但没到位 → 3。都带实测角度。"""
    b = _FakeBoat(gain=0.2)                       # 极弱
    rc, logs = _run(b, deg=90.0, left=True, timeout=3.0)
    assert rc == 4, "几乎没转应报 4，实际 %s" % rc
    assert any("超时" in s for s in logs)
    assert any("实测只转了" in s for s in logs)
    assert b.neutral_calls >= 1

    b2 = _FakeBoat(gain=8.0)                      # 能转但很慢
    rc2, _ = _run(b2, deg=90.0, left=True, timeout=3.0)
    assert rc2 == 3, "转了一部分应报 3，实际 %s" % rc2


def test_refuses_to_turn_without_telemetry():
    """**默认行为**（2026-09-18 用户定：接口只有角度）：没有遥测 → **拒转**（rc=5），
    一根转向指令都不许发（不能"以为转了 90° 其实瞎转"），但必须回中位、不能挂死。"""
    b = _FakeBoat()
    b.telemetry.yaw_deg = None
    rc, logs = _run(b, deg=90.0, left=True, timeout=5.0, wait_tel_s=0.5)
    assert rc == 5, "缺闭环量应拒转（rc=5），实际 %s" % rc
    assert not _yaw_cmds(b), "拒转时不该发出任何非零舵：%s" % _yaw_cmds(b)
    assert b.neutral_calls >= 1, "拒转也要回中位"
    assert any("拒绝转向" in s for s in logs)


def test_blind_turn_is_angle_over_rate():
    """显式 `--blind`（mode=timed）才盲转：时长必须是 **角度 ÷ 角速率假设**，不是拍脑袋的秒数。"""
    b = _FakeBoat()
    b.telemetry.yaw_deg = None
    rc, logs = _run(b, deg=90.0, left=True, timeout=5.0, wait_tel_s=0.5,
                    mode="timed", blind_rate_dps=30.0)      # 90/30 = 3.0s → 60 帧@0.05s
    assert rc == 2
    ys = _yaw_cmds(b)
    assert ys and all(y < 0 for y in ys), "盲转也要按方向给舵"
    # 60 帧 ±2 帧；发送循环是"先发后 sleep"，所以帧数≈时长/周期
    assert 58 <= len(ys) <= 62, "盲转时长应为 90/30=3s（≈60 帧），实际 %d 帧" % len(ys)
    assert len(ys) == pytest.approx(60, abs=2)
    assert any("盲转" in s for s in logs)


def test_known_sign_skips_probe():
    """已知遥测符号（imag_sign=1）时不做探向：第一条命令就是转向舵。"""
    b = _FakeBoat(imag_sign=1.0, start_yaw=0.0)
    rc, _ = _run(b, deg=30.0, left=True, timeout=10.0, imag_sign=1.0)
    assert rc == 0
    assert b.cmd[0][3] != 0.0, "应当直接开始转，而不是先给中性帧等遥测"


def test_wrap180():
    assert wrap180(0.0) == 0.0
    assert wrap180(190.0) == pytest.approx(-170.0)
    assert wrap180(-190.0) == pytest.approx(170.0)
    assert wrap180(360.0) == pytest.approx(0.0)
