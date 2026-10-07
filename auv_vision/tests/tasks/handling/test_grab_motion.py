# -*- coding: utf-8 -*-
"""tests/tasks/handling/test_grab_motion.py — 夹取运动原语（`common/motion/axis.py`）。

口径来源：`doc/记录/README_COMMUNICATION.md` §2.2（轴方向与目标：目标 = 当时测量 + 相对角）、
用户 2026-10-06 口令（抬头 30°、任务级限深下限 1.0m、先下潜再抬头）。
**本套用例只证明逻辑与安全属性，不证明水中效果。**
"""
from __future__ import annotations

import pytest

import base.cfg.settings as S
from common.motion.axis import (AxisMove, EnsureDepth, TimedDof, axis_deg, grab_cfg,
                             level_relative_deg, pitch_up_deg, roll_dump_deg)


class _Tel(object):
    def __init__(self, pitch=0.0, roll=0.0, yaw=0.0):
        self.pitch_deg, self.roll_deg, self.yaw_deg = pitch, roll, yaw


class _Uart(object):
    """假下位机：指定角度轴"立即执行"（2 次 poll 后完成），并记录发出的 DOF。"""

    def __init__(self, depth=1.2, pitch=0.0, roll=0.0, yaw=0.0, pitch_gain=1.0):
        self.telemetry = _Tel(pitch, roll, yaw)
        self.depth_m = depth
        self.extra_min_depth_m = 0.0
        self.pitch_gain = float(pitch_gain)
        self.turns = []
        self.dofs = []
        self.estop_active = False
        self._pending = 0
        self._cur = None

    @property
    def effective_min_depth_m(self):
        # 与真件同源：max(cfg 的 site 值, 任务级下限)，别写死米数
        return max(float(S.comm.depth_guard.min_depth_m), self.extra_min_depth_m)

    def set_extra_min_depth(self, m):
        self.extra_min_depth_m = max(0.0, float(m))
        return self.effective_min_depth_m

    def request_turn(self, angle_deg, turn_id=None, axis=1):
        self._cur = (int(axis), float(angle_deg))
        self._pending = 2
        return True

    def poll_turn_complete(self):
        if self._pending <= 0:
            return False
        self._pending -= 1
        if self._pending > 0:
            return False
        axis, deg = self._cur
        self.turns.append((axis, deg))
        if axis == 2:
            self.telemetry.pitch_deg = (self.telemetry.pitch_deg or 0.0) + deg * self.pitch_gain
        elif axis == 3:
            self.telemetry.roll_deg = (self.telemetry.roll_deg or 0.0) + deg
        else:
            self.telemetry.yaw_deg = (self.telemetry.yaw_deg or 0.0) + deg
        return True

    def cancel_turn(self):
        self._pending = 0

    def send_dof(self, surge=0.0, sway=0.0, heave=0.0, yaw=0.0):
        self.dofs.append((float(surge), float(sway), float(heave), float(yaw)))

    def neutral(self):
        pass


def test_pitch_up_sign_is_a_measured_cfg_value():
    """抬头角 = `up_sign × up_deg`；`up_sign` 是**现场单轴实测值**（2026-10-07 板端 = -1.0）。

    ⚠️ 别把 30 写死：极性/角度改 cfg 就变（换固件或换 IMU 安装要重测）。缺键会抛 `MissingCfg`
    —— 极性**故意没有代码兜底**（猜错方向 = 机头往反方向压，比报错危险）。
    """
    from common.motion.axis import grab_node
    from common.cfg.cfgnode import sub
    g = grab_node()
    up = float(sub(g, "pitch")["up_sign"]) * abs(float(sub(g, "pitch")["up_deg"]))
    rl = float(sub(g, "roll")["dump_sign"]) * abs(float(sub(g, "roll")["dump_deg"]))
    assert pitch_up_deg() == pytest.approx(up)
    assert roll_dump_deg() == pytest.approx(rl)
    assert abs(up) == pytest.approx(30.0), "抬头幅度是 30°（符号由实测极性决定）"


def test_level_angle_is_measured_not_hardcoded():
    """回水平的相对角 = `wrap180(ref − now)`（协议 §2.2 的反解），**不是写死的 −30°**。"""
    assert level_relative_deg(0.0, 30.0) == pytest.approx(-30.0)
    assert level_relative_deg(7.0, 31.0) == pytest.approx(-24.0)      # 中途被扰动
    assert level_relative_deg(0.0, 190.0) == pytest.approx(170.0)      # 回绕
    assert level_relative_deg(None, 30.0) is None, "没基准 ⇒ 必须返回 None（不许瞎发）"
    assert level_relative_deg(0.0, None) is None


def test_axis_deg_reads_telemetry_and_survives_missing_fields():
    u = _Uart(pitch=3.5, roll=-2.0, yaw=11.0)
    assert axis_deg(u, "pitch") == pytest.approx(3.5)
    assert axis_deg(u, "roll") == pytest.approx(-2.0)
    assert axis_deg(u, "yaw") == pytest.approx(11.0)
    assert axis_deg(object(), "pitch") is None, "没有 telemetry ⇒ None"
    u.telemetry.pitch_deg = None
    assert axis_deg(u, "pitch") is None


def test_axis_move_uses_the_right_axis_and_records_the_reference():
    u = _Uart(pitch=5.0)
    m = AxisMove("pitch", 30.0)
    for i in range(10):
        m.step(i * 100, u)
        if m.finished:
            break
    assert m.state == AxisMove.DONE
    assert u.turns == [(2, 30.0)], "pitch 必须落在 byte[7]=2 上，实际 %s" % u.turns
    assert m.ref_deg == pytest.approx(5.0), "必须记下动作前的遥测角当回位基准"


def test_axis_move_fails_closed_when_the_lower_board_rejects():
    class _NoTurn(_Uart):
        def request_turn(self, *a, **kw):
            raise TypeError("old lower board")

    m = AxisMove("pitch", 30.0, log=lambda *a: None)
    for i in range(5):
        m.step(i * 100, _NoTurn())
        if m.finished:
            break
    assert m.state == AxisMove.FAILED and m.why == "send_failed"


def test_ensure_depth_descends_until_the_floor_then_stops():
    """深度不够 ⇒ 返回下潜 DOF（heave<0）；到位就停（**不再给下潜 DOF**）。"""
    e = EnsureDepth(log=lambda *a: None)
    floor = float(e.floor)                             # ★ 从 cfg 取，别写死米数
    u = _Uart(depth=floor - 0.10)                      # 比下限浅 10cm ⇒ 该下潜
    st, dof = e.step(0, u)
    assert st == EnsureDepth.RUN and dof[2] < 0.0, "应该在下潜"
    u.depth_m = floor + 0.05                           # 到位（严格 ≥ 下限）
    st, dof = e.step(100, u)
    assert st == EnsureDepth.DONE and dof == (0.0, 0.0, 0.0, 0.0)


def test_ensure_depth_criterion_is_strict_to_the_floor():
    """★ 到位判据**严格到下限**（`>= floor`），不许拿容差去啃硬边界。

    为什么（2026-10-06 把翘头工作深度压到全局限深那个量级时暴露的）：
    原来写的是 `floor - tol` ⇒ 容差会把"到位"判到**下限之下**（而那已经在下限之外）、
    上浮又被禁 ⇒ "状态说可以抬头"和"保护说不能上浮"自相矛盾。
    """
    u = _Uart(depth=0.70)
    e = EnsureDepth(log=lambda *a: None)          # floor 取 cfg：comm.grab.depth_floor_m
    floor = float(e.floor)
    u.depth_m = floor - 0.01                      # 差 1cm ⇒ 还不算到位（继续压）
    st, dof = e.step(0, u)
    assert st == EnsureDepth.RUN and dof[2] < 0.0, "差下限一点点也必须继续压"
    u.depth_m = floor                             # 恰好到下限 ⇒ 放行
    assert e.step(100, u)[0] == EnsureDepth.DONE


def test_working_depth_is_the_shallowest_the_invariant_allows():
    """★ 翘头工作深度（用户定的 0.6~0.5）落在**机制允许的最浅值**上，且不含糊地卡住 0.5。"""
    import base.cfg.settings as S
    from common.motion.axis import grab_cfg
    site = float(S.comm.depth_guard.min_depth_m)
    floor = float(grab_cfg()["depth_floor_m"])
    assert 0.5 <= floor <= 0.6, "用户要的工作深度区间是 0.6~0.5"
    assert floor >= site, "任务级下限**不许低于**现场定死的全局下限"
    u = _Uart(depth=1.0)
    assert u.set_extra_min_depth(0.50) == pytest.approx(site), \
        "写 0.5 会被全局下限抬回 %.2f（这就是 0.50 做不到的原因）" % site
    assert u.set_extra_min_depth(floor) == pytest.approx(floor)


def test_ensure_depth_fails_without_telemetry_instead_of_guessing():
    u = _Uart(depth=None)
    e = EnsureDepth(log=lambda *a: None)
    st, dof = e.step(0, u)
    assert st == EnsureDepth.FAILED and dof == (0.0, 0.0, 0.0, 0.0)
    assert e.why == "no_depth_telemetry"


def test_ensure_depth_times_out_and_gives_up():
    u = _Uart(depth=0.40)                              # 永远到不了
    e = EnsureDepth(log=lambda *a: None)
    st = None
    for i in range(2000):
        st, _dof = e.step(i * 100, u)
        if st in (EnsureDepth.DONE, EnsureDepth.FAILED):
            break
    assert st == EnsureDepth.FAILED and e.why == "depth_timeout"


def test_timed_dof_only_advances_state_and_never_sends():
    """★ 下发点唯一：`TimedDof` **只**推进状态并给 DOF，绝不自己发帧。

    （原来"子动作发一帧 + 主循环又补一帧全零"，同帧两次 send_dof —— 靠 50ms 节流侥幸没把推力
    抹掉，但 `_dof_target` 已被清零。这条用例就是钉住它不会再回来。）
    """
    u = _Uart()
    t = TimedDof(dur_s=0.5, name="下压", heave=-0.35, log=lambda *a: None)
    assert t.step(1000) == TimedDof.RUN and t.dof == (0.0, 0.0, -0.35, 0.0)
    assert t.step(1499) == TimedDof.RUN
    assert t.step(1500) == TimedDof.DONE
    assert u.dofs == [], "TimedDof 不该自己发帧"


def test_grab_cfg_raises_when_the_section_is_missing(monkeypatch):
    """★ 2026-10-07 用户定：**删掉兜底** —— 删掉整个 `comm.grab` 段**必须报名字**，
    不再"静默用代码里的同值默认"（那正是"改参数要动两处"的来源）。"""
    import pytest
    from common.cfg.cfgnode import MissingCfg
    monkeypatch.setattr(S, "comm", S.Y({k: v for k, v in S.comm.items() if k != "grab"}))
    with pytest.raises(MissingCfg):
        grab_cfg()
    with pytest.raises(MissingCfg):
        pitch_up_deg()
