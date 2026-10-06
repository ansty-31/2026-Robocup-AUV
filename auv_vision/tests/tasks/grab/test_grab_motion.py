# -*- coding: utf-8 -*-
"""tests/tasks/grab/test_grab_motion.py — 夹取运动原语（`common/motion/axis.py`）。

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
        return max(0.55, self.extra_min_depth_m)

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


def test_pitch_up_sign_is_configurable_and_unverified_by_default():
    """抬头角 = `up_sign × up_deg`；up_sign 是**未验证**项，现场实测后改 cfg（不改代码）。"""
    assert pitch_up_deg() == pytest.approx(30.0)
    assert roll_dump_deg() == pytest.approx(30.0)


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
    u = _Uart(depth=0.70)
    e = EnsureDepth(log=lambda *a: None)
    st, dof = e.step(0, u)
    assert st == EnsureDepth.RUN and dof[2] < 0.0, "应该在下潜"
    u.depth_m = 1.05                                   # 到位（≥ 1.0 − tol）
    st, dof = e.step(100, u)
    assert st == EnsureDepth.DONE and dof == (0.0, 0.0, 0.0, 0.0)


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


def test_grab_cfg_falls_back_when_the_section_is_missing(monkeypatch):
    """删掉整个 `comm.grab` 段也不崩（代码同值兜底）—— 这是工程既有约定。"""
    monkeypatch.setattr(S, "comm", S.Y({k: v for k, v in S.comm.items() if k != "grab"}))
    c = grab_cfg()
    assert c["depth_floor_m"] == pytest.approx(1.0)
    assert c["timeout_ms"] == pytest.approx(45000.0)
    assert pitch_up_deg() == pytest.approx(30.0)
