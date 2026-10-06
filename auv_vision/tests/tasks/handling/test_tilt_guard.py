# -*- coding: utf-8 -*-
"""tests/tasks/handling/test_tilt_guard.py — **pitch/roll ±50° 保护**（用户 2026-10-06 定）。

三层（一层比一层外）：
1. 指令侧：`TurnCore` 把 pitch/roll 的相对角夹到 ±`comm.motion.axis.max_tilt_deg`（yaw 不受限）；
2. 姿态侧：`AxisMove` 每帧看遥测，**实测**超限就撤指令 + 判失败，让调用方安全收尾；
3. 最外层：看门狗按实测姿态停船（在 `tests/platform/test_base.py` 里验收）。
"""
from __future__ import annotations

import pytest

import base.cfg.settings as S
from common.motion.axis import AxisMove, pitch_up_deg, roll_dump_deg
from common.motion.turn_deg import TurnCore, clamp_tilt_deg, max_tilt_deg
from tests.tasks.handling.test_grab_flow import FakeUart


def test_turn_core_clamps_pitch_and_roll_but_not_yaw():
    lim = max_tilt_deg()
    assert lim == pytest.approx(50.0), "用户定的是 ±50°"
    t = TurnCore(deg=80.0, axis="pitch", log=lambda *a: None)
    assert t.deg == pytest.approx(lim) and t.clamped is True
    assert TurnCore(deg=60.0, axis="roll", log=lambda *a: None).deg == pytest.approx(lim)
    assert TurnCore(deg=30.0, axis="pitch", log=lambda *a: None).clamped is False
    y = TurnCore(deg=180.0, axis="yaw", log=lambda *a: None)
    assert y.deg == pytest.approx(180.0) and y.clamped is False, "yaw 不受限"


def test_clamp_keeps_the_sign_so_leveling_still_works():
    """只夹幅度、保号 ⇒ 「往小里修」（回水平/回正）不会被挡住。"""
    assert clamp_tilt_deg("pitch", -80.0) == (pytest.approx(-50.0), True)
    assert clamp_tilt_deg("roll", 20.0) == (20.0, False)


def test_limit_is_configurable_and_missing_key_does_not_crash(monkeypatch):
    monkeypatch.setitem(S.comm.motion.axis, "max_tilt_deg", 35.0)
    assert max_tilt_deg() == pytest.approx(35.0)
    assert TurnCore(deg=80.0, axis="pitch", log=lambda *a: None).deg == pytest.approx(35.0)
    monkeypatch.setitem(S.comm.grab, "max_tilt_deg", 20.0)      # 任务段可覆盖
    assert max_tilt_deg("grab") == pytest.approx(20.0)
    assert AxisMove("pitch", 80.0, prefix="grab", log=lambda *a: None).deg == pytest.approx(20.0)
    monkeypatch.delitem(S.comm.motion, "axis", raising=False)   # 删掉整段
    assert max_tilt_deg() == pytest.approx(50.0), "删 cfg 要回代码兜底"


def test_axis_move_aborts_when_the_measured_tilt_exceeds_the_limit():
    """★ 姿态侧兜底：下位机真把船转超了 ⇒ 撤指令 + 判失败（不停在半路硬转）。"""
    lim = max_tilt_deg()
    u = FakeUart()
    m = AxisMove("pitch", pitch_up_deg(), log=lambda *a: None)
    assert m.step(0, u) in (AxisMove.RUN, AxisMove.DONE)
    u.telemetry.pitch_deg = lim + 10.0            # 实测超限（比 ±50 多 10°）
    m.step(100, u)
    assert m.state == AxisMove.FAILED and m.why == "tilt_over_limit"
    assert u._pending == 0, "必须撤掉下位机的转角任务（cancel_turn）"
    # 限内不误判
    u2 = FakeUart()
    m2 = AxisMove("pitch", pitch_up_deg(), log=lambda *a: None)
    m2.step(0, u2)
    u2.telemetry.pitch_deg = lim - 1.0
    m2.step(100, u2)
    assert m2.state != AxisMove.FAILED
    # 没遥测 = 不判（信息不足不乱停）
    u3 = FakeUart()
    u3.telemetry.pitch_deg = None
    m3 = AxisMove("pitch", pitch_up_deg(), log=lambda *a: None)
    m3.step(0, u3)
    m3.step(100, u3)
    assert m3.state != AxisMove.FAILED


def test_pitch_up_and_roll_dump_requests_stay_within_the_limit():
    """任务侧真正会下发的两个角（抬头 30°/横倾 30°）本来就在限内；cfg 被改大也夹得住。"""
    assert pitch_up_deg() == pytest.approx(30.0) and roll_dump_deg() == pytest.approx(30.0)
    assert AxisMove("pitch", pitch_up_deg(), log=lambda *a: None).clamped is False
