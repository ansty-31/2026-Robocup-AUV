# -*- coding: utf-8 -*-
"""tests/tasks/grab/test_drop.py — 横倾放球/倒球共享件（`common/motion/drop.py`）。

它是**任务三（倒错球）与任务四（放球）共用**的那一个动作，所以验收重点是：
步骤顺序、回正按遥测算、**限深下限只抬不降且恢复运行前的值**、缺 roll 遥测就停手、
以及"能单独当脚本跑"。
"""
from __future__ import annotations

import numpy as np  # noqa: F401  （与同目录用例保持同一导入环境）
import pytest

import base.cfg.settings as S
from common.motion.axis import ZERO_DOF
from common.motion.drop import BallDropSequence, _steps_of


class _Tel(object):
    def __init__(self, roll=0.0):
        self.roll_deg = roll
        self.pitch_deg = 0.0
        self.yaw_deg = 0.0


class _Uart(object):
    """假下位机：指定角度轴 2 次 poll 完成；记录 DOF 与限深下限的历史。"""

    sim = False

    def __init__(self, roll=0.0, floor=0.0, roll_ok=True):
        self.telemetry = _Tel(roll)
        self.extra_min_depth_m = float(floor)
        self.roll_ok = roll_ok
        self.floor_calls = []
        self.turns = []
        self.dofs = []
        self.stop_hard_calls = 0
        self.neutral_calls = 0
        self._pending = 0
        self._cur = None

    @property
    def effective_min_depth_m(self):
        return max(0.55, self.extra_min_depth_m)

    def set_extra_min_depth(self, m):
        self.extra_min_depth_m = max(0.0, float(m))
        self.floor_calls.append(self.extra_min_depth_m)
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
        if axis == 3:
            self.telemetry.roll_deg = None if not self.roll_ok else \
                (self.telemetry.roll_deg or 0.0) + deg
        return True

    def cancel_turn(self):
        self._pending = 0

    def send_dof(self, surge=0.0, sway=0.0, heave=0.0, yaw=0.0):
        self.dofs.append((float(surge), float(sway), float(heave), float(yaw)))

    def close(self):
        pass

    def stop_hard(self, *a, **kw):
        self.stop_hard_calls += 1
        return True

    def neutral(self):
        self.neutral_calls += 1


def _run(seq, u, frames=400, dt=100):
    """逐帧推进；返回下发过的非零 DOF 帧序列。"""
    now, sent = 0, []
    for _ in range(frames):
        st, dof = seq.step(now, u)
        if dof != ZERO_DOF:
            sent.append(dof)
        if seq.finished:
            break
        now += dt
    return sent


# --------------------------------------------------------------------------- #
# 步骤表
# --------------------------------------------------------------------------- #
def test_dump_and_release_have_their_own_step_tables():
    dump = [s.get("name") for s in _steps_of("dump")]
    rel = [s.get("name") for s in _steps_of("release")]
    assert dump[0] == "右移" and "横倾倒出" in dump and "回正" in dump and dump[-1] == "停稳"
    assert rel == ["横倾放球", "断动力停住"], "任务四放球 = 横倾 → 断动力，不后退不回正：%s" % rel


def test_steps_come_from_cfg_and_fall_back_when_deleted(monkeypatch):
    monkeypatch.setitem(S.comm.motion.drop, "dump",
                        S.Y(dict(steps=[{"name": "只走一步", "s": 0.5}])))
    seq = BallDropSequence("dump", log=lambda *a: None)
    assert [s["name"] for s in seq.steps] == ["只走一步"]
    monkeypatch.setitem(S.comm.motion.drop, "dump", S.Y({}))       # 删掉 steps
    assert [s.get("name") for s in _steps_of("dump")][0] == "右移", "删 cfg 要回代码兜底"


# --------------------------------------------------------------------------- #
# 序列行为
# --------------------------------------------------------------------------- #
def test_dump_sequence_order_and_measured_roll_back():
    """右移 → 横倾 → 保持 → **按遥测回正** → 左移 → 后退 → 停稳。"""
    u = _Uart(roll=4.0)
    seq = BallDropSequence("dump", log=lambda *a: None)
    sent = _run(seq, u)
    assert seq.state == BallDropSequence.DONE
    assert u.turns == [(3, 30.0), (3, -30.0)], \
        "横倾 +30° 后必须回到**横倾前测到的** roll（相对角 −30°），实际 %s" % (u.turns,)
    sway_signs = [d[1] for d in sent if abs(d[1]) > 0]
    assert sway_signs[0] > 0 and sway_signs[-1] < 0, "先右移、再左移回来：%s" % sway_signs
    assert any(d[0] < 0 for d in sent), "最后要后退"
    assert [d for d in sent if abs(d[1]) > 0][0][1] == pytest.approx(0.30)


def test_release_sequence_is_tilt_then_power_cut_only():
    """任务四放球：横倾 → 断动力 3s；**不平移、不后退、不回正**。"""
    u = _Uart(roll=0.0)
    seq = BallDropSequence("release", log=lambda *a: None)
    sent = _run(seq, u)
    assert seq.state == BallDropSequence.DONE
    assert u.turns == [(3, 30.0)]
    assert sent == [], "放球阶段不该有平推/后退 DOF，实际 %s" % sent
    assert u.telemetry.roll_deg == pytest.approx(30.0), "横倾目标保持"


# --------------------------------------------------------------------------- #
# 安全属性
# --------------------------------------------------------------------------- #
def test_drop_raises_the_floor_but_restores_the_previous_value():
    """★ 横倾收紧限深（只抬不降），结束**恢复运行前的值** —— 不能把任务三自己抬的 1.0m 撤掉。"""
    u = _Uart(floor=1.0)                     # 任务三已经抬到 1.0m
    seq = BallDropSequence("dump", log=lambda *a: None)
    _run(seq, u)
    assert seq.state == BallDropSequence.DONE
    assert 1.0 in u.floor_calls, "序列期间下限必须 ≥ 1.0m"
    assert u.extra_min_depth_m == pytest.approx(1.0), \
        "结束必须恢复成运行前的 1.0m，而不是 0（撤掉任务的收紧）"

    u2 = _Uart(floor=0.0)                    # 独立运行（没人抬过）
    _run(BallDropSequence("release", log=lambda *a: None), u2)
    assert u2.extra_min_depth_m == pytest.approx(0.0), "没人抬过就恢复 0"


def test_drop_fails_closed_without_roll_telemetry():
    """拿不到 roll 遥测 ⇒ 回不了正 ⇒ **停手**（横着开船不安全），且不再继续走后面的步骤。"""
    u = _Uart(roll=0.0, roll_ok=False)
    seq = BallDropSequence("dump", log=lambda *a: None)
    _run(seq, u)
    assert seq.state == BallDropSequence.FAILED
    assert seq.why == "no_roll_telemetry"
    assert u.turns == [(3, 30.0)], "只许走完已经发出的横倾，不许再发回正/平移"


def test_drop_script_entry_runs_and_hard_stops(monkeypatch):
    """能单独当脚本跑（台架/水池只测这个动作）：退出码 0 + 收尾硬停。"""
    import sys as _sys
    from common.motion import drop as D

    monkeypatch.setitem(S.comm.motion.drop, "release",
                        S.Y(dict(steps=[{"name": "空步", "s": 0.0}])))
    u = _Uart(roll=0.0)
    monkeypatch.setattr("base.hw.uart.UartController", lambda *a, **kw: u)
    monkeypatch.setattr(_sys, "argv", ["drop.py", "--what", "release", "--timeout", "5"])
    assert D.main() == 0
    assert u.stop_hard_calls >= 1, "脚本收尾必须硬停"
