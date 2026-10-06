# -*- coding: utf-8 -*-
"""tests/tasks/place/test_place_flow.py — 任务四「放置」相位机（`handling/handling_task.py`）。

验收重点（用户 2026-10-06 口径）：运输**保持 roll 平衡**（避免球滚出）→ 到点 → 横倾放球
→ 停动力 3s；以及安全闸/限深/急停/无遥测的边界。**离线用例，不证明水中效果。**
"""
from __future__ import annotations

import pytest

import base.cfg.settings as S
from handling.handling_task import PH_PLACE_RELEASE, PH_PLACE_STOP, PH_PLACE_TRANSPORT, PlaceTask, place_cfg


class _Tel(object):
    def __init__(self, roll=0.0):
        self.roll_deg = roll
        self.pitch_deg = 0.0
        self.yaw_deg = 0.0


class FakeUart(object):
    sim = False

    def __init__(self, roll=0.0, roll_ok=True):
        self.telemetry = _Tel(roll)
        self.roll_ok = roll_ok
        self.extra_min_depth_m = 0.0
        self.floor_calls = []
        self.turns = []
        self.dofs = []
        self.estop_active = False
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
        if axis == 3 and self.roll_ok:
            self.telemetry.roll_deg = (self.telemetry.roll_deg or 0.0) + deg
        return True

    def cancel_turn(self):
        self._pending = 0

    def send_dof(self, surge=0.0, sway=0.0, heave=0.0, yaw=0.0):
        self.dofs.append((float(surge), float(sway), float(heave), float(yaw)))

    def neutral(self):
        self.neutral_calls += 1


def _task(monkeypatch, uart, calibrated=True):
    monkeypatch.setitem(S.comm.place, "calibrated", bool(calibrated))
    return PlaceTask(uart, None, 1280, 720, log=lambda *a: None)


def _drive(task, uart, frames=400, dt=100):
    now, phases = 0, []
    for _ in range(frames):
        st = task.process(None, now)
        if not phases[-1:] == [task.last_info["phase"]]:
            phases.append(task.last_info["phase"])
        if st == S.STATUS_DONE:
            break
        now += dt
    return phases, now


def test_uncalibrated_is_skipped_and_never_moves(monkeypatch):
    uart = FakeUart()
    task = _task(monkeypatch, uart, calibrated=False)
    assert task.ready is False
    assert task.process(None, 0) == S.STATUS_DONE
    assert task.last_info["reason"] == "uncalibrated"
    assert uart.dofs == [] and uart.turns == []


def test_transport_keeps_roll_level_then_releases_and_stops(monkeypatch):
    """进运输前回正 → 运输（保持 roll）→ 到点放球（横倾）→ 停动力。"""
    monkeypatch.setitem(S.comm.place, "transport_s", 1.5)
    uart = FakeUart(roll=12.0)                  # 进来就是歪的
    task = _task(monkeypatch, uart)
    phases, _ = _drive(task, uart)
    assert task.last_info["reason"] == "placed" and task.placed is True
    assert PH_PLACE_TRANSPORT in phases and PH_PLACE_RELEASE in phases and PH_PLACE_STOP in phases
    assert uart.turns[0] == (3, -12.0), "先把进来的横倾回正（相对角 = 目标 − 当前）：%s" % (uart.turns[:2],)
    assert (3, 30.0) in uart.turns, "放球要横倾 30°（同一物理方向，sign 未验证）"
    assert any(d[0] > 0 for d in uart.dofs), "运输必须真的前进"
    assert uart.dofs[-1] == (0.0, 0.0, 0.0, 0.0), "停动力 = 全 0"


def test_roll_drift_is_releveled_during_transport(monkeypatch):
    """运输中横倾漂了 ⇒ 重新回正（否则球会滚出来）。"""
    monkeypatch.setitem(S.comm.place, "transport_s", 10.0)
    monkeypatch.setitem(S.comm.place, "relevel_min_s", 0.0)
    uart = FakeUart(roll=0.0)
    task = _task(monkeypatch, uart)
    task.process(None, 0)
    task.process(None, 100)
    uart.telemetry.roll_deg = 25.0               # 风浪/推力把船压歪了
    for i in range(20):
        task.process(None, 200 + i * 100)
    assert any(t == (3, -25.0) for t in uart.turns), "漂 25° 就该回正 25°：%s" % (uart.turns,)


def test_arrived_api_triggers_release(monkeypatch):
    """「到点」的正式入口（TBD 判据由装配层/上位机给）：`arrived()` 立刻进放球。"""
    monkeypatch.setitem(S.comm.place, "transport_s", 999.0)
    uart = FakeUart(roll=0.0)
    task = _task(monkeypatch, uart)
    for i in range(10):
        task.process(None, i * 100)
    assert task.last_info["phase"] == PH_PLACE_TRANSPORT
    task.arrived()
    task.process(None, 2000)
    assert task.last_info["phase"] == PH_PLACE_RELEASE


def test_depth_floor_is_raised_and_restored(monkeypatch):
    """任务级限深下限只抬不降 + 结束撤回（全局 min_depth_m 一个字没动）。"""
    monkeypatch.setitem(S.comm.place, "transport_s", 0.5)
    uart = FakeUart(roll=0.0)
    task = _task(monkeypatch, uart)
    _drive(task, uart)
    assert 1.0 in uart.floor_calls
    assert uart.extra_min_depth_m == pytest.approx(0.0)
    assert float(S.comm.depth_guard.min_depth_m) == pytest.approx(0.55)


def test_missing_roll_telemetry_stops_instead_of_driving_tilted(monkeypatch):
    uart = FakeUart(roll=0.0, roll_ok=False)
    uart.telemetry.roll_deg = None
    task = _task(monkeypatch, uart)
    assert task.process(None, 0) == S.STATUS_DONE
    assert task.last_info["reason"] == "no_roll_telemetry"
    assert uart.dofs == [], "拿不到姿态就一步都不许走"


def test_estop_and_timeout_end_the_task(monkeypatch):
    uart = FakeUart()
    task = _task(monkeypatch, uart)
    uart.estop_active = True
    assert task.process(None, 0) == S.STATUS_DONE
    assert task.last_info["reason"] == "estop"

    monkeypatch.setitem(S.comm.place, "timeout_ms", 300)
    monkeypatch.setitem(S.comm.place, "transport_s", 999.0)
    uart2 = FakeUart()
    task2 = _task(monkeypatch, uart2)
    _drive(task2, uart2, frames=20)
    assert task2.last_info["reason"] == "timeout"
    assert uart2.extra_min_depth_m == pytest.approx(0.0)


def test_task_packages_do_not_import_each_other():
    """任务包之间**不得互相 import**（依赖规则）。

    2026-10-06：夹取与放置已合并进 `handling/`（同包内共享运动逻辑），
    原来的"place 不许 import grab"不再适用 —— 改成守"任务包之间不许互相 import"：
    `handling/` 只能用 `base`/`common` 与自身包，不许 import `gate`/`task` 等其它任务包。

    （用 AST 看**真正的 import 语句**，不拿子串猜 —— 文档串里出现包名不算违规。）
    """
    import ast
    import pathlib
    for f in sorted(pathlib.Path("handling").rglob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        mods = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module:
                mods.append(n.module)
        for m in mods:
            top = m.split(".")[0]
            assert top not in ("gate", "task", "grab", "place"), \
                "%s 不许 import 其它任务包：%s" % (f, m)
