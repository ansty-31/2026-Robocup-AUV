# -*- coding: utf-8 -*-
"""tests/tasks/grab/test_handling_modes.py — 「夹取 + 放置」总调度的**模式切换**
（`handling/handling_task.py`：`mode="grab" | "place" | "full"`）。

守三件事：
① `mode="grab"` 只跑夹取并正常结束（**绝不**自己切到放置）；
② `mode="full"` 夹取成功那一刻**交接**：`_stage` 变 `place`、`holds_ball` 为真、相位进入放置流程，
   且**不必重建实例**（限深与收尾仍由同一任务持有）；
③ `mode="place"` 从放置的 INIT 起步，不接受夹取相位。
"""
import pytest

import base.cfg.settings as S
from handling.handling_task import (PH_GRAB_DONE, PH_PLACE_DONE, PH_PLACE_INIT, PH_PLACE_RELEASE,
                                    PH_PLACE_STOP, PH_PLACE_TRANSPORT, GrabTask, HandlingTask,
                                    PlaceTask)

_PLACE_PHASES = (PH_PLACE_INIT, PH_PLACE_TRANSPORT, PH_PLACE_RELEASE, PH_PLACE_STOP, PH_PLACE_DONE)
from tests.tasks.grab.test_grab_flow import (FakeUart, W, H, _Backend, _Hub, _drive,
                                             _grow, _red_frame, _task)


class _Uart(FakeUart):
    """放置流程要 roll 遥测；FakeUart 已有 roll 字段，这里只补齐缺省。"""


def test_grab_mode_never_switches_to_place(monkeypatch):
    uart = _Uart()
    task, be = _task(monkeypatch, uart)
    assert task.mode == "grab"
    _drive(task, uart, be, radius=_grow, frame=_red_frame(), phases=[])
    assert task.last_info["reason"] == "grab_ok"
    assert task._stage == "grab", "grab 模式不许自己切到放置"
    assert task._phase == PH_GRAB_DONE
    assert task.holds_ball is True


def test_full_mode_hands_over_to_place(monkeypatch):
    """夹取成功 → 同一实例交接给放置（`_stage`/相位/持球标志都要对）。"""
    monkeypatch.setitem(S.comm.place, "calibrated", True)
    uart = _Uart()
    task, be = _task(monkeypatch, uart)
    task.mode = "full"                       # 同一实例切到 full（装配层就是建 full）
    task._stage = "grab"
    _drive(task, uart, be, frames=400, radius=_grow, frame=_red_frame(), phases=[])
    assert task._stage == "place", "夹取成功后应交接到放置流程"
    assert task._holds_ball is True, "交接时要把持球标志带过去"
    assert task._phase in _PLACE_PHASES, \
        "交接后相位必须落在放置流程里（实测 %s）" % task._phase
    assert task._finished is False, "交接后任务还没结束"


def test_place_mode_starts_from_place_init(monkeypatch):
    monkeypatch.setitem(S.comm.place, "calibrated", True)
    task = PlaceTask(_Uart(), None, W, H, log=lambda *a: None)
    assert task.mode == "place" and task._stage == "place"
    assert task._phase == PH_PLACE_INIT
    assert task.wants_down() is False, "只放置时不需要下视帧"
