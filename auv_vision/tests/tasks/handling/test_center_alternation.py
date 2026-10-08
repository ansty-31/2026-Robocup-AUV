"""CENTER alternates timed, mutually exclusive translation windows."""
import pytest
import base.cfg.settings as S
from tests.tasks.handling.test_grab_flow import FakeUart, _task, _red_frame, W, H


def setup_center(monkeypatch, enabled=True):
    monkeypatch.setitem(S.comm.grab, "duty", S.Y(dict(
        enable=enabled, duty=0.35, duty_window_ms=250,
        duty_window_ms_surge=300, duty_surge_center=0.15)))
    uart = FakeUart(depth=1.3)
    task, be = _task(monkeypatch, uart)
    task._phase = "CENTER"
    be.cx, be.cy = W * 0.75, H * 0.75
    return task, be, uart


def test_center_alternates_windows_while_both_errors_remain_large(monkeypatch):
    task, be, uart = setup_center(monkeypatch)
    # Offset start deliberately: pulse timing must follow the action, not wall time.
    for elapsed, axis in [(0, 1), (50, 1), (100, None), (249, None),
                          (250, 0), (280, 0), (300, None), (549, None),
                          (550, 1), (800, 0)]:
        task.process(_red_frame(), 137 + elapsed)
        surge, sway, _, yaw = uart.dofs[-1]
        assert not (surge and sway)
        assert yaw == 0
        assert task._phase == "CENTER"
        if axis is None:
            assert (surge, sway) == (0, 0)
        else:
            assert uart.dofs[-1][axis] > 0
            assert uart.dofs[-1][1-axis] == 0


def test_center_uses_latest_frame_for_each_direction(monkeypatch):
    task, be, uart = setup_center(monkeypatch, enabled=False)
    task.process(_red_frame(), 0)
    be.cy = H * 0.25
    task.process(_red_frame(), 250)
    assert uart.dofs[-1][0] < 0 and uart.dofs[-1][1] == 0
    be.cx = W * 0.25
    task.process(_red_frame(), 550)
    assert uart.dofs[-1][1] < 0 and uart.dofs[-1][0] == 0


def test_center_lost_prediction_keeps_alternation_and_cannot_advance(monkeypatch):
    task, be, uart = setup_center(monkeypatch, enabled=False)
    task.process(_red_frame(), 0)
    monkeypatch.setattr(be, "circles", lambda frame: [])
    task.process(_red_frame(), 250)
    assert task.last_info["action"] == "lost_predict"
    assert uart.dofs[-1][0] > 0 and uart.dofs[-1][1] == 0
    assert task._phase == "CENTER"
    task.process(_red_frame(), 550)
    assert uart.dofs[-1][1] > 0 and uart.dofs[-1][0] == 0


def test_center_reentry_restarts_with_sway(monkeypatch):
    task, be, uart = setup_center(monkeypatch, enabled=False)
    task.process(_red_frame(), 0)
    task.process(_red_frame(), 250)
    task._phase = "SEARCH"
    task.process(_red_frame(), 300)
    task.process(_red_frame(), 350)
    assert uart.dofs[-1][1] > 0 and uart.dofs[-1][0] == 0


@pytest.mark.parametrize("stop", ["estop", "no_frame"])
def test_center_stop_interrupts_a_window(monkeypatch, stop):
    task, be, uart = setup_center(monkeypatch)
    task.process(_red_frame(), 0)
    if stop == "estop":
        uart.estop_active = True
        task.process(_red_frame(), 10)
        assert task.last_info["reason"] == "estop"
        assert uart.neutral_calls > 0
    else:
        task.process(None, 10)
        assert uart.dofs[-1] == (0, 0, 0, 0)


def test_center_confirmation_still_enters_approach(monkeypatch):
    task, be, uart = setup_center(monkeypatch)
    be.cx, be.cy = W / 2, H / 2
    for i in range(int(S.comm.grab.center.confirm_frames)):
        task.process(_red_frame(), i * 50)
    assert task._phase == "APPROACH"
    assert all(d == (0, 0, 0, 0) for d in uart.dofs)
