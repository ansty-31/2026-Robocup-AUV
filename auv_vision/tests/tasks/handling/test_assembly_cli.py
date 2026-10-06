# -*- coding: utf-8 -*-
"""tests/tasks/handling/test_assembly_cli.py — **装配面与命令行**（`main.py` 的 grab/place/handling）。

`python3 main.py --task grab` 是任务三下水用的那条命令。它曾经**跑不起来**（三重缺口）：
1. `main()` 的校验只看 `TASK_CLASS` ⇒ `--task grab` 直接报"未知任务"；
2. `TASK_CAM["grab"] = "down"`，但 `self.cams` 里从来没有 `down` ⇒ 即使过了校验也会 KeyError；
3. 夹取后端从没注册进 hub ⇒ `hub.extra()` 拿到 None ⇒ 感知**静默退回 legacy 模型**
   （= 用撞球那版 YOLO，正是本项目明确否掉的路）。
本套用例把这三条钉住（用假相机/假串口，不碰硬件）。
"""
from __future__ import annotations

import sys

import pytest

import base.cfg.settings as S
import main as M


class _Cam(object):
    width, height = 1280, 720


class _Uart(object):
    sim = True
    estop_active = False

    def close(self):
        pass

    def neutral(self):
        pass


class _Feeder(object):
    """假 DownCamFeeder：只记录有没有被 start（真实现会开一路下视相机）。"""

    made = []

    def __init__(self):
        _Feeder.made.append(self)
        self.started = False
        self._want = False

    def start(self):
        self.started = True

    def set_want(self, want):
        self._want = bool(want)

    def latest(self):
        return None

    def stop(self):
        pass


@pytest.fixture
def fake_hw(monkeypatch):
    _Feeder.made = []
    monkeypatch.setattr(M, "create_camera", lambda which, **kw: _Cam())
    monkeypatch.setattr(M, "UartController", lambda *a, **kw: _Uart())
    monkeypatch.setattr(M, "DownCamFeeder", _Feeder)
    monkeypatch.setattr(M.AppController, "_init_video", lambda self: False)
    return monkeypatch


def test_cli_accepts_the_grab_command():
    """`--task` 的合法名包含 grab/place/handling（校验闸别再把它们当"未知任务"）。"""
    assert set(M.TASK_NAMES) == {"ball", "gate", "grab", "place", "handling"}
    assert "grab" in M.TASK_MODE and "place" in M.TASK_MODE and "handling" in M.TASK_MODE


def test_grab_assembles_down_camera_and_cv_backend(fake_hw):
    """★ `--task grab` 的装配面：**只开下视** + CV 后端 + 正确的状态/队列。

    为什么强调"只开下视"（2026-10-06 板端实测）：USB 相机独占且吃带宽，同时开前视+下视两路
    720p 时**后开那路的 `read()` 返回 None** ⇒ `task.process()` 一帧都不被调用 ⇒ 状态机卡在
    GRAB 空转、逐帧日志一行不写、任务自己的超时也走不到。所以"不用的相机一个都不开"是**安全属性**。
    """
    ctrl = M.AppController(["grab"])
    assert set(ctrl.cams) == {"down"}, \
        "夹取只用下视 ⇒ **不许**顺带开前视（两路 USB 会互抢带宽，下视读不到帧）"
    assert list(ctrl.tasks) == ["grab"]
    task = ctrl.tasks["grab"]
    assert task.mode == "grab"
    assert task.name == M.HANDLING_NAME == "handling", \
        "hub 的键是任务实例的 name（handling），**不是 CLI 名**"
    assert ctrl.hub.extra(M.HANDLING_NAME) is not None, \
        "CV 后端必须注册在 hub 的 handling 键上，否则感知会静默退回 legacy 模型"
    assert ctrl.queue == [S.STATE_GRAB]
    assert ctrl._down_feeder is None, "已经直接开了 down ⇒ 不能再起 feeder（下视是独占设备）"
    ctrl.uart.neutral()


def test_registering_under_the_cli_name_would_be_invisible(fake_hw):
    """反面证据：注册成 `"grab"` 的话，任务查 `"handling"` **永远拿不到**（README 里那个坑）。"""
    ctrl = M.AppController(["grab"])
    ctrl.hub.register("grab", object())                 # 故意注册错名字
    assert ctrl.hub.extra("grab") is not None
    assert ctrl.hub.extra(ctrl.tasks["grab"].name) is not None  # 真名是 handling（已被正确注册）


def test_ball_and_gate_keep_their_old_assembly(fake_hw):
    """老任务装配不变：ball/gate 只开前视（不开下视）；gate 仍走 feeder。"""
    ctrl = M.AppController(["ball"])
    assert set(ctrl.cams) == {"front"} and ctrl._down_feeder is None
    # gate 后端在 PC 上没有 hbm_runtime ⇒ 换掉构造入口（这里只验相机/feeder 装配）
    fake_hw.setattr(M, "build_gate_backend", lambda: object())
    ctrl2 = M.AppController(["gate"])
    assert set(ctrl2.cams) == {"front"}
    assert ctrl2._down_feeder is not None and ctrl2._down_feeder.started is True


def test_unknown_task_is_rejected_with_the_real_choices():
    argv = sys.argv
    try:
        sys.argv = ["main.py", "--task", "bogus"]
        with pytest.raises(SystemExit) as e:
            M.main()
        assert e.value.code == 2
    finally:
        sys.argv = argv


def test_cli_refuses_to_start_when_uncalibrated(monkeypatch):
    """★ 机械闸：`calibrated=false` ⇒ `main.py` **拒绝启动**（退出码 3），而不是下水空跑。"""
    monkeypatch.setitem(S.comm.grab, "calibrated", False)

    class _Task(object):
        mode = "grab"
        calibrated = False

    class _StubCtl(object):
        tasks = {"grab": _Task()}

        def __init__(self, tasks=None):
            pass

        def check_ready(self, tasks):
            return ["grab"]

        def _model_desc(self):
            return "-"

        def close(self):
            pass

    monkeypatch.setattr(M, "AppController", _StubCtl)
    argv = sys.argv
    try:
        sys.argv = ["main.py", "--task", "grab"]
        with pytest.raises(SystemExit) as e:
            M.main()
        assert e.value.code == 3
    finally:
        sys.argv = argv


def test_run_grab_script_exists_and_points_at_the_right_command():
    """现场脚本：`bash task/run_grab.sh` = 单任务夹取（对标 run_gate.sh），且检查项写清。"""
    import pathlib
    sh = pathlib.Path("task/run_grab.sh")
    assert sh.is_file(), "任务三需要与 run_gate.sh 对齐的单任务下水脚本"
    text = sh.read_text(encoding="utf-8")
    assert "main.py --task grab" in text
    assert "calibrated" in text and "从视" not in text
