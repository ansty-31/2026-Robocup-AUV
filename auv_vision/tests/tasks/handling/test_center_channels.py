# -*- coding: utf-8 -*-
"""tests/tasks/handling/test_center_channels.py — 小球**居中**两个通道（用户 2026-10-07 定）。

口径：
1. 居中 = **dx → sway + dy → surge**（不用 yaw：抬头时转 yaw 会让目标在画面里绕圈）；
2. **sway 与 gate/ball 同一套** —— 直接取共用 `comm.motion.pid_sway`（gate 用
   `comm.gate.pid_sway | comm.motion.pid_sway` 的同一条路子）；
3. **surge 的 PID 小一点**（默认 = 共用 sway × 0.5；**死区不缩**，缩了会推不动执行器）。
"""
from __future__ import annotations

import pytest

import base.cfg.settings as S
from handling.handling_task import PH_GRAB_CENTER
#: 定深区间内的一个深度 —— **从 cfg 取**（板端把区间改成别的值也不该让用例假红）
from common.motion.depth_hold import hold_cfg as _hold_cfg
_IN_BAND = float(_hold_cfg("grab")["target_m"])
from common.motion.axis import grab_node
from common.motion.depth_hold import hold_cfg
from handling.motion.actions import _cvnode
from tests.tasks.handling.test_grab_flow import H, W, FakeUart, _red_frame, _task


def _center_dofs(monkeypatch, dx, dy, depth=None):
    uart = FakeUart(depth=depth)                 # 区间内 ⇒ 定深不动手，heave 应为 0
    task, be = _task(monkeypatch, uart)
    task._phase = PH_GRAB_CENTER
    task._ensure_pids()
    be.cx = W / 2.0 + dx * W / 2.0
    be.cy = H / 2.0 + dy * H / 2.0
    task.process(_red_frame(), 0)
    return uart.dofs[-1], task


def test_center_maps_dx_to_sway_and_dy_to_surge(monkeypatch):
    """★ dx→sway、dy→surge，**不发 yaw**；heave 归定深（区间内 = 0）。"""
    (surge, sway, heave, yaw), _ = _center_dofs(monkeypatch, dx=0.5, dy=0.4)
    assert sway > 0.0, "dx>0 应该有横向输出"
    assert surge > 0.0, "dy>0 应该有前后输出（下视相机里画面上方 = 前方）"
    assert yaw == 0.0, "夹取任务不需要旋转"
    assert heave == 0.0, "heave 归定深，居中不许碰"

    (surge2, sway2, _h, _y), _ = _center_dofs(monkeypatch, dx=-0.5, dy=-0.4)
    assert sway2 < 0.0 and surge2 < 0.0, "反向偏差应该反向输出"


def test_center_sway_is_the_grab_tasks_own_smaller_set(monkeypatch):
    """★ 夹取 sway = **本任务自己那套** `comm.grab.pid_sway`（用户 2026-10-07：
    「夹取的 sway 单独设置一个，小一点的」），三个平移相位共用；**缺键才退回共用** 那份。

    用**结构性**判据（不是"输出比例"）：PID 输出会被 out_max 饱和，比例测法在饱和区不成立。
    """
    from common.cfg.cfgnode import motion_pid, pid_kw
    from common.motion.axis import grab_node
    uart = FakeUart(depth=_IN_BAND)
    task, _be = _task(monkeypatch, uart)
    task._ensure_pids()
    own = pid_kw(grab_node()["pid_sway"], "comm.grab.pid_sway")
    shared = motion_pid("pid_sway")
    for k in ("kp", "ki", "kd", "out_max", "deadzone"):
        if own.get(k) is not None:
            assert getattr(task._pid_yaw, k) == pytest.approx(own[k]), \
                "居中 sway 的 %s 没取本任务那一套" % k
    assert float(own["out_max"]) < float(shared["out_max"]), \
        "夹取那套必须比共用小（否则没必要单独设一个是吧）"
    # 三个平移相位同一套
    assert task._pid_sway.kp == pytest.approx(own["kp"])
    assert task._pid_asw.kp == pytest.approx(own["kp"])
    # 缺键 ⇒ 退回共用（gate/ball 那套）
    monkeypatch.delitem(S.comm.grab, "pid_sway", raising=False)
    u2 = FakeUart(depth=_IN_BAND)
    t2, _b2 = _task(monkeypatch, u2)
    t2._ensure_pids()
    assert t2._pid_yaw.kp == pytest.approx(shared["kp"]), "缺任务覆盖时应退回共用那份"


def test_center_surge_pid_is_smaller_than_sway_but_keeps_the_deadzone(monkeypatch):
    """★ surge 比 sway 小一点（增益/幅值减半），但**死区不变**（死区是执行器物理量）。"""
    uart = FakeUart(depth=_IN_BAND)
    task, _be = _task(monkeypatch, uart)
    task._ensure_pids()
    sw, su = task._pid_yaw, task._pid_csu
    ratio = float(_cvnode("center").get("surge_ratio", 0.5))   # cfg 派生：别写死 0.5
    assert su.kp == pytest.approx(sw.kp * ratio)
    assert su.out_max < sw.out_max
    assert su.deadzone == pytest.approx(sw.deadzone), "死区别跟着缩，否则掉进死区推不动"
    # 注：**不要**拿"执行器死区 0.138"去给前进定下限。0.138 是 gate 那条线量出来的死区
    # （见 common/motion/PID.py / search_scan.py / doc/记录/2026-10-07-gate-阈值与运动重设.md），
    # 但"刚出死区的幅值"对**前进**已经太猛（该机前进推力大，用户 2026-10-07 明确纠正过）
    # ⇒ 前进靠 `duty_surge_*` + `duty_window_ms_surge` 的**短脉冲**做慢，而不是靠贴死区的幅值。


def test_center_gains_are_overridable_from_cfg_without_touching_shared(monkeypatch):
    """逃生口：`comm.grab.center.sway/surge` 可整块覆盖（含增益）；缺键则走"共用增益 + 系数缩放"。）

    注：`surge` 的**基准永远是共用 `pid_sway`**（× `surge_ratio`），不跟着 sway 的覆盖走 ——
    否则"sway 和 gate/ball 一致"这条语义会被一个局部覆盖悄悄改掉。
    """
    monkeypatch.setitem(S.comm.grab["center"], "sway",
                        dict(kp=3.0, ki=0.0, kd=0.0, out_max=0.2, deadzone=0.02))
    (surge, sway, _h, _y), task = _center_dofs(monkeypatch, dx=0.5, dy=0.4)
    assert task._pid_yaw.kp == pytest.approx(3.0), "整块覆盖生效（含增益，逃生口）"
    # surge 的基准**永远**是本任务那套（× surge_ratio），不跟着 sway 的整块覆盖走。
    own = grab_node()["pid_sway"]
    ratio = float(_cvnode("center").get("surge_ratio", 0.5))
    assert task._pid_csu.kp == pytest.approx(float(own["kp"]) * ratio, rel=0.01)
    assert sway != 0.0 and surge != 0.0
    assert _cvnode("center")["eps"] > 0.0
