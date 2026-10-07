# -*- coding: utf-8 -*-
"""tests/tasks/handling/test_grab_duty.py — **平移脉冲闸门**（用户 2026-10-07：「速度还是太快了」）。

执行器死区 ~0.138 ⇒ 幅值压到那以下就推不动（实测 sweep 0.20 已很弱）⇒ "再慢"只能靠**占空比脉冲**：
幅值不动、平均速度降下来（同 sweep 那招「推一小段停一下」）。
作用域必须**只**是 CENTER/APPROACH/ALIGN 的 sway+surge：`heave` 归定深、`yaw` 恒 0、SEARCH 有自己的 duty。
"""
from __future__ import annotations

import pytest

import base.cfg.settings as S
from common.motion.search_sweep import duty_gate
from handling.handling_task import PH_GRAB_CENTER, PH_GRAB_SEARCH
from handling.motion.actions import _cvnode
from tests.tasks.handling.test_grab_flow import H, W, FakeUart, _red_frame, _task


def test_duty_gate_math():
    assert duty_gate(0.3, 0, 1.0, 250) == 0.3, "duty=1 ⇒ 原样通过（等于关掉）"
    assert duty_gate(0.3, 0, 0.0, 250) == 0.0, "duty=0 ⇒ 全关"
    on, off = 10, 130                                  # 250ms 窗口、0.5 ⇒ 前 125ms 推、后 125ms 停
    assert duty_gate(0.3, on, 0.5, 250) == 0.3
    assert duty_gate(0.3, off, 0.5, 250) == 0.0


def _center_frame(monkeypatch, now_ms, **duty):
    if duty:
        monkeypatch.setitem(S.comm.grab, "duty", S.Y(dict(duty)))
    uart = FakeUart(depth=float(_cvnode("center") and S.comm.grab.pid_sway and
                                 __import__("common.motion.depth_hold", fromlist=["x"])
                                 .hold_cfg("grab")["target_m"]))
    task, be = _task(monkeypatch, uart)
    task._phase = PH_GRAB_CENTER
    be.cx = W / 2.0 + 0.5 * W / 2.0
    be.cy = H / 2.0
    task.process(_red_frame(), now_ms)
    return uart.dofs[-1], task


def test_gate_pulses_the_translation_in_the_closed_loop_phases(monkeypatch):
    """★ `duty=0.5/250ms`：125ms 内推、后 125ms 停 —— 幅值**不变**（不掉死区）。"""
    on, _ = _center_frame(monkeypatch, 10, enable=True, duty=0.5, duty_window_ms=250)
    off, _ = _center_frame(monkeypatch, 130, enable=True, duty=0.5, duty_window_ms=250)
    assert abs(on[1]) > 0.0, "窗口前半段该推"
    assert off[1] == 0.0, "窗口后半段该停"
    assert abs(on[1]) == pytest.approx(float(S.comm.grab.pid_sway["out_max"])), \
        "脉冲期幅值就是 cfg 那个上限（没有被偷偷改小）"


def test_surge_can_pulse_more_sparsely_than_sway(monkeypatch):
    """★ 前进单独的占空比：`duty_surge=0.3` ⇒ 窗口 30% 后前进就停，而横向（0.5）还在推。"""
    monkeypatch.setitem(S.comm.grab, "duty",
                        S.Y(dict(enable=True, duty=0.5, duty_surge=0.3, duty_window_ms=250)))
    from common.motion.depth_hold import hold_cfg
    uart = FakeUart(depth=float(hold_cfg("grab")["target_m"]))
    task, be = _task(monkeypatch, uart)
    task._phase = PH_GRAB_CENTER
    task._ensure_pids()
    be.cx = W / 2.0 + 0.5 * W / 2.0                     # dx>0 ⇒ 有 sway
    be.cy = H / 2.0 + 0.4 * H / 2.0                     # dy>0 ⇒ 有 surge
    task.process(_red_frame(), 100)                     # 100/250 = 40%：过 30%、未过 50%
    surge, sway, _h, _y = uart.dofs[-1]
    assert surge == 0.0, "前进该停了（30% 之后）"
    assert abs(sway) > 0.0, "横向还在推（50% 之内）"


def test_surge_duty_is_per_phase_and_approach_is_not_extra_gated(monkeypatch):
    """★ 前进占空比**按相位**（居中 ≤ 对准 < 进近），且**时刻判据必须从 cfg 算**，不许写死 ms。

    "居中时前进要很小"只能靠**脉冲**实现：幅值贴死区反而可能是"猛"或"完全不走"（用户 2026-10-07）。
    """
    from common.motion.search_sweep import duty_gate
    d = S.comm.grab.duty
    assert float(d["duty_surge_center"]) <= float(d["duty_surge_align"]) \
        < float(d["duty_surge_approach"]), "居中该最稀、进近该最不稀"
    w_sw, w_su = float(d["duty_window_ms"]), float(d["duty_window_ms_surge"])
    t = None
    for cand in range(0, int(max(w_sw, w_su)) + 1, 5):
        if duty_gate(1.0, cand, float(d["duty"]), w_sw) and \
                not duty_gate(1.0, cand, float(d["duty_surge_center"]), w_su):
            t = cand
            break
    assert t is not None, "配置里找不到「横向在推、前进已停」的时刻 ⇒ 前进没被削"

    from common.motion.depth_hold import hold_cfg
    def _run(phase, now_ms):
        uart = FakeUart(depth=float(hold_cfg("grab")["target_m"]))
        task, be = _task(monkeypatch, uart)
        task._phase = phase
        task._ensure_pids()
        be.cx = W / 2.0 + 0.3 * W / 2.0        # dx>0 ⇒ 有 sway
        be.cy = H / 2.0 + 0.6 * H / 2.0        # dy 大 ⇒ 一定有 surge
        task.process(_red_frame(), now_ms)
        return uart.dofs[-1]

    c = _run(PH_GRAB_CENTER, t)
    assert c[0] == 0.0, "居中：这个时刻前进该停了"
    assert abs(c[1]) > 0.0, "横向此时还在推"
    a = _run("APPROACH", t)
    assert abs(a[0]) > 0.0, "进近的前进不该被额外削（否则永远接近不了）"


def test_surge_can_use_a_shorter_pulse_window(monkeypatch):
    """★ 前进可以用**更短的窗口** ⇒ 每一下只推一小步（该机前进推力大，慢要靠脉冲不是靠贴死区）。"""
    monkeypatch.setitem(S.comm.grab, "duty", S.Y(dict(
        enable=True, duty=0.5, duty_surge=0.5, duty_window_ms=400, duty_window_ms_surge=100)))
    uart = FakeUart(depth=float(__import__("common.motion.depth_hold", fromlist=["x"])
                                .hold_cfg("grab")["target_m"]))
    task, be = _task(monkeypatch, uart)
    task._phase = PH_GRAB_CENTER
    task._ensure_pids()
    be.cx = W / 2.0 + 0.3 * W / 2.0
    be.cy = H / 2.0 + 0.6 * H / 2.0
    task.process(_red_frame(), 60)          # 60ms：surge 窗口 100ms 的 60% ⇒ 已过 50% 该停
    surge, sway, _h, _y = uart.dofs[-1]
    assert surge == 0.0, "前进按自己的短窗口已经停了"
    assert abs(sway) > 0.0, "横向仍在 400ms 窗口的前半段 ⇒ 还在推"


def test_gate_can_be_turned_off_with_one_key(monkeypatch):
    """`enable: false` ⇒ 回到连续输出（一行退回旧行为）。"""
    a, _ = _center_frame(monkeypatch, 130, enable=False, duty=0.5, duty_window_ms=250)
    assert abs(a[1]) > 0.0


def test_gate_does_not_touch_heave_or_search(monkeypatch):
    """闸门**只**碰 sway/surge：定深照常出 heave；SEARCH 用自己的 duty（不受这个闸门影响）。"""
    monkeypatch.setitem(S.comm.grab, "duty", S.Y(dict(enable=True, duty=0.5, duty_window_ms=250)))
    from common.motion.depth_hold import hold_cfg
    band = hold_cfg("grab")
    uart = FakeUart(depth=float(band["band_hi_m"]) + 0.3)      # 明显太深 ⇒ 定深要上浮
    task, be = _task(monkeypatch, uart)
    task._phase = PH_GRAB_CENTER
    be.cx = W / 2.0 + 0.5 * W / 2.0
    be.cy = H / 2.0
    task.process(_red_frame(), 130)                            # 闸门"停"的半区
    surge, sway, heave, _yaw = uart.dofs[-1]
    assert sway == 0.0, "平移该被闸住"
    assert heave > 0.0, "heave 归定深，不许被闸门碰"

    # SEARCH（t=400ms：本闸门处于"停"半区，而 sweep 自己的窗口正处于"推"半区）
    u2 = FakeUart(depth=float(band["target_m"]))
    t2, be2 = _task(monkeypatch, u2)
    monkeypatch.setattr(be2, "circles", lambda f: [])
    t2._phase = PH_GRAB_SEARCH
    t2.process(_red_frame(), 400)
    assert u2.dofs[-1][1] != 0.0, "SEARCH 走 sweep 自己的 duty，不受这个闸门影响"
