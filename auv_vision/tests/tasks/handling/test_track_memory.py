# -*- coding: utf-8 -*-
"""tests/tasks/handling/test_track_memory.py — **历史位置推演**（用户 2026-10-07 定）。

"同时对比前几帧之间位置，历史位置推演，防止球丢失找不到。"

两半都要钉住：
* 模块（`handling/percept/track_memory.py`）：匀速外推、样本不足退化、有效期到点失效、幅值时效折减；
* 接线：丢目标时按推演给**小幅**修正（`action="lost_predict"`）、**不推进相位判据**、
  过期回搜索且**先去上次看到的那一侧**；没轨迹时仍是原来的"保持不动"。
"""
from __future__ import annotations

import pytest

import base.cfg.settings as S
from handling.handling_task import PH_GRAB_CENTER, PH_GRAB_SEARCH
from common.motion.depth_hold import hold_cfg
from handling.percept.track_memory import TrackMemory, mem_cfg
from tests.tasks.handling.test_grab_flow import H, W, FakeUart, _red_frame, _task


# ---- 模块本身 --------------------------------------------------------------- #
def test_no_samples_no_prediction_and_side_hint_none():
    m = TrackMemory(log=lambda *a: None)
    assert m.predict(0) is None and m.side_hint() is None


def test_velocity_needs_enough_samples_then_predicts_forward():
    m = TrackMemory(log=lambda *a: None)
    m.add(0, -0.30, 0.0)
    m.add(50, -0.25, 0.0)
    assert m.velocity() == (0.0, 0.0), "样本不足只退化成最后位置（不发明速度）"
    m.add(100, -0.20, 0.0)
    vx, vy = m.velocity()
    assert vx > 0.0 and vy == 0.0
    dx, dy, sc = m.predict(150)
    assert dx > -0.20 and sc <= mem_cfg()["decay"], "外推要往前、且幅值折减"


def test_prediction_expires_and_scale_decays_with_age():
    m = TrackMemory(log=lambda *a: None)
    for t, dx in ((0, -0.1), (50, -0.1), (100, -0.1)):
        m.add(t, dx, 0.0)
    fresh = m.predict(120)[2]
    older = m.predict(400)[2]
    assert older < fresh, "越久没看到，置信折减越小"
    assert m.predict(100 + mem_cfg()["horizon_s"] * 1000.0 + 1) is None, "超有效期 ⇒ 交给搜索"


def test_history_is_capped_and_side_hint_tracks_the_last_seen_side():
    m = TrackMemory(log=lambda *a: None)
    for i in range(50):
        m.add(i * 10, -0.1, 0.0)
    assert m.seen == int(mem_cfg()["history"]), "轨迹长度有上限"
    m.add(1000, 0.3, 0.0)
    assert m.side_hint() == "right"
    m.clear()
    assert m.seen == 0 and m.side_hint() is None


# ---- 接线行为 --------------------------------------------------------------- #
def _feed_ball_moving_left(task, be, n=6, now=0):
    """让球在画面里从右往左飘几帧（同时给 task 帧）。"""
    out = now
    for i in range(n):
        be.cx = W * (0.75 - 0.06 * i)      # 一路往左：+0.5 → −0.1（跨过中线）
        be.cy = H / 2.0
        task.process(_red_frame(), out)
        out += 50
    return out


def test_lost_with_history_predicts_a_small_correction_without_faking_centering(monkeypatch):
    """★ 丢球后按轨迹往**球跑的方向**小幅修，而且**不许**误判成"已居中"（那会导致盲冲）。"""
    uart = FakeUart(depth=hold_cfg("grab")["target_m"])      # 区间内：别让定深插进来
    task, be = _task(monkeypatch, uart)
    task._phase = PH_GRAB_CENTER
    now = _feed_ball_moving_left(task, be)
    assert task._mem.seen >= 3, "看得见时要记轨迹"
    monkeypatch.setattr(be, "circles", lambda f: [])          # 球不见了
    predicted_dofs = []
    for _ in range(20):                                       # 越过 grace_frames=15
        task.process(_red_frame(), now)
        predicted_dofs.append(uart.dofs[-1])
        now += 50
    assert task.last_info["action"] == "lost_predict"
    surge, sway, heave, yaw = uart.dofs[-1]
    assert any(d[1] < 0.0 for d in predicted_dofs), "左右窗口必须向左修正"
    assert all(d[1] <= 0.0 and not (d[0] and d[1]) for d in predicted_dofs)
    assert abs(sway) <= 0.25 + 1e-9, "丢目标时只能小幅修（有硬上限）"
    assert abs(surge) <= 0.12 + 1e-9, "surge 上限更小：绝不盲冲"
    assert heave == 0.0 and yaw == 0.0
    assert task.last_info["phase"] in (PH_GRAB_CENTER, "APPROACH"), \
        "推演不许自己推进相位（误判已居中 ⇒ 盲冲）"


def test_lost_without_history_still_just_holds(monkeypatch):
    """没有轨迹 ⇒ 仍是原来的"保持不动"（安全默认不能被推演功能改掉）。"""
    uart = FakeUart(depth=hold_cfg("grab")["target_m"])
    task, be = _task(monkeypatch, uart)
    task._phase = PH_GRAB_CENTER
    monkeypatch.setattr(be, "circles", lambda f: [])
    now = 0
    for _ in range(20):
        task.process(_red_frame(), now)
        now += 50
    assert task.last_info["action"] == "hold"
    assert uart.dofs[-1] == (0.0, 0.0, 0.0, 0.0)


def test_when_the_window_expires_search_goes_to_the_side_the_ball_went(monkeypatch):
    """★ 保持窗口过期 ⇒ 回搜索，且**先扫上次看到球的那一侧**（而不是每次先右扫）。"""
    uart = FakeUart(depth=hold_cfg("grab")["target_m"])
    task, be = _task(monkeypatch, uart)
    task._phase = PH_GRAB_CENTER
    now = _feed_ball_moving_left(task, be)
    assert task._mem.side_hint() == "left"
    monkeypatch.setattr(be, "circles", lambda f: [])
    for _ in range(200):                                      # 越过 grace + hold_s 窗口
        task.process(_red_frame(), now)
        now += 50
        if task.last_info["phase"] == PH_GRAB_SEARCH:
            break
    assert task.last_info["phase"] == PH_GRAB_SEARCH, "过期必须回搜索（不能一直保持）"
    assert task._sweep.first == -1.0, "先去球最后出现的那一侧找"
