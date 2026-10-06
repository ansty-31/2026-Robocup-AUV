# -*- coding: utf-8 -*-
"""tests/tasks/handling/test_search_sweep.py — **纯左右平移扫视**（`common/motion/search_sweep.py`）。

用户 2026-10-06 定：任务三/四的 search 用平移扫视，**不用** `search_scan` 的 yaw 慢扫。
形状照 gate 那套（`gate/motion/channels.py::_search_sweep`：右→停→左→停，每轮时长翻倍）。
"""
from __future__ import annotations

import pytest

import base.cfg.settings as S
from common.motion.search_sweep import Sweep, sweep_cfg
from tests.tasks.handling.test_grab_flow import FakeUart, _Backend, _Hub, _task
from handling.motion.params import PH_GRAB_SEARCH


def _seq(sweep, t_end_ms, dt=250):
    return [(float(t), sweep.step(t)) for t in range(0, int(t_end_ms), dt)]


def test_segments_are_right_pause_left_pause():
    """段序严格是 右 → 停 → 左 → 停，每段时长由 cfg 决定。"""
    s = Sweep(prefix="grab", log=lambda *a: None)
    s.duty = 1.0                      # 这一条只验段序；占空比单独有用例
    c = sweep_cfg("grab")
    leg = float(c["sweep_s"])
    pause = float(c["pause_s"])
    v = float(c["sway"])
    # 段中点取样（避开边界）
    assert s.step(0) == pytest.approx(v)                       # 右
    assert s.step(int((leg + pause / 2) * 1000)) == 0.0        # 停
    # 段布局：右 [0, leg) → 停 [leg, leg+pause) → 左 [leg+pause, 2leg+pause) → 停 [2leg+pause, 2leg+2pause)
    assert s.step(int((1.5 * leg + pause) * 1000)) == pytest.approx(-v)           # 左（取段中点）
    assert s.step(int((2 * leg + pause + pause / 2) * 1000)) == 0.0               # 停


def test_duty_makes_each_leg_pulse_instead_of_one_shot():
    """★ 用户 2026-10-06：搜索**速度慢一点、节奏缓一点，同一方向不一次到位**。

    做法是**占空比**（沿用工程里 `search_scan.duty` 的同一惯用法）：窗口内前 `duty` 比例推、
    其余发 0 —— **幅值不动** ⇒ 平均速度降下来但**不掉执行器死区**（死区 0.138）。
    """
    s = Sweep(prefix="grab", log=lambda *a: None)
    c = sweep_cfg("grab")
    v = float(c["sway"])
    win = float(c["duty_window_ms"])
    vals = [s.step(t) for t in range(0, int(4 * win), 50)]
    on = [x for x in vals if x != 0.0]
    off = [x for x in vals if x == 0.0]
    assert on and off, "同一方向必须是脉冲（推一段 + 停一下），不能整段恒推"
    assert all(abs(x) == pytest.approx(v) for x in on), "幅值必须恒定（否则会掉进死区）"
    assert 0.25 <= len(on) / float(len(vals)) <= 0.75, "占空比大致就是 cfg 的 duty"
    assert v >= 0.15, "幅值别低于死区附近的 ~0.15（cfg 里写清了这条下限）"


def test_leg_length_doubles_every_round_and_is_capped():
    """单程时长每轮翻倍（门的形状），到 `sweep_max_s` 封顶。"""
    s = Sweep(prefix="grab", log=lambda *a: None)
    c = sweep_cfg("grab")
    leg, pause = float(c["sweep_s"]), float(c["pause_s"])
    # 第 1 轮：右段长 leg；第 2 轮：右段长 2×leg（用它中点是否仍在右向来判）
    t_round2_right_mid = (2 * (leg + pause) + leg + 0.25 * leg) * 1000.0
    assert s.step(int(t_round2_right_mid)) == pytest.approx(float(c["sway"])), \
        "第 2 轮右段应该更长（翻倍）"
    # 封顶：把上限压到很小，单程不会超过它
    s.duty = 1.0
    monkey_sweep = Sweep(prefix="nonexistent", log=lambda *a: None)
    monkey_sweep.sweep_max_s = monkey_sweep.sweep_s
    assert monkey_sweep._sweep_len(5) == pytest.approx(monkey_sweep.sweep_s * 1000.0)


def test_max_ms_stops_sweeping():
    """`max_ms` 到点 ⇒ 不再扫（保持静止，等目标自己出现）。"""
    s = Sweep(prefix="grab", log=lambda *a: None)
    s.max_ms = 1000.0
    assert s.step(0) != 0.0
    assert s.step(500) != 0.0
    assert s.step(1500) == 0.0 and s.done is True


def test_zero_sweep_disables_it():
    s = Sweep(prefix="grab", log=lambda *a: None)
    s.sweep_s = 0.0
    assert all(v == 0.0 for _t, v in _seq(s, 3000))


def test_reset_restarts_from_the_first_leg():
    s = Sweep(prefix="grab", log=lambda *a: None)
    s.step(0)
    s.step(10_000)
    assert s.rounds > 0
    s.reset(20_000)
    assert s.rounds == 0 and s.step(20_000) == pytest.approx(float(sweep_cfg("grab")["sway"]))


def test_cfg_missing_section_raises(monkeypatch):
    """★ 2026-10-07 用户定：**删掉兜底** —— 删掉 `comm.motion.search_sweep` ⇒ 报名字。"""
    import pytest
    from common.cfg.cfgnode import MissingCfg
    monkeypatch.setattr(S, "comm", S.Y({k: v for k, v in S.comm.items() if k != "motion"}))
    with pytest.raises(MissingCfg):
        sweep_cfg("grab")
    with pytest.raises(MissingCfg):
        Sweep(prefix="grab")


def test_task_section_can_override():
    s = Sweep(prefix="grab", log=lambda *a: None)
    assert s.sway == pytest.approx(float(sweep_cfg("grab")["sway"]))


def test_handling_search_is_translation_only_never_yaw(monkeypatch):
    """★ 用户口径的直接验收：夹取的 SEARCH **只出 sway，yaw 恒 0**（不再用 search_scan）。"""
    uart = FakeUart()
    task, be = _task(monkeypatch, uart)
    task._phase = PH_GRAB_SEARCH
    surge, sway, heave, yaw = task._step_search(0, None)
    assert yaw == 0.0, "SEARCH 不许转 yaw（抬头时转 yaw 会让目标绕圈跑）"
    assert surge == 0.0 and heave == 0.0
    assert sway != 0.0, "SEARCH 必须真的在左右平移"
    # 而且源码里不该再有 search_scan 的影子
    import pathlib
    for f in ("handling/motion/phases.py", "handling/handling_task.py"):
        src = pathlib.Path(f).read_text(encoding="utf-8")
        assert "search_scan" not in src or "#" in src.split("search_scan")[0].split("\n")[-1], \
            "%s 还在用 search_scan" % f
        assert "telemetry_yaw" not in src, "%s 还在依赖 yaw 遥测" % f
