# -*- coding: utf-8 -*-
"""tests/_bite_probe.py — 临时文件：**"会咬"自检**（跑完即删，不属于交付套件）。

每条探针 = 「在内存里改一个 cfg 值 → 直接调用目标用例 → 它必须变红」。
若某条探针**意外通过**（= 目标用例没红），说明那个用例怎么改都不咬，是假测试。
跑法：python3 -m pytest tests/_bite_probe.py -q
"""
import pytest

import base.settings as S


def test_bite__loiter_timeout_is_used(monkeypatch):
    """改 comm.gate.loiter.timeout_ms → 门口兜底用例必须红。"""
    from tests.tasks.test_gate_flow import test_loiter_timeout_commits_through as target
    monkeypatch.setitem(S.comm.gate, "loiter",
                        S.Y(dict(S.comm.gate.loiter, timeout_ms=10 ** 9)))
    with pytest.raises(AssertionError):
        target()


def test_bite__search_sweep_value_is_pinned(monkeypatch):
    """改 comm.gate.search.sweep_s → 配置守卫用例必须红（波形用例只验形状，验不了数值）。"""
    from tests.tasks.test_gate_flow import test_gate_defaults_match_cfg as target
    monkeypatch.setitem(S.comm.gate, "search",
                        S.Y(dict(S.comm.gate.search, sweep_s=0.5)))
    with pytest.raises(AssertionError):
        target()


def test_bite__motion_shared_value_is_pinned(monkeypatch):
    """改 comm.motion.surge_fast → 配置守卫用例必须红（共用值不能被偷偷改）。"""
    from tests.tasks.test_gate_flow import test_gate_defaults_match_cfg as target
    monkeypatch.setitem(S.comm.motion, "surge_fast", 0.99)
    with pytest.raises(AssertionError):
        target()


def test_bite__depth_guard_threshold_is_pinned(monkeypatch):
    """改 comm.depth_guard.min_depth_m → **现场定死的 0.55** 不变量用例必须红。"""
    from tests.platform.test_base import test_settings_loads_real_yaml_values as target
    monkeypatch.setitem(S.comm.depth_guard, "min_depth_m", 0.99)
    with pytest.raises(AssertionError):
        target()


def test_no_bite__depth_guard_behaviour_is_relative_to_cfg(monkeypatch):
    """反向对照：限深**行为**用例按 cfg 相对判定，改阈值它仍应通过（这是对的，不是假红）。"""
    from tests.platform.test_base import test_depth_guard_blocks_surfacing_only as target
    monkeypatch.setitem(S.comm.depth_guard, "min_depth_m", 0.40)
    target(monkeypatch)


def test_bite__ball_dash_ratio_is_used(monkeypatch):
    """改 comm.ball.dash_ratio → 撞球命中用例必须红（冲刺条件真的被读）。"""
    from tests.tasks.test_ball import test_ball_growing_detection_hits as target
    from tests.conftest import FakeUart
    monkeypatch.setitem(S.comm.ball, "dash_ratio", 2.0)      # 不可能达到 → 永无 DASH
    with pytest.raises(AssertionError):
        target(FakeUart())


def test_bite__hdg_fresh_window_is_used(monkeypatch):
    """改 comm.gate.hdg.fresh_ms → 航向测量新鲜度用例必须红。"""
    from tests.tasks.test_motion import test_p3p_frames_do_not_feed_the_filter as target
    monkeypatch.setitem(S.comm.gate, "hdg",
                        S.Y(dict(S.comm.gate.hdg, fresh_ms=-1.0)))   # 任何测量都算过期
    with pytest.raises(AssertionError):
        target()


def test_bite__through_confirm_ms_is_used(monkeypatch):
    """改 comm.gate.through.confirm_ms → 配置守卫用例必须红。

    （`test_through_duration_is_time_based` 自己 monkeypatch 出 500ms 来验"按时间不按帧数"，
      所以它验的是语义而不是数值 —— 数值由配置守卫钉住。）
    """
    from tests.tasks.test_gate_flow import test_gate_defaults_match_cfg as target
    monkeypatch.setitem(S.comm.gate, "through",
                        S.Y(dict(S.comm.gate.through, confirm_ms=100000)))
    with pytest.raises(AssertionError):
        target()
