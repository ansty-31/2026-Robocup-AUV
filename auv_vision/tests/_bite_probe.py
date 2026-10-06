# -*- coding: utf-8 -*-
"""tests/_bite_probe.py — 临时文件：**"会咬"自检**（跑完即删，不属于交付套件）。
（详细用法、判据与实测见 doc/注释历史.md）"""
import pytest

import base.cfg.settings as S

def test_bite__loiter_center_window_is_used(monkeypatch):
    """改 `comm.gate.loiter.dx_max`（门口"安全带"宽）→ 门口兜底用例必须红。"""
    from tests.tasks.gate.test_gate_flow import (
        test_loiter_timeout_commits_with_loosened_criteria as target)
    monkeypatch.setitem(S.comm.gate, "loiter",
                        S.Y(dict(S.comm.gate.loiter, dx_max=0.0)))   # 安全带收成 0 ⇒ 永远不算"在门口"
    with pytest.raises(AssertionError):
        target(monkeypatch)          # 新用例签名带 fixture

def test_bite__search_sweep_value_is_pinned(monkeypatch):
    """改 comm.gate.search.sweep_s → 配置守卫用例必须红（波形用例只验形状，验不了数值）。"""
    from tests.tasks.gate.test_gate_flow import test_gate_defaults_match_cfg as target
    monkeypatch.setitem(S.comm.gate, "search",
                        S.Y(dict(S.comm.gate.search, sweep_s=0.5)))
    with pytest.raises(AssertionError):
        target()

def test_bite__motion_shared_value_is_pinned(monkeypatch):
    """改 comm.motion.surge_fast → 配置守卫用例必须红（共用值不能被偷偷改）。"""
    from tests.tasks.gate.test_gate_flow import test_gate_defaults_match_cfg as target
    monkeypatch.setitem(S.comm.motion, "surge_fast", 0.99)
    with pytest.raises(AssertionError):
        target()

def test_bite__depth_guard_threshold_is_pinned(monkeypatch):
    """改 comm.depth_guard.min_depth_m → **现场定死值**（现 0.50）的不变量用例必须红。"""
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

def test_bite__hdg_measure_needs_trusted_full_frame(monkeypatch):
    """改 `vision.gate.keypoint.conf_thr` → "psi 只来自被采信的 full 帧"用例必须红。"""
    from tests.tasks.test_motion import test_p3p_frames_do_not_feed_the_filter as target
    monkeypatch.setitem(S.vision.gate["keypoint"], "conf_thr", 0.99)   # 高于用例里 full 帧的 0.95
    with pytest.raises(AssertionError):
        target()


def test_bite__turn_scale_shapes_the_issued_angle(monkeypatch):
    """改 `comm.gate.hdg.turn_scale` → 下发角成形用例必须红。"""
    from tests.tasks.gate.test_gate_flow import (
        test_turn_scale_and_max_step_shape_the_issued_angle as target)
    monkeypatch.setitem(S.comm.gate["hdg"], "turn_scale", 1.0)   # 不再缩放 ⇒ 下发角变了
    with pytest.raises(AssertionError):
        target()


def test_bite__post_sway_kpt_min_is_used(monkeypatch):
    """改 `comm.gate.hdg.post_sway_kpt_min` → postsway 角点数用例必须红。"""
    from tests.tasks.gate.test_gate_flow import (
        test_post_sway_exits_as_soon_as_the_configured_keypoints_appear as target)
    monkeypatch.setitem(S.comm.gate["hdg"], "post_sway_kpt_min", 99)   # 永远看不到这么多
    with pytest.raises(AssertionError):
        target()

def test_bite__through_confirm_ms_is_used(monkeypatch):
    """改 comm.gate.through.confirm_ms → 配置守卫用例必须红。"""
    from tests.tasks.gate.test_gate_flow import test_gate_defaults_match_cfg as target
    monkeypatch.setitem(S.comm.gate, "through",
                        S.Y(dict(S.comm.gate.through, confirm_ms=100000)))
    with pytest.raises(AssertionError):
        target()
