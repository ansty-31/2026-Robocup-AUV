# -*- coding: utf-8 -*-
"""tests/tasks/handling/test_grab_targets.py — 目标色集合（`grab.targets`）与"每色一套阈值"的取用闸。

用户口径（2026-10-06）：现在用**红球**测逻辑与运动（红球外都毙）；比赛用**粉球/黄球** ⇒
先把阈值槽位准备好。**本套用例钉的是"未标定不许参与"这条闸**，不是某个颜色的数值。
"""
from __future__ import annotations

import pytest

import base.cfg.settings as S
from handling.percept.cv_ball import Params, active_targets


def test_red_is_the_only_active_target_for_now():
    """现在只有红球生效（红球 = 测试用目标；"红球外的可以都毙了"）。"""
    assert active_targets() == ["red"]
    assert Params.from_target("red").dom_min == pytest.approx(
        float(S.get("vision.grab.cv.dom_min", 30)))


def test_grab_cfg_is_really_wired_to_vision_yaml(monkeypatch):
    """★ 回归：`grab.*` 参数**真的**从 `cfg/vision.yaml` 读（漏 `vision.` 前缀会静默取默认值）。

    2026-10-06 之前 `_cfg` 传的是 `"grab.cv.dom_min"`，而 `S.get()` 只认 `vision.`/`comm.`
    开头 ⇒ 整个 `grab.*` 段从未生效（只因 yaml 值与代码默认值相同才没被发现）。
    """
    monkeypatch.setitem(S.vision.grab.cv, "dom_min", 77)
    monkeypatch.setitem(S.vision.grab, "kind", "probe_kind")
    from handling.percept.grab_detector import GrabBallDetector
    assert Params.from_cfg().dom_min == pytest.approx(77), "cfg 改了没生效 ⇒ 又断了"
    assert GrabBallDetector().__dict__["kind"] == "probe_kind"


def test_uncalibrated_colours_do_not_exist_yet(monkeypatch):
    """★ 粉/黄未标定 ⇒ `None`（不参与检测），**绝不回退到红球的默认值**。

    这是最要防的一坑：`Params.from_cfg()` 的 dataclass 默认值就是红的工作点，
    谁要是直接 `from_cfg("grab.targets.pink.")` 就等于拿红阈值去认粉球。
    """
    assert Params.from_target("pink") is None
    assert Params.from_target("yellow") is None
    assert Params.from_target("green") is None
    # 只开 enable、没标定 ⇒ 还是不许用
    monkeypatch.setitem(S.vision.grab.targets.pink, "enable", True)
    assert Params.from_target("pink") is None
    assert active_targets() == ["red"]


def test_calibrated_colour_is_picked_up_with_its_own_values(monkeypatch):
    """标定齐了才认，而且用的是**它自己的**值（不是红的）。"""
    node = S.vision.grab.targets.yellow
    monkeypatch.setitem(node, "enable", True)
    monkeypatch.setitem(node, "calibrated", True)
    monkeypatch.setitem(node, "dom_min", 11)
    monkeypatch.setitem(node, "s_min", 22)
    monkeypatch.setitem(node, "v_min", 33)
    p = Params.from_target("yellow")
    assert p is not None
    assert (p.dom_min, p.s_min, p.v_min) == (11, 22, 33)
    assert p.dom_min != Params.from_target("red").dom_min, "不许沿用红的阈值"
    monkeypatch.setitem(S.vision.grab.targets, "active", ["red", "yellow"])
    assert active_targets() == ["red", "yellow"]


def test_missing_section_falls_back_to_red_only(monkeypatch):
    """整个 `grab.targets` 段删掉也不崩（代码兜底 = 只认红）。"""
    monkeypatch.setattr(S.vision, "grab",
                        S.Y({k: v for k, v in S.vision.grab.items() if k != "targets"}))
    assert active_targets() == ["red"]


def test_yellow_can_not_reuse_the_red_criterion_shape():
    """把"黄球不能照抄红判据"写成可执行的说明：红判据 `R−max(G,B)` 对黄色恒 ≈0。"""
    import numpy as np
    yellow = np.zeros((1, 1, 3), dtype=np.uint8)
    yellow[0, 0] = (0, 240, 250)              # BGR：黄 = 高 R、高 G、低 B
    b, g, r = int(yellow[0, 0, 0]), int(yellow[0, 0, 1]), int(yellow[0, 0, 2])
    assert r - max(g, b) <= 10, "红占优判据对黄色本来就 ≈0 ⇒ 必须换判据形式（见 vision.yaml 注释）"
