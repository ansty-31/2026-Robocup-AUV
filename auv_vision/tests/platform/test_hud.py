# -*- coding: utf-8 -*-
"""tests/platform/test_hud.py — HUD 的「偏转角」显示行（`main.psi_line`）。

为什么单独测：现场最常盯的就是这一行，而它有两个**会被误读**的坑 ——
  1. 还没测到 psi 时必须显示 `--` 而**不是 0**（0 会被当成"已经正了"）；
  2. `hdg_skip` 有值时（这一趟跳过了正航向）必须**变色并写明原因**，
     否则现场会以为"航向没问题"，而实际上是根本没测过。
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import psi_line                                            # noqa: E402

GREEN = (0, 220, 0)
YELLOW = (0, 190, 255)
GRAY = (165, 165, 165)
ORANGE = (0, 140, 255)


def test_no_measurement_shows_dashes_not_zero():
    txt, col = psi_line({"hdg": None}, 8.0)
    assert "--" in txt and "0.0" not in txt, "没测到必须显示 --，不能显示 0：%r" % txt
    assert col == GRAY
    assert "no-full-frame" in txt


def test_within_tolerance_is_green():
    txt, col = psi_line({"hdg": -3.4, "hdg_state": "done"}, 8.0)
    assert col == GREEN and "-3.4" in txt and "done" in txt


def test_outside_tolerance_is_yellow():
    txt, col = psi_line({"hdg": 12.3, "hdg_state": "turn"}, 8.0)
    assert col == YELLOW and "+12.3" in txt and "turn" in txt


def test_exactly_on_tolerance_counts_as_converged():
    assert psi_line({"hdg": 8.0}, 8.0)[1] == GREEN
    assert psi_line({"hdg": 8.01}, 8.0)[1] == YELLOW


def test_iteration_shown_only_when_nonzero():
    assert "it=" not in psi_line({"hdg": 1.0, "hdg_i": 0}, 8.0)[0]
    assert "it=2" in psi_line({"hdg": 1.0, "hdg_i": 2}, 8.0)[0]


def test_skip_reason_is_highlighted():
    txt, col = psi_line({"hdg": 1.0, "hdg_state": "measure",
                         "hdg_skip": "mode=p3p(只有 full 帧的 psi 可用)"}, 8.0)
    assert col == ORANGE, "跳过正航向必须变色（否则会被当成航向正常）"
    assert "SKIP" in txt and "p3p" in txt


def test_tolerance_is_printed():
    assert "(tol 8.0)" in psi_line({"hdg": 0.0}, 8.0)[0]
    assert "(tol 10.0)" in psi_line({"hdg": 0.0}, 10.0)[0]


def test_missing_keys_do_not_crash():
    for info in ({}, {"hdg_state": None}, {"hdg": 0, "hdg_i": None}):
        txt, col = psi_line(info, 8.0)
        assert isinstance(txt, str) and len(col) == 3
