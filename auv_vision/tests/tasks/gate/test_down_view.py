# -*- coding: utf-8 -*-
"""下视辅助过门：HSV 红色横向横杆检测。"""
import numpy as np
from gate.percept.down_view import detect_red_bar, red_mask


def _frame(red_band=False, w=640, h=360):
    f = np.zeros((h, w, 3), np.uint8)
    if red_band:
        bar_h = 20
        bar_w = int(w * 0.6)
        x0 = int((w - bar_w) / 2)
        f[h - bar_h:, x0:x0 + bar_w] = (20, 20, 220)   # 红(BGR)
    return f


def test_red_mask_finds_red_band():
    assert int(red_mask(_frame(True)).sum()) > 0
    assert int(red_mask(_frame(False)).sum()) == 0


def test_detect_red_bar():
    assert detect_red_bar(_frame(True)) is True
    assert detect_red_bar(_frame(False)) is False
    # 竖条/小块不是"横向横杆"
    f = np.zeros((360, 640, 3), np.uint8)
    f[100:300, 310:330] = (20, 20, 220)
    assert detect_red_bar(f) is False
