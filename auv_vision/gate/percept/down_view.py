# -*- coding: utf-8 -*-
"""gate/percept/down_view.py — 下视辅助过门：HSV 识别门框底部的红色横向横杆
（详细用法、判据与实测见 doc/注释历史.md）"""
from __future__ import annotations

import numpy as np


def _cfg():
    try:
        import base.cfg.settings as S
        return S.get("vision.color.line_red", None)
    except Exception:
        return None


def red_mask(frame, cfg=None):
    """HSV 双区间红掩码（uint8，0/255）。"""
    import cv2
    c = cfg or _cfg() or {}
    hsv = c.get("hsv", {}) or {}
    h1 = hsv.get("h1", [0, 12])
    h2 = hsv.get("h2", [168, 180])
    s_min = int(hsv.get("s_min", 90))
    v_min = int(hsv.get("v_min", 90))
    f = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(f, (int(h1[0]), s_min, v_min), (int(h1[1]), 255, 255))
    m = cv2.bitwise_or(m, cv2.inRange(f, (int(h2[0]), s_min, v_min),
                                      (int(h2[1]), 255, 255)))
    return m


def detect_red_bar(frame, cfg=None):
    """画面里有没有红色**横向**横杆（门框底部）。返回 bool。"""
    import cv2
    if frame is None:
        return False
    c = cfg or _cfg() or {}
    h, w = frame.shape[:2]
    min_area = float(c.get("min_area_ratio", 0.002)) * float(h * w)
    m = red_mask(frame, c)
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if cv2.contourArea(cnt) < min_area:
            continue
        _x, _y, bw, bh = cv2.boundingRect(cnt)
        if bw < 0.4 * float(w):
            continue
        if bh <= 0 or (bw / float(bh)) < 3.0:
            continue
        return True
    return False
