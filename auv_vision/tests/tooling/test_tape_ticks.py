# -*- coding: utf-8 -*-
"""tests/tooling/test_tape_ticks.py — 卷尺刻度周期测焦距（`tools/analyze/tape_ticks.py`）的校验。

为什么这样测：这条方法是**唯一不依赖点击**的标尺测量（几十个刻度一起估计周期，
精度 ~0.5%），所以核心用例是**合成往返**：按已知 `f` 画一条带厘米刻度的卷尺 →
工具测出的 `f` 必须对得上。另加"透视梯度"用例（靶面没正对时，光轴处的局部比例才对）。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.analyze import tape_ticks as TT                          # noqa: E402


# ------------------------------------------------------------------ 合成
def synth_image(path, f_px, z_m, y_c=300, thick=12, x0=200, x1=950,
                w=1280, h=720, tape=210, bg=30, grad=0.0):
    """画一条卷尺：亮带 + 每 `f/z*0.01` px 一条暗刻度（`grad` 可加线性透视梯度）。"""
    import cv2
    img = np.full((h, w, 3), bg, np.uint8)
    img[y_c:y_c + thick, x0:x1] = tape
    p0 = f_px * 0.01 / z_m
    x = x0
    k = 0
    while x < x1:
        # 2 px 宽 + 中间更暗：贴近真实（光学/JPEG 把刻度糊成平滑的暗带）
        for dc, val in ((-1, tape - 60), (0, bg + 5), (1, tape - 60)):
            c = int(round(x)) + dc
            if 0 <= c < w:
                img[y_c:y_c + thick, c] = val
        k += 1
        x = x0 + p0 * k * (1.0 + grad * (x - w / 2.0) / w)
    cv2.imwrite(str(path), img)
    return img


def test_find_ticks_subpixel():
    pr = np.full(200, 200.0)
    for x in (20.0, 40.5, 61.0, 81.5):
        pr[int(x)] = 150.0
        pr[int(x) + 1] = 150.0
    t = TT.find_ticks(pr, min_prom=5.0)
    assert len(t) == 4
    assert t[1] == pytest.approx(40.5, abs=0.6)


def test_fit_index_drops_outliers():
    x = np.arange(0, 200, 10.0)
    x = np.append(x, 95.0)                      # 误检（正好卡在两条之间）
    fit = TT.fit_index(np.sort(x))
    assert fit is not None
    assert fit["period_med"] == pytest.approx(10.0, abs=0.2)
    assert fit["rms"] < 0.6
    assert fit["n"] <= len(x) - 1, "离群点应被剔掉"


def test_period_at_handles_perspective_gradient():
    """间距随 x 线性变化（靶面没正对）时，`period_at` 要给出该点的**局部**比例。"""
    p0, slope = 10.0, 0.02
    ks = np.arange(0, 60)
    x = ks * p0 + slope * (ks ** 2) / 2.0       # dx/dk = p0 + slope*k
    fit = TT.fit_index(x)
    assert fit is not None
    for target_k in (10, 30, 50):
        target_x = p0 * target_k + slope * target_k ** 2 / 2.0
        p, _ = TT.period_at(fit, target_x)
        assert p == pytest.approx(p0 + slope * target_k, rel=0.02)


def test_f_from_period():
    assert TT.f_from_period(10.78, 1.0) == pytest.approx(1078.0, rel=1e-6)
    assert TT.f_from_period(10.78, 1.0, c_m=-0.08) == pytest.approx(10.78 * 0.92 / 0.01)
    assert TT.f_from_period(5.39, 2.0, grad_m=0.005) == pytest.approx(2156.0, rel=1e-6)


def test_detect_band_finds_tape():
    import cv2
    img = np.full((400, 800, 3), 30, np.uint8)
    img[150:165, 100:700] = 210
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    band = TT.detect_band(g, 120, 680)
    assert band is not None
    y0, y1 = band
    assert y0 <= 150 and y1 >= 164


# ------------------------------------------------------------------ 端到端
@pytest.mark.parametrize("f_px,z", [(1100.0, 1.0), (1078.0, 1.0), (1100.0, 2.0)])
def test_measure_one_recovers_focal(tmp_path, f_px, z):
    p = tmp_path / "cap_001.jpg"
    synth_image(p, f_px, z)
    r = TT.measure_one(str(p), z, xr=(200, 950), band=(298, 314))
    assert r is not None
    assert r["period_px"] == pytest.approx(f_px * 0.01 / z, rel=0.02)
    assert r["f_px"] == pytest.approx(f_px, rel=0.025)
    assert r["rms_px"] < 1.0


def test_measure_one_with_perspective_still_close(tmp_path):
    """有透视梯度时，取光轴处的局部比例 ⇒ 仍应接近真值（整条平均则会偏）。"""
    p = tmp_path / "cap_001.jpg"
    synth_image(p, 1100.0, 1.0, grad=0.15, x0=250, x1=1050)
    r = TT.measure_one(str(p), 1.0, xr=(250, 1050), band=(298, 314))
    assert r is not None
    assert r["f_px"] == pytest.approx(1100.0, rel=0.05)


def test_measure_one_no_tape_returns_none(tmp_path):
    import cv2
    p = tmp_path / "blank.jpg"
    cv2.imwrite(str(p), np.full((200, 300, 3), 25, np.uint8))
    assert TT.measure_one(str(p), 1.0) is None      # 没卷尺 ⇒ 行带都找不到


def test_main_writes_records_and_reports(tmp_path, capsys):
    for i in (1, 2):
        synth_image(tmp_path / ("cap_%03d.jpg" % i), 1080.0, 1.0)
    out = tmp_path / "ruler_ticks.jsonl"
    assert TT.main(["--z", "1.0", "--x", "200,950", "--band", "298,314",
                    "--out", str(out), str(tmp_path / "*.jpg")]) == 0
    txt = capsys.readouterr().out
    assert "f = 1080" in txt or "f = 1079" in txt or "f = 1081" in txt
    lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2
    import json
    r = json.loads(lines[0])
    assert r["kind"] == "ruler" and r["method"] == "tick_period"
    assert r["span_m"] == pytest.approx(0.01)


def test_ticks_output_feeds_ruler_calib(tmp_path):
    """刻度法产出的记录必须能被 `ruler_calib` 直接吃（端到端闭环）。"""
    import json
    from tools.analyze import ruler_calib as RC
    for z in (1.0, 3.0):
        d = tmp_path / ("z%d" % int(z * 100))
        d.mkdir()
        synth_image(d / "cap_001.jpg", 1080.0, z)
    out = str(tmp_path / "ticks.jsonl")
    assert TT.main(["--z", "1.0", "--x", "200,950", "--band", "298,314",
                    "--out", out, str(tmp_path / "z100" / "*.jpg")]) == 0
    recs, _ = RC.load_records([out])
    assert len(recs) == 1
    res = RC.solve(recs + [{"src": "x", "z": 3.0, "du": 1080.0 * 0.5 / 3.0, "span": 0.5}])
    assert res["focal"]["x"] == pytest.approx(1080.0, rel=0.03)
