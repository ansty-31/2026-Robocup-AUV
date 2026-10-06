# -*- coding: utf-8 -*-
"""夹取小球感知层用例（**纯软件，无硬件/无相机/无权重**）。
（详细用法、判据与实测见 doc/注释历史.md）"""
import numpy as np
import pytest

from handling.percept.ball_tracker import BallTracker
from handling.percept.cv_ball import Params, detect, red_mask, red_mask_fast, red_mask_ref

W, H = 640, 360


def synth(radius=40, center=(320, 180), bg=(90, 90, 90)):
    """合成图：灰底 + 一个纯红圆（BGR）。"""
    img = np.zeros((H, W, 3), np.uint8)
    img[:, :] = bg
    import cv2
    cv2.circle(img, center, radius, (20, 20, 230), -1)     # BGR 红
    return img


def test_fast_mask_matches_reference_on_synthetic():
    img = synth()
    p_fast = Params(fast_mask=True)
    p_ref = Params(fast_mask=False)
    m_fast, _ = red_mask_fast(img, p_fast)
    m_ref, _ = red_mask_ref(img, p_ref)
    diff = int((m_fast != m_ref).sum())
    assert diff == 0, "快掩膜与参考实现不一致：%d 个像素" % diff


def test_fast_mask_matches_reference_on_noise():
    """带噪点/渐变/其它颜色：等价性不能只在干净合成图上成立。"""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    for p in (Params(), Params(dom_min=60, rel_min=0.4, s_min=100, v_min=30),
              Params(use_rel=False), Params(use_hue=False, s_min=0, v_min=0)):
        m_fast, _ = red_mask_fast(img, Params(**{**p.__dict__, "fast_mask": True}))
        m_ref, _ = red_mask_ref(img, Params(**{**p.__dict__, "fast_mask": False}))
        assert int((m_fast != m_ref).sum()) == 0


@pytest.mark.parametrize("t", [0.20, 0.30, 0.33, 0.40, 0.50])
def test_rel_lut_is_exactly_the_rational_rule(t):
    """相对红占优的 LUT 必须与除法判据**逐格等价**（这是"快路径不掉精度"的根）。"""
    from handling.percept.cv_ball import _rel_lut
    lut = _rel_lut(t)
    mx = np.arange(256, dtype=np.int32)
    r = np.arange(256, dtype=np.int32)
    dom = np.maximum(0, r[:, None] - mx[None, :])          # 与实现一致：饱和减
    got = dom > lut[None, :]
    rel = (r[:, None] - mx[None, :]) / (r[:, None] + mx[None, :] + 1)
    assert int((got != (rel >= t)).sum()) == 0, "LUT 与除法判据不等价（t=%.2f）" % t


@pytest.mark.parametrize("radius", [30, 40, 80, 150])
def test_detects_red_disc(radius):
    img = synth(radius=radius)
    dets = detect(img, Params())
    assert len(dets) == 1
    d = dets[0]
    assert abs(d.cx - 320) <= 2 and abs(d.cy - 180) <= 2
    assert abs(d.r - radius) <= max(2.0, radius * 0.08)


def test_rejects_thin_red_strip():
    """细长红条（= 板端实测的"瓦缝红光"）必须被拦：面积够但圆度/张角不对。"""
    import cv2
    img = synth(radius=0)
    cv2.rectangle(img, (100, 170), (500, 178), (20, 20, 230), -1)   # 400x8 长条
    assert detect(img, Params()) == []


def test_rejects_too_small():
    """r=12 的小红点（池底图案）在 r_min=25 下必须被拦。"""
    img = synth(radius=12)
    assert detect(img, Params()) == []


def test_ignores_orange():
    """橙色导轨（H≈20–30）不能进掩膜 —— 色相门限的意义就在这。"""
    import cv2
    img = synth(radius=0)
    cv2.circle(img, (320, 180), 60, (0, 140, 255), -1)              # BGR 橙
    assert detect(img, Params()) == []


def test_fills_specular_highlight():
    """球心一块白斑（镜面高光）时，仍要给出**整个球**：圆半径不能只按红环算。"""
    import cv2
    img = synth(radius=60)
    cv2.circle(img, (320, 180), 18, (250, 250, 250), -1)            # 高光
    dets = detect(img, Params())
    assert len(dets) == 1
    assert abs(dets[0].r - 60) <= 6, "高光孔没被填：r=%.1f" % dets[0].r


def test_tracker_equals_full_frame_and_uses_roi():
    tr = BallTracker(Params(), margin=64, proc_side=0)     # proc_side=0：结果应与全图**完全一致**
    frames = [synth(radius=r, center=(320 + i * 20, 180)) for i, r in enumerate([40, 45, 50, 55, 60])]
    for i, f in enumerate(frames):
        d_track, info = tr.step(f)
        d_full = detect(f, Params())
        assert d_track is not None
        assert len(d_full) == 1
        assert abs(d_track.cx - d_full[0].cx) <= 0.5
        assert abs(d_track.r - d_full[0].r) <= 0.5
        assert info.mode == ("full" if i == 0 else "roi")
    assert tr.n_roi == 4 and tr.stats()["fallback_rate"] == 0.0


def test_tracker_proc_side_keeps_center_close():
    """尺度归一化（proc_side>0）允许有量化误差，但必须在亚像素~2px 量级。"""
    tr = BallTracker(Params(), margin=64, proc_side=256)
    for _ in range(4):
        d, _info = tr.step(synth(radius=150, center=(320, 180)))
    ref = detect(synth(radius=150, center=(320, 180)), Params())[0]
    assert abs(d.cx - ref.cx) <= 2 and abs(d.cy - ref.cy) <= 2
    assert abs(d.r / ref.r - 1.0) <= 0.03


def test_tracker_recovers_after_lost():
    """球消失（全灰帧）→ 连续丢帧后回到全图搜索，再出现时能重新捕获。"""
    tr = BallTracker(Params(), margin=64, max_lost=2)
    tr.step(synth(radius=50))
    blank = synth(radius=0)
    for _ in range(3):
        d, _ = tr.step(blank)
        assert d is None
    d, info = tr.step(synth(radius=50, center=(400, 200)))
    assert d is not None and info.mode == "full"      # 丢干净了 → 全图重捕获
