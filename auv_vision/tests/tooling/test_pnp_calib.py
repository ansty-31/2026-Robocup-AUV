# -*- coding: utf-8 -*-
"""tests/tooling/test_pnp_calib.py — tools/analyze/calib/pnp_calib.py 的**合成往返**校验。
（详细用法、判据与实测见 doc/注释历史.md）"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

# 工程根 = 向上第一个含 `cfg/` 的目录（**别写死层级**：脚本搬过位置，写死会静默指错）
ROOT = os.path.dirname(os.path.abspath(__file__))
while ROOT != os.path.dirname(ROOT) and not os.path.isdir(os.path.join(ROOT, "cfg")):
    ROOT = os.path.dirname(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gate.percept.geometry import CameraModel, object_points          # noqa: E402
from tools.analyze.calib import pnp_calib as PC# noqa: E402

CAM = CameraModel.pinhole(1280, 720, fx=782.5, fy=779.8, cx=640.0, cy=360.0)
W_TRUE, H_TRUE = 0.60, 0.40        # "真实"门（开口内缘的口径）


# ------------------------------------------------------------------ 造数据
def _rvec_for(yaw_deg, pitch_deg=0.0):
    """**船体**偏航 → 门在相机系里的姿态。"""
    return np.array([0.0, -np.radians(yaw_deg), np.radians(pitch_deg)],
                    np.float64).reshape(3, 1)


def _tvec_for(z, lat=0.0, up=0.0):
    # 相机系：x 右、y 下、z 前；门在光轴上方 (up>0) ⇒ y 为负
    return np.array([lat, -up, z], np.float64).reshape(3, 1)


def make_dump(path, z, lat=0.0, up=0.0, yaw=0.0, n=40, W_model=W_TRUE,
              H_model=H_TRUE, drop=(), jitter_px=0.0, seed=0, img_w=1280, img_h=720,
              lead_full=0):
    """按"真实门尺寸"投影生成 dump（PnP 侧要用的尺寸由调用方另给，模拟 cfg 与实际不符）。"""
    rng = np.random.RandomState(int(seed))
    obj = object_points(W_model, H_model)
    rvec, tvec = _rvec_for(yaw), _tvec_for(z, lat, up)
    uv = CAM.project(obj, rvec, tvec)
    lines = []
    for i in range(n):
        k = uv + (rng.randn(*uv.shape) * jitter_px if jitter_px else 0.0)
        conf = [0.95] * 4
        if i >= lead_full:
            for d in drop:
                conf[d] = 0.10                  # 低于 conf_thr → 视为不可见
        x0, y0 = float(k[:, 0].min()), float(k[:, 1].min())
        x1, y1 = float(k[:, 0].max()), float(k[:, 1].max())
        rec = {"t": round(i * 0.1, 3), "frame": i + 1, "n_det": 1, "fps": 10.0,
               "dets": [{"kind": "gate", "score": 0.9,
                         "bbox": [int(x0), int(y0), int(max(1, x1 - x0)),
                                  int(max(1, y1 - y0))],
                         "kpts": [[round(float(a), 2), round(float(b), 2)] for a, b in k],
                         "kpt_conf": conf}]}
        lines.append(json.dumps(rec))
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return path


# ------------------------------------------------------------------ 1. 文件名真值
@pytest.mark.parametrize("name,exp", [
    ("pnp_z150.jsonl", (1.50, 0.0, 0.0, 0.0)),
    ("pnp_z075.jsonl", (0.75, 0.0, 0.0, 0.0)),
    ("pnp_z150_lat+25.jsonl", (1.50, 0.25, 0.0, 0.0)),
    ("pnp_z150_lat-25.jsonl", (1.50, -0.25, 0.0, 0.0)),
    ("pnp_z200_up+10.jsonl", (2.00, 0.0, 0.10, 0.0)),
    ("pnp_z150_yaw+20.jsonl", (1.50, 0.0, 0.0, 20.0)),
    ("pnp_z400_yaw-30.jsonl", (4.00, 0.0, 0.0, -30.0)),
    ("tag_run3_z150_lat+10_yaw-10.jsonl", (1.50, 0.10, 0.0, -10.0)),
])
def test_parse_gt_from_name(name, exp):
    g = PC.parse_gt_from_name("/x/log/" + name)
    assert g is not None, name
    assert (g.z_m, g.lat_m, g.up_m, g.yaw_deg) == pytest.approx(exp)
    assert g.fronto == all(abs(v) < 1e-9 for v in exp[1:])


def test_parse_gt_from_name_without_z():
    assert PC.parse_gt_from_name("pnp_run1.jsonl") is None, "没 z 真值就该返回 None（去用 --gt）"


# ------------------------------------------------------------------ 2/3. 标尺反演
def test_calib_recovers_correct_scale_when_cfg_matches(tmp_path):
    """cfg 门宽 == 真实门宽 → a≈1、b≈0、相对误差≈0，建议的 frame_w 就是原值。"""
    files = [make_dump(str(tmp_path / ("pnp_z%03d.jsonl" % (z * 100))), z, seed=int(z * 100))
             for z in (0.75, 1.00, 1.50, 2.00, 3.00)]
    per_file, res = PC.run(files, camera=CAM, W=W_TRUE, H=H_TRUE,
                           conf_thr=0.7, reproj_px=20.0, do_invert=True, verbose=False)
    assert len(per_file) == 5
    for f in per_file:
        assert f["z_n"] > 0, "合成数据应该每帧都解出 full 位姿"
        assert abs(f["z_rel_p50"]) < 0.02, \
            "%s 深度相对误差 %.3f 应≈0" % (os.path.basename(f["path"]), f["z_rel_p50"])
    fit = res["fit"]
    a, b, r2, n = fit
    assert a == pytest.approx(1.0, abs=0.02)
    assert b == pytest.approx(0.0, abs=0.02)
    assert r2 > 0.99
    assert res["inv"]["W_star"] == pytest.approx(W_TRUE, abs=0.02), \
        "标尺本来正确 → 反演给出的门宽应还是 %.3f，实际 %.3f" % (W_TRUE, res["inv"]["W_star"])


def test_calib_recovers_real_door_size_from_wrong_cfg(tmp_path):
    """**核心用例**：cfg 标 0.70×0.50（外轮廓），真实是 0.60×0.40（开口内缘）。"""
    W_cfg, H_cfg = 0.70, 0.50
    files = [make_dump(str(tmp_path / ("pnp_z%03d.jsonl" % (z * 100))), z, seed=int(z * 100))
             for z in (1.00, 1.50, 2.00, 3.00)]
    per_file, res = PC.run(files, camera=CAM, W=W_cfg, H=H_cfg,
                           conf_thr=0.7, reproj_px=20.0, do_invert=True, verbose=False)
    a = res["fit"][0]
    assert a == pytest.approx(1.17, abs=0.04), "应报出 z 系统性偏大约 17%%，实际 a=%.3f" % a
    assert res["inv"]["W_star"] == pytest.approx(W_TRUE, abs=0.02), \
        "建议门宽应回到真实 %.3f，实际 %.3f" % (W_TRUE, res["inv"]["W_star"])
    best = min(res["inv"]["ar_table"], key=lambda t: t[2])
    assert best[1] == pytest.approx(H_TRUE, abs=0.03), \
        "建议门高应回到真实 %.3f，实际 %.3f（宽高比 %.2f）" % (H_TRUE, best[1], best[0])
    assert best[2] < 0.02, "校正后深度相对误差应落到 2% 以内，实际 %.1f%%" % (100 * best[2])


def test_calib_reports_near_range_rejection_from_wrong_aspect(tmp_path):
    """**近距会被 reproj 门槛整档拒掉**：宽高比不对时，近距离的残差（px）更大。"""
    near = make_dump(str(tmp_path / "pnp_z060.jsonl"), 0.60, seed=60)
    far = make_dump(str(tmp_path / "pnp_z200.jsonl"), 2.00, seed=200)
    per_file, _ = PC.run([near, far], camera=CAM, W=0.70, H=0.50,
                         conf_thr=0.7, reproj_px=20.0, do_invert=False, verbose=False)
    by = {os.path.basename(f["path"]): f for f in per_file}
    assert by["pnp_z060.jsonl"]["z_n"] in (None, 0), "近距 + 错比例 → 位姿应被全部拒掉"
    assert by["pnp_z060.jsonl"]["n_mode"]["full"] == 40, "但 mode 判定仍是 full（角点都在）"
    assert by["pnp_z200.jsonl"]["z_n"] == 40, "同参数在 2 m 处应能解出"


# ------------------------------------------------------------------ 4/5. 横向/竖向/航向
def test_calib_lateral_vertical_and_heading_signs(tmp_path):
    """t_x 跟 lat 同号；t_y 跟 up **反号**；psi ≈ -yaw（符号约定最容易搞反的地方）。"""
    f1 = make_dump(str(tmp_path / "pnp_z150_lat+25.jsonl"), 1.50, lat=0.25)
    f2 = make_dump(str(tmp_path / "pnp_z150_up+10.jsonl"), 1.50, up=0.10)
    f3 = make_dump(str(tmp_path / "pnp_z150_yaw+20.jsonl"), 1.50, yaw=20.0)
    per_file, _ = PC.run([f1, f2, f3], camera=CAM, W=W_TRUE, H=H_TRUE,
                         conf_thr=0.7, reproj_px=20.0, do_invert=False, verbose=False)
    by = {os.path.basename(f["path"]): f for f in per_file}
    a = by["pnp_z150_lat+25.jsonl"]
    assert a["tx_p50"] == pytest.approx(0.25, abs=0.02), "门在光轴右侧 ⇒ t_x 应为正"
    b = by["pnp_z150_up+10.jsonl"]
    assert b["ty_p50"] == pytest.approx(-0.10, abs=0.02), "门在光轴上方 ⇒ t_y 应为负"
    c = by["pnp_z150_yaw+20.jsonl"]
    assert c["psi_p50"] == pytest.approx(-20.0, abs=3.0), \
        "船右转 20° ⇒ psi 应≈ -20°（门法向偏向画面左），实际 %.1f" % c["psi_p50"]


# ------------------------------------------------------------------ 6. p3p 分类与偏差
def test_p3p_needs_prev_or_fails(tmp_path):
    """3 角 PnP **必须有上帧猜值**：全程只有 3 角时 mode=p3p 但位姿解不出来。"""
    no_prev = make_dump(str(tmp_path / "pnp_z200_drop.jsonl"), 2.00, n=20, drop=(3,), seed=2)
    per_file, _ = PC.run([no_prev], camera=CAM, W=W_TRUE, H=H_TRUE,
                         conf_thr=0.7, reproj_px=20.0, do_invert=False, verbose=False)
    f = per_file[0]
    assert f["n_mode"]["p3p"] == 20, "3 角帧应全部归到 p3p"
    assert f["z_n"] in (None, 0), "没有上帧猜值 → 3 点解不出来（应有 z_n=0/None）"

    # 前面给几帧全角点（建立 prev）→ 后面的 3 角帧就能解出位姿
    with_prev = make_dump(str(tmp_path / "pnp_z200_drop2.jsonl"), 2.00, n=20,
                          drop=(3,), seed=3, lead_full=5)
    per_file, res = PC.run([with_prev], camera=CAM, W=W_TRUE, H=H_TRUE,
                           conf_thr=0.7, reproj_px=20.0, sweep=True, verbose=False)
    f = per_file[0]
    assert f["n_mode"]["full"] == 5 and f["n_mode"]["p3p"] == 15
    assert f["z_any_n"] == 20, \
        "有 prev 之后 15 帧 p3p 也应解出位姿（合计 20 帧），实际 %s" % f["z_any_n"]
    assert f["z_n"] == 5, "z_n 只统计 full 档（5 帧）"
    rows = {(r["file"], r["mode"]): r for r in res["sweeps"]["p3p"]}
    assert ("pnp_z200_drop2.jsonl", "p3p") in rows, "p3p 的行必须单独报出来"
    assert rows[("pnp_z200_drop2.jsonl", "p3p")]["n"] == 15


# ------------------------------------------------------------------ 7. CSV 输出
def test_csv_outputs(tmp_path):
    files = [make_dump(str(tmp_path / ("pnp_z%03d.jsonl" % (z * 100))), z, seed=int(z * 100))
             for z in (1.00, 2.00)]
    per_file, _ = PC.run(files, camera=CAM, W=W_TRUE, H=H_TRUE,
                         conf_thr=0.7, reproj_px=20.0, do_invert=False, verbose=False)
    c1 = str(tmp_path / "sum.csv")
    c2 = str(tmp_path / "frames.csv")
    PC.write_summary_csv(c1, per_file)
    PC.write_frames_csv(c2, per_file)
    n1 = len(open(c1, encoding="utf-8").read().strip().splitlines())
    n2 = len(open(c2, encoding="utf-8").read().strip().splitlines())
    assert n1 == 3, "2 档 + 表头"
    assert n2 == 1 + 40 * 2, "表头 + 40 帧 × 2 档"


# ------------------------------------------------------------------ 8. 报告可生成
def test_report_renders(tmp_path):
    files = [make_dump(str(tmp_path / ("pnp_z%03d.jsonl" % (z * 100))), z, seed=int(z * 100))
             for z in (1.00, 2.00)]
    per_file, res = PC.run(files, camera=CAM, W=W_TRUE, H=H_TRUE,
                           conf_thr=0.7, reproj_px=20.0, sweep=True, verbose=False)
    md = PC.render_md(per_file, {"conf_thr": 0.7, "reproj_px": 20.0},
                      res["cam"], res["W"], res["H"], res["inv"], res["sweeps"])
    for key in ("# 过门 PnP 位姿 / 深度标定报告", "## 1. 深度", "## 2. 横向",
                "## 3. 航向", "## 4. 选参扫描", "## 5. 结论与待办"):
        assert key in md, "报告缺章节：%s" % key
