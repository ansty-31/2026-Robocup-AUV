# -*- coding: utf-8 -*-
"""tests/tooling/test_label_corners.py — 手工标注工具（`tools/analyze/label_corners.py`）的校验。

为什么这样测：这条「岸上路子」的价值全在于 —— **手工标的 4 个点能喂通 `pnp_calib`，
并反演出正确的深度标尺**。所以核心用例是一次**端到端往返**：
    已知位姿 → 合成角点 → 走 label_corners 写成 dump → pnp_calib 复算 → 深度对得上。
GUI 部分无法在无显示器环境测（已用 `--coords` 离线模式覆盖其数据通路）。

覆盖：
  1. 吸附到红管（snap_to_red）：点到偏了会吸到管子中心；非红区域不乱跳；
  2. 记录 schema 与 `preview_detect --dump` 一致（pnp_calib 能读）；
  3. **端到端**：coords 模式 → dump → `pnp_calib.run()` → z 正确、`frame_w` 反演正确；
  4. `--resume` 按文件名跳过已标注；
  5. bbox 由 4 点算出。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gate.geometry import CameraModel, object_points                    # noqa: E402
from tools.analyze import label_corners as LC                           # noqa: E402
from tools.analyze import pnp_calib as PC                               # noqa: E402

CAM = CameraModel.pinhole(1280, 720, fx=782.5, fy=779.8, cx=640.0, cy=360.0)
W_TRUE, H_TRUE = 0.60, 0.40


# ------------------------------------------------------------------ 1. 吸附
def test_snap_to_red_snaps_onto_the_pipe():
    """红管在 x=100..108：点在管子**左侧外** (98,50) → 吸到管子上、y 不变；
    点在管子上 → 原地不动；点在远离管子处 → 保持点击（不乱吸）。"""
    img = np.full((120, 200, 3), 40, np.uint8)
    img[40:60, 100:109] = (30, 30, 210)                 # BGR：红管（9 px 宽）
    x, y = LC.snap_to_red(img, 98, 50, r=6, min_gain=25)
    assert 100.0 <= x <= 108.0, "应吸到管子上（x∈[100,108]），实际 %.1f" % x
    assert x > 98.0, "必须往管子方向动"
    assert y == 50.0, "竖直方向不该动"
    # 已经在管子上 → 原地（同红度取最近 = 自己）
    x2, y2 = LC.snap_to_red(img, 104, 50, r=6, min_gain=25)
    assert (x2, y2) == (104.0, 50.0)
    # 离管子太远（窗口里没有红）→ 保持点击
    x3, y3 = LC.snap_to_red(img, 60, 50, r=6, min_gain=25)
    assert (x3, y3) == (60.0, 50.0)


def test_snap_to_red_keeps_click_on_non_red():
    """灰度图上没有红 → 保持原点击坐标（不许因为噪声乱吸）。"""
    img = np.full((120, 200, 3), 128, np.uint8)
    img[::7, ::5] = 150                                  # 一点纹理
    x, y = LC.snap_to_red(img, 33, 77, r=6, min_gain=25)
    assert (x, y) == (33.0, 77.0)


# ------------------------------------------------------------------ 2. schema
def test_record_schema_matches_preview_dump():
    rec = LC.make_record(3, [(10, 20), (110, 22), (112, 92), (8, 90)], src="pnp_z150.jpg")
    assert set(rec) >= {"t", "frame", "n_det", "dets"}
    d = rec["dets"][0]
    assert set(d) == {"kind", "score", "bbox", "kpts", "kpt_conf"}
    assert d["kind"] == "gate" and d["score"] == 1.0
    assert len(d["kpts"]) == 4 and all(len(p) == 2 for p in d["kpts"])
    assert d["kpt_conf"] == [1.0] * 4
    assert rec["src"] == "pnp_z150.jpg" and rec["label"] == "manual"
    # 下游读取器认的是这几个键（pnp_calib.read_frames / _det_key）
    assert rec["dets"][0]["kind"] == "gate"
    assert isinstance(rec["t"], float) and isinstance(rec["frame"], int)


def test_bbox_from_points():
    b = LC.bbox_of([(10.4, 20.6), (110.0, 22.0), (112.0, 92.0), (8.0, 90.0)])
    # x0/y0 取整、宽高按未取整的差再取整（与 preview_detect 的 bbox 语义一致）
    assert b == [8, 21, 104, 71]


# ------------------------------------------------------------------ 3. 端到端
def _write_dump_with_kpts(path, z, lat=0.0, up=0.0, yaw=0.0, n=6, W_model=W_TRUE,
                          H_model=H_TRUE):
    """直接造"手工标注"式 dump：已知位姿投影出角点 → 写成 label_corners 的格式。"""
    obj = object_points(W_model, H_model)
    rvec = np.array([0.0, -np.radians(yaw), 0.0], np.float64).reshape(3, 1)
    tvec = np.array([lat, -up, z], np.float64).reshape(3, 1)
    uv = CAM.project(obj, rvec, tvec)
    lines = []
    for i in range(n):
        rec = LC.make_record(i + 1, uv, src="pnp_z%03d_%02d.jpg" % (z * 100, i))
        lines.append(json.dumps(rec, ensure_ascii=False))
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def test_labeled_dump_is_read_by_pnp_calib_and_depth_is_right(tmp_path):
    """**核心**：手工标注的 dump 喂 pnp_calib → 深度正确、标尺反演正确。"""
    f = str(tmp_path / "pnp_z150.jsonl")
    _write_dump_with_kpts(f, 1.50)
    per_file, res = PC.run([f], camera=CAM, W=W_TRUE, H=H_TRUE,
                           conf_thr=0.7, reproj_px=20.0, do_invert=False, verbose=False)
    assert len(per_file) == 1
    s = per_file[0]
    assert s["n_mode"]["full"] == 6, "标注点 conf=1.0 → 必须判成 full 档"
    assert s["z_p50"] == pytest.approx(1.50, abs=0.02), \
        "手工标注的深度应精确等于真值，实际 %.3f" % s["z_p50"]
    assert s["usable"] == pytest.approx(1.0)


def test_end_to_end_scale_recovery_from_labeled_dumps(tmp_path):
    """多档标注 → 反演出正确的 frame_w/frame_h（模拟真实标定流程）。"""
    files = []
    for z in (1.00, 1.50, 2.00, 3.00):
        p = str(tmp_path / ("pnp_z%03d.jsonl" % (z * 100)))
        _write_dump_with_kpts(p, z)
        files.append(p)
    per_file, res = PC.run(files, camera=CAM, W=0.70, H=0.50,     # 故意给错标尺
                           conf_thr=0.7, reproj_px=20.0, do_invert=True, verbose=False)
    assert res["fit"][0] == pytest.approx(0.70 / W_TRUE, abs=0.03), \
        "标尺错 1.167× 应被拟合出来，实际 a=%.3f" % res["fit"][0]
    assert res["inv"]["W_star"] == pytest.approx(W_TRUE, abs=0.02)
    best = min(res["inv"]["ar_table"], key=lambda t: t[2])
    assert best[1] == pytest.approx(H_TRUE, abs=0.03)


def test_coords_cli_writes_one_record(tmp_path):
    """`--coords` 离线模式：不依赖相机/窗口，写一条合法记录（GUI 的数据通路）。"""
    out = str(tmp_path / "pnp_z150.jsonl")
    rc = LC.main(["--coords", "10,20;110,22;112,92;8,90", "--out", out,
                  "--src", "pnp_z150.jpg"])
    assert rc == 0
    rec = json.loads(open(out, encoding="utf-8").read().strip())
    assert rec["dets"][0]["kpts"][0] == [10.0, 20.0]
    assert rec["src"] == "pnp_z150.jpg"
    # 文件名真值也能被 pnp_calib 解析
    assert PC.parse_gt_from_name("pnp_z150.jpg").z_m == pytest.approx(1.50)


# ------------------------------------------------------------------ 4. resume
def test_load_done_and_resume(tmp_path):
    out = str(tmp_path / "pnp_z150.jsonl")
    LC.main(["--coords", "1,2;3,4;5,6;7,8", "--out", out, "--src", "cap_001.jpg"])
    LC.main(["--coords", "1,2;3,4;5,6;7,8", "--out", out, "--src", "cap_002.jpg"])
    done = LC.load_done(out)
    # 去重键：有 src_path 用路径，老记录退回 basename ⇒ 两个集合的并集就是"已处理过的图"
    assert (done["paths"] | done["legacy"]) == {"cap_001.jpg", "cap_002.jpg"}
    n = len([l for l in open(out, encoding="utf-8") if l.strip()])
    assert n == 2, "两次调用应追加成 2 行，实际 %d" % n


def test_same_basename_across_dirs_does_not_collide(tmp_path):
    """**2026-09-22 现场踩过**：采图目录里的文件名永远是 cap_001.jpg…
    ⇒ 按 basename 去重会让 `ruler_z100` 的记录被 `ruler_z300` 整行删掉，
    `--resume` 更会直接跳过整档。必须按**完整路径**去重。"""
    out = str(tmp_path / "ruler.jsonl")
    a = {"src": "cap_001.jpg", "src_path": "log/pnp_0922/ruler_z100/cap_001.jpg",
         "kind": "ruler", "span_m": 0.5, "z_tape": 1.0, "du": 540.0}
    b = {"src": "cap_001.jpg", "src_path": "log/pnp_0922/ruler_z300/cap_001.jpg",
         "kind": "ruler", "span_m": 0.5, "z_tape": 3.0, "du": 180.0}
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(a) + "\n" + json.dumps(b) + "\n")
    done = LC.load_done(out)
    assert len(done["paths"]) == 2
    assert LC.is_done(done, "log/pnp_0922/ruler_z100/cap_001.jpg", "cap_001.jpg")
    assert not LC.is_done(done, "log/pnp_0922/ruler_z200/cap_001.jpg", "cap_001.jpg")
    # 删 3 m 的那条，1 m 的必须留下
    assert LC.remove_src(out, "log/pnp_0922/ruler_z300/cap_001.jpg") == 1
    rows = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
    assert [r["z_tape"] for r in rows] == [1.0]


def test_ticks_option_sets_span(tmp_path, capsys):
    """`--ticks 10,50,90` ⇒ span 自动 = 0.80 m（避免手填 span 出错）。"""
    img = tmp_path / "cap_001.jpg"
    import numpy as np
    import cv2
    cv2.imwrite(str(img), np.zeros((40, 200, 3), np.uint8))
    out = str(tmp_path / "ruler.jsonl")
    # 不起窗口（无显示器环境）→ 只验证参数解析与 span 推导被打印
    LC.main(["--measure", "--images", str(tmp_path / "none*.jpg"), "--out", out,
             "--ticks", "10,50,90"])
    assert "span = 0.800 m" in capsys.readouterr().out
    assert LC.main(["--measure", "--images", str(tmp_path / "none*.jpg"), "--out", out,
                    "--ticks", "10,50"]) == 2


def test_parse_coords_rejects_wrong_count():
    with pytest.raises(SystemExit):
        LC.parse_coords("1,2;3,4;5,6")

def test_remove_src_replaces_old_record_on_relabel(tmp_path):
    """回上一张重标时必须**替换**旧记录，而不是追加 —— 否则同一张图两行会把它在中位数里加权两次。"""
    out = str(tmp_path / "pnp_z150.jsonl")
    LC.main(["--coords", "10,20;110,22;112,92;8,90", "--out", out, "--src", "cap_001.jpg"])
    LC.main(["--coords", "11,21;111,23;113,93;9,91", "--out", out, "--src", "cap_001.jpg"])
    assert LC.remove_src(out, "cap_001.jpg") == 2, "两条旧记录都要被删掉"
    LC.main(["--coords", "12,22;112,24;114,94;10,92", "--out", out, "--src", "cap_001.jpg"])
    rows = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
    assert len(rows) == 1 and rows[0]["dets"][0]["kpts"][0] == [12.0, 22.0]
    _d = LC.load_done(out)
    assert (_d["paths"] | _d["legacy"]) == {"cap_001.jpg"}
