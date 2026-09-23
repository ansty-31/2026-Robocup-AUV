# -*- coding: utf-8 -*-
"""tests/tooling/test_ruler_calib.py — 「卷尺刻度靶子」测量与解算（§A0b / §B4）的校验。

为什么这样测：这条实验的目的只有一个 —— **从已知长度的刻度里把 `fx` 与 `c` 解出来**，
它是把"相机/介质比例"和"门框口径"分开的唯一手段（只看门框时两者简并）。
所以核心用例是**合成往返**：给定期望的 `(fx, c)` → 合成 Δu → 走工具链 → 解回同一个 `(fx, c)`。

覆盖：
  1. `span_stats`：3 个刻度点的跨度/半跨/斜视诊断（含正对 vs 斜视）；
  2. `make_ruler_record` schema（`kind: "ruler"`，含 z 真值解析）；
  3. `load_records`：z 从 `src` 兜底解析、跳过非靶子行；
  4. `solve`：**精确与带噪往返**、2 档可解、<2 档报错、≥3 档给留一校验；
  5. `report`：含 fx/c 与 `k_med = fx_cfg/fx`，skew 超 2% 会告警；
  6. 端到端：JSONL → `main()` → 报告落盘。
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

from tools.analyze import label_corners as LC                          # noqa: E402
from tools.analyze import ruler_calib as RC                            # noqa: E402


# ------------------------------------------------------------------ 合成
def synth(fx, c, zs, span=1.0, noise=0.0, seed=0):
    """按 `Δu = fx·L/(z+c)` 合成靶子读数。"""
    rng = np.random.default_rng(seed)
    out = []
    for z in zs:
        du = fx * span / (z + c) * (1.0 + rng.normal(0.0, noise))
        out.append({"src": "cap_%03d.jpg" % int(z * 100), "kind": "ruler",
                    "span_m": span, "z_tape": z, "du": du, "skew": 0.0})
    return out


# ------------------------------------------------------------------ span_stats
def test_span_stats_even_ticks():
    st = LC.span_stats([(100, 300), (350, 300), (600, 300)])
    assert st["du_l"] == pytest.approx(250.0)
    assert st["du_r"] == pytest.approx(250.0)
    assert st["du"] == pytest.approx(500.0)
    assert st["skew"] == pytest.approx(0.0)


def test_span_stats_yaw_probe():
    """靶面绕竖轴偏 → 近侧半跨变大：skew 有符号、约等于 (L/2)·sinψ/z。"""
    st = LC.span_stats([(100, 300), (355, 300), (600, 300)])       # 左半 255、右半 245
    assert st["skew"] == pytest.approx((245.0 - 255.0) / 500.0)
    assert st["skew"] < 0                                          # 右半更小 = 右侧更远
    assert st["du"] == pytest.approx(500.0)


def test_span_stats_wrong_point_count():
    with pytest.raises(ValueError):
        LC.span_stats([(0, 0), (10, 0)])


def test_ruler_record_schema_and_z_parse():
    rec = LC.make_ruler_record("ruler_z200/cap_001.jpg", [(100, 300), (350, 300), (600, 300)],
                               z_tape=None, span_m=1.0)
    assert rec["kind"] == "ruler"
    assert rec["du"] == pytest.approx(500.0)
    assert rec["span_m"] == 1.0
    assert LC._z_from_name("log/pnp_0922/ruler_z150/cap_001.jpg") == 1.5
    assert LC._z_from_name("cap_001.jpg") is None


# ------------------------------------------------------------------ load
def test_load_records_z_fallback_and_skip(tmp_path):
    p = tmp_path / "ruler.jsonl"
    rows = [
        {"src": "ruler_z200/cap_1.jpg", "kind": "ruler", "span_m": 1.0,
         "p": [[100, 300], [350, 300], [600, 300]]},                # 没有 z_tape → 从 src 抠
        {"src": "x.jpg", "kind": "ruler", "z_tape": None, "du": 100.0},   # 抠不出 z → 跳过
        {"src": "gate.jpg", "n_det": 1, "dets": []},                      # 非靶子行 → 跳过
        {"src": "ruler_z300/cap_1.jpg", "kind": "ruler", "span_m": 1.0, "z_tape": 3.0, "du": 260.0},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    recs, skipped = RC.load_records([str(p)])
    assert [r["z"] for r in recs] == [2.0, 3.0]
    assert recs[0]["du"] == pytest.approx(500.0)                   # 由 p 三点算出
    assert skipped == 2


# ------------------------------------------------------------------ solve
@pytest.mark.parametrize("fx,c", [(1110.0, 0.0), (1010.0, -0.16), (782.5, 0.0), (946.0, 0.05)])
def test_solve_recovers_parameters(fx, c):
    res = RC.solve(synth(fx, c, [1.5, 2.0, 3.0]))
    assert res["fx"] == pytest.approx(fx, rel=1e-6)
    assert res["c"] == pytest.approx(c, abs=1e-9)
    assert res["rms_px"] < 1e-6


def test_solve_two_levels_only():
    """2 档就能解开 (fx, c) —— 这是"只做 1.5 与 3.0"能成立的理由。"""
    res = RC.solve(synth(946.0, 0.0, [1.5, 3.0]))
    assert res["fx"] == pytest.approx(946.0, rel=1e-6)
    assert res["c"] == pytest.approx(0.0, abs=1e-9)


def test_solve_single_level_raises():
    with pytest.raises(ValueError):
        RC.solve(synth(1000.0, 0.0, [2.0]))


def test_solve_noise_is_within_budget():
    """1% 的 Δu 噪声 → fx 优于 3%（判据）。"""
    res = RC.solve(synth(1110.0, 0.0, [1.5, 2.0, 3.0], noise=0.01, seed=7))
    assert abs(res["fx"] - 1110.0) / 1110.0 < 0.03


def test_solve_leave_one_out_present():
    res = RC.solve(synth(1110.0, 0.0, [1.5, 2.0, 3.0]))
    assert len(res["loo"]) == 3
    assert all(abs(focal["x"] - 1110.0) < 1.0 for _, focal, _ in res["loo"])


def test_report_mentions_k_and_skew_warning():
    recs = synth(1110.0, 0.0, [1.5, 2.0, 3.0])
    recs[1]["skew"] = 0.05                                        # 故意歪一档
    res = RC.solve([r for r in recs if abs(r["skew"]) < 0.02])    # 解算只用正对的档
    txt = RC.report(recs, 0, res, fx_cfg_value=782.5)
    assert "k = 0.705" in txt                     # 对比 cfg 的比例（每轴一行）
    assert "skew > 2%" in txt


# ------------------------------------------------------------------ CLI 接线
def test_cli_registers_measure_option():
    """`--measure` 必须注册在 `parse_args` **之前** —— 否则 `--help` 里看不到它、
    传进去还会报 "unrecognized arguments"（2026-09-22 我自己踩过这个坑）。"""
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
        LC.main(["--help"])
    txt = buf.getvalue()
    assert "--measure" in txt and "--span-m" in txt


def test_measure_no_images_returns_2(tmp_path):
    """`--measure` 被认出（不是 SystemExit），只是找不到图 → 返回 2。"""
    assert LC.main(["--measure", "--images", str(tmp_path / "none"),
                    "--out", str(tmp_path / "r.jsonl")]) == 2


def test_measure_requires_out(tmp_path):
    assert LC.main(["--measure", "--images", str(tmp_path)]) == 2


def test_span_stats_vertical_span_not_zero():
    """**2026-09-22 实测踩过**：卷尺竖着放时旧代码用 x 差 → 379 px 的真实跨度被算成 2 px。
    现在用欧氏距离，任意方向都成立。"""
    st = LC.span_stats([(679, 456), (677, 275), (681, 77)])
    assert st["du"] == pytest.approx(379.0, abs=0.05)
    assert abs(st["theta_deg"]) > 85                       # 竖直
    assert st["skew"] == pytest.approx((379 - 181 - 181) / 379, abs=0.005)


def test_span_stats_diagonal_span():
    st = LC.span_stats([(100, 100), (300, 300), (500, 500)])
    assert st["du"] == pytest.approx(np.hypot(400, 400), abs=0.01)
    assert st["skew"] == pytest.approx(0.0, abs=1e-9)
    assert st["theta_deg"] == pytest.approx(45.0, abs=0.01)


def test_span_stats_reports_mid_off_line():
    """中点没点在两端连线上（卷尺有折角/点错刻度）→ off_mid 报警。"""
    st = LC.span_stats([(0, 0), (100, 30), (400, 0)])
    assert st["off_mid"] > 20
    assert st["mid_frac"] < 0.5


def test_load_records_heals_stale_du_and_skew(tmp_path):
    """老版本写下的 `du`/`skew`（竖向时是垃圾）在 `p` 还在时会被重算 —— 记录自动被治好。"""
    p = tmp_path / "ruler.jsonl"
    p.write_text(json.dumps({"src": "cap_001.jpg", "kind": "ruler", "span_m": 1.0,
                             "z_tape": 3.0, "du": 2.0, "skew": 3.0,
                             "p": [[679, 456], [677, 275], [681, 77]]}) + "\n",
                 encoding="utf-8")
    recs, _ = RC.load_records([str(p)])
    assert recs[0]["du"] == pytest.approx(379.0, abs=0.05)
    assert abs(recs[0]["skew"]) < 0.1


def test_solve_same_distance_twice_is_rejected():
    """同一个距离量 3 遍解不开 (f, c)：要明确报错，而不是给出 c=−3 m 这种垃圾。"""
    recs = synth(1100.0, 0.0, [3.0, 3.0, 3.0])
    with pytest.raises(ValueError, match="同一个距离"):
        RC.solve(recs)


# ------------------------------------------------------------------ 联立（靶子 + 门框）
def gate_synth(fx, c, W, zs):
    """合成门框档（Δu = fx·W/(z+c)）。"""
    return [{"src": "A_pnp_z%d.jsonl" % int(z * 100), "z": z,
             "du": fx * W / (z + c), "n": 5} for z in zs]


def test_joint_recovers_focal_c_and_width():
    fy, c, W, ratio = 1107.0, -0.08, 0.727, 782.537 / 779.788
    ruler = [{"src": "cap_1.jpg", "z": 3.0, "du": fy * 1.0 / (3.0 + c), "span": 1.0}]
    gate = gate_synth(fy * ratio, c, W, [1.5, 2.0, 2.5])
    best = RC.solve_joint(ruler, gate, ratio=ratio)
    assert best["fy"] == pytest.approx(fy, rel=0.02)
    assert best["c"] == pytest.approx(c, abs=0.02)
    assert best["W"] == pytest.approx(W, rel=0.03)
    assert best["rms_rel"] < 0.005


def test_joint_needs_gate_and_ruler():
    with pytest.raises(ValueError):
        RC.solve_joint([{"src": "a", "z": 3.0, "du": 379.0, "span": 1.0}], [])
    with pytest.raises(ValueError):
        RC.solve_joint([], gate_synth(1100.0, 0.0, 0.75, [1.5]))


def test_load_gate_parses_names_and_skips_offaxis(tmp_path):
    """门框 dump 的真值来自文件名；`lat/up/yaw` 非 0 的档不能用于联立。"""
    def wr(name, kpts):
        (tmp_path / name).write_text(
            json.dumps({"dets": [{"kpts": kpts, "kpt_conf": [1.0] * 4}]}) + "\n",
            encoding="utf-8")
    wr("A_pnp_z150.jsonl", [[100, 300], [500, 300], [500, 500], [100, 500]])
    wr("A_pnp_z150_lat+25.jsonl", [[100, 300], [480, 300], [480, 500], [100, 500]])
    wr("A_nogate.jsonl", [[100, 300], [500, 300], [500, 500], [100, 500]])
    g = RC.load_gate([str(tmp_path / "*.jsonl")])
    assert [(x["z"], x["du"], x["n"]) for x in g] == [(1.5, 400.0, 1)]


def test_joint_end_to_end_main(tmp_path):
    p = tmp_path / "ruler.jsonl"
    p.write_text(json.dumps({"src": "cap_1.jpg", "kind": "ruler", "span_m": 1.0,
                             "z_tape": 3.0, "du": 1137.0}) + "\n", encoding="utf-8")
    gdir = tmp_path / "g"
    gdir.mkdir()
    for z, du in ((1.5, 570.0), (2.0, 431.0), (2.5, 329.0)):
        (gdir / ("A_pnp_z%d.jsonl" % int(z * 100))).write_text(
            json.dumps({"dets": [{"kpts": [[100, 300], [100 + du, 300],
                                           [100 + du, 500], [100, 500]],
                                  "kpt_conf": [1.0] * 4}]}) + "\n", encoding="utf-8")
    rep = tmp_path / "joint.md"
    assert RC.main([str(p), "--gate", str(gdir / "*.jsonl"), "--fx-cfg", "782.5",
                    "--report", str(rep), "--title", "合成联立"]) == 0
    txt = rep.read_text(encoding="utf-8")
    assert "W_eff" in txt and "k_air" in txt and "简并剖面" in txt


# ------------------------------------------------------------------ 端到端
def test_end_to_end_main_writes_report(tmp_path):
    p = tmp_path / "ruler.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in synth(1010.0, -0.16, [1.5, 2.0, 3.0])) + "\n",
                 encoding="utf-8")
    rep = tmp_path / "ruler.md"
    assert RC.main([str(p), "--report", str(rep), "--fx-cfg", "782.5"]) == 0
    txt = rep.read_text(encoding="utf-8")
    assert "1010.0" in txt and "-0.160" in txt and "留一校验" in txt


def test_main_no_usable_rows(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    assert RC.main([str(p)]) == 2


def test_main_skip_level(tmp_path):
    p = tmp_path / "ruler.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in synth(1110.0, 0.0, [1.5, 2.0, 3.0])) + "\n",
                 encoding="utf-8")
    assert RC.main([str(p), "--skip", "3.0", "--fx-cfg", "782.5"]) == 0


# ------------------------------------------------- 混合轴向（fx 与 fy 分开解）
def _rec(src, z, span, du, theta, axis=None):
    r = {"src": src, "kind": "ruler", "span_m": span, "z_tape": z, "du": du}
    if theta is not None:
        r["theta_deg"] = theta
    if axis is not None:
        r["axis"] = axis
    return r


def test_axis_of():
    assert RC.axis_of({"theta_deg": 0.3}) == "x"
    assert RC.axis_of({"theta_deg": 89.0}) == "y"
    assert RC.axis_of({"theta_deg": -89.0}) == "y"        # 竖着从上往下点也是 y
    assert RC.axis_of({"theta_deg": 46.0}) == "y"         # ≥45 归 y
    assert RC.axis_of({"theta_deg": 44.0}) == "x"
    assert RC.axis_of({"axis": "y"}) == "y"               # 没 theta 时看 axis 键
    assert RC.axis_of({}) == "x"                          # 兜底


def test_solve_separates_fx_and_fy():
    """横向档量到 fx、竖向档量到 fy —— 不再需要「两轴等比」这个假设。"""
    fx, fy, c = 1110.0, 1130.0, -0.08               # 故意各向异性 1.8%
    recs = [_rec("horiz_z150.jpg", 1.5, 0.5, fx * 0.5 / (1.5 + c), 0.5),
            _rec("horiz_z300.jpg", 3.0, 0.5, fx * 0.5 / (3.0 + c), -1.0),
            _rec("vert_z300.jpg", 3.0, 1.0, fy * 1.0 / (3.0 + c), 89.0)]
    res = RC.solve(recs)
    assert res["focal"]["x"] == pytest.approx(fx, rel=1e-6)
    assert res["focal"]["y"] == pytest.approx(fy, rel=1e-6)
    assert res["c"] == pytest.approx(c, abs=1e-9)
    assert res["axes"] == ["x", "y"]


def test_solve_rank_deficient_is_rejected():
    """同一轴上重复同一距离 ⇒ 该列与 c 列线性相关 ⇒ 3 未知量实际只有 2 个自由度。
    必须报错，而不是像 2026-09-22 自检里那样给出 fx=-223 px 这种垃圾。"""
    recs = [_rec("h1.jpg", 2.0, 0.5, 270.0, 0.0), _rec("h2.jpg", 2.0, 0.5, 270.0, 0.0),
            _rec("v.jpg", 3.0, 1.0, 370.0, 89.0)]
    with pytest.raises(ValueError, match="秩不足"):
        RC.solve(recs)


def test_solve_same_distance_same_axis_is_rejected():
    """只有同一距离的档 → 先撞上「同一个距离」这条更直白的保护。"""
    with pytest.raises(ValueError, match="同一个距离"):
        RC.solve([_rec("h1.jpg", 2.0, 0.5, 270.0, 0.0), _rec("h2.jpg", 2.0, 0.5, 271.0, 0.0)])


def test_vertical_points_without_theta_still_become_y_axis(tmp_path):
    """手写记录（只有 p）也要能判出轴线 —— 三条竖着的点 → y。"""
    p = tmp_path / "ruler.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"src": "a.jpg", "kind": "ruler", "span_m": 1.0, "z_tape": 3.0,
         "p": [[679, 456], [677, 275], [681, 77]]},
        {"src": "b.jpg", "kind": "ruler", "span_m": 1.0, "z_tape": 1.5,
         "p": [[679, 600], [679, 300], [679, 30]]}]) + "\n", encoding="utf-8")
    recs, _ = RC.load_records([str(p)])
    assert [r["axis"] for r in recs] == ["y", "y"]
