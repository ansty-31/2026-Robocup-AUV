#!/usr/bin/env python3
"""
畸变实验 · 标定体检（这份 front_camera.yaml 到底能用在哪里）

## 为什么要单独体检

去畸变实验的上限由标定决定。评测前必须先回答三个问题，否则 A/C 组的结论
可能只是"这份标定错了"，而不是"去畸变没用"：

1. **畸变模型可逆吗？** `r_d = r_u·(1+k1r_u²+k2r_u⁴+k3r_u⁶)/(1+…)` 是径向函数；
   若它在某个 `r_turn` 处取极大 `r_d_max`，则 `r_d > r_d_max` 的原始像素
   **在模型下没有去畸变解**——`cv2.undistortPoints` 在这里会静默返回垃圾
   （本工程实测：原始像素 (0,360) 返回自己，(80,360) 跳到 -525px）。
2. **画面有几成像素真的可逆？** 把可逆上限换算成像素半径，再和"到主点的最远
   像素"比。
3. **棋盘实际覆盖到哪个半径？** 标定只在棋盘出现过的区域受数据约束；外圈是
   多项式外推，可能给出枕形/桶形的**过度校正**（本工程实测就是过度校正）。

只要 2) 或 3) 的答案覆盖不满画面，A/C 组的外圈结论就不成立，必须缩小到
可信区域重做，或者重标（`--board` 会对比不同畸变模型）。

## 用法

    # 只体检现有标定
    python experiment/scripts/exp_distortion/calib_diagnose.py

    # 顺带扫棋盘素材，给出覆盖半径 + 用 5 参数/8 参数(rational) 重标并对比
    python experiment/scripts/exp_distortion/calib_diagnose.py \
        --board data/AUV_1/board --cols 11 --rows 8 --square-mm 20 \
        --refit 5 8 --save-refit experiment/runs/exp_distortion/calib

判读标准（脚本会直接给结论）：
* `可逆半径 ≥ 画面角点半径` → 通过；
* `棋盘观测半径 ≥ 0.9 × 画面角点半径` → 覆盖合格（理想是四角都有棋盘）；
* 拒绝 `undershoot`：可逆半径覆盖了画面但棋盘没去过 → 外圈是外推，不可信。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from distortion_geometry import (Geometry, invertible_limit, load_calibration,  # noqa: E402
                                 radial_curve, distort_normalized)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
TERM = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
DETECT_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE


def scan_board(directory: Path, cols: int, rows: int, K, every: int):
    """扫棋盘素材，返回观测到的归一化畸变半径数组"""
    files = sorted(f for f in directory.iterdir() if f.suffix.lower() in IMG_EXTS)
    rs, hits, frames = [], 0, 0
    for f in files[::every]:
        img = cv2.imread(str(f))
        if img is None:
            continue
        frames += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, c = cv2.findChessboardCorners(gray, (cols, rows), DETECT_FLAGS)
        if not ok:
            continue
        hits += 1
        c = cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), TERM).reshape(-1, 2)
        rs.extend(np.hypot((c[:, 0] - K[0, 2]) / K[0, 0],
                           (c[:, 1] - K[1, 2]) / K[1, 1]).tolist())
    return np.array(rs), hits, frames


def _select_views(directory: Path, cols: int, rows: int, every: int, views: int,
                  min_shift: float):
    """复用 calibrate_camera.py 的择优逻辑，保证重标与现行 yaml 是同一批视图。

    ⚠️ 不这么做的话拟合会被运动模糊/近重复帧带偏（实测 RMS 从 0.7px 变成 5.4px），
    那种对比毫无意义。
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import calibrate_camera as cc
    files = sorted(f for f in directory.iterdir() if f.suffix.lower() in IMG_EXTS)[::every]
    pool = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cc._detect_one(gray, cols, rows)
        if corners is not None:
            pool.append((float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                         f"图片[{directory.name}] {f}", corners))
    return cc._select_best(pool, views, min_shift), len(files)


def _refine(views, obj, size, max_rms: float, min_views: int, flags: int):
    """与 calibrate_camera.py 同款的离群剔除精修，但可指定畸变模型 flags"""
    labels = [v[1] for v in views]
    pts = [v[2] for v in views]
    while True:
        rms, K, d, rvecs, tvecs = cv2.calibrateCamera(
            [obj] * len(pts), pts, size, None, None, flags=flags)
        errs = []
        for o, p, r, t in zip([obj] * len(pts), pts, rvecs, tvecs):
            proj, _ = cv2.projectPoints(o, r, t, K, d)
            errs.append(float(np.mean(np.linalg.norm(
                p.reshape(-1, 2) - proj.reshape(-1, 2), axis=1))))
        if rms <= max_rms or len(pts) <= min_views:
            return rms, K, d, len(pts)
        worst = int(np.argmax(errs))
        del labels[worst]
        del pts[worst]


def refit(directory: Path, cols: int, rows: int, square_mm: float, size,
          views: int, min_shift: float, max_rms: float, n_coef: int,
          every: int = 1):
    """用棋盘重新标定（n_coef=5 → 标准模型；8 → CALIB_RATIONAL_MODEL）"""
    obj = np.zeros((rows * cols, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    picked, n_files = _select_views(directory, cols, rows, every, views, min_shift)
    if len(picked) < 10:
        return None
    flags = 0 if n_coef == 5 else cv2.CALIB_RATIONAL_MODEL
    rms, K, d, used = _refine(picked, obj, size, max_rms, 15, flags)
    return {"n_scanned": n_files, "n_selected": len(picked), "n_used": used,
            "rms": float(rms), "K": np.asarray(K).tolist(),
            "dist": np.asarray(d).ravel().tolist(), "flags": int(flags)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="标定体检：畸变模型可逆性 / 可逆像素占比 / 棋盘覆盖半径",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--calibration", type=Path,
                    default=PROJECT_ROOT / "configs" / "front_camera.yaml")
    ap.add_argument("--size", type=int, default=640, help="训练/推理的方形输入尺寸")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--raw-size", type=int, nargs=2, default=None, metavar=("W", "H"))
    ap.add_argument("--board", type=Path, default=None, help="棋盘素材目录（可选）")
    ap.add_argument("--cols", type=int, default=11, help="棋盘内角点列数")
    ap.add_argument("--rows", type=int, default=8, help="棋盘内角点行数")
    ap.add_argument("--square-mm", type=float, default=20.0)
    ap.add_argument("--every", type=int, default=100, help="扫棋盘时每隔多少张测一次")
    ap.add_argument("--refit", type=int, nargs="*", default=None,
                    choices=[5, 8], help="用棋盘重标并对比（5=标准，8=rational）")
    ap.add_argument("--refit-views", type=int, default=35)
    ap.add_argument("--refit-every", type=int, default=1,
                    help="重标时扫全部棋盘帧（1=每张都检；--every 只管 ② 的覆盖统计）")
    ap.add_argument("--min-shift", type=float, default=40.0,
                    help="视图去重的最小角点位移(px)，与 calibrate_camera.py 一致")
    ap.add_argument("--max-rms", type=float, default=0.8,
                    help="离群剔除的 RMS 目标，与 calibrate_camera.py 一致")
    ap.add_argument("--save-refit", type=Path, default=None,
                    help="把重标结果写到该目录（<n>coef.yaml）")
    ap.add_argument("--out", type=Path, default=None, help="体检报告 json 路径")
    a = ap.parse_args()

    geo = Geometry(a.calibration, a.size, a.alpha,
                   tuple(a.raw_size) if a.raw_size else None)
    raw_size = geo.expected_raw_size()
    if raw_size is None:
        sys.exit("❌ 标定文件没有 image_width/height，请用 --raw-size 指定")
    g = geo.for_frame(*raw_size)
    K, dist = g.K, g.dist
    r_turn, r_d_max = invertible_limit(dist)
    stats = g.mappable_stats()
    corner_r = stats["frame_corner_radius_px"]
    limit_px = stats["raw_radius_limit_px"]

    report = {"calibration": str(a.calibration), "raw_size": list(raw_size),
              "size": a.size, "alpha": a.alpha,
              "dist": dist.tolist(), "n_coeff": int(dist.size),
              "r_turn": r_turn, "r_d_max": r_d_max,
              "raw_radius_limit_px": limit_px,
              "frame_corner_radius_px": corner_r,
              "b640_mappable_frac": stats["b640_mappable_frac"]}

    print(f"📄 {a.calibration}")
    print(f"   标定分辨率 {raw_size[0]}x{raw_size[1]}   dist({dist.size}): "
          f"{np.array2string(dist, precision=5)}")
    print(f"\n① 畸变模型可逆性")
    print(f"   径向函数在 r_u={r_turn:.4f} 处取极大 r_d_max={r_d_max:.4f}")
    print(f"   → 可逆的原始像素半径上限 ≈ {limit_px:.0f}px；画面四角半径 {corner_r:.0f}px"
          f"  {'✅ 覆盖整幅' if limit_px >= corner_r else '❌ 覆盖不满'}")
    print(f"   → B 组画面里可去畸变的像素占比 {stats['b640_mappable_frac'] * 100:.1f}%")

    mag = g.magnification_profile()
    report["magnification"] = mag
    print("\n①b 去畸变的局部放大倍率（|d(a640)/d(b640)|，越大说明外圈被拉伸得越狠）")
    print(f"   {'半径(b640 px)':>16s} {'像素占比':>8s} {'中位':>7s} {'p90':>7s} {'最大':>7s}")
    for r in mag["bins"]:
        print(f"   {r['r_lo']:8.0f}-{r['r_hi']:<7.0f} {r['frac'] * 100:7.1f}% "
              f"{r['mag_median']:7.2f} {r['mag_p90']:7.2f} {r['mag_max']:7.2f}")
    for k, v in mag["safe"].items():
        print(f"   {k:10s}: 占 B 画面 {v['frac_of_b_frame'] * 100:5.1f}%，"
              f"半径 ≤ {v['max_radius_b640_px']:.0f}px")
    print("   ⚠️ 倍率 ≈1 的区域去畸变基本是恒等变换 —— 「能算」不等于「有意义」：")
    print("      畸变效应主要活在倍率大的外圈，而外圈正是本标定没数据的地方。")

    ok1 = limit_px >= corner_r
    ok2 = True
    if a.board and a.board.is_dir():
        rs, hits, frames = scan_board(a.board, a.cols, a.rows, K, a.every)
        if rs.size:
            r_max_obs = float(rs.max())
            report.update({"board_dir": str(a.board), "board_frames_tested": frames,
                           "board_hits": hits, "board_corners": int(rs.size),
                           "board_r_p50": float(np.percentile(rs, 50)),
                           "board_r_p95": float(np.percentile(rs, 95)),
                           "board_r_max": r_max_obs,
                           "board_coverage_frac": r_max_obs / (corner_r / K[0, 0])})
            print(f"\n② 棋盘覆盖（{a.board}）")
            print(f"   抽测 {frames} 帧，命中 {hits}，角点 {rs.size}")
            print(f"   观测归一化半径 p50={np.percentile(rs, 50):.3f} "
                  f"p95={np.percentile(rs, 95):.3f} max={r_max_obs:.3f}")
            print(f"   画面角点半径（归一化）= {corner_r / K[0, 0]:.3f}"
                  f"  → 覆盖到画面半径的 {100 * r_max_obs / (corner_r / K[0, 0]):.0f}%")
            ok2 = r_max_obs >= 0.9 * (corner_r / K[0, 0])
            print(f"   {'✅ 覆盖合格' if ok2 else '❌ 只覆盖到中心区域：'}"
                  + ("" if ok2 else " 外圈的畸变参数是**多项式外推**，不是实测约束，"
                                    "A/C 组的外圈结论不可信"))
        else:
            print(f"\n② 棋盘覆盖：{a.board} 没检出棋盘")

    if a.refit:
        if not a.board or not a.board.is_dir():
            sys.exit("❌ --refit 需要 --board")
        print(f"\n③ 重标对比（{a.board}）")
        fits = {}
        for n in a.refit:
            r = refit(a.board, a.cols, a.rows, a.square_mm, raw_size,
                      a.refit_views, a.min_shift, a.max_rms, n, a.refit_every)
            if r is None:
                print(f"   {n} 参数: 有效视图不足，跳过")
                continue
            d = np.array(r["dist"])
            _, rd = invertible_limit(d)
            fits[f"{n}coef"] = {**r, "r_d_max": rd}
            tag = "标准" if n == 5 else "rational"
            print(f"   {n} 参数({tag}): RMS {r['rms']:.3f}px  "
                  f"扫描 {r['n_scanned']} / 择优 {r['n_selected']} / 采用 {r['n_used']} 视图  "
                  f"r_d_max={rd:.4f}  可逆像素半径 {rd * K[0, 0]:.0f}px"
                  f"  {'✅ 覆盖整幅' if rd * K[0, 0] >= corner_r else '❌ 仍不满'}")
            print(f"      dist = {np.array2string(d, precision=5)}")
            if a.save_refit:
                a.save_refit.mkdir(parents=True, exist_ok=True)
                out = a.save_refit / f"front_camera_{n}coef.yaml"
                fs = cv2.FileStorage(str(out), cv2.FILE_STORAGE_WRITE)
                fs.write("image_width", raw_size[0])
                fs.write("image_height", raw_size[1])
                fs.write("camera_matrix", np.array(r["K"]))
                fs.write("distortion_coefficients", d.reshape(1, -1))
                fs.write("reprojection_error", r["rms"])
                fs.release()
                print(f"      → {out}")
        report["refit"] = fits
        print("   ⚠️ **可逆 ≠ 准确**：`r_d_max` 大只说明「数学上有解」，不说明「校得对」。")
        print("      外圈没有棋盘数据时，多给几个自由度只会让外推更野。本工程实测"
              "（2026-09-22，同一批 35 视图）：")
        print("      · 5 参数 RMS 0.705px，复现出与现行 yaml 完全相同的系数；")
        print("      · 8 参数 rational RMS 0.704px，r_d_max 从 0.664 提到 412"
              "（看着‘可逆了’），但合成栅格去畸变后外圈畸变更离谱，")
        print("        A 画面里只剩 1.9% 的像素有对应（5 参数是 63.5%）——"
              "见 experiment/runs/exp_distortion/preview/grid_5coef_vs_8coef.jpg。")
        print("      ⇒ **换模型不是修复手段，重拍棋盘（棋盘进四角与边缘）才是。**")

    print("\n④ 结论")
    if ok1 and ok2:
        print("   ✅ 标定在整幅画面上可逆且被棋盘数据覆盖，A/C 组可以全画面评估。")
    else:
        if not ok1:
            print(f"   ❌ 可逆半径只到画面角点的 {100 * limit_px / corner_r:.0f}%："
                  f"外圈像素在去畸变域**没有定义**。")
        if not ok2:
            print("   ❌ 棋盘没覆盖到画面外圈：畸变模型在外圈是外推，"
                  "结果是**过度校正**（枕形）。")
        print("   → 处理办法（二选一或都做）：")
        print("      a) 评估时限定在可信区域内（`compare_kpt_spacing.py --valid-only`），"
              "并在结论里明确写「外圈未验证」；")
        print("      b) 重新下水标定：棋盘必须出现在画面**四角与边缘**，"
              "并考虑 `--refit 8`（rational 模型）。")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                         encoding="utf-8")
        print(f"\n💾 {a.out}")


if __name__ == "__main__":
    main()
