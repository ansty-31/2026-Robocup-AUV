#!/usr/bin/env python3
"""
棋盘格标定输入（视频 / 图片目录 / 图片） → 相机内参标定（去畸变参数）

做法与 exampls/calibrate_camera.py 一致：
    检测棋盘格内角点 → cornerSubPix 亚像素精化 → cv2.calibrateCamera
    → 结果写入 OpenCV FileStorage yaml：
        image_width / image_height / camera_matrix /
        distortion_coefficients / reprojection_error
    该 yaml 供 scripts/1_prepare/prepare_frames.py（训练图集）与板端 rdk_bpu_worker.py
    （推理帧）生成去畸变 remap 映射，保证训练/推理使用同一份标定参数。

采集策略（自动选择）：
    * 图片目录/图片：全池扫描 → 按清晰度(Laplacian方差)降序、位姿去重(min_shift)
      贪心选 --views 张 —— 避免视频切帧里运动模糊/低质量帧混入；
    * 标定视频：按 --frame-step 抽帧、相邻位移去重，边扫边收到 --views 张即止。
    两种输入共用“离群剔除精修”：迭代剔除单视图重投影误差最大者，直至
    RMS ≤ --max-rms 或剩余视图数 ≤ --min-views。

少于 10 张视为失败；RMS 超过 --max-rms(0.8px) 时仍写出 yaml 但退出码非 0。

用法示例：
    python scripts/1_prepare/calibrate_camera.py data/AUV_1/board \
        --cols 11 --rows 8 --square-mm 20 --output configs/front_camera.yaml \
        --views 35 --min-shift 40 --save-views qc_views/

    python scripts/1_prepare/calibrate_camera.py data/calib/checkerboard.mp4 \
        --cols 9 --rows 6 --output configs/front_camera.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # scripts/<stage>/x.py

import cv2
import numpy as np

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
DETECT_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
TERM = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def _collect_sources(inputs: list[str]):
    """输入归一化为 (视频列表, {组名: 图片列表})"""
    videos, image_groups = [], {}
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files = sorted(f for f in p.iterdir()
                           if f.suffix.lower() in IMG_EXTS)
            if files:
                image_groups.setdefault(p.name, []).extend(files)
            else:
                print(f"⚠️  目录中没有图片: {p}")
        elif p.is_file():
            if p.suffix.lower() in VIDEO_EXTS:
                videos.append(str(p))
            elif p.suffix.lower() in IMG_EXTS:
                image_groups.setdefault(p.parent.name, []).append(p)
            else:
                print(f"⚠️  不支持的文件类型: {p}")
        else:
            print(f"⚠️  路径不存在: {p}")
    return videos, image_groups


def _detect_one(gray, cols, rows):
    """半分辨率FAST预筛 + 全分辨率精检 + 亚像素。命中返回 corners，否则 None"""
    small = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2))
    ok, _ = cv2.findChessboardCorners(small, (cols, rows),
                                      cv2.CALIB_CB_FAST_CHECK)
    if not ok:
        return None
    ok, corners = cv2.findChessboardCorners(gray, (cols, rows), DETECT_FLAGS)
    if not ok:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), TERM)


def _scan_images(groups, cols, rows, progress_every=400):
    """全池扫描图片，返回 [(清晰度, 标签, corners)]"""
    pool = []
    total = sum(len(f) for f in groups.values())
    done = 0
    for name, files in groups.items():
        for f in files:
            img = cv2.imread(str(f))
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                corners = _detect_one(gray, cols, rows)
                if corners is not None:
                    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    pool.append((sharp, f"图片[{name}] {f}", corners))
            done += 1
            if done % progress_every == 0:
                print(f"  … 扫描 {done}/{total}，命中 {len(pool)}", flush=True)
    print(f"📊 扫描完成: 共 {total} 张，棋盘命中 {len(pool)} 张")
    return pool


def _scan_videos(videos, cols, rows, frame_step, views_target, min_shift):
    """流式扫描视频，边扫边按 min_shift 去重，收满 views_target 即止"""
    pool = []
    for video in videos:
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print(f"⚠️  无法打开视频，跳过: {video}")
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        print(f"🎬 扫描视频: {video} ({total} 帧)")
        idx = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % frame_step != 0:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = _detect_one(gray, cols, rows)
            if corners is None:
                continue
            if (pool and np.mean(np.linalg.norm(
                    corners.reshape(-1, 2) - pool[-1][2].reshape(-1, 2),
                    axis=1)) < min_shift):
                continue
            pool.append((0.0, f"视频[{Path(video).name}] 帧{idx}", corners))
            print(f"  ✓ 命中 {len(pool)}/{views_target}: 帧{idx}", flush=True)
            if len(pool) >= views_target:
                break
        cap.release()
    return pool


def _select_best(pool, views_target, min_shift):
    """按清晰度降序贪心：与已选视图角点位移 < min_shift 的跳过"""
    pool.sort(key=lambda v: v[0], reverse=True)
    chosen = []
    for sharp, label, corners in pool:
        if all(np.mean(np.linalg.norm(
                corners.reshape(-1, 2) - c.reshape(-1, 2), axis=1)) >= min_shift
               for _, _, c in chosen):
            chosen.append((sharp, label, corners))
        if len(chosen) >= views_target:
            break
    return chosen


def _refine_with_save(views, obj, size, max_rms, min_views):
    """迭代标定并剔除单视图误差最大者，直到 RMS≤max_rms 或剩余≤min_views。
    返回 (rms, K, dist, [(标签, corners)])"""
    labels = [v[1] for v in views]
    pts = [v[2] for v in views]
    while True:
        rms, k, dist, rvecs, tvecs = cv2.calibrateCamera(
            [obj] * len(pts), pts, size, None, None)
        errs = []
        for o, p, r, t in zip([obj] * len(pts), pts, rvecs, tvecs):
            proj, _ = cv2.projectPoints(o, r, t, k, dist)
            errs.append(float(np.mean(
                np.linalg.norm(p.reshape(-1, 2) - proj.reshape(-1, 2),
                               axis=1))))
        if rms <= max_rms or len(pts) <= min_views:
            return rms, k, dist, list(zip(labels, pts))
        worst = int(np.argmax(errs))
        print(f"  ➖ 剔除 {labels[worst]}（单视图误差 {errs[worst]:.2f}px），"
              f"剩余 {len(pts) - 1} 张")
        del labels[worst]
        del pts[worst]


def calibrate(inputs: list[str], cols: int, rows: int, square_mm: float,
              views_target: int, frame_step: int, min_shift: float,
              max_rms: float, min_views: int,
              output: Path, save_views: Path | None) -> float:
    videos, image_groups = _collect_sources(inputs)
    if not videos and not image_groups:
        raise RuntimeError("没有可用的输入（视频/图片目录/图片）")

    obj = np.zeros((rows * cols, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm

    n_img = sum(len(f) for f in image_groups.values())
    print(f"📥 输入: {n_img} 张图片 / {len(videos)} 个视频   棋盘内角: {cols}x{rows}")

    # 尺寸：以第一张图片为准（要求所有输入同分辨率）
    size = None
    for files in image_groups.values():
        for f in files:
            img = cv2.imread(str(f))
            if img is not None:
                size = (img.shape[1], img.shape[0])
                break
        if size:
            break
    if size is None and videos:
        cap = cv2.VideoCapture(videos[0])
        if cap.isOpened():
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            cap.release()
    if size is None:
        raise RuntimeError("无法确定图像尺寸")

    # 采集：图片全池择优；视频流式去重
    if image_groups:
        pool = _scan_images(image_groups, cols, rows)
        views = _select_best(pool, views_target, min_shift)
        print(f"🔍 按清晰度择优后: {len(views)} 张"
              f"（清晰度 {views[0][0]:.0f}~{views[-1][0]:.0f}）" if views else
              "🔍 无命中视图")
    else:
        views = _scan_videos(videos, cols, rows, frame_step,
                             views_target, min_shift)

    if len(views) < 10:
        raise RuntimeError(
            f"有效标定视图不足: 仅 {len(views)} 张（要求 ≥10）\n"
            "   请检查棋盘格是否完整出现在画面中、cols/rows 是否与棋盘一致")

    if save_views is not None:
        save_views = Path(save_views)
        save_views.mkdir(parents=True, exist_ok=True)

    # 离群剔除精修
    rms, k, dist, final = _refine_with_save(
        views, obj, size, max_rms, min_views)

    output.parent.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(output), cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", size[0])
    fs.write("image_height", size[1])
    fs.write("camera_matrix", k)
    fs.write("distortion_coefficients", dist)
    fs.write("reprojection_error", rms)
    fs.release()

    print(f"✅ 最终采用 {len(final)} 张视图: "
          + ", ".join(lab for lab, _ in final[:8]) + " …")
    if save_views is not None:
        for i, (lab, corners) in enumerate(final, 1):
            if not lab.startswith("图片["):
                continue          # 视频帧无法回溯重读，仅图片输入导出 QC 图
            src = Path(lab.split("] ", 1)[1])
            frame = cv2.imread(str(src))
            if frame is not None:
                cv2.drawChessboardCorners(frame, (cols, rows), corners, True)
                cv2.imwrite(str(save_views / f"view_{i:03d}.jpg"), frame)
    print(f"💾 标定结果已写入: {output}")
    return float(rms)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="棋盘格标定（视频/图片目录/图片） → 相机内参标定（FileStorage yaml）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="棋盘格视频、图片目录或图片文件（可多个）")
    ap.add_argument("--cols", type=int, default=9, help="棋盘格内角点列数")
    ap.add_argument("--rows", type=int, default=6, help="棋盘格内角点行数")
    ap.add_argument("--square-mm", type=float, default=25.0,
                    help="方格边长(mm)，仅影响物理尺度换算，不影响去畸变像素映射")
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "configs" / "front_camera.yaml",
                    help="输出标定 yaml 路径")
    ap.add_argument("--views", type=int, default=35,
                    help="目标有效视图数（≥10 才能标定）")
    ap.add_argument("--frame-step", type=int, default=3,
                    help="视频输入每隔多少帧检测一次（图片输入每张都检测）")
    ap.add_argument("--min-shift", type=float, default=40.0,
                    help="已选视图间最小角点位移(px)，剔除重复/近重复视角")
    ap.add_argument("--min-views", type=int, default=15,
                    help="离群剔除时保留的最少视图数")
    ap.add_argument("--max-rms", type=float, default=0.8,
                    help="RMS 验收阈值(px)，超过则退出码非 0")
    ap.add_argument("--save-views", type=Path, default=None,
                    help="把最终采用帧（画好角点）保存到此目录，便于人工检查")
    a = ap.parse_args()

    try:
        error = calibrate(
            a.inputs, a.cols, a.rows, a.square_mm, a.views,
            a.frame_step, a.min_shift, a.max_rms, a.min_views,
            a.output, a.save_views)
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)

    print(f"RMS reprojection error: {error:.4f}px "
          f"({'✅ ≤' + str(a.max_rms) if error <= a.max_rms else '❌ 超限'})")
    sys.exit(0 if error <= a.max_rms else 1)


if __name__ == "__main__":
    main()
