#!/usr/bin/env python3
"""
畸变实验 · 一次生成三组 640×640 训练图（A 先去畸变 / B 不去畸变 / C 最后去畸变）

## 这个脚本解决什么

「畸变对水下门框特征点识别的影响」要比较三条链路，**必须保证除几何之外的一切
都相同**（同一批原始帧、同一套白平衡/CLAHE/gamma、同一输出尺寸、同一 JPEG 质量），
而且**同一原始帧在三组里必须同名**，否则后面按帧对齐做统计就无从下手。

    A  A_undistort_first : remap@720p   -> resize640 -> enhance      ← 板端当前链路
    B  B_no_undistort    :             resize640 -> enhance
    C  C_undistort_last  :             resize640 -> enhance -> remap@640

像素操作全部复用 `scripts/1_prepare/prepare_frames.py` 里的
`enhance()` 与 `calibration_maps()`/`distortion_geometry` 的映射表（同一份
`cv2.CV_16SC2` remap），参数一律从板端镜像 `configs/vision.yaml` 读 —— 见
`distortion_geometry.py` 顶部关于「A/C 是同一套几何、只差重采样顺序」的推导。

⚠️ **去畸变的标定质量决定了 A/C 组的上限**：本工程现行
`configs/front_camera.yaml` 只在棋盘覆盖到的中心区域可信（见 `calib_diagnose.py`）。
脚本结尾会把可逆像素占比打出来，别忽略。

## 用法

    # 1) 先把四次下水挑出来的原始 720p 帧放进一个目录（可多个目录，按目录分组）
    #    例：data/exp_distortion/picked/{auv1,auv2,auv3,auv4}/

    # 2) 一次生成三组（同一帧三组同名，前缀 = 源目录名）
    python experiment/scripts/exp_distortion/prepare_distortion_groups.py \
        data/exp_distortion/picked/auv1 data/exp_distortion/picked/auv2 \
        data/exp_distortion/picked/auv3 data/exp_distortion/picked/auv4 \
        --out-root data/exp_distortion/processed \
        --config configs/vision.yaml

    # 只想先跑一组看看
    python ... --groups A C

产物：
    <out-root>/A_undistort_first/<源目录名>__<原名>.jpg
    <out-root>/B_no_undistort/<源目录名>__<原名>.jpg
    <out-root>/C_undistort_last/<源目录名>__<原名>.jpg
    <out-root>/experiment.json      ← 全部参数 + 几何自检 + 源文件映射
                                       （后续标注换算/评估脚本都用它当唯一参数源）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # 1_prepare/
sys.path.insert(0, str(Path(__file__).resolve().parent))              # exp_distortion/
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "1_prepare"))
import prepare_frames as pf                                            # noqa: E402
from distortion_geometry import (Geometry, GROUP_A, GROUP_B, GROUP_C,  # noqa: E402
                                 GROUP_DESC, ALL_GROUPS, resolve_group)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    h.update(path.read_bytes())
    return h.hexdigest()


def collect_groups(inputs: list[str]):
    """[(组名, [图片...])]，组名 = 目录名；单目录时可用 --no-prefix"""
    groups: dict[str, list[Path]] = {}
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files = sorted(f for f in p.iterdir()
                           if f.suffix.lower() in IMG_EXTS)
            if not files:
                print(f"⚠️  目录中没有图片，跳过: {p}")
                continue
            groups.setdefault(p.name, []).extend(files)
        elif p.is_file() and p.suffix.lower() in IMG_EXTS:
            groups.setdefault(p.parent.name, []).append(p)
        else:
            print(f"⚠️  无法识别（既非目录也非图片）: {p}")
    if not groups:
        sys.exit("❌ 没有可处理的图片输入")
    return list(groups.items())


def out_name(group: str, path: Path, prefix: bool) -> str:
    stem = path.stem + (path.suffix.lower() if path.suffix.lower() != ".jpeg"
                        else ".jpg")
    return f"{group}__{stem}" if prefix else stem


def process_one(raw: np.ndarray, geo_group, which: str, gains, clahe, gamma,
                size: int, c_order: str) -> np.ndarray:
    """按组名走对应链路；所有像素操作都来自 prepare_frames / distortion_geometry"""
    h, w = raw.shape[:2]
    g = geo_group.for_frame(w, h)

    if which == GROUP_A:                       # remap@720p -> resize -> enhance
        m0, m1 = g.maps_720_i16
        frame = cv2.remap(raw, m0, m1, cv2.INTER_LINEAR)
        if (w, h) != (size, size):
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
        return pf.enhance(frame, gains, clahe, gamma)

    if which == GROUP_B:                       # resize -> enhance
        frame = raw
        if (w, h) != (size, size):
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
        return pf.enhance(frame, gains, clahe, gamma)

    if which == GROUP_C:                       # resize -> [enhance] -> remap@640
        frame = raw
        if (w, h) != (size, size):
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
        if c_order == "before-enhance":
            frame = cv2.remap(frame, *g.maps_640_i16, cv2.INTER_LINEAR)
            return pf.enhance(frame, gains, clahe, gamma)
        frame = pf.enhance(frame, gains, clahe, gamma)
        return cv2.remap(frame, *g.maps_640_i16, cv2.INTER_LINEAR)

    raise ValueError(f"未知组: {which}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="一次生成三组 640×640 训练图（A 先/B 无/C 最后 去畸变）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="挑好的原始 720p 帧目录（可多个，按目录名分组）")
    ap.add_argument("--out-root", type=Path,
                    default=PROJECT_ROOT / "data" / "exp_distortion" / "processed",
                    help="输出根目录（每组一个子目录）")
    ap.add_argument("--config", type=Path,
                    default=PROJECT_ROOT / "configs" / "vision.yaml",
                    help="板端 vision.yaml 镜像：增益/CLAHE/gamma/尺寸/标定路径的唯一来源")
    ap.add_argument("--calibration", type=Path, default=None,
                    help="标定 yaml（默认取 --config 里 camera.front.calibration）")
    ap.add_argument("--size", type=int, default=None, help="输出尺寸（默认取板端 model.input_size）")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="getOptimalNewCameraMatrix 的 alpha，板端用 0（不改）")
    ap.add_argument("--c-order", choices=["after-enhance", "before-enhance"],
                    default="after-enhance",
                    help="C 组 remap 与 enhance 的先后（'最后去畸变' = after-enhance）")
    ap.add_argument("--groups", nargs="+", default=list(ALL_GROUPS),
                    type=resolve_group, choices=list(ALL_GROUPS),
                    metavar="A|B|C", help="只生成指定组（可短名 A/B/C）")
    ap.add_argument("--no-prefix", action="store_true",
                    help="输出不加以源目录名为前缀（单目录时用；多目录会重名！）")
    ap.add_argument("--quality", type=int, default=95, help="JPEG 质量（与板端一致 95）")
    ap.add_argument("--allow-size-mismatch", action="store_true",
                    help="允许帧尺寸与标定分辨率不一致（几何会失真，慎用）")
    ap.add_argument("--dry-run", action="store_true", help="只列计划，不写文件")
    a = ap.parse_args()

    # ---- 参数（唯一来源：板端 vision.yaml 镜像）----
    board = pf.load_board_config(a.config)
    gains = [float(x) for x in (board["gains"] or [1.0, 1.0, 1.0])]
    clahe = float(board["clahe"] if board["clahe"] is not None else 2.0)
    gamma = float(board["gamma"] if board["gamma"] is not None else 1.0)
    size = int(a.size or board["size"] or 640)

    if a.calibration:
        calib = Path(a.calibration)
    else:
        loc = pf.localize_calibration(board["calibration"], a.config)
        if not loc:
            sys.exit("❌ 配置里没有 camera.front.calibration，且未显式给 --calibration")
        calib = Path(loc)
    if not calib.exists():
        sys.exit(f"❌ 标定文件不存在: {calib}")

    geo = Geometry(calib, size, a.alpha)
    print(f"📄 板端参数: gains={gains} clahe={clahe} gamma={gamma} size={size}")
    print(f"📄 标定: {calib}")

    groups = collect_groups(a.inputs)
    if a.no_prefix and len(groups) > 1:
        sys.exit("❌ --no-prefix 只能用于单个输入目录（多目录会重名）")

    # ---- 尺寸校验：几何按标定分辨率算，尺寸不符必须先说清楚 ----
    probe = None
    for _, files in groups:
        for f in files:
            probe = cv2.imread(str(f))
            if probe is not None:
                break
        if probe is not None:
            break
    if probe is None:
        sys.exit("❌ 所有输入都读不出来")
    raw_h, raw_w = probe.shape[:2]
    if geo.calib_size and (raw_w, raw_h) != geo.calib_size and not a.allow_size_mismatch:
        sys.exit(f"❌ 帧尺寸 {raw_w}x{raw_h} != 标定分辨率 {geo.calib_size}。\n"
                 f"   去畸变映射必须按标定时的分辨率生成；若确实要用请加 --allow-size-mismatch")

    g = geo.for_frame(raw_w, raw_h)
    print(f"\n{geo.describe()}")

    # ---- 逐组生成 ----
    out_root = a.out_root
    manifest_groups: dict[str, dict] = {}
    sources: list[dict] = []
    t0 = time.time()

    for which in a.groups:
        sub = out_root / which
        sub.mkdir(parents=True, exist_ok=True)
        n_saved = n_fail = 0
        for gname, files in groups:
            for f in files:
                raw = cv2.imread(str(f))
                if raw is None:
                    n_fail += 1
                    continue
                if raw.shape[:2] != (raw_h, raw_w):
                    print(f"⚠️  分辨率不一致，跳过: {f} {raw.shape[:2]}")
                    n_fail += 1
                    continue
                dst = sub / out_name(gname, f, not a.no_prefix)
                if a.dry_run:
                    n_saved += 1
                    continue
                frame = process_one(raw, geo, which, gains, clahe, gamma,
                                    size, a.c_order)
                cv2.imwrite(str(dst), frame, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
                n_saved += 1
                if which == a.groups[0]:
                    sources.append({"out": dst.name, "src": str(f),
                                    "group": gname})
            print(f"   {which} · {gname}: 累计 {n_saved} 张", flush=True)
        manifest_groups[which] = {"dir": str(sub), "n": n_saved,
                                  "n_failed": n_fail, "desc": GROUP_DESC[which]}
        print(f"✅ {which}: {n_saved} 张 -> {sub}"
              + (f"（失败 {n_fail}）" if n_fail else ""))

    if a.dry_run:
        print("\n🔎 dry-run：未写任何文件")
        return

    # ---- 清单 ----
    checks = {**g.self_test(), **g.check_equivalence()}
    stats = g.mappable_stats()
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vision_config": str(a.config),
        "calibration": str(calib),
        "calibration_sha1": sha1(calib),
        "size": size,
        "alpha": a.alpha,
        "raw_size": [raw_w, raw_h],
        "gains_bgr": gains, "clahe": clahe, "gamma": gamma,
        "jpeg_quality": a.quality,
        "c_order": a.c_order,
        "groups": manifest_groups,
        "geometry": {
            "K": g.K.tolist(), "dist": g.dist.tolist(),
            "newK720": g.newK720.tolist(), "newK640": g.newK640.tolist(),
            "sx": g.sx, "sy": g.sy,
        },
        "checks": checks,
        "mappable_stats": stats,
        "sources": sources,
    }
    (out_root / "experiment.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n🧪 几何自检（{raw_w}x{raw_h}）")
    print(f"   解析畸变模型 vs cv2  : {checks['distort_model_vs_cv2_max']:.2e}")
    print(f"   newK640 vs S·newK720 : {checks['newK640_vs_S_newK720_max']:.2e}"
          "   ← 0 表示 A/C 是同一套几何")
    print(f"   解析逆解 vs remap 表 : {checks['map_vs_newton_inverse_px']:.3f} px")
    print(f"   A 域 vs C 域 最大偏差: {checks['domain_a_vs_c_px']:.4f} px")
    print(f"⚠️  标定可逆性: B 组画面里只有 "
          f"{stats['b640_mappable_frac'] * 100:.1f}% 的像素存在去畸变解；"
          f"原始像素半径上限 ≈ {stats['raw_radius_limit_px']:.0f}px"
          f"（四角 {stats['frame_corner_radius_px']:.0f}px）")
    print(f"    → 跨组标注换算 / C′ 组在超出该半径的区域**无定义**，评估脚本会显式标记")
    print(f"\n🎉 三组预处理完成，用时 {time.time() - t0:.0f}s")
    print(f"   清单: {out_root / 'experiment.json'}")
    print("   下一步：拿 A 组的图去标注（只标一套），再用 transform_pose_labels.py 推到 B/C")


if __name__ == "__main__":
    main()
