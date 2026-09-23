#!/usr/bin/env python3
"""
录制文件 → 逐帧切出原始图片（统一入口，自动识别格式）

支持两类输入，自动判断，参数统一：

  1) 容器视频（.mp4/.avi/.mov/.mkv）      → cv2 逐帧解码保存
  2) 裸 MJPEG 流（.mjpeg/.mjpg，文件头 FF D8 FF，JPEG 首尾直接拼接）
                                          → 按 JPEG 标记【无损】字节切割
                                            （不解码、零画质损失、速度快）
     若存在 <文件>.timestamps（逐帧时间戳），可用 --every-seconds 按时间抽帧

用途：先切出原始帧 → 人工分类（红球/蓝球/门/标定板…）→
      再用 scripts/1_prepare/prepare_frames.py 统一预处理（去畸变 + 补偿 + 640x640）。

用法示例：
    python scripts/1_prepare/extract_frames.py data/AUV_2/auv_xxx.mjpeg              # 裸流全量切
    python scripts/1_prepare/extract_frames.py data/AUV_2/auv_xxx.mjpeg --every-seconds 1
    python scripts/1_prepare/extract_frames.py data/AUV_1/rec_front.mp4 --step 5     # 容器视频抽帧
    python scripts/1_prepare/extract_frames.py <文件> --dry-run                       # 只统计不写盘

输出: <同目录>/<文件名>_frames/frame_000001.jpg ...（原始分辨率）
"""

from __future__ import annotations

import argparse
import mmap
import sys
from pathlib import Path

import cv2

IMG_EXTS = (".mjpeg", ".mjpg")          # 视为裸 MJPEG 流的扩展名
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
SOI = b"\xff\xd8\xff"                    # JPEG Start Of Image


# ---------------------------------------------------------------- 格式识别
def is_raw_mjpeg(path: Path) -> bool:
    """文件头为 FF D8 FF 即裸 MJPEG 流（单帧 JPEG 或拼接流）"""
    with open(path, "rb") as f:
        return f.read(3) == SOI


def read_timestamps(path: Path) -> list[float]:
    """读录制器的逐帧时间戳文件 -> [arrival_unix, ...]"""
    ts = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                ts.append(float(parts[1]))
            except ValueError:
                pass
    return ts


# ---------------------------------------------------------------- 裸 MJPEG
def extract_mjpeg(src: Path, out_dir: Path, step: int, start: int,
                  max_frames: int, every_seconds: float) -> tuple[int, int]:
    """无损切割裸 MJPEG 流：帧 = SOI(i) 到 SOI(i+1)，返回 (保存数, 跳过数)"""
    ts_path = Path(str(src) + ".timestamps")
    stamps = read_timestamps(ts_path) if ts_path.exists() else None

    with open(src, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        offsets, pos = [], 0
        while (i := mm.find(SOI, pos)) >= 0:     # 扫描所有帧起点
            offsets.append(i)
            pos = i + 3
        total = len(offsets)
        info = f"，{src.stat().st_size / 1e9:.2f} GB"
        if stamps:
            info += (f"，时间戳 {len(stamps)} 帧 / {stamps[-1] - stamps[0]:.1f}s")
        print(f"🎞  裸 MJPEG: {src.name}  共 {total} 帧{info}")

        selected = list(range(start, total))
        if every_seconds > 0:
            if not stamps:
                mm.close()
                raise RuntimeError("--every-seconds 需要 <文件>.timestamps")
            picked, next_t = [], None
            for i in selected:
                if i >= len(stamps):
                    break
                if next_t is None or stamps[i] >= next_t:
                    picked.append(i)
                    next_t = stamps[i] + every_seconds
            selected = picked
        elif step > 1:
            selected = selected[::step]
        if max_frames:
            selected = selected[:max_frames]

        out_dir.mkdir(parents=True, exist_ok=True)
        saved = skipped = 0
        for i in selected:
            end = offsets[i + 1] if i + 1 < total else len(mm)
            data = mm[offsets[i]:end]
            if not data.rstrip(b"\x00").endswith(b"\xff\xd9"):  # 帧尾校验
                skipped += 1
                continue
            saved += 1
            (out_dir / f"frame_{saved:06d}.jpg").write_bytes(data)
            if saved % 2000 == 0:
                print(f"   … 已保存 {saved}/{len(selected)}", flush=True)
        mm.close()
    return saved, skipped


# ---------------------------------------------------------------- 容器视频
def extract_video(src: Path, out_dir: Path, step: int, start: int,
                  max_frames: int, quality: int) -> tuple[int, int]:
    """cv2 逐帧解码保存（重编码为 JPEG）"""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {src}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"🎬 容器视频: {src.name}  {w}x{h}  共 {total} 帧")
    print(f"   输出: {out_dir}/  step={step}  start={start}")

    out_dir.mkdir(parents=True, exist_ok=True)
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    saved, idx = 0, start
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - start) % step == 0:
            saved += 1
            cv2.imwrite(str(out_dir / f"frame_{saved:06d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, quality])
            if saved % 500 == 0:
                print(f"   … 已保存 {saved} 帧 (idx={idx})", flush=True)
            if max_frames and saved >= max_frames:
                break
        idx += 1
    cap.release()
    return saved, 0


# ---------------------------------------------------------------- 入口
def main() -> None:
    ap = argparse.ArgumentParser(
        description="录制文件 → 逐帧原始图片（自动识别容器视频 / 裸 MJPEG 流）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("files", nargs="+", help="一个或多个视频/裸 MJPEG 流文件")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="输出目录（默认: 同目录/<文件名>_frames/）")
    ap.add_argument("--step", type=int, default=1,
                    help="每隔多少帧取一帧（1=全部）")
    ap.add_argument("--every-seconds", type=float, default=0.0,
                    help="按时间戳每隔 S 秒取一帧（仅裸 MJPEG 且有 .timestamps）")
    ap.add_argument("--start-frame", type=int, default=0, help="起始帧号")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="每个文件最多输出帧数（0=不限）")
    ap.add_argument("--quality", type=int, default=95,
                    help="JPEG 质量（仅容器视频需要重编码时生效）")
    ap.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    a = ap.parse_args()

    for name in a.files:
        src = Path(name)
        if not src.is_file():
            print(f"⚠️  文件不存在，跳过: {src}")
            continue
        out = a.out_dir or src.with_name(src.stem + "_frames")
        try:
            if is_raw_mjpeg(src):
                if a.dry_run:
                    with open(src, "rb") as f:
                        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                        n = 0
                        pos = 0
                        while (i := mm.find(SOI, pos)) >= 0:
                            n += 1
                            pos = i + 3
                        mm.close()
                    print(f"🔎 dry-run: {src.name} 裸 MJPEG 约 {n} 帧 -> {out}")
                    continue
                saved, skipped = extract_mjpeg(
                    src, out, a.step, a.start_frame, a.max_frames,
                    a.every_seconds)
                msg = f"✅ 已输出 {saved} 帧 -> {out}"
                if skipped:
                    msg += f"（跳过 {skipped} 个不完整帧）"
                print(msg)
            else:
                if src.suffix.lower() not in VIDEO_EXTS:
                    print(f"ℹ️  {src.name} 非典型容器扩展名，仍按视频解码尝试")
                if a.dry_run:
                    cap = cv2.VideoCapture(str(src))
                    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    cap.release()
                    print(f"🔎 dry-run: {src.name} 容器视频约 {n} 帧 -> {out}")
                    continue
                saved, _ = extract_video(
                    src, out, a.step, a.start_frame, a.max_frames, a.quality)
                print(f"✅ 已输出 {saved} 帧 -> {out}")
        except RuntimeError as e:
            print(f"❌ {e}")
            sys.exit(1)

    print("   下一步：人工分类后运行 "
          "python scripts/1_prepare/prepare_frames.py <分类目录> --config configs/vision.yaml")


if __name__ == "__main__":
    main()
