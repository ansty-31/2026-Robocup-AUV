#!/usr/bin/env python3
"""
畸变实验 · 挑图用的接触印相表（contact sheet）

从一次下水里挑"有门框、清晰、姿态有代表性"的帧，靠一张张点开看太慢。
本脚本把一批原始帧缩成小图拼成网格（每格带帧名与清晰度），几千张一眼扫完：

    python experiment/scripts/exp_distortion/make_contact_sheet.py \
        data/AUV_2/auv_20260910_172208_frames \
        --out experiment/runs/exp_distortion/sheets/auv2.jpg \
        --tile 240 --cols 8 --per-sheet 200

产物：`<out>` 一张大图（帧数超过 `--per-sheet` 时自动拆成 `<out>_p2.jpg, _p3.jpg …`），
并在同目录写 `<out>.csv`：每格 → 原始文件名，挑完把文件名抄进挑选清单即可。

挑图建议（本实验口径）：
* **门框四角都在画面内**、`TL/TR/BR/BL` 都分得清 → 标注才可用；
* 门框要覆盖**不同画面位置**（中心 / 偏左 / 偏右 / 偏上 / 偏下）与不同**距离**
  —— 畸变误差是径向的，只有偏心样本才能把"外圈更差"这个效应测出来；
* 避开运动模糊帧（格子下方 `Lap` 是拉普拉斯方差，越大越清晰）。
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
LABEL_H = 22


def main() -> None:
    ap = argparse.ArgumentParser(
        description="原始帧 → 接触印相表（挑图用）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="帧目录或图片（可多个）")
    ap.add_argument("--out", type=Path, required=True, help="输出大图路径（.jpg）")
    ap.add_argument("--tile", type=int, default=240, help="每格最长边像素")
    ap.add_argument("--cols", type=int, default=8, help="每行格数")
    ap.add_argument("--per-sheet", type=int, default=200, help="每张最多多少格")
    ap.add_argument("--start", type=int, default=0, help="从第几张开始")
    ap.add_argument("--every", type=int, default=1, help="每隔多少张取一张")
    ap.add_argument("--quality", type=int, default=88)
    a = ap.parse_args()

    files: list[Path] = []
    for item in a.inputs:
        p = Path(item)
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS)
        elif p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    if not files:
        sys.exit("❌ 没有图片输入")
    files = files[a.start::max(1, a.every)]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    rows = math.ceil(a.per_sheet / a.cols)
    canvas_h = rows * (a.tile + LABEL_H)
    canvas_w = a.cols * a.tile
    print(f"🖼  {len(files)} 张 → 每格 {a.tile}px，每张 {a.per_sheet} 格"
          f"（{rows}×{a.cols}）")

    rows_csv: list[dict] = []
    sheet_idx = 0
    placed = 0
    canvas = np.full((canvas_h, canvas_w, 3), 18, np.uint8)

    def flush():
        p = (a.out if sheet_idx == 0 else
             a.out.with_name(f"{a.out.stem}_p{sheet_idx + 1}{a.out.suffix}"))
        cv2.imwrite(str(p), canvas, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
        print(f"   → {p}")

    for f in files:
        if placed >= a.per_sheet:
            flush()
            sheet_idx += 1
            placed = 0
            canvas = np.full((canvas_h, canvas_w, 3), 18, np.uint8)
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = a.tile / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        r, c = divmod(placed, a.cols)
        y0, x0 = r * (a.tile + LABEL_H), c * a.tile
        ih, iw = img.shape[:2]
        canvas[y0:y0 + ih, x0:x0 + iw] = img
        cv2.putText(canvas, f"{f.stem}  Lap{lap:.0f}", (x0 + 3, y0 + ih + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 255), 1)
        rows_csv.append({"sheet": sheet_idx, "row": r, "col": c, "file": f.name,
                         "path": str(f), "lap_var": round(lap, 1)})
        placed += 1
    flush()

    csv_path = a.out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["sheet", "row", "col", "file", "path", "lap_var"])
        w.writeheader()
        w.writerows(rows_csv)
    print(f"✅ 共 {sheet_idx + 1} 张印相表；索引 {csv_path}")


if __name__ == "__main__":
    main()
