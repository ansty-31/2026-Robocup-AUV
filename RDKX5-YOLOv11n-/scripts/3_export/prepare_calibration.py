#!/usr/bin/env python3
"""准备 PTQ 量化的校准集：从图片目录均匀抽样，转成 640×640 RGB float32 的 `.rgb`。

用法：
    python scripts/3_export/prepare_calibration.py --coco-path <代表性图片目录>
    # 例：data/datasets/AUV_1_AUV.yolov11（比 COCO 更贴近场景）或 /path/to/coco/val2017

约定：letterbox 保持宽高比、补 114 灰边（与 YOLO 训练预处理一致）；输出目录默认
<项目根>/calibration_data，detect / pose 用各自的 --output-dir，别混用。
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # scripts/<stage>/x.py

import cv2
import numpy as np
from glob import glob


def letterbox_resize(img, target_size=640):
    """Letterbox缩放：保持宽高比，填充到目标尺寸（与YOLO训练预处理一致）"""
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def prepare_calibration_data(image_dir, output_dir, target_size=640,
                             num_images=100):
    os.makedirs(output_dir, exist_ok=True)

    image_files = glob(os.path.join(image_dir, '*.jpg')) + \
                  glob(os.path.join(image_dir, '*.png'))
    if not image_files:
        print(f"❌ 在 {image_dir} 中未找到图片")
        return False

    # 均匀抽样，保证覆盖不同场景
    step = max(1, len(image_files) // num_images)
    image_files = image_files[::step][:num_images]

    print(f"开始处理 {len(image_files)} 张图片...")
    for i, img_path in enumerate(image_files):
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️  跳过损坏的图片: {img_path}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = letterbox_resize(img_rgb, target_size)
        img_chw = img_resized.transpose(2, 0, 1)          # HWC -> CHW
        img_float = img_chw.astype(np.float32)
        out_path = os.path.join(output_dir,
                                f'{os.path.basename(img_path).rsplit(".", 1)[0]}.rgb')
        img_float.tofile(out_path)
        if (i + 1) % 10 == 0:
            print(f"  进度: {i + 1}/{len(image_files)}")

    print(f"\n✅ 校准数据准备完成 → {output_dir}")
    print(f"   文件数量: {len(image_files)}")
    print(f"   文件格式: RGB float32, shape=(3, {target_size}, {target_size})")
    return True


def main():
    ap = argparse.ArgumentParser(description='准备 PTQ 量化校准集（RGB float32 CHW .rgb）')
    ap.add_argument('--coco-path', type=str, required=True,
                    help='代表性图片目录（含 *.jpg/*.png），如 COCO val2017 或本项目帧集')
    ap.add_argument('--output-dir', type=str, default=str(PROJECT_ROOT / 'calibration_data'))
    ap.add_argument('--num-images', type=int, default=100)
    ap.add_argument('--target-size', type=int, default=640)
    a = ap.parse_args()

    ok = prepare_calibration_data(
        image_dir=a.coco_path, output_dir=a.output_dir,
        target_size=a.target_size, num_images=a.num_images)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
