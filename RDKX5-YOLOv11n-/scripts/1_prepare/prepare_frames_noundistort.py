#!/usr/bin/env python3
"""
图片集 → YOLO 训练图集（**不做去畸变**：640x640 + 画面补偿）

与 `prepare_frames.py` 完全同链路，唯一区别是**跳过去畸变 remap**：

    prepare_frames.py            去畸变 remap → resize(640) → enhance
    prepare_frames_noundistort   (跳过)        → resize(640) → enhance   ← 本脚本

## 为什么要这个变体

板端推理正常情况下会先 `cv2.remap` 去畸变，但**去畸变依赖标定文件与相机**：
标定 yaml 缺失/失效、`vision.yaml` 里 `undistort: false`、或相机被换过/磕碰过时，
送进模型的其实是**未校正的桶形畸变图**。若训练集只有去畸变图，模型在畸变图上的
精度会明显掉。

把「不去畸变」的图**按比例混入训练集**，模型同时见过两种几何分布，韧性更好
（配合 `select_frames.py --mix-ratio` 使用）。

## ⚠️ 标注注意

不去畸变会**保留几何畸变**，因此同一帧的「去畸变版」和「不去畸变版」**标注框不通用**，
必须**分别标注**（两版文件名不同，见 `select_frames.py --mix-prefix`）。

参数（白平衡增益/CLAHE/gamma/size）仍以板端 `cfg/vision.yaml` 为唯一来源
（`--config`，镜像见 configs/README.md）；`image.undistort` 在本脚本中**被忽略**
（本脚本的定义就是不去畸变）。

用法示例：
    # 与 prepare_frames.py 完全一样的用法，只是不做去畸变
    python scripts/1_prepare/prepare_frames_noundistort.py \\
        data/AUV_3/auv_20260911_201754_frames \\
        --config configs/vision.yaml \\
        --out-dir data/AUV_3/processed_640_noundistort

输出: <out_dir>/<来源目录名>/frame_000001.jpg ...（每张 640x640，未去畸变）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # scripts/<stage>/x.py

# 复用 prepare_frames.py 的同一套实现，保证两个变体**只有去畸变这一步不同**
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_frames as pf  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="图片集 → (不去畸变)+640x640+画面补偿 训练图集（去畸变变体的补充）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="图片目录或图片文件（可多个/通配符；目录名作为分组）")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("processed_640_noundistort"),
                    help="输出根目录（每个输入组一个子目录）")
    ap.add_argument("--size", type=int, default=640,
                    help="输出/模型输入尺寸（板端 model.input_size）")
    ap.add_argument("--wb-gains", type=pf.parse_gains, default=[1.0, 1.0, 1.0],
                    help="白平衡通道增益 B,G,R（板端 image.white_balance_bgr）")
    ap.add_argument("--clahe", type=float, default=2.0,
                    help="CLAHE clipLimit（板端 image.clahe_clip）")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="gamma 值（板端 image.gamma）")
    ap.add_argument("--config", type=Path, default=None,
                    help="板端 vision.yaml：读取增益/CLAHE/gamma/尺寸"
                         "（其中 image.undistort 被忽略，本脚本就是不去畸变）")
    a = ap.parse_args()

    if a.config:
        if not a.config.exists():
            sys.exit(f"❌ 配置文件不存在: {a.config}")
        board = pf.load_board_config(a.config)
        if board["gains"]:
            a.wb_gains = [float(x) for x in board["gains"]]
        if board["clahe"] is not None:
            a.clahe = float(board["clahe"])
        if board["gamma"] is not None:
            a.gamma = float(board["gamma"])
        if board["size"]:
            a.size = int(board["size"])
        print(f"📄 已从 {a.config} 读取参数: gains={a.wb_gains} "
              f"clahe={a.clahe} gamma={a.gamma} size={a.size} "
              f"(board undistort={board['undistort']} → 本脚本忽略，不做去畸变)")

    groups = pf.collect_image_sets(a.inputs)
    # calibration=None → pf.prepare 跳过去畸变，其余（640 缩放 + 补偿）完全相同
    n = pf.prepare(groups, a.out_dir, None, a.wb_gains, a.clahe, a.gamma, a.size)
    print(f"\n🎉 共生成 {n} 张**未去畸变**训练图片（{a.size}x{a.size}），"
          f"位于 {a.out_dir}/")
    print("下一步：与去畸变图一起混入训练集（选图时加 --mix-ratio）:")
    print("   python scripts/1_prepare/select_frames.py <去畸变目录> <本目录> \\")
    print("       --weights weights/yolo11n.pt --mix-ratio 0.35 --keep 2000 ...")


if __name__ == "__main__":
    main()
