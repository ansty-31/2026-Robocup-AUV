#!/usr/bin/env python3
"""
按需生成某个域数据集的**变体**（只重渲染图像，标签直接复制）—— 用于快速敏感性试验

为什么要它：§17.3 的「只关 CLAHE」结论是用「保留 WB+gamma、跳过 CLAHE」的测试图 + 现有模型
推理得到的（不需要重训）。当时那段生成代码是临时的、没存下来；本脚本把它固化，
使得那些临时数据集可以随时删除、随时重建。

用法：
    # 生成 C 域的「无 CLAHE」测试集（跳过 CLAHE，WB+gamma 保留）
    python experiment/scripts/exp_distortion/make_variant_dataset.py \
        --base experiment/data/pose_C --variant noclahe --out experiment/runs/domain/pose_C_noclahe
    # 生成「完全无 enhance」的测试集
    python experiment/scripts/exp_distortion/make_variant_dataset.py \
        --base experiment/data/pose_C --variant noenh --out experiment/runs/domain/pose_C_noenh_test
    # 只做 test split（默认），或全量
    ... --splits test          # 默认
    ... --splits train,valid,test
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "1_prepare"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import prepare_frames as pf  # noqa: E402

WB, CLAHE, GAMMA = [1.0, 1.05, 1.15], 0.5, 0.85
CALIB = "configs/front_camera.yaml"
RAW = PROJECT_ROOT / "data/mapped/raw"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="experiment/data/pose_C",
                    help="基准域数据集（用它的 manifest / 标签 / 划分）")
    ap.add_argument("--variant", required=True, choices=["noclahe", "noenh", "raw640"],
                    help="noclahe=WB+gamma 跳过 CLAHE；noenh=只缩放；raw640=只缩放不补偿")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="test")
    a = ap.parse_args()

    base = PROJECT_ROOT / a.base
    out = PROJECT_ROOT / a.out
    man = {r["name"]: r for r in csv.DictReader(open(base / "manifest.csv"))}
    need_maps = any(r["src"] in ("AUV_4dir",) or True for r in list(man.values())[:1])
    m720 = pf.calibration_maps(CALIB, 1280, 720) if need_maps else None

    n = 0
    for name, r in man.items():
        if r["split"] not in a.splits.split(","):
            continue
        src = RAW / r["src"] / f"frame_{int(r['num_src']):06d}.jpg"
        raw = cv2.imread(str(src))
        if raw is None:
            continue
        f = cv2.resize(cv2.remap(raw, *m720, cv2.INTER_LINEAR), (640, 640),
                       interpolation=cv2.INTER_LINEAR)
        if a.variant == "noclahe":
            f = np.clip(f.astype(np.float32) * np.array(WB, np.float32), 0, 255).astype(np.uint8)
            f = cv2.LUT(f, pf._gamma_lut(GAMMA))
        elif a.variant == "noenh":
            pass
        d = out / r["split"]
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "labels").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "images" / f"{name}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (d / "labels" / f"{name}.txt").write_text(
            (base / r["split"] / "labels" / f"{name}.txt").read_text())
        n += 1
    try:
        shown = out.relative_to(PROJECT_ROOT)
    except ValueError:
        shown = out
    print(f"生成 {a.variant} 变体 {n} 张（splits={a.splits}）→ {shown}")


if __name__ == "__main__":
    main()
