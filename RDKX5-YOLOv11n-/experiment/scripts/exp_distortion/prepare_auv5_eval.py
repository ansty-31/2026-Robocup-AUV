#!/usr/bin/env python3
"""
备好 AUV_5（清水）验证集：E1 里人工标过的 16 张门框帧

`experiment/runs/exp_distortion/e1_labels_B.jsonl` 里有 16 张 AUV_5 gate-* 帧的四角点人工标注，
坐标在 `e1_picked/<图>`（640x640，**B 域**：不去畸变 + enhance）里。
本脚本把它们整理成一张 provenance 表 + YOLO pose 数据集，之后就能用
`map_pose_dataset.py` 直接生成 B/C/D 三域版本，用于「中/浊水新权重推理清水」的判定。

输出：
    runs/auv5_eval/eval/{images,labels}/          ← 源域(B)图与标签
    runs/prov/provenance_auv5_eval.csv            ← 供 map_pose_dataset.py 使用
    data/mapped/raw/AUV_5gate/frame_*.jpg         ← 原始帧缓存（供三域渲染复用）
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "1_prepare"))

import cv2  # noqa: E402

SIZE = 640


def main():
    recs = [json.loads(l) for l in
            (PROJECT_ROOT / "experiment/runs/exp_distortion/e1_labels_B.jsonl").read_text().splitlines()]
    auv5 = [r for r in recs if "gate-" in r["src"]]
    print(f"AUV_5 (gate-*) 已标帧 {len(auv5)} 张")

    ds = PROJECT_ROOT / "runs/auv5_eval/eval"
    (ds / "images").mkdir(parents=True, exist_ok=True)
    (ds / "labels").mkdir(parents=True, exist_ok=True)
    rawdir = PROJECT_ROOT / "data/mapped/raw/AUV_5gate"
    rawdir.mkdir(parents=True, exist_ok=True)

    rows, n_ok = [], 0
    for i, r in enumerate(auv5):
        m = r["src"]                      # e1_picked__gate-5__frame_000058.jpg
        parts = m.split("__")
        gate, fname = parts[-2], parts[-1]
        num = int(fname.split("_")[1].split(".")[0])
        src_img = PROJECT_ROOT / "experiment/runs/exp_distortion/e1_picked" / "__".join(parts[1:])
        raw_img = PROJECT_ROOT / "data/AUV_5/raw-data" / gate / fname
        if not src_img.exists() or not raw_img.exists():
            print(f"  ⚠️ 缺文件，跳过 {m}")
            continue
        idx = 50000 + i                       # 合成唯一序号（gate 目录间帧号会撞）
        im = cv2.imread(str(raw_img))
        if im is None:
            continue
        cv2.imwrite(str(rawdir / f"frame_{idx:06d}.jpg"), im,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        name = f"auv5_{gate.replace('-', '')}_{num:06d}"
        cv2.imwrite(str(ds / "images" / f"{name}.jpg"),
                    cv2.imread(str(src_img)), [cv2.IMWRITE_JPEG_QUALITY, 95])
        d = r["dets"][0]
        k = d["kpts"]
        x1, y1, x2, y2 = d["bbox"]
        parts_l = ["0", f"{(x1+x2)/2/SIZE:.6f}", f"{(y1+y2)/2/SIZE:.6f}",
                   f"{(x2-x1)/SIZE:.6f}", f"{(y2-y1)/SIZE:.6f}"]
        for (x, y) in k:
            parts_l += [f"{x/SIZE:.6f}", f"{y/SIZE:.6f}", "2"]
        (ds / "labels" / f"{name}.txt").write_text(" ".join(parts_l) + "\n")
        rows.append(dict(img=f"{name}.jpg", split="eval", num_label=idx,
                         chain="nound", src="AUV_5gate", num_src=idx,
                         mad128=0.0, d48=0, accept=1))
        n_ok += 1

    prov = PROJECT_ROOT / "runs/prov/provenance_auv5_eval.csv"
    with open(prov, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"写出 {n_ok} 张 → {ds.relative_to(PROJECT_ROOT)}")
    print(f"provenance → {prov.relative_to(PROJECT_ROOT)}")
    print("下一步：")
    for dom in ("B", "C", "D"):
        print(f"  python scripts/1_prepare/pose/map_pose_dataset.py --prov {prov.relative_to(PROJECT_ROOT)} "
              f"--dataset runs/auv5_eval --domain {dom} --out runs/auv5_eval/pose_{dom}")


if __name__ == "__main__":
    main()
