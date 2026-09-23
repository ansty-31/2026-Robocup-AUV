#!/usr/bin/env python3
"""
标注质量审计：用 GT 角点自身的 PnP 重投影 RMS 找出可能标错的帧

原理
----
门框是已知尺寸（0.77 x 0.56 m）的平面矩形。把 GT 的 4 个角点拿去解 PnP，
再重投影回图像，若标注正确则 RMS 很小（正确标注实测 ~1~10 px）；
标错的帧（角点点错位置、顺序错、或门框其实不是那个四边形）RMS 会显著偏大。

注意：畸变模型不完美也会抬高 RMS，所以判据用**同一批数据内的分布**（分位数），
不要用绝对阈值。且本审计只标"可疑"，不自动删——由人复核。

用法：
    python experiment/scripts/exp_distortion/audit_labels.py \
        --dataset experiment/data/pose_B --domain B --out experiment/runs/domain/label_audit_B.csv
    # 出可疑帧的拼图供人工复核
    python experiment/scripts/exp_distortion/audit_labels.py --dataset experiment/data/pose_B \
        --domain B --sheet experiment/runs/domain/label_audit_sheet.jpg
    # AUV_5 评估集用的是 eval split
    python experiment/scripts/exp_distortion/audit_labels.py --dataset runs/auv5_eval/pose_B \
        --domain B --splits eval
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "1_prepare"))
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "scripts" / "exp_distortion"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "1_prepare" / "pose"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from eval_domain_arms import Cam, implied_wh, obj_points, CALIB_A, CALIB_C  # noqa: E402
from map_pose_dataset import read_pose_label  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="experiment/data/pose_B")
    ap.add_argument("--domain", default="B", choices=["B", "C", "D", "AOLD"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--splits", default="train,valid,test",
                    help="逗号分隔；AUV_5 评估集用 eval")
    a = ap.parse_args()

    ds = PROJECT_ROOT / a.dataset
    calib = CALIB_A if a.domain == "AOLD" else (None if a.domain == "B" else CALIB_C)
    cam = Cam(a.domain, calib or CALIB_C)
    if a.domain == "B":
        cam = Cam("B", CALIB_C)

    rows = []
    for split in a.splits.split(","):
        for f in sorted((ds / split / "labels").glob("*.txt")):
            lab = read_pose_label(f)
            if lab is None:
                continue
            k, box, vis = lab
            if (vis > 0).sum() != 4:            # 只审计四角都可见的帧
                continue
            p = implied_wh(*cam.pnp_pts(k), obj_points())
            if p is None:
                continue
            W, H, rms, dep = p
            rows.append(dict(img=f.stem + ".jpg", split=split, rms=round(rms, 2),
                             W=round(W, 4), H=round(H, 4), depth=round(dep, 3),
                             dW=round(abs(W - 0.77) / 0.77, 4),
                             dH=round(abs(H - 0.56) / 0.56, 4)))
    if not rows:
        print("没有可审计的帧（四角可见者）")
        return
    rows.sort(key=lambda r: -r["rms"])
    rms = np.array([r["rms"] for r in rows])
    print(f"审计 {ds.name}（域 {a.domain}）：四角可见帧 {len(rows)}")
    print(f"  GT 重投影 RMS：中位 {np.median(rms):.2f}  p90 {np.percentile(rms,90):.2f}  "
          f"p99 {np.percentile(rms,99):.2f}  最大 {rms.max():.2f} px")
    for th in (20, 30, 50):
        n = int((rms > th).sum())
        print(f"  RMS > {th} px 的帧: {n}（{100*n/len(rows):.1f}%）")
    print(f"\n最可疑的 {a.top} 帧：")
    for r in rows[:a.top]:
        print(f"  {r['split']:<5} {r['img'][:44]:<46} rms={r['rms']:7.2f}  "
              f"W={r['W']:.3f} H={r['H']:.3f} 门距={r['depth']:.2f}")

    if a.out:
        p = PROJECT_ROOT / a.out
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n→ {p.relative_to(PROJECT_ROOT)}")

    if a.sheet:
        pan = []
        for r in rows[:a.top]:
            im = cv2.imread(str(ds / r["split"] / "images" / r["img"]))
            if im is None:
                continue
            lab = read_pose_label(ds / r["split"] / "labels" / (Path(r["img"]).stem + ".txt"))
            k = lab[0].astype(int)
            cv2.polylines(im, [k], True, (0, 255, 0), 2)
            for (x, y) in k:
                cv2.circle(im, (x, y), 4, (0, 0, 255), -1)
            cv2.rectangle(im, (0, 0), (240, 26), (0, 0, 0), -1)
            cv2.putText(im, f"rms={r['rms']:.1f}", (6, 19), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 255), 2)
            pan.append(cv2.resize(im, (300, 300)))
        if pan:
            while len(pan) % 4:
                pan.append(np.zeros_like(pan[0]))
            grid = np.vstack([np.hstack(pan[i:i + 4]) for i in range(0, len(pan), 4)])
            cv2.imwrite(str(PROJECT_ROOT / a.sheet), grid,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"→ {a.sheet}（最可疑 {len(pan)} 帧拼图）")


if __name__ == "__main__":
    main()
