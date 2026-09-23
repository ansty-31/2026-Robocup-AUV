#!/usr/bin/env python3
"""
清水（AUV_5）检出率探针：无 GT，只看模型在真实清水帧上的表现

E1 挑的 16 张是**故意挑的外圈难帧**，用它判断"清水能不能用"会有偏。
本脚本在 AUV_5 的 gate-* 裸帧里**随机抽样**，渲染到指定域后跑模型，统计：

  · 有检出帧比例（conf ≥ 阈值）
  · 四角齐全帧比例（kpt conf ≥ 0.5 且非 (0,0) 占位）—— 板端 PnP 必须要 4 个角
  · 检出框面积中位（判断是不是把小目标/杂物当成门）

用法：
    python experiment/scripts/exp_distortion/probe_auv5_detect.py \
        --weights weights/domain_B.pt --domain B --n 120
    python experiment/scripts/exp_distortion/probe_auv5_detect.py \
        --weights weights/yolo11n-pose.pt --domain AOLD --n 120 \
        --render-dir experiment/runs/domain/auv5_probe
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "1_prepare"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "1_prepare" / "pose"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from map_pose_dataset import Domain, SRC_PATH  # noqa: E402

AOLD = "configs/backup/front_camera_AUV1_water_fx782.yaml"
CALIB_C = "configs/front_camera.yaml"


def calib_for(domain: str) -> str | None:
    if domain == "B":
        return None
    if domain == "AOLD":
        return AOLD
    return CALIB_C


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--domain", default="B", choices=["B", "C", "D", "AOLD"])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render-dir", default=None,
                    help="给定时把抽样帧渲染结果落盘（供目视）")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    deleted = set()
    dl = PROJECT_ROOT / "data/AUV_5/raw-data/_deleted_gray_frames.txt"
    if dl.exists():
        for line in dl.read_text().splitlines():
            if line.strip():
                deleted.add(line.strip())
    pool = []
    for d in sorted((PROJECT_ROOT / "data/AUV_5/raw-data").glob("gate-*")):
        for p in sorted(d.glob("*.jpg")):
            if p.name not in deleted:
                pool.append(p)
    random.seed(a.seed)
    picks = random.sample(pool, min(a.n, len(pool)))
    print(f"清水 gate 帧池 {len(pool)} 张，抽样 {len(picks)} 张；域={a.domain}，权重={a.weights}")

    # A-old 域 = P1 链路 + 标定 A（Domain 里用 "C" 形态表示 P1）
    dom = Domain("C" if a.domain == "AOLD" else a.domain, calib_for(a.domain), True)
    out_dir = Path(PROJECT_ROOT / a.render_dir) if a.render_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(str(PROJECT_ROOT / a.weights))

    res_rows = []
    batch_imgs, batch_paths = [], []
    for i in range(0, len(picks), 32):
        chunk = picks[i:i + 32]
        imgs, names = [], []
        for p in chunk:
            raw = cv2.imread(str(p))
            if raw is None:
                continue
            img = dom.render(raw)
            imgs.append(img)
            names.append(p)
            if out_dir:
                cv2.imwrite(str(out_dir / f"{a.domain}_{p.parent.name}_{p.name}"), img,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not imgs:
            continue
        out = model.predict(imgs, imgsz=640, conf=a.conf, verbose=False, device=0)
        for p, r in zip(names, out):
            nbox = 0 if r.boxes is None else len(r.boxes)
            best4 = 0
            area = 0.0
            bconf = 0.0
            if nbox:
                j = int(np.argmax(r.boxes.conf.cpu().numpy()))
                bconf = float(r.boxes.conf.cpu().numpy()[j])
                b = r.boxes.xyxy.cpu().numpy()[j]
                area = float((b[2] - b[0]) * (b[3] - b[1]))
                try:
                    pc = r.keypoints.conf.cpu().numpy()[j]
                    pk = r.keypoints.xy.cpu().numpy()[j]
                    best4 = int(((pc >= 0.5) & ~((pk[:, 0] < 0.5) & (pk[:, 1] < 0.5))).sum() == 4)
                except Exception:
                    best4 = 0
            res_rows.append(dict(img=f"{p.parent.name}/{p.name}", nbox=nbox, conf=bconf,
                                 area=area, all4=best4))

    n = len(res_rows)
    det = [r for r in res_rows if r["nbox"] > 0]
    a4 = [r for r in res_rows if r["all4"]]
    tag = a.tag or f"{Path(a.weights).stem}@{a.domain}"
    print(f"\n{'方案':<28}{'抽样':>6}{'有检出':>10}{'四角齐全':>10}{'面积中位(px²)':>14}{'最高conf中位':>13}")
    def pct(k, d=n):
        return f"{k}/{d} ({100*k/max(d,1):.0f}%)"
    print(f"{tag:<28}{n:>6}{pct(len(det)):>10}{pct(len(a4)):>10}"
          f"{(np.median([r['area'] for r in det]) if det else 0):>14.0f}"
          f"{(np.median([r['conf'] for r in det]) if det else 0):>13.2f}")
    import json
    outj = PROJECT_ROOT / "experiment/runs/domain" / f"auv5_probe_{tag.replace('@','_').replace('/','_')}.json"
    outj.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(tag=tag, n=n, n_det=len(det), n_all4=len(a4), rows=res_rows),
              open(outj, "w"), indent=1)
    print(f"→ {outj.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
