#!/usr/bin/env python3
"""
挑 AUV_5 清水「补标候选帧」，并把模型预测的门框四边形**画在图上**供人工核对

为什么这样做
------------
`tools/analyze/label_corners.py` 支持交互标注（含「吸附到红管」）与 `--resume` 去重，
但**不支持从已有记录预填角点**。所以最高效的补标工作流是：

    本脚本挑帧 + 把模型预测的四边形画上去  →  人工只需核对/微调（而不是从零找角点）
                                          →  label_corners.py --images ... --out ...

挑帧策略（避免又挑出"外圈难帧"这类有偏样本）：
  · 在 AUV_5 全部 gate-* 帧里按 --every 抽样，渲染到指定域后跑模型；
  · 只保留**模型有检出**（conf ≥ --conf）的帧作为候选（大概率真有门框）；
  · 按「检出框面积」三分位分层抽样（近/中/远都有），再按清晰度/曝光过滤；
  · 输出：候选图（带预测叠加）、candidates.csv、拼图、以及可直接复制的标注命令。

用法：
    python experiment/scripts/exp_distortion/pick_auv5_label_candidates.py \
        --weights experiment/runs/domain/A_B/weights/best.pt --domain B --every 3 --n 150
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "1_prepare"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "1_prepare" / "pose"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from map_pose_dataset import Domain  # noqa: E402


def rel(p):
    """项目内则给相对路径，否则给绝对路径（--out 允许给项目外）"""
    try:
        return p.relative_to(PROJECT_ROOT)
    except ValueError:
        return p

AOLD = "configs/backup/front_camera_AUV1_water_fx782.yaml"
CALIB_C = "configs/front_camera.yaml"


def calib_for(dom):
    return {"B": None, "C": CALIB_C, "D": CALIB_C, "AOLD": AOLD}[dom]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="experiment/runs/domain/A_B/weights/best.pt")
    ap.add_argument("--domain", default="D", choices=["B", "C", "D", "AOLD"])
    ap.add_argument("--enhance", default="wb", choices=["on", "wb", "off"],
                    help="on=旧(WB+CLAHE+gamma)；wb=新(WB+gamma,无 CLAHE)；off=不做")
    ap.add_argument("--every", type=int, default=3, help="每 N 帧取 1 帧")
    ap.add_argument("--n", type=int, default=150, help="最终候选张数")
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--out", default="data/AUV_5/label_candidates_Dwb")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    outdir = PROJECT_ROOT / a.out
    imgdir = outdir / "images"
    imgdir.mkdir(parents=True, exist_ok=True)

    # 排除已知损坏/全灰帧
    deleted = set()
    dl = PROJECT_ROOT / "data/AUV_5/raw-data/_deleted_gray_frames.txt"
    if dl.exists():
        deleted = {l.strip() for l in dl.read_text().splitlines() if l.strip()}
    pool = []
    for d in sorted((PROJECT_ROOT / "data/AUV_5/raw-data").glob("gate-*")):
        for p in sorted(d.glob("*.jpg")):
            if p.name not in deleted:
                pool.append(p)
    pool = pool[::max(a.every, 1)]
    print(f"清水 gate 池 {len(pool)} 帧（每 {a.every} 帧取 1），域={a.domain} enhance={a.enhance}")

    dom = Domain("C" if a.domain == "AOLD" else a.domain, calib_for(a.domain),
                 {"on": True, "wb": "wb", "off": False}[a.enhance])
    from ultralytics import YOLO
    model = YOLO(str(PROJECT_ROOT / a.weights))

    recs = []
    for i in range(0, len(pool), 32):
        chunk = pool[i:i + 32]
        imgs, meta = [], []
        for p in chunk:
            raw = cv2.imread(str(p))
            if raw is None:
                continue
            imgs.append(dom.render(raw))
            meta.append(p)
        if not imgs:
            continue
        res = model.predict(imgs, imgsz=640, conf=a.conf, verbose=False, device=0)
        for img, p, r in zip(imgs, meta, res):
            if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
                continue
            j = int(np.argmax(r.boxes.conf.cpu().numpy()))
            k = r.keypoints.xy.cpu().numpy()[j]
            pc = r.keypoints.conf.cpu().numpy()[j]
            b = r.boxes.xyxy.cpu().numpy()[j]
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lap = float(cv2.Laplacian(g, cv2.CV_32F).var())
            recs.append(dict(path=p, img=img, k=k, kconf=pc, conf=float(r.boxes.conf[j]),
                             area=float((b[2] - b[0]) * (b[3] - b[1])),
                             bright=float(g.mean()), sharp=lap,
                             nconf4=int((pc >= 0.5).sum())))
    print(f"有检出 {len(recs)} 帧")
    if not recs:
        print("没有候选，退出")
        return
    # 清晰度/曝光过滤（与 E1 挑帧同口径）
    keeps = [r for r in recs if r["sharp"] >= 30 and 25 <= r["bright"] <= 210]
    print(f"过滤清晰度/曝光后 {len(keeps)} 帧")
    # 按面积三分位分层，等量抽取
    keeps.sort(key=lambda r: r["area"])
    parts = np.array_split(np.array(keeps, dtype=object), 3)
    random.seed(a.seed)
    per = max(a.n // 3, 1)
    picks = []
    for part in parts:
        # 分层内优先取模型最有把握的帧（4 角齐全 > conf），让初始叠加尽量完整
        ordered = sorted(part, key=lambda r: (r["nconf4"], r["conf"]), reverse=True)
        picks += ordered[:per]
    picks = picks[:a.n]

    rows = []
    for r in picks:
        name = f"{r['path'].parent.name}_{r['path'].name}"
        k = r["k"]
        vis = r["img"].copy()
        cv2.polylines(vis, [k.astype(np.int32)], True, (0, 255, 0), 2)      # 模型预测=绿
        for (x, y), c in zip(k, r["kconf"]):
            col = (0, 0, 255) if c >= 0.5 else (0, 165, 255)
            cv2.circle(vis, (int(x), int(y)), 4, col, -1)
        strip = vis[0:46, 0:430].copy()
        cv2.rectangle(strip, (0, 0), (430, 46), (0, 0, 0), -1)
        vis[0:46, 0:430] = cv2.addWeighted(vis[0:46, 0:430], 0.35, strip, 0.65, 0)
        cv2.putText(vis, f"MODEL GUESS conf{r['conf']:.2f} 4k{r['nconf4']}",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(vis, "verify & correct in label_corners.py",
                    (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.imwrite(str(imgdir / f"{name}"), vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append(dict(name=name, src=str(r["path"].relative_to(PROJECT_ROOT)),
                         conf=round(r["conf"], 3), n_kpt_conf_ge05=r["nconf4"],
                         area=int(r["area"]), bright=round(r["bright"], 1),
                         sharp=round(r["sharp"], 1),
                         kpts=";".join(f"{x:.1f},{y:.1f}" for x, y in k)))

    with open(outdir / "candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 拼图（前 12 张）
    pan = []
    for r in rows[:12]:
        im = cv2.imread(str(imgdir / r["name"]))
        pan.append(cv2.resize(im, (300, 300)))
    while len(pan) % 4:
        pan.append(np.zeros_like(pan[0]))
    cv2.imwrite(str(outdir / "sheet.jpg"),
                np.vstack([np.hstack(pan[i:i + 4]) for i in range(0, len(pan), 4)]),
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    n4 = sum(1 for r in rows if r["n_kpt_conf_ge05"] == 4)
    print(f"\n候选 {len(rows)} 张（面积三分位各 ~{per}）→ {rel(imgdir)}")
    print(f"  其中模型给出 4 个角(conf≥0.5)的 {n4} 张（{100*n4/len(rows):.0f}%）")
    print(f"  拼图预览: {rel(outdir/'sheet.jpg')}")
    print(f"  清单:     {rel(outdir/'candidates.csv')}")
    print("\n下一步（人工核对/微调模型预测，不要盲信）：")
    print(f"  cd /home/ansty/RDKX5/auv_vision && "
          f"/home/ansty/anaconda3/envs/yolov8/bin/python auv_vision/tools/analyze/label_corners.py \\")
    print(f"      --images {imgdir} --out runs/auv5_labeled.jsonl --resume")


if __name__ == "__main__":
    main()
