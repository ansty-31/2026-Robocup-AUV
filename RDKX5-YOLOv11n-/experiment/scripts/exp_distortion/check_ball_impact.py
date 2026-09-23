#!/usr/bin/env python3
"""
预处理换方案对**球（检测模型）**的影响检查

背景：板端预处理要从「C 域（720p 去畸变）+ 完整 enhance（CLAHE 0.5）」
换成「D 域（640 去畸变）+ 只 WB+gamma（无 CLAHE）」。pose 模型重训即可，
但**检测模型（blue_ball/gate/red_ball）也在吃同一条预处理链路**，必须确认它不受影响。

做法（无 GT 的 A/B，同一批裸流帧、同一个权重）：
  1. 把同一批原始帧分别按 旧链路 与 新链路 渲染；
  2. 同一个检测权重分别推理；
  3. 比较：每类检出数、置信度分布、框一致率（IoU≥0.5 的匹配比例）、
     帧级"检出≥1 个该类"的一致率、以及两链路的图片像素差。

判据（经验）：
  · 每类检出数变化 < 5%、置信度中位变化 < 0.02、帧级一致率 > 95% ⇒ 判"无影响"；
  · 任一类检出数掉 > 10% 或帧级一致率 < 90% ⇒ 判"有影响"，需要重训检测模型。

用法：
    python experiment/scripts/exp_distortion/check_ball_impact.py --n-per-dive 40
    # 指定权重/链路
    ... --weights weights/yolo11n.pt --old C --new D --new-enhance wb
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "1_prepare"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "1_prepare" / "pose"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from map_pose_dataset import Domain  # noqa: E402

CALIB = "configs/front_camera.yaml"
CLASSES = ["blue_ball", "gate", "red_ball"]


def make(dom_name, enhance):
    calib = None if dom_name == "B" else CALIB
    e = {"on": True, "wb": "wb", "off": False}[enhance]
    return Domain("C" if dom_name == "AOLD" else dom_name, calib, e)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="weights/yolo11n.pt")
    ap.add_argument("--old", default="C", help="旧链路域（板端当前 = C + 完整 enhance）")
    ap.add_argument("--old-enhance", default="on")
    ap.add_argument("--new", default="D", help="新链路域（= D + wb）")
    ap.add_argument("--new-enhance", default="wb")
    ap.add_argument("--n-per-dive", type=int, default=40)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    raw_root = PROJECT_ROOT / "data/mapped/raw"
    rng = np.random.default_rng(a.seed)
    frames = []
    for d in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        fs = sorted(d.glob("*.jpg"))
        if fs:
            idx = rng.choice(len(fs), size=min(a.n_per_dive, len(fs)), replace=False)
            frames += [fs[i] for i in sorted(idx)]
    print(f"抽样 {len(frames)} 帧（来源 {[p.name for p in sorted(raw_root.iterdir()) if p.is_dir()]}）")

    from ultralytics import YOLO
    model = YOLO(str(PROJECT_ROOT / a.weights))
    dom_old, dom_new = make(a.old, a.old_enhance), make(a.new, a.new_enhance)

    res = {"old": [], "new": []}
    px_diff = []
    for tag, dom in (("old", dom_old), ("new", dom_new)):
        B = 32
        imgs = []
        for f in frames:
            raw = cv2.imread(str(f))
            if raw is None:
                continue
            imgs.append((f, dom.render(raw)))
        if tag == "old":
            old_imgs = [im for _, im in imgs]
        else:
            for (f, im), imo in zip(imgs, old_imgs):
                px_diff.append(float(np.abs(im.astype(np.int16) - imo.astype(np.int16)).mean()))
        for i in range(0, len(imgs), B):
            chunk = imgs[i:i + B]
            outs = model.predict([im for _, im in chunk], imgsz=640, conf=a.conf,
                                 verbose=False, device=0)
            for (f, _), r in zip(chunk, outs):
                dets = []
                if r.boxes is not None and len(r.boxes):
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    cls = r.boxes.cls.cpu().numpy().astype(int)
                    cf = r.boxes.conf.cpu().numpy()
                    dets = [(int(c), float(s), b) for c, s, b in zip(cls, cf, xyxy)]
                res[tag].append((f.name, dets))

    print(f"\n两链路图片像素差：均值 {np.mean(px_diff):.2f} /255（新 vs 旧）")
    hdr = f"{'类别':<12}{'旧检出':>8}{'新检出':>8}{'变化':>9}{'旧conf中位':>11}{'新conf中位':>11}{'框一致率':>10}{'帧级一致':>10}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for ci, cname in enumerate(CLASSES):
        n_old = sum(1 for _, d in res["old"] if any(c == ci for c, _, _ in d))
        n_new = sum(1 for _, d in res["new"] if any(c == ci for c, _, _ in d))
        co = [s for _, d in res["old"] for c, s, _ in d if c == ci]
        cn = [s for _, d in res["new"] for c, s, _ in d if c == ci]
        agree_box, tot = 0, 0
        agree_frame = 0
        for (nm, do), (_, dn) in zip(res["old"], res["new"]):
            bo = [b for c, _, b in do if c == ci]
            bn = [b for c, _, b in dn if c == ci]
            if bo and bn:
                tot += 1
                if max(iou(x, y) for x in bo for y in bn) >= 0.5:
                    agree_box += 1
            if bool(bo) == bool(bn):
                agree_frame += 1
        delta = (n_new - n_old) / max(n_old, 1) * 100
        print(f"{cname:<12}{n_old:>8}{n_new:>8}{delta:>8.1f}%"
              f"{(np.median(co) if co else float('nan')):>11.3f}"
              f"{(np.median(cn) if cn else float('nan')):>11.3f}"
              f"{(agree_box/max(tot,1)*100 if tot else float('nan')):>9.0f}%"
              f"{agree_frame/len(res['old'])*100:>9.0f}%")
    print("\n判据：变化<5% / conf 中位差<0.02 / 帧级一致>95% ⇒ 无影响；"
          "掉>10% 或一致<90% ⇒ 有影响，需重训检测模型")


if __name__ == "__main__":
    main()
