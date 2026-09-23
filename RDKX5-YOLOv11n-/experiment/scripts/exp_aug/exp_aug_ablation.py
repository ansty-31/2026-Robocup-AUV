#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_aug_ablation.py — 实验：**增强方式**是否是"门角系统性内缩"的成因

背景（见 runs/auv5/REPORT_auv5_pose.md §11.5）：
    模型四边形 / GT 标签的面积比中位 fp32 0.960、int8 0.954（同域、四角可见帧）
    ⇒ 模型相对自己的标签系统性**内缩 ~2% 线性**，且与量化无关。
    本实验要分清这 2% 是"增强方式"造成的，还是"数据分布/不确定性"造成的。

设计（受控 A/B）：
    同一份数据（data/AUV_5/gate-pose.yolov8）、同一起点（coco.pt，中性、无门先验）、
    同一 seed / epochs / batch / imgsz，**只改增强**：
      armA = ultralytics 默认（mosaic=1.0, scale=0.5, translate=0.1, erasing=0.4）
      armB = 收缩诱导型增强关掉/收窄（mosaic=0, scale=0.2, translate=0.05, erasing=0）
    训练后对两臂 + 参考模型（当前 auv5 权重）用**同一指标**评分：
      · Pose mAP50-95（val split）
      · 四边形面积比（预测 / GT 标签，同域 640，四角全可见帧，中心最近配对）
      · 按水质分层（清水/中水/浊水，按文件名前缀）

判据：
    armB 的面积比中位明显回升到 ~1.00 而 armA ≈0.96  ⇒ **增强是成因** ⇒ 改增强即可，不必补标
    两臂都 ≈0.96                                    ⇒ 增强不是主因 ⇒ 需按"近距/中浊"轴补标

用法：
    python experiment/scripts/exp_aug/exp_aug_ablation.py                 # 全流程
    python experiment/scripts/exp_aug/exp_aug_ablation.py --eval-only     # 只评估（跳过训练）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT = PROJECT_ROOT / "experiment" / "runs" / "aug_ablation"
DATA = PROJECT_ROOT / "runs" / "auv5" / "eval" / "gate_pose_abs.yaml"

WATER = {"AUV_5": "清水", "AUV_4": "中水", "AUV_1": "中水", "AUV_2": "浊水", "AUV_3": "浊水"}

ARMS = {
    "armA_default": dict(mosaic=1.0, scale=0.5, translate=0.1, erasing=0.4, close_mosaic=10),
    "armB_reduced": dict(mosaic=0.0, scale=0.2, translate=0.05, erasing=0.0, close_mosaic=10),
}


def qarea(q):
    """四边形面积（鞋带公式）"""
    import numpy as np
    x, y = q[:, 0], q[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def eval_model(weights: Path, tag: str, split: str = "test") -> dict:
    """返回 {mAP, 面积比统计, 分水质}"""
    import numpy as np
    from ultralytics import YOLO

    m = YOLO(str(weights))
    r = m.val(data=str(DATA), split=split, imgsz=640, batch=4, device=0, verbose=False,
              project=str(OUT / "eval"), name=tag, plots=False, exist_ok=True)
    res = dict(tag=tag, weights=str(weights),
               pose_map50=round(float(r.pose.map50), 4), pose_map=round(float(r.pose.map), 4),
               box_map50=round(float(r.box.map50), 4))

    # 逐帧面积比（用 predict，才能拿到实例级角点）
    test_dir = PROJECT_ROOT / "data" / "AUV_5" / "gate-pose.yolov8" / split
    per_water, ratios = {}, []
    for img in sorted((test_dir / "images").glob("*.jpg")):
        lab = test_dir / "labels" / (img.stem + ".txt")
        if not lab.exists():
            continue
        gts = []
        for line in lab.read_text().split("\n"):
            p = line.split()
            if len(p) != 17:
                continue
            f = list(map(float, p))
            k4 = np.array(f[5:]).reshape(4, 3)
            if (k4[:, 2] > 0).sum() < 4:          # 只用四角全可见的 GT
                continue
            gts.append(k4[:, :2] * 640.0)
        if not gts:
            continue
        out = m.predict(str(img), imgsz=640, conf=0.25, verbose=False, device=0)[0]
        if out.keypoints is None or len(out.keypoints) == 0:
            continue
        bi = int(np.argmax(out.boxes.conf.cpu().numpy()))
        kp = out.keypoints.xy.cpu().numpy()[bi]
        cen = kp.mean(0)
        g = min(gts, key=lambda q: np.linalg.norm(cen - q.mean(0)))
        if np.linalg.norm(cen - g.mean(0)) > 150:   # 配对失败（双门/漏检）
            continue
        ratio = qarea(kp) / qarea(g)
        ratios.append(ratio)
        w = WATER.get(next((k for k in WATER if img.name.startswith(k)), None), "?")
        per_water.setdefault(w, []).append(ratio)

    A = np.array(ratios)
    if len(A):
        res.update(n=int(len(A)), ratio_median=round(float(np.median(A)), 4),
                   ratio_mean=round(float(A.mean()), 4),
                   linear_scale=round(float(np.sqrt(np.median(A))), 4),
                   shrink_frac=round(float((A < 1).mean()), 3),
                   per_water={k: dict(n=len(v), median=round(float(np.median(v)), 4))
                              for k, v in per_water.items()})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--start", default=str(PROJECT_ROOT / "weights" / "yolo11n-pose.coco.pt"))
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--amp", action="store_true", dest="amp", default=True,
                    help="AMP（默认开）；本机 GPU 与桌面共用，遇 CUDA 异常可 --no-amp 排除")
    ap.add_argument("--no-amp", action="store_false", dest="amp")
    ap.add_argument("--ref", default=str(PROJECT_ROOT / "weights" / "yolo11n-pose.pt"),
                    help="参考模型（当前 auv5 权重）")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if not DATA.exists():
        print(f"✗ 缺 data.yaml: {DATA}")
        return 1

    results = {}
    if not a.eval_only:
        from ultralytics import YOLO
        for arm, aug in ARMS.items():
            print(f"\n{'='*70}\n=== 训练 {arm}: {aug}\n{'='*70}", flush=True)
            run_dir = OUT / "train" / arm
            last = run_dir / "weights" / "last.pt"
            if last.exists():
                # 断点续训：上一次可能在 CUDA 异常/被抢占时中断（本机 GPU 与桌面共用）
                print(f"  检测到未完成的运行 {last} → 续训（resume=True）", flush=True)
                model = YOLO(str(last))
                model.train(resume=True)
            else:
                model = YOLO(a.start)
                model.train(data=str(DATA), epochs=a.epochs, imgsz=640, batch=4, workers=2,
                            cache="ram", device=0, project=str(OUT / "train"), name=arm,
                            seed=0, amp=a.amp, exist_ok=False, verbose=True, **aug)
            best = Path(model.trainer.save_dir) / "weights" / "best.pt"
            # 复刻 best.pt 到固定名，避免后续被覆盖
            dst = OUT / f"{arm}_best.pt"
            dst.write_bytes(best.read_bytes())
            print(f"  → {dst}", flush=True)
            results[arm] = eval_model(dst, arm)

    # 参考模型（若已训过两臂则一并评分，便于同表对比）
    ref = Path(a.ref)
    if ref.exists():
        results["ref_auv5_current"] = eval_model(ref, "ref_auv5_current")

    (OUT / "aug_ablation.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))

    # ---- 汇总 markdown ----
    lines = ["# 增强消融：门角内缩是否由增强造成", "",
             f"数据 `{DATA.name}`（625 张，same split）；起点 `{Path(a.start).name}`；"
             f"epochs {a.epochs}；seed 0；batch 4；**只改增强**", "",
             "| 模型 | 增强 | Pose mAP50-95 | 面积比中位 | 线性尺度 | 内缩帧占比 | 清水 | 中水 | 浊水 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for tag, r in results.items():
        aug = "默认(mosaic1.0/scale0.5/erasing0.4)" if tag == "armA_default" else (
            "收窄(mosaic0/scale0.2/erasing0)" if tag == "armB_reduced" else "—（当前 auv5）")
        pw = r.get("per_water", {})
        f = lambda k: (f"{pw[k]['median']:.3f}(n={pw[k]['n']})" if k in pw else "—")
        lines.append(f"| {tag} | {aug} | {r.get('pose_map','—')} | {r.get('ratio_median','—')} | "
                     f"{r.get('linear_scale','—')} | {r.get('shrink_frac','—')} | "
                     f"{f('清水')} | {f('中水')} | {f('浊水')} |")
    lines += ["", "判据：armB 面积比明显回升到 ~1.00 而 armA ≈0.96 ⇒ 增强是成因（改增强即可，不必补标）；",
              "两臂都 ≈0.96 ⇒ 增强不是主因 ⇒ 按「近距 / 中浊」轴补标。"]
    (OUT / "AUG_ABLATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n→ {OUT/'aug_ablation.json'} / {OUT/'AUG_ABLATION.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
