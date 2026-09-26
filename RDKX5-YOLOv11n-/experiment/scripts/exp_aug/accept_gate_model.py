#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""accept_gate_model.py — 门 pose 模型的**验收**（新版权重的统一判据）

为什么需要：2026-09-23 的 auv5 事件说明"mAP 好看"并不够 —— 它 val/test mAP 都不低，
但四边形相对标签**系统性内缩 2–4%**（近距离肉眼可见"角点偏中间"），根因是起点权重带偏。
所以每版新权重必须同时过**两条**判据：

  ① **精度**：Pose mAP50-95（val 与 test）不低于基线（默认 0.9242 = auv5 的 test 值）
  ② **口径**：四边形面积比中位 = 预测/GT，需落在 [0.98, 1.02]
     （<0.98 = 内缩，>1.02 = 外扩；auv5 是 0.960–0.977，从 coco 起步的 armA 是 1.022）

用法（在仓库根）：
    python experiment/scripts/exp_aug/accept_gate_model.py --weights weights/yolo11n-pose.pt
    python experiment/scripts/exp_aug/accept_gate_model.py --weights <新权重> \
        --min-map 0.9242 --json experiment/runs/aug_ablation/accept.json

输出：控制台一张表 + 退出码（0=两条都过，1=未过）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WATER = {"AUV_5": "清水", "AUV_4": "中水", "AUV_1": "中水", "AUV_2": "浊水", "AUV_3": "浊水"}


def qarea(q):
    import numpy as np
    x, y = q[:, 0], q[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def read_gt(lab: Path):
    """返回四角全可见的实例列表（640 像素空间）"""
    import numpy as np
    out = []
    if not lab.exists():
        return out
    for line in lab.read_text().split("\n"):
        p = line.split()
        if len(p) != 17:
            continue
        f = list(map(float, p))
        k4 = np.array(f[5:]).reshape(4, 3)
        if (k4[:, 2] > 0).sum() < 4:
            continue
        out.append(k4[:, :2] * 640.0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default=str(PROJECT_ROOT / "runs" / "auv5" / "eval" / "gate_pose_abs.yaml"))
    ap.add_argument("--ratio-lo", type=float, default=0.98)
    ap.add_argument("--ratio-hi", type=float, default=1.02)
    ap.add_argument("--min-map", type=float, default=0.9242, help="test split Pose mAP50-95 下限。⚠️ 默认 0.9242 是 auv5 在【重划分之前】那个有泄漏的 split 上测的，只能当参考；跨 split 比较请用同一 split 重测的权重（如 armA）")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    import numpy as np
    from ultralytics import YOLO

    w = Path(a.weights)
    if not w.exists():
        print(f"✗ 权重不存在：{w}")
        return 1
    m = YOLO(str(w))
    res = {"weights": str(w)}

    # ---- ① 精度 ----
    for split in ("val", "test"):
        r = m.val(data=a.data, split=split, imgsz=640, batch=4, device=0, verbose=False,
                  project=str(PROJECT_ROOT / "experiment" / "runs" / "aug_ablation" / "accept"),
                  name=f"{w.stem}_{split}", plots=False, exist_ok=True)
        res[f"{split}_map50"] = round(float(r.pose.map50), 4)
        res[f"{split}_map"] = round(float(r.pose.map), 4)

    # ---- ② 口径（面积比） ----
    test_dir = PROJECT_ROOT / "data" / "datasets" / "AUV_5_gate-pose.yolov8" / "test"
    ratios, per_water, n4 = [], {}, 0
    for img in sorted((test_dir / "images").glob("*.jpg")):
        gts = read_gt(test_dir / "labels" / (img.stem + ".txt"))
        if not gts:
            continue
        n4 += 1
        out = m.predict(str(img), imgsz=640, conf=a.conf, verbose=False, device=0)[0]
        if out.keypoints is None or out.boxes is None or len(out.boxes) == 0:
            continue
        kp = np.asarray(out.keypoints.xy.cpu().numpy()[int(np.argmax(out.boxes.conf.cpu().numpy()))], float)
        g = min(gts, key=lambda q: np.linalg.norm(kp.mean(0) - q.mean(0)))
        if np.linalg.norm(kp.mean(0) - g.mean(0)) > 150:
            continue
        ratio = qarea(kp) / qarea(g)
        ratios.append(ratio)
        wat = WATER.get(next((k for k in WATER if img.name.startswith(k)), None), "?")
        per_water.setdefault(wat, []).append(ratio)

    if not ratios:
        print("✗ 没有可算面积比的样本")
        return 1
    A = np.array(ratios)
    res.update(n_ratio=int(len(A)), ratio_median=round(float(np.median(A)), 4),
               ratio_linear=round(float(np.sqrt(np.median(A))), 4),
               shrink_frac=round(float((A < 1).mean()), 3),
               per_water={k: dict(n=len(v), median=round(float(np.median(v)), 4))
                          for k, v in per_water.items()},
               n_test_4corner=n4)

    ok_map = res["test_map"] >= a.min_map
    ok_ratio = a.ratio_lo <= res["ratio_median"] <= a.ratio_hi

    print("\n" + "=" * 74)
    print(f"验收：{w.name}")
    print("=" * 74)
    print(f"  ① 精度  Pose mAP50-95  val {res['val_map']:.4f} / test {res['test_map']:.4f} "
          f"(下限 {a.min_map})  → {'PASS' if ok_map else 'FAIL'}")
    print(f"  ② 口径  面积比中位 {res['ratio_median']:.4f}（线性 {res['ratio_linear']:.4f}，"
          f"内缩帧 {res['shrink_frac']*100:.0f}%）"
          f"  目标 [{a.ratio_lo}, {a.ratio_hi}]  → {'PASS' if ok_ratio else 'FAIL'}")
    print(f"     分水质：" + "  ".join(f"{k} {v['median']:.3f}(n={v['n']})" for k, v in res["per_water"].items()))
    print(f"     （test 四角可见帧 {res['n_test_4corner']}，参与面积比 {res['n_ratio']}）")
    print("=" * 74)
    print("✅ 两条都过 → 可作为当前模型" if (ok_map and ok_ratio)
          else "❌ 未通过 → 先别覆盖 canonical 槽位，把这张表发出来一起看")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"→ {a.json}")
    return 0 if (ok_map and ok_ratio) else 1


if __name__ == "__main__":
    sys.exit(main())
