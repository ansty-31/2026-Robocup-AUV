#!/usr/bin/env python3
"""
畸变实验 · 把一套 YOLO-pose 标注换算到另一个组的坐标域（标 1 套 → 推 3 套）

## 为什么能这么干

三组的差别**只是已知的坐标变换**，不是"另一个场景"：

    A ↔ B ↔ C  都经过 distortion_geometry.FrameGeometry.transform() 精确互转
    （A/C 是同一套几何、只差重采样顺序；B 是畸变域）

所以只要在一套图上人工标好 4 个角点，另外两套的标注可以由坐标换算得到，
**标注量减到 1/3**，而且三套标注在几何上自洽——实验测出来的差异只可能来自模型，
不会混进"人三次标注不一致"的噪声。

> 反过来，如果你想让三组各自独立人工标注（测的是"人眼在哪种几何下标得更一致"），
> 那就不用本脚本，直接三套标 —— 两种口径别混着用。

## 推荐流程

1. `prepare_distortion_groups.py` 生成三组图；
2. **只把 A 组（去畸变图，直线是直的，角点最好对齐）传 RoboFlow 标注**，
   1 类 `gate`、4 点、顺序 `TL,TR,BR,BL`，导出 **YOLOv8 Pose**；
3. 用本脚本把 A 的标签推到 B 和 C；
4. 三套标签各自组数据集 → 训练/评估（见 `compare_kpt_spacing.py`）。

    # A 的岩点标签（RoboFlow 导出目录里的 labels/）→ B
    python experiment/scripts/exp_distortion/transform_pose_labels.py \
        --experiment data/exp_distortion/processed \
        --labels-in  /path/to/rf_export/train/labels --roboflow \
        --src-group A --dst-group B \
        --labels-out data/exp_distortion/labels_B

## ⚠️ 两类失效点（脚本会显式统计，不许当坐标用）

1. 畸变模型不可逆：B 域里超出可逆半径的点（本工程标定 ≈ 520px，占画面 36%）；
2. 目标点落在目标画面之外（A/C 的整流图只覆盖原始画面中央）。

命中的关键点写 `v=0`（YOLO-pose 的"未标注"），并把坐标夹回画面内；
命中率会在报告里逐组打印。**如果某组关键点无效比例很高，这一组的结论就不成立**，
要么缩小到可信区域重做，要么先修标定（见 `calib_diagnose.py`）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from distortion_geometry import (Geometry, ALL_GROUPS, GROUP_A, GROUP_B,  # noqa: E402
                                 GROUP_C, GROUP_DESC, resolve_group)

RF_SUFFIX = re.compile(r"(_jpg|_jpeg|_png|_bmp)?(\.rf\.[A-Za-z0-9]+)?$", re.I)


def load_experiment(root: Path) -> dict:
    p = root / "experiment.json"
    if not p.exists():
        sys.exit(f"❌ 找不到清单 {p}（先用 prepare_distortion_groups.py 生成）")
    return json.loads(p.read_text(encoding="utf-8"))


def normalize_stem(stem: str) -> str:
    """`frame_0001_jpg.rf.AbC123` -> `frame_0001`（RoboFlow 会改名）"""
    s = RF_SUFFIX.sub("", stem)
    return s


def parse_line(line: str, size: int):
    v = line.split()
    if len(v) < 5:
        return None
    cls = int(float(v[0]))
    cx, cy, w, h = (float(x) * size for x in v[1:5])
    rest = v[5:]
    if len(rest) % 3 != 0:
        return None
    kpts = np.array([[float(rest[i]), float(rest[i + 1])]
                     for i in range(0, len(rest), 3)], np.float64) * size
    vis = np.array([float(rest[i + 2]) for i in range(0, len(rest), 3)], np.float64)
    return cls, np.array([cx, cy, w, h]), kpts, vis


def fmt_line(cls, box, kpts, vis, size) -> str:
    out = [str(cls)]
    out += [f"{x / size:.8f}" for x in box]
    for (x, y), v in zip(kpts, vis):
        out += [f"{max(0.0, min(size, x)) / size:.8f}",
                f"{max(0.0, min(size, y)) / size:.8f}", str(int(round(v)))]
    return " ".join(out)


def box_perimeter(box, size, per_edge=12):
    """把框的四条边密采样（去畸变是非线性映射，只转 4 个角会漏掉边的外凸）"""
    cx, cy, w, h = box
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - h / 2, cy + h / 2
    t = np.linspace(0, 1, per_edge)
    pts = []
    pts.append(np.stack([x0 + (x1 - x0) * t, np.full_like(t, y0)], axis=1))
    pts.append(np.stack([x0 + (x1 - x0) * t, np.full_like(t, y1)], axis=1))
    pts.append(np.stack([np.full_like(t, x0), y0 + (y1 - y0) * t], axis=1))
    pts.append(np.stack([np.full_like(t, x1), y0 + (y1 - y0) * t], axis=1))
    return np.vstack(pts)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="YOLO-pose 标注跨组换算（A/B/C 之间，标 1 套推 3 套）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--experiment", type=Path, required=True,
                    help="prepare_distortion_groups.py 的 --out-root（读 experiment.json）")
    ap.add_argument("--labels-in", type=Path, required=True,
                    help="源标签目录（*.txt，YOLO-pose 归一化格式）")
    ap.add_argument("--labels-out", type=Path, required=True, help="输出标签目录")
    ap.add_argument("--src-group", required=True, type=resolve_group,
                    choices=list(ALL_GROUPS), metavar="A|B|C",
                    help="源标签所在组（推荐 A）")
    ap.add_argument("--dst-group", required=True, type=resolve_group,
                    choices=list(ALL_GROUPS), metavar="A|B|C", help="目标组")
    ap.add_argument("--roboflow", action="store_true",
                    help="输入是 RoboFlow 导出（文件名带 _jpg.rf.<hash>），自动归一化文件名")
    ap.add_argument("--images-dir", type=Path, default=None,
                    help="目标组图片目录（默认取清单里的；用于校验文件名对得上）")
    ap.add_argument("--kpt", type=int, default=4, help="关键点数（门=4）")
    ap.add_argument("--size", type=int, default=None, help="图上尺寸（默认取清单）")
    ap.add_argument("--alpha", type=float, default=None, help="默认取清单")
    ap.add_argument("--min-valid-kpt", type=int, default=2,
                    help="一个实例至少几个关键点有效才保留（否则丢弃并计数）")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    exp = load_experiment(a.experiment)
    size = int(a.size or exp["size"])
    alpha = exp["alpha"] if a.alpha is None else a.alpha
    calib = exp["calibration"]
    geo = Geometry(calib, size, alpha, tuple(exp["raw_size"]))
    g = geo.for_frame(*exp["raw_size"])

    src, dst = a.src_group, a.dst_group
    if src == dst:
        sys.exit("❌ src-group == dst-group，不需要换算")
    if not a.labels_in.is_dir():
        sys.exit(f"❌ 标签目录不存在: {a.labels_in}")

    images_dir = a.images_dir or Path(exp["groups"][dst]["dir"])
    img_index = {}
    if images_dir.is_dir():
        for f in sorted(images_dir.iterdir()):
            img_index[normalize_stem(f.stem)] = f.name

    print(f"📐 {GROUP_DESC[src]}\n   -> {GROUP_DESC[dst]}")
    print(f"📄 标定 {calib}  size={size} alpha={alpha} raw={exp['raw_size']}")
    print(f"🖼  目标组图目录 {images_dir}（{len(img_index)} 张）")

    a.labels_out.mkdir(parents=True, exist_ok=True)
    stat = {"files": 0, "written": 0, "unmatched": 0, "instances": 0,
            "kept": 0, "dropped": 0, "kpt_total": 0, "kpt_invalid": 0,
            "kpt_out_of_frame": 0, "unmatched_names": []}
    per_file = []
    for txt in sorted(a.labels_in.glob("*.txt")):
        stem = normalize_stem(txt.stem) if a.roboflow else txt.stem
        stat["files"] += 1
        out_name = img_index.get(stem)
        if img_index and out_name is None:
            stat["unmatched"] += 1
            if len(stat["unmatched_names"]) < 10:
                stat["unmatched_names"].append(txt.name)
            continue
        target = Path(out_name).with_suffix(".txt") if out_name else f"{stem}.txt"

        lines_out, n_inst, n_kept, n_invalid, n_oof = [], 0, 0, 0, 0
        for line in txt.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parsed = parse_line(line, size)
            if parsed is None:
                continue
            cls, box, kpts, vis = parsed
            n_inst += 1
            if kpts.shape[0] != a.kpt:
                continue
            moved, ok = g.transform(kpts, src, dst)
            n_invalid += int((vis > 0).sum() - (ok & (vis > 0)).sum())
            inside = g.inside(moved)
            n_oof += int((ok & (vis > 0) & ~inside).sum())
            new_vis = np.where(ok & inside & (vis > 0), vis, 0.0)
            if int((new_vis > 0).sum()) < a.min_valid_kpt:
                n_invalid += int((new_vis > 0).sum())
                n_kept -= 0
                stat["dropped"] += 1
                continue
            # 框：密采样边 → 变换 → 外接矩形（只用有效点）
            per = box_perimeter(box, size)
            per_t, per_ok = g.transform(per, src, dst)
            per_ok &= g.inside(per_t)
            if per_ok.sum() >= 4:
                xs, ys = per_t[per_ok, 0], per_t[per_ok, 1]
                nb = np.array([(xs.min() + xs.max()) / 2,
                               (ys.min() + ys.max()) / 2,
                               xs.max() - xs.min(), ys.max() - ys.min()])
            else:
                xs, ys = moved[new_vis > 0, 0], moved[new_vis > 0, 1]
                nb = np.array([(xs.min() + xs.max()) / 2,
                               (ys.min() + ys.max()) / 2,
                               xs.max() - xs.min(), ys.max() - ys.min()])
            lines_out.append(fmt_line(cls, nb, moved, new_vis, size))
            n_kept += 1
            stat["kpt_total"] += int((vis > 0).sum())
            stat["kpt_invalid"] += n_invalid
            stat["kpt_out_of_frame"] += n_oof

        stat["instances"] += n_inst
        stat["kept"] += n_kept
        per_file.append({"file": target, "instances": n_inst, "kept": n_kept})
        if not a.dry_run:
            (a.labels_out / target).write_text("\n".join(lines_out) + "\n",
                                               encoding="utf-8")
        stat["written"] += 1

    report = {
        "src_group": src, "dst_group": dst, "labels_in": str(a.labels_in),
        "labels_out": str(a.labels_out), "images_dir": str(images_dir),
        "calibration": calib, "size": size, "alpha": alpha,
        "roboflow_names": a.roboflow, **stat,
    }
    if not a.dry_run:
        (a.labels_out / "_transform_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n📦 标注文件 {stat['written']}/{stat['files']} 写出"
          + (f"，{stat['unmatched']} 个文件名对不上" if stat["unmatched"] else ""))
    print(f"   实例 {stat['instances']} → 保留 {stat['kept']}，丢弃（有效点不足）{stat['dropped']}")
    tot = max(1, stat["kpt_total"])
    print(f"   关键点有效率 {100 * (tot - stat['kpt_invalid']) / tot:.1f}%"
          f"（无效 {stat['kpt_invalid']}，其中落在目标画面外 {stat['kpt_out_of_frame']}）")
    if stat["unmatched_names"]:
        print(f"   ⚠️ 对不上的例子: {stat['unmatched_names']}")
    if a.dry_run:
        print("\n🔎 dry-run：未写任何文件")
    else:
        print(f"✅ 标签已写入 {a.labels_out}/（含 _transform_report.json）")


if __name__ == "__main__":
    main()
