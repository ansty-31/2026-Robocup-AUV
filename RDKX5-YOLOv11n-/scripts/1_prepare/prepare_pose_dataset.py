#!/usr/bin/env python3
"""
Roboflow pose 导出 → 可直接训练的 YOLO pose 数据集（kpt_shape 归一 + flip_idx 修正）

## 为什么需要这一步

Roboflow 的 Pose 项目在"关键点 schema"里加过的槽位**不会随标注删除而消失**：
导出的 `data.yaml` 会带上全部槽位（`kpt_shape: [5, 3]`），而没标过的槽位在
每一行标签里都写成 `0 0 0`（x=y=v=0）。这类**尾随全零槽位**必须截掉，否则：

* 输出头 kpt 通道数从 12（4 点）变成 15（5 点），
  与 `configs/gate_kpt_config.yaml` 的 `kpt12`、板端 `gate_decode.py` 对不上；
* `scripts/3_export/export_onnx.py` 会告警"gate 需要 4 点"。

另外 `flip_idx` 是个**必须与既有模型保持一致的工程约定**，不能想当然改：

ultralytics 在 `data/augment.py:1466-1469` 先 `instances.fliplr(w)`（镜像坐标），
再 `keypoints[:, flip_idx]`（下标重排）。于是两种约定各自自洽：

* **恒等** `[0,1,2,3]`：翻转后索引**跟着物理角点走**（"索引 i 永远是门的第 i 个物理角"）。
* **左右交换** `[1,0,3,2]`：翻转后索引**跟着图像位置走**（"索引 0 永远是画面左上角"）。

两者只在**镜像图 / 从门背面看**时才有区别；板端 `gate/geometry.py` 的
`object_points()` 明确写着「前后完全对称、无朝向要求 → from_front 恒 true」，
`gate/gate_frontend.py` 的 mode 判定也只用到 TL/TR 与 BL/BR 的**配对**，
所以两种约定对板端**等价**。

`--flip-idx` 默认值因此是「**沿用源 data.yaml 的 flip_idx 截到目标点数**」
（Roboflow 写的是恒等 `[0,1,2,3,4]` → 截成 `[0,1,2,3]`），这样微调时
增广语义与 `weights/yolo11n-pose.pt`（现有板端模型）一致，不会互相打架。
要换成图像位置约定，显式加 `--flip-idx-lr-swap`。

> 实测依据（本仓库 2026-09-17）：用 `weights/yolo11n-pose.pt` 在
> `data/AUV_4/PNP.kpt4.yolov8/valid` 上跑，未翻转图的逐索引误差
> 「同序 32/61/81/52 px」远小于「交换 267/279/295/251 px」（89/96 张同序胜）；
> 翻转图则 112/124 张符合「索引跟物理角点」= 恒等约定。即现有模型就是
> 恒等约定训出来的，且与标注、与板端 object_points 一致。

## 做了什么

1. **截断尾随全零关键点槽位**：全数据集范围内 x=y=v=0 的尾部槽位逐个丢弃，
   直到遇到有值的槽位；结果必须等于 `--kpt`（默认 4），否则报错退出。
2. **校验角点顺序约定**：只用四角齐全的实例，检查
   `x0<x1`、`x3<x2`、`y0<y3`、`y1<y2` 的比例；低于 `--order-min-ratio`
   （默认 0.98）说明标注不是"图像空间 TL,TR,BR,BL"，此时**不猜**，直接报错。
3. **确定 flip_idx**：默认沿用源 `data.yaml` 的 flip_idx 截到目标点数（保持与既有
   板端模型一致的增广约定，见上文）；`--flip-idx-lr-swap` 切到"索引跟图像位置"约定。
4. **输出新数据集目录**：图片用硬链接（同一文件系统时零拷贝），标签重写，
   写 `data.yaml` + `normalize_report.txt`。**原导出目录一个字节都不改。**

划分（train/valid/test）不在本脚本里做：先跑本脚本，再跑
`scripts/1_prepare/resplit_dataset.py` 做序列感知划分（相邻帧不跨 split）。

## 用法

    # 先看要改什么，不写文件
    python scripts/1_prepare/prepare_pose_dataset.py data/AUV_4/PNP.yolov8 --dry-run

    # 生成规范化数据集（默认 <src 同级>/PNP.kpt4.yolov8）
    python scripts/1_prepare/prepare_pose_dataset.py data/AUV_4/PNP.yolov8

    # 指定输出目录，然后做序列感知划分
    python scripts/1_prepare/prepare_pose_dataset.py data/AUV_4/PNP.yolov8 \
        --out data/AUV_4/PNP.kpt4.yolov8
    python scripts/1_prepare/resplit_dataset.py data/AUV_4/PNP.kpt4.yolov8 --dry-run

## 输出

* `<out>/{train,valid,test}/{images,labels}/`
* `<out>/data.yaml`            — kpt_shape=[4,3], flip_idx=[1,0,3,2], nc/names 沿用
* `<out>/normalize_report.txt` — 截断了几个槽位、角点顺序统计、每 split 张数
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

SPLITS = ("train", "valid", "test")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CORNERS = ("TL", "TR", "BR", "BL")


# ---------------------------------------------------------------- 标签读写

def parse_label_line(line: str):
    """一行 YOLO-pose 标签 → (cls, [cx,cy,w,h], [(x,y,v), ...])"""
    t = line.split()
    if len(t) < 5 or (len(t) - 5) % 3 != 0:
        raise ValueError(f"字段数 {len(t)} 不是 5+3k: {line[:60]!r}")
    cls = int(t[0])
    box = [float(x) for x in t[1:5]]
    kpts = [(float(t[5 + 3 * i]), float(t[6 + 3 * i]), float(t[7 + 3 * i]))
            for i in range((len(t) - 5) // 3)]
    return cls, box, kpts


def load_dataset(src: Path):
    """读全量标签 → {(split, label_path): (cls, box, kpts)}，并统计槽位数"""
    records = {}
    slots = Counter()
    for s in SPLITS:
        lp = src / s / "labels"
        if not lp.is_dir():
            continue
        for f in sorted(lp.glob("*.txt")):
            items = []
            for line in f.read_text(encoding="utf-8").strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                cls, box, kpts = parse_label_line(line)
                items.append((cls, box, kpts))
                slots[len(kpts)] += 1
            records[(s, f)] = items
    if not records:
        sys.exit(f"❌ {src} 下没有找到任何 {SPLITS}/*/labels/*.txt")
    return records, slots


def droppable_trailing_slots(records, n_slots: int) -> int:
    """从最高位往下数：全数据集都满足 x=y=v=0 的尾随槽位个数"""
    drop = 0
    for i in range(n_slots - 1, -1, -1):
        allzero = all(
            kp[i][0] == 0.0 and kp[i][1] == 0.0 and kp[i][2] == 0.0
            for items in records.values() for (_, _, kp) in items
            if len(kp) > i
        )
        if allzero:
            drop += 1
        else:
            break
    return drop


# ---------------------------------------------------------------- 校验

def check_corner_order(records, kpt: int):
    """四角齐全的实例里，角点是否按图像空间 TL,TR,BR,BL 排"""
    if kpt != 4:
        return None
    tot = 0
    ok = Counter()
    unlabeled = Counter()
    for items in records.values():
        for (_, _, kp) in items:
            for i in range(4):
                if kp[i][2] == 0:
                    unlabeled[i] += 1
            if any(kp[i][2] == 0 for i in range(4)):
                continue
            tot += 1
            ok["x0<x1"] += kp[0][0] < kp[1][0]
            ok["x3<x2"] += kp[3][0] < kp[2][0]
            ok["y0<y3"] += kp[0][1] < kp[3][1]
            ok["y1<y2"] += kp[1][1] < kp[2][1]
    return tot, ok, unlabeled


def make_lr_swap_flip_idx(kpt: int) -> list[int]:
    """左右配对交换：[1,0,3,2,5,4,...]（"索引跟图像位置"约定）"""
    idx = []
    for i in range(kpt):
        idx.append(i + 1 if i % 2 == 0 else i - 1)
    return idx


def choose_flip_idx(src_flip_idx, kpt: int, lr_swap: bool):
    """决定本次要写进 data.yaml 的 flip_idx。"""
    if lr_swap:
        return make_lr_swap_flip_idx(kpt), "—flip-idx-lr-swap（索引跟图像位置）"
    src = list(src_flip_idx or [])
    if len(src) >= kpt:
        return src[:kpt], "沿用源 data.yaml 截到目标点数"
    return list(range(kpt)), "源 flip_idx 太短，退化为恒等"


# ---------------------------------------------------------------- 主流程

def main():
    p = argparse.ArgumentParser(
        description="Roboflow pose 导出 → kpt_shape 归一 + flip_idx 修正的 YOLO pose 数据集",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("src", type=Path, help="Roboflow 导出的数据集根目录（含 train/valid/test）")
    p.add_argument("--out", type=Path, default=None,
                   help="输出目录（默认 <src 同级>/<src 名去掉 .yolov8>.kpt<N>.yolov8）")
    p.add_argument("--kpt", type=int, default=4, help="目标关键点数（gate 4 角点）")
    p.add_argument("--flip-idx", type=int, nargs="*", default=None,
                   help="直接指定 flip_idx（覆盖默认策略）")
    p.add_argument("--flip-idx-lr-swap", action="store_true",
                   help="改用「索引跟图像位置」约定（[1,0,3,2,...]）；默认沿用源 flip_idx")
    p.add_argument("--order-min-ratio", type=float, default=0.98,
                   help="角点顺序一致性下限，低于此值直接报错（不猜）")
    p.add_argument("--copy", action="store_true", help="复制图片而不是硬链接")
    p.add_argument("--dry-run", action="store_true", help="只报告，不写任何文件")
    a = p.parse_args()

    src: Path = a.src.resolve()
    if not (src / "data.yaml").exists():
        sys.exit(f"❌ {src}/data.yaml 不存在")
    meta = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8")) or {}
    out = a.out or (src.parent / f"{src.stem.split('.')[0]}.kpt{a.kpt}.yolov8")

    print("=" * 72)
    print(f"源数据集 : {src}")
    print(f"目标数据集: {out}{'   [dry-run]' if a.dry_run else ''}")
    print("=" * 72)

    records, slots = load_dataset(src)
    print(f"\n标签实例数: {sum(len(v) for v in records.values())}   每实例关键点槽位数: {dict(slots)}")
    if len(slots) != 1:
        sys.exit(f"❌ 关键点槽位数不唯一 {dict(slots)}，无法判断 kpt_shape")
    n_slots = next(iter(slots))
    print(f"data.yaml 声明: kpt_shape={meta.get('kpt_shape')}  flip_idx={meta.get('flip_idx')}")

    # --- 1) 截断尾随全零槽位 ---
    drop = droppable_trailing_slots(records, n_slots)
    print(f"\n[1] 尾随全零关键点槽位（全数据集 x=y=v=0）: {drop} 个 → 保留 {n_slots - drop} 点")
    if n_slots - drop != a.kpt:
        sys.exit(f"❌ 截断后是 {n_slots - drop} 点，与 --kpt {a.kpt} 不符。\n"
                 f"   若确实要 {n_slots - drop} 点，用 --kpt {n_slots - drop}；"
                 f"若尾部槽位本该有标注，请回 Roboflow 补标后重新导出。")

    # --- 2) 角点顺序校验 ---
    order = check_corner_order(records, a.kpt)
    if order:
        tot, ok, unlabeled = order
        print(f"\n[2] 角点顺序校验（四角齐全实例 {tot} 个）")
        worst = 1.0
        for k in ("x0<x1", "x3<x2", "y0<y3", "y1<y2"):
            r = ok[k] / tot if tot else 0.0
            worst = min(worst, r)
            print(f"      {k}: {ok[k]}/{tot} = {100 * r:.1f}%")
        print(f"    各角 v=0（未标注，不参与 pose loss）计数: "
              + ", ".join(f"{CORNERS[i]}={unlabeled[i]}" for i in range(a.kpt)))
        if worst < a.order_min_ratio:
            sys.exit(f"❌ 角点顺序一致性 {worst:.3f} < {a.order_min_ratio}，"
                     f"标注可能不是图像空间 TL,TR,BR,BL 顺序，请人工确认后再改 flip_idx")

    # --- 3) flip_idx ---
    if a.flip_idx:
        flip_idx, why = a.flip_idx, "命令行 --flip-idx 指定"
    else:
        flip_idx, why = choose_flip_idx(meta.get("flip_idx"), a.kpt, a.flip_idx_lr_swap)
    if len(flip_idx) != a.kpt:
        sys.exit(f"❌ flip_idx 长度 {len(flip_idx)} != kpt {a.kpt}")
    print(f"\n[3] flip_idx: {meta.get('flip_idx')} → {flip_idx}   ({why})")

    if a.dry_run:
        names = meta.get("names") or [f"cls{i}" for i in range(meta.get("nc", 1))]
        print(f"\n[dry-run] 将写出: kpt_shape=[{a.kpt}, 3], flip_idx={flip_idx}, "
              f"nc={meta.get('nc', len(names))}, names={names}")
        print("[dry-run] 未写任何文件。")
        return

    # --- 4) 落盘 ---
    if out.exists():
        sys.exit(f"❌ 输出目录已存在: {out}\n   先删掉或换 --out，避免新旧标签混在一起。")
    report = [f"源数据集: {src}", f"目标数据集: {out}",
              f"槽位: {n_slots} → {a.kpt}（丢弃尾随全零 {drop} 个）",
              f"flip_idx: {meta.get('flip_idx')} → {flip_idx}", ""]
    link = not a.copy

    for s in SPLITS:
        idir = src / s / "images"
        if not idir.is_dir():
            continue
        (out / s / "images").mkdir(parents=True, exist_ok=True)
        (out / s / "labels").mkdir(parents=True, exist_ok=True)
        n_img = n_lbl = n_inst = 0
        for img in sorted(p for p in idir.iterdir() if p.suffix.lower() in IMG_EXTS):
            items = records.get((s, src / s / "labels" / f"{img.stem}.txt"))
            if items is None:
                print(f"  ⚠️  图片没有对应标签，跳过: {s}/{img.name}")
                continue
            dst_img = out / s / "images" / img.name
            if link:
                try:
                    os.link(img, dst_img)
                except OSError:
                    shutil.copy2(img, dst_img)
            else:
                shutil.copy2(img, dst_img)
            n_img += 1
            lines = []
            for cls, box, kpts in items:
                kept = kpts[:a.kpt]
                lines.append(" ".join(
                    [str(cls)] + [f"{v:.10g}" for v in box]
                    + [f"{c:.10g}" for kp in kept for c in kp]))
                n_inst += 1
            (out / s / "labels" / f"{img.stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            n_lbl += 1
        print(f"  {s}: 图片 {n_img}  标签 {n_lbl}  实例 {n_inst}")
        report.append(f"{s}: 图片 {n_img} 标签 {n_lbl} 实例 {n_inst}")

    new_meta = dict(meta)
    new_meta["kpt_shape"] = [a.kpt, 3]
    new_meta["flip_idx"] = flip_idx
    new_meta.pop("path", None)          # 让 ultralytics 以 yaml 所在目录为数据根
    (out / "data.yaml").write_text(
        yaml.safe_dump(new_meta, allow_unicode=True, sort_keys=False), encoding="utf-8")

    report += ["", f"names: {meta.get('names')}", f"nc: {meta.get('nc')}",
               "", "下一步（序列感知划分）:",
               f"  python scripts/1_prepare/resplit_dataset.py {out} --dry-run"]
    (out / "normalize_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"\n✅ 已写出: {out}")
    print(f"   data.yaml: kpt_shape=[{a.kpt}, 3]  flip_idx={flip_idx}")
    print(f"   下一步: python scripts/1_prepare/resplit_dataset.py {out} --dry-run")


if __name__ == "__main__":
    main()
