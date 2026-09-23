#!/usr/bin/env python3
"""
用已有权重筛选图片集：剔除含指定类别的画面 + 按画面质量加权随机抽样
（可选）多目录混合：第 1 个是主集，其余按 `--mix-ratio` 掺入

典型用途：把一批预处理后的帧集，用当前模型推理，
    1) 剔除检测到「不要的类别」（如 red_ball）的画面；
    2) 对剩余画面按质量分（清晰度为主、亮度异常惩罚）做**加权随机抽样**，
       保留 N 张作为下一轮训练/挖掘素材。

## 多目录混合

给多个输入目录即可：**第 1 个是主集，其余是混合集**，用 `--mix-ratio` 控制掺入比例：

    python scripts/1_prepare/select_frames.py \\
        data/AUV_3/processed_640/auv_xxx_frames \\   # 主集
        <第二个来源目录> \\                           # 混合集
        --weights weights/yolo11n.pt --drop-classes red_ball \
        --quality-dir data/AUV_3/auv_xxx_frames \
        --mix-ratio 0.35 --keep 2000 --out-dir data/AUV_3/selected_2000

混合集的输出文件会加前缀（默认 `nd_`，见 `--mix-prefix`）避免与主集同名，
例如 `frame_000123.jpg` 与 `nd_frame_000123.jpg`。

> **历史用途已废弃（2026-09-23）**：这个混合能力最初是为了把「不去畸变」变体掺进训练集，
> 配套脚本 `prepare_frames_noundistort.py` **已删除**；去畸变位置已定稿为 **D 域**
> （`resize(640) → enhance → remap@640`，见 `pose/map_pose_dataset.py`）。
> 多目录混合本身仍可用（混任何第二来源都行）；若混的是几何不同的图，
> **同一帧两版的标注框不通用，需分别标注**。

注意：**只有给了 `--drop-classes` 才会加载模型推理**（需要**原始 head.py**）。
      若此前跑过 scripts/3_export/modify_ultralytics.py（6 输出补丁），请先恢复：
          cp <site-packages>/ultralytics/nn/modules/head.py.backup  head.py
      不加 `--drop-classes` 时跳过全部推理（只按损坏帧过滤），速度快很多，
      此时 `--weights` 可以不传；CSV 报告里的 `n_det` 会是 0。

输出：<out-dir>/<文件名>.jpg（选中的 N 张）+ 可选 CSV 报告
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def list_images(path: Path):
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS)


def quality_score(img_path: Path) -> tuple[float, float, float]:
    """返回 (质量分, 清晰度laplacian方差, 亮度均值)
    质量分 = 清晰度 × 亮度合理度惩罚（过暗/过曝降权）"""
    img = cv2.imread(str(img_path))
    if img is None:
        return 0.0, 0.0, 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean = float(gray.mean())
    # 亮度惩罚：偏离 110 越远越低（全黑/过曝几乎为 0）
    penalty = max(0.05, 1.0 - abs(mean - 110.0) / 110.0)
    return sharp * penalty, sharp, mean


def weighted_sample_without_replacement(scores: np.ndarray, k: int,
                                        gamma: float, seed: int) -> np.ndarray:
    """Efraimidis-Spirakis 加权无放回抽样，返回被选索引（质量越高越易入选）"""
    rng = np.random.default_rng(seed)
    scores = np.clip(scores, 0, None)
    if len(scores) == 0 or k <= 0:
        return np.array([], dtype=int)
    if scores.max() <= 0:
        weights = np.ones_like(scores)
    else:
        weights = (scores / scores.max()) ** gamma
        weights = np.clip(weights, 1e-6, None)
    u = rng.random(len(scores))
    keys = u ** (1.0 / weights)
    return np.argsort(-keys)[:k]


def collect_entries(dirs: list[Path], mix_prefix: str):
    """把多个输入目录摊平成 (路径, 来源序号, 输出文件名) 三个平行列表。

    第 0 个目录 = 主集（文件名不加前缀）；其余 = 混合集（加 mix_prefix），
    保证两版同名帧在输出目录里不会互相覆盖。
    """
    paths: list[Path] = []
    src_of: list[int] = []
    out_names: list[str] = []
    used: set[str] = set()
    per_src: list[tuple[str, int]] = []

    for si, d in enumerate(dirs):
        if not d.is_dir():
            sys.exit(f"❌ 不是目录: {d}")
        files = list_images(d)
        if not files:
            sys.exit(f"❌ 目录中没有图片: {d}")
        prefix = mix_prefix if si > 0 else ""
        for p in files:
            name = f"{prefix}{p.name}"
            if name in used:                      # 极少数重名（同名不同扩展名）
                j = 2
                while f"{prefix}{p.stem}_{j}{p.suffix}" in used:
                    j += 1
                name = f"{prefix}{p.stem}_{j}{p.suffix}"
            used.add(name)
            paths.append(p)
            src_of.append(si)
            out_names.append(name)
        # 用末两级目录名做标签：不同来源的叶子目录常同名（如 .../xx/<组> 与
        # .../yy/<组>），只取末级会分不清来源
        label = "/".join(d.parts[-2:]) if len(d.parts) >= 2 else (d.name or str(d))
        per_src.append((label, len(files)))
    return paths, src_of, out_names, per_src


def allocate_quota(k_total: int, mix_ratio: float,
                   n_pri: int, n_mix: int) -> tuple[int, int]:
    """把 k_total 个名额按 mix_ratio 分给 主集/混合集。

    返回 (k_pri, k_mix)。一侧候选不够时把余量让给另一侧，尽量凑满 k_total。
    """
    k_mix = min(int(round(k_total * mix_ratio)), n_mix)
    k_pri = min(k_total - k_mix, n_pri)
    k_mix = min(k_total - k_pri, n_mix)
    k_pri = min(k_total - k_mix, n_pri)
    return k_pri, k_mix


def main() -> None:
    ap = argparse.ArgumentParser(
        description="按已有权重剔除指定类别画面 + 质量加权随机抽样（可混入不去畸变变体）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("images", nargs="+", type=Path,
                    help="输入图片目录（可多个；第 1 个为主集，其余为混合集）")
    ap.add_argument("--weights", default=None,
                    help="模型权重（如 weights/yolo11n.pt；建议用当前 detect 权重）；"
                         "仅在给了 --drop-classes 时需要")
    ap.add_argument("--drop-classes", default="",
                    help="要剔除的类别名，逗号分隔（如 red_ball）；空=不按类别剔除")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="判定该类别的置信度阈值（低阈值=更激进地剔除）")
    ap.add_argument("--quality-dir", type=Path, default=None,
                    help="用该目录的同名图片计算质量（建议用原始高分辨率帧；默认用输入图）")
    ap.add_argument("--mix-ratio", type=float, default=0.0,
                    help="最终保留中来自「第 2 个及以后输入目录」的比例（0~1；0=只用主集）")
    ap.add_argument("--mix-prefix", default="nd_",
                    help="混合集输出文件名前缀（避免与主集同名）")
    ap.add_argument("--keep", type=int, default=2000, help="最终保留张数")
    ap.add_argument("--gamma", type=float, default=2.0,
                    help="质量权重指数（越大越偏向高质量图；1=纯随机）")
    ap.add_argument("--seed", type=int, default=0, help="随机种子")
    ap.add_argument("--out-dir", type=Path, required=True, help="选中图片输出目录")
    ap.add_argument("--report", type=Path, default=None, help="CSV 报告路径（可选）")
    ap.add_argument("--batch", type=int, default=16, help="推理批大小")
    ap.add_argument("--chunk", type=int, default=64,
                    help="分块处理的图片数（控制内存峰值，越小越省内存）")
    ap.add_argument("--device", default="0", help="推理设备（'0'=GPU, 'cpu'）")
    a = ap.parse_args()

    if not 0.0 <= a.mix_ratio <= 1.0:
        sys.exit(f"❌ --mix-ratio 需在 0~1 之间，当前 {a.mix_ratio}")

    files, src_of, out_names, per_src = collect_entries(a.images, a.mix_prefix)
    n = len(files)
    print(f"📥 输入 {n} 张，共 {len(a.images)} 组:")
    for si, (nm, cnt) in enumerate(per_src):
        role = "主集" if si == 0 else f"混合集(前缀 '{a.mix_prefix}')"
        print(f"     [{si}] {nm}: {cnt} 张  ({role})")
    if len(a.images) > 1 and a.mix_ratio <= 0:
        print("ℹ️  有多个输入目录但 --mix-ratio=0：本次仍只用主集")

    # ---------- 1+2. 分块推理 + 质量评分（低内存，无多进程 DataLoader） ----------
    drop_names = [s.strip() for s in a.drop_classes.split(",") if s.strip()]
    model = None
    drop_ids: list[int] = []
    if drop_names:
        if not a.weights:
            sys.exit("❌ 指定了 --drop-classes 就必须给 --weights")
        from ultralytics import YOLO
        model = YOLO(a.weights)
        names = model.names
        id2name = names if isinstance(names, dict) else dict(enumerate(names))
        for nm in drop_names:
            hit = [i for i, n in id2name.items() if n == nm]
            if not hit:
                sys.exit(f"❌ 权重类别中没有 {nm!r}；可用类别: {list(id2name.values())}")
            drop_ids.extend(hit)
        print(f"🎯 模型类别 {list(id2name.values())}；"
              f"剔除类别 {drop_names} → ids {drop_ids}")
    elif a.weights:
        print("ℹ️  未指定 --drop-classes：不做类别剔除，**跳过模型推理**"
              "（只按损坏帧过滤，省去全部推理时间）")
    else:
        print("ℹ️  未指定 --drop-classes 与 --weights：不做类别剔除，跳过推理")

    qdir = a.quality_dir or a.images[0]
    dropped = np.zeros(n, dtype=bool)      # 含目标类别 / 读取失败
    det_count = np.zeros(n, dtype=np.int32)
    scores = np.zeros(n)
    sharps = np.zeros(n)
    means = np.zeros(n)

    for start in range(0, n, a.chunk):
        batch_files = files[start:start + a.chunk]
        imgs, idxs = [], []
        for j, p in enumerate(batch_files):
            i = start + j
            im = cv2.imread(str(p))
            # 质量分用原始高分辨率同名图（未被 resize 影响）
            qp = qdir / p.name
            if not qp.exists():
                qp = p
            s, sh, mn = quality_score(qp)
            scores[i], sharps[i], means[i] = s, sh, mn
            if im is None:                    # 损坏/截断帧直接剔除
                dropped[i] = True
                continue
            imgs.append(im)
            idxs.append(i)
        if imgs and model is not None:
            rs = model.predict(imgs, imgsz=640, conf=a.conf, batch=a.batch,
                               device=a.device or None, verbose=False)
            for i, r in zip(idxs, rs):
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                det_count[i] = len(r.boxes)
                cls = r.boxes.cls.cpu().numpy().astype(int)
                if any(int(c) in drop_ids for c in cls):
                    dropped[i] = True
        done = min(start + a.chunk, n)
        print(f"   … {done}/{n}  已剔除 {int(dropped[:done].sum())}", flush=True)
        if model is not None and (start // a.chunk) % 10 == 9:
            import torch
            torch.cuda.empty_cache()

    n_drop = int(dropped.sum())
    why = f"含 {drop_names} / " if drop_names else ""
    print(f"🚫 {why}损坏的图片: {n_drop} 张 → 剩余候选 {n - n_drop} 张")

    # ---------- 3. 质量加权随机抽样（多目录时按 --mix-ratio 分配名额） ----------
    cand = np.where(~dropped)[0]
    if len(cand) == 0:
        sys.exit("❌ 所有图片都被剔除，无候选")
    src_arr = np.array(src_of)
    k_total = min(a.keep, len(cand))

    if len(a.images) > 1 and a.mix_ratio > 0:
        cand_pri = cand[src_arr[cand] == 0]        # 主集（已去畸变）
        cand_mix = cand[src_arr[cand] > 0]         # 混合集（未去畸变）
        k_pri, k_mix = allocate_quota(k_total, a.mix_ratio,
                                      len(cand_pri), len(cand_mix))
        picked_pri = cand_pri[weighted_sample_without_replacement(
            scores[cand_pri], k_pri, a.gamma, a.seed)]
        picked_mix = cand_mix[weighted_sample_without_replacement(
            scores[cand_mix], k_mix, a.gamma, a.seed + 1)]
        picked = np.concatenate([picked_pri, picked_mix])
        print(f"🧩 名额分配: 主集 {len(picked_pri)} 张 + 混合集 {len(picked_mix)} 张"
              f"（目标比例 {a.mix_ratio:.0%}，实际 "
              f"{len(picked_mix) / max(len(picked), 1):.0%}）")
    else:
        picked = cand[weighted_sample_without_replacement(
            scores[cand], k_total, a.gamma, a.seed)]

    a.out_dir.mkdir(parents=True, exist_ok=True)
    for i in picked:
        shutil.copy2(files[i], a.out_dir / out_names[i])

    # ---------- 4. 统计与报告 ----------
    print(f"选中 {len(picked)}/{len(cand)} 张（质量加权 γ={a.gamma}）")
    for si, (nm, _) in enumerate(per_src):
        sel = int(np.sum(src_arr[picked] == si))
        print(f"     [{si}] {nm}: 选中 {sel} 张")
    print(f"  清晰度(Lap方差) 中位数: 全体候选 {np.median(sharps[cand]):.0f}"
          f" → 选中 {np.median(sharps[picked]):.0f}")
    print(f"  亮度均值 中位数: 候选 {np.median(means[cand]):.1f}"
          f" → 选中 {np.median(means[picked]):.1f}")

    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        sel = set(picked.tolist())
        with open(a.report, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "source", "contains_drop", "n_det", "quality",
                        "sharpness", "brightness", "selected"])
            for i, p in enumerate(files):
                w.writerow([out_names[i], per_src[src_of[i]][0], int(dropped[i]),
                            int(det_count[i]), f"{scores[i]:.2f}",
                            f"{sharps[i]:.1f}", f"{means[i]:.1f}",
                            int(i in sel)])
        print(f"📄 报告: {a.report}")

    print(f"✅ 输出目录: {a.out_dir}")


if __name__ == "__main__":
    main()
