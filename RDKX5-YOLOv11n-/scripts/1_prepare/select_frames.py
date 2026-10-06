#!/usr/bin/env python3
"""筛图：剔除含指定类别的画面 + 按画面质量加权随机抽样。

用 --weights 推理（需要**原始 head.py**，若打过 6 输出补丁先 restore）：
    1) 剔除检出 --drop-classes 中任一类别（conf ≥ --conf）的画面；
    2) 其余按质量分（清晰度 × 亮度合理性）加权无放回抽样，保留 --keep 张。

只有给了 --drop-classes 才加载模型；否则只过滤损坏帧（--weights 可不传，CSV 的 n_det 为 0）。
多个输入目录合并成一个候选池（来源只用于统计与 CSV 报告），同名文件加 _2/_3 后缀防覆盖。

用法：
    python scripts/1_prepare/select_frames.py data/frames/AUV_3_auv_xxx_frames \\
        --weights weights/yolo11n.pt --drop-classes red_ball \\
        --quality-dir data/frames/AUV_3_auv_xxx_frames \\
        --keep 2000 --out-dir _archive/auv5/derived_AUV_3_selected_2000

输出：<out-dir>/<文件名>.jpg（选中的 N 张）+ 可选 --report CSV 报告。
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
    """返回 (质量分, 清晰度 laplacian 方差, 亮度均值)；
    质量分 = 清晰度 × 亮度合理度惩罚（偏离 110 越远越低）。"""
    img = cv2.imread(str(img_path))
    if img is None:
        return 0.0, 0.0, 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean = float(gray.mean())
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


def collect_entries(dirs: list[Path]):
    """把多个输入目录摊平成 (路径, 来源序号, 输出文件名) 三个平行列表：
    多个目录合并成一个候选池（来源序号只用于统计与报告），同名文件加 _2/_3 后缀防覆盖。"""
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
        for p in files:
            name = p.name
            if name in used:                      # 极少数重名（同名不同扩展名）
                j = 2
                while f"{p.stem}_{j}{p.suffix}" in used:
                    j += 1
                name = f"{p.stem}_{j}{p.suffix}"
            used.add(name)
            paths.append(p)
            src_of.append(si)
            out_names.append(name)
        # 标签取末两级目录名：不同来源的叶子目录常同名
        label = "/".join(d.parts[-2:]) if len(d.parts) >= 2 else (d.name or str(d))
        per_src.append((label, len(files)))
    return paths, src_of, out_names, per_src


def main() -> None:
    ap = argparse.ArgumentParser(
        description="按已有权重剔除指定类别画面 + 质量加权随机抽样",
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

    files, src_of, out_names, per_src = collect_entries(a.images)
    n = len(files)
    print(f"📥 输入 {n} 张，共 {len(a.images)} 组:")
    for si, (nm, cnt) in enumerate(per_src):
        print(f"     [{si}] {nm}: {cnt} 张")

    # ---------- 分块推理 + 质量评分（分块控制内存峰值，不用 DataLoader） ----------
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

    # ---------- 质量加权随机抽样 ----------
    cand = np.where(~dropped)[0]
    if len(cand) == 0:
        sys.exit("❌ 所有图片都被剔除，无候选")
    src_arr = np.array(src_of)
    k_total = min(a.keep, len(cand))

    picked = cand[weighted_sample_without_replacement(
        scores[cand], k_total, a.gamma, a.seed)]

    a.out_dir.mkdir(parents=True, exist_ok=True)
    for i in picked:
        shutil.copy2(files[i], a.out_dir / out_names[i])

    # ---------- 统计与报告 ----------
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
