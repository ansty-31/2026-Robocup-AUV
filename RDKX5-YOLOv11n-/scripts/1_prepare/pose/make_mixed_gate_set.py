#!/usr/bin/env python3
"""跨水质融合门图 → 定稿链路渲染 → 交 Roboflow 打标的待标图集

默认构成（共 3000）：清水 1200（AUV_5）｜中水 1000（AUV_4 700 + AUV_1 300）｜
浊水 800（AUV_2 500 + AUV_3 300）。

链路（D 域 + WB + gamma，无 CLAHE，与板端 `common/preprocess.py` 逐像素一致）：
    resize(640) → WB+gamma → remap@640

筛选：等距抽样（跨整段流）→ 渲染 → 质量过滤（清晰度/曝光）→ 门框存在性过滤
（pose 模型低阈值；清水用更低阈值以免把它筛掉）→ 内容去重（同段内与上一张保留帧的
32x32 灰度平均绝对差 >= `--dedup-thr` 才保留，配额不足时逐级放宽）→ 按质量分取配额。

输出：`data/derived/AUV_5_selected_3000_Dwb/`（扁平目录，文件名 `<dive>_<原目录>_<原帧号>.jpg`）
      + `manifest.csv`（输出名 → 水质/来源/原始帧/清晰度/亮度/模型门框置信度）

用法：
    python scripts/1_prepare/pose/make_mixed_gate_set.py                 # 默认 3000
    python scripts/1_prepare/pose/make_mixed_gate_set.py --total 3000 --no-model-filter
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "1_prepare"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from map_pose_dataset import Domain  # noqa: E402

# 水质 → (来源目录, 默认配额, 模型门框阈值)
SOURCES = {
    "AUV_5": [("data/frames/AUV_5_gate_calib/gate-*", 1200, 0.10)],          # 清水（偏多）
    "AUV_4": [("data/frames/AUV_4_auv_4_frames", 700, 0.25)],              # 中水
    "AUV_1": [("data/frames/AUV_1_rec_front_frames", 300, 0.25)],          # 中水
    "AUV_2": [("data/frames/AUV_2_auv_20260910_172208_frames", 500, 0.25)],  # 浊水
    "AUV_3": [("data/frames/AUV_3_auv_20260911_201754_frames", 300, 0.25)],  # 浊水
}
WATER = {"AUV_5": "清水", "AUV_4": "中水", "AUV_1": "中水", "AUV_2": "浊水", "AUV_3": "浊水"}


def spread_sample(paths, k):
    """等距抽样（跨整段）"""
    paths = sorted(paths)
    if k >= len(paths):
        return paths
    idx = np.linspace(0, len(paths) - 1, k).astype(int)
    return [paths[i] for i in idx]


_SIG_CACHE = {}


def _sig(path):
    """32x32 灰度签名（内容去重用），带缓存"""
    k = str(path)
    s = _SIG_CACHE.get(k)
    if s is None:
        s = cv2.resize(cv2.imread(k, cv2.IMREAD_GRAYSCALE), (32, 32),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
        _SIG_CACHE[k] = s
    return s


def dedup_streams(scored, thr):
    """只在同一段录像内，按时间顺序贪心去重：与同段上一张保留帧的 mad >= thr 才保留"""
    by_stream = {}
    for item in scored:
        by_stream.setdefault(item[2].parent, []).append(item)
    out = []
    for parent, items in by_stream.items():
        items.sort(key=lambda s: s[2].name)             # 段内按帧号 = 时间序
        last = None
        for sharp, bright, m, c in items:
            g = _sig(m)
            if last is None or float(np.abs(g - last).mean()) >= thr:
                last = g
                out.append((sharp, bright, m, c))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="weights/domain_D.pt", help="门框存在性过滤用的 pose 权重")
    ap.add_argument("--total", type=int, default=3000)
    ap.add_argument("--oversample", type=float, default=1.7, help="候选超额倍数（供筛除）")
    ap.add_argument("--out-dir", default="data/derived/AUV_5_selected_3000_Dwb")
    ap.add_argument("--no-model-filter", action="store_true", help="不做门框存在性过滤")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dedup-thr", type=float, default=3.0,
                    help="同段内容去重阈值：与上一张保留帧的 32x32 灰度平均绝对差 >= 此值才保留"
                         "（0 = 关闭；参考值 2 可见变化 / 3 默认 / 4 明显位移）")
    a = ap.parse_args()

    out = PROJECT_ROOT / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    dom = Domain("D", "configs/front_camera.yaml", "wb")     # 定稿链路
    scale = a.total / sum(q for v in SOURCES.values() for _, q, _ in v)

    from ultralytics import YOLO
    model = None if a.no_model_filter else YOLO(str(PROJECT_ROOT / a.weights))

    rows = []
    for dive, specs in SOURCES.items():
        for pat, quota, thr in specs:
            quota = int(round(quota * scale))
            pool = []
            for p in sorted(PROJECT_ROOT.glob(pat)):
                pool += sorted(p.glob("*.jpg")) if p.is_dir() else [p]
            if not pool:
                print(f"⚠️ {dive} {pat}: 无原始帧，跳过")
                continue
            cand = spread_sample(pool, int(quota * a.oversample))
            print(f"{dive:<6} {WATER[dive]}  池 {len(pool):>6} → 候选 {len(cand):>5} → 目标 {quota}", flush=True)
            imgs, meta = [], []
            for f in cand:
                raw = cv2.imread(str(f))
                if raw is None or raw.shape[0] < 100:
                    continue
                imgs.append(dom.render(raw))
                meta.append(f)
            # 门框存在性过滤（低阈值；清水的阈值更低，避免把它筛空）
            if model is not None:
                keep, kconf = [], []
                for i in range(0, len(imgs), 32):
                    ch = imgs[i:i + 32]
                    outs = model.predict(ch, imgsz=640, conf=0.05, verbose=False, device=0)
                    for m, r in zip(meta[i:i + 32], outs):
                        c = float(r.boxes.conf.max()) if (r.boxes is not None and len(r.boxes)) else 0.0
                        keep.append(c >= thr)
                        kconf.append(c)
                imgs = [im for im, k in zip(imgs, keep) if k]
                meta = [m for m, k in zip(meta, keep) if k]
                confs = [c for c in kconf if c >= thr]
            else:
                confs = [float("nan")] * len(imgs)
            # 质量过滤 + 打分（清晰度为主，亮度异常惩罚）
            scored = []
            for im, m, c in zip(imgs, meta, confs):
                g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                sharp = float(cv2.Laplacian(g, cv2.CV_32F).var())
                bright = float(g.mean())
                if bright < 25 or bright > 215:
                    continue
                scored.append((sharp, bright, m, c))
            if not scored:
                print(f"   ⚠️ 过滤后为空")
                continue
            sh = np.array([s[0] for s in scored])
            cut = np.percentile(sh, 20)                     # 丢掉最模糊的 20%
            scored = [s for s in scored if s[0] >= cut]
            # 内容去重：配额不足时从 --dedup-thr 逐级放宽，取能填满配额的最严阈值
            if a.dedup_thr > 0:
                levels = [t for t in (a.dedup_thr, a.dedup_thr - 0.5, a.dedup_thr - 1.0, 1.5) if t > 0]
                levels = sorted(set(round(t, 2) for t in levels), reverse=True)
                best, used = None, None
                for thr in levels:
                    ded = dedup_streams(scored, thr)
                    if best is None or len(ded) > len(best):
                        best, used = ded, thr
                    if len(ded) >= quota:
                        best, used = ded, thr
                        break
                scored = best
                if used < a.dedup_thr:
                    print(f"   去重阈值放宽到 {used} 才够配额", flush=True)
            scored.sort(key=lambda s: s[0], reverse=True)
            for sharp, bright, m, c in scored[:quota]:
                name = f"{dive}_{m.parent.name}_{m.name}"
                cv2.imwrite(str(out / name), dom.render(cv2.imread(str(m))),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append(dict(name=name, water=WATER[dive], dive=dive,
                                 src_dir=str(m.parent.relative_to(PROJECT_ROOT)),
                                 raw_frame=m.name, sharp=round(sharp, 1),
                                 bright=round(bright, 1),
                                 gate_conf=("" if c != c else round(c, 3))))
            print(f"   去重后 {len(scored)} → 保留 {min(len(scored), quota)}", flush=True)

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n共写出 {len(rows)} 张 → {out.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    main()
