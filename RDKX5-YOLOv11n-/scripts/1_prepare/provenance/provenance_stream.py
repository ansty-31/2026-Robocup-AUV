#!/usr/bin/env python3
"""
按【裸流帧序号】直达的溯源验证（Route B 主工具）

背景（2026-09-23 查明）
----------------------
标注数据集里的文件名编号 `frame_NNNNNN` **就是原始裸流的帧序号**（1-based，
按 JPEG SOI 标记顺序）。证据：
  · AUV_2/AUV_3 里能在现存 `*_frames` 池中命中的帧，满足 N = 25*(k-1)+1
    （k 是池内序号），即池 = 每 25 帧取 1 帧、并从 1 重新编号的子集；
  · 标注号 mod 25 近似均匀 → 标注集并非只含 1fps 帧，而是含全帧率的裸流帧；
  · AUV_4 的 450 张命中全部是 offset 0（`auv_4_frames` 是全量抽取，未重编号）。

因此**不需要**内容最近邻：给定标注号 N，直接在每条裸流的第 N 帧取图比对即可。
现存 `data/AUV_x/*_frames` 目录是被降采样+重编号后的残片，不能用它做编号查找。

本脚本：扫一遍裸流（MJPEG 按字节无损切割 / mp4 顺序解码），只对"被标注号命中"
的序号做预处理链路复现 + 缩略图比对，输出 runs/prov/stream_match_<dataset>.csv。
不写任何帧到磁盘。

用法：
    conda run -n yolov8 python scripts/1_prepare/provenance/provenance_stream.py \
        --dataset data/AUV_4/PNP.kpt4.yolov8 --dataset data/AUV_3/PNP.v1i.yolov8
    # 未命中者再在序号 ±W 窗口里找（补偿裸流丢帧造成的相移）
    ... --window 3 --only-unmatched runs/prov/stream_match_PNP.kpt4.yolov8.csv
"""
from __future__ import annotations

import argparse
import csv
import mmap
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "1_prepare"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import prepare_frames as pf  # noqa: E402

SOI = b"\xff\xd8\xff"
AOLD = "configs/backup/front_camera_AUV1_water_fx782.yaml"
WB, CLAHE, GAMMA = [1.0, 1.05, 1.15], 0.5, 0.85
NTHUMB, NBIG = 48, 128

# 裸流 / 帧目录：tag -> 路径（mjpeg=字节切割，video=cv2 解码，dir=现成全量目录）
SOURCES = {
    "AUV_1": ("video", "data/AUV_1/rec_front.mp4"),
    "AUV_2": ("mjpeg", "data/AUV_2/auv_20260910_172208.mjpeg"),
    "AUV_3": ("mjpeg", "data/AUV_3/auv_20260911_201754.mjpeg"),
    "AUV_4": ("mjpeg", "data/AUV_4/auv_4.mjpeg"),
    "AUV_4dir": ("dir", "data/AUV_4/auv_4_frames"),
    "AUV_1board": ("dir", "data/AUV_1/board"),
    "AUV_2sel": ("dir", "data/AUV_2/selected_2000"),
    "AUV_3sel": ("dir", "data/AUV_3/selected_2000"),
}

_MAPS = {}


def maps_for(w, h):
    if (w, h) not in _MAPS:
        _MAPS[(w, h)] = pf.calibration_maps(AOLD, w, h)
    return _MAPS[(w, h)]


# 链路变体：区分「去畸变(A-old) / 不去畸变」×「新增强参数 / 旧增强参数」
OLD_WB, OLD_CLAHE, OLD_GAMMA = [1.15, 1.0, 0.9], 2.0, 1.0
CHAIN = "aold"
OFFSETS: dict[str, int] = {}


def chain(raw):
    """复现当年建集链路。CHAIN 取值见 --chain。"""
    h, w = raw.shape[:2]
    f = raw
    if CHAIN.startswith("aold"):
        f = cv2.remap(raw, *maps_for(w, h), cv2.INTER_LINEAR)
    if (w, h) != (640, 640):
        f = cv2.resize(f, (640, 640), interpolation=cv2.INTER_LINEAR)
    if CHAIN.endswith("noenh"):
        return f
    if CHAIN.endswith("_old"):
        return pf.enhance(f, OLD_WB, OLD_CLAHE, OLD_GAMMA)
    return pf.enhance(f, WB, CLAHE, GAMMA)


def thumbs(bgr640):
    g = cv2.cvtColor(bgr640, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(g, (NBIG, NBIG), interpolation=cv2.INTER_AREA).astype(np.int16)
    small = cv2.resize(big.astype(np.uint8), (NTHUMB, NTHUMB),
                       interpolation=cv2.INTER_AREA).astype(np.int16)
    return small.ravel(), big.ravel()


def load_query(path):
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        return None
    if im.shape[:2] != (640, 640):
        im = cv2.resize(im, (640, 640), interpolation=cv2.INTER_LINEAR)
    return thumbs(im)


def mad(q_big, cand_big):
    return float(np.abs(q_big - cand_big).mean())


def scan_mjpeg(path, need, collector, tag):
    """need: {序号: [query_idx,...]}；collector(qi, tag, idx, mad48, mad128)"""
    p = Path(path)
    with open(p, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        offsets, pos = [], 0
        while (i := mm.find(SOI, pos)) >= 0:
            offsets.append(i)
            pos = i + 3
        total = len(offsets)
        print(f"  {tag}: 裸流共 {total} 帧，需查 {len(need)} 个序号", flush=True)
        hit = 0
        for idx in sorted(need):
            if idx < 1 or idx > total:
                continue
            end = offsets[idx] if idx < total else len(mm)
            buf = np.frombuffer(mm[offsets[idx - 1]:end], np.uint8)
            raw = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if raw is None:
                continue
            s, b = thumbs(chain(raw))
            for qi in need[idx]:
                collector(qi, tag, idx, s, b)
            hit += 1
        mm.close()
    return hit


def scan_video(path, need, collector, tag, window=0):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  {tag}: 打不开 {path}")
        return 0
    want = set()
    for idx in need:
        for d in range(-window, window + 1):
            want.add(idx + d)
    i, hit = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i in want:
            s, b = thumbs(chain(frame))
            for base in range(i - window, i + window + 1):
                for qi in need.get(base, ()):
                    collector(qi, tag, i, s, b)
            hit += 1
    cap.release()
    print(f"  {tag}: 视频共 {i} 帧，命中待查序号 {hit} 个", flush=True)
    return hit


def scan_dir(path, need, collector, tag):
    hit = 0
    for idx in sorted(need):
        fp = Path(path) / f"frame_{idx:06d}.jpg"
        if not fp.exists():
            continue
        raw = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if raw is None:
            continue
        s, b = thumbs(chain(raw))
        for qi in need[idx]:
            collector(qi, tag, idx, s, b)
        hit += 1
    print(f"  {tag}: 目录内命中待查序号 {hit} 个", flush=True)
    return hit


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", required=True)
    ap.add_argument("--sources", default="AUV_1,AUV_2,AUV_3,AUV_4dir",
                    help="逗号分隔；AUV_4dir 用现成全量目录，AUV_4 走裸流")
    ap.add_argument("--max-mad", type=float, default=2.0)
    ap.add_argument("--chain", default="aold",
                    choices=["aold", "nound", "aold_old", "nound_old", "aold_noenh", "nound_noenh"],
                    help="建集链路变体：aold=去畸变(标定A)；nound=不去畸变；_old=旧增强参数")
    ap.add_argument("--unmatched-from", default=None,
                    help="只处理该结果 CSV 里 accept==0 的图（复检用）")
    ap.add_argument("--offset", action="append", default=[],
                    help="该源的「标注号→流内序号」偏移，如 AUV_1=419（可重复）")
    ap.add_argument("--window", type=int, default=0,
                    help=">0 时对每个序号在 ±window 内搜索（仅 video/mjpeg 通用）")
    a = ap.parse_args()

    global CHAIN, OFFSETS
    CHAIN = a.chain
    for spec in a.offset:
        k, v = spec.split("=")
        OFFSETS[k] = int(v)
    if OFFSETS:
        print(f"序号偏移: {OFFSETS}")
    only = None
    if a.unmatched_from:
        only = {r["img"] for r in csv.DictReader(open(a.unmatched_from))
                if r.get("accept") == "0"}
        print(f"仅复检 {len(only)} 张未命中图，链路={CHAIN}")
    outdir = PROJECT_ROOT / "runs" / "prov"
    outdir.mkdir(parents=True, exist_ok=True)
    srcs = a.sources.split(",")

    for ds in a.dataset:
        imgs = sorted(Path(ds).glob("*/images/*.jpg"))
        qs, need = [], {}
        for p in imgs:
            if only is not None and p.name not in only:
                continue
            m = re.search(r"frame_(\d+)", p.name)
            if not m:
                continue
            q = load_query(p)
            if q is None:
                continue
            qi = len(qs)
            qs.append(dict(path=p, num=int(m.group(1)), split=p.parent.parent.name,
                           qs=q[0], qb=q[1], best=None))
            need.setdefault(qs[qi]["num"], []).append(qi)
        if not qs:
            continue
        print(f"\n== {ds}: {len(qs)} 张，{len(need)} 个不同序号 ==")

        def collector(qi, tag, idx, s, b):
            d = float(np.abs(qs[qi]["qs"] - s).sum())
            m = mad(qs[qi]["qb"], b)
            if qs[qi]["best"] is None or m < qs[qi]["best"][2]:
                qs[qi]["best"] = (tag, idx, m, d)

        t0 = time.time()
        for tag in srcs:
            kind, path = SOURCES[tag]
            off = OFFSETS.get(tag, 0)
            sneed = {n + off: qis for n, qis in need.items()} if off else need
            if kind == "mjpeg":
                scan_mjpeg(path, sneed, collector, tag)
            elif kind == "video":
                scan_video(path, sneed, collector, tag, a.window)
            else:
                scan_dir(path, sneed, collector, tag)
        print(f"  扫描耗时 {time.time()-t0:.0f}s")

        rows, nacc = [], 0
        for q in qs:
            b = q["best"]
            acc = int(b is not None and b[2] < a.max_mad)
            nacc += acc
            rows.append(dict(img=q["path"].name, split=q["split"], num_label=q["num"],
                             src=b[0] if b else "", num_src=b[1] if b else "",
                             mad128=round(b[2], 3) if b else "",
                             d48=int(b[3]) if b else "", accept=acc))
        suffix = "" if a.chain == "aold" else "_" + a.chain
        out = outdir / f"stream_match_{Path(ds).name}{suffix}.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"命中 {nacc}/{len(qs)} = {100*nacc/max(len(qs),1):.1f}%  → {out.relative_to(PROJECT_ROOT)}")
        import collections
        print("  命中来源:", collections.Counter(r["src"] for r in rows if r["accept"]))


if __name__ == "__main__":
    main()
