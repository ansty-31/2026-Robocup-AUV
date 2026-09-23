#!/usr/bin/env python3
"""
标注图 → 原始帧 的溯源匹配（Route B：把已有标注重投影到新标定域）

为什么需要它
------------
现有 2100 张标好关键点的图（data/AUV_4/PNP.kpt4.yolov8 1304 张 +
data/AUV_3/PNP.v1i.yolov8 801 张）是**在旧标定域（AUV_1 水下 fx≈782.5，
记为 A-old）里做的人工标注**。换标定（→ C, AUV_5 fx≈1207.6）或换去畸变策略
（B 域 = 完全不去畸变）时必须重新生成训练图，否则模型训练域与板端推理域
不一致。重新生成图需要两个东西：

    1) 该标注图对应的【原始 1280x720 帧】(raw)  ← 本脚本负责
    2) 该标注图的 4 个角点在图上的像素坐标（labels/*.txt，YOLO 归一化）

有了 (1)(2) 就能：raw --(新链路)--> 新域图，同时把角点坐标经 A-old 链路
的逆变换映回 raw 像素，再投影到新域图。**不需要重新标注。**

方法
----
不能拿【原始帧】直接和【标注图】比像素：标注图已经过
remap(A-old) → resize(640) → enhance，域差异（去畸变形变 + 白平衡/CLAHE/gamma）
远大于帧间差异，最近邻会稳定地指向错误帧。所以索引必须建立在**与标注图相同的
处理域**上（默认 A-old 链路），再用缩略图 L1 距离找最近邻。

判定：
    stage1  48x48 灰度缩略图全池最近邻（top-5）
    stage2  对 top-5 用 128x128 复核，取最优
    接受    128x128 平均绝对差 < --max-mad（默认 2.0/255）
    歧义度  d2/d1：真命中时 d1 极小（≈0.03），d2/d1 会很大

用法：
    # 建索引（约 2~4 分钟，结果缓存到 runs/prov/）
    python scripts/1_prepare/provenance/provenance_match.py index
    # 扫描所有标注集
    python scripts/1_prepare/provenance/provenance_match.py scan
    # 只扫一个数据集
    python scripts/1_prepare/provenance/provenance_match.py scan --dataset data/AUV_4/PNP.kpt4.yolov8

输出：runs/prov/index_<domain>.npz、runs/prov/provenance_<dataset>.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "1_prepare"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import prepare_frames as pf  # noqa: E402

DEFAULT_CALIB = "configs/backup/front_camera_AUV1_water_fx782.yaml"  # A-old 域
WB, CLAHE, GAMMA = [1.0, 1.05, 1.15], 0.5, 0.85  # 板端 vision.yaml 当前值
NTHUMB = 48
NBIG = 128

# ---------------------------------------------------------------- 索引构建

_W = {}


def _init(calib: str, domain: str, query: bool = False):
    maps = None
    if domain != "raw" and not query:
        maps = pf.calibration_maps(calib, 1280, 720)
    _W["maps"] = maps
    _W["domain"] = domain
    _W["query"] = query


def _thumb(path: str):
    # 查询图（标注图）本身已在目标域里：只能缩放取缩略图，绝不能再套一次链路
    if _W.get("query"):
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is None:
            return None
        if im.shape[:2] != (640, 640):
            im = cv2.resize(im, (640, 640), interpolation=cv2.INTER_LINEAR)
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        big = cv2.resize(g, (NBIG, NBIG), interpolation=cv2.INTER_AREA).astype(np.uint8)
        small = cv2.resize(big, (NTHUMB, NTHUMB), interpolation=cv2.INTER_AREA).astype(np.uint8)
        return small.ravel(), big.ravel()
    # 必须全分辨率读入：remap 映射是按 1280x720 生成的，先降采样会让映射越界
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None or im.shape[0] < 100:
        return None
    raw = im
    if _W["maps"] is not None:
        # 索引建在 640 域的形态：全图 remap 后 squish 到 640
        raw = cv2.resize(cv2.remap(im, *_W["maps"], cv2.INTER_LINEAR), (640, 640),
                         interpolation=cv2.INTER_LINEAR)
    else:
        raw = cv2.resize(im, (640, 640), interpolation=cv2.INTER_LINEAR)
    if _W["domain"] == "enh":
        raw = pf.enhance(raw, WB, CLAHE, GAMMA)
    g = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(g, (NBIG, NBIG), interpolation=cv2.INTER_AREA).astype(np.uint8)
    small = cv2.resize(big, (NTHUMB, NTHUMB), interpolation=cv2.INTER_AREA).astype(np.uint8)
    return small.ravel(), big.ravel()


def collect_pool_paths() -> dict[str, list[str]]:
    pools = {
        "AUV_1": "data/AUV_1/rec_front_frames",
        "AUV_2": "data/AUV_2/auv_20260910_172208_frames",
        "AUV_3": "data/AUV_3/auv_20260911_201754_frames",
        "AUV_4": "data/AUV_4/auv_4_frames",
    }
    out = {}
    for tag, d in pools.items():
        fs = sorted(Path(d).glob("*.jpg"))
        out[tag] = [str(p) for p in fs]
    out["AUV_5"] = sorted(str(p) for p in Path("data/AUV_5/raw-data").glob("gate-*/*.jpg"))
    return out


def cmd_index(a) -> None:
    t0 = time.time()
    outdir = PROJECT_ROOT / "runs" / "prov"
    outdir.mkdir(parents=True, exist_ok=True)
    dst = outdir / f"index_{a.domain}.npz"
    pool = collect_pool_paths()
    paths, tags = [], []
    for tag, lst in pool.items():
        paths += lst
        tags += [tag] * len(lst)
    print(f"池内原始帧 {len(paths)} 张（{ {k: len(v) for k, v in pool.items()} }）")
    small = np.zeros((len(paths), NTHUMB * NTHUMB), np.uint8)
    big = np.zeros((len(paths), NBIG * NBIG), np.uint8)
    ok = np.zeros(len(paths), bool)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                             initargs=(a.calibration, a.domain)) as ex:
        for i, r in enumerate(ex.map(_thumb, paths, chunksize=64)):
            if r is None:
                continue
            small[i], big[i] = r
            ok[i] = True
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(paths)}  {time.time()-t0:.0f}s", flush=True)
    print(f"索引完成 {ok.sum()}/{len(paths)}，耗时 {time.time()-t0:.0f}s")
    np.savez_compressed(dst, small=small[ok], big=big[ok],
                        paths=np.array(paths)[ok], tags=np.array(tags)[ok])
    print(f"→ {dst.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------- 扫描

def cmd_scan(a) -> None:
    idx_path = PROJECT_ROOT / "runs" / "prov" / f"index_{a.domain}.npz"
    if not idx_path.exists():
        sys.exit(f"索引不存在：{idx_path}（先跑 index）")
    z = np.load(idx_path)
    S = z["small"].astype(np.int16)
    B = z["big"].astype(np.int16)
    ipath, itag = z["paths"], z["tags"]
    print(f"索引 {len(ipath)} 帧，域={a.domain}")

    def query_thumb(p):
        r = _thumb(p)
        return r

    _init(a.calibration, a.domain, query=True)
    outdir = PROJECT_ROOT / "runs" / "prov"
    datasets = a.dataset or ["data/AUV_4/PNP.kpt4.yolov8", "data/AUV_3/PNP.v1i.yolov8"]
    for ds in datasets:
        imgs = sorted(Path(ds).glob("*/images/*.jpg"))
        if not imgs:
            print(f"跳过（无图）{ds}")
            continue
        print(f"\n== {ds}  {len(imgs)} 张 ==")
        rows, t0, nacc = [], time.time(), 0
        for n, p in enumerate(imgs):
            r = query_thumb(str(p))
            if r is None:
                rows.append(dict(img=p.name, accept=0, note="unreadable"))
                continue
            q_s, q_b = r[0].astype(np.int16), r[1].astype(np.int16)
            d1 = np.abs(S - q_s).sum(1)
            top = np.argpartition(d1, 5)[:5]
            top = top[np.argsort(d1[top])]
            db = np.abs(B[top] - q_b).sum(1) / (NBIG * NBIG)
            j = top[int(np.argmin(db))]
            order = np.sort(db)
            mad = float(order[0])
            amb = float(order[1] / order[0]) if len(order) > 1 and order[0] > 1e-6 else 999.0
            num = re.search(r"frame_(\d+)", p.name)
            num = int(num.group(1)) if num else -1
            fnum = int(re.search(r"frame_(\d+)", Path(ipath[j]).name).group(1))
            acc = int(mad < a.max_mad)
            nacc += acc
            rows.append(dict(img=p.name, split=p.parent.parent.name, pool=itag[j],
                             raw=ipath[j], num_label=num, num_raw=fnum, offset=fnum - num,
                             mad128=round(mad, 3), d48=int(d1[j]), amb=round(amb, 1),
                             accept=acc, note=""))
            if (n + 1) % 200 == 0:
                print(f"  {n+1}/{len(imgs)}  命中 {nacc}  {time.time()-t0:.0f}s", flush=True)
        out = outdir / f"provenance_{Path(ds).name}.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"命中 {nacc}/{len(imgs)} = {100*nacc/max(len(imgs),1):.1f}%  → {out.relative_to(PROJECT_ROOT)}")
        offs = {}
        for r in rows:
            if r.get("accept"):
                offs[r["offset"]] = offs.get(r["offset"], 0) + 1
        print("  命中帧号偏移分布 top8:", sorted(offs.items(), key=lambda kv: -kv[1])[:8])
        pools = {}
        for r in rows:
            if r.get("accept"):
                pools[r["pool"]] = pools.get(r["pool"], 0) + 1
        print("  命中来源池:", pools)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = dict()
    for name in ("index", "scan"):
        s = sub.add_parser(name)
        s.add_argument("--calibration", default=DEFAULT_CALIB, help="索引所用标定（A-old）")
        s.add_argument("--domain", default="enh", choices=["enh", "undist", "raw"],
                       help="enh=A-old链路(remap+resize+enhance)；undist=只remap；raw=只resize")
        s.add_argument("--workers", type=int, default=8)
        if name == "scan":
            s.add_argument("--dataset", action="append", default=None)
            s.add_argument("--max-mad", type=float, default=2.0,
                           help="128x128 平均绝对差阈值（0-255），超过判为未命中")

    a = ap.parse_args()
    if a.cmd == "index":
        cmd_index(a)
    else:
        cmd_scan(a)


if __name__ == "__main__":
    main()
