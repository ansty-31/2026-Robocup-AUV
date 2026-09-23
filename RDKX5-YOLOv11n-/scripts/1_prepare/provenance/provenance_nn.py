#!/usr/bin/env python3
"""
裸流全帧内容最近邻溯源（`--step 1` 全量索引）

什么时候需要它
--------------
`provenance_stream.py` 按【裸流帧序号】直达，能解开绝大多数标注图。但当某个来源组
被 `prepare_frames.py` 重新编号过（文件名变成 1..N 的连续序号）时，标注号就不再是
裸流序号，只能靠**内容最近邻**在全量帧里找。

本脚本对每条裸流做一遍 step=1 的完整解码 → A-old 链路 → 48x48 灰度缩略图索引
（AUV_1 为 mp4 顺序解码；其余为裸 MJPEG 字节切割，可多进程），
再对未命中标注图做 NN + 128x128 复核。

用法：
    python scripts/1_prepare/provenance/provenance_nn.py \
        --dataset data/AUV_4/PNP.kpt4.yolov8 \
        --queries runs/prov/stream_match_PNP.kpt4.yolov8.csv   # 只查该 CSV 里 accept==0
        --sources AUV_1,AUV_2,AUV_3,AUV_4
输出：runs/prov/nn_match_<dataset>.csv
"""
from __future__ import annotations

import argparse
import csv
import mmap
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
import provenance_stream as ps  # noqa: E402

SOURCES = {
    "AUV_1": ("video", "data/AUV_1/rec_front.mp4"),
    "AUV_2": ("mjpeg", "data/AUV_2/auv_20260910_172208.mjpeg"),
    "AUV_3": ("mjpeg", "data/AUV_3/auv_20260911_201754.mjpeg"),
    "AUV_4": ("mjpeg", "data/AUV_4/auv_4.mjpeg"),
}
N = 48
_M = {}


def _init(chain: str):
    ps.CHAIN = chain
    _M.clear()


def _thumbs(raw):
    f = ps.chain(raw)
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (N, N), interpolation=cv2.INTER_AREA).astype(np.uint8).ravel()


def _mjpeg_chunk(job):
    """job = (path, [indices 1-based])；返回 (idx数组, thumb数组)"""
    path, idxs = job
    out = np.zeros((len(idxs), N * N), np.uint8)
    keep = []
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        offsets, pos = [], 0
        while (i := mm.find(ps.SOI, pos)) >= 0:
            offsets.append(i)
            pos = i + 3
        total = len(offsets)
        for k, ix in enumerate(idxs):
            if ix < 1 or ix > total:
                continue
            end = offsets[ix] if ix < total else len(mm)
            raw = cv2.imdecode(np.frombuffer(mm[offsets[ix - 1]:end], np.uint8), cv2.IMREAD_COLOR)
            if raw is None:
                continue
            out[k] = _thumbs(raw)
            keep.append(ix)
        mm.close()
    return np.array(keep), out[:len(keep)]


def _video_index(path, step_report=2000):
    cap = cv2.VideoCapture(str(path))
    thumbs, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        thumbs.append(_thumbs(fr))
        if i % step_report == 0:
            print(f"    … AUV_1 已索引 {i} 帧", flush=True)
    cap.release()
    return np.stack(thumbs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--queries", required=True, help="上一个结果 CSV，只查其中 accept==0")
    ap.add_argument("--sources", default="AUV_1,AUV_2,AUV_3,AUV_4")
    ap.add_argument("--chain", default="aold")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    ps.CHAIN = a.chain

    want = {r["img"] for r in csv.DictReader(open(a.queries)) if r.get("accept") == "0"}
    ds = Path(a.dataset)
    queries = []
    for p in sorted(ds.glob("*/images/*.jpg")):
        if p.name not in want:
            continue
        q = ps.load_query(p)
        if q is None:
            continue
        num = int(re.search(r"frame_(\d+)", p.name).group(1))
        queries.append(dict(path=p, num=num, split=p.parent.parent.name,
                            q48=np.asarray(q[0], np.uint8), qb=q[1]))
    print(f"待查 {len(queries)} 张（链路 {a.chain}）")
    if not queries:
        return

    Q48 = np.stack([q["q48"] for q in queries]).astype(np.int16)
    rows = [dict(img=q["path"].name, split=q["split"], num_label=q["num"],
                 src="", num_src="", mad128="", d48="", accept=0) for q in queries]

    for tag in a.sources.split(","):
        kind, path = SOURCES[tag]
        t0 = time.time()
        if kind == "video":
            print(f"  {tag}: 全量顺序解码索引…", flush=True)
            S = _video_index(path).astype(np.int16)
            ixs = np.arange(1, len(S) + 1)
        else:
            with open(path, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                total = 0
                pos = 0
                while (i := mm.find(ps.SOI, pos)) >= 0:
                    total += 1
                    pos = i + 3
                mm.close()
            print(f"  {tag}: 全量 {total} 帧，多进程索引…", flush=True)
            bounds = np.linspace(1, total + 1, a.workers + 1).astype(int)
            jobs = [(path, list(range(bounds[k], bounds[k + 1]))) for k in range(a.workers)]
            with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                                     initargs=(a.chain,)) as ex:
                res = list(ex.map(_mjpeg_chunk, jobs))
            ixs = np.concatenate([r[0] for r in res if len(r[0])])
            S = np.concatenate([r[1] for r in res if len(r[0])]).astype(np.int16)
        print(f"    索引 {len(ixs)} 帧，{time.time()-t0:.0f}s", flush=True)

        # NN: 分块算 48x48 L1
        order = np.argsort(ixs)
        ixs, S = ixs[order], S[order]
        for qi in range(len(queries)):
            d = np.abs(S - Q48[qi]).sum(1)
            j = int(np.argmin(d))
            d1 = int(d[j])
            if d1 < (int(rows[qi]["d48"]) if rows[qi]["d48"] != "" else 10**9):
                rows[qi].update(src=tag, num_src=int(ixs[j]), d48=d1)
        print(f"    完成 {tag}", flush=True)

    out = PROJECT_ROOT / "runs" / "prov" / f"nn_match_{ds.name}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    hit = sum(1 for r in rows if r["d48"] != "" and int(r["d48"]) < 300)
    print(f"d48<300（=平均差 <0.13/255，真命中量级）{hit}/{len(rows)} → {out.relative_to(PROJECT_ROOT)}")
    import collections
    print("  命中来源:", collections.Counter(r["src"] for r in rows if r["d48"] != "" and int(r["d48"]) < 300))


if __name__ == "__main__":
    main()
