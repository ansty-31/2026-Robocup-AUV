#!/usr/bin/env python3
"""
把溯源命中的原始帧落地到本地缓存（domain 映射的唯一输入）

为什么需要
----------
· AUV_2/AUV_3/AUV_4 在裸 MJPEG 里，按 SOI 字节偏移切出来是无损的；
· AUV_1 在 `rec_front.mp4` 里（重编码件），必须**顺序解码**才能帧准——
  `cap.set(CAP_PROP_POS_FRAMES, n)` 在 mp4 上不保证逐帧精确；
· B/C/D 三个域要反复渲染同一批帧，落地一次避免重复解码。

输出：data/mapped/raw/<src>/frame_<idx:06d>.jpg（原分辨率 1280x720）

用法：
    python scripts/1_prepare/provenance/materialize_raw.py --prov runs/prov/provenance_final_PNP.kpt4.yolov8.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "1_prepare"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "1_prepare" / "pose"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from map_pose_dataset import SRC_PATH, read_raw, stream_offsets  # noqa: E402

OUT = PROJECT_ROOT / "data/mapped/raw"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prov", default="runs/prov/provenance_final_PNP.kpt4.yolov8.csv")
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(PROJECT_ROOT / a.prov)) if r["accept"] == "1"]
    need = defaultdict(set)
    for r in rows:
        need[r["src"]].add(int(r["num_src"]))
    print(f"需落地 {len(rows)} 张（去重后 {sum(len(v) for v in need.values())} 帧）")
    print("  按源:", {k: len(v) for k, v in need.items()})

    for src, idxs in need.items():
        d = OUT / src
        d.mkdir(parents=True, exist_ok=True)
        idxs = sorted(idxs)
        have = {int(p.stem.split("_")[1]) for p in d.glob("frame_*.jpg")}
        todo = [i for i in idxs if i not in have]
        if not todo:
            print(f"  {src}: 已全部缓存")
            continue
        kind, path = SRC_PATH[src]
        n_ok = 0
        if kind == "video":
            # 顺序解码，1-based 计数
            want = set(todo)
            cap = cv2.VideoCapture(path)
            i = 0
            while want:
                ok, fr = cap.read()
                if not ok:
                    break
                i += 1
                if i in want:
                    want.discard(i)
                    cv2.imwrite(str(d / f"frame_{i:06d}.jpg"), fr,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    n_ok += 1
            cap.release()
            if want:
                print(f"  {src}: ⚠️ {len(want)} 帧未取到（超出视频长度）")
        else:
            for i in todo:
                fr = read_raw(src, i)
                if fr is None:
                    continue
                cv2.imwrite(str(d / f"frame_{i:06d}.jpg"), fr,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                n_ok += 1
        print(f"  {src}: 新落地 {n_ok} 帧 → {d.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
