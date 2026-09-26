#!/usr/bin/env python3
"""合并多次溯源结果为一张总表（每张标注图取最优命中）

输入 CSV 来自 provenance_stream.py（按裸流序号直达）/ provenance_nn.py（内容最近邻）；
本脚本按 mad128（无则按 d48）取最优，产出
runs/prov/provenance_final_<dataset>.csv：

    img, split, num_label, chain, src, num_src, mad128, d48, accept

用法：
    python scripts/1_prepare/provenance/provenance_merge.py \
        --out runs/prov/provenance_final_PNP.kpt4.yolov8.csv \
        runs/prov/stream_match_PNP.kpt4.yolov8.csv \
        runs/prov/stream_match_PNP.kpt4.yolov8_nound.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

FIELDS = ["img", "split", "num_label", "chain", "src", "num_src", "mad128", "d48", "accept"]


def score(r):
    """越小越好：优先 mad128，其次 d48"""
    m = r.get("mad128") or ""
    d = r.get("d48") or ""
    if m != "":
        return float(m)
    if d != "":
        return float(d) / 1000.0 + 10.0   # d48 只在没有 mad128 时兜底
    return 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-mad", type=float, default=2.0)
    ap.add_argument("csvs", nargs="+")
    a = ap.parse_args()

    best = {}
    for c in a.csvs:
        stem = Path(c).stem
        chain = "aold"
        for suffix in ("_nound_old", "_aold_old", "_nound", "_aold", "_noenh"):
            if stem.endswith(suffix):
                chain = suffix[1:]
                break
        for r in csv.DictReader(open(c)):
            img = r["img"]
            rec = dict(img=img, split=r.get("split", ""), num_label=r.get("num_label", ""),
                       chain=chain, src=r.get("src", ""), num_src=r.get("num_src", ""),
                       mad128=r.get("mad128", ""), d48=r.get("d48", ""),
                       accept=r.get("accept", "0"))
            if img not in best or score(rec) < score(best[img]):
                best[img] = rec
    # 统一判定规则
    for r in best.values():
        m = r["mad128"]
        d = r["d48"]
        r["accept"] = int((m != "" and float(m) < a.max_mad) or
                          (m == "" and d != "" and int(d) < 300))
    rows = sorted(best.values(), key=lambda r: (r["split"], r["img"]))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"总表 {len(rows)} 张 → {out}")


if __name__ == "__main__":
    main()
