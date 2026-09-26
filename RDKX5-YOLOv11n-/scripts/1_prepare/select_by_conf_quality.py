#!/usr/bin/env python3
"""按模型估计筛图：置信度分层 + 质量分层，挑 N 张并硬链接到输出目录。

用法：
    python scripts/1_prepare/select_by_conf_quality.py <图片目录> --out <输出目录> \
        [--weights weights/new/yolo11n-pose.pt] [--n 1000] [--gap 12] [--weights2 <pt>]
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "2_train"))
from eval_benchmark import instance_quality  # noqa: E402

CAPS = {"single_ok": 400, "partial": 350, "close_up": 250, "partial3": 60, "no_det": 40}
# 类别内的最小帧号间隔：单门高质量成段出现，间隔放小一点才能多选
GAPS = {"single_ok": 5}
DEFAULT_GAP = 10


def frame_no(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else -1


def measure(res) -> dict:
    """取最高分实例的指标；无检出则 conf=0。"""
    k = 0 if res.boxes is None else len(res.boxes)
    out = dict(n_det=k, conf=0.0, n_hi=0, vmin=0.0, area_pct=0.0, edge=0)
    if not k or res.keypoints is None or len(res.keypoints) != k:
        return out
    kd = res.keypoints.data.cpu().numpy()
    bx = res.boxes.xyxy.cpu().numpy()
    cf = res.boxes.conf.cpu().numpy()
    i = int(cf.argmax())
    q = instance_quality(kd[i, :, :2], kd[i, :, 2], bx[i])
    x1, y1, x2, y2 = bx[i]
    out.update(conf=round(float(cf[i]), 3), n_hi=q["n_hi"],
               vmin=round(float(kd[i, :, 2].min()), 3),
               area_pct=round(float((x2 - x1) * (y2 - y1)) / (640 * 640) * 100, 2),
               edge=int((x1 <= 2) + (y1 <= 2) + (x2 >= 638) + (y2 >= 638)))
    return out


def infer(files: list[Path], weights: str, conf: float, tag: str) -> dict:
    from ultralytics import YOLO
    lst = Path("/tmp") / f"_select_{tag}.txt"
    lst.write_text("\n".join(str(f.absolute()) for f in files), encoding="utf-8")
    model = YOLO(weights)
    out = {}
    for res in model.predict(source=str(lst), imgsz=640, conf=conf, device="0",
                             batch=16, max_det=10, stream=True, verbose=False):
        out[Path(res.path).name] = measure(res)
    return out


def classify(r: dict) -> str:
    if r["n_det"] == 0:
        return "no_det"
    if (r["area_pct"] >= 25 or r["edge"] >= 2) and r["n_hi"] >= 2:
        return "close_up"
    if r["n_hi"] <= 2:
        return "partial"
    if r["n_hi"] == 3:
        return "partial3"
    if r["n_det"] == 1:
        return "single_ok"
    return "multi_ok"


def pick(rows: list[dict], n: int, gap: int) -> list[dict]:
    """按类别额度挑；同类内按给定顺序；全局帧号间隔 >= gap 去近重复。"""
    chosen: list[dict] = []

    def take(cands: list[dict], cap: int) -> None:
        got = 0
        for r in cands:
            if len(chosen) >= n or got >= cap:
                return
            g = GAPS.get(r["cat"], gap)
            if all(abs(r["fno"] - c["fno"]) >= g for c in chosen):
                chosen.append(r)
                got += 1

    by = lambda cat: [r for r in rows if r["cat"] == cat]
    take(sorted(by("single_ok"), key=lambda r: -r["conf"]), CAPS["single_ok"])
    take(sorted(by("partial"), key=lambda r: -r["area_pct"]), CAPS["partial"])
    take(sorted(by("close_up"), key=lambda r: -r["area_pct"]), CAPS["close_up"])
    take(sorted(by("partial3"), key=lambda r: -r["area_pct"]), CAPS["partial3"])
    take(sorted(by("no_det"), key=lambda r: -r["fno"]), CAPS["no_det"])
    left = [r for r in rows if r not in chosen]
    pref = [r for r in left if r["cat"] in ("single_ok", "partial", "close_up", "partial3")]
    take(sorted(pref, key=lambda r: r["conf"]), n - len(chosen))    # 低置信补满（低→高）
    take(sorted([r for r in left if r not in chosen], key=lambda r: r["conf"]), n)
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="图片目录（递归找 jpg/png）")
    ap.add_argument("--out", required=True, help="选中图片的输出目录")
    ap.add_argument("--weights", default="weights/new/yolo11n-pose.pt")
    ap.add_argument("--weights2", default=None, help="第二个权重，只记录置信度供对照")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--conf", type=float, default=0.25, help="估计用的检出阈值")
    ap.add_argument("--gap", type=int, default=DEFAULT_GAP, help="选中帧之间的最小帧号间隔")
    ap.add_argument("--reuse", action="store_true", help="复用已存在的全量估计表，跳过推理")
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    files = sorted([p for p in src.rglob("*") if p.suffix.lower() in (".jpg", ".png")],
                   key=lambda p: p.name)
    if not files:
        raise SystemExit(f"{src} 下没有图片")
    path_of = {f.name: f for f in files}
    out.mkdir(parents=True, exist_ok=True)

    est = out.parent / f"{src.name}_estimate.csv"
    if a.reuse and est.exists():
        rows = []
        for r in csv.DictReader(open(est, encoding="utf-8")):
            for k in ("n_det", "n_hi", "edge"):
                r[k] = int(r[k])
            for k in ("conf", "vmin", "area_pct"):
                r[k] = float(r[k])
            r["fno"] = frame_no(r["name"])
            r["cat"] = classify(r)
            rows.append(r)
        print(f"复用估计表 {est}（{len(rows)} 行）")
    else:
        m1 = infer(files, a.weights, a.conf, "w1")
        m2 = infer(files, a.weights2, a.conf, "w2") if a.weights2 else {}
        rows = []
        for f in files:
            r = dict(m1[f.name])
            r.update(name=f.name, fno=frame_no(f.name),
                     conf2=m2.get(f.name, {}).get("conf", ""))
            r["cat"] = classify(r)
            rows.append(r)

    chosen = pick(rows, a.n, a.gap)
    chosen.sort(key=lambda r: r["fno"])
    for r in chosen:
        dst = out / r["name"]
        if not dst.exists():
            os.link(path_of[r["name"]], dst)

    cols = ["name", "cat", "conf", "conf2", "n_det", "n_hi", "vmin", "area_pct", "edge"]
    man = out.parent / f"manifest_{out.name}.csv"
    for target, data in ((man, chosen), (est, sorted(rows, key=lambda r: r["fno"]))):
        with open(target, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in data:
                w.writerow({k: r[k] for k in cols})

    print(f"源 {len(files)} 张 → 选中 {len(chosen)} 张 → {out}")
    print("  全量估计:", dict(sorted(Counter(r["cat"] for r in rows).items())))
    print("  选中构成:", dict(sorted(Counter(r["cat"] for r in chosen).items())))
    if chosen:
        print("  帧号跨度:", chosen[0]["fno"], "~", chosen[-1]["fno"])
    print("  清单:", man)
    print("  全量估计表:", est)


if __name__ == "__main__":
    main()
