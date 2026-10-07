#!/usr/bin/env python3
"""compare_corners.py — 手动标的角点 vs 模型预测：逐角点误差、顺序、可见性、漏检。

用法：
  python scripts/1_prepare/compare_corners.py \
      --batch data/derived/landmark_check_150 \
      --manual data/derived/landmark_check_150/manual_corners.json \
      --pred data/derived/landmark_check_150/predictions.json \
      --out data/derived/landmark_check_150/diff

配对方式：手动四边形按"包围盒 IoU 最大"匹配到模型实例（角点差值只在配上的实例上算）。
输出：diff/per_frame.csv、diff/summary.txt、diff/overlay/<组>/<帧>.jpg（手动=绿、模型=红）
"""
from __future__ import annotations
import argparse, csv, json, pathlib
import cv2
import numpy as np

NAMES = ["TL", "TR", "BR", "BL"]


def bbox_of(pts: np.ndarray):
    return [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())]


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    it = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - it
    return it / u if u > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--manual", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    B = pathlib.Path(a.batch).resolve()
    man = json.loads(pathlib.Path(a.manual).read_text(encoding="utf-8"))
    pred = json.loads(pathlib.Path(a.pred).read_text(encoding="utf-8"))
    out = pathlib.Path(a.out); (out / "overlay").mkdir(parents=True, exist_ok=True)

    rows = []
    for rel, pts in sorted(man.items()):
        M = np.array(pts, np.float64)
        insts = pred.get(rel, {}).get("instances", [])
        if not insts:
            rows.append(dict(frame=rel, matched=False, note="模型无检出", group=rel.split("/")[0]))
            continue
        ious = [iou(bbox_of(np.array(d["corners"])[:, :2]), bbox_of(M)) for d in insts]
        j = int(np.argmax(ious))
        d = insts[j]
        P = np.array(d["corners"], np.float64)          # (4,3) x,y,v
        # 顺序：允许"模型 L/R 与手动相反"（背面门）——取两种排列里误差小的那个
        cand = [P[:, :2], P[[1, 0, 3, 2], :2]]
        errs = [np.linalg.norm(c - M, axis=1) for c in cand]
        k = int(np.argmin([e.mean() for e in errs]))
        per = errs[k]
        row = dict(frame=rel, group=rel.split("/")[0], matched=True, n_det=len(insts), hit_instance=j + 1,
                   iou=round(ious[j], 3), flipped=bool(k == 1))
        for n, e in zip(NAMES, per):
            row[f"{n}_err"] = round(float(e), 2)
            row[f"{n}_v"] = float(d["corners"][NAMES.index(n)][2])
        row["mean_err"] = round(float(per.mean()), 2)
        row["max_err"] = round(float(per.max()), 2)
        rows.append(row)

        # 叠加图
        p = B / rel
        img = cv2.imread(str(p))
        if img is not None:
            for idx, (x, y) in enumerate(P[:, :2]):
                cv2.drawMarker(img, (int(x), int(y)), (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
                cv2.putText(img, f"{NAMES[idx]}", (int(x) + 6, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            for idx, (x, y) in enumerate(M):
                cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
                cv2.putText(img, f"{NAMES[idx]}", (int(x) + 6, int(y) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(img, f"{rel}  mean {per.mean():.1f}px  max {per.max():.1f}px  "
                             f"IoU {ious[j]:.2f}{'  [LR-flip]' if k == 1 else ''}",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, f"{rel}  mean {per.mean():.1f}px  max {per.max():.1f}px  "
                             f"IoU {ious[j]:.2f}{'  [LR-flip]' if k == 1 else ''}",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            d2 = out / "overlay" / rel.split("/")[0]; d2.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d2 / pathlib.Path(rel).name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    with open(out / "per_frame.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    ok = [r for r in rows if r.get("matched")]
    L = [f"手动标注 {len(man)} 张；与模型配上 {len(ok)} 张", ""]
    for g in sorted({r["group"] for r in rows}):
        gg = [r for r in ok if r["group"] == g]
        if not gg:
            L.append(f"[{g}] 无配对"); continue
        me = np.array([r["mean_err"] for r in gg]); mx = np.array([r["max_err"] for r in gg])
        L += [f"[{g}] n={len(gg)}  逐角点平均误差 中位 {np.median(me):.2f}px  p90 {np.percentile(me, 90):.2f}  "
              f"最大 {me.max():.2f} | 单点最大误差 中位 {np.median(mx):.2f}  最大 {mx.max():.2f}",
              f"       IoU 中位 {np.median([r['iou'] for r in gg]):.3f}   "
              f"L/R 翻转 {sum(1 for r in gg if r['flipped'])}/{len(gg)}   "
              f"模型判为不可见(v<0.5) 的点 {sum(1 for r in gg for n in NAMES if r[f'{n}_v'] < 0.5)}/{4 * len(gg)}"]
    L += ["", "误差最大的 10 帧："]
    L += [f"  {r['frame']:<46} mean {r['mean_err']:>6.2f}  max {r['max_err']:>6.2f}  IoU {r['iou']:.2f}"
          for r in sorted(ok, key=lambda r: -r["mean_err"])[:10]]
    L += ["", f"模型无检出但手动有 {sum(1 for r in rows if not r.get('matched'))} 张"]
    txt = "\n".join(L)
    (out / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"\n📄 {out/'per_frame.csv'}  ·  {out/'summary.txt'}  ·  叠加图 {out/'overlay'}/")


if __name__ == "__main__":
    main()
