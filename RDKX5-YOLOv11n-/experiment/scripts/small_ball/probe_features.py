#!/usr/bin/env python3
"""给候选算判别特征（为"把 CV 推到极致"做依据，不是拍脑袋调参）。

已有判据：cov(圆内红占比)、circ(圆度)、hull_fill(凸性)、arc(边界张角)、rms_rel(拟合残差/半径)、
         aspect(最小外接矩形伸长比)、score
本脚本再加几个**更贴合"球是实心圆、误检是反光/矩形/色斑"**的特征：

  rim_cov   圆环带(0.72r~1.0r)内的红占比。实心球≈1.0；拟合圆罩住矩形/色斑时**环带**先掉 ⇒ 比 cov 更灵
  core_frac 圆内"强红像素"(dom ≥ core_dom)占比。实心深红球高；水反光/浅色斑低
  h_med     圆内色相中位数（球实测≈176；偏橙/偏粉的误检会偏移）
  s_med/v_med 圆内饱和度/亮度中位数（反光斑偏亮、S 偏低）
  hough_ok  ROI 内 HoughCircles 是否也认出一个圆（refine 机制；矩形/色斑通常认不出）
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_red_ball import Params, detect, red_mask

BASE = dict(dom_min=30, use_rel=True, rel_min=0.30, s_min=40, v_min=60, close_k=9, open_k=5,
            fill_hole_frac=0.65, min_area=30, circ_min=0.30, hull_fill_min=0.55, cov_min=0.35,
            min_arc_deg=150.0, r_min=25.0, score_min=0.0, fast_mask=True)


def feats(img, mask, hsv, d, p, core_dom=80):
    h, w = mask.shape
    x1, y1 = max(0, int(d.cx - d.r) - 1), max(0, int(d.cy - d.r) - 1)
    x2, y2 = min(w, int(d.cx + d.r) + 2), min(h, int(d.cy + d.r) + 2)
    if x2 <= x1 or y2 <= y1:
        return None
    yy, xx = np.ogrid[y1:y2, x1:x2]
    dd = np.sqrt((xx - d.cx) ** 2 + (yy - d.cy) ** 2)
    disc = dd <= d.r
    ring = (dd >= 0.72 * d.r) & (dd <= d.r)
    sub = mask[y1:y2, x1:x2] > 0
    n = int(disc.sum())
    if n == 0:
        return None
    b, g, r = cv2.split(img[y1:y2, x1:x2])
    dom = cv2.subtract(r, cv2.max(g, b))
    hs = hsv[y1:y2, x1:x2]
    out = dict(rim_cov=float(sub[ring].mean()) if ring.any() else 0.0,
               core_frac=float((dom[disc] >= core_dom).mean()),
               h_med=float(np.median(hs[..., 0][disc])),
               s_med=float(np.median(hs[..., 1][disc])),
               v_med=float(np.median(hs[..., 2][disc])))
    # Hough 复核：在 ROI 里独立找圆，看能否复现这个圆
    rr = int(max(6, d.r * 0.5))
    sy0, sx0 = max(0, int(d.cy) - rr), max(0, int(d.cx) - rr)
    sub_r = dom[sy0 - y1:sy0 - y1 + 2 * rr, sx0 - x1:sx0 - x1 + 2 * rr] if False else None
    reg = dom[max(0, sy0 - y1):max(0, sy0 - y1) + 2 * rr, max(0, sx0 - x1):max(0, sx0 - x1) + 2 * rr]
    ok = 0
    if reg.size:
        reg = cv2.GaussianBlur(reg, (5, 5), 0)
        cs = cv2.HoughCircles(reg, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(max(8, d.r)),
                              param1=100, param2=26,
                              minRadius=int(max(5, d.r * 0.6)), maxRadius=int(max(8, d.r * 1.5)))
        if cs is not None and len(cs[0]):
            hx, hy, hr = cs[0][0]
            dc = np.hypot((hx + sx0) - d.cx, (hy + sy0) - d.cy)
            ok = int(dc <= 0.5 * d.r and 0.6 <= hr / max(d.r, 1) <= 1.6)
    out["hough_ok"] = ok
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    p = Params(**BASE)
    files = sorted(Path(a.src).glob("*.jpg"))
    if a.limit:
        files = files[:a.limit]
    rows = []
    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask, _ = red_mask(img, p)
        for d in detect(img, p):
            ft = feats(img, mask, hsv, d, p)
            if ft is None:
                continue
            rows.append(dict(file=f.name, cx=round(d.cx, 1), cy=round(d.cy, 1), r=round(d.r, 1),
                             score=round(d.score, 3), cov=round(d.cov, 3), circ=round(d.circ, 3),
                             hull_fill=round(d.area / max(1, d.area), 3), arc=round(d.arc_deg, 0),
                             rms_rel=round(d.rms_rel, 3), aspect=round(d.aspect, 2), **
                             {k: round(v, 3) for k, v in ft.items()}))
        if (i + 1) % 100 == 0:
            print("  … %d/%d 候选 %d" % (i + 1, len(files), len(rows)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("✅ %d 个候选 → %s" % (len(rows), a.out))


if __name__ == "__main__":
    main()
