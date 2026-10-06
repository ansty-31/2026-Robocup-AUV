#!/usr/bin/env python3
"""夹取小球 · 出报告 sheet：两法叠图对照 + CV 边际检出复核 + 汇总面板。

生成四组东西（全部落 `output/preview/small_ball/`）：
  `compare/sheet_*.jpg`   逐帧把 **CV 圆(绿)** 与 **YOLO 框(橙)** 画在同一张图上，按帧序。
                          一列看不出来就两列并排 —— 这里选叠图，因为"两法是否指同一个球"一眼可判。
  `cvonly/sheet_*.jpg`    只 YOLO 漏、CV 检出的帧（含两档 CV 参数的分层标记）→ 用来判断
                          CV 的额外召回是**真球**还是焦散误检（这一步必须人眼看，脚本不替你下结论）。
  `neither/sheet_*.jpg`   两法都没检出的帧抽样 → 用来判断"漏检"里有多少其实是**球不在画面**。
  `summary.jpg`           指标面板：检出率、两法一致度、半径分布直方图、逐段覆盖、耗时。

用法：
    /home/ansty/anaconda3/envs/yolov8/bin/python experiment/scripts/small_ball/make_report_sheets.py \
        --src data/derived/small_ball_selected_700 \
        --cv output/preview/small_ball/cv_ball_cov0.35_a30.json \
        --cv-tight output/preview/small_ball/cv_ball_cov0.45_a40.json \
        --yolo output/preview/small_ball/yolo_ball.json \
        --outdir output/preview/small_ball
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import make_sheet  # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def load(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {r["file"]: r for r in d["records"]}, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--cv", required=True, help="CV 结果 JSON（宽档）")
    ap.add_argument("--cv-tight", default=None, help="CV 结果 JSON（严档，用于标出边际检出）")
    ap.add_argument("--yolo", required=True)
    ap.add_argument("--yolo-conf", type=float, default=0.25)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--per", type=int, default=20)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--max-sheets", type=int, default=0)
    ap.add_argument("--skip-sheets", action="store_true", help="只出 summary 面板")
    args = ap.parse_args()

    src = Path(args.src)
    outdir = Path(args.outdir)
    cv, cvmeta = load(args.cv)
    yl, _ = load(args.yolo)
    tight, _ = load(args.cv_tight) if args.cv_tight else (None, None)

    files = sorted(p.name for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)

    def yolo_best(f):
        reds = [x for x in yl[f]["det"] if x["name"] == "red_ball" and x["conf"] >= args.yolo_conf]
        if not reds:
            return None
        return max(reds, key=lambda x: (x["xyxy"][2] - x["xyxy"][0]) * (x["xyxy"][3] - x["xyxy"][1]))

    def cv_best(f, table=None):
        rec = (table or cv)[f]
        keep = rec["keep"]
        if not keep:
            return None
        return max(keep, key=lambda c: c[2])

    both, cv_only, yl_only, neither = [], [], [], []
    for f in files:
        y = yolo_best(f)
        c = cv_best(f)
        (both if (y and c) else cv_only if c else yl_only if y else neither).append(f)
    print(f"两法都命中 {len(both)} | 仅CV {len(cv_only)} | 仅YOLO {len(yl_only)} | 都无 {len(neither)}")

    def sheet_group(names, sub, draw="both", extra_caption=None):
        d = outdir / sub
        d.mkdir(parents=True, exist_ok=True)
        n_sheets = min((len(names) + args.per - 1) // args.per, args.max_sheets or 10 ** 9)
        for s in range(n_sheets):
            entries = []
            for k, f in enumerate(names[s * args.per:(s + 1) * args.per]):
                img = cv2.imread(str(src / f))
                c = cv_best(f)
                y = yolo_best(f)
                caps = [f"{s*args.per+k+1}. f{f[6:12]}"]
                if draw in ("both", "cv"):
                    for cc in cv[f]["keep"]:
                        cv2.circle(img, (int(cc[0]), int(cc[1])), int(cc[2]), (0, 255, 0), 2)
                        cv2.drawMarker(img, (int(cc[0]), int(cc[1])), (0, 255, 255),
                                       cv2.MARKER_CROSS, 12, 2)
                if draw in ("both", "yolo") and y:
                    x1, y1, x2, y2 = [int(v) for v in y["xyxy"]]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(img, f"{y['conf']:.2f}", (x1, max(14, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1, cv2.LINE_AA)
                line = []
                if c is not None:
                    line.append(f"CV r={c[2]:.0f} c=({c[0]:.0f},{c[1]:.0f})")
                else:
                    line.append("CV: none")
                if y is not None:
                    x1, y1, x2, y2 = y["xyxy"]
                    line.append(f"YOLO r~{(x2-x1)/2:.0f} {y['conf']:.2f}")
                else:
                    line.append("YOLO: none")
                caps.append(" | ".join(line))
                if extra_caption:
                    caps.append(extra_caption(f))
                entries.append((img, [c2 for c2 in caps if c2], c is not None or y is not None))
            make_sheet(entries, d / f"sheet_{s+1:02d}.jpg", cols=args.cols,
                       title=f"{sub} - {s+1}/{n_sheets}  (green circle = Plan A CV, "
                             f"orange box = Plan B YOLO, conf>={args.yolo_conf})")
        print(f"  {sub}: {n_sheets} 张")

    if not args.skip_sheets:
        sheet_group(files, "compare")

    def cv_only_cap(f):
        if tight is None:
            return ""
        t = cv_best(f, tight)
        return "CV 严档也有" if t is not None else "仅宽档检出(需人工判断)"

    if not args.skip_sheets:
        sheet_group(cv_only, "cvonly", draw="cv", extra_caption=cv_only_cap)
        sheet_group(neither, "neither", draw="none")
        if yl_only:
            sheet_group(yl_only, "yoloonly", draw="both")

    # ----------------------------- 汇总面板 -----------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rc, ry = [], []
    for f in files:
        c = cv_best(f)
        if c:
            rc.append(c[2])
        y = yolo_best(f)
        if y:
            x1, y1, x2, y2 = y["xyxy"]
            ry.append((x2 - x1) / 2)

    fig = plt.figure(figsize=(17, 10), dpi=100, constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 0.85])
    fig.suptitle("AUV small-ball grasp - two detection lines on 700 selected frames "
                 "(local PC only, no export/quantize)", fontsize=15, fontweight="bold")

    ax = fig.add_subplot(gs[0, 0])
    bars = {"YOLO11n detect\n(ball-collision .pt)": len(both) + len(yl_only),
            "CV red-ball\n(classic)": len(both) + len(cv_only)}
    ax.bar(list(bars), list(bars.values()), color=["#ff8c00", "#2ca02c"])
    for i, (k2, v) in enumerate(bars.items()):
        ax.text(i, v + 8, f"{v}/700 = {v/7:.1f}%", ha="center", fontsize=10)
    ax.set_ylim(0, 760); ax.set_ylabel("frames with a ball found")
    ax.set_title("frame-level hit rate"); ax.grid(axis="y", alpha=.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.pie([len(both), len(cv_only), len(yl_only), len(neither)],
           labels=[f"both {len(both)}", f"CV only {len(cv_only)}",
                   f"YOLO only {len(yl_only)}", f"neither {len(neither)}"],
           autopct="%1.0f%%", colors=["#2ca02c", "#1f77b4", "#ff8c00", "#999999"],
           startangle=90, textprops={"fontsize": 9})
    ax.set_title("frame-level agreement")

    ax = fig.add_subplot(gs[0, 2:])
    bins = np.logspace(np.log10(15), np.log10(700), 24)
    ax.hist(rc, bins=bins, alpha=.65, label=f"CV radius (n={len(rc)})", color="#2ca02c")
    ax.hist(ry, bins=bins, alpha=.55, label=f"YOLO box half-width (n={len(ry)})", color="#ff8c00")
    ax.set_xscale("log"); ax.set_xlabel("ball size px (radius / box half-width, log)")
    ax.set_ylabel("frames"); ax.legend(fontsize=9); ax.set_title("size coverage")
    ax.grid(alpha=.3)

    ax = fig.add_subplot(gs[1, :2])
    frames = sorted(int(f[6:12]) for f in files)
    cov_y = np.zeros(len(frames)); cov_c = np.zeros(len(frames))
    for i, fr in enumerate(frames):
        f = f"frame_{fr:06d}.jpg"
        cov_y[i] = 1 if yolo_best(f) else 0
        cov_c[i] = 1 if cv_best(f) else 0
    w = 25
    ax.step(frames, np.convolve(cov_y, np.ones(w) / w, "same"), where="mid",
            color="#ff8c00", lw=1.6, label="YOLO")
    ax.step(frames, np.convolve(cov_c, np.ones(w) / w, "same"), where="mid",
            color="#2ca02c", lw=1.6, label="CV")
    ax.set_ylim(-0.03, 1.05); ax.set_xlabel("frame index")
    ax.set_ylabel(f"moving avg ({w}f)")
    ax.set_title("temporal coverage: where each line finds / loses the ball")
    ax.legend(fontsize=9); ax.grid(alpha=.3)

    ax = fig.add_subplot(gs[1, 2:])
    seg = []
    for lo in range(2000, 3300, 100):
        names = [f for f in files if lo <= int(f[6:12]) < lo + 100]
        if not names:
            continue
        seg.append((f"{lo}", sum(1 for f in names if yolo_best(f)) / len(names),
                    sum(1 for f in names if cv_best(f)) / len(names)))
    xs = np.arange(len(seg))
    ax.bar(xs - .2, [s[1] for s in seg], .4, label="YOLO", color="#ff8c00")
    ax.bar(xs + .2, [s[2] for s in seg], .4, label="CV", color="#2ca02c")
    ax.set_xticks(xs); ax.set_xticklabels([s[0] for s in seg], rotation=90, fontsize=8)
    ax.set_ylim(0, 1.12); ax.set_ylabel("hit rate"); ax.set_xlabel("frame bucket")
    ax.set_title("hit rate by frame bucket"); ax.legend(fontsize=9); ax.grid(axis="y", alpha=.3)

    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    # 两法都命中时的尺寸一致度（CV 半径 / YOLO 框半宽）
    ratios = []
    for f in files:
        c, y = cv_best(f), yolo_best(f)
        if c and y:
            x1, y1, x2, y2 = y["xyxy"]
            ratios.append(c[2] / ((x2 - x1) / 2))
    ratios = np.array(ratios) if ratios else np.array([0.0])
    P = cvmeta["params"]
    lines = [
        f"frames: 700 of 1106  (scripts/1_prepare/select_frames.py, quality-weighted gamma=1.5)   "
        f"|  Plan B weights: weights/yolo11n.pt md5 91069d76...  = the .pt behind "
        f"output/weights/yolo11n_detect_bayese_640x640_nv12.bin",
        f"YOLO hit {len(both)+len(yl_only)}/700 = {(len(both)+len(yl_only))/7:.1f}% (conf>=0.25)   "
        f"CV hit {len(both)+len(cv_only)}/700 = {(len(both)+len(cv_only))/7:.1f}%   "
        f"both {len(both)}   CV-only {len(cv_only)}   YOLO-only {len(yl_only)}   neither {len(neither)}",
        f"size agreement (matched frames, n={len(ratios)}): CV_r / YOLO_halfwidth  "
        f"p25={np.percentile(ratios,25):.2f}  p50={np.percentile(ratios,50):.2f}  p75={np.percentile(ratios,75):.2f}"
        f"   |  CV radius p5/p50/p95 = {np.percentile(rc,[5,50,95]).round(0)}   "
        f"YOLO half-w p5/p50/p95 = {np.percentile(ry,[5,50,95]).round(0)}",
        f"CV cost: {cvmeta.get('ms_median', float('nan')):.1f} ms/frame  "
        f"({cvmeta.get('fps_cpu', float('nan')):.0f} FPS on CPU 1280x720)   "
        f"|  temporal: {cvmeta['temporal']} min_track={cvmeta.get('min_track')} "
        f"link_factor={cvmeta.get('link_factor')} (was a near no-op at these settings)",
        f"CV params: dom>={P['dom_min']} rel>={P['rel_min']} S>={P['s_min']} V>={P['v_min']} "
        f"close={P['close_k']} open={P['open_k']} area>={P['min_area']} circ>={P['circ_min']} "
        f"cov>={P['cov_min']} arc>={P['min_arc_deg']} r>={P['r_min']}",
        "not verified: board-side (RDK X5 / BPU) accuracy and FPS - nothing here was quantized or run on the board",
    ]
    ax.text(0.0, 0.92, "\n".join(lines), va="top", family="monospace", fontsize=9.2)

    fig.savefig(outdir / "summary.jpg", dpi=100)
    print(f"  summary: {outdir/'summary.jpg'}")

    # 机器可读汇总
    (outdir / "summary.json").write_text(json.dumps({
        "n_frames": len(files), "both": len(both), "cv_only": len(cv_only),
        "yolo_only": len(yl_only), "neither": len(neither),
        "yolo_hit": len(both) + len(yl_only), "cv_hit": len(both) + len(cv_only),
        "size_ratio_p50": float(np.percentile(ratios, 50)),
        "cv_radius_p50": float(np.percentile(rc, 50)),
        "yolo_halfw_p50": float(np.percentile(ry, 50)),
    }, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
