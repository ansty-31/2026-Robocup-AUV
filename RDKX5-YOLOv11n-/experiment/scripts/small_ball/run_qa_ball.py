#!/usr/bin/env python3
"""夹取小球 · 新素材的**识别结果分组 + 待人工判读**产物生成（离线，PC 侧）。

做四件事：
  1. 用当前 CV 红球检测（`cv_red_ball.py`，参数与 `output/preview/small_ball/README.md`
     §2 的最终工作点一致）逐帧检测；
  2. 按"识别到 / 没识别到"把图**分成两组**，各写一份带标注的副本：
        <out>/hit/   画了圈（绿圆+黄十字+序号）
        <out>/miss/  没画圈（原图，便于你确认"确实没球"）
  3. 出 `qa.json`（逐帧全部候选）与 `qa.csv`（一行一帧，方便你在表格里排序/筛选）；
  4. 出总览 sheet（每组一页 20 张），先扫一眼再逐张判。

**"目标只有一个"的约定**：画面里只有一个小球，所以
  * `n_det = 0` ⇒ 漏检候选（也可能是球本来不在画面 —— 判读时区分）；
  * `n_det ≥ 2` ⇒ 至少有一个是误检（CSV 里 `n_det` 列会标出来，标注图里每个候选都有序号）。

两种工作模式（`--mode`）：
  * `full`  （默认）每帧整图检测 —— **人工判读就该看这个**：一帧一个判断，不受跟踪状态影响；
  * `track` 流式 ROI 跟踪 —— 与 full 的结果逐帧比对，不一致的帧在 CSV 里标 `mode_diff=1`，
            用来量化"跟踪有没有引入误差"（本素材按 11 帧等间隔抽样，跟踪链并不连续，
            所以只看结论，不按它判读）。

用法（conda yolov8）：
    python experiment/scripts/small_ball/run_qa_ball.py \
        --src data/derived/small_ball_07_500 \
        --out output/preview/small_ball_07_qa \
        --mode full --compare-track
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import make_sheet                      # noqa: E402
from cv_ball_tracker import BallTracker            # noqa: E402
from cv_red_ball import Params, detect             # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# 与最终工作点一致（见 output/preview/small_ball/README.md §2 / §8）
WORK_PARAMS = dict(dom_min=30, use_rel=True, rel_min=0.30, s_min=40, v_min=60,
                   close_k=9, open_k=5, fill_hole_frac=0.65, min_area=30,
                   circ_min=0.30, hull_fill_min=0.55, cov_min=0.35,
                   min_arc_deg=150.0, r_min=32.0, score_min=0.0, fast_mask=True)


def draw(frame, dets, tag, lines):
    """在图上画全部候选（序号+圆心+圆），右上角写文件名/自动判定。"""
    img = frame.copy()
    for i, d in enumerate(dets, 1):
        c = (int(round(d.cx)), int(round(d.cy)))
        cv2.circle(img, c, int(round(d.r)), (0, 255, 0), 2)
        cv2.drawMarker(img, c, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(img, "#%d  r=%.0f s=%.2f" % (i, d.r, d.score),
                    (c[0] - int(d.r), max(16, c[1] - int(d.r) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, "#%d  r=%.0f s=%.2f" % (i, d.r, d.score),
                    (c[0] - int(d.r), max(16, c[1] - int(d.r) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    bar = 34 * len(lines) + 8
    img[0:bar, 0:img.shape[1]] = (img[0:bar, 0:img.shape[1]] * 0.35).astype(np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (8, 26 + i * 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="帧目录")
    ap.add_argument("--out", required=True, help="输出目录（含 hit/ miss/ sheets/）")
    ap.add_argument("--mode", default="full", choices=["full", "track"])
    ap.add_argument("--compare-track", action="store_true",
                    help="额外跑一遍 track 模式，标注两种模式结果不一致的帧")
    ap.add_argument("--rms-rel-max", type=float, default=0.0,
                    help="形状判据：圆拟合残差/半径 上限（0=关）。真球 0.017~0.12，矩形色斑 0.13~0.37")
    ap.add_argument("--aspect-max", type=float, default=0.0,
                    help="形状判据：最小外接矩形伸长比 上限（0=关）。球 1.08~1.37，矩形色斑 1.66~3.20")
    ap.add_argument("--r-min", type=float, default=None,
                    help="半径下限（默认 32，= cfg/vision.yaml 的 grab.cv.r_min）")
    ap.add_argument("--h-med-min", type=float, default=0.0,
                    help="候选级验色：圆内色相中位数下限（红球用 20 砍掉橙色反光 H≈6~11）")
    ap.add_argument("--core-frac-min", type=float, default=0.0,
                    help="候选级验实心度：圆内强红(dom>=core_dom)占比下限（红球用 0.25）")
    ap.add_argument("--only-hit", action="store_true",
                    help="只出 hit/（下一轮只查检出图时用），不出 miss/ 与 miss 的 sheet")
    ap.add_argument("--sheet-per", type=int, default=20)
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    files = sorted(p.name for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not files:
        sys.exit("❌ 空目录: %s" % src)
    (out / "hit").mkdir(parents=True, exist_ok=True)
    (out / "miss").mkdir(parents=True, exist_ok=True)
    (out / "sheets").mkdir(parents=True, exist_ok=True)

    wp = dict(WORK_PARAMS)
    if args.r_min is not None:
        wp["r_min"] = args.r_min
    wp["rms_rel_max"] = args.rms_rel_max          # 0 = 关（默认，= 原工作点行为）
    wp["aspect_max"] = args.aspect_max
    wp["h_med_min"] = args.h_med_min
    wp["core_frac_min"] = args.core_frac_min
    p = Params(**wp)
    print("候选级判据: h_med_min=%.0f  core_frac_min=%.2f  rms_rel_max=%.2f  aspect_max=%.2f %s"
          % (args.h_med_min, args.core_frac_min, args.rms_rel_max, args.aspect_max,
             "(全关 = 原工作点)" if not (args.rms_rel_max or args.aspect_max
                                        or args.h_med_min or args.core_frac_min) else ""))
    tracker = BallTracker(p, margin=64, proc_side=256) if args.compare_track else None

    recs, times = [], []
    for i, name in enumerate(files):
        img = cv2.imread(str(src / name))
        if img is None:
            print("⚠️ 读不出:", name)
            continue
        t0 = time.perf_counter()
        dets = detect(img, p)
        dt = (time.perf_counter() - t0) * 1e3
        times.append(dt)
        rec = {"file": name, "n_det": len(dets), "ms": round(dt, 1),
               "det": [d.as_dict() for d in dets]}
        if tracker is not None:
            td, info = tracker.step(img)
            rec["track_mode"] = info.mode
            rec["track_det"] = ([td.cx, td.cy, td.r] if td else None)
            # 两模式是否一致（判据：都无检出，或都有且圆心距 ≤ 0.4r、半径比在 [0.7,1.4]）
            if not dets and td is None:
                same = True
            elif dets and td is not None:
                b = max(dets, key=lambda d: d.score)
                same = (np.hypot(b.cx - td.cx, b.cy - td.cy) <= 0.4 * max(td.r, 1)
                        and 0.7 <= b.r / max(td.r, 1e-6) <= 1.4)
            else:
                same = False
            rec["mode_diff"] = 0 if same else 1
            rec["track_n_roi"] = tracker.n_roi
            rec["track_n_full"] = tracker.n_full
            rec["track_n_fallback"] = tracker.n_fallback
        recs.append(rec)

        # 分组落盘（画了圈的进 hit/，没画的进 miss/；miss 用原图便于确认"确实没球"）
        best = max(dets, key=lambda d: d.score) if dets else None
        lines = ["%s   n_det=%d" % (name, len(dets))]
        if best:
            lines.append("HIT  r=%.0f score=%.2f cov=%.2f circ=%.2f arc=%.0f"
                         % (best.r, best.score, best.cov, best.circ, best.arc_deg))
            lines.append("      rms=%.2f aspect=%.2f" % (best.rms_rel, best.aspect))
            lines.append("      h_med=%.0f core=%.2f rim=%.2f"
                         % (best.h_med, best.core_frac, best.rim_cov))
            lines.append("center=(%.0f, %.0f)  border_frac=%.2f"
                         % (best.cx, best.cy, best.border_frac))
        else:
            lines.append("MISS (no candidate)")
        if tracker is not None and not rec.get("mode_diff", 0):
            pass
        elif tracker is not None:
            lines.append("⚠️ track/full 结果不一致（本帧判读以 full 为准）")
        if dets or not args.only_hit:
            vis = draw(img, dets, "hit" if dets else "miss", lines)
            (out / ("hit" if dets else "miss") / name).write_bytes(
                cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes())
        if (i + 1) % 100 == 0:
            print("  … %d/%d" % (i + 1, len(files)))

    # ---- JSON + CSV ----
    hit = [r for r in recs if r["n_det"]]
    miss = [r for r in recs if not r["n_det"]]
    summary = {"src": str(src), "n": len(recs), "hit": len(hit), "miss": len(miss),
               "params": p.__dict__, "mode": args.mode,
               "ms_p50": float(np.median(times)) if times else 0.0,
               "multi_det": sum(1 for r in recs if r["n_det"] >= 2),
               "mode_diff": sum(r.get("mode_diff", 0) for r in recs)}
    (out / "qa.json").write_text(json.dumps({"summary": summary, "records": recs},
                                            ensure_ascii=False, indent=1), encoding="utf-8")
    csv_p = out / "qa.csv"
    with open(csv_p, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["file", "auto", "n_det", "cx", "cy", "r", "score", "cov", "circ",
                    "arc_deg", "border_frac", "ms", "mode_diff", "human", "note"])
        for r in recs:
            b = max(r["det"], key=lambda d: d["score"]) if r["det"] else None
            row = [r["file"], "HIT" if r["n_det"] else "MISS", r["n_det"]]
            if b:
                row += ["%.1f" % b["cx"], "%.1f" % b["cy"], "%.1f" % b["r"],
                        "%.3f" % b["score"], "%.2f" % b["cov"], "%.2f" % b["circ"],
                        "%.0f" % b["arc_deg"], "%.3f" % b["border_frac"]]
            else:
                row += [""] * 8
            row += ["%.1f" % r["ms"], r.get("mode_diff", ""), "", ""]
            w.writerow(row)

    print("\n=== 分组结果 ===")
    print("  识别到 (hit) : %d / %d = %.1f%%" % (len(hit), len(recs), 100 * len(hit) / len(recs)))
    print("  没识别到(miss): %d / %d = %.1f%%" % (len(miss), len(recs), 100 * len(miss) / len(recs)))
    print("  多候选(>=2)的帧: %d（目标只有一个 ⇒ 这些帧至少有一个是误检）" % summary["multi_det"])
    if args.compare_track:
        print("  track/full 不一致的帧: %d（CSV 的 mode_diff 列）" % summary["mode_diff"])
    print("  单帧耗时 p50 = %.1f ms" % summary["ms_p50"])
    print("  → %s" % out)
    print("     hit/  %d 张   miss/ %d 张" % (len(hit), len(miss)))

    # ---- 总览 sheet（先扫一眼）----
    groups = (("hit", hit),) if args.only_hit else (("hit", hit), ("miss", miss))
    for grp, items in groups:
        for s in range((len(items) + args.sheet_per - 1) // args.sheet_per):
            entries = []
            for k, r in enumerate(items[s * args.sheet_per:(s + 1) * args.sheet_per]):
                vis = cv2.imread(str(out / grp / r["file"]))
                b = max(r["det"], key=lambda d: d["score"]) if r["det"] else None
                caps = ["%d. %s" % (s * args.sheet_per + k + 1, r["file"])]
                caps.append(("r=%.0f s=%.2f n=%d" % (b["r"], b["score"], r["n_det"]))
                            if b else "no candidate")
                entries.append((vis, caps, bool(b)))
            make_sheet(entries, out / "sheets" / ("%s_%02d.jpg" % (grp, s + 1)),
                       cols=args.cols,
                       title="QA %s - %d/%d - n_det=%s" %
                             (grp, s + 1, (len(items) + args.sheet_per - 1) // args.sheet_per,
                              "0" if grp == "miss" else ">=1"))
    print("  sheets/  %s" % sorted(x.name for x in (out / 'sheets').iterdir())[:6])


if __name__ == "__main__":
    main()
