#!/usr/bin/env python3
"""板端测试的汇总面板 + 可视化 sheet（产出到 output/preview/small_ball/renders/board/）。

面板内容：三条输入链路的**阶段耗时堆叠**、端到端 FPS、命中率对比、以及板端 CV 的对照。
sheet：把**板端 none-stretch 链**的框画在原始 720p 上（该链路的框就在原图坐标，可直接叠）。

用法：
    python experiment/scripts/small_ball/make_board_summary.py \
        --dir output/preview/small_ball --src data/derived/small_ball_selected_700
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHAINS = [("board_yolo.json", "A chain\n(remap@720p +\nresize + enhance)", "#d62728"),
          ("board_yolo_none-letterbox.json", "resize + letterbox\n(no domain)", "#2ca02c"),
          ("board_yolo_none-stretch.json", "resize 640 stretch\n(no domain)", "#1f77b4")]
STAGES = [("preprocess_A", "preprocess"), ("nv12", "nv12"), ("bpu_run", "BPU"),
          ("decode", "decode")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="output/preview/small_ball/renders",
                        help="工作区根（board/ 与拼版图写在这里；2026-10-06 分类后为 …/small_ball/renders")
    ap.add_argument("--src", default="data/derived/small_ball_selected_700")
    ap.add_argument("--yolo-conf", type=float, default=0.25)
    args = ap.parse_args()
    B = Path(args.dir) / "board"
    B.mkdir(parents=True, exist_ok=True)

    data = {}
    for f, tag, col in CHAINS:
        d = json.loads((B / f).read_text(encoding="utf-8"))
        st = d["stats_ms"]
        hits = {t: sum(1 for r in d["records"]
                       if any(x["kind"] == "red_ball" and x["score"] >= t for x in r["det"]))
                for t in (0.25, 0.5)}
        data[f] = dict(tag=tag, col=col, st=st, hits=hits, n=len(d["records"]))
    cvres = {}
    for t in (1, 4, 8):
        d = json.loads((B / f"board_cv_t{t}.json").read_text(encoding="utf-8"))
        cvres[t] = d
    best_cv = min(cvres.values(), key=lambda d: d["stats_ms"]["det_only_p50"])
    pc_cv = 10.6      # PC 侧实测（output/preview/small_ball/README.md §2）
    # 快掩膜（v2）与流式 ROI 跟踪的板端结果
    trk = json.loads((B / "board_cv_track_v3_ps256.json").read_text(encoding="utf-8"))
    trk_avg = 1000.0 / np.mean([r["ms"] for r in trk["records"]])
    trk_roi = 1000.0 / np.median([r["ms"] for r in trk["records"] if r["mode"] == "roi"])
    v2f = json.loads((B / "board_cv_v2_t4.json").read_text(encoding="utf-8"))
    v2_fps = v2f["stats_ms"]["fps_det_only"]
    v2_p50 = v2f["stats_ms"]["det_only_p50"]

    fig = plt.figure(figsize=(16, 9), dpi=100, constrained_layout=True)
    fig.suptitle("RDK X5 board-side: YOLO11n detect (ball-collision bin) input-chain cost vs "
                 "classic CV  —  700 frames, real board", fontsize=14, fontweight="bold")

    # 1. 阶段耗时堆叠（实机口径：不含读盘）
    ax = fig.add_subplot(2, 3, 1)
    xs = np.arange(len(CHAINS)); bottom = np.zeros(len(CHAINS))
    for key, lab in STAGES:
        vals = np.array([data[f]["st"][key]["p50"] for f, _, _ in CHAINS])
        ax.bar(xs, vals, 0.55, bottom=bottom, label=lab)
        bottom += vals
    for i, v in enumerate(bottom):
        ax.text(i, v + 1.5, f"{v:.1f} ms\n{1000/v:.0f} FPS", ha="center", fontsize=10)
    ax.set_xticks(xs); ax.set_xticklabels([t for _, t, _ in CHAINS], fontsize=7.5)
    ax.set_ylabel("ms / frame (excl. disk read)"); ax.set_ylim(0, 55)
    ax.set_title("input-chain stage cost on board (p50)", fontsize=10); ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.3)

    # 2. 纯 BPU 上限
    ax = fig.add_subplot(2, 3, 2)
    th = [1, 2, 4, 8]; fps = [67.2, 129.5, 148.2, 147.6]
    ax.bar([str(t) for t in th], fps, color="#9467bd")
    for i, v in enumerate(fps):
        ax.text(i, v + 3, f"{v:.0f}", ha="center")
    ax.axhline(data["board_yolo_none-stretch.json"]["st"]["bpu_run"]["p50"] and
               1000 / data["board_yolo_none-stretch.json"]["st"]["bpu_run"]["p50"],
               color="r", ls="--", lw=1, label="single-frame BPU (1 thread)")
    ax.set_xlabel("hrt_model_exec thread_num"); ax.set_ylabel("FPS")
    ax.set_title("BPU-only ceiling (12.2 ms/frame)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)

    # 3. YOLO vs CV 速度（板端）
    ax = fig.add_subplot(2, 3, 3)
    ys = 1000 / (data["board_yolo_none-stretch.json"]["st"]["end2end"]["p50"]
                 - data["board_yolo_none-stretch.json"]["st"]["read"]["p50"])
    ya = 1000 / (data["board_yolo.json"]["st"]["end2end"]["p50"]
                 - data["board_yolo.json"]["st"]["read"]["p50"])
    cvf = best_cv["stats_ms"]["fps_det_only"]
    pccv = 1000 / pc_cv
    labels = ["YOLO no-domain\n(board)", "YOLO A-chain\n(board)",
              "CV tracker\nROI (board)", "CV v2\nfull (board)",
              "CV numpy-ref\nfull (board)", "CV\n(PC)"]
    vals = [ys, ya, trk_avg, v2_fps, cvf, pccv]
    cols = ["#2ca02c", "#d62728", "#1f77b4", "#17becf", "#9467bd", "#aec7e8"]
    ax.bar(labels, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.5, f"{v:.1f}", ha="center")
    ax.set_ylabel("FPS (algo only, no disk read)")
    ax.set_title("end-to-end speed: YOLO vs CV (algo only)", fontsize=10)
    ax.grid(axis="y", alpha=.3); ax.tick_params(axis="x", labelsize=7)

    # 4. 命中率对比（含 PC）
    ax = fig.add_subplot(2, 3, 4)
    names, h25, h50 = [], [], []
    for f, tag, _ in CHAINS:
        names.append(tag.split("\n")[0].replace("resize + ", "resize+")
                     .replace("resize 640", "resize 640")); h25.append(data[f]["hits"][0.25]); h50.append(data[f]["hits"][0.5])
    pcpt = json.loads((Path(args.dir) / "pc_pt_on_adomain640.json").read_text(encoding="utf-8"))
    pc_h25 = sum(1 for r in pcpt["records"]
                 if any(x["name"] == "red_ball" and x["conf"] >= 0.25 for x in r["det"]))
    pc_h50 = sum(1 for r in pcpt["records"]
                 if any(x["name"] == "red_ball" and x["conf"] >= 0.5 for x in r["det"]))
    names += ["CV (board)"]; h25 += [best_cv["n_hit"]]; h50 += [None]
    xs = np.arange(len(names))
    ax.bar(xs - .2, [v / 7 for v in h25], .4, label="conf>=0.25", color="#2ca02c")
    ax.bar(xs + .2, [(v / 7) if v is not None else 0 for v in h50], .4,
           label="conf>=0.5 (board workpoint)", color="#ff8c00")
    ax.axhline(pc_h25 / 7, color="k", ls=":", lw=1,
               label=f"PC pt on A-domain640 @0.25 = {pc_h25/7:.1f}%")
    for i, v in enumerate(h25):
        ax.text(i - .2, v / 7 + .012, f"{v/7:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 0.95); ax.set_ylabel("frames with red_ball / 700")
    ax.set_title("accuracy per input chain (700 frames)", fontsize=10)
    ax.legend(fontsize=7.5); ax.grid(axis="y", alpha=.3)

    # 5. 板端 vs PC（同一 A 域输入）一致性
    ax = fig.add_subplot(2, 3, 5)
    bd = json.loads((B / "board_yolo.json").read_text(encoding="utf-8"))["records"]
    SX, SY = 1280 / 640.0, 720 / 640.0
    ious = []
    for r in bd:
        ca = [x for x in r["det"] if x["kind"] == "red_ball" and x["score"] >= 0.25]
        cb = [x for x in pcpt["records"] if x["file"] == r["file"]][0]
        cb = [x for x in cb["det"] if x["name"] == "red_ball" and x["conf"] >= 0.25]
        if ca and cb:
            A = max(ca, key=lambda x: x["score"]); C = max(cb, key=lambda x: x["conf"])
            ax1, ay1, ax2, ay2 = (A["xyxy"][0] / SX, A["xyxy"][1] / SY,
                                  A["xyxy"][2] / SX, A["xyxy"][3] / SY)
            cx1, cy1, cx2, cy2 = C["xyxy"]
            ix = max(0, min(ax2, cx2) - max(ax1, cx1)); iy = max(0, min(ay2, cy2) - max(ay1, cy1))
            inter = ix * iy
            uni = (ax2 - ax1) * (ay2 - ay1) + (cx2 - cx1) * (cy2 - cy1) - inter
            ious.append(inter / uni if uni > 0 else 0)
    ious = np.array(ious)
    ax.hist(ious, bins=np.linspace(0, 1, 21), color="#17becf")
    ax.set_xlabel("IoU  board .bin  vs  PC .pt  (same A-domain 640 input)")
    ax.set_ylabel("frames"); ax.set_title(f"int8 quantization loss: n={len(ious)}, "
                                          f"p50={np.median(ious):.3f}")
    ax.grid(alpha=.3)

    # 6. 文本
    ax = fig.add_subplot(2, 3, 6); ax.axis("off")
    st = data["board_yolo_none-stretch.json"]["st"]
    ag = data["board_yolo.json"]["st"]
    lines = [
        "board: RDK X5 (Bayes-e) / 8x A55 / Python 3.10 / cv2 4.11 / OS 3.5.0",
        "model: models/yolo11n_detect_bayese_640x640_nv12.bin",
        "   md5 a28156ba... = local output/weights/... = weights/yolo11n.pt",
        "config on board: chain=A, clahe=0.5, gamma=0.85, undistort=on,",
        "   score_threshold=0.5, labels=[blue_ball,gate,red_ball]",
        "",
        f"YOLO  A-chain   : preprocess {ag['preprocess_A']['p50']:.1f} + nv12 {ag['nv12']['p50']:.1f}"
        f" + BPU {ag['bpu_run']['p50']:.1f} + decode {ag['decode']['p50']:.1f} ms",
        f"YOLO  no-domain : preprocess {st['preprocess_A']['p50']:.1f} + nv12 {st['nv12']['p50']:.1f}"
        f" + BPU {st['bpu_run']['p50']:.1f} + decode {st['decode']['p50']:.1f} ms",
        f"        -> dropping the domain chain saves "
        f"{(1 - st['end2end']['p50'] / ag['end2end']['p50']):.0%} of end-to-end"
        f" (disk read {ag['read']['p50']:.1f} ms, same for both)",
        "",
        f"CV on board: numpy-ref full {best_cv['stats_ms']['det_only_p50']:.1f} ms /"
        f" v2-fast full {v2_p50:.1f} ms / tracker avg {1000/trk_avg:.1f} FPS"
        f" (ROI steady {trk_roi:.0f} FPS)",
        f"CV hits 549/700 = 78.4% (tracker 546) vs YOLO 419/700 = 59.9%",
        f"CV per-band ROI: r<=70 -> 71 FPS | r 200-300 -> 32 FPS   (YOLO flat 47 FPS)",
        f"CV thread scaling: 1thr {cvres[1]['stats_ms']['det_only_p50']:.0f} ms / "
        f"4thr {cvres[4]['stats_ms']['det_only_p50']:.0f} ms / "
        f"8thr {cvres[8]['stats_ms']['det_only_p50']:.0f} ms (barely scales)",
        "",
        "read() 13.6 ms = SD-card JPEG decode; a live frame comes from the camera buffer,",
        "so it is excluded from the per-frame budget above.",
        "board clock is not synced (stuck at 2000-01-01) - file timestamps only.",
        "NOT verified: live camera loop, multi-thread pipelining (BPU 2 threads = 129 FPS).",
    ]
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8.0)
    fig.savefig(B / "summary_board.jpg", dpi=100)
    print(f"✅ {B/'summary_board.jpg'}")

    # ---- sheet：板端 none-stretch 框（坐标就在原图 720p 空间，可直接叠）----
    S = Path(args.src)
    d = json.loads((B / "board_yolo_none-stretch.json").read_text(encoding="utf-8"))
    recs = [r for r in d["records"] if any(x["kind"] == "red_ball" for x in r["det"])]
    recs.sort(key=lambda r: max([(x["xyxy"][2] - x["xyxy"][0]) for x in r["det"]
                                 if x["kind"] == "red_ball"], default=0))
    per, cols = 20, 4
    for s in range(min(3, (len(recs) + per - 1) // per)):
        tiles = []
        for k, r in enumerate(recs[s * per:(s + 1) * per]):
            im = cv2.imread(str(S / r["file"]))
            best = None
            for x in r["det"]:
                if x["kind"] != "red_ball":
                    continue
                x1, y1, x2, y2 = [int(v) for v in x["xyxy"]]
                col = (0, 255, 0) if x["score"] >= args.yolo_conf else (130, 130, 130)
                cv2.rectangle(im, (x1, y1), (x2, y2), col, 2)
                cv2.putText(im, f"{x['score']:.2f}", (x1, max(14, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
                cv2.putText(im, f"{x['score']:.2f}", (x1, max(14, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
                if best is None or x["score"] > best["score"]:
                    best = x
            cap = [f"{s*per+k+1}. f{r['file'][6:12]}"]
            cap.append(f"board bin  conf={best['score']:.2f}  "
                       f"{(best['xyxy'][2]-best['xyxy'][0]):.0f}x{(best['xyxy'][3]-best['xyxy'][1]):.0f}px"
                       if best else "no det")
            tiles.append((im, cap, bool(best and best["score"] >= args.yolo_conf)))
        while len(tiles) % cols:
            tiles.append((np.zeros((720, 1280, 3), np.uint8), [""], False))
        rows = []
        for i in range(0, len(tiles), cols):
            rows.append(np.hstack([cv2.resize(t[0], (480, 270)) for t in tiles[i:i + cols]]))
        sheet = np.vstack(rows)
        for idx, (im, cap, ok) in enumerate(tiles):
            r0, c0 = divmod(idx, cols)
            y0, x0 = r0 * 270, c0 * 480
            cv2.rectangle(sheet, (x0 + 1, y0 + 1), (x0 + 479, y0 + 269),
                          (0, 200, 0) if ok else (0, 0, 255), 2)
            for j, t in enumerate(cap):
                cv2.putText(sheet, t, (x0 + 6, y0 + 18 + j * 17), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(B / f"sheet_board_nonestretch_{s+1:02d}.jpg"), sheet)
        print(f"✅ {B/f'sheet_board_nonestretch_{s+1:02d}.jpg'}")


if __name__ == "__main__":
    main()
