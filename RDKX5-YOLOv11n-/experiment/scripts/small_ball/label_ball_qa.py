#!/usr/bin/env python3
"""夹取小球 · **逐张人工判读器**（把识别对/错一张张标出来，随时可中断续标）。

配套 `run_qa_ball.py` 的产物使用：它已经把"识别到"和"没识别到"分成两组、并画好了圈。
本脚本一帧一屏地放给你看，按一个键就记一条，**每按一次都落盘**（崩溃/误关窗口不丢标注）。

## 判读口径（只记"这一帧识别得对不对"，不记类别名，避免标错）

    HIT 帧（画了圈）：圈在球上 → `1` 正确 ｜ 圈不在球上 → `2` 误检
    MISS帧（没画圈）：确实没球   → `1` 正确 ｜ 画面里有球没圈 → `3` 漏检
    看不清/球被挡一半/拿不准 → `4`（不参与正确率统计）

按键：
    `1` 正确    `2` 误检(FP)    `3` 漏检(FN)    `4` 不确定
    `space` 跳过（不记录）      `b` 回上一张并清掉它的标签
    `d` 叠加/隐藏 识别圈（对比原图）   `z` 切换放大倍率（2.5x/5x）   `q`/`Esc` 退出

## 为什么必须用 conda 的解释器

交互看图依赖 `cv2.imshow`，而 cv2 的 GUI 支持是编译期决定的：

    /home/ansty/anaconda3/envs/yolov8/bin/python    cv2 4.11 + Qt5   ✅ 能开窗
    python3（系统）                                  cv2 5.0  GUI=NONE ❌ 开不了窗

所以 **必须**用 conda 那个解释器跑（或在 conda 里 `activate yolov8`）。脚本启动时会自检并提示。

## 用法

    # 逐张判读（默认按帧序；已标过的自动跳过，随时可中断续标）
    /home/ansty/anaconda3/envs/yolov8/bin/python experiment/scripts/small_ball/label_ball_qa.py \
        --qa output/preview/small_ball_07_qa --src data/derived/small_ball_07_500

    # 只判"识别到"那组 / 只判"没识别到"那组
    ... --only hit         # 或 --only miss

    # 先看大候选（最像真球的那批），再回头扫小圆
    ... --order size

    # 不开窗也能核：把前 12 帧的界面渲染成图片sheet
    ... --self-test 12

    # 只看统计（读已有 labels.csv）
    ... --report
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
FIELDS = ["file", "group", "auto", "n_det", "r", "score", "verdict", "note", "ts"]
VERDICT_KEY = {ord("1"): "ok", ord("2"): "fp", ord("3"): "fn", ord("4"): "dunno"}
VERDICT_TXT = {"ok": "OK (1)", "fp": "FALSE POS (2)", "fn": "MISSED (3)",
               "dunno": "UNSURE (4)"}
# ⚠️ HUD 一律用 ASCII：cv2.putText 走 Hershey 字体，中文会渲染成方框（实测过）。


def check_gui() -> bool:
    """cv2 能不能开窗（不能就别让用户白等）。"""
    try:
        cv2.namedWindow("__probe__")
        cv2.destroyWindow("__probe__")
        return True
    except Exception as e:
        print("=" * 70)
        print("❌ 这个 Python 的 cv2 没有 GUI 支持，开不了窗：%s" % str(e)[:120])
        print("   请改用带 Qt5 的解释器：")
        print("     /home/ansty/anaconda3/envs/yolov8/bin/python %s ..." % sys.argv[0])
        print("   或退一步用静态版：加 --self-test 30 把界面渲染成图片来看。")
        print("=" * 70)
        return False


def load_labels(path: Path):
    """读已有的 labels.csv → {file: verdict}（支持续标）。"""
    done = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as fp:
            for row in csv.DictReader(fp):
                if row.get("file"):
                    done[row["file"]] = row.get("verdict", "")
    return done


def append_label(path: Path, row: dict):
    """一条一落盘（含 fsync）：崩溃/误关窗口都不丢标注。"""
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})
        fp.flush()
        os.fsync(fp.fileno())


def rewrite_labels(path: Path, rows: list):
    """重写整张表（用于"回上一张"清标签）。"""
    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
        fp.flush()
        os.fsync(fp.fileno())


def build_items(qa: Path, order: str, only: str):
    d = json.loads((qa / "qa.json").read_text(encoding="utf-8"))
    recs = d["records"]
    if only == "hit":
        recs = [r for r in recs if r["n_det"]]
    elif only == "miss":
        recs = [r for r in recs if not r["n_det"]]
    if order == "size":
        recs = sorted(recs, key=lambda r: -(max((x["r"] for x in r["det"]), default=-1)))
    elif order == "score":
        recs = sorted(recs, key=lambda r: -(max((x["score"] for x in r["det"]), default=-1)))
    return recs, d["summary"]


def hud(img, lines, verdict, group):
    """左上角信息条 + 底部按键条 + 右上角当前标签。"""
    h, w = img.shape[:2]
    bar = 30 * len(lines) + 10
    img[0:bar, 0:w] = (img[0:bar, 0:w] * 0.30).astype(np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (10, 24 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
    foot = ("[1]correct [2]false-pos [3]missed [4]unsure  |  [space]skip  [b]back  "
            "[d]overlay  [z]zoom  [q]quit+save")
    fb = 34
    img[h - fb:h, 0:w] = (img[h - fb:h, 0:w] * 0.30).astype(np.uint8)
    cv2.putText(img, foot, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (200, 255, 200), 1, cv2.LINE_AA)
    col = {"ok": (0, 220, 0), "fp": (0, 80, 255), "fn": (255, 120, 0)}.get(verdict, (200, 200, 200))
    tag = VERDICT_TXT.get(verdict, "-")
    (tw, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(img, (w - tw - 30, 8), (w - 8, 50), (0, 0, 0), -1)
    cv2.putText(img, tag, (w - tw - 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
    cv2.putText(img, group.upper(), (w - tw - 130, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2, cv2.LINE_AA)
    return img


def render(rec, src: Path, qa_dir: Path, overlay=True, zoom=2.5, idx=0, total=0,
           verdict="", zoom_override=None):
    """返回 (主图, 放大图)。主图 = 帧 + 识别圈 + HUD；放大图 = 以最优候选/画面中心为中心。"""
    ann = cv2.imread(str(qa_dir / ("hit" if rec["n_det"] else "miss") / rec["file"]))
    raw = cv2.imread(str(src / rec["file"]))
    if raw is None:
        raw = ann
    if ann is None:          # --only-hit 生成的包里 miss/ 是空的 → 回退到原图，别崩
        ann = raw
    best = max(rec["det"], key=lambda d: d["score"]) if rec["det"] else None
    lines = ["[%d/%d]  %s   auto=%s  n_det=%d" % (idx + 1, total, rec["file"],
                                                   "HIT" if rec["n_det"] else "MISS", rec["n_det"])]
    if best:
        lines.append("best: r=%.0f  score=%.2f  cov=%.2f  circ=%.2f  arc=%.0f  border=%.2f"
                     % (best["r"], best["score"], best["cov"], best["circ"],
                        best["arc_deg"], best["border_frac"]))
        if "rms_rel" in best:      # 形状判据（判断"圆的还是矩形/色斑"就看这两个数）
            lines.append("shape: rms=%.2f (<=0.15) asp=%.2f (<=1.5) | color: h_med=%.0f (>=20) "
                         "core=%.2f (>=0.25) rim=%.2f"
                         % (best["rms_rel"], best.get("aspect", 1.0), best.get("h_med", -1),
                            best.get("core_frac", -1), best.get("rim_cov", -1)))
        lines.append("center=(%.0f, %.0f)   detect=%.1f ms" % (best["cx"], best["cy"], rec["ms"]))
    else:
        lines.append("no candidate   (ball IS visible here? -> press 3 = missed)")
        lines.append("detect=%.1f ms" % rec["ms"])
    main = ann if overlay else raw          # d 键切换：看带圈图 / 看原图
    main = hud(main, lines, verdict, "hit" if rec["n_det"] else "miss")

    cx, cy = (best["cx"], best["cy"]) if best else (raw.shape[1] / 2, raw.shape[0] / 2)
    z = zoom_override or zoom
    r = int(min(raw.shape[1], raw.shape[0]) / (2 * z))
    x0 = int(np.clip(cx - r, 0, raw.shape[1] - 2 * r))
    y0 = int(np.clip(cy - r, 0, raw.shape[0] - 2 * r))
    crop = (ann if overlay else raw)[y0:y0 + 2 * r, x0:x0 + 2 * r].copy()
    crop = cv2.resize(crop, (int(2 * r * z), int(2 * r * z)), interpolation=cv2.INTER_NEAREST)
    crop = crop[:720, :720]
    cv2.putText(crop, "zoom %.1fx" % z, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2, cv2.LINE_AA)
    return main, crop


def do_report(path: Path):
    rows = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as fp:
            rows = [r for r in csv.DictReader(fp) if r.get("verdict")]
    if not rows:
        print("还没有标注（%s 不存在或为空）" % path)
        return
    tp = sum(1 for r in rows if r["auto"] == "HIT" and r["verdict"] == "ok")
    fp = sum(1 for r in rows if r["auto"] == "HIT" and r["verdict"] == "fp")
    fn = sum(1 for r in rows if r["auto"] == "MISS" and r["verdict"] == "fn")
    tn = sum(1 for r in rows if r["auto"] == "MISS" and r["verdict"] == "ok")
    dunno = sum(1 for r in rows if r["verdict"] == "dunno")
    print("=" * 66)
    print("已判读 %d 帧：正确 %d（其中 HIT 帧 %d / MISS 帧 %d）  误检 %d  漏检 %d  不确定 %d"
          % (len(rows), tp + tn, tp, tn, fp, fn, dunno))
    if tp + fp:
        print("  HIT 帧里判对的比例（精度）= %d/%d = %.1f%%" % (tp, tp + fp, 100 * tp / (tp + fp)))
    if tp + fn:
        print("  真实有球的帧里检出的比例（召回）= %d/%d = %.1f%%" % (tp, tp + fn, 100 * tp / (tp + fn)))
    if fp:
        rs = [float(r["r"]) for r in rows if r["auto"] == "HIT" and r["verdict"] == "fp" and r["r"]]
        if rs:
            print("  误检候选半径: p50=%.0f px  最小=%.0f  最大=%.0f  ⇒ 卡 r_min 的依据"
                  % (np.median(rs), min(rs), max(rs)))
    print("=" * 66)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa", default="output/preview/small_ball_07_qa",
                    help="run_qa_ball.py 的输出目录（含 qa.json / hit/ / miss/）")
    ap.add_argument("--src", default="data/derived/small_ball_07_500", help="原始帧目录")
    ap.add_argument("--labels", default=None, help="标注 CSV（默认 <qa>/labels.csv）")
    ap.add_argument("--only", default="all", choices=["all", "hit", "miss"])
    ap.add_argument("--order", default="seq", choices=["seq", "size", "score"])
    ap.add_argument("--redo", action="store_true", help="已标过的也再看一遍")
    ap.add_argument("--self-test", type=int, default=0,
                    help="不开窗：把前 N 帧的界面渲染成 sheet 预览（验界面用）")
    ap.add_argument("--report", action="store_true", help="只打印统计")
    args = ap.parse_args()

    qa = Path(args.qa)
    src = Path(args.src)
    lab = Path(args.labels) if args.labels else (qa / "labels.csv")

    if args.report:
        do_report(lab)
        return

    if args.only == "all" and not any((qa / "miss").glob("*.jpg")):
        print("提示：该包里没有 miss/ 图（用 --only-hit 生成的），本次只看 hit 组")
        args.only = "hit"
    recs, summary = build_items(qa, args.order, args.only)
    print("待判读 %d 帧（自 %s；分组=%s 排序=%s）" % (len(recs), qa, args.only, args.order))
    print("  自动判定统计: HIT %d / MISS %d（全素材 %d 帧）"
          % (summary["hit"], summary["miss"], summary["n"]))

    if args.self_test:
        out = qa / "ui_preview"
        out.mkdir(parents=True, exist_ok=True)
        tiles = []
        step = max(1, len(recs) // args.self_test)
        sel = recs[::step][:args.self_test]          # 均匀取，别只看开头（开头多半是没球的帧）
        for i, rec in enumerate(sel):
            main, crop = render(rec, src, qa, idx=i, total=len(recs))
            pair = np.hstack([cv2.resize(main, (1024, 576)), cv2.resize(crop, (576, 576))])
            cv2.imwrite(str(out / ("ui_%02d_%s.jpg" % (i + 1, Path(rec["file"]).stem))), pair)
            tiles.append(cv2.resize(pair, (800, 400)))
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(qa / "ui_preview_sheet.jpg"),
                    np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
        print("✅ 界面预览 → %s（每张左=主窗 右=放大窗）" % (qa / "ui_preview_sheet.jpg"))
        return

    if not check_gui():
        return 3

    done = {} if args.redo else load_labels(lab)
    rows_all = []
    if lab.exists() and args.redo is False:
        with open(lab, newline="", encoding="utf-8") as fp:
            rows_all = list(csv.DictReader(fp))

    todo = [r for r in recs if r["file"] not in done]
    print("已有标注 %d 条，本次待看 %d 帧（'q' 退出，随时可再来）" % (len(done), len(todo)))
    print("标签文件：%s" % lab)

    win_main, win_zoom = "grab-qa  [1/2/3/4 判读]", "grab-qa zoom"
    cv2.namedWindow(win_main, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_zoom, cv2.WINDOW_AUTOSIZE)
    idx = 0
    overlay, zoom = True, 2.5
    try:
        while idx < len(todo):
            rec = todo[idx]
            verdict = done.get(rec["file"], "")
            main_img, crop = render(rec, src, qa, overlay=overlay, zoom=zoom,
                                    idx=idx, total=len(todo), verdict=verdict)
            cv2.imshow(win_main, main_img)
            cv2.imshow(win_zoom, crop)
            key = cv2.waitKey(0) & 0xFF
            ch = chr(key)
            if key in VERDICT_KEY:                      # 记一条
                v = VERDICT_KEY[key]
                append_label(lab, {"file": rec["file"],
                                   "group": "hit" if rec["n_det"] else "miss",
                                   "auto": "HIT" if rec["n_det"] else "MISS",
                                   "n_det": rec["n_det"],
                                   "r": (max(rec["det"], key=lambda d: d["score"])["r"]
                                         if rec["n_det"] else ""),
                                   "score": (max(rec["det"], key=lambda d: d["score"])["score"]
                                             if rec["n_det"] else ""),
                                   "verdict": v, "note": "", "ts": time.strftime("%H:%M:%S")})
                rows_all.append({"file": rec["file"], "verdict": v,
                                 "auto": "HIT" if rec["n_det"] else "MISS",
                                 "r": (max(rec["det"], key=lambda d: d["score"])["r"]
                                       if rec["n_det"] else "")})
                done[rec["file"]] = v
                idx += 1
                if idx % 20 == 0:
                    do_report(lab)
            elif ch in (" ", "s"):                      # 跳过
                idx += 1
            elif ch == "b":                             # 回上一张并清标签
                if idx == 0:
                    continue
                idx -= 1
                prev = todo[idx]["file"]
                done.pop(prev, None)
                rows_all = [r for r in rows_all if r.get("file") != prev]
                rewrite_labels(lab, rows_all)
            elif ch == "d":
                overlay = not overlay
            elif ch == "z":
                zoom = 5.0 if zoom < 3 else 2.5
            elif ch in ("q", "\x1b"):
                break
    finally:
        cv2.destroyAllWindows()
        do_report(lab)
    print("标注文件：%s（下次运行会自动跳过已标帧）" % lab)


if __name__ == "__main__":
    main()
