#!/usr/bin/env python3
"""夹取小球 · 方案 B：720p → 640×640 letterbox → 用现有 detect 权重（撞球那版）推理。

**当前权重口径**：`weights/yolo11n.pt` = 3 类 detect（`blue_ball/gate/red_ball`，nc=3），
由 `scripts/2_train/train_yolo11n.py` 产出。本脚本只做推理，**需要原版 head.py**；
若打过 6 输出补丁先 `python scripts/3_export/modify_ultralytics.py --restore`。

板端对齐：输入 = ultralytics 默认 letterbox 到 640×640（1280×720 → 640×360 → 上下 pad 140）。
可选 `--save-640` 把这份 640 输入落盘（与板端 nv12 同几何，便于核对）。

用法（在仓库根、conda yolov8 下）：
    /home/ansty/anaconda3/envs/yolov8/bin/python experiment/scripts/small_ball/run_yolo_ball.py \
        --src data/derived/small_ball_selected_700 --weights weights/yolo11n.pt \
        --device 0 --conf 0.25 \
        --out-json output/preview/small_ball/tables/yolo_ball.json \
        --sheet-dir output/preview/small_ball/renders/sheets_yolo

输出：JSON（逐图框，原图坐标 + 640 坐标）、CSV 汇总、sheet_*.jpg 联系表。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import letterbox, make_sheet, PALETTE  # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--weights", default="weights/yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.02,
                    help="推理原始阈值（建议压低，JSON 里保留全部；展示阈值用 --sheet-conf）")
    ap.add_argument("--sheet-conf", type=float, default=0.25,
                    help="sheet 上判定「命中」的置信度阈值；低于它的框画成细灰线并标 lo")
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--save-640", default=None, help="把 640×640 letterbox 输入落盘到此目录")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--sheet-dir", default=None)
    ap.add_argument("--sheet-per", type=int, default=20)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--max-sheets", type=int, default=0, help="0=全部")
    args = ap.parse_args()

    src = Path(args.src)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not files:
        sys.exit(f"❌ 空目录: {src}")

    from ultralytics import YOLO
    model = YOLO(args.weights)
    names = model.names
    print(f"权重 {args.weights} → classes: {names}  (nc={len(names)})")
    if list(names.values())[:3] != ["blue_ball", "gate", "red_ball"]:
        print(f"⚠️ 类别顺序不是 [blue_ball, gate, red_ball]，实际 {names}；"
              f"请确认与板端 model.labels 一致，否则类别会错位")

    if args.save_640:
        Path(args.save_640).mkdir(parents=True, exist_ok=True)

    records = []
    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        if img is None:
            print(f"⚠️ 读不出: {f}")
            continue
        lb, r, px, py = letterbox(img, args.imgsz)
        if args.save_640:
            cv2.imwrite(str(Path(args.save_640) / f.name), lb,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        res = model.predict(lb, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                            device=args.device, verbose=False)[0]
        dets = []
        for b in res.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            dets.append({
                "cls": int(b.cls[0]),
                "name": names[int(b.cls[0])],
                "conf": float(b.conf[0]),
                "xyxy640": [x1, y1, x2, y2],
                "xyxy": [(x1 - px) / r, (y1 - py) / r, (x2 - px) / r, (y2 - py) / r],
            })
        records.append({"file": f.name, "wh": [img.shape[1], img.shape[0]],
                        "ratio": r, "pad": [px, py], "det": dets})
        if (i + 1) % 100 == 0:
            print(f"  … {i+1}/{len(files)}")

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"weights": args.weights, "imgsz": args.imgsz,
                               "conf": args.conf, "names": names,
                               "records": records}, ensure_ascii=False, indent=1),
                   encoding="utf-8")

    # CSV 汇总：每图一行，小球框取最大者（面积）
    csv_path = out.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["file", "n_det", "n_red_ball", "n_red_ge_thr", "red_conf", "red_x1",
                    "red_y1", "red_x2", "red_y2", "red_w", "red_h", "red_cx", "red_cy",
                    "names_all", "red_conf_maxlow"])
        thr = args.sheet_conf
        for rec in records:
            reds = [d for d in rec["det"] if d["name"] == "red_ball"]
            reds_hi = [d for d in reds if d["conf"] >= thr]
            pool = reds_hi or reds
            best = max(pool, key=lambda d: (d["xyxy"][2] - d["xyxy"][0]) * (d["xyxy"][3] - d["xyxy"][1])) if pool else None
            row = [rec["file"], len(rec["det"]), len(reds), len(reds_hi)]
            if best:
                x1, y1, x2, y2 = best["xyxy"]
                row += [f"{best['conf']:.3f}", f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}",
                        f"{y2:.1f}", f"{x2-x1:.1f}", f"{y2-y1:.1f}",
                        f"{(x1+x2)/2:.1f}", f"{(y1+y2)/2:.1f}"]
            else:
                row += [""] * 9
            row.append("|".join(sorted({d["name"] for d in rec["det"]})))
            lo = max([d["conf"] for d in reds if d["conf"] < thr], default="")
            row.append(f"{lo:.3f}" if lo != "" else "")
            w.writerow(row)

    n_red = sum(1 for rec in records
                if any(d["name"] == "red_ball" and d["conf"] >= args.sheet_conf for d in rec["det"]))
    n_red_lo = sum(1 for rec in records
                   if any(d["name"] == "red_ball" for d in rec["det"]))
    print(f"✅ JSON: {out}")
    print(f"✅ CSV : {csv_path}")
    print(f"   命中 red_ball (conf>={args.sheet_conf}): {n_red}/{len(records)} = {n_red/max(1,len(records)):.1%}")
    print(f"   放宽到任意 conf>={args.conf}          : {n_red_lo}/{len(records)} = {n_red_lo/max(1,len(records)):.1%}")

    # ---------------- sheet ----------------
    if args.sheet_dir:
        sdir = Path(args.sheet_dir)
        sdir.mkdir(parents=True, exist_ok=True)
        thr = args.sheet_conf

        def wmax(rec):
            reds = [d for d in rec["det"] if d["name"] == "red_ball" and d["conf"] >= thr]
            if not reds:
                return 0.0
            b = max(reds, key=lambda d: (d["xyxy"][2]-d["xyxy"][0])*(d["xyxy"][3]-d["xyxy"][1]))
            return (b["xyxy"][2]-b["xyxy"][0]) * (b["xyxy"][3]-b["xyxy"][1])

        # 排序：命中按框面积升序（小→大），未命中排最后
        ordered = sorted(records, key=lambda r: (0, wmax(r)) if wmax(r) > 0 else (1, 0.0))
        per, cols = args.sheet_per, args.cols
        n_sheets = min((len(ordered) + per - 1) // per, args.max_sheets or 10 ** 9)
        made = []
        for s in range(n_sheets):
            entries = []
            for k, rec in enumerate(ordered[s * per:(s + 1) * per]):
                img = cv2.imread(str(src / rec["file"]))
                reds_hi = [d for d in rec["det"] if d["name"] == "red_ball" and d["conf"] >= thr]
                best = max(reds_hi, key=lambda d: (d["xyxy"][2]-d["xyxy"][0])*(d["xyxy"][3]-d["xyxy"][1])) if reds_hi else None
                for d in rec["det"]:
                    x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
                    hi = d["conf"] >= thr
                    col = PALETTE.get(d["cls"], (255, 255, 255)) if hi else (140, 140, 140)
                    cv2.rectangle(img, (x1, y1), (x2, y2), col, 2 if hi else 1)
                    lab = f"{d['name']} {d['conf']:.2f}" if hi else f"lo {d['conf']:.2f}"
                    cv2.putText(img, lab, (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(img, lab, (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, col, 1, cv2.LINE_AA)
                if best:
                    x1, y1, x2, y2 = best["xyxy"]
                    caps = [f"{s*per+k+1}. f{rec['file'][6:12]}  red {best['conf']:.2f}",
                            f"{x2-x1:.0f}x{y2-y1:.0f}px c=({(x1+x2)/2:.0f},{(y1+y2)/2:.0f})"]
                else:
                    lo = max([d["conf"] for d in rec["det"] if d["name"] == "red_ball"], default=None)
                    caps = [f"{s*per+k+1}. f{rec['file'][6:12]}  MISS",
                            f"best lo-conf={lo:.2f}" if lo else "no red_ball output"]
                entries.append((img, caps, best is not None))
            p = make_sheet(entries, sdir / f"sheet_{s+1:02d}.jpg", cols=cols,
                           title=f"Plan B - YOLO11n detect ({args.weights}) hit thr>={thr} "
                                 f"- sorted by ball box area asc - {s+1}/{n_sheets}")
            made.append(p)
        print(f"✅ sheets: {len(made)} 张 → {sdir}/  (每张 {per} 图，{cols} 列)")


if __name__ == "__main__":
    main()
