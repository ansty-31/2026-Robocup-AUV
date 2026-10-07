#!/usr/bin/env python3
"""annotate_corners.py — 手动标门框 4 角点（键盘/鼠标极简版），存 JSON 供与模型预测对比。

用法（本机有显示时）：
  python scripts/1_prepare/annotate_corners.py --dir data/derived/landmark_check_150 \
      --out data/derived/landmark_check_150/manual_corners.json

操作：左键依次点 **TL → TR → BR → BL**（点满 4 个自动存并下一张）
  u 撤销上一点   r 重标本张   n 跳过（本张没有门）   b 上一张
  o 显示/隐藏模型框（只画框不画角点，便于确认"标哪一个门"）   q 保存退出
约定：标**画面里最近（最大）的那一个门**；坐标存 640 输入空间的像素。
"""
from __future__ import annotations
import argparse, json, pathlib
import cv2
import numpy as np

NAMES = ["TL", "TR", "BR", "BL"]
COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dir", required=True, help="图片目录（可含子目录）")
    ap.add_argument("--out", required=True, help="输出 JSON 路径")
    ap.add_argument("--pred", default=None, help="模型 predictions.json（配合 o 键显示框）")
    ap.add_argument("--start", type=int, default=0, help="从第几张开始（0 基）")
    ap.add_argument("--win", type=int, default=900, help="显示边长（像素）")
    a = ap.parse_args()

    root = pathlib.Path(a.dir).resolve()
    files = sorted(root.rglob("*.jpg"))
    if not files:
        raise SystemExit(f"❌ {root} 里没有图片")
    out = pathlib.Path(a.out)
    data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    pred = json.loads(pathlib.Path(a.pred).read_text(encoding="utf-8")) if a.pred else {}

    i, pts, show_pred = a.start, [], False
    win = "annotate"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append([round(x / scale, 1), round(y / scale, 1)])

    cv2.setMouseCallback(win, on_mouse)
    while 0 <= i < len(files):
        p = files[i]
        rel = p.relative_to(root).as_posix()
        img = cv2.imread(str(p))
        scale = a.win / img.shape[0]
        done = data.get(rel)
        while True:
            vis = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if show_pred:
                for j, d in enumerate(pred.get(rel, {}).get("instances", [])):
                    x1, y1, x2, y2 = [int(v * scale) for v in d["box"]]
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 1)
                    cv2.putText(vis, f"#{j + 1} {d['conf']:.2f}", (x1, max(14, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            shown = pts if pts else (done or [])
            for k, (x, y) in enumerate(shown[:4]):
                cv2.circle(vis, (int(x * scale), int(y * scale)), 6, COLORS[k], -1)
                cv2.putText(vis, NAMES[k] if k < len(NAMES) else "", (int(x * scale) + 8, int(y * scale) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[k], 2, cv2.LINE_AA)
            if len(shown) == 4:
                q = np.array([[int(x * scale), int(y * scale)] for x, y in shown], np.int32)
                cv2.polylines(vis, [q], True, (255, 255, 255), 1, cv2.LINE_AA)
            tag = "DONE" if done else "TODO"
            cv2.putText(vis, f"[{i + 1}/{len(files)}] {rel}  {tag}  点:{len(pts)}  "
                             f"o=模型框 q=退出", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, f"[{i + 1}/{len(files)}] {rel}  {tag}  点:{len(pts)}  "
                             f"o=模型框 q=退出", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, vis)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"已保存 {len(data)} 张 → {out}")
                cv2.destroyAllWindows()
                return
            if key == ord("o"):
                show_pred = not show_pred
            elif key == ord("u") and pts:
                pts.pop()
            elif key == ord("r"):
                pts = []
            elif key == ord("n"):
                data.pop(rel, None)
                out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
                i += 1
                pts = []
                break
            elif key == ord("b"):
                i -= 1
                pts = []
                break
            if len(pts) == 4:
                data[rel] = pts
                out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
                i += 1
                pts = []
                break


if __name__ == "__main__":
    main()
