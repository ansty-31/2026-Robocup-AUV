#!/usr/bin/env python3
"""夹取小球任务 · 两条识别线（传统 CV / YOLO）的公共工具。

只做三件事：letterbox 到 640×640（与 ultralytics 训练、板端 nv12 输入同几何）、
把 640 坐标还原回原图坐标、以及把标注结果拼成 sheet（联系表）。

**与板端对齐的口径**：`letterbox()` 复刻 ultralytics `LetterBox` 的默认行为
（scaleup=True、auto=False、stride=32、pad 用 114 灰），所以 1280×720 → 640×360 →
上下各 pad 140 px → 640×640。这正是训练时的几何，也是板端 nv12 需喂的形状。
"""

from __future__ import annotations

import cv2
import numpy as np

STRIDE = 32
PAD_VALUE = 114


def letterbox(img: np.ndarray, dst: int = 640):
    """返回 (letterboxed 图 dst×dst, ratio, pad_x, pad_y)。原图坐标 → 640 坐标：
    x640 = x * ratio + pad_x。逆变换：x = (x640 - pad_x) / ratio。"""
    h, w = img.shape[:2]
    r = min(dst / h, dst / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    dw, dh = (dst - nw) / 2, (dst - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
    return out, r, left, top


def unletterbox_points(pts, r, px, py):
    """(x,y)[640 坐标] → 原图坐标"""
    return np.asarray(pts, dtype=float).copy() * [[1 / r, 1 / r]] - [px / r, py / r]


def unletterbox_box(box, r, px, py):
    x1, y1, x2, y2 = box
    return [(x1 - px) / r, (y1 - py) / r, (x2 - px) / r, (y2 - py) / r]


# --------------------------- sheet（联系表） ---------------------------

PALETTE = {0: (255, 160, 0), 1: (0, 255, 0), 2: (0, 0, 255)}  # blue_ball, gate, red_ball


def _put(img, text, org, color=(0, 255, 255), scale=0.55, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def ascii_only(s: str) -> str:
    """cv2.putText 用的是 Hershey 字体，画不了「×」「·」等非 ASCII 字符（会变成 ?）。
    标题/图注一律先过这里。"""
    return s.encode("ascii", "replace").decode("ascii")


def make_sheet(entries, out_path, cols=4, tile=(480, 270), title=None):
    """entries: [(img_bgr, caption_lines:list[str], ok:bool, extra_color)] →
    拼成 cols 列的联系表 PNG。ok=False 的片子加一圈红框，一眼能数出漏检。"""
    tw, th = tile
    n = len(entries)
    rows = (n + cols - 1) // cols
    header = 34 if title else 0
    sheet = np.full((header + rows * th, cols * tw, 3), 32, np.uint8)
    for i, (img, caps, ok) in enumerate(entries):
        r, c = divmod(i, cols)
        tile_img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        y0 = header + r * th
        x0 = c * tw
        sheet[y0:y0 + th, x0:x0 + tw] = tile_img
        cv2.rectangle(sheet, (x0, y0), (x0 + tw - 1, y0 + th - 1),
                      (60, 60, 60), 1)
        if not ok:
            cv2.rectangle(sheet, (x0 + 1, y0 + 1), (x0 + tw - 2, y0 + th - 2),
                          (0, 0, 255), 3)
        else:
            cv2.rectangle(sheet, (x0 + 1, y0 + 1), (x0 + tw - 2, y0 + th - 2),
                          (0, 200, 0), 1)
        for k, cap in enumerate(caps):
            _put(sheet, ascii_only(cap), (x0 + 6, y0 + 18 + k * 17),
                 color=(0, 255, 0) if ok else (0, 80, 255), scale=0.45)
    if title:
        cv2.putText(sheet, ascii_only(title), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), sheet)
    return out_path
