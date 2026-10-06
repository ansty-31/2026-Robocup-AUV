# -*- coding: utf-8 -*-
"""grab/percept/cage_color.py — 颜色校验：在**下视画面**的笼区 ROI 里读"目标色像素占比(percent)"。

用户口径（2026-10-06）：夹取全程用下视相机；夹起后"检查是不是目标颜色" = **读 percent**
（笼区 ROI 内目标色像素占比），不是跑分类器。

为什么不新写掩膜：目标就是红球，`cv_ball.red_mask_fast` 那套阈值是 700 帧真图扫出来的工作点，
就地复用 ⇒ 全程只有**一份**颜色定义，改阈值只改一处（`cfg/vision.yaml` 的 `grab.cv`）。

⚠️ **ROI 与阈值都是占位**：`comm.grab.verify.roi` / `red_percent_min` 必须在下视真图上标
（方法：`preview_detect.py --grab --show` 看球进笼后画面里目标色块落在哪，再把 ROI 收紧到它）。
"""
from __future__ import annotations

import numpy as np

from handling.percept.cv_ball import Params, red_mask_fast


def _rect_from_roi(roi, w, h):
    """归一化 ROI `[x0, y0, w, h]`（占整幅比例）→ 像素矩形 `(x1, y1, x2, y2)`；非法 → None。"""
    if not roi or len(roi) != 4:
        return None
    try:
        x0, y0, rw, rh = (float(v) for v in roi)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(v) for v in (x0, y0, rw, rh)):
        return None
    x1, y1 = int(round(x0 * w)), int(round(y0 * h))
    x2, y2 = int(round((x0 + rw) * w)), int(round((y0 + rh) * h))
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def red_percent(frame, roi, params: Params | None = None):
    """ROI 内目标色（红）像素占比，∈ [0,1]。帧/ROI 非法 → `None`（**不是 0**：0 会被当成"确实不是目标色"）。"""
    if frame is None:
        return None
    try:
        h, w = frame.shape[:2]
    except Exception:
        return None
    rect = _rect_from_roi(roi, w, h)
    if rect is None:
        return None
    x1, y1, x2, y2 = rect
    sub = frame[y1:y2, x1:x2]
    if sub.size == 0:
        return None
    p = params or Params.from_cfg()
    try:
        m, _redness = red_mask_fast(sub, p)      # 返回 (mask 0/255, redness)
    except Exception:
        return None
    n = int(np.count_nonzero(m))
    return float(n) / float(m.size)
