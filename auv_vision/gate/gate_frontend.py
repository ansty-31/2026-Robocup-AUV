# -*- coding: utf-8 -*-
"""gate/gate_frontend.py — keypoint→2D/3D 对应与 mode 判定（§4.3 前端适配层）

与几何内核解耦：本模块只负责把"一次检测（bbox + 4 角点 + 每点置信度）"翻译成
下游可用的 (mode, img2, obj3) 或粗对准信息；位姿解算交给 gate.geometry。

mode 定义（对应 §4.6 降级表 / §5.3 子状态）：
  full        4 角可见 → IPPE 全位姿
  p3p         恰好 3 角可见 → P3P（需上一帧消歧）
  width       对向 2 角(TL,TR 或 BL,BR)可见 → 已知实宽测距 + 中点瞄准
  coarse      角不足但整门框可信 → 框心对准（运动仲裁在 GateTask）
  none        无可信检测
"""
from __future__ import annotations

import numpy as np

MODE_FULL = "full"
MODE_P3P = "p3p"
MODE_WIDTH = "width"
MODE_COARSE = "coarse"
MODE_NONE = "none"

# 角点语义索引（训练顺序 = object_points 顺序 TL,TR,BR,BL）
_ID_TL, _ID_TR, _ID_BR, _ID_BL = 0, 1, 2, 3


def parse_kpt_mode(kpts, kpt_conf, conf_thr=0.5, min_pair_ok=True):
    """由检测的角点/置信度判定 mode 与有效角点 id 列表。

    Returns:
        (mode, ids)：ids 与 object_points 同序索引，用于裁 2D/3D 对应。
    """
    kpts = np.asarray(kpts, np.float32)
    conf = np.asarray(kpt_conf, np.float32)
    if kpts.ndim != 2 or kpts.shape[0] < 4:
        return MODE_COARSE, []
    valid = conf >= conf_thr
    n = int(valid.sum())
    if n >= 4:
        return MODE_FULL, [_ID_TL, _ID_TR, _ID_BR, _ID_BL]
    if n == 3:
        return MODE_P3P, [int(i) for i in range(4) if valid[i]]
    if n == 2 and min_pair_ok:
        pair = [int(i) for i in range(4) if valid[i]]
        if set(pair) in ({_ID_TL, _ID_TR}, {_ID_BL, _ID_BR}):
            return MODE_WIDTH, pair        # 对向角：已知实宽测距
    if n >= 1:
        return MODE_COARSE, [int(i) for i in range(4) if valid[i]]
    return MODE_COARSE, []


def build_correspondence(obj3, ids):
    """按 ids 裁出 (obj3_sub(N,3), img2_sub(N,2))；与 kpts 顺序一致由调用方传入 img2。"""
    obj3 = np.asarray(obj3, np.float32)
    return obj3[ids]


def width_range_depth(u1, u2, fx, width_m):
    """对向 2 角测距（进近段近似正对）：Z ≈ fx·W/|Δu|。"""
    d = float(abs(u2 - u1))
    if d < 1e-6:
        return None
    return float(fx * width_m / d)


def bbox_center(det):
    """Det → 整门框中心像素（coarse 档瞄准用）。"""
    return (det.x + det.w / 2.0, det.y + det.h / 2.0)
