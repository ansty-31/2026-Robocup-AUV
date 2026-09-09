# -*- coding: utf-8 -*-
"""gate/gate_detector.py — 板端相机模型 + gate 后端装配（组合根用，非纯函数）

- board_camera()：按 settings 构造与"检测坐标域"一致的 CameraModel（§4.2）
- build_gate_backend()：按 model.mode 选择 Mock 或真实 keypoint 后端；
  真实模型缺失时返回 None（主程序据此 skip gate，不中断其它任务）
"""
from __future__ import annotations

import os

import base.settings as S
from gate.geometry import CameraModel


def board_camera():
    """板上前视相机模型（raw/rectified 与 preprocess 一致，§4.2）。"""
    front = S.vision.camera.front
    path = front.calibration
    rectified = bool(S.vision.image.undistort)
    candidates = [path]
    if path:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates.append(os.path.join(here, "cfg", os.path.basename(path)))
    for p in candidates:
        if p and os.path.exists(p):
            return CameraModel.from_yaml(p, rectified=rectified)
    # 标定缺失：退化近似（仅供 mock/联调，真实运行会先在 preprocess 报缺失）
    print("[GATE] ⚠️ 标定 %s 不存在，用近似针孔(仅 mock/联调)" % path)
    return CameraModel.pinhole(int(front.width), int(front.height),
                               fx=float(front.width) * 0.61,
                               fy=float(front.width) * 0.61,
                               cx=float(front.width) / 2.0,
                               cy=float(front.height) / 2.0)


def build_gate_backend():
    """返回 gate 检测后端或 None。

    mode=mock → MockGateBackend（无硬件闭环调试）
    其它     → 真实 keypoint 后端（需 vision.model.task_models.gate 且权重存在）
    """
    if S.vision.model.mode == "mock":
        from gate.mock import MockGateBackend
        return MockGateBackend(board_camera())
    task_cfg = S.get("vision.model.task_models.gate")
    if not task_cfg:
        print("[GATE] 未配置 model.task_models.gate → gate 跳过")
        return None
    path = task_cfg.get("path")
    if not path or not os.path.exists(path):
        print("[GATE] 权重缺失 %s → gate 跳过（就绪后配置 vision.yaml 再启用）"
              % (path or "<空>"))
        return None
    from gate.gate_decode import GateKeypointBackend
    return GateKeypointBackend(
        path=path,
        labels=list(task_cfg.get("labels", ["gate"])),
        kpt_order=list(task_cfg.get("kpt_order", ["TL", "TR", "BR", "BL"])),
        camera=board_camera())
