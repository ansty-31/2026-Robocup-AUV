# -*- coding: utf-8 -*-
"""gate/mock.py — MockGateBackend：脚本化进近/穿门检测序列（SIM/测试）

按帧序号推进一段"进近门"虚拟轨迹（可用 pose_fn 注入自定义），输出带 4 角点的
Det（角点缺失/整门消失可脚本化），供 GateTask 在无硬件下闭环验证逻辑。

用法示例：
    backend = MockGateBackend(camera=board_camera())
    hub.register("gate", backend)
"""
from __future__ import annotations

import numpy as np

from common.detector import Det
from gate.geometry import object_points


class MockGateBackend(object):
    """每 detect() 前进一帧；pose_fn(f) -> (rvec, tvec) 或 None(=目标消失)。"""

    def __init__(self, camera, frame_w=None, frame_h=None,
                 z0=5.5, step=0.06, cross_z=0.7,
                 lat_bias=0.18, align_frames=50,
                 score=0.92, conf_ok=0.92,
                 pose_fn=None,
                 # 缺角脚本：{z窗口下限: [角点id...]} → 该距离段强制 conf=0
                 drop_bands=None,
                 # 全缺角(仅 bbox 可信) 距离段：模拟"角不足但整框在" → coarse/HOLD
                 coarse_bands=None):
        self.camera = camera
        self.w = frame_w or camera.width
        self.h = frame_h or camera.height
        self.z0 = float(z0)
        self.step = float(step)
        self.cross_z = float(cross_z)
        self.lat_bias = float(lat_bias)
        self.align_frames = int(align_frames)
        self.score = score
        self.conf_ok = conf_ok
        self._pose_fn = pose_fn
        self.drop_bands = drop_bands or {}       # {下限z: [kpt idx]}
        self.coarse_bands = coarse_bands or []   # [(z_lo, z_hi), ...]
        self._f = 0
        self.last_pose = None

    # ------------------------------------------------------------ 轨迹
    def _default_pose(self, f):
        z = self.z0 - self.step * f
        if z <= self.cross_z:                     # 穿过后目标消失
            return None
        lat = self.lat_bias * max(0.0, 1.0 - f / max(self.align_frames, 1))
        rvec = np.array([0.02, -0.01, 0.0], np.float64)
        tvec = np.array([lat, 0.0, z], np.float64).reshape(3, 1)
        return rvec, tvec

    def _pose(self):
        if self._pose_fn is not None:
            return self._pose_fn(self._f)
        return self._default_pose(self._f)

    # ------------------------------------------------------------ 前端接口
    def detect(self, frame):
        self._f += 1
        pose = self._pose()
        if pose is None:
            return []
        rvec, tvec = pose
        self.last_pose = pose
        obj3 = object_points()
        uv = self.camera.project(obj3, rvec, tvec)
        z = float(tvec.ravel()[2])

        # 整门消失（全出画面/不可信）→ coarse 空
        conf = np.full(4, self.conf_ok, np.float32)
        for lo, _idx in self.drop_bands.items():   # 逐带：角点缺
            if z <= lo:
                for i in _idx:
                    conf[i] = 0.0
                break
        for lo, hi in self.coarse_bands:           # 整框可信但角点全缺
            if lo <= z <= hi:
                conf[:] = 0.0
                break

        x0, y0 = int(uv[:, 0].min()), int(uv[:, 1].min())
        x1, y1 = int(uv[:, 0].max()), int(uv[:, 1].max())
        return [Det("gate", self.score,
                    max(0, x0), max(0, y0),
                    min(self.w, x1) - max(0, x0),
                    min(self.h, y1) - max(0, y0),
                    kpts=uv, kpt_conf=conf)]
