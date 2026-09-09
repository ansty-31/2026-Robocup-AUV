# -*- coding: utf-8 -*-
"""camera.py — 相机：前视 / 下视分离，各自可 sim 或真机，真机失败可软回退 sim

配置：cfg/vision.yaml camera.front / camera.down（各自 type/device/宽高/fps/标定）。
调用：create_camera("front" | "down")；返回对象仅实现 read()。
"""
import numpy as np

import base.settings as S

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class Camera(object):
    def __init__(self, width, height, fps):
        self.width, self.height, self.fps = width, height, fps

    def read(self):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 虚拟相机（前视/下视各自仿真；下视带红色标示线供 back 真实颜色逻辑测试）
# ---------------------------------------------------------------------------
class SimCamera(Camera):
    def __init__(self, which):
        cfg = S.vision.camera[which]
        super().__init__(cfg.width, cfg.height, cfg.fps)
        self.which = which
        self._rng = np.random.default_rng(20260904)
        self._frame = 0

    def read(self):
        h, w = self.height, self.width
        base = self._rng.integers(0, S.vision.sim.noise + 1,
                                  (h, w, 3), dtype=np.uint8)
        if self.which == "down":
            line = S.vision.sim.down.red_line
            if line.enable:
                lh = max(1, line.width // 2)
                y0 = h - lh
                band_w = max(8, int(w * 0.6))
                off = max(-1.0, min(1.0, float(line.dx_offset)))
                cx = int(w * (0.5 + 0.5 * off))
                x0 = max(0, min(w - band_w, cx - band_w // 2))
                base[y0:, x0:x0 + band_w, 2] = 220
                base[y0:, x0:x0 + band_w, 0] = 20
                base[y0:, x0:x0 + band_w, 1] = 20
        self._frame += 1
        return base


# ---------------------------------------------------------------------------
# 真机后端
# ---------------------------------------------------------------------------
class UsbCamera(Camera):
    def __init__(self, cfg):
        if not HAS_CV2:
            raise RuntimeError("需要 opencv-python(cv2) 打开 USB 相机")
        super().__init__(cfg.width, cfg.height, cfg.fps)
        self.dev = cfg.device
        self._cap = cv2.VideoCapture(self.dev)
        if not self._cap.isOpened():
            raise RuntimeError("无法打开相机 %s" % self.dev)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        ok, frame = self._cap.read()
        return frame if ok else None


class MipiCamera(Camera):
    """下视 IMX415（MIPI/CSI）占位：板端接入后实现（可先经 V4L2 映射）。"""

    def __init__(self, cfg):
        super().__init__(cfg.width, cfg.height, cfg.fps)
        raise NotImplementedError(
            "下视 MIPI 相机尚未接入：确认 CSI 驱动节点后实现本类")


# ---------------------------------------------------------------------------
# 工厂：front/down 各自按 yaml type 实例化；真机失败按 fallback_sim 软回退
# ---------------------------------------------------------------------------
def create_camera(which, fallback_sim=None):
    cfg = S.vision.camera[which]
    typ = cfg.type
    fallback = S.vision.camera.fallback_sim if fallback_sim is None \
        else fallback_sim
    try:
        if typ == "sim":
            return SimCamera(which)
        if typ == "usb":
            return UsbCamera(cfg)
        if typ == "mipi":
            return MipiCamera(cfg)
        raise RuntimeError("未知相机类型 %s.%s" % (which, typ))
    except Exception as e:
        if fallback:
            print("[CAM] %s 相机(%s)不可用：%s；软回退 sim" % (which, typ, e))
            return SimCamera(which)
        raise
