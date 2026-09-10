# -*- coding: utf-8 -*-
"""camera.py — 相机：前视 / 下视分离，各自可 sim 或真机，真机失败可软回退 sim

配置：cfg/vision.yaml camera.front / camera.down（各自 type/device/宽高/fps/标定）。
调用：create_camera("front" | "down")；返回对象仅实现 read()。
"""
import os

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
        pusher = get_stream_pusher()          # 仿真也能推（无硬件时验证全链路）
        if pusher is not None:
            ok, enc = cv2.imencode(".jpg", base, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                pusher.offer(enc)
        return base


# ---------------------------------------------------------------------------
# 推流（可选）：把相机原始 MJPEG 帧零转码转发给水面 PC
#   开关：cfg/vision.yaml 顶层 stream.*（手动模式由根目录 manual.sh 用环境变量覆盖）
#   实现：manual/stream.py
#   关键点：推流不另外开相机（UVC 只允许一个进程取流），而是"谁在用相机谁顺带推"，
#   因此识别（main.py）与录像（recorder.py）各自跑时都能同时推流，互不冲突。
#   环境变量覆盖：AUV_STREAM=1/0、AUV_STREAM_HOST、AUV_STREAM_PORT、
#                AUV_STREAM_FPS、AUV_STREAM_PKT
# ---------------------------------------------------------------------------
_stream_pusher = None


def get_stream_pusher():
    """惰性创建推流器；未开启或缺少依赖时返回 None（不影响主流程）。"""
    global _stream_pusher
    if _stream_pusher is not None:
        return _stream_pusher
    try:
        cfg = S.vision.stream
    except AttributeError:
        cfg = {}
    env = os.environ.get
    enable = bool(getattr(cfg, "enable", False))
    if env("AUV_STREAM") == "1":
        enable = True
    elif env("AUV_STREAM") == "0":
        enable = False
    if not enable:
        return None
    host = env("AUV_STREAM_HOST") or getattr(cfg, "host", None)
    if not host:
        print("[CAM] 推流已开启但没配 host，跳过")
        return None
    port = int(env("AUV_STREAM_PORT") or getattr(cfg, "port", 5000))
    pkt = int(env("AUV_STREAM_PKT") or getattr(cfg, "pkt", 8000))
    fps = float(env("AUV_STREAM_FPS") or getattr(cfg, "stream_fps", 30))
    try:
        from manual import stream as stream_mod        # 手动模式三件套之一（推流库）
    except ImportError:
        try:
            import stream as stream_mod                # 兼容：stream.py 与 camera.py 同目录
        except ImportError:
            print("[CAM] 找不到 manual/stream.py，跳过推流")
            return None
    try:
        _stream_pusher = stream_mod.MjpegPusher(host, port, pkt=pkt, stream_fps=fps)
        print("[CAM] 推流已开启 → %s:%d（原始 MJPEG 零转码，上限 %.0f fps）" % (host, port, fps))
    except Exception as e:
        print("[CAM] 推流开启失败（不影响识别/录像）：%s" % e)
        _stream_pusher = None
    return _stream_pusher


# ---------------------------------------------------------------------------
# 真机后端
# ---------------------------------------------------------------------------
class UsbCamera(Camera):
    def __init__(self, cfg):
        if not HAS_CV2:
            raise RuntimeError("需要 opencv-python(cv2) 打开 USB 相机")
        super().__init__(cfg.width, cfg.height, cfg.fps)
        self.dev = cfg.device
        self._cap = cv2.VideoCapture(self.dev, getattr(cv2, "CAP_V4L2", 0))
        if not self._cap.isOpened():
            raise RuntimeError("无法打开相机 %s" % self.dev)
        # OpenCV 的 V4L2 后端默认协商 YUYV：本相机 720p YUYV 只有 **9 fps**，
        # MJPG 有 60 fps。所以显式选 MJPG，并用 CONVERT_RGB=0 直接拿原始 JPEG
        # （退流零转码；识别侧再 cv2.imdecode 成 BGR，代价与让 OpenCV 内部解码相同）。
        self._raw = False
        try:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        try:
            self._cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            ok, buf = self._cap.read()
            if ok and buf is not None and bytes(buf.reshape(-1)[:3]) == b"\xff\xd8\xff":
                self._raw = True
        except Exception:
            self._raw = False
        if not self._raw:
            self._cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        # 以驱动实际协商结果为准（本相机 720p MJPG 只有 60fps 档，写 30 也按 60 出）；
        # self.fps 保持配置值不变，避免影响 recorder.py 的落盘帧率语义。
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.negotiated_fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 0.0

    def read(self):
        ok, buf = self._cap.read()
        if not ok or buf is None:
            return None
        pusher = get_stream_pusher()
        if self._raw:
            if pusher is not None:
                pusher.offer(buf)                        # 原始 JPEG：零转码（CPU≈0）
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)   # 识别/录像用 BGR
        if pusher is not None:                           # 相机不支持原始 JPEG：回退重编码
            ok2, enc = cv2.imencode(".jpg", buf, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok2:
                pusher.offer(enc)
        return buf


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
