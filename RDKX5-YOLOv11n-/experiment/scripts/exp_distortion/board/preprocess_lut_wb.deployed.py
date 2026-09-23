# -*- coding: utf-8 -*-
"""preprocess.py — 目标识别前的图像预处理（推理链路）

    去畸变 remap（可选，需标定 yaml）→ 画面补偿 enhance(WB→CLAHE→gamma)
    → 等比缩放到模型方形输入（默认 640x640）
（训练数据集制作在 PC 工程 RDKX5-YOLOv11n- 中完成，不在本工程范围）
参数源：cfg/vision.yaml（image.* / camera.*_calibration / model.input_size）
"""
import os

import numpy as np

import base.settings as S


def _cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        raise RuntimeError("本链路需要 opencv：pip install opencv-python"
                           "（板端: sudo apt install python3-opencv）")


# ---------------------------------------------------------------------------
# 链路步骤（纯函数，与板端逐条一致）
# ---------------------------------------------------------------------------
def calibration_maps(path, width, height):
    """由标定 yaml 生成去畸变 remap 映射（camera_matrix/distortion_coefficients）。"""
    cv2 = _cv2()
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError("camera calibration missing: %s" % path)
    k = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("distortion_coefficients").mat()
    fs.release()
    if k is None or dist is None:
        raise ValueError("invalid camera calibration: %s" % path)
    new_k, _ = cv2.getOptimalNewCameraMatrix(k, dist, (width, height), 0,
                                             (width, height))
    return cv2.initUndistortRectifyMap(k, dist, None, new_k,
                                       (width, height), cv2.CV_16SC2)


_CLAHE_CACHE = {}
_GAMMA_CACHE = {}


def _clahe(clip):
    c = _CLAHE_CACHE.get(clip)
    if c is None:
        c = _cv2().createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        _CLAHE_CACHE[clip] = c
    return c


def _gamma_lut(gamma):
    lut = _GAMMA_CACHE.get(gamma)
    if lut is None:
        lut = np.array([pow(i / 255.0, gamma) * 255 for i in range(256)],
                       dtype=np.uint8)
        _GAMMA_CACHE[gamma] = lut
    return lut


_WB_CACHE = {}
_FUSED_CACHE = {}


def _wb_luts(gains):
    """白平衡的 256 项 LUT：与 np.clip(v*g,0,255).astype(uint8) 逐像素等价"""
    key = tuple(float(g) for g in gains)
    luts = _WB_CACHE.get(key)
    if luts is None:
        idx = np.arange(256, dtype=np.float32)
        luts = [np.clip(idx * g, 0, 255).astype(np.uint8) for g in key]
        _WB_CACHE[key] = luts
    return luts


def _fused_luts(gains, gamma):
    """clip==0 时把 白平衡∘gamma 融合成每通道一张 LUT（省一次 pass）"""
    key = (tuple(float(g) for g in gains), float(gamma))
    luts = _FUSED_CACHE.get(key)
    if luts is None:
        gg = _gamma_lut(gamma)
        luts = [gg[w] for w in _wb_luts(gains)]
        _FUSED_CACHE[key] = luts
    return luts


def enhance(frame, gains, clip, gamma):
    """画面补偿：白平衡通道增益 → LAB-L CLAHE(clip>0 时) → gamma LUT。

    2026-09-23 板端实测提速（输出逐像素不变）：
      · 白平衡由 float 乘/裁剪/回 uint8（~31 ms/帧）改为 256 项 LUT（~12 ms）；
      · clip==0 时再把 gamma 融合进同一张 LUT → 每通道只做一次 cv2.LUT。
    板端端到端 72.1 → 37.3 ms/帧（含 clahe_clip=0）。
    """
    cv2 = _cv2()
    if clip > 0:
        w = _wb_luts(gains)
        b, g, r = cv2.split(frame)
        f = cv2.merge((cv2.LUT(b, w[0]), cv2.LUT(g, w[1]), cv2.LUT(r, w[2])))
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
        l, a, bb = cv2.split(lab)
        l = _clahe(clip).apply(l)
        f = cv2.cvtColor(cv2.merge((l, a, bb)), cv2.COLOR_LAB2BGR)
        return cv2.LUT(f, _gamma_lut(gamma))
    w = _fused_luts(gains, gamma)
    b, g, r = cv2.split(frame)
    return cv2.merge((cv2.LUT(b, w[0]), cv2.LUT(g, w[1]), cv2.LUT(r, w[2])))



# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------
class ModelPreprocessor(object):
    """推理链路预处理对象（缓存去畸变映射）。"""

    def __init__(self, undistort=None, gains=None, clip=None, gamma=None,
                 size=None, calib_path=None):
        self.undistort = S.vision.image.undistort if undistort is None else undistort
        self.gains = S.vision.image.white_balance_bgr if gains is None else list(gains)
        self.clip = S.vision.image.clahe_clip if clip is None else clip
        self.gamma = S.vision.image.gamma if gamma is None else gamma
        self.size = S.vision.model.input_size if size is None else int(size)
        self.calib_path = calib_path
        self._maps = {}                # {(w,h): remap 映射}，按分辨率缓存

    def _maps_for(self, w, h):
        if not self.undistort or not self.calib_path:
            return None
        key = (w, h)
        if key not in self._maps:
            if not os.path.exists(self.calib_path):
                raise FileNotFoundError("undistort=true 但标定文件缺失: %s"
                                        % self.calib_path)
            self._maps[key] = calibration_maps(self.calib_path, w, h)
        return self._maps[key]

    def process(self, frame_bgr):
        """raw(BGR,任意尺寸) → 去畸变 → 缩放 → 补偿(在模型输入尺寸上做，省时)。"""
        cv2 = _cv2()
        h, w = frame_bgr.shape[:2]
        maps = self._maps_for(w, h)
        f = frame_bgr
        if maps is not None:
            f = cv2.remap(f, maps[0], maps[1], cv2.INTER_LINEAR)
        if (w, h) != (self.size, self.size):
            f = cv2.resize(f, (self.size, self.size),
                           interpolation=cv2.INTER_LINEAR)
        f = enhance(f, self.gains, self.clip, self.gamma)   # 640 上做，约 1/3 耗时
        return f

    def scale(self, frame_w, frame_h):
        """模型坐标 → 原始帧坐标的缩放系数。"""
        return frame_w / self.size, frame_h / self.size