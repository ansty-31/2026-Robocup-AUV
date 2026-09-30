# -*- coding: utf-8 -*-
"""preprocess.py — 目标识别前的图像预处理（推理链路，**单域 D+wb**）
raw → 缩放到模型方形输入（默认 640x640）→ 画面补偿 enhance(WB→CLAHE→gamma)
→ 640 上去畸变 remap（可选，需标定 yaml；与训练链路 prepare_frames.py 一致）
（A 域那条链 `remap@720p → resize640 → enhance` 已于 2026-10-01 退役归档，见 bak/retired/Adomain_20261001/）
（训练数据集制作在 PC 工程 RDKX5-YOLOv11n- 中完成，不在本工程范围）
参数源：cfg/vision.yaml（image.* / camera.*_calibration / model.input_size）"""
import os

import numpy as np

import base.cfg.settings as S

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

RAW_W, RAW_H = 1280, 720          # 标定 yaml 的原生分辨率（yaml 未写时的缺省）

def calibration_maps_640(path, size=640):
    """640 域去畸变映射：dest(去畸变 size×size) ← src(畸变 size×size)
    与 PC 工程 `scripts/1_prepare/map_pose_dataset.py` 的「D 域」严格同式：
    S = diag(size/RAW_W, size/RAW_H, 1)
    nk640 = S @ nk720 ;  K640 = S @ K
    initUndistortRectifyMap(K640, dist, None, nk640, (size,size))
    几何上与「先 remap@720p 再缩放」等价（因为 nk640 = S·nk720）。"""
    cv2 = _cv2()
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError("camera calibration missing: %s" % path)
    k = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("distortion_coefficients").mat()
    w = int(fs.getNode("image_width").real() or RAW_W)
    h = int(fs.getNode("image_height").real() or RAW_H)
    fs.release()
    if k is None or dist is None:
        raise ValueError("invalid camera calibration: %s" % path)
    sc = np.diag([size / float(w), size / float(h), 1.0])
    nk720, _ = cv2.getOptimalNewCameraMatrix(k, dist, (w, h), 0, (w, h))
    return cv2.initUndistortRectifyMap(sc @ k, dist, None, sc @ nk720,
                                       (size, size), cv2.CV_16SC2)

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
    """白平衡的 256 项 LUT：与原 float 式 np.clip(v*g,0,255).astype(uint8) 逐像素等价"""
    key = tuple(float(g) for g in gains)
    luts = _WB_CACHE.get(key)
    if luts is None:
        idx = np.arange(256, dtype=np.float32)
        luts = [np.clip(idx * g, 0, 255).astype(np.uint8) for g in key]
        _WB_CACHE[key] = luts
    return luts

def _fused_luts(gains, gamma):
    """clip==0 时把 白平衡∘gamma 融合成每通道一张 LUT（省一次整幅 pass）"""
    key = (tuple(float(g) for g in gains), float(gamma))
    luts = _FUSED_CACHE.get(key)
    if luts is None:
        gg = _gamma_lut(gamma)
        luts = [gg[w] for w in _wb_luts(gains)]
        _FUSED_CACHE[key] = luts
    return luts

def enhance(frame, gains, clip, gamma):
    """画面补偿：白平衡通道增益 → LAB-L CLAHE(clip>0 时) → gamma LUT。
    融合进同一张 LUT，每通道只做一次 cv2.LUT。
    CLAHE 对象与 gamma LUT 缓存复用（原先每帧 createCLAHE / 重建 LUT）。"""
    cv2 = _cv2()
    if clip <= 0:                      # 无 CLAHE：WB 与 gamma 融合，单次 LUT
        w = _fused_luts(gains, gamma)
        b, g, r = cv2.split(frame)
        return cv2.merge((cv2.LUT(b, w[0]), cv2.LUT(g, w[1]), cv2.LUT(r, w[2])))
    w = _wb_luts(gains)
    b, g, r = cv2.split(frame)
    f = cv2.merge((cv2.LUT(b, w[0]), cv2.LUT(g, w[1]), cv2.LUT(r, w[2])))
    if clip > 0:
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = _clahe(clip).apply(l)
        f = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return cv2.LUT(f, _gamma_lut(gamma))

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


    def process(self, frame_bgr):
        """raw(BGR,任意尺寸) → 缩放 → 画面补偿 → 去畸变(640 域)，即 **D 链**（单域，2026-10-01 起）。
        顺序有讲究：enhance 本来就在 640 上做（与训练链路一致，PC 工程只产 D+wb）。
        几何：kpt 回缩放后落在**去畸变 720p** 空间，相机模型一律 `rectified=True` 的 nk720，
        **不要**改成 raw K/D。（改链路前先看 doc/设计/gate_pose_decode_spec.md 与权重代次。）"""
        cv2 = _cv2()
        h, w = frame_bgr.shape[:2]
        f = frame_bgr
        if (w, h) != (self.size, self.size):
            f = cv2.resize(f, (self.size, self.size),
                           interpolation=cv2.INTER_LINEAR)
        f = enhance(f, self.gains, self.clip, self.gamma)
        maps = self._maps640()
        if maps is not None:
            f = cv2.remap(f, maps[0], maps[1], cv2.INTER_LINEAR)
        return f

    def _maps640(self):
        """640 域去畸变映射（None = 不做去畸变）"""
        if not self.undistort or not self.calib_path:
            return None
        if "m640" not in self._maps:
            if not os.path.exists(self.calib_path):
                raise FileNotFoundError("undistort=true 但标定文件缺失: %s"
                                        % self.calib_path)
            self._maps["m640"] = calibration_maps_640(self.calib_path, self.size)
        return self._maps["m640"]

    def scale(self, frame_w, frame_h):
        """模型坐标 → 原始帧坐标的缩放系数。"""
        return frame_w / self.size, frame_h / self.size