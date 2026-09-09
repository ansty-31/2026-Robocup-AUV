# -*- coding: utf-8 -*-
"""gate/geometry.py — 门框位姿几何内核（纯函数，前端无关）

移植自 BumblebeeAS pose_estimator（NUS 水下机器人）：
  - utils/PinholeCamera.py        → CameraModel（raw / rectified 双域）
  - utils/pose_estimator.py       → gate_pose / plane_from_pose / backproject_to_plane
  - config/object_points.py       → object_points（门框外轮廓四角, z=0 平面）
设计文档：doc/算法说明-gate-PnP移植方案.md（§3/§4.2/§4.4/§4.5）

坐标约定（与文档一致）：
  - 图像/门框坐标系：x 向右、y 向下；门框外轮廓矩形 0.70(宽) × 0.50(高)，原点=矩形中心；
  - 4 角顺序 TL,TR,BR,BL（keypoint 训练顺序），z=0 平面 = 门平面，法向 +z 朝 AUV；
  - 相机域：PnP/反投影必须在"检测坐标所在域"做——
      undistort=true（板上默认）→ rectified K（getOptimalNewCameraMatrix, D=0）；
      undistort=false          → 原始 K + 原始 D（solvePnP 内部自行去畸变点）。
  依赖：numpy + cv2（缺 cv2 时在用到处报明确错误）。
"""
from __future__ import annotations

import os

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# 门框外轮廓尺寸（§3 已定，单位 m）
GATE_FRAME_W = 0.70
GATE_FRAME_H = 0.50

# 位姿可信深度范围（§4.4 质量检查初值）
Z_BOUNDS = (0.2, 15.0)
# 默认重投影合格阈值(px)
REPROJ_THR_PX = 20.0


def _need_cv2():
    if not HAS_CV2:
        raise RuntimeError("gate/geometry.py 需要 opencv：pip install opencv-python"
                           "（板端: sudo apt install python3-opencv）")


# ---------------------------------------------------------------------------
# 相机模型
# ---------------------------------------------------------------------------
class CameraModel(object):
    """针孔相机封装：原始域(带畸变) / 去畸变域(rectified K, D=0) 二选一。

    对应 Bumblebee `PinholeCamera`；`rectified=True` 时按板上 preprocess.py 同一
    方式计算 new_K（getOptimalNewCameraMatrix alpha=0）。
    """

    def __init__(self, width, height, fx, fy, cx, cy, dist=None, rectified=False):
        self.width = int(width)
        self.height = int(height)
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.rectified = bool(rectified)
        if dist is None:
            dist = [0.0] * 5
        if self.rectified:
            dist = [0.0] * 5          # 去畸变域：视为理想针孔
        self.d = np.asarray(dist, dtype=np.float64).reshape(-1)
        self._K = np.array([[self.fx, 0, self.cx],
                            [0, self.fy, self.cy],
                            [0, 0, 1]], dtype=np.float64)

    # ------------------------------------------------------------ 工厂
    @classmethod
    def from_yaml(cls, path, rectified=False):
        """由标定 yaml（FileStorage，calibrate_camera_video.py 输出格式）构造。

        rectified=True：与原图等尺寸的 rectified K（同 preprocess.calibration_maps）。"""
        _need_cv2()
        if not os.path.exists(path):
            raise FileNotFoundError("相机标定缺失: %s" % path)
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise FileNotFoundError("无法打开标定 yaml: %s" % path)
        try:
            k = fs.getNode("camera_matrix").mat()
            dist = fs.getNode("distortion_coefficients").mat()
            w = int(fs.getNode("image_width").real())
            h = int(fs.getNode("image_height").real())
        finally:
            fs.release()
        if k is None or dist is None:
            raise ValueError("标定 yaml 缺 camera_matrix/distortion_coefficients: %s" % path)
        if rectified:
            new_k, _ = cv2.getOptimalNewCameraMatrix(k, dist, (w, h), 0, (w, h))
            k, dist = new_k, np.zeros((1, 5), np.float64)
        return cls(w, h, k[0, 0], k[1, 1], k[0, 2], k[1, 2],
                   dist=dist.reshape(-1).tolist(), rectified=rectified)

    @classmethod
    def pinhole(cls, width, height, fx, fy, cx, cy):
        """无畸变理想针孔（合成测试/调试用）。"""
        return cls(width, height, fx, fy, cx, cy, dist=[0.0] * 5, rectified=True)

    # ------------------------------------------------------------ 访问
    def camera_matrix(self):
        return self._K

    def dist_coeffs(self):
        return self.d

    # ------------------------------------------------------------ 投影/反投影
    def project(self, obj3, rvec, tvec):
        """3D 点(相机系外, Nx3) → 像素(Nx2)。"""
        _need_cv2()
        uv, _ = cv2.projectPoints(np.asarray(obj3, np.float32),
                                  np.asarray(rvec, np.float64),
                                  np.asarray(tvec, np.float64),
                                  self.camera_matrix(), self.dist_coeffs())
        return np.asarray(uv).reshape(-1, 2)

    def backproject_z(self, x, y, Z):
        """已知深度 Z 反投影像素 → 相机系 3D 点（对应 Bumblebee backproject_pixel）。

        先经 undistortPoints 校正（raw 域带畸变时生效），再按 Z 缩放。"""
        _need_cv2()
        xn, yn = cv2.undistortPoints(
            np.array([[[x, y]]], dtype=np.float32),
            self.camera_matrix(), self.dist_coeffs())[0][0]
        return np.array([float(xn) * Z, float(yn) * Z, float(Z)], dtype=np.float64)


# ---------------------------------------------------------------------------
# 门框 3D 角点表（§3；顺序 = keypoint 训练顺序 TL,TR,BR,BL；mm 约定已转 m）
# ---------------------------------------------------------------------------
def object_points(W_m=GATE_FRAME_W, H_m=GATE_FRAME_H, from_front=True):
    """门框外轮廓四角 3D 点 (4,3)，z=0 平面（门平面），原点=矩形中心。

    from_front：前后完全对称、无朝向要求 → 恒 true（参数保留向后兼容/对称核对）。"""
    s = 1.0 if from_front else -1.0
    pts = [(-s * W_m / 2, -H_m / 2, 0.0),   # TL
           (s * W_m / 2, -H_m / 2, 0.0),    # TR
           (s * W_m / 2, H_m / 2, 0.0),     # BR
           (-s * W_m / 2, H_m / 2, 0.0)]    # BL
    return np.asarray(pts, dtype=np.float32)


def _rodrigues(rvec):
    _need_cv2()
    r, _ = cv2.Rodrigues(np.asarray(rvec, np.float64))
    return r


def reproj_rms(camera, obj3, img2, rvec, tvec):
    """重投影 RMS(px)：候选位姿质量度量。"""
    uv = camera.project(obj3, rvec, tvec)
    img = np.asarray(img2, np.float64)
    return float(np.sqrt(np.mean((uv - img) ** 2)))


# ---------------------------------------------------------------------------
# 位姿解算（§4.4）
# ---------------------------------------------------------------------------
def _iter_candidates(camera, obj3, img2, prev=None):
    """生成候选 (rvec,tvec) 列表。

    - ≥4 点（共面矩形）：solvePnPGeneric(IPPE)（cv2 4/5 兼容，取 rvecs/tvecs）；
      失败退回单次 solvePnP。
    - 恰好 3 点：必须有上帧猜值 prev（ITERATIVE + guess 收敛，无猜值 3 点不可靠）；
      为兼容 cv2 4 先试 solvePnPGeneric(P3P) 多解，失败再走 guess 路径。
    """
    K = camera.camera_matrix()
    D = camera.dist_coeffs()
    obj3 = np.asarray(obj3, np.float32)
    img2 = np.asarray(img2, np.float32)
    n = obj3.shape[0]
    cands = []
    if n >= 4:
        try:
            res = cv2.solvePnPGeneric(obj3, img2, K, D, flags=cv2.SOLVEPNP_IPPE)
            rvecs, tvecs = res[1], res[2]
            cands = list(zip(rvecs, tvecs))
        except Exception:
            cands = []
        if not cands:
            ok, r, t = cv2.solvePnP(obj3, img2, K, D, flags=cv2.SOLVEPNP_SQPNP)
            if ok:
                cands = [(r, t)]
    elif n == 3:
        try:                               # cv2 4.x：P3P 多解（cv2 5 此路径会 assert 失败）
            res = cv2.solvePnPGeneric(obj3, img2, K, D, flags=cv2.SOLVEPNP_P3P)
            rvecs, tvecs = res[1], res[2]
            cands = list(zip(rvecs, tvecs))
        except Exception:
            cands = []
        if not cands and prev is not None:
            r0, t0 = prev
            ok, r, t = cv2.solvePnP(obj3, img2, K, D,
                                    np.asarray(r0, np.float64),
                                    np.asarray(t0, np.float64),
                                    useExtrinsicGuess=True,
                                    flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                cands = [(r, t)]
    return cands


def gate_pose(camera, obj3, img2, prev=None,
              reproj_thr=REPROJ_THR_PX, z_bounds=Z_BOUNDS, refine=True):
    """门框位姿：由 2D↔3D 角点对估计 (rvec, tvec)（§4.4）。

    Args:
        camera: CameraModel（与检测坐标同域）
        obj3:   (N,3) 门框角点（object_points 顺序）
        img2:   (N,2) 像素角点（同序）
        prev:   (rvec_prev, tvec_prev) 上帧位姿；**3 点时必需**（消歧/收敛，§4.7）
        reproj_thr: 候选合格重投影 RMS(px)
        z_bounds:   tvec.z 可信范围 (min,max)
        refine:  是否 solvePnPRefineLM 精调（≥4 点建议开；3 点视 cv2 支持容错跳过）

    Returns:
        (rvec(3,1), tvec(3,1)) 或 None（点数<3 / 无合格候选 → 弃帧保持上帧）
    """
    _need_cv2()
    obj3 = np.asarray(obj3, np.float32)
    img2 = np.asarray(img2, np.float32)
    if obj3.shape[0] < 3 or obj3.shape[0] != img2.shape[0]:
        return None
    cands = _iter_candidates(camera, obj3, img2, prev=prev)
    if not cands:
        return None

    lo, hi = z_bounds
    best, best_score = None, float("inf")
    for r, t in cands:
        r = np.asarray(r, np.float64)
        t = np.asarray(t, np.float64)
        tz = float(t.ravel()[2])
        if not (lo <= tz <= hi):
            continue
        rms = reproj_rms(camera, obj3, img2, r, t)
        if rms > reproj_thr:
            continue
        score = rms
        if prev is not None:                 # 上帧距离惩罚：多解/歧义时倾向连贯
            score += 0.3 * float(np.linalg.norm(t.ravel() - np.asarray(prev[1], np.float64).ravel()))
        if score < best_score:
            best, best_score = (r, t), score
    if best is None:
        return None

    r, t = best
    if refine:
        try:
            if obj3.shape[0] >= 4 or prev is not None:
                ok, r, t = cv2.solvePnPRefineLM(obj3, img2, camera.camera_matrix(),
                                               camera.dist_coeffs(), r, t)
                if not ok:
                    r, t = best
        except Exception:
            r, t = best                        # 某些 cv2 版本/点数限制：容错保留原值
    return r, t


# ---------------------------------------------------------------------------
# 门平面与反投影（§4.5）
# ---------------------------------------------------------------------------
def plane_from_pose(rvec, tvec):
    """门平面方程（相机系）：n_c·p = rho。

    门框原点=矩形中心，门平面 z=0、法向 +z 朝 AUV → n_c = R·[0,0,1]，rho = n_c·tvec。"""
    R = _rodrigues(rvec)
    n = np.asarray(R[:, 2], np.float64).reshape(3)   # 门平面法向(相机系)
    rho = float(np.dot(n, np.asarray(tvec, np.float64).reshape(3)))
    return n, rho


def backproject_to_plane(u, v, camera, n, rho):
    """像素 (u,v) 反投影到门平面 → **相机系** 3D 点（§4.5 核心）。

    p = λ·K⁻¹[u,v,1]，λ = rho/(n·K⁻¹[u,v,1])；分母≈0（视线平行门平面）返回 None。
    注意：返回点已在相机系——控制直接用 p；若要画回图像做叠加调试，
    用零位姿投影（project(p, rvec=0, tvec=0)），不要再施加门位姿（否则双重变换）。"""
    _need_cv2()
    K_inv = np.linalg.inv(camera.camera_matrix())
    dir_ = np.asarray(K_inv @ np.array([u, v, 1.0], np.float64), np.float64).reshape(3)
    den = float(np.dot(n.reshape(3), dir_))
    if abs(den) < 1e-12:
        return None
    p = (rho / den) * dir_
    return p


# ---------------------------------------------------------------------------
# 调试/自检小工具
# ---------------------------------------------------------------------------
def project_gate(camera, rvec, tvec, obj3=None):
    """把门框四角（或给定 3D 点）投影到图像（叠加调试用，§7.2）。"""
    if obj3 is None:
        obj3 = object_points()
    return camera.project(obj3, rvec, tvec)
