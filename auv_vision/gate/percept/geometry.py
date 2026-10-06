# -*- coding: utf-8 -*-
"""gate/percept/geometry.py — 门框位姿几何内核（纯函数，前端无关）
（详细用法、判据与实测见 doc/注释历史.md）"""
from __future__ import annotations

import os

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# 运行期真值以 `cfg/vision.yaml → vision.gate.geometry` 为准，两处必须同值
GATE_FRAME_W = 0.70
GATE_FRAME_H = 0.50

# 位姿可信深度范围（§4.4 质量检查初值）
Z_BOUNDS = (0.2, 15.0)
# 默认重投影合格阈值(px)
REPROJ_THR_PX = 20.0


def _need_cv2():
    if not HAS_CV2:
        raise RuntimeError("gate/percept/geometry.py 需要 opencv：pip install opencv-python"
                           "（板端: sudo apt install python3-opencv）")


class CameraModel(object):
    """针孔相机封装：原始域(带畸变) / 去畸变域(rectified K, D=0) 二选一。"""

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
        """由标定 yaml（FileStorage，calibrate_camera_video.py 输出格式）构造。"""
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
        """已知深度 Z 反投影像素 → 相机系 3D 点（对应 Bumblebee backproject_pixel）。"""
        _need_cv2()
        xn, yn = cv2.undistortPoints(
            np.array([[[x, y]]], dtype=np.float32),
            self.camera_matrix(), self.dist_coeffs())[0][0]
        return np.array([float(xn) * Z, float(yn) * Z, float(Z)], dtype=np.float64)


def object_points(W_m=GATE_FRAME_W, H_m=GATE_FRAME_H, from_front=True):
    """门框外轮廓四角 3D 点 (4,3)，z=0 平面（门平面），原点=矩形中心。"""
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


def _finite(r, t):
    """候选必须是有限值：IPPE 在退化配置(门正对、某些 z)会返回 nan 解。"""
    return np.all(np.isfinite(np.asarray(r, np.float64))) and \
        np.all(np.isfinite(np.asarray(t, np.float64)))


def _iter_candidates(camera, obj3, img2, prev=None):
    """生成候选 (rvec,tvec) 列表。"""
    K = camera.camera_matrix()
    D = camera.dist_coeffs()
    obj3 = np.asarray(obj3, np.float32)
    img2 = np.asarray(img2, np.float32)
    n = obj3.shape[0]
    cands = []
    if n >= 4:
        try:
            res = cv2.solvePnPGeneric(obj3, img2, K, D, flags=cv2.SOLVEPNP_IPPE)
            cands = [(r, t) for r, t in zip(res[1], res[2]) if _finite(r, t)]
        except Exception:
            cands = []
        if not cands:
            try:
                ok, r, t = cv2.solvePnP(obj3, img2, K, D,
                                        flags=cv2.SOLVEPNP_SQPNP)
                if ok and _finite(r, t):
                    cands = [(r, t)]
            except Exception:
                pass
        if not cands:                      # 最后兜底：ITERATIVE（有上帧猜值更快收敛）
            try:
                if prev is not None:
                    ok, r, t = cv2.solvePnP(obj3, img2, K, D, prev[0], prev[1],
                                            useExtrinsicGuess=True,
                                            flags=cv2.SOLVEPNP_ITERATIVE)
                else:
                    ok, r, t = cv2.solvePnP(obj3, img2, K, D,
                                            flags=cv2.SOLVEPNP_ITERATIVE)
                if ok and _finite(r, t):
                    cands = [(r, t)]
            except Exception:
                pass
    elif n == 3:
        try:                               # cv2 4.x：P3P 多解（cv2 5 此路径会 assert 失败）
            res = cv2.solvePnPGeneric(obj3, img2, K, D, flags=cv2.SOLVEPNP_P3P)
            cands = [(r, t) for r, t in zip(res[1], res[2]) if _finite(r, t)]
        except Exception:
            cands = []
        if not cands and prev is not None:
            r0, t0 = prev
            ok, r, t = cv2.solvePnP(obj3, img2, K, D,
                                    np.asarray(r0, np.float64),
                                    np.asarray(t0, np.float64),
                                    useExtrinsicGuess=True,
                                    flags=cv2.SOLVEPNP_ITERATIVE)
            if ok and _finite(r, t):
                cands = [(r, t)]
    return cands


def gate_pose(camera, obj3, img2, prev=None,
              reproj_thr=REPROJ_THR_PX, z_bounds=Z_BOUNDS, refine=True):
    """门框位姿：由 2D↔3D 角点对估计 (rvec, tvec)（§4.4）。"""
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
        if not _finite(r, t):            # nan/inf 解直接丢
            continue
        tz = float(t.ravel()[2])
        if not (lo <= tz <= hi):
            continue
        rms = reproj_rms(camera, obj3, img2, r, t)
        if not np.isfinite(rms) or rms > reproj_thr:
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


def plane_from_pose(rvec, tvec):
    """门平面方程（相机系）：n_c·p = rho。"""
    R = _rodrigues(rvec)
    n = np.asarray(R[:, 2], np.float64).reshape(3)   # 门平面法向(相机系)
    rho = float(np.dot(n, np.asarray(tvec, np.float64).reshape(3)))
    return n, rho


def gate_normal_angles_deg(rvec, tvec=None):
    """门法向 n=R·[0,0,1]（相机系）相对光轴的**航向/俯仰角（度）**。"""
    n, _rho = plane_from_pose(rvec, tvec if tvec is not None else np.zeros((3, 1)))
    n = np.asarray(n, np.float64).reshape(3)
    if n[2] < 0:                     # 镜像解 → 翻到朝向相机那一侧
        n = -n
    yaw = float(np.degrees(np.arctan2(n[0], n[2])))
    pitch = float(np.degrees(np.arctan2(n[1], n[2])))
    return yaw, pitch


def backproject_to_plane(u, v, camera, n, rho):
    """像素 (u,v) 反投影到门平面 → **相机系** 3D 点（§4.5 核心）。"""
    _need_cv2()
    K_inv = np.linalg.inv(camera.camera_matrix())
    dir_ = np.asarray(K_inv @ np.array([u, v, 1.0], np.float64), np.float64).reshape(3)
    den = float(np.dot(n.reshape(3), dir_))
    if abs(den) < 1e-12:
        return None
    p = (rho / den) * dir_
    return p


def project_gate(camera, rvec, tvec, obj3=None):
    """把门框四角（或给定 3D 点）投影到图像（叠加调试用，§7.2）。"""
    if obj3 is None:
        obj3 = object_points()
    return camera.project(obj3, rvec, tvec)
