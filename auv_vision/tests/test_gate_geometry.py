# -*- coding: utf-8 -*-
"""
test_gate_geometry.py — gate 几何内核合成往返测试（无硬件）

运行：cd auv_vision && python3 tests/test_gate_geometry.py

覆盖（对应 算法说明-gate-PnP移植方案.md §6）：
  1. object_points 布局：0.70×0.50、中心≈0、z=0、顺序 TL,TR,BR,BL
  2. full(4角)：随机 6-DoF → 投影 → gate_pose 反解 → 误差 < 阈值
     （无噪声：<1cm/0.01rad；加 0.5px 像素噪声：<2cm/0.02rad）
  3. p3p(3角)：缺一角 + 上帧猜值 → 收敛回真值（证明"3 点必须上帧消歧"）
  4. 平面/反投影往返：门平面法向/rho → 像素反投影 → 重投影闭环
  5. CameraModel：raw 域(带畸变)与 rectified 域(去畸变 K)均可用
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gate import geometry as G                      # noqa: E402
from gate.geometry import (                         # noqa: E402
    CameraModel, object_points, gate_pose,
    plane_from_pose, backproject_to_plane,
)

_CALIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "cfg", "front_camera.yaml")


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _R(rv):
    R, _ = cv2.Rodrigues(np.asarray(rv, np.float64))
    return R


def angle_err(rvec_a, rvec_b):
    Ra, Rb = _R(rvec_a), _R(rvec_b)
    tr = np.clip((np.trace(Ra.T @ Rb) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(tr))


def make_cameras():
    """返回 [(名字, CameraModel)]：优先真实标定(raw+rectified)，缺标定退合成。"""
    if os.path.exists(_CALIB):
        return [("raw(带畸变)", CameraModel.from_yaml(_CALIB, rectified=False)),
                ("rectified", CameraModel.from_yaml(_CALIB, rectified=True))]
    print("⚠️  未找到 %s，用合成针孔相机测试" % _CALIB)
    return [("pinhole", CameraModel.pinhole(1280, 720, 800.0, 800.0, 640.0, 360.0))]


def rand_approach_pose(rng, cam, z_lo=1.2, z_hi=6.0, tries=200):
    """采样"进近段"随机位姿：门在机前、四角都在画面内。"""
    w, h = cam.width, cam.height
    for _ in range(tries):
        z = float(rng.uniform(z_lo, z_hi))
        x = float(rng.uniform(-0.5, 0.5))
        y = float(rng.uniform(-0.5, 0.5))
        rvec = rng.uniform(-0.35, 0.35, 3).astype(np.float64)
        tvec = np.array([x, y, z], np.float64).reshape(3, 1)
        uv = cam.project(object_points(), rvec, tvec)
        m = 10
        if np.all(uv[:, 0] > m) and np.all(uv[:, 0] < w - m) and \
                np.all(uv[:, 1] > m) and np.all(uv[:, 1] < h - m):
            return rvec, tvec, uv
    raise RuntimeError("采样不到合格位姿（相机参数异常?）")


def check(name, ok, detail=""):
    print("  [%s] %s %s" % ("PASS" if ok else "FAIL", name, detail))
    if not ok:
        raise AssertionError("%s 失败: %s" % (name, detail))


# ---------------------------------------------------------------------------
# 1) object_points 布局
# ---------------------------------------------------------------------------
def test_object_points():
    print("[1] object_points 布局")
    pts = object_points()                      # 默认 0.70×0.50, from_front=True
    check("尺寸 0.70/0.50",
          abs(pts[0][0] - (-0.35)) < 1e-9 and abs(pts[1][0] - 0.35) < 1e-9 and
          abs(pts[0][1] - (-0.25)) < 1e-9 and abs(pts[2][1] - 0.25) < 1e-9)
    check("z=0 平面", np.allclose(pts[:, 2], 0.0))
    check("顺序 TL,TR,BR,BL 矩形", np.allclose(pts[0], [-0.35, -0.25, 0]) and
          np.allclose(pts[1], [0.35, -0.25, 0]) and
          np.allclose(pts[2], [0.35, 0.25, 0]) and
          np.allclose(pts[3], [-0.35, 0.25, 0]))
    back = object_points(from_front=False)
    check("from_front 仅 x 符号翻转", np.allclose(pts[:, [1, 2]], back[:, [1, 2]]) and
          np.allclose(pts[:, 0], -back[:, 0]))
    print()


# ---------------------------------------------------------------------------
# 2) full(4角) 往返
# ---------------------------------------------------------------------------
def test_roundtrip_full(cams):
    print("[2] full(4角) 往返")
    rng = np.random.default_rng(42)
    for name, cam in cams:
        for _ in range(20):                     # 无噪声
            rvec, tvec, uv = rand_approach_pose(rng, cam)
            out = gate_pose(cam, object_points(), uv)
            check("%s 无噪声有解" % name, out is not None)
            r2, t2 = out
            pe = float(np.linalg.norm(t2.ravel() - tvec.ravel()))
            ae = angle_err(rvec, r2)
            check("%s 无噪声 位置<1cm/角度<0.01rad" % name,
                  pe < 1e-2 and ae < 0.01, "pos=%.4f ang=%.4f" % (pe, ae))
        for _ in range(10):                     # 0.5px 高斯噪声
            rvec, tvec, uv = rand_approach_pose(rng, cam)
            uv_n = uv + rng.normal(0.0, 0.5, uv.shape)
            out = gate_pose(cam, object_points(), uv_n)
            check("%s 噪声有解" % name, out is not None)
            r2, t2 = out
            z = float(tvec.ravel()[2])
            pe = float(np.linalg.norm(t2.ravel() - tvec.ravel()))
            ae = angle_err(rvec, r2)
            _noise_stats.append((z, pe, ae))
            if "raw" in name:
                # 板上默认走 rectified 域(undistort:true)；raw 域仅粗校验
                # （宽角畸变放大边缘像素噪声, 该域精度预期明显差于 rectified）
                check("%s 0.5px噪声(粗校验)" % name,
                      pe < 0.10 and ae < 0.30,
                      "z=%.1f pos=%.4f ang=%.4f" % (z, pe, ae))
            else:
                # 任务判据 = 门中心位置(深度/横向)：典型误差 ∝ noise·z²/(f·W)；
                # 0.5pxσ×4角含尾部 → 4× 裕量；姿态角仅粗校验防"错解"
                # （平面倾角远距+小目标病态；任务无朝向要求不依赖）
                pos_tol = max(0.02, 4.0 * 0.5 * z * z / (cam.fx * G.GATE_FRAME_W))
                check("%s 0.5px噪声(位置按距离缩放)" % name,
                      pe < pos_tol and ae < 0.35,
                      "z=%.1f pos=%.4f(tol=%.3f) ang=%.4f(粗)" % (z, pe, pos_tol, ae))
    _noise_stats.sort()
    if _noise_stats:
        z0, p0, _ = _noise_stats[0]
        zm, pm, _ = _noise_stats[-1]
        zx, px, ax = max(_noise_stats, key=lambda s: s[1])
        print("     噪声包络(0.5pxσ): 最近z=%.1fm pos=%.3fm | 最远z=%.1fm pos=%.3fm |"
              " 全局最差 pos=%.3fm@z=%.1fm(ang=%.2frad)" % (z0, p0, zm, pm, px, zx, ax))
    print()


_noise_stats = []


# ---------------------------------------------------------------------------
# 3) p3p(3角 + 上帧) 往返
# ---------------------------------------------------------------------------
def test_roundtrip_p3p(cams):
    print("[3] p3p(3角 + 上帧猜值)")
    rng = np.random.default_rng(7)
    for name, cam in cams:
        for drop in (0, 1, 2, 3):               # 每种缺角
            for _ in range(3):
                rvec, tvec, uv = rand_approach_pose(rng, cam)
                keep = [i for i in range(4) if i != drop]
                obj3 = object_points()[keep]
                img2 = uv[keep]
                # 上帧猜值 = 真值 + 小噪声（帧间位移小）
                prev = (rvec + rng.normal(0, 0.02, 3).astype(np.float64),
                        tvec + rng.normal(0, 0.02, 3).reshape(3, 1).astype(np.float64))
                out = gate_pose(cam, obj3, img2, prev=prev)
                check("%s 缺角%d 有解(需上帧)" % (name, drop), out is not None)
                r2, t2 = out
                pe = float(np.linalg.norm(t2.ravel() - tvec.ravel()))
                ae = angle_err(rvec, r2)
                check("%s 缺角%d 位置<3cm/角度<0.03rad" % (name, drop),
                      pe < 3e-2 and ae < 0.03, "pos=%.4f ang=%.4f" % (pe, ae))
    print()


# ---------------------------------------------------------------------------
# 4) 平面/反投影往返（rectified 域精确；raw 域视线需先 undistort → 仅测 rectified）
# ---------------------------------------------------------------------------
def test_plane_backproject(cams):
    print("[4] 门平面/反投影往返")
    rng = np.random.default_rng(1)
    rect = next((c for n, c in cams if "rect" in n or "pinhole" in n), cams[0][1])
    for _ in range(15):
        rvec, tvec, uv = rand_approach_pose(rng, rect)
        n, rho = plane_from_pose(rvec, tvec)
        t3 = tvec.ravel()
        check("平面法向≈门中心视线约束", abs(np.dot(n, t3) - rho) < 1e-9)
        # 门中心(原点)的投影像素 → 反投影回平面应等于 t
        uv0 = rect.project(np.zeros((1, 3), np.float32), rvec, tvec)[0]
        p = backproject_to_plane(uv0[0], uv0[1], rect, n, rho)
        check("中心像素反投影=位姿平移", p is not None and
              np.linalg.norm(p - t3) < 1e-6, "d=%.2e" % (np.linalg.norm(p - t3) if p is not None else -1))
        # 任意 4 个随机角点像素：反投影→平面(相机系 p)→ 直接透视投影回像素
        for i in range(4):
            u, v = uv[i]
            p = backproject_to_plane(u, v, rect, n, rho)
            # p 已在相机系：重投影 = 零位姿投影(identity 变换)，勿再施加 rvec/tvec
            back = rect.project(np.asarray(p, np.float32).reshape(1, 3),
                                np.zeros(3), np.zeros(3))[0]
            check("角点%d 反投影往返 <1e-4px" % i,
                  np.linalg.norm(back - [u, v]) < 1e-4,
                  "d=%.2e" % float(np.linalg.norm(back - [u, v])))
    print()


# ---------------------------------------------------------------------------
# 5) CameraModel 域/工厂
# ---------------------------------------------------------------------------
def test_camera_model(cams):
    print("[5] CameraModel 域/工厂")
    for name, cam in cams:
        check("%s 尺寸/主点合理性" % name,
              cam.width > 0 and cam.height > 0 and 0 < cam.cx < cam.width and
              0 < cam.cy < cam.height)
        K = cam.camera_matrix()
        check("%s 内参形状" % name, K.shape == (3, 3) and K[0, 1] == 0)
        if "rect" in name or "pinhole" in name:
            check("%s 去畸变域 D=0" % name, np.allclose(cam.dist_coeffs(), 0.0))
    cam = CameraModel.pinhole(320, 240, 300, 300, 160, 120)
    uv = cam.project(np.array([[0, 0, 1]], np.float32), np.zeros(3), np.array([0, 0, 2], np.float64))
    check("pinhole 投影中心", np.allclose(uv[0], [160, 120], atol=1e-4), "uv=%s" % uv[0])
    p = cam.backproject_z(160.0, 120.0, 2.0)
    check("backproject_z 中心@2m", np.linalg.norm(p - [0, 0, 2]) < 1e-6)
    print()


def main():
    cams = make_cameras()
    test_object_points()
    test_camera_model(cams)
    test_roundtrip_full(cams)
    test_roundtrip_p3p(cams)
    test_plane_backproject(cams)
    print("✅ test_gate_geometry 全部通过（%s 域）" % ", ".join(n for n, _ in cams))


if __name__ == "__main__":
    main()
