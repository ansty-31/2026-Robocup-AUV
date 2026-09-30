# -*- coding: utf-8 -*-
"""tools/check/check_pipeline_identity.py — "域自证"脚本（本地/板端都能跑；不入运行时）

每次把代码/配置同步到另一台机器（PC ↔ 板端）后跑一次（服务整个项目，不限 gate）：
    python3 tools/check/check_pipeline_identity.py
    # 或: python3 tools/check/check_pipeline_identity.py --vision cfg/vision.yaml

校验（设备内相对一致性，不绑定 cv2 版本/哈希，避免 PC 与板端误报）：
  1. 声明项：undistort=true、input_size=640、标定分辨率==配置帧尺寸、纯拉伸（无 letterbox/crop）；
  2. 单一来源：getOptimalNewCameraMatrix(alpha=0, centerPrincipalPoint=False) 的结果
     必须与 gate.geometry.CameraModel(rectified=True) 的 K 一致；
  3. 域恒等式：A⁻¹(A·new_K)==K_full；detector decode 的逆缩放 == 1/resize 缩放
     （"关键点回缩域"与"PnP 域"必须是同一个域）。
退出码：0=一致；1=不一致（部署脚本可据此报警）。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np
import yaml

def _project_root():
    """工程根 = 从本文件向上找到含 `cfg/` 的那一层（本脚本在 tools/ 下）。

    别写死层级：脚本挪过位置（根 → tools/），写死会让 cfg 相对路径静默指错。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        if os.path.isdir(os.path.join(here, "cfg")):
            return here
        here = os.path.dirname(here)
    return os.path.dirname(os.path.abspath(__file__))


_ROOT = _project_root()                  # 工程根（含 cfg/）
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

PIPELINE_ID = "undistort_alpha0_resize640_pixel_centers_v1"


def resize_pixel_transform(src_wh, dst=640):
    sx, sy = float(dst) / float(src_wh[0]), float(dst) / float(src_wh[1])
    return np.array([[sx, 0, (sx - 1) / 2.0],
                     [0, sy, (sy - 1) / 2.0],
                     [0, 0, 1.0]], np.float64), (sx, sy)


def _check(name, ok, detail=""):
    print("  [%s] %s %s" % ("PASS" if ok else "FAIL", name, detail))
    return bool(ok)


def run(vision_path=None, calibration=None, tol=1e-6):
    import cv2
    from gate.geometry import CameraModel

    vision_path = vision_path or os.path.join(_ROOT, "cfg", "vision.yaml")
    print("=== gate 域自证 (pipeline_id=%s) ===" % PIPELINE_ID)
    with open(vision_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cam = (cfg.get("camera") or {}).get("front") or {}
    img = cfg.get("image") or {}
    model = cfg.get("model") or {}
    w, h = int(cam.get("width", 0)), int(cam.get("height", 0))
    input_size = int(model.get("input_size", 0))
    calibration = calibration or cam.get("calibration")
    ok = True

    ok &= _check("声明 undistort=true", img.get("undistort") is True,
                 "image.undistort=%r" % img.get("undistort"))
    ok &= _check("声明 input_size=640", input_size == 640, "input_size=%r" % input_size)
    ok &= _check("声明纯拉伸（无 letterbox/crop/alpha!=0）",
                 img.get("undistort_alpha", 0) == 0
                 and img.get("resize_mode", "stretch") == "stretch"
                 and img.get("crop", False) in (False, None)
                 and img.get("center_principal_point", False) is False,
                 "alpha=%r resize=%r crop=%r" % (img.get("undistort_alpha", 0),
                                                 img.get("resize_mode", "stretch"),
                                                 img.get("crop", False)))

    if not calibration or not os.path.exists(str(calibration)):
        cand = os.path.join(_ROOT, "cfg", os.path.basename(str(calibration)))
        if os.path.exists(cand):
            print("  [info] 标定路径 %s 不存在，回退使用 %s" % (calibration, cand))
            calibration = cand
    ok &= _check("标定文件存在", bool(calibration) and os.path.exists(str(calibration)),
                 str(calibration))
    if not ok:
        return 1

    text = open(str(calibration), encoding="utf-8-sig").read()
    fs = cv2.FileStorage(text, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
    try:
        K = fs.getNode("camera_matrix").mat()
        D = fs.getNode("distortion_coefficients").mat()
        cw = int(fs.getNode("image_width").real())
        ch = int(fs.getNode("image_height").real())
    finally:
        fs.release()
    ok &= _check("标定分辨率 == 配置帧尺寸", (cw, ch) == (w, h),
                 "calib=%dx%d cfg=%dx%d" % (cw, ch, w, h))
    if not ok:
        return 1

    new_K, _roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0, (w, h),
                                                centerPrincipalPoint=False)
    cm = CameraModel.from_yaml(str(calibration), rectified=True)
    K_full = cm.camera_matrix()
    ok &= _check("CameraModel(rectified).K == getOptimalNewCameraMatrix",
                 np.allclose(K_full, new_K, rtol=tol, atol=1e-6),
                 "max|Δ|=%.2e" % float(np.max(np.abs(K_full - new_K))))

    A, (sx, sy) = resize_pixel_transform((w, h), input_size)
    back = np.linalg.inv(A) @ (A @ new_K)
    ok &= _check("A⁻¹(A·new_K) == K_full", np.allclose(back, K_full, rtol=tol, atol=1e-6),
                 "max|Δ|=%.2e" % float(np.max(np.abs(back - K_full))))
    ok &= _check("detector decode 逆缩放 == 1/resize 缩放",
                 abs((w / input_size) - (1.0 / sx)) < 1e-9
                 and abs((h / input_size) - (1.0 / sy)) < 1e-9,
                 "decode=(%.4f,%.4f) 1/s=(%.4f,%.4f)"
                 % (w / input_size, h / input_size, 1 / sx, 1 / sy))

    print("  pipeline_id : %s" % PIPELINE_ID)
    print("  K_full      : fx=%.3f fy=%.3f cx=%.3f cy=%.3f"
          % (K_full[0, 0], K_full[1, 1], K_full[0, 2], K_full[1, 2]))
    print("  resize A    : sx=%.4f sy=%.4f" % (sx, sy))
    print("  sha256(vision) : %s"
          % hashlib.sha256(open(vision_path, "rb").read()).hexdigest()[:16])
    print("  sha256(calib)  : %s"
          % hashlib.sha256(open(str(calibration), "rb").read()).hexdigest()[:16])
    print("=== %s ===" % ("域自证通过" if ok else "域自证失败"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="gate 域自证（移植后运行，不改任何东西）")
    ap.add_argument("--vision", default=None)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args()
    sys.exit(run(a.vision, a.calibration, a.tol))


if __name__ == "__main__":
    main()
