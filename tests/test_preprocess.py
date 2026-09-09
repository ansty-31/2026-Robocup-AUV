# -*- coding: utf-8 -*-
"""test_preprocess.py — 训练/推理同源预处理测试
运行：cd auv_vision && python3 tests/test_preprocess.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                    # noqa: E402

import base.settings as S      # noqa: E402
import common.preprocess as pp        # noqa: E402

PASS = []


def check(name, cond, extra=""):
    if not cond:
        raise AssertionError("FAIL: %s %s" % (name, extra))
    PASS.append(name)
    print("PASS: %s" % name)


def test_default_chain():
    # 全幅常数色图（验证中性链路 = 纯缩放且保色）
    color = np.array([90, 150, 60], dtype=np.uint8)   # B,G,R
    raw = np.zeros((180, 320, 3), dtype=np.uint8)
    raw[...] = color
    pre = pp.ModelPreprocessor(gains=[1.0, 1.0, 1.0], clip=0.0,
                              gamma=1.0, size=640)   # 显式中性参数，不受 YAML 实测调参影响
    out = pre.process(raw)
    check("shape_640", out.shape == (640, 640, 3), out.shape)
    # 确定性：同输入两次结果一致（训练帧/推理帧同一函数）
    check("deterministic", np.array_equal(out, pre.process(raw)))
    mean = out.mean(axis=(0, 1))
    check("color_kept",
          all(abs(float(mean[i]) - float(color[i])) <= 1.0 for i in range(3)),
          mean)
    # 缩放系数
    sx, sy = pre.scale(320, 180)
    check("scale_ratio", abs(sx - 0.5) < 1e-9 and abs(sy - 0.28125) < 1e-9,
          (sx, sy))


def test_enhance_active():
    raw = np.full((64, 64, 3), 100, dtype=np.uint8)
    raw[:, :, 2] = 200                 # 偏红图
    neutral = pp.ModelPreprocessor(gains=[1, 1, 1], clip=0.0, gamma=1.0)
    enhanced = pp.ModelPreprocessor(gains=[1.2, 1.0, 0.8], clip=0.0,
                                    gamma=1.0)
    a = neutral.process(raw)
    b = enhanced.process(raw)
    check("enhance_differs", not np.array_equal(a, b))
    # 中性链路只缩放：常数图输出仍常数（LUT gamma=1 恒等，允许取整差异）
    mean = a.mean(axis=(0, 1))
    src = np.array([100, 100, 200], dtype=np.float64)   # B,G,R 与 raw 一致
    check("neutral_constant", all(abs(float(mean[i]) - src[i]) <= 1.0
                                  for i in range(3)), mean)
    # CLAHE 分支可正常执行
    clahe = pp.ModelPreprocessor(gains=[1, 1, 1], clip=2.0, gamma=1.0)
    c = clahe.process(raw)
    check("clahe_runs", c.shape == a.shape)


def test_undistort_requires_calib():
    pre = pp.ModelPreprocessor(undistort=True, calib_path="/nonexistent.yaml")
    try:
        pre.process(np.zeros((100, 100, 3), dtype=np.uint8))
        check("undistort_missing_raises", False)
    except FileNotFoundError:
        check("undistort_missing_raises", True)


def main():
    test_default_chain()
    test_enhance_active()
    test_undistort_requires_calib()
    print("\n全部通过 %d 项" % len(PASS))


if __name__ == "__main__":
    main()