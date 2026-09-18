# -*- coding: utf-8 -*-
"""tests/test_common.py — common 层：PID / 图像预处理 / Det 与目标挑选。"""
import numpy as np

import base.settings as S
from common.PID import PID
from common.detector import Det, pick_target
from common.preprocess import ModelPreprocessor, enhance


def test_pid_deadzone_clamp_and_derivative():
    """PID：死区清零、输出限幅、D 反应、积分抗 windup、reset。"""
    # 死区：|error| <= deadzone → 输出 0，并把积分清零
    p = PID(kp=1.0, ki=1.0, kd=0.0, out_min=-1.0, out_max=1.0, deadzone=0.05)
    assert p.update(0.04) == 0.0
    assert p.last_out == 0.0
    assert p.update(0.2) > 0.2          # now_ms=None → dt=0.05，积分已累积
    assert p._integral > 0.0
    assert p.update(0.03) == 0.0
    assert p._integral == 0.0           # 死区内积分清零（防抱死）

    # 限幅：比例项远超量程 → 夹在 out_min/out_max
    p2 = PID(kp=10.0, ki=0.0, kd=0.0, out_min=-1.0, out_max=1.0, deadzone=0.0)
    assert p2.update(0.5) == 1.0
    assert p2.update(-0.5) == -1.0

    # D 项：误差由 0 → 0.2、dt=0.05s → 微分 4.0
    p3 = PID(kp=0.0, ki=0.0, kd=1.0, out_min=-10.0, out_max=10.0, deadzone=0.0)
    assert p3.update(0.0, now_ms=0) == 0.0
    assert p3.update(0.2, now_ms=50) == 4.0

    # 积分抗 windup：恒定误差下输出饱和但不越界
    p4 = PID(kp=0.0, ki=1.0, kd=0.0, out_min=-1.0, out_max=1.0, deadzone=0.0)
    outs = [p4.update(1.0) for _ in range(40)]
    assert max(outs) == 1.0
    assert min(outs) >= 0.0
    p4.reset()
    assert p4.last_out == 0.0
    assert p4._integral == 0.0


def test_preprocess_enhance_shape_dtype_and_gamma():
    """预处理：形状/dtype 保持、gamma<1 提亮、任意尺寸 → 模型方形输入。"""
    rng = np.random.default_rng(20260917)
    frame = rng.integers(20, 200, (32, 40, 3), dtype=np.uint8)

    # 无 CLAHE、gamma<1 → 提亮；形状与 dtype 不变
    bright = enhance(frame, [1.0, 1.0, 1.0], 0.0, 0.5)
    assert bright.shape == frame.shape
    assert bright.dtype == np.uint8
    assert float(bright.mean()) > float(frame.mean())

    # 白平衡增益 + CLAHE 同样保持形状/dtype
    comp = enhance(frame, [1.0, 1.05, 1.15], 0.5, 0.85)
    assert comp.shape == frame.shape
    assert comp.dtype == np.uint8

    # ModelPreprocessor：任意输入尺寸 → (size, size, 3)，并可换算原图坐标
    pre = ModelPreprocessor(undistort=False, size=64)
    out = pre.process(np.zeros((48, 96, 3), np.uint8))
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8
    assert pre.scale(96, 48) == (96 / 64.0, 48 / 64.0)
    # 默认 size 来自配置
    assert ModelPreprocessor(undistort=False).size == S.vision.model.input_size


def test_det_helpers_and_pick_target():
    """Det 的 area/ratio/center/kpts 与按类别挑选（绝不因别类分高而换目标）。"""
    d = Det("blue_ball", 0.7, 10, 20, 30, 40)
    assert d.area == 1200.0
    assert d.ratio(300, 400) == 0.01
    assert d.center == (25.0, 40.0)
    assert d.kpts is None and d.kpt_conf is None       # bbox 任务向后兼容

    kp = Det("gate", 0.9, 0, 0, 10, 10,
             kpts=[[1, 2], [3, 4], [5, 6], [7, 8]], kpt_conf=[1, 1, 1, 0.2])
    assert kp.kpts.shape == (4, 2)
    assert kp.kpts.dtype == np.float32
    assert np.allclose(kp.kpt_conf, [1.0, 1.0, 1.0, 0.2])

    dets = [Det("red_ball", 0.95, 0, 0, 10, 10),
            Det("blue_ball", 0.60, 0, 0, 10, 10),
            Det("blue_ball", 0.80, 0, 0, 10, 10)]
    best = pick_target(dets, "blue_ball")
    assert best is not None
    assert best.kind == "blue_ball"
    assert best.score == 0.80                          # 同类取最高分，不选红球
    assert pick_target(dets, "gate") is None           # 目标缺席 → None（继续搜索）
    assert pick_target([], "red_ball") is None
