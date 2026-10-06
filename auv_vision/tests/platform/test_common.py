# -*- coding: utf-8 -*-
"""tests/platform/test_common.py — common 层：PID / 图像链路（含 NV12 打包与 squish 缩放）/ Det 与目标挑选 /
（详细用法、判据与实测见 doc/注释历史.md）"""
import numpy as np
import pytest

import base.cfg.settings as S
from common.motion.PID import PID
from common.vision.detector import Det, pick_target
from common.cfg.cfgnode import (MissingCfg, flag, merge, motion_num, motion_pid,
                            num, nums, pid_kw, sub)
from common.vision.detector import bgr_to_packed_nv12, bgr_to_packed_nv12_fast
from common.vision.preprocess import ModelPreprocessor, enhance


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


def test_pid_bias_lifts_output_above_actuator_deadzone():
    """★ 2026-10-07：`bias` = **执行器死区补偿**。

    推进器 `|DOF| < 0.138` 时一点推力都没有（cfg 自注）⇒ 只降 kp 会把"一直顶满"
    换成"一直没推力"。`bias` 让输出一出生就在死区之上，之后随 kp 线性升到 out_max。
    `bias=0` 时与加这个参数之前完全一致（下面是回归断言）。
    """
    DEAD_ACT = 0.138
    p = PID(kp=1.0, ki=0.0, kd=0.0, out_min=-0.55, out_max=0.55, deadzone=0.05,
            bias=DEAD_ACT)
    assert p.update(0.04) == 0.0, "死区内不补偿，必须是 0"
    assert p.update(0.051) == pytest.approx(DEAD_ACT + 0.051, abs=1e-9),         "刚出死区就该是『有推力的最小指令』"
    a = abs(p.update(0.10, now_ms=0))
    b = abs(p.update(0.20, now_ms=100))
    assert b > a > DEAD_ACT, (a, b)
    p.reset()
    assert p.update(-0.20) == pytest.approx(-(DEAD_ACT + 0.20), abs=1e-9), "负误差要对称"
    p.reset()
    assert p.update(0.90) == pytest.approx(0.55), "上限仍是 out_max"
    # bias=0 ⇒ 旧行为（回归）
    q = PID(kp=1.0, ki=0.0, kd=0.0, out_min=-0.55, out_max=0.55, deadzone=0.05)
    assert q.update(0.20) == pytest.approx(0.20)
    assert q.update(0.9) == pytest.approx(0.55)


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


def test_preprocess_is_d_chain_only():
    """识别链路**只剩 D 链**（A 域/`chain: A` 于 2026-10-01 退役归档，见 bak/retired/Adomain_20261001/）。"""
    import os
    import cv2
    from common.vision.preprocess import calibration_maps_640

    rng = np.random.default_rng(20261001)
    raw = rng.integers(0, 256, (48, 96, 3), dtype=np.uint8)      # 非方形 → 一定会走缩放

    with pytest.raises(TypeError):
        ModelPreprocessor(undistort=False, size=64, chain="D")

    # ② 无去畸变：resize → enhance
    pre = ModelPreprocessor(undistort=False, size=64)
    manual = enhance(cv2.resize(raw, (64, 64), interpolation=cv2.INTER_LINEAR),
                     pre.gains, pre.clip, pre.gamma)
    assert np.array_equal(pre.process(raw), manual), "D 链序应为 resize → enhance"

    # ③ 有去畸变（用仓库里的前视标定）：resize → enhance → remap@640（顺序不能倒）
    calib = S.vision.camera.front.calibration
    assert os.path.exists(calib), "本用例需要 %s" % calib
    pre2 = ModelPreprocessor(calib_path=calib, size=64)
    assert pre2.size == 64 and pre2.undistort
    mx, my = calibration_maps_640(calib, 64)
    manual2 = cv2.remap(manual, mx, my, cv2.INTER_LINEAR)
    assert np.array_equal(pre2.process(raw), manual2), \
        "D 链序应为 resize → enhance → remap@640（老 A 序是先 remap 再缩放/补偿）"


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


H = W = 640


def _test_frame():
    """确定性合成帧：含平滑渐变 + 饱和色块，能同时压到亮度和色度通道。"""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.zeros((H, W, 3), np.float32)
    img[..., 0] = 40 + 120 * (xx / W)
    img[..., 1] = 30 + 150 * (yy / H)
    img[..., 2] = 200 - 100 * (xx / W)
    img[80:240, 80:240] = (30, 40, 220)      # 红块（BGR）
    img[400:560, 380:560] = (230, 200, 40)   # 青块
    return np.clip(img, 0, 255).astype(np.uint8)
def _nv12_to_bgr(packed):
    import cv2
    n = W * H
    y = packed[:n].reshape(H, W)
    uv = packed[n:]
    u = uv[0::2].reshape(H // 2, W // 2)
    v = uv[1::2].reshape(H // 2, W // 2)
    i420 = np.concatenate([y.reshape(-1), u.reshape(-1), v.reshape(-1)])
    return cv2.cvtColor(i420.reshape(H * 3 // 2, W), cv2.COLOR_YUV2BGR_I420)

def test_selected_nv12_path_is_color_faithful():
    """**当前配置选中的**那条 NV12 路径必须是色度忠实的。"""
    fast = bool(S.vision.model.fast_nv12)
    packer = bgr_to_packed_nv12_fast if fast else bgr_to_packed_nv12
    img = _test_frame()
    back = _nv12_to_bgr(packer(img, W, H))
    err = float(np.abs(back.astype(np.int32) - img.astype(np.int32)).mean())
    assert err < 3.0, (
        "vision.model.fast_nv12=%s 选中的 NV12 路径色度失真（往返误差 %.2f 灰阶）。"
        "这条路径会改变模型输入的颜色分布，请改用 fast 版或先修 numpy 版的色度系数。"
        % (fast, err))



def test_cfgnode_node_readers(monkeypatch):
    """缺键/类型不对一律兜底，绝不抛异常；flag 按字符串语义解析。"""
    assert sub({"a": 1}, "a") == {}                     # 子节点不是 dict → {}
    assert sub({"a": {"b": 2}}, "a") == {"b": 2}
    assert sub({}, "a") == {}
    assert merge({"kp": 3.0}, {"kp": 1.0, "kd": 0.5}) == {"kp": 3.0, "kd": 0.5}
    assert merge({"kp": None}, {"kp": 1.0}) == {"kp": 1.0}      # None 也算缺键
    assert merge(None, {"kp": 1.0}) == {"kp": 1.0}
    assert num({"x": "2.5"}, "x", 9.0) == 2.5
    assert num({"x": "abc"}, "x", 9.0) == 9.0
    assert num({}, "x", 9.0) == 9.0                            # 缺键 → 默认值
    assert nums({"kp": 4.0}, {"kp": 1.0, "ki": 0.0}) == {"kp": 4.0, "ki": 0.0}

    assert flag({"e": False}, "e", True) is False
    assert flag({"e": "false"}, "e", True) is False
    assert flag({"e": "0"}, "e", True) is False
    assert flag({"e": "off"}, "e", True) is False
    assert flag({"e": "true"}, "e", False) is True
    assert flag({"e": 0}, "e", True) is False
    assert flag({}, "e", True) is True                          # 缺键 → 默认

    # PID 参数：out_max 同时是 ±限幅；★ 2026-10-07 起**五个键必填**（无兜底）
    kw = pid_kw({"kp": 2.0, "ki": 0.0, "kd": 0.0, "out_max": 0.5, "deadzone": 0.05})
    assert kw["kp"] == 2.0
    assert kw["out_min"] == -kw["out_max"] == -0.5
    assert kw["deadzone"] == 0.05
    with pytest.raises(MissingCfg):                     # 缺键 ⇒ 报名字，不再兜底
        pid_kw({"kp": 2.0})

    # **共用运动参数**：正常取值 == cfg（一处调、两个任务同时生效）
    for k in ("surge_fast", "surge_slow", "loss_inertia_surge", "search_yaw"):
        assert motion_num(k) == pytest.approx(float(S.comm.motion[k])), k
    for k in ("pid_sway", "pid_heave"):
        assert motion_pid(k)["kp"] == pytest.approx(float(S.comm.motion[k]["kp"])), k
    # ★ 2026-10-07 用户定：**删掉兜底** —— cfg 缺键 ⇒ 报名字（不是静默用默认值）
    monkeypatch.delitem(S.comm.motion, "surge_fast", raising=False)
    with pytest.raises(MissingCfg):
        motion_num("surge_fast")
