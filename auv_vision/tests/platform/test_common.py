# -*- coding: utf-8 -*-
"""tests/platform/test_common.py — common 层：PID / 图像链路（含 NV12 打包与 squish 缩放）/ Det 与目标挑选 /
cfgnode 配置读取（含 ball·gate 共用的 comm.motion 参数）。"""
import numpy as np
import pytest

import base.settings as S
from common.PID import PID
from common.detector import Det, pick_target
from common.cfgnode import (MOTION_DEFAULTS, flag, merge, motion_num, motion_pid,
                            num, nums, pid_kw, sub)
from common.detector import bgr_to_packed_nv12, bgr_to_packed_nv12_fast
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


# ==============================================================
# 图像链路：NV12 打包保真（模型输入的色度）
# ==============================================================
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
    """**当前配置选中的**那条 NV12 路径必须是色度忠实的。

    `vision.model.fast_nv12=false` 会切到 numpy 版，而那一版的色度被放大 4 倍
    （实测往返误差 ~15 灰阶、最大 161）。BPU 用这批 NV12 反解 RGB 喂网络，
    模型看到的颜色会整体偏 —— 不报错，只掉精度。这条断言把取舍摆到测试里。
    """
    fast = bool(S.vision.model.fast_nv12)
    packer = bgr_to_packed_nv12_fast if fast else bgr_to_packed_nv12
    img = _test_frame()
    back = _nv12_to_bgr(packer(img, W, H))
    err = float(np.abs(back.astype(np.int32) - img.astype(np.int32)).mean())
    assert err < 3.0, (
        "vision.model.fast_nv12=%s 选中的 NV12 路径色度失真（往返误差 %.2f 灰阶）。"
        "这条路径会改变模型输入的颜色分布，请改用 fast 版或先修 numpy 版的色度系数。"
        % (fast, err))



# ======================================================================
# cfgnode：配置节点读取 + **ball/gate 共用运动参数**（comm.motion）
# ======================================================================
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

    # ⚠️ 必须按字符串语义解析：bool("false") 是 True —— 板端就因此把 kpt_mem 的一个
    #    拼写错值当成了"开"，这条断言守的就是那个坑
    assert flag({"e": False}, "e", True) is False
    assert flag({"e": "false"}, "e", True) is False
    assert flag({"e": "0"}, "e", True) is False
    assert flag({"e": "off"}, "e", True) is False
    assert flag({"e": "true"}, "e", False) is True
    assert flag({"e": 0}, "e", True) is False
    assert flag({}, "e", True) is True                          # 缺键 → 默认

    # PID 参数：out_max 同时是 ±限幅；任务级覆盖优先、其余取共用默认
    kw = pid_kw({"kp": 2.0}, MOTION_DEFAULTS["pid_sway"])
    assert kw["kp"] == 2.0
    assert kw["out_min"] == -kw["out_max"] == -MOTION_DEFAULTS["pid_sway"]["out_max"]
    assert kw["deadzone"] == MOTION_DEFAULTS["pid_sway"]["deadzone"]

    # **共用运动参数**：正常取值 == cfg（一处调、两个任务同时生效）
    for k in ("surge_fast", "surge_slow", "loss_inertia_surge", "search_yaw"):
        assert motion_num(k) == pytest.approx(float(S.comm.motion[k])), k
    for k in ("pid_sway", "pid_heave"):
        assert motion_pid(k)["kp"] == pytest.approx(float(S.comm.motion[k]["kp"])), k
    # cfg 缺键 → 代码兜底（不是 KeyError）
    monkeypatch.delitem(S.comm.motion, "surge_fast", raising=False)
    assert motion_num("surge_fast") == pytest.approx(MOTION_DEFAULTS["surge_fast"])
