# -*- coding: utf-8 -*-
"""tests/tasks/test_gate_vision.py — gate 视觉侧：PnP 几何往返 / keypoint 解码约定 /
非等比缩放回投 / mode 降级 / kpt_mem 开关 / GateTask(mock) 无硬件闭环。
GateTask 相位机（MockGateBackend 无硬件闭环）。"""
import numpy as np
import pytest

import base.settings as S
from common.detector import Det
from gate.gate_decode import decode_yolo11_kpt
from gate.gate_detector import board_camera
from gate.gate_frontend import (MODE_COARSE, MODE_FULL, MODE_P3P, MODE_WIDTH,
                                bbox_center, parse_kpt_mode, width_range_depth)
from gate.gate_task import (PH_ALIGN, PH_APPROACH, PH_THROUGH, SUB_GOLDEN,
                            GateTask)
from gate.geometry import (CameraModel, backproject_to_plane, gate_pose,
                           object_points, plane_from_pose, reproj_rms)
from gate.kpt_memory import (ENV_ENABLE, KptMemory, build_kpt_memory,
                             kpt_mem_enabled)
from gate.mock import MockGateBackend


def test_pnp_roundtrip_recovers_injected_pose():
    """合成针孔相机：object_points → project → gate_pose 还原注入位姿，RMS 极小。"""
    cam = CameraModel.pinhole(640, 480, fx=600.0, fy=600.0, cx=320.0, cy=240.0)
    obj3 = object_points()
    assert obj3.shape == (4, 3)
    rvec = np.array([0.02, -0.01, 0.0], np.float64)
    tvec = np.array([0.05, -0.03, 3.0], np.float64).reshape(3, 1)

    uv = cam.project(obj3, rvec, tvec)
    assert uv.shape == (4, 2)

    res = gate_pose(cam, obj3, uv)
    assert res is not None
    r_est, t_est = res
    assert abs(float(t_est.ravel()[2]) - 3.0) < 1e-2       # ≈ 注入的 tvec.z
    np.testing.assert_allclose(t_est.ravel(), tvec.ravel(), atol=1e-2)
    np.testing.assert_allclose(r_est.ravel(), rvec, atol=1e-3)
    assert reproj_rms(cam, obj3, uv, r_est, t_est) < 0.5   # 重投影 RMS 很小

    # backproject_z：已知深度反投影 → 再投影回原像素
    p = cam.backproject_z(400.0, 300.0, 2.5)
    assert abs(float(p[2]) - 2.5) < 1e-9
    np.testing.assert_allclose(
        cam.project(p, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]).ravel(),
        [400.0, 300.0], atol=1e-3)

    # 门平面 + 反投影：反投影像素落在平面上，且沿同一视线回到原像素
    n, rho = plane_from_pose(r_est, t_est)
    p_plane = backproject_to_plane(uv[0, 0], uv[0, 1], cam, n, rho)
    assert p_plane is not None
    assert abs(float(np.dot(n, p_plane)) - rho) < 1e-6
    np.testing.assert_allclose(
        cam.project(p_plane, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]).ravel(),
        uv[0], atol=1e-3)


def test_parse_kpt_mode_degrades_with_missing_keypoints():
    """keypoint 置信度 → mode 降级表：4→full / 3→p3p / 对向 2→width / 其余→coarse。"""
    kp = np.array([[10, 10], [20, 10], [20, 20], [10, 20]], np.float32)

    def conf(*v):
        return np.array(v, np.float32)

    assert parse_kpt_mode(kp, conf(0.9, 0.9, 0.9, 0.9)) == (MODE_FULL,
                                                            [0, 1, 2, 3])
    assert parse_kpt_mode(kp, conf(0.9, 0.9, 0.9, 0.1)) == (MODE_P3P,
                                                            [0, 1, 2])
    # 对向 2 角（TL,TR 或 BR,BL）→ width：可测距
    assert parse_kpt_mode(kp, conf(0.9, 0.9, 0.1, 0.1)) == (MODE_WIDTH, [0, 1])
    assert parse_kpt_mode(kp, conf(0.1, 0.1, 0.9, 0.9)) == (MODE_WIDTH, [2, 3])
    # 对角 2 点不可测宽 → coarse
    assert parse_kpt_mode(kp, conf(0.9, 0.1, 0.9, 0.1)) == (MODE_COARSE, [0, 2])
    # 全不可信 / 点数不足 → coarse
    assert parse_kpt_mode(kp, conf(0.1, 0.1, 0.1, 0.1)) == (MODE_COARSE, [])
    assert parse_kpt_mode(np.zeros((2, 2), np.float32),
                          conf(0.9, 0.9)) == (MODE_COARSE, [])
    # 阈值可调
    assert parse_kpt_mode(kp, conf(0.4, 0.4, 0.4, 0.4),
                          conf_thr=0.3)[0] == MODE_FULL

    # 已知实宽测距：Z ≈ fx·W/Δu；重合点无解
    assert width_range_depth(100.0, 300.0, 600.0, 0.7) == pytest.approx(2.1)
    assert width_range_depth(100.0, 100.0, 600.0, 0.7) is None

    # coarse 档瞄准用 = Det 中心
    det = Det("gate", 0.9, 10, 20, 30, 40)
    assert bbox_center(det) == det.center == (25.0, 40.0)


def test_kpt_memory_switch_precedence_and_smoothing(monkeypatch):
    """kpt_mem 可选开关：env > cfg.enable > False，force 最高；开启后抖动被抑制。"""
    monkeypatch.delenv(ENV_ENABLE, raising=False)

    assert kpt_mem_enabled({"enable": False}) is False
    assert build_kpt_memory({"enable": False}) is None      # 关 → None

    monkeypatch.setenv(ENV_ENABLE, "0")                     # env 压过 cfg.enable=true
    assert build_kpt_memory({"enable": True}) is None
    monkeypatch.setenv(ENV_ENABLE, "1")                     # env 也能压过 enable=false
    assert isinstance(build_kpt_memory({"enable": False}), KptMemory)
    monkeypatch.setenv(ENV_ENABLE, "0")                     # force 压过 env
    assert isinstance(build_kpt_memory({"enable": False}, force=True), KptMemory)

    # 工程真实配置：**2026-09-18 用户定 kpt_mem 真关掉**（enable=false）→ 返回 None
    monkeypatch.delenv(ENV_ENABLE, raising=False)
    assert build_kpt_memory(S.vision.gate.kpt_mem) is None, \
        "cfg/vision.yaml 里 kpt_mem.enable 应该是 false（用户现场决定）"
    # 下面要验"融合行为"本身 → 显式 force 打开（不改配置）
    real = build_kpt_memory(S.vision.gate.kpt_mem, force=True)
    assert isinstance(real, KptMemory)

    # 抖动抑制：±6px 交替抖动，融合后位置标准差明显小于原始
    base_kp = np.array([[100.0, 200.0], [300.0, 200.0],
                        [300.0, 300.0], [100.0, 300.0]], np.float32)
    jitter = np.array([[6.0, 4.0], [-6.0, -4.0]], np.float32)
    raw, fused = [], []
    now = 0
    for i in range(40):
        kp = base_kp + jitter[i % 2]
        now += 33
        out_k, _ = real.update(kp, np.full(4, 0.9, np.float32), now)
        raw.append(kp[0].copy())
        fused.append(out_k[0].copy())
    raw_std = float(np.std(np.array(raw[5:]), axis=0).mean())
    fused_std = float(np.std(np.array(fused[5:]), axis=0).mean())
    assert fused_std < 0.5 * raw_std

    # 短消失回忆：连续缺角帧不把位置拉向垃圾输入，置信度保持 ≥ recall_conf
    mem = build_kpt_memory(S.vision.gate.kpt_mem, force=True)
    seen = np.array([[100.0, 100.0], [200.0, 100.0],
                     [200.0, 200.0], [100.0, 200.0]], np.float32)
    now = 0
    for _ in range(5):
        now += 33
        mem.update(seen, np.full(4, 0.9, np.float32), now)
    out_k, out_c = mem.update(np.zeros((4, 2), np.float32),
                              np.zeros(4, np.float32), now + 33)
    assert float(out_c.min()) >= float(S.vision.gate.kpt_mem.recall_conf)
    np.testing.assert_allclose(out_k, seen, atol=1.0)


class _GateHub(object):
    """hub 包装：把 mock 后端注册成 gate 的专用后端（GateTask.ready 依赖）。"""

    def __init__(self, backend):
        self.backend = backend

    def has_extra(self, task):
        return task == "gate"

    def detect_list(self, task, frame):
        return self.backend.detect(frame)


def test_gate_task_mock_reaches_through_and_counts_pass(fake_uart):
    """GateTask 相位机：mock 进近 → ALIGN(GOLDEN) → APPROACH → THROUGH → pass。"""
    cam = board_camera()
    backend = MockGateBackend(cam)
    task = GateTask(fake_uart, _GateHub(backend), cam.width, cam.height)
    assert task.ready is True
    frame = np.zeros((cam.height, cam.width, 3), np.uint8)

    phases, substates = [], []
    status = S.STATUS_RUNNING
    for i in range(600):
        status = task.process(frame, 1000 + 33 * i)
        info = task.last_info
        if not phases or phases[-1] != info["phase"]:
            phases.append(info["phase"])
        if info["substate"]:
            substates.append(info["substate"])
        if status == S.STATUS_DONE:
            break

    assert status == S.STATUS_DONE
    assert phases == [PH_ALIGN, PH_APPROACH, PH_THROUGH]
    assert SUB_GOLDEN in substates                 # 位姿对准子状态确实走到
    assert task.last_info["reason"] == "pass"
    assert task.last_info["pass"] == 1             # 达 comm.gate.pass_target
    assert task.last_info["mode"] == MODE_FULL     # 4 角可见 → 全位姿

    surges = [f[0] for f in fake_uart.frames]
    assert S.comm.gate.surge.through in surges     # 穿门冲刺速度真的下发
    assert fake_uart.neutral_calls >= 1            # 结束时回中性


# ==============================================================
# keypoint 头解码约定（合成张量；训练侧 ONNX 与板端唯一的耦合面）
# ==============================================================
G = 80                                   # 用 stride=8 那一层做算术最直观
STRIDE = 640 / G                         # = 8.0
INPUT = 640
REG_MAX = 16
KPT_DIM = 4
LABELS = ["gate"]



def _blank_outputs(g=G, nc=1, kpt_dim=KPT_DIM, reg_max=REG_MAX):
    """一个尺度的空输出：{通道数: [1,g,g,C]}。

    cls 初值给 -10（sigmoid≈4.5e-5）而不是 0：sigmoid(0)=0.5 会全部越过 conf 阈值，
    合成用例里必须显式把背景压下去，否则解出几千个空框。
    """
    cls = np.full((1, g, g, nc), -10.0, np.float32)
    return {64: np.zeros((1, g, g, 4 * reg_max), np.float32),
            nc: cls,
            kpt_dim * 3: np.zeros((1, g, g, kpt_dim * 3), np.float32)}
def _set_dfl(out, gx, gy, dist=3):
    """把 (gx,gy) 格的 DFL 分布设成 one-hot 在 bin=dist → 解码距离恒为 dist 格。"""
    a = out[64][0, gy, gx]
    a[:] = 0.0
    for side in range(4):
        a[side * REG_MAX + dist] = 40.0          # softmax → 几乎 one-hot
def _set_score(out, gx, gy, logit=6.0):
    out[1][0, gy, gx, 0] = logit                 # sigmoid(6) ≈ 0.9975
def _set_kpt_cell(out, gx, gy, i, cell_x, cell_y, v_logit=6.0):
    """把第 i 个角点放在 **cell 坐标 (cell_x, cell_y)**。

    注意：ONNX 输出的 kpt x/y **已经是 cell 坐标**（= `raw*2 + 网格索引`，
    见 scripts/3_export/modify_ultralytics.py 的 POSE_FORWARD），
    所以这里直接写 cell 坐标；板端只做 `×stride`。
    """
    a = out[KPT_DIM * 3][0, gy, gx]
    a[3 * i + 0] = cell_x
    a[3 * i + 1] = cell_y
    a[3 * i + 2] = v_logit

def test_kpt_grid_term_is_plus_index():
    """网格项必须是 **+索引**，不是 +索引-0.5。

    把角点编码在 cell 坐标 = 网格索引本身（raw=0）处：
        正确 → 像素 = gx * stride
        多写 -0.5 → 像素 = (gx-0.5) * stride，即 stride=8 上偏 4 px
    """
    gx, gy = 10, 20
    out = _blank_outputs()
    _set_score(out, gx, gy)
    _set_dfl(out, gx, gy, dist=3)
    for i in range(4):
        _set_kpt_cell(out, gx, gy, i, cell_x=gx, cell_y=gy)

    dets = decode_yolo11_kpt(out, LABELS, INPUT, INPUT,
                             conf=0.25, iou=0.45, input_w=INPUT, input_h=INPUT)
    assert len(dets) == 1, "合成张量应当解出 1 个目标"
    k = np.asarray(dets[0].kpts)

    # +索引 约定：cell 坐标 == 网格索引 → 像素 == 索引×stride
    np.testing.assert_allclose(k[:, 0], gx * STRIDE, atol=1e-3)
    np.testing.assert_allclose(k[:, 1], gy * STRIDE, atol=1e-3)

    # 显式反证：-0.5 约定会给出不同的值，且差正好半个 cell
    np.testing.assert_allclose(k[:, 0], (gx - 0.5) * STRIDE + 0.5 * STRIDE, atol=1e-3)
    assert abs(k[0, 0] - (gx - 0.5) * STRIDE) > 3.9, "若成立说明网格项被写成了 -0.5"

def test_kpt_channel_layout_is_xyv_interleaved():
    """通道序必须是 [x,y,v]×K，且 K 个点各自独立（reshape(g*g, K, 3) 的语义）。"""
    gx, gy = 5, 6
    cells = [(gx + 0.5, gy + 0.5), (gx + 2.5, gy + 0.5),
             (gx + 2.5, gy + 2.5), (gx + 0.5, gy + 2.5)]
    out = _blank_outputs()
    _set_score(out, gx, gy)
    _set_dfl(out, gx, gy, dist=4)
    for i, (cx, cy) in enumerate(cells):
        _set_kpt_cell(out, gx, gy, i, cx, cy, v_logit=(6.0 if i < 3 else -6.0))

    d = decode_yolo11_kpt(out, LABELS, INPUT, INPUT, conf=0.25, iou=0.45,
                          input_w=INPUT, input_h=INPUT, vis_thr=0.5)[0]
    k = np.asarray(d.kpts)
    for i, (cx, cy) in enumerate(cells):
        np.testing.assert_allclose(k[i], [cx * STRIDE, cy * STRIDE], atol=1e-3)
    # v 经 sigmoid 后阈值化：第 4 点 logit=-6 → sigmoid≈0.0025 < 0.5 → conf 置 0
    c = np.asarray(d.kpt_conf)
    assert c[0] > 0.9 and c[3] == 0.0, "可见度应 sigmoid 后按 vis_thr 阈值化"

def test_box_decode_is_dfl_times_stride_around_grid_center():
    """框：dist 由 DFL 期望给出，中心 = (索引+0.5)×stride。"""
    gx, gy, dist = 12, 9, 5
    out = _blank_outputs()
    _set_score(out, gx, gy)
    _set_dfl(out, gx, gy, dist=dist)
    for i in range(4):
        _set_kpt_cell(out, gx, gy, i, cell_x=gx + 0.5, cell_y=gy + 0.5)

    d = decode_yolo11_kpt(out, LABELS, INPUT, INPUT,
                          conf=0.25, iou=0.45, input_w=INPUT, input_h=INPUT)[0]
    cx, cy = (gx + 0.5) * STRIDE, (gy + 0.5) * STRIDE
    # Det.x/y/w/h 是 int（见 common/detector.py:34），所以用 w/h 反推中心并放宽 1px
    got_cx = d.x + d.w / 2.0
    got_cy = d.y + d.h / 2.0
    assert abs(got_cx - cx) <= 1.0 and abs(got_cy - cy) <= 1.0
    assert abs(d.w - 2 * dist * STRIDE) <= 2 and abs(d.h - 2 * dist * STRIDE) <= 2

def test_scale_back_to_frame_is_squish():
    """回投必须是 squish（x/y 用不同系数），因为板端预处理是直接 resize 到 640×640。"""
    gx, gy = 10, 20
    out = _blank_outputs()
    _set_score(out, gx, gy)
    _set_dfl(out, gx, gy, dist=3)
    for i in range(4):
        _set_kpt_cell(out, gx, gy, i, cell_x=gx + 0.5, cell_y=gy + 0.5)

    d = decode_yolo11_kpt(out, LABELS, 1280, 720,
                          conf=0.25, iou=0.45, input_w=640, input_h=640)[0]
    k = np.asarray(d.kpts)
    np.testing.assert_allclose(k[:, 0], (gx + 0.5) * STRIDE * (1280 / 640), atol=1e-3)
    np.testing.assert_allclose(k[:, 1], (gy + 0.5) * STRIDE * (720 / 640), atol=1e-3)
    # 若误用等比（两个方向都乘 2）→ y 会差 720/640=1.125 倍，这里显式反证
    assert abs(k[0, 1] - (gy + 0.5) * STRIDE * 2.0) > 1.0

def test_multiscale_outputs_are_merged_and_nms_dedups():
    """3 个尺度的字典应合并，重叠目标被 NMS 压成一个。"""
    out = {}
    for g in (20, 40, 80):
        o = _blank_outputs(g=g)
        gx = gy = g // 4
        _set_score(o, gx, gy, logit=6.0)
        _set_dfl(o, gx, gy, dist=2)
        for i in range(4):
            _set_kpt_cell(o, gx, gy, i, cell_x=gx + 0.5, cell_y=gy + 0.5)
        out.update(o)

    dets = decode_yolo11_kpt(out, LABELS, INPUT, INPUT,
                             conf=0.25, iou=0.45, input_w=INPUT, input_h=INPUT)
    assert len(dets) == 1, f"多层同位置目标应被 NMS 合并，实际 {len(dets)}"
    assert dets[0].kind == "gate"
    assert dets[0].kpts.shape == (4, 2) and dets[0].kpt_conf.shape == (4,)

def test_low_score_cells_are_filtered_by_conf():
    """cls 低于 conf 的格子必须被丢掉（否则 NMS 前会爆炸）。"""
    out = _blank_outputs()
    _set_score(out, 3, 3, logit=-6.0)            # sigmoid ≈ 0.0025 < 0.25
    _set_dfl(out, 3, 3, dist=2)
    for i in range(4):
        _set_kpt_cell(out, 3, 3, i, cell_x=3.5, cell_y=3.5)
    assert decode_yolo11_kpt(out, LABELS, INPUT, INPUT, conf=0.25,
                             input_w=INPUT, input_h=INPUT) == []
