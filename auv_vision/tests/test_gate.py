# -*- coding: utf-8 -*-
"""tests/test_gate.py — gate：PnP 几何往返 / keypoint mode 降级 / kpt_mem 开关 /
GateTask 相位机（MockGateBackend 无硬件闭环）。"""
import numpy as np
import pytest

import base.settings as S
from common.detector import Det
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
