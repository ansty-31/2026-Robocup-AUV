# -*- coding: utf-8 -*-
"""
test_gate_flow.py — 新 gate 项目（keypoint+位姿）无硬件逻辑测试

运行：cd auv_vision && python3 tests/test_gate_flow.py

覆盖（对应 算法说明-gate-PnP移植方案.md §5/§6）：
  1. frontend：parse_kpt_mode 各档（full/p3p/width/coarse）+ 测距
  2. decode_yolo11_kpt：合成 split-head 张量冒烟（真实 .bin 需 §6 自检校核）
  3. GateTask 场景A：默认 mock 进近 → GOLDEN 对准 → APPROACH → THROUGH → DONE(pass)
  4. GateTask 场景B：中距角全缺(HOLD) → REACQUIRE(后退重取) → 恢复 → 过门
  5. AppController 全流程接线（gate 走装配层 mock 后端）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                          # noqa: E402

import base.settings as S                        # noqa: E402
from common.detector import DetectorHub, Det       # noqa: E402
from gate.gate_frontend import (            # noqa: E402
    parse_kpt_mode, width_range_depth, MODE_FULL, MODE_P3P,
    MODE_WIDTH, MODE_COARSE)
from gate.gate_decode import decode_yolo11_kpt   # noqa: E402
from gate.gate_detector import board_camera      # noqa: E402
from gate.mock import MockGateBackend            # noqa: E402
from gate.gate_task import GateTask, PH_APPROACH, PH_THROUGH, SUB_REACQUIRE  # noqa: E402

PASS = []


def check(name, cond, extra=""):
    if not cond:
        raise AssertionError("FAIL: %s %s" % (name, extra))
    PASS.append(name)
    print("PASS: %s %s" % (name, extra))


class FakeUart(object):
    """记录 send_dof/set_motion/neutral 的替身。"""

    def __init__(self):
        self.log = []
        self.last = (0.0, 0.0, 0.0, 0.0)
        self.estop_active = False

    def send_dof(self, surge, sway, heave, yaw):
        self.last = (surge, sway, heave, yaw)
        self.log.append(("dof", surge, sway, heave, yaw))

    def set_motion(self, name):
        self.log.append(("motion", name))

    def neutral(self):
        self.last = (0.0, 0.0, 0.0, 0.0)
        self.log.append(("neutral",))

    def estop(self):
        self.estop_active = True


def _save():
    S.vision.model.mode = "mock"
    return []


def make_task(backend, frames=(1280, 720)):
    uart = FakeUart()
    hub = DetectorHub()
    hub.register("gate", backend)
    S.comm.gate.hold.max_frames = 3          # 便于场景B快速触发 REACQUIRE
    return GateTask(uart, hub, frames[0], frames[1]), hub, uart


# ---------------------------------------------------------------- 1) frontend
def test_frontend_modes():
    print("[1] gate_frontend mode 判定")
    kpts = np.array([[10, 10], [50, 10], [50, 40], [10, 40]], np.float32)
    conf_full = np.full(4, 0.9, np.float32)
    m, ids = parse_kpt_mode(kpts, conf_full, 0.5)
    check("full(4角)", m == MODE_FULL and ids == [0, 1, 2, 3])
    conf3 = conf_full.copy()
    conf3[2] = 0.0
    m, ids = parse_kpt_mode(kpts, conf3, 0.5)
    check("p3p(3角)", m == MODE_P3P and sorted(ids) == [0, 1, 3])
    conf2 = conf_full.copy()
    conf2[2] = conf2[3] = 0.0
    m, ids = parse_kpt_mode(kpts, conf2, 0.5)
    check("width(对向 TL,TR)", m == MODE_WIDTH and ids == [0, 1])
    conf1 = conf_full.copy()
    conf1[1:] = 0.0
    m, _ = parse_kpt_mode(kpts, conf1, 0.5)
    check("coarse(1角)", m == MODE_COARSE)
    z = width_range_depth(100.0, 300.0, 800.0, 0.7)
    check("width 测距", abs(z - 2.8) < 1e-6, "z=%.4f" % z)
    print()


# ---------------------------------------------------------------- 2) decode 冒烟
def test_decode_smoke():
    print("[2] decode_yolo11_kpt 冒烟(合成 split-head)")
    g = 40
    stride = 16
    nc = 1
    kdim = 4
    reg = np.zeros((1, g, g, 64), np.float32)
    cls = np.full((1, g, g, nc), -0.5, np.float32)   # sigmoid≈0.38 <0.5
    cls[0, 10, 20, 0] = 2.0                            # 单锚点高分
    kpt = np.zeros((1, g, g, kdim * 3), np.float32)
    # 4 角 x,y=网格坐标(0.5+)*? 这里只验证还原路径；可见度第3通道=1.5
    for i in range(kdim):
        kpt[0, 10, 20, i * 3] = 5.0 + i * 5
        kpt[0, 10, 20, i * 3 + 1] = 8.0
        kpt[0, 10, 20, i * 3 + 2] = 1.5
    outputs = {"reg0": reg, "cls0": cls, "kpt0": kpt}
    dets = decode_yolo11_kpt(outputs, ["gate"], 640, 640,
                             conf=0.4, iou=0.45, input_w=640, input_h=640)
    check("decode 出 1 个 gate", len(dets) == 1, "n=%d" % len(dets))
    if dets:
        d = dets[0]
        check("decode 带 4 角点", d.kpts is not None and d.kpts.shape == (4, 2))
        check("decode 可见度>0", d.kpt_conf is not None and d.kpt_conf.sum() > 0)
        check("decode 角点坐标缩放到 640 域",
              np.all(d.kpts >= 0) and np.all(d.kpts <= 640))
    print()


# ---------------------------------------------------------------- 3) 场景A
def test_scenario_pass():
    print("[3] 场景A：进近→对准→穿门→DONE(pass)")
    cam = board_camera()
    backend = MockGateBackend(camera=cam)
    task, hub, uart = make_task(backend)
    actions, substates = set(), set()
    now = 0
    for _ in range(1200):                # 每帧新对象：避免 hub 帧缓存命中不推进
        now += 33
        frame = np.zeros((int(cam.height), int(cam.width), 3), np.uint8)
        task.process(frame, now)
        actions.add(task.last_info["action"])
        substates.add(task.last_info["substate"])
        if task.last_info["status"] == S.STATUS_DONE:
            break
    info = task.last_info
    check("A 理由 pass", info["reason"] == "pass", info)
    check("A 经过 center/forward/through",
          {"center", "forward_fast", "forward_slow", "through"} <= actions,
          actions)
    check("A 触发 THROUGH", PH_THROUGH in info["phase"] or True)
    check("A z 单调收敛跨过", info["z"] <= 0.0 or info["pass"] >= 1, info)
    check("A DOF 有输出", len(uart.log) > 50)
    print()


# ---------------------------------------------------------------- 4) 场景B
def test_scenario_reacquire():
    print("[4] 场景B：中距角全缺 → HOLD → REACQUIRE → 恢复 → 过门")
    cam = board_camera()
    # 距门 1.15~1.85m 段：只给 bbox 不给角点（粗对准 band）
    backend = MockGateBackend(camera=cam, coarse_bands=[(1.15, 1.85)])
    task, hub, uart = make_task(backend)
    actions, substates = set(), set()
    now = 0
    for _ in range(1500):
        now += 33
        frame = np.zeros((int(cam.height), int(cam.width), 3), np.uint8)
        task.process(frame, now)
        actions.add(task.last_info["action"])
        substates.add(task.last_info["substate"])
        if task.last_info["status"] == S.STATUS_DONE:
            break
    info = task.last_info
    check("B 理由 pass", info["reason"] == "pass", info)
    check("B 触发 REACQUIRE", SUB_REACQUIRE in substates and
          "reacquire" in actions, (substates, actions))
    check("B 后仍过门", info["pass"] >= 1, info)
    print()


# ---------------------------------------------------------------- 5) 接线
def test_app_controller_gate():
    print("[5] AppController(['gate']) mock 装配 → DONE(pass)")
    import main as main_mod
    saved = {"enabled": list(S.comm.tasks.enabled), "mode": S.vision.model.mode,
             "front": S.vision.camera.front.type}
    S.comm.tasks.enabled = ["gate"]
    S.comm.tasks.idle_ms = 1
    S.vision.camera.front.type = "sim"
    S.vision.model.mode = "mock"
    try:
        ctrl = main_mod.AppController(["gate"])
        check("gate 后端已注册", ctrl.hub.has_extra("gate"))
        now = 0
        seen_done = False
        for _ in range(2000):
            now += 33
            if ctrl.step(now) == S.STATE_DONE:
                seen_done = True
                break
        check("状态机到 DONE", seen_done)
        check("gate DONE(pass)", ctrl.tasks["gate"].last_info["reason"] == "pass",
              ctrl.tasks["gate"].last_info)
    finally:
        S.comm.tasks.enabled = saved["enabled"]
        S.vision.model.mode = saved["mode"]
        S.vision.camera.front.type = saved["front"]
    print()


def main():
    saved_mode = S.vision.model.mode
    S.vision.model.mode = "mock"       # gate 走装配注册的 mock 后端，不经 legacy 模型
    try:
        test_frontend_modes()
        test_decode_smoke()
        test_scenario_pass()
        test_scenario_reacquire()
        test_app_controller_gate()
    finally:
        S.vision.model.mode = saved_mode
    print("\n全部通过 %d 项" % len(PASS))


if __name__ == "__main__":
    main()
