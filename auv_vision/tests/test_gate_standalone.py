# -*- coding: utf-8 -*-
"""test_gate_standalone.py — “单独下水测 gate 任务”的接线回归测试

运行：cd auv_vision && python3 tests/test_gate_standalone.py（或 pytest tests/）

覆盖用户诉求：`python3 main.py --task gate` 必须
  1. 用 gate 专用权重（hub.detect_list("gate")），而不是 ball 单权重模型；
  2. **不加载 ball 权重**（DetectorHub 惰性加载）→ 门权重缺失/ball 权重缺失都能单独跑 gate；
  3. 后端不可用时**拒绝启动**（退出码 3），而不是入水后空跑 skip；
  4. 画框用任务本帧已有的检测（含门角点），不额外跑一遍模型；
  5. gate 后端输入名从模型元数据解析（防 `Input name "input" is invalid` 复发）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                          # noqa: E402

import base.settings as S                        # noqa: E402
from common.detector import DetectorHub, Det      # noqa: E402
from gate.gate_decode import find_model_input     # noqa: E402

PASS = []


def check(name, cond, extra=""):
    if not cond:
        raise AssertionError("FAIL: %s %s" % (name, extra))
    PASS.append(name)
    print("PASS: %s %s" % (name, extra))


class _Recorder(object):
    """假装单权重检测器：只记录“是否被构造过”。"""

    built = 0

    def __init__(self):
        _Recorder.built += 1

    def detect(self, frame):
        return [Det("blue_ball", 0.9, 10, 10, 20, 20)]


class _FakeGateBackend(object):
    """一个总能返回 4 角点门的替身后端（不依赖相机/权重）。"""

    def detect(self, frame):
        h, w = frame.shape[:2]
        k = np.array([[w * 0.3, h * 0.3], [w * 0.7, h * 0.3],
                      [w * 0.7, h * 0.7], [w * 0.3, h * 0.7]], np.float32)
        return [Det("gate", 0.9, w * 0.3, h * 0.3, w * 0.4, h * 0.4,
                    kpts=k, kpt_conf=np.full(4, 0.9, np.float32))]


class _Model(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)


# ---------------------------------------------------------------- 1) 惰性加载
def test_hub_lazy_real():
    print("[1] DetectorHub 惰性加载单权重模型")
    saved_mode, saved_real = S.vision.model.mode, DetectorHub._REAL
    _Recorder.built = 0
    S.vision.model.mode = "hbm_runtime"
    DetectorHub._REAL = {"hbm_runtime": _Recorder}
    try:
        hub = DetectorHub()
        check("构造 hub 不加载 ball 权重", _Recorder.built == 0,
              "built=%d" % _Recorder.built)
        check("未加载时 detect_list(gate) 不触发", not hub.has_extra("gate"))
        frame = np.zeros((64, 64, 3), np.uint8)
        hub.detect_all(frame)
        check("真正用到时才加载", _Recorder.built == 1, "built=%d" % _Recorder.built)
        hub.detect_all(np.zeros((64, 64, 3), np.uint8))
        check("只加载一次", _Recorder.built == 1, "built=%d" % _Recorder.built)
    finally:
        S.vision.model.mode, DetectorHub._REAL = saved_mode, saved_real
    print()


# ---------------------------------------------------------------- 2) gate-only 不加载 ball
def test_gate_only_never_loads_ball():
    print("[2] AppController(['gate']) 全程不加载 ball 权重，且画框不额外推理")
    import main as main_mod
    saved = {"mode": S.vision.model.mode, "front": S.vision.camera.front.type,
             "idle": S.comm.tasks.idle_ms, "real": DetectorHub._REAL,
             "build": main_mod.build_gate_backend, "debug": S.DEBUG}
    S.DEBUG = False                              # 本机无 GUI，别刷屏
    _Recorder.built = 0
    S.vision.model.mode = "hbm_runtime"          # 真机档：单权重模型会走 _REAL
    S.vision.camera.front.type = "sim"
    S.comm.tasks.idle_ms = 1
    DetectorHub._REAL = {"hbm_runtime": _Recorder}
    main_mod.build_gate_backend = lambda: _FakeGateBackend()
    try:
        # 板端 cv2 是 Qt 构建：无显示时 imshow/namedWindow 会直接 abort 进程（不是
        # 异常）→ 测试里把 GUI 函数全换成空实现，保证任何机器上都能跑完
        import common.detector as _det_mod
        import cv2 as _cv2
        gui = {}
        for _fn in ("imshow", "waitKey", "namedWindow", "destroyAllWindows"):
            gui[_fn] = getattr(_cv2, _fn, None)
            setattr(_cv2, _fn, lambda *a, **k: None)
        try:
            ctrl = main_mod.AppController(["gate"])
            check("gate 后端已注册", ctrl.hub.has_extra("gate"))
            check("构造后仍未加载 ball", _Recorder.built == 0,
                  "built=%d" % _Recorder.built)
            ctrl._video_on = True                # 强制走 _draw（画框路径）
            now = 0
            for _ in range(30):
                now += 33
                ctrl.step(now)
            check("跑 30 帧(含画框)后仍不加载 ball", _Recorder.built == 0,
                  "built=%d" % _Recorder.built)
            check("画框用的是任务本帧检测", len(ctrl.tasks["gate"].last_dets) == 1
                  and ctrl.tasks["gate"].last_dets[0].kpts is not None)
            ctrl.close()
        finally:
            for _fn, _orig in gui.items():
                setattr(_cv2, _fn, _orig)
    finally:
        S.vision.model.mode = saved["mode"]
        S.vision.camera.front.type = saved["front"]
        S.comm.tasks.idle_ms = saved["idle"]
        DetectorHub._REAL = saved["real"]
        main_mod.build_gate_backend = saved["build"]
        S.DEBUG = saved["debug"]
    print()


# ---------------------------------------------------------------- 3) 拒绝空跑
def test_gate_refuses_to_start_without_backend():
    print("[3] gate 后端不可用 → 退出码 3（拒绝入水空跑）")
    import main as main_mod
    saved = {"front": S.vision.camera.front.type,
             "build": main_mod.build_gate_backend,
             "argv": list(sys.argv)}
    S.vision.camera.front.type = "sim"
    main_mod.build_gate_backend = lambda: None     # 模拟权重缺失
    try:
        sys.argv = ["main.py", "--task", "gate"]
        try:
            main_mod.main()
            check("应当 SystemExit", False)
        except SystemExit as e:
            check("退出码 3", int(e.code or 0) == 3, "code=%s" % e.code)
    finally:
        S.vision.camera.front.type = saved["front"]
        main_mod.build_gate_backend = saved["build"]
        sys.argv = saved["argv"]
    print()


# ---------------------------------------------------------------- 4) 权重自检
def test_gate_weight_exists():
    print("[4] cfg 里 gate 权重路径已配置且存在（板端）/ 结构完整（PC）")
    cfg = S.get("vision.model.task_models.gate", None) or {}
    check("已配置 task_models.gate", bool(cfg), cfg)
    check("有 path/kind/labels/kpt_order",
          all(k in cfg for k in ("path", "kind", "labels", "kpt_order")), cfg)
    check("kind=keypoint", cfg.get("kind") == "keypoint", cfg.get("kind"))
    check("kpt_order 4 角", list(cfg.get("kpt_order", [])) == ["TL", "TR", "BR", "BL"],
          cfg.get("kpt_order"))
    on_board = os.path.exists(cfg.get("path", ""))
    print("     path=%s （本机%s）" % (cfg.get("path"), "存在" if on_board else "不存在，板端再验"))
    print()


# ---------------------------------------------------------------- 5) 输入名解析
def test_find_model_input():
    print("[5] 输入名从模型元数据解析（防硬编码复发）")
    saved = S.vision.model.input_name
    S.vision.model.input_name = "images"
    check("dict 命中 cfg", find_model_input(
        _Model(input_names={"gate_kpt_bayese_640x640_nv12": ["images"]})) == "images")
    check("dict 未命中 → 用模型自身", find_model_input(
        _Model(input_names={"m": ["data"]})) == "data")
    check("list", find_model_input(_Model(input_names=["foo"])) == "foo")
    check("str", find_model_input(_Model(input_name="bar")) == "bar")
    check("无元数据 → 回落 cfg", find_model_input(_Model()) == "images")
    S.vision.model.input_name = saved
    print()


# ---------------------------------------------------------------- 6) 编排脚本
def test_run_gate_script():
    print("[6] run_gate.sh 存在且只跑 gate 任务")
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "task1_2", "run_gate.sh")
    check("脚本存在", os.path.exists(p), p)
    check("可执行", os.access(p, os.X_OK), p)
    txt = open(p, encoding="utf-8").read()
    check("调用 main.py --task gate", 'main.py --task gate' in txt)
    check("有残留进程清理", "pkill -f" in txt)
    check("有下水前权重自检", "权重自检" in txt)
    print()


# ---------------------------------------------------------------- 7) 穿门保护
def test_z_jump_guard():
    print("[7] 穿门保护：z 跳变弃帧 + 连续帧确认（防错解满速冲出去）")
    from gate.gate_task import GateTask, PH_THROUGH
    from gate.mock import MockGateBackend
    from gate.gate_detector import board_camera

    cam = board_camera()

    class U(object):
        estop_active = False

        def send_dof(self, *a):
            pass

        def set_motion(self, n):
            pass

        def neutral(self):
            pass

    def make(pose_fn):
        hub = DetectorHub()
        hub.register("gate", MockGateBackend(camera=cam, pose_fn=pose_fn))
        return GateTask(U(), hub, int(cam.width), int(cam.height))

    # (a) 远/近交替的“错解”序列：必须一次都不触发 THROUGH
    def alt(f):
        z = 5.0 if f % 2 else 0.5
        return np.array([0.02, -0.01, 0.0]), np.array([0.0, 0.0, z]).reshape(3, 1)

    task = make(alt)
    now = 0
    for _ in range(80):
        now += 33
        task.process(np.zeros((int(cam.height), int(cam.width), 3), np.uint8), now)
    check("交替错解不穿门", task._pass_cnt == 0 and task.phase != PH_THROUGH,
          "pass=%d phase=%s" % (task._pass_cnt, task.phase))

    # (b) 正常平滑进近：仍然能穿门（保护没有挡住真过门）
    task2 = make(None)                    # None → mock 默认平滑轨迹
    now = 0
    for _ in range(400):
        now += 33
        task2.process(np.zeros((int(cam.height), int(cam.width), 3), np.uint8), now)
        if task2.last_info["status"] == S.STATUS_DONE:
            break
    check("平滑进近仍过门", task2.last_info["reason"] == "pass",
          task2.last_info["reason"])
    check("穿门确认帧可配", int(S.comm.gate.z.get("cross_confirm_frames", 2)) >= 2,
          S.comm.gate.z.get("cross_confirm_frames"))
    print()


# ---------------------------------------------------------------- 8) 运动策略
class _ScriptedGate(object):
    """按脚本逐帧返回 (z, x偏移) 或 None；用真实相机模型投影出 4 角点。"""

    def __init__(self, camera, script):
        self.camera = camera
        self.script = list(script)
        self._f = 0

    def detect(self, frame):
        i = self._f
        self._f += 1
        item = self.script[i] if i < len(self.script) else self.script[-1]
        if item is None:
            return []
        z, x = item
        from gate.geometry import object_points
        obj3 = object_points()
        tvec = np.array([x, 0.0, z], np.float64).reshape(3, 1)
        uv = self.camera.project(obj3, np.zeros(3), tvec)
        return [Det("gate", 0.9, int(uv[:, 0].min()), int(uv[:, 1].min()),
                    int(uv[:, 0].max() - uv[:, 0].min()),
                    int(uv[:, 1].max() - uv[:, 1].min()),
                    kpts=uv, kpt_conf=np.full(4, 0.92, np.float32))]


def _scripted_task(camera, script):
    from gate.gate_task import GateTask

    class U(object):
        estop_active = False

        def send_dof(self, *a):
            pass

        def set_motion(self, n):
            pass

        def neutral(self):
            pass

    hub = DetectorHub()
    hub.register("gate", _ScriptedGate(camera, script))
    return GateTask(U(), hub, int(camera.width), int(camera.height))


def test_gate_motion_strategy():
    print("[8] 运动策略(对齐撞球)：两档限速 + 冲刺前/冲刺中微调 + 丢失轻微后退")
    from gate.gate_detector import board_camera
    cam = board_camera()
    G = S.comm.gate

    def nf():
        # 每帧新对象：hub 按帧身份缓存，复用同一对象会让后端只被调用一次
        return np.zeros((int(cam.height), int(cam.width), 3), np.uint8)

    # 速度阶梯：0 < creep ≤ slow < fast（避免太快）
    check("速度阶梯 creep≤slow<fast",
          0 < G.surge.creep <= G.surge.slow < G.surge.fast,
          dict(G.surge))
    check("远处丢失后退 ≤ 慢速", 0 < G.surge.lost_backward <= G.surge.slow,
          G.surge.lost_backward)
    # 横向 PID 带阻尼（对齐撞球 kp1.0/kd0.15/死区0.04）
    check("横向 PID 带阻尼 kd>0", G.pid.kd > 0, dict(G.pid))

    # (a) 远处(z=2.5)短暂丢失 → 轻微后退
    t = _scripted_task(cam, [(2.5, 0.0)] * 12 + [None] * 4)
    now = 0
    for _ in range(12):
        now += 100
        t.process(nf(), now)
    check("(a) 已进近", t.phase == "APPROACH", t.phase)
    now += 100
    t.process(nf(), now)                       # 丢失第一帧
    check("(a) 远处丢失→backward_slow", t.last_info["action"] == "backward_slow",
          t.last_info["action"])
    check("(a) 后退量为负", t.last_info["surge"] < 0, t.last_info["surge"])

    # (b) 近距(z=0.9≤near_lost_m)丢失 → 立刻判过门，不在门口后退
    t2 = _scripted_task(cam, [(0.9, 0.0)] * 12 + [None] * 10)
    now = 0
    for _ in range(12):
        now += 100
        t2.process(nf(), now)
    check("(b) 已进近", t2.phase == "APPROACH", t2.phase)
    now += 100
    t2.process(nf(), now)
    check("(b) 近距丢失→through", t2.last_info["action"] == "through",
          t2.last_info["action"])
    check("(b) 冲刺速度=through", abs(t2.last_info["surge"] - G.surge.through) < 1e-6,
          t2.last_info["surge"])

    # (c) 冲刺前微调 / 穿门直行：z≤cross 第一帧=只微调不前进；第二帧(确认够)=只前进不微调
    t3 = _scripted_task(cam, [(0.9, 0.0)] * 12 + [(0.65, 0.15)] * 2)
    now = 0
    for _ in range(12):
        now += 100
        t3.process(nf(), now)
    check("(c) 已进近", t3.phase == "APPROACH", t3.phase)
    now += 100
    t3.process(nf(), now)                      # 冲刺前最后一帧（偏 0.15m）
    check("(c) 冲刺前: 不前进只微调",
          t3.last_info["action"] == "center" and abs(t3.last_info["surge"]) < 1e-9
          and abs(t3.last_info["sway"]) > 0,
          (t3.last_info["action"], t3.last_info["surge"], t3.last_info["sway"]))
    now += 100
    t3.process(nf(), now)                      # 确认够 → 穿门
    check("(c) 进入 THROUGH", t3.phase == "THROUGH", t3.phase)
    check("(c) 穿门只前进不微调",
          abs(t3.last_info["surge"] - G.surge.through) < 1e-6
          and abs(t3.last_info["sway"]) < 1e-9 and abs(t3.last_info["heave"]) < 1e-9,
          (t3.last_info["surge"], t3.last_info["sway"], t3.last_info["heave"]))
    # 穿门途中门再次出现（且偏置）→ 仍然直行，不受影响
    t4 = _scripted_task(cam, [(0.9, 0.0)] * 12 + [None] + [(0.9, 0.45)] * 2)
    now = 0
    for _ in range(13):
        now += 100
        t4.process(nf(), now)
    check("(d) 近距丢失已进 THROUGH", t4.phase == "THROUGH", t4.phase)
    now += 100
    t4.process(nf(), now)
    check("(d) 穿门途中门可见也不微调",
          abs(t4.last_info["sway"]) < 1e-9 and abs(t4.last_info["heave"]) < 1e-9,
          (t4.last_info["sway"], t4.last_info["heave"]))
    print()


def main():
    test_hub_lazy_real()
    test_gate_only_never_loads_ball()
    test_gate_refuses_to_start_without_backend()
    test_gate_weight_exists()
    test_find_model_input()
    test_run_gate_script()
    test_z_jump_guard()
    test_gate_motion_strategy()
    print("\n全部通过 %d 项" % len(PASS))


if __name__ == "__main__":
    main()

