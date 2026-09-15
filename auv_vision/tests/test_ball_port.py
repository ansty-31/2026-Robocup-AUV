# -*- coding: utf-8 -*-
"""test_ball_port.py — 撞球运动逻辑"融合独立实验"后的框架性回归

覆盖（对应这次改造的验收点）：
  1. 20s 总时限生效；"其余时限"按预算缩放（不做严格 1/15）
  2. 参数面：新增 dash_ratio/stop_hold_s/approach_pid 等；旧的命中/后退/入场补丁已删
  3. APPROACH 用独立实验的 sway 环路参数（kp12/out0.45/dz0.05）
  4. 相位不越界：CENTER 不前进不横移 / APPROACH 不转不升降 / DASH 只前进 / STOP 全 0
  5. ball_fwd 全链路已删除
  6. 命中后 main 会保持停止 done_hold_ms（"稳定保持停止"）
"""
import contextlib
import io
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                     # noqa: E402
import base.settings as S              # noqa: E402
from common.detector import Det        # noqa: E402
from task1_2.ball import (BallTask, PH_SEARCH, PH_CENTER, PH_APPROACH,  # noqa: E402
                          PH_DASH, PH_STOP)

PASS = []


def check(name, cond, extra=""):
    if not cond:
        raise AssertionError("FAIL: %s %s" % (name, extra))
    PASS.append(name)
    print("PASS: %s" % name)


class FakeUart(object):
    def __init__(self):
        self.dof = []

    def send_dof(self, *a):
        self.dof.append(a)

    def neutral(self):
        self.dof.append(("neutral",))

    def set_motion(self, n):
        self.dof.append(("motion", n))

    def estop(self):
        pass


class FakeHub(object):
    def __init__(self):
        self.t = None

    def ready(self, name):
        return True

    def detect(self, name, frame):
        return self.t

    def detect_all(self, frame):
        return [self.t] if self.t is not None else []


W, H = 1280, 720
FRAME = np.zeros((H, W, 3), np.uint8)


def det_ratio(ratio, cx=W / 2.0, cy=H / 2.0):
    side = int(round(math.sqrt(max(ratio, 0.0) * W * H)))
    return Det("red_ball", 0.9, int(cx - side / 2), int(cy - side / 2),
               side, side)


# ---------------------------------------------------------------- 1) 时限
def test_time_limits():
    print("[1] 时限：总 20s + 其余按预算缩放")
    check("总时限 20s", int(S.comm.ball.timeout_ms) == 20000,
          S.comm.ball.timeout_ms)
    check("搜索前进探测触发点缩小", S.comm.ball.search_advance_after_s <= 6,
          S.comm.ball.search_advance_after_s)
    check("阶段窗口有下限(不被 1/15 压成 0.1s)",
          S.comm.ball.search_advance_dur_s >= 0.3
          and S.comm.ball.engage_hold_s >= 0.3,
          (S.comm.ball.search_advance_dur_s, S.comm.ball.engage_hold_s))
    check("冲刺 2s / 停稳保持", abs(S.comm.ball.dash_dur_s - 2.0) < 1e-9
          and S.comm.ball.stop_hold_s > 0,
          (S.comm.ball.dash_dur_s, S.comm.ball.stop_hold_s))
    print()


def test_total_timeout_fires():
    print("[2] 一直找不到球 → 20s 总时限收尾")
    u, h = FakeUart(), FakeHub()
    t = BallTask(u, h, W, H)
    h.t = None
    now = 0
    while t.last_info["status"] != S.STATUS_DONE and now < 40000:
        now += 33
        t.process(FRAME, now)
        if now > 25000:
            break
    check("理由=timeout", t.last_info["reason"] == "timeout", t.last_info)
    check("耗时≈20s", 19.0 <= now / 1000.0 <= 21.0, now / 1000.0)
    print()


# ---------------------------------------------------------------- 3) 参数面
def test_param_surface():
    print("[3] 参数面：新增项在、旧补丁已删")
    b = S.comm.ball
    for k in ("dash_ratio", "dash_confirm_frames", "dash_dur_s", "stop_hold_s",
              "approach_pid", "lost_inertia_surge", "r_slow", "growth_eps"):
        check("有 %s" % k, k in b, sorted(b.keys()))
    for k in ("search_total_stop_s", "r_hit", "r_near", "hit_confirm_frames",
              "backward_surge", "dash_surge", "entry_look_s", "edge_dx"):
        check("已删 %s" % k, k not in b, k)
    ap = b.approach_pid
    check("APPROACH 用独立实验参数",
          abs(ap.kp - 12.0) < 1e-9 and abs(ap.out_max - 0.45) < 1e-9
          and abs(ap.deadzone - 0.05) < 1e-9, dict(ap))
    check("dash_ratio = 0.25", abs(b.dash_ratio - 0.25) < 1e-9, b.dash_ratio)
    print()


# ---------------------------------------------------------------- 4) 相位不越界
def test_phase_invariants():
    print("[4] 相位不越界（喂一条「接近→达标→冲刺」的合成轨迹）")
    u, h = FakeUart(), FakeHub()
    t = BallTask(u, h, W, H)
    now = 0
    seen = set()
    ratio = 0.01
    h.t = None
    for _ in range(20):                      # 先搜索几帧
        now += 33
        t.process(FRAME, now)
    for i in range(400):
        now += 33
        if t.last_info["status"] == S.STATUS_DONE:
            break
        # 目标：小面积居中 → 面积缓慢增长到超过阈值
        if i < 20:
            ratio = 0.02
        else:
            ratio = min(0.40, ratio + 0.01)
        h.t = det_ratio(ratio, cx=W / 2.0 + 60)
        t.process(FRAME, now)
        ph, li = t.last_info["phase"], t.last_info
        seen.add(ph)
        if ph == PH_CENTER:
            check("CENTER 不前进", abs(li["surge"]) < 1e-9, li)
            check("CENTER 不横移", abs(li["sway"]) < 1e-9, li)
        elif ph == PH_APPROACH:
            check("APPROACH 不转向", abs(li["yaw"]) < 1e-9, li)
            check("APPROACH 不升降", abs(li["heave"]) < 1e-9, li)
            check("APPROACH 前进>0", li["surge"] > 0, li)
        elif ph == PH_DASH:
            check("DASH 速度=surge_fast",
                  abs(li["surge"] - S.comm.ball.surge_fast) < 1e-9, li)
            check("DASH 只前进",
                  abs(li["sway"]) < 1e-9 and abs(li["yaw"]) < 1e-9
                  and abs(li["heave"]) < 1e-9, li)
        elif ph == PH_STOP:
            check("STOP 全 0",
                  abs(li["surge"]) < 1e-9 and abs(li["sway"]) < 1e-9
                  and abs(li["yaw"]) < 1e-9 and abs(li["heave"]) < 1e-9, li)
    check("走完 居中/接近/冲刺/停稳",
          {PH_CENTER, PH_APPROACH, PH_DASH, PH_STOP} <= seen, seen)
    check("结束 DONE(hit)", t.last_info["reason"] == "hit", t.last_info)
    print()


# ---------------------------------------------------------------- 5) ball_fwd 已删
def test_ball_fwd_removed():
    print("[5] ball_fwd 全链路已删除")
    import main as main_mod
    check("TASK_CLASS 无 ball_fwd", "ball_fwd" not in main_mod.TASK_CLASS,
          sorted(main_mod.TASK_CLASS))
    check("无 STATE_BALL_FWD", not hasattr(S, "STATE_BALL_FWD"))
    check("comm 无 ball_forward 段", "ball_forward" not in S.comm,
          [k for k in S.comm.keys() if "ball" in k])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in ("task1_2/ball_forward.py", "task1_2/run_ball_forward.sh",
                "tests/test_ball_forward.py"):
        check("已删文件 %s" % rel, not os.path.exists(os.path.join(root, rel)))
    print()


# ---------------------------------------------------------------- 6) 命中后保持停止
def test_done_hold_after_hit():
    print("[6] 命中后 main 保持停止 done_hold_ms")
    import main as main_mod
    saved = {"mode": S.vision.model.mode, "front": S.vision.camera.front.type,
             "enabled": list(S.comm.tasks.enabled), "idle": S.comm.tasks.idle_ms,
             "debug": S.DEBUG, "sleep": time.sleep}
    S.vision.model.mode = "mock"
    S.vision.camera.front.type = "sim"
    S.comm.tasks.enabled = ["ball"]
    S.comm.tasks.idle_ms = 1
    S.DEBUG = False
    time.sleep = lambda s: None                # 别真睡，缩短测试
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            ctrl = main_mod.AppController(["ball"])
            ctrl.run()
        out = buf.getvalue()
        check("命中后任务为 DONE(hit)",
              ctrl.tasks["ball"].last_info["reason"] == "hit",
              ctrl.tasks["ball"].last_info)
        check("main 打印了保持停止", "保持停止" in out, out[-200:])
        check("退出前最后是 stop/neutral",
              ctrl.uart.last_motion in ("stop", "dof"),
              getattr(ctrl.uart, "last_motion", None))
        motions = [d for d in ctrl.tasks["ball"].last_info.get("_x", [])]
        _ = motions
    finally:
        time.sleep = saved["sleep"]
        S.vision.model.mode = saved["mode"]
        S.vision.camera.front.type = saved["front"]
        S.comm.tasks.enabled = saved["enabled"]
        S.comm.tasks.idle_ms = saved["idle"]
        S.DEBUG = saved["debug"]
    check("done_hold_ms 已配置且 >0", int(S.comm.tasks.done_hold_ms) > 0,
          S.comm.tasks.done_hold_ms)
    print()


def main():
    test_time_limits()
    test_total_timeout_fires()
    test_param_surface()
    test_phase_invariants()
    test_ball_fwd_removed()
    test_done_hold_after_hit()
    print("\n全部通过 %d 项" % len(PASS))


if __name__ == "__main__":
    main()
