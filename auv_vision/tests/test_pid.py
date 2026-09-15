# -*- coding: utf-8 -*-
"""
test_pid.py — 撞球视觉居中 PID 测试
运行：cd auv_vision && python3 tests/test_pid.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base.settings as S  # noqa: E402
from common.PID import PID       # noqa: E402
from task1_2.ball import BallTask  # noqa: E402
from common.detector import Det             # noqa: E402
from base.uart import UartController      # noqa: E402

PASS = []


def check(name, cond, extra=""):
    if not cond:
        raise AssertionError("FAIL: %s %s" % (name, extra))
    PASS.append(name)
    print("PASS: %s" % name)


def test_pid_basics():
    p = PID(kp=1.0, ki=0.1, kd=0.0, out_min=-0.6, out_max=0.6, deadzone=0.04)
    # 正向误差 → 正输出且单调增长（含积分累积）
    t = 0
    out = [p.update(0.3, t + i * 50) for i in range(5)]
    check("pid_positive", out[-1] > 0.3 and all(b >= a for a, b in zip(out, out[1:])),
          out)
    # 死区 → 输出 0 且积分清零
    out_dead = p.update(0.01, t + 500)
    check("pid_deadzone_zero", out_dead == 0.0)
    # 限幅
    out_maxed = p.update(1.0, t + 550)
    check("pid_clamp", out_maxed <= 0.6 + 1e-9, out_maxed)


class _FakeHub(object):
    def __init__(self, cx_ratio=0.5, cy_ratio=0.5):
        self._cx = cx_ratio
        self._cy = cy_ratio
        self.w = 0

    def set_center(self, cx_ratio, cy_ratio):
        self._cx, self._cy = cx_ratio, cy_ratio

    def ready(self, task):
        return True

    def detect(self, task, frame):
        fw, fh = frame.shape[1], frame.shape[0]
        self.w = fw
        s = 60
        cx = int(fw * self._cx)
        cy = int(fh * self._cy)
        return Det("blue_ball", 0.9, cx - s // 2, cy - s // 2, s, s)

    def detect_all(self, frame):          # main._draw 走这里（真实 DetectorHub 同 API）
        return [self.detect("ball", frame)]


def test_ball_centering():
    """CENTER 用 yaw+heave（不横移不前进）；APPROACH 只用 sway（不转不升降）。"""
    saved = (S.comm.ball.align_x, S.comm.ball.align_y, S.DEBUG,
             S.comm.ball.pid.kp, S.comm.ball.pid.ki, S.comm.ball.pid.kd)
    S.DEBUG = False
    S.comm.ball.pid.kp, S.comm.ball.pid.ki, S.comm.ball.pid.kd = 0.9, 0.05, 0.15
    try:
        import numpy as np
        from task1_2.ball import PH_CENTER, PH_APPROACH
        frame = np.zeros((360, 640, 3), dtype=np.uint8)

        # (a) 球远离中心（dx=+0.5 > center_eps）→ 一直停在 CENTER，只用 yaw 右转
        uart = UartController(sim=True)
        hub = _FakeHub(cx_ratio=0.75)
        task = BallTask(uart, hub, 640, 360)
        now = 1000
        yaws, surges = [], []
        for _ in range(60):
            now += 33
            task.process(frame, now)
            assert task.last_info["phase"] == PH_CENTER, task.last_info
            yaws.append(task.last_info["yaw"])
            surges.append(task.last_info["surge"])
        check("centering_yaw_right", any(y > 0.02 for y in yaws[5:]), yaws[-3:])
        check("centering_no_sway", task.last_info["sway"] == 0.0,
              task.last_info["sway"])
        check("centering_never_advances", all(abs(s) < 1e-9 for s in surges),
              surges[-3:])
        # 球跑到左边（dx=-0.5）→ 转向反向
        hub.set_center(0.25, 0.5)
        for _ in range(30):
            now += 33
            task.process(frame, now)
        check("centering_yaw_left", task.last_info["yaw"] < -0.02,
              task.last_info["yaw"])

        # (b) 球在 eps 内（dx=+0.12）→ 居中确认后进 APPROACH：仅 sway 修正
        uart2 = UartController(sim=True)
        hub2 = _FakeHub(cx_ratio=0.56)
        task2 = BallTask(uart2, hub2, 640, 360)
        now2 = 1000
        for _ in range(S.comm.ball.center_confirm_frames + 3):
            now2 += 33
            task2.process(frame, now2)
        check("approach_phase_reached", task2.last_info["phase"] == PH_APPROACH,
              (task2.last_info["phase"], task2.last_info["action"]))
        check("approach_sway_left", task2.last_info["sway"] < -1e-6,
              task2.last_info["sway"])
        check("approach_no_yaw", abs(task2.last_info["yaw"]) < 1e-9,
              task2.last_info["yaw"])
        check("approach_no_heave", abs(task2.last_info["heave"]) < 1e-9,
              task2.last_info["heave"])
        check("approach_advances", task2.last_info["surge"] > 0,
              task2.last_info["surge"])
    finally:
        (S.comm.ball.align_x, S.comm.ball.align_y, S.DEBUG,
         S.comm.ball.pid.kp, S.comm.ball.pid.ki, S.comm.ball.pid.kd) = saved


def main():
    test_pid_basics()
    test_ball_centering()
    print("\n全部通过 %d 项" % len(PASS))


if __name__ == "__main__":
    main()