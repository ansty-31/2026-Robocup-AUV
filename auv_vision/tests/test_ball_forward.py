# -*- coding: utf-8 -*-
"""test_ball_forward.py — 撞球简化版（无命中识别）：搜索→居中→恒速前进, 累计前进10s停"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                       # noqa: E402
import base.settings as S                # noqa: E402
from common.detector import Det          # noqa: E402
from task1_2.ball_forward import BallForwardTask   # noqa: E402

PASS = []


def check(name, cond, extra=""):
    if not cond:
        raise AssertionError("FAIL: %s %s" % (name, extra))
    PASS.append(name)
    print("PASS: %s" % name)


class FakeUart(object):
    def __init__(self):
        self.dof = []
        self.neutral_n = 0

    def send_dof(self, *a):
        self.dof.append(a)

    def neutral(self):
        self.neutral_n += 1

    def set_motion(self, n):
        pass

    def estop(self):
        pass


class FakeHub(object):
    def __init__(self):
        self.t = None

    def ready(self, name):
        return True

    def detect(self, name, frame):
        return self.t


W, H = 1280, 720
FRAME = np.zeros((H, W, 3), np.uint8)
FWD = S.comm.ball_forward.forward_surge
TOTAL_MS = S.comm.ball_forward.forward_total_s * 1000


def det(side, cx=W / 2.0, cy=H / 2.0):
    return Det("red_ball", 0.9, int(cx - side / 2), int(cy - side / 2),
               side, side)


def test_search_when_no_target():
    u, h = FakeUart(), FakeHub()
    t = BallForwardTask(u, h, W, H)
    h.t = None
    t.process(FRAME, 0)
    check("no_target_search", t.last_info["action"] == "search",
          t.last_info["action"])
    check("no_target_not_counted", t._fwd_ms == 0, t._fwd_ms)


def test_center_first_then_forward():
    u, h = FakeUart(), FakeHub()
    t = BallForwardTask(u, h, W, H)
    h.t = det(200, cx=W / 2.0 + 300)          # 明显偏右 → 先居中
    now = 0
    t.process(FRAME, now); now += 100
    check("center_label", t.last_info["action"] == "center",
          t.last_info["action"])
    check("center_no_surge", t.last_info["surge"] == 0.0, t.last_info["surge"])
    check("center_not_counted", t._fwd_ms == 0, t._fwd_ms)
    # 移到正中并保持 → 居中确认后开始恒速前进
    h.t = det(200)
    seen_forward = False
    for _ in range(S.comm.ball_forward.center_confirm_frames + 2):
        t.process(FRAME, now); now += 100
        if t.last_info["action"] == "forward":
            seen_forward = True
            check("forward_constant_speed",
                  abs(t.last_info["surge"] - FWD) < 1e-9, t.last_info["surge"])
            break
    check("forward_started", seen_forward)


def test_forward_total_stops_task():
    u, h = FakeUart(), FakeHub()
    t = BallForwardTask(u, h, W, H)
    h.t = det(200)                             # 一直在正中
    now = 0
    done = False
    max_surge = 0.0
    for _ in range(400):
        t.process(FRAME, now); now += 100
        if t.last_info["action"] == "forward":
            max_surge = max(max_surge, abs(t.last_info["surge"]))
        if t.last_info["status"] == S.STATUS_DONE:
            done = True
            break
    check("stop_reason_forward_time",
          done and t.last_info["reason"] == "forward_time",
          (t.last_info["status"], t.last_info["reason"]))
    check("accumulated_total", t._fwd_ms >= TOTAL_MS, t._fwd_ms)
    check("speed_kept_constant", abs(max_surge - FWD) < 1e-9, max_surge)
    check("neutral_on_stop", u.neutral_n >= 1, u.neutral_n)


def test_total_stop_caps_search():
    """没球时不能无限搜索：总时限到点 → DONE(total_stop)。"""
    u, h = FakeUart(), FakeHub()
    t = BallForwardTask(u, h, W, H)
    h.t = None                                 # 一直无目标 → 只搜索
    saved = S.comm.ball_forward.total_stop_s
    S.comm.ball_forward.total_stop_s = 1.0     # 临时把总时限压到 1s
    try:
        now = 0
        done = False
        for _ in range(40):
            t.process(FRAME, now); now += 100
            if t.last_info["status"] == S.STATUS_DONE:
                done = True
                break
    finally:
        S.comm.ball_forward.total_stop_s = saved
    check("total_stop_reason", done and t.last_info["reason"] == "total_stop",
          (t.last_info["status"], t.last_info["reason"]))
    check("total_stop_no_forward", t._fwd_ms == 0, t._fwd_ms)
    check("total_stop_neutral", u.neutral_n >= 1, u.neutral_n)


def main():
    test_search_when_no_target()
    test_center_first_then_forward()
    test_forward_total_stops_task()
    test_total_stop_caps_search()
    print("\n全部通过 %d 项" % len(PASS))


if __name__ == "__main__":
    main()
