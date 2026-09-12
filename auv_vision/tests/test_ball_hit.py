# -*- coding: utf-8 -*-
"""test_ball_hit.py — 撞球命中判定新逻辑：
   - 近距消失 → 后退(不再算命中)
   - 连续近距(stop)达阈值 → 最后一次冲刺(居中+前进) → 冲刺完成才算 hit
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                     # noqa: E402
import base.settings as S              # noqa: E402
from common.detector import Det        # noqa: E402
from task1_2.ball import BallTask      # noqa: E402

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


def det_ratio(ratio, cx=W / 2.0, cy=H / 2.0):
    side = int(round(math.sqrt(ratio * W * H)))
    return Det("red_ball", 0.9, int(cx - side / 2), int(cy - side / 2),
               side, side)


def test_near_loss_backward():
    u, h = FakeUart(), FakeHub()
    t = BallTask(u, h, W, H)
    # 0.46：< r_hit(0.50) 不算命中；丢球瞬间 ema≈0.7*0.46=0.32 ≥ r_near(0.30) → 应后退
    h.t = det_ratio(0.46)
    now = 0
    for _ in range(10):
        t.process(FRAME, now); now += 100
    h.t = None                               # 近距消失
    t.process(FRAME, now)
    check("near_loss_action_backward", t.last_info["action"] == "backward",
          t.last_info["action"])
    check("near_loss_surge_negative", t.last_info["surge"] < 0,
          t.last_info["surge"])
    check("near_loss_not_done", t.last_info["status"] != S.STATUS_DONE)


def test_stop_then_dash_then_hit():
    u, h = FakeUart(), FakeHub()
    t = BallTask(u, h, W, H)
    h.t = det_ratio(0.90)                    # >= r_hit(0.75)
    now = 0
    # 先居中所需帧
    for _ in range(S.comm.ball.center_confirm_frames + 1):
        t.process(FRAME, now); now += 100
    check("first_stop_not_done", t.last_info["status"] != S.STATUS_DONE,
          (t.last_info["action"], t.last_info["status"]))
    # 注：hit_confirm_frames=1 时，贴近第一帧即触发 dash（没有单独的 stop 帧）
    # 继续喂近距帧 → 达 hit_confirm_frames 应触发 dash
    dash_seen = False
    done = False
    for _ in range(40):
        t.process(FRAME, now); now += 100
        if t.last_info["action"] == "dash":
            dash_seen = True
            if t.last_info["surge"] <= 0:
                raise AssertionError("dash surge 应>0: %s" % t.last_info)
        if t.last_info["status"] == S.STATUS_DONE:
            done = True
            break
    check("dash_triggered", dash_seen)
    check("dash_then_hit_done", done and t.last_info["reason"] == "hit",
          (t.last_info["status"], t.last_info["reason"]))


def test_hit_cnt_reset_when_far():
    u, h = FakeUart(), FakeHub()
    t = BallTask(u, h, W, H)
    h.t = det_ratio(0.90)
    now = 0
    for _ in range(S.comm.ball.center_confirm_frames + 1):
        t.process(FRAME, now); now += 100
    check("hit_cnt_started", t._hit_cnt >= 1, t._hit_cnt)
    h.t = det_ratio(0.40)                    # 变远(<r_hit)：EMA 回落后应清零
    for _ in range(12):
        t.process(FRAME, now); now += 100
        if t._hit_cnt == 0:
            break
    check("hit_cnt_reset", t._hit_cnt == 0, t._hit_cnt)


def main():
    test_near_loss_backward()
    test_stop_then_dash_then_hit()
    test_hit_cnt_reset_when_far()
    print("\n全部通过 %d 项" % len(PASS))


if __name__ == "__main__":
    main()
