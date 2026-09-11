# -*- coding: utf-8 -*-
"""test_ball_search.py — 搜索→切入行为：
   - 搜索脉冲中出现目标 → 立刻切出(search 立即结束)
   - 目标在画面边缘 → 用 yaw 转向球(而非无效的横移)
   - 短暂看到后丢失 → 进入 hold(暂停搜索脉冲)，而不是马上继续转/停
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


def det(cx, cy=H / 2.0, side=200):
    return Det("red_ball", 0.9, int(cx - side / 2), int(cy - side / 2),
               side, side)


def test_entry_look_then_search_then_detect():
    u, h = FakeUart(), FakeHub()
    t = BallTask(u, h, W, H)
    h.t = None
    t.process(FRAME, 0)
    check("entry_look_first", t.last_info["action"] == "look",
          t.last_info["action"])
    t.process(FRAME, 300)                  # 入场窗口内：继续只识别不动作
    check("still_look", t.last_info["action"] == "look", t.last_info["action"])
    t.process(FRAME, 1000)                 # 过了入场窗口仍无球 → 搜索
    check("search_after_look", t.last_info["action"] == "search",
          t.last_info["action"])
    h.t = det(W / 2.0)                     # 出现目标
    t.process(FRAME, 1100)
    check("detect_switches_out_of_search",
          t.last_info["action"] != "search", t.last_info["action"])


def test_entry_detect_immediately_centers():
    u, h = FakeUart(), FakeHub()
    t = BallTask(u, h, W, H)
    h.t = det(W / 2.0 + 420)               # 一入场就看到边缘球
    t.process(FRAME, 0)
    check("entry_detect_center", t.last_info["action"] == "center",
          t.last_info["action"])
    check("entry_no_motion_until_detect", u.dof[-1][3] > 0, u.dof[-1])


def test_edge_ball_uses_yaw():
    u, h = FakeUart(), FakeHub()
    t = BallTask(u, h, W, H)
    h.t = det(W / 2.0 + 420)               # 球在画面右侧边缘(dx≈0.66)
    t.process(FRAME, 0)
    check("edge_label_center", t.last_info["action"] == "center",
          t.last_info["action"])
    surge, sway, heave, yaw = u.dof[-1]
    check("edge_yaw_to_ball", yaw > 0, yaw)          # dx>0 → 右转
    check("edge_no_sway", abs(sway) < 1e-9, sway)


def test_brief_detect_holds_not_search():
    u, h = FakeUart(), FakeHub()
    t = BallTask(u, h, W, H)
    h.t = det(W / 2.0)
    t.process(FRAME, 0)                    # 看到一帧 → 设置切入保持窗口
    h.t = None
    now = 100
    labels = []
    for _ in range(40):
        t.process(FRAME, now); now += 100
        labels.append(t.last_info["action"])
    check("hold_present", "hold" in labels, labels[:25])
    idx = labels.index("search") if "search" in labels else None
    # 保持窗口(engage_hold_s=2s)内不应回到 search
    check("search_not_immediate", idx is None or idx >= 19, idx)


def main():
    test_entry_look_then_search_then_detect()
    test_entry_detect_immediately_centers()
    test_edge_ball_uses_yaw()
    test_brief_detect_holds_not_search()
    print("\n全部通过 %d 项" % len(PASS))


if __name__ == "__main__":
    main()
