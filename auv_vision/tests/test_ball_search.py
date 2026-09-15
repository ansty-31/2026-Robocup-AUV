# -*- coding: utf-8 -*-
"""test_ball_search.py — 搜索 / 居中 / 丢目标 行为（融合版运动链）

覆盖：
  1. 无目标 → 搜索脉冲（+ 周期性前进探测）；出现目标 → 立刻进 CENTER
  2. CENTER：只用 yaw(转向) + heave(升降)，**surge 恒 0、sway 恒 0**
  3. 边缘球 → yaw 转向球（dx>0 → 右转），不做横移
  4. 连续居中达标 → APPROACH：**仅 sway** 修水平（yaw=0、heave=0），surge>0
  5. 丢目标阶梯：短时 hold（不前进）→ hold 窗口 → 回 SEARCH
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                     # noqa: E402
import base.settings as S              # noqa: E402
from common.detector import Det        # noqa: E402
from task1_2.ball import BallTask, PH_CENTER, PH_APPROACH   # noqa: E402

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

    def detect_all(self, frame):          # main._draw 走这里（真实 DetectorHub 同 API）
        return [self.t] if self.t is not None else []


W, H = 1280, 720
FRAME = np.zeros((H, W, 3), np.uint8)


def det(cx, cy=H / 2.0, side=200):
    return Det("red_ball", 0.9, int(cx - side / 2), int(cy - side / 2),
               side, side)


def det_ratio(ratio, cx=W / 2.0, cy=H / 2.0):
    side = int(round(math.sqrt(ratio * W * H)))
    return Det("red_ball", 0.9, int(cx - side / 2), int(cy - side / 2),
               side, side)


# ---------------------------------------------------------------- 1) 搜索→居中
def test_search_then_detect_centers():
    u, h = FakeUart(), FakeHub()
    t = BallTask(u, h, W, H)
    h.t = None
    t.process(FRAME, 0)
    check("no_target_search", t.last_info["action"] in ("search", "search_advance"),
          t.last_info["action"])
    check("search_phase", t.last_info["phase"] == "SEARCH", t.last_info["phase"])
    h.t = det(W / 2.0 + 420)               # 边缘球出现
    t.process(FRAME, 100)
    check("detect_enters_center", t.last_info["action"] == "center",
          t.last_info["action"])
    check("center_phase", t.last_info["phase"] == PH_CENTER, t.last_info["phase"])


# ---------------------------------------------------------------- 2) 居中不前进
def test_center_never_advances():
    u, h = FakeUart(), FakeHub()
    t = BallTask(u, h, W, H)
    h.t = det(W / 2.0 + 420)               # 偏右很多，需持续转向
    now = 0
    for _ in range(30):
        now += 33
        t.process(FRAME, now)
        assert t.last_info["phase"] == PH_CENTER, t.last_info
        check_surge = t.last_info["surge"]
        if check_surge != 0.0:
            raise AssertionError("FAIL: center 阶段出现了前进 %s" % check_surge)
    check("center_never_advances", True)
    check("center_no_sway", all(abs(d[1]) < 1e-9
                                for d in u.dof if len(d) == 4), u.dof[-1])
    check("center_uses_yaw", t.last_info["yaw"] > 0, t.last_info["yaw"])
    check("center_uses_heave_channel", abs(t.last_info["heave"]) >= 0.0)


# ---------------------------------------------------------------- 3) 边缘球用 yaw
def test_edge_ball_uses_yaw():
    u, h = FakeUart(), FakeHub()
    t = BallTask(u, h, W, H)
    h.t = det(W / 2.0 + 420)               # dx≈0.66
    t.process(FRAME, 0)
    surge, sway, heave, yaw = u.dof[-1]
    check("edge_yaw_to_ball", yaw > 0, yaw)          # dx>0 → 右转
    check("edge_no_sway", abs(sway) < 1e-9, sway)
    check("edge_no_surge", abs(surge) < 1e-9, surge)


# ---------------------------------------------------------------- 4) 进近仅 sway
def test_center_done_then_approach_sway_only():
    u, h = FakeUart(), FakeHub()
    t = BallTask(u, h, W, H)
    # 球略微偏右但已在 center_eps 内 → 连续 center_confirm_frames 帧后进 APPROACH
    h.t = det_ratio(0.02, cx=W / 2.0 + 80)
    now = 0
    for _ in range(S.comm.ball.center_confirm_frames + 2):
        now += 33
        t.process(FRAME, now)
    check("approach_phase", t.last_info["phase"] == PH_APPROACH,
          (t.last_info["phase"], t.last_info["action"]))
    check("approach_forwards", t.last_info["surge"] > 0, t.last_info["surge"])
    check("approach_uses_sway", t.last_info["sway"] < -1e-6, t.last_info["sway"])
    check("approach_no_yaw", abs(t.last_info["yaw"]) < 1e-9, t.last_info["yaw"])
    check("approach_no_heave", abs(t.last_info["heave"]) < 1e-9,
          t.last_info["heave"])


# ---------------------------------------------------------------- 5) 丢目标阶梯
def test_lost_ladder_hold_then_search():
    u, h = FakeUart(), FakeHub()
    t = BallTask(u, h, W, H)
    h.t = det(W / 2.0 + 420)
    t.process(FRAME, 0)                    # 看到一帧 → CENTER
    dof_before = len(u.dof)
    h.t = None
    now = 300
    labels = []
    for _ in range(40):
        t.process(FRAME, now)
        now += 100
        labels.append(t.last_info["action"])
        if t.last_info["action"] == "hold":
            check("hold_no_forward", u.dof[-1][0] == 0.0, u.dof[-1])
    check("hold_present", "hold" in labels, labels[:20])
    check("lost_before_search", "search" in labels, labels[-6:])
    check("lost_did_not_advance_in_center",
          all(d[0] <= 1e-9 for d in u.dof[dof_before:] if len(d) == 4),
          u.dof[dof_before:dof_before + 3])
    idx = labels.index("search") if "search" in labels else None
    check("search_after_hold_window", idx is not None and idx > 0, idx)


def main():
    test_search_then_detect_centers()
    test_center_never_advances()
    test_edge_ball_uses_yaw()
    test_center_done_then_approach_sway_only()
    test_lost_ladder_hold_then_search()
    print("\n全部通过 %d 项" % len(PASS))


if __name__ == "__main__":
    main()
