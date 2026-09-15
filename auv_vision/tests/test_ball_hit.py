# -*- coding: utf-8 -*-
"""test_ball_hit.py — 冲刺(DASH)与停稳(STOP)：**不再检测有没有撞到**

覆盖：
  1. 面积(EMA) ≥ dash_ratio 连续 dash_confirm_frames 帧 → 进 DASH，速度 = 分级最高速
  2. DASH 持续 dash_dur_s（不检测目标；冲刺中丢目标也照冲到底）
  3. DASH 结束 → STOP：全 0 保持 stop_hold_s（稳定停住）→ DONE("hit")
  4. 全程不会出现后退（没有"近距消失→后退"那套判定）
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                     # noqa: E402
import base.settings as S              # noqa: E402
from common.detector import Det        # noqa: E402
from task1_2.ball import BallTask, PH_DASH, PH_STOP        # noqa: E402

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


def det_ratio(ratio, cx=W / 2.0, cy=H / 2.0):
    side = int(round(math.sqrt(ratio * W * H)))
    return Det("red_ball", 0.9, int(cx - side / 2), int(cy - side / 2),
               side, side)


def _run_to_approach(task, hub, ratio=0.02, dt=33, n=12):
    """喂若干帧"已居中的小球"，让任务从 CENTER 进到 APPROACH。"""
    hub.t = det_ratio(ratio)
    now = 0
    for _ in range(n):
        now += dt
        task.process(FRAME, now)
    return now


def _run_until_dash(task, hub, now, ratio=None, dt=33, max_frames=60):
    """持续喂"面积已达标"的球（EMA 需要几帧收敛），直到进 DASH。"""
    ratio = S.comm.ball.dash_ratio + 0.05 if ratio is None else ratio
    hub.t = det_ratio(ratio)
    for _ in range(max_frames):
        now += dt
        task.process(FRAME, now)
        if task.last_info["phase"] == PH_DASH:
            break
    return now


# ---------------------------------------------------------------- 1) 面积触发冲刺
def test_area_triggers_dash():
    u, h = FakeUart(), FakeHub()
    t = BallTask(u, h, W, H)
    now = _run_to_approach(t, h)
    check("可进近", t.last_info["phase"] == "APPROACH", t.last_info["phase"])
    now = _run_until_dash(t, h, now)
    check("面积达标进入 DASH", t.last_info["phase"] == PH_DASH,
          (t.last_info["phase"], t.last_info["action"]))
    check("冲刺速度=分级最高速",
          abs(t.last_info["surge"] - S.comm.ball.surge_fast) < 1e-9,
          (t.last_info["surge"], S.comm.ball.surge_fast))
    check("冲刺只用前进", abs(t.last_info["sway"]) < 1e-9
          and abs(t.last_info["yaw"]) < 1e-9 and abs(t.last_info["heave"]) < 1e-9,
          t.last_info)


# ---------------------------------------------------------------- 2) 冲刺时长/盲冲
def test_dash_duration_and_blind():
    u, h = FakeUart(), FakeHub()
    t = BallTask(u, h, W, H)
    now = _run_to_approach(t, h)
    now = _run_until_dash(t, h, now)
    t0 = now
    h.t = None                              # 冲刺中丢目标：必须照冲到底
    while t.last_info["phase"] == PH_DASH and now - t0 < 5000:
        now += 33
        t.process(FRAME, now)
    dur = (now - t0) / 1000.0
    check("冲刺时长≈dash_dur_s",
          abs(dur - S.comm.ball.dash_dur_s) <= 0.2,
          "dur=%.2f 期望=%.2f" % (dur, S.comm.ball.dash_dur_s))
    check("冲刺后进 STOP", t.last_info["phase"] == PH_STOP, t.last_info["phase"])


# ---------------------------------------------------------------- 3) 停稳后 DONE
def test_stop_holds_then_done():
    u, h = FakeUart(), FakeHub()
    t = BallTask(u, h, W, H)
    now = _run_to_approach(t, h)
    now = _run_until_dash(t, h, now)
    h.t = None
    stop_t0 = None
    zeros = 0
    while t.last_info["status"] != S.STATUS_DONE and now < 20000:
        now += 33
        t.process(FRAME, now)
        if t.last_info["phase"] == PH_STOP:
            if stop_t0 is None:
                stop_t0 = now
            if len(u.dof[-1]) == 4 and all(abs(v) < 1e-9 for v in u.dof[-1]):
                zeros += 1
    check("DONE 理由 hit", t.last_info["reason"] == "hit", t.last_info)
    hold = (now - stop_t0) / 1000.0
    check("STOP 保持≈stop_hold_s",
          stop_t0 is not None and abs(hold - S.comm.ball.stop_hold_s) <= 0.2,
          "hold=%.2f 期望=%.2f" % (hold, S.comm.ball.stop_hold_s))
    check("STOP 期间持续发全 0", zeros >= 5, "zeros=%d" % zeros)
    check("命中后已回中位", ("neutral",) in u.dof, u.dof[-2:])


# ---------------------------------------------------------------- 4) 不会后退
def test_never_backward():
    u, h = FakeUart(), FakeHub()
    t = BallTask(u, h, W, H)
    now = _run_to_approach(t, h)
    now = _run_until_dash(t, h, now)        # 面积达标 → 触发冲刺
    h.t = None
    while t.last_info["status"] != S.STATUS_DONE and now < 20000:
        now += 33
        t.process(FRAME, now)
    surges = [d[0] for d in u.dof if len(d) == 4]
    check("全程无后退", all(s >= 0 for s in surges),
          [s for s in surges if s < 0][:5])
    check("结束时 DONE(hit)", t.last_info["reason"] == "hit", t.last_info)


def main():
    test_area_triggers_dash()
    test_dash_duration_and_blind()
    test_stop_holds_then_done()
    test_never_backward()
    print("\n全部通过 %d 项" % len(PASS))


if __name__ == "__main__":
    main()
