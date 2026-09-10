# -*- coding: utf-8 -*-
"""test_return_handover.py — 返回策略B（交下位机接管握手）无硬件测试"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                      # noqa: E402
from base.uart import UartController, build_frame_with_header   # noqa: E402
from common.detector import Det         # noqa: E402
from task1_2 import return_handover as RH   # noqa: E402

PASS = []


def check(name, cond, extra=""):
    if not cond:
        raise AssertionError("FAIL: %s %s" % (name, extra))
    PASS.append(name)
    print("PASS: %s" % name)


def test_frame_header():
    f = build_frame_with_header(0xAA, surge=1.0, sway=0.0, heave=0.0, yaw=0.0)
    check("frame_len_11", len(f) == 11, len(f))
    check("frame_header_AA", f[0] == 0xAA, hex(f[0]))
    check("frame_surge_axis1", f[2] > 128, f[2])   # 帧=[头,axis0(yaw),axis1(surge),...]


def test_rx_flag():
    u = UartController(sim=True)
    u._rxbuf.extend(b"\x11\x22\xaa\x55\xff")
    check("rx_has_hit", u.rx_has(bytes.fromhex("aa55")) is True)
    check("rx_consumed", u.rx_has(bytes.fromhex("aa55")) is False)


def test_wait_flag():
    u = UartController(sim=True)
    ticks = {"n": 0}

    def tick():
        ticks["n"] += 1

    def inject():
        time.sleep(0.15)
        u._rxbuf.extend(b"\xde\xad\xaa\x55\xff")

    threading.Thread(target=inject, daemon=True).start()
    got = u.wait_flag(bytes.fromhex("aa55"), timeout_ms=2000, tick=tick,
                      tick_interval_ms=50)
    check("wait_flag_true", got is True)
    check("wait_flag_ticked", ticks["n"] >= 1, ticks["n"])

    u2 = UartController(sim=True)
    check("wait_flag_timeout",
          u2.wait_flag(bytes.fromhex("aa55"), timeout_ms=100) is False)


class FakeUart(object):
    def __init__(self):
        self.dof = []

    def send_dof(self, *a):
        self.dof.append(a)

    def send_dof_header(self, *a, **k):
        self.dof.append(("hdr",) + a)


class FakeCam(object):
    width, height, fps = 640, 360, 30

    def read(self):
        return np.zeros((360, 640, 3), np.uint8)


class FakeHub(object):
    def __init__(self, seq):
        self.seq = list(seq)

    def detect(self, task, frame):
        if not self.seq:
            return None
        return self.seq.pop(0)


def _det_at(cx, cy):
    # 构造一个球心在 (cx,cy) 的 Det（w=h=20）
    return Det("red_ball", 0.9, int(cx - 10), int(cy - 10), 20, 20)


def test_center_loop_yaw():
    u = FakeUart()
    # 目标在右侧 → yaw 应为正(右转)；随后居中
    hub = FakeHub([_det_at(560, 180), _det_at(330, 180),
                   _det_at(320, 180), _det_at(320, 180), _det_at(320, 180)])
    ok = RH.center_loop(u, hub, FakeCam(), "yaw", "dx", 1.0, 0.8, 0.5,
                        0.10, 3, 5, "t")
    check("yaw_center_ok", ok is True)
    yaws = [d[3] for d in u.dof if len(d) == 4]
    check("yaw_positive_for_right_target", any(y > 0 for y in yaws), yaws[:3])
    check("yaw_zero_when_centered", yaws[-1] == 0.0, yaws[-1])


def test_center_loop_sway():
    u = FakeUart()
    hub = FakeHub([_det_at(560, 180), _det_at(330, 180),
                   _det_at(320, 180), _det_at(320, 180), _det_at(320, 180)])
    ok = RH.center_loop(u, hub, FakeCam(), "sway", "dx", -1.0, 0.9, 0.6,
                        0.10, 3, 5, "t")
    check("sway_center_ok", ok is True)
    sw = [d[1] for d in u.dof if len(d) == 4]
    check("sway_negative_for_right_target", any(s < 0 for s in sw), sw[:3])


def test_hex_to_bytes():
    check("hex_ok", RH.hex_to_bytes("aa55ff") == b"\xaa\x55\xff")
    check("hex_empty", RH.hex_to_bytes("") == b"")
    check("hex_odd", RH.hex_to_bytes("abc") == b"\x0a\xbc")


def main():
    test_frame_header()
    test_rx_flag()
    test_wait_flag()
    test_center_loop_yaw()
    test_center_loop_sway()
    test_hex_to_bytes()
    print("\n全部通过 %d 项" % len(PASS))


if __name__ == "__main__":
    main()
