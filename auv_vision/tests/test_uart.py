# -*- coding: utf-8 -*-
"""
test_uart.py — 串口协议 v2 测试（pty 真串口回环，无需硬件）

覆盖：11B 帧结构、DOF→轴映射（真值：LX=yaw LY=surge RX=heave RY=sway）、
附加字节安全默认、极性/钳位、心跳节流、ramp 平滑、急停。
运行：cd auv_vision && python3 tests/test_uart.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base.settings as S  # noqa: E402
import base.uart as uart                          # noqa: E402
from base.uart import (UartController, build_frame_from_dof,  # noqa: E402
                  build_neutral_frame, describe_motion, dof_to_axis_bytes)

PASS = []


def check(name, cond, extra=""):
    if not cond:
        raise AssertionError("FAIL: %s %s" % (name, extra))
    PASS.append(name)
    print("PASS: %s" % name)


def read_exact(fd, n, timeout=1.5):
    buf = b""
    t0 = time.time()
    os.set_blocking(fd, False)
    while len(buf) < n and time.time() - t0 < timeout:
        try:
            buf += os.read(fd, n - len(buf))
        except (BlockingIOError, OSError):
            pass
        time.sleep(0.01)
    os.set_blocking(fd, True)
    return buf


def expect_axis(surge=0.0, sway=0.0, heave=0.0, yaw=0.0):
    """独立参考实现：按 config 真值表推导期望的 7 轴字节。"""
    axes = [S.comm.frame.axis_mid] * 7
    for i, v in S.comm.frame.aux_axis.items():
        axes[i] = int(v)
    for name, val in (("surge", surge), ("sway", sway),
                      ("heave", heave), ("yaw", yaw)):
        ch = S.comm.dof_map.get(name)
        if ch is None:
            continue
        sign = ch.get("sign", 1)
        axes[ch["axis"]] = int(max(0, min(255, round(
            S.comm.frame.axis_mid + sign * val * S.comm.frame.axis_range))))
    return axes


def test_motion_desc():
    check("desc_stop", describe_motion(build_neutral_frame()) == "停止")
    fwd = build_frame_from_dof(1.0, 0, 0, 0)     # surge=1 → 前进
    check("desc_fwd", "前进" in describe_motion(fwd), describe_motion(fwd))
    mix = build_frame_from_dof(0.5, -0.5, 0, 0)  # 前进+左移
    s = describe_motion(mix)
    check("desc_mix", "前进" in s and "左移" in s, s)
    up = build_frame_from_dof(0, 0, 1.0, 0)
    check("desc_up", "上浮" in describe_motion(up), describe_motion(up))


def test_mapping():
    # 真值表通道：LX=yaw(轴0) LY=surge(轴1) RX=heave(轴2) RY=sway(轴3)
    ch = S.comm.dof_map
    check("map_yaw_axis0", ch["yaw"]["axis"] == 0, ch)
    check("map_surge_axis1", ch["surge"]["axis"] == 1, ch)
    check("map_heave_axis2", ch["heave"]["axis"] == 2, ch)
    check("map_sway_axis3", ch["sway"]["axis"] == 3, ch)

    axes = dof_to_axis_bytes(1.0, 1.0, 1.0, 1.0)
    check("full_deflection", axes == expect_axis(1.0, 1.0, 1.0, 1.0),
          (axes, expect_axis(1.0, 1.0, 1.0, 1.0)))
    for val in (0.5, -0.5, 1.0, -1.0):
        check("surge_%s" % val,
              dof_to_axis_bytes(val, 0, 0, 0) == expect_axis(val, 0, 0, 0))

    check("neutral_axes", dof_to_axis_bytes(0, 0, 0, 0) ==
          uart.neutral_axis_bytes())
    fr = build_neutral_frame()
    check("neutral_frame_len", len(fr) == 11 and
          fr[0] == S.comm.frame.header)
    check("neutral_motion_mid", list(fr[1:5]) == [128] * 4)
    check("neutral_aux", list(fr[5:8]) ==
          [S.comm.frame.aux_axis[i] for i in (4, 5, 6)])
    check("neutral_btns", list(fr[8:11]) == [0, 0, 0])


def test_pty_roundtrip():
    master, slave = os.openpty()
    slave_name = os.ttyname(slave)
    os.close(slave)

    saved_ramp = S.comm.ramp.speed_per_s
    S.comm.ramp.speed_per_s = 0          # 关闭平滑使帧可预期
    u = UartController(dev=slave_name, baud=115200, sim=False)
    check("serial_opened", not u.sim)
    try:
        u.neutral()
        got = read_exact(master, 11)
        check("neutral_pty", got == build_neutral_frame(), got.hex())

        # forward_fast → surge=1 → axis1(LY)=255
        u.set_motion("forward_fast", force=True)
        got = read_exact(master, 11)
        exp = expect_axis(1.0, 0, 0, 0)
        check("forward_frame", list(got[1:8]) == exp, got.hex())

        # 心跳节流
        u.set_motion("stop", force=False)
        time.sleep(S.comm.heartbeat.interval_ms / 1000.0 + 0.02)
        check("throttle_ok", u.set_motion("stop", force=False) is True)
        check("throttle_suppressed",
              u.set_motion("stop", force=False) is False)

        # 急停
        u.estop()
        check("estop_active", u.estop_active is True)
        check("estop_blocks_motion",
              u.set_motion("forward_fast") is False)
    finally:
        S.comm.ramp.speed_per_s = saved_ramp
        u.close()
        os.close(master)


def test_ramp():
    """ramp：目标满偏需多个心跳周期逐步逼近（无阶跃）。"""
    saved = S.comm.ramp.speed_per_s
    saved_debug = S.DEBUG
    S.comm.ramp.speed_per_s = 220.0
    S.DEBUG = False
    try:
        ctrl = UartController(sim=True)
        ctrl.set_motion("forward_fast")            # surge 目标 1.0
        ax = S.comm.dof_map["surge"]["axis"]
        first = ctrl._axes[ax]
        check("ramp_first_not_full", first < 200, first)
        for _ in range(16):                        # ~0.8s 收敛
            time.sleep(0.05)
            ctrl.send_motion(name="forward_fast")
        check("ramp_converged", ctrl._axes[ax] == 255, ctrl._axes[ax])
        ctrl.neutral()
        time.sleep(0.6)
        for _ in range(16):
            time.sleep(0.05)
            ctrl.send_motion(name="stop")
        check("ramp_back_mid", ctrl._axes[ax] <= 130, ctrl._axes[ax])
    finally:
        S.comm.ramp.speed_per_s = saved
        S.DEBUG = saved_debug


def main():
    test_motion_desc()
    test_mapping()
    test_pty_roundtrip()
    test_ramp()
    print("\n全部通过 %d 项" % len(PASS))


if __name__ == "__main__":
    main()