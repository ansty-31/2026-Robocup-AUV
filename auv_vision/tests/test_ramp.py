# -*- coding: utf-8 -*-
"""test_ramp.py — 线性斜坡（common/ramp.py）与串口 DOF 斜坡集成

要点：速度必须**线性逼近**目标，不能一步激增；rate<=0 或 dt<=0 时直通。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base.settings as S                  # noqa: E402
from common.ramp import LinearRamp, DofRamp   # noqa: E402
from base.uart import UartController       # noqa: E402

PASS = []


def check(name, cond, extra=""):
    if not cond:
        raise AssertionError("FAIL: %s %s" % (name, extra))
    PASS.append(name)
    print("PASS: %s" % name)


def test_linear_ramp():
    r = LinearRamp(rate=1.0)
    # 0 起步，每 0.1s 走 0.1（线性）
    vals = [r.update(1.0, 0.1) for _ in range(5)]
    check("ramp_linear", all(abs(v - (i + 1) * 0.1) < 1e-9
                             for i, v in enumerate(vals)), vals)
    check("ramp_not_jump", vals[0] < 0.2, vals[0])
    for _ in range(5):
        r.update(1.0, 0.1)
    check("ramp_reach_target", abs(r.value - 1.0) < 1e-9, r.value)
    # 不过冲
    for _ in range(5):
        r.update(1.0, 0.1)
    check("ramp_no_overshoot", r.value == 1.0, r.value)
    # 反向也线性
    v = r.update(-1.0, 0.25)
    check("ramp_reverse", abs(v - 0.75) < 1e-9, v)
    # rate<=0 / dt<=0 → 直通
    r0 = LinearRamp(rate=0.0)
    check("ramp_disabled_passthrough", r0.update(0.6, 0.1) == 0.6)
    r1 = LinearRamp(rate=1.0)
    check("ramp_dt0_passthrough", r1.update(0.6, 0.0) == 0.6)
    # 限幅
    rc = LinearRamp(rate=1.0, value=0.0, lo=-0.5, hi=0.5)
    check("ramp_clamp", rc.update(1.0, 10.0) == 0.5, rc.value)


def test_dof_ramp():
    d = DofRamp(rate=2.0)
    out = d.update((1.0, -1.0, 0.0, 0.5), 0.1)     # 每步 0.2
    check("dof_step", abs(out[0] - 0.2) < 1e-9 and abs(out[1] + 0.2) < 1e-9
          and out[2] == 0.0 and abs(out[3] - 0.2) < 1e-9, out)
    d.reset((0.1, 0.2, 0.3, 0.4))
    check("dof_reset", d.values == (0.1, 0.2, 0.3, 0.4), d.values)
    d.update((0.0, 0.0, 0.0, 0.0), 1.0)            # rate=2 → 一步走 2，直接到位
    check("dof_reach", d.values == (0.0, 0.0, 0.0, 0.0), d.values)


def test_uart_dof_ramp():
    """串口层：ramp.dof_per_s>0 时，轴字节线性上升（不是一步到满）。"""
    saved = (S.get("comm.ramp.dof_per_s", 0.0), S.comm.ramp.speed_per_s)
    try:
        S.comm.ramp.speed_per_s = 0                # 只看 DOF 斜坡
        S.comm.ramp.dof_per_s = 1.0
        u = UartController(sim=True)
        ax = S.comm.dof_map["surge"]["axis"]
        mid = S.comm.frame.axis_mid
        rng = S.comm.frame.axis_range
        u._dof_target = (1.0, 0.0, 0.0, 0.0)
        # 首次 0.1s → surge≈0.1 → 轴≈mid+0.1*rng
        u._last_update_ms -= 100
        u._ramp_step()
        v1 = u._axes[ax]
        check("uart_ramp_first_partial", mid + 0.05 * rng < v1 < mid + 0.2 * rng,
              v1)
        # 再给 1s → 到位
        u._last_update_ms -= 1000
        u._ramp_step()
        check("uart_ramp_reach_full", u._axes[ax] == mid + rng, u._axes[ax])
        # 关闭 DOF 斜坡 → 一步到满
        S.comm.ramp.dof_per_s = 0
        u2 = UartController(sim=True)
        check("uart_no_dof_ramp", u2._dof_ramp is None)
        u2._dof_target = (1.0, 0.0, 0.0, 0.0)
        u2._ramp_step()
        check("uart_jump_when_disabled", u2._axes[ax] == mid + rng, u2._axes[ax])
        # neutral() 复位斜坡（停车不等斜坡）
        S.comm.ramp.dof_per_s = 1.0
        u3 = UartController(sim=True)
        u3._dof_target = (1.0, 0.0, 0.0, 0.0)
        u3._last_update_ms -= 1000
        u3._ramp_step()
        u3.neutral()
        check("neutral_resets_ramp", u3._dof_ramp.values == (0.0,) * 4,
              u3._dof_ramp.values)
    finally:
        S.comm.ramp.dof_per_s, S.comm.ramp.speed_per_s = saved


def main():
    test_linear_ramp()
    test_dof_ramp()
    test_uart_dof_ramp()
    print("\n全部通过 %d 项" % len(PASS))


if __name__ == "__main__":
    main()
