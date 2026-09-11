# -*- coding: utf-8 -*-
"""
test_logic.py — 无硬件全流程逻辑测试（任务一撞球；记忆返回见 test_return_by_memory）

运行：cd auv_vision && python3 tests/test_logic.py

覆盖：
  1. ball：搜索 → 远/中/近 分级调速 → 越球判定 hit → DONE
  2. 主状态机：IDLE → BALL → DONE（顺序迁移）
  3. 任务超时兜底；E-STOP 状态锁
  （返回出发区 = 记忆返回：tests/test_return_by_memory.py；gate：tests/test_gate_*.py）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base.settings as S                # noqa: E402
from main import AppController           # noqa: E402

PASS = []


def check(name, cond, extra=""):
    if not cond:
        raise AssertionError("FAIL: %s %s" % (name, extra))
    PASS.append(name)
    print("PASS: %s" % name)


# ---- 配置快照/恢复（测试间不互相污染） -----------------------------------
_PATHS = ("S.DEBUG", "S.SIM_MODE", "S.vision.camera.front.type",
          "S.vision.model.mode", "S.comm.tasks.enabled",
          "S.comm.tasks.idle_ms", "S.comm.ball.timeout_ms",
          "S.comm.tasks.fault_action")
_SAVED = {}


def _getp(expr):
    parts = expr[2:].split(".")
    if parts[0] in ("vision", "comm"):
        node = getattr(S, parts[0])
        for k in parts[1:]:
            node = node[k]
        return node
    return getattr(S, parts[0])


def _setp(expr, val):
    parts = expr[2:].split(".")
    if parts[0] in ("vision", "comm"):
        node = getattr(S, parts[0])
        for k in parts[1:-1]:
            node = node[k]
        node[parts[-1]] = val
        return
    setattr(S, parts[0], val)


def cfg_save():
    for e in _PATHS:
        _SAVED[e] = _getp(e)


def cfg_restore():
    for e, v in _SAVED.items():
        _setp(e, v)


def make_ctrl(tasks):
    S.SIM_MODE = True
    S.vision.camera.front.type = "sim"
    S.vision.model.mode = "mock"
    S.DEBUG = False
    S.comm.tasks.idle_ms = 1
    S.comm.tasks.enabled = list(tasks)
    S.comm.ball.timeout_ms = 300000
    return AppController(tasks)


# ---- 1) 撞球全阶段 --------------------------------------------------------
def test_ball_full_stages():
    ctrl = make_ctrl(["ball"])
    actions = set()
    peak = 0.0
    now = 0
    for _ in range(4000):
        now += 33
        ctrl.step(now)
        info = ctrl.tasks["ball"].last_info
        actions.add(info["action"])
        peak = max(peak, info["ratio"])
        if ctrl.state == S.STATE_DONE:
            break
    info = ctrl.tasks["ball"].last_info
    if info["reason"] != "hit":
        print("ball info:", info)
    check("ball_reason_hit", info["reason"] == "hit")
    check("ball_actions", {"forward_fast", "forward_slow", "stop"} <= actions,
          actions)
    check("ball_entry_state", bool(actions & {"look", "search"}), actions)
    # 命中由“最后一次冲刺”确认：DONE 时球已被撞飞，故看过程峰值而非终值
    check("ball_peak_ratio", peak > 0.5, "peak=%.3f" % peak)


# ---- 2) 状态机：单任务顺序 -------------------------------------------------
def test_state_machine_single():
    ctrl = make_ctrl(["ball"])
    seen, now = [], 0
    for _ in range(4000):
        now += 33
        before = ctrl.state
        ctrl.step(now)
        if ctrl.state != before:
            seen.append(ctrl.state)
        if ctrl.state == S.STATE_DONE:
            break
    check("machine_order", seen[:2] == [S.STATE_BALL, S.STATE_DONE], seen)
    check("ball_reason_hit", ctrl.tasks["ball"].last_info["reason"] == "hit",
          ctrl.tasks["ball"].last_info)


# ---- 3) 超时 / E-STOP ------------------------------------------------------
def test_ball_timeout():
    ctrl = make_ctrl(["ball"])
    S.comm.ball.timeout_ms = 600
    now = 0
    for _ in range(500):
        now += 33
        ctrl.step(now)
        if ctrl.state == S.STATE_DONE:
            break
    check("ball_timeout", ctrl.tasks["ball"].last_info["reason"] == "timeout",
          ctrl.tasks["ball"].last_info)


def test_estop():
    ctrl = make_ctrl(["ball"])
    ctrl.step(1000)
    ctrl.uart.estop()
    check("estop_state", ctrl.step(1033) == S.STATE_ESTOP)
    check("estop_locked", ctrl.step(1066) == S.STATE_ESTOP)
    check("estop_flag", ctrl.uart.estop_active)


def main():
    cfg_save()
    try:
        test_ball_full_stages()
        test_state_machine_single()
        test_ball_timeout()
        test_estop()
    finally:
        cfg_restore()
    print("\n全部通过 %d 项" % len(PASS))


if __name__ == "__main__":
    main()
