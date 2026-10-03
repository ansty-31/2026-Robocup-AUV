#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""相对角度旋转：带编号重发目标角，由下位机执行，等待15字节遥测完成标志。
上位机只发"相对角 + 编号"、等完成反馈；**运动参数全在下位机**（上位机不再有 PID/限幅/死区）。
`yaw_sign()`/`wrap180()` 保留：诊断工具（check_dof_sign 等）判读遥测 yaw 还要用。"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

# 工程根 = common/<类>/x.py 往上三级（本文件在子目录里，可直接当脚本跑）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from common.cfg.cfgnode import nums                                     # noqa: E402
from base.log.turn_log import turn_log                                  # noqa: E402 转弯调用日志

_FW_RIGHT_YAW_SIGN = -1.0

def wrap180(deg):
    """把角度差归一化到 (-180, 180]。"""
    x = math.fmod(deg + 180.0, 360.0)
    if x <= 0:
        x += 360.0
    return x - 180.0


def yaw_sign():
    """σ：`psi = σ × 遥测 yaw`（**定死的乘式**，不是测量值）。返回 (σ, 说明串)。

    σ = `_FW_RIGHT_YAW_SIGN` × `dof_map.yaw.sign` × `telemetry.yaw_sign`：
      · 前两项决定"+yaw 命令"落在轴字节的哪一边（半边 = 右转，手动挡 `turn_right` 已验），
      · `telemetry.yaw_sign` 决定回传 yaw 的符号（它本来就是给这件事配的旋钮）。
    改任一旋钮 σ 自动跟着变，不会出现"配置与代码各记一套符号"。
    """
    sd, st = 1.0, 1.0
    try:
        import base.cfg.settings as S
        ch = (S.comm.get("dof_map", None) or {}).get("yaw", None) or {}
        sd = float(ch.get("sign", 1.0) or 1.0)
        st = float(S.get("comm.telemetry.yaw_sign", 1.0) or 1.0)
    except Exception:
        pass
    return _FW_RIGHT_YAW_SIGN * sd * st, "固件%+g × dof_map%+g × telemetry%+g" % (
        _FW_RIGHT_YAW_SIGN, sd, st)


def stop_hard(uart, log=print, verify=True):
    """把船**真正**停住（不是发一帧 neutral 就完事）。
    **字节级平滑**，`neutral()` 的 `force=True` **只绕过心跳节流、不绕过 ramp** → 单帧 neutral
    发出时轴字节还在半路（yaw 84→128 只到 ~95 = 仍在转）；而**下位机没有无帧超时停车**，
    随后 `close()` 一关串口，船就锁在那个值上一直转。
    优先用 `uart.stop_hard()`（连发中性帧走完 ramp + 遥测 yaw 验证）；没有该方法就自己连发。"""
    sh = getattr(uart, "stop_hard", None)
    try:
        if callable(sh):
            ok = sh(verify=verify)
            if ok is False:
                log("[TURN] ⚠️ 硬停未确认（遥测显示仍在转）——请人工确认船已停")
            return ok
        for _ in range(14):                     # 旧对象：自己连发足够帧走完 ramp
            uart.neutral()
        return True
    except TypeError:
        try:
            sh(verify=False)
            return True
        except Exception:
            uart.neutral()
            return False


class TurnCore(object):
    """下位机执行相对转角；上位机重复发送同一编号、等待匹配完成或超时。

    2026-10-02：**只认角大小与方向**——限幅/PID/死区/归一化这些"上位机运动参数"已整块删除
    （旋转由下位机执行，参数都在下位机那侧）。
    """
    IDLE, RUN, DONE, TIMEOUT, ABORTED = "idle", "run", "done", "timeout", "aborted"

    def __init__(self, deg=90.0, left=True, timeout=20.0, log=None):
        self.deg = abs(float(deg))
        if not math.isfinite(self.deg) or self.deg > 180:
            raise ValueError("转角大小必须在 0～180 度内")
        self.left = bool(left)
        self.d = -1.0 if self.left else 1.0
        self.timeout = float(timeout)
        self.log = log or (lambda *a: None)
        self.state = self.IDLE
        self.why = ""
        self.t_start = None
        self._sent = False
        self._uart = None

    def start(self, now_ms):
        self.t_start = now_ms
        self.state = self.RUN
        return self.state

    def finished(self):
        return self.state in (self.DONE, self.TIMEOUT, self.ABORTED)

    def abort(self, why="external"):
        if not self.finished():
            self.state, self.why = self.ABORTED, why
            if self._uart is not None:
                self._uart.cancel_turn()

    def step(self, now_ms, uart):
        if self.finished():
            return self.state, 0.0
        if self.state == self.IDLE:
            self.start(now_ms)
        self._uart = uart
        if getattr(uart, "estop_active", False):
            self.abort("external")
            return self.state, 0.0
        if not self._sent:
            self._sent = True  # 重发由 UART 层管理，保持同一个编号
            try:
                ok = uart.request_turn(self.deg * self.d)
            except Exception as exc:
                self.log("[TURN] 下发角度失败: %s" % exc)
                ok = False
            if not ok:
                self.abort("send_failed")
                return self.state, 0.0
            self.log("[TURN] 下位机执行相对转角 %+.2f°" % (self.deg * self.d))
        if now_ms - self.t_start > self.timeout * 1000.0:
            self.state, self.why = self.TIMEOUT, "completion_timeout"
            uart.cancel_turn()
        elif uart.poll_turn_complete():
            self.state = self.DONE
            self.log("[TURN] 下位机回报转向完成")
        return self.state, 0.0  # 执行期间手动 yaw 永远回中


def turn(uart, deg=90.0, left=True, timeout=20.0, log=print, now=None, sleep=None):
    """脚本入口：把 `deg/left` 交给下位机执行，等完成反馈。0=完成，3=等待超时，5=下发失败/中止。

    运动参数（限幅 PID 增益、死区…）**一律不在这里**：旋转由下位机执行（2026-10-02 定）。
    """
    now, sleep = now or time.time, sleep or time.sleep
    core = TurnCore(deg=deg, left=left, timeout=timeout, log=log)
    try:
        while not core.finished():
            core.step(int(now() * 1000), uart)
            uart.send_dof(0.0, 0.0, 0.0, 0.0)
            if not core.finished():
                sleep(0.05)
    finally:
        stop_hard(uart, log)
    return 0 if core.state == core.DONE else (3 if core.state == core.TIMEOUT else 5)


def main():
    """**手动入口**：命令行给一个角度就转过去（与 gate 自动接受同一条执行链）。

    两个入口共用 `turn()` → `TurnCore` → `uart.request_turn()`（下发"相对角 + 编号"，等完成反馈）：
      · **手动**：`--deg/--dir`（本 CLI；`task1_2/run_ball_reverse.sh` 也走这条）；
      · **自动**：gate 的 ALIGN.HDG 用测到的 ψ 当目标角调 `TurnCore`。

    ⚠️ **只暴露三个参数**（用户 2026-10-02 定）：`--deg` / `--dir` / `--timeout`。
    其余运动参数（限幅、PID 增益、死区、归一化…）**都归下位机**：新模型下上位机只发"相对角 + 编号"，
    所以这里**不再接受**旧的 `--out-max/--kp/--kd/--norm-deg/--imag-sign` 等旋钮（老脚本请一并去掉）。
    """
    ap = argparse.ArgumentParser(description="相对角度原地转（手动给角度；下位机执行、完成标志确认）")
    ap.add_argument("--deg", type=float, default=90.0, help="角度大小（正数，度）")
    ap.add_argument("--dir", choices=("left", "right"), default="left",
                    help="left=左转(−角) / right=右转(+角)")
    ap.add_argument("--timeout", type=float, default=20.0, help="等待完成反馈的超时秒数")
    args = ap.parse_args()

    from base.hw.uart import UartController
    u = UartController()
    if u.sim:
        print("!! 串口处于 SIM（只打印）：[turn] 不会真正驱动电机")
    try:
        print("[turn] 手动目标角：%s %.1f°（超时 %.0fs）"
              % ("左转" if args.dir == "left" else "右转", args.deg, args.timeout))
        rc = turn(u, deg=args.deg, left=(args.dir == "left"), timeout=args.timeout)
    except KeyboardInterrupt:
        print("\n[turn] Ctrl-C → 硬停")
        rc = 130
        try:
            stop_hard(u, print)
        except Exception:
            pass
    finally:
        u.close()
    print("[turn] 退出码 %d（0=下位机回报完成 / 3=等完成超时 / 5=下发失败或中止）" % rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
