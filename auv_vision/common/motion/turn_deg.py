"""相对角度旋转（指定角度轴）：带编号重发目标角，由下位机执行，等待15字节遥测完成标志。

轴由协议 byte[7] 选择：`yaw`(1) / `pitch`(2) / `roll`(3) —— 见
`doc/记录/README_COMMUNICATION.md`（帧布局 §2、相对角度编码 §2.1、轴方向与目标 §2.2）。
`pitch`/`roll` 的 `deg/left` 是**相对角**：`right` = 正值 = 下位机 IMU 该轴测量值增大。
**物理方向（抬头/向左倾）必须现场实测确认**，协议正值不等于物理抬头。
（详细用法、判据与实测见 doc/注释历史.md）"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

# 工程根 = common/<类>/x.py 往上三级（本文件在子目录里，可直接当脚本跑）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from common.cfg.cfgnode import nums                                     # noqa: E402
from base.hw.uart import turn_axis_id                                   # noqa: E402 轴名→byte[7] 唯一来源
from base.log.turn_log import turn_log                                  # noqa: E402 转弯调用日志

_FW_RIGHT_YAW_SIGN = -1.0
_AXIS_NAMES = {1: "yaw", 2: "pitch", 3: "roll"}

def wrap180(deg):
    """把角度差归一化到 (-180, 180]。"""
    x = math.fmod(deg + 180.0, 360.0)
    if x <= 0:
        x += 360.0
    return x - 180.0


def yaw_sign():
    """σ：`psi = σ × 遥测 yaw`（**定死的乘式**，不是测量值）。返回 (σ, 说明串)。"""
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
    """把船**真正**停住（不是发一帧 neutral 就完事）。"""
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

    `axis`：`"yaw"`(默认) / `"pitch"` / `"roll"`，或 1/2/3。
    * yaw：`left=True` → 负角（左转），沿用既有约定（不要动）；
    * pitch/roll：`left=True` → **负角**，`left=False` → 正角（= 下位机 IMU 该轴测量值增大）。
      符号的**物理**含义（抬头/低头、向左倾/向右倾）**未验证**，上板前先做单轴实测。
    """

    IDLE, RUN, DONE, TIMEOUT, ABORTED = "idle", "run", "done", "timeout", "aborted"

    def __init__(self, deg=90.0, left=True, timeout=20.0, log=None, axis="yaw"):
        self.deg = abs(float(deg))
        if not math.isfinite(self.deg) or self.deg > 180:
            raise ValueError("转角大小必须在 0～180 度内")
        self.left = bool(left)
        self.d = -1.0 if self.left else 1.0
        self.axis = turn_axis_id(axis)          # 1=yaw / 2=pitch / 3=roll（协议 byte[7]）
        self.axis_name = _AXIS_NAMES.get(self.axis, "axis%d" % self.axis)
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
                if self.axis == 1:
                    # 既有 yaw 路径：**参数表不变**（老下位机对象/测试替身只认一个角度参数）
                    ok = uart.request_turn(self.deg * self.d)
                else:
                    ok = uart.request_turn(self.deg * self.d, axis=self.axis)
            except Exception as exc:
                # 不支持该轴（旧下位机对象 / 旧固件）→ 直接放弃，**不退回 yaw**（防动作走错轴）
                self.log("[TURN] 下发角度失败: %s" % exc)
                ok = False
            if not ok:
                self.abort("send_failed")
                return self.state, 0.0
            self.log("[TURN] 下位机执行相对转角 %+.2f°（轴=%s）"
                     % (self.deg * self.d, self.axis_name))
        if now_ms - self.t_start > self.timeout * 1000.0:
            self.state, self.why = self.TIMEOUT, "completion_timeout"
            uart.cancel_turn()
        elif uart.poll_turn_complete():
            self.state = self.DONE
            self.log("[TURN] 下位机回报转向完成")
        return self.state, 0.0  # 执行期间手动 yaw 永远回中


def turn(uart, deg=90.0, left=True, timeout=20.0, log=print, now=None, sleep=None,
         axis="yaw"):
    """脚本入口：把 `deg/left/axis` 交给下位机执行，等完成反馈。
    0=完成，3=等待超时，5=下发失败/中止。"""
    now, sleep = now or time.time, sleep or time.sleep
    core = TurnCore(deg=deg, left=left, timeout=timeout, log=log, axis=axis)
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
    """**手动入口**：命令行给一个角度就转/抬过去（与任务自动路径同一条执行链）。

    ⚠️ 运动参数仍**只有这三个**（用户 2026-10-02 定）：`--deg` / `--dir` / `--timeout`
    （限幅/PID/归一化等一律归下位机）。`--axis` 是**通道选择**，不是运动参数，
    2026-10-06 为"指定角度轴"协议（byte[7]）新增，默认 yaw ⇒ 原命令行为不变。
    """
    ap = argparse.ArgumentParser(description="指定角度轴相对转动（下位机执行、完成标志确认）")
    ap.add_argument("--deg", type=float, default=90.0, help="角度大小（正数，度）")
    ap.add_argument("--dir", choices=("left", "right"), default="left",
                    help="yaw: left=左转(−角) / right=右转(+角)；pitch/roll: 仅表示角的正负")
    ap.add_argument("--timeout", type=float, default=20.0, help="等待完成反馈的超时秒数")
    ap.add_argument("--axis", choices=("yaw", "pitch", "roll"), default="yaw",
                    help="指定角度轴：1=yaw / 2=pitch / 3=roll（协议 byte[7]）")
    args = ap.parse_args()

    from base.hw.uart import UartController
    u = UartController()
    if u.sim:
        print("!! 串口处于 SIM（只打印）：[turn] 不会真正驱动电机")
    try:
        print("[turn] 轴=%s 手动目标角：%+.1f°（超时 %.0fs）"
              % (args.axis, args.deg if args.dir == "right" else -args.deg, args.timeout))
        if args.axis != "yaw":
            print("       ⚠️ pitch/roll 的角符号只表示下位机 IMU 测量值的增减，"
                  "**物理方向（抬头/低头、左倾/右倾）未验证**，第一次上板请单轴小幅实测")
        rc = turn(u, deg=args.deg, left=(args.dir == "left"), timeout=args.timeout,
                  axis=args.axis)
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
