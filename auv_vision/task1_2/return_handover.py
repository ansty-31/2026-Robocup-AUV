# -*- coding: utf-8 -*-
"""return_handover.py — 返回出发区 策略B（视觉居中 + 交下位机接管握手）

撞球任务 DONE 后执行（由 run_ball_return.sh 的 AUV_RETURN=handover 选择）：
  1) 后退 rev1_s
  2) 原地 yaw 居中：转动使小球球心水平居中（y_align=true 时也用 heave 修垂直）
  3) 换帧头 handover.header(0xAA) 发中性帧 → 表示“暂时交给下位机接管”
     待命：持续以该帧头发中性帧心跳（standby_neutral，≥20Hz）
  4) 等下位机回传“特殊标志” handover.resume_flag（内容待定；空则不判定内容并跳过等待）
  5) 上位机收权：**不再动 yaw**，仅 sway 左右平移使球心居中（y_align 时也用 heave）
  6) 居中后 → 直接后退 rev2_s → 回中位

配置：cfg/comm.yaml `handover:` 段。

⚠️ 固件依赖：当前下位机 `RC_InputByte()` 只认帧头 0xA5，**不认识 0xAA 交接帧头、
也不会回传特殊标志**。因此本模块是“上位机侧可配置骨架”——需下位机固件实现
`识别 handover.header 接管` 与 `回传 resume_flag` 后才能真正联调。

用法：
  python3 task1_2/return_handover.py                 # 真机执行（需回车确认）
  python3 task1_2/return_handover.py --yes           # 无人值守（不回车）
  python3 task1_2/return_handover.py --sim           # 串口仿真
  python3 task1_2/return_handover.py --dry-run       # 只打印流程，不执行
  python3 task1_2/return_handover.py --flag aa55ff   # 临时覆盖 resume_flag(hex)
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base.settings as S                      # noqa: E402
from base.uart import UartController           # noqa: E402
from base.camera import create_camera          # noqa: E402
from common.detector import DetectorHub        # noqa: E402


# ---------------------------------------------------------------------------
# 配置/工具
# ---------------------------------------------------------------------------
def cfg(name, default=None):
    """读 comm.handover.<name>。"""
    return S.get("comm.handover." + name, default)


def hex_to_bytes(s):
    s = (s or "").strip().replace(" ", "").replace("0x", "")
    if not s:
        return b""
    if len(s) % 2:
        s = "0" + s
    try:
        return bytes.fromhex(s)
    except ValueError:
        print("!! resume_flag 不是合法十六进制: %r" % s)
        return b""


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def hold_dof(uart, surge=0.0, sway=0.0, heave=0.0, yaw=0.0, dur=0.0, period=0.05):
    """持续发送给定 DOF 目标 dur 秒（走 ramp 平滑）；dur<=0 不执行。"""
    if dur <= 0:
        return
    t0 = time.time()
    while time.time() - t0 < dur:
        uart.send_dof(surge, sway, heave, yaw)
        time.sleep(period)


def read_ball(hub, cam):
    """取一帧并检测目标球；返回 (frame, det|None)。"""
    frame = cam.read()
    if frame is None:
        return None, None
    return frame, hub.detect("ball", frame)


def center_loop(uart, hub, cam, out_axis, err_axis, sign, kp, out_max,
                eps, confirm, timeout, tag):
    """比例居中：out_axis∈{yaw,sway,heave}，err_axis∈{dx,dy}。

    居中判据：|err|<=eps 连续 confirm 帧。返回 True=已居中，False=超时/放弃。
    """
    cnt = 0
    t0 = time.time()
    period = 1.0 / max(5.0, float(getattr(cam, "fps", 20) or 20))
    w = float(getattr(cam, "width", 640) or 640)
    h = float(getattr(cam, "height", 360) or 360)
    while True:
        if timeout and (time.time() - t0) >= timeout:
            print("[%s] 超时未居中" % tag)
            return False
        frame, det = read_ball(hub, cam)
        if det is not None:
            cx, cy = det.center
            dx = (cx - w / 2.0) / (w / 2.0)
            dy = (cy - h / 2.0) / (h / 2.0)
            err = dx if err_axis == "dx" else dy
            if abs(err) <= eps:
                cnt += 1
            else:
                cnt = 0
            if cnt >= confirm:
                uart.send_dof(0.0, 0.0, 0.0, 0.0)
                print("[%s] 已居中(%s=%.3f)" % (tag, err_axis, err))
                return True
            out = clamp(sign * kp * err, -out_max, out_max)
            if out_axis == "yaw":
                uart.send_dof(0.0, 0.0, 0.0, out)
            elif out_axis == "sway":
                uart.send_dof(0.0, out, 0.0, 0.0)
            else:                                   # heave
                uart.send_dof(0.0, 0.0, out, 0.0)
        else:
            cnt = 0
            uart.send_dof(0.0, 0.0, 0.0, 0.0)       # 丢目标：停止等待
        time.sleep(period)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(uart, cam, hub, header, flag, P):
    # 1) 先后退
    print("[1/6] 后退 %.1fs (surge=%.2f)" % (P["rev1_s"], P["rev_speed"]))
    hold_dof(uart, surge=P["rev_speed"], dur=P["rev1_s"])

    # 2) 原地 yaw 居中
    print("[2/6] 原地 yaw 居中")
    center_loop(uart, hub, cam, "yaw", "dx", P["yaw_sign"], P["yaw_kp"],
                P["yaw_max"], P["center_x_eps"], P["center_confirm_frames"],
                P["center_timeout_s"], "yaw-center")
    if P["y_align"]:
        print("[2b] 用 heave 修垂直居中")
        center_loop(uart, hub, cam, "heave", "dy", P["sway_sign"],
                    P["sway_kp"], P["sway_max"], P["center_x_eps"],
                    P["center_confirm_frames"], P["center_timeout_s"],
                    "heave-center")

    # 3) 交接：换帧头发中性帧
    print("[3/6] 交接：以帧头 0x%02X 发中性帧 → 交给下位机接管" % header)
    uart.rx_clear()
    uart.send_dof_header(header, 0.0, 0.0, 0.0, 0.0, force=True)

    # 4) 待命：中性心跳 + 等下位机特殊标志
    if flag:
        print("[4/6] 待命：等标志 %s（timeout=%ss；期间持续发中性心跳）"
              % (flag.hex(), P["resume_timeout_s"] or "∞"))
        tick = None
        if P["standby_neutral"]:
            tick = lambda: uart.send_dof_header(header, 0.0, 0.0, 0.0, 0.0,
                                                force=True)
        got = uart.wait_flag(flag, timeout_ms=int(P["resume_timeout_s"] * 1000),
                             tick=tick)
        print("[4/6] 收到标志=%s" % got)
    else:
        print("[4/6] resume_flag 未配置 → 跳过等待(骨架模式)，直接收权")

    # 5) 收权：不再动 yaw，仅 sway 居中
    print("[5/6] 收权：仅 sway 左右平移居中")
    center_loop(uart, hub, cam, "sway", "dx", P["sway_sign"], P["sway_kp"],
                P["sway_max"], P["center_x_eps"], P["center_confirm_frames"],
                P["center_timeout_s"], "sway-center")
    if P["y_align"]:
        center_loop(uart, hub, cam, "heave", "dy", P["sway_sign"],
                    P["sway_kp"], P["sway_max"], P["center_x_eps"],
                    P["center_confirm_frames"], P["center_timeout_s"],
                    "heave-center")

    # 6) 直接后退
    print("[6/6] 后退 %.1fs (surge=%.2f)" % (P["rev2_s"], P["rev_speed"]))
    hold_dof(uart, surge=P["rev_speed"], dur=P["rev2_s"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="串口仿真（不发真串口）")
    ap.add_argument("--dry-run", action="store_true", help="只打印流程，不执行")
    ap.add_argument("--yes", action="store_true", help="跳过回车确认（无人值守）")
    ap.add_argument("--flag", default=None, help="临时覆盖 resume_flag(hex)")
    a = ap.parse_args()

    header = int(cfg("header", 0xAA))
    flag = hex_to_bytes(a.flag if a.flag is not None else cfg("resume_flag", ""))
    P = dict(
        rev1_s=float(cfg("rev1_s", 1.0)),
        rev2_s=float(cfg("rev2_s", 3.0)),
        rev_speed=float(cfg("rev_speed", -0.35)),
        yaw_kp=float(cfg("yaw_kp", 0.8)),
        yaw_max=float(cfg("yaw_max", 0.5)),
        yaw_sign=float(cfg("yaw_sign", 1)),
        sway_kp=float(cfg("sway_kp", 0.9)),
        sway_max=float(cfg("sway_max", 0.6)),
        sway_sign=float(cfg("sway_sign", -1)),
        y_align=bool(cfg("y_align", False)),
        center_x_eps=float(cfg("center_x_eps", 0.10)),
        center_confirm_frames=int(cfg("center_confirm_frames", 3)),
        center_timeout_s=float(cfg("center_timeout_s", 20)),
        resume_timeout_s=float(cfg("resume_timeout_s", 0)),
        standby_neutral=bool(cfg("standby_neutral", True)),
    )
    print("=" * 66)
    print(" 返回策略B：后退%.1fs → yaw居中 → 交接(0x%02X) → 待命等标志 → sway居中 → 后退%.1fs"
          % (P["rev1_s"], header, P["rev2_s"]))
    print(" resume_flag=%s  resume_timeout=%ss  standby_neutral=%s"
          % (flag.hex() or "(未配置)", P["resume_timeout_s"] or "∞", P["standby_neutral"]))
    print("=" * 66)

    if a.dry_run:
        print("[dry-run] 不执行")
        return 0

    if not a.yes:
        input("回车开始（请确认周围安全，Ctrl-C 中断）...")

    uart = UartController(sim=True if a.sim else None)
    if not a.sim and uart.sim:
        print("!! 串口未打开，无法发送——检查 comm.yaml serial")
        return 1
    cam = create_camera("front")
    hub = DetectorHub()
    try:
        run(uart, cam, hub, header, flag, P)
    except KeyboardInterrupt:
        print("\n[HANDOVER] 手动中断")
    finally:
        uart.neutral()
        uart.close()
        print("[HANDOVER] 已回中位停止")
    return 0


if __name__ == "__main__":
    sys.exit(main())
