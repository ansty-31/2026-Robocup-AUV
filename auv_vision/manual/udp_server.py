# -*- coding: utf-8 -*-
"""
udp_server.py - RDK-side UDP remote-control bridge for keyboard testing.

Run this script on the RDK. It receives PC keyboard commands over UDP and sends
the existing 0xA5 11-byte RC-compatible frame to STM32 through uart.py.

Packet format:
    surge,sway,heave,yaw

Each value is a float in [-1.0, 1.0].

Examples:
    1,0,0,0      forward
    0,-1,0,0     sway left
    0,0,1,0      heave up
    0,0,0.0,0.6  yaw right
    stop         neutral

Typical RDK command:
    python3 udp_server.py

Dry run without STM32 serial hardware:
    python3 udp_server.py --sim

Telemetry (STM32 -> RDK): UartController 收 14B 遥测帧(0xAA55 + 深度/姿态 + 校验和，
协议见 base/telemetry.py)，并按 cfg/comm.yaml → comm.depth_guard 做**限深保护**：
深度 ≤ min_depth_m(默认 0.3m) 时禁止上浮——键盘按"上浮"也不会把机身顶出水面。
遥测打印：[UART←] depth=...（节流 comm.debug.tel_log_ms）。
"""
import os as _os, sys as _sys
if __package__ in (None, ""):        # 支持直接 python3 manual/xxx.py 运行
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import socket
import time

import base.settings as S
from base.uart import UartController


def clamp(value, lo=-1.0, hi=1.0):
    return max(lo, min(hi, float(value)))


def parse_motion_packet(data):
    text = data.decode("utf-8", errors="ignore").strip().lower()
    if text in ("", "stop", "neutral", "0"):
        return 0.0, 0.0, 0.0, 0.0

    parts = text.replace(" ", "").split(",")
    if len(parts) != 4:
        raise ValueError("expected: surge,sway,heave,yaw")
    return tuple(clamp(part) for part in parts)


def apply_runtime_overrides(args):
    """Command-line overrides only affect this process; YAML files are untouched."""
    if args.sim:
        S.SIM_MODE = True
    elif args.real:
        S.SIM_MODE = False

    if args.uart_dev is not None:
        S.comm.serial.device = args.uart_dev
    if args.baud is not None:
        S.comm.serial.baud = int(args.baud)
    if args.ramp is not None:
        S.comm.ramp.speed_per_s = float(args.ramp)
    if args.debug is not None:
        S.DEBUG = bool(args.debug)


def main():
    parser = argparse.ArgumentParser(
        description="UDP keyboard-control bridge for RDK X5 AUV upper computer"
    )
    parser.add_argument("--bind", default="0.0.0.0",
                        help="UDP bind address, default: 0.0.0.0")
    parser.add_argument("--port", type=int, default=9000,
                        help="UDP port, default: 9000")
    parser.add_argument("--uart-dev", default=None,
                        help="override STM32 UART device from cfg/comm.yaml")
    parser.add_argument("--baud", type=int, default=None,
                        help="override STM32 UART baud from cfg/comm.yaml")
    parser.add_argument("--timeout-ms", type=int, default=300,
                        help="neutral timeout after PC packets stop, default: 300")
    parser.add_argument("--ramp", type=float, default=None,
                        help="override axis ramp speed; 0 disables ramp")
    parser.add_argument("--sim", action="store_true",
                        help="do not open the serial port; print frames")
    parser.add_argument("--real", action="store_true",
                        help="force real serial mode, overriding settings.SIM_MODE")
    parser.add_argument("--quiet", dest="debug", action="store_false",
                        help="reduce debug logging")
    parser.set_defaults(debug=None)
    args = parser.parse_args()

    apply_runtime_overrides(args)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind, args.port))
    sock.settimeout(0.02)

    uart = UartController()

    target = (0.0, 0.0, 0.0, 0.0)
    last_rx = time.time()
    last_log = 0.0
    timeout_s = max(1, args.timeout_ms) / 1000.0

    print("[UDP] listening on %s:%d" % (args.bind, args.port))
    print("[UDP] serial=%s baud=%d sim=%s timeout=%dms"
          % (S.comm.serial.device, S.comm.serial.baud, S.SIM_MODE,
             args.timeout_ms))
    print("[UDP] packet format: surge,sway,heave,yaw")
    print("[UDP] depth_guard=%s min_depth=%.2fm（深度来自下位机 14B 遥测 0xAA55；"
          "≤min_depth 时禁止上浮）"
          % (S.get("comm.depth_guard.enable", True),
             float(S.get("comm.depth_guard.min_depth_m", 0.3) or 0.0)))

    try:
        while True:
            now = time.time()
            try:
                data, addr = sock.recvfrom(256)
                target = parse_motion_packet(data)
                last_rx = now
                if now - last_log >= 0.25:      # 收到 PC 包就打印（诊断手动控制是否收到指令）
                    print("[UDP] %s -> surge=%.2f sway=%.2f heave=%.2f yaw=%.2f"
                          % (addr[0], target[0], target[1], target[2],
                             target[3]))
                    last_log = now
            except socket.timeout:
                pass
            except ValueError as exc:
                print("[UDP] bad packet: %r (%s)" % (bytes(data[:32]), exc))

            if now - last_rx > timeout_s:
                target = (0.0, 0.0, 0.0, 0.0)

            uart.send_dof(*target)     # 发帧时顺带收遥测/限深（base/uart.py:_drain_rx）
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("\n[UDP] keyboard interrupt")
    finally:
        try:
            uart.neutral()
            uart.estop()
        finally:
            uart.close()
            sock.close()
            print("[UDP] stopped")


if __name__ == "__main__":
    main()
