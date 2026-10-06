#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/check/check_search_scan.py — **单独跑搜索扫描**（只看转圈/不转圈，不下水任务）
用法（工程根执行）：
⚠️ 安全：脚本退出（含 Ctrl-C）一定先 `stop_hard` 回中位再关串口。
（详细用法、判据与实测见 doc/注释历史.md）"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import base.cfg.settings as S                      # noqa: E402
from common.motion.search_scan import Scan, telemetry_yaw   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="单独跑搜索扫描（只动 yaw）")
    ap.add_argument("--duration", type=float, default=30.0, help="跑多久（秒），默认 30")
    ap.add_argument("--hz", type=float, default=10.0,
                    help="下发频率（Hz）；默认 10 = 主任务帧率（睁眼的节奏）")
    ap.add_argument("--span", type=float, default=None,
                    help="单侧幅度（度）；不给就用 cfg 的 motion.search_scan.span_deg")
    ap.add_argument("--pause", type=float, default=None,
                    help="每段到位后停顿（ms）；不给就用 cfg 值")
    ap.add_argument("--sim", action="store_true", help="只打印、不驱动电机（台架）")
    a = ap.parse_args()

    if a.sim:
        os.environ["AUV_SIM_MODE"] = "1"

    node = None
    if a.span is not None or a.pause is not None:
        cur = S.get("comm.motion.search_scan", {}) or {}
        node = S.Y(dict(cur))
        if a.span is not None:
            node["span_deg"] = float(a.span)
        if a.pause is not None:
            node["pause_ms"] = float(a.pause)

    print("=" * 66)
    print(" 单独搜索扫描：只动 yaw，其余三轴恒 0")
    print("=" * 66)

    scan = Scan(node=node, log=lambda *x: print("  " + (x[0] % x[1:]) if len(x) > 1 else "  " + str(x[0])))
    print("  span=%.0f° ⇒ 总扫幅 %.0f° | pause=%.0fms | tol=%.0f° | out_max=%.2f | σ=%+.0f"
          % (scan.span, scan.span_total_deg, scan.pause_ms, scan.tol,
             scan._pid.out_max, scan.sigma))

    if a.sim:
        print("  ⚠ AUV_SIM_MODE=1：只打印指令，不会真正驱动电机")
    u = None
    if not a.sim:
        from base.hw.uart import UartController
        u = UartController()
        if getattr(u, "sim", False):
            print("  !! 串口处于 SIM（只打印）：不会真正驱动电机")

    dt = 1.0 / max(1e-6, a.hz)
    t0 = time.time()
    now_ms = 0.0
    n_no_tel = 0
    try:
        while time.time() - t0 < a.duration:
            tel = telemetry_yaw(u) if u is not None else 0.0     # --sim 时用假遥测（不动电机，只看波形）
            if tel is None:
                n_no_tel += 1
            yaw = scan.step(now_ms, tel)
            if u is not None:
                u.send_dof(0.0, 0.0, 0.0, float(yaw))
            print("  t=%5.1fs psi=%7.1f° 段目标=%7s 残差=%7s yaw=%+.3f %s"
                  % (time.time() - t0,
                     (scan.sigma * float(tel)) if tel is not None else float("nan"),
                     ("%.1f" % scan.target_psi) if scan.target_psi is not None else "-",
                     ("%+.1f" % (scan.target_psi - scan.sigma * float(tel)))
                     if (scan.target_psi is not None and tel is not None) else "-",
                     float(yaw),
                     "（无遥测）" if tel is None else ""))
            time.sleep(dt)
            now_ms += dt * 1000.0
    except KeyboardInterrupt:
        print("\n  收到 Ctrl-C")
    finally:
        if u is not None:
            try:
                u.stop_hard(verify=False)      # ⚠️ 必须 stop_hard：走完 ramp 才算真回中位
            except Exception:
                pass
            try:
                u.close()
            except Exception:
                pass
            print("  [ok] 已回中位（硬停）并关串口")
    print("  走了 %d 段 | 无遥测帧 %d | 结束" % (scan.legs_done, n_no_tel))
    if n_no_tel and n_no_tel > 0.9 * max(1, int(a.duration / dt)):
        print("  !! 几乎每帧都没有遥测 ⇒ 闭环没有输入量，船不会转：查下位机 15B 上行帧/串口")


if __name__ == "__main__":
    main()
