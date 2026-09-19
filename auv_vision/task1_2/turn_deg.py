#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""turn_deg.py — 按**额定角度**原地转（既有 `common/PID.py` + 下位机遥测 yaw 闭环）。

闭环量：下位机回传的**绝对航向** `yaw_deg`（0.01° 分辨率，`base/telemetry.py`）。
为什么闭环：我们不知道"多少秒 = 90°"——yaw 角速度随电量/水流/负载变（manual 的 `turn_right`
只是 0.4 的固定舵量）。遥测给了绝对航向，于是可以把"还剩多少度"喂给 PID。

**控制器 = 项目既有的 `common/PID.PID`**（与 `task1_2/ball.py` 的居中转向同一个件、同一套用法：
位置式、死区清零积分、限幅防 windup）。参数走 **`comm.motion.turn_pid` 这一套独立值**
（2026-09-18 用户定：转角是独立动作，不跟小球那套联动；"0.3 有点大"→ 现 kp=0.2）。

    comm.motion.turn_pid.{kp,ki,kd,out_max,deadzone_deg,norm_deg}   ← 唯一来源
    代码内置默认（同值）                                            ← cfg 缺键时

两个旋钮各管什么（`err_norm = 剩余角度 / norm_deg`）：
    **kp**      管"多早开始收力/收敛多紧"（大=晚收力、残差小、近目标凶）
    **out_max** 管"最大转速（满舵推力）"（想整体更慢就降它，别只降 kp）

误差单位：PID 吃的是**归一化误差** `err_norm = 剩余角度 / norm_deg`
（小球/过门那套增益是给"归一化误差"标的，所以这里必须归一化，否则量纲不对）。
输出单位：DOF，直接就是 `send_dof(..., yaw=out)`；**+ = 右转**。

⚠️ **`out_max` 不能取 0.15 那一档**（= `ball.edge_yaw_max`）：15% 推力恰好卡在执行器
   死区 0.138 边上 ⇒ 转大角度会"还剩二十几度就没推力"（现场"yaw 调不动"就是这个）。
   可用区间 0.35~0.5。当前算式（kp=0.2, out_max=0.45, norm_deg=15）：
       err=90° → err_norm=6.0 → out=min(1.2, 0.45)=0.45（45% 推力）
       err=34° → 2.25 → 0.45（满舵到这儿）
       err=20° → 1.33 → 0.27
       err=15° → 1.00 → 0.20
       err≈10.4° → 0.69 → 0.139（刚过死区，还修得动）
       err≤6°  → 进 PID 死区 → 输出 0 ⇒ **收敛残差 ≈6~10°**（执行器死区给的物理下限）

符号约定（沿用既有约定，不自创）：**`+yaw DOF = 右转`**（`manual/udp_server.py` 的
`0,0,0,0.6 = yaw right`）。遥测 yaw 正方向固件没文档，所以先给 0.6s 小舵**探向**：
遥测随 +DOF 增大 → `imag_sign=+1`；之后用 `psi = imag_sign × yaw`（随 +DOF 增大）做闭环，
PID 输出直接就是 DOF，符号天然正确。

用法（**接口只有角度**：角度 + 方向，不涉及时长）：
    python3 task1_2/turn_deg.py --deg 90 --dir left
    python3 task1_2/turn_deg.py --deg 90 --dir right --out-max 0.5 --kp 0.25

没有遥测怎么办（2026-09-18 用户定：**不该有时长参数**）：**拒转**并报错（退出码 5）。
   盲转（不看遥测、按时长开环转）必须**显式**要：`--blind`（用"速率假设"
   `comm.motion.turn_pid.blind_rate_dps` 把角度换算成时长）。默认不允许 ——
   盲转的落点不可信，水里可能把船甩到池壁/门上。

退出码：0=到达目标角 ｜ 2=盲转完成(--blind) ｜ 3=超时未到达 ｜ 4=舵效不足（几乎没转）
        ｜ 5=**无遥测 → 拒转**（默认行为；要盲转就加 --blind）
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

# ⚠️ 直接跑 `python3 task1_2/turn_deg.py` 时，Python 的 `sys.path[0]` 是**脚本所在目录**
#    （task1_2/），**不含项目根** → 板上（Python 3.10）会
#    `ModuleNotFoundError: No module named 'base'`（2026-09-18 板上实测踩到；
#    本地 Python 3.13 恰好把 CWD 也算进去了，所以本地没暴露）。
#    必须在 import base 之前执行。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 代码内兜底（cfg 与 ball 都取不到时用；正常走 comm.ball.edge_yaw_*）
_D_TURN_PID = dict(kp=0.2, ki=0.0, kd=0.0, out_max=0.45,
                   deadzone_deg=6.0, norm_deg=15.0,
                   # 只用于 `--blind`（不看遥测的开环盲转）：把角度换算成时长用的
                   # "yaw 角速率假设"。它是**物理量假设**，不是控制参数；想盲转就得标它。
                   blind_rate_dps=20.0)


def wrap180(deg):
    """把角度差归一化到 (-180, 180]。"""
    x = math.fmod(deg + 180.0, 360.0)
    if x <= 0:
        x += 360.0
    return x - 180.0


def _num(node, key, default):
    try:
        v = node[key]
    except (KeyError, TypeError):
        return float(default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def turn_pid_cfg():
    """解析额定转角的 PID 参数（与 gate_task 解析 align_yaw 同一套路，缺啥都不崩）。

    返回 (cfg, src)；src 用于打印"这一跑用的是哪一套增益"（现场一眼可查）。
    """
    out = dict(_D_TURN_PID)
    src = "代码内置"
    try:
        import base.settings as S
        # **独立一套**（2026-09-18 用户定）：转角是独立动作，不跟小球/门那套联动。
        # 只读 `comm.motion.turn_pid`；缺键 → 用代码默认（不抛异常）。
        node = (S.comm.get("motion", None) or {}).get("turn_pid", None)
        if isinstance(node, dict):
            for k in out:
                if node.get(k) is not None:
                    out[k] = _num(node, k, out[k])
            src = "turn_pid(独立)"
    except Exception:
        pass
    return out, src


def _stop(uart, log, verify=True):
    """把船**真正**停住（不是发一帧 neutral 就完事）。

    为什么必须（2026-09-18 用户水里实测 + 代码核实）：`base/uart.py::_ramp_step` 对
    yaw/surge/sway/heave 做**字节级平滑**，`neutral()` 的 `force=True` **只绕过心跳节流、
    不绕过 ramp** → 单帧 neutral 发出时轴字节还在半路（yaw 84→128 只到 ~95 = 仍在转）；
    而**下位机没有无帧超时停车**，随后 `close()` 一关串口，船就锁在那个值上一直转。
    所以这里调 `uart.stop_hard()`：连发中性帧走完 ramp，再用**遥测 yaw** 确认停住。
    兼容：对象若没有 stop_hard（旧版/假对象）就退回连发 neutral。
    """
    sh = getattr(uart, "stop_hard", None)
    try:
        if callable(sh):
            ok = sh(verify=verify)
            if ok is False:
                log("[TURN] ⚠️ 硬停未确认（遥测显示仍在转）——请人工确认船已停")
        else:
            for _ in range(14):                 # 旧对象：自己连发足够帧走完 ramp
                uart.neutral()
    except TypeError:
        sh(verify=False) if callable(sh) else uart.neutral()


def turn(uart, deg=90.0, left=True, out_max=None, timeout=20.0, tol_deg=None,
         kp=None, ki=None, kd=None, deadzone_deg=None, norm_deg=None, confirm=3,
         probe_s=0.6, probe_dof=0.3, wait_tel_s=3.0, imag_sign=None,
         mode="closed", blind_rate_dps=None, period=0.05, log=print,
         now=None, sleep=None):
    """转到额定角度。返回退出码（0/2/3/4）。可注入 now/sleep 以便离线测试。"""
    from common.PID import PID

    now = now or (lambda: time.time())
    sleep = sleep or time.sleep
    cfg, src = turn_pid_cfg()
    om = float(out_max if out_max is not None else cfg["out_max"])
    kp = float(kp if kp is not None else cfg["kp"])
    ki = float(ki if ki is not None else cfg["ki"])
    kd = float(kd if kd is not None else cfg["kd"])
    dz = float(deadzone_deg if deadzone_deg is not None else cfg["deadzone_deg"])
    nd = max(0.1, float(norm_deg if norm_deg is not None else cfg["norm_deg"]))
    rate = max(1.0, float(blind_rate_dps if blind_rate_dps is not None
                          else cfg.get("blind_rate_dps", 20.0)))
    tol = float(tol_deg if tol_deg is not None else dz)   # 到达容差默认=死区
    tel = getattr(uart, "telemetry", None)

    def _yaw():
        return None if tel is None else getattr(tel, "yaw_deg", None)

    d = -1.0 if left else 1.0                 # 期望的 DOF 符号：左 =-（+yaw=右转）

    # ---------------- 盲转（**显式**要求才做：开环、按时长、落点不可信）----------------
    if mode == "timed" or str(imag_sign) == "timed":
        dur = float(deg) / rate                 # ← 接口是角度；时长由"速率假设"换算
        log("[TURN] ⚠️ 盲转（开环）：%g° ÷ %.1f°/s = %.1fs @ yaw=%+.2f"
            "（**落点不可信**，速率假设见 comm.motion.turn_pid.blind_rate_dps）"
            % (deg, rate, dur, d * om))
        t0 = now()
        try:
            while now() - t0 < dur:
                uart.send_dof(0.0, 0.0, 0.0, d * om)
                sleep(period)
        finally:
            _stop(uart, log)
        return 2

    # ---------------- 等遥测 ----------------
    t0 = now()
    while _yaw() is None and now() - t0 < wait_tel_s:
        uart.send_dof(0.0, 0.0, 0.0, 0.0)
        sleep(period)
    if _yaw() is None:
        # **拒转**（2026-09-18 用户定）：接口只有角度，闭环缺了闭环量就不该转。
        # 盲转要显式 --blind（上层 mode="timed"），否则这里直接失败，避免"以为转了 90° 其实瞎转"。
        log("[TURN] ⛔ %.1fs 内没拿到遥测 yaw → **拒绝转向**（闭环量缺失）。"
            "查：下位机是否在回传（控制台应有 [UART←] depth=… yaw=…）；"
            "确实要开环盲转就加 --blind" % wait_tel_s)
        _stop(uart, log, verify=False)
        return 5

    # ---------------- 探向：遥测 yaw 的正方向相对我们的 +yaw 约定 ----------------
    y_ref = _yaw()
    if imag_sign in (None, 0) or str(imag_sign) == "auto":
        y0 = y_ref
        t0 = now()
        try:
            while now() - t0 < probe_s:
                uart.send_dof(0.0, 0.0, 0.0, probe_dof * d)
                sleep(period)
        finally:
            _stop(uart, log, verify=False)     # 探向也是一次转向 → 同样要停住
        sleep(0.15)                            # 停稳再读
        y1 = _yaw() or y0
        dy = wrap180((y1 or 0.0) - (y0 or 0.0))
        if abs(dy) < 0.3:
            log("[TURN] ⚠️ 探向 Δyaw=%.2f°（几乎没动）→ 舵效不足或遥测不更新" % dy)
            imag_sign = 1.0
        else:
            imag_sign = 1.0 if (dy * d) > 0 else -1.0
            log("[TURN] 探向：给 yaw=%+.2f×%.1fs 时 Δyaw=%+.2f° → 遥测正方向=%s"
                % (probe_dof * d, probe_s, dy,
                   "与 +yaw(右转) 同向" if imag_sign > 0 else "与 +yaw(右转) 反向"))
    else:
        imag_sign = float(imag_sign)

    # 换算到"我方约定朝向"：psi 随 +DOF 增大；目标 = 起始 psi + deg×d
    # ⚠️ 基准取**探向之前**的角度：探向已转掉的部分必须计入总量（否则白送几度）
    psi0 = wrap180(imag_sign * (y_ref or 0.0))
    psi_tgt = wrap180(psi0 + deg * d)
    pid = PID(kp=kp, ki=ki, kd=kd, out_min=-om, out_max=om,
              deadzone=dz / nd)                 # PID 吃归一化误差 → 死区也要归一化
    log("[TURN] PID 闭环（%s）：目标 psi=%+.1f°（%s %g°，从 %+.1f° 起）"
        % (src, psi_tgt, "左转" if left else "右转", deg, psi0))
    log("[TURN]   增益 kp=%.3f kd=%.3f ki=%.3f ｜限幅 ±%.2f ｜死区 %.1f° ｜"
        "归一化 norm_deg=%.1f ｜到达容差 %.1f°"
        % (kp, kd, ki, om, dz, nd, tol))

    t0 = now()
    last_log = t0
    on_target = 0
    out = 0.0
    max_dev = 0.0
    try:
        while now() - t0 < timeout:
            y = _yaw()
            if y is None:                      # 遥测中断：保持中性等下一帧，不盲转
                uart.send_dof(0.0, 0.0, 0.0, 0.0)
                sleep(period)
                continue
            psi = wrap180(imag_sign * y)
            err = wrap180(psi_tgt - psi)       # 还剩多少度（度）
            max_dev = max(max_dev, abs(err))
            out = pid.update(err / nd, int(now() * 1000.0))   # ← 既有 PID：吃归一化误差
            uart.send_dof(0.0, 0.0, 0.0, out)
            sleep(period)
            if now() - last_log >= 0.5:
                log("[TURN]   err=%+6.1f° → yaw=%+.3f（已转 %+.1f°）"
                    % (err, out, wrap180(psi - psi0)))
                last_log = now()
            on_target = on_target + 1 if abs(err) <= tol else 0
            if on_target >= max(1, int(confirm)):
                log("[TURN] ✅ 到达：err=%+.1f°（≤容差 %.1f° 连续 %d 帧）｜实测已转 %+.1f°"
                    "（目标 %+.1f°）" % (err, tol, on_target, wrap180(psi - psi0), deg * d))
                _stop(uart, log)               # ← 必须先真停住再返回
                return 0
    finally:
        pass
    _stop(uart, log)
    yy = _yaw()
    done_deg = wrap180(wrap180(imag_sign * yy) - psi0) if yy is not None else float("nan")
    left_deg = wrap180(psi_tgt - wrap180(imag_sign * yy)) if yy is not None else float("nan")
    log("[TURN] ❌ 超时 %.0fs：还差 %+.1f°｜实测只转了 %+.1f°（目标 %+.1f°）"
        "→ 舵效不足/卡住/增益过小" % (timeout, left_deg, done_deg, deg * d))
    # 4 = 几乎没转（舵效/接线/推进器问题）；3 = 转了一部分但没到位（增益/时间不够/被挡）
    return 4 if abs(done_deg) < 5.0 else 3


def main():
    ap = argparse.ArgumentParser(description="额定角度原地转（既有 PID + 遥测 yaw 闭环）")
    ap.add_argument("--deg", type=float, default=90.0, help="角度大小（正数）")
    ap.add_argument("--dir", choices=("left", "right"), default="left")
    ap.add_argument("--out-max", type=float, default=None,
                    help="转向限幅 DOF（默认取 comm.motion.turn_pid.out_max）")
    ap.add_argument("--yaw", type=float, default=None,
                    help="同 --out-max（兼容旧脚本写法）")
    ap.add_argument("--kp", type=float, default=None, help="覆盖 PID kp（默认=ball.edge_yaw_kp）")
    ap.add_argument("--ki", type=float, default=None)
    ap.add_argument("--kd", type=float, default=None)
    ap.add_argument("--deadzone-deg", type=float, default=None,
                    help="PID 死区（度）；也是「到达容差」的默认值")
    ap.add_argument("--norm-deg", type=float, default=None,
                    help="误差归一化分母（度）：err_norm = 剩余角度 / 它")
    ap.add_argument("--timeout", type=float, default=20.0, help="闭环超时秒数")
    ap.add_argument("--blind", action="store_true",
                    help="**开环盲转**（不看遥测，按角度÷速率定时转）；默认关，缺遥测直接拒转")
    ap.add_argument("--rate-dps", type=float, default=None,
                    help="盲转用的角速率假设（度/秒），仅 --blind 时用")
    ap.add_argument("--imag-sign", default="auto",
                    help="遥测 yaw 正方向相对 +yaw(右转)：auto/1/-1（已知就别探向）")
    args = ap.parse_args()

    from base.uart import UartController
    u = UartController()
    if u.sim:
        print("!! 串口处于 SIM（只打印）：[turn] 不会真正驱动电机")
    try:
        rc = turn(u, deg=args.deg, left=(args.dir == "left"),
                  out_max=(args.out_max if args.out_max is not None else args.yaw),
                  timeout=args.timeout, kp=args.kp, ki=args.ki, kd=args.kd,
                  deadzone_deg=args.deadzone_deg, norm_deg=args.norm_deg,
                  mode=("timed" if args.blind else "closed"),
                  blind_rate_dps=args.rate_dps, imag_sign=args.imag_sign)
    except KeyboardInterrupt:
        print("\n[turn] Ctrl-C → 硬停")
        rc = 130
        try:
            _stop(u, print)
        except Exception:
            pass
    finally:
        u.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
