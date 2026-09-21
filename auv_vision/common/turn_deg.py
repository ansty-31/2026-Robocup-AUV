#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""turn_deg.py — 按**额定角度**原地转（公用件：既有 `common/PID.py` + 下位机遥测 yaw 闭环）。

**2026-09-18 由 task1_2/ 移到这里**（用户定：转弯是一个公用运动原语，脚本和 gate 任务都要用）。
两种驱动方式共用**同一套状态机**（`TurnCore`），保证"探向 / 增益 / 到位判据 / 硬停"只有一份实现：

    · **阻塞版** `turn(uart, deg, ...)`     —— 给脚本用（自己 sleep 循环，跑完才返回）
    · **逐帧版** `TurnCore`                —— 给主循环用（每帧 `step()` 一次，不阻塞相机/日志/fps）

闭环量：下位机回传的**绝对航向** `yaw_deg`（0.01° 分辨率，`base/telemetry.py`）。
为什么必须闭环：我们不知道"多少秒 = 90°"——yaw 角速度随电量/水流/负载变（manual 的 `turn_right`
只是 0.4 的固定舵量）。遥测给了绝对航向，于是可以直接把"还剩多少度"喂给 PID。

**接口只有角度**（角度 + 方向），不涉及时长：
    · 没有遥测 → **拒转**（返回码 5 / `TurnCore` 直接 aborted），要开环盲转必须显式 `--blind`，
      那时才用"角速率假设" `motion.turn_pid.blind_rate_dps` 把角度换算成时长。

符号约定（沿用既有约定，不自创）：**`+yaw DOF = 右转`**（`manual/udp_server.py` 的
`0,0,0,0.6 = yaw right`）。遥测 yaw 正方向固件没文档，所以先给 0.6s 小舵**探向**：
遥测随 +DOF 增大 → `imag_sign=+1`；之后用 `psi = imag_sign × yaw`（随 +DOF 增大）做闭环，
PID 输出直接就是 DOF，符号天然正确。**探向结果可以复用**（`imag_sign` 传进来就不用再探）。

PID：`common/PID.PID`（与 ball 的居中转向同一个件、同一套用法），参数走
`comm.motion.turn_pid`（**独立一套**，2026-09-18 用户定"0.3 有点大"→ 现 kp=0.2）：
    kp      管"多早开始收力/收敛多紧"    out_max 管"最大转速（满舵推力）"
    ⚠️ out_max 不能取 0.15 那一档（15% 推力恰卡在执行器死区 0.138 边上 ⇒ 转大角度会
       "还剩二十几度就没推力"）。可用区间 0.35~0.5。

退出码（阻塞版 main）：0=到达 ｜ 2=盲转完成(--blind) ｜ 3=超时未到达 ｜ 4=舵效不足（几乎没转）
                        ｜ 5=**无遥测 → 拒转**（默认；要盲转加 --blind）
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.PID import PID                                          # noqa: E402
from common.cfgnode import nums                                     # noqa: E402

# 代码内兜底（cfg 缺失时用；正常走 comm.motion.turn_pid）
_D_TURN_PID = dict(kp=0.2, ki=0.0, kd=0.05, out_max=0.45,
                   deadzone_deg=6.0, norm_deg=15.0,
                   blind_rate_dps=20.0)


def wrap180(deg):
    """把角度差归一化到 (-180, 180]。"""
    x = math.fmod(deg + 180.0, 360.0)
    if x <= 0:
        x += 360.0
    return x - 180.0


def turn_cfg():
    """读 `comm.motion.turn_pid`（独立一套；缺键 → 代码默认，不抛异常）。返回 (cfg, src)。"""
    out = dict(_D_TURN_PID)
    src = "代码内置"
    try:
        import base.settings as S
        node = (S.comm.get("motion", None) or {}).get("turn_pid", None)
        if isinstance(node, dict):
            out.update(nums(node, _D_TURN_PID))
            src = "turn_pid(独立)"
    except Exception:
        pass
    return out, src


def stop_hard(uart, log=print, verify=True):
    """把船**真正**停住（不是发一帧 neutral 就完事）。

    为什么必须（2026-09-18 用户水里实测 + 代码核实）：`base/uart.py::_ramp_step` 对
    yaw/surge/sway/heave 做**字节级平滑**，`neutral()` 的 `force=True` **只绕过心跳节流、
    不绕过 ramp** → 单帧 neutral 发出时轴字节还在半路（yaw 84→128 只到 ~95 = 仍在转）；
    而**下位机没有无帧超时停车**，随后 `close()` 一关串口，船就锁在那个值上一直转。
    优先用 `uart.stop_hard()`（连发中性帧走完 ramp + 遥测 yaw 验证）；没有该方法就自己连发。
    """
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
    """按角度原地转的**状态机**（逐帧推进；阻塞版只是它外面套了个 sleep 循环）。

    状态：idle → probe(可选) → run → done ｜ timeout ｜ aborted ｜ blind
    用法（逐帧）：
        core = TurnCore(deg=25.0, left=True, imag_sign=1.0)   # imag_sign=None → 先探向
        st, yaw_cmd = core.step(now_ms, yaw_telemetry_deg)    # yaw_telemetry 为 None 则等待
        if core.finished(): ...
    """

    IDLE, WAIT_TEL, PROBE, RUN, DONE, TIMEOUT, ABORTED, BLIND = (
        "idle", "wait_tel", "probe", "run", "done", "timeout", "aborted", "blind")

    def __init__(self, deg=90.0, left=True, cfg=None, imag_sign=None,
                 probe_s=0.6, probe_dof=0.3, wait_tel_s=3.0, timeout=20.0,
                 blind_rate_dps=None, log=None):
        c, src = turn_cfg()
        if cfg:
            c.update(cfg)
        self.cfg = c
        self.src = src
        self.deg = abs(float(deg))
        self.left = bool(left)
        self.d = -1.0 if self.left else 1.0        # 期望 DOF 符号：左 =-（+yaw=右转）
        self.om = float(c["out_max"])
        self.dz = float(c["deadzone_deg"])
        self.nd = max(0.1, float(c["norm_deg"]))
        self.rate = max(1.0, float(blind_rate_dps if blind_rate_dps is not None
                                   else c.get("blind_rate_dps", 20.0)))
        self.probe_s = float(probe_s)
        self.probe_dof = float(probe_dof)
        self.wait_tel_s = float(wait_tel_s)
        self.timeout = float(timeout)
        self.imag_sign = None if imag_sign in (None, 0, "auto") else float(imag_sign)
        self.log = log or (lambda *a: None)
        self.state = self.IDLE
        self.pid = None
        self.psi0 = None
        self.psi_tgt = None
        self.t_start = None
        self.t_probe = None
        self.t_run = None
        self.y_probe = None
        self.on_target = 0
        self.out = 0.0
        self.max_err = 0.0
        self.done_deg = 0.0        # 实测已转角度（区分 4=几乎没转 / 3=转了一部分）
        self.blind_until = None

    # ------------------------------------------------------------------
    def start(self, now_ms):
        self.state = self.WAIT_TEL if self.imag_sign is None else self.PROBE
        if self.imag_sign is not None:
            self._begin_run(now_ms, yaw_telemetry=None)   # 用第一帧遥测定基准
        self.t_start = now_ms
        return self.state

    def finished(self):
        """终态才返回 True。BLIND 是**进行中**的状态（按时间自己走完 → DONE）。"""
        return self.state in (self.DONE, self.TIMEOUT, self.ABORTED)

    def abort(self):
        """外部要求中止（例如转向中整门丢失）→ 立刻停转。"""
        if not self.finished():
            self.state = self.ABORTED
            self.out = 0.0

    def _begin_run(self, now_ms, yaw_telemetry):
        self.psi0 = wrap180((self.imag_sign or 1.0) * (yaw_telemetry or 0.0))
        self.psi_tgt = wrap180(self.psi0 + self.deg * self.d)
        self.pid = PID(kp=self.cfg["kp"], ki=self.cfg["ki"], kd=self.cfg["kd"],
                       out_min=-self.om, out_max=self.om, deadzone=self.dz / self.nd)
        self.t_run = None
        self.state = self.RUN

    def step(self, now_ms, yaw_telemetry):
        """推进一帧。返回 (state, yaw_cmd)。`yaw_telemetry` 为 None 表示本帧没有遥测。"""
        if self.finished():
            return self.state, 0.0
        if self.state == self.IDLE:
            self.start(now_ms)

        # ---- 等遥测 ----
        if self.state == self.WAIT_TEL:
            if yaw_telemetry is None:
                if now_ms - self.t_start > self.wait_tel_s * 1000.0:
                    self.state = self.ABORTED
                    self.log("[TURN] ⛔ %.1fs 内没拿到遥测 yaw → **拒绝转向**（闭环量缺失）"
                             % self.wait_tel_s)
                return self.state, 0.0
            self.y_probe = yaw_telemetry
            self.t_probe = now_ms
            self.state = self.PROBE
            self.log("[TURN] 探向：给 yaw=%+.2f × %.1fs（判定遥测正方向）"
                     % (self.probe_dof * self.d, self.probe_s))
            return self.PROBE, self.probe_dof * self.d

        # ---- 探向（给一小段固定舵，看遥测 yaw 往哪边动）----
        if self.state == self.PROBE:
            if now_ms - self.t_probe < self.probe_s * 1000.0:
                return self.PROBE, self.probe_dof * self.d
            dy = wrap180((yaw_telemetry if yaw_telemetry is not None
                          else self.y_probe) - self.y_probe)
            if abs(dy) < 0.3:
                self.log("[TURN] ⚠️ 探向 Δyaw=%.2f°（几乎没动）→ 舵效不足或遥测不更新" % dy)
                self.imag_sign = 1.0
            else:
                self.imag_sign = 1.0 if (dy * self.d) > 0 else -1.0
                self.log("[TURN] 探向结果：Δyaw=%+.2f° → 遥测正方向=%s"
                         % (dy, "与 +yaw(右转) 同向" if self.imag_sign > 0 else "与 +yaw(右转) 反向"))
            # 探向也转掉了一点：基准取**探向之前**那个角度，把它计入总量
            self.psi0 = wrap180(self.imag_sign * self.y_probe)
            self.psi_tgt = wrap180(self.psi0 + self.deg * self.d)
            self.pid = PID(kp=self.cfg["kp"], ki=self.cfg["ki"], kd=self.cfg["kd"],
                           out_min=-self.om, out_max=self.om, deadzone=self.dz / self.nd)
            self.state = self.RUN
            return self.RUN, 0.0                      # 本帧先回中性（探向刚结束）

        # ---- 盲转（开环；只有显式要求才会走到这，用"角度÷速率"定时）----
        if self.state == self.BLIND:
            if now_ms < self.blind_until:
                self.done_deg = self.deg * self.d            # 盲转：只能按设定值记账
                return self.BLIND, self.d * self.om
            self.state = self.DONE
            return self.DONE, 0.0

        # ---- 闭环转到目标角 ----
        if self.state == self.RUN:
            if yaw_telemetry is None:                 # 遥测中断：保持中性等，不盲转
                return self.RUN, 0.0
            if self.psi0 is None:
                self.psi0 = wrap180(self.imag_sign * yaw_telemetry)
                self.psi_tgt = wrap180(self.psi0 + self.deg * self.d)
            if self.t_run is None:
                self.t_run = now_ms
            psi = wrap180(self.imag_sign * yaw_telemetry)
            err = wrap180(self.psi_tgt - psi)
            self.max_err = max(self.max_err, abs(err))
            self.done_deg = wrap180(psi - self.psi0) if self.psi0 is not None else 0.0
            self.out = self.pid.update(err / self.nd, int(now_ms))
            if abs(err) <= self.dz:
                self.on_target += 1
                if self.on_target >= 3:
                    self.state = self.DONE
                    done = wrap180(psi - self.psi0) if self.psi0 is not None else 0.0
                    self.log("[TURN] ✅ 到达：err=%+.1f°（目标 %+.1f°）｜实测已转 %+.1f°"
                             % (err, self.deg * self.d, done))
                    return self.DONE, 0.0
            else:
                self.on_target = 0
            if now_ms - self.t_run > self.timeout * 1000.0:
                self.state = self.TIMEOUT
                self.log("[TURN] ❌ 超时 %.0fs：还差 %+.1f°｜实测只转了 %+.1f°"
                         "（本轮最大偏差 %.1f°）→ 舵效不足/卡住/增益过小"
                         % (self.timeout, err, self.done_deg, self.max_err))
                return self.TIMEOUT, 0.0
            return self.RUN, self.out
        return self.state, 0.0

    # ---- 盲转入口（显式才用）----
    def start_blind(self, now_ms):
        self.blind_until = now_ms + (self.deg / self.rate) * 1000.0
        self.state = self.BLIND
        self.log("[TURN] ⚠️ 盲转（开环）：%g° ÷ %.1f°/s = %.1fs @ yaw=%+.2f（落点不可信）"
                 % (self.deg, self.rate, self.deg / self.rate, self.d * self.om))
        return self.state


# ---------------------------------------------------------------------------
def turn(uart, deg=90.0, left=True, out_max=None, timeout=20.0, tol_deg=None,
         kp=None, ki=None, kd=None, deadzone_deg=None, norm_deg=None,
         probe_s=0.6, probe_dof=0.3, wait_tel_s=3.0, imag_sign=None,
         mode="closed", blind_rate_dps=None, period=0.05, log=print,
         now=None, sleep=None):
    """阻塞版：转到额定角度后返回。退出码 0/2/3/4/5（含义见模块头）。"""
    now = now or (lambda: time.time())
    sleep = sleep or time.sleep
    cfg_over = {}
    for k, v in (("out_max", out_max), ("kp", kp), ("ki", ki), ("kd", kd),
                 ("deadzone_deg", deadzone_deg), ("norm_deg", norm_deg)):
        if v is not None:
            cfg_over[k] = float(v)
    core = TurnCore(deg=deg, left=left, cfg=cfg_over, imag_sign=imag_sign,
                    probe_s=probe_s, probe_dof=probe_dof, wait_tel_s=wait_tel_s,
                    timeout=timeout, blind_rate_dps=blind_rate_dps, log=log)
    tel = getattr(uart, "telemetry", None)

    def _yaw():
        return None if tel is None else getattr(tel, "yaw_deg", None)

    t0 = now()
    if mode == "timed":
        core.start_blind(int(t0 * 1000.0))
    log("[TURN] %s ｜ 增益 kp=%.3f kd=%.3f ki=%.3f 限幅 ±%.2f 死区 %.1f° norm_deg=%.1f"
        % (core.src, core.cfg["kp"], core.cfg["kd"], core.cfg["ki"],
           core.cfg["out_max"], core.cfg["deadzone_deg"], core.cfg["norm_deg"]))
    try:
        while not core.finished():
            st, out = core.step(int(now() * 1000.0), _yaw())
            if st == TurnCore.ABORTED:
                break
            uart.send_dof(0.0, 0.0, 0.0, out)
            sleep(period)
            if now() - t0 > timeout + wait_tel_s + probe_s + 5.0:   # 外层兜底，防死循环
                core.abort()
                break
    finally:
        stop_hard(uart, log)
    if core.state == TurnCore.DONE:
        return 2 if mode == "timed" else 0
    if core.state == TurnCore.ABORTED:
        return 5
    # 4 = 几乎没转（舵效/接线/推进器问题）；3 = 转了一部分没到位（增益/时间/被挡）
    return 4 if abs(core.done_deg) < 5.0 else 3


def main():
    ap = argparse.ArgumentParser(description="额定角度原地转（既有 PID + 遥测 yaw 闭环）")
    ap.add_argument("--deg", type=float, default=90.0, help="角度大小（正数）")
    ap.add_argument("--dir", choices=("left", "right"), default="left")
    ap.add_argument("--out-max", type=float, default=None,
                    help="转向限幅 DOF（默认取 comm.motion.turn_pid.out_max）")
    ap.add_argument("--yaw", type=float, default=None, help="同 --out-max（兼容旧脚本写法）")
    ap.add_argument("--kp", type=float, default=None, help="覆盖 PID kp")
    ap.add_argument("--ki", type=float, default=None)
    ap.add_argument("--kd", type=float, default=None)
    ap.add_argument("--deadzone-deg", type=float, default=None,
                    help="PID 死区（度）；也是「到位判据」")
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
            stop_hard(u, print)
        except Exception:
            pass
    finally:
        u.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
