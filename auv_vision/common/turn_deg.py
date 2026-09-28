#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""turn_deg.py — 按给定角度原地转：**下位机遥测 yaw + PID 全闭环**（没有任何开环段）。

闭环量 = 下位机回传的绝对航向 `yaw_deg`（`base/telemetry.py`）。我们不知道"多少秒 = 90°"
（角速度随电量/水流/负载变），所以把"还剩多少度"喂给 PID —— PID 只负责**逐步收敛到目标角**。

**极性（σ：`psi = σ × 遥测 yaw`）是定死的常量，不现场测**，和普通 yaw 环一样：
    σ = 固件约定 × `comm.dof_map.yaw.sign` × `comm.telemetry.yaw_sign`
固件约定 = **船右转时原始 yaw 减小**（一次实测确定，换固件/IMU 才要复核）。
当前 cfg 两个旋钮都是 +1 ⇒ σ = -1。
旧版靠"固定舵探向"定这个符号，已删除（开环探向本身会白转十几度，实测数字见 `doc/注释历史.md`）。

    · **阻塞版** `turn(uart, deg, ...)` —— 给脚本用（自己 sleep 循环，跑完才返回）
    · **逐帧版** `TurnCore`            —— 给主循环用（每帧 step() 一次，不阻塞相机/日志/fps）

参数唯一来源 `comm.motion.turn_pid`（缺键 → 代码兜底）：
    kp 管"多早开始收力/收敛多紧"，out_max 管"最大转速"（想整体更慢就降它，别只降 kp；
    ⚠️ 别降到 0.15 那一档：15% 推力恰卡在执行器死区 0.138 边上 ⇒ "还剩二十几度就没推力"）。
    deadzone_deg 兼「到位判据」，norm_deg 是误差归一化分母（err_norm = 剩余角 / norm_deg）。

退出码（阻塞版 main）：0=到达 ｜ 3=超时未到达 ｜ 4=舵效不足（几乎没转）
                      ｜ 5=**无遥测 → 拒转**（没有闭环量就一根舵都不发；**没有盲转备案**）
                      ｜ 6=**方向自证失败**（命令朝一边、船朝另一边 ⇒ 停转，查极性/接线）
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
from base.turn_log import turn_log                                  # noqa: E402 转弯调用日志

# 固件约定：**船右转时原始 yaw 减小** ⇒ 该常量 = -1（一次实测确定，换固件/IMU 才要复核）。
_FW_RIGHT_YAW_SIGN = -1.0

# 代码内兜底（cfg 缺失时用；正常走 comm.motion.turn_pid）
_D_TURN_PID = dict(
    # ⚠️ kd 必须 0、deadzone_deg 必须 ≥ 执行器死区折算角（见 cfg/comm.yaml 的 turn_pid 注释）：
    #   内层 20Hz 下 kd=0.1 的 D 项就超过 out_max ⇒ 每帧出力正负反转、原地极限环。
    kp=0.2, ki=0.0, kd=0.0, out_max=0.30, deadzone_deg=6.0, norm_deg=15.0,
    # div_deg：**方向自证**（唯一一条兜底，非测量）：命令一直朝目标方向、误差却和初始同号
    #   且比初始还大这么多度 ⇒ 说明船在朝反方向转 ⇒ 停转（不发瞎舵）。0 = 关掉这条兜底。
    div_deg=15.0,
    # 转向闭环推进周期(s)：脚本的 turn(period) 与 gate 转向内层循环共用同一节拍
    period=0.05,
    # sat_max_s：**满舵保护**（用户 2026-09-27 定）——命令饱和到 ±out_max 连续多久 ⇒ **立刻停**。
    #   为什么必须：舵效不对 / 遥测不动 / 被外力顶住时，闭环会一直满舵（现场"一开始就转个不停、
    #   超时保护也不起作用"）；保护不能只指望 turn 自己的超时（那个要求每帧都被 step 到）。
    #   **只满舵还不算**——必须同时"没进展"（这段时间误差改善 < `sat_progress_deg`）才停，
    #   否则慢船的正常长饱和会被误杀（用例 test_motion 实测过）。
    #   0 = 关掉这条保护。实测：健康转向的饱和段通常 < 0.5s。
    sat_max_s=2.0, sat_progress_deg=5.0)

# 方向自证要求的最小命令量（DOF·s；∫out·dt，只算朝目标方向的）：0.3 DOF·s ≈ 1s 满舵。
_DIV_U = 0.3


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
        import base.settings as S
        ch = (S.comm.get("dof_map", None) or {}).get("yaw", None) or {}
        sd = float(ch.get("sign", 1.0) or 1.0)
        st = float(S.get("comm.telemetry.yaw_sign", 1.0) or 1.0)
    except Exception:
        pass
    return _FW_RIGHT_YAW_SIGN * sd * st, "固件%+g × dof_map%+g × telemetry%+g" % (
        _FW_RIGHT_YAW_SIGN, sd, st)


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

    为什么必须（水里实测 + 代码核实）：`base/uart.py::_ramp_step` 对 yaw/surge/sway/heave 做
    **字节级平滑**，`neutral()` 的 `force=True` **只绕过心跳节流、不绕过 ramp** → 单帧 neutral
    发出时轴字节还在半路（yaw 84→128 只到 ~95 = 仍在转）；而**下位机没有无帧超时停车**，
    随后 `close()` 一关串口，船就锁在那个值上一直转。
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

    状态：idle → run → done ｜ timeout ｜ aborted（run 里没有遥测就等着，超时即拒转）
    用法（逐帧）：
        core = TurnCore(deg=25.0, left=True, log=print)
        st, yaw_cmd = core.step(now_ms, yaw_telemetry_deg)   # 遥测为 None → 本帧不发舵
        if core.finished(): ...
    """

    IDLE, RUN, DONE, TIMEOUT, ABORTED = "idle", "run", "done", "timeout", "aborted"

    def __init__(self, deg=90.0, left=True, cfg=None, imag_sign=None,
                 wait_tel_s=3.0, timeout=20.0, log=None, tag="turn"):
        c, src = turn_cfg()
        if cfg:
            c.update(cfg)
        self.cfg = c
        self.src = src
        self.deg = abs(float(deg))
        self.left = bool(left)
        self.d = -1.0 if self.left else 1.0     # 命令方向：左=-（+yaw = 右转，项目约定）
        self.om = float(c["out_max"])
        self.dz = float(c["deadzone_deg"])
        self.nd = max(0.1, float(c["norm_deg"]))
        self.sat_max_s = float(c.get("sat_max_s", _D_TURN_PID["sat_max_s"]) or 0.0)
        self.sat_progress_deg = float(c.get("sat_progress_deg",
                                            _D_TURN_PID["sat_progress_deg"]) or 0.0)
        self._sat_ms = 0.0            # 连续满舵累计时长(ms)
        self._sat_err0 = None         # 这段饱和开始时的 |误差|(°)（判有没有进展）
        self.div_deg = float(c.get("div_deg", 15.0) or 0.0)      # 0 = 关掉方向自证
        self.wait_tel_s = float(wait_tel_s)
        self.timeout = float(timeout)
        # σ：定死的乘式；imag_sign 仅供用例注入（复现/验证方向自证）
        sigma, ssrc = yaw_sign()
        self.imag_sign = float(imag_sign) if imag_sign not in (None, 0) else sigma
        self.sigma_src = ssrc if imag_sign in (None, 0) else "用例注入"
        self.log = log or (lambda *a: None)
        self.tag = tag                       # 调用方标记（cli / gate …），只进 turn 日志
        self.state = self.IDLE
        self.pid = None
        self.yaw0 = None
        self.psi0 = None
        self.psi_tgt = None
        self.err0 = None
        self.t_start = None
        self.t_run = None
        self._t_last = None
        self._u_d = 0.0              # 朝目标方向已施加的命令量（DOF·s，方向自证用）
        self.on_target = 0
        self.out = 0.0
        self.max_err = 0.0
        self.done_deg = 0.0          # 已按目标方向转过的角度（正 = 朝目标方向）
        self.why = ""                # ABORTED 的原因：no_tel | div | external

    # ------------------------------------------------------------------
    def start(self, now_ms):
        self.t_start = now_ms
        self.state = self.RUN
        turn_log("enter", src=self.tag, deg=self.deg, left=self.left, d=self.d,
                 sig=self.imag_sign, sig_src=self.sigma_src, kp=self.cfg["kp"],
                 kd=self.cfg["kd"], out_max=self.om, dz=self.dz, norm=self.nd,
                 timeout=self.timeout)
        self.log("[TURN] %s ｜ 闭环 %s %.1f° ｜ 极性 σ=%+.0f（%s）｜ 增益 kp=%.3f kd=%.3f ki=%.3f"
                 " ｜ 限幅 ±%.2f ｜ 死区 %.1f° ｜ norm_deg=%.1f"
                 % (self.src, "左转" if self.left else "右转", self.deg,
                    self.imag_sign, self.sigma_src,
                    self.cfg["kp"], self.cfg["kd"], self.cfg["ki"],
                    self.om, self.dz, self.nd))
        return self.state

    def finished(self):
        return self.state in (self.DONE, self.TIMEOUT, self.ABORTED)

    def abort(self, why="external"):
        """外部要求中止（例如转向中整门丢失）→ 立刻停转。"""
        if not self.finished():
            self.state = self.ABORTED
            self.why = why
            self.out = 0.0
            turn_log("exit", src=self.tag, state="aborted", why=why,
                     done_deg=self.done_deg)

    def _make_pid(self):
        """转向 PID（out 限幅 = ±out_max；deadzone 用角度阈值归一化）。"""
        return PID(kp=self.cfg["kp"], ki=self.cfg["ki"], kd=self.cfg["kd"],
                   out_min=-self.om, out_max=self.om, deadzone=self.dz / self.nd)

    def _anchor(self, now_ms, yaw0):
        self.yaw0 = float(yaw0)
        self.psi0 = wrap180(self.imag_sign * self.yaw0)
        self.psi_tgt = wrap180(self.psi0 + self.deg * self.d)
        self.err0 = self.deg
        self.pid = self._make_pid()
        self.t_run = now_ms
        self._t_last = now_ms
        self._u_d = 0.0
        turn_log("anchor", src=self.tag, yaw0=self.yaw0, psi_tgt=self.psi_tgt,
                 deg=self.deg, left=self.left, sig=self.imag_sign)
        self.log("[TURN] 锚点：遥测 yaw0=%+.2f° ⇒ 目标 psi=%+.1f°（%s %.1f°）"
                 % (self.yaw0, self.psi_tgt, "左转" if self.left else "右转", self.deg))

    def step(self, now_ms, yaw_telemetry):
        """推进一帧。返回 (state, yaw_cmd)。`yaw_telemetry` 为 None 表示本帧没有遥测。"""
        if self.finished():
            return self.state, 0.0
        if self.state == self.IDLE:
            self.start(now_ms)
        if self.state != self.RUN:
            return self.state, 0.0

        # ---- 等遥测：保持中性，**一根舵都不发**（没有闭环量就不许转）----
        if yaw_telemetry is None:
            if now_ms - self.t_start > self.wait_tel_s * 1000.0:
                self.abort("no_tel")
                self.log("[TURN] ⛔ %.1fs 内没拿到遥测 yaw → **拒绝转向**（闭环量缺失）"
                         % self.wait_tel_s)
            return self.state, 0.0
        yaw = float(yaw_telemetry)
        if self.yaw0 is None:
            self._anchor(now_ms, yaw)

        dt = max(0.001, (now_ms - (self._t_last if self._t_last is not None else now_ms)) / 1000.0)
        self._t_last = now_ms
        err = wrap180(self.psi_tgt - wrap180(self.imag_sign * yaw))
        self.out = self.pid.update(err / self.nd, int(now_ms))
        self.max_err = max(self.max_err, abs(err))
        self.done_deg = wrap180(self.imag_sign * self.d * wrap180(yaw - self.yaw0))

        # ★ 满舵保护（用户 2026-09-27 定）：命令饱和到 ±out_max 持续 `sat_max_s` ⇒ **立刻停**。
        #   加在"算完 out"之后、"到位/超时"判断之前 ⇒ 它比超时更早生效（现场要的就是这个）。
        if self.sat_max_s > 0.0 and abs(self.out) >= 0.9 * self.om:
            if self._sat_ms <= 0.0:
                self._sat_err0 = abs(err)          # 进入饱和段：记起点误差
            self._sat_ms += dt * 1000.0
            if self._sat_ms >= self.sat_max_s * 1000.0:
                prog = (self._sat_err0 or abs(err)) - abs(err)
                if prog < self.sat_progress_deg:
                    _out_abs = abs(self.out)          # 先记下来（abort() 会把 out 清零）
                    self.abort("sat")
                    self.log("[TURN] ⛔ 满舵 %.1fs 且**没进展**（误差只改善 %.1f° < %.1f°；"
                             "|out|=%.2f，还差 %+.1f°）⇒ **立刻停转**（舵效/遥测/接线？）"
                             % (self._sat_ms / 1000.0, prog, self.sat_progress_deg,
                                _out_abs, err))
                    return self.state, 0.0
                self._sat_ms = 0.0                 # 有进展 ⇒ 重新计一段（慢船不被误杀）
                self._sat_err0 = abs(err)
        else:
            self._sat_ms = 0.0
            self._sat_err0 = None

        # ---- 方向自证（不是"测符号"：σ 已定死。只回答"船有没有按命令的方向转"）----
        self._u_d += max(0.0, self.out * self.d) * dt
        if (self.div_deg > 0 and self._u_d >= _DIV_U and self.done_deg <= -self.div_deg):
            self.abort("div")
            self.log("[TURN] ⛔ 方向自证失败：已朝%s施加 %.2f DOF·s，遥测却显示船朝**反方向**"
                     "转了 %.1f° ⇒ 停转。查：dof_map.yaw.sign / comm.telemetry.yaw_sign /"
                     " 推进器接线（σ=%+.0f，%s）"
                     % ("左" if self.left else "右", self._u_d, self.done_deg,
                        self.imag_sign, self.sigma_src))
            return self.state, 0.0

        if abs(err) <= self.dz:
            self.on_target += 1
            if self.on_target >= 3:
                self.state = self.DONE
                turn_log("exit", src=self.tag, state="done", done_deg=self.done_deg,
                         max_err=self.max_err, ms=now_ms - (self.t_run or now_ms))
                self.log("[TURN] ✅ 到达：err=%+.1f°（目标 %+.1f°）｜实测已转 %+.1f°（最大偏差 %.1f°）"
                         % (err, self.deg * self.d, self.done_deg, self.max_err))
                return self.DONE, 0.0
        else:
            self.on_target = 0

        if now_ms - self.t_run > self.timeout * 1000.0:
            self.state = self.TIMEOUT
            turn_log("exit", src=self.tag, state="timeout", done_deg=self.done_deg,
                     max_err=self.max_err, ms=now_ms - (self.t_run or now_ms))
            self.log("[TURN] ❌ 超时 %.0fs：还差 %+.1f°｜实测只转了 %+.1f°（目标 %+.1f°；"
                     "最大偏差 %.1f°）→ 舵效不足/卡住/增益过小"
                     % (self.timeout, err, self.done_deg, self.deg * self.d, self.max_err))
            return self.TIMEOUT, 0.0
        return self.RUN, self.out


# ---------------------------------------------------------------------------
def turn(uart, deg=90.0, left=True, out_max=None, timeout=20.0, tol_deg=None,
         kp=None, ki=None, kd=None, deadzone_deg=None, norm_deg=None, div_deg=None,
         wait_tel_s=3.0, imag_sign=None, period=0.05, log=print,
         now=None, sleep=None, sat_max_s=None, sat_progress_deg=None):
    """阻塞版：转到额定角度后返回。退出码 0/3/4/5/6（含义见模块头）。"""
    now = now or (lambda: time.time())
    sleep = sleep or time.sleep
    cfg_over = {}
    for k, v in (("out_max", out_max), ("kp", kp), ("ki", ki), ("kd", kd),
                 ("deadzone_deg", deadzone_deg), ("norm_deg", norm_deg),
                 ("div_deg", div_deg), ("sat_max_s", sat_max_s),
                 ("sat_progress_deg", sat_progress_deg)):
        if v is not None:
            cfg_over[k] = float(v)
    core = TurnCore(deg=deg, left=left, cfg=cfg_over, imag_sign=imag_sign,
                    wait_tel_s=wait_tel_s, timeout=timeout, log=log, tag="cli")

    def _yaw():
        tel = getattr(uart, "telemetry", None)
        return None if tel is None else getattr(tel, "yaw_deg", None)

    turn_log("call", src="cli", deg=deg, left=left, timeout=timeout, period=period)
    t0 = now()
    try:
        while not core.finished():
            st, out = core.step(int(now() * 1000.0), _yaw())
            if st == TurnCore.ABORTED:
                break
            uart.send_dof(0.0, 0.0, 0.0, out)
            sleep(period)
            if now() - t0 > timeout + wait_tel_s + 5.0:   # 外层兜底，防死循环
                core.abort("loop_guard")
                break
    finally:
        stop_hard(uart, log)
    turn_log("return", src="cli", state=core.state, why=core.why,
             done_deg=core.done_deg, ms=(now() - t0) * 1000.0)
    if core.state == TurnCore.DONE:
        return 0
    if core.state == TurnCore.ABORTED:
        return 6 if core.why == "div" else 5
    # 4 = 几乎没转（舵效/接线/推进器问题）；3 = 转了一部分没到位（增益/时间/被挡）
    return 4 if abs(core.done_deg) < 5.0 else 3


def main():
    ap = argparse.ArgumentParser(description="额定角度原地转（遥测 yaw + PID 全闭环）")
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
    ap.add_argument("--imag-sign", type=float, default=None,
                    help="只用来自检：强制 σ（默认不传 ⇒ 用定死的乘式）")
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
                  imag_sign=args.imag_sign)
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
