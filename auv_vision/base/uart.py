# -*- coding: utf-8 -*-
"""uart.py — 串口 v2：11B 帧(0xA5+7轴+3键)；上层 DOF 目标→DOF_MAP 真值表→轴字节(速度平滑)，
模拟遥控输出；心跳 ≥20Hz；急停=连发中性帧。帧语义见 README 与 docs/CUP_AUV。

上行（下位机 → 上位机）：`base/telemetry.py` 的 14B 遥测帧（0xAA55 + 深度/姿态 + 校验和），
每次发帧时顺带读空接收缓冲 → `self.telemetry.depth_m`；`comm.depth_guard` 据此**禁止上浮**
（深度 ≤ min_depth_m 时把 heave 清零、heave 轴立刻回中），保证机身不冒出水面。
"""
import os
import sys
import time

import base.settings as S
import base.telemetry as TEL

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


def _now_ms():
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# 运动状态描述（把 11B 帧翻译成人类可读的运动）
# ---------------------------------------------------------------------------
_CHAN_NAMES = [  # (通道名, 正向标签, 反向标签)
    ("surge", "前进", "后退"),
    ("sway", "右移", "左移"),
    ("heave", "上浮", "下潜"),
    ("yaw", "右转", "左转"),
]


def describe_motion(frame=None, axes=None):
    """由帧(或7轴字节)生成[运动状态]，如：前进80%, 左移35%；全中位=停止。"""
    if frame is None and axes is None:
        return "停止"
    if frame is not None:
        axes = list(frame)[1:1 + S.comm.frame.axis_count]
    parts = []
    for name, pos_lab, neg_lab in _CHAN_NAMES:
        ch = S.comm.dof_map.get(name)
        if ch is None:
            continue
        axis = ch["axis"]
        sign = ch.get("sign", 1)
        v = axes[axis] - S.comm.frame.axis_mid
        d = v * sign                      # 通道正向位移
        if abs(d) <= 3:
            continue
        lab = pos_lab if d > 0 else neg_lab
        parts.append("%s%d%%" % (lab, abs(d) / S.comm.frame.axis_range * 100))
    return "停止" if not parts else ", ".join(parts)


# ---------------------------------------------------------------------------
# 帧构建（纯函数，测试可直接使用）
# ---------------------------------------------------------------------------
def _axis_of(name, default):
    """DOF 通道名 → 轴字节下标（缺映射时用 default）。"""
    ch = S.comm.dof_map.get(name)
    return int(ch["axis"]) if ch else int(default)


def dof_to_axis_bytes(surge=0.0, sway=0.0, heave=0.0, yaw=0.0,
                      aux=None, btns=None):
    """DOF 目标(归一化[-1,1]) → 7 轴字节。附加字节按 config 安全默认。"""
    axes = [S.comm.frame.axis_mid] * S.comm.frame.axis_count
    aux = dict(S.comm.frame.aux_axis) if aux is None else dict(aux)
    for i, v in aux.items():
        axes[i] = int(max(0, min(255, v)))
    mov = {"surge": surge, "sway": sway, "heave": heave, "yaw": yaw}
    for name, val in mov.items():
        ch = S.comm.dof_map.get(name)
        if ch is None:
            if val and S.DEBUG:
                print("[UART] 通道 %s 未映射，忽略输出" % name)
            continue
        sign = ch.get("sign", 1)
        axes[ch["axis"]] = int(max(0, min(255, round(
            S.comm.frame.axis_mid + sign * val * S.comm.frame.axis_range))))
    return axes


def neutral_axis_bytes():
    """无动作中性帧的 7 轴字节（运动轴中位 + 附加默认）。"""
    axes = [S.comm.frame.axis_mid] * S.comm.frame.axis_count
    for i, v in S.comm.frame.aux_axis.items():
        axes[i] = int(v)
    return axes


def build_frame_from_dof(surge=0.0, sway=0.0, heave=0.0, yaw=0.0):
    axes = dof_to_axis_bytes(surge, sway, heave, yaw)
    return bytes([S.comm.frame.header] + axes + list(S.comm.frame.btn_values))


def build_neutral_frame():
    return bytes([S.comm.frame.header] + neutral_axis_bytes() +
                 list(S.comm.frame.btn_values))


# ---------------------------------------------------------------------------
# 控制器
# ---------------------------------------------------------------------------
class UartController(object):
    def __init__(self, dev=None, baud=None, sim=None):
        self.dev = dev if dev is not None else S.comm.serial.device
        self.baud = baud if baud is not None else S.comm.serial.baud
        self.sim = sim if sim is not None else S.SIM_MODE
        self._ser = None
        self._estop = False
        self._axes = neutral_axis_bytes()        # 当前输出轴（平滑后）
        self._dof_target = (0.0, 0.0, 0.0, 0.0)  # 期望 DOF（上层原始请求）
        self._dof_out = (0.0, 0.0, 0.0, 0.0)     # 实际下发 DOF（限深 + 下潜补偿后）
        self._last_update_ms = _now_ms()   # 首帧平滑从当前时刻起算
        self.last_send_ms = 0
        self.last_motion = "stop"
        self._last_tx_log = 0
        # ---- 下位机遥测（深度等）与限深保护状态 ----
        self.telemetry = TEL.TelemetryReceiver()
        self._last_tel_log_ms = 0
        self._last_guard_log_ms = 0
        self._warned_no_tel = False
        self.guard_active = False      # 最近一帧是否因限深压掉了上浮（供叠加/日志）
        self.guard_blocks = 0          # 累计被限深压掉的上浮帧数
        self._dof_log_f = None
        self._last_dof_t = time.monotonic()
        log_path = os.environ.get("AUV_DOF_LOG")
        if log_path:
            try:
                self._dof_log_f = open(log_path, "w", encoding="utf-8")
                self._dof_log_f.write("dt_s,a0,a1,a2,a3\n")
                print("[UART] 轨迹记录 → %s" % log_path)
            except OSError as e:
                print("[UART] 轨迹记录失败：%s" % e)
        if not self.sim:
            if not HAS_SERIAL:
                print("[UART] 未安装 pyserial，退回 SIM 模式")
                self.sim = True
                return
            try:
                self._ser = serial.Serial(self.dev, self.baud,
                                          timeout=0.05, write_timeout=0.1)
                print("[UART] 打开 %s @ %d" % (self.dev, self.baud))
            except Exception as e:
                print("[UART] 打开失败：%s；退回 SIM 模式" % e)
                self.sim = True

    # ---------------- 帧/发送 ----------------
    def _current_frame(self):
        return bytes([S.comm.frame.header] + self._axes +
                     list(S.comm.frame.btn_values))

    def _ramp_step(self):
        """轴字节级平滑（模拟摇杆手感）：

        按 `ramp.speed_per_s`(字节/秒) 让输出连续逼近目标，避免速度直接激增；
        `<=0` 直通。

        限深保护在平滑**之前**生效（`_apply_depth_guard`）：深度不足时上浮分量被清零，
        且 heave 轴**当帧直接回中**（`depth_guard.snap`，不等 ramp）——否则平滑惯性还会
        继续往上浮，等不到下一帧就已经冒出水面了。
        """
        now = _now_ms()
        dt = max(0.0, (now - self._last_update_ms) / 1000.0)
        self._last_update_ms = now
        self._dof_out = self._apply_depth_guard(*self._dof_target, now_ms=now)
        # 下潜动力放大（底层共用，gate/ball 都吃）：放在限深之后——限深只压上浮(heave>0)，
        # 本项只放大下潜(heave<0)，两者互不干涉。
        self._dof_out = self._apply_dive_boost(*self._dof_out)
        tgt = dof_to_axis_bytes(*self._dof_out)
        step_max = (256.0 if S.comm.ramp.speed_per_s <= 0
                    else S.comm.ramp.speed_per_s * dt)
        h_axis = _axis_of("heave", 2)
        snap = self.guard_active and bool(S.get("comm.depth_guard.snap", True))
        for i in range(4):                    # 仅平滑双摇杆区
            if snap and i == h_axis:
                self._axes[i] = tgt[i]        # 限深触发：heave 立刻回中
                continue
            diff = tgt[i] - self._axes[i]
            if abs(diff) <= step_max:
                self._axes[i] = tgt[i]
            else:
                self._axes[i] = int(round(
                    self._axes[i] + (step_max if diff > 0 else -step_max)))
        for i in (4, 5, 6):                   # 附加字节直接取目标
            self._axes[i] = tgt[i]

    # ---------------- 接收：下位机遥测（深度等） ----------------
    def _drain_rx(self):
        """读空串口接收缓冲 → 遥测解析（深度/姿态）。无串口、无数据 = 空操作。

        发送路径每帧都调它（见 `_write`），所以遥测的实时性跟着心跳（≥20Hz）。
        """
        if self.sim or self._ser is None:
            return 0
        try:
            n = self._ser.in_waiting
        except Exception:
            return 0
        if not n:
            return 0
        try:
            data = self._ser.read(n)
        except Exception as e:
            print("[UART] 遥测读取失败：%s" % e)
            return 0
        return self.feed_telemetry(data)

    def feed_telemetry(self, data, now_ms=None):
        """喂入下位机上行字节（14B 遥测帧，可任意切分）→ 更新深度等。

        真串口由 `_drain_rx` 调用；测试/台架模拟下位机可直接调它。返回有效帧数。
        """
        got = self.telemetry.feed(data, now_ms=now_ms)
        if got:
            self._log_telemetry(now_ms)
        return got

    def _log_telemetry(self, now_ms=None):
        """打印遥测（节流：`comm.debug.tel_log_ms`，0=每帧都打）。"""
        if not S.DEBUG:
            return
        now = now_ms or _now_ms()
        interval = float(S.get("comm.debug.tel_log_ms", 200) or 0)
        if interval > 0 and now - self._last_tel_log_ms < interval:
            return
        self._last_tel_log_ms = now
        t = self.telemetry
        print("[UART←] depth=%.2fm target=%.2fm roll=%.2f pitch=%.2f yaw=%.2f"
              % (t.depth_m, t.target_m or 0.0, t.roll_deg or 0.0,
                 t.pitch_deg or 0.0, t.yaw_deg or 0.0))

    # ---------------- 限深保护（不得浮出水面） ----------------
    @property
    def depth_m(self):
        """下位机回传的当前深度(m)；None = 还没收到有效遥测。"""
        return self.telemetry.depth_m

    def depth_fresh(self, now_ms=None):
        """深度数据是否新鲜（`comm.depth_guard.stale_ms` 内有更新）。"""
        stale = float(S.get("comm.depth_guard.stale_ms", 500) or 0)
        return self.telemetry.fresh(stale, now_ms)

    def _apply_dive_boost(self, surge, sway, heave, yaw):
        """**下潜动力单独放大**（底层共用机制，撞球/过门都吃）。

        只处理 `heave < 0`（下潜）：乘上 `dive_scale` 后夹到 [-1, 0]；上浮/悬停/平移/转向
        一律原样 —— 所以它是"单独调下潜这一路"的旋钮，不影响其它通道。

        背景（为什么要单独放大）：DOF→字节→下位机 `RC_Matching*`（映射值 ≤35 直接返回 0）
        使实际推力远小于 DOF 数字（0.20→20%、0.30→30%、0.60→60%，见 gate 文档 §1.1）；
        下潜偏弱时就把这一路放大，比在任务层加补偿/脉冲简单得多。

        注：很小的下潜指令（如 −0.05）即使放大也仍可能落在死区(<0.138)内 —— 要连很小的
        指令也变成有效推力，`dive_scale` 得给足（≈3~4）。
        """
        c = S.get("comm.dof_comp", None) or {}
        if not bool(c.get("enable", True)) or heave >= 0.0:
            return (surge, sway, heave, yaw)
        k = abs(float(c.get("dive_scale", 1.0) or 1.0))
        if k <= 1.0:
            return (surge, sway, heave, yaw)
        return (surge, sway, max(-1.0, heave * k), yaw)

    def _apply_depth_guard(self, surge, sway, heave, yaw, now_ms=None):
        """上浮(heave>0)限深：当前深度 ≤ `min_depth_m` → 本帧禁止上浮。

        只改 heave，**不动 surge/sway/yaw**（别让保护破坏对准/前进）；
        `heave<=0`（下潜/悬停）与深度充足时原样放行。
        遥测缺失/超时：按 `depth_guard.stale_action` 处理——`pass`(默认)放行，
        `block_up` 连"盲上浮"也不许（要求下位机持续回传；SIM/无串口始终放行，免得台架被锁死）。
        """
        self.guard_active = False
        if heave <= 0.0 or not bool(S.get("comm.depth_guard.enable", True)):
            return (surge, sway, heave, yaw)
        if not self.depth_fresh(now_ms):
            if not self._warned_no_tel:
                self._warned_no_tel = True
                if not self.sim:
                    print("[UART] ⚠️ 限深保护已开但没有新鲜深度遥测(0xAA55…)→按 "
                          "depth_guard.stale_action=%s 处理，确认下位机在上行遥测"
                          % S.get("comm.depth_guard.stale_action", "pass"))
            if self.sim or str(S.get("comm.depth_guard.stale_action",
                                     "pass")).lower() != "block_up":
                return (surge, sway, heave, yaw)
            self.guard_active = True                # 断链也不盲上浮（保守档）
            self.guard_blocks += 1
            self._log_guard(heave, now_ms, why="无新鲜深度遥测")
            return (surge, sway, 0.0, yaw)
        limit = float(S.get("comm.depth_guard.min_depth_m", 0.3) or 0.0)
        if self.telemetry.depth_m > limit:
            return (surge, sway, heave, yaw)
        self.guard_active = True
        self.guard_blocks += 1
        self._log_guard(heave, now_ms,
                        why="depth=%.2fm ≤ %.2fm" % (self.telemetry.depth_m, limit))
        return (surge, sway, 0.0, yaw)          # 上浮清零，其余照旧

    def _log_guard(self, heave, now_ms=None, why=""):
        """限深触发告警（节流：`comm.depth_guard.log_ms`，0=每次都打）。"""
        now = now_ms or _now_ms()
        interval = float(S.get("comm.depth_guard.log_ms", 1000) or 0)
        if interval > 0 and now - self._last_guard_log_ms < interval:
            return
        self._last_guard_log_ms = now
        print("[UART] 限深保护：%s → 禁止上浮(heave %.2f→0)" % (why, heave))

    def _log_tx(self, frame, now_ms=None):
        """终端打印发出的运动帧：发一次打一次；帧后附[运动状态]。

        sim 沿用全局 DEBUG；真串口按 comm.debug.tx_frame_hex，
        若配置了 tx_frame_log_ms>0 则按该毫秒数节流。
        """
        if self.sim and not S.DEBUG:
            return
        if not self.sim and not S.get("comm.debug.tx_frame_hex", True):
            return
        now = now_ms or _now_ms()
        interval = S.get("comm.debug.tx_frame_log_ms", 0)
        if not self.sim and interval and now - self._last_tx_log < interval:
            return
        self._last_tx_log = now
        extra = ""
        if self.guard_active:
            extra = "  [限深保护 depth=%.2fm]" % (self.telemetry.depth_m or 0.0)
        print("[UART→] %s  [运动状态] %s%s"
              % (frame.hex(), describe_motion(frame), extra))

    def _log_dof_row(self, frame, now_ms=None):
        """把每帧 4 个运动轴写入轨迹日志（仅设了环境变量 AUV_DOF_LOG 时）。"""
        if self._dof_log_f is None:
            return
        t = time.monotonic()
        dt = t - self._last_dof_t
        self._last_dof_t = t
        axes = list(frame)[1:1 + S.comm.frame.axis_count]
        self._dof_log_f.write("%.4f,%d,%d,%d,%d\n"
                              % (dt, axes[0], axes[1], axes[2], axes[3]))
        self._dof_log_f.flush()

    def _write(self, frame, force=False):
        now = _now_ms()
        if not force and (now - self.last_send_ms) < S.comm.heartbeat.interval_ms:
            return False
        self.last_send_ms = now
        self._log_tx(frame, now)
        self._log_dof_row(frame, now)
        if self.sim:
            return True
        try:
            self._ser.write(frame)
            self._drain_rx()
            return True
        except Exception as e:
            print("[UART] 发送失败：%s" % e)
            return False

    def send_frame_bytes(self, frame, force=False):
        """原样发送一帧字节（跳过平滑，可绕过心跳节流）。"""
        if self._estop and frame != build_neutral_frame():
            return False
        return self._write(frame, force=force)

    def send_motion(self, name=None, force=False):
        """按当前平滑后的轴输出一帧（心跳节流）。

        发帧**前**先收一次遥测：限深保护要用最新深度判定（发完再收就慢一帧）。
        """
        if self._estop:
            return False
        if name is not None:
            self.last_motion = name
        self._drain_rx()
        self._ramp_step()
        return self._write(self._current_frame(), force=force)

    def set_motion(self, name, force=False):
        """动作预设 → DOF 目标 → 持续平滑输出（每帧调用即可）。"""
        if name not in S.comm.motion.presets:
            print("[UART] 未知运动预设：%s" % name)
            name = "stop"
        self._dof_target = S.comm.motion.presets[name]
        self.last_motion = name
        return self.send_motion(name=name, force=force)

    def send_dof(self, surge=0.0, sway=0.0, heave=0.0, yaw=0.0, force=False):
        """直接给定 DOF 目标（供测试/上层精细控制）。"""
        self._dof_target = (surge, sway, heave, yaw)
        self.last_motion = "dof"
        return self.send_motion(name="dof", force=force)

    def neutral(self):
        """回中性并强发一帧。"""
        self._dof_target = (0.0, 0.0, 0.0, 0.0)
        self.last_motion = "stop"
        return self.send_motion(name="stop", force=True)

    # ---------------- 安全 ----------------
    def estop(self):
        if self._estop:
            return
        self._estop = True
        print("[UART] E-STOP 触发，连发中性帧")
        for _ in range(5):
            self._write(build_neutral_frame(), force=True)
            time.sleep(S.comm.estop.repeat_ms / 1000.0)

    @property
    def estop_active(self):
        return self._estop

    @property
    def dof_target(self):
        return tuple(self._dof_target)   # 供测试/日志读取当前 DOF 目标

    @property
    def dof_out(self):
        """限深保护**之后**实际下发的 DOF（对照 dof_target 看被压掉了多少）。"""
        return tuple(self._dof_out)

    # ---------------- 收尾 ----------------
    def close(self):
        """关闭 DOF 轨迹日志与串口（可重复调用）。"""
        if self._dof_log_f is not None:
            try:
                self._dof_log_f.close()
            except Exception:
                pass
            self._dof_log_f = None
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None


def install_signal_handlers(uart):
    """Ctrl-C / SIGTERM → estop（板上建议再加 GPIO 急停输入）。"""
    import signal

    def _handler(sig, _f):
        print("[MAIN] 收到信号 %s" % sig)
        try:
            uart.estop()
        finally:
            sys.exit(130 if sig == signal.SIGINT else 143)

    signal.signal(signal.SIGINT, _handler)
    try:
        signal.signal(signal.SIGTERM, _handler)
    except Exception:
        pass