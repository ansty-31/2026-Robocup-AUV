# -*- coding: utf-8 -*-
"""uart.py — 串口 v2：11B 帧(0xA5+7轴+3键)；上层 DOF 目标→DOF_MAP 真值表→轴字节(速度平滑)，
（详细用法、判据与实测见 doc/注释历史.md）"""
import math
import os
import sys
import threading
import time

import base.cfg.settings as S
import base.hw.telemetry as TEL

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# 限深保护的兜底阈值（m）：**必须与 cfg/comm.yaml 的 `depth_guard.min_depth_m` 同值**。
#   2026-10-06 随 cfg 从 0.55 下调到 0.50（用户要求夹取翘头工作深度能到 0.5）。
_D_MIN_DEPTH_M = 0.50

# pitch/roll 姿态限幅的兜底（度）：与 `comm.motion.axis.max_tilt_deg` / common/motion/turn_deg 同源同值。
#   （base 层不 import common ⇒ 这里直接按 cfg 路径读，**只有一个数**，不各写一份）
_D_MAX_TILT_DEG = 50.0

# 指定角度轴（协议 byte[7]）：0=不发起新任务, 1=yaw, 2=pitch, 3=roll。
# 帧布局 / 编码 / 执行帧语义见 doc/记录/README_COMMUNICATION.md §2、§2.1、§2.3。
# `comm.frame.turn_axis` 可覆盖这张表；删掉该键也不崩（用这里的同值兜底）。
_D_TURN_AXIS = {"yaw": 1, "pitch": 2, "roll": 3}


def turn_axis_id(axis=1):
    """轴名（`yaw`/`pitch`/`roll`）或编号 → byte[7]（1/2/3）。非法值抛 ValueError。"""
    table = dict(_D_TURN_AXIS)
    try:
        for k, v in (S.get("comm.frame.turn_axis", None) or {}).items():
            table[str(k)] = int(v)
    except Exception:
        pass
    if isinstance(axis, str):
        key = axis.strip().lower()
        if key not in table:
            raise ValueError("未知指定角度轴：%s（可选 %s）"
                             % (axis, "/".join(sorted(table))))
        a = int(table[key])
    else:
        a = int(axis)
    if a not in (1, 2, 3):
        raise ValueError("指定角度轴只能是 1=yaw / 2=pitch / 3=roll，收到 %r" % (axis,))
    return a


def _now_ms():
    return int(time.time() * 1000)


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


def _wrap180(deg):
    """角度差归一化到 (-180, 180]（与 task1_2/turn_deg.py 的 wrap180 同语义）。"""
    x = math.fmod(float(deg) + 180.0, 360.0)
    if x <= 0:
        x += 360.0
    return x - 180.0


def neutral_axis_bytes():
    """无动作中性帧的 7 轴字节（运动轴中位 + 附加默认）。"""
    axes = [S.comm.frame.axis_mid] * S.comm.frame.axis_count
    for i, v in S.comm.frame.aux_axis.items():
        axes[i] = int(v)
    return axes


def build_frame_from_dof(surge=0.0, sway=0.0, heave=0.0, yaw=0.0):
    axes = dof_to_axis_bytes(surge, sway, heave, yaw)
    return bytes([S.comm.frame.header] + axes + [0, 0, S.comm.frame.btn_values[2]])


def encode_turn_angle(angle_deg):
    """相对转角：负=左、正=右；[-180,180] → [0,32767]，四舍五入。"""
    angle = float(angle_deg)
    if not math.isfinite(angle) or not -180.0 <= angle <= 180.0:
        raise ValueError("相对转角必须在 [-180, 180] 度内")
    return int(math.floor((angle + 180.0) * 32767.0 / 360.0 + 0.5))


def build_turn_frame(angle_deg, turn_id=1, axis=1):
    """带编号的可重发执行帧；四个手动运动轴回中。

    `axis` 显式写入 byte[7]（1=yaw / 2=pitch / 3=roll）。协议要求**执行帧自己覆盖该字节**
    （doc/记录/README_COMMUNICATION.md §2、§6-1），不能只靠 `frame.aux_axis` 的默认值 ——
    否则把中性帧的 byte[7] 改成 0 就会静默变成"不发起任务"。默认 1(yaw) 与既有 yaw 路径
    **逐字节一致**（`aux_axis[6]` 现为 1）。
    """
    if not 1 <= turn_id <= 255:
        raise ValueError("旋转编号必须在 1～255")
    a = turn_axis_id(axis)
    code = encode_turn_angle(angle_deg)
    frame = bytearray(build_neutral_frame())
    frame[7] = a
    frame[8] = 0x80 | ((code >> 8) & 0x7f)
    frame[9] = code & 0xff
    frame[10] = turn_id
    return bytes(frame)


def build_neutral_frame():
    return bytes([S.comm.frame.header] + neutral_axis_bytes() +
                 [0, 0, S.comm.frame.btn_values[2]])


def build_turn_cancel_frame(turn_id=0):
    """执行位0、角度低字节1：取消指定编号；编号0为同步并取消。"""
    frame = bytearray(build_neutral_frame())
    frame[9], frame[10] = 1, turn_id
    return bytes(frame)


class UartController(object):
    def __init__(self, dev=None, baud=None, sim=None):
        self.dev = dev if dev is not None else S.comm.serial.device
        self.baud = baud if baud is not None else S.comm.serial.baud
        self.sim = sim if sim is not None else S.SIM_MODE
        self._ser = None
        self._estop = False
        self._turn_frame = None
        self._cancel_turn_id = None
        self._turn_phase = None
        self._turn_counter = 0
        self._turn_id = None
        self._turn_axis = 1                      # 本次指定角度任务的轴（1=yaw/2=pitch/3=roll）
        self._turn_retry_at = 0.0
        self._sync_frames = 0
        self._sync_sent = False
        self._axes = neutral_axis_bytes()        # 当前输出轴（平滑后）
        self._dof_target = (0.0, 0.0, 0.0, 0.0)  # 期望 DOF（上层原始请求）
        self._dof_out = (0.0, 0.0, 0.0, 0.0)     # 实际下发 DOF（限深 + 下潜补偿后）
        self._last_update_ms = _now_ms()   # 首帧平滑从当前时刻起算
        self.last_send_ms = 0
        self.last_motion = "stop"
        self._last_tx_log = 0
        # ---- 下位机遥测（深度等）与限深保护状态 ----
        self.telemetry = TEL.TelemetryReceiver()
        self._tel_playback = None
        _tp = os.environ.get("AUV_SIM_TEL_JSONL")
        if _tp:
            if not self.sim:
                print("[TEL] ⚠️ 设了 AUV_SIM_TEL_JSONL 但不是 SIM 模式（AUV_SIM_MODE=0）→ 忽略回放，"
                      "用真串口遥测（要回放请同时设 AUV_SIM_MODE=1）")
            else:
                try:
                    self._tel_playback = TEL.TelPlayback(_tp)
                except Exception as e:
                    print("[TEL] 遥测回放装载失败：%s" % e)
                    self._tel_playback = None
        self._last_tel_log_ms = 0
        self._last_guard_log_ms = 0
        self._warned_no_tel = False
        self.guard_active = False      # 最近一帧是否因限深压掉了上浮（供叠加/日志）
        self.guard_blocks = 0          # 累计被限深压掉的上浮帧数
        self.extra_min_depth_m = 0.0   # 任务级**只抬不降**的有效限深下限（set_extra_min_depth）
        self._dof_log_f = None
        self._last_dof_t = time.monotonic()
        self._tx_lock = threading.Lock()   # 串口写锁：看门狗线程也要写（硬停），不能与主线程交错
        self._watchdog = None              # 轴饱和看门狗（下面按配置/环境启动；早退分支也要有这个属性）
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

        # ---- 轴饱和看门狗（独立线程；默认：真串口开、SIM 关）----
        self._watchdog = None
        try:
            self._start_watchdog()
        except Exception as e:
            print("[UART] 轴看门狗启动失败：%s" % e)

    def _start_watchdog(self):
        """按 `comm.watchdog.*` 与环境变量 `AUV_WATCHDOG=0/1` 决定是否启动（SIM 默认不开）。"""
        env = os.environ.get("AUV_WATCHDOG")
        on = bool(S.get("comm.watchdog.enable", True))
        if env is not None:
            on = env.strip().lower() in ("1", "on", "true", "yes")
        elif self.sim:
            return                      # SIM（无硬件/用例）：默认不开，免得 os._exit 打断 pytest
        if not on:
            print("[UART] 轴看门狗：关闭（comm.watchdog.enable=false 或 AUV_WATCHDOG=0）")
            return
        self._watchdog = _AxisWatchdog(
            self,
            max_same_dir_s=float(S.get("comm.watchdog.max_same_dir_s", 10.0) or 0.0),
            min_off_b=int(S.get("comm.watchdog.min_off_b", 13) or 13),
            poll_ms=int(S.get("comm.watchdog.poll_ms", 50) or 50),
            axes=S.get("comm.watchdog.axes", ["yaw"])).start()

    # ---------------- 帧/发送 ----------------
    def _current_frame(self):
        return bytes([S.comm.frame.header] + self._axes +
                     [0, 0, S.comm.frame.btn_values[2]])

    def _ramp_step(self):
        """轴字节级平滑（模拟摇杆手感）："""
        now = _now_ms()
        dt = max(0.0, (now - self._last_update_ms) / 1000.0)
        self._last_update_ms = now
        self._dof_out = self._apply_depth_guard(*self._dof_target, now_ms=now)
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
        """读空串口接收缓冲 → 遥测解析（深度/姿态）。无串口、无数据 = 空操作。"""
        if self._tel_playback is not None:
            return self._tel_playback.pump(self.telemetry)
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
        """喂入下位机上行字节（15B 遥测帧，可任意切分）→ 更新深度等。"""
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

    @property
    def effective_min_depth_m(self):
        """本帧真正生效的限深下限(m) = `max(cfg 的 min_depth_m, 任务级 extra)`。

        ⚠️ `comm.depth_guard.min_depth_m`(现场定死 0.55) **不因任务级下限而改变** ——
        两者取 max，所以任务级下限只能把保护**收紧**、永远不能放宽。
        """
        base = float(S.get("comm.depth_guard.min_depth_m", _D_MIN_DEPTH_M) or 0.0)
        return max(base, float(getattr(self, "extra_min_depth_m", 0.0) or 0.0))

    def set_extra_min_depth(self, m):
        """任务级有效限深下限（m）：**只抬不降**。返回生效后的 `effective_min_depth_m`。

        用途：pitch 抬头之类的姿态动作会把机身最高点抬高（**没有 heave 指令也会靠近水面**），
        此时需要的是"更深才允许上浮"，而不是放宽保护。传 0 或比 cfg 更小的数 = 撤回请求，
        保护值仍是 cfg 的 `min_depth_m`。`estop`/`neutral` 不重置它，任务结束由调用方显式清零。
        """
        try:
            v = float(m)
        except (TypeError, ValueError):
            v = 0.0
        self.extra_min_depth_m = max(0.0, v) if math.isfinite(v) else 0.0
        return self.effective_min_depth_m

    def _apply_depth_guard(self, surge, sway, heave, yaw, now_ms=None):
        """上浮(heave>0)限深：当前深度 ≤ 有效下限 → 本帧禁止上浮。

        有效下限 = `max(comm.depth_guard.min_depth_m, extra_min_depth_m)`（见
        `effective_min_depth_m` / `set_extra_min_depth`）。
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
        limit = self.effective_min_depth_m
        if self.telemetry.depth_m > limit:
            return (surge, sway, heave, yaw)
        self.guard_active = True
        self.guard_blocks += 1
        why = "depth=%.2fm ≤ %.2fm" % (self.telemetry.depth_m, limit)
        if getattr(self, "extra_min_depth_m", 0.0) > 0.0:
            why += "（含任务级下限 %.2fm）" % self.extra_min_depth_m
        self._log_guard(heave, now_ms, why=why)
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
        """终端打印发出的运动帧：发一次打一次；帧后附[运动状态]。"""
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
        frame = bytearray(frame)
        if self._estop:
            frame = bytearray(build_turn_cancel_frame(0))
        elif self._cancel_turn_id is not None and not (frame[8] & 128):
            frame[8:11] = bytes([0, 1, self._cancel_turn_id])
        frame = bytes(frame)
        now = _now_ms()
        if not force and (now - self.last_send_ms) < S.comm.heartbeat.interval_ms:
            return False
        self.last_send_ms = now
        self._log_tx(frame, now)
        self._log_dof_row(frame, now)
        if self.sim:
            return True
        try:
            with self._tx_lock:                # 看门狗线程可能同时写（硬停），必须串行化
                written = self._ser.write(frame)
                if written != len(frame):
                    raise IOError("串口写入不完整: %s/%s" % (written, len(frame)))
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
        """按当前平滑后的轴输出一帧（心跳节流）。"""
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

    def request_turn(self, angle_deg, axis=1):
        """先同步，再按100ms重发同一编号；写失败保留请求供重试。

        `axis`：1=yaw（既有路径，默认）/ 2=pitch / 3=roll —— 也接受 `"pitch"` 这类轴名。
        pitch/roll 的 `angle_deg` 是**相对角**，符号含义 = 下位机 IMU 测量值的增减方向
        （正值使 IMU pitch/roll 增大），**不保证等于物理抬头/向左倾**，物理方向以现场实测为准
        （doc/记录/README_COMMUNICATION.md §2.2）。
        """
        encode_turn_angle(angle_deg)
        a = turn_axis_id(axis)
        if self._estop or self._turn_phase is not None:
            return False
        self._drain_rx()
        self.telemetry.buf.clear()
        self._cancel_turn_id = None
        self._turn_counter = self._turn_counter % 255 + 1
        self._turn_id = self._turn_counter
        self._turn_axis = a
        self._turn_frame = build_turn_frame(angle_deg, self._turn_id, a)
        self._turn_phase = "sync"
        self._sync_frames = self.telemetry.frames
        self._sync_sent = False
        self.telemetry.begin_turn(self._turn_id)
        self._dof_target = self._dof_out = (0.0, 0.0, 0.0, 0.0)
        self._axes = neutral_axis_bytes()
        self._turn_retry_at = 0.0
        self.poll_turn_complete()
        return True

    def poll_turn_complete(self):
        self._drain_rx()
        if self._estop or self._turn_phase is None:
            return self.telemetry.turn_complete
        if (self._turn_phase == "sync" and self._sync_sent and
                self.telemetry.frames > self._sync_frames and
                self.telemetry.turn_id == 0 and self.telemetry.turn_done is False):
            self._turn_phase = "run"
            self.telemetry.begin_turn(self._turn_id)
            self._turn_retry_at = 0.0
        if self._turn_phase == "run" and self.telemetry.turn_complete:
            self._turn_phase = None
            self._turn_frame = None
            return True
        now = time.monotonic()
        if now >= self._turn_retry_at:
            self._turn_retry_at = now + 0.1
            if self._turn_phase == "sync":
                self._sync_sent = True
                self._write(build_turn_cancel_frame(0), force=True)
            else:
                self._write(self._turn_frame, force=True)
        return False

    def cancel_turn(self):
        """取消本次自动转向，后续普通帧继续携带取消直到新任务。"""
        if self._turn_phase is None:
            return
        cancel_id = 0 if self._turn_phase == "sync" else self._turn_id
        self._turn_phase = None
        self._turn_frame = None
        self.telemetry._turn_waiting = False
        self.telemetry.turn_complete = False
        self._cancel_turn_id = cancel_id
        self._write(build_turn_cancel_frame(cancel_id), force=True)

    def neutral(self):
        """回中性并强发一帧。"""
        self._dof_target = (0.0, 0.0, 0.0, 0.0)
        self.last_motion = "stop"
        return self.send_motion(name="stop", force=True)

    def stop_hard(self, dt=0.05, verify=True, settle_s=0.6, tol_deg=2.0,
                  max_extra=3, quiet=False):
        """**真正停住**：连发中性帧走完 ramp，再用遥测 yaw 验证它停下来了。"""
        self.cancel_turn()
        mid = int(S.comm.frame.axis_mid)
        spd = float(S.comm.ramp.speed_per_s or 0.0)
        # 最大偏离（±127 字节）走完需要多少帧，再留 2 帧余量
        ticks = 2 if spd <= 0 else int(127.0 / max(1.0, spd * dt)) + 3
        if not quiet:
            off = [int(a) - mid for a in self._axes[:4]]
            if any(off):
                print("[UART] 硬停：轴偏离中位 %s（字节）→ 连发 %d 帧中性 + 遥测验证"
                      % (off, ticks))
        self._dof_target = (0.0, 0.0, 0.0, 0.0)
        self.last_motion = "stop"
        for _ in range(ticks):
            self.send_motion(name="stop", force=True)
            try:
                time.sleep(dt)
            except Exception:
                pass
        ok = all(int(a) == mid for a in self._axes[:4])
        if not verify:
            return ok
        # ② 遥测 yaw 闭环验证：静置窗口内 yaw 不应再明显变化
        y0 = getattr(self.telemetry, "yaw_deg", None)
        if y0 is None:
            if not quiet:
                print("[UART] 硬停：回中位 %s；**无遥测 yaw，停住与否未验证**" % ok)
            return ok
        for k in range(max(1, int(max_extra))):
            t_end = time.time() + max(0.1, settle_s)
            y1 = y0
            while time.time() < t_end:
                self.send_motion(name="stop", force=True)
                y = getattr(self.telemetry, "yaw_deg", None)
                if y is not None:
                    y1 = y
                try:
                    time.sleep(dt)
                except Exception:
                    pass
            d = _wrap180((y1 or 0.0) - (y0 or 0.0))
            if abs(d) <= float(tol_deg):
                if not quiet:
                    print("[UART] 硬停：✅ 已停住（%.1fs 内 yaw 变化 %+.2f° ≤ %.1f°）"
                          % (settle_s, d, tol_deg))
                return ok
            if not quiet:
                print("[UART] 硬停：⚠️ 仍在转（%.1fs 内 yaw 变化 %+.2f°）→ 补发中性帧"
                      % (settle_s, d))
            y0 = y1
        return False

    # ---------------- 安全 ----------------
    def estop(self):
        if self._estop:
            return
        self._estop = True
        self.cancel_turn()
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
        self.cancel_turn()
        mid = int(S.comm.frame.axis_mid)
        if any(int(a) != mid for a in self._axes[:4]):
            try:
                self.stop_hard(quiet=False)
            except Exception as e:
                print("[UART] close 前硬停失败：%s" % e)
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


class _AxisWatchdog(object):
    """**轴饱和看门狗**（用户 2026-09-27 定，阈值 5.0s）：**只看 yaw 轴** —— 同一方向**连续**发 yaw ≥"""

    AXES = ("yaw", "surge", "heave", "sway")
    # ★ 不走手动轴字节的轴：pitch/roll 是「指定角度轴任务」（byte[7..10]），
    #   老的「同向饱和」判据对它们**根本不适用** ⇒ 另开一路看 IMU 姿态（见 `check_tilt`）。
    TILT_AXES = ("pitch", "roll")

    @staticmethod
    def _resolve_axes(axes):
        """轴名/下标 → 字节下标元组（名字走 `comm.dof_map`，不硬编码）。空列表 ⇒ 看门狗不盯任何轴。"""
        idx = {}
        try:
            for k, v in (S.comm.get("dof_map", None) or {}).items():
                idx[str(k)] = int((v or {}).get("axis", -1))
        except Exception:
            idx = {}
        out = []
        for a in (axes or ()):
            if isinstance(a, int):
                out.append(a)
            else:
                i = idx.get(str(a), None)
                if i is None:
                    print("[UART] 看门狗：未知轴名 %r（可用 %s）→ 忽略" % (a, sorted(idx)))
                    continue
                out.append(i)
        return tuple(sorted(set(i for i in out if 0 <= i < 4)))

    def axis_names(self):
        """实际盯的轴名（日志/用例用）。"""
        return tuple(self.AXES[i] if i < len(self.AXES) else ("axis%d" % i) for i in self.axes)

    def __init__(self, uart, max_same_dir_s=10.0, min_off_b=13, poll_ms=50,
                 on_trip=None, log=None, axes=None):
        self.uart = uart
        self.max_same_dir_s = float(max_same_dir_s or 0.0)
        self.min_off_b = int(min_off_b or 0)
        self.poll_ms = max(10, int(poll_ms or 50))
        # 只看哪些轴：轴名或下标。名单里出现 pitch/roll ⇒ 走**姿态**那一路（`check_tilt`），
        # 其余走**手动轴字节饱和**那一路。缺省只盯 yaw。
        names = list(axes if axes is not None else ("yaw",))
        self.tilt_axes = tuple(str(n).strip().lower() for n in names
                               if not isinstance(n, int) and str(n).strip().lower() in self.TILT_AXES)
        self.axes = self._resolve_axes([n for n in names
                                        if isinstance(n, int) or str(n).strip().lower() not in self.TILT_AXES])
        # 姿态限幅/保持时长：阈值复用 `comm.motion.axis.max_tilt_deg`（**单一来源**）
        try:
            self.max_tilt_deg = abs(float(S.get("comm.motion.axis.max_tilt_deg", _D_MAX_TILT_DEG)))
        except (TypeError, ValueError):
            self.max_tilt_deg = _D_MAX_TILT_DEG
        try:
            self.tilt_hold_s = max(0.0, float(S.get("comm.watchdog.tilt_hold_s", 0.5) or 0.0))
        except (TypeError, ValueError):
            self.tilt_hold_s = 0.5
        self._tilt_t0 = {}
        self._tilt_ref = None          # 上电首次读到的 pitch/roll（诊断用：看是不是安装偏置）
        self.on_trip = on_trip
        self.log = log or print
        self._stop = threading.Event()
        self._warned_err = False
        self._thread = threading.Thread(target=self._run, name="axis-watchdog", daemon=True)

    def tilt_names(self):
        return self.tilt_axes

    def start(self):
        if not self.axes and not self.tilt_axes:
            self.log("[UART] 轴看门狗：comm.watchdog.axes 为空 ⇒ 不盯任何轴（不启动）")
            return self
        if self.axes and self.max_same_dir_s <= 0:
            self.log("[UART] 轴看门狗：手动轴饱和判据关闭（max_same_dir_s=0）")
        parts = []
        if self.axes:
            parts.append("手动轴 %s：同一方向连续发轴 ≥ %.1fs ⇒ 硬停 + 强制退出"
                         "（|偏离中位| ≥ %d 字节）"
                         % ("/".join(self.axis_names()), self.max_same_dir_s, self.min_off_b))
        if self.tilt_axes:
            parts.append("姿态 %s：|遥测角| > %.0f° 连续 ≥ %.1fs ⇒ 硬停 + 强制退出"
                         % ("/".join(self.tilt_axes), self.max_tilt_deg, self.tilt_hold_s))
        self.log("[UART] 轴看门狗（巡检 %dms）｜ %s" % (self.poll_ms, " ； ".join(parts)))
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def _run(self):
        try:
            mid = int(S.comm.frame.axis_mid)
        except Exception:
            mid = 128
        dirs = {}          # 字节下标 → 当前方向（只记盯的轴）
        t0 = {}            # 字节下标 → 该方向起始时刻
        while not self._stop.wait(self.poll_ms / 1000.0):
            try:
                ax = list(self.uart._axes)
            except Exception:
                continue
            try:
                _now = time.monotonic()
                if self.tilt_axes and self.check_tilt(now=_now):
                    return               # trip 已经处理过（on_trip 或强制退出）
                self._poll_once(ax, mid, now=_now, dirs=dirs, t0=t0)
            except Exception as e:
                if not self._warned_err:
                    self._warned_err = True
                    self.log("[UART] 看门狗巡检异常（继续跑）：%r" % (e,))
            continue

    def _poll_once(self, ax, mid, now, dirs, t0):
        """单次巡检（抽出来便于整体兜异常 + 单测）。"""
        for i in self.axes:                      # ★ 只看 comm.watchdog.axes（缺省 yaw）
            if i >= len(ax):
                continue
            b = ax[i]
            try:
                off = int(b) - mid
            except Exception:
                continue
            if abs(off) < self.min_off_b:
                dirs[i], t0[i] = 0, None
                continue
            d = 1 if off > 0 else -1
            if dirs.get(i) != d:
                dirs[i], t0[i] = d, now
                continue
            if t0.get(i) is not None and (now - t0[i]) >= self.max_same_dir_s:
                self._trip(i, int(b), now - t0[i])
                return

    def check_tilt(self, now=None):
        """★ **姿态看守**（pitch/roll）：`|遥测角| > max_tilt_deg` 且**连续** ≥ `tilt_hold_s` ⇒ trip。

        为什么必须另开一路：pitch/roll 走「指定角度轴任务」，**不经过手动轴字节**，老的同向饱和
        判据对它们不适用。没有遥测时**不判**（信息不足不乱停船）。返回被 trip 的轴名或 None。

        ⚠️ 判据是**绝对** IMU 角：若 IMU 安装有偏置（上电静止时就 > 限值），会立刻停船 ——
        那本身就是要先解决的问题（日志会把上电基准打出来）。
        """
        if not self.tilt_axes or self.max_tilt_deg <= 0:
            return None
        t = time.monotonic() if now is None else float(now)
        tel = getattr(self.uart, "telemetry", None)
        vals = {}
        for name in self.tilt_axes:
            v = getattr(tel, "%s_deg" % name, None) if tel is not None else None
            vals[name] = None if v is None else float(v)
        if self._tilt_ref is None and any(v is not None for v in vals.values()):
            self._tilt_ref = dict(vals)
            self.log("[UART] 看门狗基准（上电首次读到）：%s"
                     % " ".join("%s=%s" % (k, "-" if v is None else "%.1f°" % v)
                                for k, v in vals.items()))
        for name, v in vals.items():
            if v is None:
                self._tilt_t0.pop(name, None)      # 没遥测 = 不判
                continue
            if abs(v) > self.max_tilt_deg:
                if name not in self._tilt_t0:
                    self._tilt_t0[name] = t
                if t - self._tilt_t0[name] >= self.tilt_hold_s:
                    self._trip_tilt(name, v, t - self._tilt_t0[name])
                    return name
            else:
                self._tilt_t0.pop(name, None)
        return None

    def _trip_tilt(self, name, deg, held_s):
        ref = (self._tilt_ref or {}).get(name, None)
        self.log("[UART] ⛔ **轴看门狗·姿态**：%s 实测 %+.1f° 超过 ±%.0f° 已持续 %.1fs"
                 "（上电基准 %s）⇒ **硬停 + 强制退出**"
                 % (name, deg, self.max_tilt_deg, held_s,
                    "n/a" if ref is None else "%+.1f°" % ref))
        try:
            self.uart.estop()
        except Exception:
            pass
        try:
            if getattr(self.uart, "_ser", None) is not None:
                with self.uart._tx_lock:
                    for _ in range(20):
                        self.uart._ser.write(build_neutral_frame())
                        time.sleep(0.02)
        except Exception:
            pass
        if self.on_trip is not None:
            try:
                self.on_trip(name, held_s, deg)
            except Exception:
                pass
            return
        for _f in (sys.stdout, sys.stderr):
            try:
                _f.flush()
            except Exception:
                pass
        os._exit(9)

    def _trip(self, i, byte, held_s):
        mid = int(S.comm.frame.axis_mid)
        name = self.AXES[i] if i < len(self.AXES) else ("axis%d" % i)
        self.log("[UART] ⛔ **轴看门狗**：%s 轴沿同一方向连续发了 %.1fs（字节 %d ⇒ 偏离中位 %+d）"
                 " ⇒ **硬停 + 强制退出**（任务可能卡死了；先让船停下再退）"
                 % (name, held_s, byte, byte - mid))
        try:
            self.uart.estop()                    # 连发原始中性帧（绕过 ramp）
        except Exception:
            pass
        try:
            if getattr(self.uart, "_ser", None) is not None:
                with self.uart._tx_lock:
                    for _ in range(20):
                        self.uart._ser.write(build_neutral_frame())
                        time.sleep(0.02)
        except Exception:
            pass
        if self.on_trip is not None:
            try:
                self.on_trip(name, held_s, byte)
            except Exception:
                pass
            return
        for _f in (sys.stdout, sys.stderr):
            try:
                _f.flush()
            except Exception:
                pass
        os._exit(9)                              # 强制退出（用户要求：把线程掐死）


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
