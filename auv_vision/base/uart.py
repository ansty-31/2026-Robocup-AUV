# -*- coding: utf-8 -*-
"""uart.py — 串口 v2：11B 帧(0xA5+7轴+3键)；上层 DOF 目标→DOF_MAP 真值表→轴字节(ramp 平滑)，
模拟遥控输出；心跳 ≥20Hz；急停=连发中性帧。帧语义见 README 与 docs/CUP_AUV。"""
import os
import sys
import time

import base.settings as S

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


def build_frame_with_header(header, surge=0.0, sway=0.0, heave=0.0, yaw=0.0):
    """以自定义帧头构建 11B 帧（交接/待命用；如 0xAA 表示交给下位机接管）。"""
    axes = dof_to_axis_bytes(surge, sway, heave, yaw)
    return bytes([header] + axes + list(S.comm.frame.btn_values))


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
        self._dof_target = (0.0, 0.0, 0.0, 0.0)  # 期望 DOF
        self._last_update_ms = _now_ms()   # 首帧 ramp 从当前时刻起算
        # DOF 级线性斜坡（模拟人手推杆的连续变化；ramp.dof_per_s<=0 直通）
        self._dof_ramp = None
        _dof_rate = S.get("comm.ramp.dof_per_s", 0.0) or 0.0
        if _dof_rate > 0:
            try:
                from common.ramp import DofRamp      # 惰性导入，避免 base↔common 耦合
                self._dof_ramp = DofRamp(_dof_rate)
            except Exception as _e:
                print("[UART] DOF 线性斜坡不可用(%s)；改为直通" % _e)
        self.last_send_ms = 0
        self.last_motion = "stop"
        self._last_tx_log = 0
        self._dof_log_f = None
        self._last_dof_t = time.monotonic()
        self._rxbuf = bytearray()                # 接收缓存（“特殊标志”判定用）
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
        """平滑推进运动输出（模拟摇杆手感）：

        ① DOF 级线性斜坡：按 `ramp.dof_per_s`(归一化DOF/秒) 把期望 DOF 线性逼近，
           避免速度直接激增（人手推杆的连续变化感）；`<=0` 直通；
        ② 轴字节级平滑：按 `ramp.speed_per_s` 再平滑一次（保留原手感）。
        """
        now = _now_ms()
        dt = max(0.0, (now - self._last_update_ms) / 1000.0)
        self._last_update_ms = now
        dof = (self._dof_ramp.update(self._dof_target, dt)
               if self._dof_ramp is not None else self._dof_target)
        tgt = dof_to_axis_bytes(*dof)
        step_max = (256.0 if S.comm.ramp.speed_per_s <= 0
                    else S.comm.ramp.speed_per_s * dt)
        for i in range(4):                    # 仅平滑双摇杆区
            diff = tgt[i] - self._axes[i]
            if abs(diff) <= step_max:
                self._axes[i] = tgt[i]
            else:
                self._axes[i] = int(round(
                    self._axes[i] + (step_max if diff > 0 else -step_max)))
        for i in (4, 5, 6):                   # 附加字节直接取目标
            self._axes[i] = tgt[i]

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
        print("[UART→] %s  [运动状态] %s"
              % (frame.hex(), describe_motion(frame)))

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
        """原样发送一帧字节（轨迹回放用；跳过 ramp，可绕过心跳节流）。"""
        if self._estop and frame != build_neutral_frame():
            return False
        return self._write(frame, force=force)

    def send_motion(self, name=None, force=False):
        """按当前平滑后的轴输出一帧（心跳节流）。"""
        if self._estop:
            return False
        if name is not None:
            self.last_motion = name
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
        if self._dof_ramp is not None:
            self._dof_ramp.reset((0.0, 0.0, 0.0, 0.0))   # 停车不走 DOF 斜坡
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

    # ---------------- 接收 ----------------
    def _drain_rx(self):
        if self.sim or self._ser is None:
            return
        try:
            n = self._ser.in_waiting
            if n:
                data = self._ser.read(n)
                self._rxbuf.extend(data)
                if len(self._rxbuf) > 4096:          # 防无限增长
                    del self._rxbuf[:-1024]
                if S.DEBUG:
                    print("[UART<-] %s" % data.hex())
        except Exception:
            pass

    # ---------------- 接收标志（交接握手用） ----------------
    def rx_clear(self):
        """清空接收缓存。"""
        self._rxbuf = bytearray()

    def rx_has(self, flag):
        """接收缓存是否已出现 flag(字节串)；命中后丢弃该段(含)之前的缓存。"""
        if not flag:
            return False
        i = bytes(self._rxbuf).find(flag)
        if i < 0:
            return False
        del self._rxbuf[:i + len(flag)]
        return True

    def send_dof_header(self, header, surge=0.0, sway=0.0, heave=0.0, yaw=0.0,
                        force=True):
        """以自定义帧头发一帧（交接/待命心跳；跳过 ramp）。"""
        return self.send_frame_bytes(
            build_frame_with_header(header, surge, sway, heave, yaw),
            force=force)

    def wait_flag(self, flag, timeout_ms=0, tick=None, tick_interval_ms=50):
        """等待接收缓存出现 flag；期间每 tick_interval_ms 调 tick()（如发心跳）。

        返回 True=收到标志；False=超时。timeout_ms<=0 = 不限时。
        """
        t0 = _now_ms()
        last_tick = 0
        while True:
            self._drain_rx()
            if self.rx_has(flag):
                return True
            now = _now_ms()
            if timeout_ms and (now - t0) >= timeout_ms:
                return False
            if tick is not None and (now - last_tick) >= tick_interval_ms:
                last_tick = now
                tick()
            time.sleep(0.01)

    def close(self):
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