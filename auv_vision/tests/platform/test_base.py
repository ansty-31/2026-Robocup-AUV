# -*- coding: utf-8 -*-
"""tests/platform/test_base.py — base 层：settings 加载 / 11B 串口帧与控制器 / 遥测上行与限深保护 / 相机工厂。
（详细用法、判据与实测见 doc/注释历史.md）"""
import os
import time

import numpy as np
import pytest

import base.cfg.settings as S
import base.hw.telemetry as T
from base.hw import uart as U
from base.hw.camera import SimCamera, create_camera


def test_settings_loads_real_yaml_values():
    """配置真的从 cfg/*.yaml 读进来了（抽查几个关键真实值），缺路径返回 None。"""
    #   钉死具体数字就违背本用例自己的约定（改参数 = 每次都要改测试 = 假红）。
    assert int(S.comm.gate.pass_target) >= 1
    #    所以这里只断言**类型与不变量**，不钉死数值（钉死 = 每次调参都要改测试 = 假红）。
    assert 0.0 < float(S.vision.gate.keypoint.conf_thr) <= 1.0
    assert isinstance(bool(S.vision.gate.kpt_mem.enable), bool)
    # 不变量：回忆点置信度必须低于任务侧阈值，否则"回忆点"会被判成真实可见（见 vision.yaml 注释）
    if S.vision.gate.kpt_mem.enable:
        assert float(S.vision.gate.kpt_mem.recall_conf) < \
            float(S.vision.gate.keypoint.conf_thr)
    assert S.vision.gate.pnp.z_max == 15.0
    assert S.comm.frame.header == 0xA5
    #    这一条专门钉**数值**：改了它必须有人来解释。
    assert float(S.comm.depth_guard.min_depth_m) == pytest.approx(0.55), \
        "comm.depth_guard.min_depth_m 是现场定死的 0.55，不准改（详见 README 限深保护一节）"
    assert bool(S.comm.depth_guard.enable) is True
    assert S.get("vision.gate.kpt_mem.recall_conf") == 0.45
    assert S.get("vision.no_such_key") is None
    assert S.get("comm.gate.deep.missing") is None
    assert S.get("bogus.path") is None

    # AUV_SIM_MODE → S.SIM_MODE（conftest 用 setdefault 保证测试默认 SIM）
    assert "AUV_SIM_MODE" in os.environ
    raw = os.environ["AUV_SIM_MODE"].strip().lower()
    expect_sim = raw not in ("0", "false", "no", "off", "sim_off")
    assert S.SIM_MODE is expect_sim


def test_uart_dof_axis_bytes_and_motion_text():
    """DOF→轴字节真值表、越界夹紧、中性帧，以及帧→可读运动状态。"""
    mid = S.comm.frame.axis_mid
    rng = S.comm.frame.axis_range

    # 全 0 == 中性轴字节（含 aux 安全默认）
    assert U.dof_to_axis_bytes() == U.neutral_axis_bytes()
    # 附加字节来自 cfg（aux_axis）：滚轮中值 165 / tri 中位 1 钉死；
    # index6 = 整帧 byte[7] = 指定角度轴，按协议 §6-2 中性值为 0（转向不受它影响，见 test_turn_axes）
    aux = dict(S.comm.frame.aux_axis)
    assert (aux[4], aux[5], aux[6]) == (165, 1, 0)
    assert U.neutral_axis_bytes() == [128, 128, 128, 128, aux[4], aux[5], aux[6]]

    # dof_map：yaw→axis0, surge→axis1, heave→axis2, sway→axis3
    assert U.dof_to_axis_bytes(yaw=1.0)[0] == mid + rng
    assert U.dof_to_axis_bytes(surge=1.0)[1] == mid + rng
    assert U.dof_to_axis_bytes(heave=1.0)[2] == mid + rng
    assert U.dof_to_axis_bytes(sway=1.0)[3] == mid + rng
    assert U.dof_to_axis_bytes(surge=-1.0)[1] == mid - rng

    # 越界输入被夹到 [0,255]，永远是 7 轴
    for v in (-5.0, 5.0):
        axes = U.dof_to_axis_bytes(surge=v, sway=v, heave=v, yaw=v)
        assert len(axes) == 7
        assert all(0 <= int(a) <= 255 for a in axes)
    assert U.dof_to_axis_bytes(surge=5.0)[1] == 255
    assert U.dof_to_axis_bytes(surge=-5.0)[1] == 0

    # 帧 → 人类可读运动状态
    assert U.describe_motion(U.build_frame_from_dof(surge=1.0)) == "前进100%"
    assert U.describe_motion(U.build_neutral_frame()) == "停止"




def test_uart_controller_sim_setmotion_neutral_estop():
    """SIM 控制器：预设→DOF 目标、未知预设回 stop、neutral 回中、急停锁死运动帧。"""
    u = U.UartController(sim=True)
    assert u.sim is True
    assert u.estop_active is False
    assert u.dof_target == (0.0, 0.0, 0.0, 0.0)

    assert u.set_motion("stop") is True
    assert u.last_motion == "stop"
    assert u.dof_target == (0.0, 0.0, 0.0, 0.0)     # 取 comm.motion.presets（表里只留生产用得到的 stop）

    u.set_motion("no_such_preset")                  # 未定义 → 回落 stop
    assert u.last_motion == "stop"
    assert u.dof_target == (0.0, 0.0, 0.0, 0.0)

    assert u.neutral() is True
    assert u.last_motion == "stop"
    assert u.dof_target == (0.0, 0.0, 0.0, 0.0)

    u.estop()
    assert u.estop_active is True
    assert u.send_motion(force=True) is False       # 急停后拒绝运动帧
    assert u.send_frame_bytes(U.build_frame_from_dof(surge=1.0),
                              force=True) is False
    # 中性帧仍放行（急停=连发中性）
    assert u.send_frame_bytes(U.build_neutral_frame(), force=True) is True

    u.close()                                       # 收尾可重复调用
    u.close()


def test_create_camera_sim_returns_configured_frame(monkeypatch):
    """create_camera 工厂：SIM 相机按配置尺寸出帧（BGR uint8）。"""
    # 配置里 front 是 usb；本机无相机 → 测试强制 sim，避免打开真实设备
    monkeypatch.setitem(S.vision.camera.front, "type", "sim")
    front = create_camera("front")
    assert isinstance(front, SimCamera)
    assert (front.width, front.height) == (S.vision.camera.front.width,
                                           S.vision.camera.front.height)
    f = front.read()
    assert f is not None
    assert f.dtype == np.uint8
    assert f.shape == (front.height, front.width, 3)

    # down 现在是 usb（2026-10-02 板端 /dev/video2）；同样强制 sim，避免打开真实设备
    monkeypatch.setitem(S.vision.camera.down, "type", "sim")
    down = create_camera("down")
    d = down.read()
    assert d.shape == (S.vision.camera.down.height,
                       S.vision.camera.down.width, 3)


def test_telemetry_frame_roundtrip_and_resync():
    """造帧↔拆帧往返；半帧/粘包/错位/校验错可重同步（串口按字节切分是常态）。"""
    f = T.build_telemetry_frame(0.42, 0.30, 1.5, -2.25, 88.0)
    assert len(f) == T.TEL_LEN == 15          # 2026-10-02：加了 turn_done，14B → 15B
    assert f[:2] == T.TEL_HEADER == b"\xAA\x55"
    assert T.check_telemetry(f) is True
    assert T.decode_telemetry(f)[:5] == pytest.approx((0.42, 0.30, 1.5, -2.25, 88.0))
    assert T.decode_telemetry(f)[5:] == (False, 0)     # (turn_done, turn_id)

    # 转向反馈：编号 + 完成标志要能原样往返
    ft = T.build_telemetry_frame(0.42, 0.30, 1.5, -2.25, 88.0, turn_done=True, turn_id=7)
    assert T.check_telemetry(ft) is True
    assert T.decode_telemetry(ft)[5:] == (True, 7)

    # 半帧不产出；补齐后才产出
    buf = bytearray(f[:6])
    assert T.parse_telemetry_frames(buf) == []
    buf.extend(f[6:])
    assert T.parse_telemetry_frames(buf)[0][:5] == pytest.approx(
        (0.42, 0.30, 1.5, -2.25, 88.0))
    assert len(buf) == 0                       # 整帧被消费掉

    # 粘包：一次喂两帧 → 两行
    buf = bytearray(T.build_telemetry_frame(0.10) +
                    T.build_telemetry_frame(1.20))
    rows = T.parse_telemetry_frames(buf)
    assert [r[0] for r in rows] == pytest.approx([0.10, 1.20])

    # 前置垃圾 + 错位字节：能找到帧头并重新对齐
    buf = bytearray(b"\x01\x02\x03\xAA")       # 末尾半个帧头要留住
    assert T.parse_telemetry_frames(buf) == []
    assert bytes(buf) == b"\xAA"
    buf.extend(b"\x55" + T.build_telemetry_frame(0.33)[2:])
    assert [r[0] for r in T.parse_telemetry_frames(buf)] == pytest.approx([0.33])

    # 校验错：本帧丢弃（计数 bad），后面那帧仍能解出来
    bad = bytearray(T.build_telemetry_frame(9.99))
    bad[-1] ^= 0xFF
    stats = {"ok": 0, "bad": 0}
    rows = T.parse_telemetry_frames(bytearray(bad + T.build_telemetry_frame(0.55)),
                                    stats)
    assert [r[0] for r in rows] == pytest.approx([0.55])
    assert stats == {"ok": 1, "bad": 1}
    assert T.check_telemetry(bytes(bad)) is False




class _FakeSerial(object):
    """假串口：只实现 _drain_rx 用到的 in_waiting/read/write/close。"""

    def __init__(self, rx=b""):
        self.rx = bytearray(rx)
        self.written = []

    @property
    def in_waiting(self):
        return len(self.rx)

    def read(self, n):
        out = bytes(self.rx[:n])
        del self.rx[:n]
        return out

    def write(self, frame):
        self.written.append(bytes(frame))
        return len(frame)

    def close(self):
        pass




def test_uart_real_serial_pty_telemetry_and_guard(monkeypatch):
    """真串口路径（pty + pyserial）：下行 11B 帧发得出去，上行遥测收到并触发限深。"""
    pytest.importorskip("serial")            # 板上没装 pyserial 就跳过（无硬件测试）
    import select
    import pty
    import time

    monkeypatch.setitem(S.comm.ramp, "speed_per_s", 0.0)
    master, slave = pty.openpty()
    u = U.UartController(dev=os.ttyname(slave), sim=False)
    try:
        assert u.sim is False
        os.write(master, T.build_telemetry_frame(0.22))
        time.sleep(0.05)                     # pty 主→从要过线规层，等一拍才可读
        u.send_dof(0.0, 0.0, 1.0, 0.0)       # 上浮指令：发帧前收遥测 → 深度不足 → 被压掉
        assert u.depth_m == pytest.approx(0.22)
        assert u.dof_out[2] == 0.0
        assert list(u._current_frame())[3] == S.comm.frame.axis_mid   # heave 轴回中

        ready, _, _ = select.select([master], [], [], 1.0)
        assert ready, "11B 帧没有真正写到串口"
        got = os.read(master, 64)
        assert len(got) == 11                       # 0xA5 + 7 轴 + 3 键
        assert got[0] == S.comm.frame.header

        os.write(master, T.build_telemetry_frame(0.60))    # 深度够了 → 恢复上浮
        time.sleep(0.05)
        u.send_dof(0.0, 0.0, 1.0, 0.0, force=True)   # 上浮 + 强制发帧
        assert u.depth_m == pytest.approx(0.60)
        assert u.dof_out[2] == 1.0
    finally:
        u.close()
        os.close(master)
        os.close(slave)


# 限深保护：深度不足时禁止上浮（机身不得冒出水面）
def _guard_uart(monkeypatch, sent, dof_comp=False):
    """SIM 控制器 + 关 ramp（轴直通目标）+ 截获实际发出的帧。"""
    monkeypatch.setitem(S.comm.ramp, "speed_per_s", 0.0)
    monkeypatch.setitem(S.comm.depth_guard, "enable", True)
    monkeypatch.setitem(S.comm.dof_comp, "enable", bool(dof_comp))
    u = U.UartController(sim=True)
    monkeypatch.setattr(u, "_write", lambda frame, force=False:
                        (sent.append(bytes(frame)), True)[1])
    return u




def test_depth_guard_blocks_surfacing_only(monkeypatch):
    """深度 ≤ min_depth_m：上浮 >0 被清零，surge/sway/yaw 照旧；下潜照常。"""
    sent = []
    u = _guard_uart(monkeypatch, sent)
    mid = S.comm.frame.axis_mid

    lim0 = float(S.comm.depth_guard.min_depth_m)
    shallow = round(lim0 - 0.05, 3)          # 明确"浅于下限"（阈值可能被改，别写死）
    u.feed_telemetry(T.build_telemetry_frame(shallow), now_ms=U._now_ms())
    assert u.depth_m == pytest.approx(shallow)

    u.send_dof(surge=0.4, sway=-0.3, heave=0.8, yaw=0.2, force=True)
    assert u.dof_target == (0.4, -0.3, 0.8, 0.2)      # 上层请求不变
    assert u.dof_out == pytest.approx((0.4, -0.3, 0.0, 0.2))  # 只压掉上浮
    assert u.guard_active is True and u.guard_blocks == 1
    axes = list(sent[-1])[1:8]
    assert len(sent[-1]) == 11
    assert axes[2] == mid                             # heave 轴当帧回中（snap）
    assert axes[1] > mid and axes[3] < mid and axes[0] > mid   # 其余通道照发

    # 下潜(-heave)/悬停不受限
    u.send_dof(heave=-0.6, force=True)
    assert u.dof_out[2] == pytest.approx(-0.6)
    assert u.guard_active is False
    assert list(sent[-1])[3] < mid

    # 边界：判据是"深度 ≥ min_depth_m 才放行"，恰好等于下限仍算触底 → 禁止上浮
    lim = float(S.comm.depth_guard.min_depth_m)
    u.feed_telemetry(T.build_telemetry_frame(lim), now_ms=U._now_ms())
    u.send_dof(heave=0.5, force=True)
    assert u.dof_out[2] == 0.0, "深度正好等于下限 %.2fm 时应禁止上浮" % lim
    # 略深于下限 → 放行
    u.feed_telemetry(T.build_telemetry_frame(lim + 0.01), now_ms=U._now_ms())
    u.send_dof(heave=0.5, force=True)
    assert u.dof_out[2] == pytest.approx(0.5), "深度 %.2fm > 下限 %.2fm 应放行" % (lim + 0.01, lim)
    assert u.guard_active is False
    u.close()


def test_dive_boost_scales_downward_only(monkeypatch):
    """**下潜动力单独放大**（底层共用，撞球/过门都吃）：只乘 heave<0，其余通道原样。"""
    sent = []
    u = _guard_uart(monkeypatch, sent, dof_comp=True)
    mid = S.comm.frame.axis_mid
    k = float(S.comm.dof_comp.dive_scale)
    assert k > 1.0, "dive_scale 应大于 1（否则等于没放大）"

    u.send_dof(heave=-0.2, force=True)                 # 下潜 → 按倍数放大
    assert u.dof_out[2] == pytest.approx(-0.2 * k)
    assert list(sent[-1])[3] < mid, "heave 轴字节应低于中位（下潜方向）"

    u.send_dof(heave=-0.9, force=True)                 # 放大后超范围 → 夹到 -1.0
    assert u.dof_out[2] == pytest.approx(-1.0)

    u.send_dof(heave=0.2, force=True)                  # 上浮 → 不动
    assert u.dof_out[2] == pytest.approx(0.2), "上浮不该被放大"
    u.send_dof(heave=0.0, sway=-0.2, yaw=0.2, force=True)   # 悬停/平移/转向 → 不动
    assert u.dof_out[2] == pytest.approx(0.0)
    assert u.dof_out[1] == pytest.approx(-0.2), "平移不该被放大"
    assert u.dof_out[3] == pytest.approx(0.2), "转向不该被放大"
    u.close()




def test_depth_guard_passthrough_when_stale_or_disabled(monkeypatch):
    """遥测超时/从未收到/开关关闭 → 放行（仿真与台架不能被保护锁死）。"""
    sent = []
    u = _guard_uart(monkeypatch, sent)

    # 从未收到遥测 → 放行
    u.send_dof(heave=0.7, force=True)
    assert u.dof_out[2] == pytest.approx(0.7)
    assert u.guard_active is False

    # 遥测超时（stale_ms 默认 500）→ 放行
    u.feed_telemetry(T.build_telemetry_frame(0.10),
                     now_ms=U._now_ms() - 5000)
    assert u.depth_fresh() is False
    u.send_dof(heave=0.7, force=True)
    assert u.dof_out[2] == pytest.approx(0.7)

    # 开关关闭 → 即使深度很浅也放行
    monkeypatch.setitem(S.comm.depth_guard, "enable", False)
    u.feed_telemetry(T.build_telemetry_frame(0.10), now_ms=U._now_ms())
    u.send_dof(heave=0.9, force=True)
    assert u.dof_out[2] == pytest.approx(0.9)
    assert u.guard_active is False
    u.close()


def test_depth_guard_stale_action_block_up(monkeypatch):
    """可选保守档 stale_action=block_up：断链/超时也不盲上浮（SIM 仍放行，台架不被锁死）。"""
    sent = []
    u = _guard_uart(monkeypatch, sent)
    monkeypatch.setitem(S.comm.depth_guard, "stale_action", "block_up")

    u.send_dof(heave=0.8, force=True)                 # SIM + 无遥测 → 仍放行
    assert u.dof_out[2] == pytest.approx(0.8)

    # 真串口 + 无数据（断链）→ 保守档压掉上浮
    monkeypatch.setattr(u, "sim", False)
    monkeypatch.setattr(u, "_ser", _FakeSerial())
    u.send_dof(heave=0.8, force=True)
    assert u.dof_out[2] == 0.0
    assert u.guard_active is True and u.guard_blocks == 1
    assert list(sent[-1])[3] == S.comm.frame.axis_mid
    u.send_dof(heave=-0.8, force=True)                # 下潜不受影响
    assert u.dof_out[2] == pytest.approx(-0.8)

    # 遥测新鲜且深度充足 → 恢复上浮
    u.feed_telemetry(T.build_telemetry_frame(0.80), now_ms=U._now_ms())
    u.send_dof(heave=0.8, force=True)
    assert u.dof_out[2] == pytest.approx(0.8)
    u.close()



# ---------------------------------------------------------------- 硬停（安全）
class _CountWrite(object):
    def __init__(self, u):
        self.u = u
        self.n = 0
        self._orig = u._write
        u._write = self._hook

    def _hook(self, frame, force=False):
        self.n += 1
        return self._orig(frame, force=force)


def test_stop_hard_ramps_axes_back_to_neutral():
    """stop_hard() 必须把四个轴字节**真的**带回中位（不是发一帧就算）。"""
    u = U.UartController(sim=True)
    mid = S.comm.frame.axis_mid
    for _ in range(5):
        u.send_dof(0.0, 0.0, 0.0, 0.4, force=True)      # 持续右转
        time.sleep(0.04)                                # 让 ramp 真的走一点
    assert any(int(a) != mid for a in u._axes[:4]), \
        "前提：此时轴应偏离中位（实际 %s）" % u._axes[:4]
    u.stop_hard(verify=False)
    assert all(int(a) == mid for a in u._axes[:4]), \
        "stop_hard 后轴字节应全回中位，实际 %s" % u._axes[:4]
    assert list(u._current_frame()[1:5]) == [mid] * 4, "实际发出的帧也必须是中性"
    u.close()


def test_close_hard_stops_the_boat_when_moving(monkeypatch):
    """兜底：带着非零舵直接 close() → 关串口前自动硬停（否则船一直转）。"""
    monkeypatch.setitem(S.comm.ramp, "speed_per_s", 5000.0)
    u = U.UartController(sim=True)
    mid = S.comm.frame.axis_mid
    u.send_dof(0.0, 0.0, 0.0, -0.45, force=True)
    time.sleep(0.02)                     # ramp 要真实时间才会动（首帧 dt≈0）
    u.send_dof(0.0, 0.0, 0.0, -0.45, force=True)
    assert any(int(a) != mid for a in u._axes[:4]), \
        "前提：轴应偏离中位（实际 %s）" % u._axes[:4]
    u.close()
    assert all(int(a) == mid for a in u._axes[:4]), \
        "close() 必须先把轴带回中位再关串口，实际 %s" % u._axes[:4]




def test_stop_hard_reports_not_stopped_when_yaw_keeps_changing(monkeypatch):
    """遥测 yaw 仍在变（船真的没停）→ stop_hard 返回 False，不做假确认。"""
    monkeypatch.setitem(S.comm.ramp, "speed_per_s", 5000.0)
    u = U.UartController(sim=True)
    u.stop_hard(verify=False)

    class _Drift(object):
        def __init__(self):
            self.yaw_deg = 0.0

        def tick(self):
            self.yaw_deg = (self.yaw_deg or 0.0) + 30.0     # 每帧涨 30° = 明显还在转

    d = _Drift()
    u.telemetry = d
    orig = u.send_motion

    def _send(name=None, force=False):
        d.tick()
        return orig(name=name, force=force)

    u.send_motion = _send
    assert u.stop_hard(verify=True, settle_s=0.1, max_extra=1) is False




# ---------------------------------------------------------------- 轴饱和看门狗
class _FakeAxesUart(object):
    """只给看门狗用的最小对象：轴字节 + estop 计数。"""

    def __init__(self):
        self._axes = [128] * 7
        self.estops = 0
        self._ser = None

    def estop(self):
        self.estops += 1


def test_axis_watchdog_trips_when_one_axis_is_held_same_direction():
    """★ **轴饱和看门狗**（用户 2026-09-27 定）：同一轴、同一方向连续发轴 ≥ max_same_dir_s"""
    u = _FakeAxesUart()
    trips = []
    w = U._AxisWatchdog(u, max_same_dir_s=0.15, min_off_b=13, poll_ms=20,
                        on_trip=lambda *a: trips.append(a), log=lambda *a: None)
    w.start()
    u._axes[0] = 128 + 38              # yaw 一直朝一个方向（0.30 DOF = 转向满舵）
    t0 = time.monotonic()
    while time.monotonic() - t0 < 2.0 and not trips:
        time.sleep(0.01)
    w.stop()
    assert trips, "同一方向持续发 yaw 必须触发看门狗"
    assert trips[0][0] == "yaw", trips[0]
    assert u.estops == 1, "触发时必须**先硬停**（否则强制退出后船锁在最后那个非中位字节上）"


def test_axis_watchdog_ignores_reversals_deadzone_and_short_pulses():
    """反向对照：**反向交替 / 幅度在判据以内 / 时长不够** 一律不许触发（保护不能误杀正常动作）。"""
    u = _FakeAxesUart()
    trips = []
    w = U._AxisWatchdog(u, max_same_dir_s=0.4, min_off_b=13, poll_ms=20,
                        on_trip=lambda *a: trips.append(a), log=lambda *a: None)
    w.start()
    u._axes[0] = 128 + 12              # ① 幅度 ±12 字节 < min_off_b ⇒ 不算"在发轴"
    time.sleep(0.5)
    t0 = time.monotonic()              # ② 同方向但每 100ms 反向一次 ⇒ 连续同向从未到 0.4s
    while time.monotonic() - t0 < 0.6:
        u._axes[0] = 128 + (25 if int((time.monotonic() - t0) * 10) % 2 == 0 else -25)
        time.sleep(0.05)
    u._axes[0] = 128                   # 清计时
    time.sleep(0.1)
    u._axes[0] = 128 + 25              # ③ 同方向只持续 0.15s < 0.4s
    time.sleep(0.15)
    u._axes[0] = 128
    time.sleep(0.3)
    w.stop()
    assert not trips, "正常动作被误杀：%s" % trips


def test_axis_watchdog_only_watches_yaw():
    """★ **只盯 yaw**（用户 2026-09-27 明确："看门狗只管 yaw 轴的，剩下的不能管"）。"""
    u = _FakeAxesUart()
    trips = []
    w = U._AxisWatchdog(u, max_same_dir_s=0.15, min_off_b=13, poll_ms=20,
                        on_trip=lambda *a: trips.append(a), log=lambda *a: None)
    assert w.axes == (0,) and w.axis_names() == ("yaw",), (w.axes, w.axis_names())
    w.start()
    # 三个非 yaw 轴全部长时间同向（远超阈值）——一个都不许触发
    u._axes[1], u._axes[2], u._axes[3] = 128 + 38, 128 + 38, 128 + 38
    time.sleep(0.6)
    w.stop()
    assert not trips, "非 yaw 轴被误杀（用户定：剩下的不能管）：%s" % trips
    # 对照：同一时刻把 yaw 也顶到同向 ⇒ 立刻触发（证明看门狗本身在工作）
    u2 = _FakeAxesUart(); trips2 = []
    w2 = U._AxisWatchdog(u2, max_same_dir_s=0.15, min_off_b=13, poll_ms=20,
                         on_trip=lambda *a: trips2.append(a), log=lambda *a: None)
    w2.start()
    u2._axes[1], u2._axes[2], u2._axes[3] = 128 + 38, 128 + 38, 128 + 38
    u2._axes[0] = 128 - 38
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.5 and not trips2:
        time.sleep(0.01)
    w2.stop()
    assert trips2 and trips2[0][0] == "yaw", trips2


def test_axis_watchdog_axes_come_from_cfg():
    """盯哪些轴由 `comm.watchdog.axes` 决定（缺省只 yaw；乱写轴名忽略并告警）。"""
    # 而**代码默认**永远只盯 yaw（这才是"配不上也不能变宽"的保证）。
    _cfg_axes = S.get("comm.watchdog.axes", None)
    assert _cfg_axes in (None, ["yaw"]), _cfg_axes
    assert U._AxisWatchdog(_FakeAxesUart(), log=lambda *a: None).axis_names() == ("yaw",)
    w = U._AxisWatchdog(_FakeAxesUart(), axes=["yaw", "sway"], log=lambda *a: None)
    assert w.axis_names() == ("yaw", "sway"), w.axis_names()
    assert U._AxisWatchdog(_FakeAxesUart(), axes=["pitch"], log=lambda *a: None).axes == ()


def test_axis_watchdog_names_match_dof_map():
    """看门狗报的轴名必须与 `comm.dof_map` 的字节序一致（不然现场日志会指错轴）。"""
    want = {int(v["axis"]): k for k, v in S.comm.dof_map.items()}
    assert [U._AxisWatchdog.AXES[i] for i in range(4)] == [want[i] for i in range(4)]


def test_axis_watchdog_auto_start_policy(monkeypatch):
    """开机策略：SIM 默认**不开**（免得 os._exit 打断用例）；`comm.watchdog.enable=false` 与"""
    monkeypatch.delenv("AUV_WATCHDOG", raising=False)
    assert U.UartController(sim=True)._watchdog is None, "SIM 默认不该开看门狗"
    monkeypatch.setenv("AUV_WATCHDOG", "0")
    assert U.UartController(sim=True)._watchdog is None
    monkeypatch.setitem(S.comm, "watchdog", S.Y(dict(S.comm.get("watchdog", {}), enable=False)))
    monkeypatch.delenv("AUV_WATCHDOG", raising=False)
    u = U.UartController(sim=True)
    u.sim = False                      # 假装真串口（不真开串口）：配置关 ⇒ 仍然不开
    u._watchdog = None
    u._start_watchdog()
    assert u._watchdog is None, "comm.watchdog.enable=false 必须真的关掉"
    monkeypatch.setenv("AUV_WATCHDOG", "1")   # 环境变量优先级最高
    u._start_watchdog()
    assert isinstance(u._watchdog, U._AxisWatchdog)
    assert u._watchdog.axis_names() == ("yaw",), u._watchdog.axis_names()
    u._watchdog.stop()
