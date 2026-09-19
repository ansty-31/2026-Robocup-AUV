# -*- coding: utf-8 -*-
"""tests/test_base.py — base 层：settings 加载 / 11B 串口帧与控制器 / 遥测上行与限深保护 / 相机工厂。

只跑 SIM 路径：UartController(sim=True)、相机 type 强制 sim，不碰串口与摄像头。
遥测/限深用 `UartController.feed_telemetry()` 或假串口喂字节，不发真帧。
"""
import os
import time

import numpy as np
import pytest

import base.settings as S
import base.telemetry as T
from base import uart as U
from base.camera import SimCamera, create_camera


def test_settings_loads_real_yaml_values():
    """配置真的从 cfg/*.yaml 读进来了（抽查几个关键真实值），缺路径返回 None。"""
    assert S.comm.gate.pass_target == 1
    # ⚠️ 这几个是"板端现场值"（2026-09-18 用户定：以板端为准写回本地），会随现场调参变，
    #    所以这里只断言**类型与不变量**，不钉死数值（钉死 = 每次调参都要改测试 = 假红）。
    assert 0.0 < float(S.vision.gate.keypoint.conf_thr) <= 1.0
    assert isinstance(bool(S.vision.gate.kpt_mem.enable), bool)
    # 不变量：回忆点置信度必须低于任务侧阈值，否则"回忆点"会被判成真实可见（见 vision.yaml 注释）
    if S.vision.gate.kpt_mem.enable:
        assert float(S.vision.gate.kpt_mem.recall_conf) < \
            float(S.vision.gate.keypoint.conf_thr)
    assert S.vision.gate.pnp.z_max == 15.0
    assert S.comm.frame.header == 0xA5
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
    assert U.neutral_axis_bytes() == [128, 128, 128, 128, 165, 1, 1]

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


def test_uart_frame_layout_is_11_bytes():
    """11B 帧布局 = 0xA5 + 7 轴 + 3 键，且 DOF 落到正确轴位。"""
    neutral = U.build_neutral_frame()
    assert len(neutral) == 11
    assert neutral[0] == S.comm.frame.header == 0xA5
    assert list(neutral[1:8]) == U.neutral_axis_bytes()
    assert list(neutral[8:11]) == list(S.comm.frame.btn_values) == [0, 0, 0]

    fwd = U.build_frame_from_dof(surge=1.0)
    assert len(fwd) == 11
    assert fwd[0] == 0xA5
    assert fwd[1] == S.comm.frame.axis_mid      # yaw 未动 → 中位
    assert fwd[2] == 255                        # surge 满速
    assert fwd[8:] == bytes(S.comm.frame.btn_count)   # 3 个按键字节全 0


def test_uart_controller_sim_setmotion_neutral_estop():
    """SIM 控制器：预设→DOF 目标、未知预设回 stop、neutral 回中、急停锁死运动帧。"""
    u = U.UartController(sim=True)
    assert u.sim is True
    assert u.estop_active is False
    assert u.dof_target == (0.0, 0.0, 0.0, 0.0)

    assert u.set_motion("forward_fast") is True
    assert u.last_motion == "forward_fast"
    assert u.dof_target == (1.0, 0.0, 0.0, 0.0)     # 取 comm.motion.presets
    assert tuple(S.comm.motion.presets["forward_fast"]) == (1.0, 0.0, 0.0, 0.0)

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

    # down 配置本身即 sim
    assert S.vision.camera.down.type == "sim"
    down = create_camera("down")
    d = down.read()
    assert d.shape == (S.vision.camera.down.height,
                       S.vision.camera.down.width, 3)


# ---------------------------------------------------------------------------
# 下位机遥测上行（14B 0xAA55 + 深度/姿态 + 校验和）
# ---------------------------------------------------------------------------
def test_telemetry_frame_roundtrip_and_resync():
    """造帧↔拆帧往返；半帧/粘包/错位/校验错可重同步（串口按字节切分是常态）。"""
    f = T.build_telemetry_frame(0.42, 0.30, 1.5, -2.25, 88.0)
    assert len(f) == T.TEL_LEN == 14
    assert f[:2] == T.TEL_HEADER == b"\xAA\x55"
    assert T.check_telemetry(f) is True
    assert T.decode_telemetry(f) == pytest.approx((0.42, 0.30, 1.5, -2.25, 88.0))

    # 半帧不产出；补齐后才产出
    buf = bytearray(f[:6])
    assert T.parse_telemetry_frames(buf) == []
    buf.extend(f[6:])
    assert T.parse_telemetry_frames(buf)[0] == pytest.approx(
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


def test_telemetry_receiver_freshness_and_reset():
    """TelemetryReceiver：最新值、新鲜度(超时)、坏帧不污染、reset。"""
    rx = T.TelemetryReceiver()
    assert rx.depth_m is None
    assert rx.fresh(500, now_ms=1000) is False          # 从未收到 → 不可用
    assert rx.age_ms(1000) is None

    assert rx.feed(T.build_telemetry_frame(0.5, 0.4, 1.0), now_ms=1000) == 1
    assert rx.depth_m == pytest.approx(0.5)
    assert rx.target_m == pytest.approx(0.4)
    assert rx.frames == 1 and rx.bad == 0
    assert rx.age_ms(1200) == 200
    assert rx.fresh(500, now_ms=1200) is True
    assert rx.fresh(500, now_ms=1600) is False          # 超过 stale_ms → 不新鲜

    bad = bytearray(T.build_telemetry_frame(0.05))
    bad[-1] ^= 0xFF
    assert rx.feed(bytes(bad), now_ms=2000) == 0
    assert rx.bad >= 1
    assert rx.depth_m == pytest.approx(0.5)             # 坏帧绝不更新深度

    rx.reset()
    assert rx.depth_m is None and rx.frames == 0 and rx.age_ms() is None


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


def test_uart_drain_rx_parses_depth_from_serial(monkeypatch):
    """真串口路径：发帧时顺带收遥测 → uart.depth_m（不用喂 feed_telemetry）。"""
    u = U.UartController(sim=True)
    monkeypatch.setattr(u, "sim", False)
    monkeypatch.setattr(u, "_ser", _FakeSerial(
        T.build_telemetry_frame(0.37, 0.30) + b"\xAA"))     # 末尾半个帧头
    assert u.depth_m is None
    u.send_dof(surge=0.2, force=True)                        # _write → _drain_rx
    assert u.depth_m == pytest.approx(0.37)
    assert u.telemetry.frames == 1
    assert len(u._ser.written) == 1                          # 帧照常发出
    assert u.depth_fresh() is True
    assert u._drain_rx() == 0                                # 没有新数据 = 空操作
    u.close()


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
        u.set_motion("up")                   # 发帧前收遥测 → 深度不足 → 上浮被压掉
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
        u.set_motion("up", force=True)
        assert u.depth_m == pytest.approx(0.60)
        assert u.dof_out[2] == 1.0
    finally:
        u.close()
        os.close(master)
        os.close(slave)


# ---------------------------------------------------------------------------
# 限深保护：深度不足时禁止上浮（机身不得冒出水面）
# ---------------------------------------------------------------------------
def _guard_uart(monkeypatch, sent, dof_comp=False):
    """SIM 控制器 + 关 ramp（轴直通目标）+ 截获实际发出的帧。

    `dof_comp`：默认**关掉下潜放大**，让"限深保护"的用例只测保护本身（两者互不干扰）；
    要测下潜放大的用例传 `dof_comp=True`。
    """
    monkeypatch.setitem(S.comm.ramp, "speed_per_s", 0.0)
    monkeypatch.setitem(S.comm.depth_guard, "enable", True)
    monkeypatch.setitem(S.comm.dof_comp, "enable", bool(dof_comp))
    u = U.UartController(sim=True)
    monkeypatch.setattr(u, "_write", lambda frame, force=False:
                        (sent.append(bytes(frame)), True)[1])
    return u


def test_depth_guard_blocks_surfacing_only(monkeypatch):
    """深度 ≤ min_depth_m：上浮 >0 被清零，surge/sway/yaw 照旧；下潜照常。

    阈值**从配置读**（不写死）：用户把下限定在 0.55，写死 0.3 会随配置变化而失效。
    """
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
    """**下潜动力单独放大**（底层共用，撞球/过门都吃）：只乘 heave<0，其余通道原样。

    背景：DOF→字节→下位机 `RC_Matching*`（映射≤35→0）使实际推力远小于 DOF 数字
    （0.20→20%、0.30→30%、0.60→60%，见 gate 文档 §1.1）→ 下潜偏弱就把这一路放大。
    """
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


def test_dive_boost_switch_off(monkeypatch):
    """开关可关：`dof_comp.enable=false` → 下潜原样下发。"""
    sent = []
    u = _guard_uart(monkeypatch, sent, dof_comp=True)
    monkeypatch.setitem(S.comm.dof_comp, "enable", False)
    u.send_dof(heave=-0.2, force=True)
    assert u.dof_out[2] == pytest.approx(-0.2)
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
# 2026-09-18 用户水里实测：转角脚本最后只发了一帧 neutral 就关串口 → 船一直转。
# 根因：_ramp_step 是**字节级平滑**，neutral() 的 force 只绕过心跳节流、不绕过 ramp；
#       而下位机没有"无帧超时停车"（comm.yaml heartbeat 注释）→ 锁在最后一个非零字节上。
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
    """stop_hard() 必须把四个轴字节**真的**带回中位（不是发一帧就算）。

    注意：ramp 是"字节/秒"，需要**真实时间**流逝（`step_max=speed×dt`），
    所以这里必须 sleep —— 紧凑循环里 dt≈0，轴根本不会动。
    """
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
    """兜底：带着非零舵直接 close() → 关串口前自动硬停（否则船一直转）。

    这里把 ramp 调快（5000 字节/秒）让用例快跑；"按 ramp 速率发够帧"本身由
    上一个用例用真实 ramp 验证。
    """
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


def test_close_sends_nothing_extra_when_already_neutral(monkeypatch):
    """已在中位时 close() **一帧都不多发**（不影响既有行为/用例）。"""
    monkeypatch.setitem(S.comm.ramp, "speed_per_s", 5000.0)
    u = U.UartController(sim=True)
    u.stop_hard(verify=False)
    c = _CountWrite(u)
    u.close()
    assert c.n == 0, "已在中位时 close() 不该再发帧，实际发了 %d 帧" % c.n


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


def test_uart_close_is_idempotent():
    u = U.UartController(sim=True)
    u.send_dof(0.0, 0.0, 0.0, 0.3, force=True)
    u.close()
    u.close()                                   # 第二次不应抛异常
