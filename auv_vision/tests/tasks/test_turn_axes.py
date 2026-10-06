# -*- coding: utf-8 -*-
"""tests/tasks/test_turn_axes.py — 指定角度轴（协议 byte[7]：yaw/pitch/roll）与"只抬不降"的任务级限深下限。

口径来源：`doc/记录/README_COMMUNICATION.md` §2（帧布局）/§2.1（相对角度编码）/§2.2（轴方向与目标）
/§2.4（取消）/§6-1（执行帧必须显式覆盖 byte[7]）。
（详细用法、判据与实测见 doc/注释历史.md）"""
from __future__ import annotations

import sys

import pytest

import base.cfg.settings as S
import base.hw.telemetry as T
from base.hw import uart as U

from common.motion.turn_deg import TurnCore, turn


def _hex(b):
    return " ".join("%02X" % x for x in b)


# --------------------------------------------------------------------------- #
# 帧层：与协议文档给出的示例逐字节一致
# --------------------------------------------------------------------------- #
def test_turn_frames_match_the_protocol_examples():
    """**执行帧逐字节等于文档 §4 的示例**（含 pitch/roll 两条新轴）。"""
    mid = S.comm.frame.axis_mid
    assert mid == 0x80, "示例帧假定四个手动轴回中 = 0x80，cfg 改了要同步改本用例的期望"
    assert _hex(U.build_turn_frame(90.0, 1, "yaw")) == "A5 80 80 80 80 A5 01 01 DF FF 01"
    assert _hex(U.build_turn_frame(-90.0, 2, "yaw")) == "A5 80 80 80 80 A5 01 01 A0 00 02"
    assert _hex(U.build_turn_frame(30.0, 3, "pitch")) == "A5 80 80 80 80 A5 01 02 CA AA 03"
    assert _hex(U.build_turn_frame(-30.0, 4, "roll")) == "A5 80 80 80 80 A5 01 03 B5 55 04"
    # 取消：执行位 0 + 角度低字节 1 + 编号（协议 §2.4，不依赖 byte[7]）
    cancel = U.build_turn_cancel_frame(4)
    assert cancel[8] == 0 and cancel[9] == 1 and cancel[10] == 4
    assert U.build_turn_cancel_frame(0)[10] == 0, "编号 0 = 同步并取消所有任务"


def test_execution_frame_writes_the_axis_byte_explicitly(monkeypatch):
    """**执行帧自己写 byte[7]**（协议 §6-1）：把中性帧的 byte[7] 改成 0，转向照样能发起。"""
    monkeypatch.setitem(S.comm.frame.aux_axis, 6, 0)          # 有人把普通帧的 byte[7] 归 0
    assert U.build_neutral_frame()[7] == 0
    assert U.build_turn_frame(90.0, 1, "yaw")[7] == 1, "yaw 执行帧不能被中性默认值拖走"
    assert U.build_turn_frame(30.0, 2, "pitch")[7] == 2
    assert U.build_turn_frame(-30.0, 3, "roll")[7] == 3


def test_neutral_frames_follow_cfg_and_turns_do_not(monkeypatch):
    """普通帧 byte[7] 跟随 cfg（协议 §6-2 设 0）；**转向与它解耦** ⇒ 改它不影响任何转向。"""
    assert U.neutral_axis_bytes()[6] == 0, "普通/中性帧 byte[7] 按协议 §6-2 设 0"
    assert U.build_neutral_frame()[7] == 0
    for i in (1, 2, 3):
        assert U.build_turn_frame(30.0, 1, i)[7] == i
    monkeypatch.setitem(S.comm.frame.aux_axis, 6, 1)     # 即使有人改回 1（或任意值）
    assert U.build_turn_frame(30.0, 1, "yaw")[7] == 1, "yaw 执行帧不能被中性默认值拖走"
    assert U.build_turn_frame(-30.0, 1, "roll")[7] == 3


def test_turn_axis_id_maps_names_and_rejects_junk():
    assert [U.turn_axis_id(a) for a in ("yaw", "pitch", "roll")] == [1, 2, 3]
    assert [U.turn_axis_id(a) for a in (1, 2, 3)] == [1, 2, 3]
    assert U.turn_axis_id(" PITCH ") == 2, "轴名应大小写/空白无关"
    for bad in ("pan", "0", 0, 4, -1, None, "1.5"):
        with pytest.raises((ValueError, TypeError)):
            U.turn_axis_id(bad)


def test_angle_encoding_is_unchanged():
    """编码公式不因轴而变：[-180,180]→[0,32767]，四舍五入（协议 §2.1）。"""
    assert U.encode_turn_angle(0.0) == 16384
    assert U.encode_turn_angle(-180.0) == 0
    assert U.encode_turn_angle(180.0) == 32767
    with pytest.raises(ValueError):
        U.encode_turn_angle(180.1)


# --------------------------------------------------------------------------- #
# 控制器层：轴真的上了线
# --------------------------------------------------------------------------- #
def _sim_uart(monkeypatch, sent):
    """SIM 控制器 + 关 ramp + 截获实际发出的帧（同 tests/platform/test_base.py 的做法）。"""
    monkeypatch.setitem(S.comm.ramp, "speed_per_s", 0.0)
    monkeypatch.setitem(S.comm.depth_guard, "enable", True)
    u = U.UartController(sim=True)
    monkeypatch.setattr(u, "_write", lambda frame, force=False:
                        (sent.append(bytes(frame)), True)[1])
    return u


def test_request_turn_puts_the_axis_byte_on_the_wire(monkeypatch):
    """`request_turn(deg, axis=2)`：重发的那一帧 byte[7]=2、执行位置 1、编号一致。"""
    sent = []
    u = _sim_uart(monkeypatch, sent)
    assert u.request_turn(30.0, axis="pitch") is True
    assert u._turn_frame[7] == 2 and (u._turn_frame[8] & 0x80)
    # 同步阶段：先写同步帧；喂一帧"编号 0/未完成"的遥测后进入执行阶段 → 写执行帧
    u.feed_telemetry(T.build_telemetry_frame(turn_id=0, turn_done=False),
                     now_ms=U._now_ms())
    u._turn_retry_at = 0.0
    u.poll_turn_complete()
    execs = [f for f in sent if f[8] & 0x80 and f[10] == u._turn_id]
    assert execs, "执行帧没有上线: %s" % [hex(f[7]) for f in sent]
    assert execs[-1][7] == 2, "byte[7] 必须是 pitch(2)，实际 %d" % execs[-1][7]
    assert execs[-1][9] == U.encode_turn_angle(30.0) & 0xFF
    u.close()


def test_request_turn_rejects_a_bad_axis_before_touching_the_axis_state(monkeypatch):
    sent = []
    u = _sim_uart(monkeypatch, sent)
    with pytest.raises(ValueError):
        u.request_turn(30.0, axis="pan")
    assert u._turn_phase is None and not sent, "轴非法不许进入转向状态、不许发帧"
    u.close()


def test_cancel_frame_does_not_depend_on_the_axis_byte(monkeypatch):
    """取消按编号识别（协议 §2.4）：cancel 帧的 byte[7] 不参与判定。"""
    sent = []
    u = _sim_uart(monkeypatch, sent)
    u.request_turn(30.0, axis="roll")
    u.feed_telemetry(T.build_telemetry_frame(turn_id=0, turn_done=False),
                     now_ms=U._now_ms())
    u._turn_retry_at = 0.0
    u.poll_turn_complete()                     # sync → run（此时才带上本次编号）
    assert u._turn_phase == "run"
    u.cancel_turn()
    cancels = [f for f in sent if not (f[8] & 0x80) and f[9] == 1 and f[10] == u._turn_id]
    assert cancels, "取消帧没有上线: %s" % [_hex(f) for f in sent]
    assert u._turn_phase is None and u._turn_frame is None
    u.close()


# --------------------------------------------------------------------------- #
# TurnCore / turn()：yaw 老路径不变、新轴不许静默回落到 yaw
# --------------------------------------------------------------------------- #
class _Lower(object):
    """假下位机，记录 (编号, 相对角, 轴)。`accept_axis=False` = 只认一个角度参数的旧对象。"""

    def __init__(self, accept_axis=True):
        self.accept_axis = accept_axis
        self.reqs = []
        self._id = 0
        self._pending = 0
        self.estop_active = False

    def request_turn(self, angle_deg, turn_id=None, **kw):
        if not self.accept_axis and kw:
            raise TypeError("request_turn() got an unexpected keyword argument %r" % list(kw))
        self._id = int(turn_id) if turn_id is not None else self._id + 1
        self.reqs.append((self._id, float(angle_deg), kw.get("axis", 1)))
        self._pending = 2
        return True

    def poll_turn_complete(self):
        if self._pending <= 0:
            return False
        self._pending -= 1
        return self._pending <= 0

    def cancel_turn(self):
        self._pending = 0

    def stop_hard(self, *a, **kw):
        return True

    def send_dof(self, *a, **kw):
        pass


def _run(core, lower, frames=50):
    now = 0
    for _ in range(frames):
        core.step(now, lower)
        if core.finished():
            break
        now += 50
    return core


def test_yaw_turn_keeps_the_single_argument_call():
    """**既有 yaw 路径参数表不变**：老测试替身/老下位机对象（只认一个角度参数）照样能转。"""
    low = _Lower(accept_axis=False)
    core = _run(TurnCore(deg=90.0, left=True, log=lambda *a: None), low)
    assert core.state == TurnCore.DONE
    assert low.reqs == [(1, -90.0, 1)]


def test_pitch_turn_is_signed_and_carries_axis_two():
    """pitch/roll：`left=False` → 正角（下位机 IMU 该轴增大），轴 = 2/3。"""
    low = _Lower()
    core = _run(TurnCore(deg=30.0, left=False, log=lambda *a: None, axis="pitch"), low)
    assert core.state == TurnCore.DONE
    assert low.reqs == [(1, 30.0, 2)]
    low2 = _Lower()
    _run(TurnCore(deg=30.0, left=True, log=lambda *a: None, axis="roll"), low2)
    assert low2.reqs == [(1, -30.0, 3)]


def test_unsupported_axis_aborts_instead_of_falling_back_to_yaw():
    """**安全属性**：下位机/替身不认这个轴 → 直接放弃（aborted），绝不退回 yaw 瞎转。"""
    low = _Lower(accept_axis=False)
    core = _run(TurnCore(deg=30.0, left=False, log=lambda *a: None, axis="pitch"), low)
    assert core.state == TurnCore.ABORTED and core.why == "send_failed"
    assert low.reqs == [], "不许发出任何转动请求"


def test_script_entry_forwards_the_axis():
    low = _Lower()
    assert turn(low, deg=30.0, left=False, timeout=5.0, log=lambda *a: None,
                axis="pitch") == 0
    assert low.reqs and low.reqs[0][1:] == (30.0, 2)


def test_manual_cli_can_target_pitch(monkeypatch):
    """手动入口新增 `--axis`（通道选择，不是运动参数）；默认仍是 yaw。"""
    import sys as _sys
    from common.motion import turn_deg as TD

    low = _Lower()

    class _Ctl(object):
        sim = False

        def __init__(self, *a, **kw):
            pass

        def close(self):
            pass

        def __getattr__(self, name):
            return getattr(low, name)

    monkeypatch.setattr("base.hw.uart.UartController", _Ctl)
    monkeypatch.setattr("sys.argv", ["turn_deg.py", "--deg", "30", "--dir", "right",
                                     "--axis", "pitch"])
    assert TD.main() == 0
    assert [(r[1], r[2]) for r in low.reqs] == [(30.0, 2)], low.reqs

    low.reqs = []
    monkeypatch.setattr("sys.argv", ["turn_deg.py", "--deg", "45", "--dir", "right"])
    assert TD.main() == 0
    assert [(r[1], r[2]) for r in low.reqs] == [(45.0, 1)], "不给 --axis 时默认 yaw"


# --------------------------------------------------------------------------- #
# 任务级限深下限：**只抬不降**
# --------------------------------------------------------------------------- #
def _depth_uart(monkeypatch, sent, depth):
    u = _sim_uart(monkeypatch, sent)
    u.feed_telemetry(T.build_telemetry_frame(depth), now_ms=U._now_ms())
    assert u.depth_m == pytest.approx(depth)
    return u


def test_extra_floor_blocks_surfacing_at_a_higher_threshold(monkeypatch):
    """任务级下限抬高 → 在 cfg 下限之上也禁止上浮（这是抬头动作要的那件事）。"""
    sent = []
    u = _depth_uart(monkeypatch, sent, 0.70)
    base = float(S.comm.depth_guard.min_depth_m)
    assert u.effective_min_depth_m == pytest.approx(base)
    u.send_dof(heave=0.5, force=True)
    assert u.dof_out[2] == pytest.approx(0.5), "0.70m 比 cfg 下限深 → 本来放行"

    assert u.set_extra_min_depth(0.90) == pytest.approx(0.90)
    u.send_dof(heave=0.5, force=True)
    assert u.dof_out[2] == 0.0, "抬到 0.90m 后，0.70m 深度不许上浮"
    assert u.guard_active is True
    u.close()


def test_extra_floor_can_never_loosen_the_site_floor(monkeypatch):
    """**不变量**：传 0 / 更小 / 非法值都不会放宽保护 —— 有效下限恒 ≥ cfg 的 0.55。"""
    sent = []
    u = _depth_uart(monkeypatch, sent, 0.30)
    base = float(S.comm.depth_guard.min_depth_m)
    for req in (0.0, -5.0, 0.10, None, "abc", float("nan")):
        assert u.set_extra_min_depth(req) == pytest.approx(base), "req=%r 放宽了保护" % (req,)
        u.set_extra_min_depth(req)
        u.send_dof(heave=0.5, force=True)
        assert u.dof_out[2] == 0.0, "0.30m 比 cfg 下限浅，任何情况下都不许上浮"
    u.close()


def test_extra_floor_is_ignored_when_the_guard_is_off(monkeypatch):
    """`depth_guard.enable: false` 时任务级下限也无效（保护整体关闭，行为与改动前一致）。"""
    sent = []
    u = _depth_uart(monkeypatch, sent, 0.20)
    monkeypatch.setitem(S.comm.depth_guard, "enable", False)   # 必须在建控制器之后关
    u.set_extra_min_depth(0.90)
    u.send_dof(heave=0.5, force=True)
    assert u.dof_out[2] == pytest.approx(0.5)
    u.close()


def test_estop_and_neutral_do_not_reset_the_extra_floor(monkeypatch):
    """`neutral()`/`estop` 不改任务级下限：任务的收紧只能由任务自己撤回。"""
    sent = []
    u = _depth_uart(monkeypatch, sent, 0.70)
    u.set_extra_min_depth(0.90)
    u.neutral()
    assert u.effective_min_depth_m == pytest.approx(0.90)
    assert u.set_extra_min_depth(0.0) == pytest.approx(float(S.comm.depth_guard.min_depth_m))
    u.close()
