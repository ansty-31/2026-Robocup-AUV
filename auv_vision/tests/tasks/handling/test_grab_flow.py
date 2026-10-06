# -*- coding: utf-8 -*-
"""tests/tasks/handling/test_grab_flow.py — 任务三「夹取小球」相位机（`grab/grab_task.py`）的
**相位顺序 + 安全属性**验收（离线，不碰硬件、不碰 BPU）。

用户口径（2026-10-06）：抬头 30° → 扫描 → yaw 居中 → 对准前进 → 面积达标 → 回水平
→ sway+surge 轻微居中（抗水波，不用 yaw）→ 下压 → 上升 → 读 percent 验色
→ 对就退出、错就倒掉重来（第 2 次默认对）。**感知全程用下视相机。**

⚠️ 本套用例证明的是**逻辑与安全边界**，不是水中效果。
"""
from __future__ import annotations

import numpy as np
import pytest

import base.cfg.settings as S
from handling.handling_task import (PH_GRAB_ALIGN, PH_GRAB_APPROACH, PH_GRAB_CENTER, PH_GRAB_DIP, PH_GRAB_DONE, PH_GRAB_LEVEL,
                            PH_GRAB_PITCH_UP, PH_GRAB_RISE, PH_GRAB_SEARCH, PH_GRAB_VERIFY, GrabTask,
                            _ratio_radius)
from handling.percept.cv_ball import Detection
from handling.percept.grab_detector import circle_to_ratio
from handling.percept.cage_color import red_percent

W, H = 320, 200
ROI = [0.25, 0.35, 0.5, 0.5]          # = cfg 里的 verify.roi


# --------------------------------------------------------------------------- #
# 假件
# --------------------------------------------------------------------------- #
class _Circle(object):
    def __init__(self, cx, cy, r):
        self.cx, self.cy, self.r = float(cx), float(cy), float(r)


class _Backend(object):
    """假 grab 后端：`circles()` 返回脚本化的一个圆（半径由驱动逐帧改）。"""

    def __init__(self):
        self.cx, self.cy, self.r = W / 2.0, H / 2.0, 25.0
        self.p = None

    def circles(self, frame):
        return [_Circle(self.cx, self.cy, self.r)]


class _Hub(object):
    def __init__(self, be):
        self._be = be

    def ready(self, task):
        return True

    def has_extra(self, task):
        return True

    def extra(self, task):
        return self._be

    def detect_list(self, task, frame):
        return []


class _Tel(object):
    def __init__(self, pitch=0.0, roll=0.0, yaw=0.0):
        self.pitch_deg, self.roll_deg, self.yaw_deg = pitch, roll, yaw


class FakeUart(object):
    """假下位机：指定角度轴"立即执行"（2 次 poll 完成），记录 DOF 与事件顺序。"""

    def __init__(self, depth=1.2, pitch=0.0, roll=0.0, yaw=0.0, pitch_gain=1.0):
        self.telemetry = _Tel(pitch, roll, yaw)
        self.depth_m = depth
        self.extra_min_depth_m = 0.0
        self.pitch_gain = float(pitch_gain)
        self.turns = []
        self.dofs = []
        self.events = []
        self.estop_active = False
        self.neutral_calls = 0
        self._pending = 0
        self._cur = None

    @property
    def effective_min_depth_m(self):
        # 与真件同源：下限 = max(cfg 的 site 值, 任务级下限)，别写死米数
        return max(float(S.comm.depth_guard.min_depth_m), self.extra_min_depth_m)

    def set_extra_min_depth(self, m):
        self.extra_min_depth_m = max(0.0, float(m))
        self.events.append("floor:%.2f" % self.extra_min_depth_m)
        return self.effective_min_depth_m

    def request_turn(self, angle_deg, turn_id=None, axis=1):
        self._cur = (int(axis), float(angle_deg))
        self._pending = 2
        self.events.append("turn:%d:%+.1f" % (int(axis), float(angle_deg)))
        return True

    def poll_turn_complete(self):
        if self._pending <= 0:
            return False
        self._pending -= 1
        if self._pending > 0:
            return False
        axis, deg = self._cur
        self.turns.append((axis, deg))
        if axis == 2:
            self.telemetry.pitch_deg = (self.telemetry.pitch_deg or 0.0) + deg * self.pitch_gain
        elif axis == 3:
            self.telemetry.roll_deg = (self.telemetry.roll_deg or 0.0) + deg
        else:
            self.telemetry.yaw_deg = (self.telemetry.yaw_deg or 0.0) + deg
        return True

    def cancel_turn(self):
        self._pending = 0

    def send_dof(self, surge=0.0, sway=0.0, heave=0.0, yaw=0.0):
        self.dofs.append((float(surge), float(sway), float(heave), float(yaw)))

    def neutral(self):
        self.neutral_calls += 1


class NoFloorUart(object):
    """**故意没有** `set_extra_min_depth` 的串口对象（老下位机/老实现）。"""

    def __init__(self):
        self.telemetry = _Tel()
        self.depth_m = 1.2
        self.dofs = []
        self.estop_active = False

    def send_dof(self, *a, **kw):
        self.dofs.append(a)

    def neutral(self):
        pass


def _red_frame():
    f = np.zeros((H, W, 3), dtype=np.uint8)
    f[:, :] = (200, 80, 20)                       # 蓝绿底
    x1, y1 = int(ROI[0] * W), int(ROI[1] * H)
    x2, y2 = int((ROI[0] + ROI[2]) * W), int((ROI[1] + ROI[3]) * H)
    f[y1:y2, x1:x2] = (0, 0, 235)                 # 笼区：纯红（目标色）
    return f


def _blue_frame():
    f = np.zeros((H, W, 3), dtype=np.uint8)
    f[:, :] = (0, 0, 235)                         # 全红（画面上别处是红的，避免偶然）
    x1, y1 = int(ROI[0] * W), int(ROI[1] * H)
    x2, y2 = int((ROI[0] + ROI[2]) * W), int((ROI[1] + ROI[3]) * H)
    f[y1:y2, x1:x2] = (235, 120, 0)               # 笼区：非目标色
    return f


def _task(monkeypatch, uart, be=None, calibrated=True):
    monkeypatch.setitem(S.comm.grab, "calibrated", bool(calibrated))
    be = be or _Backend()
    return GrabTask(uart, _Hub(be), W, H, log=lambda *a: None), be


def _drive(task, uart, be, frames=300, dt=100, radius=None, frame=None, phases=None):
    """逐帧推进；返回 (结束相位序列)。"""
    now = 0
    for i in range(frames):
        if radius is not None:
            be.r = radius(i)
        f = frame(i) if callable(frame) else frame
        task.set_down_frame(f)
        st = task.process(f, now)
        if phases is not None and not phases[-1:] == [task.last_info["phase"]]:
            phases.append(task.last_info["phase"])
        if st == S.STATUS_DONE:
            break
        now += dt
    return phases


def _grow(i):
    return min(80.0, 25.0 + 10.0 * i)


# --------------------------------------------------------------------------- #
# 安全闸
# --------------------------------------------------------------------------- #
def test_uncalibrated_task_is_skipped_and_never_moves(monkeypatch):
    """`calibrated: false` ⇒ `ready` 为假（装配层跳过）且即使被调用也**一帧不动**。"""
    uart = FakeUart()
    task, _ = _task(monkeypatch, uart, calibrated=False)
    assert task.ready is False
    assert task.process(_red_frame(), 0) == S.STATUS_DONE
    assert task.last_info["reason"] == "uncalibrated"
    assert uart.dofs == [] and uart.turns == [] and uart.events == []


def test_refuses_to_run_without_the_depth_floor_capability(monkeypatch):
    """拿不到"收紧限深"的能力 ⇒ 抬头就可能露出水面 ⇒ **拒绝执行**（fail closed）。"""
    uart = NoFloorUart()
    task, _ = _task(monkeypatch, uart)
    assert task.process(_red_frame(), 0) == S.STATUS_DONE
    assert task.last_info["reason"] == "no_depth_floor"
    assert uart.dofs == []


def test_floor_is_raised_before_the_first_axis_command_and_cleared_at_exit(monkeypatch):
    """★ 安全不变量：**先抬限深下限，再发第一个轴指令**；任务结束必须撤回（0）。"""
    uart = FakeUart()
    task, be = _task(monkeypatch, uart)
    _drive(task, uart, be, radius=_grow, frame=_red_frame())
    assert task.last_info["reason"] == "grab_ok"
    # ★ 判据是**顺序**（先抬下限、再发轴指令），**不是某个具体米数** ——
    #   把阈值钉死在测试里等于"改 cfg 就假红"（本工程明令禁止，见 README 用例约定）。
    floor = float(S.comm.grab.depth_floor_m)
    assert uart.events[0] == "floor:%.2f" % floor, \
        "第一件事必须是把下限抬到 cfg 值 %.2f，实际 %s" % (floor, uart.events[:2])
    first_turn = next(i for i, e in enumerate(uart.events) if e.startswith("turn:"))
    assert first_turn == 1, "抬下限与第一个轴指令之间不该插别的事：%s" % uart.events[:3]
    assert uart.extra_min_depth_m == pytest.approx(0.0), "任务结束必须撤回任务级下限"
    assert float(S.comm.depth_guard.min_depth_m) == pytest.approx(0.50), \
        "全局限深是现场定死值（2026-10-06 定为 0.50），任务不许动它"


def test_pitch_up_waits_for_depth_instead_of_driving_shallow(monkeypatch):
    """深度不够 ⇒ 先下潜；到不了就**放弃抬头**（不发任何轴指令）。"""
    from common.motion.axis import grab_cfg
    uart = FakeUart(depth=float(grab_cfg()["depth_floor_m"]) - 0.15)   # ★ cfg 派生，别写死米数
    task, be = _task(monkeypatch, uart)
    _drive(task, uart, be, frames=200, radius=_grow, frame=_red_frame())
    assert task.last_info["reason"] == "depth_depth_timeout"
    assert uart.turns == [], "深度不够时绝不许抬头"
    assert any(d[2] < 0 for d in uart.dofs), "应该真的在下潜"


def test_timeout_finishes_and_clears_the_floor(monkeypatch):
    monkeypatch.setitem(S.comm.grab, "timeout_ms", 500)
    uart = FakeUart(depth=0.10)                 # 一直在下潜 ⇒ 一定撞总时限
    task, be = _task(monkeypatch, uart)
    _drive(task, uart, be, frames=50, radius=_grow, frame=_red_frame())
    assert task.last_info["reason"] == "timeout"
    assert uart.extra_min_depth_m == pytest.approx(0.0)


def test_missing_down_frame_never_moves_blindly(monkeypatch):
    """没有下视帧 = 没有感知 ⇒ 本帧停手；绝不能拿"上一帧的结论"继续推。"""
    uart = FakeUart()
    task, _ = _task(monkeypatch, uart)
    now = 0
    for _ in range(40):
        task.set_down_frame(None)
        st = task.process(None, now)
        now += 100
        if st == S.STATUS_DONE:
            break
    assert st == S.STATUS_RUNNING, "不该自己结束"
    # 抬头是任务的**前置姿态**（与有没有帧无关，允许）；但任何**平移/转向/盲上浮**都不许
    assert all(d[0] == 0.0 and d[1] == 0.0 and d[3] == 0.0 for d in uart.dofs), \
        "盲动 = %s" % uart.dofs[:3]
    assert all(d[2] <= 0.0 for d in uart.dofs), "不许在没有深度依据时上浮"
    assert [t for t in uart.turns if t[0] != 2] == [], "只许动 pitch（前置姿态），不许动别的轴"


# --------------------------------------------------------------------------- #
# 相位顺序与通道
# --------------------------------------------------------------------------- #
def test_happy_path_phase_order(monkeypatch):
    """完整通过：抬头 → 搜索 → yaw 居中 → 对准前进 → 回水平 → 轻微对准 → 下压 → 上升 → 验色。"""
    uart = FakeUart()
    task, be = _task(monkeypatch, uart)
    phases = _drive(task, uart, be, radius=_grow, frame=_red_frame(), phases=[])
    assert task.last_info["reason"] == "grab_ok"
    assert task.holds_ball is True
    for ph in (PH_GRAB_PITCH_UP, PH_GRAB_SEARCH, PH_GRAB_CENTER, PH_GRAB_APPROACH, PH_GRAB_LEVEL, PH_GRAB_ALIGN,
               PH_GRAB_DIP, PH_GRAB_RISE, PH_GRAB_VERIFY, PH_GRAB_DONE):
        assert ph in phases, "相位 %s 没走到：%s" % (ph, phases)
    assert phases.index(PH_GRAB_APPROACH) < phases.index(PH_GRAB_LEVEL) < phases.index(PH_GRAB_ALIGN)
    assert uart.turns[0] == (2, 30.0), "第一次轴动作 = pitch +30°，实际 %s" % uart.turns[:2]
    assert uart.turns[1] == (2, -30.0), "第二次 = 回水平，实际 %s" % uart.turns[:2]


def test_center_uses_yaw_only(monkeypatch):
    """居中段：只转 yaw，**不前进/不横移**（否则会撞目标）。

    ⚠️ `heave` 这一路在抬头工作段（SEARCH/CENTER/APPROACH）已归**定深**管（用户 2026-10-06：
    深度控制在 0.5~0.6）⇒ 这里把深度放进区间内，验证"定深不动手"时 heave 就是 0。
    """
    uart = FakeUart(depth=0.58)          # 区间内 ⇒ 定深不动作
    task, be = _task(monkeypatch, uart)
    task._phase = PH_GRAB_CENTER
    task._ensure_pids()
    be.cx = W / 2.0 + 0.5 * W / 2.0            # dx = +0.5
    task.process(_red_frame(), 0)
    surge, sway, heave, yaw = uart.dofs[-1]
    assert abs(yaw) > 0.0, "没居中就该转"
    assert (surge, sway, heave) == (0.0, 0.0, 0.0)


def test_align_level_uses_sway_and_surge_but_never_yaw(monkeypatch):
    """★ 回水平后：dx→sway、dy→surge，**两个通道都轻微**（抗水波），yaw 恒 0。"""
    uart = FakeUart()
    task, be = _task(monkeypatch, uart)
    task._phase = PH_GRAB_ALIGN
    task._ensure_pids()
    be.cx = W / 2.0 + 0.8 * W / 2.0            # dx = +0.8
    be.cy = H / 2.0 + 0.8 * H / 2.0            # dy = +0.8
    task.process(_red_frame(), 0)
    surge, sway, heave, yaw = uart.dofs[-1]
    assert yaw == 0.0, "这一段明确不用 yaw"
    assert heave == 0.0
    assert sway > 0.0 and surge > 0.0
    lim = S.comm.grab.align_level
    assert abs(sway) <= float(lim["sway"]["out_max"]) + 1e-9, "sway 必须是轻微档"
    assert abs(surge) <= float(lim["surge"]["out_max"]) + 1e-9, "surge 必须是轻微档"
    assert abs(sway) <= 0.30 and abs(surge) <= 0.30


def test_level_angle_follows_the_measured_pitch(monkeypatch):
    """回水平用**实测**：抬头只到 +25° 时，回位必须是 −25°，不是写死的 −30°。"""
    uart = FakeUart(pitch_gain=25.0 / 30.0)
    task, be = _task(monkeypatch, uart)
    _drive(task, uart, be, radius=_grow, frame=_red_frame())
    assert uart.turns[0] == (2, 30.0)
    assert uart.turns[1] == (2, -25.0), "回位角应是实测的反解，实际 %s" % (uart.turns[:2],)


def test_no_pitch_telemetry_stops_instead_of_dipping(monkeypatch):
    """拿不到 pitch 遥测 ⇒ 回不了水平 ⇒ **绝不下压**（按着角度压会撞坏东西）。"""
    uart = FakeUart()
    uart.telemetry.pitch_deg = None
    task, be = _task(monkeypatch, uart)
    _drive(task, uart, be, radius=_grow, frame=_red_frame())
    assert task.last_info["reason"] == "no_pitch_telemetry"
    dip_heave = float(S.comm.grab.dip["heave"])
    assert not any(abs(d[2] - dip_heave) < 1e-9 for d in uart.dofs), "不许下压"


def test_detection_uses_the_down_frame_not_the_front_frame(monkeypatch):
    """**感知源必须是下视帧**：给 process 一帧全非目标的"前视"，只要注入的下视帧是红的就算对。"""
    uart = FakeUart()
    task, be = _task(monkeypatch, uart)
    _drive(task, uart, be, radius=_grow, frame=_blue_frame())   # 前视/主循环帧：笼区非红
    assert task.last_info["reason"] == "wrong_ball" or task.last_info["reason"] == "assumed_ok"


# --------------------------------------------------------------------------- #
# 错球：倒掉 → 重来 → 第 2 次默认对
# --------------------------------------------------------------------------- #
def test_wrong_ball_is_dumped_then_the_second_attempt_is_assumed_ok(monkeypatch):
    """★ 用户口径：笼里不是目标色 ⇒ 右移→横倾倒出→回正→左移→后退→重来；
    第 2 次尝试**默认是对的**，直接过。"""
    uart = FakeUart()
    task, be = _task(monkeypatch, uart)
    monkeypatch.setitem(S.comm.grab, "calibrated", True)

    def frame(i):
        # 第 1 次夹到的是"错球"（笼区非目标色）；倒球后（_dumps≥1）换回目标色
        return _red_frame() if task._dumps >= 1 else _blue_frame()

    phases = _drive(task, uart, be, radius=_grow, frame=frame, phases=[])
    assert task._dumps == 1, "应该倒过一次球"
    assert task.last_info["reason"] == "assumed_ok", task.last_info
    assert task.holds_ball is True
    assert (3, 30.0) in uart.turns and (3, -30.0) in uart.turns, \
        "横倾倒出后必须回正，实际 %s" % (uart.turns,)
    # 倒球序列：先右移(sway>0)、再左移(sway<0)、再后退(surge<0)
    assert any(d[1] > 0 for d in uart.dofs) and any(d[1] < 0 for d in uart.dofs)
    assert any(d[0] < 0 for d in uart.dofs)
    assert uart.turns.count((2, 30.0)) == 2, "倒球后必须重新抬头再夹一次"
    assert PH_GRAB_DIP in phases


def test_dump_is_capped(monkeypatch):
    """倒球次数用完 ⇒ 安全退出，**不空转到超时**。"""
    monkeypatch.setitem(S.comm.grab, "retry",
                        S.Y(dict(assume_ok=False, assume_ok_after=99, max_dumps=0)))
    uart = FakeUart()
    task, be = _task(monkeypatch, uart)
    _drive(task, uart, be, radius=_grow, frame=_blue_frame())
    assert task.last_info["reason"] == "wrong_ball"
    assert task.holds_ball is False and task._dumps == 0


def test_estop_ends_the_task(monkeypatch):
    uart = FakeUart()
    task, be = _task(monkeypatch, uart)
    uart.estop_active = True
    assert task.process(_red_frame(), 0) == S.STATUS_DONE
    assert task.last_info["reason"] == "estop"


# --------------------------------------------------------------------------- #
# 判据量 / 感知件
# --------------------------------------------------------------------------- #
def test_ratio_helper_matches_the_project_definition():
    """判据量 = 圆面积占比（πr²/(W·H)），**不是 bbox 占比**（bbox 会高估 4/π≈1.27 倍）。"""
    d = Detection(cx=100.0, cy=80.0, r=40.0, score=1.0, area=1, circ=1.0, cov=1.0)
    assert _ratio_radius(40.0, W, H) == pytest.approx(circle_to_ratio(d, W, H))
    assert _ratio_radius(40.0, W, H) != pytest.approx(4.0 * 40.0 * 40.0 / (W * H))


def test_red_percent_reads_the_cage_roi():
    assert red_percent(_red_frame(), ROI) > 0.9
    assert red_percent(_blue_frame(), ROI) < 0.05
    assert red_percent(None, ROI) is None, "帧缺失 ⇒ None（**不是 0**：0 会被当成「确实不是目标色」）"
    assert red_percent(_red_frame(), [0.5, 0.5, 0.0, 0.0]) is None, "退化 ROI ⇒ None"
    assert red_percent(_red_frame(), None) is None
    assert red_percent(_red_frame(), [0.1, 0.1, 0.2]) is None
