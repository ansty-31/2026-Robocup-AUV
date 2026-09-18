# -*- coding: utf-8 -*-
"""tests/test_gate_dash.py — gate「对准好就直接冲」出口 + 配置接线（无硬件闭环）

覆盖（都是 2026-09-17 那轮改动新增/修好的行为）：
  1. width 档：对准确认 + z≤width.z_max → 直冲（旧版这里是**永久 HOLD 到超时**）；
  2. width 档：一上来 z 就小于 cross 附近时同样能冲（不再"没触发 APPROACH 就等死"）；
  3. width 档：**没对准就不冲**（保持静止等，而不是蹭杆）；
  4. coarse 档：够近(占比≥dash_ratio) + 对准 → 直冲；
  5. coarse 档：**只有未对准才后退** —— 对准时不下发后退速度（实测"来回退"的主因）；
  6. `dash: false` 开关能退回旧行为（可回退）；
  7. 配置缺键不崩（删掉 comm.gate.* / vision.gate.pnp 也能跑）；
  8. `keypoint.conf_thr` 真的接到解码器 vis_thr（旋钮不是摆设）。

全部用合成检测（真实标定相机模型 + 手工 2D 角点），不依赖权重/串口/水池。
"""
import numpy as np
import pytest

import base.settings as S
from common.detector import Det
from gate.gate_detector import board_camera
from gate.gate_task import PH_THROUGH, SUB_HOLD, GateTask
from gate.geometry import object_points

CAM = board_camera()
OBJ3 = object_points()


# --------------------------------------------------------------- 装置
class _Uart(object):
    def __init__(self):
        self.frames = []
        self.neutral_calls = 0

    def send_dof(self, surge=0.0, sway=0.0, heave=0.0, yaw=0.0):
        self.frames.append((float(surge), float(sway), float(heave), float(yaw)))

    def neutral(self):
        self.neutral_calls += 1

    def set_motion(self, name, force=False):
        pass


class _Hub(object):
    """把"每帧检测"做成可调用对象（GateTask.ready 依赖 has_extra）。"""

    def __init__(self, fn):
        self.fn = fn

    def has_extra(self, task):
        return task == "gate"

    def detect_list(self, task, frame):
        return self.fn()


def _det(z, px=None, kconf=(0.95, 0.95, 0.95, 0.95), cam=CAM):
    """整门框 bbox（4 角投影）+ 指定角点置信度。

    px=None → 门原点放在画面正中（= 已对准：dxn/dyn≈0）；
    px=(u,v) → 放在指定像素（用于构造"没对准"）。
    """
    if px is None:
        px = (cam.width / 2.0, cam.height / 2.0)
    tvec = cam.backproject_z(float(px[0]), float(px[1]), float(z))
    uv = cam.project(OBJ3, np.zeros(3), tvec)
    x0, y0 = float(uv[:, 0].min()), float(uv[:, 1].min())
    x1, y1 = float(uv[:, 0].max()), float(uv[:, 1].max())
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(cam.width), x1), min(float(cam.height), y1)
    return [Det("gate", 0.95, x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0),
                kpts=uv, kpt_conf=np.asarray(kconf, np.float32))]


def _run(task, n_frames, t0=1000, dt=100):
    """跑 n 帧，返回 (最终 status, 每帧信息快照)。"""
    frame = np.zeros((CAM.height, CAM.width, 3), np.uint8)
    hist = []
    st = S.STATUS_RUNNING
    for i in range(n_frames):
        st = task.process(frame, t0 + dt * i)
        hist.append(dict(task.last_info))
        if st == S.STATUS_DONE:
            break
    return st, hist


def _resolved_gains():
    """构造一个 GateTask，返回它**实际解析出来**的三套修正增益。

    gate 的 cfg 里**不再写增益数字**（共用 `comm.ball` 那一套），所以只能问任务本身。
    """
    t = _task(lambda: [])
    return t


@pytest.fixture
def dash_on(monkeypatch):
    """显式打开 width/coarse 的直冲出口③④。

    **出厂配置是关的**（`width.dash`/`coarse.dash: false`，2026-09-18 用户现场决定
    "暂时不在 width/coarse 触发 dash"）→ 测"能力"的用例必须自己打开，测"默认行为"的
    用例则断言关着时的表现。
    """
    for nm in ("width", "coarse"):
        # 必须包成 S.Y：任务里是 `G.width.dash` 这种属性访问，普通 dict 会 AttributeError
        monkeypatch.setitem(S.comm.gate, nm,
                            S.Y(dict(S.comm.gate.get(nm, {}), dash=True)))
    return True


def _task(fn):
    return GateTask(_Uart(), _Hub(fn), CAM.width, CAM.height)


# --------------------------------------------------------------- 1/2/3. width 档
def test_width_aligned_dashes_instead_of_hanging(dash_on):
    """width 档对准好 → 直冲（旧版：z≤cross+0.4 后 surge 恒 0，卡到任务超时）。"""
    z = {"v": 3.0}

    def det_fn():
        z["v"] = max(0.8, z["v"] - 0.02)
        return _det(z["v"], kconf=(0.95, 0.95, 0.0, 0.0))   # 只 TL,TR → width 档

    task = _task(det_fn)
    st, hist = _run(task, 300)
    phases = [h["phase"] for h in hist]
    modes = {h["mode"] for h in hist}

    assert "width" in modes
    assert PH_THROUGH in phases, "对准后必须出现直冲出口"
    assert st == S.STATUS_DONE and task.last_info["reason"] == "pass"
    assert task.last_info["pass"] == 1
    # 直冲速度 = width.surge（比 surge.through 保守），且**不是** cre/0
    surges = [f[0] for f in task.uart.frames]
    assert pytest.approx(S.comm.gate.width.surge) in surges
    assert max(surges) <= S.comm.gate.width.surge + 1e-9
    # 全程没出现过 HOLD 到死的迹象：THROUGH 之前的"creep/center"都有正 surge
    assert max(surges) > 0


def test_width_dashes_even_when_already_near_cross(dash_on):
    """一上来 z 就小于 cross 附近：仍要能冲（不依赖先走 APPROACH→creep）。"""
    def det_fn():
        return _det(0.9, kconf=(0.95, 0.95, 0.0, 0.0))

    task = _task(det_fn)
    st, hist = _run(task, 60)
    assert PH_THROUGH in [h["phase"] for h in hist]
    assert st == S.STATUS_DONE and task.last_info["reason"] == "pass"
    # 不要求经过 APPROACH：这条路径就是"近距也能直接冲"
    assert task.last_info["mode"] == "width"


def test_width_misaligned_does_not_dash(dash_on):
    """没对准（水平偏出安全带）→ 不冲、也不前进（保持静止等下一次对准）。"""
    def det_fn():
        # 门心偏到画面右侧 1/4 半宽处 → dxn≈0.25 ≫ width.dx_max(0.08)
        return _det(1.2, px=(CAM.width * 0.5 + CAM.width * 0.25, CAM.height / 2.0),
                    kconf=(0.95, 0.95, 0.0, 0.0))

    task = _task(det_fn)
    st, hist = _run(task, 30)
    assert PH_THROUGH not in [h["phase"] for h in hist]
    assert st == S.STATUS_RUNNING
    assert all(abs(f[0]) < 1e-9 for f in task.uart.frames), "没对准时不该有前向速度"
    assert hist[-1]["substate"] == SUB_HOLD


def test_width_dash_off_restores_old_behaviour(monkeypatch):
    """开关可回退：dash=false → 回到"远距 creep、近距 HOLD"（不再直冲）。"""
    monkeypatch.setitem(S.comm.gate, "width",
                        dict(S.comm.gate.width, dash=False))

    def det_fn():
        return _det(1.2, kconf=(0.95, 0.95, 0.0, 0.0))

    task = _task(det_fn)
    st, hist = _run(task, 60)
    assert PH_THROUGH not in [h["phase"] for h in hist]
    assert st == S.STATUS_RUNNING
    assert all(f[0] <= 0.0 for f in task.uart.frames)


# --------------------------------------------------------------- 4/5. coarse 档
def _ratio_of(det):
    return float(det.w) / float(CAM.width)


def test_coarse_near_and_aligned_dashes(dash_on):
    """coarse：框占比 ≥ dash_ratio 且对准 → 直冲，速度 = coarse.surge。"""
    def det_fn():
        return _det(1.0, kconf=(0.0, 0.0, 0.0, 0.0))        # 角点全缺 → coarse

    task = _task(det_fn)
    assert _ratio_of(det_fn()[0]) >= S.comm.gate.coarse.dash_ratio
    st, hist = _run(task, 60)
    assert PH_THROUGH in [h["phase"] for h in hist]
    assert st == S.STATUS_DONE and task.last_info["reason"] == "pass"
    surges = [f[0] for f in task.uart.frames]
    assert pytest.approx(S.comm.gate.coarse.surge) in surges


def test_coarse_backs_off_only_when_misaligned(monkeypatch):
    """对准时不下发后退；未对准时才会 HOLD→REACQUIRE（负 surge）。"""
    # ---- 对准、中距（框占比介于 far_ratio 与 near_ratio 之间）：creep，不后退
    def det_aligned():
        return _det(2.2, kconf=(0.0, 0.0, 0.0, 0.0))

    task = _task(det_aligned)
    assert S.comm.gate.coarse.far_ratio < _ratio_of(det_aligned()[0]) \
        < S.comm.gate.coarse.near_ratio
    _run(task, 40)
    surges = [f[0] for f in task.uart.frames]
    assert min(surges) >= 0.0, "对准时不该出现后退"
    assert max(surges) > 0.0, "对准时应慢速靠近（creep）"

    # ---- 同样距离但没对准：HOLD 到 hold.max_frames → REACQUIRE（负 surge）
    def det_misaligned():
        return _det(2.2, px=(CAM.width * 0.5 + CAM.width * 0.3, CAM.height / 2.0),
                    kconf=(0.0, 0.0, 0.0, 0.0))

    task2 = _task(det_misaligned)
    _run(task2, int(S.comm.gate.hold.max_frames) + 10)
    surges2 = [f[0] for f in task2.uart.frames]
    assert min(surges2) < 0.0, "未对准且迟迟拿不到角点 → 应后退重取"


def test_horizontal_channel_is_exclusive_and_signed(monkeypatch):
    """居中阶段的水平修正：**yaw 与 sway 同一帧只发一个**，且符号各自正确。

    约定（三重一致）：+yaw = 右转（`manual/udp_server.py` 的 `0,0,0,0.6 = yaw right`、
    SEARCH 脉冲注释"右转"）；+sway = 右移（下位机 `command_left_right=RC[7]-RC[8]`）。
    默认策略（2026-09-18 改）：**像素档以 sway 为主**（`align_yaw.in_px: false`）——
    原因是 yaw 默认增益推不动（见 `test_yaw_centering_channel_needs_gain_above_deadzone`）。
    本用例测的是"显式打开 in_px 时，互斥与符号仍然正确"（能力用例，不是默认行为）。
    """
    monkeypatch.setitem(S.comm.gate, "align_yaw",
                        S.Y(dict(S.comm.gate.get("align_yaw", {}), in_px=True)))
    right_px = (CAM.width * 0.5 + CAM.width * 0.25, CAM.height / 2.0)
    left_px = (CAM.width * 0.5 - CAM.width * 0.25, CAM.height / 2.0)

    # 像素档（显式 in_px=true）：门在右 → yaw>0、sway==0；门在左反之
    for px, side, sign in ((right_px, "右", 1), (left_px, "左", -1)):
        task = _task(lambda px=px: _det(1.2, px=px,
                                        kconf=(0.0, 0.0, 0.0, 0.0)))
        _run(task, 3)
        surge, sway, heave, yaw = task.uart.frames[-1]
        assert yaw * sign > 0, "门在%s → yaw 符号错了（yaw=%.3f）" % (side, yaw)
        assert abs(sway) < 1e-9, "yaw 生效时不许同时发 sway（sway=%.3f）" % sway


def test_yaw_only_during_align(monkeypatch, dash_on):
    """yaw **只在居中(ALIGN)阶段**下发：THROUGH 段必须 yaw=0（冲的时候不许转头）。"""
    seq = {}

    def det_fn():
        # 先给"对准的 coarse"让它攒够确认帧 → 直冲（出口④），此后一直发同一个检测
        return _det(1.0, kconf=(0.0, 0.0, 0.0, 0.0))

    task = _task(det_fn)
    _run(task, 40)
    assert task.last_info["phase"] == "THROUGH"
    # 取 THROUGH 之后的帧：yaw 必须为 0（`_tick_through` 只发 surge）
    through_frames = [f for f in task.uart.frames[-8:]]
    assert all(abs(f[3]) < 1e-9 for f in through_frames), \
        "THROUGH 段不该有 yaw：%s" % [f[3] for f in through_frames]
    assert all(f[0] > 0 for f in through_frames), "THROUGH 段应持续前进"


def test_align_yaw_off_restores_sway(monkeypatch):
    """开关可回退：`align_yaw.enable=false`（**当前默认**）→ 像素档只用 sway（旧行为）。"""
    monkeypatch.setitem(S.comm.gate, "align_yaw",
                        S.Y(dict(S.comm.gate.get("align_yaw", {}), enable=False)))
    right_px = (CAM.width * 0.5 + CAM.width * 0.25, CAM.height / 2.0)
    task = _task(lambda: _det(1.2, px=right_px, kconf=(0.0, 0.0, 0.0, 0.0)))
    _run(task, 3)
    surge, sway, heave, yaw = task.uart.frames[-1]
    assert sway > 0.0 and abs(yaw) < 1e-9


def test_yaw_centering_converges_in_coarse(monkeypatch):
    """**闭环**：coarse 档靠转向居中必须真的收敛到 aligned（旧版没有 yaw → 完不成居中）。

    简化被控对象：机身航向 psi 随下发的 yaw 指令积分；门在画面里的水平误差
        err = 0.30 - 1.5*psi   （psi 增大 → 门往画面中心走）
    判据：若干帧后 |dx| < coarse.align_x 且进入 CREEP（说明居中真的完成了）。
    """
    monkeypatch.setitem(S.comm.gate, "align_yaw",
                        S.Y(dict(S.comm.gate.get("align_yaw", {}), enable=True)))
    st = {"psi": 0.0}

    def det_fn():
        err = 0.30 - 1.5 * st["psi"]
        px = (CAM.width / 2.0 * (1.0 + err), CAM.height / 2.0)
        return _det(1.5, px=px, kconf=(0.0, 0.0, 0.0, 0.0))   # 角点全缺 → coarse

    task = _task(det_fn)
    frame = np.zeros((CAM.height, CAM.width, 3), np.uint8)
    for i in range(60):
        task.process(frame, 1000 + 100 * i)
        surge, sway, heave, yaw = task.uart.frames[-1]      # 用本帧指令推进被控对象
        st["psi"] += yaw * 1.0 * 0.1                        # 1.0 = 角速度增益, 0.1s = 帧间隔
    assert abs(task.last_info["dx"]) < S.comm.gate.coarse.align_x, \
        "转向居中没收敛：dx=%.3f" % task.last_info["dx"]
    assert task.last_info["action"] == "creep", \
        "收敛后应判 aligned → CREEP，实际 %s" % task.last_info["action"]


# 下位机链路实测死区：DOF→字节(128+127v)→Rc2MyRcKey→RC_Matching*(映射≤35→0)
# ⇒ |DOF| < 0.138 一点推力都没有（数字由固件公式算出，见 gate 文档 §1.1）
# 注：**下潜**的死区补偿在底层（base/uart.py::_apply_dive_comp），用例在 tests/test_base.py；
#     这里只留"速度档位不许落在死区里"这条配置守卫。
ACT_DEADZONE = 0.138


def test_no_speed_preset_inside_actuator_deadzone():
    """没有任何速度档位落在执行器死区里（0.12 那种"看着有值、实际 0 推力"）。

    `surge.creep`/`lost_backward` 原来都是 0.12 → 有效推力 **0%**（蠕进/轻微后退从未发生）。
    允许的值：0（明确表示"不要这个动作"）或 ≥ 死区线。
    """
    g = S.comm.gate
    bad = []
    for k in ("fast", "slow", "creep", "lost_backward", "reacquire", "through"):
        v = abs(float(g.surge[k]))
        if 0.0 < v < ACT_DEADZONE:
            bad.append("surge.%s=%.3f" % (k, v))
    for nm in ("width", "coarse"):
        v = abs(float(g[nm]["surge"]))
        if 0.0 < v < ACT_DEADZONE:
            bad.append("%s.surge=%.3f" % (nm, v))
    assert not bad, "这些档位落在执行器死区内（发出去等于 0 推力）：%s" % bad


def test_gate_gains_share_the_tuned_ball_set():
    """三套修正增益**共用小球那一套**（gate 里不写数字，直接取 `comm.ball`）。

    小球那套是实船实测过的；两套各自漂移正是 sway 反号那类 bug 的温床。
    要单独一套，就在 comm.gate 里显式写 pid_sway/pid_heave 或 align_yaw.{kp,kd,out_max}，
    届时本用例会失败，提醒你"这是有意分叉，请更新用例"。
    """
    t = _resolved_gains()
    b = S.comm.ball
    ks = ("kp", "ki", "kd", "out_max", "deadzone")
    assert t._gain_src.startswith("ball"), "增益来源应为 ball(共用)，实际 %s" % t._gain_src
    assert {k: t._pid_sway_kw[k] for k in ks} == dict(b.approach_pid), \
        "sway 应共用 ball.approach_pid"
    assert {k: t._pid_heave_kw[k] for k in ks} == dict(b.pid), \
        "heave 应共用 ball.pid"
    assert (t._yaw_kw["kp"], t._yaw_kw["kd"], t._yaw_kw["out_max"]) == \
        (float(b.edge_yaw_kp), float(b.edge_yaw_kd), float(b.edge_yaw_max)), \
        "yaw 应共用 ball.edge_yaw_kp/kd/max"
    assert t._yaw_kw["ki"] == 0.0 and t._yaw_kw["deadzone"] == 0.0, \
        "yaw 的 ki/deadzone 应与 ball.py 的转向 PID 一致（都是 0）"


def test_gate_gains_follow_ball_changes(monkeypatch):
    """**共用的证据**：改 `comm.ball` 的增益，gate 构造出来立刻跟着变（没有第二份数字）。

    这条保证"调小球 = 同时调过门"，也保证不会再出现两套参数各自漂移
    （sway 反号那次就是"两任务各一套"的产物）。
    """
    monkeypatch.setitem(S.comm.ball, "approach_pid",
                        S.Y(dict(S.comm.ball.approach_pid, kp=3.25)))
    monkeypatch.setitem(S.comm.ball, "edge_yaw_max", 0.42)
    t = _resolved_gains()
    assert float(t._pid_sway_kw["kp"]) == pytest.approx(3.25), \
        "改了 ball.approach_pid.kp，gate 的 sway 增益没跟着变 → 还在抄第二份数字"
    assert float(t._yaw_kw["out_max"]) == pytest.approx(0.42), \
        "改了 ball.edge_yaw_max，gate 的 yaw 限幅没跟着变"


def test_align_yaw_deadzone_below_align_threshold():
    """配置关系守卫：转向死区必须小于"算作对准"的门槛，否则永远判不了 aligned。"""
    g = S.comm.gate
    yaw_dz = float(_resolved_gains()._yaw_kw["deadzone"])
    thr = min(float(g.align.px_x), float(g.coarse.align_x))
    assert yaw_dz < thr, (
        "yaw deadzone=%.3f ≥ 对准门槛 %.2f → 转到位前就停，coarse 永远差一点" % (yaw_dz, thr))


def test_dash_counter_resets_on_mode_switch(monkeypatch, dash_on):
    """档位跳变（width↔coarse）不能把不相邻的"对准帧"算成连续 → 不许误直冲。

    注：必须关掉 kpt_mem 才能让档位真的交替——融合的"回忆"会把刚消失的角点
    置信度保持为 `max(前值, recall_conf)`（≤max_missing_frames 帧内），
    于是"整框但无角点"的那一帧仍被判成 width 档。
    """
    monkeypatch.setenv("AUV_GATE_KPT_MEM", "0")
    seq = {"i": 0}

    def det_fn():
        i = seq["i"]
        seq["i"] += 1
        if i % 2 == 0:                      # width 档：对准
            return _det(1.2, kconf=(0.95, 0.95, 0.0, 0.0))
        # coarse 档：整框但明显没对准（水平偏出）
        return _det(1.2, px=(CAM.width * 0.5 + CAM.width * 0.35, CAM.height / 2.0),
                    kconf=(0.0, 0.0, 0.0, 0.0))

    task = _task(det_fn)
    st, hist = _run(task, 20)
    assert {"width", "coarse"} <= {h["mode"] for h in hist}, "档位应真的交替"
    assert PH_THROUGH not in [h["phase"] for h in hist]
    assert st == S.STATUS_RUNNING


# --------------------------------------------------------------- 6/7. 配置健壮性
@pytest.mark.parametrize("drop", ["gate_section", "pnp_key", "reacquire_key"])
def test_gate_task_survives_missing_cfg(monkeypatch, drop):
    """配置缺键/整段缺失都不许崩（默认值 = 当前 cfg 值）。"""
    if drop == "gate_section":
        monkeypatch.setitem(S.comm, "gate", {})
        monkeypatch.setitem(S.vision, "gate", {})
    elif drop == "pnp_key":
        monkeypatch.setitem(S.vision.gate, "pnp", None)
    else:
        monkeypatch.setitem(S.comm.gate, "reacquire", None)

    def det_fn():
        return _det(2.0)                                    # 4 角 → 走 PnP 位姿路径

    task = _task(det_fn)
    st, hist = _run(task, 10)
    assert st == S.STATUS_RUNNING
    assert hist[-1]["mode"] == "full"
    assert hist[-1]["action"] in ("center", "forward_slow", "forward_fast")


# --------------------------------------------------------------- 8. 解码器接线
def _kpt_output_one_cell(vis_logit):
    """按 X5 split-head 约定造一个单格输出（stride=8 层，kpt 通道 = [x,y,v]×4）。"""
    g, reg_max, kpt_dim = 80, 16, 4
    cls = np.full((1, g, g, 1), -10.0, np.float32)
    out = {64: np.zeros((1, g, g, 4 * reg_max), np.float32),
           1: cls,
           kpt_dim * 3: np.zeros((1, g, g, kpt_dim * 3), np.float32)}
    gx, gy = 10, 20
    out[1][0, gy, gx, 0] = 6.0                              # sigmoid≈0.9975
    for side in range(4):
        out[64][0, gy, gx, side * reg_max + 3] = 40.0       # DFL → 距离 3 格
    for i in range(4):
        out[kpt_dim * 3][0, gy, gx, 3 * i + 0] = gx         # x（cell 坐标）
        out[kpt_dim * 3][0, gy, gx, 3 * i + 1] = gy         # y（cell 坐标）
        out[kpt_dim * 3][0, gy, gx, 3 * i + 2] = vis_logit  # 可见度 logit
    return out


def test_decode_vis_thr_precedence(monkeypatch):
    """解码期角点阈值优先级：显式参数 > keypoint.vis_thr > conf_thr > 0.5。

    这条把"解码阈值"与"模式判定阈值"解耦：板端曾把 conf_thr 设成 0.95（本意是让
    **模式判定**更严），解码期硬门限跟着变 0.95 后，会把 conf 0.5~0.94 的角点在
    解码阶段直接清 0 —— 预览里就"看不到标识点"了。
    """
    from gate import gate_decode

    monkeypatch.setattr(gate_decode.GateKeypointBackend, "_init_backend",
                        lambda self: None)

    def build(**kw):
        return gate_decode.GateKeypointBackend(path="<not-loadable>",
                                               labels=["gate"],
                                               kpt_order=["TL", "TR", "BR", "BL"],
                                               camera=CAM, **kw)

    # ① 默认（没配 vis_thr）：解码阈值 = conf_thr（历史行为，向后兼容）
    monkeypatch.setitem(S.vision.gate, "keypoint", {"conf_thr": 0.95})
    assert build()._vis_thr == pytest.approx(0.95)
    # ② 配了 vis_thr：解码与模式判定解耦
    monkeypatch.setitem(S.vision.gate, "keypoint",
                        {"conf_thr": 0.95, "vis_thr": 0.5})
    assert build()._vis_thr == pytest.approx(0.5)
    # ③ 显式参数最高（预览工具传 0.0 = 看模型原始置信度）
    assert build(vis_thr=0.0)._vis_thr == pytest.approx(0.0)
    # ④ 非法值逐级回退，不许崩
    monkeypatch.setitem(S.vision.gate, "keypoint",
                        {"conf_thr": 0.95, "vis_thr": "abc"})
    assert build()._vis_thr == pytest.approx(0.95)
    monkeypatch.setitem(S.vision.gate, "keypoint", {"conf_thr": "??"})
    assert build()._vis_thr == pytest.approx(0.5)


def test_decode_thr_decides_whether_090_corner_survives():
    """conf≈0.90 的角点：vis_thr=0.95 被清 0（板端现象），0.5 / 0.0 保留。

    对应板端历史 dump 里真实出现的 conf 0.56 / 0.9125 / 0.9685 那类角点。
    """
    from gate import gate_decode

    logit_090 = float(np.log(0.9 / 0.1))
    out = _kpt_output_one_cell(logit_090)
    for thr, kept in ((0.95, False), (0.5, True), (0.0, True)):
        dets = gate_decode.decode_yolo11_kpt(out, ["gate"], 640, 640,
                                            input_w=640, input_h=640,
                                            vis_thr=thr)
        assert (float(dets[0].kpt_conf.max()) > 0.0) is kept, \
            "vis_thr=%.2f 时该角点保留应为 %s" % (thr, kept)


def test_decode_vis_thr_follows_keypoint_conf_thr(monkeypatch):
    """`keypoint.conf_thr` 必须真的传给解码器（否则调低它完全没效果）。"""
    from gate import gate_decode

    # 构造不加载模型（无权重也能验接线）
    monkeypatch.setattr(gate_decode.GateKeypointBackend, "_init_backend",
                        lambda self: None)
    monkeypatch.setitem(S.vision.gate, "keypoint", {"conf_thr": 0.30})
    bk = gate_decode.GateKeypointBackend(path="<not-loadable>", labels=["gate"],
                                        kpt_order=["TL", "TR", "BR", "BL"],
                                        camera=CAM)
    assert bk._vis_thr == pytest.approx(0.30)

    # 解码层：可见度 sigmoid 后 ≈0.40 的角点在两个阈值下结果不同
    logit_040 = float(np.log(0.4 / 0.6))
    out = _kpt_output_one_cell(logit_040)
    keep = gate_decode.decode_yolo11_kpt(out, ["gate"], 640, 640,
                                        input_w=640, input_h=640, vis_thr=0.30)
    drop = gate_decode.decode_yolo11_kpt(out, ["gate"], 640, 640,
                                        input_w=640, input_h=640, vis_thr=0.50)
    assert float(keep[0].kpt_conf.min()) > 0.0
    assert float(drop[0].kpt_conf.max()) == 0.0


# ------------------------------------------------- 9. 居中判据：全档位一套像素门槛
def test_pose_path_centers_in_pixels_not_meters(monkeypatch):
    """位姿档也**按像素**居中（2026-09-18）：偏 0.15 半屏现在算对准 → 进 APPROACH。

    同时守住"米制遗留键被忽略"：配置里塞 `xy_m=0.001`（米制下=几乎必须零误差）
    不产生任何影响，行为只由 `px_x/px_y` 决定。
    """
    from gate.gate_task import PH_APPROACH
    # 故意塞米制遗留键 + 明确写像素门槛（两者都在配置里）
    monkeypatch.setitem(S.comm.gate, "align", S.Y(dict(
        S.comm.gate.get("align", {}), confirm_frames=4,
        px_x=0.20, px_y=0.18, xy_m=0.001, scale_m=99.0)))

    # 门心偏到画面右侧 0.15 半屏、竖直偏下 0.15 半屏（像素量）
    def det_fn():
        return _det(1.2, px=(CAM.width * 0.5 + CAM.width * 0.5 * 0.15,
                             CAM.height * 0.5 + CAM.height * 0.5 * 0.15))

    task = _task(det_fn)
    st, hist = _run(task, 30)
    assert hist[0]["mode"] == "full", "整门框 4 角可信 → 应走位姿档"
    assert PH_APPROACH in [h["phase"] for h in hist], \
        "0.15 半屏在 px 门槛(0.20)内 → 必须判对准并进 APPROACH（米制遗留键不许干扰）"
    # 像素→运动：门在右(+dxn) → sway>0（右移）；门在下(+dyn) → heave<0（下潜）
    f = [x for x in task.uart.frames if abs(x[1]) > 1e-9]
    assert f and all(x[1] > 0 for x in f), "门在画面右侧 → sway 应为正（右移）"
    h = [x for x in task.uart.frames if abs(x[2]) > 1e-9]
    assert h and all(x[2] < 0 for x in h), "门在画面下方 → heave 应为负（下潜）"
    # 位姿档没前进之前（ALIGN 段）不许有 surge
    assert all(abs(x[0]) < 1e-9 for x in task.uart.frames[:2]), "ALIGN 里不该前进"


def test_align_threshold_is_loosened_but_shaped_like_margins():
    """2026-09-18 放宽的回归守卫 + 形状：横向门槛不小于竖向（竖向余量更小）。"""
    g = S.comm.gate
    px_x, px_y = float(g.align.px_x), float(g.align.px_y)
    assert px_x >= 0.15 and px_y >= 0.13, "被改回收紧值了？机身小，要放宽"
    # ⚠️ 两轴不能直接比数字：1 单位 dxn = 0.818·z 米、dyn = 0.462·z 米（见 cfg/front_camera.yaml
    #    标定）。要比**物理允许偏差**：竖向余量只有横向的一半 → 竖向物理偏差必须更小。
    m_x = px_x * (1280 / 2.0) / 782.54        # dxn → tan(角)，×z 后是米
    m_y = px_y * (720 / 2.0) / 779.79
    assert m_y <= m_x, "竖向允许的物理偏差比横向还大 → 方向错了（竖向余量本来就小）"
    # coarse 的框心是最粗的估计 → 不该比位姿/width 档更严
    assert float(g.coarse.align_x) >= px_x - 1e-9
    assert float(g.coarse.align_y) >= px_y - 1e-9
    # 直冲安全带必须比"对准门槛"更严（敢冲 vs 能动，职责分离）
    assert float(g.width.dx_max) < px_x
    assert float(g.coarse.dx_max) < float(g.coarse.align_x)


def test_safety_invariants_were_not_loosened():
    """放宽"居中成功"时**不许顺手放宽**这些防单帧错解的保命项（本次一个都没动）。"""
    z = S.comm.gate.z
    assert int(z.cross_confirm_frames) >= 2, "z 连续确认帧数不许降到 1"
    assert float(z.near_lost_m) >= 1.0, "丢门判过门的距离门槛不许缩小"
    pnp = S.vision.gate.pnp
    assert float(pnp.max_z_jump_m) <= 0.8, "z 跳变保护不许放宽"
    assert float(S.comm.depth_guard.min_depth_m) == pytest.approx(0.55)
    assert int(S.comm.gate.align.confirm_frames) >= 4, "对准连续帧数不许降"


# ------------------------------------------------- 10. 水平通道选择 / 执行器死区
# 下位机链路：DOF→字节(128+127v)→Rc2MyRcKey→RC_Matching*(映射≤35→0)
#   ⇒ |DOF| < 0.138 一点推力都没有（数字由固件公式算出，见 gate 文档 §1.1）
_ACT_DEADZONE = 0.138


def test_default_centering_channel_is_sway_in_px_modes():
    """**默认**：width/coarse 用 sway 居中（in_px=false），位姿档也用 sway（in_pose=false）。

    现场观察"coarse/width 往中间居中很费力"的原因 = yaw 默认增益推不动：
        yaw 命令 = kp(0.3)·dxn，要 > 0.138 才有推力 ⇒ |dxn| > 0.46（近半个屏）
        sway 命令 = kp(8.0)·dxn ⇒ |dxn| > 0.017
    所以像素档必须落到 sway 上（否则小误差时等于没有水平控制）。
    """
    assert bool(S.comm.gate.align_yaw.get("in_px", True)) is False, \
        "像素档又切回 yaw 了？先看 yaw 增益能不能过死区"
    assert bool(S.comm.gate.align_yaw.get("in_pose", True)) is False
    # 门偏 1/8 屏（0.25）：sway 必须真的出力（yaw 那种"看着有指令其实没推力"不算）
    task = _task(lambda: _det(1.2, px=(CAM.width * 0.5 + CAM.width * 0.125,
                                       CAM.height / 2.0),
                              kconf=(0.0, 0.0, 0.0, 0.0)))
    _run(task, 3)
    surge, sway, heave, yaw = task.uart.frames[-1]
    assert abs(sway) > _ACT_DEADZONE, "sway 落进执行器死区了（sway=%.3f）" % sway
    assert abs(yaw) < 1e-9, "默认不该发 yaw"


def test_yaw_centering_channel_needs_gain_above_deadzone():
    """**不变量**：只要哪一档启用了 yaw 居中，它在"对准门槛"处的输出就必须超过执行器死区。

    这是"两条死区线"里的第 ② 条（第 ① 条是 PID 内 deadzone < 对准门槛，另有用例）。
    默认增益 kp=0.3 时 0.3×0.25 = 0.075 < 0.138 → **该通道无效**，所以 in_px/in_pose 必须为 false；
    要启用就得把 kp 抬到 ≥ 0.138/对准门槛（≈0.55 以上才有一点点，1.0 才舒服）。
    """
    g = S.comm.gate
    yaw_kp = float(_resolved_gains()._yaw_kw["kp"])
    yaw_max = float(_resolved_gains()._yaw_kw["out_max"])
    for flag, thr in (("in_px", float(g.coarse.align_x)),
                      ("in_pose", float(g.align.px_x))):
        if not bool(g.align_yaw.get(flag, False)):
            continue
        eff = min(yaw_kp * thr, yaw_max)
        assert eff > _ACT_DEADZONE, (
            "%s=true 但 yaw 在门槛 %.2f 处只有 %.3f（<死区 %.3f）→ 通道是摆设；"
            "要么把 kp 抬过 %.2f，要么把 %s 设回 false"
            % (flag, thr, eff, _ACT_DEADZONE, _ACT_DEADZONE / thr, flag))


def test_yaw_centering_with_raised_gain_actually_moves_the_boat(monkeypatch):
    """把 kp 抬到能过死区后，像素档 yaw 居中**真的能收敛**（被控对象含执行器死区）。"""
    monkeypatch.setitem(S.comm.gate, "align_yaw",
                        S.Y(dict(S.comm.gate.get("align_yaw", {}),
                                 enable=True, in_px=True, kp=1.5)))
    st = {"psi": 0.0}

    def det_fn():
        err = 0.30 - 1.5 * st["psi"]
        return _det(1.5, px=(CAM.width / 2.0 * (1.0 + err), CAM.height / 2.0),
                    kconf=(0.0, 0.0, 0.0, 0.0))          # 角点全缺 → coarse

    task = _task(det_fn)
    frame = np.zeros((CAM.height, CAM.width, 3), np.uint8)
    moved = 0
    for i in range(60):
        task.process(frame, 1000 + 100 * i)
        _s, _sw, _h, yaw = task.uart.frames[-1]
        eff = 0.0 if abs(yaw) < _ACT_DEADZONE else yaw   # ← 真实执行器：死区内不出力
        if abs(eff) > 0:
            moved += 1
        st["psi"] += eff * 1.0 * 0.1
    assert moved > 0, "抬了 kp 还是没推力？"
    assert abs(task.last_info["dx"]) < 0.10, \
        "抬高 kp 后应能收敛，实际 dx=%.3f" % task.last_info["dx"]


# ------------------------------------------------- 11. 档位翻转 & 航向通道
def test_align_confirm_survives_full_p3p_flip():
    """性质守卫：full(4角)↔p3p(3角) 帧间翻转**不影响**对准确认 → 仍能进 APPROACH。

    机理（已核实）：`_on_pose()` 之后紧跟 `return`，所以"档位变化清零"那段只在
    **位姿失败退化**的路径上执行；full↔p3p 都是位姿档、根本不经过它。
    本用例的价值是**防将来重构**：谁要是把清零挪到位姿分派之前，这条会红
    （那才是"永远进不了 APPROACH → ALIGN 里 surge=0 → 原地不动"的真陷阱）。
    """
    from gate.gate_task import PH_APPROACH
    seq = {"i": 0}

    def det_fn():
        i = seq["i"]
        seq["i"] += 1
        # 门心始终在画面正中（已对准），只是第 4 个角点隔帧可见
        kc = (0.95, 0.95, 0.95, 0.95) if i % 2 == 0 else (0.95, 0.95, 0.95, 0.0)
        return _det(1.6, kconf=kc)

    task = _task(det_fn)
    _st, hist = _run(task, 20)
    modes = {h["mode"] for h in hist}
    assert modes == {"full", "p3p"}, "用例前提：两种位姿档应交替出现，实际 %s" % modes
    assert PH_APPROACH in [h["phase"] for h in hist], \
        "full↔p3p 翻转把对准确认清了 → 永远进不了 APPROACH"


def test_align_confirm_still_resets_across_mode_class():
    """但跨类（位姿档 ↔ width/coarse）必须清零：那才是"证据不同"。"""
    seq = {"i": 0}

    def det_fn():
        i = seq["i"]
        seq["i"] += 1
        # 交替：位姿档(对准) ↔ coarse(整框、也"对准")
        if i % 2 == 0:
            return _det(1.6)
        return _det(1.6, kconf=(0.0, 0.0, 0.0, 0.0))

    task = _task(det_fn)
    _st, hist = _run(task, 20)
    from gate.gate_task import PH_APPROACH
    # coarse 档没有 yaw/位姿，sanity：这里只要求"没有因为翻转而误判进 APPROACH 后直冲"
    assert PH_THROUGH not in [h["phase"] for h in hist]


def test_pose_heading_error_is_measured_and_signed_right(monkeypatch):
    """位姿档的**航向**通道：能测出"机身歪了多少度"，且转向符号正确。

    用合成位姿注入已知航向误差（把门/相机绕**垂直轴**转 delta），
    断言：（1）`gate_normal_angles_deg` 读出 ≈ -delta（机身右转 → 门法向偏左）；
          （2）按读出的角**反着转**回去，误差变小（符号物理正确，不是自证循环）。
    """
    import cv2
    from gate.geometry import gate_normal_angles_deg

    def Ry(deg):
        return cv2.Rodrigues(np.array([0.0, np.radians(float(deg)), 0.0]))[0]

    for delta in (-18.0, -7.0, 7.0, 18.0):        # delta>0 = 机身右转
        Rc = Ry(delta)
        rv = cv2.Rodrigues(Rc.T)[0].ravel()
        tv = Rc.T @ np.array([0.0, 0.0, 1.5])
        psi, pitch = gate_normal_angles_deg(rv, tv)
        assert abs(psi - (-delta)) < 0.5, "机身右转 %.0f° 应读出 psi≈%+.0f，实际 %+.1f" \
            % (delta, -delta, psi)
        assert abs(pitch) < 0.5
        # 修正方向：yaw_cmd 与 psi 同号（psi<0 → 左转）；把机身按该方向转一小步 → |psi| 变小
        step = 5.0 * (1.0 if psi > 0 else -1.0)   # 与 yaw 指令同向的一步
        Rc2 = Ry(delta + step)
        psi2, _ = gate_normal_angles_deg(cv2.Rodrigues(Rc2.T)[0].ravel(),
                                         Rc2.T @ np.array([0.0, 0.0, 1.5]))
        assert abs(psi2) < abs(psi), \
            "按 yaw 指令方向转 %.0f° 反而更歪（psi %.1f→%.1f）→ 符号反了" \
            % (step, psi, psi2)


def test_heading_channel_logs_and_stays_off_by_default(monkeypatch):
    """航向通道默认**只测不控**：日志里有 hdg（可标定 bias），但 yaw 不动。"""
    task = _task(lambda: _det(1.5))
    _run(task, 5)
    assert task.last_info["hdg"] is not None, "位姿档必须把航向误差打进日志（现场标定用）"
    assert abs(task.last_info["hdg"]) < 3.0, "正对门时航向误差应≈0，实际 %.1f°" \
        % task.last_info["hdg"]
    assert all(abs(f[3]) < 1e-9 for f in task.uart.frames), \
        "默认 heading.enable=false → 不该发 yaw"


def test_heading_channel_turns_the_boat_when_enabled(monkeypatch):
    """打开航向通道（并把 bias 标成 0）后：机身歪 → 真的发反向 yaw。"""
    import cv2
    monkeypatch.setitem(S.comm.gate, "align_yaw",
                        S.Y(dict(S.comm.gate.get("align_yaw", {}),
                                 enable=True, in_pose=False,
                                 heading=dict(enable=True, ema_frames=1,
                                              deadzone_deg=6.0, bias_deg=0.0))))
    delta = {"v": 15.0}                      # 机身左偏 15° → 需要右转(yaw>0)

    def det_fn():
        Rc = cv2.Rodrigues(np.array([0.0, np.radians(-delta["v"]), 0.0]))[0]
        tvec = Rc.T @ np.array([0.0, 0.0, 1.5])
        uv = CAM.project(OBJ3, cv2.Rodrigues(Rc.T)[0].ravel(), tvec)
        return [Det("gate", 0.95, 0.0, 0.0, 10.0, 10.0,
                    kpts=uv, kpt_conf=np.full(4, 0.95, np.float32))]

    task = _task(det_fn)
    _run(task, 4)
    yaw = task.uart.frames[-1][3]
    assert task.last_info["hdg"] > 6.0, "应测出约 +15° 的航向误差，实际 %s" % task.last_info["hdg"]
    assert yaw > 0, "机身左偏 → 应右转(yaw>0)，实际 %.3f" % yaw


def test_heading_channel_is_full_only_not_p3p(monkeypatch):
    """航向通道**只认 4 角(full)**：p3p 的航向实测是垃圾（std 64~69°），不许拿去闭环。

    依据：tools/analyze_pnp_center.py + analyze_heading.py 对真实 dump 的统计
      p3p：重投影 RMS 恒 0.0px（位姿欠定 → reproj_px 门槛失效）、深度比 full 小 28%、
           航向 std 64~69°（full 干净 dump 5.4°）。
    """
    import cv2
    monkeypatch.setitem(S.comm.gate, "align_yaw",
                        S.Y(dict(S.comm.gate.get("align_yaw", {}),
                                 enable=True, in_pose=False,
                                 heading=dict(enable=True, ema_frames=1,
                                              deadzone_deg=6.0, bias_deg=0.0))))
    # 先给几帧 4 角（full）把上帧位姿建立起来：3 点求解**必须有 prev 做初值**
    # （geometry._iter_candidates：n==3 且无 prev → 返回空 → 退化 coarse；本机 cv2 5.0 就走这条路）
    Rc = cv2.Rodrigues(np.array([0.0, np.radians(-15.0), 0.0]))[0]
    tvec = Rc.T @ np.array([0.0, 0.0, 1.5])
    uv = CAM.project(OBJ3, cv2.Rodrigues(Rc.T)[0].ravel(), tvec)
    seq = {"i": 0}

    def det_fn():
        i = seq["i"]
        seq["i"] += 1
        kc = np.array([0.95, 0.95, 0.95, 0.95 if i < 3 else 0.0], np.float32)
        return [Det("gate", 0.95, 0.0, 0.0, 10.0, 10.0, kpts=uv, kpt_conf=kc)]

    task = _task(det_fn)
    _st, hist = _run(task, 8)
    modes = [h["mode"] for h in hist]
    assert modes[0] == "full" and "p3p" in modes, \
        "用例前提：应先 full 后 p3p，实际 %s" % modes
    i_p3p = modes.index("p3p")
    # full 段（航向可信）应当真的出了 yaw；p3p 段必须为 0
    assert any(abs(f[3]) > 0 for f in task.uart.frames[:i_p3p]), \
        "full 段航向通道应该出手（否则本用例证明不了 p3p 被挡住）"
    assert all(abs(f[3]) < 1e-9 for f in task.uart.frames[i_p3p:]), \
        "p3p 下不许用航向通道发 yaw（航向不可信）：%s" \
        % [f[3] for f in task.uart.frames[i_p3p:]]


# ------------------------------------------------- 12. 到门口丢门：判过门而不是回 SEARCH
def _task_no_det():
    """先给可达门口的检测、再整门丢失的装置（coarse 档：角点全缺）。"""
    return None


def test_align_near_lost_declares_pass_instead_of_search():
    """**现场 bug 回归**：width/coarse 的 creep 把船送到门口后整门丢失时，
    ALIGN 相位必须**判过门并直冲**，而不是防抖后回 SEARCH（原先就是后者：0 分 + 原地旋转）。

    判据 = 最后一次检测到的**框占比** ≥ `z.near_lost_ratio`（门快装不下/机身已进门框）。
    """
    from gate.gate_task import PH_SEARCH, PH_THROUGH
    seq = {"i": 0}

    def det_fn():
        seq["i"] += 1
        if seq["i"] <= 6:
            # z=0.6 → 框宽/屏宽 ≈0.71，且角点全缺 → coarse 档、aligned → creep 靠近
            return _det(0.6, kconf=(0.0, 0.0, 0.0, 0.0))
        return []                       # 到门口：整门丢失

    task = _task(det_fn)
    # 帧数要够 THROUGH 走完（confirm_ms=2500 / dt=100ms → 25 帧），故给 60 帧
    _st, hist = _run(task, 60)
    phases = [h["phase"] for h in hist]
    ratio = max(abs(h["ratio"]) for h in hist)
    assert ratio >= float(S.comm.gate.z.near_lost_ratio), \
        "用例前提：框占比应达到判据 %.2f，实际 %.2f" % (
            float(S.comm.gate.z.near_lost_ratio), ratio)
    assert PH_THROUGH in phases, \
        "到门口丢门应判过门并直冲，实际相位序列=%s" % phases[-8:]
    assert task.last_info["pass"] >= 1, "应计到一次过门，实际 pass=%s" % task.last_info["pass"]
    assert hist[-1]["action"] != "search", "不该回 SEARCH（原地旋转 = 0 分）"


def test_align_far_lost_still_searches():
    """远距丢门（框占比小）**仍走原路径**：防抖后回 SEARCH —— 不许误判过门。"""
    from gate.gate_task import PH_SEARCH
    seq = {"i": 0}

    def det_fn():
        seq["i"] += 1
        if seq["i"] <= 4:
            return _det(4.0, kconf=(0.0, 0.0, 0.0, 0.0))   # 远：占比≈0.11
        return []

    task = _task(det_fn)
    _st, hist = _run(task, 40)
    assert PH_SEARCH in [h["phase"] for h in hist], "远距丢门应回 SEARCH 重搜"
    assert task.last_info["pass"] == 0, "远距丢门不许计过门"
    # 允许 coarse 的慢 creep（0.20），但不许出现冲刺速度（= 误判过门后的直冲）
    assert all(f[0] < float(S.comm.gate.surge.through) for f in task.uart.frames), \
        "远距丢门不许直冲"


# ------------------------------------------------- 13. THROUGH 时长（不是帧数）
def _through_frames(ms, monkeypatch, dt=100, n=40):
    from gate.gate_task import PH_THROUGH
    monkeypatch.setitem(S.comm.gate, "through",
                        S.Y(dict(S.comm.gate.get("through", {}), confirm_ms=ms)))
    # z=0.6 ≤ z.cross(0.7) 连 2 帧 → 出口① → THROUGH
    task = _task(lambda: _det(0.6))
    _st, hist = _run(task, n, dt=dt)
    return sum(1 for h in hist if h["phase"] == PH_THROUGH), task


def test_through_duration_is_time_based(monkeypatch):
    """冲刺时长由 `through.confirm_ms` 决定（**与帧数无关**）：500ms/100ms 帧间隔 → 约 5 帧。"""
    n5, task = _through_frames(500, monkeypatch)
    assert 3 <= n5 <= 7, "confirm_ms=500、dt=100ms 应 ≈5 帧，实际 %d 帧" % n5
    assert task.last_info["pass"] >= 1


def test_through_ms_zero_falls_back_to_frames(monkeypatch):
    """兼容开关：`confirm_ms<=0` → 退回旧的**帧数**语义（confirm_frames=8）。"""
    n_frames, _t = _through_frames(0, monkeypatch)
    assert n_frames >= 8, "退回帧数语义后应到 confirm_frames(8) 才结束，实际 %d" % n_frames


# ------------------------------------------------- 14. z 新鲜度：陈旧 z 不许参与穿门判定
def test_coarse_invalidates_stale_z():
    """coarse 档没有任何测距 → 必须把 `_z_ms` 清空（`_z_last` 留着给日志看）。

    现场日志（gate_run1.jsonl）里 width 档算出的 9.25 垃圾 z 一路带到穿门判定；
    反向更危险：陈旧的小 z（≤ near_lost_m）会让"远处丢门"被误判成"已过门"→ 满速盲冲。
    """
    seq = {"i": 0}

    def det_fn():
        seq["i"] += 1
        # 先 width 档（会更新 z），再 coarse 档（没有测距）
        if seq["i"] <= 2:
            return _det(1.2, kconf=(0.95, 0.95, 0.0, 0.0))
        return _det(1.2, kconf=(0.0, 0.0, 0.0, 0.0))

    task = _task(det_fn)
    _run(task, 2)
    z_after_width = task._z_last
    assert task._z_ms is not None, "width 档应打 z 时间戳"
    _run(task, 3)
    assert task._z_ms is None, "coarse 档必须让 z 失效（没有测距）"
    assert task._z_last == z_after_width, "但 _z_last 要原样留着（日志/诊断用）"


def test_stale_z_does_not_fake_a_pass():
    """**回归**：宽度档留下的小 z（0.95 ≤ near_lost_m）在 coarse 阶段**不得**再用来判过门。

    序列：width 档近距（z≈0.95）→ coarse 档远距（占比 0.14）→ 整门丢失。
    正确行为：不算过门（z 已失效、占比远小于阈值）→ 防抖后回 SEARCH。
    """
    from gate.gate_task import PH_SEARCH
    seq = {"i": 0}

    def det_fn():
        seq["i"] += 1
        if seq["i"] <= 3:
            return _det(0.95, kconf=(0.95, 0.95, 0.0, 0.0))   # width：z≈0.95（新鲜且 ≤1.0）
        if seq["i"] <= 8:
            return _det(3.0, kconf=(0.0, 0.0, 0.0, 0.0))      # coarse：远，占比≈0.14
        return []                                            # 整门丢失

    task = _task(det_fn)
    _st, hist = _run(task, 40)
    assert task.last_info["pass"] == 0, \
        "陈旧 z 不该判过门（pass=%s）" % task.last_info["pass"]
    assert PH_SEARCH in [h["phase"] for h in hist], "远距丢门应回 SEARCH"


# ------------------------------------------------- 15. 在门口超时兜底
def test_loiter_timeout_commits_through():
    """**现场回归**：已在门口（占比≥阈值）+ 对准，但**一直没丢检**（门大到仍能整框检出）→
    超过 `loiter.timeout_ms` 必须自己拍板判过门并直冲。

    依据 log/gate_run1.jsonl：coarse 档 commit 出口关着 + 门在占比 0.97 时仍检出
    → 在门口 creep 了 18.7s，最后只靠丢检补一次 1.3s 冲刺（没冲出去）。
    """
    from gate.gate_task import PH_THROUGH
    L = S.comm.gate.loiter
    timeout_ms = float(L.timeout_ms)
    dt = 100
    # 门心始终正中 → in-band；占比 ≈0.71 ≥ near_lost_ratio；**永远不丢检**
    task = _task(lambda: _det(0.6, kconf=(0.0, 0.0, 0.0, 0.0)))
    # 帧数要够：等 timeout_ms + THROUGH 自己走完 confirm_ms（默认 2500ms → 25 帧）
    need = int(timeout_ms / dt) + int(float(S.comm.gate.through.confirm_ms) / dt) + 10
    _st, hist = _run(task, need, dt=dt)
    phases = [h["phase"] for h in hist]
    idx = phases.index(PH_THROUGH) if PH_THROUGH in phases else None
    assert idx is not None, "在门口超时未兜底 → 又会在门口无限 creep"
    assert idx >= int(timeout_ms / dt) - 2, \
        "兜底不该提前触发（应在 %.1fs 之后，实际第 %d 帧）" % (timeout_ms / 1000.0, idx)
    assert task.last_info["pass"] >= 1, "兜底应判过门"
    assert max(f[0] for f in task.uart.frames) >= float(S.comm.gate.surge.through), \
        "兜底后应真的用冲刺速度"


def test_loiter_needs_alignment():
    """在门口但**没对准**（偏出安全带）→ 兜底不许拍板（否则就是闭眼撞杆）。"""
    from gate.gate_task import PH_THROUGH
    dt = 100
    n = int(float(S.comm.gate.loiter.timeout_ms) / dt) + 30
    # 门心偏到右侧 1/4 屏（dxn≈0.5 ≫ dx_max 0.14），但框很大（在门口）
    task = _task(lambda: _det(0.6, px=(CAM.width * 0.5 + CAM.width * 0.25,
                                       CAM.height / 2.0),
                              kconf=(0.0, 0.0, 0.0, 0.0)))
    _st, hist = _run(task, n, dt=dt)
    assert PH_THROUGH not in [h["phase"] for h in hist], \
        "没对准时兜底不许直冲（应继续 HOLD/REACQUIRE）"


def test_loiter_can_be_disabled(monkeypatch):
    """开关可回退：`loiter.enable=false` → 回到"在门口一直 creep"的旧行为。"""
    from gate.gate_task import PH_THROUGH
    monkeypatch.setitem(S.comm.gate, "loiter",
                        S.Y(dict(S.comm.gate.get("loiter", {}), enable=False)))
    task = _task(lambda: _det(0.6, kconf=(0.0, 0.0, 0.0, 0.0)))
    _st, hist = _run(task, 60)
    assert PH_THROUGH not in [h["phase"] for h in hist], "关掉开关后不该兜底直冲"
    assert any(abs(f[0]) > 0 for f in task.uart.frames), "但应该还在 creep（不是完全不动）"


def test_loiter_needs_being_at_the_gate():
    """还没到门口（占比小）→ 兜底不许触发（那是"远距"，该继续靠近/搜索）。"""
    from gate.gate_task import PH_THROUGH
    task = _task(lambda: _det(4.0, kconf=(0.0, 0.0, 0.0, 0.0)))   # 占比≈0.11
    _st, hist = _run(task, 60)
    assert PH_THROUGH not in [h["phase"] for h in hist]
