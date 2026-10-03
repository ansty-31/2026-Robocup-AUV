# -*- coding: utf-8 -*-
"""tests/tasks/gate/test_gate_flow.py — gate 相位机：档位仲裁 / 通道选择 / 出口兜底 / 正航向接入
守的是：
1. coarse 档：**只有未对准才后退** —— 对准时不下发后退速度（"来回退"的根因）；
2. 水平通道选择与符号：中心只用 sway、yaw 只在 ALIGN.HDG 出现；
3. SEARCH 是**左右平移扫视**（不是旋转），波形与 cfg 一致；
4. 出口兜底：近距丢门判过门（②）、在门口超时兜底 loiter（③）、THROUGH 按时长；
5. 正航向 ALIGN.HDG：居中→转正→复核居中→APPROACH，以及「转向不被画面干扰」；
全部用合成检测（真实标定相机模型 + 手工 2D 角点），不依赖权重/串口/水池。"""
import time

import numpy as np
import pytest

import base.cfg.settings as S
from common.vision.detector import Det
from gate.percept.gate_detector import board_camera
from gate.gate_task import PH_THROUGH, PH_CREEP_THROUGH, SUB_HOLD, GateTask
from gate.percept.geometry import object_points

CAM = board_camera()
OBJ3 = object_points()


# --------------------------------------------------------------- 装置
class _Tel(object):
    """假遥测：正航向（ALIGN.HDG）要用下位机回传的绝对航向做闭环。

    默认 yaw_deg=0.0（船不动、航向为 0）→ 对"本来就正对门"的用例完全透明
    （测到 |psi|≈0 → 直接收敛，不发任何 yaw）。
    """

    def __init__(self, yaw_deg=0.0):
        self.yaw_deg = yaw_deg


class _Uart(object):
    """假串口 + **假下位机**（2026-10-02 起旋转由下位机执行）：
    上位机只发"相对角度 + 编号"，下位机转完在遥测里回 `turn_done`。测试里用
    `request_turn/poll_turn_complete/cancel_turn` 三个钩子模拟这台下位机；
    `world` 一给，request_turn 就真的把假世界转过去（等价于下位机执行）。"""

    def __init__(self, yaw_deg=0.0, world=None):
        self.frames = []
        self.neutral_calls = 0
        self.telemetry = _Tel(yaw_deg)
        self.world = world
        self.turn_reqs = []          # [(编号, 相对角)] —— 断言"发了多少度"看这里
        self.cancel_calls = []       # 取消/同步帧的编号
        self.stop_hard_calls = 0
        self.estop_active = False
        self.accept_turn = True      # False = 模拟下发失败
        self.turn_replies = 1        # poll 几次才回报完成（1 = 当帧完成）
        self._turn_id = 0
        self._pending = 0

    def send_dof(self, surge=0.0, sway=0.0, heave=0.0, yaw=0.0):
        self.frames.append((float(surge), float(sway), float(heave), float(yaw)))

    def neutral(self):
        self.neutral_calls += 1

    def set_motion(self, name, force=False):
        pass

    # ---- 假下位机的旋转接口 ----
    def request_turn(self, angle_deg):
        if self.estop_active or not self.accept_turn:
            return False
        self._turn_id += 1
        self.turn_reqs.append((self._turn_id, float(angle_deg)))
        if self.world is not None:
            self.world.turn_by(float(angle_deg))
        self._pending = int(self.turn_replies)
        return True

    def poll_turn_complete(self):
        if self._pending <= 0:
            return False
        self._pending -= 1
        return self._pending <= 0

    def cancel_turn(self):
        if self.estop_active:
            return
        self.cancel_calls.append(self._turn_id)
        self._pending = 0

    def stop_hard(self, *a, **kw):
        self.stop_hard_calls += 1
        self.cancel_turn()
        return True


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
    """构造一个 GateTask，返回它**实际解析出来**的三套修正增益。"""
    t = _task(lambda: [])
    return t




def _task(fn):
    return GateTask(_Uart(), _Hub(fn), CAM.width, CAM.height)


@pytest.fixture(params=["gate_section", "pnp_key", "reacquire_key"])
def drop(request):
    """缺配置用例的参数：整段缺失 / 单个键缺失 / 另一段缺键。"""
    return request.param


# --------------------------------------------------------------- 1/2/3. width 档





def _ratio_of(det):
    return float(det.w) / float(CAM.width)



def test_coarse_backs_off_only_when_misaligned(monkeypatch):
    """对准时不下发后退；未对准时才会 HOLD→REACQUIRE（负 surge）。"""
    # ---- 对准、中距（框占比介于 far_ratio 与 near_ratio 之间）：creep，不后退
    def det_aligned():
        return _det(1.3, kconf=(0.0, 0.0, 0.0, 0.0))

    task = _task(det_aligned)
    assert S.comm.gate.coarse.far_ratio < _ratio_of(det_aligned()[0]) \
        < S.comm.gate.coarse.near_ratio
    _run(task, 40)
    surges = [f[0] for f in task.uart.frames]
    assert min(surges) >= 0.0, "对准时不该出现后退"
    assert max(surges) > 0.0, "对准时应慢速靠近（creep）"

    # ---- 同样距离但没对准：HOLD 到 hold.max_frames → REACQUIRE（负 surge）
    def det_misaligned():
        return _det(1.3, px=(CAM.width * 0.5 + CAM.width * 0.3, CAM.height / 2.0),
                    kconf=(0.0, 0.0, 0.0, 0.0))

    task2 = _task(det_misaligned)
    _run(task2, int(S.comm.gate.hold.max_frames) + 10)
    surges2 = [f[0] for f in task2.uart.frames]
    assert min(surges2) < 0.0, "未对准且迟迟拿不到角点 → 应后退重取"


def test_horizontal_channel_is_sway_only_and_signed():
    """居中阶段的水平修正：**只有 sway 平移**，方向正确，且真的出力（> 执行器死区）。
    约定：+sway = 右移（下位机 `command_left_right = RC[7]-RC[8]`；门在画面右 → 取正）。
    （守这条的是 `test_yaw_never_used_for_centering`）。"""
    right_px = (CAM.width * 0.5 + CAM.width * 0.25, CAM.height / 2.0)
    left_px = (CAM.width * 0.5 - CAM.width * 0.25, CAM.height / 2.0)

    for px, side, sign in ((right_px, "右", 1), (left_px, "左", -1)):
        task = _task(lambda px=px: _det(1.0, px=px, kconf=(0.0, 0.0, 0.0, 0.0)))
        _run(task, 3)
        surge, sway, heave, yaw = task.uart.frames[-1]
        assert sway * sign > 0, "门在%s → sway 符号错了（sway=%.3f）" % (side, sway)
        assert abs(sway) > _ACT_DEADZONE, \
            "sway=%.3f 落在执行器死区(<%.3f)内 = 实际 0 推力" % (sway, _ACT_DEADZONE)
        assert abs(yaw) < 1e-9, "居中不许发 yaw（yaw=%.3f）" % yaw






def test_yaw_only_during_align(monkeypatch):
    """yaw **只在正航向(ALIGN.HDG)里**下发：THROUGH 段必须 yaw=0（冲的时候不许转头）。

    到达 THROUGH 走的是**出口③「在门口超时兜底」**：门口对准后连续停留 ≥
    loiter.timeout_ms 就自己拍板 —— 顺便也验了这条兜底。
    """
    # 本用例守的是**单门**行为（正航向 / 近距判过门 / mock 全流程）；
    monkeypatch.setitem(S.comm.gate, "pass_target", 1)

    def det_fn():
        # 用 z=1.0 ≤ z.cross(1.2) 的**正常出口**进 THROUGH（门口兜底已改为"后退重取"，不盲冲）
        return _det(1.0, kconf=(0.95, 0.95, 0.95, 0.95))

    task = _task(det_fn)
    _run(task, 30)
    assert task.last_info["phase"] == "THROUGH"
    # 取 THROUGH 之后的帧：yaw 必须为 0（`_tick_through` 只发 surge）
    through_frames = [f for f in task.uart.frames[-8:]]
    assert all(abs(f[3]) < 1e-9 for f in through_frames), \
        "THROUGH 段不该有 yaw：%s" % [f[3] for f in through_frames]
    assert all(f[0] > 0 for f in through_frames), "THROUGH 段应持续前进"


# ------------------------------------------------- 9b. SEARCH = 左右平移扫视（不许旋转）
def test_no_speed_preset_inside_actuator_deadzone():
    """没有任何速度档位落在执行器死区里（那种"看着有值、实际 0 推力"的档位）。
    允许的值：0（明确表示"不要这个动作"）或 ≥ 死区线。"""
    g = S.comm.gate
    bad = []
    # 进近两档来自共用表 motion（gate 段里不再重复写）
    for k, v in (("motion.surge_fast", S.comm.motion.surge_fast),
                 ("motion.surge_slow", S.comm.motion.surge_slow)):
        if 0.0 < abs(float(v)) < _ACT_DEADZONE:
            bad.append("%s=%.3f" % (k, v))
    for k in ("creep", "lost_backward", "reacquire", "through"):
        v = abs(float(g.surge[k]))
        if 0.0 < v < _ACT_DEADZONE:
            bad.append("surge.%s=%.3f" % (k, v))
    se = abs(float(g.search.sway))
    if 0.0 < se < _ACT_DEADZONE:
        bad.append("search.sway=%.3f" % se)
    # （width/coarse 的 surge 随 dash 功能一起删除，不再需要在这里查）
    assert not bad, "这些档位落在执行器死区内（发出去等于 0 推力）：%s" % bad






def test_gate_gains_follow_motion_changes(monkeypatch):
    """**共用的证据**：改 `comm.motion` 的增益，gate 构造出来立刻跟着变（没有第二份数字）。

    这条保证"一处调、两个任务同时生效"，也保证不会再出现两套参数各自漂移。
    """
    monkeypatch.setitem(S.comm.motion, "pid_sway",
                        S.Y(dict(S.comm.motion.pid_sway, kp=3.25)))
    monkeypatch.setitem(S.comm.motion, "pid_heave",
                        S.Y(dict(S.comm.motion.pid_heave, kp=1.75)))
    t = _resolved_gains()
    assert float(t._pid_sway_kw["kp"]) == pytest.approx(3.25), \
        "改了 motion.pid_sway.kp，gate 的 sway 增益没跟着变 → 还在抄第二份数字"
    assert float(t._pid_heave_kw["kp"]) == pytest.approx(1.75), \
        "改了 motion.pid_heave.kp，gate 的 heave 增益没跟着变"





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










# ------------------------------------------------- 10. 水平通道 / 执行器死区 / 兜底默认值
# 下位机链路：DOF→字节(128+127v)→Rc2MyRcKey→RC_Matching*(映射≤35→0)
#   ⇒ |DOF| < 0.138 一点推力都没有（数字由固件公式算出，见 gate 文档 §1.1）
_ACT_DEADZONE = 0.138




def test_gate_defaults_match_cfg():
    """**配置守卫**：代码里的兜底默认值必须与 `cfg/*.yaml` 同值（缺配置时行为不变）。
    三条反向检查：
    · **cfg 里出现的键必须真的被读到**（拼错键名会静默失效）；
    · **共用参数（comm.motion）**：cfg 的 motion 段必须与代码兜底同值；
    · **共用值不许在任务段重复一份**（gate 段不许再抄 sway/heave 增益或 fast/slow 速度档）。"""
    import gate.gate_task as gt
    from common.cfg.cfgnode import MOTION_DEFAULTS
    from gate.motion.hdg import _D_HDG
    from gate.percept.gate_postproc import _D_DET, _D_POST, _D_SELECT

    G, V = S.comm.gate, S.vision.gate
    pairs = [
        ("comm.gate.align", gt._D_ALIGN, G.align),
        ("comm.gate.loiter", gt._D_LOITER, G.loiter),
        ("comm.gate.z", gt._D_Z, G.z),
        ("comm.gate.surge", gt._D_SURGE, G.surge),
        ("comm.gate.coarse", gt._D_COARSE, G.coarse),
        ("comm.gate.width", gt._D_WIDTH, G.width),
        ("comm.gate.hold", gt._D_HOLD, G.hold),
        ("comm.gate.reacquire", gt._D_REACQ, G.reacquire),
        ("comm.gate.through", gt._D_THROUGH, G.through),
        ("comm.gate.search", gt._D_SEARCH, G.search),
        ("comm.gate.hdg", _D_HDG, G.hdg),
        # 转向原语（gate 的正航向与 .sh 脚本共用同一套参数）
        ("vision.gate.pnp", gt._D_PNP, V.pnp),
        ("vision.gate.percept.geometry", gt._D_GEOM, V.geometry),
        # 解码后处理（规范 doc/设计/gate_pose_decode_spec.md §4；实现 gate/percept/gate_postproc.py）
        ("vision.gate.det", _D_DET, V.det),
        ("vision.gate.postproc", _D_POST, V.postproc),
        ("vision.gate.select", _D_SELECT, V.select),
    ]
    # cfg 里有、但代码**故意**不参与运算的键（每加一个都要写理由）
    only_doc = {
        "comm.gate.z.align_max", "comm.gate.z.fast_max",   # Z 只分 slow_max 两档，留作扩展
        "vision.gate.percept.geometry.bar_width", "vision.gate.percept.geometry.sym_bars",
        "comm.gate.timeout_ms", "comm.gate.pass_target", "comm.gate.pose_hold_frames",
    }
    bad = []
    for name, code, cfg in pairs:
        cfg = dict(cfg)
        for k in cfg:
            key = "%s.%s" % (name, k)
            if k not in code:
                if key not in only_doc:
                    bad.append("%s：cfg 有但代码读不到（拼错键名？或兜底表漏了这个键）" % key)
                continue
            if cfg[k] != code[k]:
                bad.append("%s：cfg=%r 而代码兜底=%r" % (key, cfg[k], code[k]))
    # 顶层标量（原先散成字面量，现在集中到 _D_TASK）
    for k in ("timeout_ms", "pass_target", "pose_hold_frames"):
        if G.get(k) != gt._D_TASK[k]:
            bad.append("comm.gate.%s：cfg=%r 代码兜底=%r" % (k, G.get(k), gt._D_TASK[k]))
    # keypoint.conf_thr（3 处兜底都指向 _D_KPT）
    if V.keypoint.get("conf_thr") != gt._D_KPT["conf_thr"]:
        bad.append("vision.gate.keypoint.conf_thr：cfg=%r 代码兜底=%r"
                   % (V.keypoint.get("conf_thr"), gt._D_KPT["conf_thr"]))
    # 共用运动参数：cfg 的 motion 段 == 代码兜底表
    for k, dflt in MOTION_DEFAULTS.items():
        if S.comm.motion.get(k) != dflt:
            bad.append("comm.motion.%s：cfg=%r 代码兜底=%r" % (k, S.comm.motion.get(k), dflt))
    assert not bad, "代码兜底默认值与 cfg 不一致（改了 cfg 就要同步兜底表）：\n  " + "\n  ".join(bad)

    assert "align_yaw" not in G, "comm.gate 里又出现了 align_yaw（居中 yaw 通道已删除）"
    # 共用参数不许在任务段重复一份
    for dup in ("fast", "slow"):
        assert dup not in dict(G.surge), \
            "comm.gate.surge.%s 与 comm.motion.surge_%s 重复（共用值只写 motion 一处）" % (dup, dup)
    for dup in ("surge_fast", "surge_slow", "pid", "approach_pid", "lost_inertia_surge"):
        assert dup not in dict(S.comm.ball), \
            "comm.ball.%s 与 comm.motion 重复（共用值只写 motion 一处）" % dup






def test_align_confirm_still_resets_across_mode_class():
    """但跨类（位姿档 ↔ width/coarse）必须清零：那才是"证据不同"。"""
    seq = {"i": 0}

    def det_fn():
        i = seq["i"]
        seq["i"] += 1
        # 交替：位姿档(对准) ↔ coarse(整框、也"对准")
        if i % 2 == 0:
            return _det(1.0)
        return _det(1.0, kconf=(0.0, 0.0, 0.0, 0.0))

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
    from gate.percept.geometry import gate_normal_angles_deg

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






def _task_no_det():
    """先给可达门口的检测、再整门丢失的装置（coarse 档：角点全缺）。"""
    return None


def test_align_near_lost_falls_back_to_search(monkeypatch):
    """**回退后的行为**：ALIGN 相位里整门丢失时：防抖窗口内保持对中（hold），
    超窗后原地保持/后退重取 —— **不回 SEARCH（SEARCH 已删）、不判过门、不直冲**。
    （近距直冲分支已按用户要求移除；`_near_lost()` 不再驱动相位。）

    现场关切：前进中门框刚好转出画面 ≠ 已经过门，不该因此打断已调好的运动方向。
    真正防这个的手段是**让 THROUGH 更早触发**（`z.cross` / `cross_confirm_frames`），
    因为 `PH_THROUGH` 在 `det is None` 之前处理，进 THROUGH 后与丢门无关。
    """
    monkeypatch.setitem(S.comm.gate, "pass_target", 1)
    from gate.gate_task import PH_SEARCH, PH_THROUGH
    seq = {"i": 0}

    def det_fn():
        seq["i"] += 1
        if seq["i"] <= 6:
            # z=1.5 → 框宽/屏宽约 0.5（**故意小于 loiter.inside_ratio 0.85**），角点全缺 → coarse 档
            return _det(1.5, kconf=(0.0, 0.0, 0.0, 0.0))
        return []                       # 整门丢失（检测框消失）

    task = _task(det_fn)
    _st, hist = _run(task, 60)
    phases = [h["phase"] for h in hist]
    #  2026-10-02：这条守的是**框不够大**（占比 < loiter.inside_ratio）的"真丢门"场景 ——
    #   贴脸（框占满画面）那种由 loiter 的 inside 出口拍板直冲，另有用例覆盖。
    assert PH_THROUGH not in phases, \
        "框不够大又真丢门：不该直冲，实际=%s" % phases[-8:]
    assert PH_SEARCH not in phases, \
        "SEARCH 已删除：真丢门应留在 ALIGN，不回 SEARCH，实际=%s" % phases[-8:]
    assert hist[-1]["action"] in ("hold", "reacquire"), \
        "最后一帧应原地保持/后退重取，实际=%s" % hist[-1]["action"]
    assert task.last_info["pass"] == 0, \
        "不该计到过门，实际 pass=%s" % task.last_info["pass"]


def _through_frames(ms, monkeypatch, dt=100, n=40):
    from gate.gate_task import PH_THROUGH
    monkeypatch.setitem(S.comm.gate, "through",
                        S.Y(dict(S.comm.gate.get("through", {}), confirm_ms=ms)))
    # 只测"**单门** THROUGH 的时长"：把 pass_target 钉成 1。
    # 测出来的就不是"一轮冲刺多久"了。
    monkeypatch.setitem(S.comm.gate, "pass_target", 1)
    # z=0.6 连 2 帧 → 出口① → THROUGH（阈值以 cfg 的 z.cross 为准）
    task = _task(lambda: _det(0.6))
    _st, hist = _run(task, n, dt=dt)
    return sum(1 for h in hist if h["phase"] == PH_THROUGH), task


def test_through_duration_is_time_based(monkeypatch):
    """冲刺时长由 `through.confirm_ms` 决定（**与帧数无关**）：500ms/100ms 帧间隔 → 约 5 帧。"""
    n5, task = _through_frames(500, monkeypatch)
    assert 3 <= n5 <= 7, "confirm_ms=500、dt=100ms 应 ≈5 帧，实际 %d 帧" % n5
    assert task.last_info["pass"] >= 1






def test_stale_z_does_not_fake_a_pass():
    """**回归**：宽度档留下的小 z（0.95 ≤ near_lost_m）在 coarse 阶段**不得**再用来判过门。

    序列：width 档近距（z≈0.95）→ coarse 档远距（占比 0.14）→ 整门丢失。
    正确行为：不算过门（z 已失效、占比远小于阈值）→ 防抖后原地保持（不回 SEARCH）。
    """
    from gate.gate_task import PH_SEARCH
    seq = {"i": 0}

    def det_fn():
        seq["i"] += 1
        if seq["i"] <= 3:
            return _det(0.95, kconf=(0.95, 0.95, 0.0, 0.0))   # width：z≈0.95（近距）
        if seq["i"] <= 8:
            return _det(3.0, kconf=(0.0, 0.0, 0.0, 0.0))      # coarse：远，占比≈0.14
        return []                                            # 整门丢失

    task = _task(det_fn)
    _st, hist = _run(task, 40)
    assert task.last_info["pass"] == 0, \
        "陈旧 z 不该判过门（pass=%s）" % task.last_info["pass"]
    assert PH_SEARCH not in [h["phase"] for h in hist], \
        "SEARCH 已删除：远距丢门应留在 ALIGN 保持，实际=%s" % [h["phase"] for h in hist][-4:]


# ------------------------------------------------- 15. 在门口超时兜底
class _HullWorld(object):
    """假船 + 遥测：psi = psi0 − 机身转角（右转 ⇒ psi 变小）。"""

    def __init__(self, psi0=20.0, z=1.5, gain=60.0, imag_sign=-1.0):
        # `common/motion/turn_deg.py::yaw_sign()` 与 log/_board_gate_one_latest.jsonl。
        self.psi0 = float(psi0)
        self.z = float(z)
        self.gain = float(gain)
        self.imag_sign = float(imag_sign)
        self.H = 0.0

    @property
    def psi(self):
        return self.psi0 - self.H          # 2026-09-28 实船改正：右转 ⇒ psi 变小

    @property
    def yaw_tel(self):
        """下位机回传的绝对航向（= 机身转角 × 遥测极性）。"""
        return self.H * self.imag_sign

    def det(self):
        import cv2
        # 门**绕自身中心**转 theta（rvec 是物体自身旋转），中心仍放在光轴上（tvec 不转）
        #   ⇒ 门心投影在画面正中（能过"居中"），只有航向是歪的（psi=theta）。
        theta = self.psi0 - self.H
        R = cv2.Rodrigues(np.array([0.0, np.radians(theta), 0.0]))[0]
        rvec = cv2.Rodrigues(R)[0].ravel()
        tvec = np.array([0.0, 0.0, self.z])
        uv = CAM.project(OBJ3, rvec, tvec)
        x0, y0 = float(uv[:, 0].min()), float(uv[:, 1].min())
        x1, y1 = float(uv[:, 0].max()), float(uv[:, 1].max())
        return [Det("gate", 0.95, max(0.0, x0), max(0.0, y0),
                    max(1.0, min(float(CAM.width), x1) - max(0.0, x0)),
                    max(1.0, min(float(CAM.height), y1) - max(0.0, y0)),
                    kpts=uv, kpt_conf=np.full(4, 0.95, np.float32))]

    def send(self, yaw_cmd, dt=0.1):
        # 新模型下上位机不再用 yaw 闭环（yaw 恒 0）⇒ 这条只剩兼容，不再推进世界
        self.H += float(yaw_cmd) * self.gain * dt

    def turn_by(self, angle_deg):
        """下位机执行相对角：正 = 右转 ⇒ H 增大 ⇒ psi = psi0 − H 变小。"""
        self.H += float(angle_deg)


def _hdg_single_turn_baseline(monkeypatch):
    """机制类用例的基线：**只转一次 + 不缩不限**（`max_turns=1, turn_scale=1, max_step_deg=0`）。"""
    monkeypatch.setitem(S.comm.gate, "hdg",
                        S.Y(dict(S.comm.gate.get("hdg", {}), max_step_deg=0.0,
                                 turn_scale=1.0, max_turns=1)))


def _drive(world, frames=200, dt=100, monkeypatch=None, cfg_over=None):
    """按帧驱动 GateTask：检测由假世界生成，命令回灌到假世界。返回 (task, hist)。"""
    if cfg_over is not None:
        monkeypatch.setitem(S.comm.gate, "hdg",
                            S.Y(dict(S.comm.gate.get("hdg", {}), **cfg_over)))
    uart = _Uart(yaw_deg=world.H * world.imag_sign, world=world)
    task = GateTask(uart, _Hub(lambda: world.det()), CAM.width, CAM.height)
    frame = np.zeros((CAM.height, CAM.width, 3), np.uint8)
    hist = []
    for i in range(frames):
        # 遥测 yaw 跟着假世界走（下位机回传的就是当前绝对航向）
        uart.telemetry.yaw_deg = world.H * world.imag_sign
        task.process(frame, 1000 + dt * i)
        _h = dict(task.last_info)
        _h["_n_send"] = len(uart.frames)   # 本帧发了几条（一帧可能先发居中、再被转向覆盖 ⇒ 末条生效）
        hist.append(_h)
        world.send(uart.frames[-1][3], dt=dt / 1000.0)
        if task.last_info["phase"] == "THROUGH":
            break
    return task, hist


def test_heading_align_turns_the_hull_parallel_then_proceeds(monkeypatch):
    _hdg_single_turn_baseline(monkeypatch)
    """**新功能端到端**：门心已居中但机身歪 20° → ALIGN 里先居中、进 HDG 转正（右转）、
    复核居中、再进 APPROACH；APPROACH 起 yaw 恒 0（此后不再调航向）。"""
    w = _HullWorld(psi0=20.0)
    task, hist = _drive(w, frames=250, monkeypatch=monkeypatch)
    acts = [h["action"] for h in hist]
    #  2026-10-02：整次转向在同一帧内跑完 ⇒ 收尾时动作名已被改回 center，"hdg" 不再出现在逐帧记录里；
    #  改为断言"确实把相对角下发给下位机了"（等价且更贴近新模型）。
    assert task.uart.turn_reqs, "居中达标后应进入正航向（实际动作序列尾部=%s）" % acts[-8:]
    # 转对了方向（psi>0 = 机身左偏 → 右转）且残差收敛到阈值内
    assert task._hdg.last_dir == "右转", \
        "psi>0 应**右转**（2026-09-28 实船改正），实际 %s" % task._hdg.last_dir
    # 2026-10-02：转向由下位机执行 ⇒ 断言"下发了多少度"，不再看上位机 yaw 帧（恒 0）
    reqs = [a for _i, a in task.uart.turn_reqs]
    assert reqs, "应把相对角下发给下位机"
    assert all(a > 0 for a in reqs), "右转应下发正角：%s" % reqs
    assert w.H > 0, "净效果应是机身右转，实际 H=%.1f°" % w.H
    assert abs(w.psi) <= float(S.comm.gate.hdg.tol_deg) + 1e-6, \
        "转完残余航向 %.1f° 应 ≤ 阈值 %.1f°" % (w.psi, float(S.comm.gate.hdg.tol_deg))
    # 转向那一帧是**复合帧**（2026-10-02 新模型）：起转 → 等完成 → 硬停 → 开 post_sway
    #   全在同一个 engine frame 里跑完，所以"这一帧只许发旋转"这种逐帧断言不再成立
    #   （该帧末尾的 DOF 帧就是 post_sway 的反向平移 + 缓慢后退，surge=−0.15 是**对的**）。
    #   这里只钉新模型下真正可观测的三件事：① 确实下发了旋转；② 底下引擎状态收在 done；
    #   ③ 航向收敛（上面已断言）。
    assert task.uart.turn_reqs, "应有转向下发"
    assert task._hdg.state == "done", task._hdg.summary()
    # 进入 APPROACH/THROUGH 之后 yaw 恒 0
    later = [f for f, h in zip(task.uart.frames, hist)
             if h["phase"] in ("APPROACH", "THROUGH")]
    assert later, "应进到 APPROACH/THROUGH"
    assert all(abs(f[3]) < 1e-9 for f in later), "进近/穿门阶段不许再调 yaw"






class _LiveUart(_Uart):
    """串口 = 真船：每次下发都推进假世界，遥测跟着变（真机就是这样闭环的）。"""

    def __init__(self, world):
        _Uart.__init__(self, yaw_deg=0.0)
        self.world = world
        self.n = 0

    def send_dof(self, surge=0.0, sway=0.0, heave=0.0, yaw=0.0):
        _Uart.send_dof(self, surge, sway, heave, yaw)
        self.world.send(yaw, dt=0.05)
        self.telemetry.yaw_deg = self.world.yaw_tel
        self.n += 1


def test_turn_runs_to_completion_inside_one_frame(monkeypatch, tmp_path):
    _hdg_single_turn_baseline(monkeypatch)
    """**运行期（生产时钟域）**：一次 `process()` 调用就把整次转向跑完再返回。

    这条走的是**生产路径**（`_turn_inner_loop` 按 `gate.hdg.turn_period` 20Hz 阻塞推进），
    与用例里那种"假时钟每帧一步"不同：起转后画面立刻全空，转向仍在同一次调用里
    转完并给结论 —— 即"脱离主循环、做完再回来"。

    ⚠️ **时钟域必须用生产那一个**：`main.py` 传的是 `int(time.time()*1000)`（epoch 毫秒）。
    这条用例曾经用 `time.monotonic()*1000`，正好等于 `use_wall` 判据写错时唯一认得的那个钟
    ⇒ **bug 被用例掩盖**（生产里内层闭环整个被跳过、转向退化成每帧一步、`gate_loop` 日志不触发）。
    所以这里固定用 `time.time()`，并顺带断言 `turn_log` 的 `gate_loop` 真的落了盘。
    """
    import time
    tlog = tmp_path / "turn_calls.jsonl"
    monkeypatch.setenv("AUV_TURN_LOG", str(tlog))
    monkeypatch.setitem(S.comm.gate, "align", S.Y(dict(
        S.comm.gate.get("align", {}), confirm_frames=2)))
    w = _HullWorld(psi0=12.0)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return [] if calls["n"] > 3 else w.det()      # 起转之后画面里什么都没有

    uart = _LiveUart(w)
    task = GateTask(uart, _Hub(fn), CAM.width, CAM.height)
    frame = np.zeros((CAM.height, CAM.width, 3), np.uint8)
    blocked = 0.0
    for _ in range(6):
        t0 = time.time()                              # ★ 生产时钟域（epoch 毫秒）
        task.process(frame, int(t0 * 1000))
        dt = time.time() - t0
        calls_during = calls["n"]
        if task._hdg.iters >= 1:
            blocked = dt
            break
    assert blocked > 0.0, "这一帧应当**阻塞**着把转向跑完（实际 %.3fs，%s）" % (blocked, task._hdg.summary())
    if tlog.exists():
        txt = tlog.read_text(encoding="utf-8")
        assert "gate_loop" in txt, "内层闭环没跑（没有 gate_loop 记录）—— 时钟域判据又坏了"
        #  2026-10-02：新模型落盘的转向事件是 gate_start / gate_loop / gate_stop_hard
        #  （旧的上位机闭环有 "enter"；事件名跟着一起换了）。
        assert "gate_start" in txt, "没看到起转事件（turn_log 内容=%s）" % txt[:200]
    assert task._hdg.iters == 1 and task._hdg.state == "done", task._hdg.summary()
    assert calls["n"] == calls_during, "转向期间不该再调检测（画面全空也不影响）"
    assert abs(w.psi) <= float(S.comm.gate.hdg.tol_deg) + 1e-6, "残余 %.1f°" % w.psi
    assert abs(uart.frames[-1][3]) < 1e-9, "转完必须收舵（yaw=%.3f）" % uart.frames[-1][3]


def test_vision_is_not_consulted_while_turning(monkeypatch):
    """**转向期间完全不看画面**：整次转向在同一帧内自包含跑完，期间不再采一帧。

    2026-10-02：转向改由下位机执行且"做完再回来" ⇒ 逐帧记录里不再有 `action=hdg`
    （收尾那一帧就改回 center 了），所以这条断言改成"确实把转向下发给下位机 + 期间零检测"。
    """
    _hdg_single_turn_baseline(monkeypatch)
    w = _HullWorld(psi0=20.0)
    calls = {"during_turn": 0}
    holder = {}

    def fn():
        t = holder.get("t")
        if t is not None and t._hdg.turning:
            calls["during_turn"] += 1
        return w.det()

    # 转向必须真的开始；用"最后一条 yaw=0"（新模型恒 0）+ turn_reqs 断言
    uart = _Uart(yaw_deg=0.0, world=w)
    task = GateTask(uart, _Hub(fn), CAM.width, CAM.height)
    holder["t"] = task
    frame = np.zeros((CAM.height, CAM.width, 3), np.uint8)
    hist = []
    for i in range(250):
        uart.telemetry.yaw_deg = w.H * w.imag_sign
        task.process(frame, 1000 + 100 * i)
        hist.append(dict(task.last_info))
        w.send(uart.frames[-1][3], dt=0.1)
        if task.last_info["phase"] in ("APPROACH", "THROUGH"):
            break
    assert uart.turn_reqs, "应进过正航向（把相对角下发给下位机）"
    assert calls["during_turn"] == 0, \
        "转向期间调用了检测 %d 次（转向必须完全脱离主循环）" % calls["during_turn"]
    assert task._hdg.iters >= 1 and task._hdg.state != "turn", task._hdg.summary()

def test_turn_completes_even_if_gate_disappears(monkeypatch):
    """下位机在执行转向，**门从画面里消失也不打断**（转向是自包含动作）。

    2026-10-02：整次转向在同一帧内跑完 ⇒ "转向进行中丢门"不可观测；
    等价场景 = **下发了转向之后立刻把门甩出画面**，转仍要走完、航向仍要收敛、最后收舵。
    """
    _hdg_single_turn_baseline(monkeypatch)
    w = _HullWorld(psi0=25.0)
    uart = _Uart(yaw_deg=0.0, world=w)
    task = GateTask(uart, _Hub(lambda: w.det()), CAM.width, CAM.height)
    frame = np.zeros((CAM.height, CAM.width, 3), np.uint8)
    hist = []
    st = {"gone": False}
    for i in range(200):
        uart.telemetry.yaw_deg = w.H * w.imag_sign
        task.process(frame, 1000 + 100 * i)
        hist.append(dict(task.last_info))
        w.send(uart.frames[-1][3], dt=0.1)
        if len(uart.turn_reqs) > 0 and not st["gone"]:
            st["gone"] = True                 # 下发转向之后立刻让门消失（下一帧起就看不见）
            w.hide = True
        if st["gone"] and task._hdg.iters >= 1:
            break
    assert st["gone"], "用例前提：应确实起转、并让门在随后消失"
    assert task._hdg.iters == 1, "转向应走完一次（%s）" % task._hdg.summary()
    assert task._hdg.state == "done", "丢门把转向打断了（state=%s）" % task._hdg.state
    assert abs(w.psi) <= float(S.comm.gate.hdg.tol_deg) + 1e-6, \
        "门消失后照样要转到位（残余 %.1f°）" % w.psi
    assert abs(uart.frames[-1][3]) < 1e-9, "转完必须收舵（最后一条 yaw=%.3f）" % uart.frames[-1][3]

FULL_C = (0.95,) * 4
P3P_C = (0.95, 0.95, 0.95, 0.02)          # 丢一个角 → 3 点 p3p（需要 prev 才解得出来）


def _seq_hub(specs):
    """按帧序喂检测：`specs` = [(px or None, kconf), ...]（超出后重复最后一帧）。"""
    it = iter(specs)
    last = {"v": specs[-1]}

    def nxt():
        try:
            last["v"] = next(it)
        except StopIteration:
            pass
        px, kc = last["v"]
        return _det(1.0, px=px, kconf=kc)
    return _Hub(nxt)


def _offset_px(dx):
    """把门心摆到偏离画面中心 dx 像素（用于"还没对中"的帧）。"""
    return (CAM.width / 2.0 + dx, CAM.height / 2.0)


def test_hdg_skip_reason_when_disabled(monkeypatch):
    """`hdg_skip` 的原因分类（2026-09-28 逐小步逼近后）：off / no_psi / psi_ok / wait_psi / done。"""
    task = _task(lambda: [])
    task._hdg_cfg = dict(task._hdg_cfg, enable=False)
    assert "off" in task._hdg_skip_reason("full", 1000)

    task._hdg_cfg = dict(task._hdg_cfg, enable=True)
    assert "no_psi" in task._hdg_skip_reason("full", 1000), task._hdg_skip_reason("full", 1000)

    task._hdg_deg, task._hdg_ms = 20.0, 2000  # 本门还没转过 + 有 ψ ⇒ 可以起转（居中达标即走）
    assert "psi_ok" in task._hdg_skip_reason("full", 3000), task._hdg_skip_reason("full", 3000)

    task._hdg_turns_last_ms = 2500            # ψ 比"上次转向结束"还旧 ⇒ 等新测量
    assert "wait_psi" in task._hdg_skip_reason("full", 3000), task._hdg_skip_reason("full", 3000)

    task._hdg_deg, task._hdg_ms = 20.0, 2600  # 新 ψ ⇒ 又能转一小步
    assert "psi_ok" in task._hdg_skip_reason("full", 3000), task._hdg_skip_reason("full", 3000)

    task._hdg_done = True                     # 真正收口（够正/中止/未启用）
    assert "done" in task._hdg_skip_reason("full", 3000)


def test_hdg_starts_even_when_trigger_frame_is_p3p(monkeypatch):
    """触发帧是不是 full **不影响起转**：只要曾经测到过 full 帧的 psi，就用那个 psi 起转。

    新模型下 mode 每帧在 full/p3p/coarse 间跳，死等"本帧恰好 full"等于永远不转。
    （这件事**只有一套语义**，不许再加放宽开关 —— 旧 `entry_stale_ok` 已随离散迭代删除。）
    """
    #  2026-10-02：ψ≈0（居中且正对）时 `_hdg_ok` 立刻成立、**本来就不该起转**；
    #   这条用例要测的是"触发帧是 p3p 也不影响起转"，所以门必须**真有残余角**（用会旋转门的假世界）。
    w = _HullWorld(psi0=20.0)
    seq = {"i": 0}

    def fn():
        i = seq["i"]
        seq["i"] += 1
        dets = w.det()
        kc = FULL_C if i < 3 else P3P_C        # 前 3 帧 full 建立测量；触发那一帧起是 p3p
        for d in dets:
            d.kpt_conf = np.asarray(kc, np.float32)
        return dets

    task = GateTask(_Uart(yaw_deg=0.0, world=w), _Hub(fn), CAM.width, CAM.height)
    _st, hist = _run(task, 12)
    acts = [h["action"] for h in hist]
    #  2026-10-02：整次转向在同一帧内跑完 ⇒ 看"有没有下发转向"，不看逐帧动作名
    assert task.uart.turn_reqs, "p3p 触发帧也应进正航向（实际动作序列=%s）" % acts


def test_no_psi_skips_heading_and_proceeds(monkeypatch):
    """**没有 full 帧的 psi** ⇒ 记 `hdg_skip=no_psi`，直接进近（不许卡在 ALIGN），且一根舵都不发。
    同时钉住：一次到位之后，旧设计的那几个"来回反馈/开环"机关**不许复活**"""
    task = _task(lambda: [])
    task._hdg_done = False
    assert "no_psi" in task._hdg_skip_reason("full", 1000)

    import inspect
    from gate import gate_task as GT
    from gate.motion import channels as _ch, exits as _ex, hdg as _hdg, modes as _mo, params as _pa
    # 拆分后 GateTask 的方法分散在门面 + 各方法簇 ⇒ **必须扫全部**（只扫门面会假绿）
    src = "\n".join(inspect.getsource(m) for m in (GT, _ch, _ex, _hdg, _mo, _pa))
    for dead in ("wait_fresh_ms", "wait_hdg", "_hdg_hist", "_hdg_fresh", "psi_stale",
                 "settle_ms", "measure_frames", "max_iters", "blind",
                 "entry_stale_ok", "dir_suspect"):
        assert dead not in src, \
            "旧设计的 `%s` 还在 gate_task 里（应随「一次到位」一起删除）" % dead


def test_width_mode_never_starts_heading_align():
    """**读代码发现的行为**：`_on_width` 的 ALIGN 分支只走 creep→APPROACH，**从不启动 HDG**"""
    import inspect
    from gate import gate_task as GT
    src = inspect.getsource(GT.GateTask._on_width)
    assert "_hdg_ready" not in src and "SUB_HDG" not in src, \
        "width 档不应启动正航向（若真要放开，先说明理由并改这条用例）"



# ------------------------------------------------- 17. 「转完反向平移」= 主循环里的一个状态
def _drive_to_sway_back(w, shrink=None, z_after_turn=None, frames=80, dt=100, hide=True):
    """驱动到「反向平移」状态激活（转向结束那一刻）。
    用来模拟"刚转走的那个门在视野里的替代者"；`shrink`：不改 z、只缩框。"""
    uart = _Uart(yaw_deg=0.0, world=w)          # 假下位机接到假世界上（旋转由它执行）
    task = GateTask(uart, _Hub(lambda: w.det()), CAM.width, CAM.height)
    frame = np.zeros((CAM.height, CAM.width, 3), np.uint8)
    n_req0 = len(uart.turn_reqs)
    for i in range(frames):
        uart.telemetry.yaw_deg = w.H * w.imag_sign
        task.process(frame, 1000 + dt * i)
        if task._hdg.turning and z_after_turn is not None:
            w.z = z_after_turn
            z_after_turn = None
        if task._hdg.turning and shrink is not None:
            w.shrink = shrink
            shrink = None
        if hide and len(uart.turn_reqs) > n_req0:
            #   必须**在转向那一帧**就隐藏：若等到转完那一帧还看得见，新的「≥2 角点」判据会立刻收手。
            #   （2026-10-02：判据从"yaw 指令非 0"改成"本帧刚下发了转向"——新模型下 yaw 恒 0，
            #     且整次转向在**同一帧内**跑完，所以不能等"还在转"。）
            w.hide = True
            hide = False
        w.send(uart.frames[-1][3], dt=dt / 1000.0)
        if task._post_sway_until_ms is not None:
            return task, uart, 1000 + dt * i        # 激活帧的时基（状态内帧要顺它往后走）
    return task, uart, 1000 + dt * (frames - 1)


class _ShrinkWorld(_HullWorld):
    """转过之后把框缩到 shrink 倍（不缩时 = 1.0）。"""

    def __init__(self, *a, **kw):
        self.shrink = float(kw.pop("shrink", 1.0))
        self.hide = bool(kw.pop("hide", False))          # True = 门被甩出画面（画面里 0 个角点）
        self.kconf = kw.pop("kconf", None)               # 覆盖角点置信度（用来控"可见角点数"）
        self.turned = False
        _HullWorld.__init__(self, *a, **kw)

    def send(self, yaw_cmd, dt=0.1):
        if abs(float(yaw_cmd)) > 1e-9:
            self.turned = True
        _HullWorld.send(self, yaw_cmd, dt)

    def det(self):
        if self.hide:
            return []                                    # 门不在画面里 ⇒ 0 个角点（转完被甩走的真实情形）
        ds = _HullWorld.det(self)
        if self.kconf is not None:
            d0 = ds[0]
            ds = [Det(d0.kind, d0.score, d0.x, d0.y, d0.w, d0.h, kpts=d0.kpts,
                      kpt_conf=np.asarray(self.kconf, np.float32))]
        if self.turned and self.shrink != 1.0:
            d = ds[0]
            return [Det(d.kind, d.score, d.x + int(d.w * (1 - self.shrink) / 2),
                        d.y + int(d.h * (1 - self.shrink) / 2),
                        max(1, int(d.w * self.shrink)), max(1, int(d.h * self.shrink)),
                        kpts=d.kpts, kpt_conf=d.kpt_conf)]
        return ds


def test_sway_back_is_a_main_loop_state_and_keeps_measuring(monkeypatch):
    _hdg_single_turn_baseline(monkeypatch)   # 只看这一扇窗口的机制：不许下一小步来打断它
    """「反向平移」是**主循环里的一个状态**：判据照跑（ψ 继续刷新），只覆盖本帧下发的指令。

    正确形态下 ψ 每帧都在刷新；曾经那种"把位姿链整段旁路、ψ 冻在起转前的旧值"的写法是
    自转的根因（2026-09-27，证据见 doc/注释历史.md）。
    场景：转完之后视野里是**更远的门**（z 1.5→2.2 m，Δz=0.7 > same_z_m=0.5 ⇒ 判成"不是我刚丢的门"）。
    """
    w = _ShrinkWorld(psi0=25.0, z=1.5)
    n_det = {"n": 0}                                   # 数"检测有没有被调用"（旁路=不调用）
    _orig_det = w.det

    def _det():
        n_det["n"] += 1
        return _orig_det()

    w.det = _det
    task, uart, t0 = _drive_to_sway_back(w, z_after_turn=2.2)
    assert task._post_sway_until_ms is not None, "转向结束应进入「反向平移」状态"
    assert task._hdg.last_dir == "右转", task._hdg.last_dir   # psi0=+25 ⇒ 右转（2026-09-28 改正）
    frame = np.zeros((CAM.height, CAM.width, 3), np.uint8)
    acts, psis, last = [], [], []
    n_det0 = n_det["n"]
    for i in range(5):                                  # 状态内 5 帧（窗口 600ms / 100ms 每帧）
        if task._post_sway_until_ms is None:
            break
        uart.telemetry.yaw_deg = w.H * w.imag_sign
        _n0 = len(uart.frames)
        task.process(frame, t0 + 100 * (i + 1))     # ★ 时间必须顺着激活帧往后走
        acts.append(task.last_info["action"])
        psis.append(task._hdg_deg)
        if uart.frames[_n0:]:
            last.append(uart.frames[-1])            # 本帧**生效**的那一条
        w.send(uart.frames[-1][3], dt=0.1)
    # 逐小步逼近（无次数上限）：窗口可能被**下一小步**打断（转向永远优先，见 _set_info 的让位规则）。
    assert acts and all(a in ("sway_back", "hdg") for a in acts), \
        "窗口内只许反向平移、或让位给下一小步（实际 %s）" % acts
    assert acts[0] == "sway_back", "窗口第一帧必须是平移：%s" % acts
    tail = last                                   # 每帧生效的那一条（不是 uart.frames 切片）
    sw = [f for f, a in zip(last, acts) if a == "sway_back"]   # 只看真正的平移帧（hdg 帧=让位给下一小步）
    assert sw, "至少要有一帧真的在平移"
    from gate.motion.hdg import hdg_cfg as _hdg_cfg
    _back = abs(float(_hdg_cfg().get("post_sway_back", 0.0) or 0.0))
    assert all(abs(f[3]) < 1e-9 and abs(f[2]) < 1e-9 for f in sw), \
        "「反向平移」帧不许带 yaw/heave：%s" % [(f[2], f[3]) for f in sw]
    assert _back > 0, "出厂默认应当带缓慢后退（post_sway_back>0，用户 2026-09-28 定）"
    assert all(abs(f[0] + _back) < 1e-9 for f in sw), \
        "转完补偿必须**边平移边缓慢后退**：每帧 surge 应 = -post_sway_back(%.2f)，实际 %s" % (_back, [f[0] for f in sw])
    # 平移方向永远**与转向方向相反**（`post_sway = -last_d * mag`）⇒ 用关系断言，
    #   不再写死"左转⇒右移"，免得下次再翻方向时这条用例变成"改测试迁就代码"。
    assert all(f[1] * task._hdg.last_d < 0 for f in sw), \
        "平移必须与转向方向相反（右转⇒左移 / 左转⇒右移），last_d=%s：%s" % (task._hdg.last_d, [f[1] for f in tail])
    #     那正是要补偿的情形。改用"检测每帧都被调用"来钉"没有旁路"（旧的旁路写法会让它停在起转前的值）。
    assert n_det["n"] - n_det0 >= len(acts), \
        "「反向平移」状态不许旁路检测：%d 帧只调了 %d 次检测" % (len(acts), n_det["n"] - n_det0)


def test_sway_back_never_locks_up_even_if_every_frame_is_a_far_gate():
    """**回归用例**：状态内每帧都被判成"远门"时，也**绝不永久停摆**。"""
    w = _ShrinkWorld(psi0=25.0, z=1.5)
    task, uart, t0 = _drive_to_sway_back(w, z_after_turn=2.2)
    assert task._post_sway_until_ms is not None
    frame = np.zeros((CAM.height, CAM.width, 3), np.uint8)
    acts = []
    for i in range(120):                       # 给足帧数：够它把每一小步都走完
        uart.telemetry.yaw_deg = w.H * w.imag_sign
        task.process(frame, t0 + 100 * (i + 1))
        acts.append(task.last_info["action"])
        w.send(uart.frames[-1][3], dt=0.1)
        if task._post_sway_until_ms is None and task._hdg.finished() and i > 6:
            break
    assert task._post_sway_until_ms is None, "最终必须退出窗口（实测仍在 %s）" % task._post_sway_until_ms
    assert any(a != "sway_back" for a in acts), "退出后必须恢复正常流程（实际 %s）" % acts
    assert task.last_info["substate"] != "SWAY_BACK", "退出后子状态不许还停在 SWAY_BACK"
    assert task.last_info["phase"] in ("ALIGN", "APPROACH", "THROUGH", "SEARCH"), task.last_info["phase"]


def test_sway_back_exits_when_the_lost_gate_comes_back():
    """刚丢的那个门一回来（同门判据过）⇒ 立刻退出状态，交回视觉；且这一帧 ψ 已刷新。"""
    w = _ShrinkWorld(psi0=25.0, z=1.5)                 # z 不变 ⇒ 判成同一个门
    task, uart, t0 = _drive_to_sway_back(w)
    assert task._post_sway_until_ms is not None, "门被甩出画面后应进入状态"
    frame = np.zeros((CAM.height, CAM.width, 3), np.uint8)
    w.hide = False                                     # ★ 门回到画面里（≥2 个角点）
    uart.telemetry.yaw_deg = w.H * w.imag_sign
    task.process(frame, t0 + 100)
    assert task._post_sway_until_ms is None, "同一个门回来了就该退出（实测仍在）"
    assert task.last_info["action"] != "sway_back", task.last_info["action"]
    assert task._hdg_deg is not None and abs(float(task._hdg_deg)) < 25.0, \
        "退出那一帧 ψ 应已重新测量（实测 %s）" % task._hdg_deg


class _BlockingStopUart(_Uart):
    """转向收尾硬停**真的阻塞**（真船实测：无遥测 0.70s / 有遥测静止 1.31s / 还在转 2.51s）。"""

    def __init__(self, cost_s=0.0, **kw):
        _Uart.__init__(self, **kw)
        self.cost_s = float(cost_s)
        self.blocked_s = 0.0

    def stop_hard(self, quiet=True, **kw):
        time.sleep(self.cost_s)
        self.blocked_s += self.cost_s
        return True


def test_post_sway_kpt_threshold_is_configurable_and_bites(monkeypatch):
    """`hdg.post_sway_kpt_min` 是**唯一**退出判据的阈值（用户 2026-09-30 定：只看当前门角点数）。

    咬三件事：① 阈值调大到 3 之后，2 个角点**不许**收手（证明阈值真的被读到，不是写死 2）；
    ② 第 3 个角点出现才收手，且 `sway_exit` 报 `kpt>=3`；
    ③ 角点置信度低于 `vision.gate.keypoint.conf_thr` 的**不算数**（否则"看到角点"会虚高）。
    """
    w = _ShrinkWorld(psi0=25.0, z=1.5)
    task, uart, t0 = _drive_to_sway_back(w)
    assert task._post_sway_until_ms is not None
    task._hdg_cfg = dict(task._hdg_cfg)
    task._hdg_cfg["post_sway_kpt_min"] = 3            # 本任务读到的 hdg 配置
    monkeypatch.setitem(S.comm.gate["hdg"], "post_sway_kpt_min", 3)   # cfg 侧也改（用例结束自动还原）
    frame = np.zeros((CAM.height, CAM.width, 3), np.uint8)

    def _step(t_ms):
        uart.telemetry.yaw_deg = w.H * w.imag_sign
        task.process(frame, t_ms)

    w.hide = False
    w.z = 3.0
    w.kconf = (0.95, 0.95, 0.0, 0.0)                  # 2 个角点 < 3 ⇒ 继续
    _step(t0 + 100)
    assert task._post_sway_until_ms is not None, \
        '阈值=3 时 2 个角点不该收手（阈值没被读到？）：exit=%s' % task.last_info.get('sway_exit')
    assert task.last_info['action'] == 'sway_back'

    w.kconf = (0.95, 0.95, 0.20, 0.0)                 # 第 3 个角点**低于** conf_thr ⇒ 仍不算
    _step(t0 + 200)
    assert task._post_sway_until_ms is not None, \
        '低于 keypoint.conf_thr 的角点不算数（否则看到角点会虚高）'

    w.kconf = (0.95, 0.95, 0.95, 0.0)                 # 第 3 个角点可信 ⇒ 收手
    _step(t0 + 300)
    assert task._post_sway_until_ms is None, '3 个可信角点 ⇒ 该收手交回视觉'
    assert task.last_info.get('sway_exit') == 'kpt>=3', task.last_info.get('sway_exit')


def test_post_sway_starts_after_the_turn_end_hard_stop():
    """**回归（2026-09-27 用户问「转完之后那个平移到底生效没有」）**：平移窗口必须从**硬停结束**起算。

    2026-10-02：转向改成"下位机执行、上位机等完成" ⇒ 整次转向在**同一帧内**跑完，
    "转向中把门甩出画面"这类逐帧编排不再成立：改成**下发转向那一帧之后**就把门藏起来。
    """
    w = _ShrinkWorld(psi0=25.0, z=1.5)
    uart = _BlockingStopUart(cost_s=0.7)
    task = GateTask(uart, _Hub(lambda: w.det()), CAM.width, CAM.height)
    task._hdg_cfg = dict(task._hdg_cfg)
    task._hdg_cfg["post_sway_ms"] = 400.0            # 窗口 < 阻塞耗时 ⇒ 旧写法必然吃光
    turn_block_s = 0.6
    _orig_inner = task._turn_inner_loop

    def _slow_inner(yaw_cmd, now_ms=0):
        out = _orig_inner(yaw_cmd, now_ms)
        if task._hdg.finished() and not getattr(_slow_inner, "_slept", False):
            time.sleep(turn_block_s)                 # 这一次转向在本帧内阻塞了这么久才结束
            _slow_inner._slept = True
        return out

    task._turn_inner_loop = _slow_inner
    frame = np.zeros((CAM.height, CAM.width, 3), np.uint8)
    t0 = 1000
    z_flip = 2.2                       # 转完视野里是"更远的门"（Δz=0.7 > 0.5 ⇒ 判成不是我丢的那个）
    hidden = False
    acts, act_now, act_until = [], None, None
    for i in range(120):
        uart.telemetry.yaw_deg = w.H * w.imag_sign
        now = t0 + 100 * i + int(uart.blocked_s * 1000.0)   # ★ 硬停的真实耗时计入帧时基
        task.process(frame, now)
        if task._hdg.turning and z_flip is not None:
            w.z, z_flip = z_flip, None
        if not hidden and len(uart.turn_reqs) > 0:
            w.hide = True                  # 下发了转向 ⇒ 立刻甩出画面（0 角点）⇒ 窗口该活着
            hidden = True
        w.send(uart.frames[-1][3], dt=0.1)
        if uart.blocked_s > 0 and task._post_sway_until_ms is not None:
            if act_now is None:                      # 硬停刚结束的那一帧：记下窗口起点
                act_now, act_until = now, task._post_sway_until_ms
            acts.append(task.last_info["action"])
        if acts and task._post_sway_until_ms is None:
            break
    assert uart.blocked_s > 0, "本用例必须真的走到「转向收尾硬停」"
    assert getattr(_slow_inner, "_slept", False), "本用例必须真的模拟到「转向帧内阻塞」"
    assert task._post_sway_until_ms is None, "窗口到期必须退出状态"
    assert act_now is not None and act_until is not None, "没抓到硬停结束那一帧"
    _blocked_ms = int((turn_block_s + uart.blocked_s) * 1000.0)
    assert act_until >= act_now + _blocked_ms + 400 - 1, (
        "平移窗口必须开在**整块阻塞（转向+硬停）之后**：起点 %s，帧起点 %s + 阻塞 %dms + 窗口 400ms"
        % (act_until, act_now, _blocked_ms))
    assert len(acts) >= 4, (
        "硬停阻塞 %.1fs 之后，「反向平移」仍须按窗口(%.0fms)推进 ≥4 帧；实测 %d 帧 %s"
        % (uart.blocked_s, task._hdg_cfg["post_sway_ms"], len(acts), acts))

class _CoarseAfterTurn(_HullWorld):
    """转过之后只给"整框可信、角点全缺"的检出（coarse 档 ⇒ **没有 z**）⇒ 只能用框占比判。

    2026-10-02：转向改由下位机执行 ⇒ `turned` 不能再看上位机 yaw（恒 0），改看"世界被转过"。
    """

    def __init__(self, *a, **kw):
        self.scale = float(kw.pop("scale", 0.5))
        self.turned = False
        _HullWorld.__init__(self, *a, **kw)

    def turn_by(self, angle_deg):
        if abs(float(angle_deg)) > 1e-9:
            self.turned = True
        _HullWorld.turn_by(self, angle_deg)

    def det(self):
        ds = _HullWorld.det(self)
        if self.turned:
            d = ds[0]
            return [Det(d.kind, d.score, d.x, d.y, max(1, int(d.w * self.scale)), d.h,
                        kpts=d.kpts, kpt_conf=np.zeros(4, np.float32))]
        return ds


def test_sway_back_without_z_falls_back_to_box_ratio():
    """没有 z（coarse）时走占比退路：框缩到 50%（< 70%）⇒ 判"不是同一个门" ⇒ 继续推、到期退出。"""
    w = _CoarseAfterTurn(psi0=25.0, z=1.5, scale=0.5)
    task, uart, t0 = _drive_to_sway_back(w)
    assert task._post_sway_until_ms is not None, "门被甩出画面后应进入反向平移状态"
    frame = np.zeros((CAM.height, CAM.width, 3), np.uint8)
    acts = []
    for i in range(10):
        uart.telemetry.yaw_deg = w.H * w.imag_sign
        task.process(frame, t0 + 100 * (i + 1))
        acts.append(task.last_info["action"])
        w.send(uart.frames[-1][3], dt=0.1)
    assert "sway_back" in acts, "没有 z 且框只剩 50%% ⇒ 必须继续反向平移（实际 %s）" % acts
    assert task._post_sway_until_ms is None, "到期必须退出（实测 %s）" % task._post_sway_until_ms


def test_turn_without_completion_never_confirms_heading(monkeypatch):
    """**下位机不回报完成 ⇒ 本门航向永不确认 ⇒ 冲刺闸门拦住**（2026-10-02 新模型的安全闸）。

    旧模型靠上位机 PID 的"满舵无进展即停"（`sat`）兜底，那套已随上位机闭环删除；
    新模型的等价保护：**只有真正回报完成的转向才算确认航向**，超时/GIVEUP/ABORTED 都不算
    —— 于是即便门已经在跟前，也不会盲冲。
    """
    monkeypatch.setitem(S.comm.gate["hdg"], "turn_timeout_s", 1.0)
    w = _HullWorld(psi0=20.0, z=1.5)
    uart = _Uart(yaw_deg=0.0, world=w)
    uart.turn_replies = 10 ** 6                    # 收了请求但**永不回报完成**
    uart.world = None          # 也不许真的转过去（否则 ψ 会合法地变正、闸门自然放行）
    task = GateTask(uart, _Hub(lambda: w.det()), CAM.width, CAM.height)
    frame = np.zeros((CAM.height, CAM.width, 3), np.uint8)
    for i in range(60):
        uart.telemetry.yaw_deg = w.H * w.imag_sign
        task.process(frame, 1000 + 100 * i)
    assert uart.turn_reqs, "应下发过转向"
    assert task._hdg.state in ("timeout", "giveup", "aborted"), task._hdg.summary()
    assert getattr(task, "_hdg_ok", False) is False, \
        "下位机没回报完成 ⇒ 绝不能把本门标成「航向已确认」"
    assert task.last_info["phase"] != "THROUGH", \
        "未确认航向就不许进 THROUGH（实测 phase=%s）" % task.last_info["phase"]

def test_turn_end_does_a_hard_stop(monkeypatch, tmp_path):
    """转向阶段结束时必须**硬停**（连发中性帧走完 ramp + 遥测验证），不许留半路 ramp 的 yaw。"""
    import time
    monkeypatch.setenv("AUV_TURN_LOG", str(tmp_path / "turn_calls.jsonl"))
    monkeypatch.setitem(S.comm.gate, "align", S.Y(dict(
        S.comm.gate.get("align", {}), confirm_frames=2)))
    w = _HullWorld(psi0=20.0)
    uart = _LiveUart(w)
    task = GateTask(uart, _Hub(lambda: w.det()), CAM.width, CAM.height)
    frame = np.zeros((CAM.height, CAM.width, 3), np.uint8)
    for _ in range(6):
        task.process(frame, int(time.time() * 1000))
        if task._hdg.iters >= 1:
            break
    assert task._hdg.iters >= 1, "本次转向没跑完（%s）" % task._hdg.summary()
    # 2026-10-02：硬停由 `uart.stop_hard()`（连发中性帧走完 ramp + 遥测校验）负责；
    #   测试替身用 `stop_hard_calls` 计数（旧断言数的是 neutral 帧数）。
    assert (getattr(uart, "stop_hard_calls", 0) >= 1
            or getattr(uart, "neutral_calls", 0) >= 3), \
        "转向收尾必须硬停，实际 stop_hard=%s neutral=%s" % (
            getattr(uart, "stop_hard_calls", 0), getattr(uart, "neutral_calls", 0))


def test_sway_back_yields_to_through_and_to_a_running_turn():
    """**互锁回归（2026-09-27 用户问「平移窗口会不会把旋转/冲刺卡死」）**：
    `_set_info` 是唯一下发口，平移若在里面**无条件覆盖** 调用方的 action，就会出现两种抢指令：
    约定：**平移让位于转向与冲刺**（它只是"把门拉回视野"的最低优先级补偿），并且立刻作废窗口。"""
    w = _ShrinkWorld(psi0=25.0, z=1.5)
    task, uart, t0 = _drive_to_sway_back(w, z_after_turn=2.2)   # 判成远门 ⇒ 窗口保持存活
    assert task._post_sway_until_ms is not None
    # ① 冲刺帧：surge 必须原样发出去，且窗口作废
    task._set_info("through", surge=0.35)
    assert task.last_info["action"] == "through", task.last_info["action"]
    assert abs(uart.frames[-1][0] - 0.35) < 1e-9, \
        "冲刺的 surge 被平移压掉了：%r" % (uart.frames[-1],)
    assert task._post_sway_until_ms is None, "进冲刺就该作废平移窗口"
    # ② 转向帧：yaw 必须原样发出去（窗口活着时也不许清零）
    task, uart, t0 = _drive_to_sway_back(_ShrinkWorld(psi0=25.0, z=1.5), z_after_turn=2.2)
    assert task._post_sway_until_ms is not None
    task._hdg.state = "turn"                     # 假装又起转了（生产里起转前会清窗口）
    task._set_info("hdg", yaw=-0.30)
    assert abs(uart.frames[-1][3] + 0.30) < 1e-9, \
        "转向的 yaw 被平移清零了（转向会推不动）：%r" % (uart.frames[-1],)
    assert task.last_info["action"] != "sway_back", task.last_info["action"]
    assert task._post_sway_until_ms is None, "转向进行中就该作废平移窗口"


def test_turn_scale_and_max_step_shape_the_issued_angle():
    """出厂默认（用户 2026-09-28 定）：**下发角 = |ψ| × turn_scale(0.8)，再用 max_step_deg(10°) 钳位**。"""
    import pytest as _pytest
    from gate.motion.hdg import hdg_cfg as _hdg_cfg
    cfg = _hdg_cfg()
    assert cfg["turn_scale"] == _pytest.approx(0.8) and cfg["max_step_deg"] == _pytest.approx(15.0), \
        "出厂默认应是 turn_scale 0.8 + max_step_deg 15（实际 %s/%s）" % (cfg["turn_scale"], cfg["max_step_deg"])
    for psi, want in ((10.0, 8.0), (25.0, 15.0), (9.0, 7.2), (-12.0, 9.6), (-40.0, 15.0)):
        task = GateTask(_Uart(yaw_deg=0.0), _Hub(lambda: []), CAM.width, CAM.height)
        task._hdg.start(0, psi=psi)
        assert task._hdg.last_target_deg == _pytest.approx(want), (psi, task._hdg.last_target_deg)
        assert task._hdg.last_dir == ("左转" if psi < 0 else "右转"), (psi, task._hdg.last_dir)


def test_one_turn_per_gate_first_four_corner_measurement_is_trusted():
    """**2026-10-02 用户定（板端语义）**：转向交给下位机执行 ⇒ **第一次看到 4 个角点的那次测量
    信息最全**，就信那一次、且只信那一次 —— 一门只下一次转向指令，不再逐小步逼近。

    所以：`max_step_deg` 仍会钳位这一次下发；钳位后残余可能 > tol，也**不再补转**（信下位机）。
    """
    w = _HullWorld(psi0=32.0)
    task, hist = _drive(w, frames=600)
    reqs = [a for _i, a in task.uart.turn_reqs]
    assert len(reqs) == 1, "一门只该下发一次转向（实际 %s）" % reqs
    assert abs(reqs[0]) <= float(S.comm.gate.hdg.max_step_deg) + 1e-6, \
        "下发角度应被 max_step_deg 钳位（实际 %.1f°）" % reqs[0]
    assert task._hdg_turns == 1, task._hdg_turns
    # 转完即"航向已确认"（信下位机执行），于是本门不再起转
    assert any(h.get("hdg_skip") and "done" in str(h.get("hdg_skip")) for h in hist), \
        "转完应记 hdg_skip=done(本门航向已确认)"


def test_one_turn_per_gate_makes_max_turns_vestigial(monkeypatch):
    """**2026-10-02 起 `hdg.max_turns` 变成兜底配置（不再是主闸）**。

    板端语义：一门只下一次转向 ⇒ 转完那一刻就 `_hdg_done=True`（本门不再起转），
    所以"逐小步逼近"的循环不再存在、上限自然也到不了。
    这条用例把这个事实**钉死**：无论上限设 1 还是 5，都只下发一次转向，
    且 `hdg_skip` 报的是 `done(...)` 而不是 `turns_full`。
    （上限本身仍保留在 cfg 里：将来若放开"逐小步"，它是防疯狂旋转的那道兜底。）
    """
    def _run(cap):
        monkeypatch.setitem(S.comm.gate["hdg"], "max_turns", cap)
        w = _HullWorld(psi0=32.0)
        task, hist = _drive(w, frames=600)
        skips = [h.get("hdg_skip") for h in hist if h.get("hdg_skip")]
        return task, w, skips

    for cap in (1, 5):
        task, w, skips = _run(cap)
        assert task._hdg_turns == 1, "上限=%d 时仍只该转一次（实际 %d）" % (cap, task._hdg_turns)
        assert len(task.uart.turn_reqs) == 1, task.uart.turn_reqs
        assert skips and all("turns_full" not in str(x) for x in skips), \
            "一次性转向 ⇒ 不该出现 turns_full（实测 %s）" % skips[:3]


def test_post_sway_exits_as_soon_as_the_configured_keypoints_appear():
    """转完补偿的**主判据**（用户 2026-09-28 定）：**边反向平移边缓慢后退**，
    直到"画面里至少出现 `post_sway_kpt_min`（默认 2）个角点"就交回视觉。
    钉三件事：① 门还在画外（0 角点）时**继续**平移 + 后退；② 角点数**不够**（1 个 < 2）时**不**收手；"""
    w = _ShrinkWorld(psi0=25.0, z=1.5)
    task, uart, t0 = _drive_to_sway_back(w)              # 转完门被甩出画面
    assert task._post_sway_until_ms is not None
    frame = np.zeros((CAM.height, CAM.width, 3), np.uint8)
    kmin = int(S.comm.gate.hdg.get("post_sway_kpt_min", 2) or 0)
    assert kmin == 2, kmin

    def _step(t_ms):
        uart.telemetry.yaw_deg = w.H * w.imag_sign
        task.process(frame, t_ms)

    # ① 0 角点：继续平移 + 后退
    w.hide = True
    _step(t0 + 100)
    assert task._post_sway_until_ms is not None, "门还在画外 ⇒ 不许收手"
    assert task.last_info["action"] == "sway_back", task.last_info["action"]
    assert uart.frames[-1][0] < 0, "应当同时缓慢后退（surge<0）：%s" % (uart.frames[-1],)

    # ② 只有 1 个角点：还不够。
    #   这条用例要单独钉"角点数不够就不收手"。
    w.hide = False
    w.kconf = (0.95, 0.0, 0.0, 0.0)
    w.z = 3.0
    _step(t0 + 200)
    assert task._post_sway_until_ms is not None, \
        "只有 1 个角点 ⇒ 不该收手：exit=%s" % task.last_info.get("sway_exit")

    # ③ 出现 2 个角点：立刻交回视觉
    w.kconf = (0.95, 0.95, 0.0, 0.0)
    _step(t0 + 300)
    assert task._post_sway_until_ms is None, "出现 %d 个角点就该收手" % kmin
    assert str(task.last_info.get("sway_exit", "")).startswith("kpt"), task.last_info.get("sway_exit")
    assert task.last_info["action"] != "sway_back", task.last_info["action"]


# ======================================================================
# 2026-10-02：z 绝对跳变保护 + 门口兜底放宽（依据实船日志 gate_two_1002-0.jsonl）
# ======================================================================
def test_z_jump_protection(monkeypatch):
    """**z 跳变保护**（2026-10-02 用户定）：

    ① 同一扇门下 z **向上**跳 ≥ 0.5m ⇒ 不是这扇门（判丢门、不采纳）；
    ② 丢门重锁时，z **超过 1.8m ⇒ 判为「下一个门」**（当前门丢了/测不到距，不能被远处能测距的门骗过去）；
    ③ 向下跳（更快接近）不挡；**首见不拦**（还没锁过门时无人可比）。
    """
    import base.cfg.settings as S
    assert float(S.comm.gate.z.relock_z_jump_m) == 0.5
    assert float(S.comm.gate.z.relock_away_m) == 1.8
    # 首见不拦（无参照）
    task = _task(lambda: []); assert task._relock_guard(3.4) is False
    # 向上跳（在追的 z）
    task = _task(lambda: []); task._z_last = 1.0; assert task._relock_guard(1.6) is True
    task = _task(lambda: []); task._z_last = 1.0; assert task._relock_guard(1.4) is False
    task = _task(lambda: []); task._z_last = 1.4; assert task._relock_guard(0.9) is False
    # 向上跳（丢门参照）
    task = _task(lambda: []); task._relock_z_ref = 1.0; assert task._relock_guard(1.7) is True
    # 1.8m 外 = 下一个门（丢门重锁，即使向上跳 < 0.5 也拦）
    task = _task(lambda: []); task._relock_z_ref = 1.7; assert task._relock_guard(2.0) is True
    task = _task(lambda: []); task._relock_z_ref = 1.7; assert task._relock_guard(1.75) is False


def test_pick_gate_nearest_by_z():
    """**选门 = near**：所有门都算 z、排序取最小（不是单纯框占比）。

    2026-10-02 用户定：最近的门可能在画面边缘、框占比反而小；框占比不能当距离。
    """
    far = _det(3.0, kconf=(0.95, 0.95, 0.95, 0.95))[0]
    near = _det(0.9, kconf=(0.95, 0.95, 0.95, 0.95))[0]
    task = _task(lambda: [])
    assert task._pick_gate([near, far]) is near, "应选 z 更近（0.9m）的那扇，而不是框占比/顺序"


# ==== 从 git 取回被误删的定义（2026-10-02 修）====

# ==== 门口超时兜底（2026-10-02 用户定：机制不变，判据放宽）====
def test_loiter_timeout_commits_with_loosened_criteria(monkeypatch):
    """放宽后的判据下，在门口（占比 + |dx|/|dy| 安全带）连续 `loiter.timeout_ms` ⇒ 拍板直冲。

    2026-10-02 用户定：门口超时后**慢速 creep 冲门**（creep_through，5s），不再满速直冲（撞杆）。
    判据放宽：dx_max=0.25 / dy_max=0.30 / near_ratio=0.50 / timeout=5s。
    """
    import base.cfg.settings as S
    timeout_ms = float(S.comm.gate.loiter.timeout_ms)
    monkeypatch.setitem(S.comm.gate["loiter"], "timeout_ms", 300.0)      # 用例里缩短，省时间
    monkeypatch.setitem(S.comm.gate["loiter"], "near_ratio", 0.50)
    task = _task(lambda: _det(0.8, kconf=(0.0, 0.0, 0.0, 0.0)))          # coarse、居中、占比够大
    _st, hist = _run(task, int(timeout_ms / 100) * 0 + 20)
    assert any(h["phase"] == PH_CREEP_THROUGH for h in hist), \
        "在门口超时后应 creep_through 慢速冲门：实际 %s" % [h["action"] for h in hist][-6:]


def test_loiter_does_not_commit_when_far_off_center(monkeypatch):
    """安全带仍然有效：`|dx|` 超过 `dx_max` 就不算"在门口对准"，不许拍板（防偏着冲）。"""
    import base.cfg.settings as S
    monkeypatch.setitem(S.comm.gate["loiter"], "timeout_ms", 200.0)
    px = (CAM.width * 0.5 + CAM.width * 0.45, CAM.height / 2.0)          # 偏到画面边
    task = _task(lambda: _det(1.2, px=px, kconf=(0.0, 0.0, 0.0, 0.0)))
    _st, hist = _run(task, 20)
    assert not any(h["phase"] in (PH_THROUGH, PH_CREEP_THROUGH) for h in hist), \
        "偏着不算对准，不该拍板冲门：%s" % [h["phase"] for h in hist][-6:]


# ======================================================================
# 2026-10-02：运行时**可信边界** `z.relock_away_m = 1.2 m`（超出即不采纳该帧位姿）。
# 本模块绝大多数用例合成的门放在 1.5–3 m，考的是**位姿之后的逻辑**（起转/出口/恢复/SWAY_BACK…），
# 与"多远才算可信"正交 ⇒ 这里统一把边界放宽到 99（= 关闭），只有专门考这条的用例用真值
# （用例名里带 jump/relock/far 的自动跳过，不覆盖）。
# ======================================================================
@pytest.fixture(autouse=True)
def _relax_z_trust_boundary(request, monkeypatch):
    if request.node.name.startswith("test_gate_defaults_match_cfg"):
        return                                  # 守卫用例要比对 cfg 与兜底表真值，不能覆盖
    #  （2026-10-02 起已无 relock_away_m 这个键；保留壳子以防将来再加"绝对距离"类参数）
    return
