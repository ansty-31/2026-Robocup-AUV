# -*- coding: utf-8 -*-
"""tests/tasks/test_gate_flow.py — gate 相位机：档位仲裁 / 通道选择 / 出口兜底 / 正航向接入

守的是：
  1. coarse 档：**只有未对准才后退** —— 对准时不下发后退速度（实测"来回退"的主因）；
  2. 水平通道选择与符号：中心只用 sway、yaw 只在 ALIGN.HDG 出现；
  3. SEARCH 是**左右平移扫视**（不是旋转），波形与 cfg 一致；
  4. 出口兜底：近距丢门判过门（②）、在门口超时兜底 loiter（③）、THROUGH 按时长；
  5. 正航向 ALIGN.HDG：居中→转正→复核居中→APPROACH，以及「转向不被画面干扰」；
  6. 配置守卫：代码兜底 == cfg，且**共用参数只写在 comm.motion**（见 `test_gate_defaults_match_cfg`）。

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
class _Tel(object):
    """假遥测：正航向（ALIGN.HDG）要用下位机回传的绝对航向做闭环。

    默认 yaw_deg=0.0（船不动、航向为 0）→ 对"本来就正对门"的用例完全透明
    （测到 |psi|≈0 → 直接收敛，不发任何 yaw）。
    """

    def __init__(self, yaw_deg=0.0):
        self.yaw_deg = yaw_deg


class _Uart(object):
    def __init__(self, yaw_deg=0.0):
        self.frames = []
        self.neutral_calls = 0
        self.telemetry = _Tel(yaw_deg)

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

    gate 的 cfg 里**不再写增益数字**（共用 `comm.motion` 那一套），所以只能问任务本身。
    """
    t = _task(lambda: [])
    return t




def _task(fn):
    return GateTask(_Uart(), _Hub(fn), CAM.width, CAM.height)


@pytest.fixture(params=["gate_section", "pnp_key", "reacquire_key"])
def drop(request):
    """缺配置用例的参数：整段缺失 / 单个键缺失 / 另一段缺键。

    ⚠️ 这个夹具曾被一次删夹具的正则误伤（连它上面的 `@pytest.fixture` 装饰器一起被吃掉），
    症状是 `fixture 'drop' not found` —— 以后用正则删函数时注意别跨过装饰器行。
    """
    return request.param


# --------------------------------------------------------------- 1/2/3. width 档





def _ratio_of(det):
    return float(det.w) / float(CAM.width)



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


def test_horizontal_channel_is_sway_only_and_signed():
    """居中阶段的水平修正：**只有 sway 平移**，方向正确，且真的出力（> 执行器死区）。

    约定：+sway = 右移（下位机 `command_left_right = RC[7]-RC[8]`；门在画面右 → 取正）。
    ⚠️ 2026-09-20 用户定：**删除像素档/位姿档的 yaw 居中通道**（原 `align_yaw.*`）——
    所以本用例只查 sway；`yaw` 在 gate 里只允许由 ALIGN.HDG 的正航向产生
    （守这条的是 `test_yaw_never_used_for_centering`）。
    """
    right_px = (CAM.width * 0.5 + CAM.width * 0.25, CAM.height / 2.0)
    left_px = (CAM.width * 0.5 - CAM.width * 0.25, CAM.height / 2.0)

    for px, side, sign in ((right_px, "右", 1), (left_px, "左", -1)):
        task = _task(lambda px=px: _det(1.2, px=px, kconf=(0.0, 0.0, 0.0, 0.0)))
        _run(task, 3)
        surge, sway, heave, yaw = task.uart.frames[-1]
        assert sway * sign > 0, "门在%s → sway 符号错了（sway=%.3f）" % (side, sway)
        assert abs(sway) > _ACT_DEADZONE, \
            "sway=%.3f 落在执行器死区(<%.3f)内 = 实际 0 推力" % (sway, _ACT_DEADZONE)
        assert abs(yaw) < 1e-9, "居中不许发 yaw（yaw=%.3f）" % yaw






def test_yaw_only_during_align():
    """yaw **只在正航向(ALIGN.HDG)里**下发：THROUGH 段必须 yaw=0（冲的时候不许转头）。

    到达 THROUGH 走的是**出口③「在门口超时兜底」**（原 dash 出口已删除）：门口对准后
    连续停留 ≥ loiter.timeout_ms 就自己拍板 —— 顺便也验了这条兜底。
    """
    def det_fn():
        # 整框（coarse）且门很大（z=0.6 → 占比≈0.71 ≥ near_lost_ratio 0.60）→ 已到门口
        return _det(0.6, kconf=(0.0, 0.0, 0.0, 0.0))

    task = _task(det_fn)
    _run(task, int(float(S.comm.gate.loiter.timeout_ms) / 100) + 40)
    assert task.last_info["phase"] == "THROUGH"
    # 取 THROUGH 之后的帧：yaw 必须为 0（`_tick_through` 只发 surge）
    through_frames = [f for f in task.uart.frames[-8:]]
    assert all(abs(f[3]) < 1e-9 for f in through_frames), \
        "THROUGH 段不该有 yaw：%s" % [f[3] for f in through_frames]
    assert all(f[0] > 0 for f in through_frames), "THROUGH 段应持续前进"


# ------------------------------------------------- 9b. SEARCH = 左右平移扫视（不许旋转）
def test_search_is_lateral_sweep_not_rotation():
    """**2026-09-20 用户定：过门过程中不允许旋转搜索** → SEARCH 是左右平移扫视。

    守三件事：
      ① **yaw 恒 0**（再也不旋转）；
      ② sway **左右交替**（左右都出现过，且符号都要能过执行器死区 = 真的会动）；
      ③ 波形是**对称**的（右总时长 == 左总时长）——这是"原地左右扫、不漂到池壁"的唯一保证。
    """
    task = _task(lambda: [])            # 全轮无检测 → 一直在 SEARCH
    n = 60                              # dt=100ms → 6s = 一个完整周期(2·(2+1))
    _run(task, n)
    frames = task.uart.frames
    yaws = [f[3] for f in frames]
    assert all(abs(y) < 1e-9 for y in yaws), \
        "SEARCH 还在旋转（yaw=%s）" % sorted(set(round(y, 3) for y in yaws))
    surges = [f[0] for f in frames]
    assert all(abs(s) < 1e-9 for s in surges), "SEARCH 不该前进/后退"
    sw = [f[1] for f in frames]
    assert any(s > 0 for s in sw) and any(s < 0 for s in sw), \
        "SEARCH 没有左右交替（sway 只出现 %s）" % sorted(set(round(s, 3) for s in sw))
    n_right = sum(1 for s in sw if s > 1e-9)
    n_left = sum(1 for s in sw if s < -1e-9)
    assert abs(n_right - n_left) <= 1, \
        "左右扫不对称（右 %d 帧 / 左 %d 帧）→ 净位移不为 0，会一路漂" % (n_right, n_left)
    # 出力必须过执行器死区（否则"扫视"等于原地不动）
    assert min(abs(s) for s in sw if abs(s) > 1e-9) > _ACT_DEADZONE, \
        "平移速度落在执行器死区里 → 实际 0 推力"


def test_search_sweep_waveform_matches_cfg():
    """波形与 cfg 对得上：右 sweep_s → 停 pause_s → 左 sweep_s → 停 pause_s。"""
    g = S.comm.gate.search
    sweep_ms = int(float(g.sweep_s) * 1000)
    pause_ms = int(float(g.pause_s) * 1000)
    task = _task(lambda: [])
    # 每 50ms 采一帧，覆盖一个完整周期
    hist = []
    frame = np.zeros((CAM.height, CAM.width, 3), np.uint8)
    for i in range(int(2 * (sweep_ms + pause_ms) / 50) + 2):
        task.process(frame, 1000 + 50 * i)
        hist.append((50 * i, task.uart.frames[-1][1]))
    # 找"有推力"的两段（>0 和 <0）与两段停顿（==0）
    pos = [t for t, s in hist if s > 1e-9]
    neg = [t for t, s in hist if s < -1e-9]
    zero = [t for t, s in hist if abs(s) < 1e-9]
    assert pos and neg and zero, "一个周期里应包含 右/左/停 三段"
    assert min(pos) == 0, "第一段应是向右（t=0 就开始）"
    assert min(neg) - min(pos) >= sweep_ms, \
        "向右段时长不足 sweep_s（右段起点 %d，左段起点 %d）" % (min(pos), min(neg))
    assert any(t > sweep_ms for t in zero), "向右之后应有停顿（pause_s）"




# 下位机链路实测死区：DOF→字节(128+127v)→Rc2MyRcKey→RC_Matching*(映射≤35→0)
# ⇒ |DOF| < 0.138 一点推力都没有（数字由固件公式算出，见 gate 文档 §1.1）
# 注：**下潜**的死区补偿在底层（base/uart.py::_apply_dive_comp），用例在 tests/platform/test_base.py；
#     这里只留"速度档位不许落在死区里"这条配置守卫。
ACT_DEADZONE = 0.138


def test_no_speed_preset_inside_actuator_deadzone():
    """没有任何速度档位落在执行器死区里（0.12 那种"看着有值、实际 0 推力"）。

    `surge.creep`/`lost_backward` 原来都是 0.12 → 有效推力 **0%**（蠕进/轻微后退从未发生）。
    允许的值：0（明确表示"不要这个动作"）或 ≥ 死区线。
    ⚠️ 2026-09-20 起 **`gate.search.sway` 也一起查**（SEARCH 的平移扫视同样要真的出力）。
    """
    g = S.comm.gate
    bad = []
    # 进近两档来自共用表 motion（gate 段里不再重复写）
    for k, v in (("motion.surge_fast", S.comm.motion.surge_fast),
                 ("motion.surge_slow", S.comm.motion.surge_slow)):
        if 0.0 < abs(float(v)) < ACT_DEADZONE:
            bad.append("%s=%.3f" % (k, v))
    for k in ("creep", "lost_backward", "reacquire", "through"):
        v = abs(float(g.surge[k]))
        if 0.0 < v < ACT_DEADZONE:
            bad.append("surge.%s=%.3f" % (k, v))
    se = abs(float(g.search.sway))
    if 0.0 < se < ACT_DEADZONE:
        bad.append("search.sway=%.3f" % se)
    # （width/coarse 的 surge 随 dash 功能一起删除，不再需要在这里查）
    assert not bad, "这些档位落在执行器死区内（发出去等于 0 推力）：%s" % bad






def test_gate_gains_follow_motion_changes(monkeypatch):
    """**共用的证据**：改 `comm.motion` 的增益，gate 构造出来立刻跟着变（没有第二份数字）。

    这条保证"一处调、两个任务同时生效"，也保证不会再出现两套参数各自漂移
    （sway 反号那次就是"两任务各一套"的产物）。
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










    # （原先还有「直冲安全带必须比对准门槛更严」两条断言 —— dash 功能已整体删除，
    #   安全带 dx_max/dy_max 不再是配置项，故随之去掉。）




# ------------------------------------------------- 10. 水平通道 / 执行器死区 / 兜底默认值
# 下位机链路：DOF→字节(128+127v)→Rc2MyRcKey→RC_Matching*(映射≤35→0)
#   ⇒ |DOF| < 0.138 一点推力都没有（数字由固件公式算出，见 gate 文档 §1.1）
_ACT_DEADZONE = 0.138




def test_gate_defaults_match_cfg():
    """**配置守卫**：代码里的兜底默认值必须与 `cfg/*.yaml` 同值（缺配置时行为不变）。

    为什么值得一条用例：这些字面量曾经悄悄漂开，而漂开的方向有的很危险 ——
      · `_D_SURGE.creep/lost_backward = 0.12` → 落在执行器死区里 = 该动作**0 推力**；
      · `_D_PNP.reproj_px = 8.0` → cfg 实测已放宽到 20，8px 会拒掉绝大多数候选位姿；
      · `_D_THROUGH.confirm_ms = 900` → cfg 现场已改 2500（"没冲出去"的主因）；
      · `_D_ALIGN.px_x/px_y` 0.25/0.26 → 用户已收紧到 0.20/0.25；
      · `timeout_ms` 180000 → cfg 300000。
    三条反向检查：
      · **cfg 里出现的键必须真的被读到**（拼错键名会静默失效 —— `hdg.fresh_ms` 就这么漏过）；
      · **共用参数（comm.motion）**：cfg 的 motion 段必须与代码兜底同值；
      · **共用值不许在任务段重复一份**（gate 段不许再抄 sway/heave 增益或 fast/slow 速度档）。
    """
    import gate.gate_task as gt
    from common.cfgnode import MOTION_DEFAULTS
    from gate.heading_align import _D_HDG

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
        ("vision.gate.pnp", gt._D_PNP, V.pnp),
        ("vision.gate.geometry", gt._D_GEOM, V.geometry),
    ]
    # cfg 里有、但代码**故意**不参与运算的键（每加一个都要写理由）
    only_doc = {
        "comm.gate.z.align_max", "comm.gate.z.fast_max",   # Z 只分 slow_max 两档，留作扩展
        "vision.gate.geometry.bar_width", "vision.gate.geometry.sym_bars",
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

    # 居中**只有 sway**：gate 段不许出现 yaw 居中通道（2026-09-20 用户定，别加回来）
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




# ------------------------------------------------- 13. THROUGH 时长（不是帧数）
def _through_frames(ms, monkeypatch, dt=100, n=40):
    from gate.gate_task import PH_THROUGH
    monkeypatch.setitem(S.comm.gate, "through",
                        S.Y(dict(S.comm.gate.get("through", {}), confirm_ms=ms)))
    # z=0.6 ≤ z.cross(0.77) 连 2 帧 → 出口① → THROUGH
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








# ------------------------------------------------- 16. 正航向（ALIGN.HDG）端到端
class _HullWorld(object):
    """假世界：机身航向 H（正=右）随 yaw 指令积分；门的航向误差 psi = psi0 − H。

    检测用旋转后的位姿投影生成（4 角 → full），门心始终投影在画面正中（所以先"居中"能过），
    只有**航向**是歪的 —— 正是这条新功能要处理的场景。
    """

    def __init__(self, psi0=20.0, z=1.5, gain=60.0, imag_sign=1.0):
        self.psi0 = float(psi0)
        self.z = float(z)
        self.gain = float(gain)
        self.imag_sign = float(imag_sign)
        self.H = 0.0

    @property
    def psi(self):
        return self.psi0 - self.H

    def det(self):
        import cv2
        # 门**绕自身中心**转 theta（rvec 是物体自身旋转），中心仍放在光轴上（tvec 不转）
        #   ⇒ 门心投影在画面正中（能过"居中"），只有航向是歪的（psi=theta）。
        #   ⚠️ 若把中心位置也一起转（tvec=R@[0,0,z]），门心会偏出画面 fx·tanθ 像素 → 永远居不中。
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
        self.H += float(yaw_cmd) * self.gain * dt


def _drive(world, frames=200, dt=100, monkeypatch=None, cfg_over=None):
    """按帧驱动 GateTask：检测由假世界生成，命令回灌到假世界。返回 (task, hist)。"""
    if cfg_over is not None:
        monkeypatch.setitem(S.comm.gate, "hdg",
                            S.Y(dict(S.comm.gate.get("hdg", {}), **cfg_over)))
    uart = _Uart(yaw_deg=world.H * world.imag_sign)
    task = GateTask(uart, _Hub(lambda: world.det()), CAM.width, CAM.height)
    frame = np.zeros((CAM.height, CAM.width, 3), np.uint8)
    hist = []
    for i in range(frames):
        # 遥测 yaw 跟着假世界走（下位机回传的就是当前绝对航向）
        uart.telemetry.yaw_deg = world.H * world.imag_sign
        task.process(frame, 1000 + dt * i)
        hist.append(dict(task.last_info))
        world.send(uart.frames[-1][3], dt=dt / 1000.0)
        if task.last_info["phase"] == "THROUGH":
            break
    return task, hist


def test_heading_align_turns_the_hull_parallel_then_proceeds(monkeypatch):
    """**新功能端到端**：门心已居中但机身歪 20° → ALIGN 里先居中、进 HDG 转正（右转）、
    复核居中、再进 APPROACH；APPROACH 起 yaw 恒 0（此后不再调航向）。"""
    w = _HullWorld(psi0=20.0)
    task, hist = _drive(w, frames=250, monkeypatch=monkeypatch)
    acts = [h["action"] for h in hist]
    assert "hdg" in acts, "居中达标后应进入正航向（实际动作序列尾部=%s）" % acts[-8:]
    # 转对了方向（psi>0 = 机身左偏 → 右转）且残差收敛到阈值内
    ys = [f[3] for f in task.uart.frames if abs(f[3]) > 1e-9]
    assert task._hdg.last_dir == "右转", \
        "psi>0（机身左偏）应右转，实际 %s" % task._hdg.last_dir
    big = [y for y in ys if abs(y) > 0.05]          # 转向主体（近目标会有极小反向修正，属正常）
    assert big and all(y > 0 for y in big), "转向主体应为正（+yaw=右转）：%s" % sorted(set(big))
    assert w.H > 0, "净效果应是机身右转，实际 H=%.1f°" % w.H
    assert abs(w.psi) <= float(S.comm.gate.hdg.tol_deg) + 1e-6, \
        "转完残余航向 %.1f° 应 ≤ 阈值 %.1f°" % (w.psi, float(S.comm.gate.hdg.tol_deg))
    # HDG 期间只转：不前进/不平移/不升降
    hdg_frames = [f for f, h in zip(task.uart.frames, hist) if h["action"] == "hdg"]
    assert hdg_frames, "应有 HDG 帧"
    assert all(abs(f[0]) < 1e-9 and abs(f[1]) < 1e-9 and abs(f[2]) < 1e-9
               for f in hdg_frames), "HDG 期间只许转（surge/sway/heave 必须为 0）"
    # 进入 APPROACH/THROUGH 之后 yaw 恒 0
    later = [f for f, h in zip(task.uart.frames, hist)
             if h["phase"] in ("APPROACH", "THROUGH")]
    assert later, "应进到 APPROACH/THROUGH"
    assert all(abs(f[3]) < 1e-9 for f in later), "进近/穿门阶段不许再调 yaw"






def test_heading_turn_is_not_interrupted_by_degraded_mode(monkeypatch):
    """**转向期间不被画面干扰**（用户定）：先居中再转，转的时候信任测好的航向角 ——
    档位退化到 width（只剩 2 角）、位姿被拒，都**不许打断**正在进行的转向。

    构造：门心居中、机身歪 25° → 进 HDG 开始转；转到一半把检测换成"只有上边 2 角"（width 档）
    → 转向必须照常走完（遥测 yaw 闭环），最终机身转到与门法向平行。
    """
    w = _HullWorld(psi0=25.0)
    phase = {"turning_seen": 0}
    sticky = {"on": False}

    def fn():
        # **一旦开始转向就持续降级**（sticky，之后不再恢复 full）—— 模拟真实的
        #   「档位退化 + 位姿被拒」。⚠️ 若只做间歇降级，打不打补丁都能转完，用例守不住行为。
        fr = task_holder["t"].uart.frames if task_holder["t"] is not None else []
        if sticky["on"] or (fr and abs(fr[-1][3]) > 1e-9):
            sticky["on"] = True
            phase["turning_seen"] += 1
            d = w.det()[0]
            return [Det("gate", 0.95, d.x, d.y, d.w, d.h,
                        kpts=d.kpts, kpt_conf=np.array([0.95, 0.95, 0.0, 0.0], np.float32))]
        return w.det()

    task_holder = {"t": None}
    uart = _Uart(yaw_deg=0.0)
    task = GateTask(uart, _Hub(fn), CAM.width, CAM.height)
    task_holder["t"] = task
    frame = np.zeros((CAM.height, CAM.width, 3), np.uint8)
    hist = []
    for i in range(250):
        uart.telemetry.yaw_deg = w.H * w.imag_sign
        task.process(frame, 1000 + 100 * i)
        hist.append(dict(task.last_info))
        w.send(uart.frames[-1][3], dt=0.1)
        if task.last_info["phase"] in ("APPROACH", "THROUGH"):
            break
    assert phase["turning_seen"] > 0, "用例前提：应确实在转向"
    # 转向必须**走完**（而不是被打断在半路）：迭代次数≥1、且已离开 turn 状态。
    #   注：转完要重测，而 sticky 降级下没有 full 帧 → 它会停在 SETTLE 等重测（这是对的，
    #   不是失败）；要验的就是"这一段转向没被画面打断"。
    assert task._hdg.iters >= 1, "应完成至少一次转向，实际 iters=%d" % task._hdg.iters
    assert task._hdg.state != "turn", \
        "档位退化把转向卡在半路了（state=%s，psi 残余=%.1f°）" % (task._hdg.state, w.psi)
    assert abs(w.psi) <= float(S.comm.gate.hdg.tol_deg) + 1e-6, \
        "转向应走完并转正，残余 %.1f°" % w.psi


# ---------------------------------------------- HDG 跳过原因 / wait_fresh_ms
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
        return _det(1.5, px=px, kconf=kc)
    return _Hub(nxt)


def _offset_px(dx):
    """把门心摆到偏离画面中心 dx 像素（用于"还没对中"的帧）。"""
    return (CAM.width / 2.0 + dx, CAM.height / 2.0)


def test_hdg_skip_reason_when_trigger_frame_is_p3p(monkeypatch):
    """**2026-09-22 现场问题**：居中达标**那一刻**若不是 full 帧 → 正航向被跳过、直接进近。

    帧序：2 帧"没对中但 full"（建立 prev 与 psi）→ 1 帧 full 对中 → 3 帧 p3p 对中
    ⇒ 第 4 个对中帧是 p3p ⇒ 应记录 `hdg_skip=mode=p3p`、进 APPROACH、且全程不发 yaw。
    """
    monkeypatch.setitem(S.comm.gate, "hdg",
                        S.Y(dict(S.comm.gate.get("hdg", {}), wait_fresh_ms=0)))
    specs = [(_offset_px(320), FULL_C), (_offset_px(320), FULL_C),
             (None, FULL_C), (None, P3P_C), (None, P3P_C), (None, P3P_C)]
    task = GateTask(_Uart(), _seq_hub(specs), CAM.width, CAM.height)
    st, hist = _run(task, 12)
    skips = [h.get("hdg_skip") for h in hist if h.get("hdg_skip")]
    assert skips, "应记录跳过正航向的原因（实际没有 hdg_skip；动作=%s）" % [h["action"] for h in hist]
    assert "mode=p3p" in skips[0], "原因应指出帧模式，实际 %r" % skips[0]
    assert task.phase == "APPROACH", "跳过正航向应直接进近，实际 %s" % task.phase
    assert all(abs(f[3]) < 1e-9 for f in task.uart.frames), "跳过后不该发任何 yaw"


def test_hdg_skip_reason_when_disabled(monkeypatch):
    task = _task(lambda: [])
    task._hdg_cfg = dict(task._hdg_cfg, enable=False, wait_fresh_ms=0)
    assert "off" in task._hdg_skip_reason("full", 1000)
    task._hdg_cfg = dict(task._hdg_cfg, enable=True)
    task._hdg_done = True
    assert "done" in task._hdg_skip_reason("full", 1000)
    task._hdg_done = False
    task._hdg_deg, task._hdg_ms = 3.0, 900
    assert "stale" in task._hdg_skip_reason("full", 2000)


def test_wait_fresh_holds_then_aligns_when_a_full_frame_arrives(monkeypatch):
    """`wait_fresh_ms>0`：触发帧不是 full 时**原地等**（动作 wait_hdg、不下发 surge），
    等到 full 帧就进正航向 —— 斜门/远距只在部分帧出 4 角时正需要这个。"""
    monkeypatch.setitem(S.comm.gate, "hdg",
                        S.Y(dict(S.comm.gate.get("hdg", {}), wait_fresh_ms=1500)))
    specs = [(_offset_px(320), FULL_C), (_offset_px(320), FULL_C),
             (None, FULL_C)] + [(None, P3P_C)] * 4 + [(None, FULL_C)] * 6
    task = GateTask(_Uart(), _seq_hub(specs), CAM.width, CAM.height)
    st, hist = _run(task, 30)
    acts = [h["action"] for h in hist]
    assert "wait_hdg" in acts, "应出现等待动作（实际 %s）" % acts[:12]
    waits = [f for f, h in zip(task.uart.frames, hist) if h["action"] == "wait_hdg"]
    assert all(abs(f[0]) < 1e-9 for f in waits), "等待期间不许下发 surge"
    assert "hdg" in acts, "等到 full 后应进入正航向（实际 %s）" % acts[:16]


def test_wait_fresh_times_out_then_proceeds(monkeypatch):
    """等超时了仍然要往下走（不能卡死在 GOLDEN）。"""
    monkeypatch.setitem(S.comm.gate, "hdg",
                        S.Y(dict(S.comm.gate.get("hdg", {}), wait_fresh_ms=300)))
    specs = [(_offset_px(320), FULL_C), (_offset_px(320), FULL_C),
             (None, FULL_C)] + [(None, P3P_C)] * 10
    task = GateTask(_Uart(), _seq_hub(specs), CAM.width, CAM.height)
    st, hist = _run(task, 25)
    acts = [h["action"] for h in hist]
    assert "wait_hdg" in acts, "应先等待（实际 %s）" % acts[:12]
    assert task.phase == "APPROACH", \
        "等超时应进 APPROACH（动作尾部=%s，phase=%s）" % (acts[-5:], task.phase)


def test_width_mode_never_starts_heading_align():
    """**读代码发现的行为**：`_on_width` 的 ALIGN 分支只走 creep→APPROACH，**从不启动 HDG**
    ⇒ 只剩对向 2 角时不会有正航向。这条守着它别被"顺手加上"。"""
    import inspect
    from gate import gate_task as GT
    src = inspect.getsource(GT.GateTask._on_width)
    assert "_hdg_ready" not in src and "SUB_HDG" not in src, \
        "width 档不应启动正航向（若真要放开，先说明理由并改这条用例）"
