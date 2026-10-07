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


def _aligned_px(cam=CAM):
    """**判据意义上的"已对准"落点**（像素）。

    ★ 2026-10-07 用户定：横向基准是**主点**（`cx,cy`）、纵向另有**零点偏置**
    `comm.gate.align.dy_target`（数据实测 −0.30：门恒在光轴下方/上方一侧，
    `dyn` 中位 = −0.30）。所以"已对准"不再是画面正中，而是
    `dxn = 0` 且 `dyn = dy_target` 的那个落点。
    """
    dy = float((S.comm.gate.align or {}).get("dy_target", 0.0) or 0.0)
    return (float(cam.cx), float(cam.cy) + dy * float(cam.height) / 2.0)


def _det(z, px=None, kconf=(0.95, 0.95, 0.95, 0.95), cam=CAM):
    """整门框 bbox（4 角投影）+ 指定角点置信度。

    px=None → 门原点放在**判据意义上的"已对准"**落点（见 `_aligned_px`）；
    px=(u,v) → 放在指定像素（用于构造"没对准"）。
    """
    if px is None:
        px = _aligned_px(cam)
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




def test_coarse_never_creeps_forward_except_the_doorstep_fallback():
    """★★ 2026-10-07 用户定：**coarse 除了门口超时兜底，一律不许触发前进**。

    原来 coarse 在"对准"时会 `creep` 往前蹭（注释写着"争取露出角点"）—— **已删**：
    盲着往前顶既露不出角点，又离撞门更近。

    现在 coarse 的 ALIGN：
      · 远距小框 / 中距 ⇒ 只做**横向**（`surge = 0`）或原地 HOLD
      · 未对准且贴脸 / HOLD 超时(`coarse.hold_timeout_ms=5s`) ⇒ **后退**（负 surge）+ 按角点反向慢移
      · 唯一的前进 = `_loiter_commit`（门口超时兜底）或下视红杆 ⇒ `creep_through`
    """
    # ① 对准 + 中距：**不许前进**（原来这里会 creep）
    def det_aligned():
        return _det(1.3, kconf=(0.0, 0.0, 0.0, 0.0))

    task = _task(det_aligned)
    assert S.comm.gate.coarse.far_ratio < _ratio_of(det_aligned()[0]) \
        < S.comm.gate.coarse.near_ratio
    _run(task, 40)
    surges = [f[0] for f in task.uart.frames]
    assert max(surges) <= 0.0, "coarse 对准时**不许前进**（实际最大 surge=%s）" % max(surges)

    # ② 未对准 + HOLD 超时 ⇒ 后退（负 surge）
    def det_misaligned():
        return _det(1.3, px=(CAM.width * 0.5 + CAM.width * 0.3, CAM.height / 2.0),
                    kconf=(0.0, 0.0, 0.0, 0.0))

    task2 = _task(det_misaligned)
    task2._hold_start_ms = None
    n = 12
    dt = int(float(S.comm.gate.coarse.hold_timeout_ms) / n) + 60
    _run(task2, n, dt=dt)
    surges2 = [f[0] for f in task2.uart.frames]
    assert min(surges2) < 0.0, "未对准 + HOLD 超时 5s → 应后退重取（实际 %s）" % surges2
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
        # ★ 2026-10-07：门的 z 要**小于 `z.cross`**（现 1.2，与 z.slow_max/width.z_max 同值）
        #   才走得到"正常出口"进 THROUGH；这里取 0.6 留足余量。
        return _det(0.6, kconf=(0.95, 0.95, 0.95, 0.95))

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






def test_gate_sway_no_wasted_thrust_and_no_bang_bang_at_deadzone():
    """sway 通道的**不变量**（与具体调参无关，别再断言"比例区要占几档"那种调参假设）。

    守住四条真正该守的：
      ① 死区内 = 0（不然会原地抖）
      ② 出了死区**就给能推动船的推力**（≥ 执行器死区 0.138）—— 否则"误差有了却推不动"
         （当年 turn_deg 的教训：输出顶在死区上等于没推力）
      ③ 输出**单调不减**、且封顶 `out_max`（不能随误差反而变小）
      ④ 配了 `bias` 死区补偿（只降 kp 会把"一直顶满"换成"一直没推力"）

    关于"比例区多宽"：那取决于 kp/bias/out_max 的取舍（现场标定项），**不是本用例该钉的**。
    """
    task = _task(lambda: [])
    kw = task._pid_sway_kw
    om, kp = float(kw["out_max"]), float(kw["kp"])
    dz = float(kw["deadzone"])
    bias = float(kw.get("bias", 0.0) or 0.0)
    DEAD_ACT = 0.138                       # cfg 自注：低于它推进器无推力

    # ④ 死区补偿
    assert bias >= DEAD_ACT - 1e-9, (
        "必须配了死区补偿（bias=%.3f）——只降 kp 会把『一直顶满』换成『一直没推力』" % bias)

    errs = (0.0, dz * 0.5, dz * 0.9, dz * 1.1, 0.05, 0.06, 0.09, 0.15, 0.30, 0.42, 0.60)
    outs = []
    for i_, e in enumerate(errs):
        task._pid_sway_px.reset()
        task._pid_sway_px.update(e, now_ms=0)          # 预热（消掉 D 项启动跳变）
        outs.append(abs(task._lateral_out(e, 1000 + 100 * i_)))

    # ① 死区内为 0
    for e, o in zip(errs, outs):
        if e <= dz + 1e-9:
            assert o == 0.0, "死区内(|e|=%.3f≤%.3f)必须 0：%s" % (e, dz, outs)
    # ② 出了死区就有能推得动的推力
    for e, o in zip(errs, outs):
        if e > dz + 1e-9:
            assert o >= DEAD_ACT - 1e-9, (
                "|e|=%.3f 有误差但输出 %.3f 低于执行器死区 %.3f ⇒ 白给：%s" % (e, o, DEAD_ACT, outs))
    # ③ 单调不减 + 封顶
    nz = [o for o in outs if o > 1e-9]
    assert nz == sorted(nz), "输出必须随误差单调不减：%s" % outs
    assert max(outs) <= om + 1e-9, "不许超 out_max：%s" % outs
    assert kp > 0 and om > 0
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




def test_gate_cfg_keys_are_all_readable():
    """**配置守卫（无兜底版）**：`cfg/*.yaml` 里出现的每个键，代码都必须**真的读到**。

    ★ 2026-10-07 用户定：**删掉代码里的兜底默认值**（改参数要同步两处，纯鸡肋）。
    删掉之后"同值比较"没有意义了，但这条**反向检查仍然值钱**：
    键名拼错 ⇒ 代码 `req()` 读不到 ⇒ 要么启动报缺、要么 cfg 里的值静默失效。
    所以这里逐个键核对"它在代码的键名清单里"。
    """
    import gate.gate_task as gt
    from gate.motion.params import (_K_ALIGN, _K_LOITER, _K_Z, _K_SURGE, _K_COARSE,
                                    _K_WIDTH, _K_HOLD, _K_REACQ, _K_THROUGH, _K_SEARCH,
                                    _K_HDG, _K_PNP, _K_GEOM, _K_TASK, _K_KPT)
    from gate.percept.gate_postproc import _K_DET, _K_POST, _K_SELECT

    G, V = S.comm.gate, S.vision.gate
    pairs = [
        ("comm.gate.align", _K_ALIGN, G.align),
        ("comm.gate.loiter", _K_LOITER, G.loiter),
        ("comm.gate.z", _K_Z, G.z),
        ("comm.gate.surge", _K_SURGE, G.surge),
        ("comm.gate.approach", (("tier",), ), None),
        ("comm.gate.coarse", _K_COARSE, G.coarse),
        ("comm.gate.width", _K_WIDTH, G.width),
        ("comm.gate.hold", _K_HOLD, G.hold),
        ("comm.gate.reacquire", _K_REACQ, G.reacquire),
        ("comm.gate.through", _K_THROUGH, G.through),
        ("comm.gate.search", _K_SEARCH, G.search),
        ("comm.gate.hdg", _K_HDG, G.hdg),
        ("vision.gate.pnp", _K_PNP, V.pnp),
        ("vision.gate.percept.geometry", _K_GEOM, V.geometry),
        ("vision.gate.det", _K_DET, V.det),
        ("vision.gate.postproc", _K_POST, V.postproc),
        ("vision.gate.select", _K_SELECT, V.select),
    ]
    # cfg 里有、但代码**故意**不参与运算的键（每加一个都要写理由）
    only_doc = {
        "comm.gate.z.align_max", "comm.gate.z.fast_max",   # Z 只分 slow_max 两档，留作扩展
        "vision.gate.percept.geometry.bar_width", "vision.gate.percept.geometry.sym_bars",
    }
    bad = []
    for name, keys, cfg in pairs:
        if cfg is None:
            continue
        for k in dict(cfg):
            key = "%s.%s" % (name, k)
            if k not in keys and key not in only_doc:
                bad.append("%s：cfg 有但代码读不到（拼错键名？或键名清单漏了它）" % key)
    assert not bad, "cfg 与代码键名清单不一致：\n  " + "\n  ".join(bad)
    # 顶层标量（现在集中成 _K_TASK）
    for k in ("timeout_ms", "pass_target", "pose_hold_frames"):
        assert G.get(k) is not None, "comm.gate.%s 缺失（代码已无兜底）" % k
        assert k in _K_TASK
    # keypoint.conf_thr
    assert V.keypoint.get("conf_thr") is not None and "conf_thr" in _K_KPT
    # approach.tier（字符串档位）
    assert (S.comm.gate.get("approach") or {}).get("tier"), "comm.gate.approach.tier 缺失"
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
        "相位机不回 SEARCH（仍是 ALIGN；旋转搜索由 _scan 在 ALIGN 内驱动 yaw），实际=%s" % phases[-8:]
    # ★ 2026-10-04 用户定：还没稳定锁上门 ⇒ 用公共慢扫**旋转搜索**（不是干等）
    assert hist[-1]["action"] == "search", \
        "最后一帧应是旋转搜索（search），实际=%s" % hist[-1]["action"]
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






def test_through_is_two_stage_fast_then_slow(monkeypatch):
    """★ 2026-10-07 用户定：**两段式冲刺**。

    `cross` 1.2→1.5 之后起冲点更远（要走完 ~2.5m），单段满速要么冲不够、要么过冲撞对面 ⇒
    段① `fast_ms` 用 `surge.through` 快冲、段② 到 `confirm_ms` 降到 `surge.creep` 保持。
    `fast_ms=0` ⇒ 整段满速（旧行为，一行回退）。
    """
    from gate.gate_task import PH_THROUGH
    import base.cfg.settings as _S

    def _surges(confirm_ms, fast_ms):
        monkeypatch.setitem(_S.comm.gate, "through",
                            _S.Y(dict(_S.comm.gate.get("through", {}),
                                      confirm_ms=confirm_ms, fast_ms=fast_ms)))
        monkeypatch.setitem(_S.comm.gate, "pass_target", 1)
        task = _task(lambda: _det(0.6))
        _st, hist = _run(task, 40, dt=100)
        return [h["surge"] for h in hist if h["phase"] == PH_THROUGH],                [h.get("through_stage") for h in hist if h["phase"] == PH_THROUGH]

    fast = float(_S.comm.gate.surge.through)
    slow = float(_S.comm.gate.surge.creep)
    assert slow < fast, "用例前提：降速段必须比快冲段慢（creep < through）"

    # 段①（0~200ms 快）→ 段②（200~500ms 慢）：2 帧快 + 3 帧慢
    su, stg = _surges(500, 200)
    assert su, "应进过 THROUGH"
    assert su[0] == pytest.approx(fast), "第 1 帧应是快冲段：%s" % su
    assert su[-1] == pytest.approx(slow), "末帧应是降速段：%s" % su
    assert "fast" in stg and "slow" in stg, "两段都要出现：%s" % stg
    assert stg[0] == "fast" and stg[-1] == "slow"
    assert all(s <= fast + 1e-9 for s in su), "任何一帧都不该超过快冲档：%s" % su

    # 单段（fast_ms=0）：全程满速 —— 旧行为，可一行回退
    su0, stg0 = _surges(500, 0)
    assert su0 and all(s == pytest.approx(fast) for s in su0),         "fast_ms=0 时应整段满速：%s" % su0
    assert set(stg0) == {"fast"}, stg0


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

    def __init__(self, psi0=20.0, z=2.0, gain=60.0, imag_sign=-1.0):
        # ⚠️ z 必须**明显大于** `comm.gate.z.cross`（2026-10-07 由 1.2 提到 1.5）：
        #   否则每帧都落进"够近"分支（`_on_pose` 里的 `z <= cross`），**永远走不到 ALIGN 的转向逻辑**；
        #   原来写 1.5 正好等于 cross，靠浮点尾数侥幸过关 —— 边界上不可依赖。
        #   上限是 `vision.gate.pnp.z_max`（2026-10-07 压到 3.0）。
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
        # ★ 2026-10-07：门心放到**判据意义上的"已对准"**落点（含纵向零点 dy_target），
        #   否则新判据会把这个"门心在光轴上"的假世界判成"没对准"。
        _px = _aligned_px(CAM)
        tvec = CAM.backproject_z(_px[0], _px[1], self.z).reshape(3, 1)
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
    w = _ShrinkWorld(psi0=25.0, z=2.0)
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
    w = _ShrinkWorld(psi0=25.0, z=2.0)
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


def test_sway_back_exits_when_the_lost_gate_comes_back(monkeypatch):
    """刚丢的那个门一回来（同门判据过）⇒ 立刻退出状态，交回视觉；且这一帧 ψ 已刷新。"""
    w = _ShrinkWorld(psi0=25.0, z=2.0)                 # z 不变 ⇒ 判成同一个门
    task, uart, t0 = _drive_to_sway_back(w)
    assert task._post_sway_until_ms is not None, "门被甩出画面后应进入状态"
    # ★ 2026-10-06：收手判据多了两道 —— **转向后先连停 settle 帧**（新画面才可信）
    #   和"门可见且**居中**"。本用例专注测"角点出现"，把 settle 显式设 0。
    monkeypatch.setitem(S.comm.gate["hdg"], "post_sway_settle_frames", 0)

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
    w = _ShrinkWorld(psi0=25.0, z=2.0)
    task, uart, t0 = _drive_to_sway_back(w)
    assert task._post_sway_until_ms is not None
    task._hdg_cfg = dict(task._hdg_cfg)
    task._hdg_cfg["post_sway_kpt_min"] = 3            # 本任务读到的 hdg 配置
    monkeypatch.setitem(S.comm.gate["hdg"], "post_sway_kpt_min", 3)   # cfg 侧也改（用例结束自动还原）
    # ★ 2026-10-06：收手判据多了两道 —— **转向后先连停 settle 帧**（新画面才可信）
    #   和"门可见且**居中**"。本用例专注测"角点出现"，把 settle 显式设 0。
    monkeypatch.setitem(S.comm.gate["hdg"], "post_sway_settle_frames", 0)

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
    assert task.last_info.get('sway_exit') in ('kpt>=3', 'kpt>=3且居中'), task.last_info.get('sway_exit')


def test_post_sway_starts_after_the_turn_end_hard_stop():
    """**回归（2026-09-27 用户问「转完之后那个平移到底生效没有」）**：平移窗口必须从**硬停结束**起算。

    2026-10-02：转向改成"下位机执行、上位机等完成" ⇒ 整次转向在**同一帧内**跑完，
    "转向中把门甩出画面"这类逐帧编排不再成立：改成**下发转向那一帧之后**就把门藏起来。
    """
    w = _ShrinkWorld(psi0=25.0, z=2.0)
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
    w = _CoarseAfterTurn(psi0=25.0, z=2.0, scale=0.5)
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
    w = _HullWorld(psi0=20.0, z=2.0)
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
    w = _ShrinkWorld(psi0=25.0, z=2.0)
    task, uart, t0 = _drive_to_sway_back(w, z_after_turn=2.2)   # 判成远门 ⇒ 窗口保持存活
    assert task._post_sway_until_ms is not None
    # ① 冲刺帧：surge 必须原样发出去，且窗口作废
    task._set_info("through", surge=0.35)
    assert task.last_info["action"] == "through", task.last_info["action"]
    assert abs(uart.frames[-1][0] - 0.35) < 1e-9, \
        "冲刺的 surge 被平移压掉了：%r" % (uart.frames[-1],)
    assert task._post_sway_until_ms is None, "进冲刺就该作废平移窗口"
    # ② 转向帧：yaw 必须原样发出去（窗口活着时也不许清零）
    task, uart, t0 = _drive_to_sway_back(_ShrinkWorld(psi0=25.0, z=2.0), z_after_turn=2.2)
    assert task._post_sway_until_ms is not None
    task._hdg.state = "turn"                     # 假装又起转了（生产里起转前会清窗口）
    task._set_info("hdg", yaw=-0.30)
    assert abs(uart.frames[-1][3] + 0.30) < 1e-9, \
        "转向的 yaw 被平移清零了（转向会推不动）：%r" % (uart.frames[-1],)
    assert task.last_info["action"] != "sway_back", task.last_info["action"]
    assert task._post_sway_until_ms is None, "转向进行中就该作废平移窗口"


def test_turn_scale_and_max_step_shape_the_issued_angle():
    """下发角 = `min(|psi| × turn_scale, max_step_deg)` —— 按 **cfg 现配**核对（不含默认值断言）。

    ⚠️ 2026-10-07 沿革：一时改成"不缩放不钳位"（1.0/0），用户随后按现场需要又调回
    **0.8 / 15.0**（cfg 是唯一来源，这里只跟 cfg 对齐，不再硬编码期望值）。
    """
    import pytest as _pytest
    from gate.motion.hdg import hdg_cfg
    cfg = hdg_cfg()
    ts, cap = float(cfg["turn_scale"]), float(cfg["max_step_deg"])
    assert ts > 0
    for psi in (10.0, 25.0, 42.0, -12.0):
        task = _task(lambda: [])
        task._hdg.start(0, psi=psi)
        want = abs(psi) * ts
        if cap > 0:
            want = min(want, cap)
        assert abs(task._hdg.last_target_deg) == _pytest.approx(want), (psi, task._hdg.last_target_deg)
        assert str(task._hdg.last_dir) == ("右转" if psi > 0 else "左转")
def test_one_turn_per_gate_first_four_corner_measurement_is_trusted():
    """**2026-10-02 用户定（板端语义）**：转向交给下位机执行 ⇒ **第一次看到 4 个角点的那次测量
    信息最全**，就信那一次、且只信那一次 —— 一门只下一次转向指令，不再逐小步逼近。

    所以：`max_step_deg` 仍会钳位这一次下发；钳位后残余可能 > tol，也**不再补转**（信下位机）。
    """
    w = _HullWorld(psi0=32.0)
    task, hist = _drive(w, frames=600)
    reqs = [a for _i, a in task.uart.turn_reqs]
    assert len(reqs) == 1, "一门只该下发一次转向（实际 %s）" % reqs
    cap = float(S.comm.gate.hdg.max_step_deg)
    assert cap <= 0 or abs(reqs[0]) <= cap + 1e-6, \
    "max_step_deg 配正数才钳；现配 0=不钳，下发的应是完整 ψ（实际 %s°）" % abs(reqs[0])
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


def test_post_sway_exits_as_soon_as_the_configured_keypoints_appear(monkeypatch):
    """转完补偿的**主判据**（用户 2026-09-28 定）：**边反向平移边缓慢后退**，
    直到"画面里至少出现 `post_sway_kpt_min`（默认 2）个角点"就交回视觉。
    钉三件事：① 门还在画外（0 角点）时**继续**平移 + 后退；② 角点数**不够**（1 个 < 2）时**不**收手；"""
    w = _ShrinkWorld(psi0=25.0, z=2.0)
    task, uart, t0 = _drive_to_sway_back(w)              # 转完门被甩出画面
    assert task._post_sway_until_ms is not None
    # ★ 2026-10-06：收手判据多了两道 —— **转向后先连停 settle 帧**（新画面才可信）
    #   和"门可见且**居中**"。本用例专注测"角点出现"，把 settle 显式设 0。
    monkeypatch.setitem(S.comm.gate["hdg"], "post_sway_settle_frames", 0)

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
    ③ 向下跳（更快接近）不挡；**首见不拦**（还没锁过门时无人可比）；
    ④ **2.5m 开外完全不相信**（2026-10-04 用户定）：无条件第一条，不看参照、不看跳没跳。
    """
    import base.cfg.settings as S
    assert float(S.comm.gate.z.relock_z_jump_m) == 0.5
    assert float(S.comm.gate.z.relock_away_m) == 1.8
    # ★ 2026-10-05 用户定：**加回来** —— 2.5m 开外**完全不相信**（无条件，首见也拦）
    # ★ 2026-10-07 用户定：dist_max_m = **2.5**（"远端阈值保持 2.5"；3 米开外本就不准）
    assert float(S.comm.gate.z.dist_max_m) == 2.5
    task = _task(lambda: []); assert task._relock_guard(4.2) is True
    task = _task(lambda: []); assert task._relock_guard(2.8) is True
    # 首见不拦（无参照）
    task = _task(lambda: []); assert task._relock_guard(2.0) is False
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


def test_gate_lock_keeps_same_gate_and_draws_only_it():
    """★ 2026-10-04 用户定：**选门后锁定**，且 HUD 只画锁定那扇门。

    · 第 1 帧按 near 选中左侧那扇（z=2.0）→ 锁定；
    · 第 4 帧起右侧那扇变成 z=1.0（比锁定门"更近"）→ **不许换过去**
      （否则转向目标角会跳到"后边那扇门"的偏移角上）；
    · `last_dets` 每帧最多 1 个 = 只画锁定那扇门的框。
    """
    stable_n = int(S.comm.gate.lock.stable_frames)
    seq = {"i": 0}

    def det_fn():
        seq["i"] += 1
        a = _det(2.0, px=(CAM.width * 0.30, CAM.height / 2.0))      # 左：先近 → 应被锁定
        b = _det(3.5 if seq["i"] <= stable_n + 2 else 1.0,
                 px=(CAM.width * 0.75, CAM.height / 2.0))           # 右：**稳定锁定之后**才变"更近"
        return a + b

    task = _task(det_fn)
    frame = np.zeros((CAM.height, CAM.width, 3), np.uint8)
    drawn = []
    for i in range(stable_n + 6):
        task.process(frame, 1000 + 100 * i)
        drawn.append(list(task.last_dets))

    # ① 每帧最多画一个框（锁定后不再画别的门）
    assert all(len(d) <= 1 for d in drawn), \
        "锁定后画面只该画锁定那扇门，实际每帧框数=%s" % [len(d) for d in drawn]
    # ② 第 1 帧锁定左侧近门，之后**一直**是它（右边更近也不换）
    assert drawn[0], "第 1 帧该选到一扇门"
    cx0 = drawn[0][0].x + drawn[0][0].w * 0.5
    assert cx0 < CAM.width * 0.5, "第 1 帧该锁定左侧那扇（框中心 x=%.0f）" % cx0
    for i, ds in enumerate(drawn):
        assert ds, "帧 %d 锁定门丢了（不该丢）" % i
        cx = ds[0].x + ds[0].w * 0.5
        assert cx < CAM.width * 0.5, \
            "帧 %d 换到了右边那扇门（中心 x=%.0f）—— 锁定失效，会误转后边门的偏移角" % (i, cx)


def test_hdg_ok_latch_requires_sustained_stability(monkeypatch):
    """★ 2026-10-04 用户定：ψ 必须**连续 ok_frames 帧**都在 tol 内才锁存"航向 OK"。

    动机（实船 `log/rungate_1.jsonl`）：ψ 在 1.5m 处标准差 22°、船真实 yaw 标准差只 3°；
    |ψ|≤8° 的最长连续段只有 4 帧。旧的"单帧落进阈值就锁"被噪声钉死 ⇒ 该转的不转。
    """
    task = _task(lambda: [])
    g = S.comm.gate
    n = int(g.hdg.ok_frames)
    assert n >= 2
    # 连续 n-1 帧达标 → **还不许**锁存
    for _ in range(n - 1):
        task._hdg_deg = 1.0
        task._hdg_ok_tick(8.0)
    assert task._hdg_ok is False, "只连续 %d 帧达标就锁存了（ok_frames=%d）" % (n - 1, n)
    # 第 n 帧达标 → 锁存
    task._hdg_deg = 1.0
    task._hdg_ok_tick(8.0)
    assert task._hdg_ok is True, "连续 %d 帧达标该锁存" % n
    # 破了要清零重数
    task2 = _task(lambda: [])
    for _ in range(n - 1):
        task2._hdg_deg = 1.0
        task2._hdg_ok_tick(8.0)
    task2._hdg_deg = 30.0            # 中间破一次
    task2._hdg_ok_tick(8.0)
    assert task2._hdg_ok_cnt == 0 and task2._hdg_ok is False, "破了该清零重数"
    task2._hdg_deg = 1.0
    task2._hdg_ok_tick(8.0)
    assert task2._hdg_ok is False, "清零后单帧达标不该锁存"


def test_gate_lock_k_consistency_rejects_a_different_gate(monkeypatch):
    """★ 2026-10-05 用户定：**k 一致性检验 = 保护锁定**（与跳变保护同性质），位置在 `_gate_lock`。

    同一扇门 k = z×框占比 ≈ 常数（实测 full 0.599 / p3p 0.457，理论 k_true=0.560）。
    候选的 k 离锁定参照太远 ⇒ 那个框**不是锁定那扇门**（哪怕框中心恰好靠得近）⇒ 不匹配。
    """
    task = _task(lambda: [])
    frame_w = CAM.width
    a = _det(1.5, px=(CAM.width * 0.5, CAM.height / 2.0))[0]        # 锁定门
    b = _det(1.5, px=(CAM.width * 0.5, CAM.height / 2.0))[0]        # 同位干扰门
    # a 的 k 正常；b 让 _z_est 报一个崩小的 z ⇒ k 明显偏小 ⇒ 必须被判"不是同一扇"
    # ★ 2026-10-07：`k_lo_ratio` 0.75→0.30 ⇒ 干扰门的 k 要明显更小才构得成"偏小"（0.20 而非 0.45）
    monkeypatch.setattr(task, "_z_est", lambda d, c: 1.5 if d is a else 0.20)
    task._gate_lock([a], frame_w)                     # 锁上 a，建立 k 参照
    assert task._lock_k_ref is not None, "刚锁上就该建立 k 参照"
    k_lo = float(S.comm.gate.select.k_lo_ratio)
    assert task._k_of(b) < k_lo * task._lock_k_ref, "构造的干扰门 k 应该明显偏小（否则用例没意义）"
    task._gate_lock([b], frame_w)                     # 只给 b
    assert task._locked_det is a, \
        "k 明显不一致的框仍被当成锁定门 ⇒ k 一致性保护失效（会把别的门当成锁定门）"


def test_gate_lock_only_becomes_exclusive_after_stable(monkeypatch):
    """★ 2026-10-04 用户定：**锁定前务必做好检查 —— 锁定稳定后再不参与选门**。

    · 稳定前：选到别的门 → 换锁（选错必须能纠正）；
    · 连续跟住 `lock.stable_frames` 帧后：别的门（哪怕 z 更小）再也抢不走；
    · 但"锁定门这帧没被检测到"仍计时，连续 `miss_frames` 帧才解锁。
    """
    stable_n = int(S.comm.gate.lock.stable_frames)
    task = _task(lambda: [])
    frame_w = CAM.width
    a = _det(1.6, px=(CAM.width * 0.45, CAM.height / 2.0))[0]      # 锁定门
    b = _det(3.0, px=(CAM.width * 0.85, CAM.height / 2.0))[0]      # 干扰门（**画面另一侧**，不该匹配上）
    # 第 1 次只是"初次选中"，之后每帧 +1 ⇒ 需要 stable_n+1 次才稳定
    for _ in range(stable_n + 1):
        task._gate_lock([a], frame_w)
    assert task._lock_stable is True, "连续 %d 帧后该稳定" % stable_n
    # 稳定后：即便只给"别的门"，也不换（会按没检测到计时，但阈值内仍保留锁定）
    task._gate_lock([b], frame_w)
    assert task._locked_det is a, "稳定后不该被别的门顶掉"


def test_gate_rotates_to_search_when_no_gate_and_not_locked_stable():
    """★ 2026-10-04 用户定：门任务**新增旋转搜索**（替换掉原来被删的左右 sway 扫）。

    无门可锁（det 全空）且**还没稳定锁定** ⇒ 用公共慢扫旋转找门：
    · yaw 由 `_scan` 给（闭环慢扫），不是 0（不是干等）；
    · **绝不转圈**：净转动量夹在 ±span 预算内；
    · **不碰 hdg**：hdg 仍在"门出现且 ψ 需要修"时走它自己那条路（另有用例守）。
    """
    task = _task(lambda: [])
    task.uart.telemetry.yaw_deg = 0.0          # 闭环要遥测才动
    frame = np.zeros((CAM.height, CAM.width, 3), np.uint8)
    for i in range(60):
        task.process(frame, 1000 + 100 * i)
    yaws = [f[3] for f in task.uart.frames]
    assert any(abs(y) > 1e-9 for y in yaws), "无门时该旋转搜索，实际一直不动（等于没实现）"
    assert task.last_info["action"] == "search", \
        "无门且未稳定锁定 ⇒ 动作应是 search，实际=%s" % task.last_info["action"]
    span = float(S.comm.motion.search_scan.span_deg)
    net = abs(task.uart.telemetry.yaw_deg)
    assert net <= 2.0 * span + 1e-6, \
        "净转了 %.1f°（超过 ±span=%.0f°）⇒ 有转圈/失控风险" % (net, span)


def test_distance_refs_survive_gate_loss_but_clear_on_new_round():
    """★ 2026-10-05 用户定：判断远近的基准**只在开启新一轮时清除**，丢门时必须保留。

    实船坑：`_tick_lost` 开头刚记下丢门前基准（`_relock_z_ref=0.83`/`_relock_ratio_ref=0.83`），
    尾部"远处丢门 → `_start_search()`"又全部清零 ⇒ 3.27m 的远门进来时三条判据都"没有参照"，
    被"首见不拦"放行；框占比 0.83→0.228 的暴跌也没人管。
    """
    task = _task(lambda: [])
    task._z_last = 0.83
    task._relock_z_ref = 0.83
    task._relock_ratio_ref = 0.83
    task._pose_hist = [(1000, 0.83, 0.0, 0.0)]
    task._start_search()                      # ← 丢门（默认 new_round=False）
    assert task._z_last == 0.83, "丢门时不该清 _z_last（清了几何参照就没了）"
    assert task._relock_z_ref == 0.83, "丢门时不该清 _relock_z_ref"
    assert task._relock_ratio_ref == 0.83, "丢门时不该清 _relock_ratio_ref"
    assert task._pose_hist, "丢门时不该清位姿历史（门口判据要用）"
    task._start_search(new_round=True)        # ← 过完门 = 新一轮
    assert task._z_last is None and task._relock_z_ref is None \
        and task._relock_ratio_ref is None and task._pose_hist == [], \
        "开启新一轮时才该把这些基准清干净"


def test_area_ref_is_tracked_continuously(monkeypatch):
    """★ 2026-10-05 用户定：**选门优先 z、锁门用面积**；面积参照必须**持续跟踪**。

    实船坑：门是"被判成另一扇"而不是"完全丢"（det 一直有），`_tick_lost` 从没跑过
    ⇒ `_relock_ratio_ref` 一直 None ⇒ 占比 0.65→0.27 的暴跌没人管。
    """
    task = _task(lambda: _det(1.5))          # 一直有门（不丢）
    frame = np.zeros((CAM.height, CAM.width, 3), np.uint8)
    for i in range(4):
        task.process(frame, 1000 + 100 * i)
    assert getattr(task, "_ratio_last", None) is not None, \
        "被采纳的帧必须记下「最近被采纳的占比」（面积参照源）"
    ref = task._ratio_last
    # 模拟"换到另一扇门"：占比暴跌到 0.3× ⇒ 面积保护必须命中（= 判为另一个门）
    task._relock_ratio_ref = ref
    task._dbg_ratio = ref * 0.3
    far = float(S.comm.gate.z.relock_far_ratio)
    assert task._dbg_ratio < far * task._relock_ratio_ref, \
        "占比摔到 0.3× 参考值时，面积保护该命中（relock_far_ratio=%.2f）" % far
    assert task._relock_guard_ratio() is True, \
        "占比暴跌（0.65→0.27 那种）没被面积保护拦住 ⇒ 会朝另一扇门开过去"


def test_lock_guard_runs_before_center_matching(monkeypatch):
    """★ 2026-10-05 用户定：锁定保护（k 一致性 + 面积）**前移到框中心匹配之前**，
    且"能测出来就用"：k 能算就用 k，算不出才退回面积。

    实船没启动的根因：老写法要等 `best_d <= match_ratio×画面宽` 才算 `matched`，
    换到远处那扇门时中心差太远 ⇒ `matched=None` ⇒ **k 检验整段被跳过**，
    直接走丢帧→解锁→重选，把远门选走。
    """
    task = _task(lambda: [])
    frame_w = CAM.width
    a = _det(1.5, px=(CAM.width * 0.45, CAM.height / 2.0))[0]     # 锁定门（近、框大）
    b = _det(4.0, px=(CAM.width * 0.90, CAM.height / 2.0))[0]     # 远处那扇（框小、在另一侧）
    # ★ 关键：b 报一个**崩小的 z**（实船那个坑）—— 框很小(0.14)却自称 1.0m ⇒
    #   k = 1.0×0.14 = 0.14，而锁定门 k ≈ 1.5×0.373 = 0.56 ⇒ 比 0.25 < k_lo(0.75) ⇒ 必须拦
    monkeypatch.setattr(task, "_z_est", lambda d, c: 1.5 if d is a else 1.0)
    task._gate_lock([a], frame_w)                                  # 锁 a
    assert task._lock_k_ref is not None, "锁上后该建立 k 参照"
    # ① k 能测 ⇒ 用 k：b 的 k 明显偏小，必须在匹配前被排除
    assert task._k_of(b) is not None, "b 应能算出 k（本用例前提）"
    assert task._lock_guard_reject(b) is True, \
        "更远那扇（k 明显偏小）必须在框中心匹配前就被排除 —— 否则 k 检验又会被跳过"
    assert task._lock_guard_reject(a) is False, "锁定门自己不该被排除"
    # ② k 测不出 ⇒ 退回面积
    task._locked_det = a
    task._lock_k_ref = None
    task._relock_ratio_ref = 0.55                                  # 参照占比
    small = _det(3.5, px=(CAM.width * 0.45, CAM.height / 2.0))[0]  # 同位但更小（更远）
    monkeypatch.setattr(task, "_k_of", lambda d: None)             # 模拟 coarse：测不出 k
    assert task._lock_guard_reject(small) is True, \
        "k 测不出时必须退回比面积，占比小太多=更远那扇门"


def test_through_requires_the_mode_own_centering_range():
    """★ 2026-10-07 用户定：**务必在"各自要求的居中范围"内触发过门**。

    过门居中闸 = **本档位自己的对中带** × `through.loose`，再与 `through.center_x/center_y` 取 AND。
    ⚠️ 不要假设"width 的带一定比位姿档宽" —— 那是某一轮的值；现在 width(0.15/0.10) 的竖直带
    **比 align(0.15/0.20) 更紧**。本用例只断言"按各自带判、且与 center_x/y 取严"。
    """
    task = _task(lambda: [])
    al = S.comm.gate.align
    w = S.comm.gate.width
    tc = S.comm.gate.through
    # 位姿档：按 align 的带
    assert task._center_ok(float(al.px_x) * 0.9, float(al.px_y) * 0.9, al) is True
    assert task._center_ok(float(al.px_x) * 1.1, 0.0, al) is False
    # width 档：按 width 的带（**与位姿档不同**，各判各的）
    wx, wy = float(w.align_x), float(w.align_y)
    assert task._center_ok(wx * 0.9, wy * 0.9, w) is True
    assert task._center_ok(wx * 1.1, 0.0, w) is False
    # 与 through.center_x/center_y 取 AND ⇒ 只会更严
    assert task._center_ok(0.0, 0.0, w) is True
    assert task._center_ok(float(tc.center_x) + 0.01, 0.0, w) is False
    assert task._center_ok(0.0, float(tc.center_y) + 0.01, w) is False
def test_through_main_exit_requires_sustained_centering(monkeypatch):
    """★ 2026-10-05 用户定：**主出口（够近那条）重新加回居中闸门** —— 必须"居中成功"
    **连续 through.center_frames 帧**才许冲刺。实船报的"没居中就冲刺"。

    两条阈值是分开的（用户定）：
      · align.px_x/px_y = 0.15/0.20 —— 管"对中动作精细度"
      · through.center_x/center_y = 0.20/0.25 —— 管"够不够正才敢冲"，**更松**（抗运动抖动）
    """
    task = _task(lambda: [])
    task._hdg_ok = True                      # 航向闸先放行，单测居中闸
    need = int(S.comm.gate.through.center_frames)
    assert need >= 2
    # 居中不足 ⇒ 不许冲
    task._center_ok_cnt = need - 1
    assert task._start_through() is False, "居中不足（%d/%d 帧）竟然放行冲刺" % (need - 1, need)
    assert "off_center" in str(task.last_info.get("through_block")), \
        "该在 through_block 里写明是被居中闸拦的，实际=%s" % task.last_info.get("through_block")
    # 连续达标 ⇒ 放行
    task._center_ok_cnt = need
    assert task._start_through() is True, "连续 %d 帧居中了还不放行" % need
    # ⚠️ 兜底出口（bypass_hdg=True）不受居中闸限制
    t2 = _task(lambda: [])
    t2._center_ok_cnt = 0
    assert t2._start_through(bypass_hdg=True) is True, \
        "兜底出口（门口超时那条救命路）不该被居中闸卡死"
    # 阈值必须是"更松"的那套
    cx, cy = float(S.comm.gate.through.center_x), float(S.comm.gate.through.center_y)
    px, py = float(S.comm.gate.align.px_x), float(S.comm.gate.align.px_y)
    assert cx >= px and cy >= py, \
        "冲刺居中阈值(%.2f/%.2f)不该比 align(%.2f/%.2f) 更严" % (cx, cy, px, py)


def test_area_guard_yields_near_door_and_when_box_touches_frame_edge():
    """★ 2026-10-06 用户定：面积保护在 **(a) z≤cross（贴门口)**、**(c) 检测框触边（门框出画）**
    时必须让位 —— 那两种情况下"框占比"已经不代表距离。

    实船误判现场：船贴近时门框撑出画面，检测 bbox 只剩可见部分 ⇒ 占比**必然**腰斩
    （帧134：0.398→0.177，z 已冻在 1.17 ≤ cross）⇒ 老的面积判据把**贴脸的门自己**
    判成"更远那扇门" ⇒ 锁定门被自己人拒收 ⇒ 一路 hold/creep，永远进不了 THROUGH。
    """
    task = _task(lambda: [])
    task._locked_det = _det(1.5, px=(CAM.width * 0.5, CAM.height / 2.0))[0]
    task._lock_ratio_ref = 0.40
    task._lock_k_ref = None                     # 只看面积那层
    small = _det(3.5, px=(CAM.width * 0.5, CAM.height / 2.0))[0]     # 占比明显更小
    assert small.w / CAM.width < 0.7 * 0.40, "用例前提：这个框占比要明显小于 0.7×参照"
    # 基线：不豁免时必须拒（否则用例没意义）
    task._z_last = 3.0
    assert task._lock_guard_reject(small) is True, "基线：远离门口时占比变小该拒"
    # (a) 贴门口（锁定门最近采信 z ≤ z.cross）⇒ 不做面积判据
    task._z_last = float(S.comm.gate.z.cross) - 0.05
    assert task._lock_guard_reject(small) is False, \
        "(a) 贴门口不该再拿框占比判距离（门框撑出画面，占比必然变小）"
    # (c) 检测框触到画面边缘（门框被裁掉）⇒ 同样不做面积判据
    task._z_last = 3.0
    edge = _det(3.5, px=(CAM.width * 0.015, CAM.height / 2.0))[0]
    assert task._lock_guard_reject(edge) is False, \
        "(c) 框触边=门框出画，框占比不代表距离，不该做面积判据"


def test_pick_gate_keeps_only_largest_k(monkeypatch):
    """★ 2026-10-06 用户定：选门时**只留 k 最大的那一档**（k ≥ k_max_ratio × max_k），其余排除。

    实船：同一扇门**框占比稳定而 z 抖 ±35%** ⇒ k 在两簇间跳（0.48 / 0.87）。
    k 小 = z 报得**比实际近**（测距崩了）⇒ 会被"z_eff 最小"误选成最近的门
    （一开始锁到**后边那扇**就是这个）。所以先按 k 把"明显测不准的"剔掉，再按 z_eff 选。
    """
    import base.cfg.settings as _S          # 局部导入，避开作用域问题
    task = _task(lambda: [])
    a = _det(1.5, px=(CAM.width * 0.5, CAM.height / 2.0))[0]
    b = _det(1.5, px=(CAM.width * 0.5, CAM.height / 2.0))[0]   # 同框位置 ⇒ 同占比
    # a 报崩小的 z(1.0) ⇒ k 小；b 报 2.0 ⇒ k 大
    monkeypatch.setattr(task, "_z_est", lambda d, c: 1.0 if d is a else 2.0)
    k_ratio = float(_S.comm.gate.select.k_max_ratio)
    assert k_ratio > 0
    # ① 有 k 最大档筛选 ⇒ 小 k 那扇被排除，选 b
    assert task._pick_gate([a, b]) is b, \
        "小 k（=测距崩小、看着比实际近）那扇没被排除 ⇒ 会锁错门"
    # ② 关掉筛选（k_max_ratio=0）⇒ 退回"z_eff 最小"，会选到小 k 那扇（反证筛选在起作用）
    monkeypatch.setitem(_S.comm.gate, "select",
                        _S.Y(dict(_S.comm.gate.select, k_max_ratio=0.0)))
    assert task._pick_gate([a, b]) is a, "k_max_ratio=0 时应退回 z_eff 最小（选到 a）"
    # ③ 两扇 k 接近（同档）⇒ 都保留，仍按 z_eff 选
    monkeypatch.setitem(_S.comm.gate, "select",
                        _S.Y(dict(_S.comm.gate.select, k_max_ratio=k_ratio)))
    monkeypatch.setattr(task, "_z_est", lambda d, c: 1.0 if d is a else 1.05)
    assert task._pick_gate([a, b]) is a, "同档 k 时不该误排除，仍按 z_eff 最小选"


def test_width_mode_also_enforces_jump_protection():
    """★ 2026-10-06 用户定：**"2.5m 开外完全不相信"必须贯彻始终** —— width 档也得有。

    实船 bug：width 档**唯一没接跳变保护**，直接采纳了 `z=3.691`（> dist_max 2.5），
    此后一路追着第三/第四扇门跑。width 的 z 来自 `fx·W/Δu`，比 PnP 更粗，**最需要这道闸**。
    """
    task = _task(lambda: [])
    # 只给对向上边一对角点 ⇒ width 档；门放很远（z≈3.7 > dist_max=2.5）
    far = _det(3.7, kconf=(0.95, 0.95, 0.0, 0.0))[0]
    assert task._relock_guard(3.7) is True, "前提：3.7m > dist_max(2.5) 该被拦"
    frame = np.zeros((CAM.height, CAM.width, 3), np.uint8)
    task.process(frame, 1000)
    z_before = task._z_last
    task._on_width(far, 1100, [0, 1])          # 直接走 width 档
    assert task._z_last == z_before, \
        "width 档采纳了 %.3f 这个 >dist_max 的 z（跳变保护没贯彻到 width）" % (task._z_last or -1)
    assert task.last_info["action"] == "hold", \
        "width 档遇到超 dist_max 的 z 应判丢门、原地 hold，实际=%s" % task.last_info["action"]


def test_z_door_distance_is_one_value_for_three_keys():
    """★★ **三键同值**（用户 2026-10-07 定；历史依据见 git commit 52512bb / d3e660d）：

        `z.cross`（冲刺触发）、`z.slow_max`（近了就慢）、`width.z_max`（width 档"够近了"）
        —— 三个都表示"门框在画面里大到这个程度 = 已到门口"，用户当时要求**直接划等号**
        （09-30 起手工写成同一个数 1.2），现在用 **YAML 锚点 `&door_m`** 实现：
        **只改 cfg 里 `slow_max: &door_m <值>` 一处**，三个键同时变。

    本用例守住这条：任一键被单独改掉、或锚点被拆散，立刻红。
    """
    z = S.comm.gate.z
    w = S.comm.gate.width
    cross, slow, wz = float(z.cross), float(z.slow_max), float(w.z_max)
    assert cross == pytest.approx(slow) == pytest.approx(wz), (
        "z.cross(%s) / z.slow_max(%s) / width.z_max(%s) 必须同值 —— "
        "它们是同一个'到门口'距离的三种用法（cfg 里用 YAML 锚点 &door_m 绑定）"
        % (cross, slow, wz))


def test_z_door_distance_anchor_is_not_broken():
    """锚点实现本身的守卫：cfg 原文里必须是 `&door_m` 一处 + 两处 `*door_m`。"""
    import io, os, re
    here = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    path = os.path.join(here, "cfg", "comm.yaml")
    src = io.open(path, encoding="utf-8").read()
    assert re.search(r"slow_max:\s*&door_m\s+[0-9.]+", src), \
        "cfg 里必须保留 `slow_max: &door_m <值>`（三键同值的唯一来源）"
    assert src.count("*door_m") >= 2, \
        "`cross` 与 `width.z_max` 必须都写成 `*door_m` 别名（否则又会散成两个数）"


def test_z_dist_max_matches_the_ranging_limit():
    """★ 2026-10-07 用户定：**远端阈值保持 2.5**（"3 米开外测距不准，但远端不放大"）。

    注：数据上 z 在 1.958–3.123 之间是**空带** ⇒ dist_max 取该区间内任何值行为完全相同；
    2.5 落在空带正中，与 relock_away_m=1.8 同侧，是数据支持的取值。
    """
    assert float(S.comm.gate.z.dist_max_m) == pytest.approx(2.5)
    task = _task(lambda: [])
    assert task._relock_guard(2.8) is True, "2.8m > dist_max(2.5) 必须被无条件拦下"
    assert task._relock_guard(2.4) is False, "2.4m < dist_max(2.5) 不该被这一条拦"


def test_pick_gate_excludes_all_far_candidates_in_one_pass(monkeypatch):
    """★ 2026-10-07 用户定：选门必须**一帧内一次把"实际远的门"全排掉**，再搬出结果。

    为什么（用户实测）：原来的"选中→排除→重选"迭代里 `k_max` 是**逐帧重算**的 ⇒
    同一个门这帧是最大 k、下帧就不是 ⇒ 反复进出候选 ⇒ **反复选错、反复解锁、耗时**。
    本用例断言：三个候选里两扇 k 崩掉 ⇒ **一次调用**就把它们全排掉，直接选出 k 达标那扇。
    """
    import base.cfg.settings as _S
    task = _task(lambda: [])
    # 三扇门（同框位置、同占比），只有 z 不同 ⇒ k 不同
    a = _det(1.5, px=(CAM.width * 0.5, CAM.height / 2.0))[0]
    b = _det(1.5, px=(CAM.width * 0.5, CAM.height / 2.0))[0]
    c = _det(1.5, px=(CAM.width * 0.5, CAM.height / 2.0))[0]
    # a 崩小(k 低)、b 正常(k 最高)、c 也崩小
    zmap = {id(a): 0.5, id(b): 1.5, id(c): 0.6}
    monkeypatch.setattr(task, "_z_est", lambda d, ct: zmap[id(d)])
    assert float(_S.comm.gate.select.k_max_ratio) > 0
    picked = task._pick_gate([a, b, c])
    assert picked is b, "应一次排除两扇 k 崩掉的、直接选 k 最高那扇（实际选了 %s）" % picked


def test_pick_gate_never_returns_none_while_a_gate_exists(monkeypatch):
    """★ **有门就绝不返回 None** —— 否则 `_gate_lock` 会走丢帧计数 ⇒ 解锁 ⇒ 用户报的"重复触发/耗时"。"""
    task = _task(lambda: [])
    ds = [_det(1.5, px=(CAM.width * 0.5, CAM.height / 2.0))[0] for _ in range(3)]
    # 三扇门的 k 全都算不出来（模拟 coarse：没有 z）
    monkeypatch.setattr(task, "_z_est", lambda d, ct: None)
    got = task._pick_gate(ds)
    assert got is not None, "有检测框却返回 None ⇒ 会制造'丢门→解锁'空转"


def test_lock_area_ref_is_inter_frame(monkeypatch):
    """★★ 2026-10-07 用户定：**锁门的面积参照 = 上一帧被采纳的占比**（帧间），不是 max。

    判据：`本帧占比 ≥ relock_far_ratio(0.7) × 上一帧占比` ⇒ 还是这扇门、可信、不着急换门。

    为什么不能用 max（见过的最大）：参照只增不减 ⇒ `0.7×max` 越抬越高 ⇒ **船正常往回退**
    也会被判成"更远那扇门"⇒ 误拒 + 反复 hold/解锁。

    ⚠️ 这**只是锁门层**的参照；`_relock_ratio_ref`（丢门重锁那套）是另一回事，用户明确不要动。
    """
    task = _task(lambda: [])
    d1 = _det(1.5, px=(CAM.width * 0.5, CAM.height / 2.0))[0]
    task._locked_det = d1
    task._lock_k_ref = None
    task._lock_note_accepted(d1)
    assert task._lock_ratio_ref == pytest.approx(float(d1.w) / CAM.width)
    ref_big = float(task._lock_ratio_ref)
    # 往回退一帧（占比变小）⇒ 参照必须**跟着降到本帧值**，而不是停在见过的最大的那个
    d2 = _det(1.9, px=(CAM.width * 0.5, CAM.height / 2.0))[0]
    assert d2.w < d1.w
    task._lock_note_accepted(d2)
    assert task._lock_ratio_ref == pytest.approx(float(d2.w) / CAM.width), \
        "参照必须等于「上一帧被采纳的占比」(帧间)，不能停在 max=%.4f" % ref_big


def test_lock_guard_inter_frame_tolerates_shrink_rejects_sudden_drop():
    """帧间判据的两个方向：小幅缩小**放行**（不着急换门）／骤降**拒收**（真换了门）。"""
    task = _task(lambda: [])
    big = _det(1.5, px=(CAM.width * 0.5, CAM.height / 2.0))[0]
    task._locked_det = big
    task._lock_k_ref = None
    task._lock_ratio_ref = float(big.w) / CAM.width
    mild = _det(1.75, px=(CAM.width * 0.5, CAM.height / 2.0))[0]     # ~0.85×
    if float(mild.w) / CAM.width >= 0.7 * task._lock_ratio_ref:
        assert task._lock_guard_reject(mild) is False, "0.85× 的小幅缩小不该判成换门"
    far = _det(3.0, px=(CAM.width * 0.5, CAM.height / 2.0))[0]
    if float(far.w) / CAM.width < 0.7 * task._lock_ratio_ref:
        assert task._lock_guard_reject(far) is True, "骤降到 0.7× 以下必须判成更远那扇门"

def test_ratio_guard_does_not_clear_the_z_reference():
    """★ 2026-10-07 用户定：`_relock_guard_ratio()` **只判、不改状态**。

    它由 `_step` **每帧**调用（只要这帧看着正常），原来顺手把 `_relock_z_ref`
    （= `_relock_guard()` 判"z 比丢门前远 ≥0.5m"用的丢门参照）也清了 ⇒ 那个参照活不过一帧
    ⇒ 该判据形同虚设。**职责越界**，已改：只清/更新自己那套（面积）。
    """
    task = _task(lambda: _det(1.5))
    task._relock_z_ref = 1.9
    task._relock_ratio_ref = 0.50
    task._dbg_ratio = 0.55
    assert task._relock_guard_ratio() is False           # 同量级 ⇒ 判"还是这扇门"
    assert task._relock_z_ref == pytest.approx(1.9), "判据动了 z 参照（越界）"
    assert task._relock_ratio_ref == pytest.approx(0.50), "纯函数不该改任何参照"
    task._dbg_ratio = 0.10                                # 骤降 ⇒ 判"另一个门"
    assert task._relock_guard_ratio() is True
    assert task._relock_z_ref == pytest.approx(1.9)
    assert task._relock_ratio_ref == pytest.approx(0.50), "判据命中也不许改参照（生命周期由别处管）"


def test_relock_ratio_model_is_switchable(monkeypatch):
    """★★ 2026-10-07 用户定：面积参照（丢门重锁那套）**三个模型可切换** ——

    · `interframe` 帧间占比突变：参照 = 上一帧被采纳的占比（只挡单帧骤降）
    · `preloss`    丢门前占比突变：参照只在丢门那一刻固定，被采纳帧**不更新**
    · `ema`        平滑跟踪：参照 = 0.7×旧 + 0.3×本帧
    """
    task = _task(lambda: _det(1.5))
    gz = S.comm.gate.z

    # ① interframe：参照直接等于本帧（不平滑）
    monkeypatch.setitem(gz, "relock_ratio_model", "interframe")
    task._relock_ratio_ref = None
    task._relock_ratio_note_accepted(0.40)
    assert float(task._relock_ratio_ref) == pytest.approx(0.40)
    task._relock_ratio_note_accepted(0.10)
    assert float(task._relock_ratio_ref) == pytest.approx(0.10), "帧间模型下参照应等于本帧"

    # ② ema：0.7×旧 + 0.3×新
    monkeypatch.setitem(gz, "relock_ratio_model", "ema")
    task._relock_ratio_ref = 0.40
    task._relock_ratio_note_accepted(0.10)
    assert float(task._relock_ratio_ref) == pytest.approx(0.7 * 0.40 + 0.3 * 0.10, abs=1e-9)

    # ③ preloss：被采纳帧**不许**改参照（只在丢门那一刻固定）
    monkeypatch.setitem(gz, "relock_ratio_model", "preloss")
    task._relock_ratio_ref = 0.40
    task._relock_ratio_note_accepted(0.10)
    assert float(task._relock_ratio_ref) == pytest.approx(0.40), \
        "preloss 模型下被采纳帧不该改参照（否则就退化成帧间了）"

    # ④ 非法模型值 ⇒ 报名字（代码无兜底）
    import pytest as _pytest
    from common.cfg.cfgnode import MissingCfg
    monkeypatch.setitem(gz, "relock_ratio_model", "nonsense")
    with _pytest.raises(MissingCfg):
        task._relock_ratio_model()


def test_width_retreats_to_reacquire_instead_of_creeping():
    """★★ 2026-10-07 用户定：**width 是降级档（只有 2 角、信息不足）—— 不往前蹭，一律回退重取**，
    把门重新拉远、拿回 4 角点，交给 full/p3p 的正常链路。

    三分支（都在 ALIGN 内）：
      · **居中成功** ⇒ 退（width 里硬冲不安全，退回去让整门重新进视野）
      · **贴脸(`z ≤ width.z_max`)且没对准** ⇒ 退（与 coarse 档同一套）
      · 还远且没对准 ⇒ 原地 HOLD（先对中，别乱动）
    且**已在重取中**时继续走闭环退（`_tick_reacquire` 拿到 ratio ⇒ 退到框明显变小就停），
    不会每帧重新 `_enter_reacquire` 把 `max_times` 烧光。
    """
    import numpy as np
    from gate.gate_task import PH_ALIGN
    task = _task(lambda: [])
    frame = np.zeros((CAM.height, CAM.width, 3), np.uint8)
    task.phase = PH_ALIGN
    task._reacquire_last_ms = None
    task._reacquire_cnt = 0

    # ① 居中成功（px 放画面中心 ⇒ dxn/dyn≈0）⇒ 必须回退，且 surge < 0
    det_ok = _det(1.0, px=(CAM.width * 0.5 + CAM.cx - CAM.width * 0.5,
                           CAM.height * 0.5 + CAM.cy - CAM.height * 0.5),
                  kconf=(0.95, 0.95, 0.0, 0.0))[0]
    task._kpt_s = None
    task._on_width(det_ok, 2000, [0, 1])
    assert task.substate == "REACQUIRE" or task.last_info.get("action") == "reacquire", \
        "width 居中成功后必须进回退重取（实际 action=%s substate=%s）" % (
            task.last_info.get("action"), task.substate)
    assert float(task.last_info.get("surge") or 0.0) < 0.0, \
        "回退重取必须发负 surge（实际 %s）" % task.last_info.get("surge")


def test_width_retreat_respects_max_times_and_then_holds():
    """`max_times` 用尽后**改原地保持**，不许无限后退（复用 reacquire 既有放弃机制）。"""
    import numpy as np
    from gate.gate_task import PH_ALIGN
    task = _task(lambda: [])
    task.phase = PH_ALIGN
    det = _det(1.0, kconf=(0.95, 0.95, 0.0, 0.0))[0]
    task._kpt_s = None
    mx = int(S.comm.gate.reacquire.max_times)
    t = 1000
    for k in range(mx + 3):
        task.substate = ""                       # 强制每帧重新触发（模拟反复判"居中成功"）
        task._on_width(det, t, [0, 1])
        t += 50
    assert int(task._reacquire_cnt) > mx
    assert task.substate in ("HOLD", "") or task.last_info.get("action") == "hold", \
        "超过 max_times 后必须放弃后退、改原地保持（实际 %s/%s）" % (
            task.substate, task.last_info.get("action"))


def _coarse_det(kconf, z=2.0, off=None):
    """coarse 档检测：kconf 顺序 = (TL, TR, BR, BL)（`gate_frontend._ID_*`）。

    ⚠️ `off` 用来把门**推出居中带**（coarse 在 aligned 时有一条既有分支"往前蹭争取露角点"，
    那条先于 HOLD 被走到 ⇒ 测 HOLD 超时必须喂**未对准**的门）。
    """
    px = None
    if off is not None:
        px = (CAM.width * 0.5 + float(off), CAM.height * 0.5)
    return _det(z, px=px, kconf=kconf)[0]


def test_coarse_hold_timeout_triggers_retreat():
    """★ 2026-10-07 用户定：coarse 的 HOLD **超时 5s** 就触发后退（原来按帧数 20 帧≈2s，偏紧）。"""
    from gate.gate_task import PH_ALIGN
    task = _task(lambda: [])
    task.phase = PH_ALIGN
    det = _coarse_det((0.0, 0.0, 0.95, 0.95), off=CAM.width * 0.30)   # 只有下边两角 + **未对准**
    task._ids_now = [2, 3]
    task._hold_start_ms = 1000
    to = float(S.comm.gate.coarse.hold_timeout_ms)
    assert to == pytest.approx(5000.0), "现值应为 5s（改了就同步这条）"
    # 还没到 5s ⇒ 仍 HOLD
    task._on_coarse(det, 1000 + int(to) - 100)
    assert task.substate == "HOLD", "没到超时不该退（实际 %s）" % task.substate
    # 过了 5s ⇒ 进后退重取
    task._hold_start_ms = 1000
    task._on_coarse(det, 1000 + int(to) + 100)
    assert task.substate == "REACQUIRE" or task.last_info.get("action") == "reacquire", \
        "超时后必须触发后退重取（实际 %s/%s）" % (task.substate, task.last_info.get("action"))
    assert float(task.last_info.get("surge") or 0.0) < 0.0, "必须是负 surge（后退）"


def test_coarse_retreat_heaves_toward_the_missing_corners():
    """按可见角点决定上浮/下潜：只见下边角(BR/BL) ⇒ 门在上方 ⇒ **上浮**；只见上边角 ⇒ 下潜。

    用户原话："只看到一个角点（BR），后退同时上浮一点，直到看到两个角点进入 width 为止"。
    """
    from gate.gate_task import PH_ALIGN
    amp = float(S.comm.gate.coarse.back_heave)
    assert amp >= 0.138, "back_heave 必须高于执行器死区 0.138 才有推力（现 %.3f）" % amp

    for ids, want_sign, why in (([2], +1, "只见 BR ⇒ 上浮"), ([3], +1, "只见 BL ⇒ 上浮"),
                                ([0], -1, "只见 TL ⇒ 下潜"), ([1], -1, "只见 TR ⇒ 下潜")):
        task = _task(lambda: [])
        task.phase = PH_ALIGN
        task.substate = "REACQUIRE"
        task._reacquire_start = 1000
        task._reacquire_ratio0 = 0.30
        task._ids_now = ids
        task._on_coarse(_coarse_det((0.95, 0.0, 0.0, 0.0) if ids == [0] else (0.0,) * 4), 1500)
        hv = float(task.last_info.get("heave") or 0.0)
        assert hv * want_sign > 0, "%s：heave 应为 %s 号（实际 %.3f）" % (why, "+" if want_sign > 0 else "−", hv)


def test_coarse_retreat_stops_once_two_corners_are_back():
    """**够 2 个角点（升到 width）就收手** —— 不追求 full（width 已能测距/居中）。"""
    from gate.gate_task import PH_ALIGN
    task = _task(lambda: [])
    task.phase = PH_ALIGN
    task.substate = "REACQUIRE"
    task._reacquire_start = 1000
    task._ids_now = [2, 3]                      # 两角 ⇒ 已达 width
    task._on_coarse(_coarse_det((0.0, 0.0, 0.95, 0.95)), 1500)
    assert task.substate == "HOLD", "够 2 角就该停退（实际 %s）" % task.substate
    assert float(task.last_info.get("surge") or 0.0) == 0.0, "停退时 surge 必须是 0"
