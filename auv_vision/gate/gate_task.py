# -*- coding: utf-8 -*-
"""gate/gate_task.py — 任务三 过门（GateTask 相位机）

门 = 对称平面矩形门框(卷尺实测外缘 **0.77×0.56 m**，管外径 5cm)，悬空，无朝向要求；任务 = 机身穿过开口。

相位：SEARCH → ALIGN → APPROACH → THROUGH(计数) → (下一门)/DONE
ALIGN 子状态（按前端 mode 与门框占屏比仲裁）：
  GOLDEN     full/p3p 位姿对准（sway/heave PID，无 surge）
  HDG        居中达标后的**离散正航向**（测→转→停稳→再测，见 gate/heading_align.py）
  CREEP      远距小框：慢速靠近 + 框心对中
  HOLD       中/近角不足：surge=0 保持对中；≥hold.max_frames → REACQUIRE
  REACQUIRE  短后退重取整门；超限 → SEARCH
直冲出口三条（任一成立即进 THROUGH：忽略角点丢失直行、不再做横向微调）：
  ① 位姿可信且 Z ≤ z.cross 连续 cross_confirm_frames 帧（**实船已验证，勿动**）；
  ② 近距（Z ≤ z.near_lost_m 且 z 新鲜，或框占比 ≥ z.near_lost_ratio）整门丢失；
  ③ 在门口超时兜底 `loiter.*`（门口 + 安全带内停留超时 → 自己拍板直冲）。
位姿跳变保护：帧间 z 变化 > pnp.max_z_jump_m 的帧按错解弃掉（防平面 PnP 错解直冲）。

依赖：gate.geometry（位姿）、gate.gate_frontend（mode）、common.PID、common/cfgnode。
参数：cfg/vision.yaml `gate.*`、cfg/comm.yaml `gate.*`；读取一律走 cfgnode 兜底，
**配置缺键时用"等于当前 cfg 的默认值"**，不会 AttributeError。
实测依据、参数演进与调参记录见 `gate/过门-状态机与参数.md`（本文件注释只留
"这个值是什么 + 为什么不能乱改"）。
"""
from __future__ import annotations

import numpy as np

import base.settings as S
from common.PID import PID
from common.cfgnode import (flag, merge, motion_node, motion_num, num, pid_kw,
                            sub)
from gate.gate_detector import board_camera
from gate.kpt_memory import ENV_ENABLE, build_kpt_memory, enable_source
from gate.gate_frontend import (parse_kpt_mode, width_range_depth, bbox_center,
                                MODE_FULL, MODE_P3P, MODE_WIDTH, MODE_COARSE)
from gate.geometry import (object_points, gate_pose, gate_normal_angles_deg,
                            GATE_FRAME_W, GATE_FRAME_H)
from gate.heading_align import HeadingAligner, hdg_cfg

PH_SEARCH = "SEARCH"
PH_ALIGN = "ALIGN"
PH_APPROACH = "APPROACH"
PH_THROUGH = "THROUGH"

SUB_HDG = "HDG"
SUB_GOLDEN = "GOLDEN"
SUB_CREEP = "CREEP"
SUB_HOLD = "HOLD"
SUB_REACQUIRE = "REACQUIRE"

# ---------------------------------------------------------------- 缺键兜底
# 本表是"cfg 缺键/缺文件"时的兜底，取值**等于当前 cfg**（用例 test_gate_defaults_match_cfg
# 钉住"兜底 == cfg"）。
# ⚠️ **共用参数不在这里**：sway/heave 两套 PID 与进近速度档 fast/slow 是 ball/gate 共用的，
#    唯一来源 = `comm.motion.*`（代码兜底 `common/cfgnode.MOTION_DEFAULTS`）。gate 只在
#    `comm.gate.pid_sway/pid_heave`、`comm.gate.surge.fast/slow` 里写"确实要单独一套"的覆盖。
#    gate 里**用 yaw 的地方只有一处**：ALIGN.HDG 离散正航向（增益 comm.motion.turn_pid）；
#    居中一律 sway（见 `_lateral_out`）。
_D_ALIGN = dict(confirm_frames=4, px_x=0.20, px_y=0.25)
# 「在门口超时兜底」= 出口③：已在门口(占比≥z.near_lost_ratio) + 在安全带内，
# 连续超过 timeout_ms 仍未 commit → 自己判过门并直冲（现场：门占比 0.97 时整框仍检测得到，
# 没有这条就会在门口一直 creep）。实测记录见 md §3。
_D_LOITER = dict(enable=True, timeout_ms=3000, dx_max=0.14, dy_max=0.20)
# z.cross：实船验证过的穿门判据，**不要动**。near_lost_* 是"到门口了"的两条判据（见出口②）。
# cross 取 **0.77** = 陆上实验（2026-09-22 21:00 前）确认的最终值，与标尺 0.77/0.56 配对。
_D_Z = dict(cross=0.77, cross_confirm_frames=2, slow_max=1.3, near_lost_m=1.0,
            near_lost_ratio=0.60, z_stale_ms=1500)
_D_SURGE = dict(creep=0.20, lost_backward=0.20, reacquire=0.25, through=0.6)
# 进近两档 fast/slow 取自 comm.motion.surge_fast/surge_slow（与撞球共用同一份，见 _speed）。
# ⚠️ creep/lost_backward 曾取 0.12 —— 那是**执行器死区以内(0 推力)**，等于这两个动作
#    从未发生（现 0.20）；用例 test_no_speed_preset_inside_actuator_deadzone 守着。
_D_COARSE = dict(far_ratio=0.18, near_ratio=0.55, align_x=0.33, align_y=0.29)
_D_WIDTH = dict(z_max=1.5)
_D_TASK = dict(timeout_ms=300000, pass_target=1, pose_hold_frames=10)
# ⚠️ keypoint.conf_thr 别改回 0.5：cfg 现场实测 0.7（0.5 会让鬼点/倒影被当成真角点）。
_D_KPT = dict(conf_thr=0.7)
_D_HOLD = dict(max_frames=20)
_D_REACQ = dict(max_ms=800, max_times=2, stop_ratio=0.75, reset_after_ms=3000)
# THROUGH 结束条件按**时间**（帧数语义下冲刺时长随 fps 漂，而"要冲多远"是距离问题）；
# confirm_ms<=0 → 退回旧的帧数语义。
_D_THROUGH = dict(confirm_frames=8, confirm_ms=2500)
# SEARCH = **左右平移扫视**（过门过程中不允许旋转搜索，2026-09-20 用户定）。
# ⚠️ 波形**必须对称**（正反等时长）：遥测没有横向位置反馈，单向扫会一路漂到池壁；
#    等时长 ⇒ 净位移≈0。残余漂移未验证，靠 timeout_ms 兜总时长。细节见 md §2.1。
_D_SEARCH = dict(sweep_s=2.0, pause_s=1.0, sway=0.3)
# ⚠️ reproj_px 曾写 8.0：8px 门槛会拒掉 ~88% 候选位姿，现场实测放宽到 20。
_D_PNP = dict(reproj_px=20.0, z_min=0.2, z_max=15.0, refine=True,
              max_z_jump_m=0.8)
_D_GEOM = dict(frame_w=GATE_FRAME_W, frame_h=GATE_FRAME_H,
               body_center_offset=0.0)


def _dof_clip(v):
    return float(max(-1.0, min(1.0, v)))



class GateTask(object):
    name = "gate"

    def __init__(self, uart, hub, frame_w, frame_h):
        self.uart = uart
        self.hub = hub
        self.w = int(frame_w)
        self.h = int(frame_h)
        G = S.get("comm.gate", None) or {}
        V = S.get("vision.gate", None) or {}
        self._G, self._V = G, V

        geo = merge(sub(V, "geometry"), _D_GEOM)
        self.frame_w_m = float(geo["frame_w"])
        self.frame_h_m = float(geo["frame_h"])
        # 相机—机身安装偏置(m)：**只读入、不参与运算**（未标定；接进 sway/heave 目标
        # 等于改对准基准）。要启用先实测标定。
        self.body_center_offset = float(geo["body_center_offset"])
        self.obj3 = object_points(self.frame_w_m, self.frame_h_m)
        self.camera = board_camera()
        # 角点短时记忆（**可选开关，2026-09-18 起默认关**）：小漂移/短消失用最近窗口稳住，
        # 抗水面倒影，避免"点不全/姿态不稳"误触发 REACQUIRE 后退。
        # 临时开启：AUV_GATE_KPT_MEM=1（优先级高于 vision.gate.kpt_mem.enable）。
        km = V.get("kpt_mem", None) or {}
        order = S.get("vision.model.task_models.gate.kpt_order", None) or \
            sub(V, "keypoint").get("kpt_order") or [0, 1, 2, 3]
        self._kpt_mem = build_kpt_memory(km, n_kpt=len(order) or 4)
        if self._kpt_mem is None:
            print("[GATE] kpt_mem 关闭 → 角点单帧直用（%s；临时开启用 %s=1）"
                  % (enable_source(km), ENV_ENABLE))
        else:
            # 打印**实际生效**的值（自己写兜底会打出与 kpt_memory.DEFAULTS 不一致的数字）
            m = self._kpt_mem
            conf_thr = num(sub(V, "keypoint"), "conf_thr", _D_KPT["conf_thr"])
            print("[GATE] kpt_mem 开启（逐点软融合，%s）：alpha=%s beta=%s "
                  "recall_conf=%s（必须 < keypoint.conf_thr=%.2f）"
                  % (enable_source(km), m.alpha, m.beta, m.recall_conf, conf_thr))
        self._kpt_s = None            # 本帧平滑后的角点（供 _on_width 等复用）
        self._kconf_s = None
        # 下水前自检：标定分辨率必须与实际帧一致，否则 PnP 深度整体缩放错
        if (self.camera.width, self.camera.height) != (self.w, self.h):
            print("[GATE] ⚠️ 标定尺寸 %dx%d ≠ 实际帧 %dx%d：PnP 距离会系统性偏差，"
                  "请按当前分辨率重新标定 (cfg/front_camera.yaml)"
                  % (self.camera.width, self.camera.height, self.w, self.h))

        # ---- 修正增益：**共用 comm.motion 那两套 PID**（同船/同推进器，实船实测过）----
        # 两层：comm.gate.pid_sway/pid_heave（一般都不写，只在确实要单独一套时才写）
        #   兜底 → comm.motion.pid_sway / pid_heave（**与撞球共用同一份**，一处调两处生效）。
        G_sway, G_heave = sub(G, "pid_sway"), sub(G, "pid_heave")
        self._gain_src = "gate(显式覆盖)" if (G_sway or G_heave) else "motion(共用)"
        self._pid_sway_kw = pid_kw(G_sway, motion_node("pid_sway"))
        self._pid_heave_kw = pid_kw(G_heave, motion_node("pid_heave"))
        # sway / heave 各一套增益；**全档位共用同一对像素 PID**（一套阈值 + 一套控制器，
        # 不会随 mode 在 full↔coarse 间跳而交替工作）。居中只有 sway（见 `_lateral_out`）。
        self._pid_sway_px = PID(**self._pid_sway_kw)
        self._pid_heave_px = PID(**self._pid_heave_kw)
        # 启动日志：现场一眼看到"跑的是哪一套"（共用小球 or 显式覆盖）
        _ks = ("kp", "ki", "kd", "out_max", "deadzone")
        print("[GATE] 修正增益来源=%s | sway %s | heave %s | 转向=仅正航向(%s)"
              % (self._gain_src,
                 {k: self._pid_sway_kw[k] for k in _ks},
                 {k: self._pid_heave_kw[k] for k in _ks},
                 "enable" if hdg_cfg().get("enable", True) else "off"))
        # 离散正航向（ALIGN.HDG；见 gate/heading_align.py）。每门只做一次，收敛或放弃后锁定。
        self._hdg = HeadingAligner(log=(print if S.DEBUG else (lambda *a: None)))
        self._hdg_cfg = hdg_cfg()
        self._hdg_done = True             # True=本门不需要再正航向（未启用或已出结论）
        self._hdg_done = not flag(self._hdg_cfg, "enable", True)
        # 航向误差测量（门法向 vs 光轴；**只有 full 帧可信**，p3p 实测 std 64~69°）
        self._hdg_f = None                # EMA 状态（只吃 full 帧，避免 p3p 垃圾污染）
        self._hdg_deg = None              # 最近一次 full 帧测到的航向误差（度）
        self._hdg_ms = None               # 上面那个值的时刻（新鲜度；离散正航向要用）
        self._wait_hdg_ms = None          # 「原地等一帧新鲜 full」的起始时刻（wait_fresh_ms>0 用）
        self._hdg_skip_logged = False     # 本门是否已报过"跳过正航向"的原因（每门一次）

        self.last_info = {"phase": PH_SEARCH, "substate": "", "mode": "",
                          "action": "stop", "z": 0.0, "dx": 0.0, "dy": 0.0,
                          "sway": 0.0, "heave": 0.0, "surge": 0.0, "yaw": 0.0,
                          "pass": 0, "kpt": 0, "kpt_raw": 0, "ratio": 0.0,
                          "hdg": None,          # 航向误差(度)：+ = 门法向偏画面右 = 机身左偏
                          "hdg_skip": None,     # 居中达标却跳过正航向的原因（下一步：进近）
                          "hdg_state": "",      # HDG 内部状态：idle/settle/measure/turn/done/giveup
                          "hdg_i": 0,           # 正航向已迭代次数（ALIGN.HDG）
                          "status": S.STATUS_RUNNING, "reason": ""}
        self.reset_state()

    # ------------------------------------------------------------ 状态复位
    def reset_state(self):
        self.frames = 0
        self._start_ms = None
        self.phase = PH_SEARCH
        self.substate = ""
        self.mode = ""
        self._last_pose = None            # (rvec, tvec) 上一帧位姿（消歧/防抖）
        self._z_last = None
        self._z_ms = None                 # `_z_last` 的更新时刻（判它新不新鲜，见 _near_lost）
        self._z_guard = False             # 上一帧是否有可信位姿（z 跳变保护基准）
        self._lost_cnt = 0
        self._center_cnt = 0
        self._hold_cnt = 0
        self._cross_cnt = 0              # 连续 z≤cross 帧数（穿门确认，防单帧错解）
        self._through_frames = 0
        self._through_start_ms = None    # THROUGH 起始时间戳（时长判据）
        self._hdg.reset()
        self._hdg_done = not flag(self._hdg_cfg, "enable", True)
        self._wait_hdg_ms = None
        self._hdg_skip_logged = False
        self._loiter_start_ms = None     # 「已在门口」的起始时刻（超时兜底）
        self._reacquire_start = None
        self._reacquire_ratio0 = None    # 进入 REACQUIRE 时的框占比（闭环后退基准）
        self._reacquire_cnt = 0          # 连续 REACQUIRE 次数（超限不再后退）
        self._reacquire_last_ms = None   # 上次真正后退的时刻（时间窗重置计数）
        self._give_up_until_ms = None    # 放弃后退后的 HOLD 窗口截止          # 连续 REACQUIRE 次数（超限不再后退）
        self._dbg_ratio = 0.0            # 本帧门框宽/屏宽（诊断+叠加）
        self._dbg_kpt = 0                # 本帧有效角点数（**记忆后**，用于判 mode）
        self._dbg_kpt_raw = 0            # 本帧原始有效角点数（诊断：看倒影污染程度）
        self._search_entry_ms = None
        self._pass_cnt = 0
        self._finished = False
        self._reason = ""
        self.last_dets = []               # 本帧门检测（供 main._draw 画角点）

    @property
    def ready(self):
        return self.hub.has_extra(self.name)

    # ------------------------------------------------------------ 工具
    def _reset_lateral_pids(self):
        for p in (self._pid_sway_px, self._pid_heave_px):
            p.reset()

    def _lateral_out(self, dxn, now_ms):
        """居中阶段的水平修正：**只用 sway 平移**（全档位共用一个像素 PID）。

        ⚠️ **别把 yaw 加回来做居中**（2026-09-20 用户定，原 `comm.gate.align_yaw` 已删）：
        位置误差该用平移修 —— sway 命令 = kp(8.0)·dxn，|dxn|>0.017 就能过执行器死区(0.138)；
        yaw 命令 = kp(0.3)·dxn，要 |dxn|>0.46（近半屏）才有推力 ⇒ 小误差时"一点力都没有"。
        而且 yaw 是**方位**控制，把门拉到光轴上 ≠ 机身与门平行。姿态由 ALIGN.HDG 负责。
        """
        return _dof_clip(self._pid_sway_px.update(dxn, now_ms))

    def _hdg_skip_reason(self, mode, now_ms):
        """居中达标却没进正航向的**原因**（水里复盘看 `hdg_skip` 这个字段）。

        ⚠️ `fresh_ms` 其实是空转的：**每个 full 帧都会刷新 `_hdg_ms`** ⇒
        "psi 新鲜"真正等价于"**当前这一帧是 full**"。所以最常见的原因就是 mode 不是 full
        （远距/斜门只在部分帧拿到 4 角）。另外 `_on_width` 的 ALIGN 分支从不启动 HDG。
        """
        if not flag(self._hdg_cfg, "enable", True) or not self._hdg.enabled:
            return "off(未启用)"
        if self._hdg_done:
            return "done(本门已做过正航向)"
        if mode != MODE_FULL:
            return "mode=%s(只有 full 帧的 psi 可用)" % mode
        if self._hdg_deg is None or self._hdg_ms is None:
            return "no_psi(还没测到过 full 的 psi)"
        age_s = (now_ms - self._hdg_ms) / 1000.0
        return "stale(psi 已 %.2fs > fresh_ms=%.2fs)" % (
            age_s, float(self._hdg_cfg.get("fresh_ms", 800.0)) / 1000.0)

    def _note_hdg_skip(self, why):
        """居中达标却跳过正航向 → 写进日志字段 + 终端报一次（每门一次）。"""
        self.last_info["hdg_skip"] = why
        if self._hdg_skip_logged:
            return
        self._hdg_skip_logged = True
        print("[GATE] ⚠️ 居中达标但跳过正航向：%s ⇒ 带残余航向直接进近"
              "（想让它等一帧新鲜 full：把 comm.gate.hdg.wait_fresh_ms 设成 1500~2000）" % why)

    def _hdg_abort(self, why):
        """中止正在进行的正航向转向（只有**整门丢失 / 进冲刺**才调）。

        档位退化（width/coarse）不触发中止：转向期间视觉链路整个被绕开（见 `_step`），
        那才是"不被画面干扰"的正确做法。原地旋转看不到门就不该继续盲转 ——
        `TurnCore.abort()` 会立刻把本帧 yaw 归零，`stop_hard` 由调用方在收尾时负责。
        """
        if self.substate == SUB_HDG and not self._hdg.finished():
            self._hdg.abort(why)
            self._hdg_done = True
            self.substate = SUB_GOLDEN
            self._center_cnt = 0
            return True
        return False

    def _hdg_fresh(self, mode, now_ms):
        """本帧的航向测量是否可用作样本：必须 full + 有测量 + **未过期**。

        只判"有过测量"不够 —— 陈旧的 psi（几秒前、船已经动过）当样本会算出错误的目标角。
        """
        if mode != MODE_FULL or self._hdg_deg is None or self._hdg_ms is None:
            return False
        fresh_ms = float(self._hdg_cfg.get("fresh_ms", 800.0))
        return (now_ms - self._hdg_ms) <= fresh_ms

    def _hdg_ready(self, mode, now_ms):
        """是否可以（或必须）进正航向：启用 + 本门还没做 + 本帧测量新鲜。"""
        if self._hdg_done or not self._hdg.enabled:
            return False
        return self._hdg_fresh(mode, now_ms)

    def _uart_yaw(self):
        """下位机回传的绝对航向（度；没有就 None）。"""
        tel = getattr(self.uart, "telemetry", None)
        return None if tel is None else getattr(tel, "yaw_deg", None)

    def _set_info(self, action, mode="", substate="",
                  z=None, dx=0.0, dy=0.0, sway=0.0, heave=0.0,
                  surge=0.0, yaw=0.0, kpt=0, ratio=None, kpt_raw=None,
                  hdg=None, hdg_skip=None, hdg_state=None):
        # hdg_skip：居中达标却跳过正航向的原因（只在那个瞬间有意义；传 None 表示本帧不涉及）
        if hdg_skip is not None:
            self.last_info["hdg_skip"] = hdg_skip
        if hdg_state is not None:
            self.last_info["hdg_state"] = str(hdg_state)
        self.last_info.update({
            "phase": self.phase, "substate": substate or self.substate,
            "mode": mode or self.mode, "action": action,
            "z": round(z if z is not None else (self._z_last or 0.0), 3),
            "dx": round(float(dx), 3), "dy": round(float(dy), 3),
            "sway": round(float(sway), 3), "heave": round(float(heave), 3),
            "surge": round(float(surge), 3), "yaw": round(float(yaw), 3),
            "pass": self._pass_cnt, "kpt": int(kpt),
            "kpt_raw": int(self._dbg_kpt_raw if kpt_raw is None else kpt_raw),
            "ratio": round(float(self._dbg_ratio if ratio is None else ratio), 3),
            "hdg": None if hdg is None else round(float(hdg), 1)})
        self.uart.send_dof(_dof_clip(surge), _dof_clip(sway),
                           _dof_clip(heave), _dof_clip(yaw))

    def _start_through(self):
        """进入 THROUGH：只前进、不微调（横向修正全部留在冲刺前完成）。

        速度 = `comm.gate.surge.through`（三个出口都走这一条，见 `_through_speed`）。
        """
        self._hdg_abort("进入冲刺")
        self.phase = PH_THROUGH
        self.substate = ""
        self._through_frames = 0
        self._lost_cnt = 0
        self._z_guard = False
        self._through_start_ms = None     # 第一帧 tick 时打时间戳（见 _tick_through）
        self._loiter_start_ms = None      # 已进入冲刺 → 门口的计时作废
        self._reset_lateral_pids()

    def _through_speed(self):
        """本次冲刺速度 = `comm.gate.surge.through`。"""
        return num(sub(self._G, "surge"), "through", _D_SURGE["through"])

    def _speed(self, tier):
        """进近速度档：`gate.surge.<tier>` 显式覆盖 → `motion.surge_<tier>`（**与撞球共用**）→ 代码兜底。"""
        return num(sub(self._G, "surge"), tier, motion_num("surge_" + tier))

    def _start_search(self):
        self.phase = PH_SEARCH
        self.substate = ""
        # 新的一门：重新允许正航向（每门只做一次）
        self._hdg.reset()
        self._hdg_done = not flag(self._hdg_cfg, "enable", True)
        self._wait_hdg_ms = None
        self._hdg_skip_logged = False
        self._search_entry_ms = None
        self._last_pose = None
        self._z_last = None
        self._z_guard = False
        self._cross_cnt = 0
        self._lost_cnt = 0
        self._hold_cnt = 0
        if self._kpt_mem is not None:
            self._kpt_mem.reset()

    def _search_sweep(self, now_ms):
        """SEARCH：左右平移扫视（波形见 md §2.1；2026-09-20 用户定：过门过程中不允许旋转搜索）。

        波形（周期 T = 2·(sweep+pause)）：
            段0 [0, sweep)               → +sway（右移）
            段1 [sweep, sweep+pause)     → 0（停：让检测有静止帧）
            段2 [.., 2·sweep+pause)      → -sway（左移）
            段3 余下                      → 0（停）
        ⚠️ **必须正反等时长**：没有横向位置反馈，单向扫会一路漂到池壁；等时长 ⇒ 净位移≈0。
        方向约定与 sway 一致：**+ = 右移**。`sweep_s<=0` → 只停不扫（等于原地待机）。
        """
        s = merge(sub(self._G, "search"), _D_SEARCH)
        sweep = num(s, "sweep_s", _D_SEARCH["sweep_s"]) * 1000.0
        pause = max(0.0, num(s, "pause_s", _D_SEARCH["pause_s"]) * 1000.0)
        if sweep <= 0:
            return 0.0
        v = num(s, "sway", _D_SEARCH["sway"])
        if self._search_entry_ms is None:
            self._search_entry_ms = now_ms
        ph = (now_ms - self._search_entry_ms) % (2.0 * (sweep + pause))
        if ph < sweep:                       # 段0：向右
            return _dof_clip(v)
        if ph < sweep + pause:               # 段1：停
            return 0.0
        if ph < 2.0 * sweep + pause:         # 段2：向左
            return _dof_clip(-v)
        return 0.0                           # 段3：停

    # ------------------------------------------------------------ 主入口
    def process(self, frame, now_ms):
        self.frames += 1
        if self._start_ms is None:
            self._start_ms = now_ms
        if not self.ready:
            self.uart.neutral()
            self.last_info.update({"status": S.STATUS_DONE,
                                   "reason": "no_backend"})
            return S.STATUS_DONE
        if self._finished:
            self.uart.neutral()
            self.last_info.update({"status": S.STATUS_DONE,
                                   "reason": self._reason})
            return S.STATUS_DONE

        dets = self.hub.detect_list(self.name, frame)
        self.last_dets = dets
        det = self._pick_gate(dets)
        self._step(det, now_ms)

        if (self._start_ms is not None and
                now_ms - self._start_ms >= num(self._G, "timeout_ms", _D_TASK["timeout_ms"])):
            self._finish("timeout")
        self.last_info["status"] = S.STATUS_DONE if self._finished \
            else S.STATUS_RUNNING
        return self.last_info["status"]

    def _finish(self, reason):
        self._finished = True
        self._reason = reason
        self.uart.neutral()
        self.last_info.update({"status": S.STATUS_DONE, "reason": reason,
                               "pass": self._pass_cnt})
        if S.DEBUG:
            print("[GATE] DONE(%s) pass=%d frames=%d" % (reason, self._pass_cnt,
                                                          self.frames))

    def _pick_gate(self, dets):
        """选目标门：**优先角点更全、更好的框**，其次才比 score。

        水中倒影常让角点落到倒影上（有时形成第二个门框）：只按 score 选可能把
        倒影当真门 → 位姿错、甚至 z≤cross 直接冲门。角点数量/置信度和更能反映
        "哪个是真门框"。（再配合上帧位姿消歧 + z 跳变保护一起用。）
        """
        conf_thr = num(sub(self._V, "keypoint"), "conf_thr", _D_KPT["conf_thr"])
        best, best_key = None, None
        for d in dets:
            if d.kind != "gate":
                continue
            if d.kpt_conf is None:
                key = (-1, 0.0, float(d.score))     # 无角点信息 → 排最后
            else:
                kc = np.asarray(d.kpt_conf, np.float64)
                key = (int((kc >= conf_thr).sum()), float(kc.sum()),
                       float(d.score))
            if best_key is None or key > best_key:
                best, best_key = d, key
        return best

    # ------------------------------------------------------------ 单帧逻辑
    def _step(self, det, now_ms):
        if self.phase == PH_THROUGH:
            self._tick_through(now_ms)      # 穿门=直行，不做横向微调
            return
        if det is None:
            self._tick_lost(now_ms)
            return
        # ---- 正航向的「转」：绕开整条视觉链路（先居中再转，转的时候别被画面干扰）
        #   只要在 ALIGN.HDG 且正在转，本帧只按「遥测 yaw + 冻结的目标角」驱动：
        #   档位退化(width/coarse)、位姿被拒、单帧角点闪烁都不打断它。
        #   ⚠️ 必须放在**档位分派之前**，否则一退化就走进 _on_width/_on_coarse 打断转向。
        #   唯一中止条件 = 整门丢失（走上面的 `det is None` → `_tick_lost` → abort）。
        if self.phase == PH_ALIGN and self.substate == SUB_HDG \
                and not self._hdg.finished() and self._hdg.turning:
            _st, yaw_cmd = self._hdg.step(now_ms, psi_deg=None, psi_fresh=False,
                                          yaw_telemetry=self._uart_yaw(), gate_lost=False)
            if self._hdg.finished():
                self._hdg_done = True
                self.substate = SUB_GOLDEN
                self._center_cnt = 0
            self.last_info["hdg_i"] = self._hdg.iters
            self._set_info("hdg" if not self._hdg.finished() else "center",
                           mode=self.mode or "", z=self._z_last,
                           dx=self.last_info.get("dx", 0.0),
                           dy=self.last_info.get("dy", 0.0),
                           sway=0.0, heave=0.0, surge=0.0, yaw=yaw_cmd,
                           kpt=self._dbg_kpt, hdg=self._hdg_deg)
            return
        self._lost_cnt = 0
        V = self._V
        conf_thr = num(sub(V, "keypoint"), "conf_thr", _D_KPT["conf_thr"])
        # 诊断量：本帧框占比 + 有效角点数（进叠加与 REACQUIRE 日志）
        self._dbg_ratio = float(det.w) / float(self.w)
        if det.kpt_conf is not None:
            self._dbg_kpt_raw = int(
                (np.asarray(det.kpt_conf) >= conf_thr).sum())
        else:
            self._dbg_kpt_raw = 0

        # 角点短时记忆：小漂移→窗口平均；短消失→回忆；鬼点→剔除
        kpts, kconf = det.kpts, det.kpt_conf
        if self._kpt_mem is not None:
            kpts, kconf = self._kpt_mem.update(kpts, kconf, now_ms)
        self._kpt_s, self._kconf_s = kpts, kconf
        self._dbg_kpt = int((np.asarray(kconf) >= conf_thr).sum()) \
            if kconf is not None else 0

        # 前端 → mode（用记忆后的角点）
        if kpts is not None and kpts.shape[0] >= 4 and kconf is not None:
            mode, ids = parse_kpt_mode(kpts, kconf, conf_thr)
            n = len(ids)
        else:
            mode, ids, n = MODE_COARSE, [], 0

        if mode in (MODE_FULL, MODE_P3P) and n >= 3:
            obj3s = self.obj3[ids]
            img2s = np.asarray(kpts)[ids]
            pnp = merge(sub(V, "pnp"), _D_PNP)
            # 上帧位姿用于消歧：平面 IPPE 双解在远距/小目标时 RMS 接近，纯靠 RMS 会帧间跳
            prev = self._last_pose
            res = gate_pose(self.camera, obj3s, img2s, prev=prev,
                            reproj_thr=num(pnp, "reproj_px", _D_PNP["reproj_px"]),
                            z_bounds=(num(pnp, "z_min", _D_PNP["z_min"]),
                                      num(pnp, "z_max", _D_PNP["z_max"])),
                            refine=flag(pnp, "refine", True))
            if res is not None:
                # z 跳变保护：上一帧也有可信位姿时，z 突变按错解弃帧
                # （错解若给出 z ≤ z.cross 会直接触发 THROUGH → 满速冲出去）
                max_jump = num(pnp, "max_z_jump_m", _D_PNP["max_z_jump_m"])
                if self._z_guard and self._z_last is not None and max_jump > 0:
                    z_new = float(res[1].ravel()[2])
                    if abs(z_new - self._z_last) > max_jump:
                        if S.DEBUG:
                            print("[GATE] 弃帧: z 跳变 %.2f→%.2f m (>%.2f)"
                                  % (self._z_last, z_new, max_jump))
                        res = None
            if res is not None:
                self._z_guard = True
                self._on_pose(det, res, now_ms, mode=mode, kpt=n)
                return
            mode = MODE_COARSE          # 位姿校验失败/跳变 → 退化 coarse
        # 本帧没有可用位姿：解除跳变判据（下个位姿重新建立基准，防卡死），
        # 并清零穿门确认计数 → 只有"连续帧"都判就近才算过门
        self._z_guard = False
        self._cross_cnt = 0
        if self.mode != mode:
            # 只在**位姿失败/退化**这条路径上执行（位姿成功时上面已 return），
            # 所以 full↔p3p 的帧间翻转不会清 _center_cnt；真正会走到的是
            # "位姿校验失败→coarse" 与 "width↔coarse 互换"。
            self._center_cnt = 0
        if mode == MODE_WIDTH:
            self._on_width(det, now_ms, ids)
            return
        self._on_coarse(det, now_ms)    # coarse / 其它退化

    # ---------------- full/p3p 位姿可用 ----------------
    def _on_pose(self, det, pose, now_ms, mode, kpt):
        """位姿档：**位姿只提供深度 z 与"可信"这一事实**；居中一律用**像素误差**。

        深度 z 有 ~20% 系统性尺度误差（实测），拿 t_x(m) 做居中等于把尺度误差灌进控制回路；
        而且米制/像素两套阈值+两套 PID 会在 mode 于 full↔coarse 间跳时交替工作 → 收敛不了。
        统一成像素后：**一套阈值(align.px_x/px_y) + 一套 PID**，全档位可比。
        门框中心 = 用位姿把门原点(0,0,0)投影回图像（p3p 缺角时也比角点均值准）。
        """
        G = self._G
        al = merge(sub(G, "align"), _D_ALIGN)
        zc = merge(sub(G, "z"), _D_Z)
        sg = merge(sub(G, "surge"), _D_SURGE)
        rvec, tvec = pose
        self._last_pose = pose
        self._reacquire_cnt = 0          # 拿到可信位姿 → 后退重取计数清零
        self.mode = mode
        z = float(tvec.ravel()[2])
        self._z_last = z
        self._z_ms = now_ms
        c = self.camera.project(np.zeros((1, 3), np.float32), rvec, tvec)[0]
        dxn = float((c[0] - self.w / 2.0) / (self.w / 2.0))
        dyn = float((c[1] - self.h / 2.0) / (self.h / 2.0))
        # 朝向误差（只有位姿档能测）：门法向 n=R·[0,0,1] → 机身相对门的航向/俯仰角。
        # dxn 是**方位**（门在画面里偏多少）→ 修它用平移；航向是**姿态** → 修它才用转向。
        # ⚠️ 只有 4 角(full) 才可信（p3p 的 psi 实测 std 64~69° = 噪声）⇒ 非 full 帧
        #    不喂滤波器、不更新 _hdg_deg，只留 last-good + 时间戳。
        if mode == MODE_FULL:
            psi_deg, _pit_deg = gate_normal_angles_deg(rvec, tvec)
            self._hdg_f = psi_deg if self._hdg_f is None else \
                self._hdg_f + 0.4 * (psi_deg - self._hdg_f)     # EMA(≈4 帧)
            self._hdg_deg = self._hdg_f
            self._hdg_ms = now_ms
        # 符号：图像 x 向右 = 机身向右 → sway 取正；图像 y 向下 → heave 取负。
        # 这里 yaw 恒 0（居中只用 sway；姿态交给 ALIGN.HDG，见 `_lateral_out`）。
        sway, yaw = self._lateral_out(dxn, now_ms), 0.0
        heave = -_dof_clip(self._pid_heave_px.update(dyn, now_ms))
        aligned = abs(dxn) <= num(al, "px_x", _D_ALIGN["px_x"]) and \
            abs(dyn) <= num(al, "px_y", _D_ALIGN["px_y"])
        dx_m, dy_m = dxn, dyn            # 日志里 dx/dy **统一是像素归一化**
        cross = num(zc, "cross", _D_Z["cross"])
        if z <= cross:
            # 穿门确认：单帧就近不冲（平面 PnP 偶发错解若给出 z≤cross，直接 THROUGH = 满速冲出），
            # 连续 n 帧才判过门。
            self._cross_cnt += 1
            need = int(num(zc, "cross_confirm_frames", _D_Z["cross_confirm_frames"]))
            if self._cross_cnt >= max(1, need):
                # 冲刺：只前进、不带横向微调（微调已在上一帧做完）
                self._start_through()
                self._set_info("through", mode=mode, z=z,
                               surge=self._through_speed(),
                               dx=dx_m, dy=dy_m, kpt=kpt)
            else:
                # 冲刺前最后一帧：不前进，只做横向微调对准
                self._set_info("center", mode=mode, z=z, dx=dx_m, dy=dy_m,
                               sway=sway, heave=heave, yaw=yaw, kpt=kpt)
            return
        self._cross_cnt = 0

        if self.phase in (PH_SEARCH,):
            self.phase = PH_ALIGN
            self.substate = SUB_GOLDEN
            self._center_cnt = 0
        if self.phase == PH_ALIGN:
            # ---- ① 正航向（SUB_HDG）：居中达标后进来，转完回 GOLDEN 复核居中 ----
            if self.substate == SUB_HDG and not self._hdg.finished():
                st_h, yaw_cmd = self._hdg.step(
                    now_ms, psi_deg=self._hdg_deg,
                    psi_fresh=self._hdg_fresh(mode, now_ms),
                    yaw_telemetry=self._uart_yaw(), gate_lost=False)
                if self._hdg.finished():
                    self._hdg_done = True
                    self.substate = SUB_GOLDEN
                    self._center_cnt = 0          # 转向会动到门在画面里的位置 → 复核居中
                # HDG 期间**只转**：sway/heave/surge 全 0（测量要求船静止）
                self.last_info["hdg_i"] = self._hdg.iters
                self._set_info("hdg" if not self._hdg.finished() else "center",
                               mode=mode, z=z, dx=dx_m, dy=dy_m,
                               sway=0.0, heave=0.0, surge=0.0, yaw=yaw_cmd, kpt=kpt,
                               hdg=self._hdg_deg, hdg_state=self._hdg.state)
                return
            self.substate = SUB_GOLDEN
            if self._loiter_commit(dxn, dyn, float(det.w) / float(self.w), now_ms):
                return
            if aligned:
                self._center_cnt += 1
                if self._center_cnt >= int(num(al, "confirm_frames", _D_ALIGN["confirm_frames"])):
                    if self._hdg_ready(mode, now_ms):
                        # 居中达标 → **先正航向**（离散：测→转→停稳→再测）
                        self._wait_hdg_ms = None
                        self.last_info["hdg_skip"] = None    # 已经进来了，清掉"跳过"标记免得日志误读
                        self.substate = SUB_HDG
                        self._hdg.start(now_ms)
                        self.last_info["hdg_i"] = self._hdg.iters
                        self._set_info("hdg", mode=mode, z=z, dx=dx_m, dy=dy_m,
                                       sway=0.0, heave=0.0, yaw=0.0, kpt=kpt,
                                       hdg=self._hdg_deg, hdg_state=self._hdg.state)
                        return
                    # 没进正航向：先看要不要"原地等一帧新鲜 full"（wait_fresh_ms>0）
                    why = self._hdg_skip_reason(mode, now_ms)
                    wait_ms = float(self._hdg_cfg.get("wait_fresh_ms", 0.0) or 0.0)
                    if wait_ms > 0:
                        if self._wait_hdg_ms is None:
                            self._wait_hdg_ms = now_ms
                        if (now_ms - self._wait_hdg_ms) < wait_ms:
                            # GOLDEN 本来就不下发 surge ⇒"等"=原地保持对中（代价只是时间）
                            self._set_info("wait_hdg", mode=mode, z=z, dx=dx_m, dy=dy_m,
                                           sway=sway, heave=heave, yaw=0.0, kpt=kpt,
                                           hdg=self._hdg_deg, hdg_skip=why,
                                           hdg_state=self._hdg.state)
                            return
                    self._note_hdg_skip(why)
                    self._wait_hdg_ms = None
                    self.phase = PH_APPROACH
                    self.substate = ""
            else:
                self._center_cnt = 0
                self._wait_hdg_ms = None       # 又偏出去 → 等待计时作废
            self._set_info("center", mode=mode, z=z, dx=dx_m, dy=dy_m,
                           sway=sway, heave=heave, yaw=yaw, kpt=kpt,
                           hdg=self._hdg_deg, hdg_state=self._hdg.state)
        elif self.phase == PH_APPROACH:
            # 进近两档（远→快 / 近→慢）；分档点 = z.slow_max。速度档与撞球共用（见 `_speed`）。
            # 注：z.fast_max / z.align_max 当前**未参与运算**（见 cfg 注释）
            s_slow = self._speed("slow")
            if z > num(zc, "slow_max", _D_Z["slow_max"]):
                surge = self._speed("fast")
            else:
                surge = s_slow
            self._set_info("forward_%s" % ("fast" if surge > s_slow else "slow"),
                           mode=mode, z=z, dx=dx_m, dy=dy_m,
                           sway=sway, heave=heave, surge=surge, kpt=kpt,
                           hdg=self._hdg_deg)
        else:
            self._set_info("center", mode=mode, z=z, dx=dx_m, dy=dy_m,
                           kpt=kpt)

    # ---------------- width（对向 2 角：测距 + 中点对中 → 慢 creep 靠近） ----
    def _on_width(self, det, now_ms, ids):
        """width 档：只有对向 2 角（上边或下边），信息只够"水平中点 + 框心竖直"，不解 PnP。

          ① z > width.z_max（还远）且对准 → 慢 creep 靠近（换取角点/整门）；
          ② 否则 HOLD（surge=0；未对中时宁可等，不蹭杆）。
        出口仍然只有 `_tick_lost`（近距丢失）与 `_loiter_commit`（门口超时）两条。
        z 由 fx·W/Δu 粗估，只用来判"该不该靠近"，不参与闭环。
        """
        G = self._G
        W = merge(sub(G, "width"), _D_WIDTH)
        al = merge(sub(G, "align"), _D_ALIGN)
        sg = merge(sub(G, "surge"), _D_SURGE)
        self.mode = MODE_WIDTH
        kp = self._kpt_s if self._kpt_s is not None else det.kpts
        u1, v1 = kp[ids[0]]
        u2, v2 = kp[ids[1]]
        z = width_range_depth(u1, u2, float(self.camera.fx), self.frame_w_m)
        if z is None:
            self._on_coarse(det, now_ms)
            return
        z = float(np.clip(z, _D_PNP["z_min"], _D_PNP["z_max"]))
        self._z_last = z
        self._z_ms = now_ms
        aim_x = (u1 + u2) / 2.0
        cx, cy = bbox_center(det)
        dxn = (aim_x - self.w / 2.0) / (self.w / 2.0)
        dyn = (cy - self.h / 2.0) / (self.h / 2.0)
        # 对中判据（归一化像素偏差）：**全档位统一**（位姿档也用这一套，见 _on_pose）
        aligned = abs(dxn) <= num(al, "px_x", _D_ALIGN["px_x"]) and \
            abs(dyn) <= num(al, "px_y", _D_ALIGN["px_y"])
        if self.phase == PH_SEARCH:
            self.phase = PH_ALIGN
            self._center_cnt = 0
        # 水平修正：**一律 sway**（居中不用 yaw；见 `_lateral_out`）
        yaw = 0.0
        sway = self._lateral_out(dxn, now_ms)
        heave = -_dof_clip(self._pid_heave_px.update(dyn, now_ms))

        if self.phase == PH_ALIGN:
            # 在门口超时兜底（实测：门占比 0.97 时整框仍检测得到 → 不走丢门分支）
            if self._loiter_commit(dxn, dyn, float(det.w) / float(self.w), now_ms):
                return
            # 出口①：距门仍远且对中 → 慢 creep（有限信息下安全推进）
            if aligned and z > num(W, "z_max", _D_WIDTH["z_max"]):
                self.substate = SUB_CREEP
                self._center_cnt += 1
                surge = num(sg, "creep", _D_SURGE["creep"])
                if self._center_cnt >= int(num(al, "confirm_frames", _D_ALIGN["confirm_frames"])):
                    self.phase = PH_APPROACH
                    self.substate = ""
            else:
                self.substate = SUB_HOLD
                surge = 0.0
            self._set_info("creep" if surge > 0 else "hold", mode=MODE_WIDTH,
                           z=z, dx=dxn, dy=dyn, sway=sway, heave=heave,
                           surge=surge, yaw=yaw)
            return
        # APPROACH / 其它：等下一次"对准确认"再冲（上面的出口对本相位同样生效）
        self._set_info("center", mode=MODE_WIDTH, z=z, dx=dxn, dy=dyn,
                       sway=sway, heave=heave, yaw=yaw)

    # ---------------- coarse（整门框可信/角不足）：走位仲裁 ----------------
    def _on_coarse(self, det, now_ms):
        """coarse 档：角点不足，只有整框可信。**只有未对准才后退**
        （实测来回退的主因是"一看框大就退"，与是否对准无关）：
          ① 对准 → 慢 creep 靠近（争取露出角点）；
          ② 未对准：远距只对中；中距 HOLD 超限 → REACQUIRE；很近(框装不下) → REACQUIRE。
        出口同样只有 `_loiter_commit` 与 `_tick_lost` 两条。
        """
        G = self._G
        C = merge(sub(G, "coarse"), _D_COARSE)
        H = merge(sub(G, "hold"), _D_HOLD)
        sg = merge(sub(G, "surge"), _D_SURGE)
        self.mode = MODE_COARSE
        # coarse 档**没有任何测距**：`_z_last` 会一直是上次 width/位姿留下的陈旧值
        # （现场留下过 9.25 的垃圾值，一路带到穿门判定）→ 标它失效、不参与判据；
        # 只失效**判据**，不动 `_z_last` 本身（日志仍要能看到它实际是多少）。
        self._z_ms = None
        ratio = float(det.w) / float(self.w)
        # 已在 REACQUIRE：闭环后退（退到框够小即停）或超时回 search
        if self.phase == PH_ALIGN and self.substate == SUB_REACQUIRE:
            self._tick_reacquire(now_ms, ratio)
            return
        cx, cy = bbox_center(det)
        dxn = (cx - self.w / 2.0) / (self.w / 2.0)
        dyn = (cy - self.h / 2.0) / (self.h / 2.0)
        aligned = abs(dxn) <= num(C, "align_x", _D_COARSE["align_x"]) and \
            abs(dyn) <= num(C, "align_y", _D_COARSE["align_y"])

        if self.phase == PH_SEARCH:
            self.phase = PH_ALIGN
            self._hold_cnt = 0
        if self.phase == PH_APPROACH:
            # 进近中角全失 → 保守降回 ALIGN 仲裁（避免盲冲）
            self.phase = PH_ALIGN
            self._hold_cnt = 0

        # 水平修正：**一律 sway**（coarse 也用平移；居中不用 yaw，见 `_lateral_out`）
        yaw = 0.0
        sway = self._lateral_out(dxn, now_ms)
        heave = -_dof_clip(self._pid_heave_px.update(dyn, now_ms))

        if self.phase != PH_ALIGN:
            self._set_info("hold", z=self._z_last, dx=dxn, dy=dyn,
                           sway=sway, heave=heave, kpt=self._dbg_kpt)
            return

        if self._loiter_commit(dxn, dyn, ratio, now_ms):
            return
        if aligned:
            # 对准：能靠近就靠近（争取露出角点），不再中距干等
            self.substate = SUB_CREEP
            self._hold_cnt = 0
            self._set_info("creep", z=self._z_last, dx=dxn, dy=dyn,
                           sway=sway, heave=heave, yaw=yaw,
                           surge=num(sg, "creep", _D_SURGE["creep"]), kpt=self._dbg_kpt)
            return

        if ratio < num(C, "far_ratio", _D_COARSE["far_ratio"]):       # 远距小框：没对准就别冲
            self.substate = SUB_CREEP
            self._set_info("center", z=self._z_last, dx=dxn, dy=dyn,
                           sway=sway, heave=heave, yaw=yaw, kpt=self._dbg_kpt)
        elif ratio <= num(C, "near_ratio", _D_COARSE["near_ratio"]):   # 中距 → HOLD（无位姿即无进展）
            self.substate = SUB_HOLD
            self._hold_cnt += 1
            if self._hold_cnt >= int(num(H, "max_frames", _D_HOLD["max_frames"])):
                self._enter_reacquire(now_ms, ratio)
            else:
                self._set_info("hold", z=self._z_last, dx=dxn, dy=dyn,
                               sway=sway, heave=heave, yaw=yaw, kpt=self._dbg_kpt)
        elif self._give_up_until_ms is not None and \
                now_ms < self._give_up_until_ms:
            # 刚放弃过后退：窗口内只原地保持（等位姿恢复或时间窗过期再试）
            self.substate = SUB_HOLD
            self._set_info("hold", z=self._z_last, dx=dxn, dy=dyn,
                           sway=sway, heave=heave, yaw=yaw, kpt=self._dbg_kpt)
        else:                                        # 近距装不下且没对准 → 后退重取
            self._enter_reacquire(now_ms, ratio)

    def _enter_reacquire(self, now_ms, ratio=None):
        """进入后退重取；同一段里连续超 max_times 次就**放弃后退，改原地保持**。

        倒影持续干扰时"退-进-退"会来回震荡；退了几次仍拿不到可用角点，说明再退也没用。
        """
        G = self._G
        R = merge(sub(G, "reacquire"), _D_REACQ)
        # 距上次后退够久 → 计数重新开始（否则一次倒影干扰会把后退永久锁死）
        reset_after = int(num(R, "reset_after_ms", _D_REACQ["reset_after_ms"]))
        if self._reacquire_last_ms is None or \
                now_ms - self._reacquire_last_ms >= reset_after:
            self._reacquire_cnt = 0
        self._reacquire_cnt += 1
        max_times = int(num(R, "max_times", _D_REACQ["max_times"]))
        if max_times > 0 and self._reacquire_cnt > max_times:
            self.substate = SUB_HOLD
            self._hold_cnt = 0
            # 放弃后退：原地保持 reset_after_ms，之后再允许重新尝试（不再反复退）
            self._give_up_until_ms = now_ms + reset_after
            if S.DEBUG:
                print("[GATE] REACQUIRE 放弃(第 %d 次 > max_times=%d, ratio=%.2f "
                      "kpt=%d) → 原地 HOLD %.1fs"
                      % (self._reacquire_cnt, max_times, self._dbg_ratio,
                         self._dbg_kpt, reset_after / 1000.0))
            self._set_info("hold", z=self._z_last, kpt=self._dbg_kpt)
            return
        self.substate = SUB_REACQUIRE
        self._reacquire_start = now_ms
        self._reacquire_last_ms = now_ms      # 只有真正后退才更新（放弃不算）
        self._give_up_until_ms = None
        # 闭环基准：进入时的框占比（退到它明显变小就停）
        self._reacquire_ratio0 = float(ratio) if ratio is not None else None
        if S.DEBUG:
            print("[GATE] REACQUIRE #%d: 角不足/过近 (ratio=%.2f kpt=%d hold=%d)"
                  " → 后退重取" % (self._reacquire_cnt, self._dbg_ratio,
                                  self._dbg_kpt, self._hold_cnt))

    def _tick_reacquire(self, now_ms, ratio=None):
        """后退重取：**闭环** —— 退到框够小就停，不再固定退满 max_ms。"""
        G = self._G
        R = merge(sub(G, "reacquire"), _D_REACQ)
        sg = merge(sub(G, "surge"), _D_SURGE)
        r0 = self._reacquire_ratio0
        stop_ratio = num(R, "stop_ratio", _D_REACQ["stop_ratio"])
        if ratio is not None and r0 and ratio <= r0 * stop_ratio:
            if S.DEBUG:
                print("[GATE] REACQUIRE 已退够(ratio=%.2f<=%.2f=%.2f×%.2f) → 回对准"
                      % (ratio, r0 * stop_ratio, r0, stop_ratio))
            self.substate = SUB_HOLD
            self._hold_cnt = 0
            self._set_info("hold", z=self._z_last, kpt=self._dbg_kpt)
            return
        dur = now_ms - (self._reacquire_start or now_ms)
        if dur >= num(R, "max_ms", _D_REACQ["max_ms"]):
            self._start_search()
            self._set_info("search")
            return
        # 后退方向/量可配（dof sign 需水池实测；负 surge = 后退）
        surge = -num(sg, "reacquire", _D_SURGE["reacquire"])
        self._set_info("reacquire", substate=SUB_REACQUIRE, surge=surge,
                       kpt=self._dbg_kpt)

    # ---------------- 丢目标 / 穿门 ----------------
    def _loiter_commit(self, dxn, dyn, ratio, now_ms, kpt=None):
        """在门口超时兜底：**不看档位**，只要「人在门口 + 对准」持续太久就自己拍板直冲。

        三个条件都成立才算命中（命中 → `_start_through()` 并返回 True）：
          · 框占比 ≥ `z.near_lost_ratio`（与"丢门判过门"同一个"在门口"定义）
          · |dxn| ≤ `loiter.dx_max`、|dyn| ≤ `loiter.dy_max`（安全带，归一化像素偏差）
          · 上述状态**连续** ≥ `loiter.timeout_ms`（任一条不成立就重新计时）
        为什么需要：门在占比 0.97（≈0.5m）时整框仍检测得到 → 走不到"丢门"分支，没有这条
        就会贴着门口一直 creep（现场实测爬了 18.7s）。实测记录见 md §3。
        """
        L = merge(sub(self._G, "loiter"), _D_LOITER)
        if not flag(L, "enable", True):
            self._loiter_start_ms = None
            return False
        near_r = num(sub(self._G, "z"), "near_lost_ratio", _D_Z["near_lost_ratio"])
        ok = (float(ratio or 0.0) >= near_r and
              abs(float(dxn)) <= num(L, "dx_max", _D_LOITER["dx_max"]) and
              abs(float(dyn)) <= num(L, "dy_max", _D_LOITER["dy_max"]))
        if not ok:
            self._loiter_start_ms = None
            return False
        if self._loiter_start_ms is None:
            self._loiter_start_ms = now_ms
            return False
        wait = num(L, "timeout_ms", _D_LOITER["timeout_ms"])
        if (now_ms - self._loiter_start_ms) < wait:
            return False
        if S.DEBUG:
            print("[GATE] 在门口停留 %.1fs 仍未 commit（占比=%.2f dx=%+.2f dy=%+.2f）"
                  " → 兜底判过门并直冲"
                  % ((now_ms - self._loiter_start_ms) / 1000.0, float(ratio or 0.0),
                     float(dxn), float(dyn)))
        self._start_through()
        self._set_info("through", z=self._z_last, dx=dxn, dy=dyn,
                       surge=self._through_speed(), kpt=self._dbg_kpt if kpt is None else kpt,
                       ratio=ratio)
        return True

    def _near_lost(self, now_ms, zc):
        """整门丢失时：判断是不是**已经到门口了**（≈机身已进门框）。两条判据取 OR：

          ① `z ≤ z.near_lost_m` **且 z 新鲜**：coarse 档没有测距、时间戳被清空，
             所以陈旧的垃圾 z（现场出现过 9.25）不会影响判定；
          ② 末次检测到的**框占比 ≥ z.near_lost_ratio**：全档位可用、每帧都算，
             门快装不下/机身进门框时必然成立。
        返回命中的判据名（空串 = 没到门口）。
        """
        near_m = num(zc, "near_lost_m", _D_Z["near_lost_m"])
        near_r = num(zc, "near_lost_ratio", _D_Z["near_lost_ratio"])
        stale = num(zc, "z_stale_ms", _D_Z["z_stale_ms"])
        z_fresh = (self._z_ms is not None and self._z_last is not None and
                   stale > 0 and (now_ms - self._z_ms) <= stale)
        if z_fresh and self._z_last <= near_m:
            return "z=%.2f" % self._z_last
        if float(self._dbg_ratio or 0.0) >= near_r:
            return "ratio=%.2f" % float(self._dbg_ratio or 0.0)
        return ""

    def _tick_lost(self, now_ms):
        G = self._G
        zc = merge(sub(G, "z"), _D_Z)
        sg = merge(sub(G, "surge"), _D_SURGE)
        self._lost_cnt += 1
        self._z_guard = False             # 丢目标→解除 z 跳变基准
        self._cross_cnt = 0
        pose_hold = int(num(G, "pose_hold_frames", _D_TASK["pose_hold_frames"]))
        if self.phase in (PH_ALIGN,) and self.substate == SUB_REACQUIRE:
            self._tick_reacquire(now_ms)      # 已丢目标：无 ratio，按时间退完
            return
        if self.phase == PH_ALIGN:
            self._hdg_abort("整门丢失")
            # 近距丢失 = 已过门（旧版这里一律防抖后回 SEARCH，于是"creep 到门口 → 整门丢失
            # → SEARCH → 原地不动"，永远数不到过门）。判据用**最后一次检测到的框占比**
            # （每帧更新、比 z 更可靠且全档位可用）；远处丢检（占比小）仍走 SEARCH。
            why = self._near_lost(now_ms, zc)
            if why:
                if S.DEBUG:
                    print("[GATE] ALIGN 近距丢失(%s) → 判过门" % why)
                self._start_through()
                self._set_info("through", surge=self._through_speed())
            elif self._lost_cnt <= pose_hold:
                # 帧间防抖：沿用上一帧对中保持
                self._set_info("hold", z=self._z_last)
            else:
                self._start_search()
                self._set_info("search")
            return
        if self.phase == PH_APPROACH:
            why = self._near_lost(now_ms, zc)
            if why:
                # 已到近距却整门丢失：门占满视野/机身进入门框 = 真过门的典型现象
                # → 直接判过门并直行穿越（不等防抖、不后退），否则永远数不到过门。
                if S.DEBUG:
                    print("[GATE] 近距丢失(%s) → 判过门" % why)
                self._start_through()
                self._set_info("through", surge=self._through_speed())
            elif self._lost_cnt <= pose_hold and self._z_last is not None:
                # 还在远处、只是短暂丢失：轻微后退换回重新锁定/更大视野（不带速度盲冲），
                # 退满防抖窗口仍无目标 → 转 SEARCH
                self._set_info("backward_slow", z=self._z_last,
                               surge=-num(sg, "lost_backward", _D_SURGE["lost_backward"]))
            else:
                self._start_search()
                self._set_info("search")
            return
        # SEARCH：左右平移扫视（不许旋转，见 `_search_sweep`）
        sway = self._search_sweep(now_ms)
        self._set_info("search", sway=sway)

    def _tick_through(self, now_ms):
        """穿门：只前进、不微调（横向修正全部留在冲刺前完成）。

        结束条件优先**时长** `through.confirm_ms`（帧数语义下冲刺时长随 fps 漂，而"能不能
        冲出去"取决于跑了多远）；`confirm_ms<=0` → 退回旧的帧数语义 `confirm_frames`。
        ⚠️ confirm_ms 要按实测航速标定：需要冲的距离 ≈ z.cross(0.77，真 0.77 m 的触发点）
        ~0.58m) + 机身长度 → confirm_ms ≈ 1000·(0.58+L)/v，再留 30% 余量。
        """
        G = self._G
        T = merge(sub(G, "through"), _D_THROUGH)
        self._through_frames += 1
        self._lost_cnt += 1               # confirm_ms<=0 时按帧数兜底
        if self._through_start_ms is None:
            self._through_start_ms = now_ms
        ms = num(T, "confirm_ms", _D_THROUGH["confirm_ms"])
        if ms > 0:
            done = (now_ms - self._through_start_ms) >= ms
        else:
            done = self._lost_cnt >= max(1, int(num(T, "confirm_frames", _D_THROUGH["confirm_frames"])))
        # 直行穿越；满足结束条件 → 判机身过门
        if done:
            self._pass_cnt += 1
            if S.DEBUG:
                print("[GATE] 通过第 %d/%d 门"
                      % (self._pass_cnt, int(num(G, "pass_target", _D_TASK["pass_target"]))))
            if self._pass_cnt >= int(num(G, "pass_target", _D_TASK["pass_target"])):
                self._finish("pass")
            else:
                self._start_search()
                self._set_info("search")
            return
        self._set_info("through", surge=self._through_speed())
