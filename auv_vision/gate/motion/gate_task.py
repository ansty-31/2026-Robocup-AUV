# -*- coding: utf-8 -*-
"""gate/motion/gate_task.py — 任务三 过门（GateTask 相位机）
门 = 对称平面矩形门框(外缘 0.77×0.56 m，管外径 5cm)，悬空，无朝向要求；任务 = 机身穿过开口。
相位：SEARCH → ALIGN → APPROACH → THROUGH(计数) → (下一门)/DONE
ALIGN 子状态（按前端 mode 与门框占屏比仲裁）：
GOLDEN      full/p3p 位姿对准（sway/heave PID，无 surge）
CREEP       远距小框：慢速靠近 + 框心对中
HOLD        中/近角不足：surge=0 保持对中；≥hold.max_frames → REACQUIRE
REACQUIRE   短后退重取整门；超限 → SEARCH
SWAY_BACK   转向结束后反向平移一段，找回刚转走的那个门（判据见 _D_SWAY_BACK）
直冲出口三条（任一成立即进 THROUGH：忽略角点丢失直行、不再做横向微调）：
① 位姿可信且 Z ≤ z.cross 连续 cross_confirm_frames 帧；
② 近距（Z ≤ z.near_lost_m 且 z 新鲜，或框占比 ≥ z.near_lost_ratio）整门丢失；
③ 在门口超时兜底 `loiter.*`（门口 + 安全带内停留超时 → 自己拍板直冲）。
三条都要先过 `through.require_align_deg`（冲刺前的航向闸门）。
位姿跳变保护：帧间 z 变化 > pnp.max_z_jump_m 的帧按错解弃掉（防平面 PnP 错解直冲）。
参数：cfg/vision.yaml `gate.*`、cfg/comm.yaml `gate.*`；读取一律走 cfgnode 兜底，
**配置缺键时用"等于当前 cfg 的默认值"**，不会 AttributeError。"""

from __future__ import annotations

import numpy as np

import base.cfg.settings as S
from common.motion.PID import PID
from base.log.turn_log import turn_log              # 转向调用日志（只写字，不参与控制）
from common.cfg.cfgnode import (flag, merge, motion_node, motion_num, num, pid_kw,
                            sub)
from gate.percept.gate_detector import board_camera
from gate.percept.kpt_memory import ENV_ENABLE, build_kpt_memory, enable_source
from gate.percept.gate_frontend import (parse_kpt_mode, width_range_depth, bbox_center,
                                MODE_FULL, MODE_P3P, MODE_WIDTH, MODE_COARSE)
from gate.percept.geometry import (object_points, gate_pose, gate_normal_angles_deg,
                            GATE_FRAME_W, GATE_FRAME_H)
from gate.motion.heading_align import HeadingAligner, _D_HDG, hdg_cfg
from gate.percept.gate_postproc import (det_cfg, postproc_cfg, select_cfg,
                                pick as postproc_pick)

PH_SEARCH = "SEARCH"
PH_ALIGN = "ALIGN"
PH_APPROACH = "APPROACH"
PH_THROUGH = "THROUGH"

SUB_HDG = "HDG"
SUB_GOLDEN = "GOLDEN"
SUB_CREEP = "CREEP"
SUB_HOLD = "HOLD"
SUB_REACQUIRE = "REACQUIRE"
# 「转完反向平移」= 主循环里的**一个状态**（不是 `_step` 顶端的旁路）：
#   进入：转向结束那一刻（`_start_post_sway`）；每帧：检测/PnP/ψ/居中判据**照跑**，
#   本状态只覆盖**本帧下发什么指令**（只发 sway，不发 yaw）；退出：到期 或 认出刚丢的那个门。
SUB_SWAY_BACK = "SWAY_BACK"

# ---------------------------------------------------------------- 缺键兜底
# cfg 缺键/缺文件时的兜底，取值**等于当前 cfg**（用例 test_gate_defaults_match_cfg 钉住"兜底 == cfg"）。
#    唯一来源 = `comm.motion.*`（代码兜底 `common/cfgnode.MOTION_DEFAULTS`）。gate 只在
#    `comm.gate.pid_sway/pid_heave`、`comm.gate.surge.fast/slow` 里写"确实要单独一套"的覆盖。
# gate 里**用 yaw 的地方只有一处**：ALIGN.HDG 正航向（增益 comm.motion.turn_pid）；居中一律 sway。
_D_ALIGN = dict(confirm_frames=4, px_x=0.20, px_y=0.25)
# confirm_frames：连续对准帧数（防抖，**别拿降帧数当"放宽"**）—— 判定"居中成功"。
#    阈值设到死区以下就只能靠惯性滑进窗口（可能永远确认不了居中）。要更准就动 pid_heave。
# 「在门口超时兜底」= 出口③：已在门口(占比≥z.near_lost_ratio) + 在安全带内，
# 连续超过 timeout_ms 仍未 commit → 自己判过门并直冲（没有这条会在门口一直 creep）。
_D_LOITER = dict(enable=True, near_ratio=0.60, timeout_ms=3000, dx_max=0.14, dy_max=0.20)
# z.cross：穿门判据（出口①）。near_lost_* 是"到门口了"的两条判据（见出口②）。
_D_Z = dict(cross=1.0, cross_confirm_frames=2, slow_max=1.3, near_lost_m=1.2,
            near_lost_ratio=0.50, z_stale_ms=1500)
# near_lost_m / near_lost_ratio：判"已经到门口了"的两条 OR 判据（出口②）。
_D_SURGE = dict(creep=0.20, lost_backward=0.20, reacquire=0.25, through=0.6)
# 进近两档 fast/slow 取自 comm.motion.surge_fast/surge_slow（与撞球共用同一份，见 _speed）。
_D_COARSE = dict(far_ratio=0.18, near_ratio=0.55, align_x=0.33, align_y=0.29)
# 框占比阈值是**像素比**，与配置 fx 无关 ⇒ 不参与米制映射（本身也不动）。
_D_WIDTH = dict(z_max=1.5)
_D_TASK = dict(timeout_ms=180000, pass_target=4, pose_hold_frames=10)
#   ① **优先用距离**：|本帧新鲜 z − 转走那一刻的 z| ≤ `post_sway_same_z_m` ⇒ 同一个门。
#   ② **z 拿不到时才退回框占比**：本帧占比 ≥ `post_sway_far_ratio` × 转走那一刻的占比 ⇒ 同一个门。
#      z 拿不到 = 当帧 coarse / 位姿已超 `gate.z.z_stale_ms` / 转前没测到过 z。
#   ③ 两道闸都设 0 ⇒ 关掉判据 ⇒ 任何检出都算"门回来了"。
#   ④ 本帧没检出 ⇒ 不算"门回来了"（继续推，由到期判据兜底）。
#      但**默认值**取自 `_D_HDG` 同一份 ⇒ cfg ↔ 兜底的一致性只由一处守（test_gate_defaults_match_cfg）。
_D_SWAY_BACK = dict(post_sway_same_z_m=_D_HDG["post_sway_same_z_m"],
                    post_sway_far_ratio=_D_HDG["post_sway_far_ratio"])
# conf_thr：**任务侧**角点可信下限（= doc/设计/gate_pose_decode_spec.md §6 的 V_MIN）。
#   `mode == full` 要求 4 个角点都 ≥ 它，而 **psi 只吃 full 帧** ⇒ 直接决定正航向能不能启动。
_D_KPT = dict(conf_thr=0.8)
_D_HOLD = dict(max_frames=20)
_D_REACQ = dict(max_ms=800, max_times=2, stop_ratio=0.75, reset_after_ms=3000)
# THROUGH 结束条件按**时间**（帧数语义下冲刺时长随 fps 漂，而"要冲多远"是距离问题）；
# confirm_ms<=0 → 退回旧的帧数语义。
_D_THROUGH = dict(confirm_frames=8, confirm_ms=3000, near_lost_confirm_ms=5000,
                  require_align_deg=8.0)
# require_align_deg（冲刺前的航向闸门）：最近一次测到的 psi 必须 ≤ 它才允许进 THROUGH
#   （≤0 = 关闸）；从没测到过 psi 时放行（没得判）。
# SEARCH = **左右平移扫视**（不允许旋转搜索）。
# sweep_s 是第一轮时长；之后每轮 ×2 封顶 sweep_max_s（固定时长只会原地横跳）。
_D_SEARCH = dict(sweep_s=2.0, sweep_max_s=32.0, pause_s=1.0, sway=0.6)
#    是代码兜底表漂了 —— 用例 test_gate_defaults_match_cfg 挡的就是这种事）。
#    **必须与 `cfg/comm.yaml → comm.gate.search.sway` 同值**（同一条守卫用例）。
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
        self.body_center_offset = float(geo["body_center_offset"])
        self.obj3 = object_points(self.frame_w_m, self.frame_h_m)
        self.camera = board_camera()
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
        _ks = ("kp", "ki", "kd", "out_max", "deadzone")
        print("[GATE] 修正增益来源=%s | sway %s | heave %s | 转向=仅正航向(%s)"
              % (self._gain_src,
                 {k: self._pid_sway_kw[k] for k in _ks},
                 {k: self._pid_heave_kw[k] for k in _ks},
                 "enable" if hdg_cfg().get("enable", True) else "off"))
        # 正航向（ALIGN.HDG；见 gate/motion/heading_align.py）：每门只做一次，出结论后锁定。
        self._hdg = HeadingAligner(log=(print if S.DEBUG else (lambda *a: None)))
        self._hdg_cfg = hdg_cfg()
        # ---- 选门策略（规范 doc/设计/gate_pose_decode_spec.md §4；实现 gate/percept/gate_postproc.py）----
        self._select_cfg = select_cfg()          # near(默认) | corner(旧行为)
        self._post = postproc_cfg()
        print("[GATE] 选门策略=%s | 去重/几何=%s | 角点可信下限=%.2f | 候选 conf=%.2f"
              % (self._select_cfg["mode"],
                 ("edge_tol=%.2f/min_edges=%d/geom=%s"
                  % (self._post["edge_tol"], self._post["min_edges"],
                     self._post["geom_check"])),
                 num(sub(V, "keypoint"), "conf_thr", _D_KPT["conf_thr"]),
                 det_cfg()))
        self._hdg_done = True             # True=本门不需要再正航向（未启用或已出结论）
        self._hdg_done = not flag(self._hdg_cfg, "enable", True)
        self._hdg_f = None                # EMA 状态（只吃 full 帧，避免 p3p 垃圾污染）
        self._hdg_deg = None              # 最近一次 full 帧测到的航向误差（度）
        self._hdg_ms = None               # 上面那个值的时刻
        self._hdg_turns = 0               # 本门已转向次数（**无上限**；逐小步逼近，仅用于日志）
        self._hdg_turns_last_ms = None    # 上一次转向**结束**的时刻（判"有没有新的 ψ"用）
        self._hdg_skip_logged = False     # 本门是否已报过"跳过正航向"的原因（每门一次）
        self._post_sway_until_ms = None   # 「转完反向平移」状态：非 None = 激活（到期时刻）
        self._post_sway = 0.0
        self._post_sway_back = 0.0        # 补偿里的**缓慢后退**分量（负=后退）
        self._post_sway_back = 0.0        # 转完补偿里的**缓慢后退**分量（负=后退）             # 该状态里下发的 sway（= 转向的反方向）
        self._post_sway_z_ref = None      # 转走那一刻的 z（**主判据**）
        self._post_sway_ratio_ref = None  # 转走那一刻的框占比（z 拿不到时的退路）
        self._now_ms = None               # 本帧时基（process 每帧写入）
        self._det_now = None              # 本帧选中的门（状态判"门回来了吗"用当帧数据）

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
        self._hdg_skip_logged = False
        self._hdg_turns = 0
        self._hdg_turns_last_ms = None
        self._post_sway_until_ms = None
        self._post_sway = 0.0
        self._post_sway_back = 0.0        # 补偿里的**缓慢后退**分量（负=后退）
        self._post_sway_back = 0.0        # 转完补偿里的**缓慢后退**分量（负=后退）
        self._post_sway_z_ref = None
        self._post_sway_ratio_ref = None
        self._loiter_start_ms = None     # 「已在门口」的起始时刻（超时兜底）
        self._reacquire_start = None
        self._reacquire_ratio0 = None    # 进入 REACQUIRE 时的框占比（闭环后退基准）
        self._reacquire_cnt = 0          # 连续 REACQUIRE 次数（超限不再后退）
        self._reacquire_last_ms = None   # 上次真正后退的时刻（时间窗重置计数）
        self._give_up_until_ms = None    # 放弃后退后的 HOLD 窗口截止
        self._through_block_logged = False  # 本门是否已报过「航向闸门拦下冲刺」（每门一次）          # 连续 REACQUIRE 次数（超限不再后退）
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
        位置误差该用平移修 —— sway 命令 = kp(8.0)·dxn，|dxn|>0.017 就能过执行器死区(0.138)；
        而且 yaw 是**方位**控制，把门拉到光轴上 ≠ 机身与门平行。姿态由 ALIGN.HDG 负责。"""
        return _dof_clip(self._pid_sway_px.update(dxn, now_ms))

    def _hdg_new_psi_since_turn(self):
        """这次起转能不能用**当前这个 ψ**：必须是"上一次转向结束之后"测到的。
        之后防自转的闸门：没有它，转向一结束就会拿"起转前那个旧 ψ"立刻再转一次，无限循环。
        够不够正不在这里判（`|ψ| ≤ tol` 时 `HeadingAligner.start()` 自己给 DONE、不转）。
        基准是 `_hdg_turns_last_ms`（本门上一次转向结束时刻），与"多久没测量"无关。"""
        if self._hdg_deg is None or self._hdg_ms is None:
            return False
        if self._hdg_turns_last_ms is None:
            return True                      # 本门还没转过 ⇒ 任何 ψ 都算"新"
        return self._hdg_ms > self._hdg_turns_last_ms

    def _hdg_skip_reason(self, mode, now_ms):
        """居中达标却没进正航向的**原因**（水里复盘看 `hdg_skip` 这个字段）。"""
        if not flag(self._hdg_cfg, "enable", True) or not self._hdg.enabled:
            return "off(未启用)"
        if self._hdg_done:
            return "done(本门正航向已收口：够正/中止/未启用)"
        if self._hdg_deg is None or self._hdg_ms is None:
            return "no_psi(还没测到过 full 帧的 psi)"
        if not self._hdg_new_psi_since_turn():
            return "wait_psi(等转向之后的新 ψ 测量)"
        return "psi_ok(其实可以起转 —— 若出现说明是居中没达标)"

    def _note_hdg_skip(self, why):
        """居中达标却跳过正航向 → 写进日志字段 + 终端报一次（每门一次）。"""
        self.last_info["hdg_skip"] = why
        if self._hdg_skip_logged:
            return
        self._hdg_skip_logged = True
        print("[GATE] ⚠️ 居中达标但跳过正航向：%s ⇒ 带残余航向直接进近（这一门不会再转）" % why)

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

    def _turn_blocking(self, now_ms, target_deg=None, left=None):
        """★ **唯一的阻塞路径**（`turn` 专用）：**输入目标角度，输出"阶段结束"标志**。
        不传（每帧续跑）就沿用 aligner 里已冻结的目标角。
        · **输出**：`True = 这次转向阶段已结束`（本门不会再转，调用方可继续往下走）；
        `False = 尚未结束`（本帧先回主循环，下一帧从同一个冻结目标角续跑）。
        · 全程只看「冻结的目标角 + 遥测 yaw」：画面丢失、门转出视野、档位退化、位姿被拒
        **都不打断**；唯一中止来自 `TurnCore`（缺遥测 / 方向自证 / 超时）。
        · 收尾（`_hdg_done` / 回 GOLDEN / 挂"转完反向平移"）**只做一次**。"""
        import time as _time
        _t0 = _time.monotonic()
        if target_deg is not None and self._hdg.state == "idle":
            self._hdg.start(now_ms, psi=float(target_deg))
        # ② 闭环推进：本帧一步 +（真实时钟域下）阻塞到终态
        yaw_cmd = 0.0
        if self.substate == SUB_HDG:
            _st, yaw_cmd = self._hdg.step(now_ms, yaw_telemetry=self._uart_yaw(),
                                          gate_lost=False)
            yaw_cmd = self._turn_inner_loop(yaw_cmd, now_ms)
        ended = bool(self._hdg.finished())
        # ③ 转向调用日志（每帧一行；ms 很大 = 这一帧被这次转向占住了）
        turn_log("gate_loop", src="gate", frame=self.frames, t_ms=now_ms,
                 state=self._hdg.state, yaw=yaw_cmd, tyaw=self._uart_yaw(),
                 tgt=self._hdg.last_target_deg, dir=self._hdg.last_dir,
                 iters=self._hdg.iters, done=ended, ended=ended,
                 ms=(_time.monotonic() - _t0) * 1000.0)
        # ④ 收尾（只做一次）
        if ended and self.substate == SUB_HDG:
            #   然后等"转向之后的新 ψ 测量"；若残余仍 > tol 就自动再走一小步（见 `_hdg_ready`）。
            self._hdg_turns += 1
            self.substate = SUB_GOLDEN
            print("[GATE] 本门第 %d 次转向结束（等新的 ψ 测量决定要不要下一小步）"
                  % self._hdg_turns)
            self._center_cnt = 0          # 转向会动到门在画面里的位置 → 之后复核居中
            #   （84→128 只到 ~95 = 仍在转）；这时一旦断流/关串口，船就**锁在那个值上一直转**。
            #   开关：`comm.gate.hdg.stop_hard`（默认 true；嫌慢可设 false，但风险自负）。
            # 硬停完成后拿到**折算过的帧时基**，平移窗口从那一刻起算（见 `_stop_hard_framed`）。
            _now_stop, _stop_ok = self._stop_hard_framed(now_ms, quiet=not S.DEBUG)
            if _stop_ok is False:
                print("[GATE] ⚠️ 转向收尾硬停**未确认**（遥测显示仍在转）—— 平移窗口照开，注意船姿态")
            #   两者在同一帧里先后阻塞、中间**没有任何一帧**（`_t0` → `_turn_inner_loop` 20Hz
            #   阻塞闭环 1~3s → `_stop_hard_framed` 再阻塞 0.7~2.5s → 本帧才收尾）。
            #   只折算硬停 ⇒ 窗口起点仍**落后"这次转向的耗时"**，而窗口只有 `post_sway_ms`(600ms)
            #   ≪ 转向耗时 ⇒ **窗口一出世就已过期**，生产里照旧只发得出 1 帧平移。
            #   （用例 `test_post_sway_starts_after_the_turn_end_hard_stop` 同时模拟"转向阻塞"与
            #     "硬停阻塞"，只折算一半就会红。）
            _now_true = int(now_ms + (_time.monotonic() - _t0) * 1000.0)
            self._now_ms = _now_true      # 本帧后续（`_set_info` 的窗口判据 / z 新鲜度）都用真实时刻
            self._hdg_turns_last_ms = _now_true   # 判"有没有新的 ψ"的基准（必须晚于它）
            self._start_post_sway(_now_true)
        self.last_info["hdg_i"] = self._hdg.iters
        self.last_info["turn_end"] = ended        # ★ 阶段结束标志（消费方/日志都看得到）
        self._set_info("hdg" if not ended else "center",
                       mode=self.mode or "", z=self._z_last, dx=0.0, dy=0.0,
                       sway=0.0, heave=0.0, surge=0.0, yaw=yaw_cmd,
                       kpt=self._dbg_kpt, hdg=self._hdg_deg,
                       hdg_state=self._hdg.state)
        return ended

    def _stop_hard_framed(self, now_ms, quiet=False):
        """转向收尾**硬停** + **帧时钟域显式折算**（与 `_turn_inner_loop` 同一套做法）。
        `base/hw/uart.py::stop_hard` 是**帧内阻塞**的，它自己按 `dt=0.05`（20Hz）连发中性帧、
        （用例 `test_post_sway_starts_after_the_turn_end_hard_stop` 咬的就是这个次序）；
        Returns:
        "平移窗口从整块阻塞结束起算"由 `_turn_blocking` 用整帧流逝时间再折算一次（见那里）。"""
        import time as _time
        if not flag(sub(self._G, "hdg"), "stop_hard", True):
            return now_ms, None
        t0 = _time.monotonic()
        ok = True
        try:
            _sh = getattr(self.uart, "stop_hard", None)
            if callable(_sh):
                ok = _sh(quiet=quiet)
            else:
                for _ in range(14):              # 旧对象：自己连发中性帧走完 ramp
                    self.uart.neutral()
        except Exception as e:
            ok = False
            print("[GATE] 转向收尾硬停失败：%s" % e)
        dt_ms = (_time.monotonic() - t0) * 1000.0
        turn_log("gate_stop_hard", src="gate", frame=self.frames, t_ms=now_ms,
                 ms=dt_ms, now_after=int(now_ms + dt_ms), ok=ok)
        return int(now_ms + dt_ms), ok

    def _start_post_sway(self, now_ms):
        """转向结束 → **进入「转完反向平移」状态**（用户 2026-09-27 定；主循环里的一个状态）。
        反方向**平移一小段把它拉回视野；一旦重新检出到"刚丢的那个门"就立刻交回视觉。
        检测、PnP、ψ 更新、居中确认、丢门处理一律照跑。"""
        if self._post_sway_until_ms is not None or not self._hdg.last_d:
            return
        win = float(self._hdg_cfg.get("post_sway_ms", 600.0) or 0.0)
        mag = float(self._hdg_cfg.get("post_sway", 0.20) or 0.0)
        if win <= 0 or mag <= 0:
            return
        back = float(self._hdg_cfg.get("post_sway_back", 0.0) or 0.0)
        self._post_sway = _dof_clip(-self._hdg.last_d * mag)     # 反向平移
        self._post_sway_back = _dof_clip(-abs(back))             # ★ 同时**缓慢后退**（2026-09-28 定）
        self._post_sway_until_ms = now_ms + win
        # 参照量：判"这帧的门是不是刚转走的那个"（见 _post_sway_same_gate）。
        #    （这期间不测位姿）⇒ 加了新鲜度必然为 None、z 闸形同虚设。新鲜度只对**当帧**的 z 有意义。
        self._post_sway_z_ref = None if self._z_last is None else float(self._z_last)
        self._post_sway_ratio_ref = self._dbg_ratio if self._dbg_ratio > 1e-6 else None

    def _fresh_z(self, now_ms):
        """当帧可用的新鲜 z（受 `gate.z.z_stale_ms` 限制）；不新鲜 ⇒ None。"""
        if self._z_last is None or self._z_ms is None:
            return None
        stale = num(sub(self._G, "z"), "z_stale_ms", _D_Z["z_stale_ms"])
        if not stale or stale <= 0 or (now_ms - self._z_ms) > stale:
            return None
        return float(self._z_last)

    def _post_sway_same_gate(self, det, z=None):
        """「反向平移」状态里，本帧的检出算不算**刚被转走的那个门**？
        **② z 拿不到就退回框占比**（用户原话：如果 z 一直被"新鲜帧"卡着，就相信框占比）：
        z 拿不到 = 当帧 coarse（无位姿）/ 唯一那次位姿已超 `gate.z.z_stale_ms` / 转前根本没测到过 z。"""
        if det is None:
            # 本帧没有检出 ⇒ 绝不是"刚丢的那个门回来了"。
            return False
        jump = num(sub(self._G, "hdg"), "post_sway_same_z_m",
                   _D_SWAY_BACK["post_sway_same_z_m"]) or 0.0
        far = num(sub(self._G, "hdg"), "post_sway_far_ratio",
                  _D_SWAY_BACK["post_sway_far_ratio"]) or 0.0
        if jump <= 0 and far <= 0:
            return True                      # 两道闸都关 ⇒ 任何检出都算"门回来了"
        # ① 距离优先
        if jump > 0 and z is not None and self._post_sway_z_ref is not None:
            return abs(float(z) - float(self._post_sway_z_ref)) <= float(jump)
        # ② z 拿不到 ⇒ 退框占比（不得小于原 70%）
        if far > 0 and self._post_sway_ratio_ref and self.w:
            return (float(det.w) / float(self.w)) >= float(far) * float(self._post_sway_ratio_ref)
        return False                         # 没有任何可比信息 ⇒ 继续推（到期兜底）

    def _turn_inner_loop(self, yaw_cmd, now_ms=0):
        """转向期间按 `comm.motion.turn_pid.period`（默认 0.05s=20Hz）推进 HDG 到终态。
        同「进带 3 帧收舵」。返回最后一次 yaw 指令。只在**真实时钟域**阻塞推进（用例的假时钟"""
        import time as _time
        # 缺配置也要能跑（用例里 motion 段可能没有 turn_pid）
        try:
            period = float(num(motion_node("turn_pid"), "period", 0.05) or 0.05)
        except Exception:
            period = 0.05
        period = max(0.01, min(0.2, period))
        # 兜底：单个转向最多推进 turn_timeout_s + 1s（TurnCore 自身也有超时）
        #   基准取传入的 now_ms —— 绝不能把 time.monotonic() 的绝对值喂进 HDG，否则时间跳变
        t_wall0 = _time.monotonic()
        budget = float(self._hdg_cfg.get("turn_timeout_s", 8.0)) + 1.0
        # 时钟域自适应：now_ms 与真实时钟同域（运行期）→ 真按 20Hz 睡；
        #   不同域（用例的假时钟）→ **不真睡**，改用合成周期推进，测试才不会被拖成真时间。
        # 时钟域判据：`now_ms` 来自**真实时钟**就算真域 —— 生产是 `main.py` 的
        # `int(time.time()*1000)`（epoch 毫秒，板端 ~1.79e12），用例里也可能是
        # `time.monotonic()*1000`（开机毫秒）。**两个都要认**；假时钟（用例 1000+100i）两者都远。
        #    ⇒ 内层闭环**整个被跳过**（转向退化成"每相机帧一步"、`gate_loop` 日志永不触发、
        _wall = abs(now_ms - _time.time() * 1000.0)
        _mono = abs(now_ms - _time.monotonic() * 1000.0)
        use_wall = min(_wall, _mono) < 60000.0
        if not use_wall:
            # 假时钟（无硬件用例：假船按"每帧一步"积分航向）⇒ 保持旧的每帧一步，
            # 高频内层循环是**运行期**行为，只在真实时钟域生效（不引入任何新逻辑）。
            return yaw_cmd
        n = 0
        while (not self._hdg.finished()) and (_time.monotonic() - t_wall0) < budget:
            n += 1
            #   没有它，一次不收敛的转向会把主循环占满 `turn_timeout_s+1`≈9s，操作员插不进来。
            if getattr(self.uart, "estop_active", False):
                self._hdg.abort("estop(Ctrl-C)")
                break
            if use_wall:
                _time.sleep(period)
                now2 = int(now_ms + (_time.monotonic() - t_wall0) * 1000.0)
            else:
                now2 = int(now_ms + n * period * 1000.0)
            st_h, yaw_cmd = self._hdg.step(
                now2, yaw_telemetry=self._uart_yaw(), gate_lost=False)
            self.last_info["hdg_i"] = self._hdg.iters
            self._set_info("hdg" if not self._hdg.finished() else "center",
                           mode=self.mode, z=self._z_last, dx=0.0, dy=0.0,
                           sway=0.0, heave=0.0, surge=0.0, yaw=yaw_cmd,
                           kpt=self._dbg_kpt, hdg=self._hdg_deg,
                           hdg_state=self._hdg.state)
        return yaw_cmd

    def _hdg_ready(self, mode, now_ms):
        """是否可以（或必须）进正航向：启用 + 本门还没做 + **测到过 full 帧的 psi**。
        新模型下 full 只占约 27%，死等 full 那一刻等于永远不转。"""
        if self._hdg_done or not self._hdg.enabled:
            return False
        # 每步幅度由 max_step_deg/turn_scale 限；够正时 start() 自己给 DONE（不转）。
        return self._hdg_new_psi_since_turn()

    def _hdg_psi_first(self, now_ms):
        """★ **航向优先**入口（2026-09-26 用户定）：居中还没达标就先转正。
        最后靠出口②在 psi=+30.6°、dy=0.42 时满速冲出去。
        触发条件（全部满足）：
        · 本门还没做过正航向、HDG 未在跑、`hdg.enable`；
        · **|psi| > `hdg.tol_deg`**（航向已经够正 → 不抢，交给常规「先居中」路径）；
        （新模型下 z 常常不可用/是垃圾值，占比是全档位都有的量）。"""
        if self._hdg_done or not self._hdg.enabled:
            return False
        if not self._hdg_new_psi_since_turn():
            return False                      # 还没等到"转向之后的新 ψ" ⇒ 不抢
        if self.substate == SUB_HDG and not self._hdg.finished():
            return False                      # 已经在跑 → 交给常规路径
        if not flag(self._hdg_cfg, "entry_psi_first", True):
            return False
        if self._hdg_deg is None:
            return False
        if abs(self._hdg_deg) <= float(self._hdg_cfg.get("tol_deg", 8.0)):
            return False
        zmax = float(self._hdg_cfg.get("psi_first_z_max", 2.5) or 0.0)
        zr = float(self._hdg_cfg.get("psi_first_ratio", 0.30) or 0.0)
        stale = num(sub(self._G, "z"), "z_stale_ms", _D_Z["z_stale_ms"])
        z_fresh = (self._z_ms is not None and self._z_last is not None and
                   stale > 0 and (now_ms - self._z_ms) <= stale)
        if z_fresh and zmax > 0 and self._z_last <= zmax:
            return True
        return zr > 0 and float(self._dbg_ratio or 0.0) >= zr

    def _hdg_start(self, now_ms, reason):
        """起转（两条入口共用）：置子状态 + 把**当前 psi 冻结成目标角**交给 aligner。"""
        self.substate = SUB_HDG
        self.last_info["hdg_skip"] = None
        self.last_info["hdg_entry"] = reason   # golden=居中达标 | psi_first=航向优先
        turn_log("gate_start", src="gate", frame=self.frames, t_ms=now_ms, reason=reason,
                 psi=self._hdg_deg, psi_ms=self._hdg_ms, n_turn=self._hdg_turns + 1)
        if self._hdg.finished():
            # 逐小步逼近的下一小步：aligner 还在终态 ⇒ 先退回 IDLE 才能再 start 一次
            self._hdg.rearm()
        self._hdg.start(now_ms, psi=self._hdg_deg)
        turn_log("gate_start_done", src="gate", frame=self.frames,
                 state=self._hdg.state, dir=self._hdg.last_dir,
                 tgt=self._hdg.last_target_deg, iters=self._hdg.iters)
        if not self._hdg.finished():
            #   阻塞发生在本帧内（"自包含动作"）：主循环/相机/日志在这段时间里不进新帧。
            self._turn_blocking(now_ms, target_deg=self._hdg.psi_meas,
                                left=(self._hdg.last_d < 0))
        if self._hdg.finished():
            # 已经够正（|psi| ≤ tol）或没得转 ⇒ 本门不再正航向。
            #    不同步就会在 ALIGN 里反复起转（死循环）。
            self._hdg_done = True
            self.substate = SUB_GOLDEN
        self.last_info["hdg_i"] = self._hdg.iters
        return self._hdg.state

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
        # ---- 「转完反向平移」状态：主循环每帧在这里推进（`_set_info` 是全任务**唯一**下发口）----
        # 只覆盖"本帧发什么"：检测/PnP/ψ/居中判据在调用本函数之前**已经跑完**（ψ 因此是新鲜的）。
        if self._post_sway_until_ms is not None:
            _now = self._now_ms if self._now_ms is not None else 0
            # 「画面里至少出现 N 个原始角点」就退出（N 可设；0 = 关掉这条判据）
            _kmin = int(num(sub(self._G, "hdg"), "post_sway_kpt_min",
                            _D_HDG["post_sway_kpt_min"]) or 0)
            #   本帧没有检出（门还在画外）= 0 个角点 ⇒ 继续平移 + 后退。
            _kraw = 0
            if self._det_now is not None and getattr(self._det_now, "kpt_conf", None) is not None:
                _kthr = num(sub(self._V, "keypoint"), "conf_thr", _D_KPT["conf_thr"])
                _kraw = int((np.asarray(self._det_now.kpt_conf) >= _kthr).sum())
            if _now >= self._post_sway_until_ms:
                self._post_sway_until_ms = None         # 到期 ⇒ 退出状态（绝不永久停摆）
                self.last_info["sway_exit"] = "expire"
            elif action == "through" or self._hdg.turning:
                #   ① 旧写法是 `elif action != "through" and 同门判据` —— `through` 会让这个 elif
                #      为假、**掉进 else**，被改写成 sway_back（surge/yaw 清零）最长一个窗口 ⇒ 冲刺被压住。
                #   ② 转向侧同理：`_turn_inner_loop` 每 20Hz（真实时钟域，帧内阻塞）经 `_set_info` 发 yaw，
                #      只要窗口还活着就会把 yaw 清零 ⇒ 转向**推不动**（不是永久死锁，但白等/触发满舵保护）。
                #   现在：**平移永远让位于转向与冲刺**（平移只是"把门拉回视野"的补偿，优先级最低）。
                self._post_sway_until_ms = None
                self.last_info["sway_exit"] = "yield(%s)" % ("through" if action == "through" else "turning")
            elif _kmin > 0 and _kraw >= _kmin:
                #   用**原始**角点数（检测器当帧给的），不是记忆后/几何过滤后的个数。
                self._post_sway_until_ms = None
                self.last_info["sway_exit"] = "kpt>=%d" % _kmin
            elif self._post_sway_same_gate(self._det_now, self._fresh_z(_now)):
                self._post_sway_until_ms = None         # 刚丢的那个门回来了 ⇒ 退出，交回视觉
                self.last_info["sway_exit"] = "same_gate"
            else:
                action, sway, heave, surge, yaw = ("sway_back", self._post_sway, 0.0,
                                                   self._post_sway_back, 0.0)
                substate = SUB_SWAY_BACK
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
            "hdg": None if hdg is None else round(float(hdg), 1),
            # 转向排查用：`img_sign`=本次转向用的极性 σ（turn_deg 里定死的乘式）、
            # `dir`/`tgt`=本次转向方向与目标角、`hdg_entry`=两条入口（golden/psi_first）。
            "img_sign": self._hdg.imag_sign,
            "hdg_dir": self._hdg.last_dir,
            "hdg_tgt": round(float(self._hdg.last_target_deg or 0.0), 1)})
        self.uart.send_dof(_dof_clip(surge), _dof_clip(sway),
                           _dof_clip(heave), _dof_clip(yaw))

    def _start_through(self, approach_near_lost=False, near_lost=False):
        """进入 THROUGH：只前进、不微调（横向修正全部留在冲刺前完成）。
        速度 = `comm.gate.surge.through`（三个出口都走这一条，见 `_through_speed`）。
        APPROACH near-loss bypasses the heading guard; near-loss entries use
        through.near_lost_confirm_ms, independently of normal confirm_ms.
        · 从没测到过（None）→ **放行**（没得判）；
        · |psi| ≤ 阈值 → 放行；
        Returns:
        True = 真的进了 THROUGH。**三个出口都必须看返回值**，别无条件当成功。"""
        need = num(sub(self._G, "through"), "require_align_deg",
                   _D_THROUGH["require_align_deg"])
        # Only APPROACH near-loss may bypass the heading guard.
        bypass_heading = approach_near_lost and self.phase == PH_APPROACH
        if not bypass_heading and need and need > 0 and self._hdg_deg is not None and abs(self._hdg_deg) > need:
            self.last_info["through_block"] = "hdg=%+.1f>%.1f" % (self._hdg_deg, need)
            if not self._through_block_logged:
                self._through_block_logged = True
                print("[GATE] 拦下冲刺：最近一次 psi=%+.1f° > %.1f°"
                      " → 留在原地继续对准/正航向（不切 THROUGH）"
                      % (self._hdg_deg, need))
            return False
        self.last_info["through_block"] = None
        self._hdg_abort("进入冲刺")
        self._through_near_lost = bool(near_lost or approach_near_lost)
        self.phase = PH_THROUGH
        self.substate = ""
        self._through_frames = 0
        self._lost_cnt = 0
        self._z_guard = False
        # Near-loss duration starts at the first THROUGH command.
        self._through_start_ms = getattr(self, "_now_ms", None) if self._through_near_lost else None
        self._loiter_start_ms = None      # 已进入冲刺 → 门口的计时作废
        self._reset_lateral_pids()
        return True

    def _through_speed(self):
        """本次冲刺速度 = `comm.gate.surge.through`。"""
        return num(sub(self._G, "surge"), "through", _D_SURGE["through"])

    def _speed(self, tier):
        """进近速度档：`gate.surge.<tier>` 显式覆盖 → `motion.surge_<tier>`（**与撞球共用**）→ 代码兜底。"""
        return num(sub(self._G, "surge"), tier, motion_num("surge_" + tier))

    def _start_search(self):
        self.phase = PH_SEARCH
        self.substate = ""
        self._post_sway_until_ms = None      # 换门/重新搜索 → 反向平移状态作废
        self._post_sway_z_ref = None
        self._post_sway_ratio_ref = None
        self._hdg_f = None
        self._hdg_deg = None
        self._hdg_ms = None
        self._hdg_turns = 0               # 新门 ⇒ 逐小步逼近的次数重新计
        self._hdg_turns_last_ms = None
        self._hdg.reset()
        self._hdg_done = not flag(self._hdg_cfg, "enable", True)
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
        """SEARCH：左右平移扫视，**每轮时长按 2 的次幂递增**（用户 2026-09-27 定）。
        一轮（k=0,1,2,…）的波形：
        段3 余下               → 0（停）
        方向约定与 sway 一致：**+ = 右移**。`sweep_s<=0` → 只停不扫（等于原地待机）。"""
        s = merge(sub(self._G, "search"), _D_SEARCH)
        base = num(s, "sweep_s", _D_SEARCH["sweep_s"]) * 1000.0
        pause = max(0.0, num(s, "pause_s", _D_SEARCH["pause_s"]) * 1000.0)
        if base <= 0:
            return 0.0
        cap = max(base, num(s, "sweep_max_s", _D_SEARCH["sweep_max_s"]) * 1000.0)
        v = num(s, "sway", _D_SEARCH["sway"])
        if self._search_entry_ms is None:
            self._search_entry_ms = now_ms
        t = now_ms - self._search_entry_ms
        k, t0, sweep = 0, 0.0, base          # 先定位当前落在第几轮（每轮时长翻倍）
        while k < 24:
            cyc = 2.0 * (sweep + pause)
            if t < t0 + cyc:
                break
            t0 += cyc
            k += 1
            sweep = min(base * (2.0 ** k), cap)
        ph = t - t0
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
        self._now_ms = now_ms        # 本帧时基：`_set_info` 里推进「反向平移」状态要用
        # 本帧**没有跑视觉**时（转向期间 / 还没走到检测）不许沿用上一帧的门：
        #   否则"转完那一帧"会拿着上一帧的门当作"门还在画面里"，反向平移状态刚进就退。
        self._det_now = None
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

        if (self._hdg.turning and not self._hdg.finished()):
            #   转向期间**连检测都不做**（画面丢了/门转出视野/水花糊了都不影响它），
            #   按 `turn_pid.period` 跑完再回主循环。
            self.last_dets = []            # 本帧没有视觉结果 → 叠加层不许画上一帧的框
            self._turn_blocking(now_ms)
        else:
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
        """选目标门：按 `vision.gate.select.mode`（规范 `doc/设计/gate_pose_decode_spec.md` §4）。
        用 **框宽**当测距代理（`z ≈ fx·W/w`，单调）；并列再比可信角点数 / 置信和 / score。
        "水面倒影可能形成第二个门框，只按 score 选会选到倒影"。要回退就是把 cfg 改成一行。
        去重（同一个门被两个尺度各检出一次）与四角几何合法性**不在这里** ——"""
        conf_thr = num(sub(self._V, "keypoint"), "conf_thr", _D_KPT["conf_thr"])
        return postproc_pick(dets, conf_thr=conf_thr, cfg=self._select_cfg)

    # ------------------------------------------------------------ 单帧逻辑
    def _step(self, det, now_ms):
        self._det_now = det          # 本帧的门（「反向平移」状态判"门回来了吗"用当帧数据）
        # ---- 正航向的「转」：**一个自包含动作**，跑完才回来 ----
        # 只要转入已经开始，本帧就只按「遥测 yaw + 冻结的目标角」把它推完/推进一步：
        # 不看画面、不等检测、**丢门也不中止**（转 50° 时门必然转出视野，这是正常现象）。
        if self._hdg.turning and not self._hdg.finished():
            self._turn_blocking(now_ms)
            return
        if self.phase == PH_THROUGH:
            self._tick_through(now_ms)      # 穿门=直行，不做横向微调
            return
        if det is None:
            self._tick_lost(now_ms)
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

        #   放在档位分派之前 ⇒ 任何档位都能起转（新模型下 full 只占 27%，死等 full 等于永远不转）。
        if self.phase == PH_ALIGN and self._hdg_psi_first(now_ms):
            self._hdg_start(now_ms, "psi_first")
            self._set_info("hdg", mode=mode, z=self._z_last, dx=0.0, dy=0.0,
                           sway=0.0, heave=0.0, surge=0.0, yaw=0.0,
                           kpt=self._dbg_kpt, hdg=self._hdg_deg,
                           hdg_state=self._hdg.state)
            return

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
        而且米制/像素两套阈值+两套 PID 会在 mode 于 full↔coarse 间跳时交替工作 → 收敛不了。
        统一成像素后：**一套阈值(align.px_x/px_y) + 一套 PID**，全档位可比。
        门框中心 = 用位姿把门原点(0,0,0)投影回图像（p3p 缺角时也比角点均值准）。"""
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
            if self._cross_cnt >= max(1, need) and self._start_through():
                # 冲刺：只前进、不带横向微调（微调已在上一帧做完）
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
            # ① 正航向（SUB_HDG）：**转入一旦开始就由 `_step` 顶部接管**（自包含动作，
            #    不看画面）；这里只处理"已经在 ALIGN、需要决定要不要起转"的正常路径。
            self.substate = SUB_GOLDEN
            if self._loiter_commit(dxn, dyn, float(det.w) / float(self.w), now_ms):
                return
            if aligned:
                self._center_cnt += 1
                if self._center_cnt >= int(num(al, "confirm_frames", _D_ALIGN["confirm_frames"])):
                    if self._hdg_ready(mode, now_ms):
                        # 居中达标 → **正航向**（PnP 目标角 → turn_deg 转一次 → 结束）
                        # （_hdg_start 会清掉「跳过」标记，免得日志误读）
                        self._hdg_start(now_ms, "golden")
                        self._set_info("hdg", mode=mode, z=z, dx=dx_m, dy=dy_m,
                                       sway=0.0, heave=0.0, yaw=0.0, kpt=kpt,
                                       hdg=self._hdg_deg, hdg_state=self._hdg.state)
                        return
                    self._note_hdg_skip(self._hdg_skip_reason(mode, now_ms))
                    self.phase = PH_APPROACH
                    self.substate = ""
            else:
                self._center_cnt = 0
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
    def _tick_hdg_degraded(self, now_ms):
        """档位退化（width/coarse）但 HDG 正在跑 → **只推进 HDG**，别让 CREEP/HOLD 冲掉子状态。
        Returns: True = 本帧已被 HDG 接管。"""
        if not (self.phase == PH_ALIGN and self.substate == SUB_HDG
                and not self._hdg.finished()):
            return False
        _st, yaw_cmd = self._hdg.step(now_ms, yaw_telemetry=self._uart_yaw(),
                                      gate_lost=False)
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
                       kpt=self._dbg_kpt, hdg=self._hdg_deg,
                       hdg_state=self._hdg.state)
        return True

    def _on_width(self, det, now_ms, ids):
        """width 档：只有对向 2 角（上边或下边），信息只够"水平中点 + 框心竖直"，不解 PnP。
        ① z > width.z_max（还远）且对准 → 慢 creep 靠近（换取角点/整门）；
        出口仍然只有 `_tick_lost`（近距丢失）与 `_loiter_commit`（门口超时）两条。
        z 由 fx·W/Δu 粗估，只用来判"该不该靠近"，不参与闭环。"""
        if self._tick_hdg_degraded(now_ms):
            return
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
        ① 对准 → 慢 creep 靠近（争取露出角点）；
        ② 未对准：远距只对中；中距 HOLD 超限 → REACQUIRE；很近(框装不下) → REACQUIRE。
        出口同样只有 `_loiter_commit` 与 `_tick_lost` 两条。"""
        if self._tick_hdg_degraded(now_ms):
            return
        G = self._G
        C = merge(sub(G, "coarse"), _D_COARSE)
        H = merge(sub(G, "hold"), _D_HOLD)
        sg = merge(sub(G, "surge"), _D_SURGE)
        self.mode = MODE_COARSE
        # coarse 档**没有任何测距**：`_z_last` 会一直是上次 width/位姿留下的陈旧值
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
        surge = -num(sg, "reacquire", _D_SURGE["reacquire"])
        self._set_info("reacquire", substate=SUB_REACQUIRE, surge=surge,
                       kpt=self._dbg_kpt)

    # ---------------- 丢目标 / 穿门 ----------------
    def _loiter_commit(self, dxn, dyn, ratio, now_ms, kpt=None):
        """在门口超时兜底：**不看档位**，只要「人在门口 + 对准」持续太久就自己拍板直冲。
        三个条件都成立才算命中（命中 → `_start_through()` 并返回 True）：
        · 框占比 ≥ `z.near_lost_ratio`（与"丢门判过门"同一个"在门口"定义）
        · |dxn| ≤ `loiter.dx_max`、|dyn| ≤ `loiter.dy_max`（安全带，归一化像素偏差）
        · 上述状态**连续** ≥ `loiter.timeout_ms`（任一条不成立就重新计时）"""
        L = merge(sub(self._G, "loiter"), _D_LOITER)
        if not flag(L, "enable", True):
            self._loiter_start_ms = None
            return False
        near_r = num(L, "near_ratio", num(sub(self._G, "z"), "near_lost_ratio", _D_Z["near_lost_ratio"]))
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
        if not self._start_through():
            # 航向闸门拦下：**不算 commit**（返回 False 让调用方继续原逻辑），
            # 门口的计时保留，下一帧会再评估一次。
            return False
        self._set_info("through", z=self._z_last, dx=dxn, dy=dyn,
                       surge=self._through_speed(), kpt=self._dbg_kpt if kpt is None else kpt,
                       ratio=ratio)
        return True

    def _near_lost(self, now_ms, zc):
        """整门丢失时：判断是不是**已经到门口了**（≈机身已进门框）。两条判据取 OR：
        ① `z ≤ z.near_lost_m` **且 z 新鲜**：coarse 档没有测距、时间戳被清空，
        ② 末次检测到的**框占比 ≥ z.near_lost_ratio**：全档位可用、每帧都算，
        门快装不下/机身进门框时必然成立。
        返回命中的判据名（空串 = 没到门口）。"""
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
            #   画面丢失/门转出视野都由它兜住）。能走到这一行的，转向一定已经结束。
            # ---- 转完补偿：刚转完那一下、又看不见门 → 朝转向反方向平移一小段（把门拉回视野）----
            if self._post_sway_until_ms is not None and now_ms < self._post_sway_until_ms:
                self._set_info("sway_back", z=self._z_last, sway=self._post_sway)
                return
            # → SEARCH → 原地不动"，永远数不到过门）。判据用**最后一次检测到的框占比**
            # （每帧更新、比 z 更可靠且全档位可用）；远处丢检（占比小）仍走 SEARCH。
            why = self._near_lost(now_ms, zc)
            if why and self._start_through(near_lost=True):
                if S.DEBUG:
                    print("[GATE] ALIGN 近距丢失(%s) → 判过门" % why)
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
            if why and self._start_through(approach_near_lost=True):
                # 已到近距却整门丢失：门占满视野/机身进入门框 = 真过门的典型现象
                # → 直接判过门并直行穿越（不等防抖、不后退），否则永远数不到过门。
                if S.DEBUG:
                    print("[GATE] 近距丢失(%s) → 判过门" % why)
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
        结束条件优先**时长** `through.confirm_ms`（帧数语义下冲刺时长随 fps 漂，而"能不能"""
        G = self._G
        T = merge(sub(G, "through"), _D_THROUGH)
        self._through_frames += 1
        self._lost_cnt += 1               # confirm_ms<=0 时按帧数兜底
        if self._through_start_ms is None:
            self._through_start_ms = now_ms
        ms = num(T, "confirm_ms", _D_THROUGH["confirm_ms"])
        if getattr(self, "_through_near_lost", False):
            ms = num(T, "near_lost_confirm_ms", _D_THROUGH["near_lost_confirm_ms"])
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
