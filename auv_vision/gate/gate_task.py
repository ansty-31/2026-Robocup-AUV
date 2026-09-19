# -*- coding: utf-8 -*-
"""gate/gate_task.py — 任务三 过门（GateTask，§5.3 相位机）

门 = 对称平面矩形门框(0.70×0.50 m)，悬空，无朝向要求；任务 = 机身穿过开口。

相位机：SEARCH → RANGE_ALIGN → APPROACH → THROUGH(计数) → (下一门)/DONE
RANGE_ALIGN 子状态（按前端 mode 与门框占屏比仲裁）：
  GOLDEN   full/p3p 位姿对准（sway/heave PID，无 surge）
  CREEP    远距小框（coarse far / width）：慢速 creep + 框心对中
  HOLD     中/近角不足：surge=0 保持对中；≥hold.max_frames → REACQUIRE
  REACQUIRE 短后退重取整门（拉距→4 角重现）；超限 → SEARCH
穿过/直冲出口（任一成立即 THROUGH，忽略角丢失直行、不做横向微调）：
  ① 位姿可信且 Z ≤ z.cross 连续 cross_confirm_frames 帧（**实船已验证，勿动**）；
  ② 进近中已到近距(Z ≤ z.near_lost_m)后整门丢失（门占满视野/机身入门框）；
  ③ **width 档**：对准确认 + 偏差在安全带内 + Z ≤ width.z_max → 直冲
     （只 2 对向角时靠"两角中点 + 框心"对中；测距只用来判"该不该冲"，不参与闭环）；
  ④ **coarse 档**：对准确认 + 门框占比 ≥ coarse.dash_ratio（够近）→ 直冲；
     未对准才 HOLD/后退（**不再"一看框大就后退"**，那是实测里来回退的主因）。
  ③④ 是"低帧率 + 机身抖动"下的懒人策略：对准好就冲、由下位机 PID 保姿态，
  不叠加多帧高精度闭环（约束越多越容易被抖动打断）。
  ⚠️ 直冲速度 width/coarse 各自可配且低于 surge.through —— 撞杆与否主要由
     **冲前是否对中**与**冲刺速度**决定，两者都要水池实测标定。
位姿跳变保护：pnp.max_z_jump_m 内的帧间变化才可信，突变帧弃帧退化 coarse，
防平面 PnP 错解(尤其偶发 z≤cross)把船直接推出去。

依赖：gate.geometry（位姿）、gate.gate_frontend（mode）、common.PID。
参数：cfg/vision.yaml gate.* 与 cfg/comm.yaml gate.*。
所有 cfg 读取都走 _sub/_cfg/_num 兜底：**配置缺键时用"等于当前 cfg 的默认值"**，
不会 AttributeError（允许 AUV_CFG_DIR 指向缺新键的旧配置）。
"""
from __future__ import annotations

import numpy as np

import base.settings as S
from common.PID import PID
from gate.gate_detector import board_camera
from gate.kpt_memory import ENV_ENABLE, build_kpt_memory, enable_source
from gate.gate_frontend import (parse_kpt_mode, width_range_depth, bbox_center,
                                MODE_FULL, MODE_P3P, MODE_WIDTH, MODE_COARSE)
from gate.geometry import (object_points, gate_pose, gate_normal_angles_deg,
                            GATE_FRAME_W, GATE_FRAME_H)

PH_SEARCH = "SEARCH"
PH_ALIGN = "ALIGN"
PH_APPROACH = "APPROACH"
PH_THROUGH = "THROUGH"

SUB_GOLDEN = "GOLDEN"
SUB_CREEP = "CREEP"
SUB_HOLD = "HOLD"
SUB_REACQUIRE = "REACQUIRE"

# ---------------------------------------------------------------- 缺键兜底
# 下面这些默认值**等于当前 cfg/comm.yaml 的取值** → 配置缺键时行为与现在一致。
# 增益一律沿用**撞球那边已标定的值**（同一套推进器/图像链路，量纲都是"归一化偏差 ±1"）
#   平移 sway ← `comm.ball.approach_pid`（接近段水平修正，实测调好）
#   升降 heave ← `comm.ball.pid`（居中垂直）
#   转向 yaw  ← `comm.ball.edge_yaw_*`（居中转向：kp0.3 / kd0 / 限幅0.15 / deadzone 0）
# 下面这三个**只是"连 ball 都取不到"时的最后兜底**（正常配置永远走 comm.ball）
_D_PID_SWAY = dict(kp=8.0, ki=0.01, kd=0.05, out_max=0.45, deadzone=0.05)
_D_PID_HEAVE = dict(kp=1.0, ki=0.0, kd=0.15, out_max=1.0, deadzone=0.04)
# 「居中成功」判据：**全档位一套，单位=归一化像素偏差**（2026-09-18）。
#   px_x/px_y 2026-09-18 演进 0.10/0.10 → ... → 0.30/0.26 → 现 **0.25/0.26**（横向用户收回）：
#   以小球那套实测判据 `ball.center_eps=0.25` 为基准，门开口比机身大得多 → 放宽一档，竖向略紧。
#   按门 70x50cm 外轮廓/管径 0.05 → 开口 0.60x0.40、机身 ≈0.30x0.25 估单边余量 15/7.5cm，
#   工作假设（2026-09-18 定，暂无法实测）：开口 0.60x0.40（外轮廓 0.70x0.50 - 管径 0.05）、
#   机身 0.35x0.30（用户口述[面积不到门框一半]，取悲观端）-> 单边余量 12.5/5.0cm。
#   0.30/0.26 换算到 z=0.7m 是 17.2/8.4cm -- **门槛本来就大于余量**，因为它只是[开始同时
#   前进+修偏]的起点误差，穿越时的误差由 APPROACH 段闭环决定（流程：居中 -> 前进 -> 冲刺）。
#   这套假设下真正的竖向瓶颈是**死区**（上浮 0.138 约 4.5cm @0.7m，已占余量 90%），
#   加减 px_y 都动不了它 -> 调参决策规则见 cfg/comm.yaml 的 align 段注释。
#   宽高比: 竖向余量更小且 1 单位 dyn = 0.462*z 米(横向 0.818*z) → dyn 阈值取略小于 dxn。
_D_ALIGN = dict(confirm_frames=4, px_x=0.25, px_y=0.26)
# 「在门口超时兜底」：已在门口(占比≥z.near_lost_ratio) + 在安全带内，连续超过 timeout_ms
# 仍未 commit → 自己判过门并直冲。2026-09-18 现场：coarse 档没有 commit 通道（dash 关着）、
# 而门在占比 0.97 时整框仍检测得到 → 在门口 creep 了 18.7s。
_D_LOITER = dict(enable=True, timeout_ms=3000, dx_max=0.14, dy_max=0.20)
_D_Z = dict(cross=0.7, cross_confirm_frames=2, slow_max=1.3, near_lost_m=1.0,
            near_lost_ratio=0.60, z_stale_ms=1500)
_D_SURGE = dict(fast=0.35, slow=0.15, creep=0.12, lost_backward=0.12,
                reacquire=0.25, through=0.6)
_D_COARSE = dict(far_ratio=0.18, near_ratio=0.55, align_x=0.33, align_y=0.29,
                 dash=True, dash_ratio=0.35, dx_max=0.08, dy_max=0.12,
                 confirm_frames=3, surge=0.25)
_D_WIDTH = dict(dash=True, dx_max=0.08, dy_max=0.12, confirm_frames=3,
                z_max=1.5, surge=0.30)
# ALIGN 阶段的水平修正通道选择。
# ⚠️ 2026-09-18 实测结论：**默认增益下 yaw 通道"几乎推不动"**——
#   yaw 命令 = kp(0.3)·误差，而执行器死区是 |DOF| ≥ 0.138（见配置 §1.1/§1.2）
#   ⇒ 只有 |dxn| > 0.138/0.3 = **0.46（近半个屏）** 才有推力；
#   相比之下 sway：kp=8.0 → |dxn| > 0.017 就能动。
#   用户现场观察"coarse/width 往中间居中很费力"就是这个原因（小误差时 yaw 一点推力都没有）。
#   ⇒ `in_px` 默认改为 **false**（像素档以 sway 为主）；`in_pose` 保持 false（实船验证过的路径）。
#   要重新启用 yaw 居中，必须**同时**把 kp 抬到能过死区（用例 `test_yaw_centering_channel_...` 守着）。
_D_ALIGN_YAW = dict(enable=True, in_px=False, in_pose=False,
                    kp=0.3, ki=0.0, kd=0.0, out_max=0.15, deadzone=0.0,
                    # **航向通道**（只有位姿档 full/p3p 能测）：门法向 → 机身朝向误差 → yaw
                    #   enable: 默认 false（要先做一次"把机身摆正读 bias"的标定，见 cfg 注释）
                    #   即使 enable，它也只做**慢速微调**（out_max 15%）且与 sway 并存
                    heading=dict(enable=False, ema_frames=4, deadzone_deg=6.0,
                                 bias_deg=0.0))
_D_HOLD = dict(max_frames=20)
_D_REACQ = dict(max_ms=800, max_times=2, stop_ratio=0.75, reset_after_ms=3000)
# THROUGH 的结束条件：**按时间**（2026-09-18 由帧数改为时长）——
# 帧数语义会让冲刺时长随 fps 漂（8~11fps → 0.7~1.0s；掉到 6fps → 1.3s；30fps → 0.27s），
# 而「要冲多远才真的出得来」是**距离/时间**问题。confirm_ms<=0 → 退回旧的帧数语义。
_D_THROUGH = dict(confirm_frames=8, confirm_ms=900)
_D_SEARCH = dict(spin_s=2.0, pause_s=1.0, yaw=0.35)
_D_PNP = dict(reproj_px=8.0, z_min=0.2, z_max=15.0, refine=True,
              max_z_jump_m=0.8)
_D_GEOM = dict(frame_w=GATE_FRAME_W, frame_h=GATE_FRAME_H,
               body_center_offset=0.0)


def _sub(node, key):
    """取子配置节点；节点缺失/类型不对 → {}（后续用默认值，不抛 AttributeError）。"""
    try:
        v = node[key]
    except (KeyError, TypeError):
        return {}
    return v if isinstance(v, dict) else {}


def _cfg(node, defaults):
    """子节点 + 默认值合并成普通 dict：缺键/None → 默认值，其余原样（保类型）。"""
    out = dict(defaults)
    for k in defaults:
        try:
            v = node[k]
        except (KeyError, TypeError):
            continue
        if v is not None:
            out[k] = v
    return out


def _num(node, key, default):
    """取数值参数：缺失/非数值 → default（float）。"""
    try:
        v = node[key]
    except (KeyError, TypeError):
        return float(default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _dof_clip(v):
    return float(max(-1.0, min(1.0, v)))


def _ig(node, dflt):
    """把一个（可能不全的）配置节点规范化成**完整缺省表**：缺键用 dflt 补。

    用途：`_pid_kw(显式节点, ball 的那一套)` —— 缺省值直接取 ball 的实时值，
    而不是在代码里再抄一份数字（那样又会变成两套参数各自漂移）。
    """
    return dict((k, _num(node, k, dflt[k])) for k in dflt)


def _pid_kw(node, d):
    """PID 构造参数（out_max 同时作为 ±限幅）。"""
    om = _num(node, "out_max", d["out_max"])
    return dict(kp=_num(node, "kp", d["kp"]), ki=_num(node, "ki", d["ki"]),
                kd=_num(node, "kd", d["kd"]),
                out_min=-om, out_max=om,
                deadzone=_num(node, "deadzone", d["deadzone"]))


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

        geo = _cfg(_sub(V, "geometry"), _D_GEOM)
        self.frame_w_m = float(geo["frame_w"])
        self.frame_h_m = float(geo["frame_h"])
        # 相机—机身安装偏置(m)：**当前只读入、不参与运算**（接进 sway/heave 目标
        # 等于改对准基准 = 改逻辑，且该值未标定）。要启用先实测标定，见 vision.yaml。
        self.body_center_offset = float(geo["body_center_offset"])
        self.obj3 = object_points(self.frame_w_m, self.frame_h_m)
        self.camera = board_camera()
        # 角点短时记忆（**可选开关**）：小漂移/短消失用最近窗口的平均值稳住（抗水面倒影），
        # 避免"点不全/姿态不稳"的误判去触发 REACQUIRE 后退。
        # 默认开启（vision.gate.kpt_mem.enable=true，= 周一 09-14 那版行为）→ 逐点融合；
        # 要临时关掉：AUV_GATE_KPT_MEM=0 python3 main.py --task gate
        km = V.get("kpt_mem", None) or {}
        order = S.get("vision.model.task_models.gate.kpt_order", None) or \
            _sub(V, "keypoint").get("kpt_order") or [0, 1, 2, 3]
        self._kpt_mem = build_kpt_memory(km, n_kpt=len(order) or 4)
        if self._kpt_mem is None:
            print("[GATE] kpt_mem 关闭 → 角点单帧直用（%s；临时开启用 %s=1）"
                  % (enable_source(km), ENV_ENABLE))
        else:
            # 打印**实际生效**的值（原先用 km.get(k, 0.55) 兜底，缺键时会打出与
            # kpt_memory.DEFAULTS 不一致的数字，排查时被带偏）
            m = self._kpt_mem
            conf_thr = _num(_sub(V, "keypoint"), "conf_thr", 0.5)
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

        # ---- 修正增益：**共用小球那一套**（同船/同推进器，实船实测过）----
        # 解析只有两层，够用且好维护：
        #   comm.gate.pid_sway / pid_heave（**一般都不写** → 显式覆盖时才写）
        #   comm.ball.approach_pid / comm.ball.pid（默认走这条；ball 一调，过门跟着变）
        # 前面那个作为"缺省表"传给 _pid_kw：连 ball 都没有的键才用代码里的最后兜底。
        B = S.get("comm.ball", None) or {}
        self._gain_src = "gate(显式覆盖)" if (_sub(G, "pid_sway") or _sub(G, "pid_heave")) \
            else "ball(共用)"
        self._pid_sway_kw = _pid_kw(_sub(G, "pid_sway"),
                                    _ig(_sub(B, "approach_pid"), _D_PID_SWAY))
        self._pid_heave_kw = _pid_kw(_sub(G, "pid_heave"),
                                     _ig(_sub(B, "pid"), _D_PID_HEAVE))
        # sway / heave **各一套增益**（都沿用 ball 已标定值）；**全档位共用同一对像素 PID**
        # （2026-09-18 统一按像素算：不再有米制那套，切档没有跨量纲 D 尖峰问题）。
        self._pid_sway_px = PID(**self._pid_sway_kw)
        self._pid_heave_px = PID(**self._pid_heave_kw)
        # ALIGN 的**转向居中** PID（单独一套，不与 sway 共用状态）：
        # 图像水平误差大、或 width/coarse 拿不到可靠横向行程时，"转过去"比"平移过去"
        # 更容易把门拉回画面中心（旧版没有 yaw → coarse 一直差一点、完不成居中）。
        # 规则（用户定）：**yaw 与 sway 同一帧只用其一**；yaw **只在居中(ALIGN)** 用，
        # APPROACH/THROUGH 一律 yaw=0。
        # yaw 同样**共用小球**：ball 的转向增益是平铺键 edge_yaw_kp/kd/max（见 ball.py），
        # 未在 gate.align_yaw 里显式覆盖时就用它（ki/deadzone 与 ball.py 一致 = 0/0）。
        yg = _sub(G, "align_yaw")
        yaw_def = dict(_D_ALIGN_YAW)
        yaw_def.update(kp=_num(B, "edge_yaw_kp", yaw_def["kp"]),
                       kd=_num(B, "edge_yaw_kd", yaw_def["kd"]),
                       out_max=_num(B, "edge_yaw_max", yaw_def["out_max"]))
        self._yaw_kw = _pid_kw(yg, yaw_def)          # gate 显式 > ball（缺省）
        if any(k in yg for k in ("kp", "kd", "out_max", "deadzone")):
            self._gain_src = "gate(显式覆盖)"
        self._pid_yaw_px = PID(**self._yaw_kw)
        # 启动日志：现场一眼看到"跑的是哪一套"（共用小球 or 显式覆盖）
        _ks = ("kp", "ki", "kd", "out_max", "deadzone")
        print("[GATE] 修正增益来源=%s | sway %s | heave %s | yaw %s"
              % (self._gain_src,
                 {k: self._pid_sway_kw[k] for k in _ks},
                 {k: self._pid_heave_kw[k] for k in _ks},
                 {k: self._yaw_kw[k] for k in _ks}))
        self._yaw_cfg = yg
        # 航向通道配置（只有位姿档能测出航向；见 _pose_horizontal）
        self._hdg_cfg = _sub(yg, "heading")
        self._hdg_f = None                # 航向误差 EMA 状态（度）
        self._hdg_deg = None              # 最近一次测到的航向误差（度；日志/叠加用）

        self.last_info = {"phase": PH_SEARCH, "substate": "", "mode": "",
                          "action": "stop", "z": 0.0, "dx": 0.0, "dy": 0.0,
                          "sway": 0.0, "heave": 0.0, "surge": 0.0, "yaw": 0.0,
                          "pass": 0, "kpt": 0, "kpt_raw": 0, "ratio": 0.0,
                          "hdg": None,          # 航向误差(度)：+ = 门法向偏画面右 = 机身左偏
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
        self._w_cnt = 0                  # width 档"对准"连续帧数（独立计数，防跨档污染）
        self._c_cnt = 0                  # coarse 档"对准"连续帧数
        self._through_surge = None       # 本次冲刺速度（None → surge.through）
        self._cross_cnt = 0              # 连续 z≤cross 帧数（穿门确认，防单帧错解）
        self._through_frames = 0
        self._through_start_ms = None    # THROUGH 起始时间戳（时长判据）
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
        for p in (self._pid_sway_px, self._pid_heave_px, self._pid_yaw_px):
            p.reset()

    def _horizontal_out(self, dxn, now_ms, in_pose=False):
        """居中阶段的水平修正：**yaw 与 sway 互斥，只返回一个非零**。

        规则（cfg `comm.gate.align_yaw`）：
          enable=false            → 永远用 sway（= 旧行为，可回退）
          in_pose/in_px 分别控制位姿档(full/p3p)与像素档(width/coarse)是否允许用 yaw
        返回 (sway, yaw)；同一帧两者只有一个非零，另一路 PID 复位（防跨通道 D 尖峰）。

        ⚠️ 用的是**同一个误差 dxn** 驱动两个通道 → 只能"二选一"（否则同一误差被
        放大两次、且切换瞬间有 D 尖峰）。**位置误差就该用平移（sway）**：
        yaw 是**方位**控制（把门拉到光轴上 ≠ 机身与门平行），而且默认增益下
        yaw 在 |dxn|<0.46 时根本推不动（见文件头 _D_ALIGN_YAW 注释）。
        真正需要转向的"机身歪"由 `_pose_horizontal` 的**航向通道**处理（信号不同、可并存）。
        """
        yg = self._yaw_cfg
        use_yaw = bool(yg.get("enable", True)) and             (bool(yg.get("in_pose", False)) if in_pose
             else bool(yg.get("in_px", False)))
        if use_yaw:
            self._pid_sway_px.reset()
            yaw = _dof_clip(self._pid_yaw_px.update(dxn, now_ms))
            return 0.0, yaw
        self._pid_yaw_px.reset()
        # **所有档位都用同一个像素 PID**（单位统一，切档不再有跨量纲 D 尖峰）
        return _dof_clip(self._pid_sway_px.update(dxn, now_ms)), 0.0

    def _pose_horizontal(self, dxn, hdg_deg, now_ms, centering, pose_mode=""):
        """位姿档的水平修正：**位置误差 → sway**，**朝向误差 → yaw**（两个独立通道，可同时发）。

        为什么能同时发（与 `_horizontal_out` 的互斥规则不矛盾）：这里两路吃的是**不同信号**
        （`dxn`=门在画面里偏多少；`hdg`=门法向与光轴差多少度），不存在同一误差被放大两次。

        `hdg_deg` 是 `gate_normal_angles_deg(rvec)` 测出的航向误差（度；=0 表示机身正对门）。
        必须满足才启用：`align_yaw.enable` 且 `align_yaw.heading.enable` 且处于居中阶段。
        噪声实测 ≈3~5°/帧（EMA 4 帧后 ≈3°）→ 用 `deadzone_deg`(默认 6°) 挡掉噪声，
        只对"真的歪了"出手；出手也是慢速微调（沿用 yaw 那套 out_max=0.15 = 15% 推力）。
        """
        hc = self._hdg_cfg
        # ⚠️ 航向通道**只认 4 角(full)**：3 角(p3p)时位姿是欠定的（3 点解 6-DoF），
        #    实测（tools/analyze_pnp_center.py + analyze_heading.py，真实 dump）：
        #      p3p 最优解重投影 RMS 恒为 **0.0px** → `reproj_px` 筛不掉任何东西；
        #      p3p 深度比 full 小 ~28%（1.25 vs 1.73m）；航向角 std **64~69°**（full 干净 dump 只有 5.4°）
        #    → p3p 能用的只有"横向误差 dxn"（实测与 bbox 中心一致、0 帧符号相反）与"这一帧有门"。
        use_head = bool(hc.get("enable", False)) and centering \
            and pose_mode == MODE_FULL
        if not use_head or hdg_deg is None:
            return self._horizontal_out(dxn, now_ms, in_pose=True)
        sway = _dof_clip(self._pid_sway_px.update(dxn, now_ms))
        if abs(hdg_deg) < _num(hc, "deadzone_deg", 6.0):
            self._pid_yaw_px.reset()
            return sway, 0.0
        # 把角度误差换算成**与 dxn 同量纲**的等效偏差：绕垂直轴转 psi 会让门心在画面上
        # 移动 ≈ -fx·tan(psi) 像素 ⇒ hdg_n = tan(psi)·fx/(w/2)。这样 yaw 那套增益
        # （kp/out_max，来自小球实测）可以直接复用，符号也与已验证的 dxn 通道一致
        # （数值验证：机身右转 → psi<0 → yaw<0 = 左转，即修正方向正确）。
        hdg_n = float(np.tan(np.radians(hdg_deg)) * self.camera.fx / (self.w / 2.0))
        hdg_n = float(max(-1.5, min(1.5, hdg_n)))
        return sway, _dof_clip(self._pid_yaw_px.update(hdg_n, now_ms))

    def _set_info(self, action, mode="", substate="",
                  z=None, dx=0.0, dy=0.0, sway=0.0, heave=0.0,
                  surge=0.0, yaw=0.0, kpt=0, ratio=None, kpt_raw=None,
                  hdg=None):
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

    def _start_through(self, surge=None):
        """进入 THROUGH（只前进、不微调）。

        surge=None → 用 `comm.gate.surge.through`（**位姿档/近距丢失两条原有出口的
        行为逐字节不变**）；width/coarse 档可传各自更保守的直冲速度（见 cfg 注释）。
        """
        self.phase = PH_THROUGH
        self.substate = ""
        self._through_frames = 0
        self._lost_cnt = 0
        self._z_guard = False
        self._through_surge = None if surge is None else float(surge)
        self._through_start_ms = None     # 第一帧 tick 时打时间戳（见 _tick_through）
        self._loiter_start_ms = None      # 已进入冲刺 → 门口的计时作废
        self._reset_lateral_pids()

    def _through_speed(self):
        """本次冲刺速度：显式指定优先，否则 surge.through。"""
        if self._through_surge is not None:
            return float(self._through_surge)
        return _num(_sub(self._G, "surge"), "through", 0.6)

    def _start_search(self):
        self.phase = PH_SEARCH
        self.substate = ""
        self._search_entry_ms = None
        self._last_pose = None
        self._z_last = None
        self._z_guard = False
        self._cross_cnt = 0
        self._lost_cnt = 0
        self._hold_cnt = 0
        self._w_cnt = 0
        self._c_cnt = 0
        self._through_surge = None
        if self._kpt_mem is not None:
            self._kpt_mem.reset()

    def _search_yaw_pulse(self, now_ms):
        """SEARCH：原地旋转脉冲（转 spin_s → 停 pause_s）。"""
        s = _cfg(_sub(self._G, "search"), _D_SEARCH)
        spin = _num(s, "spin_s", 2.0) * 1000
        pause = _num(s, "pause_s", 1.0) * 1000
        if spin + pause <= 0:
            return 0.0
        if self._search_entry_ms is None:
            self._search_entry_ms = now_ms
        ph = (now_ms - self._search_entry_ms) % (spin + pause)
        return _num(s, "yaw", 0.35) if ph <= spin else 0.0

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
                now_ms - self._start_ms >= _num(self._G, "timeout_ms", 180000)):
            self._finish("timeout")
        elif self._finished:
            pass
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
        conf_thr = _num(_sub(self._V, "keypoint"), "conf_thr", 0.5)
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
        self._lost_cnt = 0
        V = self._V
        conf_thr = _num(_sub(V, "keypoint"), "conf_thr", 0.5)
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
            pnp = _cfg(_sub(V, "pnp"), _D_PNP)
            # 有上帧位姿就用于消歧（平面 IPPE 双解在远距/小目标时 RMS 接近，
            # 纯靠 RMS 会帧间来回跳；上帧距离惩罚把解锁在连续分支上）
            prev = self._last_pose
            res = gate_pose(self.camera, obj3s, img2s, prev=prev,
                            reproj_thr=_num(pnp, "reproj_px", 8.0),
                            z_bounds=(_num(pnp, "z_min", 0.2),
                                      _num(pnp, "z_max", 15.0)),
                            refine=bool(pnp.get("refine", True)))
            if res is not None:
                # z 跳变保护：上一帧也有可信位姿时，z 突变视为错解弃帧
                # （错解若给出 z ≤ z.cross 会直接触发 THROUGH → 满速冲出去）
                max_jump = _num(pnp, "max_z_jump_m", 0.8)
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
        # 本帧没有可用位姿：解除跳变判据，下个位姿重新建立基准（防“卡死”）；
        # 同时清零穿门确认计数 → 只有“连续帧”都判就近才算过门
        self._z_guard = False
        self._cross_cnt = 0
        if self.mode != mode:
            # 直冲计数按"哪一个档"独立 → 任何档位变化都清零（否则 width→coarse→width
            # 会把不相邻的帧算成"连续对准"→误直冲；用例 test_dash_counter_resets_...）
            self._w_cnt = 0
            self._c_cnt = 0
            self._center_cnt = 0
            # 注（2026-09-18 核实）：**本块只在"位姿失败/退化"的路径上执行** ——
            # 位姿解成功时 `_on_pose()` 之后立刻 `return`（见上面几行），
            # 所以 full(4角)↔p3p(3角) 帧间翻转**不会**清 `_center_cnt`
            # （曾误以为会清、并据此改过一版"按类清零"，实测是空操作，已退回）。
            # 真正会走到这里的是：位姿校验失败→退化 coarse、width↔coarse 互换。
        if mode == MODE_WIDTH:
            self._on_width(det, now_ms, ids)
            return
        self._on_coarse(det, now_ms)    # coarse / 其它退化

    # ---------------- full/p3p 位姿可用 ----------------
    def _on_pose(self, det, pose, now_ms, mode, kpt):
        """位姿档：**位姿只提供深度 z 与"可信"这一事实**；居中一律用**像素误差**。

        为什么要统一成像素（2026-09-18）：
          * 深度 z 有系统性尺度误差（实测 PnP z 比框宽粗估大 ~20%，且去畸变/标注口径不完美）
            → 用 t_x(m) 做居中，等于把"深度的误差"直接灌进控制回路；
          * 米制/像素两套阈值 + 两套 PID 实例，会让 mode 在 full↔coarse 之间跳时两套控制器
            交替工作（各拿一半帧）→ 收敛不了；
          * 统一后：**一套阈值(`align.px_x/px_y`)、一套 PID（像素档那套）**，全档位可比。
        门框中心取"用位姿把门原点(0,0,0)投影回图像"——p3p(3 角)时也比角点均值更准（缺角不偏）。
        """
        G = self._G
        al = _cfg(_sub(G, "align"), _D_ALIGN)
        zc = _cfg(_sub(G, "z"), _D_Z)
        sg = _cfg(_sub(G, "surge"), _D_SURGE)
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
        # 朝向误差（只有位姿档能测）：门法向 n=R·[0,0,1] → 机身相对门的航向/俯仰角
        # 注意：dxn 是**方位**（门在画面里偏多少），修它用平移；航向是**姿态**，修它才用转向。
        psi_deg, _pit_deg = gate_normal_angles_deg(rvec, tvec)
        hc = self._hdg_cfg
        k_ema = max(1, int(_num(hc, "ema_frames", 4)))
        self._hdg_f = psi_deg if self._hdg_f is None else \
            self._hdg_f + (2.0 / (k_ema + 1.0)) * (psi_deg - self._hdg_f)
        self._hdg_deg = self._hdg_f - _num(hc, "bias_deg", 0.0)   # 去掉安装/场地偏置
        # 符号沿用（图像 x 向右 = 机身向右 → sway/yaw 取正；图像 y 向下 → heave 取负）
        sway, yaw = self._pose_horizontal(
            dxn, self._hdg_deg, now_ms,
            centering=self.phase in (PH_SEARCH, PH_ALIGN), pose_mode=mode)
        heave = -_dof_clip(self._pid_heave_px.update(dyn, now_ms))
        aligned = abs(dxn) <= _num(al, "px_x", _D_ALIGN["px_x"]) and \
            abs(dyn) <= _num(al, "px_y", _D_ALIGN["px_y"])
        dx_m, dy_m = dxn, dyn            # 日志里 dx/dy 现在**统一是像素归一化**
        cross = _num(zc, "cross", 0.7)
        if z <= cross:
            # 穿门确认：单帧就近不冲（平面 PnP 偶发错解若给出 z≤cross，
            # 直接 THROUGH 会让船满速冲出去），连续 n 帧才判过门
            self._cross_cnt += 1
            need = int(_num(zc, "cross_confirm_frames", 2))
            if self._cross_cnt >= max(1, need):
                # 冲刺：只前进，不带横向微调（微调已在上一帧做完）
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
            self.substate = SUB_GOLDEN
            if self._loiter_commit(dxn, dyn, float(det.w) / float(self.w), now_ms):
                return
            if aligned:
                self._center_cnt += 1
                if self._center_cnt >= int(_num(al, "confirm_frames", 6)):
                    self.phase = PH_APPROACH
                    self.substate = ""
            else:
                self._center_cnt = 0
            self._set_info("center", mode=mode, z=z, dx=dx_m, dy=dy_m,
                           sway=sway, heave=heave, yaw=yaw, kpt=kpt,
                           hdg=self._hdg_deg)
        elif self.phase == PH_APPROACH:
            # 进近两档（远→快 / 近→慢）；分档点 = z.slow_max
            # 注：z.fast_max / z.align_max 当前**未参与运算**（见 cfg 注释）
            s_slow = _num(sg, "slow", 0.15)
            if z > _num(zc, "slow_max", 1.3):
                surge = _num(sg, "fast", 0.35)
            else:
                surge = s_slow
            self._set_info("forward_%s" % ("fast" if surge > s_slow else "slow"),
                           mode=mode, z=z, dx=dx_m, dy=dy_m,
                           sway=sway, heave=heave, surge=surge, kpt=kpt,
                           hdg=self._hdg_deg)
        else:
            self._set_info("center", mode=mode, z=z, dx=dx_m, dy=dy_m,
                           kpt=kpt)

    # ---------------- width（对向 2 角：测距+中点对中 → 对准好就直接冲） ----
    def _on_width(self, det, now_ms, ids):
        """width 档：只有对向 2 角（上边或下边）。

        信息只够"水平中点 + 框心竖直"，不够 6-DoF（所以不在这里解 PnP）。策略按
        实船经验收敛成两条，**不再"对准了也一直 HOLD 到超时"**：
          ① z > width.z_max（还远）且对准 → 慢 creep 靠近（换取角点/整门）；
          ② z ≤ width.z_max 且**连续 confirm_frames 帧都在安全带内** → 直接冲
             （`_start_through`，只前进不微调）。z 由 fx·W/Δu 粗估，**只用来判
             "该不该冲"**，不参与闭环——低帧率/抖动下越少依赖测距越稳。
        防撞杆：安全带 `dx_max/dy_max`（偏离大就不冲）+ 直冲速度 `surge`（低于
        surge.through）+ 未对中时保持静止（宁可等，不蹭杆）。
        """
        G = self._G
        W = _cfg(_sub(G, "width"), _D_WIDTH)
        al = _cfg(_sub(G, "align"), _D_ALIGN)
        sg = _cfg(_sub(G, "surge"), _D_SURGE)
        self.mode = MODE_WIDTH
        kp = self._kpt_s if self._kpt_s is not None else det.kpts
        u1, v1 = kp[ids[0]]
        u2, v2 = kp[ids[1]]
        z = width_range_depth(u1, u2, float(self.camera.fx), self.frame_w_m)
        if z is None:
            self._on_coarse(det, now_ms)
            return
        z = float(np.clip(z, 0.2, 15.0))
        self._z_last = z
        self._z_ms = now_ms
        aim_x = (u1 + u2) / 2.0
        cx, cy = bbox_center(det)
        dxn = (aim_x - self.w / 2.0) / (self.w / 2.0)
        dyn = (cy - self.h / 2.0) / (self.h / 2.0)
        # 对中判据（归一化像素偏差）：**全档位统一**（位姿档也用这一套，见 _on_pose）
        aligned = abs(dxn) <= _num(al, "px_x", _D_ALIGN["px_x"]) and \
            abs(dyn) <= _num(al, "px_y", _D_ALIGN["px_y"])
        if self.phase == PH_SEARCH:
            self.phase = PH_ALIGN
            self._center_cnt = 0
            self._w_cnt = 0
        # 水平修正：**居中阶段才允许 yaw**，且 yaw 与 sway 同一帧只发一个
        if self.phase == PH_ALIGN:
            sway, yaw = self._horizontal_out(dxn, now_ms)
        else:
            yaw = 0.0
            self._pid_yaw_px.reset()
            sway = _dof_clip(self._pid_sway_px.update(dxn, now_ms))
        heave = -_dof_clip(self._pid_heave_px.update(dyn, now_ms))

        # 职责分离（都不是死旋钮）：
        #   aligned → creep / HOLD 判据（较严，决定"要不要动"）
        #   in_band → 直冲安全带（决定"敢不敢冲"）
        in_band = abs(dxn) <= _num(W, "dx_max", 0.08) and \
            abs(dyn) <= _num(W, "dy_max", 0.12)
        self._w_cnt = self._w_cnt + 1 if in_band else 0
        if bool(W.get("dash", True)) and \
                self._w_cnt >= max(1, int(_num(W, "confirm_frames", 3))) and \
                z <= _num(W, "z_max", 1.5):
            # 出口③：够近 + 连续在安全带内 → 直冲（复用 THROUGH：只前进、不微调）
            self._start_through(surge=_num(W, "surge", 0.30))
            if S.DEBUG:
                print("[GATE] width 对准直冲: z=%.2f dx=%.2f dy=%.2f 连续 %d 帧"
                      % (z, dxn, dyn, self._w_cnt))
            self._set_info("through", mode=MODE_WIDTH, z=z, dx=dxn, dy=dyn,
                           surge=self._through_speed(), kpt=self._dbg_kpt)
            return

        if self.phase == PH_ALIGN:
            # 在门口超时兜底（实测：门占比 0.97 时整框仍检测得到 → 不走丢门分支）
            if self._loiter_commit(dxn, dyn, float(det.w) / float(self.w), now_ms):
                return
            # 出口①：距门仍远且对中 → 慢 creep（有限信息下安全推进）
            if aligned and z > _num(W, "z_max", 1.5):
                self.substate = SUB_CREEP
                self._center_cnt += 1
                surge = _num(sg, "creep", 0.12)
                if self._center_cnt >= int(_num(al, "confirm_frames", 6)):
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

    # ---------------- coarse（整门框可信/角不足）：三层仲裁 ----------------
    def _on_coarse(self, det, now_ms):
        """coarse 档：角点不足，只有整框可信。

        仲裁按"能不能冲"分层，**只有未对准才后退**（实测里来回退的主因是"一看框
        大就退"，与是否对准无关）：
          ① 够近（框占比 ≥ dash_ratio）且对准确认 → 直接冲（出口④）；
          ② 对准但还远 → 慢 creep 靠近（不再中距干等）；
          ③ 未对准：中距 HOLD 超限 → REACQUIRE；很近(框装不下) → REACQUIRE。
        "够近"用框占比（不依赖测距），与"测距只用于判该不该冲"的策略一致。
        """
        G = self._G
        C = _cfg(_sub(G, "coarse"), _D_COARSE)
        H = _cfg(_sub(G, "hold"), _D_HOLD)
        sg = _cfg(_sub(G, "surge"), _D_SURGE)
        self.mode = MODE_COARSE
        # coarse 档**没有任何测距**：`_z_last` 会一直是上一次 width/位姿留下的陈旧值
        # （现场实测留下过 9.25 的垃圾值，一路带到穿门判定里）→ 标它失效，别让它参与判据。
        # 注意：只失效**判据**，不动 `_z_last` 本身（日志仍要能看到它实际是多少）。
        self._z_ms = None
        ratio = float(det.w) / float(self.w)
        # 已在 REACQUIRE：闭环后退（退到框够小即停）或超时回 search
        if self.phase == PH_ALIGN and self.substate == SUB_REACQUIRE:
            self._tick_reacquire(now_ms, ratio)
            return
        cx, cy = bbox_center(det)
        dxn = (cx - self.w / 2.0) / (self.w / 2.0)
        dyn = (cy - self.h / 2.0) / (self.h / 2.0)
        aligned = abs(dxn) <= _num(C, "align_x", _D_COARSE["align_x"]) and \
            abs(dyn) <= _num(C, "align_y", _D_COARSE["align_y"])

        if self.phase == PH_SEARCH:
            self.phase = PH_ALIGN
            self._hold_cnt = 0
            self._c_cnt = 0
        if self.phase == PH_APPROACH:
            # 进近中角全失 → 保守降回 ALIGN 仲裁（避免盲冲）
            self.phase = PH_ALIGN
            self._hold_cnt = 0

        # 水平修正：**居中阶段才允许 yaw**（coarse 拿不到可靠横向行程，靠转向把门拉回
        # 画面中心），且 yaw 与 sway 同一帧只发一个；非居中阶段一律 sway、yaw=0。
        if self.phase == PH_ALIGN:
            sway, yaw = self._horizontal_out(dxn, now_ms)
        else:
            yaw = 0.0
            self._pid_yaw_px.reset()
            sway = _dof_clip(self._pid_sway_px.update(dxn, now_ms))
        heave = -_dof_clip(self._pid_heave_px.update(dyn, now_ms))

        if self.phase != PH_ALIGN:
            self._set_info("hold", z=self._z_last, dx=dxn, dy=dyn,
                           sway=sway, heave=heave, kpt=self._dbg_kpt)
            return

        # 职责分离（与 width 档同构）：align_x/y → 走位判据；dx_max/dy_max → 直冲安全带
        in_band = abs(dxn) <= _num(C, "dx_max", 0.08) and \
            abs(dyn) <= _num(C, "dy_max", 0.12)
        self._c_cnt = self._c_cnt + 1 if in_band else 0
        if bool(C.get("dash", True)) and \
                self._c_cnt >= max(1, int(_num(C, "confirm_frames", 3))) and \
                ratio >= _num(C, "dash_ratio", 0.35):
            # 出口④：够近 + 连续在安全带内 → 直冲
            self._start_through(surge=_num(C, "surge", 0.25))
            if S.DEBUG:
                print("[GATE] coarse 对准直冲: ratio=%.2f dx=%.2f dy=%.2f 连续 %d 帧"
                      % (ratio, dxn, dyn, self._c_cnt))
            self._set_info("through", z=self._z_last, dx=dxn, dy=dyn,
                           surge=self._through_speed(), kpt=self._dbg_kpt)
            return

        if self._loiter_commit(dxn, dyn, ratio, now_ms):
            return
        if aligned:
            # 对准：能靠近就靠近（争取露出角点），不再中距干等
            self.substate = SUB_CREEP
            self._hold_cnt = 0
            self._set_info("creep", z=self._z_last, dx=dxn, dy=dyn,
                           sway=sway, heave=heave, yaw=yaw,
                           surge=_num(sg, "creep", 0.12), kpt=self._dbg_kpt)
            return

        if ratio < _num(C, "far_ratio", 0.18):       # 远距小框：没对准就别冲
            self.substate = SUB_CREEP
            self._set_info("center", z=self._z_last, dx=dxn, dy=dyn,
                           sway=sway, heave=heave, yaw=yaw, kpt=self._dbg_kpt)
        elif ratio <= _num(C, "near_ratio", 0.55):   # 中距 → HOLD（无位姿即无进展）
            self.substate = SUB_HOLD
            self._hold_cnt += 1
            if self._hold_cnt >= int(_num(H, "max_frames", 20)):
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
        """进入后退重取；**同一段里连续超 max_times 次就放弃后退，改原地保持**。

        倒影持续干扰时"退-进-退"会来回震荡（看着就是一直后退又一直对准）；
        退了好几次仍拿不到可用角点，说明再退也没用 → 停下来等（超时另算）。
        """
        G = self._G
        R = _cfg(_sub(G, "reacquire"), _D_REACQ)
        # 距上次后退够久 → 计数重新开始（否则一次倒影干扰会把后退永久锁死）
        reset_after = int(_num(R, "reset_after_ms", 3000))
        if self._reacquire_last_ms is None or \
                now_ms - self._reacquire_last_ms >= reset_after:
            self._reacquire_cnt = 0
        self._reacquire_cnt += 1
        max_times = int(_num(R, "max_times", 2))
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
        R = _cfg(_sub(G, "reacquire"), _D_REACQ)
        sg = _cfg(_sub(G, "surge"), _D_SURGE)
        r0 = self._reacquire_ratio0
        stop_ratio = _num(R, "stop_ratio", 0.75)
        if ratio is not None and r0 and ratio <= r0 * stop_ratio:
            if S.DEBUG:
                print("[GATE] REACQUIRE 已退够(ratio=%.2f<=%.2f=%.2f×%.2f) → 回对准"
                      % (ratio, r0 * stop_ratio, r0, stop_ratio))
            self.substate = SUB_HOLD
            self._hold_cnt = 0
            self._set_info("hold", z=self._z_last, kpt=self._dbg_kpt)
            return
        dur = now_ms - (self._reacquire_start or now_ms)
        if dur >= _num(R, "max_ms", 800):
            self._start_search()
            self._set_info("search")
            return
        # 后退方向/量可配（dof sign 需水池实测；负 surge = 后退）
        surge = -_num(sg, "reacquire", 0.25)
        self._set_info("reacquire", substate=SUB_REACQUIRE, surge=surge,
                       kpt=self._dbg_kpt)

    # ---------------- 丢目标 / 穿门 ----------------
    def _loiter_commit(self, dxn, dyn, ratio, now_ms, kpt=None):
        """在门口超时兜底：**不看档位**，只要「人在门口 + 对准」持续太久就自己拍板直冲。

        条件（三个都成立）：
          · 框占比 ≥ `z.near_lost_ratio`（= 与"丢门判过门"同一个"在门口"定义）
          · |dxn| ≤ `loiter.dx_max`、|dyn| ≤ `loiter.dy_max`（安全带，与 width/coarse 直冲同量纲）
          · 上述状态**连续** ≥ `loiter.timeout_ms`（任一条不成立就重新计时）
        一旦命中 → `_start_through()`（判过门 + 直冲），返回 True，调用方直接 return。

        为什么需要（2026-09-18 现场日志）：coarse/width 唯一能 commit 的出口③④被
        `dash: false` 关掉，而门在占比 0.97（≈0.5m）时**整框仍检测得到** → 走不到"丢门"
        分支 → 船贴着门口以 creep(0.20) 爬了 **18.7 秒**，最后只靠丢检补了一次 1.3s 冲刺。
        这条兜底就是那段缺失的"决策出口"，且与档位解耦（位姿/width/coarse 都能用）。
        """
        L = _cfg(_sub(self._G, "loiter"), _D_LOITER)
        if not bool(L.get("enable", True)):
            self._loiter_start_ms = None
            return False
        near_r = _num(_sub(self._G, "z"), "near_lost_ratio", 0.60)
        ok = (float(ratio or 0.0) >= near_r and
              abs(float(dxn)) <= _num(L, "dx_max", 0.14) and
              abs(float(dyn)) <= _num(L, "dy_max", 0.20))
        if not ok:
            self._loiter_start_ms = None
            return False
        if self._loiter_start_ms is None:
            self._loiter_start_ms = now_ms
            return False
        wait = _num(L, "timeout_ms", 3000.0)
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
        """整门丢失时：判断「是不是已经到门口了（≈机身已进门框）」。

        两条判据取 OR（现场数据驱动的设计）：
          ① `z ≤ z.near_lost_m`（**要求 z 新鲜**）：位姿/width 档每帧都会更新时间戳；
             coarse 档没有测距 → 时间戳被清空 → 陈旧垃圾 z 不会再影响判定
             （现场日志里正是 9.25 的陈旧值一路带到了穿门判定）。
          ② 末次检测到的**框占比 ≥ z.near_lost_ratio**：全档位可用、每帧都算，
             门快装不下/机身进门框时必然成立。
        返回命中的判据名（空串 = 没到门口）。
        """
        near_m = _num(zc, "near_lost_m", 1.0)
        near_r = _num(zc, "near_lost_ratio", 0.60)
        stale = _num(zc, "z_stale_ms", 1500.0)
        z_fresh = (self._z_ms is not None and self._z_last is not None and
                   stale > 0 and (now_ms - self._z_ms) <= stale)
        if z_fresh and self._z_last <= near_m:
            return "z=%.2f" % self._z_last
        if float(self._dbg_ratio or 0.0) >= near_r:
            return "ratio=%.2f" % float(self._dbg_ratio or 0.0)
        return ""

    def _tick_lost(self, now_ms):
        G = self._G
        zc = _cfg(_sub(G, "z"), _D_Z)
        sg = _cfg(_sub(G, "surge"), _D_SURGE)
        self._lost_cnt += 1
        self._z_guard = False             # 丢目标→解除 z 跳变基准
        self._cross_cnt = 0
        pose_hold = int(_num(G, "pose_hold_frames", 10))
        if self.phase in (PH_ALIGN,) and self.substate == SUB_REACQUIRE:
            self._tick_reacquire(now_ms)      # 已丢目标：无 ratio，按时间退完
            return
        if self.phase == PH_ALIGN:
            # ★2026-09-18 修★ 这里原先**没有"近距丢失=已过门"的判断**，一律防抖后回 SEARCH。
            #   后果（现场现象一次解释完）：width/coarse 的 creep（surge=0.20，**无距离上限**）
            #   会把船一路送到门口、且**始终留在 ALIGN**；到门口门占满视野→整门丢失→走这条
            #   分支→防抖 10 帧→`_start_search()`。于是"到门口突然 SEARCH + 从来没进 THROUGH
            #   + 原地不动"（SEARCH 只有 yaw 脉冲、surge=0）。
            #   判据用**最后一次检测到的框占比**（每帧更新、比 z 更可靠且全档位可用）：
            #   门快装不下/机身已进门框 ⇒ 真过门。远处丢检（占比小）仍走原 SEARCH 路径。
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
                # 进近已到近距(z≤near_lost_m)却整门丢失：门已占满视野/机身进入门框，
                # 这是**真过门**的典型现象 → 直接判过门并直行穿越（不等防抖、不后退）；
                # 否则只会退回 SEARCH，永远数不到过门（旧版这里靠 PnP 错解偶然给出
                # z≤cross 才“过门”）
                if S.DEBUG:
                    print("[GATE] 近距丢失(%s) → 判过门" % why)
                self._start_through()
                self._set_info("through", surge=self._through_speed())
            elif self._lost_cnt <= pose_hold and self._z_last is not None:
                # 还在远处、只是短暂丢失：**轻微后退**（学撞球：丢目标时不带速度盲冲，
                # 退一点换取重新锁定/更大视野），退满防抖窗口仍无目标 → 转 SEARCH
                self._set_info("backward_slow", z=self._z_last,
                               surge=-_num(sg, "lost_backward", 0.12))
            else:
                self._start_search()
                self._set_info("search")
            return
        # SEARCH
        yaw = self._search_yaw_pulse(now_ms)
        self._set_info("search", yaw=yaw)

    def _tick_through(self, now_ms):
        """穿门：**只前进、不微调**（横向修正全部留在冲刺前完成）。

        结束条件（2026-09-18 改）：**按 `through.confirm_ms` 计时**（默认 900ms ≈ 旧 8 帧@9fps），
        `confirm_ms<=0` 时退回旧的帧数语义（`confirm_frames`）。为什么改：帧数语义下的冲刺
        时长随 fps 漂（8~11fps → 0.7~1.0s），而「能不能冲出去」取决于**跑了多远**；低帧率或
        偶发卡顿时时长会不够 → 船停在门口（现场：through 没冲出去，像原地没动）。
        ⚠️ 900ms 只是**保持原行为**的等价值，真正够不够要按实测速度标定：
           需要冲的距离 ≈ z.cross(0.7；注意 z 高估 ~20% ⇒ 实际 ~0.58m) + 机身长度
           → 若实测航速 v(m/s)，confirm_ms ≈ 1000·(0.58+L)/v，再留 30% 余量。
        """
        G = self._G
        T = _cfg(_sub(G, "through"), _D_THROUGH)
        self._through_frames += 1
        if self._lost_cnt is not None:
            self._lost_cnt += 1
        if self._through_start_ms is None:
            self._through_start_ms = now_ms
        # 结束条件（2026-09-18 改）：优先**时长**(confirm_ms>0)，否则退回旧的**帧数**
        ms = _num(T, "confirm_ms", 0.0)
        if ms > 0:
            done = (now_ms - self._through_start_ms) >= ms
        else:
            done = self._lost_cnt >= max(1, int(_num(T, "confirm_frames", 8)))
        # 直行穿越；满足结束条件 → 判机身过门
        if done:
            self._pass_cnt += 1
            if S.DEBUG:
                print("[GATE] 通过第 %d/%d 门"
                      % (self._pass_cnt, int(_num(G, "pass_target", 1))))
            if self._pass_cnt >= int(_num(G, "pass_target", 1)):
                self._finish("pass")
            else:
                self._start_search()
                self._set_info("search")
            return
        self._set_info("through", surge=self._through_speed())
