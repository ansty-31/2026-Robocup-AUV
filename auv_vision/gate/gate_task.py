# -*- coding: utf-8 -*-
"""gate/gate_task.py — 任务三 过门（GateTask，§5.3 相位机骨架）

门 = 对称平面矩形门框(0.70×0.50 m)，悬空，无朝向要求；任务 = 机身穿过开口。

相位机：SEARCH → RANGE_ALIGN → APPROACH → THROUGH(计数) → (下一门)/DONE
RANGE_ALIGN 子状态（按前端 mode 与门框占屏比仲裁）：
  GOLDEN   full/p3p 位姿对准（sway/heave PID，无 surge）
  CREEP    远距小框（coarse far / width）：慢速 creep + 框心对中
  HOLD     中/近角不足：surge=0 保持对中；≥hold.max_frames → REACQUIRE
  REACQUIRE 短后退重取整门（拉距→4 角重现）；超限 → SEARCH
穿过例外（两条，任一成立即 THROUGH，忽略角丢失直行）：
  ① 位姿可信且 Z ≤ z.cross 连续 cross_confirm_frames 帧；
  ② 进近中已到近距(Z ≤ z.near_lost_m)后整门丢失（门占满视野/机身入门框）。
  THROUGH 内再按 through.confirm_frames 帧确认 → 计数 +1（达 pass_target → DONE）。
  位姿跳变保护：pnp.max_z_jump_m 内的帧间变化才可信，突变帧弃帧退化 coarse，
  防平面 PnP 错解(尤其偶发 z≤cross)把船直接推出去。

**分层**（v1.4，只搬文件不改运动）：
    quality.py        质量分 Q 的两条去向：先验→kpt_memory 权重/增益；后验→谨慎度
    perception.py     感知半场：选目标→先验→融合→mode→线索→位姿→后验 ⇒ Sighting
    phase_degrade.py  降级路径相位段（width / coarse / REACQUIRE 闭环）
    phase_recover.py  搜索 / 丢目标 / 穿门收尾 与 线索接管
    gate_task.py      本文件：装配 + 帧守卫 + 相位选路 + 位姿/像素对准档 + 复位/诊断
本文件只保留"决策与动作"：一帧的"看到什么"全在 `perception.GatePerception`。

依赖：gate.vision.geometry（位姿）、gate.vision.perception（感知半场）、gate.motion.phases（相位常量）、
      gate.data.quality / gate.data.cues（质量分与线索）、common.streak / common.frame_stamp（共用件）、
      common.PID。参数：cfg/vision.yaml gate.* 与 cfg/comm.yaml gate.*
      （策略件默认在代码内，可用 gate.data.quality / gate.data.cues / gate.kpt_mem 覆盖，无需改代码）。
"""
from __future__ import annotations

import numpy as np

import base.settings as S
from common.PID import PID
from gate.vision.gate_detector import board_camera
from gate.vision.gate_frontend import MODE_FULL, MODE_P3P, MODE_WIDTH, MODE_COARSE
from gate.vision.geometry import object_points, GATE_FRAME_W, GATE_FRAME_H
from gate.vision.perception import GatePerception, pick_gate       # 感知半场（含质量分）
from gate.data.quality import caution_from_q                     # 质量分后验 → 谨慎度
from gate.data.cues import (cue_dof, blend_dof, cue_mode)         # 线索动作 + 主导权混合
from gate.motion.phases import (PH_SEARCH, PH_ALIGN, PH_APPROACH, PH_THROUGH,  # noqa: F401
                         SUB_GOLDEN, SUB_HOLD, SUB_CUE, SUB_REACQUIRE)
from gate.motion.phase_degrade import DegradeTicks                 # 降级路径相位段
from gate.motion.phase_recover import RecoverTicks                 # 搜索/丢失/穿门/线索
from common.streak import Streak                            # 共用：双条件确认
from common.frame_stamp import FrameStamp, FrameValidator, StampCfg  # 共用：唯一帧

_ALIGN_SCALE = 0.5        # 位置误差归一化标尺(m)：±0.5m 满量程 → DOF ±1


def _dof_clip(v):
    return float(max(-1.0, min(1.0, v)))


class GateTask(DegradeTicks, RecoverTicks):
    """过门相位机。相位段实现见 phase_degrade.py / phase_recover.py（同一份状态）。"""

    name = "gate"
    supports_frame_stamp = True     # process() 接受 (frame_id, captured_at)（v1.3）

    def __init__(self, uart, hub, frame_w, frame_h):
        self.uart = uart
        self.hub = hub
        self.w = int(frame_w)
        self.h = int(frame_h)
        G = S.comm.gate
        V = S.vision.gate
        self._G, self._V = G, V

        geo = V.geometry
        self.frame_w_m = float(geo.get("frame_w", GATE_FRAME_W))
        self.frame_h_m = float(geo.get("frame_h", GATE_FRAME_H))
        self.body_center_offset = float(geo.get("body_center_offset", 0.0))
        self.obj3 = object_points(self.frame_w_m, self.frame_h_m)
        self.camera = board_camera()

        # 感知半场：选目标→质量先验→逐点软融合→mode→线索→位姿→质量后验（gate/vision/perception.py）
        order = S.get("vision.model.task_models.gate.kpt_order", None) or \
            V.keypoint.get("kpt_order") or [0, 1, 2, 3]
        self._per = GatePerception.from_settings(S, self.camera, self.obj3,
                                                 (self.w, self.h),
                                                 n_kpt=len(order) or 4)
        self._kpt_mem = self._per.kpt_mem     # 兼容旧引用（诊断/测试）
        self._kpt_s = None            # 本帧平滑后的角点（供 _on_width 等复用）
        self._kconf_s = None

        # ===== v1.3 策略升级件（不改底层运动：相位机/速度阶梯/确认计数保持原样）=====
        # 共用件：帧有效性（唯一帧/帧龄/断点）与双条件确认放在 common/ 下，供全项目复用。
        self._stamp_cfg = StampCfg.from_settings(S)
        self._qcfg = self._per.tracker.cfg if self._per.tracker is not None else None
        self._cuecfg = self._per.cue_cfg
        self._validator = FrameValidator(self._stamp_cfg)
        self._st_cue = Streak()        # 线索动作确认（新策略，用共用 Streak）
        self._auto_id = 0              # 未传 frame_id 时的自增序号
        self._last_dof = (0.0, 0.0, 0.0, 0.0)
        self._q = None                 # 本帧质量分（诊断）
        self._caution = 1.0            # 本帧谨慎度（0=最谨慎，1=正常）
        self._sp = 1.0                 # 速度缩放 = speed_scale(caution)
        self._dz = 1.0                 # 死区/对准阈值缩放 = deadzone_scale(caution)
        self._cue = None               # 本帧线索动作
        self._cue_reason = ""
        self._pose_gap = 0             # 位姿连续不可用帧数（线索接管门槛）
        self._age = 0.0
        self._frame_kind = "ok"
        self._countable = True         # 本帧是否可计入确认（过期帧只估计、不计确认）
        # 下水前自检：标定分辨率必须与实际帧一致，否则 PnP 深度整体缩放错
        if (self.camera.width, self.camera.height) != (self.w, self.h):
            print("[GATE] ⚠️ 标定尺寸 %dx%d ≠ 实际帧 %dx%d：PnP 距离会系统性偏差，"
                  "请按当前分辨率重新标定 (cfg/front_camera.yaml)"
                  % (self.camera.width, self.camera.height, self.w, self.h))

        pid_kw = dict(kp=G.pid.kp, ki=G.pid.ki, kd=G.pid.kd,
                      out_min=-G.pid.out_max, out_max=G.pid.out_max,
                      deadzone=G.pid.deadzone)
        # 两套横向 PID：位姿档喂"t_x/0.5(m) 归一化"，像素档喂"框心偏差 归一化"。
        # 单位不同必须分开：合用一个 PID 时 kd>0 会在档位切换瞬间产生跨量纲 D 尖峰
        # （撞球只有像素档，所以它一套就够）。
        self._pid_sway_pose = PID(**pid_kw)
        self._pid_heave_pose = PID(**pid_kw)
        self._pid_sway_px = PID(**pid_kw)
        self._pid_heave_px = PID(**pid_kw)
        # 过门不做原地转向（机身穿过开口即可，无朝向要求）→ 无 yaw PID

        self.last_info = {"phase": PH_SEARCH, "substate": "", "mode": "",
                          "action": "stop", "z": 0.0, "dx": 0.0, "dy": 0.0,
                          "sway": 0.0, "heave": 0.0, "surge": 0.0, "yaw": 0.0,
                          "pass": 0, "kpt": 0, "kpt_raw": 0, "ratio": 0.0,
                          # v1.3/v1.4 策略诊断量
                          "Q": 0.0, "Qmem": 0.0, "gain": 1.0, "caution": 0.0,
                          "cue": "", "cue_w": 0.0, "cue_mode": "",
                          "visible": "", "source": "",
                          "frame_kind": "ok", "age": 0.0, "rms": 0.0,
                          "agree": "",
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
        self._z_guard = False             # 上一帧是否有可信位姿（z 跳变保护基准）
        self._lost_cnt = 0
        self._center_cnt = 0
        self._hold_cnt = 0
        self._cross_cnt = 0              # 连续 z≤cross 帧数（穿门确认，防单帧错解）
        self._through_frames = 0
        self._reacquire_start = None
        self._reacquire_ratio0 = None    # 进入 REACQUIRE 时的框占比（闭环后退基准）
        self._reacquire_cnt = 0          # 连续 REACQUIRE 次数（超限不再后退）
        self._reacquire_last_ms = None   # 上次真正后退的时刻（时间窗重置计数）
        self._give_up_until_ms = None    # 放弃后退后的 HOLD 窗口截止
        self._dbg_ratio = 0.0            # 本帧门框宽/屏宽（诊断+叠加）
        self._dbg_kpt = 0                # 本帧有效角点数（**记忆后**，用于判 mode）
        self._dbg_kpt_raw = 0            # 本帧原始有效角点数（诊断：看倒影污染程度）
        self._search_entry_ms = None
        self._pass_cnt = 0
        self._finished = False
        self._reason = ""
        self.last_dets = []               # 本帧门检测（供 main._draw 画角点）
        # v1.3：帧有效性 / 线索确认 / 诊断量
        self._validator.reset()
        self._st_cue.reset()
        self._per.reset()
        self._pose_gap = 0
        self._q = None
        self._caution = 1.0
        self._sp = self._dz = 1.0
        self._cue = None
        self._cue_reason = ""
        self._age = 0.0
        self._frame_kind = "ok"
        self._countable = True

    @property
    def ready(self):
        return self.hub.has_extra(self.name)

    # ------------------------------------------------------------ 工具
    def _reset_lateral_pids(self):
        for p in (self._pid_sway_pose, self._pid_heave_pose,
                  self._pid_sway_px, self._pid_heave_px):
            p.reset()

    def _set_info(self, action, mode="", substate="",
                  z=None, dx=0.0, dy=0.0, sway=0.0, heave=0.0,
                  surge=0.0, yaw=0.0, kpt=0, ratio=None, kpt_raw=None):
        self.last_info.update({
            "phase": self.phase, "substate": substate or self.substate,
            "mode": mode or self.mode, "action": action,
            "z": round(z if z is not None else (self._z_last or 0.0), 3),
            "dx": round(float(dx), 3), "dy": round(float(dy), 3),
            "sway": round(float(sway), 3), "heave": round(float(heave), 3),
            "surge": round(float(surge), 3), "yaw": round(float(yaw), 3),
            "pass": self._pass_cnt, "kpt": int(kpt),
            "kpt_raw": int(self._dbg_kpt_raw if kpt_raw is None else kpt_raw),
            "ratio": round(float(self._dbg_ratio if ratio is None else ratio), 3)})
        self._last_dof = (_dof_clip(surge), _dof_clip(sway),
                          _dof_clip(heave), _dof_clip(yaw))
        self.uart.send_dof(*self._last_dof)

    def _set_quality_info(self, sight):
        """本帧质量/线索诊断量（只写 last_info，不改变下发 DOF）。

        Q/Qmem/gain 分别是：后验质量分、喂下一帧先验的历史证据、kpt_memory 实际
        用到的帧级增益（gain<1 ⇔ 质量分真的压了这一帧的融合增益）。
        """
        q = sight.quality
        mem = self._per.kpt_mem
        self.last_info.update({
            "Q": round(float(q.total), 3) if q is not None else 0.0,
            "Qmem": round(float(self._per.tracker.q_mem), 3)
            if self._per.tracker is not None else 0.0,
            "gain": round(float(mem.gain), 3) if mem is not None else 1.0,
            "caution": round(float(sight.caution), 3),
            "cue": sight.cue or "",
            "cue_mode": cue_mode(self._cuecfg),
            "visible": "".join("1" if v else "0" for v in sight.mask),
            "rms": round(float(sight.rms), 2) if sight.rms is not None else 0.0,
            "agree": "" if sight.agree is None else bool(sight.agree)})

    def _start_through(self):
        self.phase = PH_THROUGH
        self.substate = ""
        self._through_frames = 0
        self._lost_cnt = 0
        self._z_guard = False
        self._reset_lateral_pids()

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
        if self._kpt_mem is not None:
            self._kpt_mem.reset()

    # ------------------------------------------------------------ 主入口
    def process(self, frame, now_ms, frame_id=None, captured_at=None):
        """单帧推进。frame_id/captured_at 可选（v1.3）：
        * 不传：按"每次调用=一帧"（旧调用方式完全兼容，行为不变）；
        * 传了：启用唯一帧/帧龄/断点守卫（重复帧不计数、过期帧只估计不计确认）。"""
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

        stamp = self._make_stamp(now_ms, frame_id, captured_at)
        verdict = self._validator.validate(stamp, float(now_ms) / 1000.0)
        self._age, self._frame_kind = verdict.age, verdict.kind
        self._countable = bool(verdict.countable)   # 过期帧 usable=True 但 countable=False
        self.last_info.update({"age": round(verdict.age, 3),
                               "frame_kind": verdict.kind})
        if not verdict.usable:
            # 重复/乱序/非法帧：不估计、不计数；沿用上一帧指令保持连续
            self.uart.send_dof(*self._last_dof)
            self.last_info["action"] = "hold_frame(%s)" % verdict.kind
            return self._tail(now_ms)
        if verdict.discontinuity:
            # 采集断点：只重置"正在累计的确认"（不动相位、不动速度/阶梯）
            self._center_cnt = 0
            self._cross_cnt = 0
            self._hold_cnt = 0
            self._st_cue.reset()

        dets = self.hub.detect_list(self.name, frame)
        self.last_dets = dets
        det = pick_gate(dets, self._per.conf_thr)
        self._step(det, now_ms, verdict)

        return self._tail(now_ms)

    def _tail(self, now_ms):
        if (self._start_ms is not None and
                now_ms - self._start_ms >= self._G.timeout_ms):
            self._finish("timeout")
        self.last_info["status"] = S.STATUS_DONE if self._finished \
            else S.STATUS_RUNNING
        return self.last_info["status"]

    def _make_stamp(self, now_ms, frame_id, captured_at):
        """帧标识：不传时按"每次调用一帧"自增（后向兼容）。"""
        if frame_id is None:
            self._auto_id += 1
            frame_id = self._auto_id
        if captured_at is None:
            captured_at = float(now_ms) / 1000.0
        return FrameStamp(int(frame_id), float(captured_at))

    def _finish(self, reason):
        self._finished = True
        self._reason = reason
        self.uart.neutral()
        self.last_info.update({"status": S.STATUS_DONE, "reason": reason,
                               "pass": self._pass_cnt})
        if S.DEBUG:
            print("[GATE] DONE(%s) pass=%d frames=%d" % (reason, self._pass_cnt,
                                                          self.frames))

    # ------------------------------------------------------------ 单帧逻辑
    def _step(self, det, now_ms, verdict):
        """相位选路：THROUGH 直行 / 无目标 / 有目标 → 感知半场 → 位姿档或降级档。"""
        if self.phase == PH_THROUGH:
            self._tick_through(now_ms)      # 穿门=直行，不做横向微调
            return
        if det is None:
            self._tick_lost(now_ms)
            return
        self._lost_cnt = 0
        # 诊断量：本帧框占比 + 原始有效角点数（进叠加与 REACQUIRE 日志）
        self._dbg_ratio = float(det.w) / float(self.w)
        # 感知半场（含质量先验→融合、mode、线索、位姿、质量后验）
        sight = self._per.observe(det, now_ms, age=verdict.age,
                                  prev_pose=self._last_pose,
                                  z_guard=self._z_guard, prev_z=self._z_last)
        self._dbg_kpt_raw = sight.kpt_raw
        self._dbg_kpt = int((np.asarray(sight.kconf) >= self._per.conf_thr).sum()) \
            if sight.kconf is not None else 0
        self._kpt_s, self._kconf_s = sight.kpts, sight.kconf
        self._q, self._caution = sight.quality, sight.caution
        self._sp, self._dz = sight.sp, sight.dz
        self._cue, self._cue_reason = sight.cue, sight.cue_reason
        self._set_quality_info(sight)
        self.last_info["source"] = "pose" if sight.pose is not None else "degrade"

        if sight.pose is not None:
            self._z_guard = True
            self._z_last = sight.z
            self._on_pose(det, sight, now_ms)
            return
        # 本帧没有可用位姿：解除跳变判据，下个位姿重新建立基准（防“卡死”）；
        # 同时清零穿门确认计数 → 只有“连续帧”都判就近才算过门
        self._z_guard = False
        self._cross_cnt = 0
        self._pose_gap += 1
        if self._maybe_cue(sight, now_ms):      # 位姿长期不可用 → 线索接管
            return
        if sight.mode == MODE_WIDTH:
            self._on_width(det, now_ms, sight.ids)
            return
        self._on_coarse(det, now_ms)            # coarse / 其它退化

    # ---------------- full/p3p 位姿可用：对准 / 进近 ----------------
    def _on_pose(self, det, sight, now_ms):
        G = self._G
        mode, kpt = sight.mode, sight.n
        tvec = sight.pose[1]
        self._last_pose = sight.pose
        self._reacquire_cnt = 0          # 拿到可信位姿 → 后退重取计数清零
        self._pose_gap = 0               # v1.3：位姿恢复 → 线索接管门槛清零
        self.mode = mode
        z = float(sight.z)
        dx_m, dy_m = float(tvec.ravel()[0]), float(tvec.ravel()[1])
        # 横向微调（位姿档 PID，统一量纲：t_x/0.5 → ±1）；速度受质量分【谨慎度】缩放
        sway = -_dof_clip(self._pid_sway_pose.update(
            max(-1, min(1, dx_m / _ALIGN_SCALE)), now_ms) * self._sp)
        heave = -_dof_clip(self._pid_heave_pose.update(
            max(-1, min(1, dy_m / _ALIGN_SCALE)), now_ms) * self._sp)
        aligned = abs(dx_m) <= G.align.xy_m * self._dz and \
            abs(dy_m) <= G.align.xy_m * self._dz
        if z <= G.z.cross:
            # 穿门确认：单帧就近不冲（平面 PnP 偶发错解若给出 z≤cross，
            # 直接 THROUGH 会让船满速冲出去），连续 n 帧才判过门
            if self._countable:
                self._cross_cnt += 1
            need = int(G.z.get("cross_confirm_frames", 2))
            if self._cross_cnt >= max(1, need):
                # 冲刺：只前进，不带横向微调（微调已在上一帧做完）；
                # 线索**不掺和**穿门帧（已判定过门，直行穿越）
                self.last_info["cue_w"] = 0.0
                self._start_through()
                self._set_info("through", mode=mode, z=z,
                               surge=G.surge.through * self._sp,
                               dx=dx_m, dy=dy_m, kpt=kpt)
            else:
                # 冲刺前最后一帧：不前进，只做横向微调对准（可掺线索）
                self._set_info("center", mode=mode, z=z, dx=dx_m, dy=dy_m,
                               kpt=kpt,
                               **self._pose_dof(sight, now_ms, 0.0, sway, heave))
            return
        self._cross_cnt = 0

        if self.phase in (PH_SEARCH,):
            self.phase = PH_ALIGN
            self.substate = SUB_GOLDEN
            self._center_cnt = 0
        if self.phase == PH_ALIGN:
            self.substate = SUB_GOLDEN
            if aligned:
                if self._countable:
                    self._center_cnt += 1
                if self._center_cnt >= G.align.confirm_frames:
                    self.phase = PH_APPROACH
                    self.substate = ""
            else:
                self._center_cnt = 0
            self._set_info("center", mode=mode, z=z, dx=dx_m, dy=dy_m,
                           kpt=kpt,
                           **self._pose_dof(sight, now_ms, 0.0, sway, heave))
        elif self.phase == PH_APPROACH:
            # 进近两档（远→快 / 近→慢）；分档点 = z.slow_max
            # 注：z.fast_max 暂未分档（如需三档在此加中间档）
            if z > G.z.slow_max:
                surge = G.surge.fast * self._sp
            else:
                surge = G.surge.slow * self._sp
            self._set_info("forward_%s" % ("fast" if surge > G.surge.slow else "slow"),
                           mode=mode, z=z, dx=dx_m, dy=dy_m, kpt=kpt,
                           **self._pose_dof(sight, now_ms, surge, sway, heave))
        else:
            self._set_info("center", mode=mode, z=z, dx=dx_m, dy=dy_m,
                           kpt=kpt,
                           **self._pose_dof(sight, now_ms, 0.0, sway, heave))

    # ---------------- 位姿档 DOF 与线索的主导权混合（v1.5） ----------------
    def _pose_dof(self, sight, now_ms, surge, sway, heave, yaw=0.0):
        """位姿档 DOF；主导档（`cues.mode` = auto/cue）按权重与线索动作逐通道混合。

        返回 4 通道字典（供 `_set_info(**dof)`），并把本帧线索权重写进诊断 `cue_w`。
        `pose` 档下 `_cue_lead` 恒返回 (None, 0) → 与升级前逐位一致。
        """
        pose = {"surge": surge, "sway": sway, "heave": heave, "yaw": yaw}
        act, wc = self._cue_lead(sight, now_ms)
        self.last_info["cue_w"] = round(float(wc), 3)
        if act is None or wc <= 0.0:
            return pose
        if S.DEBUG:
            print("[GATE] 线索主导 wc=%.2f %s（位姿 Q=%.2f）"
                  % (wc, act, sight.quality.total if sight.quality else -1.0))
        return blend_dof(pose, cue_dof(act, self._cue_speed()), wc)
