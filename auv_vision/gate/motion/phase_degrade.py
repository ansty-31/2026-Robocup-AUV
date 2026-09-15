# -*- coding: utf-8 -*-
"""gate/motion/phase_degrade.py — 无完整位姿时的降级路径（GateTask 相位段实现之一）

从 `gate_task.py` 按功能拆出，方法体与原实现**逐行一致**（只拆文件，不改运动）：
  * `_on_width`     对向 2 角（width 档）：单目测距 + 中点/框心对中，远距允许慢 creep
  * `_on_coarse`    整框可用但角点不足：远/中/近三层仲裁（creep / HOLD / REACQUIRE）
  * `_enter_reacquire` / `_tick_reacquire`   闭环后退重取（退到框够小即停，
                    同一段连续超 max_times 次则放弃后退、改原地保持）

以 mixin 形式并入 GateTask：这些方法天然要读写 GateTask 的状态与工具
（`self._G`、`self._set_info`、`self._start_search`、`self._dbg_*`、`self._sp/_dz`、
两套横向 PID），用 mixin 才能保持"同一份状态、同一段逻辑"，而不是拷贝状态。
"""
from __future__ import annotations

import numpy as np

import base.settings as S
from gate.vision.gate_frontend import width_range_depth, bbox_center, MODE_WIDTH, MODE_COARSE
from gate.motion.phases import (PH_SEARCH, PH_ALIGN, PH_APPROACH,
                         SUB_CREEP, SUB_HOLD, SUB_REACQUIRE)


def _dof_clip(v):
    return float(max(-1.0, min(1.0, v)))


class DegradeTicks(object):
    """width / coarse / REACQUIRE 三个降级档的相位段。"""

    # ---------------- width（对向 2 角：测距+中点对中，可慢 creep） ----------------
    def _on_width(self, det, now_ms, ids):
        G = self._G
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
        aim_x = (u1 + u2) / 2.0
        cx, cy = bbox_center(det)
        dxn = (aim_x - self.w / 2.0) / (self.w / 2.0)
        dyn = (cy - self.h / 2.0) / (self.h / 2.0)
        sway = -_dof_clip(self._pid_sway_px.update(dxn, now_ms) * self._sp)
        heave = -_dof_clip(self._pid_heave_px.update(dyn, now_ms) * self._sp)
        aligned = abs(dxn) <= G.align.xy_m * self._dz and \
            abs(dyn) <= G.align.xy_m * self._dz
        if self.phase == PH_SEARCH:
            self.phase = PH_ALIGN
            self._center_cnt = 0
        if self.phase == PH_ALIGN:
            # 距门仍远且对中 → 允许慢 creep（有限信息下安全推进）
            if aligned and z > G.z.cross + 0.4:
                self.substate = SUB_CREEP
                self._center_cnt += 1
                surge = G.surge.creep * self._sp
                if self._center_cnt >= G.align.confirm_frames:
                    self.phase = PH_APPROACH
                    self.substate = ""
            else:
                self.substate = SUB_HOLD
                surge = 0.0
            self._set_info("creep" if surge > 0 else "hold", mode=MODE_WIDTH,
                           z=z, dx=dxn, dy=dyn, sway=sway, heave=heave,
                           surge=surge)
            return
        self._set_info("center", mode=MODE_WIDTH, z=z, dx=dxn, dy=dyn,
                       sway=sway, heave=heave)

    # ---------------- coarse（整门框可信/角不足）：三层仲裁 ----------------
    def _on_coarse(self, det, now_ms):
        G = self._G
        self.mode = MODE_COARSE
        ratio = float(det.w) / float(self.w)
        # 已在 REACQUIRE：闭环后退（退到框够小即停）或超时回 search
        if self.phase == PH_ALIGN and self.substate == SUB_REACQUIRE:
            self._tick_reacquire(now_ms, ratio)
            return
        cx, cy = bbox_center(det)
        dxn = (cx - self.w / 2.0) / (self.w / 2.0)
        dyn = (cy - self.h / 2.0) / (self.h / 2.0)
        sway = -_dof_clip(self._pid_sway_px.update(dxn, now_ms) * self._sp)
        heave = -_dof_clip(self._pid_heave_px.update(dyn, now_ms) * self._sp)
        aligned = abs(dxn) <= G.coarse.get("align_x", 0.05) * self._dz and \
            abs(dyn) <= G.coarse.get("align_y", 0.10) * self._dz

        if self.phase == PH_SEARCH:
            self.phase = PH_ALIGN
            self._hold_cnt = 0
        if self.phase == PH_APPROACH:
            # 进近中角全失 → 保守降回 ALIGN 仲裁（避免盲冲）
            self.phase = PH_ALIGN
            self._hold_cnt = 0

        if self.phase != PH_ALIGN:
            self._set_info("hold", z=self._z_last, dx=dxn, dy=dyn,
                           sway=sway, heave=heave, kpt=self._dbg_kpt)
            return

        if ratio < G.coarse.far_ratio:               # 远距小框 → creep 换取角点
            self.substate = SUB_CREEP
            surge = (G.surge.creep * self._sp) if aligned else 0.0
            self._set_info("creep" if surge > 0 else "center",
                           z=self._z_last, dx=dxn, dy=dyn,
                           sway=sway, heave=heave, surge=surge,
                           kpt=self._dbg_kpt)
        elif ratio <= G.coarse.near_ratio:           # 中距 → HOLD（无位姿即无进展）
            self.substate = SUB_HOLD
            if self._countable:
                self._hold_cnt += 1
            if self._hold_cnt >= G.hold.max_frames:
                self._enter_reacquire(now_ms, ratio)
            else:
                self._set_info("hold", z=self._z_last, dx=dxn, dy=dyn,
                               sway=sway, heave=heave, kpt=self._dbg_kpt)
        elif self._give_up_until_ms is not None and \
                now_ms < self._give_up_until_ms:
            # 刚放弃过后退：窗口内只原地保持（等位姿恢复或时间窗过期再试）
            self.substate = SUB_HOLD
            self._set_info("hold", z=self._z_last, dx=dxn, dy=dyn,
                           sway=sway, heave=heave, kpt=self._dbg_kpt)
        else:                                        # 近距装不下 → 后退重取
            self._enter_reacquire(now_ms, ratio)

    def _enter_reacquire(self, now_ms, ratio=None):
        """进入后退重取；**同一段里连续超 max_times 次就放弃后退，改原地保持**。

        倒影持续干扰时"退-进-退"会来回震荡（看着就是一直后退又一直对准）；
        退了好几次仍拿不到可用角点，说明再退也没用 → 停下来等（超时另算）。
        """
        G = self._G
        # 距上次后退够久 → 计数重新开始（否则一次倒影干扰会把后退永久锁死）
        reset_after = int(G.reacquire.get("reset_after_ms", 3000))
        if self._reacquire_last_ms is None or \
                now_ms - self._reacquire_last_ms >= reset_after:
            self._reacquire_cnt = 0
        self._reacquire_cnt += 1
        max_times = int(G.reacquire.get("max_times", 2))
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
        r0 = self._reacquire_ratio0
        stop_ratio = float(G.reacquire.get("stop_ratio", 0.75))
        if ratio is not None and r0 and ratio <= r0 * stop_ratio:
            if S.DEBUG:
                print("[GATE] REACQUIRE 已退够(ratio=%.2f<=%.2f=%.2f×%.2f) → 回对准"
                      % (ratio, r0 * stop_ratio, r0, stop_ratio))
            self.substate = SUB_HOLD
            self._hold_cnt = 0
            self._set_info("hold", z=self._z_last, kpt=self._dbg_kpt)
            return
        dur = now_ms - (self._reacquire_start or now_ms)
        if dur >= G.reacquire.max_ms:
            self._start_search()
            self._set_info("search")
            return
        # 后退方向/量可配（dof sign 需水池实测；负 surge = 后退）
        surge = -float(G.surge.reacquire) * self._sp
        self._set_info("reacquire", substate=SUB_REACQUIRE, surge=surge,
                       kpt=self._dbg_kpt)
