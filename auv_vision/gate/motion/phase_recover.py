# -*- coding: utf-8 -*-
"""gate/motion/phase_recover.py — 搜索 / 丢目标 / 穿门收尾 + 恢复线索接管

从 `gate_task.py` 按功能拆出，方法体与原实现**逐行一致**（只拆文件，不改运动）：
  * `_search_yaw_pulse`  SEARCH 的原地旋转脉冲（转 spin_s → 停 pause_s）
  * `_tick_lost`         丢目标：ALIGN 防抖保持 → SEARCH；APPROACH 近距丢失=真过门、
                         远处短暂丢失=轻微后退
  * `_tick_through`      穿门：只前进不微调，目标消失确认满 → 计数/下一门/结束
  * `_maybe_cue` / `_on_cue`  恢复线索接管（位姿长期不可用时的低速方向动作；
                         动作确认用共用件 `common/streak.py`，速度受谨慎度缩放）

以 mixin 形式并入 GateTask（与 phase_degrade.py 同理：共享同一份任务状态）。
"""
from __future__ import annotations

import base.settings as S
from gate.motion.phases import (PH_SEARCH, PH_ALIGN, PH_APPROACH, PH_THROUGH,
                         SUB_CUE, SUB_REACQUIRE)
from gate.vision.gate_frontend import MODE_COARSE
from gate.data.cues import (cue_dof, cue_weight, blend_dof, cue_mode, is_lead,
                           BACKWARD, TURN_RIGHT, TURN_LEFT, DESCEND, ASCEND)  # noqa: F401
# 方向符号表在 `gate/data/cues.py::cue_dof`（必须水池实测，先用 cues.record_only 核对）。


class RecoverTicks(object):
    """SEARCH / 丢目标 / THROUGH 收尾 与 线索接管。"""

    def _search_yaw_pulse(self, now_ms):
        """SEARCH：原地旋转脉冲（转 spin_s → 停 pause_s）。"""
        spin = self._G.search.spin_s * 1000
        pause = self._G.search.pause_s * 1000
        if self._search_entry_ms is None:
            self._search_entry_ms = now_ms
        ph = (now_ms - self._search_entry_ms) % (spin + pause)
        return self._G.search.yaw if ph <= spin else 0.0

    # ---------------- 恢复线索 ↔ 降级链互相自检 ----------------
    def _cue_phase_ok(self):
        """线索允许说话的相位：SEARCH/ALIGN；`cues.allow_approach` 时也含 APPROACH。

        （THROUGH 永不介入：那时已判定过门、直行穿越，任何横向动作都是干扰。）
        """
        if self.phase in (PH_SEARCH, PH_ALIGN):
            return True
        return bool(self._cuecfg.allow_approach) and self.phase == PH_APPROACH

    def _cue_speed(self):
        """线索速度：主导档（auto/cue）用 `speed_lead`，`pose` 档用 `speed`。

        `pose` 档乘 `_sp`（位姿差 → 线索也慢，保守，与 v1.3 逐位一致）；
        主导档默认**不乘**——谨慎度本就是"位姿链的质量分"，用它拖慢线索等于让
        不可信的位姿链继续决定线索的力度。
        """
        cfg = self._cuecfg
        if not is_lead(cfg):
            return float(cfg.speed) * self._sp       # pose 档：与 v1.3 逐位一致
        v = float(cfg.speed_lead)
        return v * self._sp if cfg.speed_uses_caution else v

    def _maybe_cue(self, sight, now_ms):
        """**无位姿**时的线索接管：相位允许 + 连续无位姿够久 + 未设 record_only。"""
        cue = sight.cue
        if cue is None or self._cuecfg.record_only:
            return False
        if not self._cue_phase_ok():
            return False
        if self._pose_gap < int(self._cuecfg.cue_after_frames):
            return False
        self.last_info["source"] = "cue"
        self._on_cue(sight, now_ms)
        return True

    def _cue_lead(self, sight, now_ms):
        """**有位姿**时的线索主导权：返回 (动作|None, 权重 wc)，不改 last_info。

        权重来自 `cues.cue_weight()`（三档见 `cues.cue_mode`）：`auto` 按位姿质量 Q
        连续交接、`cue` 直接主导；权重 < `lead_min_w` 就不掺和（免得两链各出一半力来回抖）。
        `pose` 档（含别名）恒返回 0 → 位姿在位时线索不驱动。
        """
        cfg = self._cuecfg
        if sight.cue is None or cfg.record_only:
            return None, 0.0
        if not is_lead(cfg):
            return None, 0.0                 # pose 档：线索只在位姿不可用时兜底
        if not self._cue_phase_ok():
            return None, 0.0
        q = sight.quality.total if sight.quality is not None else None
        wc = cue_weight(cfg, q_pose=q, cue=sight.cue, pose_ok=True, z=sight.z)
        if wc < float(cfg.lead_min_w):
            return None, 0.0
        if self._countable:                  # 主导档同样不认重复/过期帧
            self._st_cue.add(sight.cue, now_ms / 1000.0)
        if not self._st_cue.ready(now_ms / 1000.0, cfg.cue_confirm_s,
                                  cfg.cue_confirm_frames):
            return None, 0.0
        return sight.cue, wc

    def _on_cue(self, sight, now_ms):
        """低速方向动作；确认由共用 Streak 提供（避免单帧噪声直接推动）。"""
        action = sight.cue
        self.substate = SUB_CUE
        self.mode = MODE_COARSE
        if self._countable:
            self._st_cue.add(action, now_ms / 1000.0)
        if self._st_cue.ready(now_ms / 1000.0, self._cuecfg.cue_confirm_s,
                              self._cuecfg.cue_confirm_frames):
            self._set_info("cue_%s" % action.lower(), z=self._z_last,
                           kpt=self._dbg_kpt, **cue_dof(action, self._cue_speed()))
        else:
            self._set_info("cue_wait", z=self._z_last, kpt=self._dbg_kpt)

    # ---------------- 丢目标 / 穿门 ----------------
    def _tick_lost(self, now_ms):
        G = self._G
        if self._countable:
            self._lost_cnt += 1
        self._z_guard = False             # 丢目标→解除 z 跳变基准
        self._cross_cnt = 0
        if self.phase in (PH_ALIGN,) and self.substate == SUB_REACQUIRE:
            self._tick_reacquire(now_ms)      # 已丢目标：无 ratio，按时间退完
            return
        if self.phase == PH_ALIGN:
            if self._lost_cnt <= G.pose_hold_frames:
                # 帧间防抖：沿用上一帧对中保持
                self._set_info("hold", z=self._z_last)
            else:
                self._start_search()
                self._set_info("search")
            return
        if self.phase == PH_APPROACH:
            near_lost_m = float(G.z.get("near_lost_m", 1.0))
            if self._z_last is not None and self._z_last <= near_lost_m:
                # 进近已到近距(z≤near_lost_m)却整门丢失：门已占满视野/机身进入门框，
                # 这是**真过门**的典型现象 → 直接判过门并直行穿越（不等防抖、不后退）；
                # 否则只会退回 SEARCH，永远数不到过门（旧版这里靠 PnP 错解偶然给出
                # z≤cross 才“过门”）
                if S.DEBUG:
                    print("[GATE] 近距丢失(z=%.2f≤%.2f) → 判过门"
                          % (self._z_last, near_lost_m))
                self._start_through()
                self._set_info("through", surge=G.surge.through * self._sp)
            elif self._lost_cnt <= G.pose_hold_frames and self._z_last is not None:
                # 还在远处、只是短暂丢失：**轻微后退**（学撞球：丢目标时不带速度盲冲，
                # 退一点换取重新锁定/更大视野），退满防抖窗口仍无目标 → 转 SEARCH
                self._set_info("backward_slow", z=self._z_last,
                               surge=-float(G.surge.lost_backward))
            else:
                self._start_search()
                self._set_info("search")
            return
        # SEARCH
        yaw = self._search_yaw_pulse(now_ms)
        self._set_info("search", yaw=yaw)

    def _tick_through(self, now_ms):
        """穿门：**只前进、不微调**（横向修正全部留在冲刺前完成）。"""
        G = self._G
        if self._countable:
            self._through_frames += 1
            if self._lost_cnt is not None:
                self._lost_cnt += 1
        # 已过 Z_pass：直行穿越；目标消失 ≥confirm → 判机身过门
        if self._lost_cnt >= G.through.confirm_frames:
            self._pass_cnt += 1
            if S.DEBUG:
                print("[GATE] 通过第 %d/%d 门" % (self._pass_cnt, G.pass_target))
            if self._pass_cnt >= G.pass_target:
                self._finish("pass")
            else:
                self._start_search()
                self._set_info("search")
            return
        self._set_info("through", surge=G.surge.through * self._sp)
