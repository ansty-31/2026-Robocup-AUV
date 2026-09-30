# -*- coding: utf-8 -*-
"""gate/motion/exits.py — 出口与终局：三条直冲出口、THROUGH 计时、丢门判定、门口超时兜底
"""
from __future__ import annotations

import base.cfg.settings as S
from common.cfg.cfgnode import flag, merge, num, sub
from gate.motion.params import PH_ALIGN, PH_APPROACH, PH_THROUGH, SUB_REACQUIRE, _D_LOITER, _D_Z, _D_SURGE, _D_TASK, _D_THROUGH

class GateExits(object):
    def _start_through(self):
        """进入 THROUGH：只前进、不微调（横向修正全部留在冲刺前完成）。

        速度 = `comm.gate.surge.through`（三个出口都走这一条，见 `_through_speed`）。

        ★ 2026-09-26 新增**冲刺前的航向闸门**（用户定：「冲刺前确保历史最后一次的 psi 收敛在
        8 度以内才行」）：`comm.gate.through.require_align_deg`（默认 8.0；≤0 = 关闸）。
        判据 = `self._hdg_deg`（最近一次 full 帧测到的 psi，EMA 平滑后的历史值）：
          · 从没测到过（None）→ **放行**（没得判）；
          · |psi| ≤ 阈值 → 放行；
          · 否则**拦下**：不切 THROUGH、写日志字段 `through_block`，任务留在原相位继续对准。
        Returns:
            True = 真的进了 THROUGH。**三个出口都必须看返回值**，别无条件当成功。
        """
        need = num(sub(self._G, "through"), "require_align_deg",
                   _D_THROUGH["require_align_deg"])
        if need and need > 0 and self._hdg_deg is not None and abs(self._hdg_deg) > need:
            self.last_info["through_block"] = "hdg=%+.1f>%.1f" % (self._hdg_deg, need)
            if not self._through_block_logged:
                self._through_block_logged = True
                print("[GATE] 拦下冲刺：最近一次 psi=%+.1f° > %.1f°"
                      " → 留在原地继续对准/正航向（不切 THROUGH）"
                      % (self._hdg_deg, need))
            return False
        self.last_info["through_block"] = None
        self._hdg_abort("进入冲刺")
        self.phase = PH_THROUGH
        self.substate = ""
        self._through_frames = 0
        self._lost_cnt = 0
        self._z_guard = False
        self._through_start_ms = None     # 第一帧 tick 时打时间戳（见 _tick_through）
        self._loiter_start_ms = None      # 已进入冲刺 → 门口的计时作废
        self._reset_lateral_pids()
        return True
    def _tick_through(self, now_ms):
        """穿门：只前进、不微调（横向修正全部留在冲刺前完成）。

        结束条件优先**时长** `through.confirm_ms`（帧数语义下冲刺时长随 fps 漂，而"能不能
        冲出去"取决于跑了多远）；`confirm_ms<=0` → 退回旧的帧数语义 `confirm_frames`。
        ⚠️ confirm_ms 要按实测航速标定：需要冲的距离 ≈ 触发距离(2026-09-23 后的 z.cross=1.10 m
        读数=真值) + 机身长度 L → confirm_ms ≈ 1000·(1.10+L)/v，再留 30% 余量。
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
    def _tick_lost(self, now_ms):
        G = self._G
        zc = merge(sub(G, "z"), _D_Z)
        sg = merge(sub(G, "surge"), _D_SURGE)
        self._lost_cnt += 1
        self._z_guard = False             # 丢目标→解除 z 跳变基准
        if getattr(self, "_relock_z_ref", None) is None and self._z_last is not None:
            self._relock_z_ref = float(self._z_last)   # 记住丢门前的 z（重锁时判是不是同一个门）
        if getattr(self, "_relock_ratio_ref", None) is None and self._dbg_ratio > 1e-6:
            self._relock_ratio_ref = float(self._dbg_ratio)   # z 拿不到时的退路：丢门前的框占比
        self._cross_cnt = 0
        pose_hold = int(num(G, "pose_hold_frames", _D_TASK["pose_hold_frames"]))
        if self.phase in (PH_ALIGN,) and self.substate == SUB_REACQUIRE:
            self._tick_reacquire(now_ms)      # 已丢目标：无 ratio，按时间退完
            return
        if self.phase == PH_ALIGN:
            # ⚠️ 这里**不再中止正航向**：转向是自包含动作（`_step` 顶部已把转向接管，
            #   画面丢失/门转出视野都由它兜住）。能走到这一行的，转向一定已经结束。
            # ---- 转完补偿：刚转完那一下、又看不见门 → 朝转向反方向平移一小段（把门拉回视野）----
            if self._post_sway_until_ms is not None and now_ms < self._post_sway_until_ms:
                self._set_info("sway_back", z=self._z_last, sway=self._post_sway)
                return
            # 近距丢失 = 已过门（旧版这里一律防抖后回 SEARCH，于是"creep 到门口 → 整门丢失
            # → SEARCH → 原地不动"，永远数不到过门）。判据用**最后一次检测到的框占比**
            # （每帧更新、比 z 更可靠且全档位可用）；远处丢检（占比小）仍走 SEARCH。
            if self._lost_cnt <= pose_hold:
                # 帧间防抖：沿用上一帧对中保持
                self._set_info("hold", z=self._z_last)
            else:
                self._start_search()
                self._set_info("search")
            return
        if self.phase == PH_APPROACH:
            if self._lost_cnt <= pose_hold and self._z_last is not None:
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
