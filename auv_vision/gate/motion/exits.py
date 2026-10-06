# -*- coding: utf-8 -*-
"""gate/motion/exits.py — 出口与终局：THROUGH 计时、CREEP_THROUGH 计时、丢门判定、门口超时兜底
"""
from __future__ import annotations

import base.cfg.settings as S
from common.motion.search_scan import telemetry_yaw
from common.cfg.cfgnode import flag, merge, num, sub
from gate.motion.params import (PH_ALIGN, PH_APPROACH, PH_THROUGH, PH_CREEP_THROUGH, SUB_REACQUIRE,
                                _D_ALIGN, _D_LOITER, _D_Z, _D_SURGE, _D_TASK, _D_THROUGH,
                                _D_CREEP_THROUGH)

class GateExits(object):
    def _start_through(self, bypass_hdg=False):
        """进入 THROUGH：只前进、不微调（横向修正全部留在冲刺前完成）。

        速度 = `comm.gate.surge.through`。

        ★ 冲刺前的航向闸门（**只管「够近」这条出口**）：`comm.gate.through.require_align_deg`
        （默认 8.0；≤0 = 关闸）。本门 `_hdg_ok` 必须已经为真；它只由有效 full 帧的
        `abs(psi) <=` 阈值，或 HeadingAligner 真正执行完成设置。转向超时、GIVEUP、ABORTED，
        以及从未取得有效航向测量，都不放行这条出口。

        ⚠️ **两条出口是独立的**（2026-10-02 用户定）：门口超时兜底走 `bypass_hdg=True`，
        它自己已经有"在门口 + 在中央安全带内持续 N 秒"三重条件，**不再看航向**——
        否则 width/coarse 档（没有 PnP ⇒ 永远没有 ψ）会连兜底都冲不出去，卡在门口到超时。
        Args:
            bypass_hdg: True = 跳过航向闸门（仅供门口兜底那条出口用）。
        Returns:
            True = 真的进了 THROUGH。**各出口都必须看返回值**，别无条件当成功。
        """
        need = num(sub(self._G, "through"), "require_align_deg",
                   _D_THROUGH["require_align_deg"])
        # ★ 2026-10-05 用户定：**主出口重新加回居中闸门** —— 必须"居中成功"**连续 center_frames 帧**
        #   才许冲刺。⚠️ 阈值用**冲刺自己那套**（through.center_x/center_y = 0.20/0.25），
        #   比 align(0.15/0.20) **松** —— 运动抖动下要求那么严会永远冲不出去。
        #   兜底出口（bypass_hdg=True，门口超时那条救命路）**不受此限**。
        if not bypass_hdg:
            T0 = merge(sub(self._G, "through"), _D_THROUGH)
            cneed = int(num(T0, "center_frames", _D_THROUGH["center_frames"]))
            if cneed > 0 and int(getattr(self, "_center_ok_cnt", 0)) < cneed:
                self.last_info["through_block"] = "off_center(连续居中 %d/%d 帧)" % (
                    int(getattr(self, "_center_ok_cnt", 0)), cneed)
                return False
        hdg_enabled = bool(getattr(self._hdg, "enabled", True))
        # HDG 关闭时无需航向确认；启用时必须由 hdg DONE 或有效 full 帧的
        # |psi|<=require_align_deg 设置本门锁存标志。
        # 门口兜底（bypass_hdg=True）不受这条约束：它是独立的第二条出口。
        if not bypass_hdg and hdg_enabled and need > 0 and not getattr(self, "_hdg_ok", False):
            psi = self._hdg_deg
            detail = "none" if psi is None else "hdg=%+.1f>%.1f" % (psi, need)
            self.last_info["through_block"] = detail
            if not self._through_block_logged:
                self._through_block_logged = True
                print("[GATE] 拦下冲刺：本门航向尚未确认(%s)"
                      " → 留在原地继续对准/正航向（不切 THROUGH）" % detail)
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
                self._start_search(new_round=True)
                self._set_info("hold")
            return
        self._set_info("through", surge=self._through_speed())
    def _start_creep_through(self):
        """进入 CREEP_THROUGH：**慢速** creep 冲门（= 门口过门出口）。

        与 `through`（满速、正常过门）是**两条出口**：正常过门走 through，门口过门走
        creep_through。coarse/width 没有位姿 ⇒ 没有航向闸门；这里只做"慢速直行"，
        由 `creep_through.creep_ms`（默认 5s）计时，到点判过门。撞杆教训：门口不再满速冲。
        """
        self.phase = PH_CREEP_THROUGH
        self.substate = ""
        self._creep_through_start_ms = None
        self._lost_cnt = 0
        self._loiter_start_ms = None      # 已进入冲刺 → 门口的计时作废
        self._reset_lateral_pids()
        return True
    def _tick_creep_through(self, now_ms):
        """creep_through：只前进（`surge.creep`），时长 `gate.creep_through.creep_ms` 后判过门。"""
        G = self._G
        T = merge(sub(G, "creep_through"), _D_CREEP_THROUGH)
        sg = merge(sub(G, "surge"), _D_SURGE)
        if self._creep_through_start_ms is None:
            self._creep_through_start_ms = now_ms
        if (now_ms - self._creep_through_start_ms) >= num(T, "creep_ms", _D_CREEP_THROUGH["creep_ms"]):
            self._pass_cnt += 1
            if S.DEBUG:
                print("[GATE] creep_through 通过第 %d/%d 门"
                      % (self._pass_cnt, int(num(G, "pass_target", _D_TASK["pass_target"]))))
            if self._pass_cnt >= int(num(G, "pass_target", _D_TASK["pass_target"])):
                self._finish("pass")
            else:
                self._start_search(new_round=True)
                self._set_info("hold")
            return
        self._set_info("creep_through", surge=num(sg, "creep", _D_SURGE["creep"]))
    def _tick_lost(self, now_ms):
        G = self._G
        zc = merge(sub(G, "z"), _D_Z)
        sg = merge(sub(G, "surge"), _D_SURGE)
        # ★ 2026-10-04 用户定：**还没稳定锁上门** ⇒ 用公共慢扫**旋转搜索**
        #   （左 span → 右 2span → 左 2span …；手动 DOF + 遥测 yaw 闭环、睁眼非阻塞、有防转圈预算）。
        #   稳定锁上门之后就不再扫（交给视觉），与本任务 hdg 完全无关（hdg 仍走"下发目标角/下位机定角度"）。
        # ⚠️ 有「转完反向平移」窗口时必须**让位**：那个状态是靠 `_set_info` 推进的（唯一派发口），
        #   这里若早退去跑扫描，sway_back 永远走不完、窗口也关不掉（实测用例就是这么挂的）。
        if (self._post_sway_until_ms is None) and not getattr(self, "_lock_stable", False):
            self._lost_cnt += 1
            yaw = self._scan.step(now_ms, telemetry_yaw(self.uart))
            self._set_info("search", substate=self.substate, yaw=yaw)
            return
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
            near_r = num(sub(self._G, "z"), "near_lost_ratio", _D_Z["near_lost_ratio"])
            last_r = float(getattr(self, "_dbg_ratio", 0.0) or 0.0)
            if self._lost_cnt <= pose_hold:
                # 帧间防抖：沿用上一帧对中保持
                self._set_info("hold", z=self._z_last)
            elif last_r >= near_r:
                # ★ 2026-10-02 用户定：**门口丢角点 ⇒ 只 hold + 后退重取，不回 SEARCH**。
                #   在门口回 SEARCH 等于原地左右扫视，永远冲不过去（实船日志里就是这样卡的）。
                if S.DEBUG:
                    print("[GATE] 门口（最后占比 %.2f ≥ %.2f）丢角点 → 原地保持 + 后退重取（不回 SEARCH）"
                          % (last_r, near_r))
                self._enter_reacquire(now_ms, last_r)
                self._set_info("reacquire", z=self._z_last)
            else:
                # 远处丢检（占比小）才回 SEARCH 扫视
                self._start_search()
                self._set_info("hold")
            return
        if self.phase == PH_APPROACH:
            if self._lost_cnt <= pose_hold and self._z_last is not None:
                # 还在远处、只是短暂丢失：轻微后退换回重新锁定/更大视野（不带速度盲冲），
                # 退满防抖窗口仍无目标 → 转 SEARCH
                self._set_info("backward_slow", z=self._z_last,
                               surge=-num(sg, "lost_backward", _D_SURGE["lost_backward"]))
            elif float(getattr(self, "_dbg_ratio", 0.0) or 0.0) >= num(sub(self._G, "z"), "near_lost_ratio", _D_Z["near_lost_ratio"]):
                # 同上：**门口丢角点不回 SEARCH**，原地后退重取（APPROACH 也守同一条）
                self._enter_reacquire(now_ms, self._dbg_ratio)
                self._set_info("reacquire", z=self._z_last)
            else:
                self._start_search()
                self._set_info("hold")
            return
        # SEARCH：左右平移扫视（不许旋转，见 `_search_sweep`）
        sway = self._search_sweep(now_ms)
        self._set_info("search", sway=sway)
    def _loiter_commit(self, dxn, dyn, ratio, now_ms, kpt=None):
        """在门口超时兜底：**不看档位**，「人在门口 + 对准」持续太久就自己拍板直冲。

        2026-10-02 用户定：**机制不变，只把判据放宽一点**（门口常是 coarse；框占满画面时门心
        本就偏出画面，卡太紧会永远 commit 不了）。三条都成立才算命中：
        占比 ≥ `loiter.near_ratio`；|dxn| ≤ `loiter.dx_max`、|dyn| ≤ `loiter.dy_max`；
        连续 ≥ `loiter.timeout_ms`（任一条不成立就重新计时）。
        """
        L = merge(sub(self._G, "loiter"), _D_LOITER)
        if not flag(L, "enable", True):
            self._loiter_start_ms = None
            return False
        near_r = num(L, "near_ratio", num(sub(self._G, "z"), "near_lost_ratio", _D_Z["near_lost_ratio"]))
        r_now = float(ratio or 0.0)
        ok = (r_now >= near_r and
              abs(float(dxn)) <= num(L, "dx_max", _D_LOITER["dx_max"]) and
              abs(float(dyn)) <= num(L, "dy_max", _D_LOITER["dy_max"]))
        if not ok and r_now >= near_r:
            # ★ 近距历史判断（2026-10-03 用户定）：coarse/width 在门口、下视看不到红横杆、
            #   主视当前帧又没位姿时，用**本轮**最近采信的位姿（z 中位数）判"已到门口"。
            #   只取本轮：_start_search() 里已清 _pose_hist，换门不串。
            hm = num(L, "pose_hist_ms", _D_LOITER["pose_hist_ms"])
            hn = num(L, "pose_hist_min", _D_LOITER["pose_hist_min"])
            hist = [h for h in getattr(self, "_pose_hist", []) if (now_ms - h[0]) <= hm]
            if len(hist) >= int(hn):
                zs = sorted(h[1] for h in hist)
                z_m = zs[len(zs) // 2]
                zc = num(sub(self._G, "z"), "cross", _D_Z["cross"])
                if z_m <= zc:
                    if S.DEBUG:
                        print("[GATE] 下视/主视都无信息 → 按本轮 %d 帧历史判门口（z 中位=%.2f ≤ %.2f）"
                              % (len(hist), z_m, zc))
                    ok = True
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
            print("[GATE] 在门口停留 %.1fs 仍未过门（占比=%.2f dx=%+.2f dy=%+.2f）→ creep_through 慢速冲门"
                  % ((now_ms - self._loiter_start_ms) / 1000.0, r_now, float(dxn), float(dyn)))
        # ★ 2026-10-02 用户定：门口过门 = **creep_through**（慢速 creep），不再满速直冲（撞杆教训）。
        return self._start_creep_through()