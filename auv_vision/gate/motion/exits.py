# -*- coding: utf-8 -*-
"""gate/motion/exits.py — 出口与终局：THROUGH 计时、CREEP_THROUGH 计时、丢门判定、门口超时兜底
"""
from __future__ import annotations

import base.cfg.settings as S
from common.motion.search_scan import telemetry_yaw
from common.cfg.cfgnode import flag, merge, num, sub, req, req_flag, req_node, MissingCfg
from gate.motion.params import PH_ALIGN, PH_APPROACH, PH_THROUGH, PH_CREEP_THROUGH, SUB_REACQUIRE

class GateExits(object):
    def _start_through(self, bypass_hdg=False):
        """进入 THROUGH：只前进、不微调（横向修正全部留在冲刺前完成）。
        ⚠️ **两条出口是独立的**（2026-10-02 用户定）：门口超时兜底走 `bypass_hdg=True`，"""
        need = req(sub(self._G, "through"), "require_align_deg")
        # ★ 2026-10-05 用户定：**主出口重新加回居中闸门** —— 必须"居中成功"**连续 center_frames 帧**
        if not bypass_hdg:
            T0 = req_node(self._G, "through")
            cneed = int(req(T0, "center_frames"))
            if cneed > 0 and int(getattr(self, "_center_ok_cnt", 0)) < cneed:
                self.last_info["through_block"] = "off_center(连续居中 %d/%d 帧)" % (
                    int(getattr(self, "_center_ok_cnt", 0)), cneed)
                return False
        hdg_enabled = bool(getattr(self._hdg, "enabled", True))
        # HDG 关闭时无需航向确认；启用时必须由 hdg DONE 或有效 full 帧的
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
        #   进入冲刺那一帧由 `_on_pose` 记日志（不走 `_tick_through`），这里先把段名写上，
        #   否则"进入帧"的 through_stage 是 None（复盘时看不出它属于哪一段）。
        self.last_info["through_stage"] = "fast"
        self._loiter_start_ms = None      # 已进入冲刺 → 门口的计时作废
        self._reset_lateral_pids()
        return True
    def _tick_through(self, now_ms):
        """穿门：只前进、不微调（横向修正全部留在冲刺前完成）。

        ★★ 2026-10-07 用户定：**两段式冲刺**（`cross` 从 1.2 提到 1.5 ⇒ 起冲点更远，
        必须走完的距离从 ~1.2m 涨到 ~2.5m；单段满速要么冲不够、要么过冲撞对面）：

          段① `fast_ms`：用 `surge.through`（0.60）快冲 —— 把门甩到身后；
          段② 剩下的到 `confirm_ms`：**降到 `surge.creep`（0.20）保持**，把艇体尾段平推过去，
               同时把"过冲撞池壁/撞门"的能量降下来。

        ⚠️ **两个时长都还没标定**（这趟日志里 THROUGH 一次都没触发 ⇒ 没有任何冲刺实测）。
           理论估算：需要走完 `cross(1.5) + 艇长(0.6~0.8) + 余量(0.3) ≈ 2.4~2.6 m`；
           `surge=0.30` 实测稳态 17.3 cm/s，按有效推力外推到 0.60 ⇒ **0.29~0.49 m/s**（按阻力
           平方根则偏 0.29，线性则偏 0.49）⇒ 走完要 **4.9~8.9 s**。
           速度本身有 1.7× 不确定度，所以"确保穿过"和"不过冲"这两件事**开环时间法不可能同时保证**
           —— 现取值**优先保证穿过**（错过一次过门 = 0 分），下水实测后再按实际走过的距离收。
        `fast_ms = 0` ⇒ 整段都用 `surge.through`（= 旧行为，一行回退）。
        """
        G = self._G
        T = req_node(G, "through")
        self._through_frames += 1
        self._lost_cnt += 1               # confirm_ms<=0 时按帧数兜底
        if self._through_start_ms is None:
            self._through_start_ms = now_ms
        elapsed = now_ms - self._through_start_ms
        ms = req(T, "confirm_ms")
        if ms > 0:
            done = elapsed >= ms
        else:
            done = self._lost_cnt >= max(1, int(req(T, "confirm_frames")))
        # 直行穿越；满足结束条件 → 判机身过门
        if done:
            self._pass_cnt += 1
            if S.DEBUG:
                print("[GATE] 通过第 %d/%d 门"
                      % (self._pass_cnt, int(req(G, "pass_target"))))
            if self._pass_cnt >= int(req(G, "pass_target")):
                self._finish("pass")
            else:
                self._start_search(new_round=True)
                self._set_info("hold")
            return
        _fast_ms = float(num(T, "fast_ms", 0.0) or 0.0)
        if _fast_ms > 0 and elapsed >= _fast_ms:
            stage, surge = "slow", self._speed("creep")     # 段②：降速保持
        else:
            stage, surge = "fast", self._through_speed()    # 段①：快冲
        self.last_info["through_stage"] = stage
        self._set_info("through", surge=surge)
    def _start_creep_through(self):
        """进入 CREEP_THROUGH：**慢速** creep 冲门（= 门口过门出口）。"""
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
        T = req_node(G, "creep_through")
        sg = req_node(G, "surge")
        if self._creep_through_start_ms is None:
            self._creep_through_start_ms = now_ms
        if (now_ms - self._creep_through_start_ms) >= req(T, "creep_ms"):
            self._pass_cnt += 1
            if S.DEBUG:
                print("[GATE] creep_through 通过第 %d/%d 门"
                      % (self._pass_cnt, int(req(G, "pass_target"))))
            if self._pass_cnt >= int(req(G, "pass_target")):
                self._finish("pass")
            else:
                self._start_search(new_round=True)
                self._set_info("hold")
            return
        self._set_info("creep_through", surge=req(sg, "creep"))
    def _tick_lost(self, now_ms):
        G = self._G
        zc = req_node(G, "z")
        sg = req_node(G, "surge")
        self._lost_cnt += 1
        self._z_guard = False             # 丢目标→解除 z 跳变基准
        # ★ 2026-10-07：**参照捕获必须在任何早退之前** —— 下面"去扫描"那条路以前直接 return，
        #   于是丢门参照从来没被记下来过（`preloss` 模型会因此完全失效）。
        if getattr(self, "_relock_z_ref", None) is None and self._z_last is not None:
            self._relock_z_ref = float(self._z_last)   # 记住丢门前的 z（重锁时判是不是同一个门）
        if getattr(self, "_relock_ratio_ref", None) is None and self._dbg_ratio > 1e-6:
            self._relock_ratio_ref = float(self._dbg_ratio)   # z 拿不到时的退路：丢门前的框占比
        # ⚠️ 有「转完反向平移」窗口时必须**让位**：那个状态是靠 `_set_info` 推进的（唯一派发口），
        if (self._post_sway_until_ms is None) and not getattr(self, "_lock_stable", False):
            yaw = self._scan.step(now_ms, telemetry_yaw(self.uart))
            self._set_info("search", substate=self.substate, yaw=yaw)
            return
        self._cross_cnt = 0
        pose_hold = int(req(G, "pose_hold_frames"))
        if self.phase in (PH_ALIGN,) and self.substate == SUB_REACQUIRE:
            self._tick_reacquire(now_ms)      # 已丢目标：无 ratio，按时间退完
            return
        if self.phase == PH_ALIGN:
            # ⚠️ 这里**不再中止正航向**：转向是自包含动作（`_step` 顶部已把转向接管，
            if self._post_sway_until_ms is not None and now_ms < self._post_sway_until_ms:
                #   ⚠️ 这里**不要**自己把 action 写成 "sway_back"：`_set_info` 的平移块会在窗口
                #   仍开着时把它覆盖成 sway_back；而窗口**恰在这一帧关闭**时，预置的名字撤不回来，
                #   于是"状态已结束"却还记/发一帧 sway_back 指令（用户 2026-10-07 指出）。
                #   传中性动作，让唯一的仲裁口去决定。
                self._set_info("hold", z=self._z_last)
                return
            near_r = req(sub(self._G, "z"), "near_lost_ratio")
            last_r = float(getattr(self, "_dbg_ratio", 0.0) or 0.0)
            if self._lost_cnt <= pose_hold:
                # 帧间防抖：沿用上一帧对中保持
                self._set_info("hold", z=self._z_last)
            elif last_r >= near_r:
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
                self._set_info("backward_slow", z=self._z_last,
                               surge=-req(sg, "lost_backward"))
            elif float(getattr(self, "_dbg_ratio", 0.0) or 0.0) >= req(sub(self._G, "z"), "near_lost_ratio"):
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
        """在门口超时兜底：**不看档位**，「人在门口 + 对准」持续太久就自己拍板直冲。"""
        L = req_node(self._G, "loiter")
        if not flag(L, "enable", True):
            self._loiter_start_ms = None
            return False
        near_r = num(L, "near_ratio", req(sub(self._G, "z"), "near_lost_ratio"))
        r_now = float(ratio or 0.0)
        ok = (r_now >= near_r and
              abs(float(dxn)) <= req(L, "dx_max") and
              abs(float(dyn)) <= req(L, "dy_max"))
        if not ok and r_now >= near_r:
            hm = req(L, "pose_hist_ms")
            hn = req(L, "pose_hist_min")
            hist = [h for h in getattr(self, "_pose_hist", []) if (now_ms - h[0]) <= hm]
            if len(hist) >= int(hn):
                zs = sorted(h[1] for h in hist)
                z_m = zs[len(zs) // 2]
                zc = req(sub(self._G, "z"), "cross")
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
        wait = req(L, "timeout_ms")
        if (now_ms - self._loiter_start_ms) < wait:
            return False
        if S.DEBUG:
            print("[GATE] 在门口停留 %.1fs 仍未过门（占比=%.2f dx=%+.2f dy=%+.2f）→ creep_through 慢速冲门"
                  % ((now_ms - self._loiter_start_ms) / 1000.0, r_now, float(dxn), float(dyn)))
        # ★ 2026-10-02 用户定：门口过门 = **creep_through**（慢速 creep），不再满速直冲（撞杆教训）。
        return self._start_creep_through()