# -*- coding: utf-8 -*-
"""gate/motion/modes.py — 各档位处理：位姿档 / width 档 / coarse 档 + 退避重取与重锁守卫
"""
from __future__ import annotations

import numpy as np
import base.cfg.settings as S
from common.cfg.cfgnode import merge, num, sub, req, req_flag, req_node, MissingCfg
from gate.percept.gate_frontend import (width_range_depth,
                                              bbox_center,
                                              MODE_FULL,
                                              MODE_WIDTH,
                                              MODE_COARSE)
from gate.percept.geometry import gate_normal_angles_deg
from gate.motion.params import (PH_SEARCH, PH_ALIGN, PH_APPROACH, SUB_GOLDEN, SUB_CREEP,
                                SUB_HOLD, SUB_REACQUIRE)
from gate.motion.channels import _dof_clip

class GateModes(object):
    def _hdg_ok_tick(self, align_deg):
        """航向「已 OK」锁存判据（★ 2026-10-04 用户定：**持续稳定才锁存**）。"""
        if align_deg <= 0 or abs(float(self._hdg_deg)) <= align_deg:
            self._hdg_ok_cnt = getattr(self, "_hdg_ok_cnt", 0) + 1
            need_ok = int(req(sub(self._G, "hdg"), "ok_frames"))
            if self._hdg_ok_cnt >= max(1, need_ok):
                self._hdg_ok = True
        else:
            self._hdg_ok_cnt = 0

    def _on_pose(self, det, pose, now_ms, mode, kpt):
        """位姿档：**位姿只提供深度 z 与"可信"这一事实**；居中一律用**像素误差**。"""
        G = self._G
        al = req_node(G, "align")
        zc = req_node(G, "z")
        sg = req_node(G, "surge")
        rvec, tvec = pose
        if self._relock_guard(float(tvec.ravel()[2])):
            # 这是"另一个门"的位姿：不接受、不改基准，本帧原地 hold（不发推力）
            self._set_info("hold", z=self._z_last)
            return
        self._last_pose = pose
        self._reacquire_cnt = 0          # 拿到可信位姿 → 后退重取计数清零
        self.mode = mode
        z = float(tvec.ravel()[2])
        self._z_last = z
        self._pose_ms = now_ms
        c = self.camera.project(np.zeros((1, 3), np.float32), rvec, tvec)[0]
        _cx0, _cy0 = self._center_ref()          # ★ 基准=主点（光轴），不是画面中心
        dxn = float((c[0] - _cx0) / (self.w / 2.0))
        # ★ 2026-10-07 用户定：**纵向零点**在**算 dyn 的地方**统一减掉（全局测量约定）。
        #   它是相机装高 + 常态抬头造成的固定偏置（实测 dyn 中位 −0.30，逐档 −0.20/−0.28/−0.34）；
        #   不在这里减，`through.center_y`/`loiter.dy_max`/heave 的零点就全错位。
        dyn = float((c[1] - _cy0) / (self.h / 2.0)) - self._dy_target()
        hist = getattr(self, "_pose_hist", None)
        if hist is None:
            hist = self._pose_hist = []
        hist.append((now_ms, z, dxn, dyn))
        del hist[:-8]
        if mode == MODE_FULL:
            psi_deg, _pit_deg = gate_normal_angles_deg(rvec, tvec)
            # 快 EMA(≈4 帧)：**只用于日志/诊断**（`hdg_fast`），不再参与判据
            self._hdg_f = psi_deg if self._hdg_f is None else \
                self._hdg_f + 0.4 * (psi_deg - self._hdg_f)
            # ★★ 2026-10-07 用户定：**判据与目标角都用平滑后的 ψ**。
            #   实测：ψ 的帧间跳 7.8° 是真实航向(tyaw)帧间跳 0.33° 的 **24 倍**
            #   ⇒ ψ 在该频段是噪声主导；拿未平滑的 ψ 去比 tol，等于让噪声决定要不要起转、
            #   以及转多少度（实船 f17 就是被 26.4° 的跳变推进 tol 窗口里、下发 0° 空转一次）。
            #   窗口 `comm.gate.hdg.psi_ema_frames`（缺键 ⇒ 1 帧 = 不平滑，不崩）。
            _k = int(num(sub(self._G, "hdg"), "psi_ema_frames", 1) or 1)
            _a = 1.0 if _k <= 1 else 2.0 / (float(_k) + 1.0)
            self._hdg_s = psi_deg if getattr(self, "_hdg_s", None) is None else \
                self._hdg_s + _a * (psi_deg - self._hdg_s)
            self._hdg_deg = self._hdg_s
            self._hdg_ms = now_ms
            # 诊断：把**未参与判据**的快 EMA 也写进日志（判据用 `hdg`，两者一起看就知道平滑吞掉了多少）
            self.last_info["hdg_fast"] = round(float(self._hdg_f), 1)
            # 航向确认使用冲刺闸门的配置阈值；真正完成的 hdg DONE 也会保持确认。
            align_deg = req(sub(G, "through"), "require_align_deg")
            self._hdg_ok_tick(align_deg)          # 航向"已 OK"锁存（连续 ok_frames 帧判据）
        sway, yaw = self._lateral_out(dxn, now_ms), 0.0
        heave = -_dof_clip(self._pid_heave_px.update(dyn, now_ms))
        aligned = self._aligned(dxn, dyn, al)          # ★ 分档 + 纵向零点（见 channels._aligned）
        #   不是 align 的 px_x/px_y（用户："最后进冲刺的居中闸不要像 align 那样严"）。
        # ★ 2026-10-07 用户定：过门居中闸 = **本档位（位姿档）自己的居中范围**；
        #   与 through.center_x/center_y 取 AND（见 channels._center_ok）。
        _ok_c = self._center_ok(dxn, dyn, al)
        self._center_ok_cnt = (getattr(self, "_center_ok_cnt", 0) + 1) if _ok_c else 0
        dx_m, dy_m = dxn, dyn            # 日志里 dx/dy **统一是像素归一化**
        cross = req(zc, "cross")
        if z <= cross:
            self._cross_cnt += 1
            need = int(req(zc, "cross_confirm_frames"))
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
            #    不看画面）；这里只处理"已经在 ALIGN、需要决定要不要起转"的正常路径。
            self.substate = SUB_GOLDEN
            if self._loiter_commit(dxn, dyn, float(det.w) / float(self.w), now_ms):
                return
            # ★★ 2026-10-07 用户定：**"最近 confirm_frames 帧同时满足 align 与机身稳"（并行）**。
            #   原来只数 align（`_center_cnt`），末尾几帧机身还在晃就已经起转。
            #   ⚠️ 为什么不是"align 5 帧 + 再稳 5 帧"（串行 10 帧）：实船 385 帧里
            #   `align ∧ 机身静` 的 **10 帧窗口恒为 0**，串行等于永远不起转。
            if aligned and self._body_quiet(now_ms):
                self._center_cnt += 1
                if self._center_cnt >= int(req(al, "confirm_frames")):
                    if self._hdg_ready(mode, now_ms):
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
            # ★★ 2026-10-06 用户定：**舍弃 forward_fast** —— 到达 `z.cross` 前**一律用 creep 试探**。
            #   为什么：撞门/撞池壁的代价不可逆（"任何部位露出水面即本次比赛停止"），
            #   而门口那段 PnP/width 的 z 最不可靠；`forward_fast` 就是"满速盲冲"的来源。
            #   档位可配：`gate.approach.tier`（默认 creep；改回 fast/slow 不影响其它逻辑）。
            _ap = sub(self._G, "approach")
            _tier = str((_ap or {}).get("tier") or "")
            if not _tier:
                raise MissingCfg("cfg 缺 comm.gate.approach.tier（代码已无兜底）")
            surge = self._speed(_tier)
            self._set_info("forward_%s" % _tier,
                           mode=mode, z=z, dx=dx_m, dy=dy_m,
                           sway=sway, heave=heave, surge=surge, kpt=kpt,
                           hdg=self._hdg_deg)
        else:
            self._set_info("center", mode=mode, z=z, dx=dx_m, dy=dy_m,
                           kpt=kpt)
    def _on_width(self, det, now_ms, ids):
        """width 档：只有对向 2 角（上边或下边），信息只够"水平中点 + 框心竖直"，不解 PnP。"""
        if self._tick_hdg_degraded(now_ms):
            return
        if self._down_sees_red_bar():
            # ★ 下视看到门框底部红色横杆 ⇒ 机身已到门口/进门 ⇒ creep_through 慢速冲门
            self._start_creep_through()
            self._set_info("creep_through", mode=MODE_WIDTH)
            return
        G = self._G
        W = req_node(G, "width")
        al = req_node(G, "align")
        sg = req_node(G, "surge")
        self.mode = MODE_WIDTH
        kp = self._kpt_s if self._kpt_s is not None else det.kpts
        u1, v1 = kp[ids[0]]
        u2, v2 = kp[ids[1]]
        z = width_range_depth(u1, u2, float(self.camera.fx), self.frame_w_m)
        if z is None:
            self._on_coarse(det, now_ms)
            return
        _pnp = sub(self._V, "pnp")   # z 截断用**视觉侧**的 pnp 上下限
        z = float(np.clip(z, req(_pnp, "z_min"), req(_pnp, "z_max")))
        # ★★ 2026-10-06 用户定：**"2.5m 开外完全不相信"必须贯彻始终**（不只是 full/p3p 档），
        #   否则就是 bug。width 档的 z 是从 `fx·W/Δu` 反推的、比 PnP 更粗，**最需要这道闸**，
        #   而它此前是唯一没闸的档（实船：width 直接采纳了 z=3.691 > dist_max 2.5，
        #   然后一路追着第三/第四扇门跑）。这里补上与 `_on_pose` 完全同款的一道。
        if self._relock_guard(z):
            self._set_info("hold", z=self._z_last)      # 不采纳：判为丢门，原地 hold
            return
        self._z_last = z
        # ★ 2026-10-05 用户定：**把 `z ≤ cross` + 连续帧计数也放进 width 档**。
        #   实船：门口必然掉角点，位姿帧(full/p3p)变得稀疏断续，`z≤cross` 的窗口里只有 2 帧
        #   位姿 ⇒ `cross_confirm_frames=4` 永远凑不满 ⇒ 冲不出去。而 width 档**有真实 z**
        #   （`width_range_depth` = fx·W/Δu），让它参与计数，门口那 4 帧正好凑满。
        #   冲刺仍走**主出口**（带航向闸 + 居中闸），连续帧确认的保护不削弱。
        zc0 = req_node(self._G, "z")
        cross0 = req(zc0, "cross")
        if z <= cross0:
            self._cross_cnt += 1
            need0 = int(req(zc0, "cross_confirm_frames"))
            if self._cross_cnt >= max(1, need0) and self._start_through():
                aim0 = (u1 + u2) / 2.0
                self._set_info("through", mode=MODE_WIDTH, z=z,
                               surge=self._through_speed(),
                               dx=(aim0 - self._center_ref()[0]) / (self.w / 2.0),
                               kpt=self._dbg_kpt)
                return
        else:
            self._cross_cnt = 0
        aim_x = (u1 + u2) / 2.0
        cx, cy = bbox_center(det)
        _cx0, _cy0 = self._center_ref()          # ★ 基准=主点
        dxn = (aim_x - _cx0) / (self.w / 2.0)
        dyn = (cy - _cy0) / (self.h / 2.0) - self._dy_target()   # ★ 纵向零点（全局约定）
        # 对中判据（归一化像素偏差）：**分档**（width 有自己的带，见 channels._aligned）
        aligned = self._aligned(dxn, dyn, W, "align_x", "align_y")
        #   不是 align 的 px_x/px_y（用户："最后进冲刺的居中闸不要像 align 那样严"）。
        # ★ 2026-10-07 用户定：过门居中闸 = **本档位（width）自己的居中范围**；
        #   与 through.center_x/center_y 取 AND（见 channels._center_ok）。
        _ok_c = self._center_ok(dxn, dyn, W, "align_x", "align_y")
        self._center_ok_cnt = (getattr(self, "_center_ok_cnt", 0) + 1) if _ok_c else 0
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
            if aligned and z > req(W, "z_max"):
                self.substate = SUB_CREEP
                self._center_cnt += 1
                surge = req(sg, "creep")
                if self._center_cnt >= int(req(al, "confirm_frames")):
                    self.phase = PH_APPROACH
                    self.substate = ""
            else:
                self.substate = SUB_HOLD
                surge = 0.0
            self._set_info("creep" if surge > 0 else "hold", mode=MODE_WIDTH,
                           z=z, dx=dxn, dy=dyn, sway=sway, heave=heave,
                           surge=surge, yaw=yaw)
            return
        # ⚠️ 上面的 creep 与 `_loiter_commit` 两个出口都在 `if self.phase == PH_ALIGN:`
        self._set_info("center", mode=MODE_WIDTH, z=z, dx=dxn, dy=dyn,
                       sway=sway, heave=heave, yaw=yaw)
    def _on_coarse(self, det, now_ms):
        """coarse 档：角点不足，只有整框可信。**只有未对准才后退**"""
        if self._tick_hdg_degraded(now_ms):
            return
        if self._down_sees_red_bar():
            # ★ 下视看到门框底部红色横杆 ⇒ 机身已到门口/进门 ⇒ creep_through 慢速冲门
            self._start_creep_through()
            self._set_info("creep_through", mode=MODE_COARSE)
            return
        G = self._G
        C = req_node(G, "coarse")
        H = req_node(G, "hold")
        sg = req_node(G, "surge")
        self.mode = MODE_COARSE
        ratio = float(det.w) / float(self.w)
        # 已在 REACQUIRE：闭环后退（退到框够小即停）或超时回 search
        if self.phase == PH_ALIGN and self.substate == SUB_REACQUIRE:
            self._tick_reacquire(now_ms, ratio)
            return
        cx, cy = bbox_center(det)
        _cx0, _cy0 = self._center_ref()          # ★ 基准=主点
        dxn = (cx - _cx0) / (self.w / 2.0)
        dyn = (cy - _cy0) / (self.h / 2.0) - self._dy_target()   # ★ 纵向零点（全局约定）
        # 对中判据：**coarse 档自己的带**（`coarse.align_x/align_y`）+ 纵向零点（见 channels._aligned）
        aligned = self._aligned(dxn, dyn, C, "align_x", "align_y")

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
                           surge=req(sg, "creep"), kpt=self._dbg_kpt)
            return

        if ratio < req(C, "far_ratio"):       # 远距小框：没对准就别冲
            self.substate = SUB_CREEP
            self._set_info("center", z=self._z_last, dx=dxn, dy=dyn,
                           sway=sway, heave=heave, yaw=yaw, kpt=self._dbg_kpt)
        elif ratio <= req(C, "near_ratio"):   # 中距 → HOLD（无位姿即无进展）
            self.substate = SUB_HOLD
            self._hold_cnt += 1
            if self._hold_cnt >= int(req(H, "max_frames")):
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
        """进入后退重取；同一段里连续超 max_times 次就**放弃后退，改原地保持**。"""
        G = self._G
        R = req_node(G, "reacquire")
        # 距上次后退够久 → 计数重新开始（否则一次倒影干扰会把后退永久锁死）
        reset_after = int(req(R, "reset_after_ms"))
        if self._reacquire_last_ms is None or \
                now_ms - self._reacquire_last_ms >= reset_after:
            self._reacquire_cnt = 0
        self._reacquire_cnt += 1
        max_times = int(req(R, "max_times"))
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
        R = req_node(G, "reacquire")
        sg = req_node(G, "surge")
        r0 = self._reacquire_ratio0
        stop_ratio = req(R, "stop_ratio")
        if ratio is not None and r0 and ratio <= r0 * stop_ratio:
            if S.DEBUG:
                print("[GATE] REACQUIRE 已退够(ratio=%.2f<=%.2f=%.2f×%.2f) → 回对准"
                      % (ratio, r0 * stop_ratio, r0, stop_ratio))
            self.substate = SUB_HOLD
            self._hold_cnt = 0
            self._set_info("hold", z=self._z_last, kpt=self._dbg_kpt)
            return
        dur = now_ms - (self._reacquire_start or now_ms)
        if dur >= req(R, "max_ms"):
            self._start_search()
            self._set_info("search")
            return
        surge = -req(sg, "reacquire")
        self._set_info("reacquire", substate=SUB_REACQUIRE, surge=surge,
                       kpt=self._dbg_kpt)
    def _relock_guard(self, z):
        """**z 跳变保护**（2026-10-02 用户定）：命中 ⇒ 这一帧的位姿**不采纳**，当作"门丢了"处理"""
        jump = req(sub(self._G, "z"), "relock_z_jump_m")
        away = req(sub(self._G, "z"), "relock_away_m")
        dist_max = req(sub(self._G, "z"), "dist_max_m")
        prev = self._z_last
        ref = getattr(self, "_relock_z_ref", None)
        zf = float(z)
        why = None
        if dist_max > 0 and zf > dist_max:
            why = "z=%.2f m > %.2f m ⇒ 2.5m 开外**完全不相信**（无条件）" % (zf, dist_max)
        # 只挡"向上/更远"；向下跳（更快接近）是正常的，不挡。三条：
        elif jump > 0 and prev is not None and (zf - float(prev)) >= jump:
            why = "z 从 %.2f m 向上跳到 %.2f m（≥%.2f m）⇒ 不是当前门" % (float(prev), zf, jump)
        elif jump > 0 and ref is not None and (zf - float(ref)) >= jump:
            why = "z=%.2f 比丢门前的 %.2f m 远 ≥%.2f m ⇒ 不是那扇门" % (zf, float(ref), jump)
        elif away > 0 and ref is not None and zf > away:
            why = "z=%.2f > %.2f m（丢门重锁）⇒ 判为下一个门" % (zf, away)
        if why:
            if not getattr(self, "_relock_logged", False):
                self._relock_logged = True
                if S.DEBUG:
                    print("[GATE] z 跳变保护：%s → 判为丢门（本帧位姿不采纳）" % why)
            return True
        if ref is not None:
            self._relock_z_ref = None          # 同一个门 → 保护解除
        return False

    def _relock_guard_ratio(self):
        """**无位姿时的重锁保护（纯框占比）**：与 postsway 早期"占比退路"同思路。"""
        ref = getattr(self, "_relock_ratio_ref", None)
        ratio = float(getattr(self, "_dbg_ratio", 0.0) or 0.0)
        far = req(sub(self._G, "z"), "relock_far_ratio") or 0.0
        if ref is None or far <= 0 or ratio <= 1e-6:
            return False
        if ratio < float(far) * float(ref):
            if not getattr(self, "_relock_logged", False):
                self._relock_logged = True
                if S.DEBUG:
                    print("[GATE] 丢门重锁(占比)：本帧 %.3f < %.2f×丢门前 %.3f → 判为另一个门，拒收该帧"
                          % (ratio, float(far), float(ref)))
            return True
        self._relock_z_ref = None          # 同一个门 → 保护解除
        self._relock_ratio_ref = None
        return False
