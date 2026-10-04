# -*- coding: utf-8 -*-
"""gate/motion/channels.py — 通道与小件：唯一执行出口 `_set_info`、横向 PID、SEARCH 扫视、诊断量
"""
from __future__ import annotations

import numpy as np
from common.cfg.cfgnode import flag, merge, motion_num, num, sub
from gate.percept.gate_postproc import pick as postproc_pick
from gate.percept.gate_frontend import parse_kpt_mode, width_range_depth, MODE_FULL, MODE_P3P, MODE_WIDTH
from gate.percept.geometry import gate_pose
from gate.motion.params import (PH_SEARCH, PH_ALIGN, SUB_SWAY_BACK, _D_Z, _D_SURGE,
                                _D_KPT, _D_SEARCH, _D_HDG, _D_PNP, _D_LOCK, _D_SELECT)

def _dof_clip(v):
    return float(max(-1.0, min(1.0, v)))


class GateChannels(object):
    def _reset_lateral_pids(self):
        for p in (self._pid_sway_px, self._pid_heave_px):
            p.reset()
    def _lateral_out(self, dxn, now_ms):
        """居中阶段的水平修正：**只用 sway 平移**（全档位共用一个像素 PID）。
        位置误差该用平移修 —— sway 命令 = kp(8.0)·dxn，|dxn|>0.017 就能过执行器死区(0.138)；
        而且 yaw 是**方位**控制，把门拉到光轴上 ≠ 机身与门平行。姿态由 ALIGN.HDG 负责。"""
        return _dof_clip(self._pid_sway_px.update(dxn, now_ms))
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
            # **唯一退出判据**（用户 2026-09-30 定）：**当前门**（本帧选中的那扇 `_det_now`）里
            #   conf ≥ `vision.gate.keypoint.conf_thr` 的角点数 ≥ `hdg.post_sway_kpt_min`
            #   ⇒ 门重新进了视野、光轴对齐的目的达成 → 退出，交回视觉。
            _kmin = int(num(sub(self._G, "hdg"), "post_sway_kpt_min",
                            _D_HDG["post_sway_kpt_min"]) or 0)
            _kcur = self._post_sway_kpt_count()
            #   ★ 2026-10-02：窗口**开启那一帧**不许按角点数退出 —— 那一帧的检测是**转向之前**采的
            #   （转向现在整段在同一帧内跑完），拿它判"门回来了"会把窗口当帧清掉、补偿永远不发。
            #   要等**开启之后的新一帧**检测。
            _fresh = self.frames > int(getattr(self, "_post_sway_frame0", -1) or -1)
            if _fresh and (_kmin <= 0 or _kcur >= _kmin):
                self._post_sway_until_ms = None
                self.last_info["sway_exit"] = "kpt_off" if _kmin <= 0 else "kpt>=%d" % _kmin
            elif _now >= self._post_sway_until_ms:
                self._post_sway_until_ms = None         # 窗口到期 ⇒ **安全兜底**退出（绝不永久停摆）
                self.last_info["sway_exit"] = "expire"
            elif action == "through" or self._hdg.turning:
                #   平移优先级最低，**永远让位于转向与冲刺**（这条不是"同门判据"，是防回归的安全规则）：
                #   ① `through` 掉进下面 else 会被改写成 sway_back（surge/yaw 清零）最长一个窗口 ⇒ 冲刺被压住；
                #   ② 转向侧 `_turn_inner_loop` 每 20Hz 经 `_set_info` 发 yaw，窗口还活着就会把 yaw 清零
                #      ⇒ 转向推不动（不是死锁，但白等/触发满舵保护）。
                self._post_sway_until_ms = None
                self.last_info["sway_exit"] = "yield(%s)" % ("through" if action == "through" else "turning")
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
            # `dir`/`tgt`=本次转向方向与目标角、`hdg_entry`=入口（**只有 golden=居中达标**）。
            "hdg_dir": self._hdg.last_dir,
            "hdg_tgt": round(float(self._hdg.last_target_deg or 0.0), 1)})
        self.uart.send_dof(_dof_clip(surge), _dof_clip(sway),
                           _dof_clip(heave), _dof_clip(yaw))
    def _speed(self, tier):
        """进近速度档：`gate.surge.<tier>` 显式覆盖 → `motion.surge_<tier>`（**与撞球共用**）→ 代码兜底。"""
        return num(sub(self._G, "surge"), tier, motion_num("surge_" + tier))
    def _through_speed(self):
        """本次冲刺速度 = `comm.gate.surge.through`。"""
        return num(sub(self._G, "surge"), "through", _D_SURGE["through"])
    def _start_search(self):
        # 2026-10-03 用户定：**删掉 SEARCH 扫视** —— 直接回 ALIGN 原地等门（不左右扫）
        self.phase = PH_ALIGN
        self.substate = ""
        self._post_sway_until_ms = None      # 换门/重新搜索 → 反向平移状态作废
        self._hdg_f = None
        self._hdg_deg = None
        self._hdg_ms = None
        self._hdg_turns = 0               # 新门 ⇒ 逐小步逼近的次数重新计
        self._hdg_turns_last_ms = None
        self._hdg.reset()
        self._hdg_done = not flag(self._hdg_cfg, "enable", True)
        self._hdg_ok = False              # 新门重新确认航向
        self._hdg_ok_cnt = 0              # ★ 连续达标帧数清零（ok_frames 判据用）
        self._hdg_skip_logged = False
        self._inside_since_ms = None       # 贴脸出口的短确认计时（换门/重新搜索清零）
        self._pose_hist = []               # 最近采信的位姿历史 [(ms, z, dxn, dyn)]（换门清零）
        self._relock_z_ref = None          # 重新搜索 = 换门，重锁保护作废
        self._relock_ratio_ref = None
        self._relock_logged = False
        self._search_entry_ms = None
        self._last_pose = None
        self._z_last = None
        self._z_guard = False
        self._cross_cnt = 0
        self._lost_cnt = 0
        self._hold_cnt = 0
        self._locked_det = None            # ★ 换门/重新搜索 → 解锁，允许重新 near 选门
        self._lock_miss = 0
        if self._kpt_mem is not None:
            self._kpt_mem.reset()
    def _search_sweep(self, now_ms):
        """SEARCH：左右平移扫视，**每轮时长按 2 的次幂递增**（用户 2026-09-27 定）。
        一轮（k=0,1,2,…）的波形（S = sweep_s·2^k，p = pause_s）：
        段0 [0, S)             → +sway（右移）
        段1 [S, S+p)           → 0（停：让检测有静止帧）
        段2 [.., 2S+p)         → -sway（左移）
        段3 余下               → 0（停）
        **正反必须等时长**（遥测没有横向位置反馈，单向平移会一路漂到池壁）；
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
        # ★ 固定搜索时长：过完门/丢门后最多扫 max_ms，到点就不再扫（保持静止，等门自己出现）
        max_ms = num(s, "max_ms", 0) * 1000.0
        if max_ms > 0 and t > max_ms:
            return 0.0
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
    def _post_sway_kpt_count(self):
        """**当前门**（本帧选中的那扇 `_det_now`）里 conf ≥ `vision.gate.keypoint.conf_thr` 的角点数。

        postsway 的**唯一**退出判据（用户 2026-09-30 定）：看到当前门的 N 个角点 = 门重新进了
        视野、"对齐光轴"的目的达成。本帧没检出（门还在画外）= 0 ⇒ 继续反向平移 + 缓慢后退。
        用**当帧原始**角点置信度，不是记忆后/几何过滤后的个数。
        """
        d = self._det_now
        kc = None if d is None else getattr(d, "kpt_conf", None)
        if kc is None:
            return 0
        thr = num(sub(self._V, "keypoint"), "conf_thr", _D_KPT["conf_thr"])
        return int((np.asarray(kc) >= thr).sum())

    def _uart_yaw(self):
        """下位机回传的绝对航向（度；没有就 None）。"""
        tel = getattr(self.uart, "telemetry", None)
        return None if tel is None else getattr(tel, "yaw_deg", None)
    def _down_sees_red_bar(self):
        """下视红色横向横杆（门框底部）是否可见。仅在 `vision.gate.down.enable` 且本帧有下视帧时判。"""
        if not getattr(self, "_down_enabled", False):
            return False
        frame = getattr(self, "_down_frame", None)
        if frame is None:
            return False
        from gate.percept.down_view import detect_red_bar
        return detect_red_bar(frame)
    def _z_est(self, det, conf_thr):
        """估计一个检测门的距离 z（**选门用，近者优先**）。

        full/p3p → PnP；width（对向 2 角）→ `fx·W/Δu`；其余 → 框宽代理 `fx·W/w`。None = 估不出。
        """
        kp = getattr(det, "kpts", None)
        kc = getattr(det, "kpt_conf", None)
        if kp is not None and kc is not None and len(kp) >= 4:
            mode, ids = parse_kpt_mode(kp, kc, conf_thr)
            if mode in (MODE_FULL, MODE_P3P) and len(ids) >= 3:
                obj3s = self.obj3[ids]
                img2s = np.asarray(kp)[ids]
                res = gate_pose(self.camera, obj3s, img2s, prev=None,
                                reproj_thr=float(_D_PNP["reproj_px"]),
                                z_bounds=(float(_D_PNP["z_min"]), float(_D_PNP["z_max"])),
                                refine=bool(_D_PNP["refine"]))
                if res is not None:
                    return float(res[1].ravel()[2])
            elif mode == MODE_WIDTH and len(ids) == 2:
                z = width_range_depth(float(kp[ids[0]][0]), float(kp[ids[1]][0]),
                                      float(self.camera.fx), self.frame_w_m)
                if z is not None:
                    return float(z)
        if det.w > 0:
            return float(self.camera.fx * self.frame_w_m / det.w)
        return None
    def _pick_gate(self, dets):
        """选目标门 = **near（按有效距离最小）**。

        ★ 2026-10-04 用户定：**用 k 做一致性检验**（k = z × 框占比）。
        同一扇门 k ≈ 常数，理论值 k_true = fx·frame_w/画面宽 = 0.560（门框 70cm）。
        实测 full 0.599(k/k_true=1.07) ✓、p3p 0.457(0.82) ✓、而"远处门 z 崩小"那次
        k=0.23(0.41) ✗ —— 一眼可辨。所以 **k/k_true < k_lo_ratio ⇒ 这个 z 崩了**，
        改用框占比反推的 z（k_true/占比）顶上；k 正常则用实测 z。
        """
        conf_thr = num(sub(self._V, "keypoint"), "conf_thr", _D_KPT["conf_thr"])
        gates = [d for d in dets if getattr(d, "kind", None) == "gate"]
        if not gates:
            return None
        k_true = float(self.camera.fx) * float(self.frame_w_m) / float(self.w or 1)
        k_lo = float(num(sub(self._G, "select"), "k_lo_ratio", _D_SELECT["k_lo_ratio"]))
        best, best_z = None, None
        for d in gates:
            z = self._z_est(d, conf_thr)
            ratio = (float(d.w) / float(self.w)) if (self.w and getattr(d, "w", 0)) else 0.0
            if ratio > 1e-6 and k_true > 0:
                k_meas = (float(z) * ratio) if z is not None else None
                if k_meas is None or k_meas < k_lo * k_true:
                    z_eff = k_true / ratio          # z 崩溃/测不出 ⇒ 用框占比反推
                else:
                    z_eff = float(z)
            else:
                z_eff = z
            if best is None or (z_eff is not None and (best_z is None or z_eff < best_z)):
                best, best_z = d, z_eff
        return best
