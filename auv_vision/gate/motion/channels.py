# -*- coding: utf-8 -*-
"""gate/motion/channels.py — 通道与小件：唯一执行出口 `_set_info`、横向 PID、SEARCH 扫视、诊断量
"""
from __future__ import annotations

import numpy as np

import base.cfg.settings as S
from common.cfg.cfgnode import flag, merge, motion_num, num, sub, req, req_flag, req_node, MissingCfg
from gate.percept.gate_postproc import pick as postproc_pick
from gate.percept.gate_frontend import parse_kpt_mode, width_range_depth, MODE_FULL, MODE_P3P, MODE_WIDTH
from gate.percept.geometry import gate_pose
from gate.motion.params import PH_SEARCH, PH_ALIGN, SUB_SWAY_BACK

def _dof_clip(v):
    return float(max(-1.0, min(1.0, v)))


class GateChannels(object):
    def _reset_lateral_pids(self):
        for p in (self._pid_sway_px, self._pid_heave_px):
            p.reset()
    def _lateral_out(self, dxn, now_ms):
        """居中阶段的水平修正：**只用 sway 平移**（全档位共用一个像素 PID）。"""
        return _dof_clip(self._pid_sway_px.update(dxn, now_ms))
    def _set_info(self, action, mode="", substate="",
                  z=None, dx=0.0, dy=0.0, sway=0.0, heave=0.0,
                  surge=0.0, yaw=0.0, kpt=0, ratio=None, kpt_raw=None,
                  hdg=None, hdg_skip=None, hdg_state=None, turn_deg=None, turn_dir=None):
        # hdg_skip：居中达标却跳过正航向的原因（只在那个瞬间有意义；传 None 表示本帧不涉及）
        if hdg_skip is not None:
            self.last_info["hdg_skip"] = hdg_skip
        if hdg_state is not None:
            self.last_info["hdg_state"] = str(hdg_state)
        # ★ 2026-10-07 用户定：**把"实际下发给下位机的转向角/方向"记进逐帧日志**。
        #   为什么：`yaw`(DOF) 与 `hdg_tgt` 恒为 0，转向走的是 `uart.request_turn(deg,dir)` 另一条路
        #   ⇒ 日志里查不到"到底下发了多少度"，只能靠猜（我据此把 15° 猜错过一次）。
        #   这两项**一旦写下就常驻** `last_info`（每次转向刷新），所以任一帧都能看到最近一次的下发量。
        if turn_deg is not None:
            self.last_info["turn_deg"] = round(float(turn_deg), 2)
        if turn_dir is not None:
            self.last_info["turn_dir"] = str(turn_dir)
        if self._post_sway_until_ms is not None:
            _now = self._now_ms if self._now_ms is not None else 0
            _kmin = int(req(sub(self._G, "hdg"), "post_sway_kpt_min") or 0)
            _kcur = self._post_sway_kpt_count()
            # ★★ 2026-10-06 用户定：**转向后先连停 settle 帧，再看画面**。
            #   实测（帧 17 转完 → 帧 18 仍报 kpt=4）：转向后前若干帧**还是转向前采的旧画面**
            #   （相机/流水线滞后），拿它判"门回来了"没有意义 —— 旧代码在下一帧就
            #   `_kcur(4) >= _kmin(2)` 立刻关窗 ⇒ **sway_back 一帧都没发**（整个日志 0 次）。
            #   所以：必须等**新画面**（`frames > frame0 + settle`）才允许按画面收手。
            _settle = int(req(sub(self._G, "hdg"), "post_sway_settle_frames") or 0)
            _f0 = int(getattr(self, "_post_sway_frame0", -1) or -1)
            _fresh = self.frames > (_f0 + _settle)
            # 收手判据（方案 A）：**门可见 且 已居中**才算"光轴对齐完成"。
            #   居中判据**调用 align 的** px_x/px_y（用户 2026-10-06 定），不另立一套。
            _al = req_node(self._G, "align")
            _dxn = abs(float(self.last_info.get("dx") or 0.0))
            _dyn = abs(float(self.last_info.get("dy") or 0.0))
            _centered = (_dxn <= req(_al, "px_x")
                         and _dyn <= req(_al, "px_y"))
            if _fresh and (_kmin <= 0 or (_kcur >= _kmin and _centered)):
                self._post_sway_until_ms = None
                self.last_info["sway_exit"] = ("kpt_off" if _kmin <= 0
                                               else "kpt>=%d且居中" % _kmin)
            elif _now >= self._post_sway_until_ms:
                self._post_sway_until_ms = None         # 窗口到期 ⇒ **安全兜底**退出（绝不永久停摆）
                self.last_info["sway_exit"] = "expire"
            elif action == "through" or self._hdg.turning:
                #   平移优先级最低，**永远让位于转向与冲刺**（这条不是"同门判据"，是防回归的安全规则）：
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
    def _center_ref(self):
        """**居中判据的基准 = 相机主点（光轴）**，不是画面几何中心。

        ★★ 2026-10-07 用户定 + 实测确认：前视相机 `cx,cy = (702.27, 409.14)`，
        而画面几何中心是 `(640, 360)` —— 差 **(+62.3, +49.1) px = 归一化 (0.0973, 0.1365)**。

        用画面中心当基准时：**门完美落在光轴上**（tvec=[0,0,z]）也会读出
        `dxn=0.0973 / dyn=0.1365` ⇒ 任何 `px_y < 0.1365` 的阈值**物理上永远无法满足**
        ⇒ `aligned` 恒 False ⇒ 转向链/居中闸全不启动（实船卡在门口一动不动就是这个）。

        "居中"的物理含义本来就是"与光轴对齐"，所以**横向**基准必须是主点。
        ⚠️ **纵向还有第二个偏置**（相机装得比门中心高 + 抬头），见 `_dy_target()`。
        """
        c = self.camera
        cx = float(getattr(c, "cx", 0.0) or (self.w / 2.0))
        cy = float(getattr(c, "cy", 0.0) or (self.h / 2.0))
        return cx, cy

    def _dy_target(self):
        """**纵向零点偏置** `comm.gate.align.dy_target`（归一化）。

        ★ 2026-10-07 用户定，实测依据（`log/rungate_20261006_postsway`，主点基准下）：
        `dyn` 的中位是 **−0.30**，且逐档分别为 full −0.200 / width −0.277 / coarse −0.337
        ⇒ 门**恒在光轴下方**（≈13cm @1.8m），物理来源是"相机比门中心高 + 常态抬头"。
        不把这个偏置搬掉，任何纵向阈值都在偏置上打滑（`heave` 也会永久单向压）。
        缺键 ⇒ 0.0（= 旧语义，**不崩**）。
        """
        al = sub(self._G, "align") or {}
        return float(al.get("dy_target", 0.0) or 0.0)

    def _aligned(self, dxn, dyn, node=None, kx="px_x", ky="px_y"):
        """**对中判据（分档）**：`|dxn| ≤ kx` 且 `|dyn| ≤ ky`。

        ⚠️ `dyn` 传进来时**已经**减掉了纵向零点（`comm.gate.align.dy_target`，见各档的算法处）
        —— 那是**全局的测量约定**，不是某一条判据的私事：`through.center_y`、`loiter.dy_max`、
        `heave` 的 PID 目标全都吃同一个 `dyn`，只在 align 里减会让它们全部错位。

        ★ 2026-10-07 用户定：**各档位的居中依赖不同，不共用一套带**（原因见 cfg 注释）。
        `node` 给档位自己的 cfg 节点（如 `comm.gate.width` / `comm.gate.coarse`），
        它没有那两个键时**回退到 `comm.gate.align`**（缺键不崩）。
        """
        al = sub(self._G, "align") or {}
        nd = node or {}
        ax = nd.get(kx, al.get("px_x", 0.15))
        ay = nd.get(ky, al.get("align_y", al.get("px_y", 0.15)))
        return abs(float(dxn)) <= float(ax) and abs(float(dyn)) <= float(ay)

    def _center_ok(self, dxn, dyn, node=None, kx="px_x", ky="px_y"):
        """**过门（THROUGH）触发前的居中闸**：必须落在**本档位自己的**居中范围内。

        ★ 2026-10-07 用户定："务必在各自要求的居中范围内触发过门"。
        实现 = `本档位对中带 × through.loose`（默认 1.0 = 完全等同该档要求），
        再与 `through.center_x/center_y`（2026-10-05 定的"抗抖动"上限）取 **AND** ——
        两道都过才许冲，所以这条改动**只可能更严**，不会把旧闸放松。
        """
        f = float((sub(self._G, "through") or {}).get("loose", 1.0) or 1.0)
        al = sub(self._G, "align") or {}
        nd = node or {}
        ax = float(nd.get(kx, al.get("px_x", 0.15))) * f
        ay = float(nd.get(ky, al.get("align_y", al.get("px_y", 0.15)))) * f
        tc = sub(self._G, "through") or {}
        return (abs(float(dxn)) <= min(ax, float(tc.get("center_x", 1.0)))
                and abs(float(dyn)) <= min(ay, float(tc.get("center_y", 1.0))))

    def _body_quiet(self, now_ms=None):
        """**机身是否稳**（转向前置条件）：遥测的横滚/纵倾/航向**帧间变化**是否都在门限内。

        ★ 2026-10-07 用户定："转之前一定是 align 成功，然后稳住 —— 重点是稳定机身，
        居中本身问题不大。本质是解决 align 末尾还在乱动、不够稳健。"

        实测依据（触发帧 f17）：|Δtyaw|=0.13°、|Δtrol|=0.24°、|Δtpit|=0.09°（全在中位以下），
        而 |Δψ|=26.4° ⇒ 那次起转是**图像 ψ 的噪声**推出去的，**机身其实是静的**。
        所以"稳机身"必须看**遥测**，不能看图像；图像侧的 ψ 用 `hdg.psi_ema_frames` 平滑解决。

        门限 `comm.gate.hdg.body_delta_max_deg`（0 = 关掉这条判据，退回旧行为）。
        ⚠️ **无遥测时 fail-open（返回 True）**：不能因为遥测缺失就把转向永久锁死；
        第一帧没有基准也返回 False（这一帧不算稳）。
        """
        node = sub(self._G, "hdg") or {}
        thr = float(node.get("body_delta_max_deg", 0.0) or 0.0)
        if thr <= 0:
            return True
        tel = getattr(self.uart, "telemetry", None)
        cur = None if tel is None else (getattr(tel, "roll_deg", None),
                                        getattr(tel, "pitch_deg", None),
                                        getattr(tel, "yaw_deg", None))
        prev = getattr(self, "_body_prev", None)
        self._body_prev = cur
        if cur is None or any(v is None for v in cur):
            if not getattr(self, "_body_quiet_warned", False):
                self._body_quiet_warned = True
                if S.DEBUG:
                    print("[GATE] 遥测缺横滚/纵倾/航向 ⇒ 『机身稳』判据 fail-open（不拦转向）")
            return True
        if prev is None or any(v is None for v in prev):
            return False
        return all(abs(float(a) - float(b)) <= thr for a, b in zip(cur, prev))

    def _speed(self, tier):
        """进近速度档：`gate.surge.<tier>` 显式覆盖 → `motion.surge_<tier>`（**与撞球共用**）→ 代码兜底。

        ⚠️ `num(node, key, default)` 的 default 是**先求值**的，所以 `motion_num("surge_xxx")`
        在 `motion.surge_xxx` 不存在时会先抛 KeyError（实船踩：creep 档就崩在这）。
        这里改成"**先试、失败再退**"，保证任何档位名都不会炸。
        """
        v = sub(self._G, "surge")
        if v is not None and v.get(tier) is not None:
            return float(v.get(tier))
        try:
            return float(motion_num("surge_" + tier))
        except Exception:
            raise MissingCfg("cfg 缺 comm.gate.surge.%s（代码已无兜底）" % tier)
    def _through_speed(self):
        """本次冲刺速度 = `comm.gate.surge.through`。"""
        return req(sub(self._G, "surge"), "through")
    def _start_search(self, new_round=False):
        """回到"等门"状态。"""
        # 2026-10-03 用户定：**删掉 SEARCH 扫视** —— 直接回 ALIGN 原地等门（不左右扫）
        self.phase = PH_ALIGN
        self.substate = ""
        self._post_sway_until_ms = None      # 换门/重新搜索 → 反向平移状态作废
        self._hdg_f = None
        self._hdg_s = None            # ★ ψ 的**慢 EMA**（判据与目标角用；见 modes._on_pose）
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
        if new_round:
            self._pose_hist = []           # 最近采信的位姿历史（只有新一轮才清；丢门保留，门口判据要用）
        if new_round:
            # ★ 只有"新一轮"（过完门）才清判断远近的基准 —— 丢门时保留，重锁保护才有参照可依
            self._relock_z_ref = None
            self._relock_ratio_ref = None
            self._relock_logged = False
        self._search_entry_ms = None
        self._last_pose = None
        if new_round:
            self._z_last = None
        self._z_guard = False
        self._cross_cnt = 0
        self._lost_cnt = 0
        self._hold_cnt = 0
        self._locked_det = None            # ★ 换门/重新搜索 → 解锁，允许重新 near 选门
        self._lock_miss = 0
        if new_round:
            self._scan.reset()             # ★ 只有新一轮才重置扫描（丢门不重置，免得起"走圈"）
        if self._kpt_mem is not None:
            self._kpt_mem.reset()
    def _search_sweep(self, now_ms):
        """SEARCH：左右平移扫视，**每轮时长按 2 的次幂递增**（用户 2026-09-27 定）。"""
        s = req_node(self._G, "search")
        base = req(s, "sweep_s") * 1000.0
        pause = max(0.0, req(s, "pause_s") * 1000.0)
        if base <= 0:
            return 0.0
        cap = max(base, req(s, "sweep_max_s") * 1000.0)
        v = req(s, "sway")
        if self._search_entry_ms is None:
            self._search_entry_ms = now_ms
        t = now_ms - self._search_entry_ms
        # ★ 固定搜索时长：过完门/丢门后最多扫 max_ms，到点就不再扫（保持静止，等门自己出现）
        max_ms = req(s, "max_ms") * 1000.0
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
        """**当前门**（本帧选中的那扇 `_det_now`）里 conf ≥ `vision.gate.keypoint.conf_thr` 的角点数。"""
        d = self._det_now
        if d is None:
            _ld = getattr(self, "last_dets", None) or []
            _raw = getattr(self, "_dets_raw", None) or []
            d = _ld[0] if _ld else (_raw[0] if _raw else None)
        kc = None if d is None else getattr(d, "kpt_conf", None)
        if kc is None:
            return 0
        thr = req(sub(self._V, "keypoint"), "conf_thr")
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
        """估计一个检测门的距离 z（**选门用，近者优先**）。"""
        kp = getattr(det, "kpts", None)
        kc = getattr(det, "kpt_conf", None)
        if kp is not None and kc is not None and len(kp) >= 4:
            mode, ids = parse_kpt_mode(kp, kc, conf_thr)
            if mode in (MODE_FULL, MODE_P3P) and len(ids) >= 3:
                obj3s = self.obj3[ids]
                img2s = np.asarray(kp)[ids]
                res = gate_pose(self.camera, obj3s, img2s, prev=None,
                                reproj_thr=req(sub(self._V, "pnp"), "reproj_px"),
                                z_bounds=(req(sub(self._V, "pnp"), "z_min"),
                                          req(sub(self._V, "pnp"), "z_max")),
                                refine=req_flag(sub(self._V, "pnp"), "refine"))
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
        """选目标门 = **z 优先 + 同时相信"面积最大"**（★ 2026-10-05 用户定）。
        ⚠️ **k 一致性检验不在这里** —— 它和跳变保护同性质，是"**选定之后保护锁定**"用的，"""
        conf_thr = req(sub(self._V, "keypoint"), "conf_thr")
        gates = [d for d in dets if getattr(d, "kind", None) == "gate"]
        if not gates:
            return None
        k_true = float(self.camera.fx) * float(self.frame_w_m) / float(self.w or 1)
        # ★★ 2026-10-06 用户定：**选门还是 near（z_eff 最小）；k 一致性只做"检验"** ——
        #   先按 near 选出目标，再算当前所有候选的 k：若选中的那扇**不是当前最大 k**
        #   （k 小 = z 报得比实际近 = 测距崩了，实船同一扇门 z 抖 ±35% 导致 k 在 0.48~0.87 跳），
        #   就**退回去**（本帧不采纳、原地 hold），而不是把"看着更近的崩解"选走。
        k_max_ratio = float(req(sub(self._G, "select"), "k_max_ratio"))
        # ★★ 2026-10-06 用户定：**先按 near（z_eff 最小）选，若它的 k 不是当前最大档，
        #   就把它排除掉、在剩下的里重新选**，直到选中的那扇 k 达标（或没得选 ⇒ None）。
        #   k 小 = z 报得比实际近（测距崩了）—— 实船同一扇门框占比稳定而 z 抖 ±35%，
        #   k 在 0.48~0.87 跳，纯按 near 会把这扇"看着更近的崩解"选走（一开始就锁到了**后边那扇**）。

        def _k_of_sel(d):
            r = (float(d.w) / float(self.w)) if (self.w and getattr(d, "w", 0)) else 0.0
            zz = self._z_est(d, conf_thr)
            return (float(zz) * r) if (zz is not None and r > 1e-6) else None

        # ★★ 2026-10-07 用户定：**一帧内一次把"实际远的那扇"全排除掉，再搬出结果**。
        #   为什么改掉原来的"选中→排除→重选"迭代：那个循环里的 `k_max` 是**逐帧重算**的，
        #   同一个门在这一帧是"最大 k"、下一帧就不是 ⇒ 反复进出候选 ⇒ **反复选错、反复解锁、
        #   耗时**（用户实测就是这个）。而且迭代版"全被排除"时会返回 None ⇒ 走丢帧计数 ⇒ 解锁，
        #   于是同一位置的门在相邻帧里被选进/排除反复横跳。
        #   现在：① 先算本帧所有候选的 k，一次定出 k_max；② **一次排除**所有 k < 阈值 的；
        #        ③ 在幸存者里按 near(z_eff 最小) 选；④ 若幸存者为空 ⇒ **退回 k 最大那扇**
        #        （它是本帧最可信的），**绝不返回 None** —— 有门就不制造"丢门→解锁"的空转。
        ks = [(d, _k_of_sel(d), self._z_est(d, conf_thr)) for d in gates]
        have = [k for (_d, k, _z) in ks if k is not None]
        if k_max_ratio > 0 and have:
            kthr = k_max_ratio * max(have)
            keep = [(d, z) for (d, k, z) in ks if (k is None or k >= kthr)]
            if not keep:
                # 全被排除 ⇒ 用 k 最大那扇（最可信），而不是丢门
                dmax = max((d for (d, k, _z) in ks if k is not None),
                           key=lambda d_: _k_of_sel(d_))
                keep = [(dmax, self._z_est(dmax, conf_thr))]
                if S.DEBUG:
                    print("[GATE] 选门：全部候选的 k 都低于 %.2f×最大 ⇒ 退回 k 最大那扇 k=%.3f"
                          % (k_max_ratio, _k_of_sel(dmax)))
            elif len(keep) < len(gates) and S.DEBUG:
                print("[GATE] 选门：一次排除 %d/%d 扇（k < %.2f×最大 %.3f）"
                      % (len(gates) - len(keep), len(gates), k_max_ratio, max(have)))
        else:
            keep = [(d, self._z_est(d, conf_thr)) for d in gates]
        best, best_z = None, None
        for d, z in keep:
            ratio = (float(d.w) / float(self.w)) if (self.w and getattr(d, "w", 0)) else 0.0
            if ratio > 1e-6 and k_true > 0:
                z_area = k_true / ratio               # 面积（框占比）反推的 z
                z_eff = float(z) if (z is not None and float(z) >= z_area) else z_area
            else:
                z_eff = z
            if best is None or (z_eff is not None and (best_z is None or z_eff < best_z)):
                best, best_z = d, z_eff
        return best
