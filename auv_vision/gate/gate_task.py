# -*- coding: utf-8 -*-
"""gate/gate_task.py — 任务三 过门（GateTask，§5.3 相位机）

门 = 对称平面矩形门框(0.70×0.50 m)，悬空，无朝向要求；任务 = 机身穿过开口。

相位机：SEARCH → RANGE_ALIGN → APPROACH → THROUGH(计数) → (下一门)/DONE
RANGE_ALIGN 子状态（按前端 mode 与门框占屏比仲裁）：
  GOLDEN   full/p3p 位姿对准（sway/heave PID，无 surge）
  CREEP    远距小框（coarse far / width）：慢速 creep + 框心对中
  HOLD     中/近角不足：surge=0 保持对中；≥hold.max_frames → REACQUIRE
  REACQUIRE 短后退重取整门（拉距→4 角重现）；超限 → SEARCH
穿过例外：Z ≤ z.cross → THROUGH（忽略角丢失，直行；目标消失确认 → 计数+1）

依赖：gate.geometry（位姿）、gate.gate_frontend（mode）、common.PID。
参数：cfg/vision.yaml gate.* 与 cfg/comm.yaml gate.*。
"""
from __future__ import annotations

import numpy as np

import base.settings as S
from common.PID import PID
from gate.gate_detector import board_camera
from gate.gate_frontend import (parse_kpt_mode, width_range_depth, bbox_center,
                                MODE_FULL, MODE_P3P, MODE_WIDTH, MODE_COARSE)
from gate.geometry import object_points, gate_pose, GATE_FRAME_W, GATE_FRAME_H

PH_SEARCH = "SEARCH"
PH_ALIGN = "ALIGN"
PH_APPROACH = "APPROACH"
PH_THROUGH = "THROUGH"

SUB_GOLDEN = "GOLDEN"
SUB_CREEP = "CREEP"
SUB_HOLD = "HOLD"
SUB_REACQUIRE = "REACQUIRE"

_ALIGN_SCALE = 0.5        # 位置误差归一化标尺(m)：±0.5m 满量程 → DOF ±1


def _dof_clip(v):
    return float(max(-1.0, min(1.0, v)))


class GateTask(object):
    name = "gate"

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

        pid_kw = dict(kp=G.pid.kp, ki=G.pid.ki, kd=G.pid.kd,
                      out_min=-G.pid.out_max, out_max=G.pid.out_max,
                      deadzone=G.pid.deadzone)
        self._pid_sway = PID(**pid_kw)
        self._pid_heave = PID(**pid_kw)
        self._pid_yaw = PID(**pid_kw)

        self.last_info = {"phase": PH_SEARCH, "substate": "", "mode": "",
                          "action": "stop", "z": 0.0, "dx": 0.0, "dy": 0.0,
                          "sway": 0.0, "heave": 0.0, "surge": 0.0, "yaw": 0.0,
                          "pass": 0, "kpt": 0, "status": S.STATUS_RUNNING,
                          "reason": ""}
        self.reset_state()

    # ------------------------------------------------------------ 状态复位
    def reset_state(self):
        self.frames = 0
        self._start_ms = None
        self.phase = PH_SEARCH
        self.substate = ""
        self.mode = ""
        self._last_pose = None            # (rvec, tvec) 上一帧位姿（p3p 消歧/防抖）
        self._z_last = None
        self._lost_cnt = 0
        self._center_cnt = 0
        self._hold_cnt = 0
        self._through_frames = 0
        self._reacquire_start = None
        self._search_entry_ms = None
        self._pass_cnt = 0
        self._finished = False
        self._reason = ""

    @property
    def ready(self):
        return self.hub.has_extra(self.name)

    # ------------------------------------------------------------ 工具
    def _pid_out(self, err_m):
        """位置误差(m) → PID（带死区/限幅）。"""
        return _dof_clip(self._pid_sway.update(err_m))

    def _set_info(self, action, mode="", substate="",
                  z=None, dx=0.0, dy=0.0, sway=0.0, heave=0.0,
                  surge=0.0, yaw=0.0, kpt=0):
        self.last_info.update({
            "phase": self.phase, "substate": substate or self.substate,
            "mode": mode or self.mode, "action": action,
            "z": round(z if z is not None else (self._z_last or 0.0), 3),
            "dx": round(float(dx), 3), "dy": round(float(dy), 3),
            "sway": round(float(sway), 3), "heave": round(float(heave), 3),
            "surge": round(float(surge), 3), "yaw": round(float(yaw), 3),
            "pass": self._pass_cnt, "kpt": int(kpt)})
        self.uart.send_dof(_dof_clip(surge), _dof_clip(sway),
                           _dof_clip(heave), _dof_clip(yaw))

    def _start_through(self):
        self.phase = PH_THROUGH
        self.substate = ""
        self._through_frames = 0
        self._lost_cnt = 0
        self._pid_sway.reset()
        self._pid_heave.reset()

    def _start_search(self):
        self.phase = PH_SEARCH
        self.substate = ""
        self._search_entry_ms = None
        self._last_pose = None
        self._lost_cnt = 0
        self._hold_cnt = 0

    def _search_yaw_pulse(self, now_ms):
        """SEARCH：原地旋转脉冲（转 spin_s → 停 pause_s）。"""
        spin = self._G.search.spin_s * 1000
        pause = self._G.search.pause_s * 1000
        if self._search_entry_ms is None:
            self._search_entry_ms = now_ms
        ph = (now_ms - self._search_entry_ms) % (spin + pause)
        return self._G.search.yaw if ph <= spin else 0.0

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
        det = None
        for d in dets:
            if d.kind == "gate" and (det is None or d.score > det.score):
                det = d
        self._step(det, now_ms)

        if (self._start_ms is not None and
                now_ms - self._start_ms >= self._G.timeout_ms):
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

    # ------------------------------------------------------------ 单帧逻辑
    def _step(self, det, now_ms):
        G = self._G
        if self.phase == PH_THROUGH:
            self._tick_through(now_ms)
            return
        if det is None:
            self._tick_lost(now_ms)
            return
        self._lost_cnt = 0

        # 前端 → mode
        V = self._V
        conf_thr = V.keypoint.get("conf_thr", 0.5)
        if det.kpts is not None and det.kpts.shape[0] >= 4 and \
                det.kpt_conf is not None:
            mode, ids = parse_kpt_mode(det.kpts, det.kpt_conf, conf_thr)
            n = len(ids)
        else:
            mode, ids, n = MODE_COARSE, [], 0

        if mode in (MODE_FULL, MODE_P3P) and n >= 3:
            obj3s = self.obj3[ids]
            img2s = np.asarray(det.kpts)[ids]
            pnp = V.pnp
            prev = self._last_pose if mode == MODE_P3P else None
            res = gate_pose(self.camera, obj3s, img2s, prev=prev,
                            reproj_thr=pnp.get("reproj_px", 8.0),
                            z_bounds=(pnp.get("z_min", 0.2),
                                      pnp.get("z_max", 15.0)),
                            refine=pnp.get("refine", True))
            if res is not None:
                self._on_pose(det, res, now_ms, mode=mode, kpt=n)
                return
            mode = MODE_COARSE          # 位姿校验失败 → 退化 coarse
        if mode == MODE_WIDTH:
            self._on_width(det, now_ms, ids)
            return
        self._on_coarse(det, now_ms)    # coarse / 其它退化
        _ = G

    # ---------------- full/p3p 位姿可用 ----------------
    def _on_pose(self, det, pose, now_ms, mode, kpt):
        G = self._G
        rvec, tvec = pose
        self._last_pose = pose
        self.mode = mode
        z = float(tvec.ravel()[2])
        self._z_last = z
        t = tvec.ravel()
        dx_m, dy_m = float(t[0]), float(t[1])
        if z <= G.z.cross:
            self._start_through()
            self._set_info("through", mode=mode, z=z, surge=G.surge.through,
                           dx=dx_m, dy=dy_m, kpt=kpt)
            return
        sway = -_dof_clip(self._pid_sway.update(
            max(-1, min(1, dx_m / _ALIGN_SCALE))))
        heave = -_dof_clip(self._pid_heave.update(
            max(-1, min(1, dy_m / _ALIGN_SCALE))))
        aligned = abs(dx_m) <= G.align.xy_m and abs(dy_m) <= G.align.xy_m

        if self.phase in (PH_SEARCH,):
            self.phase = PH_ALIGN
            self.substate = SUB_GOLDEN
            self._center_cnt = 0
        if self.phase == PH_ALIGN:
            self.substate = SUB_GOLDEN
            if aligned:
                self._center_cnt += 1
                if self._center_cnt >= G.align.confirm_frames:
                    self.phase = PH_APPROACH
                    self.substate = ""
            else:
                self._center_cnt = 0
            self._set_info("center", mode=mode, z=z, dx=dx_m, dy=dy_m,
                           sway=sway, heave=heave, kpt=kpt)
        elif self.phase == PH_APPROACH:
            if z > G.z.fast_max:
                surge = G.surge.fast
            elif z > G.z.slow_max:
                surge = G.surge.fast
            else:
                surge = G.surge.slow
            self._set_info("forward_%s" % ("fast" if surge > G.surge.slow else "slow"),
                           mode=mode, z=z, dx=dx_m, dy=dy_m,
                           sway=sway, heave=heave, surge=surge, kpt=kpt)
        else:
            self._set_info("center", mode=mode, z=z, dx=dx_m, dy=dy_m,
                           kpt=kpt)

    # ---------------- width（对向 2 角：测距+中点对中，可慢 creep） ----------------
    def _on_width(self, det, now_ms, ids):
        G = self._G
        self.mode = MODE_WIDTH
        u1, v1 = det.kpts[ids[0]]
        u2, v2 = det.kpts[ids[1]]
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
        sway = -_dof_clip(self._pid_sway.update(dxn))
        heave = -_dof_clip(self._pid_heave.update(dyn))
        aligned = abs(dxn) <= G.align.xy_m and abs(dyn) <= G.align.xy_m
        if self.phase == PH_SEARCH:
            self.phase = PH_ALIGN
            self._center_cnt = 0
        if self.phase == PH_ALIGN:
            # 距门仍远且对中 → 允许慢 creep（有限信息下安全推进）
            if aligned and z > G.z.cross + 0.4:
                self.substate = SUB_CREEP
                self._center_cnt += 1
                surge = G.surge.creep
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
        # 已在 REACQUIRE：持续后退直到 角点/位姿 恢复 或 超时回 search
        if self.phase == PH_ALIGN and self.substate == SUB_REACQUIRE:
            self._tick_reacquire(now_ms)
            return
        ratio = float(det.w) / float(self.w)
        cx, cy = bbox_center(det)
        dxn = (cx - self.w / 2.0) / (self.w / 2.0)
        dyn = (cy - self.h / 2.0) / (self.h / 2.0)
        sway = -_dof_clip(self._pid_sway.update(dxn))
        heave = -_dof_clip(self._pid_heave.update(dyn))
        aligned = abs(dxn) <= 0.05 and abs(dyn) <= 0.10

        if self.phase == PH_SEARCH:
            self.phase = PH_ALIGN
            self._hold_cnt = 0
        if self.phase == PH_APPROACH:
            # 进近中角全失 → 保守降回 ALIGN 仲裁（避免盲冲）
            self.phase = PH_ALIGN
            self._hold_cnt = 0

        if self.phase != PH_ALIGN:
            self._set_info("hold", z=self._z_last, dx=dxn, dy=dyn,
                           sway=sway, heave=heave)
            return

        if ratio < G.coarse.far_ratio:               # 远距小框 → creep 换取角点
            self.substate = SUB_CREEP
            surge = G.surge.creep if aligned else 0.0
            self._set_info("creep" if surge > 0 else "center",
                           z=self._z_last, dx=dxn, dy=dyn,
                           sway=sway, heave=heave, surge=surge)
        elif ratio <= G.coarse.near_ratio:           # 中距 → HOLD（无位姿即无进展）
            self.substate = SUB_HOLD
            self._hold_cnt += 1
            if self._hold_cnt >= G.hold.max_frames:
                self._enter_reacquire(now_ms)
            else:
                self._set_info("hold", z=self._z_last, dx=dxn, dy=dyn,
                               sway=sway, heave=heave)
        else:                                        # 近距装不下 → 直接后退重取
            self._enter_reacquire(now_ms)

    def _enter_reacquire(self, now_ms):
        self.substate = SUB_REACQUIRE
        self._reacquire_start = now_ms
        if S.DEBUG:
            print("[GATE] REACQUIRE: 角不足/过近, 后退重取")

    def _tick_reacquire(self, now_ms):
        G = self._G
        dur = now_ms - (self._reacquire_start or now_ms)
        if dur >= G.reacquire.max_ms:
            self._start_search()
            self._set_info("search")
            return
        # 后退方向/量可配（dof sign 需水池实测；负 surge = 后退）
        surge = -float(G.surge.reacquire)
        self._set_info("reacquire", substate=SUB_REACQUIRE, surge=surge)

    # ---------------- 丢目标 / 穿门 ----------------
    def _tick_lost(self, now_ms):
        G = self._G
        self._lost_cnt += 1
        if self.phase in (PH_ALIGN,) and self.substate == SUB_REACQUIRE:
            self._tick_reacquire(now_ms)
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
            if self._lost_cnt <= G.pose_hold_frames and self._z_last is not None:
                self._set_info("forward_slow", z=self._z_last,
                               surge=G.surge.creep)
            else:
                self._start_search()
                self._set_info("search")
            return
        # SEARCH
        yaw = self._search_yaw_pulse(now_ms)
        self._set_info("search", yaw=yaw)

    def _tick_through(self, now_ms):
        G = self._G
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
        self._set_info("through", surge=G.surge.through)
