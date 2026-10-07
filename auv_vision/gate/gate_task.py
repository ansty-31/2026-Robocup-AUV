# -*- coding: utf-8 -*-
"""gate/gate_task.py — **任务二（过门）总调度**：GateTask 相位机编排
（详细用法、判据与实测见 doc/注释历史.md）"""
from __future__ import annotations

import numpy as np
import base.cfg.settings as S
from common.motion.PID import PID
from base.log.turn_log import turn_log
from common.cfg.cfgnode import flag, merge, motion_node, motion_num, num, pid_kw, sub, req, req_flag, req_node, MissingCfg
from gate.percept.gate_detector import board_camera
from gate.percept.kpt_memory import ENV_ENABLE, build_kpt_memory, enable_source
from gate.percept.gate_frontend import (parse_kpt_mode,
                                              width_range_depth,
                                              bbox_center,
                                              MODE_FULL,
                                              MODE_P3P,
                                              MODE_WIDTH,
                                              MODE_COARSE)
from gate.percept.geometry import (object_points,
                                         gate_pose,
                                         gate_normal_angles_deg,
                                         GATE_FRAME_W,
                                         GATE_FRAME_H)
from gate.motion.hdg import HeadingAligner, hdg_cfg
from gate.percept.gate_postproc import (det_cfg,
                                              postproc_cfg,
                                              select_cfg,
                                              pick as postproc_pick)
from gate.motion.params import (PH_SEARCH, PH_ALIGN, PH_APPROACH, PH_THROUGH, SUB_HDG,
                                SUB_GOLDEN, SUB_CREEP, SUB_HOLD, SUB_REACQUIRE, SUB_SWAY_BACK)
from common.motion.search_scan import Scan, telemetry_yaw
from gate.motion.channels import GateChannels
from gate.motion.exits import GateExits
from gate.motion.hdg import GateHdg
from gate.motion.modes import GateModes
# 兼容出口：常量（PH_*/SUB_*/_D_*）实现已在 params.py；下面两行保证旧引用仍可用
from gate.motion.params import *            # noqa: F401,F403

class GateTask(GateChannels, GateHdg, GateModes, GateExits):
    name = "gate"

    # ------------------------------------------------------------------ 启动自检
    def validate_cfg(self):
        """**启动自检：把 `_K_*` 键名清单逐项核对 cfg** —— 缺键当场报名字。

        ★ 2026-10-07 用户定：删掉代码兜底之后，必须补上这道"必填校验"。
        ⚠️ 为什么必须在**启动**时做，而不能只靠 `req()` 用到才报：
           `req()` 是惰性的 —— `z.cross` 只在**门口的位姿帧**才被读到，缺它的话
           船会一路跑到门口才抛异常（实船：那正是最不能出事的地方）。
        """
        from gate.motion.params import (_K_ALIGN, _K_LOITER, _K_Z, _K_SURGE, _K_COARSE,
                                        _K_WIDTH, _K_HOLD, _K_REACQ, _K_THROUGH, _K_SEARCH,
                                        _K_HDG, _K_LOCK, _K_SELECT, _K_PNP, _K_GEOM)
        G, V = self._G, self._V
        pairs = [("comm.gate.align", _K_ALIGN, G.get("align")),
                 ("comm.gate.loiter", _K_LOITER, G.get("loiter")),
                 ("comm.gate.z", _K_Z, G.get("z")),
                 ("comm.gate.surge", _K_SURGE, G.get("surge")),
                 ("comm.gate.coarse", _K_COARSE, G.get("coarse")),
                 ("comm.gate.width", _K_WIDTH, G.get("width")),
                 ("comm.gate.hold", _K_HOLD, G.get("hold")),
                 ("comm.gate.reacquire", _K_REACQ, G.get("reacquire")),
                 ("comm.gate.through", _K_THROUGH, G.get("through")),
                 ("comm.gate.search", _K_SEARCH, G.get("search")),
                 ("comm.gate.hdg", _K_HDG, G.get("hdg")),
                 ("comm.gate.lock", _K_LOCK, G.get("lock")),
                 ("comm.gate.select", _K_SELECT, G.get("select")),
                 ("vision.gate.pnp", _K_PNP, V.get("pnp")),
                 ("vision.gate.percept.geometry", _K_GEOM, V.get("geometry"))]
        miss = []
        for name, keys, node in pairs:
            for k in keys:
                if not isinstance(node, dict) or node.get(k) is None:
                    miss.append("%s.%s" % (name, k))
        # 顶层标量 + 字符串档位
        for k in ("timeout_ms", "pass_target", "pose_hold_frames", "heave_max"):
            if G.get(k) is None:
                miss.append("comm.gate.%s" % k)
        if (G.get("approach") or {}).get("tier") in (None, ""):
            miss.append("comm.gate.approach.tier")
        if (self._V.get("keypoint") or {}).get("conf_thr") is None:
            miss.append("vision.gate.keypoint.conf_thr")
        if miss:
            raise MissingCfg("✗ cfg 缺必填参数（代码已无兜底，请补进 cfg/*.yaml）：\n    "
                             + "\n    ".join(miss))

    def __init__(self, uart, hub, frame_w, frame_h):
        self.uart = uart
        self.hub = hub
        self.w = int(frame_w)
        self.h = int(frame_h)
        G = S.get("comm.gate", None) or {}
        V = S.get("vision.gate", None) or {}
        self._G, self._V = G, V
        # 下视辅助过门：开关 vision.gate.down.enable（相机配置上机后补）；帧由 main 每帧写入。
        self._down_enabled = flag(sub(V, "down"), "enable", False)
        self._down_frame = None

        geo = req_node(V, "geometry")
        self.frame_w_m = float(geo["frame_w"])
        self.frame_h_m = float(geo["frame_h"])
        # 相机—机身安装偏置(m)：**只读入、不参与运算**（未标定；接进 sway/heave 目标
        self.body_center_offset = float(geo["body_center_offset"])
        self.obj3 = object_points(self.frame_w_m, self.frame_h_m)
        self.camera = board_camera()
        km = V.get("kpt_mem", None) or {}
        order = S.get("vision.model.task_models.gate.kpt_order", None) or \
            sub(V, "keypoint").get("kpt_order") or [0, 1, 2, 3]
        self._kpt_mem = build_kpt_memory(km, n_kpt=len(order) or 4)
        if self._kpt_mem is None:
            print("[GATE] kpt_mem 关闭 → 角点单帧直用（%s；临时开启用 %s=1）"
                  % (enable_source(km), ENV_ENABLE))
        else:
            # 打印**实际生效**的值（自己写兜底会打出与 kpt_memory.DEFAULTS 不一致的数字）
            m = self._kpt_mem
            conf_thr = req(sub(V, "keypoint"), "conf_thr")
            print("[GATE] kpt_mem 开启（逐点软融合，%s）：alpha=%s beta=%s "
                  "recall_conf=%s（必须 < keypoint.conf_thr=%.2f）"
                  % (enable_source(km), m.alpha, m.beta, m.recall_conf, conf_thr))
        self._kpt_s = None            # 本帧平滑后的角点（供 _on_width 等复用）
        self._kconf_s = None
        # 下水前自检：标定分辨率必须与实际帧一致，否则 PnP 深度整体缩放错
        if (self.camera.width, self.camera.height) != (self.w, self.h):
            print("[GATE] ⚠️ 标定尺寸 %dx%d ≠ 实际帧 %dx%d：PnP 距离会系统性偏差，"
                  "请按当前分辨率重新标定 (cfg/front_camera.yaml)"
                  % (self.camera.width, self.camera.height, self.w, self.h))

        G_sway, G_heave = sub(G, "pid_sway"), sub(G, "pid_heave")
        self._gain_src = "gate(显式覆盖)" if (G_sway or G_heave) else "motion(共用)"
        # ★ pid_kw 现在**必填**（无兜底）：任务级没写就用 comm.motion 那份，两者都必须齐全
        self._pid_sway_kw = pid_kw(G_sway or motion_node("pid_sway"),
                                   "comm.gate.pid_sway|comm.motion.pid_sway")
        self._pid_heave_kw = pid_kw(G_heave or motion_node("pid_heave"),
                                    "comm.gate.pid_heave|comm.motion.pid_heave")
        self._pid_sway_px = PID(**self._pid_sway_kw)
        self._pid_heave_px = PID(**self._pid_heave_kw)
        _ks = ("kp", "ki", "kd", "out_max", "deadzone")
        print("[GATE] 修正增益来源=%s | sway %s | heave %s | 转向=仅正航向(%s)"
              % (self._gain_src,
                 {k: self._pid_sway_kw[k] for k in _ks},
                 {k: self._pid_heave_kw[k] for k in _ks},
                 "enable" if hdg_cfg().get("enable", True) else "off"))
        # 正航向（ALIGN.HDG；见 gate/motion/hdg.py，先居中再转向）：每门只做一次，出结论后锁定。
        self._hdg = HeadingAligner(log=(print if S.DEBUG else (lambda *a: None)))
        self._hdg_cfg = hdg_cfg()
        # ---- 选门策略（规范 doc/设计/gate_pose_decode_spec.md §4；实现 gate/percept/gate_postproc.py）----
        self._select_cfg = select_cfg()          # near(默认) | corner(旧行为)
        self._post = postproc_cfg()
        print("[GATE] 选门策略=%s | 去重/几何=%s | 角点可信下限=%.2f | 候选 conf=%.2f"
              % (self._select_cfg["mode"],
                 ("edge_tol=%.2f/min_edges=%d/geom=%s"
                  % (self._post["edge_tol"], self._post["min_edges"],
                     self._post["geom_check"])),
                 req(sub(V, "keypoint"), "conf_thr"),
                 det_cfg()))
        self._hdg_done = True             # True=本门不需要再正航向（未启用或已出结论）
        self._hdg_done = not flag(self._hdg_cfg, "enable", True)
        self._hdg_ok = False              # 本门是否满足冲刺前航向条件；每次换门清零
        self._hdg_ok_cnt = 0              # ★ 连续落进 tol_deg 的帧数（ok_frames 判据用）
        self._hdg_f = None                # 快 EMA(≈4 帧)：只用于日志 hdg_fast
        self._hdg_s = None                # ★ 慢 EMA(psi_ema_frames 帧)：**判据与目标角用这个**
        self._hdg_deg = None              # 交给判据的航向误差（度）= 上面的慢 EMA
        self._hdg_ms = None               # 上面那个值的时刻
        self._hdg_turns = 0               # 本门已转向次数（**无上限**；逐小步逼近，仅用于日志）
        self._hdg_turns_last_ms = None    # 上一次转向**结束**的时刻（判"有没有新的 ψ"用）
        self._hdg_skip_logged = False     # 本门是否已报过"跳过正航向"的原因（每门一次）
        self._post_sway_until_ms = None   # 「转完反向平移」状态：非 None = 激活（到期时刻）
        self._post_sway = 0.0
        self._post_sway = 0.0             # 该状态里下发的 sway（= 转向的反方向）
        self._post_sway_back = 0.0        # 同时叠加的**缓慢后退**分量（负=后退）
        self._locked_det = None           # ★ 换门/复位 → 解锁，允许重新 near 选门
        self._lock_miss = 0
        self._now_ms = None               # 本帧时基（process 每帧写入）
        self._det_now = None              # 本帧选中的门（状态判"门回来了吗"用当帧数据）
        self._locked_det = None           # ★ 本轮**锁定**的门（det 对象）：锁定后只跟它，不再每帧重选
        self._lock_miss = 0               # 锁定后连续没匹配上的帧数（≥lock.miss_frames 才解锁重选）
        self._lock_cnt = 0                # ★ 连续跟住同一扇的帧数（≥lock.stable_frames ⇒ 稳定后不再参与选门）
        self._lock_stable = False         # ★ 锁定是否已稳定（稳定后才不许别的门抢）
        self._lock_k_ref = None           # ★ 锁定门的 k 参照（k=z×框占比）——"保护锁定"①用
        # ★ 2026-10-07：把"丢门重锁"那套参照**显式初始化**（原来全靠 `getattr(...,None)` 兜着，
        #   属性不存在 ⇒ 外部/用例直接读会 AttributeError，也看不出它是不是该有值）
        self._hold_start_ms = None        # ★ coarse 的 HOLD 起始时刻（超时 5s ⇒ 触发反向慢移）
        self._ids_now = []                # ★ 本帧有效角点 id（coarse 反向慢移按它判上浮/下潜）
        self._relock_z_ref = None         # 丢门前的 z（`_relock_guard()` 判"比丢门前远 ≥0.5m"用）
        self._relock_ratio_ref = None     # 丢门前的框占比（`_relock_guard_ratio()` 用）
        self._relock_logged = False
        self._lock_ratio_ref = None       # ★ 锁定门的**面积**参照（框占比）——"保护锁定"②用
        self._lock_k_bad = 0
        self._scan = Scan()

        self.last_info = {"phase": PH_ALIGN, "substate": "", "mode": "",
                          "action": "stop", "z": 0.0, "dx": 0.0, "dy": 0.0,
                          "sway": 0.0, "heave": 0.0, "surge": 0.0, "yaw": 0.0,
                          "pass": 0, "kpt": 0, "kpt_raw": 0, "ratio": 0.0,
                          "hdg": None,          # 航向误差(度)：+ = 门法向偏画面右 = 机身左偏
                          "hdg_skip": None,     # 居中达标却跳过正航向的原因（下一步：进近）
                          "hdg_state": "",      # HDG 内部状态：idle/settle/measure/turn/done/giveup
                          "hdg_i": 0,           # 正航向已迭代次数（ALIGN.HDG）
                          "status": S.STATUS_RUNNING, "reason": ""}
        self.reset_state()
        self.validate_cfg()          # ★ 启动自检：缺键当场报名字（代码已无兜底）
    def reset_state(self):
        self.frames = 0
        self._start_ms = None
        self.phase = PH_ALIGN
        self.substate = ""
        self.mode = ""
        self._last_pose = None            # (rvec, tvec) 上一帧位姿（消歧/防抖）
        self._z_last = None
        self._z_guard = False             # 上一帧是否有可信位姿（z 跳变保护基准）
        self._lost_cnt = 0
        self._center_cnt = 0
        self._center_ok_cnt = 0           # ★ 连续"居中成功"帧数（主出口居中闸用）
        self._hold_cnt = 0
        self._cross_cnt = 0              # 连续 z≤cross 帧数（穿门确认，防单帧错解）
        self._through_frames = 0
        self._through_start_ms = None    # THROUGH 起始时间戳（时长判据）
        self._creep_through_start_ms = None  # CREEP_THROUGH 起始时间戳
        self._hdg.reset()
        self._hdg_done = not flag(self._hdg_cfg, "enable", True)
        self._hdg_ok = False
        self._hdg_ok_cnt = 0
        self._hdg_skip_logged = False
        self._hdg_turns = 0
        self._hdg_turns_last_ms = None
        self._hdg_f = None                # ★ 复位时把 ψ 的 EMA 状态也清掉（原来漏了）
        self._hdg_s = None
        self._hdg_deg = None
        self._hdg_ms = None
        self._post_sway_until_ms = None
        self._post_sway = 0.0
        self._post_sway_back = 0.0        # 同时叠加的**缓慢后退**分量（负=后退）
        self._loiter_start_ms = None     # 「已在门口」的起始时刻（超时兜底）
        self._reacquire_start = None
        self._reacquire_ratio0 = None    # 进入 REACQUIRE 时的框占比（闭环后退基准）
        self._reacquire_cnt = 0          # 连续 REACQUIRE 次数（超限不再后退）
        self._reacquire_last_ms = None   # 上次真正后退的时刻（时间窗重置计数）
        self._give_up_until_ms = None    # 放弃后退后的 HOLD 窗口截止
        self._through_block_logged = False  # 本门是否已报过「航向闸门拦下冲刺」（每门一次）
        self._dbg_ratio = 0.0            # 本帧门框宽/屏宽（诊断+叠加）
        self._dbg_kpt = 0                # 本帧有效角点数（**记忆后**，用于判 mode）
        self._dbg_kpt_raw = 0            # 本帧原始有效角点数（诊断：看倒影污染程度）
        self._search_entry_ms = None
        self._pass_cnt = 0
        self._finished = False
        self._reason = ""
        self.last_dets = []               # 本帧门检测（供 main._draw 画角点）
    def set_down_frame(self, frame):
        """每帧由装配层(main)写入下视帧；下视处理只在 coarse/width 门口时读它。"""
        self._down_frame = frame
    def wants_down(self):
        """本帧是否需要下视帧：下视开关开 且 当前档位是 coarse/width（门口过门才用下视判断）。"""
        return self._down_enabled and self.mode in ("coarse", "width")
    @property
    def ready(self):
        return self.hub.has_extra(self.name)
    def process(self, frame, now_ms):
        self.frames += 1
        self._now_ms = now_ms        # 本帧时基：`_set_info` 里推进「反向平移」状态要用
        self._det_now = None
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

        if (self._hdg.turning and not self._hdg.finished()):
            self.last_dets = []            # 本帧没有视觉结果 → 叠加层不许画上一帧的框
            self._turn_blocking(now_ms)
        else:
            dets = self.hub.detect_list(self.name, frame)
            self._dets_raw = dets        # 本帧原始检测（sway 退出判据的兜底用）
            fw = frame.shape[1] if frame is not None and hasattr(frame, "shape") else 0
            det = self._gate_lock(dets, fw)
            # ★ 2026-10-04 用户定：**锁定本轮的门之后，画面只打锁定那扇门的框**（别的门不画，
            self.last_dets = [det] if det is not None else []
            self._step(det, now_ms)

        if (self._start_ms is not None and
                now_ms - self._start_ms >= req(self._G, "timeout_ms")):
            self._finish("timeout")
        self.last_info["status"] = S.STATUS_DONE if self._finished \
            else S.STATUS_RUNNING
        return self.last_info["status"]
    def _gate_lock(self, dets, frame_w):
        """选门 + **锁定**（★ 2026-10-04 用户定："可以锁定，但锁定前务必做好检查 ——"""
        Lc = req_node(self._G, "lock")
        gates = [d for d in dets if getattr(d, "kind", None) == "gate"]
        miss_max = int(req(Lc, "miss_frames"))
        stable_n = int(req(Lc, "stable_frames"))
        if not gates:
            self._lock_miss += 1
            if self._lock_miss >= miss_max:
                self._unlock()
            return None
        if self._locked_det is None:
            self._locked_det = self._pick_gate(dets)
            self._lock_miss = 0
            self._lock_cnt = 0
            self._lock_stable = False
            self._lock_k_ref = self._k_of(self._locked_det) if self._locked_det is not None else None
            self._lock_ratio_ref = (float(self._locked_det.w) / float(self.w)) \
                if (self._locked_det is not None and self.w and getattr(self._locked_det, "w", 0)) else None
            return self._locked_det
        #   ⚠️ 只在**锁定已稳定**时启用：稳定之前参照还在建立（头几帧的框/姿态都还没定），
        kept = ([d for d in gates if not self._lock_guard_reject(d)]
                if self._lock_stable else list(gates))
        if not kept:
            self._lock_miss += 1
            if self._lock_miss >= miss_max:
                if not self._lock_guard_reject(self._pick_gate(dets) or self._locked_det):
                    self._unlock()
                else:
                    self._lock_miss = 0        # 重选结果也被判不是同一扇 ⇒ 不放行，继续等
                    return None
            return None if self._lock_stable else None
        gates = kept
        lx = float(self._locked_det.x) + float(self._locked_det.w) * 0.5
        ly = float(self._locked_det.y) + float(self._locked_det.h) * 0.5
        thr = float(req(Lc, "match_ratio")) * float(frame_w or 0.0)
        best, best_d = None, None
        for d in gates:
            cx = float(d.x) + float(d.w) * 0.5
            cy = float(d.y) + float(d.h) * 0.5
            dd = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
            if best is None or dd < best_d:
                best, best_d = d, dd
        matched = best if (best is not None and thr > 0 and best_d <= thr) else None
        if matched is not None:
            kz = self._k_of(matched)
            if self._lock_k_ref is None:
                self._lock_k_ref = kz                     # 刚锁上：建立参照
            elif kz is not None:
                k_lo = float(req(sub(self._G, "select"), "k_lo_ratio"))
                k_hi = float(req(sub(self._G, "select"), "k_hi_ratio"))
                if not (k_lo * self._lock_k_ref <= kz <= k_hi * self._lock_k_ref):
                    if S.DEBUG:
                        print("[GATE] k 一致性不通过：本帧 k=%.3f vs 锁定参照 %.3f（允许 ×%.2f~%.2f）"
                              "⇒ 不是锁定那扇门" % (kz, self._lock_k_ref, k_lo, k_hi))
                    matched = None                        # 当没检测到，走丢帧计数
                else:
                    self._lock_k_ref = 0.7 * self._lock_k_ref + 0.3 * kz   # 缓慢跟踪
        if matched is None:
            self._lock_miss += 1
            if self._lock_miss >= miss_max:
                self._unlock()
                return self._pick_gate(dets)
            return None if self._lock_stable else self._pick_gate(dets)
        self._lock_miss = 0
        self._lock_cnt += 1
        if self._lock_cnt >= max(1, stable_n):
            self._lock_stable = True
        if self._lock_stable:
            self._locked_det = matched          # ★ 稳定 ⇒ 不参与选门
            self._lock_note_accepted(matched)
            return matched
        pick = self._pick_gate(dets)
        if pick is not None and pick is not matched:
            self._locked_det = pick
            self._lock_cnt = 0
            self._lock_note_accepted(pick)
            return pick
        self._locked_det = matched
        self._lock_note_accepted(matched)
        return matched

    def _lock_guard_reject(self, det):
        """**锁定保护**（★ 2026-10-05 用户定：k 一致性检验与跳变保护**同一个位置**，"""
        if self._locked_det is None:
            return False
        ratio = (float(det.w) / float(self.w)) if (self.w and getattr(det, "w", 0)) else 0.0
        if ratio <= 1e-6:
            return False
        k_lo = float(req(sub(self._G, "select"), "k_lo_ratio"))
        k_hi = float(req(sub(self._G, "select"), "k_hi_ratio"))
        # ---- 保护①：k 一致性（k = z×框占比；能测才判，测不出本层跳过）----
        kz = self._k_of(det)
        k_ok = True
        if kz is not None and self._lock_k_ref:
            if not (k_lo * self._lock_k_ref <= kz <= k_hi * self._lock_k_ref):
                if S.DEBUG:
                    print("[GATE] 锁定保护①(k)：本帧 k=%.3f vs 参照 %.3f（允许 ×%.2f~%.2f）⇒ 不是锁定那扇门"
                          % (kz, self._lock_k_ref, k_lo, k_hi))
                return True
        # ★★ 2026-10-06 用户定：**贴到门口 / 门框出画时，面积保护必须让位**(方案 a+c)。
        #   (a) 船贴近时门框撑出画面，bbox 只剩可见部分 ⇒ 框占比**必然**变小 ⇒ "占比<0.7×参照"
        #       会把**贴脸的门自己**判成"更远那扇" ⇒ 锁定门被自己人拒收 ⇒ 一路 hold 进不了 THROUGH
        #       （实船帧134：占比 0.398→0.177，z 已冻在 1.17 ≤ cross）。判定"近"用锁定门最近采信的 z。
        #   (c) 检测框**触到画面边缘**= 门框被裁掉 ⇒ 占比同样不代表距离 ⇒ 本条也不做面积判据。
        _zc = req_node(self._G, "z")
        _cross = req(_zc, "cross")
        _zl = getattr(self, "_z_last", None)
        if _cross > 0 and _zl is not None and float(_zl) <= _cross:
            return False                      # (a) 门口：跳过面积保护（①k 与跳变保护仍在）
        _w = float(self.w or 0)
        _h = float(self.h or 0)
        if _w > 0 and _h > 0:
            _x0, _y0 = float(det.x), float(det.y)
            _x1, _y1 = _x0 + float(det.w), _y0 + float(det.h)
            _m = 2.0                          # 容差(px)：贴着边就算出画
            if _x0 <= _m or _y0 <= _m or _x1 >= _w - _m or _y1 >= _h - _m:
                return False                  # (c) 门框出画：框占比不代表距离，跳过面积保护
        ref = getattr(self, "_lock_ratio_ref", None)
        far = req(sub(self._G, "z"), "relock_far_ratio")
        if ref is not None and far > 0 and ratio < float(far) * float(ref):
            if S.DEBUG:
                print("[GATE] 锁定保护②(面积)：本帧占比 %.3f < %.2f×参照 %.3f ⇒ 是更远那扇门，拒收"
                      % (ratio, float(far), float(ref)))
            return True
        # ⚠️ 本函数**必须是纯函数**（只判、不改参照）：它在"候选筛选"里对每个候选都调，
        return False

    def _lock_note_accepted(self, det):
        """只对**最终采纳（=锁定）的那一扇**更新两者参照。

        ★★ 2026-10-07 用户定：**锁门的面积参照 = 上一帧被采纳的占比（帧间比较）**，
        判据即「本帧占比 ≥ `relock_far_ratio`(0.7) × 上一帧占比 ⇒ 还是这扇门、可信、
        **不着急换门**」。

        ⚠️ 与 `_relock_ratio_ref`（`_step` 里那套）**不是同一个东西**，别互相套：
          · 这里是**锁门层**的参照 —— 帧间；
          · `_relock_ratio_ref` 是丢门重锁那套，**保持用户原来的写法不动**。
        k 参照仍用 EMA（它是"k 一致性"的锚，语义不同）。
        """
        if det is None:
            return
        kz = self._k_of(det)
        if kz is not None:
            self._lock_k_ref = (kz if self._lock_k_ref is None
                                else 0.7 * self._lock_k_ref + 0.3 * kz)
        ratio = (float(det.w) / float(self.w)) if (self.w and getattr(det, "w", 0)) else None
        if ratio is not None:
            self._lock_ratio_ref = ratio        # ← 帧间：下一帧与**上一帧**比

    def _k_of(self, det):
        """这扇门的 k = z × 框占比。拿不到 z（没角点/coarse）或占比 ⇒ None ="本帧这项检验用不了"。"""
        try:
            ratio = (float(det.w) / float(self.w)) if (self.w and getattr(det, "w", 0)) else 0.0
            if ratio <= 1e-6:
                return None
            z = self._z_est(det, req(sub(self._V, "keypoint"), "conf_thr"))
            return None if z is None else float(z) * ratio
        except Exception:
            return None

    def _unlock(self):
        """解锁：允许下一帧重新 near 选门。"""
        self._locked_det = None
        self._lock_miss = 0
        self._lock_cnt = 0
        self._lock_stable = False
        self._lock_k_ref = None
        self._lock_ratio_ref = None
        self._lock_k_bad = 0


    def _step(self, det, now_ms):
        self._det_now = det          # 本帧的门（「反向平移」状态判"门回来了吗"用当帧数据）
        if self._hdg.turning and not self._hdg.finished():
            self._turn_blocking(now_ms)
            return
        if self.phase in (PH_THROUGH, PH_CREEP_THROUGH):
            if self.phase == PH_THROUGH:
                self._tick_through(now_ms)      # 正常过门=直行，不做横向微调
            else:
                self._tick_creep_through(now_ms)   # 门口过门=慢速 creep 直行
            return
        if det is None:
            self._tick_lost(now_ms)
            return
        # ★ 2026-10-05 用户定：**选门优先 z，锁门用面积**。
        #   ★★ 2026-10-07 用户定：面积参照（丢门重锁那套）做成**可切换的三个模型** ——
        #     `comm.gate.z.relock_ratio_model` = interframe(帧间) / preloss(丢门前) / ema(平滑)；
        #     具体更新在 `_relock_ratio_note_accepted()`（**只对被采纳的帧**更新），
        #     判据 `_relock_guard_ratio()` 是**纯函数**（不改任何参照）。
        #   这里只保留一个兜底：参照还空着而已经有被采纳帧 ⇒ 用它的占比起个头
        #   （`preloss` 模型下如果没丢过门，也用它当起点，避免"永远没有参照"）。
        if getattr(self, "_relock_ratio_ref", None) is None and \
                getattr(self, "_ratio_last", None) is not None:
            self._relock_ratio_ref = float(self._ratio_last)
        # 命中（= 另一个/更远的门）就原地 hold，不朝它靠近。
        if self._relock_guard_ratio():
            self._set_info("hold", z=self._z_last)
            return
        self._lost_cnt = 0
        self._ratio_last = float(det.w) / float(self.w)   # ★ 本帧通过保护 ⇒ 记"最近被采纳的占比"
        self._relock_ratio_note_accepted(self._ratio_last)   # ★ 按模型更新面积参照
        V = self._V
        conf_thr = req(sub(V, "keypoint"), "conf_thr")
        # 诊断量：本帧框占比 + 有效角点数（进叠加与 REACQUIRE 日志）
        self._dbg_ratio = float(det.w) / float(self.w)
        if det.kpt_conf is not None:
            self._dbg_kpt_raw = int(
                (np.asarray(det.kpt_conf) >= conf_thr).sum())
        else:
            self._dbg_kpt_raw = 0

        # 角点短时记忆：小漂移→窗口平均；短消失→回忆；鬼点→剔除
        kpts, kconf = det.kpts, det.kpt_conf
        if self._kpt_mem is not None:
            kpts, kconf = self._kpt_mem.update(kpts, kconf, now_ms)
        self._kpt_s, self._kconf_s = kpts, kconf
        self._dbg_kpt = int((np.asarray(kconf) >= conf_thr).sum()) \
            if kconf is not None else 0

        # 前端 → mode（用记忆后的角点）
        if kpts is not None and kpts.shape[0] >= 4 and kconf is not None:
            mode, ids = parse_kpt_mode(kpts, kconf, conf_thr)

            self._ids_now = list(ids)     # ★ 本帧有效角点 id（coarse 反向慢移按它判方向）
            n = len(ids)
        else:
            mode, ids, n = MODE_COARSE, [], 0
            self._ids_now = []           # ★ 本帧没有可信角点 ⇒ 清空（别留上一帧的）


        if mode in (MODE_FULL, MODE_P3P) and n >= 3:
            obj3s = self.obj3[ids]
            img2s = np.asarray(kpts)[ids]
            pnp = req_node(V, "pnp")
            # 上帧位姿用于消歧：平面 IPPE 双解在远距/小目标时 RMS 接近，纯靠 RMS 会帧间跳
            prev = self._last_pose
            res = gate_pose(self.camera, obj3s, img2s, prev=prev,
                            reproj_thr=req(pnp, "reproj_px"),
                            z_bounds=(req(pnp, "z_min"),
                                      req(pnp, "z_max")),
                            refine=flag(pnp, "refine", True))
            if res is not None:
                max_jump = req(pnp, "max_z_jump_m")
                if self._z_guard and self._z_last is not None and max_jump > 0:
                    z_new = float(res[1].ravel()[2])
                    if abs(z_new - self._z_last) > max_jump:
                        if S.DEBUG:
                            print("[GATE] 弃帧: z 跳变 %.2f→%.2f m (>%.2f)"
                                  % (self._z_last, z_new, max_jump))
                        res = None
            if res is not None:
                self._z_guard = True
                self._on_pose(det, res, now_ms, mode=mode, kpt=n)
                return
            mode = MODE_COARSE          # 位姿校验失败/跳变 → 退化 coarse
        self._z_guard = False
        self._cross_cnt = 0
        if self.mode != mode:
            self._center_cnt = 0
        if mode == MODE_WIDTH:
            self._on_width(det, now_ms, ids)
            return
        self._on_coarse(det, now_ms)    # coarse / 其它退化
    def _finish(self, reason):
        self._finished = True
        self._reason = reason
        self.uart.neutral()
        self.last_info.update({"status": S.STATUS_DONE, "reason": reason,
                               "pass": self._pass_cnt})
        if S.DEBUG:
            print("[GATE] DONE(%s) pass=%d frames=%d" % (reason, self._pass_cnt,
                                                          self.frames))
