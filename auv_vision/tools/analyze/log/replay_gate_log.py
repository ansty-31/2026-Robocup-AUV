# -*- coding: utf-8 -*-
"""tools/analyze/log/replay_gate_log.py — **用实船逐帧日志驱动 GateTask 的虚拟回放**

用途：回答"这趟日志如果按**现在的代码 + 现在的 cfg** 再走一遍，会发生什么"——
特别是 **THROUGH 在什么时刻进入**、**转向下发了多少度/哪个方向**、
以及 **没进 THROUGH 时是哪一道闸门拦下的**（`last_info['through_block']`）。

## 原理（重要：这是"感知回放"，不是动力学仿真）

日志里没有图像，但记下了每帧的**判据量**（mode / z / ratio / kpt / dx / dy / hdg）。
本工具按下面规则**反造一个 Det**，喂给真代码，让真代码自己走相位机：

| 日志 mode | 反造方式 | 依据 |
|---|---|---|
| `full` | 由 (z, ψ_raw, dx, dy) **合成门框位姿** → 投影 4 角点 | ψ_raw 由 `hdg` 的 EMA 反解（`alpha=0.4`，`modes.py:59`） |
| `width` | 直接摆**对向 2 角点**，令 `Δu = fx·W/z` | `gate_frontend.width_range_depth` |
| `coarse` | **角点全丢**（conf=0），bbox 按 `ratio`+`(dx,dy)` 摆 | `modes.bbox_center`（dx/dy 来自框心） |

已用合成往返验证：合成位姿 → 投影 → `gate_pose` 回解，z 与 ψ 均逐位还原（见 `--selftest`）。

⚠️ **能证明什么 / 不能证明什么**
- 能证明：相位机/闸门在**这串判据序列**下的仲裁结果（含 THROUGH 时刻、下发角、拦门原因）。
- **不能**证明：船的真实运动、以及"当时探测为什么解不出位姿"（日志 mode=coarse 而 kpt=4
  的那些帧，是 PnP 失败 ⇒ 本工具按日志记录的 mode 复现"角点不足"，**不会**重新去解 PnP）。
- 回放是**开环感知**：感知由日志强制，控制器只影响 **DOF/转向下发**，不影响下一帧的判据。

用法：
    python3 tools/analyze/log/replay_gate_log.py log/rungate_20261006_postsway.jsonl
    python3 tools/analyze/log/replay_gate_log.py <log.jsonl> --turn-follows   # 假下位机服从转向
    python3 tools/analyze/log/replay_gate_log.py <log.jsonl> --kpt-raw        # 反事实：按 kpt_raw 给角点
    python3 tools/analyze/log/replay_gate_log.py --selftest
"""
from __future__ import annotations

import json
import os
import sys

# 工程根 = 向上第一个含 `cfg/` 的目录
_ROOT = os.path.dirname(os.path.abspath(__file__))
while _ROOT != os.path.dirname(_ROOT) and not os.path.isdir(os.path.join(_ROOT, "cfg")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)

os.environ.setdefault("AUV_SIM_MODE", "1")
os.environ.setdefault("AUV_STREAM", "0")

import numpy as np                                                  # noqa: E402
import cv2                                                          # noqa: E402

import base.cfg.settings as S                                       # noqa: E402
from common.vision.detector import Det                              # noqa: E402
from gate.percept.gate_detector import board_camera                 # noqa: E402
from gate.percept.geometry import (object_points, gate_pose,        # noqa: E402
                                   gate_normal_angles_deg)
from gate.percept.gate_frontend import parse_kpt_mode, width_range_depth  # noqa: E402
from gate.gate_task import GateTask                                 # noqa: E402
from gate.motion.params import (PH_ALIGN, PH_APPROACH, PH_THROUGH,   # noqa: E402
                               PH_CREEP_THROUGH)

CAM = board_camera()
OBJ3 = object_points()
CONF_THR = float(S.get("vision.gate.keypoint.conf_thr", 0.8) or 0.8)
K_TRUE = CAM.fx * float(S.get("vision.gate.geometry.frame_w", 0.7) or 0.7) / CAM.width
# 门框 bbox 的"高/宽"比（0.7×0.5 m 门在 rectified 域下的理论值）：coarse 档造框要用
BOX_HW = (CAM.fy * float(S.get("vision.gate.geometry.frame_h", 0.5) or 0.5)) / \
         (CAM.fx * float(S.get("vision.gate.geometry.frame_w", 0.7) or 0.7))

# ★★ 基准约定（这趟日志最大的一个坑）：
#   日志期的 `_on_pose` 用 **画面几何中心 (640,360)** 当基准（HEAD 版 modes.py），
#   而现在的 `_center_ref()` 改成了 **相机主点 (cx,cy)=(702.27,409.14)**。
#   同一帧物理画面，两种基准读出的偏差恰好差一个常数：
#       dx_axis = dx_img − (cx−W/2)/(W/2) = dx_img − 0.0973
#       dy_axis = dy_img − (cy−H/2)/(H/2) = dy_img − 0.1365
#   `--ref=axis`（默认）= 把日志的 dx/dy **折算到主点基准**（物理正确，喂给现在的代码）；
#   `--ref=img`        = 把日志数字**原样**喂进去（等价于"没意识到基准变了"的错误回放）。
REF_SHIFT = {"axis": ((CAM.cx - CAM.width / 2.0) / (CAM.width / 2.0),
                      (CAM.cy - CAM.height / 2.0) / (CAM.height / 2.0)),
             "img": (0.0, 0.0)}


# ---------------------------------------------------------------- 反造检测
def pose_from(z, psi_deg, dx, dy):
    """由 (z, ψ, dx, dy) 反造门框位姿：ψ 绕相机 Y 轴，tx/ty 解出让门原点落在 (dx,dy)。"""
    rvec, _ = cv2.Rodrigues(np.array([0.0, np.radians(float(psi_deg)), 0.0]))
    tx = ((float(dx) * CAM.width / 2.0 + CAM.width / 2.0) - CAM.cx) * float(z) / CAM.fx
    ty = ((float(dy) * CAM.height / 2.0 + CAM.height / 2.0) - CAM.cy) * float(z) / CAM.fy
    return rvec, np.array([tx, ty, float(z)], np.float64).reshape(3, 1)


def _box_from_hull(uv, ratio, w_img):
    """按角点外框摆 bbox，但**宽度强制等于日志里的 ratio**（日志里框宽是检测器的量）。"""
    cxs = float(uv[:, 0].min() + uv[:, 0].max()) / 2.0
    cys = float(uv[:, 1].min() + uv[:, 1].max()) / 2.0
    w = max(2.0, float(ratio) * float(w_img))
    h = max(2.0, w * BOX_HW)
    return cxs - w / 2.0, cys - h / 2.0, w, h


def _refxy(row, ref):
    sx, sy = REF_SHIFT[ref]
    return float(row["dx"]) - sx, float(row["dy"]) - sy


def det_full(row, psi_raw, w_img, ref="axis"):
    """full 档：合成位姿 → 投影 4 角点（conf 用日志的 kpt_raw 决定）。"""
    dx, dy = _refxy(row, ref)
    rvec, tvec = pose_from(row["z"], psi_raw, dx, dy)
    uv = CAM.project(OBJ3, rvec, tvec)
    n_ok = int(row.get("kpt_raw") or 4)
    conf = np.array([0.95 if i < n_ok else 0.0 for i in range(4)], np.float32)
    x, y, w, h = _box_from_hull(uv, row["ratio"], w_img)
    return Det("gate", 0.95, x, y, w, h, kpts=uv, kpt_conf=conf)


def det_width(row, w_img, ref="axis"):
    """width 档：摆对向 2 角点（上边），使 Δu 对应日志里的 z。"""
    dxr, dyr = _refxy(row, ref)
    z = float(row["z"])
    du = CAM.fx * float(S.get("vision.gate.geometry.frame_w", 0.7) or 0.7) / max(z, 1e-6)
    aim = dxr * CAM.width / 2.0 + CAM.width / 2.0           # 2 角中点 = 水平瞄准点
    cx = row["ratio"] * w_img / 2.0 + CAM.width / 2.0
    cy = dyr * CAM.height / 2.0 + CAM.height / 2.0          # dy 来自框心（bbox_center）
    y_top = cy - (row["ratio"] * w_img * BOX_HW) / 2.0
    kp = np.array([[aim - du / 2.0, y_top], [aim + du / 2.0, y_top],
                   [0.0, 0.0], [0.0, 0.0]], np.float32)
    kc = np.array([0.95, 0.95, 0.0, 0.0], np.float32)
    w = max(2.0, row["ratio"] * w_img)
    h = max(2.0, w * BOX_HW)
    return Det("gate", 0.95, cx - w / 2.0, cy - h / 2.0, w, h, kpts=kp, kpt_conf=kc)


def det_coarse(row, w_img, with_kpts=False, ref="axis"):
    """coarse 档：角点全丢（默认）。with_kpts=True = **反事实**：按面积反推 z 给 4 个角点。"""
    dxr, dyr = _refxy(row, ref)
    w = max(2.0, row["ratio"] * w_img)
    h = max(2.0, w * BOX_HW)
    cx = dxr * CAM.width / 2.0 + CAM.width / 2.0
    cy = dyr * CAM.height / 2.0 + CAM.height / 2.0
    if with_kpts:
        z_area = K_TRUE / max(row["ratio"], 1e-6)
        rvec, tvec = pose_from(z_area, 0.0, dxr, dyr)
        uv = CAM.project(OBJ3, rvec, tvec)
        return Det("gate", 0.95, cx - w / 2.0, cy - h / 2.0, w, h,
                   kpts=uv, kpt_conf=np.full(4, 0.95, np.float32))
    return Det("gate", 0.95, cx - w / 2.0, cy - h / 2.0, w, h,
               kpts=np.zeros((4, 2), np.float32), kpt_conf=np.zeros(4, np.float32))


def raw_psi_series(rows):
    """由日志的 `hdg`（1 阶 EMA，alpha=0.4）反解每帧的**原始 ψ**（只在 full 帧有 hdg）。"""
    out, prev = [], None
    for r in rows:
        h = r.get("hdg")
        if h is None:
            out.append(prev)                     # 非 full 帧无 ψ：沿用上一帧（代码里 _hdg_deg 也是保留的）
            continue
        h = float(h)
        psi = h if prev is None else prev + (h - prev) / 0.4
        out.append(psi)
        prev = h
    return out


# ---------------------------------------------------------------- 假串口/装配
class _Telemetry(object):
    def __init__(self, yaw=None):
        self.yaw_deg = yaw


class ReplayUart(object):
    """记录 DOF 与转向下发；`follow=True` 时"假下位机照做"（否则遥测照日志给）。"""

    def __init__(self, follow=False):
        self.frames = []          # 每帧 (surge, sway, heave, yaw)
        self.turn_reqs = []       # (帧号, 角度)
        self.neutral_calls = 0
        self.stop_hard_calls = 0
        self.estop_active = False
        self.telemetry = _Telemetry(None)
        self.follow = bool(follow)
        self._pending = 0
        self._id = 0

    def send_dof(self, surge=0.0, sway=0.0, heave=0.0, yaw=0.0):
        self.frames.append((float(surge), float(sway), float(heave), float(yaw)))

    def neutral(self):
        self.neutral_calls += 1

    def set_motion(self, name, force=False):
        pass

    def request_turn(self, angle_deg):
        self._id += 1
        self.turn_reqs.append((self._id, float(angle_deg)))
        if self.follow and self.telemetry.yaw_deg is not None:
            # σ=-1：+角=右转 ⇒ 遥测航向减小（与 tests 的假下位机同约定）
            self.telemetry.yaw_deg = float(self.telemetry.yaw_deg) - float(angle_deg)
        self._pending = 1
        return True

    def poll_turn_complete(self):
        if self._pending <= 0:
            return False
        self._pending -= 1
        return True

    def cancel_turn(self):
        self._pending = 0

    def stop_hard(self, *a, **kw):
        self.stop_hard_calls += 1
        return True


class PairHub(object):
    def __init__(self):
        self.dets = []

    def has_extra(self, task):
        return task == "gate"

    def detect_list(self, task, frame):
        return list(self.dets)


# ---------------------------------------------------------------- 自检
def selftest():
    print("== 合成往返自检（full / width）==")
    ok = True
    for z, psi, dx, dy in [(1.780, -29.7, 0.236, -0.089), (1.830, -17.6, 0.129, -0.064),
                           (1.100, -34.6, 0.149, -0.036), (3.380, 0.0, 0.057, -0.136)]:
        rv, tv = pose_from(z, psi, dx, dy)
        uv = CAM.project(OBJ3, rv, tv)
        res = gate_pose(CAM, OBJ3, uv, prev=None, reproj_thr=20.0, z_bounds=(0.2, 15.0))
        if res is None:
            print("  z=%.3f ψ=%+.1f → PnP None ✗" % (z, psi)); ok = False; continue
        r2, t2 = res
        yaw2, _ = gate_normal_angles_deg(r2, t2)
        dz = abs(float(t2.ravel()[2]) - z); dpsi = abs(yaw2 - psi)
        print("  z=%.3f ψ=%+6.1f dx=%+.3f dy=%+.3f → 回解 z=%.3f ψ=%+6.1f  (Δz=%.2e Δψ=%.2e)"
              % (z, psi, dx, dy, float(t2.ravel()[2]), yaw2, dz, dpsi))
        ok = ok and dz < 1e-3 and dpsi < 1e-3
    for z in (3.123, 3.383, 3.691, 4.060):
        du = CAM.fx * 0.7 / z
        back = width_range_depth(700.0 - du / 2, 700.0 + du / 2, CAM.fx, 0.7)
        mode, ids = parse_kpt_mode(np.array([[700 - du / 2, 300], [700 + du / 2, 300],
                                             [0, 0], [0, 0]], np.float32),
                                   np.array([0.9, 0.9, 0, 0], np.float32), 0.8)
        print("  z=%.3f → Δu=%.1f px → 回解 %.3f, mode=%s ids=%s" % (z, du, back, mode, ids))
        ok = ok and abs(back - z) < 1e-3 and mode == "width"
    print("自检", "通过 ✓" if ok else "失败 ✗")
    return 0 if ok else 1


# ---------------------------------------------------------------- 主回放
def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def is_no_measurement(row):
    """日志里的"哨兵零"帧：`dx=dy=0 且 kpt=0` ⇒ **本帧没有任何可用偏差测量**。

    为什么必须这样判：`_set_info` 的 `dx/dy/kpt` 缺省值就是 `0.0`，
    "居中完美"与"这一帧根本没测"在日志里长得一模一样（见 ANALYSIS 特性清单 A/D）。
    回放时把它当"测到且正中"会凭空造出一堆 `creep`（实测一致性从 33% 掉到 63% 的差就在这里），
    所以这里一律**当成"没有检测"**（真代码走 `_tick_lost` → hold），这也与日志的 action=hold 吻合。
    """
    return (abs(float(row.get("dx") or 0.0)) < 1e-9
            and abs(float(row.get("dy") or 0.0)) < 1e-9
            and int(row.get("kpt") or 0) == 0)


def replay(rows, follow=False, kpt_raw=False, verbose=True, no_measure_none=True, ref="axis"):
    psi_raw = raw_psi_series(rows)
    uart = ReplayUart(follow=follow)
    hub = PairHub()
    task = GateTask(uart, hub, CAM.width, CAM.height)
    frame = np.zeros((CAM.height, CAM.width, 3), np.uint8)
    t0 = rows[0]["t"]

    hist = []
    for i, r in enumerate(rows):
        md = r.get("mode") or ""
        if no_measure_none and is_no_measurement(r):
            hub.dets = []                      # 哨兵零 ⇒ 当成"没有检测"
        elif md == "full":
            hub.dets = [det_full(r, psi_raw[i] if psi_raw[i] is not None else 0.0,
                                 CAM.width, ref=ref)]
        elif md == "width":
            hub.dets = [det_width(r, CAM.width, ref=ref)]
        elif md == "coarse":
            hub.dets = [det_coarse(r, CAM.width, with_kpts=kpt_raw, ref=ref)]
        else:
            hub.dets = []
        if uart.telemetry.yaw_deg is None or not follow:
            uart.telemetry.yaw_deg = r.get("tyaw")      # 遥测严格照日志（不 follow 时）
        now_ms = int(round(float(r["t"]) * 1000.0))
        st = task.process(frame, now_ms)
        info = dict(task.last_info)
        info["_f"] = i
        info["_t_rel"] = float(r["t"]) - t0
        info["_turn_n"] = len(uart.turn_reqs)
        hist.append(info)
        if st == S.STATUS_DONE:
            break
    return task, uart, hist, psi_raw


def compare(rows, hist):
    """回放 vs 日志 的逐帧一致性（判据/仲裁是否被复现）。"""
    cols = ("phase", "substate", "action", "mode")
    stat, first = {}, {}
    for c in cols:
        n = min(len(rows), len(hist))
        same = sum(1 for i in range(n) if str(rows[i].get(c) or "") == str(hist[i].get(c) or ""))
        stat[c] = (same, n)
        first[c] = next((i for i in range(n)
                         if str(rows[i].get(c) or "") != str(hist[i].get(c) or "")), None)
    return stat, first


def apply_old_behavior():
    """`--old`：把三处**日志期代码才有/没有**的行为还原，用于"按原来的参数模拟一遍"。

    1. `_center_ref` → **画面几何中心**（HEAD 版 `_on_pose` 就是这么算 dx/dy 的；
       主点版 `_center_ref` 是日志之后才加的）。
    2. `_on_width` 里的 `_relock_guard(z)`（`modes.py:161`）→ **日志期没有这道闸**，
       所以那时 width 档直接采纳了 z=3.691 / 4.060（> dist_max）。
       这里只在 mode=="width" 时短路它，`_on_pose` 的那道闸保持原样。
    3. `approach.tier` → `fast`（日志里 `action=forward_fast, surge=0.3`）。
    """
    from gate.motion import modes as _M
    from gate.motion import channels as _C
    # ⚠️ `_center_ref` 定义在 **GateChannels**（channels.py:93），不是在 GateModes 上
    _C.GateChannels._center_ref = lambda self: (self.w / 2.0, self.h / 2.0)
    _orig = _M.GateModes._relock_guard

    def _patched(self, z):
        if getattr(self, "mode", "") == _M.MODE_WIDTH:
            return False
        return _orig(self, z)
    _M.GateModes._relock_guard = _patched
    S.comm.gate["approach"]["tier"] = "fast"


def main(argv):
    if "--selftest" in argv:
        return selftest()
    # `--tier=fast|creep`：临时把 `comm.gate.approach.tier` 换成日志期的值（做保真度对照）
    for a in argv:
        if a.startswith("--tier="):
            S.comm.gate["approach"]["tier"] = a.split("=", 1)[1]
    paths = [a for a in argv if not a.startswith("--")]
    if not paths:
        print(__doc__)
        return 2
    follow = "--turn-follows" in argv
    kpt_raw = "--kpt-raw" in argv
    keep_sentinel = "--keep-sentinel" in argv   # 关掉"哨兵零⇒无检测"规则（看它影响多大）
    if "--old" in argv:
        apply_old_behavior()
    ref = "img" if "--ref=img" in argv else ("axis" if "--ref=axis" in argv else None)
    if ref is None:
        ref = "img" if "--old" in argv else "axis"
    for a in argv:
        if a.startswith("--ref="):
            ref = a.split("=", 1)[1]

    print("=" * 78)
    print("相机 fx=%.2f fy=%.2f cx=%.2f cy=%.2f（rectified=%s）  k_true=fx·W/w=%.4f"
          % (CAM.fx, CAM.fy, CAM.cx, CAM.cy, CAM.rectified, K_TRUE))
    print("cfg：z.cross=%.2f cross_confirm=%d dist_max=%.2f max_z_jump=%.2f | align px_x=%.2f px_y=%.2f"
          % (S.comm.gate.z.cross, S.comm.gate.z.cross_confirm_frames, S.comm.gate.z.dist_max_m,
             S.vision.gate.pnp.max_z_jump_m, S.comm.gate.align.px_x, S.comm.gate.align.px_y))
    print("     approach.tier=%s | through center=%d帧(%.2f/%.2f) require_align=%.1f° | hdg tol=%.1f° ok_frames=%d"
          % (S.comm.gate.approach.tier, S.comm.gate.through.center_frames,
             S.comm.gate.through.center_x, S.comm.gate.through.center_y,
             S.comm.gate.through.require_align_deg, S.comm.gate.hdg.tol_deg,
             S.comm.gate.hdg.ok_frames))
    print("模式：follow=%s  kpt_raw(反事实)=%s  基准折算 ref=%s（axis=折算到主点 / img=按画面中心原样）"
          % (follow, kpt_raw, ref))
    print("      主点相对画面中心：归一化 (%.4f, %.4f) ⇒ 日志的 dx/dy 折算到主点基准要各减这个数"
          % REF_SHIFT["axis"])
    print("=" * 78)

    for p in paths:
        rows = load(p)
        task, uart, hist, psi_raw = replay(rows, follow=follow, kpt_raw=kpt_raw,
                                           no_measure_none=not keep_sentinel, ref=ref)
        print("\n### %s（%d 帧，%.1f s）" % (p, len(rows), rows[-1]["t"] - rows[0]["t"]))

        stat, first = compare(rows, hist)
        print("--- 一致性（回放 vs 日志）---")
        for c in stat:
            a, b = stat[c]
            print("   %-9s 相同 %3d/%3d = %5.1f%%   首次分歧帧 %s"
                  % (c, a, b, 100.0 * a / max(b, 1), "-" if first[c] is None else first[c]))

        # ---- THROUGH ----
        print("--- THROUGH ---")
        th = [(h["_f"], h["_t_rel"]) for h in hist
              if h.get("phase") in (PH_THROUGH, PH_CREEP_THROUGH)
              or h.get("action") in ("through", "creep_through")]
        if th:
            f0, tt = th[0]
            print("   ✅ 首次进入 t=%.3f s（帧 %d），共 %d 帧；结束 reason=%s pass=%s"
                  % (tt, f0, len(th), hist[-1].get("reason"), hist[-1].get("pass")))
            h0 = hist[f0]
            print("      进入帧判据：z=%.3f dx=%+.3f dy=%+.3f action=%s surge=%.2f"
                  % (h0.get("z", 0), h0.get("dx", 0), h0.get("dy", 0), h0.get("action"),
                     h0.get("surge", 0)))
        else:
            print("   ❌ 全程未进 THROUGH（pass=%s reason=%s）"
                  % (hist[-1].get("pass"), hist[-1].get("reason")))
        blk = {}
        for h in hist:
            b = h.get("through_block")
            if b:
                blk.setdefault(str(b), []).append((h["_f"], round(h["_t_rel"], 3)))
        if blk:
            print("   被拦下的原因（`last_info['through_block']`）：")
            for why, lst in sorted(blk.items(), key=lambda kv: -len(kv[1])):
                print("      %-40s %4d 帧，首帧 t=%.3f s" % (why, len(lst), lst[0][1]))

        # ---- 转向 ----
        print("--- 转向下发 ---")
        if uart.turn_reqs:
            for i, (tid, deg) in enumerate(uart.turn_reqs):
                # 找这一帧（request_turn 发生在该帧处理过程中）
                print("   #%d 相对角 %+8.2f°  (第 %d 次 request_turn)" % (tid, deg, i + 1))
        else:
            print("   （无 request_turn：一次真实转向都没下发）")
        starts = [(h["_f"], h["_t_rel"], h.get("hdg"), h.get("turn_deg"), h.get("turn_dir"),
                   h.get("turn_raw_psi"), h.get("hdg_state"), h.get("hdg_entry"))
                  for h in hist if h.get("hdg_entry") or h.get("turn_deg") is not None]
        if starts:
            print("   起转事件（hdg_entry/turn_deg 首次出现的那一帧）：")
            seen = set()
            for f, tt, psi, deg, dr, rpsi, stt, ent in starts:
                key = (ent, round(deg or 0, 2))
                if key in seen:
                    continue
                seen.add(key)
                print("     帧 %-4d t=%7.3f s  ψ_ema=%s 下发角=%s 方向=%s ψ_raw=%s state=%s entry=%s"
                      % (f, tt, psi, deg, dr, rpsi, stt, ent))
        else:
            print("   （`_hdg_start` 一次都没被调用）")

        # ---- 反事实：只按 z≤cross 连续计数 ----
        cf = None
        cnt = 0
        for r in rows:
            z = float(r.get("z") or 0.0)
            if 0 < z <= S.comm.gate.z.cross:
                cnt += 1
            else:
                cnt = 0
            if cnt >= int(S.comm.gate.z.cross_confirm_frames):
                cf = (rows.index(r), float(r["t"]) - rows[0]["t"], z)
                break
        print("--- 反事实：若 `z ≤ cross` 的连续帧计数在所有档位都生效 ---")
        if cf:
            print("   首次满足 t=%.3f s（帧 %d，z=%.3f ≤ %.2f，连续 %d 帧）"
                  % (cf[1], cf[0], cf[2], S.comm.gate.z.cross, S.comm.gate.z.cross_confirm_frames))
        else:
            print("   无（日志里的 z 序列从未连续 %d 帧 ≤ %.2f）"
                  % (S.comm.gate.z.cross_confirm_frames, S.comm.gate.z.cross))
        # ---- 动作分布 ----
        print("--- 动作分布（回放 vs 日志）---")
        def cnt_of(seq, k):
            d = {}
            for x in seq:
                d[str(x.get(k) or "")] = d.get(str(x.get(k) or ""), 0) + 1
            return d
        for k in ("action", "mode"):
            a, b = cnt_of(rows, k), cnt_of(hist, k)
            keys = sorted(set(a) | set(b))
            print("   %-6s 日志 %-58s" % (k, {kk: a.get(kk, 0) for kk in keys}))
            print("   %-6s 回放 %-58s" % ("", {kk: b.get(kk, 0) for kk in keys}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
