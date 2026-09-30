# -*- coding: utf-8 -*-
"""gate_postproc.py — 检测后处理（解码之后、下游之前）：可信角点 / 几何合法 / 去重 / 选门键
不是模型契约；模型契约见该文件 §1–§3。
低 v 角点的**坐标是垃圾**（规范 §7.2）：不参与 PnP、也不用来推"门中心"，但**不能据此丢整帧** ——
门框不全时该降级处理（`parse_kpt_mode` 已经这么做）。
2. **L/R 归一 → 几何合法**（顺序见 `apply()`）：先按"门框正反两面一致"把左右标反的实例**换回来**
（`lr_normalize`，占位 `(0,0)` 不参与、只看到一侧不动），再判几何（`TL.x<TR.x`、`BL.x<BR.x`、
3. **去重（重复框）**：同一个门被两个尺度各检出一次 → 丢小框。判据 = **小框 conf 更低** 且
**≥ MIN_EDGES 条边与小框短边在 EDGE_TOL 比例内重合**。
4. **选门键**（函数在这里，**决策在 `gate_task._pick_gate`**）：
`near`（默认，规范 §4）= **近距离优先**，用**框宽做测距代理**（`z ≈ fx·W/w`，单调）；
调用点：`gate_decode.GateKeypointBackend.detect()` 在解码后调 `apply()`（几何 + 去重），"""
from __future__ import annotations

import numpy as np

from common.cfg.cfgnode import flag, num, sub

# 缺键兜底：**必须与 cfg/vision.yaml 同值**（用例 test_gate_postproc 的守卫守着）
_D_DET = dict(conf=0.5)                  # 候选框 score 阈值（只作用于 gate，不动共用的 model.score_threshold）
_D_POST = dict(edge_tol=0.10, min_edges=2, geom_check=True, lr_norm=True)
_D_SELECT = dict(mode="near")            # near(规范) | corner(旧行为)
_D_VMIN = 0.8                            # 角点可信下限的兜底（真值 = vision.gate.keypoint.conf_thr）

_BOOL_KEYS = ("geom_check", "lr_norm")

def det_cfg(conf=None, node=None):
    """候选阈值：显式参数 > cfg `vision.gate.det.conf` > 共用的 `model.score_threshold` > 0.6。"""
    if conf is not None:
        try:
            return float(conf)
        except (TypeError, ValueError):
            pass
    if node is None:
        try:
            import base.cfg.settings as S
            node = S.get("vision.gate.det", None)
        except Exception:
            node = None
    if isinstance(node, dict) and node.get("conf") is not None:
        return num(node, "conf", _D_DET["conf"])
    try:
        import base.cfg.settings as S
        v = S.get("vision.model.score_threshold", None)
        return float(v) if v is not None else _D_DET["conf"]
    except Exception:
        return _D_DET["conf"]

def postproc_cfg(node=None):
    """读 `vision.gate.postproc`（缺键 → 代码默认，不抛异常）。"""
    if node is None:
        try:
            import base.cfg.settings as S
            node = S.get("vision.gate.postproc", None)
        except Exception:
            node = None
    if not isinstance(node, dict):
        return dict(_D_POST)
    return dict(edge_tol=num(node, "edge_tol", _D_POST["edge_tol"]),
                min_edges=int(num(node, "min_edges", _D_POST["min_edges"])),
                geom_check=flag(node, "geom_check", _D_POST["geom_check"]),
                lr_norm=flag(node, "lr_norm", _D_POST["lr_norm"]))

def select_cfg(node=None):
    """读 `vision.gate.select`（缺键 → `mode: near`，规范 §4）。"""
    if node is None:
        try:
            import base.cfg.settings as S
            node = S.get("vision.gate.select", None)
        except Exception:
            node = None
    mode = "near"
    if isinstance(node, dict) and node.get("mode"):
        mode = str(node.get("mode")).strip().lower()
    if mode not in ("near", "corner"):
        mode = "near"                    # 拼错不崩，回落到规范值
    return dict(mode=mode)

def v_min(conf_thr=None):
    """角点可信下限 V_MIN：显式参数 > cfg `vision.gate.keypoint.conf_thr` > 0.8。"""
    if conf_thr is not None:
        try:
            return float(conf_thr)
        except (TypeError, ValueError):
            pass
    try:
        import base.cfg.settings as S
        v = S.get("vision.gate.keypoint.conf_thr", None)
        return float(v) if v is not None else _D_VMIN
    except Exception:
        return _D_VMIN

# ---------------------------------------------------------------------------
# 1) 可信角点
# ---------------------------------------------------------------------------
def trusted_mask(det, conf_thr=None):
    """(K,) bool：本实例里**可信**的角点。无角点信息 → 全 False。"""
    kc = getattr(det, "kpt_conf", None)
    if kc is None:
        return np.zeros(0, dtype=bool)
    return np.asarray(kc, np.float64) >= float(v_min(conf_thr))

def n_trusted(det, conf_thr=None):
    return int(trusted_mask(det, conf_thr).sum())

def conf_sum(det):
    kc = getattr(det, "kpt_conf", None)
    return 0.0 if kc is None else float(np.asarray(kc, np.float64).sum())

def vis_thr(conf_thr=None):
    """**可见性下限**：显式参数 > cfg `vision.gate.keypoint.vis_thr` > V_MIN。

    与 `trusted_mask` 的 V_MIN 口径**不是一个东西**：可见只说明"这个角点有坐标"，
    几何检验 / L-R 归一看它（最基本的检查，不该被逐角点置信度卡住）。
    """
    try:
        import base.cfg.settings as S
        v = S.get("vision.gate.keypoint.vis_thr", None)
        if v is not None:
            return float(v)
    except Exception:
        pass
    return v_min(conf_thr)

def visible_mask(det, v_thr=None, frame_wh=None):
    """(K,) bool：本实例里**可见**的角点 = 置信度 ≥ 阈值 **且坐标不是 `(0,0)` 占位**。
    模型对没找到的角点会输出 `(0,0)` 而 `v` 仍可能不低 —— 只看 `v` 会把占位当可见，"""
    kc = getattr(det, "kpt_conf", None)
    kp = getattr(det, "kpts", None)
    if kc is None or kp is None:
        return np.zeros(0, dtype=bool)
    kc = np.asarray(kc, np.float64)
    kp = np.asarray(kp, np.float64)
    thr = float(vis_thr() if v_thr is None else v_thr)
    m = kc >= thr
    if kp.ndim == 2 and kp.shape[0] == m.shape[0]:
        m = m & ~((kp[:, 0] == 0) & (kp[:, 1] == 0))
        if frame_wh:
            fw, fh = float(frame_wh[0]), float(frame_wh[1])
            m = m & (kp[:, 0] >= 0) & (kp[:, 0] < fw) & (kp[:, 1] >= 0) & (kp[:, 1] < fh)
    return m

def swap_corners(det, pairs=((0, 1), (3, 2))):
    """就地交换角点（`kpts` 与 `kpt_conf` 同步换）。默认 = **TL↔TR、BL↔BR**（L/R 归一用）。"""
    kp = getattr(det, "kpts", None)
    kc = getattr(det, "kpt_conf", None)
    if kp is None:
        return False
    for i, j in pairs:
        if i >= len(kp) or j >= len(kp):
            return False
        tmp = kp[i].copy()
        kp[i] = kp[j]
        kp[j] = tmp
        if kc is not None and i < len(kc) and j < len(kc):
            t2 = kc[i].copy()
            kc[i] = kc[j]
            kc[j] = t2
    return True

def lr_normalize(det, v_thr=None, frame_wh=None):
    """**L/R 归一**（用户 2026-09-28 定）：门框正反两面一致 ⇒ 模型把左右标反时**自己换回来**。
    判据：左列 `TL,BL` 与右列 `TR,BR` **两侧都至少有一个可见角点**，且**左列可见点的 x 均值 > 右列**
    Returns: True = 检测到标反并已交换；False = 没动。"""
    kp = getattr(det, "kpts", None)
    if kp is None:
        return False
    kp = np.asarray(kp)
    if kp.ndim != 2 or kp.shape[0] != 4:
        return False
    m = visible_mask(det, v_thr, frame_wh)
    if m.shape[0] != 4:
        return False
    li = [i for i in (0, 3) if m[i]]                  # TL, BL（左列）
    ri = [i for i in (1, 2) if m[i]]                  # TR, BR（右列）
    if not li or not ri:
        return False                                  # 只有一侧可见 ⇒ 判不出左右
    lx = float(np.mean([float(kp[i][0]) for i in li]))
    rx = float(np.mean([float(kp[i][0]) for i in ri]))
    if lx <= rx:
        return False                                  # 已经 L 在左
    return swap_corners(det)

# ---------------------------------------------------------------------------
# 2) 几何合法
# ---------------------------------------------------------------------------
def _seg_cross(p1, p2, p3, p4):
    """线段 (p1,p2) 与 (p3,p4) 是否真相交（共线/端点相触不算）。"""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1, d2 = cross(p3, p4, p1), cross(p3, p4, p2)
    d3, d4 = cross(p1, p2, p3), cross(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))

def quad_legal(kpts, ids=None):
    """四角是否**顺序合法 + 不自交** —— **只判"两端都可见"的那些关系**（2026-09-28 改）。
    Args:
    kpts: (K,2) 像素（顺序 TL,TR,BR,BL）
    ids:  **可见**角点索引（`None` = 当作全可见）
    Returns:
    True = 合法（或点数 ≠ 4 —— 非四角实例不归本函数管）
    · 自交只在**四角都可见**时判（判不了就不判）。"""
    if kpts is None:
        return True
    k = np.asarray(kpts, np.float64)
    if k.ndim != 2 or k.shape[0] != 4:
        return True
    vis = set(range(4)) if ids is None else set(int(i) for i in ids)
    tl, tr, br, bl = k[0], k[1], k[2], k[3]
    # 顺序：上边两点左→右、下边两点左→右、左边两点上→下、右边两点上→下（各关系两端都可见才判）
    if (0 in vis and 1 in vis) and not (tl[0] < tr[0]):
        return False
    if (3 in vis and 2 in vis) and not (bl[0] < br[0]):
        return False
    if (0 in vis and 3 in vis) and not (tl[1] < bl[1]):
        return False
    if (1 in vis and 2 in vis) and not (tr[1] < br[1]):
        return False
    if len(vis) < 4:
        return True
    # 自交只在四角都可见时判：非相邻边相交（TL-TR × BR-BL 或 TR-BR × BL-TL）
    if _seg_cross(tl, tr, br, bl):
        return False
    if _seg_cross(tr, br, bl, tl):
        return False
    return True

# ---------------------------------------------------------------------------
# 3) 去重（同一个门被两个尺度各检出一次）
# ---------------------------------------------------------------------------
def _edge_hits(small, big, tol_px):
    """小框与"对应边"重合的条数（左/上/右/下 各比一次）。"""
    hits = 0
    for va, vb in ((small[0], big[0]),                 # left
                   (small[1], big[1]),                 # top
                   (small[0] + small[2], big[0] + big[2]),   # right
                   (small[1] + small[3], big[1] + big[3])):  # bottom
        if abs(float(va) - float(vb)) <= tol_px:
            hits += 1
    return hits

def dedup(dets, conf_thr=None, cfg=None, log=None):
    """重复框去重：**小框 conf 更低 且 ≥min_edges 条边重合 → 丢小框**。"""
    c = postproc_cfg(cfg)
    tol_frac = float(c["edge_tol"])
    min_edges = max(1, int(c["min_edges"]))
    gates = [d for d in dets if d.kind == "gate"]
    if len(gates) < 2:
        return list(dets)
    drop = set()
    order = sorted(range(len(gates)), key=lambda i: gates[i].score, reverse=True)
    for a in range(len(gates)):
        for b in range(a + 1, len(gates)):
            da, db = gates[a], gates[b]
            small, big = (da, db) if (da.w * da.h) <= (db.w * db.h) else (db, da)
            if small.score >= big.score:          # 小框必须 conf 更低，否则不动
                continue
            tol_px = tol_frac * max(1.0, float(min(small.w, small.h)))
            if _edge_hits((small.x, small.y, small.w, small.h),
                          (big.x, big.y, big.w, big.h), tol_px) >= min_edges:
                drop.add(id(small))
                if log is not None:
                    log("[GATE] 去重：丢小框 score=%.2f %dx%d（与大框 score=%.2f %dx%d 有 ≥%d 条边重合）"
                        % (small.score, small.w, small.h, big.score, big.w, big.h, min_edges))
    # 保序输出（下游按 score 降序，别打乱）
    out = []
    for d in dets:
        if d.kind == "gate" and id(d) in drop:
            continue
        out.append(d)
    return out

# ---------------------------------------------------------------------------
# 统一入口（解码后立刻调用）
# ---------------------------------------------------------------------------
def apply(dets, conf_thr=None, cfg=None, log=None, frame_wh=None):
    """解码后处理（规范 §4 的**固定顺序**）：**① L/R 归一 → ② 几何合法 → ③ 去重**。
    · ② 只判"两端都可见"的关系（`visible_mask` 口径，**不是** V_MIN 可信口径）；
    · ③ 重复框去重（小框 conf 更低 且 ≥2 条边重合）。
    返回新列表（保序）；`Det` 对象本身会被 ① 就地改写（交换角点），因为下游 PnP 要用换好的角点。"""
    c = postproc_cfg(cfg)
    out = []
    for d in dets:
        if d.kind == "gate":
            if c.get("lr_norm", True) and lr_normalize(d, frame_wh=frame_wh):
                if log is not None:
                    log("[GATE] L/R 归一：该实例左右标反 → 已交换 TL↔TR / BL↔BR")
            if c["geom_check"]:
                ids = np.where(visible_mask(d, frame_wh=frame_wh))[0]
                if not quad_legal(d.kpts, ids):
                    if log is not None:
                        log("[GATE] 丢实例：四角顺序/自交不合法（可见 %d 个角点）" % len(ids))
                    continue
        out.append(d)
    return dedup(out, conf_thr=conf_thr, cfg=cfg, log=log)

# ---------------------------------------------------------------------------
# 4) 选门键（决策在 gate_task._pick_gate）
# ---------------------------------------------------------------------------
def near_key(det, conf_thr=None):
    """规范 §4「近距离优先」：框宽 = 测距代理（`z ≈ fx·W/w`，单调）→ 大的先。"""
    return (int(det.w), n_trusted(det, conf_thr), conf_sum(det), float(det.score))

def corner_key(det, conf_thr=None):
    if getattr(det, "kpt_conf", None) is None:
        return (-1, 0.0, float(det.score))
    return (n_trusted(det, conf_thr), conf_sum(det), float(det.score))

def pick(dets, conf_thr=None, cfg=None):
    """按 cfg `vision.gate.select.mode` 选目标门（无 gate 实例 → None）。"""
    mode = select_cfg(cfg)["mode"]
    key = near_key if mode == "near" else corner_key
    best, best_key = None, None
    for d in dets:
        if d.kind != "gate":
            continue
        k = key(d, conf_thr)
        if best_key is None or k > best_key:
            best, best_key = d, k
    return best
