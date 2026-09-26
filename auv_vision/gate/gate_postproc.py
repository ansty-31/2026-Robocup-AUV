# -*- coding: utf-8 -*-
"""gate_postproc.py — 检测后处理（解码之后、下游之前）：可信角点 / 几何合法 / 去重 / 选门键

**依据**：`doc/gate_pose_decode_spec.md` §4（2026-09-26 用户给定，与"阶段一"权重的评测/判定工作点同值）。
这份规范是**工程约束**（本仓自设），不是模型契约；模型契约见该文件 §1–§3。

四条规则（全部可配置，缺键不崩；纯函数，不碰串口/相机/状态）：

  1. **角点可信**：逐点 `v ≥ V_MIN`（= `vision.gate.keypoint.conf_thr`，现 0.8）才算"看见" ——
     与 `gate_frontend.parse_kpt_mode` **同一口径**（4 个 = full / 3 = p3p / 对向 2 = width / 否则 coarse）。
     低 v 角点的**坐标是垃圾**（规范 §7.2）：不参与 PnP、也不用来推"门中心"，但**不能据此丢整帧** ——
     门框不全时该降级处理（`parse_kpt_mode` 已经这么做）。
  2. **几何合法**：四角**不自交** + **顺序合法**（`TL.x<TR.x`、`BL.x<BR.x`、`TL.y<BL.y`、`TR.y<BR.y`）。
     两道都只在"四角可信"的实例上判；退化实例直接丢（规范实测触发率 0~1.5%）。
  3. **去重（重复框）**：同一个门被两个尺度各检出一次 → 丢小框。判据 = **小框 conf 更低** 且
     **≥ MIN_EDGES 条边与小框短边在 EDGE_TOL 比例内重合**。
     ⚠️ **不看"包含"**：远处第二个门可能落在近门框内，用包含会误删真门（规范实测误删 15 个）。
  4. **选门键**（函数在这里，**决策在 `gate_task._pick_gate`**）：
     `near`（默认，规范 §4）= **近距离优先**，用**框宽做测距代理**（`z ≈ fx·W/w`，单调）；
     并列再比可信角点数 / 置信度和 / score。`corner` = 2026-09-18 的旧行为（角点优先，防倒影抢门）。

调用点：`gate_decode.GateKeypointBackend.detect()` 在解码后调 `apply()`（几何 + 去重），
所以 **任务与预览看到的是同一批实例**；选门由 `gate_task._pick_gate()` 用 `pick()`。
"""
from __future__ import annotations

import numpy as np

from common.cfgnode import flag, num, sub

# 缺键兜底：**必须与 cfg/vision.yaml 同值**（用例 test_gate_postproc 的守卫守着）
_D_DET = dict(conf=0.6)                  # 候选框 score 阈值（只作用于 gate，不动共用的 model.score_threshold）
_D_POST = dict(edge_tol=0.10, min_edges=2, geom_check=True)
_D_SELECT = dict(mode="near")            # near(规范) | corner(旧行为)
_D_VMIN = 0.8                            # 角点可信下限的兜底（真值 = vision.gate.keypoint.conf_thr）

_BOOL_KEYS = ("geom_check",)


def det_cfg(conf=None, node=None):
    """候选阈值：显式参数 > cfg `vision.gate.det.conf` > 共用的 `model.score_threshold` > 0.6。"""
    if conf is not None:
        try:
            return float(conf)
        except (TypeError, ValueError):
            pass
    if node is None:
        try:
            import base.settings as S
            node = S.get("vision.gate.det", None)
        except Exception:
            node = None
    if isinstance(node, dict) and node.get("conf") is not None:
        return num(node, "conf", _D_DET["conf"])
    try:
        import base.settings as S
        v = S.get("vision.model.score_threshold", None)
        return float(v) if v is not None else _D_DET["conf"]
    except Exception:
        return _D_DET["conf"]


def postproc_cfg(node=None):
    """读 `vision.gate.postproc`（缺键 → 代码默认，不抛异常）。"""
    if node is None:
        try:
            import base.settings as S
            node = S.get("vision.gate.postproc", None)
        except Exception:
            node = None
    if not isinstance(node, dict):
        return dict(_D_POST)
    return dict(edge_tol=num(node, "edge_tol", _D_POST["edge_tol"]),
                min_edges=int(num(node, "min_edges", _D_POST["min_edges"])),
                geom_check=flag(node, "geom_check", _D_POST["geom_check"]))


def select_cfg(node=None):
    """读 `vision.gate.select`（缺键 → `mode: near`，规范 §4）。"""
    if node is None:
        try:
            import base.settings as S
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
        import base.settings as S
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
    """四角是否**顺序合法 + 不自交**（只在恰好 4 个点的实例上判）。

    Args:
        kpts: (K,2) 像素
        ids:  可信角点索引（None = 全部）
    Returns:
        True = 合法（或点数 ≠ 4 —— 非四角实例不归本函数管）
    """
    if kpts is None:
        return True
    k = np.asarray(kpts, np.float64)
    if k.ndim != 2 or k.shape[0] != 4:
        return True
    if ids is not None and len(ids) != 4:
        return True
    tl, tr, br, bl = k[0], k[1], k[2], k[3]
    # 顺序：上边两点左→右、下边两点左→右、左边两点上→下、右边两点上→下
    if not (tl[0] < tr[0] and bl[0] < br[0] and tl[1] < bl[1] and tr[1] < br[1]):
        return False
    # 自交：非相邻边相交（TL-TR × BR-BL 或 TR-BR × BL-TL）
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
    """重复框去重：**小框 conf 更低 且 ≥min_edges 条边重合 → 丢小框**。

    ⚠️ 判据**不是包含关系**：远处第二个门可能落在近门框内（规范实测用包含会误删真门 15 个）。
    """
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
def apply(dets, conf_thr=None, cfg=None, log=None):
    """解码后处理：几何合法（四角实例）→ 去重。返回新列表（不改动传入对象）。"""
    c = postproc_cfg(cfg)
    out = []
    for d in dets:
        if d.kind == "gate":
            ids = np.where(trusted_mask(d, conf_thr))[0]
            if c["geom_check"] and len(ids) == 4 and not quad_legal(d.kpts, ids):
                if log is not None:
                    log("[GATE] 丢实例：四角顺序/自交不合法（%d 个可信角点）" % len(ids))
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
    """2026-09-18 旧行为：角点更全/置信和更高优先，最后才比 score（防倒影抢门）。"""
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
