# -*- coding: utf-8 -*-
"""gate/data/cues.py — 恢复线索 + 线索↔位姿互相自检

作用：当**位姿链长时间不可用**（点缺失/倒影导致解算失败）时，用"哪条边可见 + 框形"
给出一个**方向明确的低速恢复动作**，避免只能干等或反复后退。

| 观察 | 动作 |
|---|---|
| 框高≥ratio 且上下贴边 | BACKWARD（最高优先级） |
| 仅 TL,BL（左列） | TURN_RIGHT |
| 仅 TR,BR（右列） | TURN_LEFT |
| 仅 TL,TR（上边） | DESCEND |
| 仅 BR,BL（下边） | ASCEND |
| 单点/对角/3 点/完整四点 | None（不猜；完整四点交给位姿链） |

它**只吃共享状态**（融合后的可见性掩码 + 检测框），不自己做检测/阈值；
默认在 `GateTask` 里只在"位姿不可用 ≥ cue_after_frames 且处于 SEARCH/ALIGN"时启用，
`record_only: true` 时只自检不驱动运动（下水前先这样核对方向符号）。

`cue_conflicts()` / `quality_agreement()`：**两条链互相自检**——线索说"太近"而位姿说
还远 → 至少一方错，此时不驱动、只把质量分打折（`gate/data/quality.py` 用它）；一致则小幅加成。
质量分再反向决定线索动作的速度与是否够胆驱动。
参数：`cfg/comm.yaml → gate.data.cues:`。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 主导权三档（`CueCfg.mode`；旧名 pose_first/blend/cue_first 仍被接受，见 _ALIASES）
MODE_POSE = "pose"     # 位姿主导：线索只在位姿不可用时兜底
MODE_AUTO = "auto"     # **默认**：依据质量自动分配（Q 差 → 线索份额上升）
MODE_CUE = "cue"       # 线索主导：有明确线索就整帧听线索
_ALL_MODES = (MODE_POSE, MODE_AUTO, MODE_CUE)
_ALIASES = {"pose_first": MODE_POSE, "pose_lead": MODE_POSE, "pose_only": MODE_POSE,
            "blend": MODE_AUTO, "quality": MODE_AUTO, "auto_quality": MODE_AUTO,
            "cue_first": MODE_CUE, "cue_lead": MODE_CUE, "cue_only": MODE_CUE}

TURN_RIGHT = "TURN_RIGHT"
TURN_LEFT = "TURN_LEFT"
DESCEND = "DESCEND"
ASCEND = "ASCEND"
BACKWARD = "BACKWARD"

# 掩码顺序：TL, TR, BR, BL
_M_LEFT = (True, False, False, True)
_M_RIGHT = (False, True, True, False)
_M_TOP = (True, True, False, False)
_M_BOTTOM = (False, False, True, True)


@dataclass
class CueCfg:
    enable: bool = True
    record_only: bool = False      # true=只自检/记录，不让线索驱动运动
    speed: float = 0.15            # 线索动作统一低速（归一化 DOF）
    too_close_height_ratio: float = 0.9
    boundary_margin_ratio: float = 0.02
    turn_right_on_left_column: bool = True
    turn_left_on_right_column: bool = True
    descend_on_top_edge: bool = True
    ascend_on_bottom_edge: bool = True
    backward_on_too_close: bool = True
    backward_agree_z_m: float = 1.5   # 线索 BACKWARD 但位姿 z 仍大于它 → 视为冲突
    agree_bonus: float = 0.15         # 线索与位姿一致 → 质量分加成（封顶 1.0）
    conflict_penalty: float = 0.5     # 线索与位姿冲突 → 质量分打折（只降权，不驱动）
    cue_after_frames: int = 15        # 位姿连续不可用多少帧后，线索才接管
    cue_confirm_s: float = 0.10       # 线索动作确认时长（秒）
    cue_confirm_frames: int = 3       # 线索动作确认帧数
    # --- 主导权（v1.5）：**三档**，见 `cue_mode()` ---
    #   auto （默认）依据质量自动分配：位姿 Q 高 → 位姿主导；Q 低 → 线索份额上升
    #   pose        位姿主导：位姿在位时线索一律不驱动，只在位姿不可用时兜底
    #   cue         线索主导：只要有明确线索动作，就整帧由线索说了算
    mode: str = "auto"
    allow_approach: bool = False      # 线索是否允许在 APPROACH 相位生效（主导时通常要 True）
    cue_pose_share: float = 0.0       # **仅 cue 档**：给位姿保留的份额（0=完全接管，
                                      # 0.2=线索没动到的通道仍留 20% 位姿修正；上限 0.5）
    speed_lead: float = 0.30          # 主导档（auto/cue）线索速度（归一化 DOF）
    speed_uses_caution: bool = False  # **仅主导档**（auto/cue）：线索速度是否再乘谨慎度
                                      # （默认不乘，否则不可信的位姿链仍会通过 Q 拖慢线索）
    lead_q: float = 0.35              # 位姿后验 Q 低于它 → 线索权重开始 > 0
    lead_full_q: float = 0.20         # 位姿 Q 低于它 → 线索权重 = 1（完全主导）
    conflict_w: float = 1.0           # **冲突时**线索权重下限（0=听位姿，1=完全听线索；
                                      # 倒影最常骗的正是位姿，"哪条边可见"更直接）
    lead_min_w: float = 0.25          # 权重小于它就不掺和（避免两链各出一点力来回抖）

    @classmethod
    def from_settings(cls, S):
        cfg = cls()
        data = S.get("comm.gate.cues") or {}
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


def cue_mode(cfg):
    """把 `cfg.mode` 归一化成三档之一（接受旧名别名；未知值 → auto）。"""
    m = str(getattr(cfg, "mode", MODE_AUTO) or MODE_AUTO).strip().lower()
    m = _ALIASES.get(m, m)
    return m if m in _ALL_MODES else MODE_AUTO


def is_lead(cfg):
    """是否处于"线索有份额"的档位（auto/cue）；pose 档只做兜底。"""
    return cue_mode(cfg) != MODE_POSE


def visible_mask(conf, conf_thr):
    c = np.asarray(conf, float).reshape(-1) if conf is not None else np.zeros(0)
    if c.size < 4:
        return (False, False, False, False)
    return tuple(bool(v) for v in (c[:4] >= float(conf_thr)))


def too_close(bbox_xyxy, image_size, cfg):
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    w, h = float(image_size[0]), float(image_size[1])
    if h <= 0 or x2 <= x1 or y2 <= y1:
        return False
    margin = h * float(cfg.boundary_margin_ratio)
    return ((y2 - y1) / h >= float(cfg.too_close_height_ratio)
            and y1 <= margin and y2 >= h - margin)


def recovery_cue(mask, bbox_xyxy, image_size, cfg):
    """返回 (action|None, reason)。"""
    if not cfg.enable:
        return None, "cues disabled"
    m = tuple(bool(v) for v in mask)
    if len(m) != 4:
        return None, "bad mask"
    if cfg.backward_on_too_close and too_close(bbox_xyxy, image_size, cfg):
        return BACKWARD, "too close (box touches top & bottom)"
    if cfg.turn_right_on_left_column and m == _M_LEFT:
        return TURN_RIGHT, "only left column visible"
    if cfg.turn_left_on_right_column and m == _M_RIGHT:
        return TURN_LEFT, "only right column visible"
    if cfg.descend_on_top_edge and m == _M_TOP:
        return DESCEND, "only top edge visible"
    if cfg.ascend_on_bottom_edge and m == _M_BOTTOM:
        return ASCEND, "only bottom edge visible"
    if all(m):
        return None, "full gate -> pose chain"
    return None, "partial gate has no unambiguous cue"


def cue_conflicts(cue_action, pose_ok, z, cfg):
    """线索链 ↔ 位姿链 的**互相自检**：返回冲突原因（无冲突返回 ""）。

    只做证据可靠的那一条：线索判"太近"（BACKWARD），而位姿说还很远 → 至少一方错，
    此时不驱动、只记录（质量分也会因此打折）。
    """
    if cue_action == BACKWARD and pose_ok and z is not None \
            and float(z) > float(cfg.backward_agree_z_m):
        return "cue=BACKWARD(too close) vs pose z=%.2fm" % float(z)
    return ""


def quality_agreement(cue_action, pose_ok, z, q_total, cfg):
    """把"线索 vs 位姿"的一致性折进质量分：返回 (q_total', agree|None)。

    * 有线索**且**有位姿：一致 → `+agree_bonus`（封顶 1.0）；冲突 → `×conflict_penalty`；
    * 只有线索（位姿还没解出来）：既不算一致也不算冲突 → 原样返回、agree=None
      —— "线索单独存在"是正常降级状态，不该被当成矛盾来压低质量分。
    """
    q = float(q_total)
    if cue_action is None or not pose_ok:
        return q, None
    conflict = cue_conflicts(cue_action, True, z, cfg)
    if conflict:
        return float(q * float(cfg.conflict_penalty)), False
    return float(min(1.0, q + float(cfg.agree_bonus))), True


# ------------------------------------------------------------------ 主导权仲裁
def cue_weight(cfg, q_pose=None, cue=None, pose_ok=False, z=None):
    """线索链的**主导权重** wc ∈ [0,1]（0=完全由位姿说了算，1=完全由线索说了算）。

    三档（`cue_mode(cfg)`；**默认 auto**）：

    | 档 | 位姿在位且 Q 高 | 位姿在位且 Q 低 | 位姿不可用 | 冲突(线索↔位姿矛盾) |
    |---|---|---|---|---|
    | `pose` 位姿主导 | 0（线索不参与） | 0 | 1（兜底接管） | 不参与（只给 Q 打折） |
    | `auto` 依据质量自动分配 | 0 | `0<wc<1` 连续交接 | 1 | `max(wc, conflict_w)` |
    | `cue` 线索主导 | `1-cue_pose_share` | 同左 | 1 | 同左 |

    * `auto` 的连续交接：`Q ≥ lead_q → 0`；`Q ≤ lead_full_q → 1`；中间线性——
      主导权随证据强弱平滑转移（与全项目的软权重风格一致，不做二值切换）。
    * `cue` 档不再看 Q：只要相位允许、有明确线索动作，就整帧听线索；
      `cue_pose_share>0` 时给位姿留一份（线索没动到的通道仍有位姿修正）。
    * **冲突**：线索判"太近"(BACKWARD) 而位姿说 z 还很大 → 至少一方错。
      `conflict_w=1.0`（默认）表示此时听线索（"哪条边可见"更直接，且后退是安全方向）；
      设 0 = 冲突时仍听位姿，0.5 = 各让一半。**`pose` 档不参与冲突仲裁**。
    """
    if not cfg.enable or cfg.record_only:
        return 0.0
    m = cue_mode(cfg)
    if not pose_ok:
        return 1.0                       # 三档一致：位姿不在，线索兜底
    if m == MODE_POSE:
        w = 0.0                          # 位姿主导：位姿在位时线索不参与
    elif m == MODE_CUE:
        w = 1.0 - float(np.clip(cfg.cue_pose_share, 0.0, 0.5))
    else:                                # MODE_AUTO：按位姿质量连续分配
        q = 1.0 if q_pose is None else float(q_pose)
        lo, hi = float(cfg.lead_full_q), float(cfg.lead_q)
        if q >= hi:
            w = 0.0
        elif q <= lo:
            w = 1.0
        else:
            w = (hi - q) / max(1e-6, hi - lo)
    if is_lead(cfg) and cue is not None and cue_conflicts(cue, True, z, cfg):
        w = max(w, float(cfg.conflict_w))   # pose 档不参与：它就是"位姿说了算"
    return float(np.clip(w, 0.0, 1.0))


def cue_dof(action, speed):
    """线索动作 → 4 通道 DOF 字典（与 motion 层 `_CUE_DOF` 同一张表）。"""
    table = {TURN_RIGHT: ("yaw", 1.0), TURN_LEFT: ("yaw", -1.0),
             DESCEND: ("heave", -1.0), ASCEND: ("heave", 1.0),
             BACKWARD: ("surge", -1.0)}
    dof = {"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}
    if action is None:
        return dof
    ch, sign = table.get(action, ("surge", 0.0))
    dof[ch] = float(sign) * float(speed)
    return dof


def blend_dof(pose_dof, cue_dofs, wc):
    """按主导权重把"位姿 DOF"与"线索 DOF"线性混合（各通道独立，结果仍在 ±1 内）。

    `wc=0` → 纯位姿（现状）；`wc=1` → 纯线索。混合而不是"谁赢谁全出"，
    是为了在主导权交接的那几帧里不产生速度跳变（会直接体现在船的姿态上）。
    """
    wc = float(np.clip(wc, 0.0, 1.0))
    out = {}
    for ch in ("surge", "sway", "heave", "yaw"):
        a = float((pose_dof or {}).get(ch, 0.0))
        b = float((cue_dofs or {}).get(ch, 0.0))
        out[ch] = float(np.clip((1.0 - wc) * a + wc * b, -1.0, 1.0))
    return out
