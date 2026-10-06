# -*- coding: utf-8 -*-
"""gate/motion/params.py — 常量与**缺键兜底表**（纯数据，无逻辑）
（详细用法、判据与实测见 doc/注释历史.md）"""
from __future__ import annotations

from gate.percept.geometry import GATE_FRAME_W, GATE_FRAME_H

PH_SEARCH = "SEARCH"
PH_ALIGN = "ALIGN"
PH_APPROACH = "APPROACH"
PH_THROUGH = "THROUGH"
PH_CREEP_THROUGH = "CREEP_THROUGH"
SUB_HDG = "HDG"
SUB_GOLDEN = "GOLDEN"
SUB_CREEP = "CREEP"
SUB_HOLD = "HOLD"
SUB_REACQUIRE = "REACQUIRE"
SUB_SWAY_BACK = "SWAY_BACK"
_K_ALIGN = (
    "confirm_frames",
    "px_x",
    "px_y"
)
_K_LOITER = (
    "dx_max",
    "dy_max",
    "enable",
    "near_ratio",
    "pose_hist_min",
    "pose_hist_ms",
    "timeout_ms"
)
_K_Z = (
    "align_max",
    "cross",
    "cross_confirm_frames",
    "dist_max_m",
    "fast_max",
    "near_lost_ratio",
    "relock_away_m",
    "relock_far_ratio",
    "relock_z_jump_m",
    "slow_max"
)
_K_APPROACH = (
    "tier",
)   # ★ 进近速档（舍弃 forward_fast，用 creep 试探）
_K_SURGE = (
    "creep",
    "lost_backward",
    "reacquire",
    "through"
)
_K_CREEP_THROUGH = (
    "creep_ms",
)   # creep_through：门口过门，慢速 creep 冲门时长(ms)
_K_COARSE = (
    "align_x",
    "align_y",
    "far_ratio",
    "near_ratio"
)
_K_WIDTH = (
    "z_max",
)
_K_TASK = (
    "timeout_ms",
    "pass_target",
    "pose_hold_frames"
)
_K_KPT = (
    "conf_thr",
)
_K_HOLD = (
    "max_frames",
)
_K_REACQ = (
    "max_ms",
    "max_times",
    "reset_after_ms",
    "stop_ratio"
)
_K_THROUGH = (
    "center_frames",
    "center_x",
    "center_y",
    "confirm_frames",
    "confirm_ms",
    "require_align_deg"
)
_K_SEARCH = (
    "max_ms",
    "pause_s",
    "sway",
    "sweep_max_s",
    "sweep_s"
)
_K_LOCK = (
    "match_ratio",
    "miss_frames",
    "stable_frames"
)   # 选门后锁定（见 cfg comm.gate.lock）
# 选门时 z 与框占比的配合（见 cfg comm.gate.select）：z_eff = max(z_pnp, ratio_z_scale × 框宽代理z)
_K_SELECT = (
    "k_check",
    "k_hi_ratio",
    "k_lo_ratio",
    "k_max_ratio"
)   # k 一致性检验（见 cfg comm.gate.select）
_K_PNP = (
    "max_z_jump_m",
    "refine",
    "reproj_px",
    "z_max",
    "z_min"
)
_K_GEOM = (
    "bar_width",
    "body_center_offset",
    "frame_h",
    "frame_w",
    "sym_bars"
)

_K_HDG = (
    "enable",
    "max_step_deg",
    "max_turns",
    "ok_frames",
    "post_sway",
    "post_sway_back",
    "post_sway_kpt_min",
    "post_sway_ms",
    "post_sway_settle_frames",
    "stop_hard",
    "timeout_ms",
    "tol_deg",
    "turn_period",
    "turn_scale",
    "turn_timeout_s"
)

_BOOL_KEYS = ("enable", "stop_hard")

__all__ = [
    '_K_HDG',
    '_BOOL_KEYS',
    'PH_SEARCH',
    'PH_ALIGN',
    'PH_APPROACH',
    'PH_THROUGH',
    'PH_CREEP_THROUGH',
    'SUB_HDG',
    'SUB_GOLDEN',
    'SUB_CREEP',
    'SUB_HOLD',
    'SUB_REACQUIRE',
    'SUB_SWAY_BACK',
]

