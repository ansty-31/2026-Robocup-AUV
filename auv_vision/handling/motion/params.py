# -*- coding: utf-8 -*-
"""handling/motion/params.py — 两个流程共用的**相位常量与兜底值表**。
PH_GRAB_* = 夹取流程相位（原 `grab/grab_task.py`）；PH_PLACE_* = 放置流程相位（原 `place/place_task.py`）。
两套原都叫 PH_INIT/PH_DONE，合并后必须前缀化（否则同一实例里分不清）。
兜底值 = 当前 `cfg/*.yaml` 的取值；缺键时行为不变。"""
from __future__ import annotations

from common.cfg.cfgnode import merge, req, req_flag, req_node, MissingCfg
from common.motion.axis import task_node




PH_GRAB_INIT = "INIT"
# Deprecated compatibility names: no grab transition enters these phases.
PH_GRAB_PITCH_UP = "PITCH_UP"
PH_GRAB_SEARCH = "SEARCH"
PH_GRAB_CENTER = "CENTER"
PH_GRAB_APPROACH = "APPROACH"
PH_GRAB_LEVEL = "LEVEL"
PH_GRAB_ALIGN = "ALIGN"
PH_GRAB_DIP = "DIP"
PH_GRAB_RISE = "RISE"
PH_GRAB_VERIFY = "VERIFY"
PH_GRAB_DUMP = "DUMP"
PH_GRAB_EXIT = "EXIT"          # 收尾前把机头放平（别把 30° 俯仰留给下一个任务）
PH_GRAB_DONE = "DONE"
_K_CENTER = (
    "eps",
    "confirm_frames",
    "pid",
    "ki",
    "kd",
    "out_max",
    "deadzone"
)
_K_APPROACH = (
    "surge_fast",
    "surge_slow",
    "slow_ratio",
    "growth_eps",
    "dip_ratio",
    "confirm_frames",
    "sway",
    "ki",
    "kd",
    "out_max",
    "deadzone"
)
_K_ALIGN = (
    "eps_x",
    "eps_y",
    "confirm_frames",
    "sway",
    "ki",
    "kd",
    "out_max",
    "deadzone",
    "surge"
)
_K_VERIFY = (
    "enable",
    "roi",
    "red_percent_min",
    "confirm_frames"
)
_K_RETRY = (
    "assume_ok",
    "assume_ok_after",
    "max_dumps"
)
_K_LOST = (
    "grace_frames",
    "hold_s",
    "ema_alpha"
)
PH_PLACE_INIT = "INIT"
PH_PLACE_TRANSPORT = "TRANSPORT"
PH_PLACE_RELEASE = "RELEASE"
PH_PLACE_STOP = "STOP"
PH_PLACE_DONE = "DONE"
_K_PLACE = (
    "calibrated",
    "timeout_ms",
    "depth_floor_m",
    "transport_s"
)


def place_cfg():
    """`comm.place` + 代码兜底（缺段/缺键都不崩）。"""
    return task_node("place")     # ★ 无兜底：调用方自己 req()
