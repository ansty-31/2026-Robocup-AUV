# -*- coding: utf-8 -*-
"""handling/motion/params.py — 两个流程共用的**相位常量与兜底值表**。
PH_GRAB_* = 夹取流程相位（原 `grab/grab_task.py`）；PH_PLACE_* = 放置流程相位（原 `place/place_task.py`）。
两套原都叫 PH_INIT/PH_DONE，合并后必须前缀化（否则同一实例里分不清）。
兜底值 = 当前 `cfg/*.yaml` 的取值；缺键时行为不变。"""
from __future__ import annotations

from common.cfg.cfgnode import merge
from common.motion.axis import task_node




PH_GRAB_INIT = "INIT"
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
_D_CENTER = dict(eps=0.20, confirm_frames=5,
                 pid=dict(kp=0.25, ki=0.0, kd=0.0, out_max=0.15, deadzone=0.0))
_D_APPROACH = dict(surge_fast=0.25, surge_slow=0.12, slow_ratio=0.12, growth_eps=0.002,
                   dip_ratio=0.15, confirm_frames=2,
                   sway=dict(kp=8.0, ki=0.0, kd=0.0, out_max=0.45, deadzone=0.05))
_D_ALIGN = dict(eps_x=0.15, eps_y=0.20, confirm_frames=4,
                sway=dict(kp=6.0, ki=0.0, kd=0.0, out_max=0.20, deadzone=0.03),
                surge=dict(kp=4.0, ki=0.0, kd=0.0, out_max=0.12, deadzone=0.03))
_D_VERIFY = dict(enable=True, roi=[0.25, 0.35, 0.5, 0.5], red_percent_min=0.30,
                 confirm_frames=3)
_D_RETRY = dict(assume_ok=True, assume_ok_after=2, max_dumps=1)
_D_LOST = dict(grace_frames=15, hold_s=0.8, ema_alpha=0.3)
PH_PLACE_INIT = "INIT"
PH_PLACE_TRANSPORT = "TRANSPORT"
PH_PLACE_RELEASE = "RELEASE"
PH_PLACE_STOP = "STOP"
PH_PLACE_DONE = "DONE"
_D_PLACE = dict(
    calibrated=False,
    timeout_ms=60000.0,
    depth_floor_m=1.0,
    transport_s=30.0,          # 占位·未标定：运输时长（真正的"到点"判据待定）
    surge=0.25,                # 占位·未标定：运输速度
    roll_target_deg=0.0,       # ⚠️ 未验证：IMU roll 目标（0 = 机身水平）
    roll_tol_deg=5.0,          # 允许的横倾偏差；超了就重新回正
    relevel_min_s=2.0,         # 两次回正之间至少隔这么久（别把转角指令刷爆）
    stop_s=3.0,                # 放球后停动力（用户定：3s）
)


def place_cfg():
    """`comm.place` + 代码兜底（缺段/缺键都不崩）。"""
    return merge(task_node("place"), _D_PLACE)
