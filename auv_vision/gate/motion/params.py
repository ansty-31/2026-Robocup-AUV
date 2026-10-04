# -*- coding: utf-8 -*-
"""gate/motion/params.py — 常量与**缺键兜底表**（纯数据，无逻辑）

兜底值 = 当前 `cfg/*.yaml` 的取值：配置缺键时行为不变（用例 `test_gate_defaults_match_cfg` 钉着）。
相位/子状态常量的语义与状态图见 doc/记录/过门-状态机与参数.md。"""
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
_D_ALIGN = dict(confirm_frames=4, px_x=0.18, px_y=0.22)
_D_LOITER = dict(enable=True, near_ratio=0.50, timeout_ms=6000, dx_max=0.25, dy_max=0.30,
                 # pose_hist_*：近距历史判断（coarse/width 在门口、下视看不到、主视也没信息时，
                 #   用本轮最近采信位姿的中位数判"已到门口"）。只取本轮，换门清零。
                 pose_hist_ms=1500.0, pose_hist_min=3)
_D_Z = dict(cross=1.2, cross_confirm_frames=4, slow_max=1.2,
            dist_max_m=2.5,   # z 超过它 ⇒ 完全不相信该帧位姿（跳变保护第一条，无条件）
            relock_z_jump_m=0.5, relock_far_ratio=0.7,
            # relock_away_m：丢门重锁时，z 超过它 ⇒ 判为「下一个门」——当前门丢了/测不到距时，
            #   不能被远处能测距的门骗过去误当当前门。只对"有丢门参照"生效，首见不拦。
            relock_away_m=1.8,
            near_lost_ratio=0.50)
_D_SURGE = dict(creep=0.20, lost_backward=0.20, reacquire=0.25, through=0.6)
_D_CREEP_THROUGH = dict(creep_ms=5000)   # creep_through：门口过门，慢速 creep 冲门时长(ms)
_D_COARSE = dict(far_ratio=0.18, near_ratio=0.55, align_x=0.33, align_y=0.29)
_D_WIDTH = dict(z_max=1.5)
_D_TASK = dict(timeout_ms=45000, pass_target=1, pose_hold_frames=10)
_D_KPT = dict(conf_thr=0.8)
_D_HOLD = dict(max_frames=20)
_D_REACQ = dict(max_ms=800, max_times=2, stop_ratio=0.75, reset_after_ms=3000)
_D_THROUGH = dict(confirm_frames=8, confirm_ms=3000,
                  require_align_deg=8.0)
_D_SEARCH = dict(sweep_s=2.0, sweep_max_s=32.0, pause_s=1.0, sway=0.6,
                 max_ms=2000)
_D_LOCK = dict(match_ratio=0.20, miss_frames=5, stable_frames=5)   # 选门后锁定（见 cfg comm.gate.lock）
# 选门时 z 与框占比的配合（见 cfg comm.gate.select）：z_eff = max(z_pnp, ratio_z_scale × 框宽代理z)
_D_SELECT = dict(k_lo_ratio=0.75, k_hi_ratio=1.40)   # k 一致性检验（见 cfg comm.gate.select）
_D_PNP = dict(reproj_px=20.0, z_min=0.2, z_max=15.0, refine=True,
              max_z_jump_m=0.8)
_D_GEOM = dict(frame_w=GATE_FRAME_W, frame_h=GATE_FRAME_H,
               body_center_offset=0.0)

_D_HDG = dict(enable=True, tol_deg=8.0,        # enable=关掉后完全不进正航向；tol_deg=够正判据(°)
              # ok_frames：**连续**多少帧 ψ 都在 tol_deg 内才锁存"航向 OK"（防单帧噪声钉死）
              ok_frames=5,
              # max_step_deg：**单次转角上限**（0 = 不设限）。先按 turn_scale 缩小测到的 ψ 再用它钳位，
              #   被钳掉的残余**由下一小步补转**（逐小步逼近，无次数上限）⇒ 它是"每步别太猛"
              #   兼"防垃圾 ψ 把船甩出去"的安全钳位，不再等于"只转一次"。
              max_step_deg=15.0,
              # turn_scale：下发角再乘的系数（只改"下发的 deg"，不改 ψ 的测量与判据）。
              #   实测偏大，先按 0.8 缩一档试；1.0 = 不缩。
              turn_scale=0.8,
              timeout_ms=30000.0, turn_timeout_s=8.0,
              # turn_period：等"下位机完成反馈"时的轮询节拍（秒）= 上位机自己的内层循环 pace。
              #   原属 `motion.turn_pid.period`；那套"上位机 PID 驱动转向"的参数 2026-10-02 已整块删除。
              turn_period=0.05,
              # post_sway(_ms)：转完门被甩出视野时（转 ≤15° 就接近 32° 半视场），在**丢门的帧**上朝
              #   **转向的反方向**平移一小段把门拉回视野；窗口 = post_sway_ms。幅度必须 > 执行器死区 0.138。
              post_sway_ms=600.0, post_sway=0.20,
              # post_sway_back：平移的同时**缓慢后退**的幅度（正值，代码取负；0 = 不后退）。
              # post_sway_kpt_min：**唯一**退出门限 —— **当前门**出现 ≥ 这么多 conf ≥
              #   `vision.gate.keypoint.conf_thr` 的角点就交回视觉（0 = 关掉这条，只靠超时兜底）。
              post_sway_back=0.15, post_sway_kpt_min=2,
              # max_turns：**本门转向上限**（0/负 = 不限）。到顶就不再起转、带残余航向进近 ——
              #   防"转向无效时自转"（2026-09-27 板的疯狂旋转 bug，见 tools/check/boat/check_hdg_lockup.py）。
              max_turns=5,
              # stop_hard：转向收尾**硬停**（连发中性帧走完 ramp + 遥测 yaw 验证）。必须 true ——
              #   下位机没有"无帧超时停车"，只发一帧中性时 yaw 轴还在 ramp 半路。
              stop_hard=True)

_BOOL_KEYS = ("enable", "stop_hard")

__all__ = [
    '_D_HDG',
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
    '_D_ALIGN',
    '_D_LOITER',
    '_D_Z',
    '_D_SURGE',
    '_D_CREEP_THROUGH',
    '_D_COARSE',
    '_D_WIDTH',
    '_D_TASK',
    '_D_KPT',
    '_D_HOLD',
    '_D_REACQ',
    '_D_THROUGH',
    '_D_SEARCH',
    '_D_PNP',
    '_D_GEOM',
]
