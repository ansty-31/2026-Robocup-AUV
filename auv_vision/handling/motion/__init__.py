# -*- coding: utf-8 -*-
"""handling/motion/ — 「夹取 + 放置」的运动方法簇（与 `gate/motion/` 同规）。

- `params.py`  相位常量（`PH_GRAB_*` / `PH_PLACE_*`）+ 兜底值表（`_D_*` / `_K_PLACE`）
- `actions.py` 关键动作与小件（PID 构造 / 执行原语 / 横倾放球 / 下发口 `_set_info`）
- `phases.py`  两个流程的相位处理（`GrabPhases` / `PlacePhases`）

通用运动原语仍不在本目录：**指定角度轴动作 / 下潜到位 / 定时推力** 在 `common/motion/axis.py`，
**横倾放球/倒球序列** 在 `common/motion/drop.py` —— 跨任务共用。
总调度在 `../handling_task.py`（`mode="grab"|"place"|"full"`）。
"""
