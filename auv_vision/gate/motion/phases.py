# -*- coding: utf-8 -*-
"""gate/motion/phases.py — 过门相位与子状态的字符串常量（GateTask 相位机）

单独一个文件是为了让"相位段实现"（phase_degrade.py / phase_recover.py）与
骨架（gate_task.py）用的是**同一份**常量，而不是各写各的字符串字面量。
`gate/gate_task.py` 会把这些名字原样再导出（`from gate.motion.phases import ...`），
所以 `from gate.gate_task import PH_APPROACH` 这类旧引用继续有效。
"""

PH_SEARCH = "SEARCH"
PH_ALIGN = "ALIGN"
PH_APPROACH = "APPROACH"
PH_THROUGH = "THROUGH"

SUB_GOLDEN = "GOLDEN"
SUB_CREEP = "CREEP"
SUB_HOLD = "HOLD"
SUB_REACQUIRE = "REACQUIRE"
SUB_CUE = "CUE"           # 恢复线索驱动中（v1.3）
