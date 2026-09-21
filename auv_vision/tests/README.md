# tests — 无硬件测试套件（按**层**分子目录）

全部都跑在无硬件环境：`conftest.py` 把工程根塞进 `sys.path`、强制 `AUV_SIM_MODE=1`（串口只打印）
与 `AUV_STREAM=0`（绝不新建推流 socket）。**秒级跑完，不需要板子/相机/水池。**

```bash
python3 -m pytest tests/ -q                     # 全部（80 例）
python3 -m pytest tests/platform -q             # 只跑平台与公共件
python3 -m pytest tests/tasks -q                # 只跑任务层
python3 -m pytest tests/tooling -q              # 只跑工具层
python3 -m pytest tests/_bite_probe.py -q        # "会咬"自检（**不被默认收集**，见下）
```

## 目录与分层

```
tests/
├── conftest.py           装置：sys.path + SIM/无推流环境 + FakeUart；
├── README.md             本文件
├── _bite_probe.py        "会咬"自检探针（名字不以 test_ 开头 → pytest **默认不收集**）
├── platform/             ① 平台与公共件（不掺任务逻辑）
│   ├── test_base.py         settings 真实取值 + **限深下限 0.55 不变量**、11B 下发帧布局/极性、
│   │                        DOF→轴字节、**真串口(pty)** 收发、0xAA55 14B 遥测解析与错位重同步、
│   │                        深度不足禁上浮/下潜照旧/超时放行、stop_hard 与急停真的停住
│   └── test_common.py       PID（死区/限幅/微分/抗 windup）、图像链路（NV12 打包色度保真、
│                            **squish 而非 letterbox**）、Det 助手与按类挑选、cfgnode（缺键兜底、
│                            字符串布尔坑、**共用 comm.motion 参数的唯一来源**）
├── tasks/                ② 任务与运动原语
│   ├── test_ball.py         撞球：SEARCH→CENTER→APPROACH→DASH→STOP→DONE(hit)；CENTER 不前进；
│   │                        APPROACH 只 sway；无目标跑满时限
│   ├── test_gate_vision.py  过门视觉：PnP 合成往返（含反投影闭环）、keypoint 解码约定
│   │                        （网格项 +索引 的 −0.5 陷阱、通道布局、DFL/框心、squish 回投、NMS）、
│   │                        mode 降级表、kpt_mem 开关优先级与平滑、mock 后端整链路
│   ├── test_gate_flow.py    过门相位机：coarse 只有未对准才后退、水平只有 sway、yaw 只在 ALIGN.HDG、
│   │                        SEARCH 是平移扫视、近距丢门判过门、loiter 门口兜底、THROUGH 按时长、
│   │                        陈旧 z 不误判；**配置守卫**（兜底==cfg、共用值只在 motion）
│   └── test_motion.py       运动原语：`TurnCore`（遥测闭环/超时/**无遥测拒转**/盲转角速率/探向）
│                            + `HeadingAligner`（测→转→停稳→再测、迭代上限、p3p 不进滤波器、
│                            新鲜度、丢门中止、一键关闭）
└── tooling/              ③ 工具层（守 `tools/` 下的脚本）
    └── test_pnp_calib.py    `tools/analyze/pnp_calib.py` 的合成往返：文件名真值解析、标尺反演
                             （故意写错门宽 20% → 必须报 a≈1.2 并建议新 `frame_w`）、t_x/t_y 尺度与符号、
                             psi≈−yaw、p3p 分类与深度偏差、CSV/报告行数
```

**分层依据**：改动影响面。`platform` 挂了全项目都得停（限深保护/串口帧）；`tasks` 挂了只影响那个任务；
`tooling` 挂了只影响离线判读，不影响下水。改完东西先跑 `tests/tasks -q` 能快速定位，再跑全量。

## `_bite_probe.py`：怎么证明测试"会咬"

一组"改配置 → 目标用例必须变红"的探针。**它不是交付套件的一部分**（文件名以 `_` 开头，
pytest 默认不收集），需要时显式跑：

```bash
python3 -m pytest tests/_bite_probe.py -q        # 8 passed = 7 条真的会咬 + 1 条反向对照
```

- 改 `comm.gate.loiter.timeout_ms` → 门口兜底用例必须红；
- 改 `comm.gate.search.sweep_s` / `comm.motion.surge_fast` / `comm.gate.through.confirm_ms`
  → 配置守卫用例必须红（这类"数值"由守卫钉，波形用例只验形状）；
- 改 `comm.depth_guard.min_depth_m` → **0.55 不变量**用例必须红；
- 改 `comm.ball.dash_ratio` → 撞球命中用例必须红；
- 改 `comm.gate.hdg.fresh_ms` → 航向新鲜度用例必须红；
- 反向对照：把 `min_depth_m` 改成 0.40 时，**限深行为用例仍应通过**（它按 cfg 相对判定，
  这是对的，不是假红）。

> 新增/改动用例后请顺手跑一次：一条"怎么改都不红"的用例等于没有测试。

## 约定

- 测试**不复制源码逻辑自证**（别把源码公式抄进断言）；
- 装置复用写进 `conftest.py` 或同层模块，跨层复用用绝对导入 `from tests.<层>.<模块> import …`；
- 断言"配置缺键/旧配置仍能跑"的地方，用代码兜底值（`gate_task._D_*`、`cfgnode.MOTION_DEFAULTS`），
  别在测试里再抄一份数字。
