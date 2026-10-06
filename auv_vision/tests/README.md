# tests — 无硬件测试套件（按**层**分子目录）

全部都跑在无硬件环境：`conftest.py` 把工程根塞进 `sys.path`、强制 `AUV_SIM_MODE=1`（串口只打印）
与 `AUV_STREAM=0`（绝不新建推流 socket）。**秒级跑完，不需要板子/相机/水池。**

```bash
python3 -m pytest tests/ -q                     # 全部（**370 例**，2026-10-07）
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
│   ├── test_common.py       PID（死区/限幅/微分/抗 windup、**执行器死区补偿 bias**）、图像链路（NV12 打包色度保真、
│   │                        **squish 而非 letterbox**、**D 链序唯一**）、Det 助手与按类挑选、
│   │                        cfgnode（缺键兜底、字符串布尔坑、**共用 comm.motion 参数的唯一来源**）
│   ├── test_paths.py        路径落点（部署脚本默认目录、cfg 绝对路径、标定 yaml 文件头）
│   └── test_hud.py          叠加层（画什么/不画什么）
├── tasks/                ② 任务与运动原语（含子目录 gate/ grab/ place/）
│   ├── test_ball.py         撞球：SEARCH→CENTER→APPROACH→DASH→STOP→DONE(hit)；CENTER 不前进；
│   │                        APPROACH 只 sway；无目标跑满时限
│   ├── gate/                过门（按主题成组）
│   │   ├── test_gate_vision.py   PnP 合成往返（含反投影闭环）、keypoint 解码约定
│   │   │                         （网格项 +索引 的 −0.5 陷阱、通道布局、DFL/框心、squish 回投、NMS）、
│   │   │                         mode 降级表、kpt_mem 开关优先级与平滑、mock 后端整链路
│   │   ├── test_gate_flow.py     相位机仲裁：coarse 只有未对准才后退、水平只有 sway、
│   │   │                         yaw 只在 ALIGN.HDG、SEARCH 是平移扫视、**丢门回 SEARCH（不再直冲）**、
│   │   │                         loiter 门口兜底、THROUGH 按时长、陈旧 z 不误判、
│   │   │                         正航向 + SWAY_BACK 全套（自包含转向/硬停/让位/角点门限/转向上限）、
│   │   │                         **配置守卫**（兜底==cfg、共用值只在 motion）
│   │   └── test_gate_postproc.py 后处理四条（L/R 归一/几何合法/去重/选门键）+ 兜底==cfg
│   └── test_motion.py       运动原语：`TurnCore`（遥测闭环/超时/**无遥测拒转**/满舵无进展自停/
│                            方向自证/超转容差；探向与盲转已删）+ `HeadingAligner`（冻结 ψ 转一步、
│                            `turn_scale`+`max_step_deg`、转向上限、p3p 不进滤波器、一键关闭）
└── tooling/              ③ 工具层（守 `tools/` 下的脚本）
    ├── test_pnp_calib.py    `tools/analyze/calib/pnp_calib.py` 的合成往返：文件名真值解析、标尺反演
    │                        （故意写错门宽 20% → 必须报 a≈1.2 并建议新 `frame_w`）、t_x/t_y 尺度与符号、
    │                        psi≈−yaw、p3p 分类与深度偏差、CSV/报告行数
    ├── test_ruler_calib.py  卷尺靶子：跨度/斜视诊断 + (fx,c) 合成往返 + 混合轴向
    ├── test_label_corners.py 手工标注：吸附/坐标域/schema/端到端喂通 pnp_calib
    ├── test_tape_ticks.py   刻度周期法：亚像素/透视梯度/谐波/交叉校验门
    └── test_cfg_yaml_head.py yaml 文件头（cfg 注释瘦身不破坏解析）
```

**分层依据**：改动影响面。`platform` 挂了全项目都得停（限深保护/串口帧）；`tasks` 挂了只影响那个任务；
`tooling` 挂了只影响离线判读，不影响下水。改完东西先跑 `tests/tasks -q` 能快速定位，再跑全量。

## `_bite_probe.py`：怎么证明测试"会咬"

一组"改配置 → 目标用例必须变红"的探针。**它不是交付套件的一部分**（文件名以 `_` 开头，
pytest 默认不收集），需要时显式跑：

```bash
python3 -m pytest tests/_bite_probe.py -q        # 10 passed = 9 条真的会咬 + 1 条反向对照
```

- 改 `comm.gate.loiter.timeout_ms` → 门口兜底用例必须红；
- 改 `comm.gate.search.sweep_s` / `comm.motion.surge_fast` / `comm.gate.through.confirm_ms`
  → 配置守卫用例必须红（这类"数值"由守卫钉，波形用例只验形状）；
- 改 `comm.depth_guard.min_depth_m` → **0.55 不变量**用例必须红；
- 改 `comm.ball.dash_ratio` → 撞球命中用例必须红；
- 改 `vision.gate.keypoint.conf_thr` → "正航向只吃可信 full 帧"用例必须红；
- 改 `comm.gate.hdg.max_turns` / `comm.gate.hdg.post_sway_kpt_min`
  → 转向上限 / SWAY_BACK 退出角点门限用例必须红；
- 反向对照：把 `min_depth_m` 改成 0.40 时，**限深行为用例仍应通过**（它按 cfg 相对判定，
  这是对的，不是假红）。

> 新增/改动用例后请顺手跑一次：一条"怎么改都不红"的用例等于没有测试。

## 2026-10-07 新增（阈值/机制重设的守卫）

| 用例 | 钉什么 |
|---|---|
| `test_gate_flow.py::test_through_requires_the_mode_own_centering_range` | 过门居中闸必须是**本档位自己的对中带**（位姿档/width/coarse 各一套），且与 `through.center_x/center_y` 取 AND |
| `test_gate_flow.py::test_through_is_two_stage_fast_then_slow` | 两段式冲刺：段① `fast_ms` 用 `surge.through`、段② 降到 `surge.creep`；`fast_ms=0` 整段满速（回退路径） |
| `test_gate_flow.py::test_gate_sway_is_proportional_not_bang_bang` | 按 cfg 反算饱和起点 `(out_max−bias)/kp`；非零输出**全部 ≥ 执行器死区 0.138**（不许"白发"指令） |
| `test_common.py::test_pid_bias_lifts_output_above_actuator_deadzone` | PID `bias`（执行器死区补偿）：死区内 0、出死区即抬到死区值、线性升、负误差对称、`bias=0` 回归 |

> 同轮把若干**夹具**按新基准对齐（不是放宽断言）：`_aligned_px()` 的"已对准"落点含纵向零点
> `align.dy_target`；`_HullWorld` 的 `z` 必须**明显大于** `z.cross`（原来写 1.5 = 新的 `cross`，
> 靠浮点尾数侥幸过关）；`gate/percept/mock.py` 的默认位姿把门画在零点上而不是光轴上。

## 约定

- 测试**不复制源码逻辑自证**（别把源码公式抄进断言）；
- 装置复用写进 `conftest.py` 或同层模块，跨层复用用绝对导入 `from tests.<层>.<模块> import …`；
- 断言"配置缺键/旧配置仍能跑"的地方，用代码兜底值（`gate/motion/params.py::_D_*`、`cfgnode.MOTION_DEFAULTS`），
  别在测试里再抄一份数字。
