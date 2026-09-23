# AUV 视觉导航项目（RDK X5 / Ubuntu 22.04 / Python）

> **当前状态（20260920）**
> - **任务三 过门（gate）= 扁平 v1.2 版**（`gate/*.py`，**9 个文件**；来自 `bak/gate_before_enhance_20260915_182038/`
>   的 v1.2 原版 + 2026-09-18 新增 `heading_align.py`），这是**定版方案**，不再有分层结构。
> - `kpt_memory`（门角点逐点软融合）是**可选开关、2026-09-18 起默认关闭**：
>   `cfg/vision.yaml → vision.gate.kpt_mem.enable: false`（板端曾因拼写错 `flase` 被
>   `bool('flase')=True` 而**实际一直开着**）。开启用 `enable: true` 或 `AUV_GATE_KPT_MEM=1`
>   （优先级 `AUV_GATE_KPT_MEM` > `enable` > 兜底 `false`；helper API 在
>   `gate/kpt_memory.py::kpt_mem_enabled/enable_source/build_kpt_memory`）。
> - **任务一 小球（ball）= `task1_2/ball.py`**（已定）。
> - **过门的两条横向/转向约定（2026-09-20 用户定）**：
>   **① 居中只有 sway 平移**（`align_yaw.{enable,in_px,in_pose}` 与其 PID **已整体删除**）；
>   **② SEARCH 是左右平移扫视，不旋转**（原旋转脉冲已删除，`gate.search.{sweep_s,pause_s,sway}`）。
>   gate 里**唯一的 yaw 来源**是 ALIGN.HDG 离散正航向（`gate/heading_align.py` +
>   `common/turn_deg.py`，增益 `comm.motion.turn_pid`）。
> - **限深保护（默认开启）**：下位机回传 14B 遥测帧（`0xAA55`+深度/姿态+校验和，
>   `base/telemetry.py`）→ 深度 ≤ `comm.depth_guard.min_depth_m`（**现场定死 0.55 m，不准改**）时
>   **禁止上浮**（只压 heave，surge/sway/yaw 不动），机身顶不出水面。详见下文「限深保护」一节。
> - **返回出发区（记忆返回）已整体移除**：`return_by_memory` 代码、交接协议与相关参数都不再存在，
>   当前撞球收尾是 `task1_2/run_ball_reverse.sh` 的**定时直线倒车**（开环，不做轨迹回放）。
> - 以下功能**已确定不需要、已删除**，本项目不再提供：v1.3/v1.4 分层 gate（`vision/`+`data/`+`motion/`）、
>   质量分 Q、线索 cues、帧守卫（`common/frame_stamp.py`）、`common/streak.py`、DOF ramp
>   （`common/ramp.py`）、`comm.handover`、`comm.back`、`STATE_BACK`；过门侧另有
>   **`comm.gate.align_yaw`（居中用 yaw 的通道）**、**SEARCH 的旋转脉冲**（`search.spin_s/yaw`）、
>   **width/coarse 的「对准就直冲」出口**（`*.dash*`）、以及 `comm.gate.align_yaw.heading`
>   连续航向通道 —— 恢复任何一项都要先说明理由并让用户拍板。下文只描述**当前实际存在**的
>   代码/工具/配置。
> - ⚠️ **跨工作区只读约束**：训练工程 `../RDKX5-YOLOv11n-/` 只有**测试脚本**可以与它共享/写入，
>   该工作区下的**其他文件一律只读**——不改它的训练/导出/量化脚本与配置（那是训练侧流水线，
>   改动会静默影响 `.bin`，见 `doc/算法说明-gate-PnP移植方案.md` §6 与训练工程自身 README）。

RoboCup AUV 赛事视觉代码。平台：**RDK X5（3.5.0）**，前视 USB + 下视 IMX415(MIPI)（下视保留，用于录素材/未来任务）。
识别：YOLO 蓝/红球 + gate（**keypoint 四角 + PnP** 新前端）；串口 11B 帧向 STM32 下发 DOF，
STM32 回传 14B 遥测帧（深度/姿态 → 限深保护）。

> 先读：`README.md`（用法）→ `doc/算法说明.md`（总体）→ `gate/过门-状态机与参数.md`（**状态/运动/参数/逻辑树速查；§9 是现场实测记录汇总**）→ `doc/算法说明-gate-PnP移植方案.md`（gate 设计与移植）→ `doc/算法说明-gate-角点逐点融合滤波.md`（角点逐点数据处理）→ `doc/实验待测-runbook.md`（**下水前后的实测阶梯与判据**）。
> 参数怎么改：`cfg/*.yaml` 注释只写"是什么 + 当前值 + 调它的后果"；**数值是怎么来的、现场发生过什么，都在 `gate/过门-状态机与参数.md` §9**。

## 目录分区（英文分区命名）

```
auv_vision/
├── main.py              # 装配/状态机入口（ball/gate；由 task1_2/run_ball_reverse.sh 或直接调用）
├── preview_detect.py    # 下水前视觉自检（带识别框；--gate-kpt 看门框 + 4 角点）
├── manual.sh            # 手动模式唯一入口：遥控桥 + 录像 + 推流
├── manual/              # 手动模式三件套：udp_server.py(遥控桥) · recorder.py(录像) · stream.py(推流/接收库)
├── base/                # 基础部件：settings.py(配置) · camera.py(前视/下视/mipi)
│                        #           uart.py(11B 下发帧) · telemetry.py(0xAA55 遥测上行：深度)
├── common/              # 通用功能：PID.py · preprocess.py(图像链路) · detector.py(检测)
│                        #           cfgnode.py(cfg 节点读取 + **ball/gate 共用参数的唯一入口**)
│                        #           turn_deg.py(按角度原地转：遥测 yaw 闭环 + 硬停；gate/脚本共用)
├── task1_2/             # 任务一（撞球）：
│   ├── ball.py          #   任务一 撞球 BallTask
│   ├── run_ball_reverse.sh  #   编排（8 步）：待机→(可选)下潜→(可选)前进→撞球→倒车→
│   │                        #     前进 2s→**左转 90°（遥测闭环）**→过门任务（可 AUV_GATE_AFTER=0 截断）
│   └── run_gate.sh      #   只跑过门（下水专测 gate 用；参数见脚本头部注释）
├── gate/                # 任务三 过门（**扁平 v1.2**：keypoint 四角 + PnP）
│   ├── gate_task.py     #   相位机骨架（唯一入口）·  __init__.py（门面）
│   ├── gate_detector.py #   board_camera() / build_gate_backend()（组合根）
│   ├── gate_decode.py   #   keypoint 头解码 decode_yolo11_kpt
│   ├── gate_frontend.py #   keypoints_to_img_pts / mode 判定（前端适配层）
│   ├── geometry.py      #   CameraModel / object_points / gate_pose / 平面·反投影
│   ├── kpt_memory.py    #   角点逐点软融合（**可选，2026-09-18 起默认关闭**；开启用 enable=true / AUV_GATE_KPT_MEM=1）
│   ├── heading_align.py #   ALIGN.HDG 离散正航向（测→转→停稳→再测；用 common/turn_deg.py 转）
│   ├── mock.py          #   MockGateBackend 仿真后端
│   └── 过门-状态机与参数.md  #   **逐状态导读**：作用/运动/参数调整/完整逻辑树（改 gate 前先看）
│                        #   细则/参数/待测项见 doc/算法说明-gate-PnP移植方案.md（§5）
│                        #   直冲出口只有三条（z.cross / 近距丢门 / 门口超时兜底 loiter.*）
├── bak/                 # 归档（gitignored，不上板）：gate v1.2 原版树 + 两份 v1.2 cfg 快照（见 bak/README.md）
├── tools/               # 工具（**按用途分三类**；除注明外都是本机驱动、不传板端，见 tools/README.md）
│   ├── deploy/          #   部署与一致性：deploy_to_board.sh(烧录) · check_board_parity.sh + board_parity.md5(清单对齐)
│   │                    #   archive_baks.sh(归档 *.bak*，**会随部署上板**) · tidy_board_bak.sh(板端 bak/ 整理)
│   ├── check/           #   自检（上板前/离线跑一次看结论）：check_kpt_decode.py(解码约定)
│   │                    #   check_gate_pose.py(位姿为何被拒) · check_pipeline_identity.py(训练/推理同域)
│   └── analyze/         #   离线判读与标定（吃日志/dump 出结论）：analyze_task_log.py(逐帧日志①~⑧)
│                        #   feature_coverage.py(哪些功能没用到) · analyze_kpt_dump.py(角点可得率)
│                        #   analyze_heading.py(PnP 安装角/航向噪声) · analyze_pnp_center.py(位姿中心 vs bbox)
│                        #   **pnp_calib.py（PnP 位姿/深度标定：真值 dump → 误差表 + 尺寸反演 + 建议 cfg）**
│                        #   **label_corners.py（手工标 4 角 → 同格式 dump；检测器不响应时的岸上路线；
│                        #                      --measure = 卷尺刻度靶子，点 3 个刻度出 Δu/斜视）**
│                        #   **ruler_calib.py（靶子读数 → 等效焦距 fx/fy 与距离口径 c；--gate 与门框联立）**
│                        #   **tape_ticks.py（刻度周期法：不点鼠标，用卷尺自带 1cm 刻度测像素比例）**
├── cfg/                 # vision.yaml · comm.yaml · front_camera.yaml（参数唯一来源）
│                        #   comm.yaml 的 **motion:** 段 = ball/gate **共用**的底层运动参数
│                        #   （两套 PID / surge_fast·surge_slow / loss_inertia_surge / turn_pid），
│                        #   任务段只放各自特有的旋钮；共用值不在两处各写一份
├── doc/                 # 算法说明.md · 算法说明-gate-PnP移植方案.md（移植总体方案）
│                        # 算法说明-gate-角点逐点融合滤波.md（kpt_memory 的数据处理）
│                        # 实验待测-runbook.md（**PnP 位姿/深度标定实验**：步骤/判据/记录表/报告模板）
│                        # **待研究-缺角位姿先验（三维信息复用）.md**（缺角时用存下来的三维补信息，**未实现**）
│                        # 前视USB相机低延迟推流方案.md（推流/手动模式）
├── models/ · tests/     # 权重(.bin) · 无硬件测试套件（**按层分子目录**，见 tests/README.md）
│                        #   tests/：138 例 / 10 个文件 + conftest
│                        #     platform/  test_base   平台：settings/11B 帧/遥测/限深保护/硬停
│                        #                test_common 公共件：PID/图像链路(NV12·squish)/Det/cfgnode
│                        #     tasks/     test_ball        撞球相位机
│                        #                test_gate_vision 解码约定/PnP 往返/角点记忆/mode 降级
│                        #                test_gate_flow   档位仲裁/出口兜底/正航向接入/配置守卫
│                        #                test_motion      额定转角 + 离散正航向
│                        #     tooling/   test_pnp_calib   **PnP/深度标定工具**的合成往返（反演标尺/符号/CSV/报告）
│                        #                test_label_corners 手工标注：吸附/坐标域/schema/端到端喂通 pnp_calib
│                        #                test_ruler_calib   卷尺靶子：跨度/斜视诊断 + (fx,c) 合成往返 + 混合轴向
│                        #                test_tape_ticks    刻度周期法：亚像素/透视梯度/谐波/交叉校验门
│                        #   （`_bite_probe.py` 是"会咬"自检探针，**不被默认收集**，
│                        #     需要时 `python3 -m pytest tests/_bite_probe.py -q`）
│                        #   分层块：`pytest tests/platform -q` / `tests/tasks` / `tests/tooling`
```

分区语义：`base`（平台基础设施）/ `common`（跨任务公用）/ `task1_2`（任务一）/ `gate`（任务三）。
依赖规则：任务代码只 import `base`/`common` 与同级任务模块；`main.py` 是唯一装配点。

## 快速开始（本机，无硬件）

```bash
python3 -m pytest tests/ -q              # 无硬件测试（89 例：base/common/ball/gate/标定与标注工具）
python3 main.py --task ball              # 只跑撞球（SIM/mock）
python3 main.py --task gate              # 试跑过门（cfg model.mode: mock）
python3 preview_detect.py --gate-kpt     # 下水前：门框 + 4 角点 + 置信度（船不动）
python3 tools/analyze/analyze_task_log.py log/*.jsonl   # 下水后：逐帧日志判读（相位/出口/通道能动力）
python3 tools/analyze/feature_coverage.py log/*.jsonl   # 下水后：这轮**哪些功能没被用到**
python3 tools/analyze/pnp_calib.py --report r.md log/pnp_*/pnp_z*.jsonl  # PnP/深度标定（文件名即真值）
```
> `tests/` 与训练工程 `../RDKX5-YOLOv11n-/` **可以共享测试脚本**（唯一允许写入对方工作区的东西）；
> 训练工程下的**其他文件只读**，不要在本工程的任务里顺手改它们。

### 真机编排（.sh，见 task1_2/）
```bash
cd task1_2 && ./run_ball_reverse.sh      # 待机→(可选)下潜→(可选)前进→撞球→倒车→前进2s→左转90°→过门
AUV_GATE_AFTER=0 ./run_ball_reverse.sh   # 只做前 7 步（分段试，不接过门）
AUV_SIM_MODE=1 ./run_ball_reverse.sh     # 台架：只打印不发串口（**改脚本后必跑一次干跑**）
./run_gate.sh                            # 只跑过门任务
```
- **撞球后返回**：`run_ball_reverse.sh` 在 `main.py --task ball` 结束后**定时直线后退**
  （开环、不依赖视觉；`AUV_REV_S`/`AUV_REV_SURGE` 调时长与速度）；
- **倒车后的三步**（2026-09-19 加）：前进 `AUV_POST_FWD_S`(2s) → **按角度左转**
  `AUV_TURN_DEG`(90°，`common/turn_deg.py` 用下位机遥测 yaw **闭环**执行，转完必硬停)
  → `main.py --task gate`。转角可用 `AUV_TURN_LEFT/AUV_TURN_KP/AUV_TURN_BLIND` 覆盖；
  **没有遥测时 turn_deg 直接拒转（退出码 5）**，要开环盲转必须显式 `AUV_TURN_BLIND=1`；
- 单独试转向：`python3 common/turn_deg.py --deg 90 --dir left`（`--help` 看全部旋钮）。
- 下视相机 `cfg/vision.yaml camera.down` 与 `base/camera.py`（sim/mipi）**保留**，仅策略不再使用。

### 只用 gate 权重跑过门（单独下水测 gate）
```bash
python3 main.py --task gate                  # 只装配 gate 后端 → hub.detect_list("gate")
python3 preview_detect.py --gate-kpt         # 下水前：门框 + 4 角点 + 置信度（船不动）
```
- `main.py --task gate` **只加载 `model.task_models.gate.path`**（即 `gate_kpt_*.bin`），
  单权重 ball 模型惰性加载、本次完全不碰；启动横幅会打印实际权重文件名；
- gate 后端不可用（权重缺失/路径错）时**直接拒绝启动（退出码 3）**，不会入水后空跑；
- 窗口叠加 = 当前任务本帧检测（gate 为门框 + 4 角点编号），不再额外跑一遍模型。

## 任务算法速览

- **任务一 撞球**（task1_2/ball.py）：**SEARCH→CENTER→APPROACH→DASH→STOP** 运动链
  （居中=仅 yaw+heave；接近=分级前进+仅 sway；面积(EMA)≥`dash_ratio` → 以分级最高速 `surge_fast`
  冲刺 `dash_dur_s`；不检测是否撞到；冲刺后 STOP 全 0 保持 `stop_hold_s` → DONE(hit)，
  总时限 `comm.ball.timeout_ms`）；
- **撞球之后**（`task1_2/run_ball_reverse.sh`）：**定时直线倒车**（开环、不依赖视觉）→ 前进 2s
  → **按角度左转 90°**（`common/turn_deg.py`，遥测 yaw 闭环 + 转完硬停）→ 过门任务；
- **任务三 过门**（`gate/gate_task.py` 骨架；**扁平 v1.2，当前定版**，细则/参数见 `doc/算法说明-gate-PnP移植方案.md`）：
  keypoint 四角 → IPPE 6-DoF，深度 `Z=tvec.z`；
  RANGE_ALIGN 子状态 GOLDEN/**HDG(离散正航向)**/CREEP/HOLD/REACQUIRE → APPROACH → THROUGH（机身过门判据）。
  门 = 闭合矩形框(红 PVC，**实测外缘 0.77×0.56 m**，管外径 5cm，悬空，对称无朝向要求)。
  **直冲出口三条**：① `Z≤z.cross` 连续确认帧（实测有效，勿动）；② 近距(`Z≤z.near_lost_m` 且 z 新鲜
  **或** 框占比 ≥ `z.near_lost_ratio`) 丢失判过门（ALIGN 与 APPROACH 都生效）；③ 在门口超时兜底
  `loiter.*`（门口 + 安全带内停留超时 → 自己拍板直冲）。
  ⚠️ 原「③ width/④ coarse 对准就直冲」**已整体删除**（2026-09-18 用户定）。
  **SEARCH 是左右平移扫视**（`gate.search.{sweep_s,pause_s,sway}`，对称来回）——2026-09-20 用户定
  「过门过程中不允许旋转搜索」；**居中只有 sway**（`align_yaw` 已随其 PID 一起删除），
  gate 里唯一的 yaw 来源是 ALIGN.HDG 正航向（增益 `comm.motion.turn_pid`）。
  coarse 仲裁原则：**只有未对准才 HOLD/后退**。
  角点逐点融合 `gate/kpt_memory.py` 为**可选功能、2026-09-18 起默认关闭**（板端曾因拼写错 `flase` 实际开着）：
  开启用 `vision.gate.kpt_mem.enable: true` 或 `AUV_GATE_KPT_MEM=1`；关闭时角点单帧直用。

## 参数（cfg/*.yaml）
`vision.yaml`：camera(front/down)、`image.*`（**ball/gate 共用图像链路，勿改**）、`model.*`
（含 `task_models.gate`）、`gate.*`（几何/keypoint/PnP/kpt_mem 开关）、`stream.*`；
`comm.yaml`：serial/dof_map/心跳/急停/ramp/**depth_guard(限深保护)**/**motion（ball/gate 共用的
底层运动：两套 PID + 速度档 + turn_pid）**/tasks.enabled（默认 `[ball]`；gate 需后端或 mock）/ball/gate 任务参数。
> **共用参数只写一份**：`comm.motion` 里的值同时被撞球与过门读取（读法见 `common/cfgnode.py`
> 的 `motion_num/motion_node/motion_pid`）；确需某个任务单独一套时，才在该任务段写覆盖键
> （如 `comm.gate.pid_sway`）。`gate.surge.fast/slow` 与 `ball` 段**不再存在**这些重复键。
模型：X5 OE `.bin`（march `bayes-e`）；`.hbm` 不兼容。

## 运行监测
串口帧打印 + 画面叠加（含 gate `phase/substate/mode/z/dx/dy/sway/heave/surge/yaw/hdg/pass`、
下位机 `depth=… guard=… min=…m`）—— cfg debug 开关；`AUV_SHOW=0` 关窗口。
> 画面里 **`yaw` 恒 0 是正常的**（居中只用 sway）；只有正航向 ALIGN.HDG 在转的时候才会非 0，
> 同时 `hdg`（航向误差°）与 `hdg_i`（已迭代次数）会跟着显示 —— 这是现场判断"正航向有没有跑"最快的办法。

## 限深保护（上浮不得越过水面）
下位机按 **14B 遥测帧**（`0xAA55` + 深度/目标深度/roll/pitch/yaw + 校验和，协议见 `base/telemetry.py`）
回传深度；`base/uart.py` 在每次发帧**前**读空串口接收缓冲 → 用最新深度限深：

> **当前深度 ≤ `comm.depth_guard.min_depth_m`（现场定死 **0.55 m**）→ 本帧禁止上浮**：heave 分量清零、
> heave 轴当帧回中（`snap: true`，不等 ramp 平滑，防惯性继续上浮）；
> **surge/sway/yaw 一律不动**——任务照常对准/前进，只是机身顶不出水面。

- 生效范围：所有下发路径（任务的 `send_dof`、`set_motion` 预设，含手动模式键盘遥控"上浮"）；
  `send_frame_bytes` 原样发字节、不过该保护（只用于中性/急停/测试帧）。
- 判据是"深度**不小于** 0.55"：深度**恰好 0.55 m 也算触底**，禁止上浮；0.56 m 起放行。
  ⚠️ 这个值**用户 2026-09-18 定死、不准改**（有不变量用例守着）；本地曾长期写 0.3，
  整文件覆盖板端配置时把板端的 0.55 冲掉过**两次** —— 所以 **cfg 不要整份推板端**（见 tools/README.md）。
- 遥测缺失或超时（`stale_ms`，默认 500 ms）→ 默认**放行**（`stale_action: pass`，台架/仿真友好），
  非 SIM 下首次告警一次；要"断链也不盲上浮"就把 `stale_action` 设成 `block_up`
  （SIM/无串口仍放行，免得台架被锁死）。板上若一直打"没有新鲜深度遥测"告警 = 下位机没在上行遥测，
  保护形同虚设（`comm.debug.tel_log_ms` 控制遥测打印节流）。
- 总开关 `comm.depth_guard.enable: false` 可退回"不做深度限制"的旧行为；参数全在 `cfg/comm.yaml`。
- 模块边界：协议/解析/重同步在 `base/telemetry.py`（纯函数 + `TelemetryReceiver`，可单独测/复用），
  保护判定与下发在 `base/uart.py`（`_apply_depth_guard`），任务代码**不感知**该保护。
- 现场可见：`[UART←] depth=0.72m target=…`、
  `[UART] 限深保护：depth=0.52m ≤ 0.55m → 禁止上浮(heave 1.00→0)`、
  发帧行附 `[限深保护 depth=0.52m]`、画面叠加 `depth=0.52m guard=on min=0.55m`。

## 画面推流 / 手动模式（端到端约 60~90 ms）
方案与实测数据见 `doc/前视USB相机低延迟推流方案.md`；根目录只有一个入口脚本 `manual.sh`，
推流/接收实现在 `manual/stream.py`（库）。

**录像放在水面 PC**（板端只推流：省板端 CPU、不受写盘拖累，帧率更高）。

```bash
# 板端：遥控桥 + 推流（默认；无串口硬件加 --sim）
./manual.sh
# 例外：确需板端本地录像时（PC 不在场）
./manual.sh --record --seconds 60 --out rec.mp4

# 水面 PC —— 用 pc/ 的总调度脚本（键盘遥控 + 本地录像，见 ../pc/README.md）
cd ../pc
./pc.sh                    # 键盘遥控 + 录像（默认直接出 mp4 到 pc/record/）
./pc.sh --show             # 边看边录（结束后自动封 mp4）
./pc.sh --raw              # 只留原始 .mjpeg
./pc.sh --view             # 只看画面
```

> 同一个 UDP 端口同一时刻只能被一个进程接收：**既要看又要录就用 ②**（`--record-raw` 在同一进程里直存 + 显示）。

**推流不会和识别抢相机**：UVC 设备同一时刻只允许一个进程取流（第二个进程报 `Device or resource busy`）。
推流因此不单独起进程，而是由 `base/camera.py` 在打开相机时顺带把**原始 JPEG** 交给 `manual/stream.py` 发出
（开关：`cfg/vision.yaml` 的 `stream.enable`，**2026-09-18 起默认 `false`**——调任务时一般在板端
直接看画面/叠加，不必推流；手动模式由 `manual.sh` 用环境变量覆盖）。
另外 `cv2.VideoCapture` 默认协商 YUYV（本相机 720p 只有 9 fps），`base/camera.py` 已强制 MJPG（60 fps 档）。

## 联调 TODO
- gate keypoint `.bin` 上板前：解码自检（§6）+ **卷尺深度曲线 0.5~6m**（判据 |误差|≤10%）；
- **遥测联通性**：下位机确认在按 `0xAA55` 14B 帧回传深度（板上应持续看到 `[UART←] depth=…`，
  看不到就是没回传、限深保护放行中）；并把下位机深度读数与人工卷尺/标尺核对（这是深度标定，
  **不是**去改 `min_depth_m` —— 它已被用户定死在 0.55，有不变量用例守着）；
- ~~门框尺寸实测~~ **已实测（09-22）：外缘到外缘 77×56 cm** → `frame_w/frame_h=0.77/0.56`；
  仍待陆上 A1 用 `a` 复核"模型眼里的门宽"是否等于外缘（可能落在管子中心线上）；
- **DOF 极性与速度标定**：surge/sway/heave 方向、`norm→m/s` 曲线、REACQUIRE 后退方向；
- **冲刺与兜底标定**：`surge.through`(0.6) 与 `through.confirm_ms`(2500) 决定"能不能冲出去"、
  `loiter.{timeout_ms,dx_max,dy_max}` 决定"在门口等多久才拍板"——
  判据 = 通过时无接触（30 分）且不超时；
- **SEARCH 平移扫视标定**：`gate.search.sway`(0.45≈25% 推力) 是否真的能把船左右挪动、
  以及一趟来回后**有没有净漂移**（开环、无横向位置反馈，见 cfg 注释里标为未验证的那条）；
- 高低门编排（`pass_target` 语义：每轮一门）；相机—机身偏置 `body_center_offset` 标定后再决定是否接进对准。
