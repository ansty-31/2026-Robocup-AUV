# AUV 视觉导航项目（RDK X5 / Ubuntu 22.04 / Python）

> **当前状态（20260926）** —— 当天全部改动/未验证/回退见 **`doc/记录/2026-09-26-改动记录-review.md`**
> - **门 pose 权重再次换代 → 阶段一 `g240_i16` 版**：
>   `models/gate_kpt_stage1_g240_i16_bayese_640x640_nv12.bin`（md5 **`ef52c18b…`**，3,938,787 B）；
>   上一代在 `bak/weights/bakup_auv5/`（`9ac61773…`，09-24 采用版）。**契约与后处理规则见
>   `doc/设计/gate_pose_decode_spec.md`**（输入 NV12/640/**squish**、输出 **9 张量 float NHWC**、
>   `x_px = x_cell × stride`、`v` 要自己 sigmoid）。上一代 auv5（`945fdd01…`）在训练工程已删除，
>   可能只剩板端一份 —— 回退前先核实。
> - **规范 §4 的"解码后处理"已落进本项目**（🆕 `gate/percept/gate_postproc.py` + 三个新配置段）：
>   ① 候选阈值 `gate.det.conf: 0.5`（**只作用于 gate**，不动 ball/gate 共用的 `model.score_threshold`；
>      规范 §6 的评测工作点是 0.6 —— 现值以 `cfg/vision.yaml` 为准）；
>   ② 角点可信下限 `V_MIN = gate.keypoint.conf_thr: 0.8`（原 0.9；0.8 是阶段一权重的评测工作点）；
>   ③ 几何合法（四角顺序 + 不自交）与**重复框去重**（小框 conf 更低 **且** ≥2 条边重合 → 丢小框，
>   **判据不是包含关系**）；④ 选门改成**近距离优先** `gate.select.mode: near`（框宽当测距代理；
>   回退旧行为 = 改成 `corner`，一行）。落地细节与未验证项见 review §2/§3/§4。
> - **本地 370 用例全绿**（2026-10-07；用例数随开发增长，以 `python3 -m pytest tests/ -q` 输出为准）。
>   **板端已同步**（2026-09-26 21:4x）：`check_board_parity.sh --board` **双向 95/95 一致**、
>   板端全量 pytest **与本地同数（当前 327 passed）**（原先靠"台架分叉"解释的 7 条失败随分叉统一而消失）。
>   ⚠️ **板端 `SIM_MODE` 现在是本地值 `True`**（串口只打印、不驱动推进器）—— 实船/下水必须
>   `AUV_SIM_MODE=0 python3 main.py --task gate`（或把那行改回 `False`）。
>   ⚠️ 板上实测：**`hbm_runtime` 返回的 9 张量字典顺序不固定**（同一 bin 连跑两次排列都不同），
>   解码必须**按形状认**（本项目 `decode_yolo11_kpt` 本来就是自描述分组，天然免疫；手写后处理别踩）。
>   ⚠️ **新权重在真实水域、有门场景下的精度/帧率仍未测**（板端只跑通了契约自检：9 张量 float32 形状全对）。
> - **ALIGN.HDG（正航向）2026-09-23 修了 4 个症状**（转不动 / 转反 / 转过头 / 节拍慢）：
>   符号在代码里定死（`dof_map.yaw.sign: +1` + `motion.turn_pid.imag_sign: +1`，**不再探向**），
>   转向下传与 `task/run_ball_reverse.sh` 同形；顺序固定为"先居中 → 达标且本帧是 full → 才起转"
>   （`gate.hdg.entry_stale_ok: false`）；转向期间由 `gate_task._turn_inner_loop` 按
>   `comm.gate.hdg.turn_period = 0.05 s`（**20 Hz**）轮询"下位机完成反馈"。
>   ⚠️ **这条链路尚未在真实水域跑通过**（最新一趟板端日志里 `mode == full` 出现 0 次 ⇒ psi 一次都没测到），
>   诊断与下一步见 review 文档 §5。
> - **角点阈值（2026-09-26，随新权重换代）**：`keypoint: {conf_thr: 0.8, vis_thr: 0.5}` + 代码兜底
>   `gate/motion/params.py::_D_KPT`（`gate_task` 再导出）同值。0.8 = 规范 `doc/设计/gate_pose_decode_spec.md` §6 的 `V_MIN`（旧值 0.9/0.7 是给
>   **上一代**权重现场试出来的）；`vis_thr` 刻意更低，只为让低 v 角点留在画面/日志里做诊断 ——
>   它进不了 mode/PnP。**cfg 不要整份推板端**（`comm.yaml` 仍有 4 处台架分叉）。
> - **任务二 过门（gate）= 扁平 v1.2 版**（`gate/` 两层结构；源自 v1.2 原版 + 2026-09-18 新增正航向 + 2026-09-26 新增后处理，
>   2026-09-30 把相位机按方法簇拆分、`gate_task.py` 提到 `gate/` 作总调度），
>   这是**定版方案**，不再有分层结构。
> - ★ **2026-10-07 过门阈值/机制按实船日志重设**（依据 `log/rungate_20261006_postsway` 385 帧；
>   成篇记录 **`doc/记录/2026-10-07-gate-阈值与运动重设.md`**，数据特性分析 `log/FEATURES_20261006_postsway.md`）：
>   `align` **分档 + 纵向零点 `dy_target=−0.30`**（`px_x/px_y/confirm_frames = 0.10/0.15/5`；
>   width `0.14/0.06`、coarse `0.18/0.10`）；`z.cross`（=`slow_max`=`width.z_max`）**1.2→1.5**；
>   `pnp.z_max` **15.0→3.0**；`through` 改**两段式**（`fast_ms=2000` 快冲 + 到 `confirm_ms=6000` 降速保持）
>   且**过门居中闸 = 本档对中带**（`through.loose=1.0`）；`hdg` 新增 `psi_ema_frames=10`（ψ 先平滑再比 tol）
>   与 `body_delta_max_deg=1.0`（转向前"机身稳"，与 align **并行**计数）；
>   `select.k_lo_ratio 0.75→0.30`；`motion.pid_sway` `kp 8→1` + 新增 `bias=0.138`（执行器死区补偿，
>   **ball 共用会一并变**）。⚠️ `through` 两个时长**尚未标定**（本趟日志 THROUGH 从未触发）。
> - `kpt_memory`（门角点逐点软融合）是**可选开关、2026-09-18 起默认关闭**：
>   `cfg/vision.yaml → vision.gate.kpt_mem.enable: false`（板端曾因拼写错 `flase` 被
>   `bool('flase')=True` 而**实际一直开着**）。开启用 `enable: true` 或 `AUV_GATE_KPT_MEM=1`
>   （优先级 `AUV_GATE_KPT_MEM` > `enable` > 兜底 `false`；helper API 在
>   `gate/percept/kpt_memory.py::kpt_mem_enabled/enable_source/build_kpt_memory`）。
> - **任务一 小球（ball）= `task/ball.py`**（已定）。
> - **过门的两条横向/转向约定（2026-09-20 用户定）**：
>   **① 居中只有 sway 平移**（`align_yaw.{enable,in_px,in_pose}` 与其 PID **已整体删除**）；
>   **② SEARCH 是左右平移扫视，不旋转**（原旋转脉冲已删除，`gate.search.{sweep_s,pause_s,sway}`）。
>   gate 里**唯一的 yaw 来源**是 ALIGN.HDG 正航向（`gate/motion/hdg.py` +
>   `common/motion/turn_deg.py`；**只给角度**，执行在下位机）。
> - **限深保护（默认开启）**：下位机回传 14B 遥测帧（`0xAA55`+深度/姿态+校验和，
>   `base/hw/telemetry.py`）→ 深度 ≤ `comm.depth_guard.min_depth_m`（**现场定死值，2026-10-06 起为 0.50 m**）时
>   **禁止上浮**（只压 heave，surge/sway/yaw 不动），机身顶不出水面。详见下文「限深保护」一节。
> - **返回出发区（记忆返回）已整体移除**：`return_by_memory` 代码、交接协议与相关参数都不再存在，
>   当前撞球收尾是 `task/run_ball_reverse.sh` 的**定时直线倒车**（开环，不做轨迹回放）。
> - 以下功能**已确定不需要、已删除**，本项目不再提供：v1.3/v1.4 分层 gate（`vision/`+`data/`+`motion/`）、
>   质量分 Q、线索 cues、帧守卫（`common/frame_stamp.py`）、`common/streak.py`、DOF ramp
>   （`common/ramp.py`）、`comm.handover`、`comm.back`、`STATE_BACK`；过门侧另有
>   **`comm.gate.align_yaw`（居中用 yaw 的通道）**、**SEARCH 的旋转脉冲**（`search.spin_s/yaw`）、
>   **width/coarse 的「对准就直冲」出口**（`*.dash*`）、以及 `comm.gate.align_yaw.heading`
>   连续航向通道 —— 恢复任何一项都要先说明理由并让用户拍板。下文只描述**当前实际存在**的
>   代码/工具/配置。
> - ⚠️ **跨工作区只读约束**：训练工程 `../RDKX5-YOLOv11n-/` 只有**测试脚本**可以与它共享/写入，
>   该工作区下的**其他文件一律只读**——不改它的训练/导出/量化脚本与配置（那是训练侧流水线，
>   改动会静默影响 `.bin`，见 `doc/记录/算法说明-gate-PnP移植方案.md` §6 与训练工程自身 README）。

RoboCup AUV 赛事视觉代码。平台：**RDK X5（3.5.0）**，前视 USB + 下视 IMX415(MIPI)（下视保留，用于录素材/未来任务）。
识别：YOLO 蓝/红球 + gate（**keypoint 四角 + PnP** 新前端）；串口 11B 帧向 STM32 下发 DOF，
STM32 回传 14B 遥测帧（深度/姿态 → 限深保护）。

> 先读：`README.md`（用法）→ `doc/设计/算法说明.md`（总体）→ **`doc/设计/过门逻辑树.md`（过门逻辑的唯一入口：相位/档位/动作的逻辑图 + 异常应对 + 代码冲突清单）** → `doc/记录/过门-状态机与参数.md`（**历史逐状态导读；§9 是现场实测记录汇总**，描述 2026-09-30 重构前的结构，逻辑以逻辑树为准）→ `doc/设计/gate_pose_decode_spec.md`（**门 pose 模型的解码契约 + 后处理规则，换 bin 必读**）→ `doc/记录/算法说明-gate-PnP移植方案.md`（gate 设计与移植）→ `doc/记录/算法说明-gate-角点逐点融合滤波.md`（角点逐点数据处理）→ `doc/记录/实验待测-runbook.md`（**下水前后的实测阶梯与判据**）→ `doc/记录/2026-09-2X-改动记录-review.md`（**逐日全记录：改了什么 / 未验证 / 怎么回退**）。
> 参数怎么改：`cfg/*.yaml` 注释只写"是什么 + 当前值 + 调它的后果"；**数值是怎么来的、现场发生过什么，都在 `doc/记录/过门-状态机与参数.md` §9**。

## 目录分区（英文分区命名）

```
auv_vision/
├── main.py              # 装配/状态机入口（ball/gate；由 task/run_ball_reverse.sh 或直接调用）
├── preview_detect.py    # 下水前视觉自检（带识别框；--gate-kpt 看门框 + 4 角点）
├── manual.sh            # 手动模式唯一入口：遥控桥 + 录像 + 推流
├── manual/              # 手动模式三件套：udp_server.py(遥控桥) · recorder.py(录像) · stream.py(推流/接收库)
├── base/                # 平台基础（**两层**）：hw/ 设备与协议 · cfg/ 配置 · log/ 运行记录
│   ├── hw/              #   camera.py(前视/下视/mipi) · uart.py(11B 下发帧 + 限深保护)
│   │                    #   telemetry.py(0xAA55 14B 遥测上行：深度/姿态)
│   ├── cfg/             #   settings.py(YAML→S.vision/S.comm；工程根自动定位)
│   └── log/             #   turn_log.py(转向调用日志：只写字、不参与控制)
├── common/              # 跨任务公共件（**两层**）：vision/ 图像 · motion/ 运动 · cfg/ 配置读取
│   ├── vision/          #   detector.py(检测) · preprocess.py(图像链路)
│   ├── motion/          #   PID.py · turn_deg.py(指定角度轴：yaw/pitch/roll，遥测完成标志 + 硬停)
│   │                    #   search_scan.py(yaw 慢扫，门用) · search_sweep.py(**任务三/四的平移扫视**) ·
│   │                    #   axis.py(轴动作/下潜到位/定时推力；pitch/roll **±50° 硬限**)
│   │                    #   drop.py(**横倾放球/倒球序列**：任务三倒错球、任务四放球共用)
│   └── cfg/             #   cfgnode.py(cfg 节点读取 + **ball/gate 共用参数的唯一入口**)
├── task/             # 任务一（撞球）：
│   ├── ball.py          #   任务一 撞球 BallTask
│   ├── run_ball_reverse.sh  #   编排（8 步）：待机→(可选)下潜→(可选)前进→撞球→倒车→
│   │                        #     前进 2s→**左转 90°（遥测闭环）**→过门任务（可 AUV_GATE_AFTER=0 截断）
│   └── run_gate.sh      #   只跑过门（下水专测 gate 用；参数见脚本头部注释）
├── gate/                # 任务二 过门（**两层**：percept/ 感知 · motion/ 运动）
│   ├── percept/         #   gate_decode.py(keypoint 解码) · gate_detector.py(组合根) · gate_frontend.py(mode 判定)
│   │                    #   gate_postproc.py(后处理：可信角点/几何合法/去重/选门) · geometry.py(CameraModel/PnP/反投影)
│   │                    #   kpt_memory.py(角点逐点软融合，**默认关**) · mock.py(仿真后端)
│   └── motion/          #   **相位机按方法簇拆分**（2026-09-30）：gate_task.py 只剩编排（~330 行）
│                        #     params.py 常量与缺键兜底表 ｜ channels.py 通道与小件（唯一执行出口 `_set_info`）
│                        #     hdg.py 正航向与转向 ｜ modes.py 各档位（位姿/width/coarse/重取） ｜ exits.py 出口与终局
│                        #     hdg.py = ALIGN.HDG（**算法核心 + 任务侧胶水**，2026-09-30 由 heading_align.py 合并而来）：
#              流程 **先居中 → 再转向 → 反向平移+缓慢后退**，直到当前门出现 N 个可信角点
#              （`comm.gate.hdg.post_sway_kpt_min`，默认 2）即认为"对齐光轴"完成、交回视觉；
#              **没有"航向优先"**（旧 `entry_psi_first` 已删）
│                        #   另有一个更粗的变体 B（4 文件）与对比：`bash bak/migration/_gate_split/switch.sh A|B`、bak/migration/SPLIT-AB.md
│                        #   （gate 的**逐状态导读**在 doc/记录/过门-状态机与参数.md；改 gate 前先看）
├── handling/            # **任务「夹取 + 放置」**（同一总调度 `handling_task.py`，mode=grab|place|full）
│                        #   motion/  motion/{params,actions,phases}.py —— 兜底值 / 关键动作 / 相位（与 gate/ 同规）
│                        #   percept/ 纯 CV 红球检测与 ROI 跟踪（下视；不走 BPU）
│                        #   `full` = 夹取完成后**同一实例交接**给放置（限深与收尾由本任务持有）
│                        #   感知全程用**下视**（`wants_down()`：grab/full 为真、place 为假）
├── bak/                 # 归档（gitignored，不上板）：**按用途分 8 类**（snapshots/ weights/ trees/ retired/\n│                        #   trials/ files/ migration/ + README.md 索引与旧→新映射表）
├── tools/               # 工具（**按用途分三类**；除注明外都是本机驱动、不传板端，见 tools/README.md）
│   ├── deploy/          #   部署与一致性：deploy_to_board.sh(烧录) · check_board_parity.sh + board_parity.md5(清单)
│   │                    #   archive_baks.sh(归档 *.bak*，**会随部署上板**) · tidy_board_bak.sh(板端 bak/ 整理)
#   （A 域分叉 sync/cfg 已于 2026-10-01 退役 → bak/retired/Adomain_20261001/）
│   ├── check/vision/    #   自检·视觉：check_kpt_decode.py(解码约定) · check_gate_pose.py(位姿为何被拒) · auv_kpt_meter.py
│   ├── check/boat/      #   自检·船体：check_dof_sign.py(舵向) · check_hdg_lockup.py(航向锁死)
│   ├── check/pipeline/  #   自检·工程：check_domain.py(两域隔离) · check_paths.py(路径) · check_pipeline_identity.py(同域自证)
│   ├── analyze/log/     #   判读：analyze_task_log.py(逐帧日志①~⑧) · feature_coverage.py · analyze_kpt_dump.py
│   │                    #         analyze_heading.py(PnP 安装角/航向噪声) · analyze_pnp_center.py(位姿中心 vs bbox)
│   └── analyze/calib/   #   标定：pnp_calib.py(位姿/深度：真值 dump → 误差表+尺寸反演) · ruler_calib.py(卷尺靶)
│                        #         tape_ticks.py(刻度周期) · label_corners.py(手工标 4 角 / --measure 刻度)
│                        #   **ruler_calib.py（靶子读数 → 等效焦距 fx/fy 与距离口径 c；--gate 与门框联立）**
│                        #   **tape_ticks.py（刻度周期法：不点鼠标，用卷尺自带 1cm 刻度测像素比例）**
├── cfg/                 # vision.yaml · comm.yaml · front_camera.yaml（参数唯一来源）
│                        #   comm.yaml 的 **motion:** 段 = ball/gate **共用**的底层运动参数
│                        #   （sway/heave 两套 PID / surge_fast·surge_slow / loss_inertia_surge），
│                        #   任务段只放各自特有的旋钮；共用值不在两处各写一份
├── doc/                 # 文档分三层（见 doc/README.md）
│   ├── 设计/            #   参考型：算法说明.md · gate_pose_decode_spec.md（门 pose 解码契约，换 bin 必读）
│   │                    #   **过门逻辑树.md（过门相位/档位/动作的逻辑图：总图 + 普通流程 + 异常应对）**
│   │                    #   **过门流程树-percept识别.svg/.pdf（对外展示：识别流程树，含 mode 释义 + cfg 判据带）**
│   │                    #   **过门流程图.svg/.pdf（对外展示：设计层面的过门流程——找门→对准→转正→进近→冲门→计数）**
│   ├── 现场/            #   现场与待办：现场卡-靶子法.md · 待研究-缺角位姿先验（三维信息复用）.md
│   ├── 记录/            #   记录与归档（**不改写历史**，见 记录/README.md）：
│   │                    #     2026-09-2X-改动记录-review.md（逐日全记录）· 过门-状态机与参数.md（逐状态导读 + §9 实测）
│   │                    #     实验待测-runbook.md（PnP 标定实验）· 算法说明-gate-PnP移植方案.md
│   │                    #     算法说明-gate-角点逐点融合滤波.md · 前视USB相机低延迟推流方案.md · psi测量步骤_20260923.txt
│   ├── 注释历史.md      #   按文件/按键的历史索引（**入口**）：实测数字 / 试错 / 参数沿革 / 注意事项
│   └── 注释历史/        #   索引的正文，按域分 7 个文件（cfg/gate/base-common/grab-task-manual/tools/tests/misc）
├── models/ · tests/     # 权重(.bin) · 无硬件测试套件（**按层分子目录**，见 tests/README.md）
│                        #   tests/：**370 例**（2026-10-07；以 `pytest tests/ -q` 输出为准）
│                        #     platform/  test_base   平台：settings/11B 帧/遥测/限深保护/硬停
│                        #                test_common 公共件：PID/图像链路(NV12·squish)/Det/cfgnode
│                        #                test_paths 路径落点 · test_hud 叠加层
│                        #     tasks/     test_ball        撞球相位机
│                        #                gate/            **(过门用例按主题成组)**
│                        #                  test_gate_vision  解码约定/PnP 往返/角点记忆/mode 降级
│                        #                  test_gate_flow    档位仲裁/出口兜底/正航向与 SWAY_BACK/配置守卫
│                        #                  test_gate_postproc 后处理：L/R 归一/几何合法/去重/选门键
│                        #                test_motion      运动原语：额定转角 + 正航向
│                        #     tooling/   test_pnp_calib   **PnP/深度标定工具**的合成往返（反演标尺/符号/CSV/报告）
│                        #                test_label_corners 手工标注：吸附/坐标域/schema/端到端喂通 pnp_calib
│                        #                test_ruler_calib   卷尺靶子：跨度/斜视诊断 + (fx,c) 合成往返 + 混合轴向
│                        #                test_tape_ticks    刻度周期法：亚像素/透视梯度/谐波/交叉校验门
│                        #   （`_bite_probe.py` 是"会咬"自检探针，**不被默认收集**，
│                        #     需要时 `python3 -m pytest tests/_bite_probe.py -q`）
│                        #   分层块：`pytest tests/platform -q` / `tests/tasks` / `tests/tooling`
```

分区语义：`base`（平台基础设施）/ `common`（跨任务公用）/ `task`（任务一 撞球）/ `gate`（任务二 过门）/
`grab`（任务三 夹取）/ `place`（任务四 放置）。
依赖规则：任务代码只 import `base`/`common` 与同级任务模块；`main.py` 是唯一装配点。
⚠️ **任务三与任务四并列，谁也不 import 谁** —— 两者共用的动作必须在 `common/motion/`
（例：横倾放球/倒球 = `common/motion/drop.py`；`tests/tasks/handling` 有用例钉着这条）。

## 快速开始（本机，无硬件）

```bash
python3 -m pytest tests/ -q              # 无硬件测试（**370 例**，2026-10-07；以实际输出为准）
python3 main.py --task ball              # 只跑撞球（SIM/mock）
python3 main.py --task gate              # 试跑过门（cfg model.mode: mock）
AUV_TASK_LOG=log/grab.jsonl python3 main.py --task grab      # 任务三 夹取（感知全程下视）
AUV_TASK_LOG=log/place.jsonl python3 main.py --task place     # 任务四 放置（运输 → 横倾放球 → 停动力 3s）
AUV_TASK_LOG=log/full.jsonl python3 main.py --task handling   # 先夹取，成功后同一实例交接给放置
#   ^ AUV_TASK_LOG 是**逐帧 JSON 日志**（相位/判据/通道/遥测），判读见下面的 analyze_task_log.py
python3 preview_detect.py --gate-kpt     # 下水前：门框 + 4 角点 + 置信度（船不动）
python3 tools/analyze/log/analyze_task_log.py log/*.jsonl   # 下水后：逐帧日志判读（相位/出口/通道能动力）
python3 tools/analyze/log/feature_coverage.py log/*.jsonl   # 下水后：这轮**哪些功能没被用到**
python3 tools/analyze/calib/pnp_calib.py --report r.md log/pnp_*/pnp_z*.jsonl  # PnP/深度标定（文件名即真值）
```
> `tests/` 与训练工程 `../RDKX5-YOLOv11n-/` **可以共享测试脚本**（唯一允许写入对方工作区的东西）；
> 训练工程下的**其他文件只读**，不要在本工程的任务里顺手改它们。

### 真机编排（.sh，见 task/）
```bash
cd task && ./run_ball_reverse.sh      # 待机→(可选)下潜→(可选)前进→撞球→倒车→前进2s→左转90°→过门
AUV_GATE_AFTER=0 ./run_ball_reverse.sh   # 只做前 7 步（分段试，不接过门）
AUV_SIM_MODE=1 ./run_ball_reverse.sh     # 台架：只打印不发串口（**改脚本后必跑一次干跑**）
./run_gate.sh                            # 只跑过门任务
./run_grab.sh                            # ★ 只跑夹取任务（任务三；标定闸自检 + 相机占用清理 + 日志默认开）
```
- **撞球后返回**：`run_ball_reverse.sh` 在 `main.py --task ball` 结束后**定时直线后退**
  （开环、不依赖视觉；`AUV_REV_S`/`AUV_REV_SURGE` 调时长与速度）；
- **倒车后的三步**（2026-09-19 加）：前进 `AUV_POST_FWD_S`(2s) → **按角度左转**
  `AUV_TURN_DEG`(90°，`common/motion/turn_deg.py` 用下位机遥测 yaw **闭环**执行，转完必硬停)
  → `main.py --task gate`。转角可用 `AUV_TURN_LEFT/AUV_TURN_KP/AUV_TURN_KD` 覆盖；
  **没有遥测时 turn_deg 直接拒转（退出码 5）** —— 转向只有遥测 yaw 闭环，没有开环/盲转备案；
- 单独试转向：`python3 common/motion/turn_deg.py --deg 90 --dir left`（`--help` 看全部旋钮）。
- 下视相机（`cfg/vision.yaml camera.down`：**usb `/dev/video0`**）**是任务三/四的感知源**
  （`--task grab|handling` 直接吃下视；`camera.down.calibration: null` = 有意不去畸变）。
  下视是**独占设备**：跑任务前先清掉残留进程，否则会被别的程序占住。

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

- **任务一 撞球**（task/ball.py）：**SEARCH→CENTER→APPROACH→DASH→STOP** 运动链
  （居中=仅 yaw+heave；接近=分级前进+仅 sway；面积(EMA)≥`dash_ratio` → 以分级最高速 `surge_fast`
  冲刺 `dash_dur_s`；不检测是否撞到；冲刺后 STOP 全 0 保持 `stop_hold_s` → DONE(hit)，
  总时限 `comm.ball.timeout_ms`）；
- **撞球之后**（`task/run_ball_reverse.sh`）：**定时直线倒车**（开环、不依赖视觉）→ 前进 2s
  → **按角度左转 90°**（`common/motion/turn_deg.py`，遥测 yaw 闭环 + 转完硬停）→ 过门任务；
- **任务二 过门**（`gate/gate_task.py` 只有编排；**两层结构** `gate/percept/` 感知 + `gate/motion/` 决策，
  2026-09-30 按方法簇拆分；**逻辑图见 `doc/设计/过门逻辑树.md`**，细则/参数见 `doc/记录/算法说明-gate-PnP移植方案.md`）：
  keypoint 四角 → IPPE 6-DoF，深度 `Z=tvec.z`；
  ALIGN 子状态 GOLDEN/**HDG(正航向)**/**SWAY_BACK(转后回找)**/CREEP/HOLD/REACQUIRE → APPROACH → THROUGH（机身过门判据）。
  门 = 闭合矩形框(红 PVC，**实测外缘 0.77×0.56 m**，管外径 5cm，悬空，对称无朝向要求)。
  **直冲出口两条**（任一成立即进 THROUGH，且都要先过 `through.require_align_deg` 航向闸门）：
  ① `Z≤z.cross` 连续 `z.cross_confirm_frames` 帧（实测有效，勿动）
  **并且**落在**本档位自己的居中范围内**连续 `through.center_frames` 帧
  （★ 2026-10-07 用户定"务必在各自要求的居中范围内触发过门"：闸 = 本档对中带 × `through.loose`，
  与 `through.center_x/center_y` 取 AND ⇒ 只会更严）；② 在门口超时兜底
  `loiter.*`（框占比 ≥ 阈值 + 在安全带内持续超时 → 自己拍板直冲）。
  ⚠️ **原「近距丢门判过门」这条出口已按用户要求移除**：驱动相位的代码先摘掉，2026-10-01 再把残留的
  死代码（`_near_lost` / `_fresh_z` / `_z_ms`）与失效 cfg 键（`z.near_lost_m` / `z.z_stale_ms`）一并删掉；
  `z.near_lost_ratio` 仍被 loiter 的 `near_ratio` 引用。
  现在 ALIGN/APPROACH 丢门是「防抖 hold → 超 `pose_hold_frames` 回 SEARCH」，**不判过门、不直冲**。
  ⚠️ 原「width/coarse 对准就直冲」**已整体删除**（2026-09-18 用户定）。
  **SEARCH 是左右平移扫视**（`gate.search.{sweep_s,pause_s,sway}`，对称来回）——2026-09-20 用户定
  「过门过程中不允许旋转搜索」；**居中只有 sway**（`align_yaw` 已随其 PID 一起删除），
  gate 里唯一的转向来源是 ALIGN.HDG 正航向（**相对角交下位机执行**，上位机不再跑 PID）。
  coarse 仲裁原则：**只有未对准才 HOLD/后退**。
  角点逐点融合 `gate/percept/kpt_memory.py` 为**可选功能、2026-09-18 起默认关闭**（板端曾因拼写错 `flase` 实际开着）：
  开启用 `vision.gate.kpt_mem.enable: true` 或 `AUV_GATE_KPT_MEM=1`；关闭时角点单帧直用。

- **任务三 夹取 + 任务四 放置**（`handling/`）：**同一个总调度两种模式** —— `mode="grab"` 走
  `INIT→PITCH_UP→SEARCH→CENTER→APPROACH→LEVEL→ALIGN→DIP→RISE→VERIFY→(DUMP)→EXIT`，
  `mode="place"` 走 `INIT→TRANSPORT→RELEASE→STOP`，`mode="full"` 夹取成功后**同一实例交接**给放置。
  感知全程用**下视**（纯 CV 红球检测 + ROI 跟踪，不走 BPU）；相位/关键动作/兜底值拆在
  `handling/motion/{phases,actions,params}.py`（与 `gate/` 同规）。细节见 `handling/README.md`。

## 参数（cfg/*.yaml）
`vision.yaml`：camera(front/down)、`image.*`（**ball/gate 共用图像链路，勿改**）、`model.*`
（含 `task_models.gate`）、`gate.*`（几何/keypoint/PnP/kpt_mem 开关）、`stream.*`；
`comm.yaml`：serial/dof_map/心跳/急停/ramp/**depth_guard(限深保护)**/**motion（ball/gate 共用的
底层运动：sway/heave 两套 PID + 速度档）**/tasks.enabled（默认 `[ball]`；gate 需后端或 mock）/ball/gate 任务参数。
> **共用参数只写一份**：`comm.motion` 里的值同时被撞球与过门读取（读法见 `common/cfg/cfgnode.py`
> 的 `motion_num/motion_node/motion_pid`）；确需某个任务单独一套时，才在该任务段写覆盖键
> （如 `comm.gate.pid_sway`）。`gate.surge.fast/slow` 与 `ball` 段**不再存在**这些重复键。
模型：X5 OE `.bin`（march `bayes-e`）；`.hbm` 不兼容。

## 运行监测
串口帧打印 + 画面叠加（含 gate `phase/substate/mode/z/dx/dy/sway/heave/surge/yaw/hdg/pass`、
下位机 `depth=… guard=… min=…m`）—— cfg debug 开关；`AUV_SHOW=0` 关窗口。
> 画面里 **`yaw` 恒 0 是正常的**（居中只用 sway）；只有正航向 ALIGN.HDG 在转的时候才会非 0，
> 同时 `hdg`（航向误差°）与 `hdg_i`（已迭代次数）会跟着显示 —— 这是现场判断"正航向有没有跑"最快的办法。

## 限深保护（上浮不得越过水面）
下位机按 **14B 遥测帧**（`0xAA55` + 深度/目标深度/roll/pitch/yaw + 校验和，协议见 `base/hw/telemetry.py`）
回传深度；`base/hw/uart.py` 在每次发帧**前**读空串口接收缓冲 → 用最新深度限深：

> **当前深度 ≤ `comm.depth_guard.min_depth_m`（现场定死值，**2026-10-06 起 0.50 m**）→ 本帧禁止上浮**：heave 分量清零、
> heave 轴当帧回中（`snap: true`，不等 ramp 平滑，防惯性继续上浮）；
> **surge/sway/yaw 一律不动**——任务照常对准/前进，只是机身顶不出水面。

- 生效范围：所有下发路径（任务的 `send_dof`、`set_motion` 预设，含手动模式键盘遥控"上浮"）；
  `send_frame_bytes` 原样发字节、不过该保护（只用于中性/急停/测试帧）。
- 判据是"深度**不小于** 该值"：**恰好等于也算触底**（禁止上浮），比它深 1cm 起放行。
  ⚠️ 这个值是**现场定死**的（有不变量用例守着）：2026-09-18 定 0.55 → **2026-10-06 用户下调为 0.50**
  （为了让夹取翘头工作深度能真到 0.5）。**改它必须同时改** `base/hw/uart.py::_D_MIN_DEPTH_M`
  与 `tests/platform/test_base.py` 的不变用例。
  ⚠️ 下调等于**放宽安全边界**（露出水面 = 比赛立即停止）：抬头 30° 时机头抬高 ≈ (机长/2)·sin30°，
  必须现场量机头（含相机壳）离水面的余量。
  ⚠️ 历史上整文件覆盖板端 cfg 把现场值冲掉过**两次** ⇒ **cfg 不要整份推板端**（见 tools/README.md）。
- 遥测缺失或超时（`stale_ms`，默认 500 ms）→ 默认**放行**（`stale_action: pass`，台架/仿真友好），
  非 SIM 下首次告警一次；要"断链也不盲上浮"就把 `stale_action` 设成 `block_up`
  （SIM/无串口仍放行，免得台架被锁死）。板上若一直打"没有新鲜深度遥测"告警 = 下位机没在上行遥测，
  保护形同虚设（`comm.debug.tel_log_ms` 控制遥测打印节流）。
- 总开关 `comm.depth_guard.enable: false` 可退回"不做深度限制"的旧行为；参数全在 `cfg/comm.yaml`。
- 模块边界：协议/解析/重同步在 `base/hw/telemetry.py`（纯函数 + `TelemetryReceiver`，可单独测/复用），
  保护判定与下发在 `base/hw/uart.py`（`_apply_depth_guard`），任务代码**不感知**该保护。
- 现场可见：`[UART←] depth=0.72m target=…`、
  `[UART] 限深保护：depth=0.47m ≤ 0.50m → 禁止上浮(heave 1.00→0)`、
  发帧行附 `[限深保护 depth=0.47m]`、画面叠加 `depth=0.47m guard=on min=0.50m`。

## 画面推流 / 手动模式（端到端约 60~90 ms）
方案与实测数据见 `doc/记录/前视USB相机低延迟推流方案.md`；根目录只有一个入口脚本 `manual.sh`，
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
推流因此不单独起进程，而是由 `base/hw/camera.py` 在打开相机时顺带把**原始 JPEG** 交给 `manual/stream.py` 发出
（开关：`cfg/vision.yaml` 的 `stream.enable`，**2026-09-18 起默认 `false`**——调任务时一般在板端
直接看画面/叠加，不必推流；手动模式由 `manual.sh` 用环境变量覆盖）。
另外 `cv2.VideoCapture` 默认协商 YUYV（本相机 720p 只有 9 fps），`base/hw/camera.py` 已强制 MJPG（60 fps 档）。

## 联调 TODO
- 🔴 **新权重上板（当前第一优先）**：阶段一 `g240_i16` 只在本地生效，板端还是旧代次 ⇒
  **代码 + `cfg/vision.yaml` + bin 一起推**（cfg 用外科补丁，别整份覆盖），然后回填
  **帧率/延迟**（R4 把 4 个节点提到 int16，算力会涨）、**`mode` 分布**、**角点误差** ⇒ 写回
  `doc/记录/2026-09-26-改动记录-review.md` §4 与 `doc/设计/gate_pose_decode_spec.md` §8「未验证」；
- 🔴 **真实水域"有位姿可用率"**：上一代权重下最新一趟板端日志 `mode == full` 出现 **0 次**（ALIGN 95 帧里有
  17 帧拿到 4 个 ≥0.7 的角点，却一帧都没解出位姿；口径 = 当时板端 `conf_thr`，22:03 快照为 0.7）⇒ psi 测不到 ⇒ HDG 整趟没被触发。
  已实测排除 `pnp.reproj_px=20`（板端 20 帧里 13/14 通过）⇒ 先给逐帧日志补"位姿拒因 + 四点 conf + 重投影 RMS"
  字段，再带录像跑一趟（判据与命令见 `doc/记录/2026-09-23-改动记录-review.md` §5）。**换权重后要重测**；
- 🔴 **新阈值/后处理参数要用实测复核**：`det.conf 0.5`、`keypoint.{conf_thr 0.8, vis_thr 0.5}`、
  `postproc.{edge_tol 0.10, min_edges 2}`、`select.mode near` —— 全部来自规范 §4/§6 的 PC 侧工作点，
  **板端/水里一次都没标定过**。手段：`preview_detect.py --gate-kpt`（不建串口、船不动）看
  `kpt_conf` 分布与实例数（`--conf` 可覆盖候选阈值）；
- ~~gate keypoint `.bin` 上板前：解码自检~~ **已做**（09-23：ONNX 9 张量核对 + 板端域自证 8/8 PASS）；
  仍待 **卷尺深度曲线 0.5~6m**（判据 |误差|≤10%）；
- **米制阈值 ×1.43 映射未回填**（陆上读数=0.70×真距、今天水里≈真距 ⇒ `z.cross` 1.10 / `near_lost_m` 1.43 /
  `slow_max` 1.86 / `width.z_max` 2.15）；**本轮只写注释不生效**。当前 `z.cross=0.77` 的实际触发点 ≈ 真 0.77 m
  （比陆上验证点晚 ~33 cm，冲门更贴门）。要回填就**整组一起改**；
- **遥测联通性**：下位机确认在按 `0xAA55` 14B 帧回传深度（板上应持续看到 `[UART←] depth=…`，
  看不到就是没回传、限深保护放行中）；并把下位机深度读数与人工卷尺/标尺核对（这是深度标定，
  **不是**去改 `min_depth_m` —— 它是现场定死值（现 0.50），有不变量用例守着）；
- ~~门框尺寸实测~~ **已实测（09-22）：外缘到外缘 77×56 cm** → `frame_w/frame_h=0.77/0.56`；
  仍待陆上 A1 用 `a` 复核"模型眼里的门宽"是否等于外缘（可能落在管子中心线上）；
- **DOF 极性与速度标定**：surge/sway/heave 方向、`norm→m/s` 曲线、REACQUIRE 后退方向。
  工具：`tools/check/boat/check_dof_sign.py`（用**视觉**当独立真值——相机是唯一不受 DOF/遥测符号影响的量）；
  yaw 侧已按手动挡既成事实定死 `+yaw = 右转`（`dof_map.yaw.sign: +1`），**不要再为转向去改它**；
- **冲刺与兜底标定**：`surge.through`(0.6) 与 `through.{fast_ms(2000), confirm_ms(6000)}` 决定"能不能冲出去"、
  `loiter.{timeout_ms,dx_max,dy_max}` 决定"在门口等多久才拍板"——
  判据 = 通过时无接触（30 分）且不超时；
  ⚠️ **2026-10-07：两段时长仍未标定**（本趟日志 THROUGH 从未触发）。按实测 `surge=0.30 → 17.3cm/s` 外推，
  2s 快冲 + 4s 慢冲合计只走 **1.01~1.25 m**，而 `z.cross=1.5` 起冲要走的几何距离是 **2.4~2.6 m**
  ⇒ 下水若冲不过去**优先加 `fast_ms`**（段② 慢冲比它差 4~5 倍）；
- **小角度过冲的收尾（2026-10-02 已随上位机闭环删除）**：原先 `turn_pid.deadzone_deg=6` 按目标缩放 /
  加最小舵效地板 `max(0.15,|out|)`（执行器死区 0.138 ⇒ 误差 <10.4° 时现在一点舵效都没有）/ 加大 `kd`；
- **SEARCH 平移扫视标定**：`gate.search.sway`(**0.45** ≈ 25% 推力档) 是否真的能把船左右挪动、
  以及一趟来回后**有没有净漂移**（开环、无横向位置反馈，见 cfg 注释里标为未验证的那条）；
- 高低门编排（`pass_target` 语义：每轮一门；板端台架目前设 2）；相机—机身偏置 `body_center_offset`
  标定后再决定是否接进对准。
