# 注释历史 · cfg/*.yaml —— 参数键的来历、实测数字、安全项（⚠️/★）


> 本文件由 `doc/注释历史.md`（4128 行）于 2026-10-05 按域拆出；**内容一字未改**，只搬了位置。
> 顶层索引与"旧路径怎么解析"见 [`doc/注释历史.md`](../注释历史.md)。
> 代码/配置里的注释只写"是什么 + 当前值"，注意事项、实测数字与历史沿革一律记在这里。

## cfg/vision.yaml


### image.chain
- 2026-09-27 · 引入 `chain` 键把两套识别方案（域）显式化：**D** = `resize640 → enhance → remap@640`
  （配阶段一 `g240_i16` 权重）；**A** = `remap@720p → resize640 → enhance`（配 AUV_4 权重 + clahe 0.5）。
  板端两个仓库分工：旧仓库 `/home/sunrise/AUV` 走 A 域、在用副本 `~/Desktop/AUV_New` 走 D 域
  → 见 `tools/README.md` §6、`tools/deploy/Adomain/sync_Adomain_repo.sh`。

### image.clahe_clip
- 2026-09-23 · 板端实测：关掉 CLAHE 对精度无影响、省约 8 ms/帧 → D 域取 0。
- 同轮 · A 域（AUV_4 权重）那代的参数一个都没改 ⇒ 它必须保持 0.5，**不是 0**。

### model.task_models.gate.path
- 2026-09-26 · 换代到**阶段一 `g240_i16`**（md5 `ef52c18b…`，3,938,787 B；量化 R4：head 末端 4 个卷积 int16）。
- 2026-09-26 · 上一代 `gate_kpt_bayese_640x640_nv12.bin`（09-24 采用版 `9ac61773…`）移入 `bak/weights/bakup_auv5/`；
  更早的 auv5（`945fdd01…`）本地已无副本（板端 `bak/backup_auv5_20260923/` 有一份）。
- 2026-09-26 · 换 bin 时**图像链路没动**（仍 D 链序 + clahe 0）⇒ 不需要改 `common/vision/preprocess.py`。

### gate.geometry.frame_w / frame_h
- 2026-09-22 · 卷尺实测外缘 **77×56 cm**（管外径 5 cm）；`ruler_calib.py --gate` 反演 `W_eff = 0.78~0.80`。
- 同轮 · 旧值 0.70/0.50 让 z 读小 9.1%（0.70/0.77），那时 `z.cross=0.7` 实际在真 0.77 m 触发；
  改成 0.77/0.56 后 `comm.gate.z.cross` 必须同步设 0.77 才是同一个物理触发点。
- 更早 · 注释里"水下用 fx=782.5 验证折射因子 k"那句**已作废**：现用水下标定 `cfg/front_camera.yaml`
  （fx≈1207.6，水陆比 k≈1.12）。

### gate.det.conf
- 2026-09-26 · 取自 `doc/设计/gate_pose_decode_spec.md` §6 的 `CONF`（= 0.6，阶段一权重的评测工作点）。
- 2026-10-01 · **现值已改成 0.5**（cfg 与代码兜底 `_D_DET` 同值，守卫用例钉住）：候选变多、
  后处理（可信角点/几何合法/去重）才是筛子；要退回评测工作点就把它写回 0.6。
- 同轮 · 决定**不复用** `model.score_threshold`：那个是 ball/gate 共用的，给 gate 调阈值不该顺手改撞球。

### gate.postproc
- 2026-09-26 · 三条常量与"不看包含关系"的判据全部来自规范 §4 的 PC 侧 339 帧人眼判定
  （用"包含"会误删落在近门框里的远处真门，实测误删 15 个）。
- 2026-09-27 · 补 `lr_norm`（左右归一）：板端 `log/kpt_s1_full.jsonl` 实测**四角可信的 175 个实例里
  86 个左右角点被标反**（约一半）⇒ 不归一就会被 `geom_check` 判"顺序不合法"整条丢掉，
  近门先被删、只剩远处那个门（现场现象："总是锁定后面的门"）。

### gate.select.mode
- 2026-09-26 · `near`（近距离优先，框宽当测距代理）由用户当天给定；`corner` 是 2026-09-18 的旧行为
  （当时理由是"水面倒影可能形成第二个门框，只按 score 选会选到倒影"）。
- 同轮 · 明确记下：`near` 用的是**代理**而非真测距（真做法是逐候选解一次 PnP 取最近，未实现）。

### gate.keypoint.conf_thr / vis_thr
- 2026-09-18 · 旧权重下的阈值实测（≥3 角可用帧占比）：0.95→23% / 0.8→87% / **0.7→94%** ⇒ 当时取 0.7，
  解码期 `vis_thr=0.5` 与它解耦。
- 2026-09-23 晚 · 板端现场把它改成 0.9 / 0.7（本地随后同步，两边同值）。
- 2026-09-26 · 随阶段一权重换代重设为 **0.8 / 0.5**：0.9/0.7 是给**上一代**权重现场试出来的，
  不是本模型的标定值；0.8 是规范 §6 的 `V_MIN`。

### gate.kpt_mem
- 2026-09-18 · 默认关（板端曾因写成 `flase`、被 `bool('flase')=True` 当成开着，实际一直在跑）。
- 同轮 · `recall_conf(0.45)` 必须 < `keypoint.conf_thr`，否则回忆点被当成真实可见 → 拿外推点解 PnP 必失败。

### gate.pnp.reproj_px
- 2026-09-18 · 实测 8 px 会拒掉约 **88%** 候选位姿（模型四角固有残差 ≈10~18 px；`kpt_v5` 候选 RMS
  p50=12.0~12.6 / p90=16.2~19.4），可用率 8px→12% / 12px→43% / 16px→79~90% / **20px→89~99%** ⇒ 放宽到 20，
  不取 25（25 会开始收下真错解）。
- 2026-09-23 · 反查过一次：板端 20 帧里 13/14 通过 `reproj_px=20`（RMS 1.4~7.6 px）
  ⇒ **它不是"总是 coarse"的瓶颈**，瓶颈在角点可得率/自洽性。

## cfg/comm.yaml


### motion.turn_pid（转向 PID，gate 与 .sh 共用）
- 2026-09-23 · 板端真实水域实测：命令 `yaw=0.45` 下遥测 yaw 逐帧 **-9.5°/0.11s ⇒ 86~106°/s**；
  命令 52.6° 却总共转了 ≈126°（驱动段 1.3 s 转 65° + 收舵后惯性又 61°）⇒ 门被甩出画面。
  结论 → `out_max` 0.45 → **0.30**（≈30°/s）、`deadzone_deg` 6 → 3。
- 2026-09-23 · gate 的 ALIGN.HDG 原先"每相机帧 step 一次"，实测 p50 **0.126 s ≈ 8 Hz**；
  同一套增益下"进带 6° → 连续 3 帧收舵"从 0.15 s 拖成 0.38 s ⇒ 过冲。
  结论 → 转向期间改用 `gate_task._turn_inner_loop` 按 `period=0.05`（20 Hz）阻塞推进。
- 2026-09-26 · `kd` 一度改成 0.25 试图刹车，实测出力**每帧正负交替**（0.3, -0.18, 0.3, -0.18…）⇒ 退回。
- 2026-09-27 · 仿真复现（`tools/check/boat/check_hdg_lockup.py`）：1.7 m / 25° 场景下 `kd=0.10`
  ⇒ D 项 = kd·Δerr/dt 达 0.4 > out_max ⇒ 原地 2 帧极限环，残余 7.5°、超时；
  `deadzone_deg=3` ⇒ 到达窗落在"一点力都没有"的区间（kp·dz/norm_deg = 0.04 < 死区 0.138）。
  结论 → `kd` 定为 **0.0**、`deadzone_deg` 定为 **6.0**。
- 2026-09-26 · 旧值沿革：`out_max` 曾 0.45、`kd` 曾 0.05；注释里的"探向 probe_s/probe_dof"已随
  探向整体删除（用户：不允许探向，符号由 `yaw_sign()` 直接算）。

### motion 速度档
- 2026-09-18 · `creep` / `lost_backward` 曾写 0.12 —— 落在**执行器死区 (0, 0.138)** 内 = 0 推力，
  等于那两个动作从未发生 ⇒ 全部提到 0.20。用例 `test_no_speed_preset_inside_actuator_deadzone` 守。
- 2026-09-18 · 下位机对映射值 ≤35 直接返回 0 ⇒ 实际推力远小于 DOF 数字（换算表在
  `doc/记录/过门-状态机与参数.md` §1.1）。

### telemetry.yaw_sign
- 2026-09-23 · 真实水域探向实测：命令 `yaw=-0.30`（左转档）→ 遥测 `Δyaw=+18.79°`
  ⇒ 探向只能测到「命令→物理」×「物理→遥测」的**积**（g·s = -1），把配置错误测成了硬件极性。
  结论 → 去掉探向，极性改由 `yaw_sign()` 直接算；本键只作**符号归一**的单一旋钮，默认 1.0 不改行为。

### depth_guard.min_depth_m
- 2026-09-18 · 本地曾长期写 0.3，**两次整份推板端把现场的 0.55 冲掉**
  ⇒ 现场定死 0.55（不准改，有不变量用例）；并立下规矩：**cfg 不再整份推板端**。

### gate.align（居中判据）
- 2026-09-26 · 板端 `log/gate_one_2026092.jsonl`（18.8 s / 162 帧）：居中达标那一帧（f=13 进 APPROACH）
  实测 **dx=-0.079 / dy=-0.167** —— 按 rectified `fx=1023.76/fy=1151.45` 换算（横 0.625·z、竖 0.313·z，z=2.16 m）：
  旧 `px_x=0.20` → 0.27 m 容差（门开口横向余量 0.16 m）、旧 `px_y=0.25` → 0.169 m（竖向余量只有 0.08 m）
  ⇒ **带着超限的偏就进 fast**。结论 → D 域收紧为 0.10 / 0.15（A 域保持老值）。
- 2026-09-26 · 同轮 `confirm_frames` 4 → 2：新模型下 `mode` 每帧在 full/coarse 间跳，
  实测 144 帧 ALIGN 里**一次都没攒到 4 连帧** ⇒ 正航向永远起不来（水波也难稳）。
- 2026-09-18 · 旧沿革：0.30/0.26 → 0.25/0.26 → 0.20/0.25；米制 `align.xy_m` 通道已删除。

### gate.z（穿门判据）
- 2026-09-26 · 板端 19.2 s 那趟：`mode` 每帧在 full47/coarse78/p3p35/width11 间跳，
  `hdg_state` 整趟只有 idle（HDG 从没启动）；f=146 靠**出口②**在 `z=0.868 ≤ near_lost_m(1.0)` 判「已过门」
  直接满速冲，当时 **psi=+30.6°、dy=0.42**、占比只有 0.498。
  结论（D 域）→ `near_lost_m` 1.0 → **0.6**、`near_lost_ratio` 0.60 → **0.75**、`cross` 0.77 → **0.65**。
- 同轮 · 那趟 `_z_last` 还带着 **5.295** 的垃圾值（width 档残留）⇒ 说明近距时米制阈值不可用，兜底才是主力。
- 2026-09-18 · `_z_last` 带过 **9.25** 的垃圾值一路进穿门判定 ⇒ 新增 `z_stale_ms=1500` 保鲜期。
- ⚠️ **（2026-10-07 用户明确：以下 09-22 那条是"废掉的提案"，不再作为约束）**
  曾经的提案：`z.cross` 必须 = `geometry.frame_w`（"同一个物理距离的两种写法"）。
  **已作废** —— 现行约束是 `z.cross` = `z.slow_max` = `width.z_max`（三键同值，YAML 锚点 `&door_m`）。
- 2026-09-22 ·（历史）`z.cross` 由 0.7 改成 0.77：门框标尺从 0.70/0.50 修正到 0.77/0.56 后，
  0.77 才是旧标尺下的同一个物理触发点。

### gate.through
- 2026-09-18 · 那轮冲刺是从门口（占比 0.97 ≈ 0.5 m）才起步，**900 ms / 0.5 DOF 没带出去**（9 帧后 DONE）
  ⇒ 时长语义 + 提速：`confirm_ms` 900 → **2500**、`surge.through` 0.5 → **0.6**；
  现场按 `1000·(起步距离+机身长)/v × 1.3` 估算为 **2~3.4 s**（本仓库现取 3000）。
- 2026-09-26 · 新增 `require_align_deg`（D 域 8.0）：那趟 19.2 s 日志里 psi=30.6° 就冲了
  ⇒ 三个出口全部加"冲刺前航向闸门"，被拦写 `through_block`。

### gate.search（平移扫视）
- 2026-09-20 · 用户定：过门过程中**不允许旋转搜索**，原「旋转脉冲」（`spin_s/yaw`）整体删除，
  改成左右平移扫视（波形必须对称，否则会一路漂到池壁）。
- 2026-09-23 · 复核 `coarse.far/near_ratio` 注释里的 "@fx782 ≈2.4m / 0.78m" 是**旧标注**；
  占比是像素比、与配置 fx 无关 ⇒ 该值不动。

## cfg/comm.yaml（瘦身搬出的注释原文）


### (文件头)
- comm.yaml — 通信与运动参数（串口帧 / DOF 映射 / 心跳急停 / 底层运动 / 任务参数）
- 分区：
- serial·frame·dof_map·heartbeat·estop·ramp   串口与 11B 帧
- depth_guard·dof_comp                        底层保护与下潜增益（ball/gate 都吃）
- motion                                      **ball/gate 共用的底层运动**（PID / 速度档 / 预设）
- tasks·ball·gate                             任务装配与**各任务特有**的旋钮
- 约定：**共用参数只写在 motion 里**，任务段不重复抄一份；任务段确实要单独一套时才写覆盖键
- （如 `gate.pid_sway`）。注释只写"是什么 + 当前值 + 单位/量纲 + 调它的后果"；
- 历史沿革与实测证据见 doc/注释历史.md，逐日改动见 doc/记录/2026-09-2X-改动记录-review.md。

### frame
- 11B 定长帧（模拟遥控）：帧头 + 7 轴 + 3 键；轴值 128 中位、±127 幅值

### frame.header
- 0xA5

### frame.aux_axis
- axis4=滚轮静止/中值, axis5/6=tri 中位（灵敏度三档 ×0.20/×0.24/×0.60）

### frame.btn_values
- 当前下位机固件未使用

### dof_map
- DOF→轴字节真值表（下位机 RC_Translate），sign 全为 1：

### dof_map.surge
- 前后←LY、左右平移←RY、升降←RX、旋转←LX

### heartbeat.interval_ms
- 心跳间隔(ms)，≥20Hz；下位机无帧超时停车 → 必须持续发

### estop.repeat_ms
- 急停连发中性帧间隔(ms)

### watchdog
- ★ **轴饱和看门狗**（base/hw/uart.py::_AxisWatchdog；用户 2026-09-27 定）

### watchdog.enable
- 独立线程（不经过任何任务控制流）：任务卡死时只有它能拦。

### watchdog.axes
- ★**只盯哪些轴**（轴名，取 dof_map 的键；空列表 = 不盯任何轴）。
- 用户 2026-09-27 定：**只管 yaw，剩下的不能管** ——
- · yaw 是闭环收敛量：正常转向越转误差越小，同向连续满舵只可能是
- 反馈卡住/舵效不对/在自转 ⇒ 该掐；
- · surge/sway/heave 可以合法长同向（冲刺几秒直行、扫视/横向对中持续平移、
- 保深持续垂向）⇒ 拿它们当"卡死"会**误杀正常动作**（长直行被掐 = 直接失败）。

### watchdog.max_same_dir_s
- 同一轴、**同一方向**连续发轴超过这么多秒 ⇒ estop 硬停 + 强制退出(9)
- （**2026-09-28 用户定 10.0s**：5.0s 会把"逐小步逼近"里的合法转向也掐掉 ——
- 每小步都可能长时间同向满舵（out_max=0.30 = 偏离中位 38 字节，
- 连续满舵时长 ≈ 本步转角/实际角速度）；小角度逐步逼近尤其容易踩到 5s。）

### watchdog.min_off_b
- 偏离中位超过这么多**字节**才算"在发轴"（≈0.10 DOF，接近执行器死区）

### watchdog.poll_ms
- 巡检周期(ms)
- ⚠️ 用 `AUV_WATCHDOG=0` 可临时关；SIM 模式默认不开（免得打断 pytest）。
- 代价：正常的**长**平移/长满舵（>max_same_dir_s）会被当成卡死掐掉 ——
- 若某个动作合法地要更久，就把这个值调大，别关掉整条保护。

### ramp.speed_per_s
- 轴字节平滑速率(字节/秒)：速度连续变化到目标，不是直接激增；0=关闭直跳

### telemetry
- 下位机上行姿态/深度的**符号归一**

### telemetry.yaw_sign
- 遥测 yaw × 它。默认 1.0 = 不改行为。
- 它就是"回传 yaw 的符号"这一个旋钮；转向环极性 σ 由代码直接算出
- （`common/motion/turn_deg.py::yaw_sign()` = 固件常量 × `dof_map.yaw.sign` × 本键）
- ⇒ 改这里，转向闭环极性自动跟着改，不会出现"配置与代码各记一套符号"。
- ⚠️ 别为了"让船转对方向"去翻它——方向不对先查 `dof_map.yaw.sign` 与接线。

### depth_guard
- 限深保护：只压上浮，不动前进/转向（base/hw/uart.py::_apply_depth_guard）

### depth_guard.enable
- 当前深度 ≤ min_depth_m → 本帧 heave 清零并当帧回中，机身顶不出水面

### depth_guard.min_depth_m
- ⚠️ **现场定死 = 0.55，不准改**（有不变量用例守着）；**别再整份覆盖板端 cfg**
- depth 正 = 水面**以下**；遥测打出负值是符号反了 → 去协议侧统一。

### depth_guard.stale_ms
- 遥测超过这么久(ms)没更新 → 视为没有深度（一直告警 = 下位机没回传）

### depth_guard.stale_action
- 无新鲜深度时：pass=放行(台架/仿真友好) | block_up=连盲上浮也不许

### depth_guard.snap
- 触发时 heave 轴当帧直接回中（不走 ramp），防惯性继续上浮

### depth_guard.log_ms
- 触发告警打印节流(ms；0=每帧都打)

### dof_comp
- **底层共用**：单独调"下潜"这一路动力（base/hw/uart.py::_apply_dive_boost）

### dof_comp.enable
- 只对 heave<0（下潜）生效；上浮/悬停/平移/转向一律原样

### dof_comp.dive_scale
- 下潜指令 × 它再夹到 [-1,0]；1.0=不放大。调法：弱→3.0/4.0，过头→1.5
- ⚠️ 下位机对很小的映射值直接返回 0；且很小的指令(如 -0.05)放大后
- 仍可能落在执行器死区(<0.138)内。

### motion
- **ball / gate 共用的底层运动参数**（同船同推进器；任务段不重复写）
- ---- 共用 PID（任务段只在"确实要单独一套"时覆盖）----

### motion.pid_sway
- 水平修正：ball 接近段 sway / gate 全档位居中（归一化像素偏差 ±1 输入）

### motion.pid_heave
- 升降居中：ball CENTER 的 heave / gate 的 heave

### motion.turn_pid.div_deg
- sat_max_s/sat_progress_deg：**满舵保护**（common/motion/turn_deg.py::TurnCore.step）——
- 命令饱和到 ±out_max 连续 sat_max_s 秒、且这段时间误差只改善 < sat_progress_deg
- ⇒ **立刻停转**并记 why=sat。只"satur 但不看进展"会误杀慢船的正常长饱和，所以两个都要。
- 实测：健康转向的饱和段通常 < 0.5s；板端 --deg 30 卡死时 2.0s 即触发。
- 0 = 关掉这条保护（超时保护仍在，但它要求每帧都被 step 到，现场卡死时不可靠）。

### motion.turn_pid.sat_max_s
- overshoot_tol_deg：**超转容差 = 10.0°**（用户 2026-09-28 定："逐步收敛，可以超转但超转不得超过
- 10 度"）——基准是**下发的 deg**，不是 ψ；**只放宽超转那一侧**：
- `0 ≤ done_deg − deg ≤ 10°` 即算这一小步到位，且在这窗里**不再触发满舵保护**；
- **超过 10° 不认**，交给下一个新鲜 ψ 带回来（典型轨迹：目标 32° ⇒ 10→20→30→38→32）。
- 为什么：满舵保护原来只看"|err| 有没有变小"，而超转会让 |err| 变大 ⇒ 正常的轻微超转
- 会被误判成"没进展/转反了"把转向掐死（用户："超了一点很正常，不应该这么被限死"）。
- 0 = 关掉这条（回到只看 err）。与 max_step_deg/turn_scale 配合：窗口按"下发的那一下"算。

### motion.turn_pid.overshoot_tol_deg
- period：转向闭环推进周期(s)，必须与 .sh 路径 `turn(period=0.05)` 一致
- （gate 的转向由 gate_task._turn_inner_loop 按它阻塞推进）。

### motion.turn_pid.period
- 「额定转角」PID（`common/motion/turn_deg.py`）：gate 的正航向与转向脚本**共用同一套用法** ——
- 调用形态就是 `turn_deg.py --deg N --dir left|right`（gate 里就是 `TurnCore(deg, left)`）。
- 闭环量 = 下位机回传的 yaw；极性 σ 由代码算出（`yaw_sign()` = 固件常量 × `dof_map.yaw.sign`
- × `telemetry.yaw_sign`）：**不需要配置、不探向、不现场测**。
- kp 管"多早开始收力/收敛更紧"，out_max 管"最大转速"（想整体更慢就降它，别只降 kp）。
- ⚠️ kd 必须为 0：内层 20Hz + 误差按 norm_deg 归一 ⇒ D 项在 30°/s 转速下就达 kd×4，
- kd=0.10 时 D 项(0.4)已超过 out_max ⇒ 出力每帧正负反转、原地 2 帧极限环、永远进不了到达窗。
- ⚠️ deadzone_deg 必须 ≥ 执行器死区折算角：kp·dz/norm_deg ≥ 0.138 ⇒ 当前增益下 dz ≥ 10.4°
- 才有推力；dz=3 等于把到达窗设在"一点力都没有"的区间里。
- ⚠️ out_max 别降到 0.15 那一档：15% 推力恰卡在执行器死区 0.138 边上。
- deadzone_deg：PID 死区(°) 兼「到达判据」；norm_deg：误差归一化分母(°，调小=更凶)。
- div_deg：方向自证 —— 命令一直朝目标方向、误差却和初始同号且比初始还大这么多度 ⇒ 停转（0=关）。
- ---- 共用速度档（归一化 DOF）----

### motion.surge_fast
- 快档：ball 分级最高速/冲刺、gate 远距进近（z > z.slow_max）

### motion.surge_slow
- 慢档：ball 接近段减速、gate 近距进近

### motion.loss_inertia_surge
- 目标短时丢失时的低速惯性（ball 接近段；gate 不用）
- ⚠️ **任何速度档都不能落在执行器死区 (0, 0.138)**：低于它一点推力都没有。

### motion.search_yaw
- 无目标时原地旋转搜索（30% 转速，右转）——**过门任务不用**（gate 是平移扫视）

### motion.presets
- 手动模式/急停用的 DOF 预设（uart.set_motion）

### motion.presets.search
- 右转；与 motion.search_yaw 一致

### tasks.enabled
- 任务一；任务二=记忆返回(task1-2/*.sh)；gate 需后端/mock

### tasks.fault_action
- 模型缺失等：skip | exit

### ball
- 任务一 撞球：运动链 SEARCH→CENTER→APPROACH→DASH→STOP
- ---- 视觉量 → 运动决策 ----

### ball.ema_alpha
- 面积占比 EMA 平滑(判据与日志都用它, 抗单帧抖动)

### ball.lost_grace_frames
- 丢目标后的"短时保持/惯性"帧数

### ball.r_slow
- 面积分级: >=该值且增长缓慢 → 用 slow; 其余用 fast

### ball.growth_eps
- 面积增长阈值: 增长快 → 用 fast(说明在快速接近)

### ball.align_x
- 居中通道：x → yaw（旋转；撞球不做左右平移）、y → heave

### ball.timeout_ms
- 总时限 30s(DASH/STOP 期间不打断, 保证命中与停稳)
- ---- SEARCH: 脉冲旋转 + 周期性前进探测（沿用原方案）----

### ball.search_spin_s
- 旋转 0.5s → 停 1.5s 循环；连续无目标 4s → 慢速前进探测一次

### ball.search_pause_s
- （探测时长 0.5s = 阶段窗口, 不按总时限等比缩; 速度 0.25）

### ball.search_advance_surge
- ---- CENTER: yaw + heave, 不前进不横移（连续居中 5 帧 → APPROACH）----

### ball.center_eps
- 水平+竖直居中判据(归一化偏差)

### ball.edge_yaw_kp
- 居中转向 P/D 增益与限幅（D 是阻尼, 防转过头）——**撞球特有**：过门居中只用 sway

### ball.edge_yaw_max
- ---- APPROACH: 仅 sway 修水平(yaw=0/heave=0) ----
- 前进速度与 sway PID 都取 motion.surge_fast/surge_slow、motion.pid_sway（共用，见上）
- ---- DASH/STOP: 面积(EMA)达标且连续 2 帧 → 冲刺 1.5s → 全 0 保持 1.0s；丢目标 hold 0.8s 回 SEARCH ----

### ball.dash_ratio
- 面积(EMA)达标阈值

### ball.dash_dur_s
- 冲刺时长（不检测是否撞到）

### gate
- 过门（相位机 SEARCH→ALIGN→APPROACH→THROUGH）

### gate.timeout_ms
- 任务级兜底（09-18 曾提到 300000，现用 180000）

### gate.pass_target
- 完成计数目标：单门 1；高低两门可设 2
- ---- 修正 PID：**默认共用 motion.pid_sway / motion.pid_heave** ----
- 这里**不写任何增益数字**：gate_task 直接取共用那两套（启动日志打印实际来源与数值）。
- 确实要单独一套时才写 pid_sway / pid_heave（同键名同量纲：归一化偏差 ±1）。转向不在这条链上（见 hdg）。

### gate.align
- 现用值（09-26 曾收到 2 / 0.10 / 0.15）
- 「居中成功」判据：**全档位一套像素门槛**（归一化像素偏差，±1=半屏；位姿档同用这套）。
- px_x/px_y：判"对准"的范围，越大越松（越容易进 APPROACH/creep）。
- ⚠️ **竖向不能再紧**：heave 是 `pid_heave.kp=1.0`，而执行器死区 0.138 ⇒ |dy|>0.138 才有推力；
- px_y 设到 ≤0.138 就等于要求船靠惯性滑进窗口 ⇒ 可能永远确认不了居中（卡在 ALIGN）。
- 想让竖向更准，要动 `pid_heave`（最小舵效地板）或 heave 的 dof_comp，**不是 px_y**。
- ⚠️ rectified 主点 cx=702.27/cy=409.14 ≠ 画面中心 640/360 ⇒ "正对光轴"的门读数就是
- dx=+0.097 / dy=+0.137（对准目标是**画面中心**，不是光轴；属 `body_center_offset` 未标定范畴）。
- ⚠️ APPROACH 阶段仍在修 sway/heave（只把 surge 加大）⇒ 容差随距离变大不会失控。
- ⚠️ 代码兜底 `gate_task._D_ALIGN` 必须与这里同值（有用例 `test_gate_defaults_match_cfg` 守）。
- confirm_frames：连续对准帧数（防抖；**别拿降帧数当"放宽"**）。
- 居中**只有 sway 平移、没有 yaw**（2026-09-20 用户定）：姿态修正由下面的 hdg.* 负责，那是 gate 唯一的 yaw 通道。

### gate.hdg.turn_timeout_s
- 「转完反向平移」（gate_task 的 SWAY_BACK 状态）—— 以下 5 个键现场可调：

### gate.hdg.post_sway_ms
- ★ 转完（硬停完成后）**边反向平移边缓慢后退**，直到「画面里至少出现 post_sway_kpt_min 个角点」：
- post_sway_back = 后退幅度（正值，代码取负；0 = 不后退）；post_sway_kpt_min = 角点门限（可设）。

### gate.hdg.post_sway_far_ratio
- 「正航向」（ALIGN 的子状态 HDG；实现 `gate/motion/hdg.py`）：让机身与门法向平行。
- ★ 2026-09-27 起是**一次到位**：起转那一刻把 full 帧的 psi（`_hdg_deg`，EMA）**冻结**成目标角
- → `common/motion/turn_deg.py` 按遥测 yaw 闭环转**一次** → 结束。
- 旧的「测→转→停稳→再测→迭代」、**开环盲转**（blind_enable/blind_rate_dps）、
- **开环探向**（probe_*）全部删除：转向只允许闭环；只有 full 帧的 psi 可用（p3p 欠定）。
- tol_deg：|psi| ≤ 它 ⇒ 不转（已经够正）。
- max_step_deg：单次转角上限（**2026-09-28 定 15.0°**；0 = 不设限）。只防"垃圾 psi 把船甩出去"。
- ⚠️ **没有「每门只转一次」的限制**（用户 2026-09-28 定：那条限制不合理）——改为**逐小步逼近**：
- 一次最多转 max_step_deg(15°)，转完**重新测 ψ**；只要来了**新的 ψ 测量**且残余仍 > tol_deg，
- 就再转一小步，直到够正。每小步之间必须有新 ψ（否则会照旧值连转、又转过头）；
- 兜底由 阶段超时(hdg.timeout_ms) + 方向自证(div_deg) + 满舵保护(sat_*) 负责。
- turn_scale：**下发角缩放**（2026-09-28 定 0.8）——实测转过 40.9° 而目标是 36.6°（偏大）⇒
- 先按 0.8 缩一档试。顺序：|ψ| × turn_scale，再用 max_step_deg 钳位。1.0 = 不缩。
- timeout_ms：正航向总时限；turn_timeout_s：单次转向的闭环超时（超时=带残余航向继续走，不卡死）。
- entry_psi_first：**航向优先**（居中还没达标也起转）——解「航向歪 ⇒ 门偏一侧 ⇒ 居中确认不了
- ⇒ 不许转 yaw」的死锁。默认 **false**（必须先居中）。
- psi_first_z_max / psi_first_ratio：航向优先的距离门（新鲜 z ≤ 它 **或** 框占比 ≥ 它）。
- ⚠️ 转向期间**整条视觉链路被绕开**，档位退化/位姿被拒/角点闪烁都不打断；只有**整门丢失**才中止。
- 一键回退：enable: false（本门不正航向，带残余航向进近；冲刺仍有 through.require_align_deg 那道闸）。
- post_sway_ms/post_sway：**转完反向平移**的窗口(ms)与幅值(DOF)。一次转向会把画面整体甩走
- （转 40° 就超出 32° 半视场），门被甩出画面后视觉纠不了 ⇒ 在窗口内朝**转向的反方向**
- 平移一小段拉回视野；同门一回来立刻交回视觉，到期也一定退出（绝不永久停摆）。
- post_sway 必须 > 执行器死区 0.138（固件 RC.c：RC_Matching 阈值 35/255）。0 = 关掉这个补偿。
- ⚠️ 窗口**从"转向收尾硬停结束"那一刻起算**：硬停是阻塞的（实测 0.70/1.31/2.51s，
- 见 base/hw/uart.py::stop_hard）——旧写法在硬停前开窗，600ms 被硬停整段吃光 =
- 生产里只发得出 1 帧 ≈ 没有平移（用例 test_post_sway_starts_after_the_turn_end_hard_stop 守着）。
- post_sway_far_ratio/post_sway_same_z_m：窗口内判"这算不算刚丢的那个门"的两道闸 —— z 优先
- （|Δz| ≤ same_z_m；门间距实测≈1.6m ⇒ 取 1/3），z 拿不到才退框占比（≥ far_ratio × 原占比）。
- 两个都设 0 = 判据关掉（任何检出都算门回来了，= 回退前那一版语义）。
- stop_hard：转向收尾**硬停**（阻塞：连发中性帧走完 ramp + 遥测 yaw 验证）。**别关**——
- 下位机没有"无帧超时停车"，只发一帧中性时 yaw 轴还在 ramp 半路，一断流/关串口船就锁住一直转。

### gate.loiter
- 「在门口超时兜底」= 直冲出口③：**不看档位**，只要 ①框占比 ≥ loiter.near_ratio
- ②在安全带内（|dx|≤dx_max、|dy|≤dy_max）③连续持续 ≥ timeout_ms → 自己拍板：判过门 + 直冲。
- 调参：等太久 → 减小 timeout_ms；误判太多（没对准就冲）→ 收紧 dx_max/dy_max。不要 → enable: false。

### gate.z
- 2026-09-28：0.7→0.9（**早点触发冲刺**）
- ★ 现用 cross=1.0（09-28 由 0.7 调上来、早点触发冲刺）；下面那组 09-26 说明对应收紧后的 0.65：
- 本仓库的老阈值与老内参（fx 782.5）是**配套**的同一套尺度。

### gate.z.cross_confirm_frames
- 连续 z≤cross 帧数才判过门(防单帧 PnP 错解冲出去)。**别降到 1**

### gate.z.near_lost_m
- ★ 现用 1.2（09-26 曾收紧到 0.6；更早 1.0）

### gate.z.near_lost_ratio
- ★ 现用 0.50（09-26 曾 0.75，更早 0.60）

### gate.z.z_stale_ms
- `_z_last` 的"保鲜期"(ms)：超过它未更新 → 不许参与穿门判定
- ⚠️ near_lost_ratio 调小 = 更早判过门（更冒险）；coarse 档**没有测距**，这一条是那一档唯一可靠的近距信号。
- ⚠️ z_stale_ms 防"陈旧的 z"：coarse 档会一直带着上一次的 z；陈旧的小 z 会把远处丢门误判成
- "已过门" → 满速盲冲。
- ⚠️ align_max / fast_max 当前**未参与运算**（Z 只分 slow_max 两档；留作三档扩展）。

### gate.surge
- 进近两档 fast/slow **取自 motion.surge_fast/surge_slow**（共用，不在这里重复）；
- 要 gate 单独一套就在本段写 fast/slow 覆盖（gate_task 优先读本段）。
- creep=coarse/width 慢蠕进；lost_backward=远处短暂丢失的轻微后退(代码取负)；reacquire=后退重取。
- through=穿门冲刺（出口①②③共用，`_start_through` 固定取它）。
- ⚠️ **任何速度档都不能落在执行器死区 (0, 0.138)**。
- ⚠️ through 是全场最高速、也是"撞门能量"最直接的旋钮（撞门 10 分 / 不撞 30 分）；
- 再加就要同时确认 THROUGH 时长够"冲出去"。

### gate.width
- width 档（只对向 2 角）：慢 creep 靠近 ／ HOLD（无直冲出口）

### gate.width.z_max
- 粗测距 z ≤ 它 = 够近了，此时只允许 creep 靠近
- 注：判"该不该判过门"用 z.near_lost_ratio（框占比），不是这里

### gate.coarse
- 没有位姿时的走位仲裁：far/near_ratio=门框宽/屏宽 远近判定；⚠️ 占比是像素比、与配置 fx 无关。
- align_x/y=对中阈值(归一化像素)，决定"creep 靠近"还是"HOLD → REACQUIRE 后退"；
- coarse 的框心是最粗的估计 ⇒ 刻意比 align.px_* 略松（有不变量用例守这个形状）。
- 仲裁要点：**只有未对准才 HOLD/后退**；对准就 creep 靠近。

### gate.hold
- HOLD 无进展帧数 → REACQUIRE(只在"未对准"时累加)

### gate.reacquire
- 单次后退上限(ms)

### gate.reacquire.max_times
- 连续这么多次仍拿不到角点 → 放弃后退, 改原地 HOLD

### gate.reacquire.stop_ratio
- 闭环后退: 框占比退到进入时的 75% 就停

### gate.reacquire.reset_after_ms
- 距上次后退超过这么久 → 重新允许 max_times 次后退

### gate.through
- ★ require_align_deg：**冲刺前的航向闸门** —— 最近一次测到的 psi
- （`_hdg_deg`）必须 ≤ 它才允许进 THROUGH。三个出口（z≤cross /
- 近距丢门 / 门口兜底）**都必须过这道闸**；被拦时写日志字段
- `through_block` 并留在原相位继续对准。≤0 = 关闸；从没测到过 psi 时放行。
- **冲刺时长**(ms) → 判过门（**按时间**，帧数语义下时长随 fps 漂）。
- confirm_ms<=0 → 退回旧的帧数语义（用 confirm_frames=8）。
- ⚠️ **下次下水按实测航速回填**：confirm_ms ≈ 1000·(起步距离+机身长)/v × 1.3。

### gate.search
- SEARCH = **左右平移扫视**（过门过程中不允许旋转搜索）。波形：向右 sweep_s → 停 pause_s → 向左同长 → 停。
- ⚠️ **必须对称**（正反等时长）：遥测没有横向位置反馈 → 单向平移会一路漂到池壁；等时长 ⇒ 净位移≈0。
- ⚠️ 这是**开环**：抵消依赖左右推力对称，水流会带来缓慢漂移（**未验证**）；
- 总时长由 `gate.timeout_ms` 兜底 —— 若现场发现一直往一侧跑，先降 `sway`。
- sway：平移速度(|DOF|)，**不经过 PID**。`sweep_s<=0` → 只停不扫（原地待机）。
- **SEARCH 不前进**（保留"看不见就别动"的保守性）。
- ★ 2026-09-28：0.45 → **0.6**（用户：现场反馈平移力度偏小）。档位参考：
- `motion.pid_sway.out_max = 0.45`（居中修正上限）、执行器死区 ≈0.138
- ⇒ 0.6 明确越过死区、且比居中修正更有力；再往上逼近满舵，单侧漂移也累积更快。
- 改这一行**必须同步** `gate/motion/params.py::_D_SEARCH`（`gate/gate_task.py` 再导出）（守卫用例 test_gate_defaults_match_cfg 挡的就是这种漂移）。

### gate.pose_hold_frames
- 帧间防抖：丢姿态时的保持帧数

### debug
- 通信调试

### debug.tx_frame_hex
- 终端打印每次发出的 11B 运动帧（真串口下）

### debug.tx_frame_log_ms
- 打印/遥测节流(毫秒，200=5Hz；0=每帧都打)：帧打印 / 下位机深度姿态打印

---

## cfg/vision.yaml（瘦身搬出的注释原文）


### (文件头)
- =============================================================
- vision.yaml — 纯视觉参数（相机 / 图像补偿 / 模型推理 / 仿真 / 任务视觉目标）
- 仅供本工程识别推理链路使用；训练数据集制作在 PC 工程（RDKX5-YOLOv11n-）
- 历史沿革与实测证据见 doc/注释历史.md
- =============================================================

### camera
- 真机相机打开失败时自动退化到该相机的 sim

### camera.front
- 前视：撞球/过门使用

### camera.front.type
- sim | usb

### camera.front.device
- usb: 0 或 /dev/videoN

### camera.front.calibration
- 去畸变标定 yaml（null=跳过）

### camera.down
- 下视(IMX415 MIPI)：保留（录素材/未来任务用）；**当前策略不使用下视**

### camera.down.type
- sim | mipi

### stream
- 前视画面推流（零转码 MJPEG；推流端 manual/stream.py + base/hw/camera.py）。
- 语义：**谁打开相机谁顺带推流**（UVC 只允许单进程取流）→ 开录像/识别都能同时推、不争相机。
- 方案与实测见 doc/记录/前视USB相机低延迟推流方案.md；临时开：AUV_STREAM=1（env 优先于本开关）。

### stream.enable
- 默认关（调任务时一般在板端直接看画面）

### stream.host
- 水面 PC 的 IP（接收脚本/ffplay 跑在这台机器上）

### stream.port
- UDP 端口（PC: ffplay -f mjpeg "udp://@:5000"）

### stream.stream_fps
- 推流帧率上限

### stream.pkt
- UDP 切片大小（1316 更保守，8000 更省 CPU）

### image
- 识别前图像链路（推理统一入口）
- ★ chain：**识别方案（域）**，决定 remap 与 resize/enhance 的先后，**必须与门权重同域**：
- D = 新方案（resize640 → enhance → remap@640）↔ 阶段一 `g240_i16` 权重（本仓库默认）
- A = 原方案（remap@720p → resize640 → enhance）↔ AUV_4 权重（配 clahe_clip: 0.5）
- ⚠️ 跨域会掉点（A 权重喂 D 链的图像，角点会飘）。板端两仓库分工见 tools/README.md §6。

### image.undistort
- true: cv2.remap 去畸变（需对应 camera.*.calibration）

### image.white_balance_bgr
- 白平衡通道增益（B,G,R）

### image.clahe_clip
- 关掉可省 ~8 ms/帧；⚠️ A 域（chain: A / AUV_4）**必须 0.5**

### image.gamma
- gamma LUT（<1 提亮，1.0=关闭）

### color.line_red
- 下视红色标示线（任务二）

### color.line_red.rgb
- 无 cv2 时的 RGB 近似

### color.line_red.hsv
- cv2 双区间

### model.mode
- mock | hbm_runtime | onnx（mock=虚拟测试）

### model.format
- RDK X5 模型必须是 X5 OE 工具链(hb_mapper makertbin)产出的 .bin
- ⚠️ .hbm 是车载 J5/J6/S100(Nash) 格式，X5(Bayes-e) 无法加载
- ===== 双权重（ball 与 gate 分离，各自独立）=====

### model.strict_x5_bin
- true: 拒绝 .hbm 后缀并提示正确编译链路

### model.input_size
- 模型方形输入

### model.input_name
- hbm_runtime 输入名（板端报错时填模型实际名）

### model.fast_nv12
- true: 用 cv2 快速 NV12(快~30x, 标准BT.601色度)；false: 原 numpy 版

### model.score_threshold
- ball/gate **共用**；gate 的候选阈值另有 gate.det.conf

### model.labels
- 类别顺序必须与训练/导出一致（决定解码类别）
- 权重② gate keypoint：ball 走 model.path、gate 走 task_models.gate，互不影响。
- 解码契约与后处理约束见 doc/设计/gate_pose_decode_spec.md；上一代权重在 bak/ 里。

### model.task_models.gate.kpt_order
- 门 keypoint 权重(4角点)

### mission.target_color
- 任务目标球颜色：red | blue

### sim
- 虚拟相机（前视/下视各自仿真）

### sim.front
- 前视虚拟：噪声底（目标识别走 mock 检测）

### sim.ball
- mock 检测器出球序列

### debug
- 运行监测（仅本工程调试用）

### debug.show_video
- 弹窗显示带识别框的相机画面（无 DISPLAY/cv2 时自动跳过）

### debug.overlay_text
- 画面叠加 状态/任务信息

### gate
- 过门（几何 / keypoint / PnP / 后处理参数）

### gate.geometry.sym_bars
- frame_w/frame_h：门框**外轮廓**尺寸(m)，PnP 深度的**唯一标尺**。
- ⚠️ 与 comm.gate.z.cross 是同一个物理距离的两种写法，改一个必须改另一个。
- body_center_offset：相机—机身安装偏置(m)，**只读入、不参与运算**（未标定）。
- bar_width/sym_bars：未参与运算（keypoint 只标外轮廓四角，留作记录）。

### gate.det
- 候选框 score 阈值，**只作用于 gate**；`preview_detect.py --conf` 会覆盖这里。
- ⚠️ 刻意不复用 model.score_threshold（那是 ball/gate 共用的）。

### gate.postproc
- 解码后处理（规范 §4；实现 gate/percept/gate_postproc.py，调用点 gate_decode.detect）：
- edge_tol  = 重复框判据里"边重合"的容忍 = 该比例 × 小框短边；
- min_edges = 至少几条边重合才算同一个门 → 丢**小框**（且要求小框 conf 更低）。
- ⚠️ 判据**不是包含关系**：远处第二个门可能落在近门框内，用包含会误删真门。
- geom_check = 四角实例的"顺序合法 + 不自交"过滤（挡退化实例）。
- lr_norm = 左右归一：两侧都可见时 L 列必须在 R 列左边，否则交换 TL↔TR/BL↔BR。
- ⚠️ 关掉它 = 回到旧行为（模型会把约一半实例的左右角点标反）。

### gate.select
- 选门策略：near = 近距离优先（框宽当测距代理 z ≈ fx·W/w，单调；并列比角点数/置信和/score）
- corner = 旧行为（角点更全更可信优先，防水面倒影形成的"第二个门框"抢门）。
- ⚠️ near 是**代理**不是真测距，斜门/倒影下需实测复核。回退：改成 corner。

### gate.keypoint
- conf_thr：**任务侧**可信下限（= doc/设计/gate_pose_decode_spec.md 的 V_MIN）——判 mode/选门/计数。
- 4 个角点都 ≥ 它才算 full；⚠️ psi 只吃 full 帧 ⇒ 直接决定正航向能否启动。
- ⚠️ 别改回 0.5（鬼点/倒影会被当真角点）。
- vis_thr ：**解码期**硬门限：低于它的角点 conf 在解码阶段被清 0，下游看不到。
- 刻意与 conf_thr 解耦且更低（0.5 < 0.8）：低 v 角点坐标不可信、不许进 PnP，
- 但要留在画面/日志里才能诊断"门为什么不全"。null = 沿用 conf_thr。
- ⚠️ 代码兜底 gate_task._D_KPT 必须与 conf_thr 同值（用例 test_gate_defaults_match_cfg 守）。

### gate.kpt_mem.max_missing_frames
- 门角点"逐点软融合"——**可选功能，默认关**（enable: false → 角点单帧直用）。
- 开关优先级：AUV_GATE_KPT_MEM > 这里 enable > 兜底 false
- ⚠️ 它是字符串语义解析（cfgnode.flag）；写成 `flase` 会被当成开。
- ⚠️ recall_conf(0.45) **必须 < keypoint.conf_thr**，否则"回忆点"会被判成
- 真实可见 → 拿外推点解 PnP 必失败。
- 算法与调参见 doc/记录/算法说明-gate-角点逐点融合滤波.md。

### gate.pnp.max_z_jump_m
- reproj_px：候选位姿重投影 RMS 上限(px)；⚠️ 收紧会丢位姿（位姿被拒 → 退化 coarse → 居中收敛不了）。
- max_z_jump_m：帧间 z 突变超过它视为错解弃帧。**保命参数，不要放宽**。

## cfg/front_camera.yaml（标定 yaml 注释搬出）


### (文件级)
- ============================================================================
- WATER-side calibration (dome on, calibrated underwater, GOOD water quality).
- The boat uses THIS file. Air / on-shore work: point vision.camera.front.calibration
- at cfg/front_camera_air.yaml instead, and switch it back before launching.
- Source : chessboard 11x8 inner corners, 20 mm -- captured underwater in the
- AUV_5 session: data/AUV_5/raw-data/calibration-{1,2,3} (1443 images).
- Fit    : scripts/1_prepare/calibrate_camera.py --views 150 --min-shift 40
- 83 views kept, RMS reprojection error 0.546 px (previous file: 0.705).
- Why this replaces the ~782.5 px file (archived in backup/):
- * 782.5 was fitted from data/AUV_1/board alone and is under-constrained: it
- fails on held-out views of its OWN set, and badly on the AUV_5 set
- (checkerboard grid-straightness residual p90 307 px, max 873 px,
- 2 frames with no solution at all).
- * this file: grid-straightness median 0.39 / p90 0.49 / max 1.27 px;
- invertible raw radius 749 px vs image corner 798 px (previous 520 vs 753);
- pixels that have an undistortion solution 93.7% (previous 63.5%).
- * it also agrees much better with the board's independent tape+PnP air
- estimate (~1078 px) than 782.5 ever did.
- KNOWN OPEN POINT: this fit reads fx ~1208 px against the air tape estimate
- ~1078 px (+12%). Both are far from 782.5. A water-side tape / PnP-depth
- cross-check against ground truth would settle the remaining 12%.
- See configs/README.md for the full comparison.
- FILE-HEAD RULE (verified on the board 2026-09-22, cv2 4.11): the first byte of a
- !!opencv-matrix yaml MUST be "%YAML:1.0". A leading comment line -- or even a
- leading blank line -- makes cv2.FileStorage throw "Input file is invalid", which
- crashes camera init (calibration_maps does not fall back). Comments are fine
- AFTER the "---" marker, as here. Keep this header ASCII-only too, on principle.
- ============================================================================

---

## cfg/front_camera_air.yaml（标定 yaml 注释搬出）


### (文件级)
- ============================================================================
- AIR / on-shore calibration (dome on, in air). Point
- vision.camera.front.calibration here for bench work and switch BACK to
- cfg/front_camera.yaml before the boat goes into the water.
- fx was measured three independent ways on 2026-09-22:
- tape tick-period method (tools/analyze/calib/tape_ticks.py)      1083 +- 7 px
- tape target + gate A1 joint fit (ruler_calib.py --gate)    1075 px
- "+-50 cm gate just leaves the frame" boundary requires    >= 1095 px
- Re-running the hand-labelled A1 dumps with these intrinsics gives depth
- 0.95 / 1.51 / 1.98 / 2.59 m against truth 1.00 / 1.50 / 2.00 / 2.50 m
- (errors within 6%), and the 1.00 m dump -- which could not be solved at all
- with the old 782.5 intrinsics -- now yields poses.
- distortion_coefficients are 0 on purpose: the k1=-0.49 fitted underwater does
- not hold in air, so on shore we work in the raw domain (identity remap).
- This is not a real calibration: it is an effective-intrinsics file derived from
- two physical measurements plus one depth cross-check. A proper chessboard
- calibration with the dome in air would supersede it.
- FILE-HEAD RULE: the first byte must be "%YAML:1.0" (see cfg/front_camera.yaml).
- ============================================================================

## cfg/comm.yaml（补充：2026-10-05 瘦身搬出的注释原文）


### motion.search_yaw
- 轨迹：0 → 左 span → 右 2span → 左 2span …（在 ±span 内往复）
- 实现：**手动 DOF + 遥测 yaw 闭环**（common/motion/search_scan.py），非阻塞"睁眼看"。
- ⚠️ pid.out_max 别取 0.15：恰好卡执行器死区 0.138，"还剩二十几度就没推力"；
- 可用区间 0.30~0.5，想更慢就降 out_max（别低于 0.30）。

### gate.hdg
- 单帧落进阈值就锁存，会被一个噪声样本钉死（实测 ψ 在 1.5m 处标准差 22°，
- 而船真实 yaw 标准差只 3°；最长一段 |ψ|≤8° 只有 4 帧）→ 之后该转的不转。

### gate.hdg.post_sway_back
- turn_period=等"下位机完成反馈"时的轮询节拍(s)：上位机自己的循环 pace，与下位机无关

### gate.hdg.post_sway_back.turn_period
- max_turns：本门转向上限（到顶就不再起转、带残余航向进近；防「转向无效时自转」）

### gate.z
- （跳变保护第一条，**无条件生效**、不看有无丢门参照）。远处 PnP 的 z 与 yaw
- 都不可信，宁可当"门丢了"原地 hold，也不让远处能测距的门骗过当前门。

### gate.z.dist_max_m
- ★ 2026-10-05 用户定：**加回来**；z 超过它 ⇒ 完全不相信该帧位姿（无条件）  # 2026-09-28：0.7→0.9；2026-09-30：→1.2（与 near_lost_m 同值）

### gate.z.cross_confirm_frames

### gate.z.relock_far_ratio
- z 拿不到时用 框占比 ≥ 0.7×丢门前占比 判"还是同一个门"。

### gate.z.relock_z_jump_m
- 比较对象 = 当前在追的 z（_z_last）或丢门前的参照；命中 ⇒ 拒收该帧位姿、原地 hold。

### gate.z.relock_away_m
- ★ 丢门重锁时 z 超过它 ⇒ 判为「下一个门」（当前门丢了/测不到距，不能被远处能测距的门骗过去）。只对有丢门参照生效，首见不拦。

### gate.through
- 连续这么多帧才许冲刺。兜底出口(超时)不受此限。
- ⚠️ center_x/center_y **比 align.px_x/px_y 宽**（用户 2026-10-05 定：
- 0.20/0.25 是**原方案值**：align 现在收到 0.15/0.20 管"对中动作精细度"，
- 而"够不够正才敢冲"因运动抖动可以松一点 —— 两件事，别共用一个阈值。

### gate.search
- 连续 miss_frames 帧跟丢才解锁重选。防止每帧重选 → 转向目标角在几扇门之间乱跳
- （"误转后边那扇门的偏移角"）。
- match_ratio：锁定门的框中心 vs 本帧候选框中心，距离 ≤ match_ratio × 画面宽度 才算同一扇。
- stable_frames：**锁定稳定**判据（2026-10-04 用户定："锁定稳定后再不参与选门"）。
- 连续跟住同一扇这么多帧才算稳定；稳定前仍正常选门（选错要能纠正），
- 稳定后别的门再也抢不走它（哪怕它这帧没 z、没角点，只要还被检测到就继续追）。

### gate.lock
- 问题：远处门的 PnP z 会崩到比实际小，甚至小于更近那扇门 → 选错门。
- 实测 k=z×框占比 在同一扇门/同一档里是常数（离散 4~7%），说明**框占比是可靠的相对判据**。
- 做法：候选门的有效距离取 z_eff = max(z_pnp, ratio_z_scale × 框宽代理z)，
- 再按 z_eff 最小选门 —— 即"z 与框占比矛盾时，以更远（更保守）的那个为准"。
- ratio_z_scale：框宽代理的缩放（1.0=直接用；调大=更多信框占比）。
- ★ k 一致性检验（2026-10-04 用户定）：k = z×框占比，同一扇门 k≈常数，
- 理论值 k_true = fx·frame_w/画面宽 = 0.560（门框 70cm）。
- full 实测 0.599(k/k_true=1.07)、p3p 0.457(0.82) 都可信；
- 而"远处门 z 崩小"那次 k=0.23(0.41) —— 一眼可辨。
- k_lo_ratio：k/k_true 低于它就判"这个 z 崩了"，改用框占比反推的 z（k_true/占比）。
