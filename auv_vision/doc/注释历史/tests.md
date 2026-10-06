# 注释历史 · tests/** —— 用例里"为什么这么钉"的说明

> **2026-10-06：`tests/tasks/grab/` 与 `tests/tasks/place/` 已合并为 `tests/tasks/handling/`**
> （代码侧 `grab/`+`place/` → `handling/`）。下面历史节里的旧路径是**当时的写法**，按此对应关系解析。


> 本文件由 `doc/注释历史.md`（4128 行）于 2026-10-05 按域拆出；**内容一字未改**，只搬了位置。
> 顶层索引与"旧路径怎么解析"见 [`doc/注释历史.md`](../注释历史.md)。
> 代码/配置里的注释只写"是什么 + 当前值"，注意事项、实测数字与历史沿革一律记在这里。

## tests/platform/test_base.py

### 限深下限数值守卫
- 2026-09-18 用户定 · 本地曾长期写 `0.3`，两次"整份推板端"把它冲掉 → 船可能浮出水面直接结束
  比赛 · 结论 → `comm.depth_guard.min_depth_m == 0.55` 是"现场定死"的唯一例外，不准改；
  行为另有用例按 cfg 相对判定，这一条专门钉数值
### 硬停（`stop_hard` / `close`）
- 2026-09-18 用户水里实测 · 转角脚本最后只发了一帧 `neutral()` 就关串口 → 船一直转 ·
  根因同上（`_ramp_step` 字节级平滑 + 下位机无"无帧超时停车"）· 结论 → `stop_hard()`；
  `close()` 前轴不在中位必须先硬停

### test_settings_loads_real_yaml_values（代码注释搬出）
- 只守"是个正计数"这一不变量：单门 1 / 高低两门 2 / 多门 4 都会随现场调参变，

### test_settings_loads_real_yaml_values（代码注释搬出）
- ⚠️ 这几个是"板端现场值"（2026-09-18 用户定：以板端为准写回本地），会随现场调参变，

### test_settings_loads_real_yaml_values（代码注释搬出）
- ⚠️ **限深下限是"现场定死"的唯一例外：0.55 不准改**（2026-09-18 用户定）。

### _CountWrite（代码注释搬出）
- （2026-09-18 用户水里实测到"船一直转"，证据见 doc/注释历史.md）

### test_axis_watchdog_axes_come_from_cfg（代码注释搬出）
- 现场 cfg 必须写 axes: [yaw]；测试 cfg 覆盖（AUV_CFG_DIR）可能没这一段 ⇒ 只要求"有就必须是 yaw"，

### test_axis_watchdog_only_watches_yaw（docstring 搬出）
- 为什么：yaw 是闭环收敛量（正常转向越转误差越小，同向连续满舵 = 反馈卡住/在自转）；
- 拿它们当"卡死"判据会**误杀正常动作**（长直行被掐 = 直接失败）。

### test_axis_watchdog_trips_when_one_axis_is_held_same_direction（docstring 搬出）
- ⇒ 先硬停（estop）再强制退出（生产里是 `os._exit(9)`；用例注入 on_trip 记录，免得带走 pytest）。
- 为什么必须是独立线程：任务可能卡在某个循环里（现场"一开始进 turn 就卡住"），写在任务控制流

### test_stop_hard_ramps_axes_back_to_neutral（docstring 搬出）
- 注意：ramp 是"字节/秒"，需要**真实时间**流逝（`step_max=speed×dt`），
- 所以这里必须 sleep —— 紧凑循环里 dt≈0，轴根本不会动。

## tests/platform/test_paths.py

### 部署脚本默认目录
- 2026-09-22 之前 · `AUV_BOARD_DIR` 默认是旧副本 `/home/sunrise/AUV`，照默认跑就把文件推/比到
  错的拷贝上 · 结论 → 默认必须是现场在用的工作副本 `/home/sunrise/Desktop/AUV_New`，
  由 `test_deploy_scripts_default_to_the_live_copy` 钉住
### cfg 绝对路径 + 标定 yaml 文件头
- 2026-09-22 板端 · (1) cfg 里写绝对路径把配置钉死在某一份拷贝上（板端有 `/home/sunrise/AUV`
  与 `~/Desktop/AUV_New` 两份）；(2) 板端 cv2 4.11 要求标定 yaml 第一个字节是 `%YAML:1.0`，
  前导空行/注释都会被判失败 · 结论 → cfg 一律相对路径；文件头由 `tools/check` 自检

### test_deploy_scripts_default_to_the_live_copy（docstring 搬出）
- 照默认跑就会把文件推到/比到错的拷贝上（沿革见 doc/注释历史.md）。

### (模块)（docstring 搬出）
- 守的是两类坑：
- 1. **cfg 里写绝对路径** ⇒ 把配置钉死在某一份拷贝上，指错一份就静默用另一个文件、
- 那份被删就直接崩；⇒ 规则：cfg 一律写**仓库内相对路径**，运行时由
- `base.cfg.settings.resolve_path()` 解析。（2026-09-22 板端踩过，证据见
- doc/注释历史.md）
- 2. **标定 yaml 的文件头**：板端 cv2 4.11 要求第一个字节是 `%YAML:1.0`。

## tests/platform/test_common.py

### `cfgnode.flag` 的字符串语义
- 板端 · `bool("false")` 是 `True`，于是一个拼写错的 kpt_mem 值被当成了"开" ·
  结论 → 必须按字符串语义解析（false/0/off → False），这条断言守的就是那个坑

### test_preprocess_chain_domain_switch（代码注释搬出）
- 非法域：直接报错（别静默回退）

### test_cfgnode_node_readers（代码注释搬出）
- ⚠️ 必须按字符串语义解析：bool("false") 是 True —— 拼写错的值会被当成"开"
- （板端踩过，见 doc/注释历史.md），这条断言守的就是那个坑

### test_selected_nv12_path_is_color_faithful（docstring 搬出）
- （实测往返误差 ~15 灰阶、最大 161）。BPU 用这批 NV12 反解 RGB 喂网络，
- 模型看到的颜色会整体偏 —— 不报错，只掉精度。这条断言把取舍摆到测试里。

### test_preprocess_chain_domain_switch（docstring 搬出）
- 3. 非法值要**当场报错**，不能静默退回某个域 —— 静默退回 = 拿错域的图喂权重。

## tests/tasks/gate/test_gate_postproc.py

### `apply` 的 L/R 归一
- 2026-09-27 补的那一步 · 以前"左右标反"被当成"顺序非法"整条丢掉（模型约一半实例会标反）
  → 丢掉的是近门、只剩远处的门 · 结论 → 先 L/R 归一（左右反 → 交换后保留）；只有
  "上下也反/自交"这种真的非法才丢
### `_det_conf` 与 `model.score_threshold`
- 配置巧合 · 曾有一段时间两者取值**恰好都是 0.6**（ball 那个值被人从 0.5 调成 0.6）；
  2026-10-01 起 gate 侧是 **0.5**、共用侧仍是 0.6 · 结论 → 断言别写"两者必须不相等"
  （配置再巧合一次就会假红），改成显式改两个值看谁生效

## tests/tasks/gate/test_gate_flow.py

### `drop` 夹具被正则误伤
- 一次"用正则删夹具"的操作把 `drop` 连同它上面的 `@pytest.fixture` 装饰器一起吃掉了，
  症状是 `fixture 'drop' not found` · 结论 → 删函数别用跨界正则，注意别跨过装饰器行
### coarse 档「来回退」
- 实测 · 对准时还下发后退速度是"来回退"的主因 · 结论 → 只有**未对准**才后退
### 配置守卫（`test_gate_defaults_match_cfg`）
- 历史漂移值 · 这些字面量曾悄悄漂开，漂开方向有的很危险：
  `_D_SURGE.creep/lost_backward = 0.12`（落在执行器死区 = 该动作 0 推力）；
  `_D_PNP.reproj_px = 8.0`（cfg 实测已放宽到 20，8px 会拒掉绝大多数候选位姿）；
  `_D_THROUGH.confirm_ms = 900`（cfg 现场已改 2500，"没冲出去"的主因）；
  `_D_ALIGN.px_x/px_y` 0.25/0.26（用户收紧到 0.20/0.25）；`timeout_ms` 180000→300000 ·
  结论 → 每组默认值都要被这条守卫钉住；**拼错键名会静默失效**（`hdg.fresh_ms` 就这么漏过）
### 门口丢门的出口
- 现场 bug · 原先到门口丢门会后回 SEARCH → 0 分 + 原地旋转 · 结论 → ALIGN 必须判过门直冲
- 出口③「在门口超时兜底」替代了原 dash 出口；THROUGH 由 loiter 兜底到达
### p3p 触发帧也能起转
- 新模型 mode 每帧在 full/p3p/coarse 间跳，死等"本帧恰好 full"等于永远不转 ·
  结论 → 只要曾测到过 full 帧的 psi 就用它起转；旧版放宽开关 `entry_stale_ok` 已随离散迭代删除，
  **不许再加第二套语义**
### 「反向平移」状态与自转
- 2026-09-27 · 自转根因：曾写成"`_step` 顶端看见门就清窗口 / 判成远门就整帧 return"，
  那条路径**把位姿链整段旁路**，ψ 冻在起转前的旧值（25.0）上 ·
  结论 → 反向平移必须是主循环里的一个状态，判据照跑、ψ 每帧刷新，只覆盖本帧下发的指令
### 共用 PID 增益
- sway 反号那次事故就是"两任务各一套参数"的产物 · 结论 → 增益只写 `comm.motion` 一处

## tests/tasks/test_motion.py

### 假船极性（真机极性）
- 板端 `log/_board_gate_one_latest.jsonl` · 命令 +0.45（右转）时遥测 yaw 从 75.07 一路降到 9.93；
  当时代码假定 +1 ⇒ 闭合成正反馈（误差 52.6° 涨到 117°，命令 30° 转到 120°+）·
  结论 → 假船默认极性取真机极性（±yaw 命令 → 遥测 yaw 减小）；`dof_map.yaw.sign` /
  `telemetry.yaw_sign` / 接线任一改动，这里会先红
### 转角 PID 极限环（`kd` / `deadzone_deg`）
- 2026-09-27 仿真抓到 · 出厂 cfg `kd=0.10 / deadzone_deg=3.0`：内层 20Hz、误差按 norm_deg
  归一化 ⇒ D 项 = kd·Δerr/dt 在 30°/s 转速下就有 ~0.4 > out_max ⇒ 出力每帧正负反转、船卡在
  离目标 7.5° 处来回摆，永远进不了 3° 到达窗（20s 超时） · 结论 → `kd=0`、`deadzone_deg=6`，
  一次转到位、出力不反号
### 角度到舵效的实测量级
- 板端实测 · 0.3 舵 ≈ 60°/s（假船 `gain=200` 按这个量级取值）
### `blind_enable`（无遥测盲转）
- 2026-09-27 用户定 · 转向只允许闭环，**不留开环盲转备案** · 结论 → `blind_enable`
  （"角度÷假设角速率"定时盲转）已删除，不许复活

### run（代码注释搬出）
- 必须"跳过死区"才复现现场那种"转过头 ⇒ 回不来 ⇒ 被满舵保护掐死"。

### test_overshoot_tolerance_is_measured_against_the_issued_deg（docstring 搬出）
- · 关掉容差 ⇒ 超转后 |err| 与"朝下发角推进"都不再改善 ⇒ 2s 后被掐（现场"被限死"那种）；
- · 超转 ≤ 10° ⇒ 认到位、正常 DONE；**超过 10° 不认**（由下一个新鲜 ψ 带回来）。

### test_no_telemetry_never_turns（docstring 搬出）
- （`blind_enable` 那种"角度÷假设角速率"定时盲转已删除，不许复活）。

### test_step_cap_clamps_single_turn_and_zero_means_unlimited（docstring 搬出）
- ⚠️ 出厂默认 2026-09-28 起是 **10.0°**（不是 0）⇒ "不设限"要显式配（见 `test_turn_scale_and_max_step_shape_the_issued_angle`）。

### test_target_angle_is_frozen_while_turning（docstring 搬出）
- now_ms / yaw_telemetry / gate_lost）⇒ 转动中测到的"假已平行"不可能提前结束转向。

### test_divergence_guard_stops_the_turn（docstring 搬出）
- 构造：假船极性与真机相反（+yaw 命令使 yaw **增大**）⇒ 闭环变正反馈。
- 必须"转一点点就停"，而不是像板端那样一路转到 120°+ 把门甩出画面。

### test_loop_converges_without_limit_cycle（docstring 搬出）
- `kd=0.10 / deadzone_deg=3.0` 及其失效过程见 doc/注释历史.md）。

### test_turn_left_reaches_rated_angle（docstring 搬出）
- 起点 170°（回绕边界附近）→ 左转 90° 会跨过 ±180，所以这条同时验回绕。

### test_yaw_sign_is_a_fixed_derivation_not_a_measurement（docstring 搬出）
- （当初把 σ 当成 +1 曾闭合成正反馈，板端实测依据见 doc/注释历史.md。）

### (模块)（docstring 搬出）
- 所以离线用**假船**测：遥测 yaw 按 `角速率 × 下发的 yaw DOF × dt` 积分，
- `telemetry.yaw_sign` / 接线改了，这里会先红。（板端实测依据见
- doc/注释历史.md）

## tests/tasks/gate/test_gate_vision.py

### kpt_mem 默认开关
- 2026-09-18 用户定 · cfg 里 `kpt_mem.enable=false`（关掉融合）· 结论 → 工程真实配置下
  `build_kpt_memory` 返回 None，验融合行为要显式 `force=True`（不改配置）

## tests/tooling/test_cfg_yaml_head.py

### 标定 yaml 必须从 `%YAML:1.0` 开始
- 2026-09-22 现场踩到（板端 cv2 4.11 实测）· `cv2.FileStorage` 靠**文件开头**判断格式 ⇒
  前面只要有一个注释行**或一个空行**，就 `Input file is invalid` → 抛 SystemError；
  而 `common.preprocess.calibration_maps` **不做兜底** ⇒ 相机初始化直接崩 ·
  结论 → 这条用例检查**字节**（本地 cv2 5.0 更宽容，"本地能读"证明不了板端能读）
### 水/空气焦距比（`test_water_and_air_differ_by_dome_factor`）
- 2026-09-23 修正方向 · 本用例原先断言"空气 fx 明显大于水下 fx（比值 1.25~1.55）"，那是拿
  **旧水下标定 fx=782.5**（错的那份）比出来的。新水下标定（棋盘、水下、83 视角、RMS 0.546 px）
  给 fx=1207.6；空气侧三法互证 fx≈1078（tape 1083±7 / ruler+gate 1075 / 出框边界 ≥1095，
  且四档深度误差 ≤6%）⇒ 实测比值 **1078/1207.6 = 0.893**，即水/空气 = 1.12 ✓ 与"罩+水收窄
  视场"一致（runbook §B4b/§B4c） · 结论 → 断言方向是"水下标定 fx 更大"，
  区间 1.05~1.20

### test_water_and_air_differ_by_dome_factor（docstring 搬出）
- 断言的比值区间由现场水/空气两份标定**实测**得出（2026-09-23 修正方向：原先的断言
- 方向是反的，具体数字与推导见 doc/注释历史.md）。

### (模块)（docstring 搬出）
- `cv2.FileStorage` 靠**文件开头**判断格式 ⇒ 前面只要有一个注释行**或一个空行**，
- 它就 `Input file is invalid` → 抛 SystemError；而 `common.vision.preprocess.calibration_maps`
- **不做兜底**（不返回恒等映射）⇒ **相机初始化直接崩**。
- 本地的 cv2 5.0 更宽容，所以"本地能读"**证明不了**板端能读 —— 这条用例因此检查**字节**。
- （2026-09-22 现场踩到，板端 cv2 4.11 实测，证据见 doc/注释历史.md）
- 所以这条规则只适用于**由 cv2 读的相机标定文件**（`cfg/front_camera*.yaml`）。

## tests/tooling/test_label_corners.py

### 按完整路径去重（`load_done` / `remove_src`）
- 2026-09-22 现场踩过 · 采图目录里的文件名永远是 `cap_001.jpg`… ⇒ 按 basename 去重会让
  `ruler_z100` 的记录被 `ruler_z300` 整行删掉，`--resume` 更会直接跳过整档 ·
  结论 → 必须按**完整路径**去重（`paths` 集合），老记录退回 basename 只作并集

### test_same_basename_across_dirs_does_not_collide（docstring 搬出）
- 跳过整档（2026-09-22 现场踩过，证据见 doc/注释历史.md）。

### (模块)（docstring 搬出）
- 为什么这样测：这条「岸上路子」的价值全在于 —— **手工标的 4 个点能喂通 `pnp_calib`，
- 并反演出正确的深度标尺**。所以核心用例是一次**端到端往返**：

## tests/tooling/test_pnp_calib.py

### 门框尺寸标定与 z 偏差
- 历史观察 · PnP 解出的 z 比粗估大 ~20%（cfg 标外轮廓 0.70×0.50、真实开口内缘 0.60×0.40 时
  工具报 a≈1.17） · 结论 → 工具必须反演标尺，报出建议门宽/门高；反演不出来这份实验就白做

### test_p3p_needs_prev_or_fails（docstring 搬出）
- （3 点走 P3P 多解，本机 cv2 5 上该路径不可用 ⇒ 要有 prev 才走 ITERATIVE-guess）。
- 现场含义：**别指望"一直只能看到 3 个角"还能有位姿**。

### test_calib_reports_near_range_rejection_from_wrong_aspect（docstring 搬出）
- 2 m 处残差缩到门槛内 → 又能解出来。**这个现象在现场会被误读成"角点太差/模型不行"**，
- 所以工具必须把它如实报出来（每个距离档一行可用率）。

### test_calib_recovers_real_door_size_from_wrong_cfg（docstring 搬出）
- 这是现场最可能的情况（门框外轮廓 vs 开口内缘）。工具必须报出

### make_dump（docstring 搬出）
- （实测本机 cv2 5 下没有 prev 的 3 点解不出来，见 `test_p3p_needs_prev_or_fails`）。

### _rvec_for（docstring 搬出）
- yaw_deg > 0 = 船体**右转**（光轴往 +x 偏）⇒ 门相对相机的姿态是绕 y 转 **负**角
- ⇒ 门法向 n = R·[0,0,1] 的 x 分量为负 ⇒ `psi = atan2(n_x,n_z) ≈ -yaw_deg`。

### (模块)（docstring 搬出）
- 为什么这样测：pnp_calib 的全部价值是"用带真值的 dump 反演出正确的门框尺寸标尺与误差表"。
- 合成数据能给出**已知真值**，于是可以逐条断言它真的反演对了 —— 而不是"跑起来没报错"。
- 5. 航向真值 → `psi ≈ -yaw`（船右转 ⇒ 门法向偏向画面左 ⇒ psi 为负）；

## tests/tooling/test_ruler_calib.py

### `--measure` 的注册时机
- 2026-09-22 踩过 · `--measure` 如果晚于 `parse_args` 注册，`--help` 里看不到它、传进去还会报
  "unrecognized arguments" · 结论 → 必须注册在 `parse_args` 之前，由用例钉住
### 跨度必须用欧氏距离（`span_stats`）
- 2026-09-22 实测踩过 · 卷尺竖着放时旧代码只取 x 差 → 379 px 的真实跨度被算成 2 px ·
  结论 → 用欧氏距离，任意方向都成立
### 秩不足必须报错（`solve`）
- 2026-09-22 自检 · 秩不足时旧代码给出 `fx=-223 px` 这种垃圾解 · 结论 → 抛 `ValueError`
  （"秩不足"），不许输出垃圾

### test_solve_rank_deficient_is_rejected（docstring 搬出）
- 必须报错，不许给个垃圾解出来。

### test_span_stats_vertical_span_not_zero（docstring 搬出）
- （只取 x 差会把真实跨度算成个位数像素，2026-09-22 实测踩过，见
- doc/注释历史.md。）

### test_cli_registers_measure_option（docstring 搬出）
- 传进去还会报 "unrecognized arguments"（踩过一次，见 doc/注释历史.md）。

### (模块)（docstring 搬出）
- 为什么这样测：这条实验的目的只有一个 —— **从已知长度的刻度里把 `fx` 与 `c` 解出来**，
- 所以核心用例是**合成往返**：给定期望的 `(fx, c)` → 合成 Δu → 走工具链 → 解回同一个 `(fx, c)`。
- 4. `solve`：**精确与带噪往返**、2 档可解、<2 档报错、≥3 档给留一校验；

## tests/tasks/gate/test_gate_flow.py


### test_yaw_only_during_align（代码注释搬出）
- `pass_target` 是现场调参（现为 4）：把范围收敛到 1 门，免得用例随它漂。

### test_search_sweep_waveform_matches_cfg（代码注释搬出）
- 下位机链路实测死区：DOF→字节(128+127v)→Rc2MyRcKey→RC_Matching*(映射≤35→0)

### test_gate_defaults_match_cfg（代码注释搬出）
- 居中**只有 sway**：gate 段不许出现 yaw 居中通道（2026-09-20 用户定，别加回来）

### test_align_near_lost_declares_pass_instead_of_search（代码注释搬出）
- `pass_target` 是现场调参（现为 4）：把范围收敛到 1 门，免得用例随它漂。

### _through_frames（代码注释搬出）
- 否则 cfg 里的多门设置（现场跑 4 门）会让本用例连着过好几个门、把各轮 THROUGH 帧累加，

### __init__（代码注释搬出）
- imag_sign=-1 = **真机极性**（板端实测：+yaw 命令使遥测 yaw 减小）。改这里之前先看

### det（代码注释搬出）
- ⚠️ 若把中心位置也一起转（tvec=R@[0,0,z]），门心会偏出画面 fx·tanθ 像素 → 永远居不中。

### test_heading_align_turns_the_hull_parallel_then_proceeds（代码注释搬出）
- ⚠️ 不能按 zip 对齐：一帧可能发多条（20Hz 内层循环 + 居中指令），只有**帧内最后一条**生效。

### test_heading_align_turns_the_hull_parallel_then_proceeds（代码注释搬出）
- ⚠️ 只看每个 HDG 帧的**最后一条**指令：同一帧里可能先发过居中指令，再被转向覆盖

### test_turn_runs_to_completion_inside_one_frame（代码注释搬出）
- 这条用例只喂 3 帧检出 ⇒ 居中确认必须 ≤2 帧（固定成代码兜底值，别依赖现场调参）

### fn（代码注释搬出）
- ★ 内层闭环真的跑过 ⇒ turn_log 里必有 gate_loop（它写在 while 循环体内）

### _drive_to_sway_back（代码注释搬出）
- ★ 一发现"开始转"就把门甩出画面（画面里 0 个角点）—— 转 25~40° 后的真实情形；

### _det（代码注释搬出）
- ★ 关键：**位姿链没被旁路** —— 状态内检测/PnP/丢门处理照跑（只是本帧不一定测得出东西）。
- ⚠️ 不能再用"ψ 每帧刷新"来钉这条：门被甩出画面（本用例的场景）时 ψ **本来就测不出来**，

### test_post_sway_starts_after_the_turn_end_hard_stop（代码注释搬出）
- ★ 模拟生产的**整帧阻塞**：转向内层（真机 20Hz 帧内闭环 1~3s）+ 硬停 0.7s 紧挨着，

### _slow_inner（代码注释搬出）
- ★ 不变量（用户 2026-09-27 要的）：窗口起点 **在硬停结束之后**，不是"转向结束"那一刻

### _step（代码注释搬出）
- ⚠️ 同时要把 z 拉开（Δz > post_sway_same_z_m）以免"同门判据"先合法收手 ——

### test_post_sway_exits_as_soon_as_the_configured_keypoints_appear（docstring 搬出）
- ③ 出现 2 个角点 ⇒ 立刻退出并交回视觉（`sway_exit` 记原因）。

### test_small_steps_converge_toward_the_target_after_the_once_per_gate_removal（docstring 搬出）
- 这里钉：① 每小步下发角 ≤ max_step_deg(15°)；② 会分**多小步**（"每门只转一次"的限制已删）；

### test_turn_scale_and_max_step_shape_the_issued_angle（docstring 搬出）
- ⚠️ 本门的转向**只做一次**（`_hdg_done`）⇒ 被钳掉的残余不会补转：|ψ|=25° 时只转 10°。

### test_sway_back_yields_to_through_and_to_a_running_turn（docstring 搬出）
- ① `action == "through"` 掉进 else 被改写成 sway_back ⇒ **冲刺的 surge 被清零**最长一个窗口；
- ② 转向内层循环（真实时钟域 20Hz，帧内阻塞）经同一入口发 yaw ⇒ yaw 被清零 ⇒ **转向推不动**。

### test_turn_aborts_immediately_when_saturated_too_long（docstring 搬出）
- 现场形态："一开始上来就不停地转、超时保护也不起作用" ⇒ 不能只指望 turn 自己的超时

### test_sway_back_z_first_then_ratio_fallback（docstring 搬出）
- 用户 2026-09-27 定：优先信息 z；如果 z 一直被"新鲜帧"卡着（当帧 coarse 无位姿 / 位姿过期 /

### test_post_sway_starts_after_the_turn_end_hard_stop（docstring 搬出）
- `comm.gate.hdg.stop_hard`（默认 true）是**阻塞**的：实测 0.70s（无遥测）/1.31s（有遥测、静止）
- /2.51s（还在转），见 `base/hw/uart.py::stop_hard`。窗口若在硬停**之前**就开（600ms），会被整段吃光
- ⇒ 生产里只发得出 1 帧平移 ≈ 等于没有平移 —— 而"转完门被甩出画面"正是这个状态要救的场景。
- 帧时基按生产的方式走（硬停真实耗时计入时间轴），因此这条用例在旧写法下必红。

### test_sway_back_never_locks_up_even_if_every_frame_is_a_far_gate（docstring 搬出）
- ⚠️ 逐小步逼近（用户 2026-09-28 定，无次数上限）之后，"窗口到期"不再是唯一的收手方式：
- 下一小步起转（转向优先）也会把窗口让掉。所以这里只钉一件事 —— **最终一定收手**、
- 流程继续往前走（曾经那种"缺到期判据 ⇒ HDG 永久锁死"的写法必须被这条挡住）。

### _drive_to_sway_back（docstring 搬出）
- `z_after_turn`：转向一开始就把门"挪远"（z 变大 ⇒ 框变小、P 3P 的 z 也变大），

### test_width_mode_never_starts_heading_align（docstring 搬出）
- ⇒ 只剩对向 2 角时不会有正航向。这条守着它别被"顺手加上"。

### test_no_psi_skips_heading_and_proceeds（docstring 搬出）
- （等待新鲜 full、历史 psi 兜底、离散迭代、盲转）。

### test_hdg_skip_reason_when_disabled（docstring 搬出）
- ⚠️ **没有"次数用尽"这一类**了 —— 用户定：每门只转一次的限制不合理，改成逐小步逼近（无次数上限）。

### _hdg_single_turn_baseline（docstring 搬出）
- 出厂默认现在是"`turn_scale=0.8` + 每步 ≤10° + 每门最多 4 步"（用户 2026-09-28 定：逐小步逼近）
- —— 那会让"从 ψ=25° 一把转正"变成分好几小步，所以**测单次转向机制**的用例要显式退回基线，
- 否则测的就是新默认值的副作用（新默认值另有专门用例 `test_turns_are_done_in_small_steps...`）。

### _HullWorld（docstring 搬出）
- ⚠️ 这条关系 2026-09-28 改正过：旧写法是 `psi0 + H`（"右转 ⇒ psi 变大"），
- 而实船证明（见 `gate/motion/hdg.py` 的注释与 `log/run_gate.jsonl`）ψ>0 时**右转**才是
- 减小 ψ 的方向 ⇒ 右转必须让 psi 变小。这条关系就是"转向方向到底对不对"在 mock 里的体现。
- 检测用旋转后的位姿投影生成（4 角 → full），门心始终投影在画面正中（所以先"居中"能过），

### test_loiter_timeout_commits_through（docstring 搬出）
- 依据 log/gate_run1.jsonl：coarse 档 commit 出口关着 + 门在占比 0.97 时仍检出

### test_gate_defaults_match_cfg（docstring 搬出）
- 为什么值得一条用例：这些字面量曾经悄悄漂开，而漂开的方向有的很危险
- （具体漂移值与后果见 doc/注释历史.md）。

### test_no_speed_preset_inside_actuator_deadzone（docstring 搬出）
- ⚠️ `gate.search.sway` 也一起查（SEARCH 的平移扫视同样要真的出力）。

### test_horizontal_channel_is_sway_only_and_signed（docstring 搬出）
- ⚠️ **像素档/位姿档的 yaw 居中通道已删除**（原 `align_yaw.*`，2026-09-20 用户定）——
- 所以本用例只查 sway；`yaw` 在 gate 里只允许由 ALIGN.HDG 的正航向产生

### drop（docstring 搬出）
- ⚠️ 别用正则删本夹具：正则很容易跨过 `@pytest.fixture` 装饰器行一起吃掉，
- 症状是 `fixture 'drop' not found`（出过一次，见 doc/注释历史.md）。

### _resolved_gains（docstring 搬出）
- gate 的 cfg 里**不再写增益数字**（共用 `comm.motion` 那一套），所以只能问任务本身。

### (模块)（docstring 搬出）
- 6. 配置守卫：代码兜底 == cfg，且**共用参数只写在 comm.motion**（见 `test_gate_defaults_match_cfg`）。

## tests/tasks/gate/test_gate_postproc.py


### test_lr_normalize_skips_placeholder_and_single_side（代码注释搬出）
- 若把占位的 x=0 算进去，均值变成 300 ≤ 400 ⇒ 会被误判成"没标反"而漏掉这次归一。

### test_pick_does_not_touch_safety_defaults（代码注释搬出）
- ⑦ 解码后端接线（不加载真实模型：把 _init_backend 换成空实现）

### test_gate_backend_wires_det_conf_and_postproc（代码注释搬出）
- ⚠️ 别写成"两者必须不相等"：两者取值可以巧合相同，那种断言会假红。

### test_apply_drops_illegal_quad_and_preserves_order（docstring 搬出）
- 旧行为会把近门整条丢掉，沿革见 doc/注释历史.md）。

### (模块)（docstring 搬出）
- 这些是**逻辑**证明：能证明规则按写的那样跑，**不能**证明它在水里/板端的效果。

## tests/tasks/gate/test_gate_vision.py


### test_kpt_memory_switch_precedence_and_smoothing（代码注释搬出）
- 工程真实配置：**kpt_mem 默认真关掉**（enable=false，2026-09-18 用户定）→ 返回 None

### test_gate_task_mock_reaches_through_and_counts_pass（代码注释搬出）
- `pass_target` 是现场调参（现为 4）：把范围收敛到 1 门，免得用例随它漂。

### _set_kpt_cell（docstring 搬出）
- 注意：ONNX 输出的 kpt x/y **已经是 cell 坐标**（= `raw*2 + 网格索引`，
- 所以这里直接写 cell 坐标；板端只做 `×stride`。

### _blank_outputs（docstring 搬出）
- 合成用例里必须显式把背景压下去，否则解出几千个空框。

## tests/_bite_probe.py


### test_bite__through_confirm_ms_is_used（docstring 搬出）
- 所以它验的是语义而不是数值 —— 数值由配置守卫钉住。）

### test_bite__through_confirm_ms_is_used（docstring 搬出）
- （`test_through_duration_is_time_based` 自己 monkeypatch 出 500ms 来验"按时间不按帧数"，

## tests/conftest.py


### (模块)（docstring 搬出）
- 必须在任何测试模块 import base.cfg.settings **之前** 执行：
- 模块以顶层包互相引用（import base.cfg.settings as S），所以工程根必须在 sys.path 上。

## tests/platform/test_hud.py


### (模块)（docstring 搬出）
- 为什么单独测：现场最常盯的就是这一行，而它有两个**会被误读**的坑 ——
- 否则现场会以为"航向没问题"，而实际上是根本没测过。

## tests/tooling/test_tape_ticks.py


### (模块)（docstring 搬出）
- 为什么这样测：这条方法是**唯一不依赖点击**的标尺测量（几十个刻度一起估计周期，
- 精度 ~0.5%），所以核心用例是**合成往返**：按已知 `f` 画一条带厘米刻度的卷尺 →

## tests/tasks/grab/test_cv_ball.py（脚本注释搬出）


```
判据：`dom/(dom+2mx+1) >= t`，LUT 用"严格大于"写成 `dom > lut(mx)`，
`lut(mx) = ceil(t(2mx+1)/(1-t)) - 1`。
踩过的坑（两条，都由这个参数化用例挡住）：
① `>=` 写法 + `r >= f(mx)`：t 大时 f 超过 255，uint8 LUT 只能 clip 到 255
⇒ "r=255 且 mx≥138" 被错误放过（纯白像素 r=mx=255、dom=0 本不该进掩膜）。
改成 `dom > lut(mx)` 后，"不可满足"能天然编码成 lut=255（`dom>255` 恒假）。
② 用 float 走 `np.ceil` 构造 LUT 会重新引入舍入（13/65536 格差 1 档），必须用整除算 ceil。
守住三条最容易悄悄坏掉的约定：
1. 快掩膜（板端路径）与 numpy 参考实现**逐帧结果一致** —— 一旦不等价，PC 上调好的参数
在板端就不成立了，而且不会报错。
2. 圆拟合/几何过滤的**边界行为**：合成一个红球必须检出、圆心/半径对得上；
细长红条（瓦缝红光）必须被拦掉。
3. ROI 跟踪与全图检测**结果一致**，且跟踪确实在走 ROI 分支（别静默退化成每帧全图）。
# --------------------------------------------------------------------------- #
# 1. 快/参考掩膜等价（板端路径的正确性）
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 2. 检出与过滤的边界行为
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 3. ROI 跟踪 == 全图（且真的走了 ROI 分支）
# --------------------------------------------------------------------------- #
```

## tests/_bite_probe.py（脚本注释搬出）


```
（原探针咬的是 `max_turns`；2026-10-02 起"一门只转一次" ⇒ 上限降级为兜底配置、
不再改变行为，咬不动了（见 `test_one_turn_per_gate_makes_max_turns_vestigial`）。）
（原探针改的是 `hdg.fresh_ms`；那个旋钮 2026-09-30 已随"航向优先"一起删除，
现在的"新鲜度"机制是"转向之后必须有一次新的 ψ 测量"，由 turn 计数/时刻驱动。）
（原先咬的是 `timeout_ms`；2026-10-02 改成"门口超时 → `creep_through` 慢速冲门"后，
超时不再是主闸、咬不动了，故改咬居中安全带。）
每条探针 = 「在内存里改一个 cfg 值 → 直接调用目标用例 → 它必须变红」。
```

## tests/conftest.py（脚本注释搬出）


```
要用闭环的用例显式置 `uart.telemetry.yaw_deg = 0.0`；此后 `send_dof` 会按
"固件约定：+yaw = 右转 ⇒ 回传 yaw 减小"（σ=-1）积分。
* AUV_SIM_MODE=1 → S.SIM_MODE=True（UartController 默认只打印，不发串口）
* AUV_STREAM=0   → SimCamera.read() 不新建 UDP 推流器（测试不碰网络）
```

## tests/platform/test_base.py（脚本注释搬出）


```
`AUV_WATCHDOG=0` 都能关；`AUV_WATCHDOG=1` 强制开。
而 surge/sway/heave **可以合法长同向**——冲刺几秒直行、扫视/横向对中持续平移、保深持续垂向。
里的保护那时根本跑不到。
这里把 ramp 调快（5000 字节/秒）让用例快跑；"按 ramp 速率发够帧"本身由
上一个用例用真实 ramp 验证。
背景：DOF→字节→下位机 `RC_Matching*`（映射≤35→0）使实际推力远小于 DOF 数字
（0.20→20%、0.30→30%、0.60→60%，见 gate 文档 §1.1）→ 下潜偏弱就把这一路放大。
阈值**从配置读**（不写死）：用户把下限定在 0.55，写死 0.3 会随配置变化而失效。
`dof_comp`：默认**关掉下潜放大**，让"限深保护"的用例只测保护本身（两者互不干扰）；
要测下潜放大的用例传 `dof_comp=True`。
只跑 SIM 路径：UartController(sim=True)、相机 type 强制 sim，不碰串口与摄像头。
遥测/限深用 `UartController.feed_telemetry()` 或假串口喂字节，不发真帧。
# ---------------------------------------------------------------------------
# 下位机遥测上行（15B 0xAA55 + turn_id + 深度/姿态 + turn_done + 校验和）
# ---------------------------------------------------------------------------
```

## tests/platform/test_common.py（脚本注释搬出）


```
`vision.model.fast_nv12=false` 会切到 numpy 版，而那一版的色度被放大 4 倍
咬两件事：
① **旧开关不许复活**：`ModelPreprocessor(chain=...)` 必须直接 `TypeError`
—— 谁把 A 链或"域开关"加回来，这条立刻红（防止两套链路又悄悄并存）；
② **链序真是 D**（缩放 → 画面补偿 → 640 去畸变）：与手工按该顺序拼出来的结果**逐像素相同**
—— 顺序被倒成老 A 序（先 remap 再缩放/补偿）就会红。
cfgnode 配置读取（含 ball·gate 共用的 comm.motion 参数）。
# ==============================================================
# 图像链路：NV12 打包保真（模型输入的色度）
# ==============================================================
# ======================================================================
# cfgnode：配置节点读取 + **ball/gate 共用运动参数**（comm.motion）
# ======================================================================
```

## tests/platform/test_hud.py（脚本注释搬出）


```
1. 还没测到 psi 时必须显示 `--` 而**不是 0**（0 会被当成"已经正了"）；
2. `hdg_skip` 有值时（这一趟跳过了正航向）必须**变色并写明原因**，
```

## tests/tasks/gate/test_gate_postproc.py（脚本注释搬出）


```
只有"上下也反/自交"这种真的非法才丢；左右反不算非法（模型约一半实例会标反，
覆盖 `gate/percept/gate_postproc.py` 四条规则里的可离线部分：
① 可信角点（V_MIN 口径） ② 四角几何合法（顺序 + 不自交）
③ 重复框去重（**不看包含**） ④ 选门键（near / corner 两种策略）
外加一条**配置守卫**：代码兜底 == cfg（与 test_gate_defaults_match_cfg 同一约定）。
# --------------------------------------------------------------------------
# ① 可信角点
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ② 四角几何合法
# --------------------------------------------------------------------------
    # 顺序变成 TL,TR,BR,BL 后：TL.x=400 < TR.x=700 ✓，但 BL.x=400 < BR.x=700 ✓，
    # TL.y=200 < BL.y=420 ✓，TR.y=200 < BR.y=420 ✓ —— 顺序检查过，靠自交检查挡
# --------------------------------------------------------------------------
# ③ 重复框去重
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ④ apply()：几何 + 去重一起，保序、不改动传入对象
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ⑤ 选门
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ⑥ 配置守卫：代码兜底 == cfg（改了 cfg 就要同步兜底表）
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
    # 候选阈值走 gate 专用的 det.conf（**不是**共用的 model.score_threshold）
    #    这里改成**显式改两个值、看谁生效**：
# --------------------------------------------------------------------------
# ⑧ 解码→后处理 集成（合成张量，不加载模型）
#    形状/语义照 doc/设计/gate_pose_decode_spec.md §2/§3：NHWC、kpt x/y 已是 cell 坐标、v 是 raw logit
# --------------------------------------------------------------------------
```

## tests/tasks/gate/test_gate_vision.py（脚本注释搬出）


```
把角点编码在 cell 坐标 = 网格索引本身（raw=0）处：
正确 → 像素 = gx * stride
多写 -0.5 → 像素 = (gx-0.5) * stride，即 stride=8 上偏 4 px
cls 初值给 -10（sigmoid≈4.5e-5）而不是 0：sigmoid(0)=0.5 会全部越过 conf 阈值，
非等比缩放回投 / mode 降级 / kpt_mem 开关 / GateTask(mock) 无硬件闭环。
GateTask 相位机（MockGateBackend 无硬件闭环）。
    #  2026-10-02：**1.8 m 外的位姿一概不信**（用户定）⇒ mock 起始距离在 1.8 m 外时，
    #  开头几帧没有可用位姿 ⇒ 先 SEARCH（靠框占比靠近），进了 1.8 m 才进 ALIGN。
# ==============================================================
# keypoint 头解码约定（合成张量；训练侧 ONNX 与板端唯一的耦合面）
# ==============================================================
# ======================================================================
# 2026-10-02：运行时**可信边界** `z.relock_away_m = 1.2 m`（超出即不采纳该帧位姿）。
# 本模块绝大多数用例合成的门放在 1.5–3 m，考的是**位姿之后的逻辑**（起转/出口/恢复/SWAY_BACK…），
# 与"多远才算可信"正交 ⇒ 这里统一把边界放宽到 99（= 关闭），只有专门考这条的用例用真值
# （用例名里带 jump/relock/far 的自动跳过，不覆盖）。
# ======================================================================
```

## tests/tasks/test_ball.py（脚本注释搬出）


```
封顶是为了"会咬"：不封顶时边长会涨到框比整幅画面还大（占比 >6，物理上不可能），
用脚本化 hub 直接喂 Det，不加载模型、不开相机；now_ms 由测试推进（不真 sleep）。
```

## tests/tasks/test_motion.py（脚本注释搬出）


```
老写法（`--out-max/--kp/--kd/--norm-deg/--imag-sign`）现在必须**报错**而不是被静默忽略 ——
静默忽略会让"以为设了限幅其实没设"，宁可让老脚本当场失败。
与自动那条对照：两者都落到 `uart.request_turn()`（同一个 `TurnCore`）。
做法：同一个 GateTask 先喂两帧 full（建立 +20° 的测量），再喂几帧 p3p（旋转过的、会测出别的值）
→ `_hdg_deg` / `_hdg_ms` 必须保持 full 那帧的值不变。
Args:
world:   给了就"真的"把假世界转过去（等价于下位机执行）。
replies: poll 几次之后才回报完成（1 = 当帧完成；很大 = 永不回报 → 触发上位机超时）。
accept:  False = 模拟下发失败（`request_turn` 返回 False）。
estop:   True = 下位机处于急停（上位机必须立刻收手）。
**2026-10-02 起模型变了**（板端先行的版本，本地对齐）：旋转**交给下位机执行**——
上位机只发"相对角度 + 编号"（`uart.request_turn`）、由 UART 层按同一编号重发，
然后等 15B 遥测里的完成标志（`uart.poll_turn_complete`）。**上位机不再跑 yaw PID 闭环**。
所以这一半测的是新契约：
· 符号：`+` = 右转、`−` = 左转（`TurnCore.d` 左 −1 / 右 +1）；
· 执行期间上位机的 yaw 恒 0（手动 yaw 永远回中，转头这件事不归它）；
· 完成/超时/下发失败/急停四条收场路径，以及超时要**取消**下位机那次转动；
· 正航向 `HeadingAligner`：起转前冻结目标角 → 交给下位机 → 等完成。
# ======================================================================
# 假下位机：**旋转的执行者**（2026-10-02 起）
#   上位机 `request_turn(相对角)` → 下位机转动 → 遥测回 `turn_done`
#   ⇒ 测试里用 `request_turn` / `poll_turn_complete` / `cancel_turn` 三个钩子模拟它。
# ======================================================================
# ======================================================================
# 额定转角（common/motion/turn_deg.py）
# ======================================================================
# ======================================================================
# 正航向（gate/motion/hdg.py）：冻结 PnP 目标角 → 交下位机转一次 → 结束
# ======================================================================
# ======================================================================
# 手动入口（CLI）：**给一个目标角度就转**（与 gate 自动接受同一条执行链）
# ======================================================================
# ======================================================================
# 2026-10-02：运行时**可信边界** `z.relock_away_m = 1.2 m`（超出即不采纳该帧位姿）。
# 本模块绝大多数用例合成的门放在 1.5–3 m，考的是**位姿之后的逻辑**（起转/出口/恢复/SWAY_BACK…），
# 与"多远才算可信"正交 ⇒ 这里统一把边界放宽到 99（= 关闭），只有专门考这条的用例用真值
# （用例名里带 jump/relock/far 的自动跳过，不覆盖）。
# ======================================================================
```

## tests/tasks/test_search_scan.py（脚本注释搬出）


```
改了 cfg 就要同步兜底表，否则"缺配置时行为不同"，而实船最难查的正是这种"改了不生效"。
（这条是被实船坑出来的：`sub(node, key)` 不吃点号路径 ⇒ Scan 一直用兜底值、cfg 全被忽略。）
这是"search 只是原地来回扫、不会转圈也不能转圈"的硬保证：船的净转动量被
2×span + runaway_slack_deg 夹住，越界即停手（不发舵）。
轨迹应严格夹在 ±span 内（允许 tol 容差），且段目标交替 -span/+span。
守四件事：
① **轨迹**：-span → +span → -span …（在 ±span 内往复），总扫幅 = 2×span_deg；
② **段到位后停 pause_ms**（停的间隙检测更稳）；
③ **没遥测就不转**（闭环量缺失 ⇒ 不乱转）；
④ **出力不超 pid.out_max，且不低于执行器死区**（死区 0.138 是物理下限）。
        # 固件约定：下发 +yaw = 右转 ⇒ **回传 yaw 减小**。σ=-1（= yaw_sign()）时：
        #   d(yaw_deg) = σ · cmd · rate · dt  ⇒ +cmd 使 yaw_deg 变小 ✔
    # 逼近目标时出力必然很小（会落进执行器死区）——这是**物理下限**，不是 bug：
    #   当年 turn_deg 文档："err≤6° → 进 PID 死区 → 输出 0 ⇒ 收敛残差 ≈6~10°"。
    #   所以这里只要求"最大出力足够过死区"（否则整段都推不动、等于没扫）。
```

## tests/tooling/test_cfg_yaml_head.py（脚本注释搬出）


```
另外：`base/cfg/settings.py` 用的 YAML 是项目自己的解析器（UTF-8 中文没问题），
```

## tests/tooling/test_label_corners.py（脚本注释搬出）


```
按 basename 去重会让 `ruler_z100` 的记录被 `ruler_z300` 整行删掉，`--resume` 更会直接
点在管子上 → 原地不动；点在远离管子处 → 保持点击（不乱吸）。
已知位姿 → 合成角点 → 走 label_corners 写成 dump → pnp_calib 复算 → 深度对得上。
GUI 部分无法在无显示器环境测（已用 `--coords` 离线模式覆盖其数据通路）。
覆盖：
1. 吸附到红管（snap_to_red）：点到偏了会吸到管子中心；非红区域不乱跳；
2. 记录 schema 与 `preview_detect --dump` 一致（pnp_calib 能读）；
3. **端到端**：coords 模式 → dump → `pnp_calib.run()` → z 正确、`frame_w` 反演正确；
4. `--resume` 按文件名跳过已标注；
5. bbox 由 4 点算出。
```

## tests/tooling/test_pnp_calib.py（脚本注释搬出）


```
这不是本工具的缺陷，而是 `geometry._iter_candidates` 的既定行为
合成：真门 0.60×0.40 用 0.70×0.50 去解 → 0.6 m 处残差 > 20 px 全被拒（可用率 0%），
① a ≈ 1.17（z 系统性偏大）② 建议的门宽 ≈ 0.60、门高 ≈ 0.40。
反演不出来 = 工具没在反演标尺，那这份实验就白做了。
`drop` = 要打掉置信度的角点 id（模拟缺角 → p3p/width/coarse）；
`lead_full` = 前多少帧仍给全 4 角 —— 给 3 点 PnP 建立 `prev` 猜值
（与 `gate_normal_angles_deg` 的约定一致：+psi = 门法向偏向画面右 = 机身相对门左偏。）
覆盖：
1. 文件名真值解析（z/lat/up/yaw 各组合）；
2. 标尺正确时：a≈1、b≈0、相对误差≈0、反演建议的 frame_w 不变；
3. **故意把 cfg 门宽写大 20%**（模拟"标称外轮廓、实际开口"）：工具必须报 a≈1.2，
并建议 `frame_w ≈ 0.70/1.2`；
4. 横向/竖向真值 → t_x/t_y 的尺度与**符号**（t_y 与 up 反号）正确；
6. 丢角点(3 角)时 mode=p3p 被正确分类，且 p3p 的深度明显偏（**p3p 不给米制阈值**的证据）；
7. 汇总/逐帧 CSV 能写出来且行数正确。
```

## tests/tooling/test_ruler_calib.py（脚本注释搬出）


```
覆盖：
1. `span_stats`：3 个刻度点的跨度/半跨/斜视诊断（含正对 vs 斜视）；
2. `make_ruler_record` schema（`kind: "ruler"`，含 z 真值解析）；
3. `load_records`：z 从 `src` 兜底解析、跳过非靶子行；
5. `report`：含 fx/c 与 `k_med = fx_cfg/fx`，skew 超 2% 会告警；
6. 端到端：JSONL → `main()` → 报告落盘。
```

## tests/tooling/test_tape_ticks.py（脚本注释搬出）


```
工具测出的 `f` 必须对得上。另加"透视梯度"用例（靶面没正对时，光轴处的局部比例才对）。
```
