# gate — 任务三「过门」模块说明

门 = 闭合矩形门框（红 PVC，**0.70 × 0.50 m**，悬空、对称、无朝向要求）；
任务 = 机身穿过开口。算法：YOLO-keypoint 四角 → PnP/反投影位姿 → 相位机 → DOF。

设计背景见 `doc/算法说明-gate-PnP移植方案.md`；本次分层的算法说明见
`doc/算法说明-gate-v1.4-分层与质量分.md`；角点融合专题见
`doc/算法说明-gate-角点逐点融合滤波.md`。

---

## 1. 目录分层（视觉处理 / 数据处理 / 运动决策）

依赖**单向**：`motion` 用 `vision`/`data`，`vision` 用 `data`，`data` 只依赖 numpy。
顶层只留装配与仿真（`gate_task.py` + `mock.py`）。

| 层 | 文件 | 职责 | 关键产出 |
|---|---|---|---|
| **顶层** | `gate_task.py` | 相位机骨架：配置装配、帧守卫、相位选路、位姿/像素对准档、复位与诊断 | DOF（`uart.send_dof`） |
| **顶层** | `mock.py` | `MockGateBackend`：脚本化进近/穿门轨迹（可注入缺角/整框/丢目标） | `Det` 序列 |
| **视觉处理** | `vision/gate_decode.py` | keypoint 头解码（模型输出 → 4 角点，含 640→帧坐标反缩放） | `kpts/kpt_conf` |
| | `vision/gate_detector.py` | `board_camera()` 相机模型（原始/校正域一致）、`build_gate_backend()` 后端装配 | `CameraModel` / 后端 |
| | `vision/gate_frontend.py` | 前端适配：`parse_kpt_mode`（full/p3p/width/coarse）、`width_range_depth`、`bbox_center` | `mode` / 测距 |
| | `vision/geometry.py` | `CameraModel`、`object_points`、`gate_pose`（IPPE+消歧）、`reproj_rms`、门平面反投影 | 6-DoF 位姿 |
| | `vision/perception.py` | **感知半场**：选目标 → 质量先验 → 融合 → mode → 线索 → 位姿/RMS → 质量后验 | `Sighting` |
| **数据处理** | `data/kpt_memory.py` | 角点逐点软融合（α-β + 柯西鲁棒权重 + 历史半径 + 滑行/衰减） | 融合角点 + 连续置信度 |
| | `data/quality.py` | 质量分 Q 的两条去向：**先验**（→融合权重/增益）与**后验**（→谨慎度/q_mem） | `w_prior` / `Quality` |
| | `data/cues.py` | 恢复线索表（可见性→方向动作）+ 线索↔位姿互相自检 | cue / 冲突判定 |
| **运动决策** | `motion/phases.py` | 相位与子状态常量（骨架与相位段共用一份） | 常量 |
| | `motion/phase_degrade.py` | 相位段：width 档、coarse 三层仲裁、REACQUIRE 闭环 | DOF |
| | `motion/phase_recover.py` | 相位段：搜索脉冲、丢目标防抖、穿门收尾、线索接管 | DOF |
| **共用件**（`common/`） | `common/streak.py` | 时间 + 帧数双条件确认（线索动作用） | `ready()` |
| | `common/frame_stamp.py` | 唯一帧 / 单调时间 / 帧龄 / 采集断点（两分法） | `Verdict` |

> 相位段以 **mixin** 并入 `GateTask`（`class GateTask(DegradeTicks, RecoverTicks)`）：
> 它们读写的是同一份任务状态，拆成独立类只会退化成 `self.task._G` 式绕路；
> 方法体与拆分前逐行一致（只搬文件）。
> `cues.py` 放在 `data/`：它是"从可见性数据里提出线索并做互检"，被运动层（驱动）
> 与质量分（打折）两侧消费，放这里才能保持依赖单向。

## 2. 一帧的数据流（质量分的两条去向）

```
                    ┌─────────────── gate/vision/perception.py（感知半场）───────────────┐
 帧 ─► 检测 ─► 选目标 ─► ①质量先验(原始conf × 帧龄) ─► kpt_memory 融合 ─► mode/可见性 ─► 恢复线索
                              │(w_prior)                    ▲                  │
                              └──────────────────────────────┘                  │
                                                                    ┌───────────┘
                                            位姿 PnP/RMS（z 跳变保护）◄─┘
                                                    │
                                    ②质量后验(RMS/框IoU + 线索↔位姿互检)
                                                    │
                          ┌─────────────────────────┴──────────────────────┐
                          ▼                                                ▼
            谨慎度 c → 速度×sp / 死区×dz                        q_mem(EMA) → 下一帧 ①
                          │
       gate/motion/* 相位机（SEARCH→ALIGN→APPROACH→THROUGH）+ 帧守卫 → DOF
```

* **Q 不参与过门判定**；`confirm_max_scale=1.0` 时它也不改确认时长，只改"给多少权重 / 开多快"。
* 帧守卫（`process(frame, now_ms, frame_id, captured_at)`）两分法：过期帧**仍用于估计**，
  但不计入任何确认；重复/乱序帧沿用上一帧指令。
* 诊断量（`last_info`，叠加可见）：`Q` 后验总分 / `Qmem` 历史证据 / `gain` 融合实际增益
  （`<1` ⇔ 质量分真的压了这一帧）/ `caution` / `cue` 线索动作 / **`cue_w` 线索主导权重**
  （0=完全听位姿，1=完全听线索）/ `cue_mode` 当前主导权档位 / `visible` / `source` /
  `frame_kind` / `age` / `rms` / `agree`。

## 3. 使用说明

### 3.1 本机无硬件（推荐先跑这些）

```bash
cd auv_vision                       # 项目根（含 main.py / cfg / tests）
python3 tests/test_gate_layers.py       # 分层 + 质量分两条去向 + 软耦合（53 项）
python3 tests/test_gate_enhance.py      # 帧守卫/双条件/质量分/线索/端到端（45 项）
python3 tests/test_gate_flow.py         # 相位机：进近→穿门 / REACQUIRE（mock）
python3 tests/test_gate_standalone.py   # 单任务接线（只加载门权重、拒绝空跑）
python3 tests/test_gate_geometry.py     # PnP/反投影合成往返
python3 tests/test_gate_kpt_memory.py   # 角点逐点融合（倒影/鬼点/滑行）
python3 check_pipeline_identity.py      # 域自证：预处理/坐标域/标定一致（移植后必跑）
```

一次跑全套：`for f in tests/test_*.py; do python3 $f || echo FAIL $f; done`

### 3.2 有相机/无船（看画面 + 角点）

```bash
python3 preview_detect.py --gate-kpt     # 门 keypoint 预览：角点、mode、可见性（不驱动运动）
```

### 3.3 无硬件闭环（mock 轨迹）

`MockGateBackend` 可脚本化缺角、整框可信、丢目标；单元测试即用它跑完整相位机：

```bash
python3 tests/test_gate_enhance.py       # 内含 Mock 端到端 → DONE(pass)（约 249 帧）
```

### 3.4 整机入口（需权重）

```bash
python3 main.py --task gate              # 只跑 gate（权重缺失会直接拒绝启动）
AUV_SHOW=0 python3 main.py --task gate   # 无显示环境
```

### 3.5 板端（RDK X5）

* 权重：`cfg/vision.yaml → model.task_models.gate.path`
  （板端 `/home/sunrise/AUV/models/gate_kpt_bayese_640x640_nv12.bin`）；**必须 `.bin`（march `bayes-e`）**，
  `.hbm` 是 Nash/J5/J6 产物，不兼容。
* 标定：`cfg/front_camera.yaml` 的分辨率必须与实际帧一致（1280×720）；不一致时
  `board_camera()` 会打印告警并按近似针孔退化 → PnP 深度整体缩放错。
* 本地与板端对齐：先各自跑 `python3 check_pipeline_identity.py`，输出的 `pipeline_id` /
  `sha256(vision)` / `sha256(calib)` 必须一致。
* 板端 `udp_server.py --real`（手动模式遥控桥）**不要动**；gate 与手动模式互斥使用。
* 串口：`S.SIM_MODE`（本机 True，板端 False）；串口帧只发 7 轴 DOF，`send_dof(surge, sway, heave, yaw)`。

## 4. 参数配置指南

参数唯一来源：`cfg/comm.yaml → gate.*`（运动/策略件）与 `cfg/vision.yaml → gate.*`（几何/感知）。
**不写就用代码默认值**（默认与 yaml 现值同值）；改配置不需要改代码。

### 4.1 运动/任务（`comm.gate.*`）— 底层运动，慎改

| 参数 | 默认 | 含义 | 调参方向 |
|---|---|---|---|
| `timeout_ms` | 180000 | 单任务总时限（超时 → DONE(timeout)） | 赛道长度变化时改 |
| `pass_target` | 1 | 需通过的门数（高低双门可设 2） | 赛制 |
| `pid.{kp,ki,kd,out_max,deadzone}` | 1.0/0/0.15/0.6/0.04 | 横向 PID（位姿档与像素档各一套） | 摆动大→降 kp/升 kd；迟钝→升 kp |
| `align.xy_m` | 0.05 | 对准判据（位姿档 \|t_x\|,\|t_y\| ≤ 它） | 偏严→对准慢但稳；注意它会 **× dz** |
| `align.confirm_frames` | 6 | 对准达标连续帧数 → APPROACH | 抖动大→加大 |
| `z.slow_max` | 1.3 | 进近两档分界（远 fast / 近 slow） | 近距冲太快→加大 |
| `z.cross` | 0.7 | 判"已穿门"的深度阈值（连续 `cross_confirm_frames` 帧） | 依门框尺寸/相机安装实测 |
| `z.cross_confirm_frames` | 2 | 防单帧 PnP 错解 | **不建议降到 1** |
| `z.near_lost_m` | 1.0 | 进近到该距离后整门丢失 → 判过门（最常见过门路径） | 依实际"门占满视野"的距离 |
| `surge.{fast,slow,creep}` | 0.35/0.15/0.12 | 进近两档 / 慢蠕进 | 会再乘 `sp`（质量分） |
| `surge.lost_backward` | 0.12 | 远处短暂丢失的轻微后退（代码取负） | 水中后退易偏 → 谨慎 |
| `surge.reacquire` | 0.25 | 后退重取速度（代码取负） | **符号需水池实测** |
| `surge.through` | 0.5 | 穿门冲刺 | 保证一次穿过 |
| `coarse.{far_ratio,near_ratio}` | 0.18/0.55 | 远/中/近三层仲裁（门框宽/屏宽） | 依检测框稳定性 |
| `coarse.{align_x,align_y}` | 0.05/0.10 | 整框对中阈值（归一化像素） | 会 **× dz** |
| `hold.max_frames` | 20 | HOLD 无进展帧数 → REACQUIRE | 太敏感→假后退 |
| `reacquire.{max_ms,max_times,stop_ratio,reset_after_ms}` | 800/2/0.75/3000 | 闭环后退：单次上限 / 同段最多几次 / 退到 75% 即停 / 多久后重新允许 | 反复后退→降 max_times |
| `through.confirm_frames` | 8 | 穿门后目标消失确认帧数 → 计数 +1 | 与帧率相关 |
| `search.{spin_s,pause_s,yaw}` | 2.0/1.0/0.35 | 搜索旋转脉冲（转/停） | 扫描覆盖率 |
| `pose_hold_frames` | 10 | 丢目标/位姿时的保持帧数（防抖） | 帧率低→加大 |

### 4.2 几何/感知（`vision.gate.*`）

| 参数 | 默认 | 含义 | 调参方向 |
|---|---|---|---|
| `geometry.frame_w/frame_h/bar_width` | 0.70/0.50/0.05 | 门框实测尺寸（米） | **上线前必须实测填值** |
| `geometry.body_center_offset` | 0.0 | 机身中心相对相机偏移 | 直穿轨迹纠偏 |
| `keypoint.conf_thr` | 0.5 | 角点置信度阈值（判可见） | 倒影多→升高 |
| `kpt_mem.enable` | true | 关掉即"单帧直用"（回退用） | 出问题先关它定位 |
| `kpt_mem.alpha/beta` | 0.4/0.1 | 位置/速度融合增益 | 抖→降 alpha；拖→升 alpha |
| `kpt_mem.k_sigma` | 3.0 | 验证门半径 = k_sigma × 该点残差 σ | 鬼点多→降 |
| `kpt_mem.r_min_px/r_max_px` | 10/80 | 半径上下限 | 近距快速逼近→升 r_max |
| `kpt_mem.sigma_init_px/sigma_lambda` | 8.0/0.15 | 残差尺度初值/慢跟踪速率 | 世界真变了跟不上→升 lambda |
| `kpt_mem.recall_conf/conf_decay/max_missing_frames/valid_ms` | 0.55/0.35/5/800 | 短消失回忆与长丢失衰减 | 回忆太久→降 max_missing |
| `kpt_mem.min_frames/w_min` | 2/0.0 | 回忆前最少见过几帧 / 权重硬下限（0=纯软） | 保持 0 |
| `kpt_mem.q_gain_floor` | 0.35 | **质量分帧级增益下限**（q_mem=0 时保留的增益比例） | 1.0=关闭质量分对融合的影响（A/B 用） |
| `pnp.{reproj_px,z_min,z_max,refine}` | 8.0/0.2/15.0/true | PnP 内点阈值/深度范围/精化 | 近距错解→收紧 z 范围 |
| `pnp.max_z_jump_m` | 0.8 | 帧间 z 突变上限（超过 → 弃帧退化 coarse） | **满速冲门风险项，勿轻易放大** |

### 4.3 策略件（`comm.gate.quality` / `comm.gate.cues` / `comm.gate.stamp`）

| 参数 | 默认 | 含义 | 调参方向 |
|---|---|---|---|
| `quality.w_{kpt,pnp,box,time}` | 1/1/0.8/0.5 | Q 的四分量权重（设 0 即关闭该分量） | 想让位姿链主导→升 w_pnp/w_box |
| `quality.rms_ref_px` | 5.0 | q_pnp 的参考尺度（该 RMS 处 ≈0.5） | 据实测 RMS 分布定 |
| `quality.q_no_pose` | 0.25 | 无位姿时的 q_pnp（降级档给多少分） | 降级档表现好→升 |
| `quality.box_iou_floor` | 0.4 | 位姿投影框与检测框 IoU 下限（≤ → q_box=0） | 倒影拼框→升 |
| `quality.q_lo/q_hi` | 0.35/0.75 | Q → 谨慎度 c 的线性区间 | **需现场数据标定** |
| `quality.speed_min_scale` | 0.4 | 最谨慎时的速度保留比例（下限，不会停死） | 太慢→升 |
| `quality.confirm_max_scale` | 1.0 | >1 表示低质量时拉长确认时长 | 保持 1.0（本次设计不改确认） |
| `quality.deadzone_max_scale` | 1.6 | 低质量时对准死区最多放大倍数 | 别太大（会不修横向） |
| `quality.prior_conf_pow` | 1.0 | 先验里原始 conf 的幂（<1 更宽容） | 与 conf_thr 配套 |
| `quality.q_time_floor` | 0.2 | 帧新鲜度下限（过期帧仍参与但权重低） | 掉帧多→升 |
| `quality.mem_lambda` | 0.3 | q_mem（历史位姿证据）EMA 速率 | 大=更敏感，小=更迟钝 |
| `cues.enable` | true | 恢复线索总开关 | 定位问题先关 |
| `cues.record_only` | **false** | true=只记录不驱动 | **首次下水建议先 true** |
| `cues.speed` | 0.15 | 线索动作低速 | 仍会 × sp |
| `cues.cue_after_frames` | 15 | 位姿连续不可用多少帧后线索才接管 | **需现场数据** |
| `cues.cue_confirm_s/frames` | 0.10/3 | 线索动作确认（时间+帧数双条件） | 防单帧噪声；主导时可放宽 |
| **`cues.mode`** | **`auto`** | 主导权三档：`auto` 依据质量自动分配 / `pose` 位姿主导 / `cue` 线索主导 | 见 §4.4（旧名 `pose_first`/`blend`/`cue_first` 仍可用） |
| `cues.allow_approach` | false | 线索是否允许在 APPROACH 相位出力 | 主导时建议 true |
| `cues.cue_pose_share` | 0.0 | **仅 `cue` 档**：给位姿保留的份额（≤0.5） | 想让线索只做"修正"→0.2 |
| `cues.speed_lead` | 0.30 | **主导档**（auto/cue）线索速度（`pose` 档仍用 `speed`） | 太猛→降；太软→升 |
| `cues.speed_uses_caution` | false | 仅主导档：线索速度是否再乘谨慎度 | 保持 false（否则位姿链仍能拖慢线索） |
| `cues.lead_q` | 0.35 | **`auto` 档**：位姿 Q 低于它 → 线索权重开始 > 0 | 想更早交接→升到 0.5~0.6 |
| `cues.lead_full_q` | 0.20 | **`auto` 档**：位姿 Q 低于它 → 线索权重 = 1 | 需 < `lead_q` |
| `cues.conflict_w` | 1.0 | **auto/cue 档**：冲突时线索权重下限（0=听位姿，1=听线索） | 想保守→0.5 |
| `cues.lead_min_w` | 0.25 | 权重小于它就不掺和（防两链各出一半力来回抖） | 抖动→升 |
| `cues.{turn_*,descend_*,ascend_*,backward_*}` | true | 各方向线索开关 | 先只留确信的方向 |
| `cues.backward_agree_z_m` | 1.5 | 线索判"太近"但位姿 z 仍大于它 → 冲突 | 依实际探测距离 |
| `cues.agree_bonus/conflict_penalty` | 0.15/0.5 | 线索↔位姿一致时 Q 加成 / 冲突时打折 | 互检力度 |
| `stamp.max_age_s` | 0.9 | 帧龄上限（超过 → stale：可估计不可计数） | 依实测延迟 |
| `stamp.max_gap_s/adaptive_gap` | 0.35/true | 采集断点判据（自适应按帧间隔） | 卡顿多→升 |
| `stamp.future_tol_s` | 0.005 | 采集时刻"微未来"容差（取整误差） | 别动（动过会误判 future） |

### 4.4 主导权三档位：线索主导 / 位姿主导 / 依据质量自动分配

一帧里"线索链与位姿链谁说了算"由 `cues.mode` 决定，**默认 `auto`（依据质量自动分配）**：

| 档 | 名字 | 位姿在位 且 Q 高 | 位姿在位 且 Q 低 | 位姿不可用 | 冲突（线索↔位姿矛盾） | 线索速度 |
|---|---|---|---|---|---|---|
| **`auto`**（默认） | 依据质量自动分配 | 线索 0（位姿主导） | `0<wc<1` 连续交接 | 接管 | wc 抬到 `conflict_w` | `speed_lead`（不乘 sp） |
| `pose` | 位姿主导 | 0 | 0 | 接管 | **不参与**（只给 Q 打折） | `speed × sp`（= v1.3 行为） |
| `cue` | 线索主导 | `1-cue_pose_share`（不看 Q） | 同左 | 接管 | 同左 | `speed_lead` |

> 旧名仍被识别：`pose_first → pose`、`blend → auto`、`cue_first → cue`（见 `data/cues.py::cue_mode`）。

每帧实际交接权重 `wc = cue_weight(...)`，最终 DOF =
`位姿DOF × (1−wc) + 线索DOF × wc`（`blend_dof`，**逐通道**，所以线索没动到的通道仍由位姿修正）。
权重可在诊断里看到：`last_info["cue_w"]`（0=完全听位姿，1=完全听线索）与 `cue_mode`。

**怎么选**：

| 你的判断 | 选哪档 |
|---|---|
| 只信位姿，线索仅作"位姿彻底不可用"时的兜底（= v1.4 行为，最保守） | `pose` |
| 位姿平时可信；倒影/缺角/解算变差时希望线索自动接手（推荐起点） | **`auto`（默认）** |
| 明确认为"看得见的边"比位姿更可信，希望线索一有动作就说话 | `cue` |

**调参阶梯（围绕 `auto` → 越来越听线索）**：

| 想要的改变 | 调哪里 |
|---|---|
| 更早交接（位姿 Q 稍降就让线索参与） | `lead_q: 0.35 → 0.5~0.6`（越大越早） |
| 交接更"果断"（不要长时间各出一半力） | `lead_full_q` 靠近 `lead_q`（如 0.45） |
| 冲突时听线索 / 各让一半 / 仍听位姿 | `conflict_w: 1.0 / 0.5 / 0.0` |
| 一丢位姿就接管（不等 15 帧） | `cue_after_frames: 15 → 3` |
| APPROACH 阶段也允许线索出力 | `allow_approach: true` |
| 线索动作更有力 | `speed_lead: 0.30 → 0.35`；并保持 `speed_uses_caution: false` |
| 去掉动作确认延迟（噪声大时别关） | `cue_confirm_frames: 1`、`cue_confirm_s: 0.0` |
| 想"整帧听线索"而不是"线索只做修正" | 直接 `mode: cue`（配 `cue_pose_share: 0.0`） |
| 反过来收紧（线索只在无位姿时兜底） | `mode: pose` |

**为什么可以放心把主导权交给它**（测试里都钉住了）：
1. **线索动作集里没有"前进"**（只有 heave / yaw / 后退），所以线索主导天然偏保守：
   最坏是"该进不进 / 绕圈"，不会冲向门框；
2. 交接是**逐通道线性混合**（`blend_dof`），不是二值切换 → 主导权在几帧内平滑转移，
   不会在船上产生速度跳变；
3. 帧守卫与确认仍在：重复/过期帧（`!countable`）不计线索确认；THROUGH 相位永不介入；
   主导档同样要过 `cue_confirm_*`。

**现场怎么选**（建议顺序）：
```bash
python3 tests/test_gate_cue_lead.py      # 77 项：三档 / 权重曲线 / 混合 / 冲突 / 安全不变量
```
① 先用 `pose` 跑一段基线并记录 `cue` / `cue_w`（此时 `cue_w` 应为 0，`cue` 仍在记录）；
② 换 `auto` 再跑：看"倒影/缺角那几帧 `cue_w` 有没有升上去、升上去以后动作方向对不对"；
③ 若 `auto` 交接太晚 → 升 `lead_q`；若发现"位姿其实是对的、线索在乱带" → 回 `pose`
或把 `conflict_w` 降到 0.5；
④ 只有确认"可见性比位姿可靠"再上 `cue`。

**改参数的推荐姿势**：改完先跑 `tests/test_gate_layers.py` + `tests/test_gate_enhance.py`
+ `tests/test_gate_cue_lead.py`（行为类），涉及几何/阈值再跑 `tests/test_gate_flow.py` 与
`check_pipeline_identity.py`。

## 5. 实验待测项（⚠️ 需现场数据/水池）

按"风险从高到低"排序；每项都给了**怎么测**与**判据**。

| # | 待测项 | 现状 | 怎么测 | 判据 / 影响 |
|---|---|---|---|---|
| 1 | **线索方向符号** | 表驱动、未实测 | `cues.record_only: true` 下水，`preview_detect.py --gate-kpt` 记录 `visible`/`cue` 与画面 | 上边可见时应下降、下边可见时应上升、左列右转、右列左转；若反了，只改 `cues.*` 开关/或在 `_CUE_DOF` 调符号 |
| 2 | `quality.q_lo/q_hi` | 0.35/0.75（保守默认） | 倒影/逆光场景录帧，回放统计 Q 分布（`last_info["Q"]`） | 正常帧 Q 应 > q_hi（c≈1）；异常帧应落入 [q_lo,q_hi] 以下 |
| 3 | `cues.cue_after_frames` | 15（≈1.4s@11fps） | 人为遮挡角点，看多久该接管 | 太短→噪声误触发；太长→干等 |
| 4 | `kpt_mem` 标定：`k_sigma/r_min_px/recall_conf/conf_decay` | 保守默认 | `preview_detect.py --gate-kpt` 逐帧记录角点，离线回放 | 鬼点应 w≈0 而不丢点；短消失应回忆、长丢失应自然降级 |
| 5 | `pnp.max_z_jump_m` 与 `z.cross/near_lost_m` | 0.8 / 0.7 / 1.0 | 池中实测帧间 z 抖动与"门占满视野"距离 | 太严→频繁弃帧；太松→单帧错解冲门 |
| 6 | `surge.reacquire` 符号与量级 | 0.25（负 surge=后退） | 手动触发 REACQUIRE，看船是否真后退 | 后退方向反了会导致"越退越近" |
| 7 | **主导权档位选择** `cues.mode` / `lead_q` / `conflict_w` | `auto`（依据质量自动分配） | 同一段倒影/缺角轨迹跑 `pose`/`auto`/`cue` 三遍，比对 `cue_w`/`cue_mode`、相位时长与过门成功率 | 倒影期 `cue_w` 应自动上升；若位姿其实是对的而线索在乱带 → 退回 `pose` 或降 `conflict_w` |
| 8 | `cues.speed_lead` 量级 | 0.30（约 2× 从属档） | 池中从 0.2 / 0.3 / 0.4 各跑 | 太小→纠正不过来；太大→过冲/来回摆 |
| 9 | `quality.q_gain_floor` 手感 | 0.35 | 用 1.0（关）与 0.35（开）各跑一趟对比 | 开 = 位姿不可信时更信历史、更平滑；关 = 更快跟随 |
| 10 | 帧守卫阈值（`stamp.*`） | 0.9/0.35/自适应 | 板端记录 `age/gap/frame_kind` 直方图 | `stale/duplicate` 应接近 0；`discontinuity` 不该频繁 |
| 11 | 预处理提速（**唯一能真提速的地方**） | 现 640 拉伸链路 | 关掉增强/合并 resize 后测帧率（见 §6） | 单帧 ≈91ms 中预处理 ≈54ms，是主要开销 |
| 12 | 门框尺寸/偏移 `geometry.*` | 0.70×0.50/0 | 卷尺实测 | 直接影响 z 绝对精度 |
| 13 | 双门赛制 `pass_target=2` | 未验证 | 连过两门 | 第二门需要 `_start_search` 正常复位 |

## 6. 性能与帧率（11 fps 现状）

单帧 ≈91 ms：读图 18 + 预处理 54 + NV12 1.3 + BPU 12.4 + 解码 3.4（ms）。
结论（本模块相关）：

* **帧有效性检查几乎不拒帧**（稳定期 `stale/duplicate` ≈ 0），所以它**不会**带来提速；
  它的价值是"过期/重复帧不许计数"，防确认被凑数。
* **两分法不丢信息量**：过期帧照常参与估计（只是权重低、不计确认）。
* 真正能提速的只有**预处理**（54ms/帧）；换 1280 域推理反而更慢（域自证脚本会拦）。
* 质量分对融合的影响是**软**的：正常帧 `gain≈0.99`（≈无影响），只有证据差时才压低。

## 7. 故障排查

| 症状 | 原因 | 处置 |
|---|---|---|
| 启动打印"标定 … 不存在，用近似针孔" | `cfg/front_camera.yaml` 不在 `cfg/` 或标定分辨率≠实际帧 | 放好标定；跑 `check_pipeline_identity.py` |
| `main.py --task gate` 拒绝启动 | 门权重 `.bin` 缺失/路径错（`.hbm` 不兼容） | 确认 `vision.model.task_models.gate.path` |
| PnP 深度整体偏大/偏小 | 标定分辨率与实际帧不一致、或 `geometry.*` 尺寸填错 | 上述两项都查 |
| 画面里 `kpt` 远小于 `kpt_raw` | 正在靠融合/回忆撑住（倒影/缺角） | 正常现象；若长期如此，看 `visible` 与 `Q` |
| `gain` 长期 < 1 | 位姿链最近不可信（RMS 大/框不一致/帧过期） | 看 `rms`/`frame_kind`；必要时调 `q_gain_floor`/`rms_ref_px` |
| 频繁 REACQUIRE（反复后退） | 角点长期不足而框占比偏大 | 调 `coarse.near_ratio`/`hold.max_frames`/`reacquire.max_times` |
| 线索动作方向相反 | `cues` 方向符号未实测 | 见 §5 第 1 项 |
| 主导档下船来回摆/画龙 | 两链交替出力、或线索速度过大 | 升 `lead_min_w`、降 `speed_lead`、`cue_confirm_frames` 回 3 |
| `cue_w` 一直为 0（想让它主导时） | `mode` 仍是 `pose`、或位姿 Q 一直高于 `lead_q`、或当前相位不被允许（`allow_approach`） | 检查 `cues.mode` / `cue_mode` 诊断；必要时升 `lead_q`、开 `allow_approach` |
| 突然直接 THROUGH 冲出去 | 单帧 PnP 错解给出了 z≤cross | 确认 `z.cross_confirm_frames ≥ 2` 且 `pnp.max_z_jump_m` 未被放大 |

## 8. 变更记录（本目录）

* **v1.2**：角点逐点融合（`data/kpt_memory.py`）、REACQUIRE 限幅闭环、过门判据完善。
* **v1.3**：帧守卫（`common/frame_stamp.py`）、双条件确认（`common/streak.py`）、
  质量分→谨慎度、恢复线索（`data/cues.py`）。
* **v1.5**：线索链**主导权三档位**（`cues.mode`）——
  **`auto` 依据质量自动分配（默认）** / `pose` 位姿主导 / `cue` 线索主导。
  `auto` 按位姿质量 Q 连续交接权重（`lead_q`/`lead_full_q`），冲突时抬权到 `conflict_w`，
  主导档速度 `speed_lead` 不再被谨慎度拖慢，`allow_approach` 可让 APPROACH 也参与，
  `cue_pose_share` 可让"线索主导"时给位姿留份额；诊断新增 `cue_w`/`cue_mode`。
  ⚠️ **默认档从 v1.4 的"线索仅兜底"变为 `auto`**（位姿差/冲突帧线索会接手）。
  自检：`tests/test_gate_cue_lead.py`（77 项）。
* **v1.4**：本目录分层（vision/data/motion + 顶层 task/mock）、质量分改为**两条去向**
  （先验→融合权重/增益；后验→谨慎度/q_mem）、`Sighting` 感知半场、
  `comm.gate.quality/cues/stamp` 显式写入 yaml；`gate_task.py` 800 → 426 行。
  相关自检：`tests/test_gate_layers.py`（53 项）。
