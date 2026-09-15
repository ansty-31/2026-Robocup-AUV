# gate v1.4/v1.5 — 分层、质量分的两条去向、主导权三档位

> 关系：`doc/算法说明-gate-PnP移植方案.md` 是**移植总体方案**（几何/前端/PnP/相位机）；
> 本文讲 **v1.4–v1.5 两次策略层更新**：目录分层、质量分 Q 的两条去向（v1.4），
> 以及"线索链 ↔ 位姿链"主导权的三档位（v1.5）。
> 参数逐项说明与实验待测项见 `gate/README.md`。

## 1. 为什么要改

v1.3 里质量分 Q 只被拿去调**运动谨慎度**（速度缩放 / 死区缩放）。这浪费了一半信息：
角点融合层（`data/kpt_memory.py`）其实最需要它——倒影帧、过期帧、位姿链解算不可信的帧，
本就不该按正常增益去拉融合状态。

同时 `gate_task.py` 涨到 800 行：配置装配、帧守卫、感知链（检测→融合→mode→线索→PnP→
质量分）、相位机、REACQUIRE、诊断全挤在一个文件里，改一处要看全局。

v1.4 因此做两件事，**都不动底层运动**（相位机、速度阶梯、确认计数、REACQUIRE 语义不变）：

1. 质量分 Q 变成"一帧证据质量"，按发生时刻分两半，各去该去的地方（§3）；
2. 目录分三层（视觉处理 / 数据处理 / 运动决策），顶层只留装配与仿真（§2）。

## 2. 分层与依赖规则

```
gate/
├── gate_task.py   相位机骨架（唯一对外入口 GateTask）
├── mock.py        仿真后端（脚本化进近/穿门）
├── vision/        视觉处理：gate_decode / gate_detector / gate_frontend / geometry / perception
├── data/          数据处理：kpt_memory / quality / cues
└── motion/        运动决策：phases / phase_degrade / phase_recover
```

**依赖单向**（下层不知道上层）：

```
data（只用 numpy） ← vision ← motion ← gate_task（装配）
```

三条取舍，写在这里免得以后反复：

* **相位段用 mixin 而不是组合**：`phase_degrade.py` / `phase_recover.py` 里的方法读写的是
  *同一个* `GateTask` 状态（相位、计数器、诊断、PID、`_sp/_dz`）。拆成独立类只会变成
  `self.task._G`、`self.task._hold_cnt` 式的绕路，可读性更差。所以
  `class GateTask(DegradeTicks, RecoverTicks)`，**方法体与拆分前逐行一致**（只搬文件）。
* **`cues.py` 放 `data/`**：它是"从可见性数据里提出线索 + 做线索↔位姿互检"，被两侧消费
  —— 运动层拿它驱动（`motion/phase_recover.py`），质量分拿它打折（`data/quality.py`）。
  放 `data/` 才能保持"两个消费者都在上层"的单向依赖。
* **`perception.py` 放 `vision/`**：它是感知半场的编排（像素→几何量），虽然会调用
  `data/` 的融合与质量分，但产出是"这一帧看到了什么"（`Sighting`），属于视觉处理的出口。

外部接口**未变**：`GateTask(uart, hub, w, h)`、
`process(frame, now_ms[, frame_id, captured_at])`、`ready/reset_state/last_info/last_dets`；
两参旧调用行为不变；`from gate.gate_task import PH_APPROACH, SUB_REACQUIRE` 等常量仍可用
（骨架把 `motion/phases.py` 的常量原样再导出）。

## 3. 质量分的两条去向（本次核心）

### 3.1 因果顺序

```
原始conf + 帧龄 ──①先验 prior──► kpt_memory 逐点权重 w_prior + 帧级增益 gain
                                        │
                                  融合角点 → mode/可见性/线索 → PnP + RMS
                                        │
                    ②后验 posterior ────┴─► 运动谨慎度 c（速度×sp / 死区×dz）
                                        └─► q_mem（历史位姿证据 EMA）──► 下一帧 ①
```

**先验必须在融合之前算**：它只能看"这一帧自己看得见的东西"（原始角点置信度）与"这帧多旧"
（帧龄），不看位姿结果——否则就是拿结果去决定过程，等于没有因果。

### 3.2 ① 先验：进融合权重与增益（`data/quality.py` + `data/kpt_memory.py`）

```
q_time        = clip(1 - age/max_age_s, q_time_floor, 1)          # 帧新鲜度（有下限）
w_prior[i]    = clip(conf_i, 0, 1) ** prior_conf_pow × q_time     # 逐点先验权重
w[i]          = clip(w_prior[i] × w_geo[i], 0, 1)                 # w_geo = 1/(1+(d/R)²)
gain          = q_gain_floor + (1 - q_gain_floor) × q_mem         # 帧级增益
α_eff, β_eff  = α×gain, β×gain                                    # 只调增益，不硬判
```

* `w_prior` 传入 `KptMemory.update(..., w_prior=...)` 时**取代**原来的 `conf_i` 因子
  （`w = w_prior × w_geo`，不重复相乘）；不传时退化为 `conf_i`，与升级前**逐位一致**。
* 语义：这一帧这个点**值多少权重**（点级）+ 这一帧**整体该多信观测**（帧级）。
* 两路都只"少拉"，**不做硬判**：点仍有效、状态不丢（与 `kpt_memory` 的"不硬判无效"一致）。
* `q_gain_floor = 1.0` 即完全关闭质量分对融合的影响（A/B 对照用）。

### 3.3 ② 后验：进运动谨慎度与历史证据

```
q_kpt   = 0.5·(可见数/4) + 0.5·平均置信度          # 融合后的角点状态
q_pnp   = 1/(1+(rms/rms_ref_px)²)                  # 重投影质量；无位姿 → q_no_pose
q_box   = 位姿投影框 vs 检测框 IoU 的归一化        # 位姿链 ↔ 线索链互检的那一条
q_time  = 帧新鲜度
Q       = Σ wᵢ qᵢ / Σ wᵢ                           # 四分量加权
Q'      = Q + agree_bonus（线索与位姿一致）/ Q × conflict_penalty（冲突）
c       = clip((Q' - q_lo)/(q_hi - q_lo), 0, 1)     # 谨慎度
sp      = speed_min_scale + (1-speed_min_scale)·c   # 速度缩放（有下限）
dz      = 1 + (deadzone_max_scale-1)·(1-c)          # 死区/对准阈值放大
q_mem   = (1-mem_lambda)·q_mem + mem_lambda·Q'      # 只在**有位姿**的帧上更新
```

* **Q 不参与过门判定**；`confirm_max_scale = 1.0` 时也不改确认时长（只改"给多少权重/开多快"）。
* `q_mem` 只在有位姿的帧更新：位姿链一时解不出来**不代表观测变差**，无位姿帧保持不惩罚；
  否则降级档会把融合越压越钝（实测过：会拖慢角点重现后的恢复）。
* 为什么互检放在质量分：线索判"太近"而位姿说还很远 → 至少一个错，此时**不驱动、只降权**，
  这条正是"两条链互相自检"的落点（`data/cues.py → quality_agreement`）。

### 3.4 软耦合：可关闭、且不影响成败

* 正常帧（Q≈0.99）→ `gain≈0.99`，与 v1.3 路径几乎无差；
* `tests/test_gate_layers.py` 的软耦合用例：默认配置与 `q_gain_floor=1.0`（关）各跑一遍 mock
  端到端，**都是 249 帧 DONE(pass)**。

## 4. 与参考实现 `gate_pnp` 的关系

只借鉴了**思路**，没有引入它的结构与硬闸门：

| gate_pnp 的做法 | 本项目的做法 |
|---|---|
| conf/RMS/歧义/尺度一律**硬拒**（拒帧） | 折成连续质量分 Q → 软权重/软增益（少信而非丢弃） |
| 640 域内解算（`K640 = A·K_full`） | 1280 校正域解算，域一致性由 `check_pipeline_identity.py` 自证 |
| 独立的 ACQUIRE 阶段 | 恢复线索与降级链**互相自检**（不新增相位） |

## 5. 验证

| 自检 | 覆盖 |
|---|---|
| `tests/test_gate_layers.py`（53 项） | 分层归属、先验随帧龄/conf 变化、先验真的进融合、帧级增益、旧行为逐位一致、`q_mem` 只吃有位姿帧、后验→谨慎度单调、真实任务 Q/Qmem/gain 动态、软耦合、配置键与标定路径回归 |
| `tests/test_gate_enhance.py`（45 项） | 帧守卫（唯一帧/过期帧两分法/微未来容差）、双条件确认、质量分、线索接管、端到端 DONE(pass) |
| 其余 16 套 | 相位机、几何、融合、接线、ball 未受影响 |
| `check_pipeline_identity.py` | 预处理/坐标域/标定一致（`pipeline_id` + sha256） |

一次跑全套：`for f in tests/test_*.py; do python3 $f || echo FAIL $f; done`

## 6. 本次暴露的两个工程坑（已修，留作教训）

1. **`__file__` 相对路径写死深度**：`vision/gate_detector.py` 原来用
   `dirname(dirname(__file__))` 找 `cfg/`，从 `gate/` 挪到 `gate/vision/` 后深度变了 →
   静默退化成"近似针孔"相机（fx 从标定的 782.5 变成 0.61×1280=780.8，主点从 633.8 变成 640）
   → 像素域整体偏移 → 测试里的宽档从 HOLD 变成 CREEP。
   现在改为**向上找含 `cfg/` 的那一层**（`_project_root()`），不再依赖文件深度。
   教训：路径推断要么基于配置根，要么显式向上搜索；测试要断言"标定确实被加载"。
2. **批量改写 import 误伤配置键字符串**：把 `gate.quality` → `gate.data.quality` 的批量替换
   同时改掉了字符串里的 `comm.gate.quality`（变成 `comm.gate.data.quality`），使配置查找
   *静默*回落到代码默认值。现在 `tests/test_gate_layers.py` 会断言
   `comm.gate.quality / cues / stamp` 三个键确实取到 dict（防再次静默失效）。
   教训：批量替换要排除点号路径出现在**配置键**里的情形；"默认值=现值"会让这种错误延迟暴露。

## 7. 仍未定（需现场数据）

线索方向符号、`q_lo/q_hi`、`cues.cue_after_frames`、`q_gain_floor` 手感、
`pnp.max_z_jump_m` 与 `z.cross/near_lost_m` 实测、`kpt_mem` 参数标定。
完整清单（含测法与判据）见 `gate/README.md` §5。

## 8. v1.5：主导权三档位（线索 ↔ 位姿）

### 8.1 问题

v1.3/v1.4 里线索链是**从属**的：位姿在位时一律不让它驱动，只在"位姿连续不可用 ≥
`cue_after_frames` 帧"时低速接管；互检（`quality_agreement`）只能给质量分打折/加成，
**不能**改变"这一帧由谁出力"。于是有一个隐含假设：*位姿永远比"哪条边可见"可信*。
倒影场景里这个假设经常不成立（倒影会拼出假框、把 PnP 带偏），而"只看到上边"这种
可见性信息反而更直接。v1.5 把这件事变成**可配的三档**。

### 8.2 三档定义（`cues.mode`，默认 `auto`）

| 档 | 名字 | 位姿在位 且 Q 高 | 位姿在位 且 Q 低 | 位姿不可用 | 冲突 | 线索速度 |
|---|---|---|---|---|---|---|
| **`auto`**（默认） | 依据质量自动分配 | 0 | `0<wc<1` | 1 | `max(wc, conflict_w)` | `speed_lead`（不乘谨慎度） |
| `pose` | 位姿主导 | 0 | 0 | 1 | 不参与（只给 Q 打折） | `speed × sp`（= v1.3） |
| `cue` | 线索主导 | `1-cue_pose_share` | 同左 | 1 | 同左 | `speed_lead` |

```
wc = cue_weight(cfg, q_pose=Q, cue=动作, pose_ok, z)          # 主导权重 ∈ [0,1]
auto:  Q ≥ lead_q → 0 ；Q ≤ lead_full_q → 1 ；中间线性
冲突:  cue=BACKWARD 且 z > backward_agree_z_m → wc = max(wc, conflict_w)
DOF  = pose_dof × (1-wc) + cue_dof × wc                       # 逐通道混合
```

* **为什么默认 `auto`**：位姿链在正常情况下（Q≈0.99）权重为 0 → 与 v1.4 完全一致；
  只有当证据变差（RMS 大 / 框不一致 / 帧过期）或两链矛盾时才自动把份额交给线索。
  即"平时不插手，出问题才接班"，不需要人工判断什么时候该信谁。
* **为什么 `pose` 档不参与冲突仲裁**：该档的语义就是"位姿说了算"，冲突时只保留
  v1.3 的"Q 打折"（少信一点），不改变出力方 —— 这样它是干净的回退档（等于 v1.4 行为）。
* **逐通道混合而不是"谁赢谁全出"**：交接发生在几帧内，二值切换会在船上产生速度跳变；
  混合还天然实现了"线索没动到的通道仍由位姿修正"。
* **`cue` 档的 `cue_pose_share`**：给位姿保留一份（默认 0 = 完全接管）。注意线索动作里
  没有前进（surge>0），所以 `wc=1` 意味着"这一帧不前进、只按线索调姿/后退"——保守。

### 8.3 安全性质（测试里固化为断言）

1. **线索动作集永不含前进**：`cue_dof()` 的 5 个动作 surge ≤ 0 → 线索主导不可能冲向门；
2. **交接平滑**：`blend_dof` 逐通道线性、结果截断 ±1；
3. **帧守卫不放松**：重复/过期帧（`!countable`）不计线索确认；THROUGH 相位永不介入；
   主导档同样要过 `cue_confirm_s/frames`；
4. **互检照旧**：冲突仍给位姿 Q 打折（`conflict_penalty`），一致仍加成（`agree_bonus`）——
   打折与"谁主导"是两套独立机制（前者动质量分，后者动出力分配）。

### 8.4 诊断与调参

* `last_info["cue_w"]`：本帧线索权重（0=完全听位姿，1=完全听线索）；
  `last_info["cue_mode"]`：当前档位。两者配合 `cue` / `action` / `Q` 曲线即可判断
  "该它出手时它出手了吗、方向对不对、有没有来回抖"。
* 抖动的典型原因：两链交替出力（升 `lead_min_w`）或线索速度过大（降 `speed_lead`）；
  交接太晚：升 `lead_q`；太早：降 `lead_q`。
* 现场对比顺序、逐项参数含义与判据见 `gate/README.md` §4.4 与 §5。

### 8.5 与 v1.4 的差异（需要注意）

| | v1.4 | v1.5 |
|---|---|---|
| 默认行为 | 线索只在位姿不可用时兜底 | **`auto`：位姿差/冲突帧线索自动接手** |
| 冲突（BACKWARD vs z 大） | 只给 Q 打折，动作仍由位姿发 | `auto`/`cue` 档把主导权给线索（后退，安全方向） |
| 线索速度 | `speed × sp` | 主导档 `speed_lead`（不乘 sp） |
| 相位 | SEARCH/ALIGN | 可配 `allow_approach` |

想回到 v1.4 行为：`cues.mode: pose`（一行配置，其余键不必动）。
