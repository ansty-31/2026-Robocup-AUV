# 门框「去畸变方案」确定实验 · 协议（针对新内参 C）

> 目的：判定 **① 去畸变该不该有 ② 放在哪一步 ③ enhance 能不能去掉**，
> 输出一套可部署的预处理链路 + 每 ms 换多少精度的账。
> 所有结论都必须能落到「改哪个文件、跑哪条命令、看哪个数」。

> **执行环境（2026-09-23）：本地做精度迭代，板端连着做速度/位置验证。**
> 板子在线（`sunrise@192.168.137.10`，实机工程 `/home/sunrise/Desktop/AUV_New`），
> 用 `bench_undistort_placement.py` 随时上板量位置差异；**精度类实验一律本地**，
> 避免每次改动都上板。板端最终验收（端到端 FPS / 实际精度）仍不可省。

---

## 0. 已确定的事实（**不要重测**）

| 项 | 结论 | 证据 |
|---|---|---|
| 标定 | **C 就位**（fx=1207.64，RMS 0.546px）；A 归档 `configs/backup/`；B 保留 `configs/front_camera_air.yaml` | `configs/README.md` 换版记录 |
| 网格直线度 | C 中位 **0.39px** / p90 0.49 vs A 中位 1.86 / **p90 307** | 180 张棋盘图 |
| 可用范围 | C 可逆半径 749px（角点 798）、93.7% 像素可去畸变；A 是 520px / 63.5% | `calib_diagnose.py` |
| **画面覆盖** | **C 覆盖原始画面 99.8%，A 只有 70.4%**（A 把外圈裁掉了） | 纯几何，见本文 §4 |
| 板端速度 | `remap@720p` 11.46 / `remap@640` 6.02 / `resize` 2.95 / **`enhance` 39.99** / NV12打包 0.31 ms | `bench_undistort_placement.py`，X5 实测 |
| 四方案耗时 | P1 **54.4** / P2 49.3 / P3 43.1 / P4 43.4 ms/帧（→ 18.4 / 20.3 / 23.2 / 23.0 FPS） | 同上 |
| DDR 流量 | P1 22.1 / P2 10.8 / P3=P4 8.3 MB/帧 | 估算，@30fps 664→249 MB/s |
| 球检测 | 换 C **不劣化**：A 检到的 C 保留 91.4%(blue) / 99.2%(red)，conf 差 ≤0.013 | 1000 帧配对 |
| 各下水可用率 | 随机 120 帧/下水：AUV_1 10% / AUV_2 47% / AUV_3 42% / AUV_4 37% / **AUV_5 35%** | `enh_ablation` 口径 |

> ⚠️ **不要再用"手工挑的门框帧"和"随机帧"互相比较** —— 前者是富集样本，
> 可用率会虚高 2–3 倍（本工程踩过：32 张手挑帧给出 81%，同口径随机帧只有 10–47%）。

---

## 1. 三问与因子设计

| 问题 | 判据 | 需要什么 |
|---|---|---|
| **Q1** PnP 几何上需不需要去畸变？ | 门距/方位的**真值误差**（卷尺） | 人工标注 + 真值，不需要训练 |
| **Q2** 模型上需不需要？ | 交叉矩阵上同域模型的外圈精度 | **重训 2–3 版** |
| **Q3** 放哪里 / enhance 留不留？ | 端到端时延 + 同域精度 | 上板测量 + 重训 |

**因子**
* 去畸变位置：`P1` 720p remap → `P2` 640 remap → `P3` 只映射 4 点 → `P4` 无
* enhance：开 / 关
* 训练域 × 推理域（交叉矩阵）
* **分层变量 ρ**（门框中心到主点的归一化距离）：`内 <0.3` / `中 0.3–0.5` / `外 >0.5`
  —— 畸变效应是径向的，**不分层就看不出差别**（本工程踩过：样本 ρ 中位只有 0.14，三组打平）

> ⚠️ **ρ 必须取「四角中最大的那个径向距离」，不是框心**（2026-09-23 实测纠正）。
> 门框本身占画面约 1/3：框心 ρ≈0.30 时远角已经到 ρ≈0.60。
> 按框心分层会把畸变应力低估约 2 倍，并且"内圈"档会显得很多（其实那些帧的角点全在外圈）。
> 实测（四角齐全的帧）：**角点最大 ρ 的中位 = 0.55–0.65，各下水一致**。
> 因此分层阈值要用**数据驱动的三分位**（本批 p33=0.565 / p67=0.688），不要拍脑袋定 0.3/0.5。

> ⚠️ **"门框出画"会把某类样本整体筛没**（2026-09-23 实测）。按"框完整在画面内"筛：
> AUV_4 300 帧里合格 68 帧，**AUV_5 只有 3 帧**（208/300 帧门框贴边或出画）。
> 也就是说**清水下门框更靠边** —— 这本身是一条结论，也解释了为什么"清水一来就废"：
> **A 标定的去畸变只覆盖画面 70.4%，靠边的门框被整体裁掉，模型根本看不见。**
> E1/E2 的挑帧判据要用「**四角关键点在画面内**」（而不是框不出画），否则 AUV_5 会被筛空。

**不变量**：imgsz=640；JPEG q95；同一批帧三组同名；除被测因子外全部保持一致。

---

## 2. E1 · 几何必要性（不训练，半天，先做）

**为什么先做**：`cv2.solvePnP(obj3, img2, K, D)` **本来就吃畸变系数**（板端
`gate/geometry.py:15` 写着 `undistort=false → 原始 K + 原始 D`）。所以 PnP 的**正确性**
不依赖去畸变，只需量清楚**精度损失**。这一步单独就能否掉/坐实"为了 PnP 去畸变"。

```bash
# 1) 备料：AUV_5 门框帧，刻意挑偏心/靠边的（ρ 各档都要有），~40 张
python experiment/scripts/exp_distortion/make_contact_sheet.py <帧目录> --out <sheet>   # 印相表辅助挑图
python experiment/scripts/exp_distortion/prepare_distortion_groups.py <挑图目录> \
    --out-root experiment/runs/exp_distortion/e1 --config configs/vision.yaml

# 2) 只标【畸变域】一套（B 组），RoboFlow: 1 类 gate / 4 点 / TL,TR,BR,BL
# 3) 真值：卷尺量门距 + 门心方位，写进 experiment/runs/exp_distortion/e1/truth.csv
```

**两条 PnP 路径（同一条标注、同一套门框 3D 尺寸）**

| 路径 | 输入 | 板端对应 |
|---|---|---|
| N1 不去畸变 | `solvePnP(obj3, pts_B, K, D)` | `undistort: false` |
| N2 去畸变 | `pts_B →(C 映射)→ pts_A`，`solvePnP(obj3, pts_A, K_new, 0)` | `undistort: true` |

**判据**：`ρ 档 × 两条路径` 的深度误差 / 方位误差（对标卷尺真值）。
* 外圈两列相当 ⇒ **PnP 不需要去畸变**，去畸变的必要性全落到模型侧
* 外圈 N2 明显更好 ⇒ 必须去畸变，且**必须先在 720p 上做**（P1）

> ❌ **不要**用 `e_opp` 的绝对值判定：门框在透视下本来就不是矩形（对边天然不等），
> `e_opp` 混淆了「透视收缩」和「镜头畸变」。它只能做**同帧配对**比较。

---

## 3. E2 · 模型必要性 + enhance 消融（重训，折进你本来要做的重训）

**同数据、同超参、同 epoch，只换预处理**，训这几版（按预算裁剪）：

| 版本 | 去畸变 | enhance | 回答 |
|---|---|---|---|
| `M_P1E` | P1 | 开 | **当前链路基线** |
| `M_P3E` | P3（只映射点，训练时仍需整幅去畸变 → 用 P2 训练、P3 推理） | 开 | 去畸变位置 |
| `M_P4E` | 无 | 开 | 模型侧去畸变有没有用 |
| `M_P1N` | P1 | **关** | enhance 有没有用（你新加的问题） |

**训练集必须包含 AUV_5**（否则测出来的是水质的锅，不是方案的锅）。
**评估：3×3（或 4×4）交叉矩阵**，每个模型在每个域上评 —— 非对角项直接量化
「换标定/换链路会让现有权重掉多少」。

指标（全部按 ρ 分层）：
* 四点全可信率（**丢弃率本身就是兼容性信号**）
* `e_opp` 同帧配对差
* 走 E1 的 **PnP 真值误差**
* 端到端时延

```bash
python experiment/scripts/exp_distortion/compare_kpt_spacing.py \
    --experiment experiment/runs/exp_distortion/<name> \
    --group A --group B --group C --group "C'" \
    --model weights/<该域模型>.pt --device 0 --kpt-conf 0.5 \
    --restrict-mappable --valid-only \
    --out-dir experiment/runs/exp_distortion/eval_<name>
```

> ⚠️ 修过的坑：`compare_kpt_spacing.py` 在「boxes 为空但 keypoints 非空」的帧上会
> `IndexError`，已加长度守卫；`(0,0)` 陷阱靠 `--kpt-conf` 过滤（默认 0.5）。

---

## 4. E3 · 位置与速度（**已完成**，X5 实测）

见 §0 表。**额外一条与模型无关的几何事实**（自查命令见文末）：

> **A 的去畸变输出只覆盖原始画面 70.4%**（remap 采样源 x∈[147,1137]、y∈[34,689]），
> **C 覆盖 99.8%**。旧标定把画面外圈裁掉了 —— 这是"换 C 后小球检出反而 +23~29%"的原因，
> 不是 C 视野变窄。

**怎么读这张表**：
* 预处理 **43–54 ms**，而 BPU 推理只要 **6.8 ms** ⇒ **真正瓶颈是 enhance（40 ms），不是去畸变位置**
* P1→P3 省 **11.3 ms（+4.8 FPS）**，DDR 流量降到 **1/2.7**
* P3 ≈ P4（差 0.3 ms，噪声内）⇒ **只映射 4 个角点几乎免费**

---

## 5. 快速检测流程（**全部在本地跑**，5 分钟一轮）

> **约定：后续实验都在本地做。** 板端只用于「一次性实测」（§4 已完成）与最终验收，
> 不作为迭代回路的一环 —— 每改一次预处理就上板，循环成本不可承受。

```bash
PY=/home/ansty/anaconda3/envs/yolov8/bin/python

# ① 标定体检（3 秒）：可逆性 / 放大倍率 / 棋盘覆盖
$PY experiment/scripts/exp_distortion/calib_diagnose.py \
    --calibration configs/front_camera.yaml --board <棋盘目录> \
    --cols 11 --rows 8 --square-mm 20 --out runs/<x>/diag.json

# ② 位置与速度（本地相对成本；绝对 ms 以上板实测为准，见 §4）
$PY experiment/scripts/exp_distortion/bench_undistort_placement.py \
    --board-root /home/ansty/RDKX5/auv_vision/auv_vision \
    --calibration configs/front_camera.yaml \
    --frames <720p 帧目录> --n 60 --out runs/<x>/e3_local.json

# ③ 几何 / 模型评估（1 分钟）
$PY experiment/scripts/exp_distortion/compare_kpt_spacing.py \
    --experiment <experiment 目录> --group A --group B --group C \
    --model <权重> --device 0 --restrict-mappable --out-dir <out>
```

**本地 vs 板端各自能回答什么**

| 量 | 本地 | 必须上板 |
|---|---|---|
| 精度（四点可信率 / `e_opp` / PnP 真值误差） | ✅ 唯一来源 | 仅最终验收 |
| 相对耗时（P1:P2:P3:P4 比值） | ✅ | |
| 绝对耗时 / FPS / DDR 带宽 / BPU 子图 | ❌ | ✅ |

**本地基线**（x86 单线程，60 张 AUV_5 帧；**只作相对参照**）

| 阶段 | PC ms | 板端 ms（§4） | 慢多少倍 |
|---|---|---|---|
| `remap@1280x720` | 0.509 | 11.46 | 22× |
| `resize 720p→640` | 0.134 | 2.95 | 22× |
| `enhance @640` | 7.274 | 39.99 | **5.5×** |
| `remap@640` | 0.386 | 6.02 | 16× |

> ⚠️ **别用 PC 比值推断板端占比**：板端 `enhance` 只慢 5.5×，其余算子慢 16–22×，
> 所以它在板端的相对占比（74%）比在 PC 上（94%）低。**位置/方案的取舍要用 §4 的板端表。**

> 🚫 **本地 `bench_undistort_placement.py` 测不出位置差异。** 实测（同一批帧）：
> P1 7.235 / P2 7.749 / P3 7.483 / P4 7.486 ms —— 四者全在噪声内，P1 甚至看着最快。
> 原因：PC 上 `enhance`(7.27ms) 把总耗时吃干，而 x86 的内存系统让多出来的 720p remap
> （0.51ms）与 DDR 流量几乎不可见。**所以本地跑这个脚本只用来确认"链路没写错"，
> 位置结论一律以 §4 的板端实测为准**（那里 P1 54.4 → P4 43.3，差异 20.8%，清晰可测）。

**验收线（照这个判，不要凭感觉）**
* 四点可信率：同域模型应 ≥ 现有基线；跨域掉 >20% 视为不可部署
* `e_opp` 外圈：同帧配对 Δ 显著为正 ⇒ 该组合外圈几何不可用
* 端到端：预处理 ≤ 15 ms 才谈得上 30 FPS

---

## 6. 判定表（把三问合成一个决定）

| E1 外圈 PnP | E2 同域模型 | E4 enhance | 结论 |
|---|---|---|---|
| 与去畸变相当 | 三域相当 | 影响小 | **P4 全去掉**（PnP 吃 D）；enhance 也去掉 → 预处理 ≈ 3 ms |
| 相当 | 去畸变域明显更好 | 需要 | **P2**（640 上 remap）+ 保留 enhance |
| 去畸变明显更好 | — | 需要 | **P1**（720p 上 remap，先做，避免缩放后再补）+ 保留 enhance |
| 任一格"enhance 影响大" | — | — | enhance **不能去掉**，转去优化它的实现（见下） |

**若 enhance 必须保留**，优化顺序（当前 40 ms，占 P1 的 74%）：
1. 白平衡合并进 gamma LUT（三次乘法 → 一次查表）
2. CLAHE 是分块局部算子，最贵；评估 `clahe_clip=0` 的代价（它本来就是"轻微" 0.5）
3. 用 NEON/`cv2.UMat` 或把三步合成一次 pass，避免多次全图读写

---

## 7. 交付物与回退

**产物**
* `experiment/runs/exp_distortion/<name>/experiment.json` — 参数/几何自检/源文件映射（唯一参数源）
* `experiment/runs/exp_distortion/eval_<name>/` — summary.json + pairwise.csv + metrics_*.csv
* 板上 `/tmp/e3_*.json`

**回退**
* 标定：`cp configs/backup/front_camera_AUV1_water_fx782.yaml configs/front_camera.yaml`
  （板端同理改 `cfg/`，两侧必须一起改）
* 换链路：改回 `cfg/vision.yaml` 的 `image.undistort`

**纪律**
* `auv_vision/auv_vision`（板端本地副本）与实机 `/home/sunrise/Desktop/AUV_New` **不是同一份**：
  实机是权威，本地副本只读（`tests/` 除外）。改配置先改实机 → 再同步 `configs/`。
* 每改一次预处理，**必须重跑 §5 的三条**，把数字贴进结论。

---

## 8. 执行现状与命名对照（2026-09-23 更新，**以本节为准**）

本协议写于实验设计阶段；实际执行时域命名与判据有更新，**新口径见
`experiment/runs/domain/EXPERIMENT_DESIGN.md`**（含预先登记的判定阈值 §13）。对照表：

| 本协议旧称 | 现在的域名 | 板端方案 | 图像链路 |
|---|---|---|---|
| 「去畸变（720p）」 | **C 域** | P1 | `remap@720p → resize(640) → enhance` |
| 「去畸变（640）」 | **D 域** | P2 | `resize(640) → enhance → remap@640` |
| 「不去畸变」 | **B 域** | P4 | `resize(640) → enhance` |
| （基线） | **A-old 域** | 部署中的现状 | 同上 P1，但标定用旧的 A（fx 782.5） |

### 已被 E1 回答（不需要再测）

- **几何上「去畸变放哪里」是恒等变换**：`solvePnP(obj,K,D)` ≡ 先去畸变再
  `solvePnP(obj,newK,0)`。48 帧实测 \|Δdepth\| 中位 **0.0007%**（外圈分层 0.0009%）。
  ⇒ 位置（P1/P2）不影响 PnP 精度，只影响板端耗时与 DDR。
- 补充实测（2026-09-23）：C 域与 D 域的**像素差只有 1.89/255**，标签差 0.0000 px
  ⇒ P2 与 P1 在几何上等价（因为协议定义 `nk640 = S·nk720`）。

### 新增的独立发现

- **标定 A 与 C 对同一相机的几何分歧达 59%**（门距 1.12 m vs 1.79 m），两者各自内部
  自洽。用部署中模型自己的预测点判定：C 几何下门框矩形一致性提高约一倍
  （W 误差 1.24%→0.66%、H 2.37%→1.56%）、reprojRMS 6.06→3.61 px
  ⇒ **A 是错的那个**，部署中的门距偏小约 37%。详见 `experiment/runs/domain/EXPERIMENT_DESIGN.md` §12。
- **A-old 去畸变会裁掉视场**：test 129 帧里有 6 个 GT 可见角点被打出画面（4 帧），
  B/C/D 均为 0；清水下恶化到 16 帧里 6 帧。

### 数据来源已可 100% 回溯

1304 张标注帧已全部溯源到裸流原始帧（`runs/prov/provenance_final_PNP.kpt4.yolov8.csv`），
可重投影到任意域而**无需重新标注**。关键事实：**标注文件名里的编号就是裸流帧序号**；
`data/AUV_x/*_frames` 是「每 25 帧取 1 帧 + 重编号」的残片，不能当全量帧用。
工具见 `scripts/1_prepare/provenance/` 与 `map_pose_dataset.py`。
