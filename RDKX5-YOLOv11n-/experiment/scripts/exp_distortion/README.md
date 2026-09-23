# 畸变实验：畸变对水下门框特征点识别的影响

目标：把「去畸变」这一步在链路里的位置当作自变量，看它对门框 4 角点
（`TL,TR,BR,BL`）识别几何质量的影响，并回答**当前板端模型兼容哪几种情况**。

- 数据集与权重都在本仓库（`RDKX5-YOLOv11n-`），不在板端工程里；
- 所有 python 脚本跑在 conda `yolov8`：
  `/home/ansty/anaconda3/envs/yolov8/bin/python`；
- 参数唯一来源是本仓库 `configs/vision.yaml` 与 `configs/front_camera.yaml`
  ——它们是**板端工程的只读镜像**（见 `configs/README.md`）。

**GPU**：训练与模型推理用 `--device 0`（RTX 4060 Laptop 8 GB，实测
`torch.cuda.is_available() == True`）。本目录里 `prepare_distortion_groups.py`、
`transform_pose_labels.py`、`make_contact_sheet.py`、`calib_diagnose.py` 都是
OpenCV/NumPy 的像素与几何运算，**本来就没有 GPU 路径**；只有
`compare_kpt_spacing.py` 带 `--model` 时会走 ultralytics 推理，那里默认
`--device 0`。GPU 显存小，训练用 `--batch 4`、`--workers 2`（8 个 worker 会与
CUDA fork 死锁）。

> 若某次 `torch.cuda.is_available()` 为 `False`，先看 `/dev/nvidia*` 在不在：
> DSH 的文件沙箱在受限策略下会挂一个**不含 GPU 设备的私有 `/dev`**，此时是沙箱
> 挡的，不是环境坏了（本会话实测：策略放开后同一个解释器立刻 `True`）。

---

## 1. 三个组（+ 一个虚拟组）

| 组 | 图目录 | 图像链路 | 模型看到的域 | 输出坐标域 | 要不要单独标注 |
|---|---|---|---|---|---|
| **A** 先去畸变 | `A_undistort_first/` | `remap@720p` → `resize640` → `enhance` | 去畸变域 | `a640` | 是（**推荐只标这一套**） |
| **B** 不去畸变 | `B_no_undistort/` | `resize640` → `enhance` | 畸变域 | `b640` | 可由 A 换算 |
| **C** 最后去畸变（链路末端） | `C_undistort_last/` | `resize640` → `enhance` → `remap@640` | 去畸变域 | `c640` | 可由 A 换算 |
| **C′** 最后去畸变（坐标后映射） | 复用 **B 的图** | 检测在 B 上做 → 输出点 `undistortPoints` | 畸变域 | `a640` | 不需要（用 B 的标注） |

A 就是**板端当前链路**（`common/preprocess.py`）；所以 A 组的结果等价于
"现在板子上跑出来是什么样"。

### 一条要写进结论的推导

去畸变是逐像素坐标映射、`resize` 是线性映射，两者在坐标上可交换：

```
S ∘ undistort(K, newK720)  ==  undistort(S·K, S·newK720) ∘ S      S = diag(640/1280, 640/720)
```

`getOptimalNewCameraMatrix(alpha=0)` 取「去畸变后有效区域」的外接矩形，该区域
在此仿射变换下也只被 `S` 缩放，于是 `newK640 == S·newK720`（实测差 0）。

> **结论：A 与 C 是同一套几何，只差重采样顺序。** C 回答的是「去畸变放在链路
> 哪一段更划算（精度/耗时）」，不是「去畸变算得对不对」。实测两者像素差
> 均值 ~2 灰阶 / 最大 ~21 灰度 / 超过 8 灰阶的像素 <2%（AUV_4 真实帧）。

⚠️ **板端是各向异性缩放**：`1280×720 → 640×640` 的 `S = diag(0.5, 0.889)`，
不是等比；所以 640 域的等效内参是 `K_sq = S·K`（`dist` 不变，它作用在归一化
坐标上）。写错这一点会让 C 组的几何全错。

---

## 2. 现状核对（2026-09-22 实测，跑之前先看这段）

### 2.1 `configs/front_camera.yaml` 只在画面中心可信 ❗

用 `calib_diagnose.py` 体检现行标定的结果：

| 项 | 数值 |
|---|---|
| 畸变模型 `r_d = r_u(1+k1r_u²+k2r_u⁴+k3r_u⁶)` 的极大 | `r_d_max = 0.664`（在 `r_u=0.962`） |
| 可逆的原始像素半径上限 | **≈ 520 px** |
| 画面四角到主点的半径 | **753 px** |
| B 组画面里"存在去畸变解"的像素占比 | **63.5 %** |
| 棋盘观测到的归一化半径（`data/AUV_1/board`，1496 角点） | p50 0.245 / p95 0.451 / **max 0.636** |

两件事同时成立：

1. 归一化半径 > 0.664 的原始像素**在模型下没有去畸变解**。`cv2.undistortPoints`
   在这里会静默返回垃圾（实测：原始像素 (0,360) 返回它自己，(80,360) 跳到 -525 px）
   ——所以本目录的代码**不用** `cv2.undistortPoints`，改用解析畸变模型 + 向量化
   牛顿法求逆，并且**带有效标志**（无效点必须丢弃，不许当坐标用）。
2. 棋盘只覆盖到画面半径的 66%（**中心区域**），外圈的畸变参数是**多项式外推**。
   把外推结果拿去做去畸变，效果是**过度校正**（枕形），见
   `experiment/runs/exp_distortion/preview/grid_undistort.jpg`：合成栅格去畸变后外圈向内弯。

**这对实验的影响**：A/C 组的外圈（左右边缘 + 四角，约 36% 画面）结论**不成立**。
处理办法：

- 评估时用 `compare_kpt_spacing.py --valid-only`，把结论限定在可信区域内，
  并在报告里写「外圈未验证」；
- 或者重新下水标定：棋盘必须出现在画面**四角与边缘**（`calib_diagnose.py --board`
  会告诉你现在覆盖到多少）。

**换畸变模型不是修复手段**。同一批 35 视图重标对比（2026-09-22 实测）：

| 模型 | RMS | `r_d_max` | 外圈表现 |
|---|---|---|---|
| 5 参数（现行 yaml） | 0.705 px | 0.664 | 外圈无解；A 画面覆盖 63.5% 像素 |
| 8 参数 rational | 0.704 px | 412 | "可逆"了，但**外圈畸变更离谱**，A 画面只剩 **1.9%** |

5 参数重标复现出与现行 yaml **逐位相同**的系数（`-0.49136, 0.37844, -0.00911,
0.00397, -0.22619`），说明标定流程本身没问题。rational 多出的 k4~k6 在本就
无数据约束的外圈自由发挥，肉眼可见更差（`experiment/runs/exp_distortion/preview/grid_5coef_vs_8coef.jpg`）。

> **`r_d_max` 大只代表"数学上有解"，不代表"校得对"。** 判据要同时看
> 「可逆」与「棋盘覆盖」，缺一不可。想修外圈只能重拍棋盘。

### 2.1b 局部放大倍率：**这条决定了实验能不能判定** ❗

去畸变的局部放大倍率 `|d(a640)/d(b640)|`（`calib_diagnose.py` 会打印）：

| 到主点半径（b640 px） | 像素占比 | 倍率中位 | p90 | 最大 |
|---|---|---|---|---|
| 0–50 | 1.9 % | 1.01 | 1.02 | 1.03 |
| 50–100 | 5.8 % | 1.04 | 1.08 | 1.11 |
| 100–150 | 9.5 % | 1.11 | 1.20 | 1.27 |
| 150–200 | 13.5 % | 1.23 | 1.42 | 1.59 |
| 200–230 | 9.8 % | 1.39 | 1.69 | 1.99 |
| 230–260 | 10.5 % | 1.52 | 2.15 | 2.92 |
| 260–290 | 7.5 % | 1.59 | 2.41 | 4.13 |
| 290–330 | 4.7 % | 1.65 | 2.69 | 6.34 |

倍率 ≤1.2 只覆盖 B 画面的 **24 %**（半径 ≤254 px）；≤1.3 覆盖 34 %；≤1.5 覆盖 47 %。

**这条把实验的可判定性摊在明面上**：

- 中心区（半径 <150 px）倍率 ≈1.0~1.2 → **A 组去畸变几乎就是恒等变换**。
  只用中心样本，三组必然"看起来差不多"，**测不出畸变的影响**；
- 畸变真正起作用的是外圈（倍率 1.5~6×）→ 但那正是本标定**没数据、且已确认
  过度校正**的地方（`coverage_confound.jpg` 的 frame_000001：B 里正常的门框，
  在 A 里被放大到几乎出框）。

⇒ **用现行 `configs/front_camera.yaml` 回答不了"畸变对门框特征点识别的影响"**：
算得对的区域没有畸变可测，有畸变的区域算得不对。

**解法只有一个：重新下水标定棋盘，让棋盘出现在画面四角与边缘**（可分多次下水、
多个距离与倾角）。之后本目录的脚本**只换 `--calibration` 一个参数**即可重跑整套实验。

### 2.2 当前 pose 模型

`weights/yolo11n-pose.pt`（2026-09-17）是在**去畸变域**的 AUV_4 数据集
（`data/AUV_4/PNP.kpt4.yolov8`，1 类 `gate`、4 点、顺序 `TL,TR,BR,BL`）上训的，
也就是 **A 域**。所以：

- "检查当前模型能否兼容三种情况" = 直接拿它跑 A/B/C 三组，比较角点几何一致性，
  **不需要重新训练**；
- 预期 A 最稳、B 会因为域不匹配而退化；C 应当接近 A（同一几何，只差重采样）；
  C′ 取决于"检测在畸变域被拉坏多少"与"坐标换算的精度"哪个更划算。

---

## 3. 操作流程

### 步骤 0 · 挑原始帧

AUV_1/2/3 的原始帧已按 ~1fps 切好（`--step 25`）：

```
data/AUV_1/rec_front_frames/                     709 张
data/AUV_2/auv_20260910_172208_frames/           995 张
data/AUV_3/auv_20260911_201754_frames/           508 张
data/AUV_4/auv_4_frames/                       10118 张（全量）
```

> 注意 AUV_2/3 是裸 mjpeg 且**没有 `.timestamps`**，所以只能用 `--step`，
> 不能用 `--every-seconds`（会直接报错）。

用接触印相表挑图（几千张一眼扫完）：

```bash
PY=/home/ansty/anaconda3/envs/yolov8/bin/python
$PY experiment/scripts/exp_distortion/make_contact_sheet.py \
    data/AUV_2/auv_20260910_172208_frames data/AUV_3/auv_20260911_201754_frames \
    --out experiment/runs/exp_distortion/sheets/pick.jpg --tile 200 --cols 8 --per-sheet 96
```

产物旁边有 `.csv`（格子 → 文件名）。把挑中的帧**复制**到
`data/exp_distortion/picked/<下水名>/`（一个目录一次下水，目录名会变成输出文件前缀）。

挑图口径：门框四角都在画面内；**不同画面位置**（中心/偏左/偏右/偏上/偏下）与
不同距离都留几张 —— 畸变误差是径向的，只有偏心样本才测得出来。

### 步骤 1 · 一次生成三组 640×640

```bash
$PY experiment/scripts/exp_distortion/prepare_distortion_groups.py \
    data/exp_distortion/picked/auv1 data/exp_distortion/picked/auv2 \
    data/exp_distortion/picked/auv3 data/exp_distortion/picked/auv4 \
    --out-root data/exp_distortion/processed \
    --config configs/vision.yaml
```

产物：

```
data/exp_distortion/processed/
├── A_undistort_first/<下水>__<原名>.jpg       ← 板端链路（与 prepare_frames.py 逐字节一致）
├── B_no_undistort/<下水>__<原名>.jpg
├── C_undistort_last/<下水>__<原名>.jpg
└── experiment.json                            ← 参数/几何/自检/源文件映射（唯一参数源）
```

**同一帧在三组里同名**，后面按帧对齐做配对统计才不会错位。
自检通过会打印：解析畸变模型 vs `cv2.projectPoints` ~4e-16、
`newK640 - S·newK720` = 0、解析逆解 vs remap 表 0.02 px、A 域 vs C 域 0.02 px。

### 步骤 2 · 标注（**只标 A 一套**）

把 `processed/A_undistort_first/` 传 RoboFlow，1 类 `gate`、4 点、顺序 `TL,TR,BR,BL`，
导出 **YOLOv8 Pose**。理由：A 的直线是直的，角点最好对齐；而且 A 是板端的目标域。

然后把这一套标注换算到 B / C：

```bash
$PY experiment/scripts/exp_distortion/transform_pose_labels.py \
    --experiment data/exp_distortion/processed \
    --labels-in /path/to/rf_export/train/labels --roboflow \
    --src-group A --dst-group B --labels-out data/exp_distortion/labels_B
$PY experiment/scripts/exp_distortion/transform_pose_labels.py \
    --experiment data/exp_distortion/processed \
    --labels-in /path/to/rf_export/train/labels --roboflow \
    --src-group A --dst-group C --labels-out data/exp_distortion/labels_C
```

脚本会报告**关键点有效率**（无效 = 超出标定可逆范围 / 落到目标画面外 → 写 `v=0`）。
有效率低说明该组结论不成立，不要硬用。

> 想三套各自独立人工标注也行（测的是"人眼在哪种几何下标得更一致"），
> 但两种口径别混着用，报告里要写清是哪种。

**这套换算已经验证过**（用现有 `data/AUV_4/PNP.kpt4.yolov8` 的 A 域标注，
挑出与 A 组图逐像素相同的帧）：

- A↔B 往返 0.014 px；A 域 vs C 域 0.016 px；
- 独立验证：在 A 图关键点取 21×21 小块，到 B 图预测位置 ±25 px 内做模板匹配，
  **NCC 中位 0.84、预测点与匹配峰值距离中位 1.0 px（最大 4.2 px）** —— 说明换算
  后的点确实落在同一物理角点上，不是"方向搞反了但往返自洽"。

### 步骤 3 · 评估（相邻关键点间距是否一致）

```bash
# ① 用当前模型跑三组（原版 head；脚本会自己检查并给恢复命令）
$PY experiment/scripts/exp_distortion/compare_kpt_spacing.py \
    --experiment data/exp_distortion/processed \
    --group A --group B --group C --group "C'" \
    --model weights/yolo11n-pose.pt --device 0 \
    --out-dir experiment/runs/exp_distortion/eval_model

# ② 用人工标注跑（测几何本身，不测模型）
$PY experiment/scripts/exp_distortion/compare_kpt_spacing.py \
    --experiment data/exp_distortion/processed \
    --group A --group B --group C --group "C'" \
    --labels-root data/exp_distortion/labels --out-dir experiment/runs/exp_distortion/eval_gt
```

判读（尺度无关，**不要跨组比像素长度**，A/C 经 `alpha=0` 整流后整体缩放）：

| 指标 | 含义 | 期望 |
|---|---|---|
| `e_opp` | `max(\|s0-s2\|/(s0+s2), \|s1-s3\|/(s1+s3))`，四条相邻边（= 门框四边） | 越接近 0 越规整 |
| `r_tb`、`r_lr` | 上/下、左/右边长比 | 随姿态缓变，不应随"门框在画面里的偏心程度"漂 |
| `corr(ρ)` | `e_opp` 与「门框中心到主点距离」的皮尔逊相关 | **最有诊断价值**：畸变没校正干净时显著为正（外圈更差），正确整流后 ≈0 |
| 配对表 | 同一帧两组相减，`frac_x_better` | 判断哪组更稳 |
| `scale_ratio_median` | 两组 `s_mean` 之比 | 只用来解释 `alpha=0` 的缩放 |

产物：`experiment/runs/exp_distortion/eval*/{summary.json, pairwise.csv, metrics_*.csv}`。

> ⚠️ 用**换算出来的**标注跑 ② 时，B/C/C′ 与 A 的差异**完全由坐标变换决定**，
> 所以那一轮只能证明"几何换算自洽"，判断不了哪组更好。要判好坏必须跑 ①（模型）
> 或者三套独立人工标注。这一点别在报告里搞混。

### 步骤 4 · （可选）每组单独训练

若要把结论从"当前模型兼容性"推进到"哪种链路值得部署"，就按组各训一版：
三套数据集分别用 `resplit_dataset.py`（序列感知）划分 →
`train_yolo11n.py --task pose --data <data.yaml> --epochs 300 --batch 4 --imgsz 640 --device 0`。
注意 `prepare_frames` 已经把图做成 640×640，校准集那步的 letterbox 是 no-op。
训练是长任务：用后台 job 跑，别在前台干等。

---

## 4. 文件清单

| 文件 | 作用 |
|---|---|
| `distortion_geometry.py` | 三组/四域互转的唯一实现；解析畸变模型 + 牛顿法求逆 + 有效性标志；`python 它` 直接跑几何自检 |
| `prepare_distortion_groups.py` | 一次生成 A/B/C 三组 640×640 + `experiment.json` |
| `transform_pose_labels.py` | 跨组标注换算（含 RoboFlow 文件名归一化），标 1 套推 3 套 |
| `compare_kpt_spacing.py` | 相邻关键点间距一致性评估 + 配对比较 + 径向相关性 |
| `calib_diagnose.py` | 标定体检：可逆性 / 可逆像素占比 / 棋盘覆盖半径 / 重标模型对比 |
| `make_contact_sheet.py` | 挑图用的接触印相表 |

## 5. 已知坑（都是本次实测踩到的）

1. **`cv2.undistortPoints` 在强畸变下会静默返回垃圾**（5 次不动点迭代不收敛）。
   本目录一律走 `distortion_geometry.undistort_pixels()`，它同时返回 `valid`。
2. **`undistortPoints` 的签名是 `(src,K,dist,dst,R,P)`**：新内参是第 6 个参数
   `P`，写成第 5 个会被当成整流矩阵 `R` 而**静默出错**。
3. **无效点不筛会得到几百 px 的假误差**：变换有两类失效（模型不可逆、目标点
   落到目标画面外），只判一类就会把假误差算进结论。
4. **`undistort_pixels` 返回的是 720p 整流域坐标**，转 `a640` 必须再乘 `S`；
   漏乘会让"是否在画面内"判错。
5. **跨组不要比绝对像素长度**：`alpha=0` 的整流会整体缩放（实测 A/B 尺度比中位
   1.09，偏心越远越大）。用 `e_opp`/`r_tb`/`r_lr` 这类尺度无关量。
6. **AUV_2/3 的裸 mjpeg 没有 `.timestamps`**，`extract_frames.py --every-seconds`
   会直接失败，只能用 `--step`。
7. **现有 `data/AUV_4/PNP.kpt4.yolov8` 的帧号不能可靠回推 `auv_4_frames` 的帧号**：
   抽查 12 帧里 7 帧与 A 组图逐像素一致、5 帧与 A/B/C 都不匹配（应另有来源）。
   拿这个数据集做验证时，先确认图能对上再用。
8. **别把 `(0,0)` 当坐标**：YOLO 对没把握的关键点输出 `(0,0)`，
   标注里未标的点也是 `(0,0)`。直接算边长会得到 `0`（两点重合），
   把"模型不确定"伪装成"几何很规整"。评估脚本默认 `--kpt-conf 0.5`
   只保留四点全可信的实例，并把丢弃数单独报一列（**丢弃率本身就是兼容性信号**）。
9. **A/C 的画面比 B 小**：同一标定下 A/C 只覆盖原始画面中央，所以 B 能检到的门
   有一部分**根本不在 A/C 画面里**。跨组比检出率必须加 `--restrict-mappable`，
   否则 A/C 白背"检出低"的锅。

---

## 6. 本次实跑记录（2026-09-22，32 张挑选帧）

输入 `experiment/runs/exp_distortion/data_raw/`（32 张 1280×720，全部与原始帧**逐字节相同**，
来源：AUV_4 18 / AUV_3 5 / AUV_2 5 / AUV_1 4，逐张 md5 溯源见同目录
`_provenance.csv`）→ 输出 `experiment/runs/exp_distortion/processed/{A_undistort_first,
B_no_undistort,C_undistort_last}/`（各 32 张，三组同名）。

用 `weights/yolo11n-pose.pt`（`--device 0`）跑四组的**初步**结果
（`experiment/runs/exp_distortion/eval_model/`，`--restrict-mappable --kpt-conf 0.5`）：

| 组 | 检出实例 | 四点全可信 | 低可信丢弃 | e_opp 中位 | e_opp 均值 | 配对：左边更好占比 |
|---|---|---|---|---|---|---|
| A | 38 | 22 | **16** | 0.0102 | 0.0168 | A−B 42 % / A−C 33 % |
| B | 21 | 21 | 0 | 0.0151 | 0.0161 | — |
| C | 22 | 22 | 0 | 0.0116 | 0.0174 | C−C′ 89 % |
| C′ | 21 | 21 | 0 | **0.0326** | 0.0421 | — |

读法（**样本只有 32 帧 / 20~22 个实例，属指示性，不作结论**）：

- **C′ 明显最差**（中位 e_opp 0.033 vs 其余 0.010~0.015，p90 0.083）。
  机理：检测在畸变域做，坐标再经非线性映射，而映射在接近可逆边界处
  局部倍率发散 → **把几个像素的检测误差放大成几十像素的几何误差**。
  它省掉一次整幅 remap 的算力，代价是几何精度 —— 值得在板端算力账上权衡。
- A/B/C 三组的中位差别很小（0.010~0.015），与 2.1b 的预期一致：
  **这批帧里门框大多落在中心区，那里去畸变本来就约等于恒等变换**。
- A 组"低可信丢弃 16/38"看着刺眼，但它混了选择偏差：`--restrict-mappable`
  把 B/C 的外圈难例剔掉了，A 的外圈难例还在。**要下结论必须用人工标注**，
  三组各标一遍后 A/B/C 的实例是一一对应的，就没有这个问题。

