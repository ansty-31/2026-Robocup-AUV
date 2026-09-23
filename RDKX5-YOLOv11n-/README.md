# RDK X5 · YOLO11 训练与部署工程

地平线 RDK X5 上的 YOLO11n 端到端工程：**数据准备 → 训练 → 导出 ONNX → PTQ 量化 → 板端部署**。
覆盖两条任务线：

| 任务 | 模型 | 输出头 | 用途 |
|---|---|---|---|
| **detect** | YOLO11n | 6 张量（3×bbox + 3×cls） | 红球 / 蓝球 / 门 检测（单权重多类别） |
| **pose** | YOLO11n-pose | 9 张量（每尺度 bbox+cls+kpt） | 门 4 角点（TL,TR,BR,BL）→ 板端 PnP 测距 |

> 环境：conda `yolov8`（`ultralytics==8.3.0`）+ OE v1.2.8 Docker 镜像（`bayes-e`）。

## ★ 运行环境（每次开始前先做这一步，勿再询问）

**所有 Python 脚本（`scripts/1_prepare` `2_train` `3_export`）都必须跑在 conda `yolov8` 环境里**，
系统自带的 `python3` 没有 torch / ultralytics，直接用会报 `ModuleNotFoundError`。

```bash
conda activate yolov8
```

- 交互式终端：先 `conda activate yolov8` 再执行脚本；
- 非交互式（脚本 / Jenkins / 自动化）：用绝对路径解释器，效果等价：
  `/home/ansty/anaconda3/envs/yolov8/bin/python scripts/2_train/train_yolo11n.py ...`
- 该环境已确认可用 **GPU**（`torch 2.3.1+cu118`，`device 0` = NVIDIA RTX 4060 Laptop 8GB），
  训练默认 `--device 0`；`nvidia-smi` 可用即说明 GPU 正常：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> 若某次调用发现 `torch.cuda.is_available() == False`，先确认 `/dev/nvidia*` 设备节点是否存在
> （驱动重载/休眠唤醒后偶发缺失），再决定是否降级到 CPU（`--device cpu`，约 112 秒/epoch，慢 20 倍以上）。

---

## 目录分区

> 整理时间 2026-09-23（晚：实验内容全部移入 `experiment/`）。顶层**九个区**（`configs` `weights`
> `scripts` `docs` `data` `raw-data` `runs` `output` `experiment`）加一个**归档区** `_archive/`；
> 散落文件已归位（原仓库根目录的 `yolov8n.pt` → `_archive/root/`）。
> **`experiment/` = 只为「去畸变时机 / enhance」那一个问题服务的一切**（脚本+对照数据+结果），
> 常规流水线不引用它 —— 分类索引见 [`experiment/README.md`](experiment/README.md)。
> 归档规则与恢复方法见 [`_archive/README.md`](_archive/README.md)。

```
RDKX5-YOLOv11n-/
├── README.md                  ← 本文件（项目总览 / 分区导航 / 全流程命令）
├── LICENSE · requirements.txt · .gitignore
├── configs/                   配置（全部集中在此，索引见 configs/README.md）
│   ├── vision.yaml            ★ 板端图像链路参数镜像（预处理与推理一致性唯一来源）
│   ├── front_camera.yaml      ★ 当前生效标定（AUV_5 水下棋盘，即"标定 C"）
│   ├── front_camera_air.yaml  空气/岸上标定（台架调试用，下水前必须切回上一份）
│   ├── yolo11n_config.yaml    detect 模型 PTQ 量化配置
│   ├── gate_kpt_config.yaml   pose 模型 PTQ 量化配置
│   └── backup/                历史标定留档（只读，勿删；见 configs/README.md）
├── weights/                   权重与模型（★ 分区说明见 weights/README.md）
├── scripts/                   脚本按阶段分三区，**只放常规流水线**（★ 索引见 scripts/README.md）
│   ├── 1_prepare/             训练前：数据准备（分类详见 scripts/README.md）
│   │   ├── (根) extract_frames · calibrate_camera · prepare_frames
│   │   │        · select_frames · resplit_dataset          ← 共用 / 常用
│   │   ├── pose/               ★ 门（4 角点）专用
│   │   │   ├── prepare_pose_dataset.py  Roboflow pose 导出 → 可训练集（kpt_shape/flip_idx）
│   │   │   ├── map_pose_dataset.py      跨去畸变域换算标注 + 生成 pose 训练集
│   │   │   │                            （`Domain` 类也是各域渲染的公共库）
│   │   │   └── make_mixed_gate_set.py   跨水质挑门图待标（新一版 selected_3000，内容去重）
│   │   └── provenance/         素材溯源（已闭合，换数据集可复用）
│   │       └── materialize_raw · provenance_{match,stream,nn,merge}.py
│   ├── 2_train/               train_yolo11n.py（--task detect|pose）
│   └── 3_export/              modify_ultralytics · export_onnx · prepare_calibration
│                              · quantize.sh · test_decode_parity
├── experiment/                ★ 实验专区（脚本+对照数据+结果，不在常规流水线上）
│   ├── README.md              分类索引：里面每个目录装什么、怎么重建
│   ├── scripts/               exp_distortion/（域对比主目录）· board/（板端对照）
│   │                          · train_domain_arms.sh · dedup_ceiling.py
│   ├── data/                  6 个对照数据集 pose_{B,C,D}{,_noenh}（各 1302 张，逐帧同划分）
│   ├── runs/                  domain/（域实验全部结果，主报告 EXPERIMENT_REPORT_20260923.md）
│   │                          · exp_distortion/（更早的畸变实验）· prov_intermediate/
│   └── logs/                  实验日志
├── docs/                      文档（教程 / 介绍 / 贡献指南）
│   ├── tutorial_zh.md         完整部署教程（含踩坑记录）
│   ├── README_ZH.md           中文项目介绍
│   ├── README_EN.md           English overview
│   └── CONTRIBUTING.md
├── data/                      数据集与素材（不进库）
│   ├── AUV_1 … AUV_5/         各次下水的素材与数据集
│   │   └── AUV_5/             selected_3000_Dwb/（待标图）· label_candidates_Dwb/（150 清水待标）
│   ├── mapped/                跨域映射出的可训练集
│   │   ├── raw/               映射前的原始帧（1303，provenance 的锚点）
│   │   └── pose_D_wb/         ★ 定稿生产集（D 域 + WB 无 CLAHE，1302）——下一步重训用这份
│   │                          （B/C/D 及 noenh 实验对照集已移入 experiment/data/）
├── raw-data/                  ★ 素材抢救区：坏卡雕出的裸流（GB 级，不进库）
├── runs/                      训练与实验输出（不进库）
│   ├── prov/                  溯源表 provenance_*.csv（流水线依赖）
│   ├── auv5_eval/             清水 16 帧回归评估集 · calib_auv5/ 标定记录
│   ├── domain/ exp_distortion/  ← 已移入 experiment/runs/
│   └── …                      其余为各阶段产物
├── output/                    量化产物 *.bin（不进库，部署时拷到板端 models/）
└── _archive/                  归档区：暂时不用但保留（不进库）
    ├── README.md              逐项说明「是什么 / 为什么在这 / 怎么恢复」
    ├── MOVES.tsv              机器可读搬移清单（含内容指纹，可据此校验恢复）
    ├── root/ · output/ · data/ · runs/ · ide/   按原位置镜像存放
    └── orphan_pyc/            ⚠️ 源文件已消失、仅存字节码的三个脚本（唯一副本，勿删）
```

### 各区状态（一眼看清哪里在动、哪里是留档）

| 区 | 状态 | 说明 |
|---|---|---|
| `scripts/` `configs/` `weights/` `docs/` | **在用** | 随版本迭代，纳入 git |
| `data/AUV_1..5` `data/mapped` `raw-data/` | **在跑** | 当前实验的输入素材与数据集，勿动 |
| `experiment/` | **在跑** | 去畸变/增强实验的全部脚本、对照数据与结果；可整体删除而不影响流水线 |
| `runs/prov` `runs/auv5_eval` `runs/calib_auv5` | **在跑** | 流水线依赖的溯源表、回归评估集、标定记录 |
| `output/*.bin` | **在用** | 当前部署产物（`gate_kpt_*` = 09-17，`yolo11n_detect_*` = 09-12） |
| `configs/backup/` | **留档** | 历史标定，只读 |
| `_archive/` | **留档** | 暂时不用但保留；含唯一副本，删除前先读其 README |

### 上一轮派生图：**保留在原位，不清理**（2026-09-23 决定）

下面这些是**上一轮的派生图**（不是原始素材）。**决定：一个都不删、位置不动** ——
它们仍可被脚本或人工引用，删掉只会让"当年这批图是怎么筛出来的"变得不可查。
本表只作**备案**：哪天真的缺空间，按「回退方式」一列重生成即可。

| 路径 | 大小 | 是什么 | 真删了的回退方式 |
|---|---|---|---|
| `data/AUV_4/selected_3000/` | 446M | AUV_4 当年的待标池（3353 张），标注在 `data/AUV_4/PNP.kpt4.yolov8/` | 从 `data/AUV_4/auv_4_frames/` 重抽 |
| `data/AUV_2/selected_2000/` | 193M | AUV_2 旧筛图产物 | 从 `*_frames/` 重抽 |
| `data/AUV_3/selected_2000/` | 251M | AUV_3 旧筛图产物 | 同上 |
| `data/AUV_1/dataset/AUV_data.zip` | 451M | 与同目录解压结果重复 | 解压目录还在 |
| `data/AUV_1/dataset/AUV.yolov11.zip` | 212M | 与 `AUV.yolov11/` 重复 | 同上 |
| `data/AUV_1/board/` | 432M | 标定 A / 空气标定的**输入照片**（结果已存 `configs/backup/`） | ⚠️ **无法复算**那只标定，只能重拍棋盘 |

合计 ≈ **1.98G**。**当前不回收。**

**分区原则**：`scripts/` 只放可执行脚本并按"训练前 / 训练 / 导出量化"三段分区；
所有配置进 `configs/`；权重与模型产物进 `weights/`；说明性文档进 `docs/`；
数据与产物不进库；"暂时不用"的一律进 `_archive/` 而不是直接删。

---

## 全流程命令（按阶段）

### 阶段 0 · 环境
```bash
conda activate yolov8                     # 必须！ultralytics 8.3.0 / torch 2.3.1+cu118（GPU 可用）
python -c "import torch; print('CUDA:', torch.cuda.is_available())"   # 期望 True；False 时先查 /dev/nvidia*
docker load < docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz   # 只需一次
```

### 阶段 1 · 训练前（数据准备，`scripts/1_prepare/`）
```bash
# 1) 切帧：容器视频 / 裸 MJPEG 流（自动识别；裸流无损切割）
python scripts/1_prepare/extract_frames.py data/AUV_2/auv_xxx.mjpeg --every-seconds 1

# 2) 相机标定（棋盘 11×8、方格 20mm；同一相机一次，动过相机必须重标）
python scripts/1_prepare/calibrate_camera.py data/AUV_1/board \
    --cols 11 --rows 8 --square-mm 20 --output configs/front_camera.yaml --save-views qc/

# 3) 预处理：去畸变 → 640×640 → 画面补偿（顺序同板端；参数取板端同一份 vision.yaml）
python scripts/1_prepare/prepare_frames.py data/AUV_2/<分类目录> --config configs/vision.yaml

# 4)（可选）筛图：用已有模型剔除不要的类别 + 按质量加权抽样
python scripts/1_prepare/select_frames.py data/AUV_2/datay/origin \
    --weights weights/yolo11n.pt --drop-classes red_ball \
    --quality-dir data/AUV_2/origin --keep 2000 --out-dir data/AUV_2/selected_2000

# 5) 标注（RoboFlow）：检测框 或 门 4 角点（规范见下）

# 6) 从 RoboFlow 拿回 pose 导出后，**必做两步**（否则 kpt_shape / 划分都会出问题）
python scripts/1_prepare/pose/prepare_pose_dataset.py data/AUV_4/PNP.yolov8 --dry-run   # 先看要改什么
python scripts/1_prepare/pose/prepare_pose_dataset.py data/AUV_4/PNP.yolov8             # → PNP.kpt4.yolov8
python scripts/1_prepare/resplit_dataset.py data/AUV_4/PNP.kpt4.yolov8             # 序列感知划分
```

> **⚠️ `weights/yolo11n.pt` 的 `gate` 类目前等于没有输出**：在 AUV_1 valid 集（152 张）上，
> 即便把 `conf` 降到 0.05，`gate` 类最高置信度仍是 **0.00**，28 个门标注框在 IoU@0.5 与 @0.3 下命中均为 **0**；
> 同一次评估 `blue_ball` 86/99、`red_ball` 76/92 都正常。所以**门检测不要用这个权重**，
> 用 `weights/yolo11n-pose.pt`（AUV_4 PnP 门 4 角点模型，2026-09-17 训练，val Pose mAP50-95 0.952）。
> 下一轮训练前请先核对训练数据集的 `names` 顺序、以及 `gate` 标签是否真的进了那一版——
> 一个类别头完全没有输出，通常不是"数据不够"，而是标签或类别顺序的问题。

### 阶段 2 · 训练（`scripts/2_train/`）
```bash
# 检测（自动恢复原版 head，训练后自动重打 detect 补丁）
python scripts/2_train/train_yolo11n.py --data data/AUV_1/dataset/AUV.yolov11/data.yaml \
    --epochs 300 --batch 4 --imgsz 640 --device 0 --cache ram

# 关键点（gate 4 角点；默认预训练 weights/yolo11n-pose.pt，训练后重打 pose 补丁）
python scripts/2_train/train_yolo11n.py --task pose --data <roboflow导出>/data.yaml \
    --epochs 300 --batch 4 --imgsz 640 --device 0
#   best.pt 自动复制为 weights/yolo11n.pt 或 weights/yolo11n-pose.pt
```

### 阶段 3 · 导出与量化（`scripts/3_export/`）
```bash
# 检测：Detect 头 → 6 输出
python scripts/3_export/modify_ultralytics.py --task detect
python scripts/3_export/export_onnx.py                        # → weights/yolo11n.onnx
# 校准集必须与 detect 训练集同分布，且单独放 calibration_data_detect/（见下方“校准集分区”）
python scripts/3_export/prepare_calibration.py \
    --coco-path data/AUV_2/AUV.yolov11/train/images --num-images 300 \
    --output-dir calibration_data_detect                      # → calibration_data_detect/*.rgb
./scripts/3_export/quantize.sh                                # → output/yolo11n_detect_*.bin

# 关键点：Pose 头 → 9 输出（4 点 → kpt 通道 12）
python scripts/3_export/modify_ultralytics.py --task pose
python scripts/3_export/export_onnx.py --task pose            # → weights/yolo11n-pose.onnx
python scripts/3_export/prepare_calibration.py \
    --coco-path data/AUV_4/PNP.kpt4.yolov8/train/images --num-images 200   # 默认写 calibration_data/
./scripts/3_export/quantize.sh configs/gate_kpt_config.yaml   # → output/gate_kpt_*.bin

# 换过解码/导出方式后，用真图对拍"板端预处理+解码"与训练侧链路（需原版 head）
cp <site-packages>/ultralytics/nn/modules/head.py.backup .../head.py
python scripts/3_export/test_decode_parity.py --n 30
python scripts/3_export/modify_ultralytics.py --task pose     # 跑完记得补回补丁版
```

> **校准集分区（勿混用）**：`quantize.sh` 现在从配置里的 `cal_data_dir` 读校准目录。
> detect 用 `calibration_data_detect/`（检测训练集图片），pose 用 `calibration_data/`（PNP 门图）。
> 两者分布完全不同，混用会静默劣化量化精度（校准分布不对 → 量化阈值偏 → int8 掉点）。

### 阶段 4 · 板端（RDK X5）
把 `output/*.bin` 放到板端 `models/`，并让 `auv_vision/cfg/vision.yaml` 对齐：

| 板端配置 | 本仓库对应产物 |
|---|---|
| `model.path: .../auv_multi.bin` | `output/yolo11n_detect_bayese_640x640_nv12.bin` |
| `model.task_models.gate.path` | `output/gate_kpt_bayese_640x640_nv12.bin` |
| `model.labels: [blue_ball, gate, red_ball]` | 训练数据集 `names` 顺序（**必须一致**） |
| `gate.keypoint … kpt_order: [TL,TR,BR,BL]` | RoboFlow 标注点顺序（**必须一致**） |
| `image.*` / `camera.front.calibration` | `configs/vision.yaml` / `configs/front_camera.yaml` |

---

## 权重与配置对照

| 文件 | 角色 | 由谁产生 / 谁使用 |
|---|---|---|
| `weights/yolo11n.pt` | 检测模型（当前 AUV v3，3 类）⚠️ `gate` 类无输出，门检测请用 pose 权重 | `2_train` 产生；供 `3_export` 导出、`select_frames` 筛图 |
| `weights/yolo11n-pose.pt` | pose 模型（门 4 角点）；实际的门检测器。**当前 = AUV_4 版**（2026-09-17，val Pose mAP50-95 **0.952** / test **0.911**） | 训练起点（上一版备份见 `weights/yolo11n-pose.auv3.pt`）；`--task pose` 导出 |
| `weights/yolo11n-pose.coco.pt` | pose 官方 COCO 预训练（17 点），从头训时用作起点 | 官方 asset |
| `weights/yolo11n.onnx` | 检测 ONNX（6 输出） | `export_onnx.py` 产出；`quantize.sh` 输入 |
| `weights/yolo11n-pose.onnx` | pose ONNX（9 输出） | 同上（`--task pose`） |
| `output/*.bin` | 板端可加载模型 | `quantize.sh` 产出 |

> `*.pt` / `*.onnx` / `*.bin` 均不进库（可再生产生），说明见 `weights/README.md`。

---

## 输出契约（与板端解码对齐，勿随意改）

| 任务 | 输出 | 规格 |
|---|---|---|
| detect | **6 个**张量 | 3×bbox `[1,g,g,4*reg_max(64)]`（DFL raw）+ 3×cls `[1,g,g,nc]`（raw logits）；NHWC |
| pose | **9 个**张量 | 每尺度 bbox(64) / cls(nc) / kpt(`3*kpt_dim`)；NHWC，g=80/40/20 |

pose 关键点语义（板端 `gate/gate_decode.py::decode_yolo11_kpt` 按此解码）：
- 通道顺序 `[x, y, v] × 4` = TL, TR, BR, BL；
- `x, y = (raw*2 + 网格索引)`，单位 = cell → 板端 `×stride` 得输入像素 → `×(frame/640)` 得原图；
- `v` = raw logit（板端 sigmoid 后与 `gate.keypoint.conf_thr` 比较）。

> ⚠️ 这里**没有 `-0.5`**：ultralytics 的 `kpts_decode` 用
> `(raw*2 + (anchors - 0.5)) * stride`，而 `make_anchors(grid_cell_offset=0.5)`
> 的 anchors 已经含 `+0.5`，两项相消后净为 **`+网格索引`**。
> 早期 `modify_ultralytics.py` 多写了一个 `-0.5`，会让角点系统性偏移
> **0.5 cell（stride 8/16/32 → 4/8/16 px）**，2026-09-12 已修正并与
> ultralytics 原版逐点比对（可见角点差 **0.03 px**）。
>
> **2026-09-17 端到端复核**（`scripts/3_export/test_decode_parity.py`，用 AUV_4 真图 + 当前 bin 对应的 ONNX）：
> * 预处理：板端 `common/preprocess.py` vs `prepare_frames.py` 链路 **逐像素最大差 0 灰阶**；
> * 解码：板端 `decode_yolo11_kpt`(ONNX 输出) vs ultralytics 预测 **平均 0.00 px / 最大 0.00 px**（38 个角点）；
> * 网格项：`+index` = 0.00 px，改成 `+index-0.5` = **22.63 px** 恒定偏差（stride 32 层）。
>
> ⚠️ 对拍时注意一个**有意的行为差异**：ultralytics 会把低置信角点写成 `(0,0)`，
> 而板端 `decode_yolo11_kpt` **保留真实坐标、只把 `kpt_conf` 置 0**
> （`gate/gate_frontend.py:parse_kpt_mode` 靠 conf 判可见性）。做数值对拍时必须把
> 参考里 `(0,0)`/低置信的点剔掉，否则会算出几百 px 的假误差。
>
> 板端侧另有一组不依赖权重的合成张量单测（约定锁定 + 自动负向对照）：
> `auv_vision/tests/test_gate_decode.py`、`tests/test_preprocess_rt.py`。

**RoboFlow 关键点标注规范**：项目类型 `Keypoint Detection`；类别 1 个 `gate`；
关键点 4 个，命名/顺序 **TL, TR, BR, BL**；允许只标可见角（未标角导出为 `v=0`，不参与 loss）；
图片用 `prepare_frames` 产出的 640 图。

> **导出格式：选 `YOLOv8 Pose` 即可，可直接用于 YOLO11-pose 训练，无需转换。**
> RoboFlow 只提供 `YOLOv5 / YOLOv8 / YOLOv26` 的 Pose 导出，但 YOLOv8-pose 与
> YOLO11-pose 的**数据集格式完全相同**（`kpt_shape` + 每行 `cls cx cy w h` 后接 `x y v`×点数），
> ultralytics 会在训练时按数据集的 `kpt_shape` 重建 Pose 头，
> 预训练权重里 17 点的关键点分支因形状不符被自动跳过（主干仍复用）。

> **⚠️ 但导出的 `data.yaml` 不能直接用**：RoboFlow 项目里删掉的关键点槽位**不会从导出里消失**，
> 常见 `kpt_shape: [5, 3]` 而第 5 点在**全部**标签里都是 `0 0 0`；`flip_idx` 也总是恒等值。
> 先过 `python scripts/1_prepare/pose/prepare_pose_dataset.py <导出目录>` 归一（截槽位 + 校验
> TL,TR,BR,BL 顺序 + 定 flip_idx），再 `resplit_dataset.py` 做序列感知划分。
> 现成示例：`data/AUV_4/PNP.kpt4.yolov8/`（`nc=1`、`names=['gate']`、`kpt_shape=[4,3]`、
> `flip_idx=[0,1,2,3]`、train/valid/test = 1043/131/130）。

---

## 注意事项

- **head.py 状态**：`训练 / predict / val`（含 `select_frames` 筛图）要**原版** head；导出 ONNX 要**补丁版**。
  `train_yolo11n.py` 自动 restore + 训练后重打补丁；手动切换：
  `python scripts/3_export/modify_ultralytics.py --task detect|pose` 与 `--restore`（原版备份在 `head.py.backup`）。
- **训练参数**：本机 8 个 dataloader worker 会与 CUDA fork 死锁 → 默认 `--workers 2`；显存小用 `--batch 4`。
- **配置以板端为准**：`configs/vision.yaml`、`configs/front_camera.yaml`，以及预处理/解码链路
  （板端 `common/preprocess.py`、`common/detector.py`、`gate/gate_decode.py`）**均以板端工程为唯一权威**，
  本仓库只做镜像与契约对齐。修改流程：**先改板端 → 再同步到 `configs/`**，然后重跑预处理；
  同步命令、差异核对与逐项一致性结论见 [configs/README.md](configs/README.md)。
- **标定质量**：棋盘必须覆盖画面**四角与边缘**（早期标定仅中央 ~5% 采样，边缘畸变靠外推，
  水质浑浊时会放大模糊/噪声，表现为“去畸变不成功”）。
- **解码是唯一的强耦合面**：改 `modify_ultralytics.py` 补丁、换导出方式、或动了板端
  `decode_yolo11_kpt` 之后，**必须**用 `scripts/3_export/test_decode_parity.py` 拿真图对拍
  （预处理逐像素 + 角点逐点 px + 网格项对照）。约定错一格不会报错，只会让 PnP 深度/姿态整体偏。
- **PTQ 校准集必须与推理同分布**：用 `prepare_frames` 产出的 **640×640** 图
  （`prepare_calibration.py` 对非方形图会 letterbox 补灰边，而板端是 squish，两者不一致）；
  detect/pose 校准目录**分开**，混用会静默掉点。校准慢是正常的（200 张一轮约 55 分钟），
  原因与"不要为此改 ONNX"见 [scripts/README.md](scripts/README.md) 的 3_export 小节。

---

## 文档

| 文档 | 内容 |
|---|---|
| [scripts/README.md](scripts/README.md) | 脚本索引（每脚本用途/输入输出/参数要点） |
| [configs/README.md](configs/README.md) | 配置来源与“**以板端为准**”同步规则、一致性核对 |
| [weights/README.md](weights/README.md) | 权重与模型分区说明 |
| [docs/tutorial_zh.md](docs/tutorial_zh.md) | 完整部署教程（含踩坑记录、板端代码） |
| [docs/README_ZH.md](docs/README_ZH.md) · [docs/README_EN.md](docs/README_EN.md) | 项目介绍（中文 / English） |
| [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) | 贡献指南 |

许可证：[MIT](LICENSE)。
