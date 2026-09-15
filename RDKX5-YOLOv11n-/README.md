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

```
RDKX5-YOLOv11n-/
├── README.md                  ← 本文件（项目总览 / 分区导航 / 全流程命令）
├── LICENSE · requirements.txt · .gitignore
├── configs/                   配置（全部集中在此）
│   ├── vision.yaml            ★ 板端图像链路参数镜像（预处理与推理一致性唯一来源）
│   ├── front_camera.yaml      相机标定（calibrate_camera.py 产出，板端共用）
│   ├── yolo11n_config.yaml    detect 模型 PTQ 量化配置
│   └── gate_kpt_config.yaml   pose 模型 PTQ 量化配置
├── weights/                   权重与模型（★ 分区说明见 weights/README.md）
│   ├── yolo11n.pt             检测权重（当前 = AUV v3 训练产物，3 类）
│   ├── yolo11n-pose.pt        pose 预训练（官方 COCO，训练起点）
│   └── yolo11n.onnx           最近一次 detect 导出（量化输入）
├── scripts/                   脚本按阶段分三区（详见 scripts/README.md）
│   ├── 1_prepare/             训练前：extract_frames · calibrate_camera · prepare_frames · select_frames
│   ├── 2_train/               训练：train_yolo11n.py（--task detect|pose）
│   └── 3_export/              导出+量化：modify_ultralytics · export_onnx · prepare_calibration · quantize.sh
├── docs/                      文档（教程/介绍/贡献指南）
│   ├── tutorial_zh.md         完整部署教程（含踩坑记录）
│   ├── README_ZH.md           中文项目介绍
│   ├── README_EN.md           English overview
│   └── CONTRIBUTING.md
├── data/                      数据集与素材（不进库）
├── output/                    量化产物 *.bin（不进库）
└── runs/                      训练输出（不进库）
```

**分区原则**：`scripts/` 只放可执行脚本并按“训练前 / 训练 / 导出量化”三段分区；
所有配置进 `configs/`；权重与模型产物进 `weights/`；说明性文档进 `docs/`；数据与产物不进库。

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
```

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
    --coco-path data/AUV_3/PNP.v1i.yolov8/train/images --num-images 300   # 默认写 calibration_data/
./scripts/3_export/quantize.sh configs/gate_kpt_config.yaml   # → output/gate_kpt_*.bin
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
| `weights/yolo11n.pt` | 检测模型（当前 AUV v3，3 类） | `2_train` 产生；供 `3_export` 导出、`select_frames` 筛图 |
| `weights/yolo11n-pose.pt` | pose 模型（先官方预训练，后为 gate 训练产物） | 训练起点；`--task pose` 导出 |
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

**RoboFlow 关键点标注规范**：项目类型 `Keypoint Detection`；类别 1 个 `gate`；
关键点 4 个，命名/顺序 **TL, TR, BR, BL**；允许只标可见角（未标角导出为 `v=0`，不参与 loss）；
图片用 `prepare_frames` 产出的 640 图。

> **导出格式：选 `YOLOv8 Pose` 即可，可直接用于 YOLO11-pose 训练，无需转换。**
> RoboFlow 只提供 `YOLOv5 / YOLOv8 / YOLOv26` 的 Pose 导出，但 YOLOv8-pose 与
> YOLO11-pose 的**数据集格式完全相同**（`kpt_shape: [4,3]` + 每行 `cls cx cy w h`
> 后接 `x y v`×4），ultralytics 会在训练时按数据集的 `kpt_shape` 重建 Pose 头，
> 预训练权重里 17 点的关键点分支因形状不符被自动跳过（主干仍复用）。
> 示例：`data/AUV_3/PNP.v1i.yolov8/`（`nc=1`、`names=['gate']`、`kpt_shape=[4,3]`）。

---

## 注意事项

- **head.py 状态**：`predict/val`（含 AMP 检查、`select_frames`）要**原版** head；导出 ONNX 要**补丁**。
  `train_yolo11n.py` 自动 restore + 训练后重打补丁；手动切换：
  `python scripts/3_export/modify_ultralytics.py --restore`。
- **训练参数**：本机 8 个 dataloader worker 会与 CUDA fork 死锁 → 默认 `--workers 2`；显存小用 `--batch 4`。
- **配置以板端为准**：`configs/vision.yaml`、`configs/front_camera.yaml`，以及预处理/解码链路
  （板端 `common/preprocess.py`、`common/detector.py`、`gate/gate_decode.py`）**均以板端工程为唯一权威**，
  本仓库只做镜像与契约对齐。修改流程：**先改板端 → 再同步到 `configs/`**，然后重跑预处理；
  同步命令、差异核对与逐项一致性结论见 [configs/README.md](configs/README.md)。
- **标定质量**：棋盘必须覆盖画面**四角与边缘**（早期标定仅中央 ~5% 采样，边缘畸变靠外推，
  水质浑浊时会放大模糊/噪声，表现为“去畸变不成功”）。

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
