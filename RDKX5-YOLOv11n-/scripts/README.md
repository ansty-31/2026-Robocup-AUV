# scripts/ 脚本索引

脚本按**流程阶段**分三区，完整流程命令与输出契约见项目根 [README.md](../README.md)。

> **运行环境**：下面所有 `python ...` 命令都必须先 `conda activate yolov8`
> （= `/home/ansty/anaconda3/envs/yolov8/bin/python`，ultralytics 8.3.0 + torch 2.3.1+cu118，GPU 可用）；
> 系统 `python3` 没装 torch/ultralytics，会直接 `ModuleNotFoundError`。

```
1_prepare/   训练前：数据准备（切帧 → 标定 → 预处理 → 筛图）
2_train/     训练：detect / pose
3_export/    导出与量化：改输出头 → ONNX → 校准数据 → PTQ 量化
```

## 1_prepare/ — 训练前（数据准备）

| 脚本 | 用途 | 常用参数 |
|---|---|---|
| `extract_frames.py` | 切帧：自动识别**容器视频 / 裸 MJPEG 流**（裸流按 JPEG 标记无损切割） | `--step N`、`--every-seconds S`、`--start-frame`、`--max-frames`、`--dry-run` |
| `calibrate_camera.py` | 相机标定：棋盘（视频或图片目录）→ 内参 yaml；全池择优 + 离群剔除，RMS≤0.8px 验收 | `--cols/--rows/--square-mm`、`--views`、`--min-shift`、`--save-views` |
| `prepare_frames.py` | 预处理：去畸变 remap → **640×640** → 白平衡/CLAHE/gamma（与板端 `common/preprocess.py` 同链路、**同顺序**；**参数以板端为准**，镜像见 [configs/README.md](../configs/README.md)） | `--config configs/vision.yaml`、`--out-dir`；或 `--calibration` + `--wb-gains/--clahe/--gamma` |
| `prepare_frames_noundistort.py` | **不去畸变**变体：resize **640×640** → 白平衡/CLAHE/gamma（其余与上一个完全同链路，复用其实现）。用于把「未校正畸变图」混入训练集，提升模型对畸变的韧性 | 同上（`image.undistort` 被忽略）；建议 `--out-dir .../processed_640_noundistort` |
| `select_frames.py` | 用模型剔除指定类别画面，再按 `清晰度×亮度合理性` 质量加权随机抽样；**可给多个输入目录把「不去畸变」变体按 `--mix-ratio` 混入**（混合集输出自动加 `--mix-prefix` 前缀防重名）。不给 `--drop-classes` 时**跳过推理**（只滤损坏帧，`--weights` 可不传） | `--drop-classes`、`--weights`、`--conf`、`--quality-dir`、`--keep`、`--gamma`、`--mix-ratio`、`--mix-prefix`、`--report` |

```bash
python scripts/1_prepare/extract_frames.py data/AUV_2/auv_xxx.mjpeg --every-seconds 1
python scripts/1_prepare/calibrate_camera.py data/AUV_1/board --cols 11 --rows 8 --square-mm 20 \
    --output configs/front_camera.yaml --save-views qc/
python scripts/1_prepare/prepare_frames.py data/AUV_2/<分类目录> --config configs/vision.yaml
python scripts/1_prepare/select_frames.py data/AUV_2/datay/origin --weights weights/yolo11n.pt \
    --drop-classes red_ball --quality-dir data/AUV_2/origin --keep 2000 --out-dir data/AUV_2/selected_2000
```

### 混入「不去畸变」图，增强模型韧性

去畸变依赖标定文件；标定失效 / `vision.yaml` 的 `undistort: false` / 相机变动时，
送进模型的是**未校正的畸变图**。把该变体按比例混入训练集，模型同时见过两种几何：

```bash
# 1) 生成不去畸变变体（与 prepare_frames.py 同链路，只少一步 remap）
python scripts/1_prepare/prepare_frames_noundistort.py data/AUV_3/auv_xxx_frames \
    --config configs/vision.yaml --out-dir data/AUV_3/processed_640_noundistort

# 2) 选图时把两个目录都给进去：第 1 个是主集，其余按 --mix-ratio 掺入
python scripts/1_prepare/select_frames.py \
    data/AUV_3/processed_640/auv_xxx_frames \
    data/AUV_3/processed_640_noundistort/auv_xxx_frames \
    --weights weights/yolo11n.pt --drop-classes red_ball \
    --quality-dir data/AUV_3/auv_xxx_frames \
    --mix-ratio 0.35 --keep 2000 --out-dir data/AUV_3/selected_2000 \
    --report data/AUV_3/select_report.csv
```

- 输出：主集 `frame_000123.jpg`，混合集 `nd_frame_000123.jpg`（前缀见 `--mix-prefix`）
- **⚠️ 标注**：不去畸变保留了几何畸变，两版**标注框不通用**，需**分别标注**

## 2_train/ — 训练

| 脚本 | 用途 | 关键参数 |
|---|---|---|
| `train_yolo11n.py` | 训练/微调（`--task detect\|pose`）：自动恢复原版 head → 训练 → best 复制到 `weights/` → 按任务重打输出头补丁 | `--data`、`--task`、`--weights`、`--epochs`、`--batch`、`--imgsz`、`--device`、`--cache ram`、`--no-repatch` |

```bash
# detect（默认权重 weights/yolo11n.pt，输出 weights/yolo11n.pt）
python scripts/2_train/train_yolo11n.py --data <data.yaml> --epochs 300 --batch 4 --device 0 --cache ram

# pose（默认权重 weights/yolo11n-pose.pt，输出 weights/yolo11n-pose.pt）
python scripts/2_train/train_yolo11n.py --task pose --data <data.yaml> --epochs 300 --batch 4 --device 0
```

要点：
- **训练必须在原版 head 下**（脚本自动 restore）；训练后自动重打补丁，衔接导出；
- 本机 8 个 dataloader worker 会与 CUDA fork 死锁 → 默认 `--workers 2`；显存小用 `--batch 4`；
- pose 数据集需 `kpt_shape: [4, 3]`（RoboFlow 导出的 YOLOv8 Pose 已含）。

## 3_export/ — 导出与量化

| 脚本 | 用途 | 关键参数 |
|---|---|---|
| `modify_ultralytics.py` | 输出头补丁：`--task detect`=6 输出 / `--task pose`=9 输出；`--restore` 回退原版（自动备份 `head.py.backup`） | `--task`、`--restore` |
| `export_onnx.py` | 导出 ONNX（opset 11 / 640）并校验张量结构 | `--task`、`--model`、`--output`、`--imgsz` |
| `prepare_calibration.py` | 生成 PTQ 校准数据（RGB float32 CHW 640 的 `.rgb`） | `--coco-path`、`--output-dir`、`--num-images` |
| `quantize.sh` | Docker 内 `hb_mapper makertbin` PTQ 量化（自动切到项目根，容器内 `/data`=项目根）；**onnx 路径、校准目录、输出前缀都从配置文件里读**，detect/pose 通用 | `[config.yaml]`（默认 `configs/yolo11n_config.yaml`） |

> **校准集按任务分开放**：detect → `calibration_data_detect/`（检测训练集图片），
> pose → `calibration_data/`（PNP 门图）。两者分布不同，**混用会静默掉点**；
> `quantize.sh` 会按配置里的 `cal_data_dir` 校验对应目录，缺了会直接报错而不是拿错数据。
>
> 校准数据建议取**训练集图片**（与推理同分布）：`prepare_calibration.py --coco-path <数据集>/train/images --output-dir <对应目录>`。
> 注意它对非方形图会 letterbox 补灰边，而板端推理是直接 squish 到 640×640；
> **图片本身已是 640×640 时 letterbox 是 no-op**，所以用 `prepare_frames` 产出的图最稳。

```bash
# detect：6 输出 → yolo11n_detect_*.bin
python scripts/3_export/modify_ultralytics.py --task detect
python scripts/3_export/export_onnx.py
python scripts/3_export/prepare_calibration.py --coco-path <检测数据集>/train/images \
    --num-images 300 --output-dir calibration_data_detect
./scripts/3_export/quantize.sh

# pose：9 输出（4 点 → kpt 通道 12）→ gate_kpt_*.bin
python scripts/3_export/modify_ultralytics.py --task pose
python scripts/3_export/export_onnx.py --task pose        # → weights/yolo11n-pose.onnx
python scripts/3_export/prepare_calibration.py --coco-path data/AUV_3/PNP.v1i.yolov8/train/images --num-images 300
./scripts/3_export/quantize.sh configs/gate_kpt_config.yaml
```

> pose 侧补充（2026-09-12 已优化）：`hb_perf` 显示 **单个 BPU 子图、6.83 ms（≈146 FPS）**，
> 与 detect（6.64 ms）基本持平；仅 9 个很小的 Concat/Reshape/Transpose 落在 CPU。
> 关键改动：`kpt` 取通道用 4D 步长切片（`kpt[:, 0::3]`）而非 `view(...,5D)` + 整数索引
> ——后者导出 rank-5 Gather，会让模型被切成 7 个子图。**不要**改成"先 permute 到 NHWC
> 再算"：layout 变换会在 DDR 实体化（实测 18→48 MB，延迟 6.8→11.5 ms）。
```

输出契约（detect 6 张量 / pose 9 张量、pose 关键点坐标语义、板端解码对应关系）见
根 [README.md 的“输出契约”](../README.md#输出契约与板端解码对齐勿随意改)。
