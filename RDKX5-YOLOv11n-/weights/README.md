# weights/ 权重与模型分区

本目录存放**所有模型文件**（`.pt` / `.onnx`）。`*.pt`、`*.onnx`、`*.bin` 均不进 git
（体积大、可再生产生），本 README 用于说明“每个文件是什么、从哪来、给谁用”。

| 文件 | 角色 | 来源 | 使用者 |
|---|---|---|---|
| `yolo11n.pt` | **检测模型**（当前 = AUV v3，3 类 `blue_ball/gate/red_ball`） | `scripts/2_train/train_yolo11n.py`（detect）产出 | 导出 detect ONNX、`select_frames.py` 筛图 |
| `yolo11n-pose.pt` | **pose 模型**：初始为官方 COCO 预训练（17 点），gate 数据训练后被替换为门 4 角点模型。**当前 = AUV_4 版**（2026-09-17 在 `data/AUV_4/PNP.kpt4.yolov8` 上微调，206 ep 早停 / best=ep106，val Pose mAP50-95 **0.952**、test **0.911**） | 官方 asset 下载 / `2_train`（pose）产出；`runs/pose/auv4/weights/best.pt` 为同源副本 | 导出 pose ONNX（`--task pose`） |
| `yolo11n-pose.coco.pt` | 官方 COCO pose 预训练（17 点）备份，从头训练时用 | 官方 asset | `2_train --task pose --weights weights/yolo11n-pose.coco.pt` |
| `yolo11n-pose.auv3.pt` | 上一版 pose 模型（AUV_3 版，= `runs/pose/PNP_v1/weights/best.pt`）。**当前进板 bin 对应的就是它**（AUV_4 训练前手动备份） | 手动备份 | 回退用 |
| `yolo11n.onnx` | 检测 ONNX（6 输出，opset 11 / 640） | `scripts/3_export/export_onnx.py` | `scripts/3_export/quantize.sh`（PTQ 输入） |
| `yolo11n-pose.onnx` | pose ONNX（9 输出，kpt 通道 = 3×点数） | `export_onnx.py --task pose` | `quantize.sh configs/gate_kpt_config.yaml` |

量化产物（板端可加载的 `.bin`）不在本目录，统一放在 **`output/`**：

| 产物 | 对应配置 | 板端用途 |
|---|---|---|
| `output/yolo11n_detect_bayese_640x640_nv12.bin` | `configs/yolo11n_config.yaml` | 球/门检测（`model.path`） |
| `output/gate_kpt_bayese_640x640_nv12.bin` | `configs/gate_kpt_config.yaml` | 门 4 角点（`model.task_models.gate.path`） |
| `output/backup_auv3/` | 上一版（AUV_3）pose 的 `*.bin` + `*_quant_info.json` | **不部署**，仅作回退与量化指标基线对比 |

> `quantize.sh` 用配置里的 `output_model_file_prefix` 命名产物 —— **同名会直接覆盖**。
> 重跑 pose 量化前先确认 `output/gate_kpt_*.bin` 是否已备份（`backup_auv3/` 就是这么来的）。

## 再生产方式

```bash
# 训练（detect / pose）
python scripts/2_train/train_yolo11n.py --data <data.yaml> --epochs 300 --batch 4 --device 0
python scripts/2_train/train_yolo11n.py --task pose --data <data.yaml> --epochs 300 --batch 4 --device 0

# 导出 ONNX（写入本目录）
python scripts/3_export/export_onnx.py               # → weights/yolo11n.onnx
python scripts/3_export/export_onnx.py --task pose   # → weights/yolo11n-pose.onnx

# 量化（写入 output/）
./scripts/3_export/quantize.sh
./scripts/3_export/quantize.sh configs/gate_kpt_config.yaml
```

## 注意

- **类别顺序必须与板端一致**：`model.labels: [blue_ball, gate, red_ball]`（决定解码类别，
  训练数据集的 `names` 顺序必须相同）。
- **pose 的关键点顺序/数量**：`kpt_shape: [4, 3]`（TL,TR,BR,BL），导出后 kpt 张量通道应为 **12**；
  `export_onnx.py --task pose` 会校验并提示。RoboFlow 的导出**未必**是 `[4,3]`
  （常见 `[5,3]` + 全零第 5 点），训练前先过 `scripts/1_prepare/prepare_pose_dataset.py`。
- **`.onnx` 只是中间产物**：`weights/*.onnx` 会被 `export_onnx.py` 直接覆盖，且板端**不加载**它
  （板端只加载 `output/*.bin`）。换 `--task` 导出前先确认 `head.py` 的补丁状态。
- **量化耗时长是正常的**：单轮 200 张校准约 55 分钟，`calibration_type: default` 通常跑多轮；
  原因（C2PSA 的 Reshape 烘死 `batch=1`）与"为什么不要为提速改 ONNX"见
  [scripts/README.md](../scripts/README.md) 的 3_export 小节。
- 若 `yolo11n-pose.pt` 被 gate 训练覆盖后，需要重新拿官方预训练做新实验时，可重新下载：
  `wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt -O weights/yolo11n-pose.pt`
