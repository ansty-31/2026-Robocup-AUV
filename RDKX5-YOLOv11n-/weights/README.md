# weights/ 权重与模型分区

本目录存放**所有模型文件**（`.pt` / `.onnx`）。`*.pt`、`*.onnx`、`*.bin` 均不进 git
（体积大、可再生产生），本 README 用于说明“每个文件是什么、从哪来、给谁用”。

| 文件 | 角色 | 来源 | 使用者 |
|---|---|---|---|
| `yolo11n.pt` | **检测模型**（当前 = AUV v3，3 类 `blue_ball/gate/red_ball`） | `scripts/2_train/train_yolo11n.py`（detect）产出 | 导出 detect ONNX、`select_frames.py` 筛图 |
| `yolo11n-pose.pt` | **pose 模型**：初始为官方 COCO 预训练（17 点），gate 数据训练后被替换为门 4 角点模型 | 官方 asset 下载 / `2_train`（pose）产出 | 导出 pose ONNX（`--task pose`） |
| `yolo11n.onnx` | 检测 ONNX（6 输出，opset 11 / 640） | `scripts/3_export/export_onnx.py` | `scripts/3_export/quantize.sh`（PTQ 输入） |
| `yolo11n-pose.onnx` | pose ONNX（9 输出，kpt 通道 = 3×点数） | `export_onnx.py --task pose` | `quantize.sh configs/gate_kpt_config.yaml` |

量化产物（板端可加载的 `.bin`）不在本目录，统一放在 **`output/`**：

| 产物 | 对应配置 | 板端用途 |
|---|---|---|
| `output/yolo11n_detect_bayese_640x640_nv12.bin` | `configs/yolo11n_config.yaml` | 球/门检测（`model.path`） |
| `output/gate_kpt_bayese_640x640_nv12.bin` | `configs/gate_kpt_config.yaml` | 门 4 角点（`model.task_models.gate.path`） |

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
  `export_onnx.py --task pose` 会校验并提示。
- 若 `yolo11n-pose.pt` 被 gate 训练覆盖后，需要重新拿官方预训练做新实验时，可重新下载：
  `wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt -O weights/yolo11n-pose.pt`
