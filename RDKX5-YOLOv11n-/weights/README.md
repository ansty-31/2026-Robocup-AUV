# weights/ 权重与模型分区

本目录存放**所有模型文件**（`.pt` / `.onnx`）。`*.pt`、`*.onnx`、`*.bin` 均不进 git
（体积大、可再生产生），本 README 用于说明“每个文件是什么、从哪来、给谁用”。

| 文件 | 角色 | 来源 | 使用者 |
|---|---|---|---|
| `yolo11n.pt` | **检测模型**（3 类 `blue_ball/gate/red_ball`） | `scripts/2_train/train_yolo11n.py`（detect）产出 | 导出 detect ONNX、`select_frames.py` 筛图 |

> ⚠️ **`yolo11n.pt` 的版本口径待确认**。整理时（2026-09-23）按 md5 核对：
> - `weights/yolo11n.pt` = `91069d76dbed46ca8639aead1f0f4cee`（mtime 09-12 14:06）
>   ＝ `_archive/runs/detect/auv_v4/weights/best.pt`（**逐字节相同**）
> - `_archive/runs/detect/auv_v3/weights/best.pt` = `d30de5dc6c3b038f415338733c9b1d40`（mtime 09-08）
>   —— **是另一个模型**，且是它的**唯一本地副本**
>
> 即：文件名/文档里说的 "AUV v3" 与实盘 md5 对不上（`weights/yolo11n.pt` 与名为 `auv_v4` 的
> 训练目录同源）。**我没有改动任何权重、也没有改版本命名**，请由你确认哪个才是真正的 v3/v4。
| `yolo11n-pose.pt` | **pose 模型（当前 = AUV_5 版）**：门 4 角点。2026-09-23 在 `data/AUV_5/gate-pose.yolov8`（625 张，清水 53%，图 = **D+wb 链路**）上从 auv4 版权重微调，早停 258/300（best=ep158），val Pose mAP50 **0.959** / mAP50-95 **0.936**，test mAP50 **0.9589** / mAP50-95 **0.9242**。md5 `f08f82ed…` | `scripts/2_train/train_yolo11n.py --task pose` 产出（best.pt 复制到本名） | 导出 pose ONNX、板端 gate 4 角点；详见 [`runs/auv5/REPORT_auv5_pose.md`](../runs/auv5/REPORT_auv5_pose.md) |
| `yolo11n-pose.auv4.pt` | 上一版 pose 模型（AUV_4 版，2026-09-17 在 `data/AUV_4/PNP.kpt4.yolov8` 上微调，206 ep 早停 / best=ep106，val Pose mAP50-95 **0.952**、test **0.911**）。md5 `3a5e87fa…`；同源训练记录 `_archive/runs/pose/auv4/` | 2026-09-23 被 auv5 顶替时按代次改名保留 | **回退用**（改名前的文件就是它） |
| `yolo11n-pose.coco.pt` | 官方 COCO pose 预训练（17 点）备份，从头训练时用 | 官方 asset | `2_train --task pose --weights weights/yolo11n-pose.coco.pt` |
| `yolo11n-pose.auv3.pt` | 更早一版 pose 模型（AUV_3 版）。**更早的进板 bin 对应的就是它** | 手动备份；同源训练记录 `_archive/runs/pose/PNP_v1/`（md5 `db2ceb28…`） | **回退用** |
| `yolo11n.onnx` | 检测 ONNX（6 输出，opset 11 / 640） | `scripts/3_export/export_onnx.py` | `scripts/3_export/quantize.sh`（PTQ 输入） |
| `yolo11n-pose.onnx` | **当前** pose ONNX（**9 输出**，NHWC，kpt 通道 12；输入 `images 1×3×640×640`，opset 11）= auv5 版导出，md5 `05fd2d8b…` | `export_onnx.py --task pose`（需先 `modify_ultralytics.py --task pose` 打补丁） | `quantize.sh configs/gate_kpt_config.yaml` |
| `yolo11n-pose.auv4.onnx` | 上一版 pose ONNX（AUV_4 导出），md5 `1e7785ca…` | 同上，被 auv5 顶替时改名保留 | **回退用** |

> **命名规则（2026-09-23 起）**：**当前版本占"规范名"**（`yolo11n-pose.pt` / `yolo11n-pose.onnx` /
> `output/gate_kpt_bayese_640x640_nv12.bin`），**历史版本加 `.<代次>` 后缀**（`.auv3` / `.auv4` / …）。
> 新模型上线三步（不删任何旧文件）：
> ```bash
> mv weights/yolo11n-pose.pt        weights/yolo11n-pose.auv4.pt      # 旧的当前 → 代次备份
> mv weights/yolo11n-pose.onnx      weights/yolo11n-pose.auv4.onnx
> cp weights/yolo11n-pose.<新代次>.pt weights/yolo11n-pose.pt        # 新的 → 当前
> cp weights/yolo11n-pose.<新代次>.onnx weights/yolo11n-pose.onnx
> ```

### 去畸变域实验权重（2026-09-23，`experiment/scripts/train_domain_arms.sh` 产出）

四条臂都在 `data/mapped/*` 上从 `yolo11n-pose.coco.pt` 起步、同日程（200 ep / seed 0），
**只换图像链路**；四个文件**不覆盖**上面的正式权重，用于"哪种去畸变时机/补偿最准"的对比。
结论与逐帧明细见 [`experiment/runs/domain/INDEX.md`](../experiment/runs/domain/INDEX.md) 与 `experiment/runs/domain/DOMAIN_RESULT.md`。

| 文件 | 链路 | val Pose mAP50-95 | 训练记录 |
|---|---|---|---|
| `domain_B.pt` | B：不去畸变 | 0.9787 @ep199 | `experiment/runs/domain/A_B/` |
| `domain_C.pt` | C：720p 上去畸变 | **0.9879** @ep195 | `experiment/runs/domain/A_C/` |
| `domain_D.pt` | D：640 上去畸变 | 0.9727 @ep196 | `experiment/runs/domain/A_D/` |
| `domain_D_noenh.pt` | D + 无 enhance（LUT 白平衡 / 无 CLAHE） | 0.9853 @ep200 | `experiment/runs/domain/A_D_noenh/` |

量化产物（板端可加载的 `.bin`）不在本目录，统一放在 **`output/`**：

| 产物 | 对应配置 | 板端用途 |
|---|---|---|
| `output/yolo11n_detect_bayese_640x640_nv12.bin` | `configs/yolo11n_config.yaml` | 球/门检测（`model.path`） |
| **`output/gate_kpt_bayese_640x640_nv12.bin`** = **当前（AUV_5 版）**，md5 `945fdd01…`，由 auv5 ONNX 量化（300 张 D+wb 校准 + `calibration_type: max`） | `configs/gate_kpt_config.yaml` | 门 4 角点（`model.task_models.gate.path`） |
| `output/gate_kpt_auv4_bayese_640x640_nv12.bin` = **上一版（AUV_4 版）**，md5 `d38b803e…` | `configs/gate_kpt_auv4_config.yaml` | **回退用**（改名前就是 `gate_kpt_bayese_640x640_nv12.bin`） |
| `output/build_auv5/` | — | auv5 量化过程的中间产物（`*_original_float/optimized_float/calibrated/quantized_model.onnx` + hbdk 编译图），**板端不用**；`check_quant_cos.py` 复核余弦时读这里 |
| `output/backup_auv3/` → 现 `_archive/output/backup_auv3/` | 更早一版（AUV_3）pose 的 `*.bin` + `*_quant_info.json` | **不部署**，仅作回退与量化指标基线对比 |

> ⚠️ **本地名 ≠ 板端名**：板端工程里模型路径由板端 `cfg` 指定。
> **2026-09-23 22:05 已上板**：板端 `models/gate_kpt_bayese_640x640_nv12.bin` = auv5（`945fdd01…`），
> 配 `common/preprocess.py`（D 链序）+ `cfg/vision.yaml`（`clahe_clip: 0`）一起换的；
> 旧 auv4 版在板端 `bak/backup_auv4/`。**板端已实测**（2026-09-23 22:1x）：
> `detect()` 31.27 ms ≈ 32 FPS、真机 18.3–19.1 FPS、int8 vs GT 中位 13.94 px@720p（量化代价 +4.74 px）。

> `quantize.sh` 用配置里的 `output_model_file_prefix` 命名产物 —— **同名会直接覆盖**。
> 重跑 pose 量化前先确认 `output/gate_kpt_*.bin` 是否已备份（2026-09-23 整理时，
> 上一版备份已移入 `_archive/output/backup_auv3/`，原因与恢复方法见 `_archive/README.md`）。

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
  （常见 `[5,3]` + 全零第 5 点），训练前先过 `scripts/1_prepare/pose/prepare_pose_dataset.py`。
- **`.onnx` 只是中间产物**：`weights/*.onnx` 会被 `export_onnx.py` 直接覆盖，且板端**不加载**它
  （板端只加载 `output/*.bin`）。换 `--task` 导出前先确认 `head.py` 的补丁状态。
- **量化耗时参考（2026-09-23 实测，本机 RTX 4060 + CPU 版工具链镜像）**：300 张校准一轮
  `makertbin` 约 **10 分钟**；随后 `check_quant_cos.py` 20 张图约 3 分钟。更早一版文档写的
  「单轮约 55 分钟」是另一台/另一版配置的旧数，按实测为准；
  原因（C2PSA 的 Reshape 烘死 `batch=1`）与"为什么不要为提速改 ONNX"见
  [scripts/README.md](../scripts/README.md) 的 3_export 小节。
- 若 `yolo11n-pose.pt` 被 gate 训练覆盖后，需要重新拿官方预训练做新实验时，可重新下载：
  `wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt -O weights/yolo11n-pose.pt`
