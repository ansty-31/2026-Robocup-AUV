> ⚠️ 本文是项目**介绍/宣传版**，目录结构已重构：脚本按阶段分 `scripts/1_prepare|2_train|3_export/`，
> 权重在 `weights/`，配置在 `configs/`。**以根目录 [README.md](../README.md) 为准**（命令、分区、输出契约）。

<div id="top"></div>

# YOLO11n 训练 + RDK X5 部署 🚀

[English](README.md) | 简体中文

<div align="center">

![YOLO11n](https://img.shields.io/badge/YOLO11-n-blue)
![RDK X5](https://img.shields.io/badge/RDK-X5-green)
![License](https://img.shields.io/badge/license-MIT-yellow)

**在地平线 RDK X5 上部署 YOLO11n：自定义数据集训练 → ONNX 导出 → PTQ 量化 → 板端推理**

</div>

---

## 目录

- [项目简介](#项目简介)
- [完整流程](#完整流程)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [关键技术](#关键技术)
- [性能参考](#性能参考)
- [许可证](#许可证)

---

## 项目简介

本项目提供在**地平线 RDK X5** 上部署 YOLO11n 的**端到端方案**，核心链路为：

```
自定义图片数据集 → 微调训练 YOLO11n → 拆分6输出头 → 导出ONNX
→ 准备校准数据 → hb_mapper PTQ量化(nv12) → *.bin → 部署到RDK X5
```

针对 YOLO11 新增 C2PSA（Softmax）导致的性能瓶颈，通过 `node_info` 将 Softmax 指定到 **BPU int16** 运行，配合 6 输出头拆分与 NV12 输入，在 RDK X5 上实现实时检测（COCO 预训练基线：BPU 延迟约 10.8ms，端到端约 47 FPS）。

---

## 完整流程

```
数据准备(data/xxx/data.yaml)         ① 训练
        ↓
scripts/2_train/train_yolo11n.py             ② 微调(基于yolo11n.pt预训练权重)
        ↓  best.pt → yolo11n.pt
scripts/3_export/modify_ultralytics.py        ③ 输出头拆分（detect=6输出 / pose=9输出，自动备份/验证）
scripts/3_export/export_onnx.py               ④ 导出 ONNX（--task detect|pose）
        ↓  yolo11n.onnx
scripts/3_export/prepare_calibration.py       ⑤ 从图片生成100张校准数据(.rgb)
        ↓  calibration_data/
scripts/3_export/quantize.sh                  ⑥ Docker内 hb_mapper PTQ量化
        ↓
output/yolo11n_detect_bayese_640x640_nv12.bin  ⑦ scp → RDK X5 板端
```

> 步骤③ 会修改当前 Python 环境里安装的 `ultralytics/nn/modules/head.py`（训练需要原始输出头，因此**必须先训练、后打补丁**；`train_yolo11n.py` 已自动处理该顺序）。

---

## 快速开始

### 环境要求

| 项目 | 要求 |
|---|---|
| 主机 | Ubuntu 22.04，Docker |
| 训练环境 | conda 环境 `yolov8`（`ultralytics==8.3.0`，GPU 可选） |
| 工具链 | OpenExplorer **v1.2.8** Docker 镜像：`openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310` |

工具链镜像加载（官方直链）：

```bash
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker load < docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
```

### ① 准备数据集

YOLO 格式数据集（参考 `data/auv/data.yaml`）：

```yaml
path: data/auv            # 图片目录（相对项目根）
train: train/images       # 训练集
val: valid/images         # 验证集
names: [red_ball, blue_ball, gate]   # 类别
```

#### （可选）真实相机：标定 + 切帧建图

若相机存在畸变/偏色，为保证**进入 YOLO 训练的图片与进入板端 YOLO 推理的图片
完全一致（去畸变 + 画面补偿 + 640x640）**，推荐链路（脚本细节见
[scripts/README.md](scripts/README.md)）：

```bash
# 1) 棋盘格（视频或图片目录）→ 相机内参 yaml（RMS≤0.8px 验收）
#    棋盘要覆盖画面四角/边缘，否则边缘畸变靠外推、去畸变会失真
python scripts/1_prepare/calibrate_camera.py data/AUV_1/board \
    --cols 11 --rows 8 --square-mm 20 --output configs/front_camera.yaml

# 2) 录制文件 → 原始帧（自动识别容器视频 / 裸 MJPEG 流，后者无损切割）
python scripts/1_prepare/extract_frames.py data/AUV_2/auv_xxx.mjpeg        # 裸 MJPEG 全量切
python scripts/1_prepare/extract_frames.py data/AUV_1/rec_front.mp4 --step 5
#    输出 <同目录>/<文件名>_frames/frame_000001.jpg ...（原始分辨率）

# 3) 分好类的图片集 → 训练图集（去畸变 + 白平衡/CLAHE/gamma 补偿 + 640x640）
python scripts/1_prepare/prepare_frames.py data/AUV_2/<分类目录> --config configs/vision.yaml
#    --calibration configs/front_camera.yaml 亦可手动指定；参数须与板端一致

# 4)（可选）用已有模型筛图：剔除不要的类别 + 按画面质量加权抽样
python scripts/1_prepare/select_frames.py data/AUV_2/datay/origin \
    --weights weights/yolo11n.pt --drop-classes red_ball \
    --quality-dir data/AUV_2/origin --keep 2000 --out-dir data/AUV_2/selected_2000
```

预处理链路与板端 `auv_vision/common/preprocess.py` 推理逐帧一致：
`cv2.remap(标定映射) 去畸变 → resize(640,640) → 白平衡通道增益 → LAB-CLAHE → gamma`。
**（注意顺序：2026-09-11 板端把画面补偿改到缩放到 640 之后做，先缩放再补偿，两步不可交换。）**
生成的图片标注后即可作为训练集（上述 data.yaml 组织方式）。

### ② 训练（基于 yolo11n.pt 预训练权重微调）

```bash
# GPU: --device 0；CPU 可不带该参数（较慢）
python scripts/2_train/train_yolo11n.py --data data/auv/data.yaml --epochs 100 --imgsz 640 --device 0
```

- 自动使用/下载根目录 `yolo11n.pt`（80 类 COCO 预训练），自定义类别数自动适配
- 训练结束后 best.pt 自动复制为根目录 `yolo11n.pt` 衔接导出，并自动重新打 6 输出头补丁
- 不训练、直接用官方预训练权重时，跳过此步即可

### ③ ④ 导出 ONNX（检测 6 输出 / 关键点 9 输出）

```bash
# 检测（Detect 头 → 6 个张量：3×bbox + 3×cls）
python scripts/3_export/modify_ultralytics.py --task detect   # 自动备份 head.py.backup 并验证
python scripts/3_export/export_onnx.py                        # yolo11n.onnx（opset 11, 640, 6 输出）

# 关键点（Pose 头 → 9 个张量：每尺度 bbox64 + cls nc + kpt 12，gate 4 角点）
python scripts/3_export/modify_ultralytics.py --task pose
python scripts/3_export/export_onnx.py --task pose            # yolo11n-pose.onnx（9 输出）
```

> pose 的 RoboFlow 标注规范（4 点 TL,TR,BR,BL / Keypoint Detection / 导出 YOLOv8 Pose）、
> 训练命令与**输出张量契约**（板端 `gate_decode.py` 解码公式）见
> [scripts/README.md](scripts/README.md#关键点gate-4-角点--pose)。

### ⑤ 准备校准数据（需要一批有代表性的图片，如 COCO val2017）

```bash
python scripts/3_export/prepare_calibration.py --coco-path /path/to/coco/val2017 --num-images 100
```

### ⑥ PTQ 量化（Docker 容器内执行 hb_mapper）

```bash
./scripts/3_export/quantize.sh   # 等价于 docker run ... hb_mapper makertbin --config configs/yolo11n_config.yaml
```

量化配置要点（`configs/yolo11n_config.yaml`）：

```yaml
model_parameters:
  node_info: {"/model.10/m/m.0/attn/Softmax": {'ON': 'BPU','InputType': 'int16','OutputType': 'int16'}}  # 核心：Softmax上BPU
input_parameters:
  input_type_rt: 'nv12'    # BPU原生NV12输入
compiler_parameters:
  compile_mode: 'latency'
  optimize_level: 'O3'
```

### ⑦ 部署到 RDK X5

```bash
scp output/yolo11n_detect_bayese_640x640_nv12.bin sunrise@<RDK_IP>:~/models/
```

---

## 项目结构

```
RDKX5-YOLOv11n-/
├── README_ZH.md / README.md        # 文档
├── LICENSE / CONTRIBUTING.md
├── requirements.txt                # 主机侧 Python 依赖
├── configs/
│   ├── yolo11n_config.yaml         # 检测模型 PTQ 量化配置（含 Softmax node_info）
│   ├── gate_kpt_config.yaml        # 门关键点(pose) 模型 PTQ 量化配置
│   └── vision.yaml                 # 板端图像链路参数镜像（预处理/推理一致）
├── scripts/                          # 各脚本用途见 scripts/README.md
│   ├── extract_frames.py           # 切帧：容器视频 / 裸 MJPEG 流（自动识别）
│   ├── calibrate_camera.py         # 标定：棋盘视频/图片 → 内参 yaml
│   ├── prepare_frames.py           # 预处理：去畸变+补偿 → 640x640（与板端一致）
│   ├── select_frames.py            # 筛图：剔类别 + 质量加权抽样
│   ├── train_yolo11n.py            # 训练（自动恢复/重打 head 补丁）
│   ├── modify_ultralytics.py       # 拆分输出头为 6 tensor
│   ├── export_onnx.py              # 导出 ONNX（6 输出）
│   ├── prepare_calibration.py      # 生成 PTQ 校准数据(.rgb)
│   └── quantize.sh                 # Docker 内 PTQ 量化
├── data/                           # 本地训练数据集（不入库）
│   └── auv/                        # 示例：AUV 水下数据集(3类, YOLO格式)
├── docs/
│   └── tutorial_zh.md              # 完整部署教程（含踩坑记录）
└── output/                         # 量化产物（不入库）
    └── yolo11n_detect_bayese_640x640_nv12.bin
```

---

## 关键技术

| 技术 | 说明 |
|---|---|
| **Softmax BPU 优化** | C2PSA 的 Softmax 默认在 CPU 运行会把模型拆成 2 个 BPU 子图，性能暴跌；用 `node_info` 指定 Softmax 到 BPU（int16）后合并为 **1 个子图** |
| **6 输出头拆分** | 修改 Detect.forward 输出 3×bbox 特征 + 3×cls 分数（如 `[1,80,80,64]`+`[1,80,80,3]`），利于 BPU 与板端后处理 |
| **NV12 输入** | BPU 原生 NV12，省去 CPU 侧 BGR→RGB 转换 |
| **O3 编译** | 编译器最高优化等级 |

---

## 性能参考

COCO 预训练 YOLO11n 量化后（640x640 nv12）：

| 指标 | 值 |
|---|---|
| BPU 子图数 | 1 |
| 输出余弦相似度 | >0.95 |
| BPU 推理延迟 | ~10.8ms（仿真 ~7ms） |
| 端到端 | ~47 FPS |

本仓库 AUV 数据集（912 训练图，3 类）微调 30 epochs：mAP50 **0.995** / mAP50-95 **0.886**。

详细实战记录见 [docs/tutorial_zh.md](docs/tutorial_zh.md)，官方工具链文档见 [RDK X5 算法工具链 V1.2.8](https://developer.d-robotics.cc/rdk_x_doc/Advanced_development/toolchain_development/rdk_x5/oe_toolchain_v1_2_8)。

---

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE)。

<div align="center">

**[⬆ 返回顶部](#top)**

</div>
