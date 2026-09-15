> ⚠️ 本文是项目**介绍/宣传版**，目录结构已重构：脚本按阶段分 `scripts/1_prepare|2_train|3_export/`，
> 权重在 `weights/`，配置在 `configs/`。**以根目录 [README.md](../README.md) 为准**（命令、分区、输出契约）。

<div id="top"></div>

# YOLO11n Train + RDK X5 Deployment 🚀

English | [简体中文](README_ZH.md)

<div align="center">

![YOLO11n](https://img.shields.io/badge/YOLO11-n-blue)
![RDK X5](https://img.shields.io/badge/RDK-X5-green)
![License](https://img.shields.io/badge/license-MIT-yellow)

**Deploy YOLO11n on Horizon RDK X5: custom dataset training → ONNX export → PTQ quantization → on-board inference**

</div>

---

## Project Overview

End-to-end workflow to run **YOLO11n** on the **Horizon RDK X5**:

```
custom dataset → fine-tune YOLO11n → split 6-output head → export ONNX
→ prepare calibration data → hb_mapper PTQ quantization (nv12) → *.bin → deploy to RDK X5
```

The Softmax operator inside the new YOLO11 C2PSA block defaults to CPU, splitting the model into 2 BPU subgraphs and killing performance. This project pins Softmax to the **BPU (int16)** via `node_info`, combines it with a 6-output head split and NV12 input, and reaches real-time inference on RDK X5 (~10.8ms BPU latency, ~47 FPS end-to-end with the COCO-pretrained baseline).

## Pipeline

```
data/xxx/data.yaml                    ① prepare dataset
        ↓
scripts/train_yolo11n.py              ② fine-tune (pretrained yolo11n.pt)
        ↓  best.pt → yolo11n.pt
scripts/modify_ultralytics.py         ③ split head into 6 tensors (auto backup/verify)
scripts/export_onnx.py                ④ export ONNX (6 outputs)
        ↓  yolo11n.onnx
scripts/prepare_calibration.py        ⑤ generate 100 calibration samples (.rgb)
scripts/quantize.sh                   ⑥ hb_mapper PTQ quantization in Docker
        ↓
output/yolo11n_detect_bayese_640x640_nv12.bin   ⑦ scp → RDK X5
```

> Step ③ patches `ultralytics/nn/modules/head.py` inside your Python environment. Training needs the original head, so **train first, patch later** — `train_yolo11n.py` handles this ordering automatically.

## Quick Start

### Requirements

| Item | Requirement |
|---|---|
| Host | Ubuntu 22.04 + Docker |
| Training env | conda env `yolov8` (`ultralytics==8.3.0`, optional GPU) |
| Toolchain | OpenExplorer **v1.2.8** image: `openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310` |

```bash
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker load < docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
```

### ① Prepare dataset (YOLO format, see `data/auv/data.yaml`)

```yaml
path: data/auv
train: train/images
val: valid/images
names: [red_ball, blue_ball, gate]
```

#### (Optional) Real camera: calibration + frame extraction

To make **training images and on-board inference images pixel-identical
(undistorted + compensated + 640x640)** (script index: [scripts/README.md](scripts/README.md)):

```bash
# 1) checkerboard (video or image folder) → intrinsics yaml (RMS<=0.8px gate)
#    the board must cover corners/edges, otherwise edge distortion is extrapolated
python scripts/calibrate_camera.py data/AUV_1/board \
    --cols 11 --rows 8 --square-mm 20 --output configs/front_camera.yaml

# 2) recording → raw frames (auto-detects container video / raw MJPEG stream)
python scripts/extract_frames.py data/AUV_2/auv_xxx.mjpeg     # raw MJPEG, lossless split
python scripts/extract_frames.py data/AUV_1/rec_front.mp4 --step 5

# 3) sorted image set → training frames (undistort + WB/CLAHE/gamma + 640x640)
python scripts/prepare_frames.py data/AUV_2/<class_dir> --config configs/vision.yaml

# 4) (optional) filter with an existing model: drop unwanted classes + quality-weighted sampling
python scripts/select_frames.py data/AUV_2/datay/origin \
    --weights runs/detect/auv_v3/weights/best.pt --drop-classes red_ball \
    --quality-dir data/AUV_2/origin --keep 2000 --out-dir data/AUV_2/selected_2000
```

The per-frame chain mirrors the board-side `auv_vision/common/preprocess.py` exactly:
`cv2.remap(undistort) → resize(640,640) → white-balance gains → LAB-CLAHE → gamma`.
**(Note the order: as of 2026-09-11 the board enhances *after* resizing to 640; the two
steps are not commutative.)**

### ② Train (fine-tune from pretrained yolo11n.pt)

```bash
python scripts/train_yolo11n.py --data data/auv/data.yaml --epochs 100 --imgsz 640 --device 0   # GPU
```

best.pt is copied to `yolo11n.pt` for export, and the 6-output patch is re-applied automatically. Skip this step to use the official COCO-pretrained weights directly.

### ③ ④ Export ONNX (detect = 6 outputs / pose = 9 outputs)

```bash
# detection (Detect head → 6 tensors: 3×bbox + 3×cls)
python scripts/modify_ultralytics.py --task detect
python scripts/export_onnx.py                     # yolo11n.onnx (opset 11, 640, 6 outputs)

# keypoints (Pose head → 9 tensors: per scale bbox64 + cls nc + kpt 12; gate 4 corners)
python scripts/modify_ultralytics.py --task pose
python scripts/export_onnx.py --task pose         # yolo11n-pose.onnx (9 outputs)
```

> RoboFlow keypoint spec (4 points TL,TR,BR,BL / Keypoint Detection / export "YOLOv8 Pose"),
> training command and the **output tensor contract** for the board-side
> `gate_decode.py` are documented in [scripts/README.md](scripts/README.md).

### ⑤ Prepare calibration data (e.g. COCO val2017)

```bash
python scripts/prepare_calibration.py --coco-path /path/to/coco/val2017 --num-images 100
```

### ⑥ Quantize

```bash
./scripts/quantize.sh                # hb_mapper makertbin with configs/yolo11n_config.yaml
```

Key settings in `configs/yolo11n_config.yaml`:

```yaml
model_parameters:
  node_info: {"/model.10/m/m.0/attn/Softmax": {'ON': 'BPU','InputType': 'int16','OutputType': 'int16'}}  # core: Softmax on BPU
input_parameters:
  input_type_rt: 'nv12'
compiler_parameters:
  compile_mode: 'latency'
  optimize_level: 'O3'
```

### ⑦ Deploy to RDK X5

```bash
scp output/yolo11n_detect_bayese_640x640_nv12.bin sunrise@<RDK_IP>:~/models/
```

## Repository Layout

```
RDKX5-YOLOv11n-/
├── README.md / README_ZH.md
├── LICENSE / CONTRIBUTING.md
├── requirements.txt
├── configs/
│   ├── yolo11n_config.yaml          # detect PTQ config (Softmax node_info)
│   ├── gate_kpt_config.yaml         # gate keypoint (pose) PTQ config
│   └── vision.yaml                  # board image-pipeline mirror (train/infer parity)
├── scripts/                         # see scripts/README.md for the full index
│   ├── extract_frames.py            # split frames: container video / raw MJPEG
│   ├── calibrate_camera.py          # checkerboard (video/images) → intrinsics yaml
│   ├── prepare_frames.py            # undistort+compensate → 640x640 (board-aligned)
│   ├── select_frames.py             # model-based filtering + quality sampling
│   ├── train_yolo11n.py             # train (auto restore/re-apply head patch)
│   ├── modify_ultralytics.py        # split head into 6 tensors
│   ├── export_onnx.py               # export ONNX (6 outputs)
│   ├── prepare_calibration.py       # PTQ calibration data (.rgb)
│   └── quantize.sh                  # PTQ quantization in Docker
├── data/                            # local datasets (git-ignored)
│   └── auv/                         # example: 3-class AUV underwater dataset
├── docs/tutorial_zh.md              # full tutorial (incl. troubleshooting)
└── output/                          # quantization artifacts (git-ignored)
    └── yolo11n_detect_bayese_640x640_nv12.bin
```

## Key Techniques

| Technique | Description |
|---|---|
| Softmax on BPU | `node_info` pins C2PSA Softmax to BPU int16 → single BPU subgraph instead of 2 |
| 6-output head | Detect.forward returns 3×bbox features + 3×cls scores (e.g. `[1,80,80,64]` + `[1,80,80,3]`) |
| NV12 input | BPU-native NV12, no CPU BGR→RGB conversion |
| O3 compile | highest compiler optimization level |

## Performance

COCO-pretrained YOLO11n after quantization (640x640 nv12): 1 BPU subgraph, output cosine similarity >0.95, ~10.8ms BPU latency (~7ms simulated), ~47 FPS end-to-end.

AUV dataset fine-tune (912 training images, 3 classes, 30 epochs): mAP50 **0.995** / mAP50-95 **0.886**.

Full write-up (Chinese): [docs/tutorial_zh.md](docs/tutorial_zh.md) · Official toolchain docs: [RDK X5 Algorithm Toolchain V1.2.8](https://developer.d-robotics.cc/rdk_x_doc/Advanced_development/toolchain_development/rdk_x5/oe_toolchain_v1_2_8).

## License

MIT — see [LICENSE](LICENSE).

<div align="center">

**[⬆ Back to top](#top)**

</div>
