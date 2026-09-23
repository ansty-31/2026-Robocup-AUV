# configs/ 配置说明

> ⚠️ **板端相关配置以板端实机为准**。本目录中的 `vision.yaml`、`front_camera.yaml` 都是
> **板端工程的只读镜像**，唯一权威来源是 **RM 板子实机**（SSH `sunrise@192.168.137.10`
> 上的 `/home/sunrise/AUV/`）。
> **修改流程：先改板端 → 再从板子读取同步到本目录**，不要只改本地。
>
> 本仓库（`RDKX5-YOLOv11n-`）对板端**只读**：只读取板端信息来对齐本地预处理过程，
> **不修改板端代码**。注意 `/home/ansty/RDKX5/auv_vision/auv_vision/` 是板端工程的
> **本地开发副本**，会与实机有意不一致（例如本地 `SIM_MODE=True` 跑虚拟、实机 `False`
> 跑真机），**不能当作权威来源**。

| 本地文件 | 来源 | 用途 |
|---|---|---|
| `vision.yaml` | 板子实机 `/home/sunrise/AUV/cfg/vision.yaml`（只读镜像） | 图像链路 / 相机 / 模型 / 任务参数；PC 端 `prepare_frames.py` 读取它，保证**训练图与板端推理同参数** |
| `front_camera.yaml` | 板子实机 `/home/sunrise/AUV/cfg/front_camera.yaml`（只读镜像） | **前视相机标定（K / dist）＝ 水下 / 上机使用的那一份**，训练侧去畸变与板端推理**共用同一份**。2026-09-23 起内容为 AUV_5 良好水质重标结果（fx≈1207.6，RMS 0.546px），详见下节 |
| `front_camera_air.yaml` | 板子实机 `/home/sunrise/AUV/cfg/front_camera_air.yaml`（只读镜像） | **岸上 / 空气有效内参**（fx≈1078，dist 全 0，由卷尺 + PnP 深度反推）。**不用于上机推理**，保留供 PnP 分析、折射因子与深度标尺解释 |
| `backup/front_camera_AUV1_water_fx782.yaml` | 归档 | 旧的水下标定（fx=782.5，来自 `data/AUV_1/board`）。**已不参与任何链路**，仅留档 |
| `backup/front_camera_board_A_fx782_20260923.yaml` | 归档（2026-09-23 换版前另存） | **与上一份是同一套内参**：`camera_matrix` 与 `distortion_coefficients` 逐项数值相同（fx=782.54、dist[0:2]=[-0.4914, 0.3784]），只是多了 `FILE-HEAD RULE` 等注释（32 行 vs 18 行）。换到 AUV_5 新标定（fx≈1207.6）前留的存底，**不参与任何链路** |
| `yolo11n_config.yaml` | 本仓库（PC 侧） | detect 模型 PTQ 量化配置（onnx 路径 / Softmax node_info / 输出前缀） |
| `gate_kpt_config.yaml` | 本仓库（PC 侧） | pose（gate 4 角点）模型 PTQ 量化配置 |

## 与板端同步（只读）

```bash
BOARD_SSH=sunrise@192.168.137.10                    # 板子实机（权威）
B=/home/sunrise/AUV

scp "$BOARD_SSH:$B/cfg/vision.yaml"       configs/vision.yaml
scp "$BOARD_SSH:$B/cfg/front_camera.yaml" configs/front_camera.yaml
# 只读参考：板端预处理实现（用于核对链路，不要改板端）
scp "$BOARD_SSH:$B/common/preprocess.py"  /tmp/board_preprocess.py

# 核对是否一致（无输出即一致）
diff <(ssh "$BOARD_SSH" "cat $B/cfg/vision.yaml")       configs/vision.yaml
diff <(ssh "$BOARD_SSH" "cat $B/cfg/front_camera.yaml") configs/front_camera.yaml
```

同步后重跑预处理即可让训练图跟上板端参数：
```bash
python scripts/1_prepare/prepare_frames.py <图片目录> --config configs/vision.yaml
```

## 前视标定换版记录（2026-09-23）

**决策：C 上位，A 归档，B 保留。** 三份内参的来历与实测对比：

| 代号 | 文件 | fx | 来历 |
|---|---|---|---|
| **A**（已归档） | `backup/front_camera_AUV1_water_fx782.yaml` | 782.5 | `data/AUV_1/board`（2298 张，水下），35 视图自拟合 RMS 0.705 |
| **B**（保留） | `front_camera_air.yaml` | ≈1078 | 卷尺三等独立测法 + PnP 深度对标真值（≤6%）；dist 全 0 |
| **C**（在用） | `front_camera.yaml` | **1207.6** | `data/AUV_5/raw-data/calibration-{1,2,3}`（1443 张，良好水质水下），83 视图，**RMS 0.546** |

**判据一：棋盘网格直线度**（去畸变后角点偏离理想方格经单应变换的 RMS，px@1280；180 张 AUV_5 棋盘图）

| 内参 | n | 中位 | p90 | 最大 | 无解/异常 |
|---|---|---|---|---|---|
| A 782.5 | 115 | 1.86 | **307.39** | **872.72** | 2 |
| B 1078 | 117 | 1.10 | 1.35 | 1.71 | 0 |
| **C 1207.6** | 117 | **0.39** | **0.49** | **1.27** | **0** |

**判据二：逐图 solvePnP 重投影交叉验证**（新棋盘图 76 张）：A 中位 1.98px 且 **20 张 >5px、2 张 >50px**；B 1.63px / 0 离群；**C 0.33px / 0 离群**。A 在**它自己的**素材上也有 2 张 >5px、1 张 >50px —— 欠约束拟合的典型特征。

**判据三：可用范围**（`experiment/scripts/exp_distortion/calib_diagnose.py`）

| 指标 | A 782.5 | **C 1207.6** |
|---|---|---|
| 可逆的原始像素半径 | 520 px（画面角点 69%） | **749 px（94%）** |
| 可去畸变的像素占比 | 63.5% | **93.7%** |
| 局部放大倍率 ≤1.2 的覆盖 | 24.0% | **74.5%** |
| 最外圈倍率最大值 | 6.34 | **2.04** |

眼观对比图：`runs/calib_auv5/compare_dives/AUV_{1..5}.jpg`（五次下水各 10 张原帧 × 4 列）与 `runs/calib_auv5/compare_zoom/calibration-{1,2,3}_zoom.jpg`（棋盘区域放大，判读最直接）。

**仍未定案的一点**：C 拟合出 fx≈1208，与板端卷尺+PnP 的空气值 1078 差 **+12%**（两者都远离 782.5）。要定案，需要在水下做一次**卷尺 / PnP 深度对标真值**（方法照 `front_camera_air.yaml` 注释里那套）。在此之前 C 仍是在水数据上表现最好的一份。

**棋盘覆盖仍未到四角**：AUV_5 棋盘角点最远只到画面角点距离的 **86%**（判据要求 ≥90%），外圈畸变参数仍是多项式外推。见 `runs/calib_auv5/diag_new.json`。


## 一致性核对结论（2026-09-11）

| 项目 | 板端权威实现 | 本仓库对应 | 结论 |
|---|---|---|---|
| 图像链路参数 | `cfg/vision.yaml`（`image.*`、`camera.front.*`、`model.input_size`） | `configs/vision.yaml` | **逐字节一致**（2026-09-11 已按板端实机状态重新同步） |
| 前视标定 | `cfg/front_camera.yaml` | `configs/front_camera.yaml` | **逐字节一致**（2026-09-23 起内容为 AUV_5 重标结果 C；旧的 782.5 已归档到双方 `backup/`） |
| 预处理链路 | `common/preprocess.py::ModelPreprocessor.process`：`calibration_maps`（`alpha=0` + `CV_16SC2`）→ `resize(640)` → `enhance`（gains → **clip>0 时** LAB-CLAHE(8×8，对象缓存) → gamma LUT（缓存）） | `scripts/1_prepare/prepare_frames.py`：同三函数、**同顺序** | **逐像素一致**（已在真实帧上验证 max diff = 0） |
| detect 解码契约 | `common/detector.py::decode_yolo11_split`：每尺度 `C=64`(reg, DFL logits) + `C=nc`(cls logits)，`(1,g,g,C)` NHWC；sigmoid/softmax/DFL 板端还原 | `export_onnx.py --task detect` 导出 6 张量 | **一致** |
| pose 解码契约 | `gate/gate_decode.py::decode_yolo11_kpt`：每尺度 64 / nc / `3*kpt_dim`，kpt `x,y=(raw*2+网格索引)`、`v` 为 logit | `export_onnx.py --task pose` 导出 9 张量 | **一致**（2026-09-12 修正了此前多写的 `-0.5`） |

> 结论：本仓库对板端只做**只读镜像 + 契约对齐**；训练侧任何参数改动都应先在板端落地，
> 再同步到 `configs/`，避免训练/推理分布漂移。

### 2026-09-11 板端预处理改动（本次已跟进）

板端 `common/preprocess.py::ModelPreprocessor.process` 把 **`enhance` 从「缩放前」挪到
「缩放后」**（在 640 上做，省约 2/3 耗时），并把 CLAHE 对象 / gamma LUT 改为缓存复用。

- CLAHE 是**分块局部算子**，`先缩放再补偿` ≠ `先补偿再缩放`，像素结果不同：
  实测真实帧上旧顺序与本仓库新顺序相差 **最大 7 灰阶 / 均值 0.75**（JPEG 前）。
- 本仓库 `prepare_frames.py` 已同步为 `remap → resize(640) → enhance`，与板端
  **逐像素一致**（真实帧验证 max diff = 0）。
- 影响评估（2026-09-11 实测）：差异很小（真实帧最大 7 灰阶 / 均值 0.75），
  已决定 **AUV_1 训练集不重跑**、**AUV_2 暂不重跑**；新素材 **AUV_3 直接用新顺序**
  生成（`data/AUV_3/processed_640/`）。

> 注：板端 `model.fast_nv12` 只影响**推理侧** BGR→NV12 转换（`detector.py` 里
> `if S.get("vision.model.fast_nv12", False)`），与训练图预处理链路无关；
> 本镜像按板端实机状态逐字节同步（当前 `true`，即启用 cv2 快速 NV12）。
