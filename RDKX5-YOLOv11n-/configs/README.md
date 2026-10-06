# configs/ 配置说明

> ⚠️ **2026-10-06 更新（在读任何一条之前先看这段）**
>
> 1. **权威来源换成 `/home/sunrise/Desktop/AUV_New/`**（在用副本，`deploy_to_board.sh` 的目标）。
>    以前本文写的 `/home/sunrise/AUV/` 是**旧树（09-17，A 域时代）**，已于 **2026-10-06 归档删除**
>    （317 MB → `AUV_New/bak/legacy_Adomain_tree_20261006.tar.gz`，md5 `3a7cda63…`），
>    连带影响与恢复方式见 [`../cleanup_record/reports/BOARD_Adomain_retire_20261006.md`](../cleanup_record/reports/BOARD_Adomain_retire_20261006.md)。
> 2. **A 域已彻底退役**：在用副本的 A 分支已于 2026-10-01 移除，旧树被删后**板端已无任何能跑 A 域的地方**。
>    规矩：识别链路只有 **D + wb**；**夹取小球只做 `resize 640×640` 送 BPU，不做任何图像处理**；
>    不要再给 `vision.image` 加 `chain` 之类的域开关。
> 3. ⚠️ **本目录的 `vision.yaml` 已确认是"混血/滞后"镜像**（部分字段来自旧树、部分来自 D 域），
>    不要再当权威；它当前只被 `../scripts/1_prepare/prepare_frames.py` 读 `image.* / camera.* / model.input_size`。
>    要权威值请按下面「与板端同步」重新拉。

> **修改流程：先改板端 → 再从板子读取同步到本目录**，不要只改本地。
>
> 本仓库（`RDKX5-YOLOv11n-`）对板端**只读**：只读取板端信息来对齐本地预处理过程，
> **不修改板端代码**（2026-10-06 那次旧树清理是用户明确要求的手工运维，记录见上）。
> 注意 `/home/ansty/RDKX5/auv_vision/auv_vision/` 是板端工程的
> **本地开发副本**，会与实机有意不一致（例如本地 `SIM_MODE=True` 跑虚拟、实机 `False`
> 跑真机），**不能当作权威来源**。

> **路径书写约定（2026-09-26 逐条核对磁盘后补）**：本文里作为「指向」用的独立路径都相对**本文件所在目录**
> （`configs/`）书写 —— 本目录内的配置文件写 `vision.yaml`、`backup/front_camera_AUV1_water_fx782.yaml`；本目录之外的写
> `../scripts/…`、`../data/…`、`../runs/…`。`bash` 代码块里的命令按仓库根执行，保持 `scripts/…` 形式。
> 板端路径（`/home/sunrise/Desktop/AUV_New/…`；旧树 `/home/sunrise/AUV/` 已于 2026-10-06 删除）
> 与三方/板端源码路径会显式标注，它们不在本仓库里。

| 本地文件 | 来源 | 用途 |
|---|---|---|
| `vision.yaml` | 板子实机 `/home/sunrise/Desktop/AUV_New/cfg/vision.yaml`（只读镜像；⚠️ 现副本是旧树时代拉的，混血滞后） | 图像链路 / 相机 / 模型 / 任务参数；PC 端 `../scripts/1_prepare/prepare_frames.py` 读取它，保证**训练图与板端推理同参数** |
| `front_camera.yaml` | 板子实机 `/home/sunrise/Desktop/AUV_New/cfg/front_camera.yaml`（只读镜像） | **前视相机标定（K / dist）＝ 水下 / 上机使用的那一份**，训练侧去畸变与板端推理**共用同一份**。2026-09-23 起内容为 AUV_5 良好水质重标结果（fx≈1207.6，RMS 0.546px），详见下节 |
| `front_camera_air.yaml` | 板子实机 `/home/sunrise/Desktop/AUV_New/cfg/front_camera_air.yaml`（只读镜像） | **岸上 / 空气有效内参**（fx≈1078，dist 全 0，由卷尺 + PnP 深度反推）。**不用于上机推理**，保留供 PnP 分析、折射因子与深度标尺解释 |
| `backup/front_camera_AUV1_water_fx782.yaml` | 归档 | 旧的水下标定（fx=782.5，来自 `../data/calib/AUV_1_board`）。**已不参与任何链路**，仅留档 |
| `backup/front_camera_board_A_fx782_20260923.yaml` | 归档（2026-09-23 换版前另存） | **与上一份是同一套内参**：`camera_matrix` 与 `distortion_coefficients` 逐项数值相同（fx=782.54、dist[0:2]=[-0.4914, 0.3784]），只是多了 `FILE-HEAD RULE` 等注释（32 行 vs 18 行）。换到 AUV_5 新标定（fx≈1207.6）前留的存底，**不参与任何链路** |
| `yolo11n_config.yaml` | 本仓库（PC 侧） | detect 模型 PTQ 量化配置（onnx 路径 / Softmax node_info / 输出前缀） |
| ~~`gate_kpt_stage1_config.yaml`~~ → **已归档** `_archive/auv6/superseded_by_v3_20260928/`（R4/g240_i16；板端仓库自带该 bin） | 已归档 | 由 `gate_kpt_stage1_v3_config.yaml` 接替 |
| ~~`gate_kpt_stage1_v2_config.yaml`~~（已随 v2 归档至 `_archive/auv6/superseded_by_v3_20260928/`） | 已归档 | 随 v2 一并归档（v3 用 `configs/gate_kpt_stage1_v3_config.yaml`） |
| `gate_kpt_stage1_clear_only_config.yaml` | 本仓库（PC 侧） | **Plan B（2026-09-28）**：阶段一权重热启动、只用 `AUV_6_clear` 单训；onnx → `weights/new/yolo11n-pose.stage1_clear_only.onnx`，校准集同 v2，前缀 `gate_kpt_stage1_clear_only_bayese_640x640_nv12` |
| `gate_kpt_config.yaml` | 本仓库（PC 侧） | pose（gate 4 角点）模型 PTQ 量化配置 |

## 与板端同步（只读）

```bash
BOARD_SSH=sunrise@192.168.137.10                           # 板子实机
B=/home/sunrise/Desktop/AUV_New                            # ★ 在用副本（权威）
# 旧路径 /home/sunrise/AUV 已于 2026-10-06 归档删除，不要再指向它

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
| **A**（已归档） | `backup/front_camera_AUV1_water_fx782.yaml` | 782.5 | `../data/calib/AUV_1_board`（2298 张，水下），35 视图自拟合 RMS 0.705 |
| **B**（保留） | `front_camera_air.yaml` | ≈1078 | 卷尺三等独立测法 + PnP 深度对标真值（≤6%）；dist 全 0 |
| **C**（在用） | `front_camera.yaml` | **1207.6** | `../data/frames/AUV_5_gate_calib/calibration-{1,2,3}`（1443 张，良好水质水下），83 视图，**RMS 0.546** |

**判据一：棋盘网格直线度**（去畸变后角点偏离理想方格经单应变换的 RMS，px@1280；180 张 AUV_5 棋盘图）

| 内参 | n | 中位 | p90 | 最大 | 无解/异常 |
|---|---|---|---|---|---|
| A 782.5 | 115 | 1.86 | **307.39** | **872.72** | 2 |
| B 1078 | 117 | 1.10 | 1.35 | 1.71 | 0 |
| **C 1207.6** | 117 | **0.39** | **0.49** | **1.27** | **0** |

**判据二：逐图 solvePnP 重投影交叉验证**（新棋盘图 76 张）：A 中位 1.98px 且 **20 张 >5px、2 张 >50px**；B 1.63px / 0 离群；**C 0.33px / 0 离群**。A 在**它自己的**素材上也有 2 张 >5px、1 张 >50px —— 欠约束拟合的典型特征。

**判据三：可用范围**（`../experiment/scripts/exp_distortion/calib_diagnose.py`）

| 指标 | A 782.5 | **C 1207.6** |
|---|---|---|
| 可逆的原始像素半径 | 520 px（画面角点 69%） | **749 px（94%）** |
| 可去畸变的像素占比 | 63.5% | **93.7%** |
| 局部放大倍率 ≤1.2 的覆盖 | 24.0% | **74.5%** |
| 最外圈倍率最大值 | 6.34 | **2.04** |

眼观对比图：`runs/auv5/calib/compare_dives/AUV_{1..5}.jpg`（五次下水各 10 张原帧 × 4 列）与 `runs/auv5/calib/compare_zoom/calibration-{1,2,3}_zoom.jpg`（棋盘区域放大，判读最直接）。

> ⚠️ **已过时（2026-09-26）：上面两个眼观对比目录已不存在。**
> `../runs/auv5/calib/` 下的 `compare_dives/`、`compare_zoom/`（连同 `compare/`、`qc_views/`、`board_all/`）
> 共 88 MB 的 QC/对比图已于 **2026-09-24 清理**（只删图、留结论），原句保留作历史记录。标定**结果**仍在：
> `../runs/auv5/calib/front_camera_good.yaml`、`../runs/auv5/calib/diag_new.json`、
> `../runs/auv5/calib/diag_old.json`、`../runs/auv5/calib/diag_auv1board.json`、
> `../runs/auv5/calib/calib.log`、`../runs/auv5/calib/prep_calib.log`（2026-09-26 逐条 `ls` 核实）；
> 清理记录见 `../cleanup_record/reports/CLEANUP_2026-09-24.md`。要重出对比图，按
> `../experiment/scripts/exp_distortion/make_contact_sheet.py`、`../experiment/scripts/exp_distortion/board/`
> 下的板端对拍脚本重跑。

**仍未定案的一点**：C 拟合出 fx≈1208，与板端卷尺+PnP 的空气值 1078 差 **+12%**（两者都远离 782.5）。要定案，需要在水下做一次**卷尺 / PnP 深度对标真值**（方法照 `front_camera_air.yaml` 注释里那套）。在此之前 C 仍是在水数据上表现最好的一份。

**棋盘覆盖仍未到四角**：AUV_5 棋盘角点最远只到画面角点距离的 **86%**（判据要求 ≥90%），外圈畸变参数仍是多项式外推。见 `../runs/auv5/calib/diag_new.json`。


## 一致性核对结论（2026-09-11）

| 项目 | 板端权威实现 | 本仓库对应 | 结论 |
|---|---|---|---|
| 图像链路参数 | `cfg/vision.yaml`（`image.*`、`camera.front.*`、`model.input_size`） | `vision.yaml` | **逐字节一致**（2026-09-11 已按板端实机状态重新同步） |
| 前视标定 | `cfg/front_camera.yaml` | `front_camera.yaml` | **逐字节一致**（2026-09-23 起内容为 AUV_5 重标结果 C；旧的 782.5 已归档到双方 `backup/`） |
| 预处理链路 | `common/preprocess.py::ModelPreprocessor.process`：`calibration_maps`（`alpha=0` + `CV_16SC2`）→ `resize(640)` → `enhance`（gains → **clip>0 时** LAB-CLAHE(8×8，对象缓存) → gamma LUT（缓存））**〔链序描述已过时，见下方 ⚠️〕** | `../scripts/1_prepare/prepare_frames.py`：同三函数、**同顺序** | **逐像素一致**（已在真实帧上验证 max diff = 0） |
| detect 解码契约 | `common/detector.py::decode_yolo11_split`：每尺度 `C=64`(reg, DFL logits) + `C=nc`(cls logits)，`(1,g,g,C)` NHWC；sigmoid/softmax/DFL 板端还原 | `../scripts/3_export/export_onnx.py --task detect` 导出 6 张量 | **一致** |
| pose 解码契约 | `gate/gate_decode.py::decode_yolo11_kpt`：每尺度 64 / nc / `3*kpt_dim`，kpt `x,y=(raw*2+网格索引)`、`v` 为 logit | `../scripts/3_export/export_onnx.py --task pose` 导出 9 张量 | **一致**（2026-09-12 修正了此前多写的 `-0.5`） |

> 结论：本仓库对板端只做**只读镜像 + 契约对齐**；训练侧任何参数改动都应先在板端落地，
> 再同步到 `configs/`，避免训练/推理分布漂移。

### 2026-09-11 板端预处理改动（本次已跟进）

板端 `common/preprocess.py::ModelPreprocessor.process` 把 **`enhance` 从「缩放前」挪到
「缩放后」**（在 640 上做，省约 2/3 耗时），并把 CLAHE 对象 / gamma LUT 改为缓存复用。

- CLAHE 是**分块局部算子**，`先缩放再补偿` ≠ `先补偿再缩放`，像素结果不同：
  实测真实帧上旧顺序与本仓库新顺序相差 **最大 7 灰阶 / 均值 0.75**（JPEG 前）。
- 本仓库 `../scripts/1_prepare/prepare_frames.py` 现为 `resize(640) → enhance → remap@640`（2026-09-23 换序后），
  与板端 `common/preprocess.py` 一致（真实帧验证 max diff = 0）。
- 影响评估（2026-09-11 实测）：差异很小（真实帧最大 7 灰阶 / 均值 0.75），
  已决定 **AUV_1 训练集不重跑**、**AUV_2 暂不重跑**；新素材 **AUV_3 直接用新顺序**
  生成（当时落在 `data/AUV_3/processed_640/`；按 2026-09-25 的 data/ 分区规则，对应 `data/frames/AUV_3_processed_640/`）。

> ⚠️ **上面两处链序描述已过时（2026-09-26 核对代码后补，原句保留作历史）**：板端预处理在
> **2026-09-23 又换过一次序**，定稿为 **D 域 / P2**：`resize(640) → enhance → remap@640`
> （`remap@640 ≈6 ms` 比 `remap@720p ≈11.5 ms` 省时，几何等价，因为 `nk640 = S·nk720`）。
> 「`calibration_maps → resize(640) → enhance`」是 **09-23 之前**的旧顺序（09-11 那次改动的产物）。
> 依 据（本仓库内可核）：`../scripts/1_prepare/prepare_frames.py`（`main()` 内的链序注释与实现）、
> 板端**本地开发副本** `/home/ansty/RDKX5/auv_vision/auv_vision/common/preprocess.py`
> （`process()` 的注释：「旧：remap@720p → resize(640) → enhance / 新：resize(640) → enhance → remap@640」，
> 并新增 `calibration_maps_640`）。**注意**：本轮**没有 SSH 板端实机复核**（与本文开头的
> 「先改板端 → 再同步」流程不冲突：这只是把本地两份实现的实际链序写清楚）。下次同步
> `vision.yaml` 时建议顺手 `diff` 一次板端 `common/preprocess.py`，确认 D 域已上机。

> ⚠️ **AUV_3 的 640 处理图目录路径不对（2026-09-26 核实）**：上文写的
> `data/frames/AUV_3_processed_640/`（= `../data/frames/AUV_3_processed_640/`）**在磁盘上不存在**
> （`../data/frames/` 下只有
> `../data/frames/AUV_1_rec_front_frames/`、`../data/frames/AUV_2_auv_20260910_172208_frames/`、
> `../data/frames/AUV_3_auv_20260911_201754_frames/`、`../data/frames/AUV_4_auv_4_frames/`、
> `../data/frames/AUV_5_gate_calib/`）。AUV_3 现存的 640×640 图是抽样后的
> `../_archive/auv5/derived_AUV_3_selected_2000/`（2000 张，实测 640×640）；原始 1280×720 帧在
> `../data/frames/AUV_3_auv_20260911_201754_frames/`（508 张）。**`processed_640` 本身已不在**，
> 要重跑就 `../scripts/1_prepare/prepare_frames.py` 现生成。

> ⚠️ **`gate_kpt_auv4_config.yaml` 已不在磁盘上（2026-09-26 核实）**：上一代（pose=AUV_4，09-17）
> 量化配置 `configs/gate_kpt_auv4_config.yaml` 在 `git status` 里是 `D`（工作区已删、未归档到
> `../_archive/`、也没进 `../cleanup_record/` 的任何 TSV），但 `../weights/README.md`（第 73 行）与
> `../output/README.md` 仍把它当作"回退用 bin 的配置"来引用。**要么取回、要么改那两处引用** —— 需人工定夺。
> 取回命令（2026-09-26 已验证可恢复：`git ls-files` 里在、`git show HEAD:<path>` 有内容；
> git 仓库根是上一级 `auv_vision/`，在项目根目录下按相对路径即可）：
> `git checkout -- configs/gate_kpt_auv4_config.yaml`。

> 注：上表还有一份 `front_camera_air.yaml`（板端 `cfg/front_camera_air.yaml` 的只读镜像），
> 上面的 `scp` 同步命令里**没有**列它 —— 它只用于岸上/PnP 分析，不在训练与推理链路上；
> 需要时按同样方式取：`scp "$BOARD_SSH:$B/cfg/front_camera_air.yaml" configs/front_camera_air.yaml`。

> 注：板端 `model.fast_nv12` 只影响**推理侧** BGR→NV12 转换（板端 `common/detector.py` 里
> `if S.get("vision.model.fast_nv12", False)`），与训练图预处理链路无关；
> 本镜像按板端实机状态逐字节同步（当前 `true`，即启用 cv2 快速 NV12）。
