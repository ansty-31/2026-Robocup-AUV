# weights/ 权重与模型分区

本目录存放**所有模型文件**（`.pt` / `.onnx`）。`*.pt`、`*.onnx`、`*.bin` 均不进 git
（体积大、可再生产生），本 README 用于说明“每个文件是什么、从哪来、给谁用”。

> 命名规则：**当前版本占“规范名”**（`yolo11n-pose.pt` / `yolo11n-pose.onnx`），
> **历史版本加 `.<代次>` 后缀**（`.auv3` / `.auv4` …）。换代不删旧文件，见文末「新模型上线三步」。

> ⚠️ 已过时（2026-09-26）：正文「本目录存放**所有**模型文件」不完全成立 —— 仓库根还有一个
> `yolov8n.pt`（md5 `95a24496…`，与 `_archive/root/yolov8n.pt` **逐字节相同**），既不在 `weights/`
> 也不在本 README 的任何一行里（无来源/用途记录，**待你确认是否归档或删除**）。
> 本 README 覆盖 `weights/` 目录本身；`.bin` 交付物在 `output/`（见下文专节）。

## 模型文件

> 下表路径一律**相对仓库根**（`2026-09-26` 逐条 `ls` 核过）。行序保持原样，新增的阶段一三个快照追加在表尾。

| 文件 | 角色 | 来源 | 使用者 |
|---|---|---|---|
| `weights/yolo11n.pt` | **检测模型**（3 类 `blue_ball/gate/red_ball`），md5 `91069d76…` | `scripts/2_train/train_yolo11n.py`（detect）产出 | 导出 detect ONNX、`scripts/1_prepare/select_frames.py` 筛图 |
| **`weights/yolo11n-pose.pt`** | **pose 模型（当前 = 2026-09-24 从 coco 重训版）**，md5 `2f36303a…` | 在 `data/datasets/AUV_5_gate-pose.yolov8`（**844 张** = train 675 / valid 85 / test 84，干净重划分；`nc=1` names `['gate']`、`kpt_shape [4,3]`）上**从官方 `weights/yolo11n-pose.coco.pt` 从零重训**，300/300 ep 无早停。val Pose mAP50-95 **0.9632** / test **0.9508**；面积比（预测/GT）中位 **0.9783**（auv5 是 0.960，内缩从 ~4% 降到 ~2%）。详见 [`runs/auv5/REPORT_auv5_pose.md`](../runs/auv5/REPORT_auv5_pose.md) §14 | 导出 pose ONNX（`--task pose`）、板端门 4 角点 |
| **`weights/yolo11n-pose.onnx`** | **pose ONNX（当前）**，md5 `f986b24f…`；已核对：**9 输出**、NHWC、kpt 通道 12、cls 通道 1、输入 `images 1×3×640×640`、opset 11 | `scripts/3_export/export_onnx.py --task pose`（需先 `scripts/3_export/modify_ultralytics.py --task pose` 打补丁） | `scripts/3_export/quantize.sh configs/gate_kpt_config.yaml`（PTQ 输入） |
| `weights/yolo11n-pose.coco.pt` | 官方 COCO pose 预训练（17 点），md5 `475cd7f6…` | 官方 asset（与官方 `yolo11n-pose.pt` release 同一文件） | **推荐的 pose 起点**：`--task pose --weights weights/yolo11n-pose.coco.pt` |
| `weights/yolo11n-pose.auv4.pt` | 上一版 pose（AUV_4 版，09-17 在 `data/datasets/AUV_4_PNP.kpt4.yolov8` 上微调，206 ep 早停 best=ep106，val mAP50-95 **0.952** / test **0.911**），md5 `3a5e87fa…`；同源训练记录 `_archive/runs/pose/auv4/` | 换代时按代次改名保留 | **回退用** |
| `weights/yolo11n-pose.auv4.onnx` | 上一版 pose ONNX（已核对 **9 输出** / kpt 通道 12 / opset 11），md5 `1e7785ca…` | 同上 | **回退用** |
| `weights/yolo11n-pose.auv3.pt` | 更早一版 pose（AUV_3 版），md5 `db2ceb28…`；同源训练记录 `_archive/runs/pose/PNP_v1/` | 手动备份 | **回退用** |
| `weights/yolo11n.onnx` | 检测 ONNX（**6 输出** = 3×(bbox 64 + cls 3)，输入 `images 1×3×640×640`、opset 11），md5 `0a2d394f…` | `scripts/3_export/export_onnx.py` | `scripts/3_export/quantize.sh`（PTQ 输入） |
| `weights/yolo11n-pose.stage1.pt` | 阶段一（线 C 分期重训）best.pt 快照，早停 240 ep / best=ep140 / `patience:100`，md5 `8b2325a6…`；与 `runs/auv5/train/stage1/weights/best.pt` **逐字节相同** | `data/datasets/AUV_5_stage1`（train 2352 / valid 233，`nc=1` gates，`kpt_shape [4,3]`）上从 `weights/yolo11n-pose.coco.pt` 起步训 300 ep 上限 | **未采用**：`output/preview/auv5_pose/tools/` 内三个脚本的默认 `--weights` |
| `weights/new/yolo11n-pose.stage1.onnx` | 阶段一 pose 的**9 输出** ONNX（opset 11，输入 `images 1×3×640×640`），10.7 MB；导出前需 `modify_ultralytics.py --task pose` | `scripts/3_export/export_onnx.py --task pose --model weights/new/yolo11n-pose.stage1_full.pt` | `configs/gate_kpt_stage1_config.yaml` → 量化 bin |
| `weights/yolo11n-pose.stage1_full.pt` | 阶段一 300 ep 全程版（`patience:0`，best 仍是 ep140），md5 `c890968f…`；**与 `stage1.pt` 在 ep140 那一整行（22 列 = epoch + 训练/val 损失 + 精度指标 + lr）逐位相同**（多跑的 60 ep 没改变 best），只有文件 md5 与尾部轮次不同 | `runs/auv5/train/stage1_full/weights/best.pt`（逐字节相同） | **未采用为产线**；`output/preview/auv5_pose/` 的当前基线（线 C） |

> ⚠️ **注（2026-09-26 补，路径/指纹已 `ls`+`md5sum` 核过）**：
> ① `weights/yolo11n.pt` 的 **`gate` 类实际无输出**（AUV_1 valid 152 张上零检出），门检测请用 pose 权重，
> 别拿这个 detect 权重当门检测器（依据见根 [`README.md`](../README.md) 注意事项）。
> ② 上表两个 `stage1*.pt` **没有对应的 ONNX / bin**（未导出、未量化），也都不是板端当前加载的模型。
> ③ **已合并（2026-09-26 用户决定）**：原 `weights/yolo11n-pose.stage1_es240.pt` 与
> `weights/yolo11n-pose.stage1.pt` 是**同一份权重**（md5 `8b2325a6…`，5,674,881 B，逐字节相同），
> 只是两个命名各被不同脚本引用 → 现只保留 `stage1.pt` 一个名字，重复名已删。
> **要恢复**：`cp runs/auv5/train/stage1/weights/best.pt weights/yolo11n-pose.stage1.pt`（同一 md5）。
> 记录见 `cleanup_record/MOVES_2026-09-26.md` §10。

> ⚠️ **已退役、不再存在的**：auv5 版 pose 权重/ONNX/bin（`f08f82ed…` / `05fd2d8b…` / `945fdd01…`）。
> 原因：auv5 按当时的静默默认**从 auv4 热身**，把 auv4 在**错误内参（A 标定）+ 旧链路**上学到的
> 「角点口径」一起继承了下来 —— 实测面积比只有 0.960–0.977（系统性内缩 ~2–4%，近距离肉眼可见
> 「点偏中间」）。删除前指纹见 [`runs/auv5/RETIRED_auv5.md`](../runs/auv5/RETIRED_auv5.md)。
> **板端目前仍跑 auv5 的 bin**，本地已无副本（详见 `output/README.md`）。

### ⚠️ 三个必须知道的口径

**① `yolo11n.pt` 的版本命名对不上（待你确认，我没有改动任何权重）**。按 md5 核对：

- `weights/yolo11n.pt` = `91069d76dbed46ca8639aead1f0f4cee`（mtime 09-12 14:06）
  ＝ `_archive/runs/detect/auv_v4/weights/best.pt`（**逐字节相同**）
- `_archive/runs/detect/auv_v3/weights/best.pt` = `d30de5dc6c3b038f415338733c9b1d40`（mtime 09-08）
  —— **是另一个模型**，且是它的**唯一本地副本**

即文档里说的 "AUV v3" 与实盘 md5 对不上（`weights/yolo11n.pt` 与名为 `auv_v4` 的训练目录同源）。

**② pose 训练的起点规则（2026-09-24 起，强制）** —— `train_yolo11n.py` 对 pose **缺省即报错**：

| 起点 | 含义 | 何时用 |
|---|---|---|
| `weights/yolo11n-pose.coco.pt` | 官方 COCO pose 17 点 | **推荐**：从零重训，不继承任何历史口径 |
| `weights/yolo11n-pose.pt` | 当前生产门模型 | 热身微调，**会继承上一代的角点口径与几何偏差** |

> 血泪教训（auv5）：按当时的静默默认从 auv4 热身 → 继承了错误内参时代的角点口径（面积比 0.960–0.977）；
> 而同数据、同增强、从 `coco.pt` 起步的对照臂面积比 **1.022（无内缩）**、mAP 还不降
> （0.9233 vs 0.9242）。完整证据见报告 §11–§12。

**③ `head.py` 的补丁生命周期（训练/推理 vs 导出，2026-09-26 起明确）** —— 这条流水线最常翻车的一处：

| 场景 | `head.py` 要哪一版 | 切换命令 |
|---|---|---|
| 训练 / `predict` / `val` / 筛图 | **原版**（未分裂输出头） | `python scripts/3_export/modify_ultralytics.py --restore` |
| 导出 ONNX | **补丁版**（detect **6** 输出 / pose **9** 输出） | `python scripts/3_export/modify_ultralytics.py --task detect`（pose 同理） |

`scripts/2_train/train_yolo11n.py` 的行为是「**restore → 训练 → 训练结束自动重打补丁**」（训练日志尾部
会打印「重新应用 pose 输出头补丁」并校验 9 个张量）。所以**训练刚结束直接 `predict` 会崩**
（输出头是分裂后的 9/6 张量），先 `--restore` 再 predict，导出前再按任务打补丁。原版备份在
ultralytics 包内同目录 `head.py.backup`（`/home/ansty/anaconda3/envs/yolov8/lib/python3.9/site-packages/ultralytics/nn/modules/head.py.backup`）。
凡遇「predict 张量形状/解包错误」「训练 loss 计算报错」，**第一嫌疑是补丁状态不对**，先核对再动模型代码。

### 去畸变域实验权重（2026-09-23，`experiment/scripts/train_domain_arms.sh` 产出）

四条臂都在 `data/mapped/*` 上从 `weights/yolo11n-pose.coco.pt` 起步、同日程（200 ep / seed 0），
**只换图像链路**；四个文件**不覆盖**正式权重，用于"哪种去畸变时机/补偿最准"的对比。
结论与逐帧明细见 [`experiment/runs/domain/INDEX.md`](../experiment/runs/domain/INDEX.md) 与
`experiment/runs/domain/reports/DOMAIN_RESULT.md`。

| 文件 | 链路 | val Pose mAP50-95 | 训练记录 |
|---|---|---|---|
| `weights/domain_B.pt`（md5 `f2b61f10…`） | B：不去畸变 | 0.9787 @ep199 | `experiment/runs/domain/A_B/` |
| `weights/domain_C.pt`（md5 `e01b7948…`） | C：720p 上去畸变 | **0.9879** @ep195 | `experiment/runs/domain/A_C/` |
| `weights/domain_D.pt`（md5 `bd229eda…`） | D：640 上去畸变 | 0.9727 @ep196 | `experiment/runs/domain/A_D/` |
| `weights/domain_D_noenh.pt`（md5 `34f490fb…`） | D + 无 enhance（LUT 白平衡 / 无 CLAHE） | 0.9853 @ep200 | `experiment/runs/domain/A_D_noenh/` |

> ⚠️ 已过时（2026-09-26）：上表的路径原文只写了文件名（`domain_B.pt` …），它们实际都在 `weights/` 下，
> 已补全为 `weights/domain_*.pt` 并补 md5；四个文件的 `ls` 结果与上表一一对应（2026-09-23 产出）。
> 表里的 mAP 数字是**当时**（200 ep 实验）的结论，保留不动。

> 注：`data/mapped/` 的对照数据集**已于 2026-09-23/24 清理**（结论已收口为 D 域 + LUT 白平衡、无 CLAHE；
> 数据集可由 `scripts/1_prepare/pose/map_pose_dataset.py` 重建，重建方式见
> [`cleanup_record/CLEANUP_2026-09-24.md`](../cleanup_record/CLEANUP_2026-09-24.md)）。
> **权重本身保留**，随时可复评。

## 量化产物（板端可加载的 `.bin`）不在本目录，统一放在 `output/`

| 产物 | 对应配置 | 板端用途 |
|---|---|---|
| `output/yolo11n_detect_bayese_640x640_nv12.bin`（md5 `a28156ba…`） | `configs/yolo11n_config.yaml` | 球/门检测（`model.path`） |
| **`output/gate_kpt_bayese_640x640_nv12.bin`** = **pose 当前采用**（09-24），md5 `9ac61773…` | `configs/gate_kpt_config.yaml` | 门 4 角点（`model.task_models.gate.path`） |
| `output/gate_kpt_auv4_bayese_640x640_nv12.bin` = AUV_4 版，md5 `d38b803e…` | `configs/gate_kpt_auv4_config.yaml` ⚠️ 见下方注 | **回退用**（改名前就是 `gate_kpt_bayese_640x640_nv12.bin`） |
| `_archive/output/backup_auv3/`（原 `output/backup_auv3/`，已在 09-23 归档） | 更早一版（AUV_3）pose 的 `*.bin` + `*_quant_info.json` | **不部署**，仅作回退与量化指标基线对比 |

> ⚠️ **`configs/gate_kpt_auv4_config.yaml` 已不在工作树**（2026-09-24 起「pose 量化只保留一份
> `configs/gate_kpt_config.yaml`」，见 `output/README.md`；`configs/backup/` 里也没有它）。
> 它仍在 git 里（`git status` 记为 `D`），需要复现 auv4 量化时从仓库根取：
> `git -C /home/ansty/RDKX5/auv_vision show HEAD:RDKX5-YOLOv11n-/configs/gate_kpt_auv4_config.yaml`
> （该配置 `onnx_model: /data/weights/yolo11n-pose.auv4.onnx`、前缀 `gate_kpt_auv4_...`，
> 与 `output/gate_kpt_auv4_bayese_640x640_nv12_quant_info.json` 对应）。**这不是路径写错，是文件被有意删除**。

> ⚠️ 已过时（2026-09-26）：`output/backup_auv3/` 的原文写法（「→ 现 `_archive/output/backup_auv3/`」）保留了
> 迁移痕迹，上表已改成**当前真实路径**；该目录下今天有 `gate_kpt_bayese_640x640_nv12.bin` +
> `..._quant_info.json` 两个文件（2026-09-12 产出）。

> ⚠️ 阶段一（`stage1*.pt`）**没有量化产物**：`output/` 顶层今天只有 3 个 `.bin`
> （detect / gate_kpt / gate_kpt_auv4），没有 `*stage1*.bin`。

> `quantize.sh` 用配置里的 `output_model_file_prefix` 命名产物 —— **同名会直接覆盖**。
> 换代时先把当前 bin 按代次改名（如 `mv output/gate_kpt_bayese_640x640_nv12.bin
> output/gate_kpt_<上一代>_bayese_640x640_nv12.bin`），别让它被新一版无声覆盖。

> ⚠️ **本地名 ≠ 板端名**：板端工程里模型路径由板端 `cfg` 指定。**板端当前仍跑 auv5 的 bin**
> （`945fdd01…`，配 D 链序 `common/preprocess.py` + `cfg/vision.yaml clahe_clip: 0`）；
> 本次新模型（`9ac61773…`）**尚未上板**。上板必须"**代码 + 新 bin 一起推**"，只推一样会掉精度。

## 再生产方式

> **两套环境别混（2026-09-26 核实）**：下面所有 PC 侧 Python 命令都在 conda `yolov8` 下跑
> —— `/home/ansty/anaconda3/envs/yolov8/bin/python`（ultralytics 8.3.0 + torch 2.3.1+cu118）；
> 系统 `python3` 没装 torch/ultralytics，直接用会 `ModuleNotFoundError`。**量化不在 conda 里跑**：
> `scripts/3_export/quantize.sh` 自己 `docker run`
> `openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310` 执行 `hb_mapper makertbin`，
> 容器内 `/data` = 仓库根，配置里的 `onnx_model` / `cal_data_dir` / `output_model_file_prefix`
> 都是**容器内路径**，不要改成宿主机路径，也不要手写 docker 命令。

```bash
# 训练（detect 可省 --weights；pose 必须显式给起点，见上文规则）
python scripts/2_train/train_yolo11n.py --data <data.yaml> --epochs 300 --batch 4 --device 0
python scripts/2_train/train_yolo11n.py --task pose --data <data.yaml> \
    --weights weights/yolo11n-pose.coco.pt --epochs 300 --batch 4 --device 0

# 导出 ONNX（写入本目录；导出前必须先打对应任务的输出头补丁）
python scripts/3_export/modify_ultralytics.py --task detect   # 或 --task pose
python scripts/3_export/export_onnx.py               # → weights/yolo11n.onnx
python scripts/3_export/export_onnx.py --task pose   # → weights/yolo11n-pose.onnx
python scripts/3_export/modify_ultralytics.py --restore       # 用完还原（predict/val 要原版）

# 量化（写入 output/；校准集按任务分区：detect 用 calibration_data_detect/，pose 用 calibration_data/）
python scripts/3_export/prepare_calibration.py --coco-path <代表性图片目录> \
    --num-images 300 --output-dir calibration_data          # pose 用（默认就是这个目录）
python scripts/3_export/prepare_calibration.py --coco-path <代表性图片目录> \
    --num-images 300 --output-dir calibration_data_detect    # detect 用
./scripts/3_export/quantize.sh
./scripts/3_export/quantize.sh configs/gate_kpt_config.yaml
```

> ⚠️ **校准集目录今天不在盘上**（2026-09-26 `ls` 无 `calibration_data/` 与 `calibration_data_detect/`）：
> `quantize.sh` 会在跑之前校验配置里的 `cal_data_dir`（`configs/gate_kpt_config.yaml` → `/data/calibration_data`、
> `configs/yolo11n_config.yaml` → `/data/calibration_data_detect`，容器内 `/data` = 仓库根，
> 由 `-v "$(pwd)":/data` 挂载），目录不存在会直接报错退出 —— 所以复现量化前**先跑
> `prepare_calibration.py` 生成对应目录**，别把「校准集缺失」误判成工具链坏了。

## 注意

- **类别顺序必须与板端一致**：`model.labels: [blue_ball, gate, red_ball]`（决定解码类别，
  训练数据集的 `names` 顺序必须相同）。
- **pose 的关键点顺序/数量**：`kpt_shape: [4, 3]`（TL,TR,BR,BL），导出后 kpt 张量通道应为 **12**；
  `export_onnx.py --task pose` 会校验并提示。RoboFlow 的导出**未必**是 `[4,3]`
  （常见 `[5,3]` + 全零第 5 点），训练前先过 `scripts/1_prepare/pose/prepare_pose_dataset.py`。
- **`.onnx` 只是中间产物**：`weights/*.onnx` 会被 `export_onnx.py` 直接覆盖，且板端**不加载**它
  （板端只加载 `output/*.bin`）。换 `--task` 导出前先确认 `head.py` 的补丁状态
  （见上文「③ `head.py` 的补丁生命周期」：导出=补丁版，predict/val=原版 `--restore`）。
- **量化耗时参考（2026-09-24 实测，本机 RTX 4060 + CPU 版工具链镜像）**：300 张校准一轮
  `makertbin`（`calibration_type: default`）**机器空闲时约 3 分钟**；被训练/其它容器抢占时会
  掉到 8–16 s/张（2 小时量级），先查占用再怀疑工具链。更早文档写的「单轮约 55 分钟」是
  被抢占时的旧数。
- **量化校验标准与脚本现状**：判据（输出层余弦 ≥ 0.995 / 逐节点 ≥ 0.999）与做法见
  [根 README「量化校验标准」](../README.md) 与 [`../_archive/quant_ideas_void_20260924/QUANT_TUNING_20260924.md`](../../_archive/quant_ideas_void_20260924/QUANT_TUNING_20260924.md)。
  `scripts/3_export/` 已于 2026-09-24 **回退到 09-15（`54e09b4`）**，工作树只留 4 个脚本
  （`export_onnx.py` / `modify_ultralytics.py` / `prepare_calibration.py` / `quantize.sh`）；
  当天新增的校验脚本已从工作树删除（思路留档），**但仍在 git HEAD 里**。要跑校验先取回 09-23 版
  （在仓库根 `/home/ansty/RDKX5/auv_vision` 的项目子目录下执行，`3a3e044` 已核实存在）：
  `git checkout 3a3e044 -- scripts/3_export/check_quant_cos.py`。
  同一状态还有 `scripts/3_export/check_quant.sh` 与 `scripts/3_export/test_decode_parity.py`
  （工作树没有、`git show HEAD:RDKX5-YOLOv11n-/scripts/3_export/<名>` 可取）。
- 若 `weights/yolo11n-pose.pt` 被 gate 训练覆盖后，需要重新拿官方预训练做新实验时，可重新下载：
  `wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt -O weights/yolo11n-pose.pt`

> ⚠️ 已过时（2026-09-26）：上面这条 `wget ... -O weights/yolo11n-pose.pt` **今天会直接覆盖产线权重**
> （`2f36303a…`，当前 pose 生产模型），且该 URL 拿到的就是官方 COCO pose 权重 —— 而它**已经在盘上**：
> `weights/yolo11n-pose.coco.pt`（md5 `475cd7f6…`）。正确做法是下载到 `...coco.pt` 这个名字，别动
> `yolo11n-pose.pt`：`wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt -O weights/yolo11n-pose.coco.pt`
> （原文照留，仅作历史记录；要覆盖前先按文末「上线三步」备份当前代次。）

## 新模型上线三步（不删任何旧文件）

```bash
mv weights/yolo11n-pose.pt        weights/yolo11n-pose.<上一代>.pt     # 旧的当前 → 代次备份
mv weights/yolo11n-pose.onnx      weights/yolo11n-pose.<上一代>.onnx
cp weights/yolo11n-pose.<新代次>.pt   weights/yolo11n-pose.pt          # 新的 → 当前
cp weights/yolo11n-pose.<新代次>.onnx weights/yolo11n-pose.onnx
```

> ⚠️ 上线的**前提**（2026-09-26 补）：那个 `.onnx` 必须是**补丁版 head** 导出的
> （pose 9 输出 / detect 6 输出，本文档已核对 `weights/yolo11n-pose.onnx` 与
> `weights/yolo11n.onnx` 分别为 9 / 6 输出），且换完权重后要 `modify_ultralytics.py --restore`
> 把 `head.py` 还原成原版，否则后续 `predict` / `val` / 筛图会因输出头分裂而报错。
> 板端上线还要连带推 `.bin` 与配套预处理代码（见上文「本地名 ≠ 板端名」）。
