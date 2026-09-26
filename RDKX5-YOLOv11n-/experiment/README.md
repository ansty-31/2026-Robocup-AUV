# `experiment/` — 实验专区

> ⚠️ **路径迁移说明（2026-09 目录归整）**
>
> 本目录在 2026-09 做过两轮归整（逐项记录见 [`../cleanup_record/`](../cleanup_record/)）。
> 下表是**已核准的旧路径 → 现状**映射。**以下各文档正文里的路径可能仍是迁移前的写法，以本表为准**；
> 本文件内部路径一律相对**仓库根**（可直接从仓库根 `ls` 验证）。
>
> | 文档里的旧路径 | 现在的位置（2026-09 复核） |
> |---|---|
> | `runs/domain/...` | `experiment/runs/domain/...` |
> | `runs/exp_distortion/...` | `experiment/runs/exp_distortion/...` |
> | `runs/prov_intermediate/...` | `experiment/runs/prov_intermediate/...` |
> | `runs/auv5_labeled.jsonl` | **已不存在**（标注记录现以 `data/datasets/*/{train,valid,test}/labels/` 与 `data/derived/AUV_5_upload_roboflow/manifest_*.csv` 为准） |
> | `weights/domain_B_noenh.pt` / `weights/domain_C_noenh.pt` | ⚠️ **与核准表不同**：本节逐条 `ls` 复核后确认这两个**从未产出**（B/C 的 noenh 臂当时没训，见 `experiment/runs/domain/reports/DOMAIN_RESULT.md`「产物完整性清单」的 ❌ 行）；现有的 `weights/domain_B.pt` / `weights/domain_C.pt` 是 **enhance 开**的 B/C 臂，**不是**由它们改名而来 |
> | `data/mapped/...`、`data/mapped/pose_D_wb`、`data/mapped/raw/...` | **整棵树已不存在**（早期实验中间产物）；"已预处理池子"现以 `data/derived/` 为准（如 `data/derived/AUV_5_selected_3000_Dwb/`；`data/derived/AUV_4_selected_3000/` **已于 2026-09-26 归档**（错域不可用））。见 `cleanup_record/` |
> | `experiment/data/pose_B` / `pose_C` / `pose_D` / `pose_B_noenh` / `pose_C_noenh` | **已不存在**（`experiment/data/` 现在只剩 `README.md`） |
> | `weights/best.pt` | **现在没有这个文件**；权重按分期命名（如 `weights/yolo11n-pose.stage1.pt`、`weights/yolo11n-pose.coco.pt`），全量见 `ls weights/` |
> | `experiment/runs/domain/tables/auv5_probe_sheet`、`pose_C_noclahe`、`smoke_B` | **已不存在**（2026-09-23 整理时删除，见 `experiment/runs/domain/INDEX.md` §6；本轮复核无同名归档副本） |
>
> 本轮 `ls` 补充核实（同口径，一并据此修正文档）：
>
> | 旧路径 | 现在的位置 |
> |---|---|
> | `data/AUV_x/*_frames` | `data/frames/AUV_1_rec_front_frames` … `data/frames/AUV_4_auv_4_frames`（见 `cleanup_record/MOVES_data_20260925.tsv`） |
> | `data/AUV_1_board` | `data/calib/AUV_1_board` |
> | `data/exp_distortion/...` | **已不存在**（早期畸变实验的中间产物，见 `cleanup_record/CLEANUP_2026-09-24.md`） |
> | `runs/auv5/eval/pose_B` / `pose_C` / `pose_D` / `pose_AOLD` | **已不存在**（2026-09-24 删除，见 `cleanup_record/CLEANUP_2026-09-24.md`；同一位置现存 `runs/auv5/eval/pose_D_wb`） |
> | `experiment/runs/domain/label_audit_B.csv`、`experiment/runs/domain/label_audit_sheet.jpg` | `experiment/runs/domain/audit/label_audit_B.csv`、`experiment/runs/domain/audit/label_audit_sheet.jpg` |

这个目录装的是**为了回答「去畸变该放在哪一步 / 要不要 enhance」这一个问题而临时产生的一切**：
支撑脚本、对照数据集、训练记录、评估结果、板端对照、日志。

**它不在常规流水线上。** 常规流水线（`scripts/1_prepare` → `scripts/2_train` → `scripts/3_export`，
数据落 `data/`，权重落 `weights/`，交付物落 `output/`）只依赖
**`data/datasets/AUV_5_gate-pose.yolov8/`（已标导出）+ `configs/`**（或按 §8 重建出的 `data/mapped/*` —— ⚠️ 该路径已不存在，见上方迁移表），
**不引用本目录任何文件**。所以本目录可以整体删除、移动或归档，而不影响"从标注图到板端 `.bin`"这条链。
反过来说：**不要在这里放流水线需要的东西。**

> **2026-09-24 更新**：`data/mapped/`（含 `pose_D_wb`）与 `experiment/data/pose_*` 对照集**已按决定清理**，
> 去畸变位置已收口为 **D 域 + LUT 白平衡（无 CLAHE）**。数据集按 §8 可重建；
> 清理与重建清单见 [`../cleanup_record/CLEANUP_2026-09-24.md`](../cleanup_record/CLEANUP_2026-09-24.md)。
> **结论、报告与权重都没有删**（`weights/domain_*.pt` 保留）。

结论本身不在这里，而在 [`experiment/runs/domain/reports/EXPERIMENT_REPORT_20260923.md`](runs/domain/EXPERIMENT_REPORT_20260923.md)
（356 行完整报告，含判定、口径、以及"哪些结论是未验证的"）。本文件只负责**分类与索引**。

---

## 1. 一页速览

| 目录 | 装什么 | 一句话 |
|---|---|---|
| `experiment/scripts/` | 实验专属脚本 | 不属常规流水线；`exp_distortion/` 是主目录，`board/` 是板端对照 |
| `experiment/data/` | 6 个**对照**数据集（各 1302 张，逐帧同划分） | B/C/D × enhance on/off 的域对照；**不是**生产集（**已于 2026-09-24 删除**，见 §3） |
| `experiment/runs/domain/` | 域对比实验的全部结果 | 报告、评估明细、结论 json、图、探针、日志、权重训练记录 |
| `experiment/runs/exp_distortion/` | 更早的"畸变本身"实验 | E1 几何/模型域评估、标定诊断、AUV_5 评估集 |
| `experiment/runs/prov_intermediate/` | 溯源中间产物 | 可重建，删除无损 |
| `experiment/logs/` | 实验用日志 | 天花板测算等 |

`data/derived/AUV_5_label_candidates_Dwb/`（150 张清水待标候选）不在这里，它在 `data/` 下 —— 因为它是
**下一步要拿去标图的素材**，属于数据准备产物，不是实验结果。同理 `weights/domain_*.pt` 留在
`weights/`：它们是"可被再次评估的权重"，不是脚本或数据。

---

## 2. `scripts/` — 实验专属脚本

### 2.1 `experiment/scripts/exp_distortion/`（域实验主目录）

| 脚本 | 用途 |
|---|---|
| `run_domain_pipeline.py` | **实验总驱动**：串起"生成域数据集 → 训练 → 评估 → 判定 → 出报告"，判定阈值硬编码在文件顶部（`TH_CERR=1.0` 等） |
| `eval_domain_arms.py` | **跨域臂评估**（核心）：`--arm tag:ds:weights:cam[:split]`，可 `--pair` 做配对 Wilcoxon，按水质/外圈关键点分层，输出 `eval_detail_*.json` |
| `make_variant_dataset.py` | 造对照数据集变体（`--variant noclahe\|noenh\|raw640`） |
| `check_ball_impact.py` | **球检测是否受影响**（用户要求项）：比对两套预处理下 blue/red ball 的框与置信度 |
| `pick_auv5_label_candidates.py` | 选清水待标候选并渲染印相表（定稿 D+wb 渲染） |
| `prepare_auv5_eval.py` | 造清水 16 帧评估集 |
| `probe_auv5_detect.py` | 120 帧探针（清水实测） |
| `audit_labels.py` | 标注体检（4 角点几何一致性） |
| `calib_diagnose.py` | 标定体检（C vs A 复投影误差） |
| `compare_kpt_spacing.py` / `distortion_geometry.py` | 相邻关键点间距 / 畸变几何分析 |
| `eval_e1_geometry.py` / `eval_e1_model_domain.py` / `pick_e1_frames.py` / `prepare_distortion_groups.py` / `transform_pose_labels.py` / `make_contact_sheet.py` / `bench_undistort_placement.py` | 更早的 E1 组工具（保留以便复算） |
| `PROTOCOL_undistort_scheme.md` | 实验协议（先写判据再跑，避免事后挑口径） |
| `README.md` | 目录索引（比本文件更细） |

### 2.2 `experiment/scripts/exp_distortion/board/` — 板端对照

`frames_run1.txt` / `frames_run2_testsplit.txt` 是钉住的取帧清单（保证两次测量可比）。
脚本覆盖：端到端计时、流水线分段计时、LUT 白平衡基准、标定切换校验、LUT 补丁校验、
板端关键点导出。`preprocess_original.py` / `preprocess_lut_wb.deployed.py` 是**当时板端
`common/preprocess.py` 的两份快照**，用于对照。细节见该目录 `README.md`。

### 2.3 `experiment/scripts/train_domain_arms.sh`

域对比的批量训练（B/C 各 200 ep / seed 0 / 同起点），产物写 `weights/domain_${ARM}.pt` 与
`experiment/runs/domain/A_${ARM}/`。**不覆盖正式权重。** `D` / `D_noenh` 两臂当时手跑。

### 2.4 `experiment/scripts/dedup_ceiling.py`

"内容去重"的天花板测算：按连续段贪心（与上一张保留帧的 32×32 灰度平均绝对差 ≥ 阈值才保留），
给出每个素材来源在各阈值下最多能保住多少张。**用于回答"能不能凑出 3000 张不相邻的图"**
（旧做法按帧号间隔，在不同素材上语义不一致，见 `scripts/1_prepare/pose/make_mixed_gate_set.py`
docstring 里的说明）。结果留档在 `logs/dedup_ceiling.log`。

### 2.5 `experiment/scripts/run_mix3000.sh`

选图 runner（带 `flock` 锁）：重复拉起是空操作，**不会 `rm -rf` 掉正在写的结果** ——
DSH 的后台命令可能被重复执行，而选图脚本一开头要清输出目录，所以必须加锁。

```bash
setsid nohup bash experiment/scripts/run_mix3000.sh > experiment/logs/mix3000.log 2>&1 &
```

> 这些脚本都靠 `Path(__file__).resolve().parents[3]` 定位项目根，目录层级保持
> `experiment/scripts/exp_distortion/*.py` 时该表达式仍等于项目根（`board/` 子目录里的脚本
> 层级更深，它们用的是绝对板端路径或自己解析，勿照搬）。

---

## 3. `experiment/data/` — 6 个对照数据集（**已于 2026-09-24 删除**，见 `experiment/data/README.md`）

> 域实验收口后，这 6 个可再生数据集（909 MB）已删；重建命令在
> [`experiment/data/README.md`](data/README.md) 与仓库根 [`cleanup_record/CLEANUP_2026-09-24.md`](../cleanup_record/CLEANUP_2026-09-24.md)。
> 以下描述保留作索引。


`pose_{B,C,D}` 与 `pose_{B,C,D}_noenh`，各 **1302 张**，**逐帧同划分**（train 1043 / valid 130 / test 129）。
来源同一批标注（`data/datasets/AUV_4_PNP.kpt4.yolov8` 1302 张），只改**预处理链路**：

| 集合 | 链路 |
|---|---|
| `pose_B` | `resize(640) → enhance`（不去畸变） |
| `pose_C` | `remap@720p → resize(640) → enhance`（先去畸变） |
| `pose_D` | `resize(640) → enhance → remap@640`（**后去畸变**，定稿方向） |
| `*_noenh` | 同域但**不做 enhance**（无 WB/CLAHE） |

`pose_D_wb`（定稿生产集：D 域 + LUT 白平衡 + 无 CLAHE，1302 张）**不在本目录**；
它曾落在 `data/mapped/pose_D_wb/`，**已于 2026-09-24 清理**（结论已收口，且下一步改用
`data/datasets/AUV_5_gate-pose.yolov8/` 重训）。需要时按下面的命令重建。

重建方式：`scripts/1_prepare/pose/map_pose_dataset.py --domain <B|C|D> --enhance <on|off>`
+ `--out <目标目录>`（输入依赖 `runs/prov/provenance_final_*.csv`）；
`--enhance wb` 即定稿链路（无 CLAHE）。

---

## 4. `experiment/runs/domain/` — 域实验全部结果

| 文件/目录 | 内容 |
|---|---|
| `EXPERIMENT_REPORT_20260923.md` | **主报告（356 行）**：0 速览 → 8 待决策路线，所有数字的唯一权威出处 |
| `EXPERIMENT_DESIGN.md` | 18 节实验设计，含 §13 预注册判定标准、§16 清水判据、§17 enhance、§18 板端测量 |
| `INDEX.md` | **结果分类索引**：每个文件是什么、对应报告哪一节、怎么重建、已删了什么 |
| `verdict.json` | 机器可读判定（winner / 各阈值 / sanity flags） |
| `DOMAIN_RESULT.md` | 自动生成的报告 + 顶部人工修正块 |
| `EVAL_*.md` + `eval_detail_*.json` | 各次评估（主测试/valid、C-D、enhance 三组、清水、基线、仪器一致性）——**md 给人看，json 给复算** |
| `A_{B,C,D,D_noenh}/` | 四臂训练的 ultralytics 记录（权重在 `weights/domain_*.pt`，此处不重复存） |
| `audit/` `probe/` `figs/` | 标注体检、120 帧探针、报告插图 |
| `board/` | 板端测量原始输出（计时/精度/关键点） |
| `logs/` `pipeline_code.md5` | 运行日志与"当时用的代码指纹"（保证结论可追溯） |
| `artifacts_before.txt` | **原始产物 md5 基线**：证明全程没动过既有交付物 |
| `NEXT_PLAN_D_wb.md` | 定稿方案（D + wb）的执行清单与同步顺序警告 |
| `README.md` | 本目录索引 |

## 5. `experiment/runs/exp_distortion/` — 更早的"畸变本身"实验

`E1_RESULT.md` 结论；`e1/` `e1_sheets/` `e1_picked/` 选帧与印相表；`calib/` 标定诊断；
`eval_auv5_*/` 清水评估；`data_raw/` 中间素材（可重建）。

## 6. `experiment/runs/prov_intermediate/`

溯源过程的中间文件（合并前的分片）。最终表在 `runs/prov/provenance_*.csv`（**流水线依赖，不在本目录**）。
删掉无损，重建靠 `scripts/1_prepare/provenance/`。

---

## 7. 与常规流水线的边界（别搞混）

| 东西 | 位置 | 归属 |
|---|---|---|
| `provenance_*.py` / `map_pose_dataset.py` / `make_mixed_gate_set.py` / `prepare_frames.py` | `scripts/1_prepare/` | 流水线（会长期用） |
| `exp_distortion/*` / `train_domain_arms.sh` / `dedup_ceiling.py` | `experiment/scripts/` | 实验 |
| `data/derived/AUV_5_selected_3000_Dwb/`、`data/datasets/AUV_5_gate-pose.yolov8/`、`data/derived/AUV_5_label_candidates_Dwb/` | `data/` | 流水线 |
| `data/mapped/*`（含 `pose_D_wb`） | ~~已清理~~ | 流水线（**按 §8 可重建**；本次清理见 `../cleanup_record/CLEANUP_2026-09-24.md`） |
| `experiment/data/pose_{B,C,D}{,_noenh}` | ~~已清理~~ | 实验对照集（**按 §8 可重建**） |
| `runs/prov/provenance_*.csv`、`runs/auv5/eval/`、`runs/auv5/calib/` | `runs/` | 流水线（溯源表 + 回归评估集 + 标定记录） |
| `experiment/runs/domain/`、`experiment/runs/exp_distortion/` | 本目录 | 实验 |
| `weights/*.pt`（含 `domain_*.pt`） | `weights/` | 交付/可复评权重，**一律不删** |
| `output/*.bin` | `output/` | 交付物 |

## 8. 复现顺序（真要从头再跑一遍）

```bash
# 0) 环境：所有 PC 脚本都用 conda yolov8
/home/ansty/anaconda3/envs/yolov8/bin/python -V

# 1) 对照数据集（6 个，逐帧同划分）
for D in B C D; do
  for E in on off; do
    python scripts/1_prepare/pose/map_pose_dataset.py --domain $D --enhance $E \
      --out experiment/data/pose_${D}$([ $E = off ] && echo _noenh)
  done
done

# 2) 四臂训练（后台，B/C 用脚本，D/D_noenh 见 README 里的手跑命令）
nohup bash experiment/scripts/train_domain_arms.sh > experiment/logs/train_arms.log 2>&1 &

# 3) 评估与判定（会重写 experiment/runs/domain/ 下的 EVAL_* 与 verdict.json）
python experiment/scripts/exp_distortion/run_domain_pipeline.py

# 4) 板端对照：见 experiment/scripts/exp_distortion/board/README.md
```

**注意**：不要用 `experiment/` 里的脚本去生成正式交付物；正式交付链路见
`scripts/3_export/`（导出 → 校准集 → `quantize.sh`）。
