# `experiment/` — 实验专区

这个目录装的是**为了回答「去畸变该放在哪一步 / 要不要 enhance」这一个问题而临时产生的一切**：
支撑脚本、对照数据集、训练记录、评估结果、板端对照、日志。

**它不在常规流水线上。** 常规流水线（`scripts/1_prepare` → `scripts/2_train` → `scripts/3_export`，
数据落 `data/`，权重落 `weights/`，交付物落 `output/`）只依赖 `data/mapped/raw/` +
`data/mapped/pose_D_wb/` + `configs/`，**不引用本目录任何文件**。所以本目录可以整体删除、
移动或归档，而不影响"从标注图到板端 `.bin`"这条链。反过来说：**不要在这里放流水线需要的东西。**

结论本身不在这里，而在 [`runs/domain/EXPERIMENT_REPORT_20260923.md`](runs/domain/EXPERIMENT_REPORT_20260923.md)
（356 行完整报告，含判定、口径、以及"哪些结论是未验证的"）。本文件只负责**分类与索引**。

---

## 1. 一页速览

| 目录 | 装什么 | 一句话 |
|---|---|---|
| `scripts/` | 实验专属脚本 | 不属常规流水线；`exp_distortion/` 是主目录，`board/` 是板端对照 |
| `data/` | 6 个**对照**数据集（各 1302 张，逐帧同划分） | B/C/D × enhance on/off 的域对照；**不是**生产集 |
| `runs/domain/` | 域对比实验的全部结果 | 报告、评估明细、结论 json、图、探针、日志、权重训练记录 |
| `runs/exp_distortion/` | 更早的"畸变本身"实验 | E1 几何/模型域评估、标定诊断、AUV_5 评估集 |
| `runs/prov_intermediate/` | 溯源中间产物 | 可重建，删除无损 |
| `logs/` | 实验用日志 | 天花板测算等 |

`data/AUV_5/label_candidates_Dwb/`（150 张清水待标候选）不在这里，它在 `data/` 下 —— 因为它是
**下一步要拿去标图的素材**，属于数据准备产物，不是实验结果。同理 `weights/domain_*.pt` 留在
`weights/`：它们是"可被再次评估的权重"，不是脚本或数据。

---

## 2. `scripts/` — 实验专属脚本

### 2.1 `scripts/exp_distortion/`（域实验主目录）

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

### 2.2 `scripts/exp_distortion/board/` — 板端对照

`frames_run1.txt` / `frames_run2_testsplit.txt` 是钉住的取帧清单（保证两次测量可比）。
脚本覆盖：端到端计时、流水线分段计时、LUT 白平衡基准、标定切换校验、LUT 补丁校验、
板端关键点导出。`preprocess_original.py` / `preprocess_lut_wb.deployed.py` 是**当时板端
`common/preprocess.py` 的两份快照**，用于对照。细节见该目录 `README.md`。

### 2.3 `scripts/train_domain_arms.sh`

域对比的批量训练（B/C 各 200 ep / seed 0 / 同起点），产物写 `weights/domain_${ARM}.pt` 与
`runs/domain/A_${ARM}/`。**不覆盖正式权重。** `D` / `D_noenh` 两臂当时手跑。

### 2.4 `scripts/dedup_ceiling.py`

"内容去重"的天花板测算：按连续段贪心（与上一张保留帧的 32×32 灰度平均绝对差 ≥ 阈值才保留），
给出每个素材来源在各阈值下最多能保住多少张。**用于回答"能不能凑出 3000 张不相邻的图"**
（旧做法按帧号间隔，在不同素材上语义不一致，见 `scripts/1_prepare/pose/make_mixed_gate_set.py`
docstring 里的说明）。结果留档在 `logs/dedup_ceiling.log`。

### 2.5 `scripts/run_mix3000.sh`

选图 runner（带 `flock` 锁）：重复拉起是空操作，**不会 `rm -rf` 掉正在写的结果** ——
DSH 的后台命令可能被重复执行，而选图脚本一开头要清输出目录，所以必须加锁。

```bash
setsid nohup bash experiment/scripts/run_mix3000.sh > experiment/logs/mix3000.log 2>&1 &
```

> 这些脚本都靠 `Path(__file__).resolve().parents[3]` 定位项目根，目录层级保持
> `experiment/scripts/exp_distortion/*.py` 时该表达式仍等于项目根（`board/` 子目录里的脚本
> 层级更深，它们用的是绝对板端路径或自己解析，勿照搬）。

---

## 3. `data/` — 6 个对照数据集

`pose_{B,C,D}` 与 `pose_{B,C,D}_noenh`，各 **1302 张**，**逐帧同划分**（train 1043 / valid 130 / test 129）。
来源同一批标注（`data/AUV_4/PNP.kpt4.yolov8` 1302 张），只改**预处理链路**：

| 集合 | 链路 |
|---|---|
| `pose_B` | `resize(640) → enhance`（不去畸变） |
| `pose_C` | `remap@720p → resize(640) → enhance`（先去畸变） |
| `pose_D` | `resize(640) → enhance → remap@640`（**后去畸变**，定稿方向） |
| `*_noenh` | 同域但**不做 enhance**（无 WB/CLAHE） |

`pose_D_wb`（定稿生产集：D 域 + LUT 白平衡 + 无 CLAHE）**不在本目录**，在 `data/mapped/pose_D_wb/`，
因为它是下一步重训要用的正式数据集。

重建方式：`scripts/1_prepare/pose/map_pose_dataset.py --domain <B|C|D> --enhance <on|off>` +
`--out experiment/data/pose_<...>`（输入依赖 `runs/prov/provenance_final_*.csv`）。

---

## 4. `runs/domain/` — 域实验全部结果

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

## 5. `runs/exp_distortion/` — 更早的"畸变本身"实验

`E1_RESULT.md` 结论；`e1/` `e1_sheets/` `e1_picked/` 选帧与印相表；`calib/` 标定诊断；
`eval_auv5_*/` 清水评估；`data_raw/` 中间素材（可重建）。

## 6. `runs/prov_intermediate/`

溯源过程的中间文件（合并前的分片）。最终表在 `runs/prov/provenance_*.csv`（**流水线依赖，不在本目录**）。
删掉无损，重建靠 `scripts/1_prepare/provenance/`。

---

## 7. 与常规流水线的边界（别搞混）

| 东西 | 位置 | 归属 |
|---|---|---|
| `provenance_*.py` / `map_pose_dataset.py` / `make_mixed_gate_set.py` / `prepare_frames.py` | `scripts/1_prepare/` | 流水线（会长期用） |
| `exp_distortion/*` / `train_domain_arms.sh` / `dedup_ceiling.py` | `experiment/scripts/` | 实验 |
| `data/mapped/raw/`、`data/mapped/pose_D_wb/`、`data/AUV_5/selected_3000_Dwb/`、`data/AUV_5/label_candidates_Dwb/` | `data/` | 流水线 |
| `experiment/data/pose_{B,C,D}{,_noenh}` | 本目录 | 实验对照集 |
| `runs/prov/provenance_*.csv`、`runs/auv5_eval/`、`runs/calib_auv5/` | `runs/` | 流水线（溯源表 + 回归评估集 + 标定记录） |
| `runs/domain/`、`runs/exp_distortion/` | 本目录 | 实验 |
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

# 3) 评估与判定（会重写 runs/domain/ 下的 EVAL_* 与 verdict.json）
python experiment/scripts/exp_distortion/run_domain_pipeline.py

# 4) 板端对照：见 experiment/scripts/exp_distortion/board/README.md
```

**注意**：不要用 `experiment/` 里的脚本去生成正式交付物；正式交付链路见
`scripts/3_export/`（导出 → 校准集 → `quantize.sh`）。
