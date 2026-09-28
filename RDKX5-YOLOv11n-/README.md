# RDK X5 · YOLO11 训练与部署工程

地平线 RDK X5 上的 YOLO11n 端到端工程：**数据准备 → 训练 → 导出 ONNX → PTQ 量化 → 板端部署**。
覆盖两条任务线：

| 任务 | 模型 | 输出头 | 用途 |
|---|---|---|---|
| **detect** | YOLO11n | 6 张量（3×bbox + 3×cls） | 红球 / 蓝球 / 门 检测（单权重多类别） |
| **pose** | YOLO11n-pose | 9 张量（每尺度 bbox+cls+kpt） | 门 4 角点（TL,TR,BR,BL）→ 板端 PnP 测距 |

> 环境：conda `yolov8`（`ultralytics==8.3.0`）+ OE v1.2.8 Docker 镜像（`bayes-e`）。

## ★ 运行环境（每次开始前先做这一步，勿再询问）

**所有 Python 脚本（`scripts/1_prepare` `2_train` `3_export`）都必须跑在 conda `yolov8` 环境里**，
系统自带的 `python3` 没有 torch / ultralytics，直接用会报 `ModuleNotFoundError`。

```bash
conda activate yolov8
```

- 交互式终端：先 `conda activate yolov8` 再执行脚本；
- 非交互式（脚本 / Jenkins / 自动化）：用绝对路径解释器，效果等价：
  `/home/ansty/anaconda3/envs/yolov8/bin/python scripts/2_train/train_yolo11n.py ...`
- 该环境已确认可用 **GPU**（`torch 2.3.1+cu118`，`device 0` = NVIDIA RTX 4060 Laptop 8GB），
  训练默认 `--device 0`；`nvidia-smi` 可用即说明 GPU 正常：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> 若某次调用发现 `torch.cuda.is_available() == False`，先确认 `/dev/nvidia*` 设备节点是否存在
> （驱动重载/休眠唤醒后偶发缺失），再决定是否降级到 CPU（`--device cpu`，约 112 秒/epoch，慢 20 倍以上）。

---

## 目录分区

> 整理时间 **2026-09-24**（上一轮 09-23：实验内容全部移入 `experiment/`）；**2026-09-26 已按磁盘现状逐条复核对齐**
> （下面目录树 / 文件清单 / 数量均由 `ls`·`find`·`du` 实测）。
> 顶层**十个区**（`cleanup_record` `configs` `data` `docs` `experiment` `output` `raw-data` `runs`
> `scripts` `weights`）加一个**归档区** `_archive/`。
> ⚠️ 已过时（2026-09-26）：原文记的是「顶层**九个区**（`configs` `weights` `scripts` `docs` `data` `raw-data`
> `runs` `output` `experiment`）加一个**归档区** `_archive/`」。现状多出 `cleanup_record/`
> （**整理记录区**：`CLEANUP_*` / `MOVES_*` / `RESTRUCTURE_*`），故为十个区。
> ⚠️ 另注（2026-09-26）：**“归档”现在有两个位置** —— 顶层 `_archive/`（历史留档）与
> `output/preview/auv5_pose/judgment/archive/`（**09-26 新建**，只装作废的判定数据）。两者分工与“以谁为准”
> 见 [`_archive/README.md`](_archive/README.md) §8。
> **`experiment/` = 只为「去畸变时机 / enhance」那一个问题服务的一切**（脚本+对照数据+结果），
> 常规流水线不引用它 —— 分类索引见 [`experiment/README.md`](experiment/README.md)。
> 归档规则与恢复方法见 [`_archive/README.md`](_archive/README.md)；09-24 清理（去重 6.4 G、根目录归位、
> 清单对齐）见 [`CLEANUP_2026-09-24.md`](cleanup_record/CLEANUP_2026-09-24.md)；
> 09-25 重组见 [`RESTRUCTURE_2026-09-25.md`](cleanup_record/RESTRUCTURE_2026-09-25.md)；
> **09-26 变更（板端视角口径修订 / T5 从 111 缩到 30 / 判定归档）见
> [`MOVES_2026-09-26.md`](cleanup_record/MOVES_2026-09-26.md)**。

```
RDKX5-YOLOv11n-/
├── README.md                  ← 本文件（项目总览 / 分区导航 / 全流程命令）
├── LICENSE · requirements.txt · .gitignore
├── cleanup_record/            整理记录（09-24 / 09-25 清理与重组 · 09-26 板端口径修订；MOVES_*.tsv 可逐项回退）
├── configs/                   配置（全部集中在此，索引见 configs/README.md）
│   ├── vision.yaml            ★ 板端图像链路参数镜像（预处理与推理一致性唯一来源）
│   ├── front_camera.yaml      ★ 当前生效标定（AUV_5 水下棋盘，即"标定 C"）
│   ├── front_camera_air.yaml  空气/岸上标定（台架调试用，下水前必须切回上一份）
│   ├── yolo11n_config.yaml    detect 模型 PTQ 量化配置
│   ├── gate_kpt_config.yaml   pose 模型 PTQ 量化配置
│   └── backup/                历史标定留档（只读，勿删；见 configs/README.md）
├── weights/                   权重与模型（★ 分区说明见 weights/README.md）
├── scripts/                   脚本按阶段分三区，**只放常规流水线**（★ 索引见 scripts/README.md）
│   ├── 1_prepare/             训练前：数据准备（分类详见 scripts/README.md）
│   │   ├── (根) extract_frames · calibrate_camera · prepare_frames
│   │   │        · select_frames · resplit_dataset · label_pose_local   ← 共用 / 常用
│   │   ├── pose/               ★ 门（4 角点）专用
│   │   │   ├── prepare_pose_dataset.py  Roboflow pose 导出 → 可训练集（kpt_shape/flip_idx）
│   │   │   ├── check_pose_export.py     回收导出：把 v=1 的角点改成 v=0
│   │   │   ├── map_pose_dataset.py      跨去畸变域换算标注 + 生成 pose 训练集
│   │   │   │                            （`Domain` 类也是各域渲染的公共库）
│   │   │   └── make_mixed_gate_set.py   跨水质挑门图待标（新一版 selected_3000，内容去重）
│   │   └── provenance/         素材溯源（已闭合，换数据集可复用）
│   │       └── materialize_raw · provenance_{match,stream,nn,merge}.py
│   ├── 2_train/               train_yolo11n.py（--task detect|pose）
│   │                          · build_stage_dataset.py  分期数据集生成（累积并入 + tag 过采样 + 排除评测保留集）
│   │                          · eval_benchmark.py       A–D 节固定基准评测
│   │                            （D 节支持 --boardview-log / --boardview-list / --boardview-subset，严格+宽松双口径）
│   └── 3_export/              modify_ultralytics · export_onnx · prepare_calibration · quantize.sh
│                              （2026-09-24 已按 git 回退到 09-15 的这 4 个脚本，
│                                校验脚本与调优记录见 _archive/quant/ideas_void_20260924/QUANT_TUNING_20260924.md）
├── experiment/                ★ 实验专区（脚本+对照数据+结果，不在常规流水线上）
│   ├── README.md              分类索引：里面每个目录装什么、怎么重建
│   ├── scripts/               exp_distortion/（域对比主目录）· exp_aug/（增强消融）
│   │                          · train_domain_arms.sh · run_mix3000.sh · dedup_ceiling.py
│   ├── data/                  ~~6 个对照数据集~~ **已于 2026-09-24 清理**（只留 README 说明怎么重建）
│   ├── runs/                  domain/（域实验全部结果，主报告 EXPERIMENT_REPORT_20260923.md）
│   │                          · exp_distortion/（更早的畸变实验）· prov_intermediate/ · aug_ablation/
│   └── logs/                  实验日志
├── docs/                      文档（教程 / 介绍 / 贡献指南）
│   ├── tutorial_zh.md         完整部署教程（含踩坑记录）
│   ├── gate_pose_decode_spec.md ★ **上位机解码约束**（9 输出布局 / DFL+kpt 解码公式 / 阈值常量 /
│   │                          选门与去重规则 / 已知坑 / 自检清单）—— 照它改板端后处理（2026-09-26）
│   ├── labeling_gate_pose.md  ★ 门 4 角点**打标规范/作业单**（发组内；2026-09-25）
│   ├── README_ZH.md           中文项目介绍
│   ├── README_EN.md           English overview
│   └── CONTRIBUTING.md
├── data/                      ★ 素材与数据集（不进库）—— **顶层按种类分 5 区**
│   │                          详细索引见 [`data/README.md`](data/README.md)
│   ├── raw/          ① 原始素材（整段视频/裸流，**不可再生**）9.8 G
│   ├── frames/       ② 切帧产物（AUV_5_gate_calib = 原 AUV_5/raw-data）3.8 G
│   ├── datasets/     ③ 标注数据集（8 个 / 703 M）—— ★ 基准集 `AUV_5_gate-pose.yolov8/`（844 张 / 96 M）
│   │                     · 阶段集 `AUV_5_stage1/`（2585 张 / 164 M）· `gate-w_clear/` · `gate-w_clear_v0fix/`（各 577 张）
│   │                     · 历史集 `AUV_1_AUV.yolov11/` · `AUV_2_AUV.yolov11/` · `AUV_3_PNP.v1i.yolov8/` · `AUV_4_PNP.kpt4.yolov8/`
│   ├── derived/      ④ 派生·筛选·待标（**793 M**，09-26 归档后）—— ★ `AUV_5_upload_roboflow/`：待标 `stage2_mid` 800 / `stage2_turbid` 429（`stage1_needlabel` 577 **已完成并归档 09-26**；
│   │                     · `stage2_mid` 800 · `stage2_turbid` 429 ＋ **09-26 纠正批 `stage1_fix_wy/` 30 张**
│   │                     `stage1_fix111/` 111 张与 `stage1_needlabel/` 577 张**已于 09-26 归档**：前者范围被取代、后者已消化进 `gate-w_clear` 导出）
│   │                     · `AUV_5_selected_3000_Dwb/` · `AUV_5_label_candidates_Dwb/` · `AUV_{2,3,4}_selected_*`
│   └── calib/        ⑤ 标定输入与 QC（`AUV_1_board/` 2298 张）439 M
│   （每区内条目一律 `<下水轮次>_<原名>`，如 `AUV_4_auv_4_frames/`）
├── raw-data/                  ⚠️ **当前是空目录**（原"素材抢救区"：坏卡雕出的裸流，GB 级、不进库）
│                              内容已于 2026-09-24 01:05 被**外部**清空（不是整理所为，见下）
├── runs/                      训练与实验输出（不进库）—— **按轮次收敛**
│   ├── prov/                  溯源表 provenance_*.csv（**流水线依赖，路径固定**）
│   └── auv5/                  ★ auv5 代次一个入口
│       ├── train/             `pose_fromcoco_844/`（09-24 重训）· `stage1/` · `stage1_full/`（★ 09-26 阶段一）
│       │                      ＋ `logs/`（各次训练 stdout，共 80 M）
│       ├── eval/              评估汇总 ＋ `clearwater16/`（清水16帧）· `pose_D_wb/`
│       ├── quant/             量化调优，**按用途分 4 组**：`cos/`（逐图余弦 json）· `kpt/`（解码级 json）
│       │                      · `logs/`（quantize / verify / checker 表）· `reports/`（QUANT_TUNING.md + QUANT_CHECK_*）
│       ├── board/             板端实测（bench / live 450 帧 / pnp 反演）
│       ├── calib/             标定 C 的记录（=`prep_calib.log` + `diag_*.json` + `front_camera_good.yaml`）
│       └── REPORT_auv5_pose.md · RETIRED_auv5.md
├── output/                    交付产物 + 分析工作区（不进库，部署时拷到板端 models/）
│   │                          索引见 [`output/README.md`](output/README.md)
│   ├── *.bin + *_quant_info.json   ★ 交付物；pose 当前采用 `gate_kpt_bayese_640x640_nv12.bin`（`9ac61773…`）
│   └── preview/               ★ 判定/调参工作区（不是流水线）
│       ├── auv5_pose/         「线 C」主工作区（**359 M**，09-26 归档旧渲染后）——**入口 [`INDEX.md`](output/preview/auv5_pose/INDEX.md)**
│       │   ├── INDEX.md · docs/   工作区分类索引 · 7 个 .md（RUNBOOK / CRITERIA / DECISIONS / TRAINING_STRATEGY …）
│       │   ├── renders/       10 个渲染目录 ＋ `_sheets/`（49 张拼版）—— annotated · boardview（旧单实例画法）
│       │   │                  · **boardview_base（282，线上权重，★ 你已决定不判）** · **boardview_stage1（339，★ 全实例画法）**
│       │   │                  · false_accept · grids · label_check · missed_gates · negatives · rejudge
│       │   ├── reports/       ★ 生成的报表 .md（09-26 从 logs/ 与 tables/ 抽出）：benchmark_stage1_boardjudged
│       │   │                  · model_vs_labels_* · stage_compare* · plan_inventory
│       │   ├── tables/        **按用途分 4 组**：`benchmarks/`（评测 json）· `boardview/`（清单/长表）
│       │   │                  · `label_analysis/`（model_vs_labels/label_check/false_accept/missed_gates）
│       │   │                  · `pool/`（pool_inventory · eval_reserved · per_image · instances · negatives_）
│       │   ├── logs/          运行日志：run.log · summary.txt · chain_*.log
│       │   ├── filelists/     **分 2 组**：`sets/`（各判定集输入清单）· `analysis/`（各次分析清单）
│       │   ├── tools/         16 个 .py：判定/渲染/分析脚本
│       │   │                  （★ `make_boardview_set.py` 09-26 重写：全实例画法 + `--extra-names-from`
│       │   │                   + 实例级长表，并修掉 46 px 错位）
│       │   ├── judgment/      5 个 .csv —— **人眼判定日志 = 结论的唯一权威来源**
│       │   │                  ＋ ★ `boardview_stage1_log.csv`（339 行，2026-09-26 你判：k 309 / y 26 / w 4）
│       │   │   ├── archive/   ★ 作废的判定数据留档（102 行 csv ＋ `buckets_singleinst_render_round1/` ＋ README）
│       │   │   │              —— 为什么作废、以谁为准见 [`_archive/README.md`](_archive/README.md) §8
│       │   │   └── buckets/   ★ **2026-09-26 归位**：判定分桶工作副本（ok 309 / one_ok_one_bad 26 / wrong_kpt 4 / …）
│       │   │                  —— 由 `triage_review.py --buckets "k=judgment/buckets/ok,…"` 写入；
│       │   │                     **结论只看 `judgment/*_log.csv`，不看目录名**
│       └── pool_other_water/  其余水质池（0.3 M，其 148 M 渲染图已于 09-26 归档）
│                              **同形归类**：`tables/` · `logs/` · `filelists/`
├── calibration_data_gate240/  ⚠️ **磁盘上已不存在**（2026-09-26 核实；详见目录树后的 ⚠️ 说明）
└── _archive/                  归档区：暂时不用但保留（不进库）—— **2026-09-28 按「项目/代次」重排为 5 组**
    │                          总索引见 [_archive/README.md](_archive/README.md)（逐项「是什么 / 为什么在这 / 怎么恢复」）
    ├── README.md              总索引（唯一顶层文件）
    ├── _records/              所有记录与旧文档统一收集：`MOVES.tsv` · `DELETED_2026-09-2{4,8}*.tsv` · `old_readme/`
    ├── legacy_2026-09-23_sweep/  第一次清扫（AUV_1..4 时代，原按位置分）：
    │                          `detect_runs_v3_v4/` · `pose_runs_auv3_auv4_smoke/` · `extract_logs/`
    │                          · `datasets_auv_history/` · `output_backup_auv3/` · `orphan_pyc/`
    ├── quant/                 所有量化相关收一处：`hb_scratch_20260923/`（含 4 个 `*_model.onnx` 唯一副本）
    │                          · `ideas_void_20260924/`（三轮调优，用户裁定**禁止再试**）· `rounds_stage1_20260926/`（R1–R3 配置）
    ├── auv5/                  AUV_5 线：`datasets_AUV_5_gate-pose_backup_20260924/`（★ 另一次划分状态，真实回退点）
    │                          · `manifest_before_realign_20260924/` · `unneeded_20260925/` · `unneeded_20260926/`
    │                          · `judgment_dupes_20260926/`（判定分桶副本，内容与 master 逐字节相同）
    └── auv6/                  AUV_6 线：`planBC_void_20260928/`（Plan B/C 作废）· `superseded_by_v3_20260928/`（被 v3 取代）
                               · `ab_v2_prebasenamefix_20260928/`（⚠️ 去留**待定**，见该组 README）
```

> ⚠️ **已过时（2026-09-26）：仓库根的 `calibration_data*` 一个都不在了。** 目录树里那一行原文是
> 「`calibration_data_gate240/` ★ 量化校验集（240 张真有门的帧，**保留**；`calibration_data/` 已删、重跑即重建）」，
> 但实测 `ls -d calibration_data*` → **无匹配**。三个目录都在**回收站**里
> （`/home/ansty/.local/share/Trash/files/`，逐项 `.trashinfo` 记了删除时间）：
>
> | 目录 | 内容 | 大小 | 删除时间（`.trashinfo`） |
> |---|---|---|---|
> | `calibration_data_gate240/` | 240 个 `.rgb`（真有门的帧） | 1.1 G | **2026-09-24T21:24:57** |
> | `calibration_data/` | 500 个 `.rgb` | 938 M | 2026-09-19T00:01:36 |
> | `calibration_data.zip` | 打包副本 | 206 M | 2026-09-19T00:01:35 |
>
> ⇒ 本文档下面凡是引用 `/data/calibration_data_gate240` 的命令（§量化校验标准）**现在跑不通**，
> **2026-09-26 用户决定：放弃恢复回收站内容**（`calibration_data/` 938 M · `calibration_data_gate240/` 1.1 G · `calibration_data.zip` 206 M · 两个 AUV_1 zip）→ 需要校验集时按 `prepare_calibration.py` **重建**，不要再去回收站捞。记录见 `cleanup_record/MOVES_2026-09-26.md` §10。
> `calibration_data/` `calibration_data_detect/` `calibration_data_*/` 都在 `.gitignore` 里（`data/` `output/` 同样不进库），
> **git 追不回来**。⚠️ 这与 `cleanup_record/CLEANUP_2026-09-24.md` 里「校验集 `calibration_data_gate240/` **保留不动**（判定基准）」
> **互相矛盾** —— 是恢复还是就此放弃，**待你定**（本轮只标注，未替你决定）。

### 各区状态（一眼看清哪里在动、哪里是留档）

| 区 | 状态 | 说明 |
|---|---|---|
| `scripts/` `configs/` `weights/` `docs/` | **在用** | 随版本迭代，纳入 git |
| `cleanup_record/` | **在用** | 整理记录（`CLEANUP_2026-09-24/25` · `RESTRUCTURE_2026-09-25` · **`MOVES_2026-09-26.md`**）；逐项可回退 |
| `data/`（`raw/ frames/ datasets/ derived/ calib/`） | **在用** | 2026-09-25 起按种类分五区；旧的 `data/AUV_1..5/` 那层已**空壳化并归档**（该空壳归档已于 **2026-09-28 删除**，0 文件）—— 本行原写作 `data/AUV_1..5`，路径已不存在，按现状改写 |
| `raw-data/`（仓库根） | ~~已空~~ | 原抢救区；**2026-09-24 01:05 被外部清空**（非本次整理），现为 0 文件空目录 |
| `experiment/` | **留档** | 去畸变/增强实验的脚本、结果与结论；可整体删除而不影响流水线 |
| `runs/prov` `runs/auv5/eval` `runs/auv5/calib` | **在跑** | 流水线依赖的溯源表、回归评估集、标定记录 |
| `runs/auv5` | **留档** | auv5 的**证据链**（报告/评估/量化校验/板端实测）；权重与 bin 已退役删除（`train/stage1*` 是 09-26 新增） |
| `output/gate_kpt_bayese_640x640_nv12.bin` | **在用** | pose 当前采用（09-24 重训版，`9ac61773…`）；板端**尚未**换到这版 |
| `output/gate_kpt_auv4_*.bin` | **在用** | pose 回退用（上一代） |
| `calibration_data_gate240/` | ⚠️ **已不在磁盘** | 原记「**留档·勿删**：量化校验集（240 张真有门的帧），是余弦判定的基准」——2026-09-26 实测已进回收站（1.1 G，2026-09-24T21:24:57），**见上方目录树后的 ⚠️ 说明** |
| `calibration_data/` | ⚠️ **已不在磁盘** | 原记「~~已删~~：纯派生校准集（2.3 G），量化前重建即可」；现同样在回收站（938 M，实测 500 个 `.rgb`） |
| `output/yolo11n_detect_*.bin` | **在用** | 当前球/门检测产物（09-12） |
| `output/preview/auv5_pose/` | **在跑** | ★「线 C」人眼判定 / 分期重训工作区（**359 M**，09-26 归档后；此前 537 M）实测）；**入口是它自己的 `INDEX.md`** |
| `output/preview/auv5_pose/judgment/archive/` | **留档·勿引用** | ★ **09-26 新增的第二个归档位置**：作废的判定日志（102 行）+ 分桶图副本（102 张）+ README；**结论不看这里**（以 `judgment/boardview_stage1_log.csv` 339 行为准）—— 见 `_archive/README.md` §8 |
| `output/preview/pool_other_water/` | **部分归档** | 只留统计小文件（0.3 M，被脚本读取）；`annotated/`+`grids/` 148 M 已入 `_archive/auv5/unneeded_20260926/` |
| `data/derived/AUV_5_upload_roboflow/` | **在跑** | 待送 Roboflow 打标两批（`stage2_mid` 800 / `stage2_turbid` 429）；`stage1_needlabel` 577 **已完成并归档**（→ `datasets/gate-w_clear/`）＋ **09-26 纠正批 `stage1_fix_wy` 30**（`stage1_fix111` 111 张范围已被取代） |
| `_archive/auv5/datasets_AUV_5_gate-pose_backup_20260924/` | **留档·勿删** | 另一次划分状态（与现行差 1358 文件），是**真实回退点**，不是冗余（**原表误记为 `data/datasets/…`；2026-09-28 归档重排后位于 `_archive/auv5/` 下**） |
| `configs/backup/` | **留档** | 历史标定，只读 |
| `_archive/` | **留档** | 暂时不用但保留；含唯一副本，删除前先读其 README。**09-26 起“归档”有两个位置**（本目录 + `output/preview/auv5_pose/judgment/archive/`），分工见 `_archive/README.md` §8 |
| `_archive/auv5/unneeded_20260926/` | **已归档（本轮）** | **16 项 / 约 850 M**（错域池子 438 M、已消化上传批次 67 M、被取代的 fix111 13 M、3 个旧渲染目录 168 M、作废 worklist 与 pycache）—— **只移动未删除**，原相对路径完整保留，恢复一条 `mv` 即可；依据与逐项见 `_archive/auv5/unneeded_20260926/README.md` |
| `_archive/auv5/unneeded_20260925/` | **待你确认后删** | 「明确不需要」归置区（19 项 / 178 M，只移动未删除）—— 核完可整目录 `rm -rf` |

### 2026-09-24 去重：同一份裸流只留一个副本

> ⚠️ **后续变化（非本次整理）**：上表里保留下来的 `raw-data/recover_all/` 等整个 `raw-data/` 区，
> 已于 **2026-09-24 01:05** 被**外部**清空（目录 mtime `01:05:33`，本仓库现为 0 文件空目录）。
> 本次整理**没有**动过它。下表仅作"当时去重过什么"的历史记录。

整理时按 **md5 逐字节**核对，删掉 6.4 G 的多余副本（**不是推测重复**，每一条都先证 md5 相同再删）：

| 删掉的多余副本 | 释放 | 内容仍在哪 |
|---|---|---|
| `data/AUV_5/raw-mjpeg/`（14 个） | 1.77 G | `raw-data/recover_all/` 同名文件（**100% 含同内容**） |
| `raw-data/recover_prio1/`（14 个） | 1.77 G | 同上（**100% 被 recover_all 覆盖**） |
| `raw-data/20260916_184757/`（流 + timestamps） | 1.92 G | `data/raw/AUV_4_auv_4.mjpeg(.timestamps)`（**被 3 个脚本引用**，保留它） |
| `recover_all/rec_492981881267_4631f_2.mjpeg` | 0.90 G | 同目录 `rec_492981881267_4631f.mjpeg` |
| `recover_all/*_2.mjpeg` ×3（**0 字节残件**） | 0 | — |

逐项记录（含 md5 与保留副本）见 [`_archive/_records/DELETED_2026-09-24_dupes.tsv`](_archive/_records/DELETED_2026-09-24_dupes.tsv)。
> 注意：`data/**/labels/*.txt` 里的 **0 字节文件是 YOLO 的「无目标」负样本**，是正常训练数据，**不要删**。

### 上一轮派生图：**保留在原位，不清理**（2026-09-23 决定）

下面这些是**上一轮的派生图**（不是原始素材）。**决定：一个都不删、位置不动** ——
它们仍可被脚本或人工引用，删掉只会让"当年这批图是怎么筛出来的"变得不可查。
本表只作**备案**：哪天真的缺空间，按「回退方式」一列重生成即可。

| 路径 | 大小 | 是什么 | 真删了的回退方式 |
|---|---|---|---|
| ~~`data/derived/AUV_4_selected_3000/`~~ | 446M | **已归档 2026-09-26**（错域不可用，MAD≈12）；AUV_4 素材改从 `data/frames/AUV_4_auv_4_frames/` 按 D+wb 重渲染。原注：标注在 `data/datasets/AUV_4_PNP.kpt4.yolov8/` | 从 `data/frames/AUV_4_auv_4_frames/` 重抽 |
| `data/derived/AUV_2_selected_2000/` | 193M | AUV_2 旧筛图产物 | 从 `*_frames/` 重抽 |
| `data/derived/AUV_3_selected_2000/` | 251M | AUV_3 旧筛图产物 | 同上 |
| `data/datasets/AUV_1_AUV_data.zip` | 451M | 与同目录解压结果重复 | 解压目录还在 |
| `data/datasets/AUV_1_AUV.yolov11.zip` | 212M | 与 `AUV.yolov11/` 重复 | 同上 |
| `data/calib/AUV_1_board/` | 432M | 标定 A / 空气标定的**输入照片**（结果已存 `configs/backup/`） | ⚠️ **无法复算**那只标定，只能重拍棋盘 |

合计 ≈ **1.98G**。**当前不回收。**

> ⚠️ 已过时（2026-09-26）：上表 6 行里**有 2 行已不在磁盘** —— `data/datasets/AUV_1_AUV_data.zip`（451M）
> 与 `data/datasets/AUV_1_AUV.yolov11.zip`（212M）都进了回收站
> （`/home/ansty/.local/share/Trash/files/`，逐项 `.trashinfo` 记的删除时间同为 **2026-09-25T22:29:41**）。
> ⚠️ **2026-09-26 更新**：`AUV_4_selected_3000`（446M）已归档；其余 3 行（`AUV_2_selected_2000` 193M · `AUV_3_selected_2000` 251M ·
> `calib/AUV_1_board` 432M）2026-09-26 实测**仍在原位、大小与表列一致**。
> 所以"保留在原位，不清理"这个 09-23 决定对那 2 个 zip 已被后续操作（非本次整理）推翻；**2026-09-26 你决定放弃恢复**，保留结论与原因见要从回收站取回**。

**分区原则**：`scripts/` 只放可执行脚本并按"训练前 / 训练 / 导出量化"三段分区；
所有配置进 `configs/`；权重与模型产物进 `weights/`；说明性文档进 `docs/`；
数据与产物不进库；"暂时不用"的一律进 `_archive/` 而不是直接删。

---

## 全流程命令（按阶段）

### 阶段 0 · 环境
```bash
conda activate yolov8                     # 必须！ultralytics 8.3.0 / torch 2.3.1+cu118（GPU 可用）
python -c "import torch; print('CUDA:', torch.cuda.is_available())"   # 期望 True；False 时先查 /dev/nvidia*
docker load < docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz   # 只需一次
```

### 阶段 1 · 训练前（数据准备，`scripts/1_prepare/`）
```bash
# 1) 切帧：容器视频 / 裸 MJPEG 流（自动识别；裸流无损切割）
python scripts/1_prepare/extract_frames.py data/raw/AUV_2_auv_xxx.mjpeg --every-seconds 1

# 2) 相机标定（棋盘 11×8、方格 20mm；同一相机一次，动过相机必须重标）
python scripts/1_prepare/calibrate_camera.py data/calib/AUV_1_board \
    --cols 11 --rows 8 --square-mm 20 --output configs/front_camera.yaml --save-views qc/

# 3) 预处理：去畸变 → 640×640 → 画面补偿（顺序同板端；参数取板端同一份 vision.yaml）
python scripts/1_prepare/prepare_frames.py data/frames/AUV_2_<分类目录> --config configs/vision.yaml

# 4)（可选）筛图：用已有模型剔除不要的类别 + 按质量加权抽样
python scripts/1_prepare/select_frames.py data/frames/AUV_2_auv_20260910_172208_frames \
    --weights weights/yolo11n.pt --drop-classes red_ball \
    --quality-dir data/frames/AUV_2_auv_20260910_172208_frames --keep 2000 --out-dir data/derived/AUV_2_selected_2000

# 5) 标注（RoboFlow）：检测框 或 门 4 角点（规范见下）

# 6) 从 RoboFlow 拿回 pose 导出后，**必做两步**（否则 kpt_shape / 划分都会出问题）
python scripts/1_prepare/pose/prepare_pose_dataset.py data/datasets/AUV_4_PNP.kpt4.yolov8 --dry-run   # 先看要改什么
python scripts/1_prepare/pose/prepare_pose_dataset.py data/datasets/AUV_4_PNP.kpt4.yolov8             # → PNP.kpt4.yolov8
python scripts/1_prepare/resplit_dataset.py data/datasets/AUV_4_PNP.kpt4.yolov8             # 序列感知划分
```

> **⚠️ `weights/yolo11n.pt` 的 `gate` 类目前等于没有输出**：在 AUV_1 valid 集（152 张）上，
> 即便把 `conf` 降到 0.05，`gate` 类最高置信度仍是 **0.00**，28 个门标注框在 IoU@0.5 与 @0.3 下命中均为 **0**；
> 同一次评估 `blue_ball` 86/99、`red_ball` 76/92 都正常。所以**门检测不要用这个权重**，
> 门 4 角点用 pose 模型（`--task pose`）。⚠️ **pose 当前 = 2026-09-24 从 `coco.pt` 从零重训版**
> （`weights/yolo11n-pose.pt`，md5 `2f36303a…`；844 张，val mAP50-95 0.9632 / test 0.9508）。
> 上一代 auv5 因「错误起点（从 auv4 热身，继承了错误内参时代的角点口径）」**已退役删除**，
> 指纹见 `runs/auv5/RETIRED_auv5.md`；**pose 训练必须显式传 `--weights`**（缺省即报错），
> 推荐 `weights/yolo11n-pose.coco.pt`。详见 `runs/auv5/REPORT_auv5_pose.md` §11–§14 与 `weights/README.md`。
> 下一轮训练前请先核对训练数据集的 `names` 顺序、以及 `gate` 标签是否真的进了那一版——
> 一个类别头完全没有输出，通常不是"数据不够"，而是标签或类别顺序的问题。
>
> ⚠️ 补充（2026-09-26）：当天又跑了一轮**阶段一**训练（`data/datasets/AUV_5_stage1`，2585 张，起点同样是
> `weights/yolo11n-pose.coco.pt`），快照落在 `weights/yolo11n-pose.stage1*.pt`，训练记录在
> `runs/auv5/train/stage1{,_full}/`。它**没有**替换上面的 09-24 版（`weights/yolo11n-pose.pt`）：
> 「线上/基准权重」仍指 09-24 版（`base`）。本轮 282 张板端视角判定图已渲染（`renders/boardview_base/`），**2026-09-26 你决定不判**（图留作备查）→ 线上权重仍无可用的板端口径数字。
> ⇒ **线上权重目前没有可用的板端口径数字**。阶段一的判定结果见
> `output/preview/auv5_pose/j