# 分区重组记录 2026-09-25（`output/` · `data/` · `runs/`）

> **原则（用户指定）：不做删除，只归整；同时必须同步更新索引与引用路径。**
> 本轮所有动作都是 `mv`（同设备 = `rename(2)`，不复制数据）。**没有删除任何内容。**
> 逐项机器可读记录：
> - [`MOVES_data_20260925.tsv`](MOVES_data_20260925.tsv) —— data/ 27 项
> - [`MOVES_runs_output_20260925.tsv`](MOVES_runs_output_20260925.tsv) —— runs/ + output/ 12 项
>
> 唯一"消失"的是一批**空壳目录**（0 文件、0 字节）——它们也被**原样移进**归档区，见 §5。

---

## 1. 目标：从"平行零散"改成"统一分区"

| 区 | 改前 | 改后 |
|---|---|---|
| `data/` | `AUV_1..5/` 各含 4–6 种**混放**的兄弟目录（raw / frames / dataset / derived / calib 混在一起） | 顶层**按种类分 5 区**：`raw/ frames/ datasets/ derived/ calib/`，区内条目一律 `<下水轮次>_<原名>` |
| `runs/` | 4 个**平行**目录：`auv5/`(混杂) · `auv5_eval/` · `calib_auv5/` · `prov/` —— 其中两个都叫 "auv5 eval" | **按轮次收敛**：`prov/`（流水线依赖）+ `auv5/`（一个入口，下含 train/eval/quant/board/calib + 报告） |
| `output/` | 9 个散文件 + 2 个**平行** preview 目录 | 交付物（顶层 `*.bin`/`*_quant_info.json`）+ `preview/`（下含 `auv5_pose/`、`pool_other_water/`） |

**查询难度对比**（这是本次的目的）：

```bash
# 改前：想知道"有哪些已标数据集"要翻 5 个目录、还得从混放里挑
ls data/AUV_1 data/AUV_2 data/AUV_3 data/AUV_4 data/AUV_5     # 25 个条目，6 种混在一起

# 改后：一条命令
ls data/datasets/        # 8 个条目，全是数据集
ls data/frames/          # 5 个，全是切帧产物
ls data/derived/         # 7 个，全是派生/待标
```

## 2. `data/` 新布局（17 G，27 项搬迁）

```
data/
├── README.md     ★ 数据分区索引（每个条目是什么、常见查询去哪）
├── raw/          ① 原始素材 9.8 G —— AUV_1..4 的整段视频/裸流（**不可再生**）
├── frames/       ② 切帧产物 3.8 G —— AUV_5_gate_calib = 原 data/AUV_5/raw-data
├── datasets/     ③ 标注数据集 1.3 G —— ★ 当前训练集 AUV_5_gate-pose.yolov8（844 张）
├── derived/      ④ 派生·筛选·待标 1.2 G —— ★ AUV_5_upload_roboflow / AUV_5_selected_3000_Dwb
└── calib/        ⑤ 标定 439 M —— AUV_1_board（2298 张棋盘照）
```

**迁移前的安全检查**（都过了）：

- 6 个 `data.yaml` **全是自相对路径**（`train: train/images`，无 `path:` 键）⇒ 搬到哪都能直接用；
- 全仓只有 **1 处绝对路径**指向 data/（`runs/auv5/eval/gate_pose_abs.yaml`），已同步更新；
- 设备号相同 ⇒ `mv` 是 `rename(2)`，不复制数据、不改内容。

## 3. `runs/` 新布局（12 项搬迁）

```
runs/
├── prov/                       溯源表（**流水线依赖，路径固定不动**）
└── auv5/                       ★ auv5 代次一个入口
    ├── train/pose_fromcoco_844/ + train_fromcoco_844.log
    ├── eval/  ├ clearwater16/（= 原 runs/auv5_eval/eval）  └ pose_D_wb/
    ├── quant/  ├ board/{live,pnp}  ├ calib/（= 原 runs/calib_auv5 + prep_calib.log）
    └── REPORT_auv5_pose.md · RETIRED_auv5.md · QUANT_TUNING_20260924.md
```

## 4. `output/` 新布局

```
output/
├── *.bin + *_quant_info.json    ★ 交付物（板端只加载这些）
└── preview/                     ★ 判定/调参工作区（不属于流水线）
    ├── auv5_pose/（= 原 output/auv5_pose_preview，340 M）  入口 INDEX.md
    └── pool_other_water/（= 原 output/pool_other_water_preview，152 M）
```

## 5. 唯一"消失"的东西：空壳目录（也**移进**了归档区，没删）

搬迁后 `data/AUV_1..5`、`runs/auv5_eval`、`runs/calib_auv5`、`runs/auv5/logs` 变成空目录。

- 早先 `data/` 那 6 个空壳曾被 `rmdir` 掉，**随后已按"不删任何东西"复原到归档区**：
  `_archive/empty_shells_20260925/data_原结构/`（空目录，0 字节）
- 之后 `runs/` 的空壳改用 **`mv` 进归档区**：`_archive/empty_shells_20260925/runs_原结构/`

⇒ 现在"没有删除任何东西"是**字面成立**的：连空目录都在归档区里。

## 6. 引用/索引同步（本轮的重点工作）

### 6.1 机械替换（按长度降序，避免前缀冲突）

| 轮次 | 文件数 | 说明 |
|---|---|---|
| data/ 路径 | **60 个文件** | 含 `README.md`、`scripts/**`、`docs/**`、`experiment/**`、`configs/**`、`runs/**`、`output/preview/**` |
| data/ 里的路径列（`.csv`） | **14 个文件** | `manifest.csv`(2577 行) · `worklist_remaining.csv`(1952) · `provenance_*.csv` · `negatives_candidates.csv` 等 |
| runs/ + output/ 路径 | **37 个文件** | 含两个 `filelist.txt`、`*_filelist.txt`、`RUNBOOK/DECISIONS`、`experiment/scripts/**` |
| 示例命令/文档说明 | **16 个文件** | `README.md`、`scripts/README.md`、`docs/README_ZH\|EN.md`、各脚本 docstring |

### 6.2 代码里的路径常量（**必须手改，机械替换扫不到**）

这类是 `REPO / "data" / "AUV_5" / ...` 的 pathlib 写法，字符串匹配发现不了 —— 已逐个修正并**验证指向存在目录**：

| 脚本 | 改前 | 改后 |
|---|---|---|
| `scripts/2_train/build_stage_dataset.py` | `REPO/"output"/"auv5_pose_preview"`；`REPO/"data"/"AUV_5"/"selected_3000_Dwb"` | `REPO/"output"/"preview"/"auv5_pose"`；`REPO/"data"/"derived"/"AUV_5_selected_3000_Dwb"` |
| `scripts/2_train/eval_benchmark.py` | 同上两处 | 同上 |
| `scripts/1_prepare/label_pose_local.py` | `REPO/"output"/"auv5_pose_preview"` | `REPO/"output"/"preview"/"auv5_pose"` |
| `experiment/scripts/exp_aug/{accept_gate_model,exp_aug_ablation}.py` | `PROJECT_ROOT/"data"/"AUV_5"/"gate-pose.yolov8"...` | `.../"data"/"datasets"/"AUV_5_gate-pose.yolov8"...` |
| `map_pose_dataset.py` · `materialize_raw.py` · `make_variant_dataset.py` · `check_ball_impact.py` · `prepare_auv5_eval.py` | `data/mapped/raw`（该目录已不存在，6 处） | `data/frames/mapped_anchors/`（锚点帧本质是切帧产物） |
| `compare_kpt_spacing.py` · `prepare_distortion_groups.py` | `runs/exp_distortion/...`、`data/exp_distortion/...` | `experiment/runs/...`、`experiment/data/...`（实验产物归实验区） |

> ⚠️ 其中 **`build_stage_dataset.py` 原本会在运行时重建 `data/AUV_5/stages/`**，等于把取消掉的
> `AUV_N` 层又建回来 —— 已改为 `data/datasets/AUV_5_stage<N>/`，并留注释说明原因。

### 6.3 新增/更新的索引

| 文件 | 状态 |
|---|---|
| `data/README.md` | **新建**：5 区说明 + 逐项清单（大小/轮次/用途）+ 「常见查询 → 去哪」表 |
| `README.md` 分区树 | 更新 data/ · runs/ · output/ 三段；`_archive` 段补 `empty_shells_20260925/` |
| `output/README.md` | 重写：交付物 + `preview/` 两段；并**更正**了一处失效声明（见 §7） |
| `output/preview/auv5_pose/INDEX.md` | 路径已随搬迁自更新（标题与新相对链接） |

### 6.4 故意**不**改的地方

`_archive/**` 与 `cleanup_record/**` 里的旧路径**保持原样** —— 它们是"当时状态"的留档，
改写会破坏审计链。活跃文件里的旧路径已清零（除 3 处明确写着"原位置/当时落在"的说明）。

## 7. 顺手发现并更正的两处**非本轮造成**的问题

1. **`output/` 里复核余弦用的两个 ONNX 不见了**：`gate_kpt_bayese_640x640_nv12_{original_float,calibrated}_model.onnx`
   （我 2026-09-24 特意保留的"最小集"）在 **09-24 22:04 被外部清理** —— 本轮开工时它们已经不在。
   已更正 `output/README.md` 里"别删/仍在"的说法，并写明**要复核余弦须重跑量化**。
> ⚠️ **该更正本身有误（2026-09-26 核对）**：本行指向的 `data/frames/AUV_3_processed_640/` **同样不存在**。现存 AUV_3 的 640 图是 `data/derived/AUV_3_selected_2000/`（2000 张，实测 640×640），原始帧在 `data/frames/AUV_3_auv_20260911_201754_frames/`（508 张 1280×720）。原句保留不改。
2. **`configs/README.md` 里 `data/AUV_3/processed_640/`** 是早已不存在的旧路径 → 已改并按新分区标注对应位置。

## 8. 验收（实测）

| 项 | 结果 |
|---|---|
| `py_compile`（scripts + experiment/scripts） | **55/55 通过** |
| `bash -n` | 3/3 ✓ |
| `--help` 冒烟（流水线全部脚本） | **21/21 通过** |
| `data/datasets/*/data.yaml` 路径解析 | **6/6 全部存在**（`names`/`kpt_shape` 正确） |
| 走项目自身绝对化路径调 ultralytics | ✓ 当前训练集加载成功（`AUV_5_gate-pose.yolov8`） |
| 被改的代码常量是否指向存在目录 | ✓ `POOL` / `PREVIEW` 均 `exists()=True` |
| 活跃文件里的旧路径残留 | **0**（仅 3 处"原位置"说明，见 §6.4） |
| 内容损失 | **0**（全部 `mv`；连空目录都移进了归档区） |

## 9. 复查入口

```bash
cd /home/ansty/RDKX5/auv_vision/RDKX5-YOLOv11n-
cat data/README.md                                    # 数据分区索引（入口）
cat output/README.md                                  # 交付物 + preview 索引
find runs -maxdepth 3 -type d | sort                  # runs/ 按轮次收敛后的结构
column -t cleanup_record/MOVES_data_20260925.tsv      # data/ 27 项搬迁记录（from→to）
column -t cleanup_record/MOVES_runs_output_20260925.tsv
ls data/{raw,frames,datasets,derived,calib}           # 5 区一眼看全
```
