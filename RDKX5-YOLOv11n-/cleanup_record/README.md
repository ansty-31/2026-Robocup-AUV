# `cleanup_record/` — 整理变更记录总索引

**这里只放记录，不放产物。** 2026-10-06 分类：`reports/`（人读的报告）＋ `moves/`（机器可读的逐项清单）
＋ `AUV_6_clear_v1fix/`（具体改动的原始留档）。

> ⚠️ **这些是历史记录，路径均为"当时"的状态，故意不改写。** 要用它们回滚时以最新一份为准；
> 目录被后续整理动过时，`to` 列可能已不是当前位置（各 `MOVES_*.tsv` 文件头有 NOTE 说明）。

```
cleanup_record/
├── README.md              ← 本文件
├── reports/               人读的报告：每次整理的"做了什么、为什么、怎么回退"
├── moves/                 机器可读清单：from / to / kind / files / bytes / why
└── AUV_6_clear_v1fix/     2026-09-26 标签修正的**原始留档**（改动前的那 48 个标签 + data.yaml.orig）
```

## `reports/` — 报告（5）

| 报告 | 讲什么 |
|---|---|
| [`CLEANUP_2026-09-24.md`](reports/CLEANUP_2026-09-24.md) | 第一次大清理：2.6 G 清理 + 6.4 G 去重 + 根目录归置 + manifest 对齐（2690→2577） |
| [`CLEANUP_2026-09-25.md`](reports/CLEANUP_2026-09-25.md) | 09-25 的分类、记录、归置 |
| [`RESTRUCTURE_2026-09-25.md`](reports/RESTRUCTURE_2026-09-25.md) | `data/` `runs/` `output/` 三分区的重组（按种类/轮次收敛） |
| [`MOVES_2026-09-26.md`](reports/MOVES_2026-09-26.md) | 板端视角口径修订 · T5 缩量；含标签修正、data.yaml 路径修正等逐项改动说明 |
| [`BOARD_Adomain_retire_20261006.md`](reports/BOARD_Adomain_retire_20261006.md) | A-domain 棋盘/分辨率那条线的除役记录 |

## `moves/` — 逐项搬迁清单（9）

格式统一为 `from / to / kind / files / bytes / why`，可据此**逐项反向 `mv` 回滚**。

| 清单 | 覆盖范围 |
|---|---|
| [`MOVES.tsv`](../_archive/_records/MOVES.tsv) | 2026-09-23 第一次归档（在 `_archive/_records/`，含内容指纹） |
| [`MOVES_data_20260925.tsv`](moves/MOVES_data_20260925.tsv) | `data/` 按种类分五区（27 项） |
| [`MOVES_runs_output_20260925.tsv`](moves/MOVES_runs_output_20260925.tsv) | `runs/` + `output/` 重排（12 项） |
| [`MOVES_auv5_pose_20260925.tsv`](moves/MOVES_auv5_pose_20260925.tsv) | `output/preview/auv5_pose/` 一级分类（48 项） |
| [`MOVES_auv5_pose_20260926.tsv`](moves/MOVES_auv5_pose_20260926.tsv) | 同上二级分类：分桶→`judgment/buckets/`、报表→`reports/`、tables/filelists 分组（61 项） |
| [`MOVES_20260926.tsv`](moves/MOVES_20260926.tsv) | 09-26 判定数据归档（作废日志与分桶副本 → `judgment/archive/`） |
| [`MOVES_20260926b.tsv`](moves/MOVES_20260926b.tsv) | `pool_other_water` 同形归类 · `runs/auv5/{quant,train}` 分组 · `labelcheck` 重复入档（59 项） |
| [`MOVES_archive_20260926.tsv`](moves/MOVES_archive_20260926.tsv) | 09-26 归档批次（16 项 / ~835 MB，进 `_archive/auv5/unneeded_20260926/`） |
| [`MOVES_archive_20260928.tsv`](moves/MOVES_archive_20260928.tsv) | 归档区按项目/代次重排为 5 组（23 项） |
| [`MOVES_20261006.tsv`](moves/MOVES_20261006.tsv) | **本次（2026-10-06）**：`runs/` 日志分类 · 作废训练归档 · `data/` 命名笔误修正 · 本目录分类（38 项） |

## `AUV_6_clear_v1fix/` — 改动原始留档（50 文件）

2026-09-26 对 `AUV_6_clear` 做的那次标签修正，**改动前的原文件**：

- 48 个标签文件的原始版本（那次把 64 个角点的 `v=1` 改成 `0`，避免不可见角点参与坐标损失）
- `data.yaml.orig`（路径修正前的原文件）

> 放这里的意义：**想知道"改之前长什么样"或要回退那次改动**，从这里取。
> 当时那份改后数据在 `data/datasets/AUV_6_clear/`。详见 `reports/MOVES_2026-09-26.md` §标签修正。

## 删除记录在别处

**"删了什么"不在这里**，在归档区的记录里（因为删除都发生在归档区）：

- [`_archive/_records/DELETED_2026-09-24_dupes.tsv`](../_archive/_records/DELETED_2026-09-24_dupes.tsv) —— 去重删除（34 项，逐项 md5 + 保留副本位置）
- [`_archive/_records/DELETED_2026-09-24_intermediates.tsv`](../_archive/_records/DELETED_2026-09-24_intermediates.tsv) —— 量化中间产物（6 项）
- [`_archive/_records/DELETED_2026-09-28.tsv`](../_archive/_records/DELETED_2026-09-28.tsv) —— 11 项 / 约 1.5 G（含逐项证据与可恢复性）
