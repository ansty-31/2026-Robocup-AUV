# 板端旧树归档删除 · A 域退役（2026-10-06）

> **本记录的对象不在本仓库里**：被删的是**板端** `/home/sunrise/AUV/`（09-17 的旧树，A 域时代产物）。
> 本文件是它的**文字记录**，落点在本仓库，防止后续又有人把 A 域捡回来。
> 板端自己的那份记录在 `/home/sunrise/Desktop/AUV_New/doc/记录/2026-10-06-A域退役与旧树清理.md`。

## A. 为什么删

**A 域已被舍弃**（2026-10-06 用户决定）：撞球 / 过门都走 **D + wb**（白平衡）；
**夹取小球只做 `resize 640×640` 直接送 BPU，不做任何图像处理**（连去畸变/增强都不要）。

但板端有两棵树，情况不一样 —— 这是本次才查清的事：

| 位置 | 规模 | A 域状态 | 结论 |
|---|---|---|---|
| `/home/sunrise/Desktop/AUV_New/`（**在用副本**，`deploy_to_board.sh` 的目标） | 75 M | **已退役**：`common/preprocess.py` / `tools/check/check_domain.py` 里已无 A 分支，`cfg/vision.yaml` 无 `chain` 键、`clahe_clip: 0`；`tests/platform/test_common.py` 留了"2026-10-01 退役"注释 | 没问题，不动 |
| `/home/sunrise/AUV/`（**旧树**，不在任何部署清单里，板端 `tools/README.md §6` 自己写了"不是权威副本、只在故意跑旧一代权重时用"） | 333 M / 583 条目 | **A 域还活着**：`chain: A`、`clahe_clip: 0.5`、gate=AUV_4、`preprocess.py` 有 A 分支 | **唯一还能把 A 域跑起来的地方 → 本次清理对象** |

清理前逐项确认过（缺一条就不删）：

1. 全树**无引用**：`grep -rl '/home/sunrise/AUV'` 只命中 `AUV_New/bak/` 里的历史快照；
2. 无 crontab、无 systemd 自启；
3. 顶层目录 mtime = 2026-09-27（就是那次手工换门权重），之后再没动过。

## B. 做了什么

```bash
# 1) 归档（板端执行，317 MB）
cd /home/sunrise && tar -czf \
  /home/sunrise/Desktop/AUV_New/bak/legacy_Adomain_tree_20261006.tar.gz AUV

# 2) 删除前置校验（脚本 experiment/scripts/small_ball/retire_Adomain_tree.py）
#    - 归档条目数 583 == find /home/sunrise/AUV | wc -l 583
#    - 抽样字节级 cmp 通过：cfg/vision.yaml、common/preprocess.py、models/yolo11n_detect_*.bin
#    - 归档 md5 3a7cda63af1f9453006e17cd3a26734f 与预期一致
# 3) 写文字记录后 rm -rf
python3 retire_Adomain_tree.py --archive <归档> --md5 3a7cda63… --apply
```

| 项 | 值 |
|---|---|
| 归档 | `/home/sunrise/Desktop/AUV_New/bak/legacy_Adomain_tree_20261006.tar.gz`（317 MB，md5 `3a7cda63af1f9453006e17cd3a26734f`） |
| 条目数 | 583（= 删除前 `find` 计数，逐项一致） |
| 释放 | 348 MB 目录 → 317 MB 归档（内容多为 JPEG/`.bin`，压缩比极低） |
| 校验 | 归档流可完整读取（`tar -tzf` 无错）、3 个文件抽样 `cmp` 相同 |

**恢复方式**（要回溯旧行为时）：
```bash
sudo tar -xzf /home/sunrise/Desktop/AUV_New/bak/legacy_Adomain_tree_20261006.tar.gz -C /home/sunrise
```

## C. 连带影响：本仓库 `configs/` 里两条悬空路径

旧树被删后，`configs/vision.yaml` 里两条**指向旧树绝对路径**的字段就悬空了（本仓库之前从旧树同步过镜像）：

| 位置 | 原值 | 处置 |
|---|---|---|
| `configs/vision.yaml:15` | `calibration: /home/sunrise/AUV/cfg/front_camera.yaml` | 改为 `front_camera.yaml`（`prepare_frames.py` 有"本机不存在→回退 configs/ 同名文件"的兜底，本地可用） |
| `configs/vision.yaml:53` | `path: /home/sunrise/AUV/models/auv_multi.bin` | **保留但标注悬空**：在用副本 `AUV_New/models/` 里**没有** `auv_multi.bin`（只有 `yolo11n_detect_*` 与两个 `gate_kpt_stage1_*`），说明这一行本来就是旧树时代的残留，不该照抄进训练参数 |

> ⚠️ **`configs/vision.yaml` 这个镜像已确认"混血/滞后"**（部分是旧树值、部分是 D 域值）。
> 它只被 `scripts/1_prepare/prepare_frames.py` 用来读 `image.* / camera.* / model.input_size`，
> 这几个字段没被动过；但**不要再把它当权威**。要权威值请从 `AUV_New/cfg/vision.yaml` 重新同步
> （同步命令与说明见 `configs/README.md`）。
> 另外：**夹取小球这条线不需要它** —— 该任务按决定只做 `resize 640×640`，不读任何域参数。

## D. 规矩（防止 A 域复活）

1. 识别链路**只有 D + wb**：`resize 640 → enhance(WB) → remap@640`，`clahe_clip: 0`。
2. 新任务**不要自带图像处理**：夹取小球 = 原图 `resize 640×640` → NV12 → BPU，仅此。
3. **不要再给 `vision.image` 加域开关**（`chain` 之类）。历史教训：A/D 两域 × 两代权重交叉喂图 →
   角点飘、球漏检，且**不报错、只静默掉点**。要换输入口径就换权重，别给同一条链路加分叉。
4. 板端 "哪棵树在用" 只认 `AUV_New`（`tools/README.md §6` 的对照表）；旧树的坑（如
   `models/gate_kpt_bayese_640x640_nv12.bin.` **文件名尾部多一个点**导致 gate 被静默跳过）随删除一并消失。
