# scripts/ 脚本索引

脚本按**流程阶段**分三区，完整流程命令与输出契约见项目根 [README.md](../README.md)。

> **路径书写约定（2026-09-26 逐条核对磁盘后补）**：正文里作为「指向」用的独立路径（反引号包起来的那个）
> 一律相对**本文件所在目录**（`scripts/`）书写 —— `1_prepare/pose/prepare_pose_dataset.py` 即
> `scripts/1_prepare/pose/prepare_pose_dataset.py`；本目录之外的写 `../configs/…`、`../runs/…`、`../data/…`。
> `bash` 代码块里的**命令示例按仓库根执行**，所以那里仍写 `python scripts/…`。
> 板端 / 三方路径会显式标注「板端」「三方」，它们不在本仓库里。

> **运行环境**：下面所有 `python ...` 命令都必须先 `conda activate yolov8`
> （= `/home/ansty/anaconda3/envs/yolov8/bin/python`，ultralytics 8.3.0 + torch 2.3.1+cu118，GPU 可用）；
> 系统 `python3` 没装 torch/ultralytics，会直接 `ModuleNotFoundError`。

```
1_prepare/   训练前：数据准备
             ├── (根) 共用·常用：切帧 / 标定 / 预处理 / 筛图 / 重划分
             ├── pose/        ★ 门（4 角点）专用：Roboflow pose 导出 → 跨域换算 → 挑待标图
             └── provenance/  素材溯源：把已标的图对回原始帧（已一次性闭合，换数据集可复用）
2_train/     训练：detect / pose
3_export/    导出与量化：改输出头 → ONNX → 校准数据 → PTQ 量化
```

> 已完成的实验脚本与其产物**不删除**，训练记录留在 `../runs/`，暂时不用的移入
> [`_archive/`](../_archive/README.md)（含恢复方法）。

## 1_prepare/ — 训练前（数据准备）

### 根目录：共用 / 后期常用（新素材的常规流程走这几个）

| 脚本 | 用途 | 常用参数 |
|---|---|---|
| `1_prepare/extract_frames.py` | 切帧：自动识别**容器视频 / 裸 MJPEG 流**（裸流按 JPEG 标记无损切割） | `--step N`、`--every-seconds S`、`--start-frame`、`--max-frames`、`--dry-run` |
| `1_prepare/calibrate_camera.py` | 相机标定：棋盘（视频或图片目录）→ 内参 yaml；全池择优 + 离群剔除，RMS≤0.8px 验收 | `--cols/--rows/--square-mm`、`--views`、`--min-shift`、`--save-views` |
| `1_prepare/prepare_frames.py` | 预处理：**640×640 → 白平衡/CLAHE/gamma → 去畸变 remap@640**（= **D 域**，2026-09-23 起定稿；与板端 `common/preprocess.py` 同链路同顺序、**参数以板端为准**，镜像见 [configs/README.md](../configs/README.md)）。历史 P1 链序（先去畸变再缩放）的分支已于 2026-09-24 删除 | `--config configs/vision.yaml`、`--out-dir`、`--size`；或 `--calibration` + `--wb-gains/--clahe/--gamma` |
| `1_prepare/select_frames.py` | 用模型剔除指定类别画面，再按 `清晰度×亮度合理性` 质量加权随机抽样；**多个输入目录合并为一个候选池**（来源仅用于统计/报告，同名加 `_2` 后缀防覆盖）。不给 `--drop-classes` 时**跳过推理**（只滤损坏帧，`--weights` 可不传） | `--drop-classes`、`--weights`、`--conf`、`--quality-dir`、`--keep`、`--gamma`、`--report` |
| `1_prepare/select_by_conf_quality.py` | 按模型估计筛图：置信度低→高 + 质量高→低，挑"单门高质量/只剩边角/贴脸"三类并硬链接到输出目录（写 manifest 与全量估计表） | `--out` `--n` `--conf` `--gap` `--reuse` |
| `1_prepare/resplit_dataset.py` | 数据集重划分：**序列感知**（同段连续帧不跨 split，避免相邻帧泄漏）+ 稀有类平衡 + 自动备份；先 `--dry-run` 看方案 | 数据集根目录、`--ratios 0.8 0.1 0.1`、`--group-gap`、`--seed`、`--dry-run`、`--force` |
| `1_prepare/label_pose_local.py` | **本地手工标 4 角点 + tag**（拖框→框内点 4 点，角点顺序自由；支持多门、不可见角点）。⚠️ 2026-09-25 实测本机环境跑不通已改走 Roboflow，**保留备查**；规范见 `../docs/labeling_gate_pose.md` | 清单 CSV/图片目录、`--images-dir`、`--out`、`--pad-frac`、`--selftest` |

> `prepare_frames.py` 的链序（D 域 / P2）：`resize(640) → enhance → remap@640`，与板端 `common/preprocess.py::process` 一致。

### `pose/` — 门 / 4 角点专用（detect 线用不到）

| 脚本 | 用途 | 关键参数 |
|---|---|---|
| `1_prepare/pose/check_pose_export.py` | 把导出标签里 **v=1 的角点改成 v=0**（写 `--out` 新目录）。约定：`v=2` 可见 / `v=0` 不可见 / `v=1` 不用 | `--out` |
| `1_prepare/pose/prepare_pose_dataset.py` | **Roboflow pose 导出 → 可训练数据集**：截断「尾随全零」关键点槽位、校验角点顺序约定、决定 `flip_idx`；图片用硬链接，**原导出目录一个字节不改** | 导出目录、`--kpt 4`、`--out`、`--flip-idx-lr-swap`、`--dry-run` |
| `1_prepare/pose/map_pose_dataset.py` | **跨去畸变域换算标注 + 生成 pose 训练集**。`--domain B\|C\|D` × `--enhance on\|off\|wb`（`wb` = 白平衡+gamma、**无 CLAHE**；默认输出目录随之带 `_noenh`/`_wb` 后缀）。**默认落 `../data/mapped/pose_D_wb`，该目录 2026-09-24 已清理**（`../data/mapped/` 整个目录当前都不存在，属按需生成），重跑会按默认名重建（原定稿生产集）。`--selfcheck` 环回校验（raw → D → raw，应 ~0.001 px）。<br>同时是**各域图像/坐标的公共库**：`Domain` 类被 detect 侧的域对照、球影响检查等复用 | `--domain`、`--enhance`、`--selfcheck`、`--out`、`--prov` |
| `1_prepare/pose/make_mixed_gate_set.py` | 新一版「selected_3000」：跨水质融合门图 → 定稿链路（D + wb）渲染 → 质量/门框存在性过滤 → **内容去重**（配额不足时自动逐级放宽阈值）→ 待标图 + `manifest.csv`（默认落 `../data/derived/AUV_5_selected_3000_Dwb/`） | `--total`、`--dedup-thr`、`--oversample`、`--out-dir`、`--no-model-filter` |

### `provenance/` — 素材溯源（已一次性闭合；换数据集可复用）

回答"这张已标的图到底是哪一段裸流的第几帧"，是 `1_prepare/pose/map_pose_dataset.py` 的上游输入。
**注意**：源帧索引（`../runs/prov/index_enh.npz`，216 MB）已删（2026-09-26 核实：文件不在磁盘上），需要时重建约 1 分钟。

| 脚本 | 用途 |
|---|---|
| `1_prepare/provenance/materialize_raw.py` | 把溯源命中的**锚点帧**落成素材（默认 `../data/frames/mapped_anchors/`，各域渲染的输入；该目录 2026-09-24 已清理，重跑即重建） |
| `1_prepare/provenance/provenance_match.py` | 图↔图溯源匹配（数据集内部对回） |
| `1_prepare/provenance/provenance_stream.py` | 在**裸 MJPEG / mp4 流**上按帧号直达；`--offset TAG=N` 修被重编号的组 |
| `1_prepare/provenance/provenance_nn.py` | 内容最近邻溯源（精确命中失败时兜底；阈值要放宽，真命中 mad128 约 0.4–0.6） |
| `1_prepare/provenance/provenance_merge.py` | 合并多轮结果 → 统一 provenance 表（下游 `1_prepare/pose/map_pose_dataset.py` 的输入） |

> ⚠️ **docstring 与代码不一致（2026-09-26 发现，本轮只同步索引、未动脚本）**：
> `1_prepare/provenance/materialize_raw.py` 的模块 docstring 写输出是 `data/mapped/raw/<src>/…`，
> 但代码里没有 `--out` 参数，落盘位置是常量 `OUT = data/frames/mapped_anchors`（该文件第 34 行）。
> 上表与正文按**代码**写；docstring 待改。

> **实验专属工具已移出本目录** → `../experiment/scripts/`（去畸变域对比、板端对照、标定体检等）：
> 只服务那一轮实验的东西全归那里。分类索引见 [`../experiment/README.md`](../experiment/README.md)。

```bash
python scripts/1_prepare/extract_frames.py data/raw/AUV_2_auv_xxx.mjpeg --every-seconds 1
python scripts/1_prepare/calibrate_camera.py data/calib/AUV_1_board --cols 11 --rows 8 --square-mm 20 \
    --output configs/front_camera.yaml --save-views qc/
python scripts/1_prepare/prepare_frames.py data/frames/AUV_2_<分类目录> --config configs/vision.yaml
python scripts/1_prepare/select_frames.py data/frames/AUV_2_auv_20260910_172208_frames --weights weights/yolo11n.pt \
    --drop-classes red_ball --quality-dir data/frames/AUV_2_auv_20260910_172208_frames --keep 2000 --out-dir data/derived/AUV_2_selected_2000
```

### ~~混入「不去畸变」图，增强模型韧性~~（已废弃）

> **2026-09-24 收口**：这条路线**连同它的参数一起删除**了 —— 配套脚本
> `prepare_frames_noundistort.py` 已删（2026-09-26 核实：磁盘上确实没有）；`1_prepare/select_frames.py` 的 `--mix-ratio` / `--mix-prefix`
> 也已删除（去畸变域定稿为 **D 域**：`resize(640) → enhance → remap@640`，见
> `1_prepare/pose/map_pose_dataset.py` 与 [`../experiment/README.md`](../experiment/README.md) 的域对比结论）。
>
> `1_prepare/select_frames.py` 仍支持**多个输入目录**，但语义简化为**合并成一个候选池**再统一抽样
> （来源只用于统计与 CSV 报告）；同名文件自动加 `_2`/`_3` 后缀防覆盖。
> 要复现旧的"按比例掺入 + `nd_` 前缀"行为：`git show 54e09b4:.../select_frames.py`（旧路径
> `scripts/1_prepare/select_frames.py`，`git cat-file -t 54e09b4` 已确认该 commit 存在）。

### Roboflow pose 导出 → 可训练数据集（标完必做两步）

```bash
# 1) 归一：kpt_shape → [4,3]（截掉尾随全零槽位）+ 校验 TL,TR,BR,BL + 定 flip_idx
python scripts/1_prepare/pose/prepare_pose_dataset.py data/datasets/AUV_4_PNP.kpt4.yolov8 --dry-run
python scripts/1_prepare/pose/prepare_pose_dataset.py data/datasets/AUV_4_PNP.kpt4.yolov8   # → data/datasets/AUV_4_PNP.kpt4.yolov8

# 2) 序列感知重划分（Roboflow 的随机划分有严重的相邻帧泄漏）
python scripts/1_prepare/resplit_dataset.py data/datasets/AUV_4_PNP.kpt4.yolov8 --dry-run
python scripts/1_prepare/resplit_dataset.py data/datasets/AUV_4_PNP.kpt4.yolov8
```

- **`kpt_shape` 必须核**：RoboFlow 项目里删掉的关键点槽位**不会从导出里消失**——`data.yaml` 常写
  `kpt_shape: [5, 3]`，而第 5 个关键点在**全部**标签里都是 `0 0 0`（x=y=v=0）。带着它训练，
  输出头 kpt 通道会从 12 变 15，与 `../configs/gate_kpt_config.yaml` 的 kpt12、板端 `gate/gate_decode.py`
  全对不上（`3_export/export_onnx.py` 会告警"gate 需要 4 点"）。脚本会自动截断并断言结果 == `--kpt`。
- **`flip_idx` 不能想当然改**：三方 `ultralytics/data/augment.py`（site-packages 内，不在本仓库）是「先镜像坐标 → 再 `keypoints[:, flip_idx]`
  重排」，两种约定各自自洽：恒等 `[0,1,2,3]` = 索引跟**物理角点**走；`[1,0,3,2]` = 索引跟**图像位置**走。
  两者只在镜像图 / 从门背面看时有区别，而板端 `gate/geometry.py:object_points()` 写明
  「前后完全对称、无朝向要求 → from_front 恒 true」，对 PnP 等价。**默认沿用 RoboFlow 的恒等值**，
  与既有已量化 bin 的约定保持一致（`[0,1,2,3]` 是历代都用的恒等值）；要换用 `--flip-idx-lr-swap`，
  但这会改变增广语义，**换之前先确认不是在做微调**（会和初始权重打架）。
- **多视频源会撞帧号**：不同视频共用 `frame_NNNNNN` 命名时，同一帧号会对应不同画面
  （实测 97 组重复帧号里 **96 组是跨视频撞号**，只有 1 组是同帧的主集/nd 对）。
  `1_prepare/resplit_dataset.py` 按帧号聚段，撞号只会把不同视频的帧**并进同一段**——这是**保守方向，不会造成泄漏**，
  代价是段变粗、val/test 与目标比例的偏差变大。**下次导出建议给文件名加视频前缀**（如 `a2_frame_001885.jpg`）。

## 2_train/ — 训练

| 脚本 | 用途 | 关键参数 |
|---|---|---|
| `2_train/eval_benchmark.py` | **固定基准评测器（验收唯一口径）**：A 人眼基准（1190 张）/ B 分层保留集（代理）/ C 负样本误检率 / D 板端视角（只判板端选中的那一个实例）+ 门槛判定。**误差只能由 labels 算，好坏只能由人眼判定**；脚本里的 `v≥V_MIN`（**2026-09-26 起为 0.8**，此前 0.9）/非自交/顺序 都是工程约束不是正确性标准。**2026-09-26 D 节改版**：渲染改为**全实例画法**（过约束实/虚线 + 编号、未过约束 `✗FAIL`、被去重 `✗DUP`），判定桶扩到 **6 个 `k`/`y`/`w`/`b`/`p`/`x`**（只有 `k` 算可用），并给**严格口径**（图上每个门都必须对）/ **宽松口径**（多门里有一个好门即可）两个不可用率，另按 `subset` / 原图桶 / 过约束实例数分层 | `--weights`、`--conf`、`--verdict`、`--json`、`--boardview-log`、`--boardview-list`、`--boardview-subset` |
| `2_train/build_stage_dataset.py` | **按 tag 生成某一期数据集**：累积（`--base` + 多个 `--add`，天然 rehearsal 防遗忘）+ 硬链接过采样 + 负样本空标签 + 排除评测保留集；`data.yaml` 原样沿用 `kpt_shape=[4,3]`/`flip_idx=[0,1,2,3]` | `--stage`、`--base`、`--add`、`--negatives`、`--human-tags`、`--replicate`、`--dry-run` |
| `2_train/train_yolo11n.py` | 训练/微调（`--task detect\|pose`）：自动恢复原版 head → 训练 → best 复制到 `../weights/` → 按任务重打输出头补丁 | `--data`、`--task`、`--weights`、`--epochs`、`--batch`、`--imgsz`、`--device`、`--cache ram`、`--no-repatch` |

> ⚠️ **已过时（2026-09-26）：D 节原口径「只判板端选中的那一个实例」作废。**
> 那条"我方排序规则会挑哪一个实例 = 板端会用的那一个"是**本仓的假设，不是板端契约**（本仓没有板端选门代码）。**板端选门规则由用户给定（2026-09-26）：近距离优先；无法测距时选置信度高或画面大的** → D 节**以严格口径为主**；
> 旧画法会把画面里第二个门整个藏掉，且顶部信息条有 46 px 错位，**该轮 39.7% 的数字已作废**。
> 现状见上文表格与 `../cleanup_record/MOVES_2026-09-26.md`、`../output/preview/auv5_pose/docs/CRITERIA.md`。
> 另外**口径不可混用**：「坏率 12.8%」（A 节旧清水重判，V_MIN=0.8）与「板端不可用 8.8%（严格）/1.2%（宽松）」（D 节 339 张）
> 回答的不是同一个问题，不能对比；线上权重（`boardview_base`，282 张）**用户决定不判**（2026-09-26）⇒ 线上权重**没有**有效的板端口径数字。

> 域对比实验的批量训练脚本 `train_domain_arms.sh` 已移到 `../experiment/scripts/`（它只服务那次对照实验，不属常规训练）。

> **D 节（板端视角）的图与判定不在本目录**：渲染器是 `../output/preview/auv5_pose/tools/make_boardview_set.py`
> （2026-09-26 重写：**全实例画法** + `--extra-names-from` + 实例级长表；修掉旧的 46 px 错位），
> 分桶判定是 `../output/preview/auv5_pose/tools/triage_review.py`，索引见
> `../output/preview/auv5_pose/INDEX.md`、口径见 `../output/preview/auv5_pose/docs/CRITERIA.md`。

```bash
# detect（默认权重 weights/yolo11n.pt，输出 weights/yolo11n.pt）
python scripts/2_train/train_yolo11n.py --data <data.yaml> --epochs 300 --batch 4 --device 0 --cache ram

# pose：**必须显式指定起点**（2026-09-24 起缺省即报错；推荐从官方预训练重训）
#   当前生产权重是 09-24 从 coco 重训版（weights/yolo11n-pose.pt, md5 2f36303a…）
python scripts/2_train/train_yolo11n.py --task pose --data <data.yaml> \
    --weights weights/yolo11n-pose.coco.pt --epochs 300 --batch 4 --device 0
```

要点：
- **训练必须在原版 head 下**（脚本自动 restore）；训练后自动重打补丁，衔接导出；
- 本机 8 个 dataloader worker 会与 CUDA fork 死锁 → 默认 `--workers 2`；显存小用 `--batch 4`；
- pose 数据集需 `kpt_shape: [4, 3]` 且 `len(flip_idx) == 4`（ultralytics 会校验长度）。
  **RoboFlow 的导出未必对**（常见 `[5,3]` + 全零第 5 点），先过
  `1_prepare/pose/prepare_pose_dataset.py` 再训，见上一节的"标完必做两步"。

## 3_export/ — 导出与量化

| 脚本 | 用途 | 关键参数 |
|---|---|---|
| `3_export/modify_ultralytics.py` | 输出头补丁：`--task detect`=6 输出 / `--task pose`=9 输出；`--restore` 回退原版（自动备份 `head.py.backup`） | `--task`、`--restore` |
| `3_export/export_onnx.py` | 导出 ONNX（opset 11 / 640）并校验张量结构 | `--task`、`--model`、`--output`、`--imgsz` |
| `3_export/prepare_calibration.py` | 生成 PTQ 校准数据（RGB float32 CHW 640 的 `.rgb`） | `--coco-path`、`--output-dir`、`--num-images` |
| `3_export/quantize.sh` | Docker 内 `hb_mapper makertbin` PTQ 量化（自动切到项目根，容器内 `/data`=项目根）；**onnx 路径、校准目录、输出前缀都从配置文件里读**，detect/pose 通用 | `[config.yaml]`（默认 `../configs/yolo11n_config.yaml`；pose 用 `../configs/gate_kpt_config.yaml`） |

> **2026-09-24：本目录已按 git 回退到 09-15（`54e09b4`）的状态，就是上面这 4 个脚本。**
> 09-19/09-23 加的 `test_decode_parity.py`、`check_quant.sh`、`check_quant_cos.py` 与
> 当天新增的解码级校验脚本**已删除**（它们不在"一周前的一般量化流程"里）。
> **思路与实测数字留档**在 [`../_archive/quant/ideas_void_20260924/QUANT_TUNING_20260924.md`](../../_archive/quant/ideas_void_20260924/QUANT_TUNING_20260924.md)
> （含校验标准、三轮结果、根因、以及"怎么把这三个校验工具重建出来"的要点）。需要时一条命令取回：
>
> ```bash
> git checkout 3a3e044 -- scripts/3_export/check_quant_cos.py     # 余弦校验（09-23 版，支持 --n/--thresh）
> git checkout 3a3e044 -- scripts/3_export/check_quant.sh         # hb_mapper checker（只看节点放置/阈值）
> git checkout 309da79 -- scripts/3_export/test_decode_parity.py  # 板端解码/预处理对拍（改解码后必跑）
> ```
>
> ⚠️ **核对（2026-09-26）**：上面的还原命令仍然有效（`git ls-tree` 确认 `3a3e044` 里有
> `check_quant.sh` / `check_quant_cos.py` / `test_decode_parity.py`，`309da79` 里有 `test_decode_parity.py`）。
> 「本目录 = `54e09b4` 状态」的说法**只差一行**：`3_export/prepare_calibration.py` 的 docstring 示例路径
> 由 `data/AUV_1/AUV_data_1` 改成了 `data/datasets/AUV_1_AUV.yolov11`（`git diff 54e09b4 -- scripts/3_export/` 仅此一处），
> 其余三个脚本与 `54e09b4` 逐字节一致。

> **校准集按任务分开放**：detect → `../calibration_data_detect/`（检测训练集图片），
> pose → `../calibration_data/`。两者分布不同，**混用会静默掉点**；
> `3_export/quantize.sh` 会按配置里的 `cal_data_dir` 校验对应目录，缺了会直接报错而不是拿错数据。
> （2026-09-26 核实配置：detect 用 `../configs/yolo11n_config.yaml` 的 `/data/calibration_data_detect`，
> pose 用 `../configs/gate_kpt_config.yaml` 的 `/data/calibration_data`；这两个目录**当前都不在磁盘上**，
> 属量化前按需生成的产物。）
>
> pose 校准集的口径（2026-09-23 起）：**用与训练/部署同链路的 640 图**（当前 = D + LUT 白平衡、无 CLAHE），
> 从 `../data/derived/AUV_5_selected_3000_Dwb/` 抽（跨水质、清水偏多），**不要**再用旧的 `PNP.kpt4` 原图
> （那是 A-old 链路，分布对不上）。张数 300、`calibration_type: default`（2026-09-24 起固定用 default：
> 同日试过的 `max` / `mix` 都更差，记录见 `../../_archive/quant/ideas_void_20260924/QUANT_TUNING_20260924.md`）。
>
> 校准数据建议取**训练集图片**（与推理同分布）：`3_export/prepare_calibration.py --coco-path <数据集>/train/images --output-dir <对应目录>`。
> 注意它对非方形图会 letterbox 补灰边，而板端推理是直接 squish 到 640×640；
> **图片本身已是 640×640 时 letterbox 是 no-op**，所以用 `1_prepare/prepare_frames.py` 产出的图最稳
> （`3_export/prepare_calibration.py` 的 `--output-dir` 默认 `<项目根>/calibration_data`、`--num-images` 默认 100，
> 见 `--help`；detect 那一路记得显式 `--output-dir calibration_data_detect`）。

```bash
# detect：6 输出 → yolo11n_detect_*.bin
python scripts/3_export/modify_ultralytics.py --task detect
python scripts/3_export/export_onnx.py
python scripts/3_export/prepare_calibration.py --coco-path <检测数据集>/train/images \
    --num-images 300 --output-dir calibration_data_detect
./scripts/3_export/quantize.sh

# pose：9 输出（4 点 → kpt 通道 12）→ gate_kpt_*.bin
python scripts/3_export/modify_ultralytics.py --task pose
python scripts/3_export/export_onnx.py --task pose        # → weights/yolo11n-pose.onnx
python scripts/3_export/prepare_calibration.py \
    --coco-path data/datasets/AUV_4_PNP.kpt4.yolov8/train/images --num-images 200   # 默认写 calibration_data/
./scripts/3_export/quantize.sh configs/gate_kpt_config.yaml
```

> pose 侧补充（2026-09-12 已优化）：`hb_perf` 显示 **单个 BPU 子图、6.83 ms（≈146 FPS）**，
> 与 detect（6.64 ms）基本持平；仅 9 个很小的 Concat/Reshape/Transpose 落在 CPU。
> 关键改动：`kpt` 取通道用 4D 步长切片（`kpt[:, 0::3]`）而非 `view(...,5D)` + 整数索引
> ——后者导出 rank-5 Gather，会让模型被切成 7 个子图。**不要**改成"先 permute 到 NHWC
> 再算"：layout 变换会在 DDR 实体化（实测 18→48 MB，延迟 6.8→11.5 ms）。

> **⚠️ 已知（2026-09-17）：C2PSA 的 Reshape 烘死了 `batch=1`，PTQ 校准只能逐张跑。**
> 节点 `/model.10/m/m.0/attn/Reshape` 的 shape 常量是 `[1,2,128,400]`，来自 ultralytics
> `Attention.forward` 的 `qkv.view(B, ...)`（`B` 导出时被 trace 成常量）。`hb_mapper` 想用
> `batch=8` 校准会被打回：`Input shape:{8,256,20,20}, requested shape:{1,2,128,400}`，
> 然后 `Reset batch_size=1 and execute forward again...`。
>
> **这是良性的**：自动回退后校准结果依然正确，只是慢（200 张一轮约 55 分钟）。理论上可把 6 处
> Reshape 的 shape 常量首维 `1` 改成 `0`（ONNX 语义=沿用输入该维）放开 batch，改完 batch=1 输出与
> 原模型**逐位一致**；**但实测这对速度没有帮助**——裸 ONNX 单线程前向 73 ms/张，batch=8 反而
> **85 ms/张**（小模型+单核，批处理只增内存流量），而校准的 5.5 s→22 s/张 是花在 hb_mapper 给每个
> 节点插入 HzCalibration 后**逐样本累计阈值统计**上，与 batch 无关。
> **要提速请减 `--num-images` 或收窄 `calibration_type` 搜索空间；不要动 Reshape。**
>
> **验证解码约定改动**（换导出方式/改 head 补丁后必跑）：
> `python scripts/3_export/test_decode_parity.py --n 30`（需先恢复原版 head）
> —— 该脚本 2026-09-24 随回退删除，需先 `git checkout 309da79 -- scripts/3_export/test_decode_parity.py` 取回。

输出契约（detect 6 张量 / pose 9 张量、pose 关键点坐标语义、板端解码对应关系）见
根 [README.md 的“输出契约”](../README.md#输出契约与板端解码对齐勿随意改)。
