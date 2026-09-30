# A 域（板端旧仓库 `/home/sunrise/AUV`）的 cfg —— canonical 副本

> 这是**板端旧仓库参数的唯一来源**。改 A 域的参数 = 改这里的文件 → 跑
> `bash tools/deploy/sync_Adomain_repo.sh`（它会把这三个文件推上去并自检）。
> **别直接改板端**：下次同步就被这里覆盖（覆盖前板端会备份到 `bak/deploy_<stamp>/`）。

## 1. 两套识别方案（域）

| | 旧仓库 `/home/sunrise/AUV`（A 域 = 原方案） | 在用副本 `~/Desktop/AUV_New`（D 域 = 新方案） |
|---|---|---|
| 链路 `image.chain` | **A**：`remap@720p → resize640 → enhance` | **D**：`resize640 → enhance → remap@640` |
| `image.clahe_clip` | **0.5** | 0 |
| 门权重 | `gate_kpt_auv4_bayese_640x640_nv12.bin`（`d38b803e…`） | `gate_kpt_stage1_g240_i16_…bin`（`ef52c18b…`） |
| 前视内参 | **老** `fx=782.5`（AUV_1 陆上单集拟合） | 新 `fx=1207.6`（09-23 水下重标） |
| 识别阈值 | **老**：`score_threshold 0.5` / `gate.det.conf 0.5` / `keypoint.conf_thr 0.7` | 新：`0.6` / `0.6` / `0.8` |
| 几何标尺 | **老** `frame_w 0.70 / frame_h 0.50` | 新 `0.77 / 0.56`（卷尺实测外缘） |
| 过门阈值 | **老**：`timeout 180s` / `align 4/0.20/0.25` / `z.cross 0.7` / `near_lost 1.0 / 0.60` | 新：`300s` / `2/0.10/0.15` / `0.65` / `0.6 / 0.75` |

**代码两边相同**：过门的解码约束（`gate/percept/gate_decode.py` + `gate/percept/gate_postproc.py`）、
视觉信号处理（`gate/percept/kpt_memory.py` / `gate/percept/geometry.py`）、运动逻辑（`gate/motion/gate_task.py` /
`gate/motion/heading_align.py` / `common/motion/turn_deg.py`）都用最新那份。
唯一的例外是上表那些**老功能的老参数**，以及下面 §2 说明的两个新功能自己的键。

## 2. 旧仓库只保留老参数 + 两个新功能（2026-09-27 用户定）

A 域那套**老功能的参数一个都不改**（含内参与阈值），只把两个**新功能**连代码带参数放进来：

| 新功能 | 代码 | 它带来的新键（取**新**值） |
|---|---|---|
| ① **转角度**（额定转角闭环 + 正航向） | `common/motion/turn_deg.py`、`gate/motion/heading_align.py`、`gate/motion/gate_task.py` 的 `_turn_inner_loop`/`_hdg_*` | `gate.hdg.*`、`gate.through.require_align_deg`、`motion.turn_pid.{kd,out_max,div_deg,period}` |
| ② **解码后约束** | `gate/percept/gate_postproc.py` + `gate_decode.detect` 接入 | `gate.postproc.{edge_tol,min_edges,geom_check}` |

- `motion.turn_pid` 的 `kd` **必须 0**、`out_max` **0.30**：这是新转角度内层循环（20 Hz、
  误差按 `norm_deg` 归一）的硬要求 —— `kd=0.05` 的 D 项会超过 `out_max` ⇒ 出力每帧正负反转、
  原地极限环、永远进不了到达窗（09-27 仿真复现）。所以它**不能**沿用老的 `0.05 / 0.45`。
- 新代码还会读几个老 cfg 里没有的"新做法"参数，只能取新值（旧机制已不在代码里）：
  `gate.search.*`（旧是 `yaw/spin_s` 原地转搜，新是 `sweep_s/sway` 左右平移扫视）。
- 反过来，老 cfg 里这几个键**新代码不再读**，已按新结构落盘：
  `ball.pid` / `ball.approach_pid` → `motion.pid_heave` / `motion.pid_sway`（数值相同：
  `1.0/0.0/0.15/1.0/0.04` 与 `8.0/0.01/0.05/0.45/0.05`），`gate.surge.fast/slow` → `motion.surge_fast/slow`
  （0.35/0.15）。所以**老的球/速度调参并没有丢**，只是换了键名。
- `gate.coarse.{far_ratio,near_ratio,align_x,align_y}`、`gate.width.z_max`、
  `gate.surge.{creep,lost_backward,reacquire,through}`、`gate.z.{align_max,fast_max,slow_max,cross_confirm_frames,z_stale_ms}`、
  `gate.hold/reacquire/loiter`、`motion.loss_inertia_surge`：老值与**代码兜底值逐个相同**，
  所以最新 cfg 把它们删了（留代码默认）——A 域因此也不需要写回，行为与老的一致。

## 3. 怎么校验

```bash
# 在板端旧仓库里跑（会打印域指纹、A 序与参考实现逐像素对比、AUV_4 角点分布）
python3 tools/check/pipeline/check_domain.py \
  --equiv-ref tools/deploy/Adomain_cfg/preprocess_A_ref_20260917.py \
  --equiv-frames log/frames --limit 4 --frames log/frames --limit 4
```

- `preprocess_A_ref_20260917.py` 是**移植前旧仓库的 `common/vision/preprocess.py` 逐字节副本**
  （md5 `7eb0dae93bd4d927c0e7a800b447cdd7`），只用于证明"新代码的 A 链序 == 老 A 实现"。
- 域指纹必须自洽：`chain A` + `clahe 0.5` + 门权重 md5 前缀 `d38b803e`。
