# tools — 工程工具（**按用途分三类**）

这里放**离线**工具：部署/对齐板端、上板前自检、下水后的日志判读与标定。
它们**不参与运行时**（`main.py` / 各 task 不会 import 这里的东西）。

```bash
# 按类跑：三个目录都可以独立工作（每个脚本自己定位工程根，在哪个 cwd 跑都行）
bash   tools/deploy/deploy_to_board.sh          # 部署（本机 → 板端）
python3 tools/check/vision/check_kpt_decode.py         # 自检（跑一次看结论）
python3 tools/analyze/log/analyze_task_log.py log/run_gate.jsonl   # 判读（吃日志/dump）
```

```
tools/
├── README.md                     # 本文件（分类索引 + 用法 + 三个部署陷阱）
├── deploy/                       # ① 部署与一致性（本机驱动；**除 archive_baks.sh 外都不传板端**）
│   ├── deploy_to_board.sh        #   烧录：按清单批量增量同步 + 板端自检（编译/pytest/配置快照）
│   ├── check_board_parity.sh     #   对齐：本地 vs 清单 vs 板端逐文件 md5
│   ├── board_parity.md5          #   清单：源文件 md5（**路径 = 部署契约**）
│   ├── archive_baks.sh           #   归档散落的 *.bak* → <工程根>/bak/（**随部署上板**，板端也用它）
│   └── tidy_board_bak.sh         #   整理**板端** bak/：按时间归档压缩 + 留几个可回滚快照
│
├── check/                        # ② 自检（上板前 / 离线跑一次看结论；只读，不动船）
│   ├── vision/                   #   视觉
│   │   ├── check_kpt_decode.py   #     keypoint 解码约定自检（-0.5 陷阱、通道布局）——**上板前必跑**
│   │   ├── check_gate_pose.py    #     PnP 位姿自检：为什么位姿被拒（角点/候选 RMS/tz）
│   │   └── auv_kpt_meter.py      #     只读相机、不碰串口：实时看门框 + 4 角点 + 置信度
│   ├── boat/                     #   船体与执行器
│   │   ├── check_dof_sign.py     #     **DOF 通道物理符号自检**：用**视觉**当独立真值定
│   │   │                         #     `dof_map.{yaw,sway}.sign`；**必须显式 --go 才发推力**，退出前补发中性帧
│   │   └── check_hdg_lockup.py   #     正航向"转不动 / 无效自转"的仿真复现（`hdg.max_turns` 的由来）
│   └── pipeline/                 #   工程链路
│       ├── check_pipeline_identity.py  # 训练/推理**同域**自证（去畸变↔回投↔解码同一条链，勿改图像链路）
│       ├── check_domain.py       #     本仓是哪个"代次"（D 链序 + wb 增益 + CLAHE + 权重 md5 指纹）
│       └── check_paths.py        #     路径/配置落点自检
│
└── analyze/                      # ③ 离线判读与标定（吃日志/dump/录制数据，出结论与参数建议）
    ├── log/                      #   判读日志
    │   ├── analyze_task_log.py   #     逐帧日志判读 ①~⑧（相位/出口/水平通道能动力/进 THROUGH 偏了多少）
    │   ├── feature_coverage.py   #     **这轮哪些功能没被用到**（相位/子状态/档位/动作 + ✔✘ 清单）
    │   ├── analyze_kpt_dump.py   #     preview --dump 的角点可得率 / PnP 命中率（标定 conf_thr/vis_thr/reproj_px）
    │   ├── analyze_heading.py    #     PnP 安装角（bias）与航向噪声底（静态 dump）
    │   └── analyze_pnp_center.py #     位姿中心 vs bbox 中心的偏差证据
    └── calib/                    #   标定
        ├── label_corners.py      #     手工标 4 角 → 同格式 dump（检测器不响应时的岸上路线）
        ├── ruler_calib.py        #     靶子读数 → 等效焦距 fx/fy 与距离口径 c（--gate 与门框联立）
        ├── tape_ticks.py         #     刻度周期法：卷尺自带 1cm 刻度测像素比例
        └── pnp_calib.py          #     **PnP 位姿/深度标定**：真值 dump → 误差表 + 门框尺寸反演 + cfg 建议
```

> `tools/` 现在是一个 **Python 包**（`tools/__init__.py` + 三个子目录的 `__init__.py`），
> 这样 `tests/tooling/test_pnp_calib.py` 可以 `from tools.analyze import pnp_calib` 直接测工具。
> **那几个 `__init__.py` 不是装饰品，别删。**

**怎么选**：

| 你想干什么 | 用哪个 |
|---|---|
| 把代码同步到板子 / 看会传哪些文件 | `deploy/deploy_to_board.sh`（`--dry-run` 先看） |
| 确认"本地 == 清单 == 板端" | `deploy/check_board_parity.sh`（加 `--board` 才连板子） |
| 把散落的 `*.bak*` 收进 `bak/` | `deploy/archive_baks.sh` |
| 板端 `bak/` 堆太多 | `deploy/tidy_board_bak.sh` |
| 权重要上板了，怕解码约定不对 | `check/check_kpt_decode.py` |
| 位姿老是解不出来 | `check/check_gate_pose.py` |
| 怀疑"该右转却左转 / 该右移却左移"（DOF 或遥测符号反了） | `check/check_dof_sign.py`（板端、水里，**加 `--go` 才发推力**） |
| 换了相机/畸变/分辨率，怕训练与推理不同域 | `check/check_pipeline_identity.py` |
| 下水回来复盘这一轮哪里不对 | `analyze/analyze_task_log.py log/<tag>gate.jsonl` |
| 想知道"这轮有哪些功能根本没被触发" | `analyze/feature_coverage.py log/<tag>gate.jsonl` |
| 深度/位姿准不准、`frame_w` 该改多少 | `analyze/pnp_calib.py`（配 `doc/记录/实验待测-runbook.md` 的流程） |

> 判读命令（本地，用拉回来的 jsonl）：
> ```bash
> python3 tools/analyze/log/analyze_task_log.py log/<tag>gate.jsonl     # 相位/出口/通道
> python3 tools/analyze/log/feature_coverage.py log/<tag>gate.jsonl     # 哪些功能没用上
> python3 tools/analyze/calib/pnp_calib.py --report r.md log/pnp_*/pnp_z*.jsonl  # PnP/深度标定（**文件名即真值**）
> ```
>
> PnP/深度标定的完整实验流程（怎么摆、录多久、判据、记录表、报告模板）见
> `doc/记录/实验待测-runbook.md`；`pnp_calib.py` 只负责「把 dump + 真值变成误差表和参数建议」。

> **过门**：当前在用的是 `gate/` **扁平 v1.2 版**（`gate/` 下 1 个总调度 + `gate/percept/`、`gate/motion/` 若干模块），
> `kpt_memory` 是**可选开关、2026-09-18 起默认关闭**
> （`cfg/vision.yaml → vision.gate.kpt_mem.enable: false`；开启用 `enable: true` 或 `AUV_GATE_KPT_MEM=1`，
> 优先级 `AUV_GATE_KPT_MEM` > `enable` > 兜底 `false`。板端曾因拼写错 `flase` 被 `bool('flase')=True`
> 而**实际一直开着**——所以开关值本身也要核对）。
> v1.3/v1.4 分层版、质量分/线索/帧守卫、过门的 yaw 居中通道与 SEARCH 旋转脉冲等已确定不需要并删除，
> 本目录不再提供任何版本切换脚本。

> SSH 包装器（含密码的 `askpass`）**不进仓库**，默认在 `/home/ansty/RDKX5/`：
> `.ssh_x5.sh` / `.scp_x5.sh` / `.tmp_askpass.sh`（密码文件 chmod 600）；`.ssh_x5_stream.sh`
> 是"保留 stdin"的版本（tar 管道要用），部署脚本缺了会自动生成。
> 全部可用环境变量覆盖：`AUV_SSH` / `AUV_SCP` / `AUV_STREAM` / `AUV_HELP_DIR` /
> `AUV_BOARD_HOST` / `AUV_BOARD_DIR`。缺包装器时脚本会打印一次性创建命令。

---

## 1. 烧录/更新板端 —— `deploy/deploy_to_board.sh`

```bash
bash tools/deploy/deploy_to_board.sh              # 全流程：同步 + 删除废弃 + 板端编译 + 全量 pytest + 终检
bash tools/deploy/deploy_to_board.sh --no-test    # 跳过板端 pytest（秒级；日常改代码用这个）
bash tools/deploy/deploy_to_board.sh --dry-run    # 只报告"将上传/删除哪些文件"，不动板端
```

做四件事，每一步都打印**累计耗时**：

1. **差异**：一次 SSH 批量取板端 md5 → 与本地比 → 得出"新/改/已一致"清单（不是一文件一次 SSH）；
2. **备份+上传**：变化的文件在板端先备份到 `bak/deploy_<时间戳>/`（保留目录结构），
   然后打成一个 tar 单包上传；
3. **废弃清理**：脚本内 `DELETED` 列表里的旧文件备份后删除（含 `__pycache__` 清理）；
4. **板端自检**：`py_compile` 关键模块 +（默认）全量 `pytest tests/ -q` + 配置快照
   （ball/gate/motion(共用)/kpt_mem(含开关)/模型/`SIM_MODE`/depth_guard/目标色），最后跑一次对齐检查并刷新清单。

**退出码**：`0` = 同步完成且板端 pytest 全绿（`PYTEST-OK`）；`1` = 已同步但板端 pytest **未全绿**
（板端打印 `PYTEST-FAIL`，末尾给 ⚠️ 提示）。

⚠️ **板端"多出来的旧文件"不会自动消失**：本地删掉/合并/搬目录过的文件会留在板上，
全量 pytest 因 import 已删除模块而收集报错（或收集到新旧两份同名模块 → import mismatch）。
这类路径一律**补进 `DELETED` 列表**（先备份到 `bak/deploy_<时间戳>/` 再删），板端才能重新与清单逐字节一致。

⚠️ **清单里的路径 = 部署契约**：清单记录的是**路径**，所以"改名 / 搬到子目录"不是免费操作，要成对做：

1. 新路径补进清单（无方括号，等板端核对后再补方括号）；
2. 旧路径加进 `DELETED`，并同步删掉清单里的旧条目；
3. 反过来也要检查：**`DELETED` 里绝不能留着一个现在还在用的路径** —— 否则部署会把刚上传的文件又删掉。

（本工程已经踩过两次：一次是分层版用例与活文件同名 `tests/tasks/gate/test_gate_flow.py`；一次是 2026-09-21
把 `tools/*`、`tests/*` 分到子目录，旧扁平路径全部进了 `DELETED`、新路径进清单。）

实测（板端已最新）：`--no-test` **约 6 s**；带 pytest 约 15 s（其中大部分是 pytest 本身）。

**幂等**：重复跑只会跳过已一致的文件，可以随时执行。

## 2. 对齐核对 —— `deploy/check_board_parity.sh`

```bash
bash tools/deploy/check_board_parity.sh                                  # 本地 vs 清单（毫秒，不用板子）
AUV_SSH=/home/ansty/RDKX5/.ssh_x5.sh bash tools/deploy/check_board_parity.sh --board
AUV_SSH=... bash tools/deploy/check_board_parity.sh --board --write       # 顺带把清单刷成当前实测
AUV_SSH=... bash tools/deploy/check_board_parity.sh --board --only-verified
AUV_SSH=... bash tools/deploy/check_board_parity.sh --board --slow        # 一文件一次 SSH（SSH 不稳时）
```

- 清单格式：`[ md5  文件 ]` = **已与板端逐字节核对一致**；无方括号 = 仅本地记录；
- `--board` 用 **1 次 SSH 批量取 md5** → 实测 **约 1.5 s**（旧实现逐文件 SSH 约 90 s）；
- 两个方向都要绿才算"板端就是这份代码"：
  `本地 == 清单`（本地没被偷偷改过）＋`板端 == 清单`（板上装的正是这份）；
- **搬过目录的文件即使内容没变，也要先去掉方括号**：板端那边还是旧路径，
  这条"已核对"就不再成立（`--write` 会按新路径重新核对并补回方括号）。

## 3. 备份归档 —— `deploy/archive_baks.sh`

```bash
bash tools/deploy/archive_baks.sh --dry-run    # 先看会移动哪些
bash tools/deploy/archive_baks.sh              # 归档：*.bak* → <工程根>/bak/（cfg/ 的进 bak/cfg/）
bash tools/deploy/archive_baks.sh --list       # 看 bak/ 现有内容
```
（脚本在 `tools/deploy/` 下，会自动以**上两级**为工程根，`bak/` 仍在工程根。
它是本目录里**唯一随部署上板**的脚本：板端路径 `<板端>/tools/deploy/archive_baks.sh`。）

---

## 4. 板端备份区整理 —— `deploy/tidy_board_bak.sh`

板端每次同步前，`deploy_to_board.sh` 都会把"将被覆盖/删除"的文件快照到 `<板端>/bak/deploy_<时间戳>/`；
跑久了 `bak/` 会堆满平铺的 `deploy_*` 目录和 `*.bak_*` 文件。这个脚本把它们**按时间归档压缩**：

```bash
bash tools/deploy/tidy_board_bak.sh --dry-run     # 先看它要做什么（不动板端）
bash tools/deploy/tidy_board_bak.sh               # 执行
KEEP=5 BIG_FILES=20 bash tools/deploy/tidy_board_bak.sh   # 多留几个回滚点
```

整理后的板端布局（脚本会顺手生成 `<板端>/bak/README.md` 索引）：

```
<板端>/bak/
├── README.md                      # 索引：每个备份有什么、怎么回滚（自动生成）
├── rollback/deploy_MMDD_HHMMSS/   # 值得留着的**原样**快照：cp -rp 回去即可回滚
│                                  #   = 最近 KEEP 个（默认 3） ∪ 文件数 ≥ BIG_FILES（默认 10）的大改动
└── archive/
    ├── deploy_YYYY-MM-DD.tar.gz   # 该天其余的 deploy_* 快照（一天一个包）
    ├── legacy_YYYY-MM-DD.tar.gz   # 散落的 *.bak_* + bak/cfg（archive_baks.sh 的产物）
    └── *.tgz                      # 板端原有归档，原样移入
```

**原则**：① **先打包再删**——除"空目录"外不丢任何内容（要彻底腾空间就手工删 `archive/`）；
② 日期用**本地时间**（板端 RTC 常年不准，`deploy_*` 目录名本身就是本地时间生成的）；
③ 幂等，随时可重跑。首次整理实测：板端 `bak/` **2.9 M / 210 个文件 → 1.3 M / 48 个文件**，
回滚点保留 5 个（含"本次大同步前"的 24 文件快照）。

---

## 5. 典型工作流

```bash
# 改完代码（本地）
python3 -m pytest tests/ -q                    # 本地全绿（216 例，约 15s）
python3 -m pytest tests/_bite_probe.py -q      # 改过 cfg 相关逻辑时：确认用例还会"咬"
bash tools/deploy/check_board_parity.sh        # 先确认"本地 == 清单"（并看有没有"新文件未入清单"）
bash tools/deploy/deploy_to_board.sh --dry-run # 看它打算传/删哪些（**新文件不在清单里就不会被传**）
bash tools/deploy/deploy_to_board.sh --no-test # 秒级同步（板端已最新时只做核对）

# 需要板端证据时
bash tools/deploy/deploy_to_board.sh           # 含板端全量 pytest
bash tools/deploy/tidy_board_bak.sh            # 可选：板端备份堆多了就整理一次
```

## 6. 板端目录对照

| 本地 | 板端 |
|---|---|
| `auv_vision/auv_vision/`（本目录，工程根） | `/home/sunrise/Desktop/AUV_New/`（**在用副本**，`deploy_to_board.sh` 的默认目标） |
| `tools/deploy/archive_baks.sh` | `/home/sunrise/Desktop/AUV_New/tools/deploy/archive_baks.sh`（板端维护用，随部署同步） |
| `tools/deploy/{deploy_to_board,check_board_parity,tidy_board_bak}.sh`、`tools/deploy/board_parity.md5`、`board_forks.txt` | 驱动类**不传**（纯本地驱动） |
| `tools/{check,analyze}/*.py`、`tools/README.md` | 随部署同步（板端可直接跑判读/自检） |
| `tests/{platform,tasks,tooling}/*.py` | 随部署同步（板端能跑同一套 pytest） |
| `cfg/*.yaml`（`vision.yaml` / `comm.yaml` / `front_camera.yaml`） | ⚠️ **默认整份同步**（在清单里，`deploy` 会覆盖板端）——但**下水现场改过板端 cfg 时，这条会把它冲掉**，见下面「陷阱 1」 |
| `models/*.bin` | `/home/sunrise/Desktop/AUV_New/models/`（权重单独管理，**不随代码同步**，见下） |
| `<板端>/bak/` | 板上备份区：`rollback/`（可直接回滚）+ `archive/`（按时间归档），由 `tidy_board_bak.sh` 维护 |

> **A 域（板端旧工程 `/home/sunrise/AUV/`，09-17 的树）已于 2026-10-01 退役**：
> 用户定「现在只用 **D+wb** 域」⇒ 它的同步通道与 canonical cfg 整体归档到
> `bak/retired/Adomain_20261001/`（含 `sync_Adomain_repo.sh`、`Adomain_cfg/`、`board_forks_Adomain.txt`、
> A 域老内参 `front_camera_AUV1_water_fx782.yaml`），`deploy` / `check_board_parity` 都不再管它。
> 当年手工运维它踩过的两个坑**仍然通用**（改任何板端 cfg 都要过一遍）：
> ① 旧代码 `build_gate_backend` 用 `os.path.exists(path)` 直接判存在、**不解析相对路径** ⇒ 手改 cfg 要写绝对路径；
> ② 文件名尾部多一个点（`….bin.`）会让 cfg 指的名字「缺失」、gate 被**静默跳过** ⇒ 换权重后 `ls -lab models/` 对一遍。

### 6.1 权重分发（`models/*.bin` 不走部署脚本）

`models/` 在清单 `find` 的排除项里（`-not -path './models/*'`），所以 **`.bin` 永远靠手工 scp**：

```bash
/home/ansty/RDKX5/.scp_x5.sh <本地.bin> sunrise@192.168.137.10:/home/sunrise/Desktop/AUV_New/models/
# 传完必须两端核对 md5（板端时钟不可靠，别看 mtime）
md5sum <本地.bin>; $AUV_SSH "md5sum /home/sunrise/Desktop/AUV_New/models/<同名>.bin"
```

换门权重时要**两处一起动**，缺一个就是"文件在、配置还指旧的"：

1. 传 `.bin` 到目标目录 `models/`；
2. 改 `cfg/vision.yaml` 的 `model.task_models.gate.path`（在用副本用**相对**路径 `models/<名字>.bin`，
   旧副本 `/home/sunrise/AUV` 用**绝对**路径）；
3. 改完在板端实跑一次后端加载 + 单帧推理（别只看文件在不在）：

```bash
$AUV_SSH "cd /home/sunrise/Desktop/AUV_New && python3 - <<'PY'
import sys; sys.path.insert(0, '.')
import cv2
from gate.gate_detector import build_gate_backend
b = build_gate_backend()
print('backend =', type(b).__name__ if b is not None else None)   # None = cfg 没指到权重
d = b.detect(cv2.imread('log/frames/pv_000024.jpg'))              # 有门的样本帧
print('det =', d)          # 角点应是 TL→TR→BR→BL 的四边形，不交叉
PY"
```

**2026-09-27 实测记录（旧副本 `/home/sunrise/AUV` 换门权重）—— ⚠️ A 域已于 2026-10-01 退役，以下仅存史**：把
`RDKX5-YOLOv11n-/output/weights/gate_kpt_auv4_bayese_640x640_nv12.bin`
（md5 `d38b803e5823b35696f5711ecd1c30b4`，3,942,297 B）scp 到该目录 `models/`，
并把 `cfg/vision.yaml` 的 `task_models.gate.path` 指到它（绝对路径）；
原 cfg 备份在板端 `bak/model_swap_0927_161157/cfg/vision.yaml`。
实测：`GateKeypointBackend` 加载成功，`log/qframes_B/q_00040.jpg` 检出 1 个门、4 角点序正确。
⚠️ 该旧副本的门此前**一直是被跳过的**（见上一节 ②），所以这次换权重会把 gate 从"跳过"变成"真跑"。

## 7. 三个部署陷阱（都真实发生过，部署前逐条过一遍）

### 陷阱 1：`deploy_to_board.sh` 会**整份覆盖板端 `cfg/*.yaml`**
`cfg/comm.yaml` / `cfg/vision.yaml` / `cfg/front_camera.yaml` **都在清单里**（`[ md5 path ]`），
`deploy` 的语义就是"清单里的文件 = 本地这份"，所以**板端被现场改过的 cfg 会被本地覆盖**。

已经因此出过事：`depth_guard.min_depth_m` 板端是现场定的 **0.55**，本地曾长期写 0.3，
两次整份推送把它冲掉。**安全做法**（改任何 cfg 前）：

```bash
export AUV_SSH=/home/ansty/RDKX5/.ssh_x5.sh
$AUV_SSH "md5sum /home/sunrise/AUV/cfg/comm.yaml /home/sunrise/AUV/cfg/vision.yaml"  # 先看是否一致
b=$(mktemp); $AUV_SSH "cat /home/sunrise/AUV/cfg/comm.yaml" > "$b"
diff -u cfg/comm.yaml "$b"        # 逐键看清差异 → 把板端现场值合并回本地，再部署
```

- **要保留板端现场值 → 先把它合并回本地**（本地是唯一参数源，两边最终必须一致，
  否则 `check_board_parity.sh` 永远不绿）；
- 覆盖前板端旧文件会自动备份到 `bak/deploy_<时间戳>/`，回滚用 `cp -rp` 即可。

> **分叉清单：`tools/deploy/board_forks.txt`**（2026-09-27 起）—— 声明"本地≠板端是**故意**的"文件。
> `deploy_to_board.sh` 命中就**不上传**（打一行 `[skip]`），`check_board_parity.sh` 命中就**不算不一致**
> （打一行 `[分叉]`）。**两个脚本读同一个文件**：只在 deploy 里加、这边不知道，`--board` 就会永远报
> `[板端不同]` 并 exit 1 —— 检查长期红着等于没有检查。
> 曾经的两条（`base/cfg/settings.py` 的 SIM_MODE、`cfg/comm.yaml` 的台架分叉）在 2026-09-26
> "全部以本地为准"时清空；**2026-09-27 把本地 `SIM_MODE=True` 推上板端、冲掉真驱动默认值**，
> 于是 `base/cfg/settings.py` 重新入列（板端默认 `False` 真发串口，台架要打印用 `AUV_SIM_MODE=1`）。
> ⚠️ 分叉粒度是**整个文件**：入列后本地对该文件的改动**不再上板端**，要改必须在板端手工同步。
> 现状：除此 1 个文件外，**板端全量跟本地一致**；上表"先 diff 再合并"的流程仍是改 cfg 前的好习惯。

### 陷阱 2：**新文件不在清单里 = 永远传不上板**
`deploy_to_board.sh` 只遍历 `board_parity.md5` 里的条目；清单由
`check_board_parity.sh --board --write` 用 `find` 重新生成（会收录新增的 `*.py/*.sh/*.md/*.yaml`）。
所以**新增源文件/新目录后必须先刷清单**，否则板端会出现"新 `gate/gate_task.py` + 缺 `gate/motion/hdg.py`"
这种半套状态 → 板端一 import 就崩。

```bash
bash tools/deploy/check_board_parity.sh                     # 会报「本地有、清单没有」的新文件
AUV_SSH=... bash tools/deploy/check_board_parity.sh --board --write   # 板端可达时才刷清单
# 板端不可达时：手工把新文件以**无方括号**形式追加（= 仅本地记录，不算板端实测证据）
```

### 陷阱 3：本地删掉的文件要同时**从清单移除 + 进 `DELETED`**
清单里仍留着本地已删除（或已搬走）的路径 → `deploy` 会拿一个不存在的文件去比对；而板端的旧副本
不会自己消失（`deploy` 只传不清）。两步都要做：

1. 从 `tools/deploy/board_parity.md5` 删掉那一行；
2. 把相对路径加进 `deploy_to_board.sh` 的 `DELETED=( … )`（部署时先备份再删板端旧文件）。
   `DELETED` 里已有的：v1.3/v1.4 分层 gate 目录、`task1_2/turn_deg.py`（已迁到 `common/motion/turn_deg.py`）、
   2026-09-20 合并前的旧用例名（`test_uart.py`/`test_gate_dash.py`/`test_gate_decode.py`/
   `test_heading_align.py`/`test_turn_deg.py`/`test_preprocess_rt.py` 等）、以及 2026-09-21 分类重整前的
   **旧扁平路径**（`tools/analyze_*.py`、`tools/check_*.py`、`tools/pnp_calib.py`、`tools/deploy/archive_baks.sh`、
   `tests/test_*.py`）。⚠️ 加旧路径的同时，**确认没有同名活文件被误列**。

> 提醒：`board_parity.md5` 的**方括号表示"已与板端逐字节核对过"**。刷清单时必须板端可达；
> 板端不可达时只能追加无方括号条目 —— 别为了让 `check` 变绿而手工补方括号（那是伪造验证）。
