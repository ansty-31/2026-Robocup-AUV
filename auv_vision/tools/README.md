# tools — 部署（烧录）与"环境/一致性"核对工具

这里放**本地驱动**的工程工具：把代码同步到 RDK X5、核对两边是否一致、归档 `*.bak*`。
它们**不参与运行时**（`main.py` 不会 import 这里的东西）。

```
tools/
├── deploy_to_board.sh        # 烧录：按清单批量增量同步到板端 + 板端自检（编译/pytest/配置快照）
├── check_board_parity.sh     # 对齐：本地 vs 清单 vs 板端逐文件 md5
├── board_parity.md5          # 清单：源文件的 md5（+ 哪些仍与板端实测核对一致）
├── tidy_board_bak.sh         # 整理**板端** bak/：按时间归档压缩 + 保留几个可直接回滚的快照
├── archive_baks.sh           # 归档**板端/本地**工程里散落的 *.bak* → <工程根>/bak/
├── .gitignore                # 不提交 __pycache__ / 含密码的 SSH 包装器
└── README.md                 # 本文件
```

> **过门**：当前在用的是 `gate/` **扁平 v1.2 版**（当前定版），`kpt_memory` 是**可选开关、默认开启**
> （`cfg/vision.yaml → vision.gate.kpt_mem.enable: true`；关掉用 `enable: false` 或 `AUV_GATE_KPT_MEM=0`，
> 优先级 `AUV_GATE_KPT_MEM` > `enable` > 兜底 `false`）。v1.3/v1.4 分层版、质量分/线索/帧守卫等
> 已确定不需要并删除，本目录不再提供任何版本切换脚本。

> SSH 包装器（含密码的 `askpass`）**不进仓库**，默认在 `/home/ansty/RDKX5/`：
> `.ssh_x5.sh` / `.scp_x5.sh` / `.tmp_askpass.sh`（密码文件 chmod 600）；`.ssh_x5_stream.sh`
> 是"保留 stdin"的版本（tar 管道要用），部署脚本缺了会自动生成。
> 全部可用环境变量覆盖：`AUV_SSH` / `AUV_SCP` / `AUV_STREAM` / `AUV_HELP_DIR` /
> `AUV_BOARD_HOST` / `AUV_BOARD_DIR`。缺包装器时脚本会打印一次性创建命令。

---

## 1. 烧录/更新板端 —— `deploy_to_board.sh`

```bash
bash tools/deploy_to_board.sh              # 全流程：同步 + 删除废弃 + 板端编译 + 全量 pytest + 终检
bash tools/deploy_to_board.sh --no-test    # 跳过板端 pytest（秒级；日常改代码用这个）
bash tools/deploy_to_board.sh --dry-run    # 只报告"将上传/删除哪些文件"，不动板端
```

做四件事，每一步都打印**累计耗时**：

1. **差异**：一次 SSH 批量取板端 md5 → 与本地比 → 得出"新/改/已一致"清单（不是一文件一次 SSH）；
2. **备份+上传**：变化的文件在板端先备份到 `bak/deploy_<时间戳>/`（保留目录结构），
   然后打成一个 tar 单包上传；
3. **废弃清理**：脚本内 `DELETED` 列表里的旧文件备份后删除（含 `__pycache__` 清理）；
4. **板端自检**：`py_compile` 关键模块 +（默认）全量 `pytest tests/ -q` + 配置快照
   （ball/gate/kpt_mem(含开关)/模型/`SIM_MODE`/depth_guard/目标色），最后跑一次对齐检查并刷新清单。

**退出码**：`0` = 同步完成且板端 pytest 全绿（`PYTEST-OK`）；`1` = 已同步但板端 pytest **未全绿**
（板端打印 `PYTEST-FAIL`，末尾给 ⚠️ 提示）。
⚠️ **板端"多出来的旧文件"不会自动消失**：本地删掉/合并过的用例（如旧的 `tests/test_uart.py`、
分层版 `test_gate_flow.py`、`test_return_*.py`）会留在板上，全量 pytest 因 import 已删除模块而
收集报错。这类文件一律**补进 `DELETED` 列表**（先备份到 `bak/deploy_<时间戳>/` 再删），
板端才能重新与清单逐字节一致。

实测（板端已最新）：`--no-test` **约 6 s**；带 pytest 约 15 s（其中大部分是 pytest 本身）。

**幂等**：重复跑只会跳过已一致的文件，可以随时执行。

## 2. 对齐核对 —— `check_board_parity.sh`

```bash
bash tools/check_board_parity.sh                                  # 本地 vs 清单（毫秒，不用板子）
AUV_SSH=/home/ansty/RDKX5/.ssh_x5.sh bash tools/check_board_parity.sh --board
AUV_SSH=... bash tools/check_board_parity.sh --board --write       # 顺带把清单刷成当前实测
AUV_SSH=... bash tools/check_board_parity.sh --board --only-verified
AUV_SSH=... bash tools/check_board_parity.sh --board --slow        # 一文件一次 SSH（SSH 不稳时）
```

- 清单格式：`[ md5  文件 ]` = **已与板端逐字节核对一致**；无方括号 = 仅本地记录；
- `--board` 用 **1 次 SSH 批量取 md5** → 实测 **约 1.5 s**（旧实现逐文件 SSH 约 90 s）；
- 两个方向都要绿才算"板端就是这份代码"：
  `本地 == 清单`（本地没被偷偷改过）＋`板端 == 清单`（板上装的正是这份）。

## 3. 备份归档 —— `archive_baks.sh`

```bash
bash tools/archive_baks.sh --dry-run    # 先看会移动哪些
bash tools/archive_baks.sh              # 归档：*.bak* → <工程根>/bak/（cfg/ 的进 bak/cfg/）
bash tools/archive_baks.sh --list       # 看 bak/ 现有内容
```
（脚本在 `tools/` 下，会自动以**上一级**为工程根，`bak/` 仍在工程根。）

---

## 4. 板端备份区整理 —— `tidy_board_bak.sh`

板端每次同步前，`deploy_to_board.sh` 都会把"将被覆盖/删除"的文件快照到 `<板端>/bak/deploy_<时间戳>/`；
跑久了 `bak/` 会堆满平铺的 `deploy_*` 目录和 `*.bak_*` 文件。这个脚本把它们**按时间归档压缩**：

```bash
bash tools/tidy_board_bak.sh --dry-run     # 先看它要做什么（不动板端）
bash tools/tidy_board_bak.sh               # 执行
KEEP=5 BIG_FILES=20 bash tools/tidy_board_bak.sh   # 多留几个回滚点
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
python3 -m pytest tests/ -q                 # 本地全绿（44 例，0.5s）
bash tools/deploy_to_board.sh --no-test     # 秒级同步（板端已最新时只做核对）
bash tools/check_board_parity.sh            # 确认"本地 == 清单"

# 需要板端证据时
bash tools/deploy_to_board.sh               # 含板端全量 pytest
bash tools/tidy_board_bak.sh                # 可选：板端备份堆多了就整理一次
```

## 6. 板端目录对照

| 本地 | 板端 |
|---|---|
| `auv_vision/auv_vision/`（本目录，工程根） | `/home/sunrise/AUV/` |
| `tools/archive_baks.sh` | `/home/sunrise/AUV/tools/archive_baks.sh`（板端维护用，随部署同步） |
| `tools/{deploy_to_board,check_board_parity,tidy_board_bak}.sh`、`board_parity.md5` | 驱动类**不传**（纯本地驱动） |
| `cfg/*.yaml`（`vision.yaml` / `comm.yaml` / `front_camera.yaml`） | 全部**随部署整份同步**（本地是唯一参数源）；覆盖前板端旧文件备份到 `bak/deploy_<时间戳>/` |
| `models/*.bin` | `/home/sunrise/AUV/models/`（权重单独管理，不随代码同步） |
| `<板端>/bak/` | 板上备份区：`rollback/`（可直接回滚）+ `archive/`（按时间归档），由 `tidy_board_bak.sh` 维护 |

> **`cfg/vision.yaml` 现在也入清单、整份覆盖**（早期为了"板端 `target_color: red`"曾把它排除在同步之外，
> 但实测两边都是 `blue`，且差异只有一个 `gate.kpt_mem.sigma_floor_px`）。**下水前若要 `red`：改本地**，
> 别再手改板端 —— 手改会在下次 `deploy` 时被覆盖（有备份，但没必要绕这一圈）。
