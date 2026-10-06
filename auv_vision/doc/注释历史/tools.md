# 注释历史 · tools/** —— 部署/自检/判读/标定脚本


> 本文件由 `doc/注释历史.md`（4128 行）于 2026-10-05 按域拆出；**内容一字未改**，只搬了位置。
> 顶层索引与"旧路径怎么解析"见 [`doc/注释历史.md`](../注释历史.md)。
> 代码/配置里的注释只写"是什么 + 当前值"，注意事项、实测数字与历史沿革一律记在这里。

## tools/check/vision/auv_kpt_meter.py

### 为什么需要它
- 2026-09-22 · 跳过"量响应阶梯"直接在新（更亮更清）水质里跑任务，只看到"门不识别"，分不清是模型弱响应还是完全没响应 · 结论 → 该脚本定位为"换水/换光/换门之后第一件事"
### 判据阈值
- 2026-09-22 · 板端 `vision.gate.keypoint.conf_thr` 现值为 0.6（当时快照） · 结论 → 判据书写为"以 cfg 为准"，注释里不再固化数值
### 归档
- 2026-09-22 · 脚本原先只存在于 `/tmp/auv_kpt_meter.py`，重启或清理即丢失，而 runbook §B1 一直引用它 · 结论 → 归档进仓库 `tools/check/`

## tools/check/boat/check_dof_sign.py

### 转向环极性 σ 的定法
- 2026-09-23 · 立项该自检；2026-09-27 明确为**一次性标定工具** · 结论 → σ = 固件常量 × `dof_map.yaw.sign` × `telemetry.yaw_sign`，不探向、不现场测，量一次写进 cfg
- 2026-09-27 · 旧版靠 `turn_deg` 的 0.6 s 探向，那段开环本身就会白转十几度 · 结论 → 该探向实现已删除，别再改回开环探向

## tools/check/pipeline/check_domain.py

（本次未改动：模块 docstring 只描述两套识别方案的配对关系与三个子命令用法，属"说明"，无日期/实测叙述。）

## tools/check/vision/check_kpt_decode.py

### 逐 cell 判定的默认 conf
- 旧版（未记日期）· `--per-cell` 里写的是 `max(0.4, conf-0.1)`，把 `--conf 0.05` 悄悄变成 0.40，于是"没有 cell ≥0.40"被误读成"模型失效" · 结论 → 改为按传入值（默认 0.05，环境变量 `AUV_KPT_PERCELL_CONF`）

## tools/check/vision/check_gate_pose.py

### 兜底值
- 未记日期 · 代码里曾抄过 0.5 / 0.7 / 0.9 等 cfg 改前的历史字面量作为兜底 · 结论 → 兜底值必须等于当前 cfg

## tools/check/boat/check_hdg_lockup.py

### post_sway 窗口把 HDG 锁死（本脚本的由来）
- 2026-09-27 · 现场故障：转向结束挂上 600 ms 的 post_sway 窗口 → 视野里出现的是更远的门（或贴画面边缘、框被裁小的同一个门）→ 判成「不是我刚丢的门」→ 那一帧 `return`（不发居中指令、不起正航向）→ 窗口过期后这个判断仍每帧成立 ⇒ 居中 / 正航向 / 丢失处理全被跳过 ⇒ HDG 永久锁死（船在池子里乱转/乱漂，串口打印的 yaw 一直是中位） · 结论 → 每个消费该窗口的方法都必须有「过期 ⇒ 收手」路径（`check_expiry`）；`_post_sway_same_gate` 必须先用 z 做判据

## tools/check/pipeline/check_paths.py

### 路径类坑
- 2026-09-22 · 板端同时存在 `/home/sunrise/AUV`（旧副本）与 `/home/sunrise/Desktop/AUV_New`（在用工作副本），cfg 里写绝对路径 ⇒ 标定/权重指错一份就静默用了另一个文件，那份被删/被搬 ⇒ 相机初始化直接崩 · 结论 → cfg 里一律写仓库内相对路径，运行时由 `base.settings.resolve_path()` 按工程根解析（检查项 B/C）
- 2026-09-22 · 相机标定 yaml 前面有注释行、甚至空行时，板端 cv2 4.11 报 `Input file is invalid`（SystemError），而 `calibration_maps` 不兜底 ⇒ 崩 · 结论 → 首字节必须是 `%YAML:1.0`，注释只能放在 `---` 之后（检查项 D）

## tools/check/pipeline/check_pipeline_identity.py

（本次未改动：模块 docstring 只列三条校验与退出码；`_project_root` 的"别写死层级"是防回归结论。）

## tools/analyze/log/analyze_heading.py

### 立项动机
- 2026-09-18 · 用户提出：ALIGN 里 yaw 用 `dxn` 驱动，那是方位控制器（把门拉到光轴上），不是航向控制器（机身与门法向平行），所以"yaw 到底有没有把机身调正"没法判断 · 结论 → 本工具用真实录制的角点量出航向估计的噪声底
- 未记日期 · 角点 RMS 有 10~18 px，平面目标的转角对像素噪声很敏感 · 结论 → 噪声 << 待修角度（~10°）才能加"航向 → yaw"通道；同量级就老实以 sway 为主

## tools/analyze/log/analyze_kpt_dump.py

（本次未改动：docstring 只有用途与用法；`max_z_jump_m(默认 0.8)` 是该脚本用于"阈值是否过严"判断的当前值。）

## tools/analyze/log/analyze_pnp_center.py

### 立项动机
- 2026-09-18 · 现场"冲歪时最强只能卡进 p3p，还是有点调不动" · 结论 → 位姿档的横向误差 `dxn` 是把门原点投影回图像算的（`gate_task._on_pose`），若 p3p 时投影退化/有偏，控制器就在消一个错的误差（表现为"怎么调都不动/越调越歪"），本工具对比 `dxn_pose` 与 `dxn_bbox`

## tools/analyze/log/analyze_task_log.py

### 居中与 yaw 的关系
- 2026-09-20 起 · 居中只有 sway 通道（`comm.gate.align_yaw` 已删除）；gate 里唯一的 yaw 来源是 ALIGN.HDG 正航向 → 增益 `comm.motion.turn_pid` · 结论 → 日志里 yaw 恒 0 属正常，不等于"转向坏了"
### 正航向跳过记录
- 2026-09-22 新加 · 记录"居中达标却跳过正航向"的帧（`hdg_skip`）：最常见是 `no_psi`（还没测到过 full 帧的 psi）或 `done`（本门已做过正航向），正航向是一次到位、不重测不迭代 · 结论 → 水里复盘第一条看这个
### 冲刺瞬间的偏心
- 现场见过 · coarse 档日志里的 z 是上一次 width/位姿留下的陈旧值（曾把 9.25 换算成"偏心 20cm"这种假数） · 结论 → 该档无测距，别用 z 换算 cm

## tools/analyze/log/feature_coverage.py

（本次未改动：日期化的"SEARCH 左右平移扫视（2026-09-20 起不许旋转）"只出现在 print 的字符串字面量里（3 处：scan 的注释、checks 列表两处），按"不改字符串字面量"的约束保持原样。）

## tools/analyze/calib/label_corners.py

### 立项动机
- 2026-09-22 · 岸上工作间空气域实测：keypoint 权重最高分 0.074、角点全 0；同一权重在水下旧录制帧上是 0.87~0.94 · 结论 → 环境不响应时仍可在岸上完成几何标定（手工点 4 角）
### 放大镜默认开关
- 2026-09-23 改 · 放大镜原先默认开，画在右上角、边长 `140*zoom`，zoom=4 时 560 px，在 640 画面上盖掉约 87%，把图挡死 · 结论 → 默认关（按 `m` 开），以 0.65 不透明度叠加，`--loupe` 才启动即开
### 吸附只保证"落在管子上"
- 2026-09-22 · 修正旧注释"误差从 ±5 px 降到 ±1 px"的过度承诺：搜索半径只有 `r=6` px，而近距档红管宽约 28 px，窗口整片落在管内、红度是平台，去平台后仍由点击位置决定 · 结论 → 想标准就自己把十字压到管中线（配放大镜）；四角都偏同一侧 = 3% 量级系统性宽度偏差
### 点在哪一条线上
- 2026-09-22 · 门框外缘 0.77×0.56 m 对应的是"管子外轮廓的角"，而吸附只在 ±6 px 内找最红像素、不会把点拉到管中线 · 结论 → 点红管中线；判据 `Δu/Δv ≈ 0.77/0.56 = 1.375`
### 靶子跨度的方向无关性
- 2026-09-22 · 现场踩出来：用坐标轴投影会在竖着量时白丢一个 cosθ（甚至直接得 0），欧氏距离才与靶线方向无关 · 结论 → `span_stats` 用欧氏距离 `|P₃−P₁|`
### z 真值在目录名上
- 2026-09-22 · 靶子图是 `log/pnp_0922/ruler_z200/cap_001.jpg`，真值在**目录名**上，只看 basename 会漏掉 · 结论 → `_z_from_name` 按"最后两段路径"扫
### --resume 去重键
- 2026-09-22 · 早期版本按 basename 去重，而采图目录里文件名永远是 `cap_001.jpg`… ⇒ 不同距离档互相覆盖（`ruler_z100` 的记录被 `ruler_z300` 整行删掉，`--resume` 直接跳过整档） · 结论 → 去重键改用 `src_path`（完整路径），老记录才退回 basename

## tools/analyze/calib/pnp_calib.py

（本次未改动：docstring 只讲输入/输出/四件事/用法。`render_md` 里"p3p 历史实测 ≈ −28%"在 print 的字符串字面量里，按约束保持原样。）

### invert_scale（docstring 搬出）
- （真值口径差、畸变残差）混进比例里，近距离档尤其容易被带偏（实测差 ~2%）。

### (模块)（docstring 搬出）
- 1. 控制台：每档一行（z 真值 / z 实测 p50 / 相对误差 / 可用率 / mode 分布）；
- 门框**尺寸标尺**决定）→ 用实测比值直接反解**真实门宽**，再 1D 扫 `frame_h/frame_w`
- 安装角偏置（均值）与噪声（std）——这是 ALIGN.HDG 能不能收敛的直接依据。
- python3 tools/analyze/calib/pnp_calib.py --report log/pnp_report.md --csv log/pnp_summary.csv \
- ⚠️ 本工具**只读**：不改 cfg、不发串口。它给出的建议值要人工确认后再写进 cfg。

## tools/analyze/calib/ruler_calib.py

### 水下等效焦距
- 未记日期 · 水下解出 `fx_water ≈ 782.5 ± 5%` ⇒ 当时判断现用标定在水里仍有效、跑船的米制阈值不用改 · 结论 → 注释只保留"与 cfg 的 fx 对比得 k_med"这条判据，具体数值回报告/本文
### 跨度必须用欧氏距离
- 2026-09-22 · 现场实测：早期版本用 `p[2][0]-p[0][0]`（x 差），卷尺竖着放时把真实 379 px 的竖向跨度算成 2 px · 结论 → `du_from_points` 用欧氏距离，并在读入时用 `p` 重算老 `du` 自愈
### 留一校验的子集约束
- 2026-09-22 自检 · 留一子集若丢掉某轴唯一的档，那个焦距就没有约束，会给出 `fx=-223` 这种垃圾 · 结论 → 子集必须保留每个轴的档才做留一

### load_records（代码注释搬出）
- ⚠️ **优先用 `p` 重算**：`du` 可能是把竖向跨度算成 x 差的旧版本写下的，

### load_records（代码注释搬出）
- ⚠️ 同理：三点在场就**重算**斜视/中点偏离。老版本在 du≈0 时把 skew

### solve（代码注释搬出）
- ⚠️ 子集必须**保留每个轴的档**，否则那个焦距在子集里根本没有约束

### report（代码注释搬出）
- ⚠️ 斜视告警按**全部输入读数**判（包括被你 `--skip` 或被过滤掉的那些档），

### solve_joint（docstring 搬出）
- 固定 `c` 时**线性** ⇒ 对 `c` 一维扫描，每个 `c` 闭式解其余参数，取相对残差最小者。
- · 靶子有横向档 → 直接解 `fx`，**不需要**任何等比假设。
- ⚠️ 为什么必须联立：**单档靶子只有 1 个观测**，`(f, c)` 是整条曲线（简并）；

### axis_of（docstring 搬出）
- 靶线水平 ⇒ 量到 fx；竖直 ⇒ fy；斜着 ⇒ 两者混合（45° 附近最糟）。

### du_from_points（docstring 搬出）
- ⚠️ 别改回 `p[2][0]-p[0][0]`（x 差）—— 卷尺竖着放时它直接得 0。靶面与光轴垂直时
- `z=const` 平面到图像是均匀缩放，所以欧氏距离 = `(f/z)·L`，与靶线在画面里的方向无关。

### z_from_name（docstring 搬出）
- ⚠️ 靶子图是 `log/pnp_0922/ruler_z200/cap_001.jpg` —— 真值在**目录名**上（`cap_001.jpg` 里没有），
- 所以按"最后两段路径"扫，别只看 basename。

### (模块)（docstring 搬出）
- ⇒ z_tape + c = fx·L/Δu        ← 对 (fx, c) **线性**，两条以上即可最小二乘
- ⚠️ 换零点只是把 `c` 变成另一个固定常数，**不会让它归零** —— 只能"零点固定成制度 + 实测一次"。
- `tools/analyze/calib/label_corners.py --measure` 产出的 JSONL（`kind: "ruler"`，一行一个档位）。
- python3 tools/analyze/calib/ruler_calib.py --report log/pnp_0922/ruler_air.md log/pnp_0922/ruler.jsonl
- python3 tools/analyze/calib/ruler_calib.py --skip 300 log/pnp_0922/ruler_wet.jsonl   # 丢掉最远档
- · 每次解算至少 **2 档**（3 档可做留一校验）；`skew` 超过 ±2% 的档**先别用**（靶面没正对）；
- · `fx` 的用途：与 `vision.camera.front.calibration` 里的 fx 比 ⇒ `k_med = fx_cfg/fx_med`；
- 水下解出的 `fx_water` 若与 cfg 差距在几个百分点内 ⇒ 现用标定在水里仍有效，米制阈值不用改；

## tools/analyze/calib/tape_ticks.py

### 刻度周期的精度
- 2026-09-22 · 实测 1 m 档三张图的周期 10.679 / 10.763 / 10.833 px（离散 1.4%），与门框那套独立证据互证 · 结论 → 周期估计法精度可做到 ±0.5%，且不用点鼠标
### 3 m 档测到的是 JPEG 块效应
- 2026-09-22 · `f≈1080` 时 3 m 处刻度周期只有 3.6 px，与 JPEG 8×8 块效应同量级；实测在 3 m 处检出的是块效应（周期 7.9 px）；刻度拟合还锁到杂线、自信地给出 `f=4252 px`，而 FFT 给 3.58 px（正确）；另一处实测该档给出 `f=1823`（真值 ~1080） · 结论 → 只信 1.0~2.0 m 的结果，3 m 用点击法；`period_at` 与 `fft_period` 必须交叉校验
### 多检杂线不能靠剔离群点
- 2026-09-22 自检实测 · 多检一格会让它后面所有序号整体平移，残差呈锯齿状，MAD 阈值刚好放它过去（rms 2.26 px 剔不掉） · 结论 → 必须先用 `drop_extra_ticks` 删杂线
### FFT 前必须高通
- 3 m 档实测 · 不高通（减滑动均值）时尺子的亮度包络/印刷数字把刻度周期压掉，FFT 给 11.6 px；高通后给 3.6 px（正确） · 结论 → `fft_period` 先高通再变换
### 挑卷尺范围不能只看"亮 + 长"
- 3 m 档实测 · 门板高光比尺子更亮更长 · 结论 → 用 `tickiness`（高通标准差）挑含周期刻度的区段
### 找连续段必须先平滑
- 1 m 档实测 · 刻度把亮尺切成 ~1 cm 的碎块，不平滑时最长段只有 11 px · 结论 → `detect_span` 先在平滑剖面上找连续段

## tools/deploy/archive_baks.sh

### 脚本位置
- 未记日期 · 脚本从 `tools/` 移到 `tools/deploy/`，头部两行位置说明互相矛盾（代码 `cd "$(dirname "$0")/../.."` 只有"上两级"是对的）· 结论 → 头部统一为"工程根 = 本脚本目录的上一级"

## tools/deploy/check_board_parity.sh

### 无方括号条目的含义
- 2026-09-18 · 板端把 kd 改成 0.05，而清单里该条目只有本地值 ⇒ 本地报"逐字节一致"（比的是本地 vs 本地） · 结论 → 无方括号 = 没有板端核对证据，必须单独暴露（`verified` 字段）
### 清单外的新文件
- 2026-09-20 加 · 新文件（`common/motion/turn_deg.py`、`gate/motion/heading_align.py` …）漏在清单外时，部署会推上"新 gate_task.py + 缺 heading_align.py"这种半套状态 → 板端 import 即崩；而原先只遍历清单的检查完全看不见这些文件（静默通过） · 结论 → [1/2] 之后加"本地有、清单没有"的检查（1b）
### 脚本位置
- 未记日期 · 头部"工程根自动定位到上一级"已过时（现为 `tools/deploy/`，上两级） · 结论 → 头部描述改为"上两级"

## tools/deploy/deploy_to_board.sh

### 性能沿革
- 未记日期 · 原先"一文件一次 SSH×2"，70+ 文件要几分钟 · 结论 → 改为板端 md5 一次 SSH 批量取 + 只打包变化文件 + 备份/删除各一次 SSH
### DELETED 列表
- 2026-09-19 · 按角度原地转的实现迁到 `common/motion/turn_deg.py`（脚本与 gate 正航向共用），板端旧 `task1_2/turn_deg.py` 必须删，否则两个同名脚本行为不同 · 结论 → 列入 DELETED
- 2026-09-21 · 分类重整：tests/tools 下移到子目录后旧扁平路径必须删，否则板端同时存在新旧两份、pytest 收集到重名模块会 import-mismatch · 结论 → 旧扁平路径列入 DELETED
- 2026-09-20 · 合并成 6 个测试文件时的旧名（`test_gate_dash.py` 等）板端仍留着，会 import 已删除模块 → 板端全量 pytest 收集即报错 · 结论 → 先备份再删，保证"板端 == 清单"
- 未记日期 · 反面教材：`tests/tasks/gate/test_gate_flow.py` 曾既是"被删的分层版用例"、又是"活文件"的同名路径，把它列进 DELETED 会把刚上传的文件再删掉 · 结论 → 加 DELETED 时看清层级（活文件在 `tests/tasks/`）
### 分叉跳过表
- 历史现场事故 · 规则原先只写在文档里（"cfg 不要整份推板端"），靠人工记得排除；漏一次就把现场值冲掉 —— `depth_guard.min_depth_m=0.55` 被冲掉过两次；板端 `SIM_MODE=False` 在 2026-09-27 被本地 True 冲掉过一次 · 结论 → 分叉清单的唯一来源是 `tools/deploy/board_forks.txt`（与 check_board_parity.sh 共读），不再是脚本里的数组
- 未记日期 · `cfg/comm.yaml` 曾按"统一为本地值"处理（板端台架分叉：`pass_target=2` / `timeout=500000` / `wait_fresh_ms=1000` / `sweep_s=3.0`），现已不在分叉清单里

## tools/deploy/tidy_board_bak.sh

### 脚本位置与默认目标
- 未记日期 · 头部写成 `tools/deploy/tidy_board_bak.sh`、默认板端目录写成 `/home/sunrise/AUV`，与代码现状（`tools/deploy/`、`/home/sunrise/Desktop/AUV_New`）不一致 · 结论 → 头部按代码现状更正（未改任何语句）

## tools/deploy/Adomain/sync_Adomain_repo.sh

### A 域口径的由来
- 2026-09-27 · 用户定：A 域那套老功能的参数一个都不改，只把两个新功能放进来（① 转角度 ② 解码后约束），它们自己的键取新值 · 结论 → 口径与逐键映射见 `tools/deploy/Adomain/Adomain_cfg/README.md`（A 域参数的 canonical 副本在该目录）

---

## tools/analyze/calib/label_corners.py


### snap_to_red（代码注释搬出）
- ⚠️ 红管是一条"脊"：同红度的像素成排出现，直接 argmax 会固定吸到**窗口左上角**那一个

### on_mouse（代码注释搬出）
- ⚠️ cv2.putText 的 Hershey 字体**只画 ASCII**，中文会变成一串方块 —— 所以

### measure（docstring 搬出）
- 与 `label()` 同一套坐标域处理（cfg 的 `undistort` 决定），所以量出来的 `fx` 与 PnP 同域。

### load_done（docstring 搬出）
- ⚠️ 别按 `basename` 去重：采图目录里的文件名永远是
- `cap_001.jpg`… ⇒ **不同距离档之间互相覆盖**（`ruler_z100` 的记录被 `ruler_z300`

### _z_from_name（docstring 搬出）
- ⚠️ 靶子图长这样：`log/pnp_0922/ruler_z200/cap_001.jpg` —— 真值在**目录名**上，
- 所以只看 `basename` 会漏掉。这里按"最后两段路径"扫：

### span_stats（docstring 搬出）
- 所以两端点的**欧氏像素距离** `du = |P₃−P₁| = (f/z)·L`，与靶线在画面里怎么转无关
- · 三个刻度点等距 ⇒ `P₂` 落在 `P₁P₃` 的**中点**。
- `off_mid`(px)、`theta_deg`（靶线在画面里的倾角，0=水平 ⇒ 量到的是 fx）。

### snap_to_red（docstring 搬出）
- 否则保持原点击坐标（避免在非红区域乱跳）。
- ⚠️ 它**只保证"点落在管子上"，不保证落在管中线上**（别信"能把误差从 ±5px 压到 ±1px"）：
- ⇒ 窗口整片落在管内、红度是平台，去平台后仍然由**点击位置**决定。所以：

### (模块)（docstring 搬出）
- 为什么需要它
- 当现场环境让 keypoint 权重不响应时（岸上空气域就是这种情况），仍然可以在岸上把几何标定
- 标注坐标必须与 PnP 域一致。`cfg/vision.yaml → image.undistort: true`（当前）⇒
- PnP 域 = **去畸变域**（`gate.percept.gate_detector.board_camera()` 用 rectified K）。所以：
- 域弄错 ⇒ 深度整体缩放错（现象像"标尺不对"，很容易误判成门宽标错）。
- ① 采图（每档距离多采几帧；图里**不要**有叠加框，所以别用 `preview_detect --save`）：
- python3 tools/analyze/calib/label_corners.py --capture log/pnp_0922/z150 --n 8
- ② 标注（顺序必须是 **TL → TR → BR → BL**，与 `gate.percept.geometry.object_points` 同序）：
- python3 tools/analyze/calib/label_corners.py --images 'log/pnp_0922/z150/*.jpg' \
- ⚠️ **放大镜默认关闭**：它画在右上角、边长 `140*zoom`，zoom=4 时是 560px ——
- python3 tools/analyze/calib/pnp_calib.py --report log/pnp_report.md log/pnp_0922/pnp_z*.jsonl
- 点**红管的中线**，不是管子的外沿/内沿。理由：门框外缘 0.77×0.56 m 对应的是"管子外轮廓的角"，
- 窗口完全落在管内，红度平台一片 ⇒ 吸附**不会**把点拉到管中线，只保证"点落在管子上"。
- 所以冷启动那一刻的点击位置就是最终答案：必要时按 `m` 开放大镜把十字压在中线上，
- **也可以拿来量卷尺刻度（§A0b 内参靶子法）**：卷尺刻度没有红色 ⇒ 吸附不生效，
- 点 0 与 100 cm 两个刻度，读控制台打印的 `TL = (x, y)` / `TR = (x, y)`，取 `Δu = |x₂−x₁|`，
- ⚠️ 标注点置信度一律给 **1.0**（这是「干净标注」，不是检测结果）——

## tools/analyze/calib/tape_ticks.py


### fft_period（代码注释搬出）
- ⚠️ 必须先**高通**（减滑动均值）：否则尺子的亮度包络/印刷数字这些低频成分

### fft_period（代码注释搬出）
- ⚠️ 亚谐波改正：**窄脉冲串的频谱是梳状**，2 倍频（周期一半）的峰可能比基频还高

### detect_span（代码注释搬出）
- ⚠️ 先在**平滑**后的剖面上找连续段：刻度本身会把亮尺切成一段段 ~1 cm 的碎块，
- 不平滑就永远拼不出一条长段（1 m 档实测最长段只有 11 px）。

### detect_band（代码注释搬出）
- ⚠️ 阈值必须用**全图**均值：若亮带占满整行，用行均值当阈值会把自己排除掉
- （合成图自检直接踩到：整行 210，阈值 225，一个像素都不算"亮"）。

### measure_one（代码注释搬出）
- 实测 3 m 档会给出一个看着像样的错数（远大于真值）。要救这一档只能显式 --band/--x。

### measure_one（docstring 搬出）
- ⚠️ 谁更好用「刻度拟合与全局 FFT 是否一致 + 用到的刻度数」判，不能一刀切：
- 实测 1 m 档整段更好（尺子几乎占满整幅），3 m 档自动收窄才对（门板纹理混进来）。

### detect_span（docstring 搬出）
- ⚠️ 光靠"亮 + 长"挑不出来：门板的高光可能比尺子更亮更长。

### period_at（docstring 搬出）
- ⚠️ 取**光轴处**（`x = cx`）而不是整条尺的平均：靶面没正对时，尺上各处的

### fit_index（docstring 搬出）
- ⚠️ 只做"常数周期划线"不行（靶面没正对时尺上比例本来就在变，实测跨度内差 6%）。

### fft_period（docstring 搬出）
- 实测 3 m 档（刻度周期只有 3.6 px，与 JPEG 8×8 块效应同量级），刻度拟合会锁到杂线、

### drop_extra_ticks（docstring 搬出）
- 说明这三格挤成了两格 ⇒ **中间那个是多余的**（印刷数字的竖笔画、污点、
- ⚠️ 这一步不能靠"拟合后剔离群点"代替：多检一格会让**它后面所有序号整体平移**，
- 残差因此呈锯齿状、MAD 阈值刚好放它过去（实测 rms 2.26 px 剔不掉）。

### (模块)（docstring 搬出）
- 为什么需要它
- `label_corners --measure` 让操作者点 3 个刻度，靠人眼定位 ⇒ 误差 ~±1.5 px/点。
- 而卷尺上**每隔 1 cm 就有一条刻度线**，一整条尺上有几十上百条 ⇒
- 模型（与 `ruler_calib` 完全一致，所以输出可以直接喂它）
- ⇒ f = period_px · (z + c) / L
- ⚠️ 前两个前提，错了结果就错：
- 自检办法：拿两个已知距离各测一次，比值应约等于距离比（与 1 cm 假设无关）；
- 而 JPEG 的 8×8 块效应正好落在这个尺度上（实测在 3 m 处检出的是块效应，周期 7.9 px）
- ⇒ **只信 1.0~2.0 m 的结果**；3 m 用点击法。
- python3 tools/analyze/calib/tape_ticks.py --z 1.00 log/pnp_0922/ruler_z100/*.jpg
- python3 tools/analyze/calib/tape_ticks.py --z 1.00 --out log/pnp_0922/ruler_ticks.jsonl             --band 118,140 --x 220,950 cap_001.jpg

## tools/analyze/log/analyze_task_log.py


### main（代码注释搬出）
- ⚠️ 水里复盘第一条就看这段：居中达标却**跳过正航向**的原因

### main（代码注释搬出）
- ⚠️ 别把"居中"当成能用 yaw：居中只有 sway 通道（`comm.gate.align_yaw` 已删除；

### main（代码注释搬出）
- ⚠️ 别用 z 换算 cm：coarse 档没有测距，日志里的 z 是上一次 width/位姿留下的

## tools/analyze/log/feature_coverage.py


### scan（代码注释搬出）
- SEARCH 的横移波形（2026-09-20 起 SEARCH = 左右平移扫视，不许旋转）

### (模块)（docstring 搬出）
- 注意：本工具只统计「日志能证明的东西」。像「深度保护是否真的压掉了上浮」这类

## tools/check/pipeline/check_domain.py


### _project_root（代码注释搬出）
- ★ 2026-09-28：D 域门权重换代 stage1_v3（ca5ba84f，3,938,806 B）。

### (模块)（docstring 搬出）
- 板端两个仓库的分工见 `tools/README.md` §6。本脚本回答三个问题：

### (模块)（docstring 搬出）
- python3 tools/check/pipeline/check_domain.py                 # ① 我这个仓库现在是哪一域？
- python3 tools/check/pipeline/check_domain.py --frames DIR    # ② 本域权重在样本帧上的角点置信度分布
- python3 tools/check/pipeline/check_domain.py                                  # ③ A 域改造前那套"两域等价"比对已删（2026-10-01）后是否逐像素一致
- ③ 的用法：`--equiv-ref` 给一份**参考实现**（例如改造前旧仓库的 `common/vision/preprocess.py`）。

## tools/check/vision/check_gate_pose.py


### (模块级)（代码注释搬出）
- 兜底值 == 当前 cfg（别在这里改回历史字面量）

### (模块级)（代码注释搬出）
- 否则"低于 0.5 的角点"根本不会出现在这张表里，就看不出"位姿为什么被拒"。
- ⚠️ 后处理（去重/几何）仍按 cfg 生效 —— 这里打印的就是**任务会看到的那批实例**。

## tools/check/vision/check_kpt_decode.py


### main（代码注释搬出）
- ⚠️ 别把 --conf 抬成 max(0.4, conf-0.1)：那会把"没有 cell ≥0.40"误读成"模型失效"；

### per_cell_report（代码注释搬出）
- ★ 先报"模型到底响不响应"（不假设布局：逐张量报最大值）

### per_cell_report（docstring 搬出）
- 所以**正确约定**应让 |bbox(kpts) - box| ≈ 几个 px；错的会差 ≈ stride 量级。

### raw_max_scores（docstring 搬出）
- **刻意不假设通道布局**（NHWC 与 CHW 两种导出都见过）：只取全局最大值，
- `>0` ⇒ 有响应（sigmoid>0.5）；`<= -4` ⇒ 基本等于没响应。**两者修法完全不同。**

### (模块)（docstring 搬出）
- # 板端（有权重 + hbm_runtime）
- python3 tools/check/vision/check_kpt_decode.py --image log/frames/pv_000040.jpg --out /tmp/decode_check.jpg

## tools/analyze/log/analyze_heading.py


### (模块)（docstring 搬出）
- （把门拉到光轴上 = 机头指向门），**不是航向控制器**（机身与门法向平行）。所以"yaw 到底

## tools/check/boat/check_dof_sign.py


### (模块)（docstring 搬出）
- 为什么需要它（**一次性标定工具**）：
- 固件常量 × `dof_map.yaw.sign` × `telemetry.yaw_sign`），不探向、不现场测；
- ⚠️ 别改回开环探向（白转十几度），也别在运行时现场测符号。
- 命令 +yaw  ⇒ 若门在画面里**左移**(dx↓) 且 psi↓ ⇒ 符号正确；反之 ⇒ dof_map.yaw.sign: -1
- 命令 +sway ⇒ 若门在画面里**左移**(dx↓)        ⇒ 符号正确；反之 ⇒ dof_map.sway.sign: -1
- 同时打印遥测 yaw 的变化 ⇒ 顺带定出遥测侧符号 s（g·s 的那个 s）。

### (模块)（docstring 搬出）
- AUV_SIM_MODE=0 python3 tools/check/boat/check_dof_sign.py --dof sway --value 0.30 --hold 2.5
- AUV_SIM_MODE=0 python3 tools/check/boat/check_dof_sign.py --dof yaw  --value 0.40 --hold 2.0
- AUV_SIM_MODE=0 python3 tools/check/boat/check_dof_sign.py --dof sway --value 0.30 --dry   # 只看基线，不发推力

## tools/check/boat/check_hdg_lockup.py


### check_expiry（docstring 搬出）
- ⚠️ 别省掉这条：窗口过期后「不是我刚丢的门」每帧照样成立，HDG 会被永久锁死。

### (模块)（docstring 搬出）
- 故障形态：消费 post_sway 窗口的方法只有「不是我刚丢的门 ⇒ return」，没有「窗口过期 ⇒ 收手」
- 路径 ⇒ 居中 / 正航向 / 丢失处理全被跳过 ⇒ HDG 永久锁死（船乱漂，而串口打印的 yaw 一直是中位）。
- 本脚本**只读**，不修改任何文件。用法（在板端工程根）：
- python3 tools/check/boat/check_hdg_lockup.py            # 检查 gate/ 与 cfg/

## tools/check/pipeline/check_paths.py


### (模块)（docstring 搬出）
- 为什么需要它（板端两类坑）
- 1. **cfg 里写绝对路径** ⇒ 把配置钉死在某一份拷贝上：标定/权重指错一份就静默用了另一个文件，
- 那份被删/被搬 ⇒ **相机初始化直接崩**。
- 2. **相机标定 yaml 的第一个字节**：板端 cv2（4.11）靠文件开头认格式 ⇒ 前面有注释行、
- **不兜底** ⇒ 崩。注释只能放在 `---` **之后**。

## tools/check/pipeline/check_pipeline_identity.py


### (模块)（docstring 搬出）
- 每次把代码/配置同步到另一台机器（PC ↔ 板端）后跑一次（服务整个项目，不限 gate）：
- 校验（设备内相对一致性，不绑定 cv2 版本/哈希，避免 PC 与板端误报）：

### (模块)（docstring 搬出）
- # 或: python3 tools/check/pipeline/check_pipeline_identity.py --vision cfg/vision.yaml
- 2. 单一来源：getOptimalNewCameraMatrix(alpha=0, centerPrincipalPoint=False) 的结果

## tools/check/vision/auv_kpt_meter.py


### (模块)（docstring 搬出）
- ⚠️ 这里把 `model.score_threshold` 压到 0.01 **只为暴露响应**，不代表能当阈值用；

### (模块)（docstring 搬出）
- cd ~/Desktop/AUV_New && python3 tools/check/vision/auv_kpt_meter.py 75      # 秒数，默认 75
- 判据：要能解 PnP 需要 >=3 个角点 conf >= `vision.gate.keypoint.conf_thr`（**该阈值以 cfg 为准**）。

## tools/deploy/deploy_to_board.sh（脚本注释搬出）


```bash
#   bash tools/deploy/deploy_to_board.sh --no-test    # 跳过板端 pytest（最快，秒级）
#   bash tools/deploy/deploy_to_board.sh --dry-run    # 只报告将上传/删除哪些文件
#
# 退出码：0 = 同步完成且板端自检通过；1 = 同步完成但**板端 pytest 未全绿**（末尾 ⚠️ 会说明）。
#   ⚠️ 板端"多出来的旧文件"（本地已删/改名的用例等）不会自动消失，全量 pytest 会因
#      import 已删除模块而收集报错 → 把这类文件补进下面的 DELETED 列表（会先备份再删）。
#
# SSH 包装器默认在 /home/ansty/RDKX5/（含密码的 askpass 不进仓库），可用环境变量改：
#   AUV_SSH / AUV_SCP / AUV_STREAM / AUV_ASKPASS
#
# 为什么快：① 板端 md5 **一次 SSH 批量取**；② 只把变化的文件打成一个 tar 传过去；
#           ③ 备份/删除各一次 SSH。每一步都打印累计耗时，慢在哪一眼能看到。
#
# 注：md5 只比对内容、**看不见文件权限**；而 tar 只上传"内容变化"的文件，
#     所以内容没变但权限丢了的（如 scp 覆盖掉 manual.sh 的 +x）永远不会被修。
#     这里在收尾步骤顺手对板端 *.sh 统一 chmod +x，杜绝 "./manual.sh: Permission denied"。
  # 按角度原地转的实现在 common/motion/turn_deg.py（脚本与 gate 正航向共用），
  # 转向调用日志 `turn_log` 已从 common/ 挪到 base/（用户 2026-09-27 定：它是排查工具，
  # v1.3/v1.4 分层版 gate/ **已确定不需要、已删除**：
  # 板端把分层目录删掉，回到扁平 gate/（扁平那 8 个文件在清单里，会自动上传）
    # [REMOVED 2026-10-04] gate/motion/__init__.py  ← 是活文件（本地存在+在清单里），列在 DELETED 会被误删
  # 分层版专用工具/测试/文档（板端旧副本一并删）
  # （否则板端会同时存在新旧两份：pytest 收集到重名模块会 import-mismatch 报错）
  # 板端遗留的**旧版/已合并**用例（合并成 6 个文件时的旧名）
  # 板端这些更早的拆分文件不在清单里、会 import 已删除模块（gate.vision/gate.data/
  # common.ramp/build_frame_with_header…）→ 板端 pytest 收集即报错、把全量自检搞红。
  # 先备份到 bak/deploy_<stamp>/ 再删，保证"板端 == 清单"。
  #    "活文件"的同名路径 —— 那种情况下把它列进 DELETED 会把刚上传的文件再删掉。
  #    分类重整后活文件是 `tests/tasks/gate/test_gate_flow.py`，扁平路径 `tests/test_gate_flow.py`
  #    才是要清的旧位置（见上面的"分类重整"段）；两者同名不同路径，**加 DELETED 时看清层级**。
  # 2026-09-27 文档归位：记录类文档从 doc/ 移到 doc/记录/，板端旧路径的副本要删
  # （新路径已在清单里 → 会自动上传；不删的话板端会同时留两份同名文档）
  # 板端遗留的改名文件：本地是 doc/记录/psi测量步骤_20260923.txt（无下划线、已在 doc/记录/），
  # 板端 doc/ 下还多一个 `psi_测量步骤_20260923.txt`（旧名 + 多一个下划线）→ 先备份再删。
  # 2026-09-30 目录两层分级：**旧扁平路径全部要清**（否则板端新旧两份并存，
  # pytest 收集到重名模块会 import-mismatch，脚本也会指错路径）。
    # [REMOVED 2026-10-04] tests/tasks/test_ball.py  ← 是活文件（本地存在+在清单里），列在 DELETED 会被误删
    # [REMOVED 2026-10-04] tests/tasks/test_motion.py  ← 是活文件（本地存在+在清单里），列在 DELETED 会被误删
# ---- 「本地/板端故意分叉」的跳过表 ----
#      （那些现场值没有任何本地副本）。
#   换目标仓库时可用 AUV_FORKS_FILE 指向另一个分叉清单（A 域那份已随 A 域退役归档）：
#     那个仓库的"故意分叉"和本仓库无关（它的 base/cfg/settings.py 也要跟着传）。
#   cfg/comm.yaml 曾按"统一为本地值"处理，现在已不在分叉清单里。
    # ★ 活文件保护（2026-10-04）：清单里的文件是活文件，绝不能被这里删掉。
    #   历史坑：gate/gate_task.py、gate/motion/__init__.py、tests/tasks/test_ball.py、
    #   tests/tasks/test_motion.py 既在 board_parity.md5 又在 DELETED 里，
    #   跑一次 deploy 就会把板端活文件删掉（且因"已一致"不会重传）→ gate 任务直接崩。
# ---- 6) 终检 + 刷新清单（一次 SSH 批量 md5）----
#    否则会把那个仓库的字节写进清单，在用副本的对照当场变红。
# 板端 pytest **未全绿要吼出来**（以前是静默放过：红着也照样"完成"，很容易误判）
# 清单已按 md5 刷新（那是"板端字节 == 本地"，与测试无关）；这里只把退出码拉红。
```

## tools/deploy/check_board_parity.sh（脚本注释搬出）


```bash
#   bash tools/deploy/check_board_parity.sh              # 本地 vs tools/deploy/board_parity.md5（无需板子，毫秒级）
#   AUV_SSH=/home/ansty/RDKX5/.ssh_x5.sh \
#     bash check_board_parity.sh --board          # 再与板端比对（**1 次 SSH 批量取 md5**，秒级）
#   ... --board --only-verified                   # 只查有板端实测证据的那批
#   ... --board --slow                            # 退回"一文件一次 SSH"的老路径（SSH 不稳时用）
#   ... --board --write                           # 顺带把 tools/deploy/board_parity.md5 刷成当前实测状态
#
# 说明：manifest 记录的是"最后一次与板端逐文件 md5 比对一致"时的字节；
#   ① 本地全绿 = 本地没被改过；
#   ② --board 也全绿 = 板端确实装的就是这份代码。
#   只读操作：不写板端、不改本地。
# 「本地≠板端是**故意的**」分叉清单（与 deploy_to_board.sh 读同一个文件）：
#   ★ 命中的文件不算不一致 —— 否则 --board 永远报 [板端不同] 并 exit 1，
  #    这类条目比的是"本地 vs 本地"，不能当作"与板端一致"（板端手改过本地是看不见的）。
# 1b) **本地有、清单没有**的源文件 —— 这类文件 deploy_to_board.sh **永远不会上传**
#        这种半套状态推上板 → 板端 import 即崩；而只遍历清单的检查**看不见这些文件**（静默通过）。
#     收录规则与 --write 里那段 find 保持一致（改了那边记得同步这里）。
    # ② 本地源码树里新增的文件
    #    -not -path './.*' 把**点目录**整个排除（.pytest_cache/、.git/、.vscode/…）：
    #    它们里面的 README.md 文件名不以点开头，只靠 `-not -name '.*'` 拦不住，
    #    一旦进清单就会被 deploy 当源码传上板。
```

## tools/deploy/tidy_board_bak.sh（脚本注释搬出）


```bash
#   bash tools/deploy/tidy_board_bak.sh               # 执行
#   KEEP=5 bash tools/deploy/tidy_board_bak.sh        # 保留最近 5 个快照不解压（默认 3）
#   BIG_FILES=10 bash tools/deploy/tidy_board_bak.sh  # 文件数 ≥ 它的算"大改动"，也保留（默认 10）
#
# 整理后的板端布局（<板端> = AUV_BOARD_DIR，默认 /home/sunrise/Desktop/AUV_New）：
#   <板端>/bak/
#   ├── README.md                      # 自动生成的索引（含每个归档的内容 + 恢复方法）
#   ├── rollback/deploy_MMDD_HHMMSS/   # 值得留着回滚的**原样**快照：cp -p 回去即可
#   │                                  #   = 最近 KEEP 个 ∪ 文件数 ≥ BIG_FILES 的大改动
#   └── archive/
#       ├── deploy_YYYY-MM-DD.tar.gz   # 该天其余 deploy_* 快照（按目录名里的 MMDD 归类）
#       ├── legacy_YYYY-MM-DD.tar.gz   # 散落的 *.bak_* 文件 + bak/cfg（archive_baks.sh 的产物）
#       └── *.tgz                      # 板端原有归档，原样移入
#
# 原则：
#   * **先打包再删**：除"空目录"外，任何内容都不会被丢掉（要彻底释放空间就手工删 archive/）；
#   * 日期一律用**本地时间**：板端 RTC 常年不准（`deploy_*` 目录名是本地时间生成的，
#     板端 `date`/mtime 却是 2000-01-01 那一类），所以归档名 = 目录名 MMDD + 本地年份；
#   * 幂等：可随时重跑；新的 `bak/deploy_<stamp>/`（deploy_to_board.sh 每次同步前生成）会被
#     自动归入 rollback/ 或 archive/；
#   * 只操作板端 `<板端>/bak/`，不动代码目录。
#
# SSH 走与 deploy_to_board.sh 相同的包装器（默认 /home/ansty/RDKX5/.ssh_x5*.sh）：
#   AUV_SSH / AUV_STREAM / AUV_BOARD_DIR / AUV_HELP_DIR
```

## tools/deploy/archive_baks.sh（脚本注释搬出）


```bash
#
# 用法（脚本位置变了也没关系，工程根自动定位；可在任意位置调用）：
#   bash tools/deploy/archive_baks.sh              # 归档（移动 *.bak* 到 <工程根>/bak/）
#   bash tools/deploy/archive_baks.sh --dry-run    # 只列出将要移动的文件，不动
#   bash tools/deploy/archive_baks.sh --list       # 查看 bak/ 现有内容
#   bash tools/deploy/archive_baks.sh --help
```

## tools/deploy/archive_board_legacy.sh（脚本注释搬出）


```bash
#   bash tools/deploy/archive_board_legacy.sh             # 备份 → 校验 → 删除 → 打包归档
#   KEEP_TAR=0 bash tools/deploy/archive_board_legacy.sh  # 不额外打整包（默认打）
#
# 为什么单独一个脚本，而不是直接跑 deploy：
#   `deploy_to_board.sh` 会**先上传本地文件**。而板端当前带着**本地仓库没有的新功能**
#   （下位机旋转执行协议 + 冲刺前航向确认锁存 `_hdg_ok`，见 doc/记录/），全量 deploy 会把它们冲掉。
#   归档旧布局不需要上传，所以这里只做"备份 + 删除"。
#
# 旧路径清单的**唯一来源**是 `deploy_to_board.sh` 里的 `DELETED=(...)`：本脚本解析它、不另抄一份
#   （抄两份必然漂移：deploy 加了、这边没加 = 永远清不干净）。
#
# 板端布局（<板端> = AUV_BOARD_DIR，默认 /home/sunrise/Desktop/AUV_New）：
#   <板端>/bak/legacy_layout_<本地时间戳>/   # 原样保留的旧文件（cp -p，可直接拷回去）
#   <板端>/bak/archive/legacy_layout_<时间戳>.tar.gz   # 上面那份的压缩包
#   <板端>/bak/board_state_<时间戳>.tar.gz             # 顺手存的"归档前板端代码全貌"（排除 bak/log/models）
#
# 原则：**先备份、再删除、后校验**；只碰 DELETED 清单里的路径与 <板端>/bak/，不动其它代码。
# ---- 护栏：本地存在同名文件 = **活文件**，绝不在板端删它 ----
#   （2026-10-02 踩过：DELETED 里误列 `gate/gate_task.py`，照单执行把板端生效的编排文件删了，
#     靠 bak/legacy_layout_*/ 才恢复。DELETED 是人写的清单，会漂；这里的判断来自文件系统。）
```

## tools/analyze/calib/label_corners.py（脚本注释搬出）


```
记录 schema 见 `make_ruler_record`（`kind: "ruler"`，**别**喂给 `pnp_calib`）。
`src` 传**完整路径**优先按 `src_path` 匹配；老记录（只有 basename）按名字匹配。
只有老记录（没有 `src_path`）才退回 basename。
既覆盖 `pnp_z150.jsonl`，也覆盖 `ruler_z200/cap_001.jpg`。
**任意方向都成立**（横着、竖着、斜着量都行）：
· 靶面只要**与光轴垂直**，`z=const` 平面到图像就是**均匀缩放** `f/z`（小孔模型下严格成立），
—— 用坐标轴投影会白丢一个 `cosθ`（竖着量更是直接得 0）。
`skew = (后半 − 前半)/du`：靶面绕竖轴/横轴偏了就会偏（近侧半段更长）。
`off_mid`：中点离 `P₁P₃` 连线的**垂距(px)** —— 点错刻度/卷尺有折角时会变大。
返回 dict：`du`(px, 欧氏)、`du_l/du_r`(px)、`skew`、`mid_frac`(应 ≈0.5)、
红度 = R − max(G, B)（BGR 输入）。邻域内最大红度比点击处高出 `min_gain` 才吸附，
搜索半径只有 `r=6` px，而近距档红管宽约 28 px
想标准，自己把十字压到管中线（配合放大镜）。四角都偏同一侧 = 系统性宽度偏差。
------------
PnP / 深度标定要的是「**像素角点 + 卷尺真值**」，检测器只是自动产出角点的手段。
做完 —— 手工点 4 个角。
坐标域（**最容易错的一点**）
----------------------------
· 用 `--capture` 采的图**已经过去畸变**（本工具按同一套标定做 remap，与运行时同源）；
· 用 `--images` 指自己拍的**原图**时，要加 `--undistort` 让工具先 remap 再显示/标注。
----
--out log/pnp_0922/pnp_z150.jsonl
键盘：左键落点 ｜ u 撤销 ｜ r 重来 ｜ s 保存并下一张 ｜ n/空格 跳过 ｜ q 退出(保存已标)
m 放大镜开关 ｜ + / - 放大镜倍率
在 640 的画面上会盖掉约 87%，把图挡死。现在按 `m` 才显示，且以 0.65
不透明度叠加（底下的画面仍可见）。`--loupe` 可让它启动即开。
③ 标完喂标定工具（文件名里的真值照旧生效）：
离线/脚本模式（不起窗口，便于批量与用例）：`--coords "x,y;x,y;x,y;x,y"`
**点在哪一条线上（直接决定标定口径）**
而 `snap_to_red` 只在 **±6 px** 邻域里找最红的像素 —— 近距档管子宽（2 m 处约 28 px），
四角都偏"同一侧"会变成一个 3% 量级的**系统性宽度偏差**（正对时 = 3% 深度偏差）。
判据：标完看 `Δu/Δv`，应落在 0.77/0.56 = **1.375** 附近（正对档）；明显偏离就先查是不是点偏了。
然后按 **q** 退出（**不要**存 jsonl —— 那不是门角点，喂给 `pnp_calib` 只会得到垃圾位姿）。
比检测结果更可信，正是标定 PnP 想要的输入。
```

## tools/analyze/calib/pnp_calib.py（脚本注释搬出）


```
① 用"正对档"的**线性拟合斜率** `a`（`z_meas = a·z_true + b`）反解真实门宽：
`W* = W0 / a`。用斜率而不是"比值的直接中位数"是关键 —— 中位数会把截距 `b`
拟合不可用时退回"比值中位数"。
② 固定 `W*`，1D 扫宽高比 `frame_h/frame_w`（正对时深度对宽高比只有弱依赖，
返回 dict（含扫描表，便于看平坦度）。
返回 dict：逐帧量 + 汇总（含 mode 分布与位姿可用率）。
它回答一个具体问题：**我们解出来的 z（和位姿）到底准不准、该把哪个参数改成多少。**
输入
----
`preview_detect.py --gate-kpt --dump` 产出的逐帧 JSONL（含 4 角点 + 置信度 + bbox）。
真值从**文件名**里读，不用手抄：
pnp_z150.jsonl          正对门，卷尺量到 z = 1.50 m
pnp_z150_lat+25.jsonl   同上，且相机光轴相对门中心横向偏 +25 cm（门在光轴右侧）
pnp_z150_up+10.jsonl    同上，且门中心在光轴**上方** 10 cm
pnp_z150_yaw+20.jsonl   同上，且船体相对门右转 20°
pnp_z075_yaw-30.jsonl   组合随便，段名可任意顺序
约定：lat/up/yaw 的**符号**都按"门相对相机"描述：
lat+ = 门在光轴右侧（相机系 t_x 应为正）
up+  = 门在光轴上方（相机系 t_y 应为**负**，因为图像 y 向下）
yaw+ = 船体右转（门法向相对光轴偏左 → psi = gate_normal_angles_deg 应为负）
文件名不带真值时，用 `--gt gt.csv`（表头：file,z_m,lat_m,up_m,yaw_deg）。
输出
----
2. `--report out.md`：完整报告（含参数校正建议、可直接粘贴的 cfg 片段）；
3. `--csv out.csv`：每档一行（喂 Excel / 贴进 runbook 记录表）；
4. `--frames-csv out.csv`：每帧一行（tz/tx/ty/psi/rms/mode，做散点图用）。
它做的四件事
------------
① **深度标尺反演**：正对门时 `z_meas = z_true · (W_cfg / W_true)`（4 角 PnP 的深度由
比值（正对时深度对宽高比只有弱依赖）拿最优 `frame_h`。
② **横向/竖向**：`t_x`（米）与 lat 真值、`t_y` 与 up 真值对比（校核尺度与符号）。
③ **航向**：`psi`（`gate_normal_angles_deg` 的 yaw 分量）与 yaw 真值对比 →
④ **选参**：`conf_thr × reproj_px` 扫描（可用率 vs 深度误差 p90）、p3p vs full 对比、
可选 kpt_mem 开/关 A/B。全部**离线复算**，不碰相机、不碰串口。
----
# 单档
python3 tools/analyze/calib/pnp_calib.py log/pnp_z150.jsonl
# 整组（通配）+ 出报告
log/pnp_z*.jsonl
# 只看 p3p/full 对比与选参扫描
python3 tools/analyze/calib/pnp_calib.py --sweep log/pnp_z*.jsonl
```

## tools/analyze/calib/ruler_calib.py（脚本注释搬出）


```
靶子 `Δu = f_轴·L/(z+c)`（水平靶线给 `fx`、竖直给 `fy`）；门 `Δu = fx·W/(z+c)`。
未知量按手头数据自动决定：
· 靶子只有竖向档 → `(fy, fx·W)`，并用 `fx = ratio·fy` 联结（`ratio` = cfg 的 fx/fy）；
门框的几档提供额外的"比例"信息，把 `c` 与 `W_eff` 一起压出来。
复用 `pnp_calib.parse_gt_from_name` 的命名约定（`pnp_z150.jsonl` → 1.50 m）。
返回 [{src, z, du, n}]，`du` = 每帧 `|x_TR − x_TL|` 的中位数（门框横向跨度）。
`recs` 里的每条记录兼容 `load_records` 的产物（`z/du/span/skew`）与手写形式
（`z_tape/du(|p)/span_m`）。返回 dict：fx, c, rms_m, rms_px, 逐档预测, 条件数。
判据用 `label_corners --measure` 记下的 `theta_deg`（画面里的倾角）。
没有 theta（手写记录）时按 `axis` 键，再不行默认 x。
要解决的问题
------------
PnP 的深度 `z_meas = fx_cfg·W_cfg/Δu` 里有两个未知量在只看门框时**简并**：
· 相机/介质带来的有效焦距 `fx_med`（罩外是空气还是水，差 ~1.4×）；
· 门框的真实口径 `W_true`（外缘 0.77 还是管子中线 0.72）。
用**已知长度的卷尺刻度**当靶子就与门无关了 —— 但还需要一个距离口径 `c`：
Δu = fx·L / (z_tape + c)      L = 刻度跨度(m)，z_tape = 从固定零点量到的读数
`c` 的物理含义：**光心相对你那个零点标记的偏置**（镜头/入瞳在零点前方则 c>0）。
输入
----
`z_tape` 优先取记录字段，缺了就从 `src`/文件名里解析（`ruler_z200` → 2.00 m）。
----
python3 tools/analyze/calib/ruler_calib.py log/pnp_0922/ruler.jsonl
判据（runbook §A0b / §B4）
-------------------------
· `c` 的用途：之后所有卷尺读数都按 `z_true = z_tape + c` 换算。
```

## tools/analyze/calib/tape_ticks.py（脚本注释搬出）


```
横向范围试**两个候选**：用户/默认的整段，和自动收窄到尺子的那段。
返回 `(y0, y1)`；找不到返回 `None`。
判据用**刻度纹理性**（`tickiness`，高通标准差）—— 尺子有周期性刻度，门板没有。
像素/厘米 本来就不同，而我们量的距离是到**尺中心**的 —— 中心处的局部比例才对。
三步（每一步都是被真实数据的失败模式逼出来的）：
1. **增量编号**：用最近几格的中位间距决定这一步跨几格（漏检 → 一次跳 2 格）；
2. **稳健拟合**（按 MAD 剔离群点）—— 用 std 会被离群点撑大阈值，剔不掉；
3. **反解序号并合并重复**：用拟合曲线反解每点的序号，同号只留离拟合最近的那个
（多检的杂线会落到同一号上被合并）；再回到 2，直到序号不再变。
返回 dict：`co`(多项式系数)、`k`、`x`、`rms`、`n`、`period_med`。
返回 `(co, keep_mask)`。
它对噪声稳，但会被"靶面没正对"带来的比例梯度糊掉（整条尺各处周期不同）；
`period_at` 则相反。**两者必须交叉校验**——只信一个会出事：
自信地给出 f=4252 px，而 FFT 给的是 3.58 px（正确）。
`pr` 是沿尺方向的平均亮度（亮尺 + 暗刻度）。返回 x 索引数组（相对 `pr` 起点）。
------------
把「1 cm 对应多少像素」当成一个**周期估计**问题，精度可以做到 ±0.5%，
而且**完全不用点鼠标**（与门框那套独立证据互证过）。
----------------------------------------------------
Δu = f·L/(z + c)     这里 L = 一个刻度间隔（标准卷尺 = 0.01 m）
1. **刻度间隔必须是 1 cm**（=0.01 m）。若这把尺细刻度是 5 mm，`f` 会**差 2 倍**。
或与门框/`--gate` 的结果对一下量级。
2. **靶线要水平**（量到的是 `fx`）。竖直放量到的是 `fy`。
3. 距太远就测不出来：刻度周期 ≈ `f·0.01/z`。`f≈1080` 时 3 m 处只有 3.6 px，
----
```

## tools/analyze/log/analyze_heading.py（脚本注释搬出）


```
问题：ALIGN 里 yaw 可以用 `dxn` 驱动 —— 那是**方位控制器**
理论上位姿档能测：门法向 n = R·[0,0,1]（相机系），
**但能不能用取决于噪声**：角点 RMS 有 10~18px，平面目标的转角对像素噪声很敏感。
* 噪声 << 待修的角度（~10°）→ 可以加"航向 → yaw"通道；
* 噪声同量级 → 测不准 → 老老实实以 sway 为主（并把估计打进日志继续观察）。
```

## tools/analyze/log/analyze_kpt_dump.py（脚本注释搬出）


```
用途：回答"放宽哪个阈值能真正提高位姿可用率"（conf_thr / vis_thr / reproj_px）。
```

## tools/analyze/log/analyze_pnp_center.py（脚本注释搬出）


```
动机：位姿档的横向误差 `dxn` 是**把门原点(0,0,0)用解出的位姿投影回图像**算的
本工具用真实录制的角点 dump 对比两套横向误差：
判据：
```

## tools/analyze/log/analyze_task_log.py（脚本注释搬出）


```
用途：下水/台架跑完后，判断"到底是识别、位姿、还是决策在卡"。
python3 tools/analyze/log/analyze_task_log.py <log.jsonl> [--task gate]
```

## tools/analyze/log/feature_coverage.py（脚本注释搬出）


```
用途：下水几轮之后，回答「我调的那些开关/分支，到底跑过没跑过」。
判定分三类（**只看日志与配置，不做推测**）：
· 用到了   = 日志里出现了该分支特有的字段取值（phase/substate/action/mode/命令非零…）
· 没用到   = 该轮日志里**一次都没出现**
· 未部署   = 代码/配置里存在，但该轮日志时间之后才加（由 --after-ms 或人工判断）
需要下位机告警行/遥测细节的，标记为「需控制台日志」，不在这里下结论。
```

## tools/check/boat/check_dof_sign.py（脚本注释搬出）


```
转向环极性 σ 是**算出来的**（`common/motion/turn_deg.py::yaw_sign()` =
这两个符号旋钮只能靠物理世界定一次 —— 相机是唯一不受 DOF/遥测符号影响的量，
本脚本就是定它们的那把尺子：量一次、写进 cfg，此后 σ 自动跟着走。
本脚本用**视觉**当独立真值：
输出：每相位的 dx/psi/遥测 yaw 均值差 → 结论 + 该改哪一行 cfg。
安全：任何退出路径都会补发中性帧（try/finally + 信号处理）；hold 期间 50Hz 心跳。
```

## tools/check/boat/check_hdg_lockup.py（脚本注释搬出）


```
新设计：postsway 的退出只看「**当前门**出现 ≥ `post_sway_kpt_min` 个可信角点」+ 超时兜底。
板端若还留着旧的 `_post_sway_same_gate`，就仍按老要求检查它（z 做主判据）。
python3 tools/check/boat/check_hdg_lockup.py                 # 本机（本仓库）
python3 tools/check/boat/check_hdg_lockup.py --root /home/sunrise/AUV   # 板端旧仓库
**结构感知**（2026-09-30 起）：本机把相位机拆成了 `gate/gate_task.py`（门面/编排）+
`gate/motion/*.py`（方法簇），`_D_HDG` 在 `gate/motion/params.py`；板端旧仓库可能还是
单文件 `gate/gate_task.py` + `gate/motion/heading_align.py`。所以这里**按新旧两套位置都扫**，
并在结尾打印实际扫了哪些文件 —— 免得"位置猜错"变成假绿/假红。
```

## tools/check/check_search_scan.py（脚本注释搬出）


```
只动 yaw 一个轴：其余三轴恒 0。用来现场确认
"左 span° → 停 pause_ms → 右 2×span° → 停 → 左 2×span° → …"
是不是**原地来回摆**，而不是**朝一个方向整圈转过去**（后者 = σ 判反了）。
python3 tools/check/check_search_scan.py                  # 默认 30s，span=cfg 值
python3 tools/check/check_search_scan.py --duration 60
python3 tools/check/check_search_scan.py --span 45 --pause 300
python3 tools/check/check_search_scan.py --sim            # 只打印、不驱动电机（台架）
判读：
· 每段末尾打印 psi 与段目标，残差应在 tol_deg(6°) 上下；
· 出现 `⚠ 判为极性/控制异常，停手` = 转不动/转反了（查 dof_map.yaw.sign / 推进器接线）；
· `没有遥测 ⇒ 不动` = 下位机没回 15B 帧（查串口/固件），闭环没有输入量。
```

## tools/check/pipeline/check_domain.py（脚本注释搬出）


```
2026-10-01 起本项目**只有一套识别方案**（A 域 / `vision.image.chain` 已退役归档，见
`bak/retired/Adomain_20261001/`），所以这里守的不再是"两域别混"，而是**同代配对**：
链路 D（resize640 → enhance → remap@640）＋ 白平衡 wb `[1.0, 1.05, 1.15]`
＋ CLAHE 关（`clahe_clip 0`）＋ gamma 0.85 ＋ **阶段一 v3 门权重** ＋ 水下重标的内参
这四样**必须同代**：换了权重没换标定（或反过来把老 AUV_4 权重配到新链上）都会跨域掉点，
只看单张图是看不出来的 —— 这就是本脚本存在的理由。
```

## tools/check/pipeline/check_paths.py（脚本注释搬出）


```
-------------------------
规则：cfg 里**一律写仓库内相对路径**（`cfg/front_camera.yaml`、`models/x.bin`），
运行时由 `base.cfg.settings.resolve_path()` 按工程根解析。
**甚至空行**都会 `Input file is invalid`（SystemError），而 `calibration_maps`
检查项
------
A 工程根自证：`main.py` / `cfg/` / `models/` 都在同一个根下
B `cfg/*.yaml` 里**没有绝对路径**（扫文本，不看注释以外的东西）
C 所有"路径类键"能解析且**文件存在**（用 `base.cfg.settings` 的实际解析结果）
D 相机标定 yaml：第一个字节是 `%YAML:1.0`，且 cv2 能读、fx 合理
E 提示同名拷贝（`/home/sunrise/AUV` 之类）存在时的混用风险
--------------------
python3 tools/check/pipeline/check_paths.py
python3 tools/check/pipeline/check_paths.py --quiet      # 只打印失败项
```

## tools/check/pipeline/check_pipeline_identity.py（脚本注释搬出）


```
python3 tools/check/pipeline/check_pipeline_identity.py
1. 声明项：undistort=true、input_size=640、标定分辨率==配置帧尺寸、纯拉伸（无 letterbox/crop）；
3. 域恒等式：A⁻¹(A·new_K)==K_full；detector decode 的逆缩放 == 1/resize 缩放
（"关键点回缩域"与"PnP 域"必须是同一个域）。
```

## tools/check/vision/auv_kpt_meter.py（脚本注释搬出）


```
用途：**换水 / 换光 / 换门之后的第一件事** —— 量出"这个权重在这种环境下、几米内还能出角点"，
每 ~4.5s 打印一行：
改判据请用 `tools/check/vision/check_kpt_decode.py`（它连原始 logit 一起报）。
```

## tools/check/vision/check_gate_pose.py（脚本注释搬出）


```
打印每帧：角点置信度 / mode / 候选解数量 / 最佳候选的 RMS 与 tz / gate_pose 最终结果。
```

## tools/check/vision/check_kpt_decode.py（脚本注释搬出）


```
YOLO-pose 的性质：目标的框 ≈ 其关键点的包围盒（门的框就是四角的外接框）。
这个判据不依赖肉眼，也不依赖 NMS 选了哪个 cell。
并把 shape 打出来便于识别哪个分支是类别。类别分支输出是 **logit**：
用该 cell 自己解出的框与目标框比"中心距离 + 尺寸"，最接近者即来源。
用途：把"角点乱飞"定性到底是 **模型** 还是 **解码约定**。
同一张图跑一次模型，把同一个 cell 的原始 kpt 输出用几种候选公式解出来，
画在同一张图上（不同颜色）并打印数值 —— 哪个公式把点钉在门框角上，就是对的。
候选公式（anchor = 选择到的网格 cell 下标；stride = input_w / g）：
V0 现在实现 : (raw)                    * stride
V1 ultralytics: (raw*2 + anchor - 0.5) * stride
V2 anchor   : (raw   + anchor)         * stride
V3 anchor-.5: (raw   + anchor - 0.5)   * stride
V4 已是像素 : (raw)                    * 1        （有些导出把解码烘进模型）
```
