# 注释历史 · 其它（跨文件的时间线、约定、历史事件）


> 本文件由 `doc/注释历史.md`（4128 行）于 2026-10-05 按域拆出；**内容一字未改**，只搬了位置。
> 顶层索引与"旧路径怎么解析"见 [`doc/注释历史.md`](../注释历史.md)。
> 代码/配置里的注释只写"是什么 + 当前值"，注意事项、实测数字与历史沿革一律记在这里。

## 2026-10-07 · 过门阈值/机制重设（跨文件的时间线）

- 依据 `log/rungate_20261006_postsway.jsonl`（385 帧 / 50.0 s / 7.7 fps，**一趟没转也没过门**）重设阈值与运动参数。
  **成篇记录（唯一入口）：`doc/记录/2026-10-07-gate-阈值与运动重设.md`**；
  数据特性分析：`log/FEATURES_20261006_postsway.md`；现行参数速查：`doc/设计/过门逻辑树.md` §8。
- 提交栈（逐级可回档）：`005b91b`（tag `checkpoint-pre-gate-tune`，动手前）→ `e28aea2`（阈值+机制第一轮）
  → `db673cd`（`through` 两段式）→ `74edf13`（快冲 2s + 慢冲 4s）→ `1750e88`（三项确认不改）
  → `802840b`（`sway` 顶满：`kp 8→1` + `bias 0.138`）。
- 三件事值得单独记住：
  ① **基准换算会静默改变一切** —— 日志期的 `dx/dy` 按画面中心算，主点是 (702.27, 409.14)，
     两套基准差归一化 (0.0973, 0.1365)；纵向另有 `dy_target=−0.30` 的零点偏置，
     **必须减在算 `dyn` 的三处**（不是只减 align）；
  ② **别用"没触发"当依据** —— `sway` 的 kp 问题与纵向零点问题都是**分布**看出来的
     （分箱 / 段长 / 符号翻转频率），不是看哪条出口没走通；
  ③ **共用的东西会连带** —— `comm.motion.pid_sway` 是 ball/gate 共用，改它等于改撞球的横向回路
     （"一处调、两个任务同时生效"是既定不变量，有用例钉住）。
- 本轮新增工具：`tools/analyze/log/replay_gate_log.py`（把日志重建成 Det 回放 `GateTask`，
  给"改前/改后"的行为一致率；`--ref=axis|img`、`--old`、`--selftest`）。

## main.py

### `psi_line` 的符号约定
- 2026-09-27 · 现场定下 psi（门法向相对光轴夹角）的方向约定 · 结论 → `+` ⇒ 该左转、
  `-` ⇒ 该右转；**没有测量时显示 `--`，别显示 0**（0 会被现场误读成"已经正了"）
### 任务逐帧日志里的遥测字段
- 2026-09-23 · 需要判"转向方向 / 摆动极性"时，只能靠"命令 vs 遥测 vs 视觉"三者对齐；
  stdout 的 `[UART←]` 没有时间戳、事后对不上帧 · 结论 → 把遥测写进 JSONL 逐帧核对
  （`tyaw`/`ttel`/`tdep`/`trol`/`tpit`）

### _init_video（代码注释搬出）
- try/except 拦不住）→ 先自己确认本地 X socket 存在，别让浮窗把整船任务搞崩

### _uart_status（代码注释搬出）
- 兜底用 base/hw/uart.py 里那个**与 cfg 同值**的常量（别再写 0.3：那会让画面
- 显示的限深阈值与真正生效的保护阈值不一致，现场会被自己的叠加骗）

### _draw（代码注释搬出）
- → hdg/hdg_i 一起显示便于现场核对

### _draw（代码注释搬出）
- 偏转角单独一行、字号更大、带颜色 —— 现场最常盯的就是它

### _draw（docstring 搬出）
- 不再调 hub.detect_all：否则 gate 任务会额外跑一遍 ball 权重，帧率腰斩。

### psi_line（docstring 搬出）
- `+` ⇒ 该**左转**、`-` ⇒ 该**右转**（2026-09-27 现场定，方向依据见
- doc/注释历史.md）。只有 `full` 帧测得出来。
- · 没有测量时显示 `--`（**别显示 0** —— 0 会被现场误读成"已经正了"）；
- · `|psi| ≤ tol` → 绿（已收敛）；否则黄（还要转）；`hdg_skip` 有值 → 橙（这一趟跳过了正航向）。

## task1_2/run_ball_reverse.sh

### 日志前缀 `AUV_LOG_TAG`
- 板子时钟不准 · 结论 → 用 `AUV_LOG_TAG=xxx` 前缀区分轮次，**别用 date 命名**
### 转角原语的位置
- 2026-09-18 · 转角原语从本脚本内联移到 `common/motion/turn_deg.py`（公用运动原语）·
  结论 → 脚本改为调用 `python3 common/motion/turn_deg.py`（不再自带一份）
### `timed_dof` 收尾必须 `stop_hard`
- 现场踩过 · 一关串口船就锁在最后一个还在动的轴字节上（下位机无帧超时停车）·
  结论 → 每步结束 `u.stop_hard(verify=False)` 连发中性帧走完 ramp

## task1_2/ball.py

### 运动链的来源
- 早期 · 三段式运动移植自一个早期独立实验目录，**该实验目录现已删除**；视觉识别一直用本项目 ·
  结论 → 注释不再提移植来源（只保留"视觉识别用本项目"）
### 丢目标阶梯
- 早期思路 · 阶梯本身沿用至今（短时丢失 → 保持/惯性 → hold → 回 SEARCH）·
  结论 → 注释去掉"沿用原思路"这类沿革说明，行为不变

### (模块)（docstring 搬出）
- 运动链（三段式运动；**视觉识别用本项目**）：

### (模块)（docstring 搬出）
- CENTER   看到球：**surge=0、sway=0**，仅 yaw(edge_yaw PID) + heave(motion.pid_heave)，
- DASH     面积(EMA) ≥ dash_ratio 连续 dash_confirm_frames 帧 → **surge_fast**（分级里

## preview_detect.py


### (模块)（docstring 搬出）
- 在板子上实测某个 .bin 权重（默认 cfg vision.model.path，即 auv_multi.bin）

### (模块)（docstring 搬出）
- PC 端： ffplay -fflags nobuffer -flags low_delay -framedrop -f mjpeg "udp://@:5000"
- python3 preview_detect.py --classes gate                       # 只打印统计(无显示)
- python3 preview_detect.py --classes gate --model models/auv_multi.bin --conf 0.4
- python3 preview_detect.py --gate-kpt --conf 0.3 --duration 20  # 降低角点置信度门槛

## 2026-10-01 · A 域（原识别方案）退役归档


- **用户定**：「不用 A 域了，现在用的都是 **D+wb** 域」⇒ A 域（`chain: A` + `clahe 0.5` + AUV_4 权重 +
  老内参 fx=782.5）整体退役。
- **归档**：`tools/deploy/Adomain/`（`sync_Adomain_repo.sh` + `Adomain_cfg/` + `board_forks_Adomain.txt`）
  与 `cfg/backup/front_camera_AUV1_water_fx782.yaml` → **`bak/retired/Adomain_20261001/`**（附 README 说明用途/恢复法），
  并在 `bak/README.md` 登记。整包回退点 `bak/snapshots/snapshot_20261001_0040_pre_Adomain_retire.tar.gz`。
- **活代码同时清掉**：`common/vision/preprocess.py` 的 `chain: A` 分支与 `chain` 参数（含只被它用的 `_maps_for`）；
  `cfg/vision.yaml` 的 `image.chain` 键；`tools/check/pipeline/check_domain.py` 从「两域指纹」改成
  **D+wb 单域代次指纹**（链序 / wb 增益 `[1.0,1.05,1.15]` / CLAHE 0 / gamma 0.85 / 权重 md5 `ca5ba84f` / fx 1207.6 同代），
  并删掉依赖 A 域参考副本的「A vs D 逐像素等价」比对；`tests/platform/test_common.py` 的
  `test_preprocess_chain_domain_switch` → `test_preprocess_is_d_chain_only`（旧开关复活即红 + D 链序逐像素钉住）。
- **顺带清掉的死代码链**（读码发现，见 `doc/设计/过门逻辑树.md` §7.1）：`_near_lost()`（近距直冲出口已按用户要求移除）
  与只被它调用的 `_fresh_z()` ⇒ `_z_ms` 变成只写不读、`comm.gate.z.near_lost_m` / `z_stale_ms` 两个键**事实上不生效**
  → 一并删除（`_D_Z` 与 `cfg/comm.yaml` 成对改，守卫用例 `test_gate_defaults_match_cfg` 钉住）；
  `gate/gate_task.py` 模块 docstring 的「直冲出口三条」改为**两条**（① z≤cross ③ loiter 兜底）。
- **注释残渣修复**：上一轮"注释瘦身"按行删掉了多行注释的前半段，留下半句（`_D_HDG`、`_search_sweep` 波形表、
  `_pick_gate` docstring、`gate_task.py` 的重复赋值/粘连注释）→ 已按语义补回；同类残留若再发现，按"整段落一起搬"处理。

## manual.sh（脚本注释搬出）


```bash
#   确需板端本地录像时再加 --record（例如 PC 不在场）。
#   三件套在 manual/：manual/udp_server.py（遥控桥）· manual/recorder.py（录像）· manual/stream.py（推流/接收库）
#
# 推流是"零转码"的：base/hw/camera.py 打开相机时顺手把原始 MJPEG 交给 manual/stream.py 发出，
# 所以**推流与录像共用同一路采集，互不抢相机**（UVC 只允许一个进程取流）。
#
# 用法（在工程根目录执行）：
#   ./manual.sh                          # 遥控桥 + 推流（录像在 PC 端做）
#   ./manual.sh --sim                    # 无串口硬件时：遥控桥只打印
#   ./manual.sh --no-udp                 # 只推流
#   ./manual.sh --record --seconds 60 --out rec.mp4    # 额外做板端本地录像
#   ./manual.sh --host 192.168.137.2 --port 5000 --stream-fps 30
#   遥控包额外支持相机切换（PC 端发到端口 9000）：cam:front / cam:down / cam（前后切换）
#   ./manual.sh --help
#
# 水面 PC 端看画面（任选）：
#   ffplay -fflags nobuffer -flags low_delay -framedrop -f mjpeg "udp://@:5000"
#   python3 -m manual.stream view --record # 本仓库自带：看画面 + 存到 PC 的 record/ 目录
#   python3 -m manual.stream view --ctrl-host 192.168.137.10  # 看画面，按 Tab 切主视/下视
#   浏览器（板端另开 HTTP 时）：python3 -m manual.stream push --host <PC> --http 8080
# 手动接管前先清理残留的视觉任务：main.py 会持续向串口发帧并占用相机，
# 导致手动遥控帧被压掉 / 推流打不开（表现为“切成手动也没法控制”）。
# 2) 录像（前台；推流由 camera.py 钩子顺带完成）或纯推流
```

## preview_detect.py（脚本注释搬出）


```
- 本脚本 **不创建 UartController**，不打开串口 → 船不会动；
- 每帧做一次推理，把识别框画在实时画面上。
显示方式（可任选/组合）：
--show            本机窗口 cv2.imshow（需要 DISPLAY；板子桌面终端可用）
--stream          把“带框画面”编码成 MJPEG，用 UDP 推给 PC 观看
--save DIR        每 N 帧存一张带框图到 DIR（默认每 30 帧）
门角点(关键点)模式 —— 用 gate 任务模型（`vision.model.task_models.gate.path`，当前为阶段一
按 Ctrl-C 或窗口里按 q/Esc 退出；结束打印各类别累计帧数。
```

## 2026-10-02 · 脚本注释瘦身（第三轮）+ 门禁正则 bug 修复


- **脚本注释瘦身**（沿用"注释只留普通说明 + 必要提示，大段描述进本文件"）：
  shell 侧 6 个脚本（`tools/deploy/{deploy_to_board,check_board_parity,tidy_board_bak,archive_baks,archive_board_legacy}.sh`、
  `manual.sh`）搬出 145 行注释；`tools/**/*.py` + `preview_detect.py` + `manual/*.py` 共 22 个脚本搬出 254 行 docstring/注释。
  规则：模块 docstring ≤6 行（用途 / 用法 / 必要提示 / 一行指针），**heredoc 内部一律不动**（那是脚本生成的输出）。
  校验：shell —— 去掉注释行后**代码逐字节相同** + `bash -n`；python —— docstring 归一化后 **AST 逐字节相同** + 逐工具 `--help` 冒烟。
- **`tools/deploy/check_board_parity.sh` 的正则 bug**：判"文件在不在清单里"用了
  `grep -qE "[[:space:]]${f}[[:space:]]*\]?[[:space:]]*$"` —— 把**路径当正则**，
  于是路径里带 `+`（如 `2026-10-02-…（下位机转向+航向确认）.md`）的文件会被误报"不在清单里"。
  已改成 `awk` 精确字符串比较（同时兼容带方括号的"板端实测"条目）。
- **`tools/presets/`**（DSH 预设与技能，agent 侧工具）**不进板端清单**，也不参与清单一致性扫描。
- **`tests/_bite_probe.py`** 随第三方 2026-10-02 的用例改名/改设计同步：
  loiter 探针改咬 `loiter.dx_max`（`timeout_ms` 不再是主闸）、上限探针改咬 `hdg.turn_scale`
  （"一门只转一次"后 `max_turns` 降级为兜底、咬不动了）；`check_hdg_lockup.py` 不再写死 `max_turns=4`。

## 为什么用 CV（2026-10-06 板端实测结论，见 `grab/README.md`）

RDK X5 上撞球那版权重（`yolo11n_detect`）对**远距小球**只有 33–42% 命中 —— 而那正是夹取任务的
接近段；同一个球 CV 在板端只要 ~14 ms/帧（71 FPS）。所以夹取小球这条线用 CV 做感知，
**把 BPU 让给门 keypoint**。

## 算法（每条都对应下面一段代码，参数在 `cfg/vision.yaml` 的 `grab.cv`）

| 步骤 | 做什么 | 为什么这么做 |
|---|---|---|
| 1 红色掩膜 | `R-max(G,B) ≥ dom_min` **或** `(R-max(G,B))/(R+max(G,B)+1) ≥ rel_min`，再叠色相/饱和度/亮度门限 | 水下红光衰减 ⇒ 暗球绝对差会掉到阈值以下、比值却仍高，所以两条判据取并；色相门限把橙色导轨（H≈20–30）挡在外面 |
| 2 填高光孔 | 只填"面积 < `fill_hole_frac` × 父轮廓"的孔 | 镜面高光把红球打成红环，不填则圆度/质心全错；真背景孔比球大，不会被误填 |
| 3 几何过滤 | 面积 / 圆度 / 凸性(hull_fill) 下限 | 滤掉瓦缝红光、池底斑点这类细长或碎块 |
| 4 鲁棒圆拟合 | Kasa 代数拟合 + 迭代剔离群点，并要求边界**张角 ≥ `min_arc_deg`** | 球常被黑瓦条挡成月牙：月牙的弧仍是真圆的一段，拟合成立；而"细长条"能拟出任意巨大的圆，用张角拦住 |
| 5 覆盖校验 | 拟合圆内红色掩膜占比 ≥ `cov_min` | 全盘 ≈0.95、被挡月牙 ≈0.5；乱拟合会很低 |
| 6 去重 | 圆 IoU > `merge_iou` 的只留高分 | 同一球被切成几块时会出多个圆 |

## 板端性能（A55，1280×720，700 帧实测）

`fast_mask=True`（默认，板端必用）与 `fast_mask=False`（numpy 参考实现）**逐帧结果完全一致**，
但板端耗时 **15 ms vs 68.6 ms** —— numpy 版每帧在做大数组 `astype(int16)`(20.5 ms) 与 float32 除法(8.6 ms)。
| 换成什么 | 为什么 |
|---|---|
| `cv2.split/max/subtract/compare`（uint8） | 少一次 2.7 MB 的 int16 分配与转换 |
| **256 项 LUT**：`r ≥ ceil(((1+t)·mx+t)/(1-t))`，`cv2.LUT` + `cv2.compare` | 与除法判据**逐位等价**，避开 `cv2.multiply(dtype=CV_16U)`（A55 上是标量循环，30 ms） |
| `cv2.inRange` 做 HSV 门限 | 全图逐像素比较换 SIMD |
| 形态学只在掩膜 bbox+margin 的 ROI 里做 | 全图 `open(5)+close(9)` 在 A55 上 35.8 ms，掩膜通常只占 <2% |
花式索引做"稀疏门控"（后者把掩膜算到 97.6 ms）。
# --------------------------------------------------------------------------- #
# 参数
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 形态学核（缓存：每帧 getStructuringElement 是纯浪费）
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 鲁棒圆拟合 / 质量度量
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 主检测
# --------------------------------------------------------------------------- #
```

## 为什么需要它

板端 A55 上**每一次全图（1280×720）uint8 pass ≈ 2.6 ms**（内存带宽受限，不是算力受限）：
全图掩膜约 26 ms、整帧 `detect()` 45–64 ms。而夹取是**跟踪**任务 —— 球在两帧之间的位移远小于
半径（本数据集实测球被夹爪带走时 ≈40 px/帧，球半径 25–270 px），所以**只有首次/跟丢后才需要全图搜索**，
之后每帧只在"上一帧球心 ± margin"的 ROI 里跑同一套检测。ROI(≈230²) 的 `detect()` 只要 4–14 ms。

## 正确性口径（必须写清，否则就是悄悄掉精度）

* ROI 含**完整球**时，结果与全图检测**完全一致**（同一套算子，只是作用域变小）。
* ROI 边界会引入人工 0 邻域 ⇒ **贴着 ROI 边界的检出不可信**；这时（以及 ROI 内无检出时）
立刻回退一次全图检测，`mode` 会记成 `fallback`。回退率是可观测指标，别让它静默涨。
* `proc_side>0` 时启用**尺度归一化**：ROI 一律缩放到 `proc_side²` 再检测，单帧成本与球大小**无关**
（否则近距大球 ROI 大、反而比全图还慢 —— 这是实测踩过的坑）。代价：中心精度按比例下降
（proc=256、ROI=664 时 1 px ≈ 2.6 px 原图），实测中心偏差 p50 0.71 px、半径比 0.998，夹取足够。
```

## main.py（脚本注释搬出）


```
dets 用**当前任务本帧已算出的检测结果**（见各 task.last_dets），
深度来自下位机 14B 遥测帧（base/hw/telemetry.py）；`guard` 用 comm.depth_guard：
开启且当前深度 ≤ min_depth_m 时禁止上浮（base/hw/uart.py::_apply_depth_guard）。
—— 主循环只取**最新帧**（非阻塞），绝不被下视相机 read 拖卡，不碰前视(主视)显示。
`psi`（`last_info["hdg"]`）= **门法向相对光轴的夹角**：0 = 机身正对门；
—— 下视 USB read 可能 50~120ms，若放进主循环会把前视(主视)显示一起拖卡/冻死。
状态流：IDLE → [选定任务依次] → DONE；任意时刻 Ctrl-C/SIGTERM → 急停。
任务分工（按目录分区）：
- 任务一 撞球          task1_2/ball.py（BallTask）—— 前视相机
- 任务三 过门          gate/（keypoint 四角 + PnP，相位机见 gate/gate_task.py（总调度））
        # 任务逐帧日志（AUV_TASK_LOG=<path>）：把 last_info 每帧存一行 JSON，
        # 供 tools/analyze/log/analyze_task_log.py 离线判读（phase/action/z/dx/dy/kpt/ratio/pass…）；
        # 默认不开，不影响运行。
            # 下位机遥测也写进日志：**转向方向/摆动极性这些只能靠"命令 vs 遥测 vs 视觉"
            # 三者对齐来判**，stdout 的 `[UART←]` 没有时间戳、事后对不上帧；写进 JSONL 后
            # 就能逐帧核对（`tyaw`=遥测绝对航向(°)、`ttel`=遥测帧年龄(ms)、`tdep`=深度(m)、
            # `trol`/`tpit`=横滚/俯仰）。
            # 任务全部结束：再持续发 stop 保持 done_hold_ms，确保"稳定保持停止"后才退出
            # （撞球命中后不会刚停就关串口；DASH 后的 STOP 相位已在任务内保持）
```
