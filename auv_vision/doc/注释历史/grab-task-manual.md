# 注释历史 · grab/** · task/** · manual/** —— 夹取任务与手动模式


> 本文件由 `doc/注释历史.md`（4128 行）于 2026-10-05 按域拆出；**内容一字未改**，只搬了位置。
> 顶层索引与"旧路径怎么解析"见 [`doc/注释历史.md`](../注释历史.md)。
> 代码/配置里的注释只写"是什么 + 当前值"，注意事项、实测数字与历史沿革一律记在这里。

## manual/stream.py


### Sink（docstring 搬出）
- save_fps=0 时按前若干帧实测帧率再建 writer（避免"30fps 写 25fps 流"造成的时长失真）。

### open_raw_camera（docstring 搬出）
- 注意：OpenCV 的 V4L2 后端默认协商 **YUYV**（本相机 720p YUYV 只有 9fps），

### (模块)（docstring 搬出）
- 板端发送（零转码）：相机输出 MJPEG，本模块把**原始 JPEG 字节**原样转发，不解码不重编码。

## manual/recorder.py


### (模块)（docstring 搬出）
- python3 manual/recorder.py --camera front --frames 60 --out rec/rec.mp4 --overlay

## manual/udp_server.py


### (模块)（docstring 搬出）
- Run this script on the RDK. It receives PC keyboard commands over UDP and sends

---

## manual/recorder.py（脚本注释搬出）


```
采集指定相机（sim/usb/mipi 占位）并保存视频：
后端 1（推荐，需 cv2）：mp4/avi（mp4v），带可选时间戳叠加
后端 2（无 cv2 时）：逐帧 .npy 落盘 + manifest.json，之后用
python3 recorder.py --assemble <帧目录> [-o out.mp4]  补出视频
python3 manual/recorder.py --camera front --seconds 10 --out rec.mp4
python3 manual/recorder.py --assemble frames_dir --out rec.mp4
```

## manual/stream.py（脚本注释搬出）


```
板端收到 `cam` 就切换主视/下视（见 manual/cam_switch.py::parse_camera_packet）。
0.3s 防抖：Tab 长按/连按也只切一次。
返回 (cap, mode)：
mode == "raw" → read() 返回 (1,N) uint8 原始 JPEG（零解码，可直接转发）
mode == "bgr" → 相机不支持原始 JPEG 输出，read() 返回 BGR（推流需重编码）
PC 端接收：按 JPEG 边界跨 UDP 数据报重组，显示/录制/统计；或直接拉 HTTP MJPEG。
给 `base/hw/camera.py` 用的接口：
pusher = stream.MjpegPusher(host, port, pkt=8000, stream_fps=30)
pusher.offer(jpeg_bytes)        # 非阻塞，只保留最新帧
给 `manual.py` 用的接口：
open_raw_camera(...), CameraSource, SimSource,
MjpegPusher, MjpegHttpServer,          # 发送
UdpReceiver, HttpReceiver, Stats, Sink # 接收/显示/录制
完整方案见 doc/记录/前视USB相机低延迟推流方案.md。
```

## manual/udp_server.py（脚本注释搬出）


```
udp_server.py - RDK-side UDP remote-control bridge for keyboard testing.
the existing 0xA5 11-byte RC-compatible frame to STM32 through uart.py.
Packet format:
surge,sway,heave,yaw
Each value is a float in [-1.0, 1.0].
1,0,0,0      forward
0,-1,0,0     sway left
0,0,1,0      heave up
0,0,0.0,0.6  yaw right
stop         neutral
Typical RDK command:
python3 udp_server.py
Dry run without STM32 serial hardware:
python3 udp_server.py --sim
Telemetry (STM32 -> RDK): UartController 收 14B 遥测帧(0xAA55 + 深度/姿态 + 校验和，
深度 ≤ min_depth_m(默认 0.3m) 时禁止上浮——键盘按"上浮"也不会把机身顶出水面。
遥测打印：[UART←] depth=...（节流 comm.debug.tel_log_ms）。
```

## grab/__init__.py（脚本注释搬出）


```
现状（2026-10-06）：**只有感知层**（纯 CV 红球检测 + ROI 跟踪，不走 BPU）。
决策/相位机尚未写 —— 先接感知，用 `pytest tests/tasks/grab` 验证后再设计运动链。
依赖规则与全工程一致：本子包只 import `base` / `common` 与同级任务模块。
（`__init__` 刻意**不 eager import** 子模块：这样 `grab.percept.cv_ball` 可以脱离工程单独 import，
便于 PC 侧离线验证与调参，与 `gate/__init__.py` 同做法。）
```

## grab/percept/cv_ball.py（脚本注释搬出）


```
月牙的有效边界只是一段弧 —— 弧必是真圆的一段，拟合成立；但**细长条**的边界也能被
最小二乘拟合成一个巨大的圆（残差还很小），此时张角很小。用它拦这类"解不唯一"的退化拟合。
真背景孔比球还大，不会被填；这一步是水下红球的关键（高光把球打成红环）。
正确性：open/close 的输出都落在"输入 bbox 外 k/2 之内"（两者都是 dilate 的子集），
取 margin = k 必然覆盖；ROI 边界外的邻域在全图里同样是 0，故边界处结果也一致。
bbox 由 1/8 缩略图求 nonzero 得到（`INTER_AREA` 是块均值，块内任一像素非零 ⇒ 均值>0，
所以缩略图 bbox 是真实 bbox 的超集），在 ~14k 像素上求 nonzero 很便宜。
板端 A55 实测（1280×720）：本函数 **~15 ms**，numpy 参考实现 **68.6 ms**。
判据 `rel = dom/(dom+2mx+1) >= t`（把 r = dom+mx 代进去）等价于
dom > lut(mx),   lut(mx) = ceil( t·(2mx+1)/(1-t) ) - 1
查表 + 一次 `cv2.compare(..., CMP_GT)`（都 uint8 SIMD）。
**为什么用"严格大于"而不是"大于等于"**：t 较大时所需 dom 下限会超过 255（t=0.4 时最大 341），
uint8 LUT 表达不了"不可满足"。用 `>` 天然能编码：lut 置 255 ⇒ `dom > 255` 恒假（dom ≤ 255）
⇒ 恰好是"不可满足"，不引入假阳性。`>=` 写法只能 clip 到 255，会把 r=255、mx≥138 的像素
（如纯白 dom=0）错误放过。
为什么绕这一步：A55 上 `cv2.multiply(..., dtype=CV_16U)` 是标量循环，除法判据要 30.3 ms，
LUT + compare 只要 ~2 ms。整数 ceil 用整除算：`t=a/b` ⇒ `ceil(a·n/(b-a)) = (a·n+(b-a)-1)//(b-a)`；
**别用 float 走 np.ceil**（会重新引入舍入，实测能让 13/65536 个格子差 1 档）。

## grab/percept/ball_tracker.py（脚本注释搬出）


```
跟踪时只在 ROI 里检测；ROI 无可信结果或跟丢超过 max_lost 帧时，本帧直接做全图搜索
（`mode` 记 'fallback' / 'full'，便于统计回退率）。
参数（`cfg/vision.yaml` 的 `grab.track.*`，缺项用这里的默认值）：
margin      ROI 在球外每边多留的像素（默认 64 ≈ 1.6× 实测最大帧间位移）
proc_side   >0 = 尺度归一化目标边长（默认 256）
max_lost    连续丢多少帧后放弃 ROI、回到全图搜索
edge_margin 检出圆离 ROI 边界近于该值(px) 判"不可信" → 回退全图
min_roi     ROI 最小边长（球很小/刚出现时也别裁太小）

## grab/percept/grab_detector.py（脚本注释搬出）


```
撞球相位机用的是 bbox 面积占比；夹取若要复用同量纲的判据（如"够近了"），用这个函数，
否则 bbox 占比会把圆形目标的面积高估 4/π≈1.27 倍。
装配层用法：
backend = build_grab_backend()
hub.register("grab", backend)
if backend is None:
print("[MAIN] ⚠️ grab 后端未启用（grab.detect.mode: mock）")
工作方式（`grab.track.enable`）：
True （默认）ROI 跟踪 —— 首次/跟丢后全图搜索，其余帧只在球附近 ROI 检测；
False        每帧全图检测 —— 慢（板端 15 FPS）但无状态，便于对照与调试。
返回的 `Det.kind` 用 `grab.kind`（默认 `red_ball`，与工程 `model.labels` 对齐）。
工程里所有感知都返回 `common.vision.detector.Det`（kind/score/x/y/w/h），
装配层用 `hub.register(task, backend)` 注册任务专用后端。本模块提供那个 backend：
# main.py 的 AppController.__init__（装配层唯一改动点，2 行）
if name == "grab":
self.hub.register("grab", GrabBallDetector())
之后 `hub.detect_list("grab", frame)` 就能拿到 `[Det]`；需要圆心/半径（夹取真正要用的量）时
调 `detector.circles(frame)`。
为什么不用 bbox 而保留圆：**夹取要的是球心与半径**（对准/接近判据），
`Det` 的 bbox 只是与工程其它任务对齐用的外接方框（圆心 = `Det.center`，两者一致）。
# --------------------------------------------------------------------------- #
# 供装配层复用的小工具（与 `gate/percept/gate_detector.py::build_gate_backend` 同形）
# --------------------------------------------------------------------------- #
```

## manual/cam_switch.py（脚本注释搬出）


```
`switch("front"|"down")` / `toggle()` 会**先关旧相机再开新相机**（不能两路同开，见模块头注释）。
接受：`cam:front` / `cam:f` → "front"；`cam:down` / `cam:d` → "down"；
`cam` / `cam:toggle` / `cam:t` → "toggle"；其余 `cam:*` → "bad"。
手动挡下，启动时用 `--camera front|down` 选一路（`udp_server.py --stream-host ... --camera X`）；
PC 端只被动接收，不做运行时切换。
（`lsusb -t`：都在 Bus 01/480M 同一个 Hub 下），**同时打开必超等时带宽** —— 不管打开顺序，
**后开的那路 `read()` 永远返回 None**（实测 front=/dev/video2 与 down=/dev/video0 都如此）。
旧实现"两个一起开"会导致被选中的那路恰好是空的 → `frames_sent=0` → PC 完全没有画面。
所以这里改成"选哪路就开哪路"，切换时先关旧再开新。
用 `base.hw.camera.create_camera("front"|"down")` 开相机（走 cfg/vision.yaml 的 camera.* 配置，
无硬件时 fallback_sim），`manual.stream.MjpegPusher` 零转码 UDP 推流（与 manual.sh 共用同一套）。
```

## task/ball.py（脚本注释搬出）


```
SEARCH 相位下（含从未见过球、或阶梯走完回到搜索）→ 继续搜索，不再套 hold。
`common/motion/search_scan.py`：左 span → 右 2span → 左 2span …（默认 span=60° ⇒ 扫 120° 视角），
**手动 DOF + 遥测 yaw 闭环**（PID 用当年 turn_pid 那套值），非阻塞、检测照跑（"睁眼看"）。
取代原来的开环时间脉冲（`search_spin_s`/`search_pause_s` 已不再使用）。
SEARCH   无目标：原地**脉冲旋转**（spin_s/pause_s）+ 周期性慢速前进探测
（转 spin_s → 停 pause_s，停的间隙让检测有静止帧）
把球在**水平与竖直**同时居中
APPROACH 稳定居中后：surge **分级**前进（远 surge_fast / 近 surge_slow）+ **仅 sway**
做水平修正（motion.pid_sway）；**yaw=0、heave=0**
STOP     冲刺结束 → 全 0 保持 stop_hold_s（**稳定停住**）→ DONE("hit")
丢目标（阶梯）：短时丢失（帧窗口）→ 保持/惯性 → hold（原地全 0，
时限：`comm.ball.timeout_ms`（当前 cfg 30s；DASH/STOP 期间不打断，保证命中与停稳都能走完）。
参数：本任务特有的在 cfg/comm.yaml 的 `ball:` 段；**与过门共用的**（两套 PID、
        # 三套 PID（量纲都是"归一化偏差 ±1"）
        # ① 居中 yaw（撞球特有：门/球居中用旋转 + 限幅，靠近自动减速；过门居中只用 sway）
```

## task/run_ball_reverse.sh（脚本注释搬出）


```bash
# 转角由脚本机械控制、按固定序列执行，不再依赖视觉连续找门 —— 更可控、机械。
#
# 用法：  ./task1_2/run_ball_reverse.sh
# 可调（环境变量，默认值在括号里）：
#   AUV_WAIT_S(10)        待机秒数
#   AUV_DESCEND_S(0)      下潜秒数（0=不下潜）
#   AUV_FWD_S(0)          撞球前前进秒数
#   AUV_FWD_SURGE(0.35)   撞球前前进速度
#   AUV_REV_S(5)          撞球后回退秒数
#   AUV_REV_SURGE(-0.5)   回退速度（负=后退）
#   AUV_POST_FWD_S(2)     回退后前进秒数
#   AUV_POST_FWD_SURGE(0.35)
#   AUV_TURN_ANGLES       转角序列（空格分隔，带符号：正=左转 / 负=右转）
#                         如 AUV_TURN_ANGLES="90 -45 90 30"；个数=过门次数
#   AUV_TURN_TIMEOUT(20)  转向等完成反馈超时（秒）
#   AUV_BALL_SKIP(0)      1=跳过撞球任务
#   AUV_REV_ON_FAIL(0)    撞球非 0 退出时仍回退（1=回退）
#   AUV_LOG_TAG(run)      日志前缀（用前缀区分轮次，别用 date：板子时钟不准）
#                         → log/<tag>ball.jsonl / log/<tag>gate_<N>.jsonl / /tmp/<tag>path.csv
# ============ ★ 现场代填：每一次"转角度→过门"的转角（度）============
# 数组长度 = 过门次数；四扇门可以填四个不同的角度。这里就是操作者下水前手填的地方。
#    所以方向由本脚本按符号拆成 --dir，不靠负号本身）
```

## task/run_gate.sh（脚本注释搬出）


```bash
# 转角由脚本机械控制、按固定序列执行 —— "四个门当一个门四次过"。
#
# 用法：  ./task1_2/run_gate.sh
# 可调（环境变量，默认值在括号里）：
#   AUV_WAIT_S(10)        待机秒数
#   AUV_DESCEND_S(0)      下潜秒数（0=不下潜）
#   AUV_FWD_S(0)          下潜后前进秒数（0=不前进）
#   AUV_FWD_SURGE(0.35)   前进速度
#   AUV_TURN_ANGLES       转角序列（空格分隔，带符号：正=左转 / 负=右转）
#                         如 AUV_TURN_ANGLES="90 -45 90 30"；个数=过门次数
#   AUV_TURN_TIMEOUT(20)  转向等完成反馈超时（秒）
#   AUV_LOG_TAG(run)      日志前缀（区分轮次，别用 date）
#                         → log/<tag>gate_<N>.jsonl
# ============ ★ 现场代填：每一次"转角度→过门"的转角（度）============
# 数组长度 = 过门次数；四扇门可以填四个不同的角度。
```

## 2026-10-06 · `grab/` + `place/` 合并为 `handling/`（同一总调度两种模式）

用户定：夹取与放下**同一性质的任务**、复用同一套运动逻辑 ⇒ 整合成一个目录，并按 `gate/` 的样式
拆成「总调度 + 相位 / 关键动作 / 兜底值 + 感知」，根目录的调度**同时支持两种流程、可切换调用**。

- **新结构**：`handling/handling_task.py`（`HandlingTask(mode="grab"|"place"|"full")` + 兼容壳
  `GrabTask`/`PlaceTask`）· `handling/motion/{params,actions,phases}.py` · `handling/percept/`（原 `grab/percept/`）。
  方法体**一字未改**，只搬位置；生成器 `bak/migration/gen_handling.py`（可复现）。
- **`full` 模式**：夹取成功那一刻 `_enter_place()` 交接 —— `_stage` 切 `place`、`holds_ball` 带过去、
  相位落到 `PH_PLACE_INIT`；**同一实例**持有任务级限深下限与收尾，不需要重建对象。
- **合并时踩到的三个坑**（都已修，留作教训）：
  1. 相位常量重名：两边都有 `PH_INIT`/`PH_DONE` ⇒ 前缀化成 `PH_GRAB_*` / `PH_PLACE_*`；
  2. **`_step_init` 同名**：两个 mixin 都有它，MRO 让 `GrabPhases` 的赢 ⇒ place 模式会跑进夹取的 INIT
     （实测：`last_info["phase"]` 变成 `PITCH_UP`）⇒ 拆成 `_step_grab_init` / `_step_place_init`；
  3. `holds_ball` 语义冲突：夹取侧是**只读属性**（笼里有没有球）、放置侧是**写入方法**（交接告知）
     ⇒ 合并为只读 `holds_ball` + 新 `set_holds_ball()`（放置侧调用点已改名）。
- **配套**：`main.py` 注册 `grab`/`place`/`handling`（`build_task()` 按名字给 mode；相机 `down`/`front`；
  状态位新增 `STATE_GRAB`/`STATE_PLACE`）；`tests/tasks/grab/test_handling_modes.py` 新增三条模式用例
  （grab 不自动切放置 / full 交接 / place 从 INIT 起步）；清单与 `DELETED` 成对跟名（旧 `grab/**`、`place/**` 板端要清）。
- `doc/记录/` 里旧路径（`grab/grab_task.py`、`place/place_task.py`）按约定不改写，映射见下面的"旧→新"。

