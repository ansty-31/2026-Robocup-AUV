# 注释历史 · base/** 与 common/** —— 平台部件与跨任务公共件


> 本文件由 `doc/注释历史.md`（4128 行）于 2026-10-05 按域拆出；**内容一字未改**，只搬了位置。
> 顶层索引与"旧路径怎么解析"见 [`doc/注释历史.md`](../注释历史.md)。
> 代码/配置里的注释只写"是什么 + 当前值"，注意事项、实测数字与历史沿革一律记在这里。

## common/vision/preprocess.py


- 2026-09-23 · 换序（「D 域 / P2」）：remap@640 ≈6 ms vs remap@720p ≈11.5 ms，且与训练链路一致 ⇒ D 域选它。
- 2026-09-23 · 板端实测提速（输出逐像素不变）：白平衡由 float 乘/裁剪/回 uint8（640×640 上约 31 ms/帧）
  改为 256 项 LUT（约 12 ms）；CLAHE 对象与 gamma LUT 也改成缓存复用（原先每帧 createCLAHE / 重建 LUT）。
- 2026-09-23 · `calibration_maps_640`：几何上与「先 remap@720p 再缩放」等价（nk640 = S·nk720），
  但少一次 720p 的整幅读写，实测省约 5 ms/帧。

### process（docstring 搬出）
- ⚠️ 跨域掉点：拿 A 域权重喂 D 域图（或反之）角点会飘 —— 代次与链路的配对见
- `doc/记录/过门-状态机与参数.md` §9.5；板端两个仓库的分工见 `tools/README.md` §6。
- D 域（`chain: D`）的理由：① remap@640 比 remap@720p 省约一半时间；
- 差别只在性能与训练分布 ⇒ A 域不是"错的链路"，是**另一代权重的配套链路**。

### process（docstring 搬出）
- | **D**（默认/新方案） | `resize(640) → enhance → remap@640` | 阶段一 `g240_i16` | 0 |
- | **A**（原方案） | `remap@720p → resize(640) → enhance` | AUV_4 `d38b803e…` | 0.5 |

### enhance（docstring 搬出）
- 白平衡由 float 乘/裁剪/回 uint8（640×640 上约 31 ms/帧）改为 256 项 LUT（约 12 ms）；clip==0 时再把 gamma

### calibration_maps_640（docstring 搬出）
- nk720 = getOptimalNewCameraMatrix(K, dist, (RAW_W,RAW_H), 0, (RAW_W,RAW_H))

### (模块)（docstring 搬出）
- 两序**几何等价**，差别是性能与训练分布；**链路必须与权重同域**，否则跨域掉点。

## common/motion/turn_deg.py


- 2026-09-27 · 极性 σ 的定法：固件约定 = **船右转时原始 yaw 减小**（板端实测，见 `log/_board_gate_one_latest.jsonl`；
  命令 +0.45 右转、遥测 yaw 从 75.07 一路降到 9.93）。当前 cfg 两个旋钮都是 +1 ⇒ σ = −1。
- 2026-09-23 · 旧版靠"0.6 s 固定舵探向"定这个符号 —— 那段是开环的，本身就会白转十几度
  （实测目标 13.7° 时探向转了 18.79°），已删除。

### (模块级)（代码注释搬出）
- 固件约定：**船右转时原始 yaw 减小** ⇒ 该常量 = -1（一次实测确定，换固件/IMU 才要复核）。

### (模块级)（代码注释搬出）
- ⚠️ kd 必须 0、deadzone_deg 必须 ≥ 执行器死区折算角（见 cfg/comm.yaml 的 turn_pid 注释）：

### (模块级)（代码注释搬出）
- sat_max_s：**满舵保护**（用户 2026-09-27 定）——命令饱和到 ±out_max 连续多久 ⇒ **立刻停**。
- 为什么必须：舵效不对 / 遥测不动 / 被外力顶住时，闭环会一直满舵（现场"一开始就转个不停、

### (模块级)（代码注释搬出）
- 否则慢船的正常长饱和会被误杀（用例 test_motion 实测过）。
- 0 = 关掉这条保护。实测：健康转向的饱和段通常 < 0.5s。

### (模块级)（代码注释搬出）
- overshoot_tol_deg：**超转容差 = 10.0°**（用户 2026-09-28 定："逐步收敛，可以超转但超转不得超过 10 度"）
- —— 判据的基准是**下发的 deg**，不是 ψ：实测**超转** `0 ≤ done_deg − deg ≤ 10°` 就算这一小步到位

### (模块级)（代码注释搬出）
- 为什么必须：满舵保护原来只看"|err| 有没有变小"，而**超转会让 |err| 变大** ⇒
- 一次正常的轻微超转会被误判成"没进展/转反了"而把转向掐死（用户："不应该这么被限死"）。

### step（代码注释搬出）
- ★ 满舵保护（用户 2026-09-27 定）：命令饱和到 ±out_max 持续 `sat_max_s` ⇒ **立刻停**。
- 加在"算完 out"之后、"到位/超时"判断之前 ⇒ 它比超时更早生效（现场要的就是这个）。

### step（代码注释搬出）
- 进展有两种口径，取更好的那个（用户 2026-09-28 定：基线是**下发的 deg**）：

### step（代码注释搬出）
- ★ 到位判据有两条，满足任一 + 连续 3 帧：

### step（代码注释搬出）
- —— 用户 2026-09-28 定："超转容差的目标是下发的 deg 才行"。基准是**我们下发的角**，

### remain_done（docstring 搬出）
- `done_deg` 是"朝目标方向实际转过的角度"（负 = 转反了）⇒ 未转够时为正、超转时为负；

### stop_hard（docstring 搬出）
- 为什么必须（水里实测 + 代码核实）：`base/hw/uart.py::_ramp_step` 对 yaw/surge/sway/heave 做

### (模块)（docstring 搬出）
- （角速度随电量/水流/负载变），所以把"还剩多少度"喂给 PID —— PID 只负责**逐步收敛到目标角**。
- **极性（σ：`psi = σ × 遥测 yaw`）是定死的常量，不现场测**，和普通 yaw 环一样：
- 固件约定 = **船右转时原始 yaw 减小**（一次实测确定，换固件/IMU 才要复核）。
- 当前 cfg 两个旋钮都是 +1 ⇒ σ = -1。
- 旧版靠"固定舵探向"定这个符号，已删除（开环探向本身会白转十几度，实测数字见 `doc/注释历史.md`）。
- kp 管"多早开始收力/收敛多紧"，out_max 管"最大转速"（想整体更慢就降它，别只降 kp；
- ⚠️ 别降到 0.15 那一档：15% 推力恰卡在执行器死区 0.138 边上 ⇒ "还剩二十几度就没推力"）。
- ｜ 6=**方向自证失败**（命令朝一边、船朝另一边 ⇒ 停转，查极性/接线）

## base/cfg/settings.py

### 路径解析（`_PATH_KEYS` / `resolve_path`）
- 2026-09-22 · cfg 里曾写绝对路径；板端有两份工程拷贝（`/home/sunrise/AUV` 旧副本 与
  `~/Desktop/AUV_New` 在用副本），绝对路径把标定/权重钉死在某一份上——那份被删/被改/被搬，
  管线立刻崩 · 结论 → cfg 路径键一律写仓库内相对路径，由本文件按工程根解析；
  绝对路径仍接受（老配置不改也能跑）但不推荐

### (模块级)（代码注释搬出）
- ⚠️ 本文件是**故意的本地/板端分叉**（出处 tools/deploy/board_forks.txt）：

### _resolve（代码注释搬出）
- 运行时在这里统一按**工程根**解析成绝对路径。为什么这么做（2026-09-22 踩过）：

### _resolve（代码注释搬出）
- 绝对路径会把标定/权重钉死在某一个拷贝上 —— 那份被删/被改/被搬，管线立刻崩。

## base/hw/telemetry.py

### `TelemetryReceiver.feed` 里的 `yaw_sign` 归一
- 2026-09-23 · 板端真实日志（SIM=false）转向探向：命令 `yaw=-0.30`（左转档）×0.6s →
  遥测 Δyaw=+18.79° ⇒ 必然是"命令符号→物理转向"与"物理转向→遥测符号"两者之一反了
  （探向只能测到二者之积 g·s=-1） · 结论 → 用 `comm.telemetry.yaw_sign` 归一；
  归一后探向应报 `imag_sign=+1`，且 `[HDG] 转向后 |psi| 应变小`；
  闭环在 g·s=-1 时仍稳定，所以这个错**不会自己暴露**，必须靠探向核对

### feed（代码注释搬出）
- ⚠️ 别把 yaw_sign 留在错的一侧：探向只能测到"命令符号→物理转向"与

### feed（代码注释搬出）
- 但 `heading_align` 驱动的是镜像量，船会**朝反方向转**（现场现象：左转后机身

### feed（代码注释搬出）
- （2026-09-23 板端实测依据见 doc/注释历史.md）

### TelPlayback（docstring 搬出）
- ⇒ HDG 的闭环（PID/超时/方向自证）与深度保护才有机会复现当时的行为。

## base/hw/uart.py

### `_D_MIN_DEPTH_M`（限深兜底阈值）
- 2026-09 之前 · 兜底常量原先写 `0.3`，而现场定死 `depth_guard.min_depth_m = 0.55`；
  一旦配置读不到，机身可以从 0.55 一路浮到 0.30 —— 露出水面即本次比赛立即停止 ·
  结论 → 兜底常量必须与 cfg 同值（`0.55`），由用例
  `test_base.py::test_depth_guard_fallback_matches_cfg` 钉住相等
### `stop_hard` / `close`（为什么不能只发一帧 neutral）
- 现场踩过 · `_ramp_step` 做字节级平滑，`force=True` 只绕心跳节流、不绕 ramp → 单帧中性
  发出时 `yaw 84→128 只走到 ~95`（仍是 −0.26 舵）；下位机又没有"无帧超时停车"，
  `close()` 一关串口船就锁在最后一个还在转的字节上，一直转/一直倒车 ·
  结论 → `stop_hard()` 连发中性帧走完 ramp + 用遥测 yaw 闭环验证；`close()` 前轴不在中位
  必须先硬停
### `_apply_dive_boost`（下潜动力单独放大）
- 背景实测 · DOF→字节→下位机 `RC_Matching*`（映射值 ≤35 直接返回 0）使实际推力远小于 DOF
  数字：`0.20→20%`、`0.30→30%`、`0.60→60%`（见 gate 文档 §1.1）；很小的下潜指令（如 −0.05）
  即使放大也可能落在死区（<0.138）内，`dive_scale` 要给足 ≈3~4 ·
  结论 → 用 `comm.dof_comp.dive_scale` 单独放大下潜这一路，比在任务层加补偿/脉冲简单

### (模块级)（代码注释搬出）
- 只在"配置缺键"时才生效，但那个方向很危险：兜底值一旦偏小，限深就从现场设定值一路放宽到

### _run（代码注释搬出）
- ★ 巡检体整体兜异常：任何意外都不许把线程弄死（线程静默退出 = 保护静默失效，
- 比没有看门狗更危险 —— 2026-09-27 用例就是靠这里抓出 KeyError 的）。

### _trip（代码注释搬出）
- ★ 退出前**必须 flush**：`os._exit()` 不走 stdio 清理 ⇒ 重定向/管道下这条最关键的日志
- （以及 estop 的消息）会**整段丢掉**。板端实测踩到过：只看到 exit=9，看不到为什么。

### _AxisWatchdog（docstring 搬出）
- `max_same_dir_s`（**缺省 10.0s**，2026-09-28 定）⇒ **硬停 + 强制退出**。
- ⚠️ **为什么只管 yaw（用户 2026-09-27 明确：剩下的不能管）**：
- 持续平移、保深是持续垂向 —— 拿它们当"卡死"判据会**误杀正常动作**（长直行被掐 = 直接失败）。
- 为什么要独立线程、要放在 `base/hw/uart.py`：
- · 任务可能**卡在某个循环里**（现场"一开始进 turn 就卡住"）——写在任务控制流里的保护
- 最后那个非中位字节上（现场踩过）⇒ 先 `estop()`（连发原始中性帧，绕过 ramp），
- · `on_trip` 可注入（用例换成记录器，避免 `os._exit` 把 pytest 带走）。

### close（docstring 搬出）
- ⚠️ 兜底：**关串口前如果轴不在中位，先 `stop_hard()`** —— 下位机没有"无帧超时停车"，
- 带着半路的 ramp 字节关串口 = 船一直转/一直倒车（现场踩过）。已在中位时一帧都不多发
- （所以既有行为与用例不受影响）。

### stop_hard（docstring 搬出）
- 为什么不能只发一帧 `neutral()`：`_ramp_step` 对 yaw/surge/sway/heave 做字节级
- 字节"上（现场踩到，见 doc/注释历史.md）。

### _apply_dive_boost（docstring 搬出）
- 一律原样 —— 所以它是"单独调下潜这一路"的旋钮，不影响其它通道。
- 背景（为什么要单独放大）：DOF→字节→下位机 `RC_Matching*`（映射值 ≤35 直接返回 0）

### _drain_rx（docstring 搬出）
- 发送路径每帧都调它（见 `_write`），所以遥测的实时性跟着心跳（≥20Hz）。

### _ramp_step（docstring 搬出）
- 限深保护在平滑**之前**生效（`_apply_depth_guard`）：深度不足时上浮分量被清零，
- 且 heave 轴**当帧直接回中**（`depth_guard.snap`，不等 ramp）——否则平滑惯性还会

## common/cfg/cfgnode.py


### pid_kw（代码注释搬出）
- 共用速度档（归一化 DOF；⚠️ 任何档都不能落在执行器死区 (0, 0.138)）

### flag（docstring 搬出）
- ⚠️ **不要用 `bool(node.get(k))`**：`bool("false")` 是 True —— 板端就因为这个

### (模块)（docstring 搬出）
- pid_kw(node, defaults)      按 (kp,ki,kd,out_max,deadzone) 造 PID 参数（out_max 兼作 ±限幅）
- 分层：本模块**不 import 任何东西**（不读文件、不碰 yaml），所以

## base/hw/camera.py


### read（docstring 搬出）
- 为什么必须这样：主循环是"读一帧→处理一帧"，真机上处理慢就自然丢帧（时间基仍是真实时间）。
- 若顺序回放不丢帧，19.8s 的片子会被拉成几十秒 ⇒ HDG 的 PID/超时/循环全变味，复现就失真了。

### VideoFileCamera（docstring 搬出）
- · 打开视频**不指定后端**（`cv2.VideoCapture(path)`）⇒ 文件与设备都能开；
- · 主循环没有节流 ⇒ 这里按 `play_fps` 自己睡，避免"以解码速度飞快跑完"导致时间基被压缩；

### (模块)（docstring 搬出）
- AUV_SIM_MODE=1 AUV_CAM_VIDEO=/path/clip.mp4 [AUV_CAM_VIDEO_FPS=30]         AUV_TASK_LOG=log/replay.jsonl python3 main.py --task gate
- · `AUV_CAM_VIDEO`：把前视相机源换成该视频文件（cfg 一个字段都不动）；
- · **必须**同时 `AUV_SIM_MODE=1`（只打印、不发串口），否则回放会真的驱动船；
- · 主循环没有帧率节流 ⇒ 回放按视频原生 fps（或 AUV_CAM_VIDEO_FPS）自己节拍，

## base/log/turn_log.py


### turn_log（docstring 搬出）
- ⚠️ **绝不抛异常**：日志坏了不能把转向/任务带崩（现场只有一次机会）。

### (模块)（docstring 搬出）
- 位置说明（用户 2026-09-27 定）：它是**排查工具**、不属于 `common/` 的共用运动/检测逻辑，所以从
- 落在本就随部署上板的基础层 `base/log/`。调用方：`gate/gate_task.py`、`gate/motion/hdg.py`、`common/motion/turn_deg.py`。
- 为什么要单独一个件：现场问题（"看到门就不停地转、手动插不进来"、"转了一点点就停"、
- · 每次的结果（state/why/实测转了deg/最大偏差/耗时ms）；
- 落盘：环境变量 `AUV_TURN_LOG` 指定的文件；没设 ⇒ `<工程根>/log/turn_calls.jsonl`（追加）。
- turn_log("enter", src="gate", deg=25.0, left=True, sig=-1.0, timeout=8.0)

## common/vision/detector.py


### bgr_to_packed_nv12_fast（docstring 搬出）
- ⚠️ 与 numpy 版 bgr_to_packed_nv12 的色度**不同**（numpy 版把 2x2 求和当平均，

## base/cfg/settings.py（脚本注释搬出）


```
import base.cfg.settings as S
# --------------------------------------------------------------- 路径解析
# cfg 里的文件路径**一律写仓库内相对路径**（`cfg/front_camera.yaml`、`models/x.bin`），
#   板端有两份拷贝（`/home/sunrise/AUV` 旧副本 与 `~/Desktop/AUV_New` 在用副本），
#   写成相对路径后，**整棵树搬到哪都能跑**（本地 / AUV_New / 旧副本 / U 盘）。
# 绝对路径仍然接受（老配置不改也能跑），但不推荐。
```

## base/hw/camera.py（脚本注释搬出）


```
返回 ["PID(命令)", ...]；`dev` 支持 "0" 与 "/dev/video0" 两种写法；没装 fuser 时返回 []。
要点：
· 读完返回 None（只打印一次），主循环空转直到任务自身超时/过门结束。
配置：cfg/vision.yaml camera.front / camera.down（各自 type/device/宽高/fps/标定）。
调用：create_camera("front" | "down")；返回对象仅实现 read()。
**视频回放**（离线复现/回归；默认行为不变，只在设了环境变量时生效）：
保证时间基与真机一致（HDG 的 PID/超时都按真实时间算）。
# ---------------------------------------------------------------------------
# 视频文件回放（离线复现：把一段录制视频当成前视相机）
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 虚拟相机（前视/下视各自仿真；下视带红色标示线供 back 真实颜色逻辑测试）
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 真机后端
# ---------------------------------------------------------------------------
        # 以驱动实际协商结果为准（本相机 720p MJPG 只有 60fps 档，写 30 也按 60 出）；
        # self.fps 保持配置值不变，避免影响 recorder.py 的落盘帧率语义。
# ---------------------------------------------------------------------------
# 工厂：front/down 各自按 yaml type 实例化；真机失败按 fallback_sim 软回退
# ---------------------------------------------------------------------------
```

## base/hw/telemetry.py（脚本注释搬出）


```
只在 `UartController` 的发送路径里被调用（单线程），无需加锁。
用途（离线复现闭环）：视觉来自视频（`AUV_CAM_VIDEO=<clip>`），航向/深度来自**当时的真实遥测**
启用：`AUV_SIM_TEL_JSONL=<当时的 task.jsonl>` + `AUV_SIM_MODE=1`。
时间轴：以 jsonl 第一条的 `t` 为 0 点，按真实经过时间线性插值；超出末尾就夹在末值。
返回 [(depth_m, target_m, roll_deg, pitch_deg, yaw_deg, turn_done, turn_id), ...]。
stats（可选 dict）：累计统计 `ok`（解析成功帧数）/ `bad`（丢弃的坏帧头/校验错次数）。
同步策略：逐字节搜 0xAA55；校验错只丢 1 字节继续搜（不整段丢），
错位/半帧后能自己重新对齐；末尾可能是半个帧头的 1 字节(0xAA)会留下等下一批。
下位机(STM32)按下面的格式持续回传**深度**等状态；`base/hw/uart.py` 在每次发帧时顺带
读空串口接收缓冲，把最新深度交给"不得浮出水面"的限深保护（`comm.depth_guard`）。
帧格式（与下位机固件、`manual/udp_server.py` 的旧解析保持一致，**小端**）：
byte0..1   : 0xAA 0x55            帧头
byte2      : turn_id：接受的旋转编号，0=同步/无任务
byte3..4   : depth_cm   int16     当前深度（cm，正=水面以下）→ /100 = m
byte5..6   : target_cm  int16     下位机目标深度（cm）→ /100 = m
byte7..8   : roll_cd    int16     横滚（0.01°）→ /100 = °
byte9..10  : pitch_cd   int16     俯仰（0.01°）→ /100 = °
byte11..12 : yaw_cd     int16     航向（0.01°）→ /100 = °
byte13     : turn_done    0=未完成，1=完成
byte14     : checksum = sum(byte0..13) & 0xFF
rx = TelemetryReceiver()
rx.feed(ser.read(ser.in_waiting))      # 原始字节（可任意切分）
rx.depth_m                             # 最新深度(m)；None = 还没收到过
rx.fresh(stale_ms=500)                 # 数据是否新鲜（限深保护据此决定是否生效）
build_telemetry_frame(0.42)            # 造帧（测试/台架模拟下位机用）
        # 姿态符号归一（`comm.telemetry.yaw_sign`，默认 +1 = 不改行为）。
        #   "物理转向→遥测符号"之积 g·s；g·s=-1 时闭环**仍然稳定**（都收敛在遥测上），
        #   反而不平行）。归一后探向应报 `imag_sign=+1`、`[HDG] 转向后 |psi| 应变小`。
```

## base/hw/uart.py（脚本注释搬出）


```
· yaw 是**闭环收敛量** —— 正常的转向应当越转误差越小，同一个方向连续满舵 5s 只可能是
"反馈卡住/舵效不对/在自转"，这才需要掐死；
· surge/sway/heave 是**可以合法长同向**的：冲刺(through)本来就是几秒直行、扫视/横向对中是
· 想看哪些轴由 `comm.watchdog.axes`（轴名列表，缺省 `[yaw]`）决定；判据本身通用。
那时根本跑不到；只有独立线程 + 串口这个唯一出口拦得住。
· 只看**实际下发的轴字节**（`uart._axes`，已过 ramp），不关心是谁发的（任务/手动/脚本）。
· 越界时**必须先硬停再退出**：下位机没有"无帧超时停车"，直接退出/关串口会把船锁在
再补一串中性帧，最后才强制退出。
平滑，`force=True` 只绕过心跳节流、**不绕过 ramp** → 单帧发出时轴字节还在半路；
而下位机**没有无帧超时停车**，随后的 `close()` 一关串口，船就锁在"最后一个还在转的
做法：① 连发中性帧，次数按 `ramp.speed_per_s` 与"最大偏离 127 字节"算够；
② 静置 `settle_s` 期间继续发中性，同时用**遥测 yaw** 看有没有还在转；
仍在转 → 再补 `max_extra` 轮（打印告警）。
Returns:
True  = 已回到中位（遥测确认停了；**无遥测时返回 True 但会打印"未验证"**）
False = 遥测显示仍在转（推进器/水流顶着，或下位机没跟上）
发帧**前**先收一次遥测：限深保护要用最新深度判定（发完再收就慢一帧）。
sim 沿用全局 DEBUG；真串口按 comm.debug.tx_frame_hex，
若配置了 tx_frame_log_ms>0 则按该毫秒数节流。
只改 heave，**不动 surge/sway/yaw**（别让保护破坏对准/前进）；
`heave<=0`（下潜/悬停）与深度充足时原样放行。
遥测缺失/超时：按 `depth_guard.stale_action` 处理——`pass`(默认)放行，
`block_up` 连"盲上浮"也不许（要求下位机持续回传；SIM/无串口始终放行，免得台架被锁死）。
只处理 `heave < 0`（下潜）：乘上 `dive_scale` 后夹到 [-1, 0]；上浮/悬停/平移/转向
使实际推力远小于 DOF 数字；下潜偏弱时就把这一路放大，比在任务层加补偿/脉冲简单得多。
注：很小的下潜指令（如 −0.05）即使放大也仍可能落在死区(<0.138)内 —— 要连很小的
指令也变成有效推力，`dive_scale` 得给足（≈3~4）。
真串口由 `_drain_rx` 调用；测试/台架模拟下位机可直接调它。返回有效帧数。
SIM + `AUV_SIM_TEL_JSONL`：不发串口，改为**回放一段录制的遥测**（离线复现闭环用）。
按 `ramp.speed_per_s`(字节/秒) 让输出连续逼近目标，避免速度直接激增；
`<=0` 直通。
继续往上浮，等不到下一帧就已经冒出水面了。
模拟遥控输出；心跳 ≥20Hz；急停=连发中性帧。帧语义见 README 与 docs/CUP_AUV。
上行（下位机 → 上位机）：`base/hw/telemetry.py` 的 15B 遥测帧（0xAA55 + 深度/姿态 + 校验和），
每次发帧时顺带读空接收缓冲 → `self.telemetry.depth_m`；`comm.depth_guard` 据此**禁止上浮**
（深度 ≤ min_depth_m 时把 heave 清零、heave 轴立刻回中），保证机身不冒出水面。
# ---------------------------------------------------------------------------
# 运动状态描述（把 11B 帧翻译成人类可读的运动）
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 帧构建（纯函数，测试可直接使用）
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 控制器
# ---------------------------------------------------------------------------
        # 离线复现用：`AUV_SIM_TEL_JSONL=<当时的 task.jsonl>` ⇒ 不发串口，改为回放那段遥测
        #   （视觉来自 `AUV_CAM_VIDEO` 的视频，航向/深度来自当时的真值 ⇒ 闭环才有机会复现）
        # 下潜动力放大（底层共用，gate/ball 都吃）：放在限深之后——限深只压上浮(heave>0)，
        # 本项只放大下潜(heave<0)，两者互不干涉。
```

## base/log/turn_log.py（脚本注释搬出）


```
· 谁调用的（src=cli/turn/gate/gate_loop）、进了几次（n 递增）、间隔多久；
from base.log.turn_log import turn_log
```

## common/cfg/cfgnode.py（脚本注释搬出）


```
延迟 import base.cfg.settings：本模块要保持"不碰文件"的性质（turn_deg 会在模块级 import 它）。
`out_max` 同时作为 ±限幅（`out_min=-out_max`）。`node` 里写了的键优先，
没写的取 `defaults` —— 于是"任务级覆盖 → 共用默认"就是一次调用。
把 `kpt_mem.enable` 的一个拼写错值当成了"开"。这里统一按字符串语义解析。
用途：`nums(显式节点, 实时取到的另一套参数)` —— 缺省值直接取那份实时值，
而不是在代码里再抄一份数字（那样又会变成两套参数各自漂移）。
cfg 是 YAML → dict，读参数的规矩只有一条：
**缺键 / 类型不对时用"等于当前 cfg 的默认值"兜底，绝不抛异常**
（允许 `AUV_CFG_DIR` 指向缺新键的旧配置，也允许用例在内存里改 cfg）。
通用读法：
sub(node, key)              取子节点；缺失/非 dict → {}
merge(node, defaults)       子节点 + 默认值 → 普通 dict（缺键/None 用默认值）
num(node, key, default)     取数值；缺失/非数值 → float(default)
nums(node, defaults)        把不全的节点规范化成完整表（缺键用 defaults 补）
flag(node, key, default)    取布尔（认 bool / "1|0|true|false|yes|no|on|off"）
**共用运动参数**（`comm.motion.*`：ball / gate / 转向脚本用的是同一份，任务段不再各抄一套）：
MOTION_DEFAULTS             代码兜底表（取值等于当前 cfg/comm.yaml 的 motion 段）
motion_node(key)            取共用表（dict）
motion_num(key)             取共用标量（float）
`common/motion/turn_deg.py` 这类要能脱离 cfg 独立跑的脚本可以安全地在模块级 import 它。
# --------------------------------------------------------------- 共用运动参数
# **ball / gate 共用**的底层运动参数（cfg/comm.yaml → comm.motion）。
# 这里存的是"配置缺键/缺文件"时的代码兜底，取值**等于当前 cfg**（有用例钉住"兜底 == cfg"）。
# 任务段只放本任务特有的旋钮；确需单独一套时才在自己的段里写覆盖键（如 gate.pid_sway）。
```

## common/motion/PID.py（脚本注释搬出）


```
供任务一(ball)/gate 共用（早期由 root tasks.py 迁移至此）。
```

## common/motion/search_scan.py（脚本注释搬出）


```
`yaw_telemetry` 为 None（本帧没遥测）⇒ 返回 0：**没有闭环量就不乱转**。
**轨迹**（入场朝向 = 0°）：
0 → 左 span_deg → 右 2×span_deg → 左 2×span_deg → 右 2×span_deg → …
= -span, +span, -span, +span …（在 ±span 内往复）
默认 `span_deg=60` ⇒ 扫完"当前 **120°** 视角"；改 `span_deg` 就改扫幅（可配）。
三条要求（2026-10-04 用户定）：
① **非阻塞、睁眼看**：`step()` 每帧调一次、返回 yaw 指令；检测/相位机照常跑，
不像 `turn_deg.py` 的 CLI 那样阻塞到转完。
② **旋转稳、频率慢**：**手动 DOF + 遥测 yaw 闭环**（既不是时间开环，也不是
`TurnCore` 那条"下位机定角度"的路）。PID 用既有 `common.motion.PID.PID`，
参数沿用当年 `comm.motion.turn_pid` 那一套（kp=0.2 / out_max=0.30 / deadzone=0.4 ...）。
闭环量 = 下位机回传的绝对航向 `uart.telemetry.yaw`（我们不知道"多少秒=多少度"）。
③ **与 gate 的 hdg 完全无关**：hdg 继续走上位机下发目标角、下位机定角度旋转
（`common/motion/turn_deg.py::TurnCore`），一行不动。本模块只**读** `yaw_sign()` 判极性。
"**还剩二十几度就没推力**"（当年 turn_deg.py 文档明确警告过，现场"yaw 调不动"就是它）。
可用区间 ≈ **0.30~0.5**；想整体更慢就降 `out_max`（但别低于 0.30），别只降 kp。
PID 的 dt 由 `now_ms` 实算，**不是**当年 turn_deg 那条独立 20Hz 内层环。
所以 `pid.kd` 必须 0（当年 20Hz 下 kd=0.1 的 D 项就超 out_max ⇒ 每帧正负反转、原地极限环；
帧率更低时更糟）。`pid` 值沿用当年 turn_pid 那一套，按帧率只调 `kp/kd`（kd 保持 0）。
符号约定（沿用既有，不自创）：**`+yaw DOF = 右转`**；`psi = σ × 遥测yaw`，σ 由
`turn_deg.yaw_sign()` 给出（固件约定 × `dof_map.yaw.sign` × `telemetry.yaw_sign`）。
于是 **右转 ⇒ psi 增大 ⇒ 左旋 = 让 psi 减小**。
    # ★ 慢速占空比：1.0=每帧都发（原始 PID 出力）。想比"死区允许的"更慢就调它 ——
    #   周期性地把 yaw **压成 0**（而不是调小幅值：幅值掉进死区 0.138 推力直接消失）。
    #   船有惯性，占空比出来的效果是"平均转速更低"，且每次推的时候都过死区。
    # ★ 分段逼近（用户 2026-10-05）：目标不一步跳到 ±span，而是按 step_deg 一格一格走
    #   （如 step_deg=10、span=50 ⇒ 10→20→30→40→50），每格都用 PID 收敛到位再走下一格。
    #   0 = 关闭（一步到位，原行为）。
    # PID：沿用当年 comm.motion.turn_pid 那一套值（用户 2026-10-04 定"pid 也采用当时的值"）
    #   deadzone 是**归一化**单位 = deadzone_deg/norm_deg = 6/15 = 0.4
        # ★★ 防转圈（**绝对预算**，2026-10-05 补）：psi 离**入场朝向**永远不许超过
        #   span + tol + abs_slack。这是比"段预算"更硬的一条 —— 段预算只看本段起点，
        #   一旦有人反复 reset（把入场朝向重取成当前朝向），船就会"一步 span 地一路走"
        #   （实船球任务实测走到 348°就是这么来的）。这条绝对闸保证**永远在 ±span 内往复**。
        # ★ 防转圈（段预算）：search 只是**原地来回扫**，绝不能整圈转下去。
        #   正常情况任一段最多转 2×span 就能到目标；超过 budget 还没到位 ⇒ 多半是极性反了
        #   （σ 错 ⇒ 正反馈 ⇒ 越转越远）⇒ **立刻停手**（不发舵），换向并进停顿。
```

## common/motion/turn_deg.py（脚本注释搬出）


```
两个入口共用 `turn()` → `TurnCore` → `uart.request_turn()`（下发"相对角 + 编号"，等完成反馈）：
· **手动**：`--deg/--dir`（本 CLI；`task1_2/run_ball_reverse.sh` 也走这条）；
· **自动**：gate 的 ALIGN.HDG 用测到的 ψ 当目标角调 `TurnCore`。
其余运动参数（限幅、PID 增益、死区、归一化…）**都归下位机**：新模型下上位机只发"相对角 + 编号"，
所以这里**不再接受**旧的 `--out-max/--kp/--kd/--norm-deg/--imag-sign` 等旋钮（老脚本请一并去掉）。
运动参数（限幅 PID 增益、死区…）**一律不在这里**：旋转由下位机执行（2026-10-02 定）。
2026-10-02：**只认角大小与方向**——限幅/PID/死区/归一化这些"上位机运动参数"已整块删除
（旋转由下位机执行，参数都在下位机那侧）。
**字节级平滑**，`neutral()` 的 `force=True` **只绕过心跳节流、不绕过 ramp** → 单帧 neutral
发出时轴字节还在半路（yaw 84→128 只到 ~95 = 仍在转）；而**下位机没有无帧超时停车**，
随后 `close()` 一关串口，船就锁在那个值上一直转。
优先用 `uart.stop_hard()`（连发中性帧走完 ramp + 遥测 yaw 验证）；没有该方法就自己连发。
σ = `_FW_RIGHT_YAW_SIGN` × `dof_map.yaw.sign` × `telemetry.yaw_sign`：
· 前两项决定"+yaw 命令"落在轴字节的哪一边（半边 = 右转，手动挡 `turn_right` 已验），
· `telemetry.yaw_sign` 决定回传 yaw 的符号（它本来就是给这件事配的旋钮）。
改任一旋钮 σ 自动跟着变，不会出现"配置与代码各记一套符号"。
上位机只发"相对角 + 编号"、等完成反馈；**运动参数全在下位机**（上位机不再有 PID/限幅/死区）。
`yaw_sign()`/`wrap180()` 保留：诊断工具（check_dof_sign 等）判读遥测 yaw 还要用。
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
```

## common/vision/detector.py（脚本注释搬出）


```
例如撞球按 mission.target_color 打蓝球：画面里红球分再高也不选，
蓝球缺席则返回 None（继续搜索），绝不自动换目标。
backend 需提供 detect(frame) -> [Det]；传 None 表示该任务无后端。
X5 编译产物把检测头拆成每尺度两个 tensor：reg 64ch(DFL 4xreg_max, logits) +
cls nc ch(logits)，concat/softmax/sigmoid/DFL 均未包含，需在此还原。
outputs: {输出名: ndarray (1,g,g,C)}；返回 Det 列表（按分降序）。
labels：类别名列表，顺序须与训练/导出一致（决定 kind）。
三目标（red_ball / blue_ball / gate）共用一个模型（cfg/vision.yaml model.path），
一次前向输出所有类别，由 DetectorHub 按任务取所需类别：
- 撞球 ball  → mission.target_color 对应类别（red_ball / blue_ball）
- 过门 gate  → gate
- 返回 back 走颜色逻辑（不加载模型）
模型文件（RDK X5）：
- 必须是 X5 OE 工具链（OpenExplorer，hb_mapper makertbin，march: bayes-e）产出的 **.bin**
- 名称 hbm_runtime 只是包名传承：X5 上它加载的仍是 .bin；
.hbm 属车载 J5/J6/S100(Nash-e/m) 路线，X5 无法解析 → 代码会直接拒绝并提示
DETECTOR 模式（vision.yaml model.mode）：
mock        虚拟测试（无权重；按任务分别模拟）
hbm_runtime RDK X5 BPU 推理：packed NV12 → model.run()
onnx        CPU onnxruntime 调试（.onnx 不经过工具链，仅联调用）
        # keypoint 扩展（gate 用；bbox 任务为 None，向后兼容）：
        #   kpts: (K,2) 像素坐标，顺序=训练顺序；kpt_conf: (K,) 每点置信度
# ---------------------------------------------------------------------------
# 预处理 / 后处理
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# X5 模型文件校验
# ---------------------------------------------------------------------------
        # 非 mock：**不在此处加载** model.path。单独跑 gate 时无需 ball 权重，
        # 首次真正用到（detect/detect_all/ready(ball)）时才构造，见 _ensure_real()。
```

## common/vision/preprocess.py（脚本注释搬出）


```
顺序有讲究：enhance 本来就在 640 上做（与训练链路一致，PC 工程只产 D+wb）。
几何：kpt 回缩放后落在**去畸变 720p** 空间，相机模型一律 `rectified=True` 的 nk720，
**不要**改成 raw K/D。（改链路前先看 doc/设计/gate_pose_decode_spec.md 与权重代次。）
融合进同一张 LUT，每通道只做一次 cv2.LUT。
CLAHE 对象与 gamma LUT 缓存复用（原先每帧 createCLAHE / 重建 LUT）。
与 PC 工程 `scripts/1_prepare/map_pose_dataset.py` 的「D 域」严格同式：
S = diag(size/RAW_W, size/RAW_H, 1)
nk640 = S @ nk720 ;  K640 = S @ K
initUndistortRectifyMap(K640, dist, None, nk640, (size,size))
几何上与「先 remap@720p 再缩放」等价（因为 nk640 = S·nk720）。
raw → 缩放到模型方形输入（默认 640x640）→ 画面补偿 enhance(WB→CLAHE→gamma)
→ 640 上去畸变 remap（可选，需标定 yaml；与训练链路 prepare_frames.py 一致）
（A 域那条链 `remap@720p → resize640 → enhance` 已于 2026-10-01 退役归档，见 bak/retired/Adomain_20261001/）
（训练数据集制作在 PC 工程 RDKX5-YOLOv11n- 中完成，不在本工程范围）
参数源：cfg/vision.yaml（image.* / camera.*_calibration / model.input_size）
# ---------------------------------------------------------------------------
# 链路步骤（纯函数，与板端逐条一致）
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------
```
