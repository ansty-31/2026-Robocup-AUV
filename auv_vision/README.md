# AUV 视觉导航项目（RDK X5 / Ubuntu 22.04 / Python）

RoboCup AUV 赛事视觉代码。平台：**RDK X5（3.5.0）**，前视 USB + 下视 IMX415(MIPI)（下视保留，用于录素材/未来任务）。
识别：YOLO 蓝/红球 + gate（**keypoint 四角 + PnP** 新前端）；串口 11B 帧向 STM32 下发 DOF。

> 先读：`README.md`（用法）→ `doc/算法说明.md`（总体）→ `doc/算法说明-gate-PnP移植方案.md`（gate 设计与移植）。

## 目录分区（英文分区命名）

```
auv_vision/
├── main.py              # 装配/状态机入口（ball/gate；由 task1_2/*.sh 或直接调用）
├── stream_server.py     # 前视画面低延迟推流（零转码 MJPEG → UDP/HTTP；可嵌入式与识别/录像共用相机）
├── stream_view.py       # 水面 PC 端接收/显示/录制（配 stream_server.py）
├── manual.sh            # 手动模式唯一入口：遥控桥 + 录像 + 推流
├── manual/              # 手动模式三件套：udp_server.py(遥控桥) · recorder.py(录像) · stream.py(推流/接收库)
├── base/                # 基础部件：settings.py(配置) · camera.py(前视/下视/mipi) · uart.py
├── common/              # 通用功能：PID.py · preprocess.py(图像链路) · detector.py(检测)
├── task1_2/             # 任务一（撞球）+ 记忆返回工具：
│   ├── ball.py          #   任务一 撞球 BallTask
│   ├── return_by_memory.py  #   记忆返回（撞球后，不依赖视觉）
│   └── run_*.sh         #   编排：待机→下潜→前进→撞球(记轨迹)→记忆返回→回退
├── gate/                # 任务三 过门（keypoint+PnP；geometry/frontend/decode/task/…）
├── cfg/                 # vision.yaml · comm.yaml（参数唯一来源）
├── doc/                 # 算法说明.md · 算法说明-gate-PnP移植方案.md
├── models/ · tests/     # 权重(.bin) · 无硬件测试
```

分区语义：`base`（平台基础设施）/ `common`（跨任务公用）/ `task1_2`（任务一 + 记忆返回工具）/ `gate`（任务三）。
依赖规则：任务代码只 import `base`/`common` 与同级任务模块；`main.py` 是唯一装配点。

## 快速开始（本机，无硬件）

```bash
python3 tests/test_uart.py              # 串口字节（pty 回环）
python3 tests/test_pid.py / test_logic.py / test_detector.py / test_preprocess.py
python3 tests/test_gate_geometry.py     # gate PnP 合成往返
python3 tests/test_gate_flow.py         # gate 相位机（进近→穿门/REACQUIRE，mock）
python3 tests/test_gate_standalone.py   # gate 单任务接线（用门权重/不加载 ball/拒绝空跑）
python3 main.py --task ball             # 只跑撞球（SIM/mock）
python3 main.py --task gate             # 试跑过门（cfg model.mode: mock）
```

### 真机编排（.sh，见 task1_2/）
```bash
cd task1_2 && ./run_ball_return.sh   # 待机45→下潜3→前进3→撞球(记轨迹)→记忆返回→回退
cd task1_2 && ./run_gate.sh          # 待机30→(可选下潜/前进)→只跑 gate（下水前自检权重）
# 手动等价：
#   AUV_DOF_LOG=/tmp/path.csv python3 main.py --task ball
#   python3 task1_2/return_by_memory.py --log /tmp/path.csv
```
- **返回出发区 = 记忆返回**（纯航位推算反向回放，见 return_by_memory.py）；
- 下视相机 `cfg/vision.yaml camera.down` 与 `base/camera.py`（sim/mipi）**保留**，仅策略不再使用。

### 只用 gate 权重跑过门（单独下水测 gate）
```bash
python3 main.py --task gate                  # 只装配 gate 后端 → hub.detect_list("gate")
python3 preview_detect.py --gate-kpt         # 下水前：门框 + 4 角点 + 置信度（船不动）
```
- `main.py --task gate` **只加载 `model.task_models.gate.path`**（即 `gate_kpt_*.bin`），
  单权重 ball 模型惰性加载、本次完全不碰；启动横幅会打印实际权重文件名；
- gate 后端不可用（权重缺失/路径错）时**直接拒绝启动（退出码 3）**，不会入水后空跑；
- 窗口叠加 = 当前任务本帧检测（gate 为门框 + 4 角点编号），不再额外跑一遍模型。

## 任务算法速览

- **任务一 撞球**（task1_2/ball.py）：面积占比分级调速 + 视觉居中 PID + 智能搜索；
- **记忆返回**（task1_2/return_by_memory.py）：轨迹反向回放（`--extra-sec` 保险余量）；
- **任务三 过门**（gate/gate_task.py）：keypoint 四角 → IPPE 6-DoF，深度 `Z=tvec.z`；
  RANGE_ALIGN 子状态 GOLDEN/CREEP/HOLD/REACQUIRE → APPROACH → THROUGH（机身过门判据）。
  门 = 闭合矩形框(红 PVC，0.70×0.50 m，悬空，对称无朝向要求)。

## 参数（cfg/*.yaml）
`vision.yaml`：camera(front/down)、`image.*`（**ball/gate 共用图像链路，勿改**）、`model.*`
（含 `task_models.gate`）、`gate.*`（几何/keypoint/PnP）；`comm.yaml`：serial/dof_map/心跳/急停、
`tasks.enabled`（默认 `[ball]`；gate 需后端或 mock）、`ball/gate` 任务参数。
模型：X5 OE `.bin`（march `bayes-e`）；`.hbm` 不兼容。

## 运行监测
串口帧打印 + 画面叠加（含 gate phase/substate/z/pass）——cfg debug 开关；`AUV_SHOW=0` 关窗口。

## 画面推流 / 手动模式（端到端约 60~90 ms）
方案与实测数据见 `doc/前视USB相机低延迟推流方案.md`；根目录只有一个入口脚本 `manual.sh`，
推流/接收实现在 `manual/stream.py`（库）。

**录像放在水面 PC**（板端只推流：省板端 CPU、不受写盘拖累，帧率更高）。

```bash
# 板端：遥控桥 + 推流（默认；无串口硬件加 --sim）
./manual.sh
# 例外：确需板端本地录像时（PC 不在场）
./manual.sh --record --seconds 60 --out rec.mp4

# 水面 PC —— 用 pc/ 的总调度脚本（键盘遥控 + 本地录像，见 ../pc/README.md）
cd ../pc
./pc.sh                    # 键盘遥控 + 录像（默认直接出 mp4 到 pc/record/）
./pc.sh --show             # 边看边录（结束后自动封 mp4）
./pc.sh --raw              # 只留原始 .mjpeg
./pc.sh --view             # 只看画面
```

> 同一个 UDP 端口同一时刻只能被一个进程接收：**既要看又要录就用 ②**（`--record-raw` 在同一进程里直存 + 显示）。

**推流不会和识别抢相机**：UVC 设备同一时刻只允许一个进程取流（第二个进程报 `Device or resource busy`）。
推流因此不单独起进程，而是由 `base/camera.py` 在打开相机时顺带把**原始 JPEG** 交给 `manual/stream.py` 发出
（开关：`cfg/vision.yaml` 的 `stream.enable`，手动模式由 `manual.sh` 用环境变量覆盖）。
另外 `cv2.VideoCapture` 默认协商 YUYV（本相机 720p 只有 9 fps），`base/camera.py` 已强制 MJPG（60 fps 档）。

## 联调 TODO
gate keypoint `.bin`（导出后做解码自检 §6）；REACQUIRE 后退 dof sign 实测；高低门编排。
