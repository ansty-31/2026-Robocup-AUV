2026 Robocup AUV

本仓库用于 2026 RoboCup AUV 项目开发，包含 RDK X5 上位机视觉与任务决策代码，以及 STM32F405 下位机姿态、深度和 8 推进器控制代码。

系统采用上下位机分工：

- `auv_vision/`：RDK X5 上位机，负责视觉识别、任务状态机、串口控制、轨迹记录与返回策略。
- `AUV/`：STM32F405 下位机，负责接收控制帧、读取 IMU/深度计、PID 控制和 PWM 输出。

## 系统架构

```text
PC / RDK task script
        |
        | UDP or local task command
        v
RDK X5 upper computer
  - Python task state machine
  - front camera vision
  - ball / gate detection
  - DOF command generation
        |
        | UART, 9600 baud
        | 0xA5 control frame
        v
STM32F405 lower computer
  - H30 IMU attitude
  - MS5837-30BA depth
  - attitude/depth PID
  - 8-thruster PWM output
```

## 目录结构

```text
2026-Robocup-AUV/
├── auv_vision/              # RDK X5 上位机代码
│   ├── main.py              # 任务调度与状态机入口
│   ├── recorder.py          # 素材录制工具
│   ├── base/                # settings / camera / uart
│   ├── common/              # detector / PID / preprocess
│   ├── task1_2/             # 撞球任务与记忆返回
│   ├── gate/                # 过门任务，keypoint + PnP
│   ├── cfg/                 # YAML 配置
│   ├── models/              # RDK X5 模型文件
│   ├── doc/                 # 算法说明
│   └── tests/               # 无硬件测试
└── AUV/                     # STM32F405 下位机固件
    ├── AUV.ioc              # STM32CubeMX 工程
    ├── MDK-ARM/             # Keil MDK 工程
    ├── Core/                # CubeMX/HAL 代码
    ├── Device/              # 设备与控制代码
    ├── algorithms/          # PID 与姿态解算
    └── Drivers/             # STM32 HAL/CMSIS
```

## 硬件平台

- 上位机：RDK X5，Ubuntu 22.04
- 下位机：STM32F405
- 相机：前视 USB 相机；下视 MIPI 相机保留用于录素材和后续任务
- IMU：H30，通过 STM32 USART1 接入
- 深度计：MS5837-30BA，通过 STM32 I2C1 接入
- 推进器：八推进器全矢量布局，由 STM32 TIM2/TIM3 输出 PWM

### 电机与 PWM 映射

| 电机 | 位置 | PWM 通道 | 引脚 |
| --- | --- | --- | --- |
| M1 | 上右后 | TIM2_CH1 | PA0 |
| M2 | 下右后 | TIM2_CH2 | PA1 |
| M3 | 上左前 | TIM2_CH3 | PB10 |
| M4 | 下左前 | TIM2_CH4 | PB11 |
| M5 | 上左后 | TIM3_CH1 | PA6 |
| M6 | 下右前 | TIM3_CH2 | PA7 |
| M7 | 上右前 | TIM3_CH3 | PB0 |
| M8 | 下左后 | TIM3_CH4 | PB1 |

## 上下位机通信

### RDK -> STM32 控制帧

RDK 通过串口发送 11 字节遥控兼容帧：

```text
0xA5 + 7 axis bytes + 3 button bytes
```

默认串口配置见 `auv_vision/cfg/comm.yaml`：

```yaml
serial:
  device: /dev/ttyS1
  baud: 9600
```

DOF 映射：

| DOF | 含义 | axis |
| --- | --- | --- |
| `surge` | 前后 | 1 |
| `sway` | 左右平移 | 3 |
| `heave` | 上浮/下潜 | 2 |
| `yaw` | 左转/右转 | 0 |

### STM32 -> RDK 遥测帧

STM32 当前通过 USART2 回传深度、目标深度和三轴姿态角：

```text
0xAA 0x55 0x0A depth_cm target_cm roll_cd pitch_cd yaw_cd checksum
```

其中 payload 为小端 `int16`：

- `depth_cm = depth_m * 100`
- `target_cm = target_depth * 100`
- `roll_cd = roll_deg * 100`
- `pitch_cd = pitch_deg * 100`
- `yaw_cd = yaw_deg * 100`

## RDK 上位机

### 安装依赖

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-yaml python3-opencv
pip3 install pyserial numpy
```

如果使用 RDK X5 真机模型推理，还需要安装板端 Horizon runtime。模型文件使用 X5 工具链导出的 `.bin`，不要使用不兼容的 `.hbm`。

### 配置文件

- `auv_vision/cfg/comm.yaml`：串口、控制帧、DOF 映射、任务参数、运动预设
- `auv_vision/cfg/vision.yaml`：相机、图像预处理、模型、任务视觉参数
- `auv_vision/cfg/front_camera.yaml`：前视相机标定

可通过环境变量切换配置：

```bash
export AUV_CFG_DIR=/path/to/cfg
export AUV_VISION_CFG=/path/to/vision.yaml
export AUV_COMM_CFG=/path/to/comm.yaml
export AUV_SHOW=0
```

### 运行任务

```bash
cd auv_vision
python3 main.py --task ball     # 撞球任务
python3 main.py --task gate     # 过门任务
python3 main.py --task all      # 按 comm.yaml 中 tasks.enabled 运行
```

运行撞球 + 记忆返回：

```bash
cd auv_vision/task1_2
./run_ball_return.sh
```

## STM32 下位机

Keil 工程：

```text
AUV/MDK-ARM/AUV.uvprojx
```

CubeMX 工程：

```text
AUV/AUV.ioc
```

主要模块：

| 文件 | 作用 |
| --- | --- |
| `Device/Src/Usermain.c` | 主控制循环、PID 叠加、PWM 输出、遥测回传 |
| `Device/Src/RC.c` | 11 字节控制帧解析 |
| `Device/Src/Motor.c` | DOF 到 8 个推进器的分配矩阵 |
| `Device/Src/Imu.c` | H30 IMU 配置与姿态数据 |
| `Device/Src/MS5837.c` | MS5837-30BA 压力/深度读取 |
| `algorithms/Src/PID.c` | PID 控制器 |
| `algorithms/Src/Angle_resolve.c` | 姿态角和 plant1/plant2 解算 |

## 控制逻辑

下位机主循环将以下输出叠加到 8 路电机：

- RDK 下发的人工/任务 DOF 指令
- `plant1` / `plant2` 姿态稳定 PID
- roll 稳定 PID
- depth 定深 PID

定深逻辑：

1. RDK 发送上浮/下潜指令时，STM32 认为当前为人工升降，暂时关闭深度 PID 输出。
2. 松开上浮/下潜指令时，STM32 将当前深度记录为 `target_depth`。
3. 之后由 `PID_depth` 将深度维持在 `target_depth` 附近。

MS5837 深度计算公式：

```c
depth_m = (pressure_01mbar - s_surface_pressure_01mbar) / 980.665f;
```

如果压力基准还在调试中，深度可能为正也可能为负。定深时最重要的是目标深度和当前深度使用同一套读数，不应单独按正负做特殊处理。

## 视觉任务

### 撞球任务

路径：`auv_vision/task1_2/ball.py`

功能：

- 识别红/蓝球
- 根据目标面积分级前进
- 使用视觉 PID 居中
- 目标丢失后执行搜索
- 记录 DOF 轨迹供记忆返回使用

### 记忆返回

路径：`auv_vision/task1_2/return_by_memory.py`

通过反向回放已记录的 DOF 轨迹返回出发区，不依赖下视相机。

### 过门任务

路径：`auv_vision/gate/`

功能：

- gate keypoint 四角检测
- PnP 位姿估计
- 距离对齐、进近、穿门相位机
- 过门完成判定

## 测试

无硬件测试位于 `auv_vision/tests/`。

```bash
cd auv_vision
python3 tests/test_uart.py
python3 tests/test_pid.py
python3 tests/test_logic.py
python3 tests/test_detector.py
python3 tests/test_preprocess.py
python3 tests/test_gate_geometry.py
python3 tests/test_gate_flow.py
```

如果安装了 `pytest`：

```bash
cd auv_vision
python3 -m pytest tests
```

## 串口接线

RDK 与 STM32 需要共地，并交叉连接 TX/RX：

```text
RDK TX  -> STM32 USART2_RX / PA3
RDK RX  <- STM32 USART2_TX / PA2
GND     <-> GND
```

RDK 40-pin 串口通常使用 `/dev/ttyS1`。如果 `/dev/ttyS0` 被 Linux console 占用，不建议作为业务串口使用。

串口权限：

```bash
sudo usermod -aG dialout $USER
```

执行后需要重新登录。
