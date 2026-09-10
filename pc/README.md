# pc/ — 水面 PC 端脚本（键盘遥控 + 本地录像）

板端（RDK X5）负责取流与推流（见 `../auv_vision/manual.sh`），PC 端只做三件事：
**收画面 → 本地录像 → 键盘遥控**。所有 PC 端操作都由 `pc.sh` 一个脚本统一调度。

```
pc/
├── pc.sh                  # 总调度（唯一入口）：键盘遥控 + 本地录像
├── pc_recorder.py         # 录像/看图（raw MJPEG 零转码直存 / ffmpeg 直录 mp4 / remux）
└── pc_keyboard_client.py  # 键盘遥控（Tk 窗口，发 UDP 到板端 udp_server）
```

## 快速开始

```bash
# 板上先起推流 + 遥控桥：  cd ~/AUV && ./manual.sh        （默认推 192.168.137.2:5000）
cd pc
./pc.sh                                   # 键盘遥控 + 录像（默认直接出 mp4，零转码）
./pc.sh --show                            # 边看边录（结束后自动把录像封成 mp4）
./pc.sh --raw                             # 只留原始 .mjpeg（不封 mp4）
./pc.sh --view                            # 只看画面（不录、不控）
./pc.sh --no-key                          # 只录像
./pc.sh --duration 60                     # 录 60 秒自动结束
./pc.sh --host 192.168.137.10 --port 5000 --ctrl-port 9000 --out-dir record
./pc.sh --dry-run                         # 只打印将执行的命令
./pc.sh --kill-stale --show               # 先清掉上次残留的录像进程（释放 UDP 端口）再启动
```

录像产物默认落在 **`pc/record/`**：

| 文件 | 说明 |
|:---|:---|
| **`auv_<时间戳>.mp4`** | **默认产物**：`ffmpeg -c:v copy` 零转码封装，画质与原始流完全一致（约 29 Mbps@720p30） |
| `auv_<时间戳>.mjpeg` | 原始 MJPEG 流（仅 `--raw` / `--keep-raw` 时保留，VLC/ffplay 可直接播） |
| `auv_<时间戳>.mjpeg.timestamps` | 逐帧到达时间轴（`帧号 到达时间 间隔ms`），便于离线分析/对齐日志 |
| `pc_<时间戳>.log` | 录像进程日志 |

```bash
# 默认就会自动封 mp4；若手里只有 .mjpeg，可事后转（自动用 timestamps 的实测帧率）
python3 pc_recorder.py --remux record/auv_xxx.mjpeg
# 一行 ffmpeg 等价写法
ffmpeg -f mjpeg -r <实测fps> -i record/auv_xxx.mjpeg -c:v copy record/auv_xxx.mp4
```

播放：任何播放器都能开 `.mp4`（编码为 MJPEG、无 B 帧，逐帧独立，便于逐帧核对）。

## 端口与协议

| 方向 | 端口 | 格式 | 对端 |
|:---|:---|:---|:---|
| 板端 → PC | UDP **5000**（或 HTTP 8080） | 原始 MJPEG（每帧一张 JPEG，跨数据报按 SOI/EOI 重组） | `../auv_vision/manual/stream.py` |
| PC → 板端 | UDP **9000** | ASCII `surge,sway,heave,yaw`，每个 `[-1,1]` 三位小数；`stop`/`0` = 中位 | `../auv_vision/manual/udp_server.py` |

键盘映射：`W/S` 前进后退 · `A/D` 左右平移 · `↑/↓` 上浮下潜 · `←/→` 偏航 · `空格` 急停 · `Esc` 退出。
板端 300 ms 收不到报文会**自动回中位**（`--timeout-ms` 可调），所以遥控窗口必须保持焦点。

## 端口冲突（Address already in use）怎么办

上一次录制/查看若异常退出，会继续占着 UDP 5000，导致新进程绑定失败。`pc.sh` 现在会**启动前预检**并给出处置办法：

```bash
ss -lunp | grep ':5000'      # 找到 PID（Windows: netstat -ano | findstr :5000）
kill <PID>                   # Windows: taskkill /PID <PID> /F
./pc.sh --kill-stale --show  # 或让 pc.sh 自动清理残留进程
./pc.sh --port 5010          # 或换端口（板端也要 ./manual.sh --port 5010）
```

## 注意事项

1. **同一 UDP 端口同时只能被一个进程接收**：要看画面又要录像 → 用 `./pc.sh --show`（同一进程里显示 + 直存）。
2. `--mp4` 模式由 ffmpeg 独占 socket，此时没有实时窗口。
3. Linux PC：想加大接收缓冲先 `sudo sysctl -w net.core.rmem_max=8388608`（默认 212992 会限制 `--rcvbuf`）。
4. Windows PC：首次运行放行防火墙的 **UDP 5000（接收）**；`python` 需可执行（`PYTHON=` 环境变量可指定解释器）。
5. 依赖：`pc.sh`/`pc_recorder.py` 的 raw 直存只需 Python 标准库；`--show`/`--view` 需要 `opencv-python`；`--mp4`/`--remux` 需要 `ffmpeg`；键盘遥控需要 Python 自带 `tkinter`。
6. 录像**全程零转码**（原始 JPEG 直接封装），不丢帧、CPU≈0，画质=相机原始画质；PC 较弱时不要用 `pc_recorder.py --mode mp4`（cv2 重编码实测只能到 ~12 fps）。
7. 体积参考：720p30 约 **29 Mbps ≈ 3.6 MB/s ≈ 13 GB/小时**，录前看好磁盘。

## 与板端的关系

```
板端 RDK X5                                  PC（水面）
  camera(MJPG 60fps, 零解码)
      │  原始 JPEG
      ▼
  manual/stream.py ──UDP 5000──►  pc_recorder.py（直存/显示）──► pc/record/*.mjpeg
  manual/udp_server.py ◄─UDP 9000── pc_keyboard_client.py（键盘）
      │ 0xA5 11B 帧
      ▼
    STM32
```
