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

## 目录约定（2026-09-23 起）：`record/` = tmp，`raw-data/` = 保存区

**录制期间**写 `pc/record/`（tmp，可随手清）；**录完自动归档**：该段的视频流与它的
`.timestamps` **一起**进 `RDKX5-YOLOv11n-/raw-data/<录像名>/`（子目录名就是时间戳，一眼对得上）。

```
pc/record/                                   ← tmp（中转；清掉不心疼）
└── pc_<时间戳>.log                          ← 进程日志（收尾时会再拷一份进归档目录）

RDKX5-YOLOv11n-/raw-data/
└── auv_20260923_000228/                     ← 一段一个时间戳子目录
    ├── auv_20260923_000228.mjpeg            ← 原始 MJPEG（或 .mp4，取决于模式）
    ├── auv_20260923_000228.mjpeg.timestamps ← 逐帧到达时间轴（与视频永远同目录）
    └── pc_20260923_000228.log               ← 该段日志副本（pc.sh 收尾时放进去）
```

| 项 | 说明 |
|:---|:---|
| `--save-dir` | 保存区（默认 `<工程根>/RDKX5-YOLOv11n-/raw-data`） |
| `--tmp-dir`（`--out-dir` 是旧名） | 中转目录（默认 `record/`） |
| `--no-archive` | 关掉归档：产物就留在 tmp（老行为） |
| `--no-adopt` | 不接管 tmp 里上次遗留的录像（默认会替它归档——进程被 kill/崩溃时来不及归档的，靠这个兜底） |
| 视频格式 | 默认 `-c:v copy` 零转码封装 mp4，画质与原始流一致（约 29 Mbps@720p30）；`--raw`/`--keep-raw` 保留原始 `.mjpeg`（VLC/ffplay 可直接播） |
| 0 字节录像 | **不归档**（启动即失败的残file 留在 tmp，不会污染保存区） |

> ⚠️ **`--show`/`--view` 必须用带 GUI 的 cv2**。本机 `python3` 可能指向 conda base
> （OpenCV 5.0.0 headless）⇒ `cv2.imshow` 抛异常、进程当场退出。正确用法：
> ```bash
> PYTHON=/home/ansty/anaconda3/envs/mate3.7/bin/python3.7 ./pc.sh --show
> ```
> 2026-09-22 的事故就是"用错解释器 → 启动失败"，而当时 `pc.sh` 还会 `rm -f record/auv_*`
> 把历史录像一起删掉（该行已删除）。

> 素材被误删后可以用 `pc/carve_mjpeg.py` 从磁盘空闲区按"连续 JPEG 链"雕刻回来
> （只读扫盘；用法见脚本头部注释）。

```bash
# 默认就会自动封 mp4；若手里只有 .mjpeg，可事后转（自动用 timestamps 的实测帧率）
python3 pc_recorder.py --remux RDKX5-YOLOv11n-/raw-data/<录像名>/auv_xxx.mjpeg
# 一行 ffmpeg 等价写法
ffmpeg -f mjpeg -r <实测fps> -i <保存区>/auv_xxx.mjpeg -c:v copy <保存区>/auv_xxx.mp4
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
  manual/stream.py ──UDP 5000──►  pc_recorder.py（直存/显示）
                                        ├─ tmp:   pc/record/（录制期间）
                                        └─ 保存:  RDKX5-YOLOv11n-/raw-data/<录像名>/*.mjpeg(+.timestamps)
  manual/udp_server.py ◄─UDP 9000── pc_keyboard_client.py（键盘）
      │ 0xA5 11B 帧
      ▼
    STM32
```
