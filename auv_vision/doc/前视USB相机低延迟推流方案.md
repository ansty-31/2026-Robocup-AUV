# 前视 USB 相机低延迟推流方案（RDK X5 / AUV）

> 目标：把**前视 USB 相机**（`/dev/video0`，`1280x720`，见 `cfg/vision.yaml` 的 `camera.front`）
> 的画面推到**水面 PC**（系缆千兆网口），**端到端延迟尽量低**，同时不抢占 BPU 推理与串口心跳的 CPU。
>
> 文中命令分两类标注：**【已本地验证】**＝在主机（Ubuntu 22.04 / ffmpeg 4.4.2 / GStreamer 1.20.3，与板端同版本）
> 用等价的模拟源实际跑通；**【已板端实测】**＝在 RDK X5 + 实际相机上跑过（数据见 §17）。

---

## 0. 结论速览

| 项 | 结论 |
|:---|:---|
| 延迟最低的做法 | **不做任何转码**：相机直接输出 MJPEG，原样透传（已落地到 `manual/stream.py` + `base/camera.py` + `manual.sh`） |
| 首选链路 | 板端 `./manual.sh`（手动模式：遥控桥+录像+推流）→ PC `python3 -m manual.stream view` / `ffplay -f mjpeg udp://@:5000` |
| 实测端到端延迟 | 采集侧 48.8 fps（**裸 JPEG 零转码**，见 §17）；H.264 硬编约 90~130 ms（方案 B）；TROS websocket 200~500 ms（方案 D，仅调试） |
| 实测带宽 | 本相机 720p MJPG **140 KB/帧** → 30 fps 推流约 **28~29 Mbps**（千兆系缆毫无压力；WiFi/百兆不行） |
| 关键坑 1 | OpenCV 默认协商 **YUYV**，本相机 720p YUYV 只有 **9 fps**；必须显式 MJPG（已修，识别从 9 → 48.8 fps） |
| 关键坑 2 | **UVC 设备同一时刻只能被一个进程取流**（`Device or resource busy`），所以推流必须与识别/录像**共用同一路采集** → §8 |
| 明确不要用的做法 | GStreamer `v4l2src ! rtpjpegpay`（1.20.3 实测报 `Invalid component`，上游已知缺陷）；两个进程各开一次 `/dev/video0`；WiFi 图传 |

---

## 1. 目标与约束

| 维度 | 现状 / 要求 |
|:---|:---|
| 相机 | 前视 UVC（**Microdia USB 2.0 Camera**，`0c45:6368`），`vision.yaml` 配 `type: usb`、`device: 0`、`1280x720`；实测 720p 只有 MJPG 60fps 档（YUYV 仅 9fps） |
| 板端 | RDK X5（Sunrise 5，8×A55 + 10 TOPS BPU），Ubuntu 22.04 arm64，镜像 3.5.0-AUV 系列 |
| 网络 | 千兆网口（支持 PoE，实测协商 1000Mb/s）；官方默认静态 IP `192.168.127.10`，本仓库实际用 `192.168.137.10`（PC 侧 `192.168.137.2`，RTT 0.16 ms） |
| 接收端 | 水面 PC（Windows 或 Linux 都可能），用来**看画面**辅助操控/调试 |
| 延迟目标 | 观感上“手眼一致”：**< 100 ms** 为优，100~150 ms 可接受，> 250 ms 已明显滞后 |
| CPU 预算 | 识别（NV12 预处理 + BPU）与串口心跳优先，推流最好 **0 转码、≤1 个核的一部分** |
| 生命线 | 推流不能拖慢主任务；推流挂了主任务必须照跑（可降级/自动重启） |

---

## 2. 平台事实核查（先对齐“板上有什么”）

以下结论来自本仓库的镜像 rootfs（`deploy/rootfs`）、deb 包索引与内核源码，不是推测：

### 2.1 镜像里已经有

| 组件 | 位置/版本 | 对推流的意义 |
|:---|:---|:---|
| `ffmpeg` / `ffplay` / `ffprobe` | `/usr/bin`，**4.4.2-0ubuntu0.22.04.1** | 方案 A/B 的主力；`v4l2` 输入、`mpjpeg` 复用器、`libx264` 都在 |
| `v4l2-ctl` | `/usr/bin` | 查相机支持的分辨率/帧率/像素格式 |
| OpenCV `cv2`（aarch64 python3.10） | `/usr/lib/python3/dist-packages` | 识别在用；**其 `videoio` 链接了 `libgstreamer-1.0` + `libgstapp`** → `cv2.VideoCapture/VideoWriter` 可直接吃 GStreamer 管线字符串 |
| GStreamer 核心 + `plugins-base` + `plugins-good` | 1.20.3（103 个插件） | `v4l2src`、`jpegdec/jpegenc`、`jpegparse`、`rtpjpegpay/depay`、`rtpjitterbuffer`、`udpsink/udpsrc`、`multipartmux`、`appsink/appsrc`、`isomp4` |
| Python GObject/GStreamer 绑定 | `gi` + `Gst-1.0.typelib` | 没有 `gst-launch` 也能用 Python 搭管线（见 §8.3） |
| TROS Humble | `/opt/tros/humble` | `hobot_usb_cam`、`hobot_codec`（硬编解码）、`websocket`（浏览器预览） |
| 硬件编码器 | `libspcdev.so` / `sp_codec.h`（ENCODER API）；TROS `hobot_codec` | X5 的 ENCODER API **支持 H264 / H265 / MJPEG** 三种编码 |
| `libx264.so.163` | 系统库，`libavcodec.so.58` 依赖它 | `ffmpeg -c:v libx264` 可用（软编，费 CPU） |

### 2.2 镜像里**没有**（别照抄网上教程）

| 缺什么 | 影响 |
|:---|:---|
| `gst-launch-1.0` CLI（`gstreamer1.0-tools` 未安装） | 所有 GStreamer 管线必须用 **Python(gi)** 或 OpenCV 承载 |
| `gstreamer1.0-plugins-bad` | 没有 `h264parse`、`mpegtsmux`、`webrtcbin` |
| `gstreamer1.0-plugins-ugly` | 没有 `x264enc` |
| `gstreamer1.0-libav` | 没有 `avdec_h264` 等 |
| RTSP 服务器（gst-rtsp-server / mediamtx） | 想用 RTSP 得先自己装/自研，不划算 |

### 2.3 相机独占：这是本方案的地基

内核源码 `source/kernel/drivers/media/usb/uvc/uvc_v4l2.c`：

- `uvc_v4l2_open()` 允许多次 `open()`，但**privileged handle 全局只有一个**：
  `uvc_acquire_privileges()` 里 `atomic_inc_return(&stream->active) != 1` 就返回 `-EBUSY`；
- 需要 privileged 的操作包括 `VIDIOC_S_FMT`、`VIDIOC_REQBUFS`，而 `uvc_ioctl_streamon()` 也要求 `uvc_has_privileges()`。

**结论：第二个进程去打 `/dev/video0`，一定在 `S_FMT`/`REQBUFS` 处拿到 `EBUSY`**——不会“各拿一半带宽”，而是直接失败。
所以推流不能简单地“再起一个 ffmpeg”，必须与识别**共用同一个采集进程**（§8）。这条约束决定了整套架构。

### 2.4 USB 侧注意

- 4 个 Type-A 口是经 GL3510 集线器扩展的 USB 3.0；相机插任一 Type-A 口都行，但**不要串在额外的 USB2.0 小 hub 上**。
- `YUYV` 是未压缩的：720p@30 YUYV ≈ **442 Mbps**，UVC 在 USB2.0 上根本给不到（社区实测 1080p YUYV 只有 5 fps）。
  → **必须用相机的 MJPEG 模式**（既省 USB 带宽，又正好是我们最想要的“已是 JPEG”）。

---

## 3. 延迟预算（把“尽量低”量化）

以方案 A（MJPEG 直通）为例，单帧 30 fps ≈ 33 ms：

| 环节 | 典型耗时 | 能不能再压 |
|:---|:---|:---|
| 相机曝光 + 内部 JPEG 编码（UVC 固件） | **33 ms**（1 帧） | 只能靠提高帧率（60fps → 16ms）；这是地板 |
| USB 传输 + 内核 UVC 队列 | 1~3 ms | 用 MJPEG/USB3.0；别用 YUYV、别串 hub |
| 板端打包（`-c:v copy`） | **< 2 ms** | 不要解码、不要重编码（`jpegdec ! jpegenc` 会加 10~30 ms + CPU） |
| 网络（千兆系缆） | 0.1~1 ms | 有线；避免 WiFi（+5~30 ms 抖动） |
| 接收端缓冲 | **0~10 ms** | `-fflags nobuffer -flags low_delay -framedrop`、GStreamer `rtpjitterbuffer latency=0 drop-on-latency=true`、VLC `--network-caching=0` |
| 解码 + 显示 | 10~25 ms | MJPEG 解码很便宜；开启 `-framedrop`，避免桌面合成器/垂直同步拖帧 |
| **合计（方案 A）** | **≈ 60~90 ms** | |
| **合计（方案 B，H.264 硬编）** | **≈ 90~130 ms** | 多 1 帧编码延迟（GOP 内无 B 帧、intra-refresh 可压到 ~1 帧） |
| **合计（方案 D，TROS websocket）** | **≈ 200~500 ms** | 浏览器侧还会再缓冲 |

> 记忆点：**任何一次“解码再编码”都是 +10~30 ms 和一份 CPU**；低延迟的第一原则是“别碰像素”。

---

## 4. 方案总览

| 方案 | 链路 | 延迟 | 带宽(720p30) | 板端 CPU | 需要与识别合进程 | 适用 |
|:---|:---|:---|:---|:---|:---|:---|
| **A** | MJPEG 直通 → UDP | **60~90 ms** | 7~11 Mbps | ≈0 | 是（或识别没跑） | **首选**：调试/观察、系缆千兆 |
| **A'** | MJPEG 直通 → HTTP/TCP | 90~150 ms | 7~11 Mbps | ≈0 | 是 | 链路有丢包、要浏览器看 |
| **B1** | ffmpeg 软编 H.264 → RTP/UDP | 110~180 ms | 2~4 Mbps | 1~2 核 | 是 | 带宽紧张（无线/百兆共享） |
| **B2** | 硬件 H.264（hobot_codec/sp_codec）→ RTP/UDP | 90~130 ms | 2~4 Mbps | ≈0 | 是 | 长期部署、要多路/要录像 |
| **C** | GStreamer 单进程 `tee`：识别 + 推流同源 | 同 A | 同 A | ≈0 | 这就是合进程方案 | 与识别共存的正解 |
| **D** | TROS `hobot_usb_cam`+`hobot_codec`+`websocket` | 200~500 ms | 受编码器控制 | 中 | 是（抢占相机） | 临时演示，不推荐 |
| ✗ | GStreamer `v4l2src ! rtpjpegpay` | — | — | — | — | **已实测失败**，见 §8.4 |
| ✗ | 两个进程各开 `/dev/video0` | — | — | — | — | **内核 EBUSY**，见 §2.3 |
| ✗ | YUYV 直传 / YUYV 走 USB2.0 | — | 442 Mbps | — | — | 物理上不成立 |

---

## 5. 方案 A：MJPEG 直通 over UDP（首选，最低延迟）

原理：UVC 相机吐出的就是 JPEG；`ffmpeg -c:v copy` 只做**搬运**（不解码不编码），每帧作为一个 UDP 数据报发出；
PC 端 `ffplay -f mjpeg` 直接解 JPEG 显示，零缓冲。

### 5.1 第 0 步：先确认相机能力（每次换相机都做）

```bash
# 列出该相机支持的所有格式/分辨率/帧率 —— 必须有 Motion-JPEG 1280x720 (30Hz)
v4l2-ctl -d /dev/video0 --list-formats-ext

# 当前格式
v4l2-ctl -d /dev/video0 --get-fmt-video
```

若相机没有 MJPEG，只有 YUYV：720p 请降到 `640x480@30` 再走本方案，否则 USB 带宽不够。

### 5.2 板端发送（【已本地验证】：命令形状与内存约束）

```bash
# 建议先跑 10 秒，确认不报错、且帧率稳定
ffmpeg -hide_banner -loglevel warning \
  -f v4l2 -input_format mjpeg -video_size 1280x720 -framerate 30 -i /dev/video0 \
  -c:v copy -flush_packets 1 -muxdelay 0 -muxpreload 0 \
  -f mjpeg "udp://192.168.137.1:5000?pkt_size=1316"
```

要点：

- `-input_format mjpeg` + `-c:v copy`：**全程不解码**，CPU 占用接近于 0，这是低延迟的关键；
- `-flush_packets 1`：每帧立即推出去，不攒缓冲；
- `pkt_size=1316`：发送端自己按 MTU 切片，避免 IP 分片；接收端（`ffplay -f mjpeg udp://`）会跨数据报按 JPEG 边界重组，
  本机实测**单帧 46 KB / 73 KB 两种情况都能完整还原（74/74 帧）**，所以它同时兼容大帧；
- 不要加 `-re`：V4L2 输入本身就是实时节奏。

> ⚠️ **本相机的真实帧大小（已板端实测）**：720p MJPG **中位 145 KB / 峰 148 KB**，
> 30 fps 推流 ≈ **28~29 Mbps**，60 fps ≈ **69 Mbps**——比常见 USB 相机大得多（相机端 JPEG 质量高）。
> 配合 `pkt=8000`（或 `pkt_size=1316`）切片，**接收端能完整还原，不存在 64KB 上限问题**（PC 端计数与发送端一致、解码失败 0）。
> 选型建议：千兆系缆直接推（无压力）；百兆/无线就改 §6 HTTP 或 §7 H.264；想省带宽可先降分辨率到 `960x540`。
> 复测命令：

```bash
# 录 10 秒 → 看单帧字节分布（帧越大，丢一个包的代价越高）
ffmpeg -hide_banner -loglevel error -f v4l2 -input_format mjpeg \
  -video_size 1280x720 -framerate 60 -i /dev/video0 -c:v copy -t 10 -y /tmp/probe.mjpeg
ffprobe -v error -select_streams v -show_entries packet=size -of csv=p=0 /tmp/probe.mjpeg \
  | sort -n | awk '{a[NR]=$1} END {printf "n=%d median=%d p95=%d max=%d\n", NR, a[int(NR/2)], a[int(NR*0.95)], a[NR]}'
```

若需要把单帧压到 ≲64 KB：降到 `-video_size 960x540`；也可改走 §6 的 HTTP/TCP（TCP 重传，抗丢包更好）。

### 5.3 PC 端接收（【已本地验证】：端到端解码出帧）

Linux / Windows（装了 ffmpeg 即可，`ffplay` 随 ffmpeg 一起发布）：

```bash
ffplay -hide_banner -loglevel warning \
  -fflags nobuffer -flags low_delay -framedrop -probesize 32 -analyzeduration 0 \
  -f mjpeg "udp://@:5000?overrun_nonfatal=1&fifo_size=1000000"
```

- `-fflags nobuffer`：不要输入缓冲；
- `-flags low_delay`：解码器不做重排等待；
- `-framedrop`：来不及显示就丢帧，**永远只显示最新帧**（低延迟观感的关键）；
- `-probesize/-analyzeduration` 压到最小：减少起播探测时间。

VLC（Windows 常用）也能看同样的流：

```bash
vlc --network-caching=0 --file-caching=0 udp://@:5000
```

> **实测踩坑（重要）**：GStreamer `udpsrc` 的 `buffer-size`（以及同类接收缓冲参数，如 ffmpeg `udp://…?fifo_size=`）
> 一旦超过内核 `net.core.rmem_max`（本机默认 **212992**），GStreamer 会直接失败：
> `Could not create a buffer of requested N bytes (Operation not permitted)`。
> 想加大接收缓冲必须先在 PC 上 `sudo sysctl -w net.core.rmem_max=8388608`（写进 `/etc/sysctl.d/99-auv.conf` 持久化）。

### 5.4 系统服务（可选，跟随开机）

```ini
# /etc/systemd/system/auv-stream.service
[Unit]
Description=AUV front camera MJPEG UDP stream (zero-transcode)
After=network-online.target

[Service]
# CPU 亲和：避开识别线程常用的核（示例：把推流放在 CPU6/7）
ExecStart=/usr/bin/taskset -c 6,7 /usr/bin/ffmpeg -hide_banner -loglevel warning \
  -f v4l2 -input_format mjpeg -video_size 1280x720 -framerate 30 -i /dev/video0 \
  -c:v copy -flush_packets 1 -muxdelay 0 -muxpreload 0 -f mjpeg udp://192.168.137.1:5000?pkt_size=1316
Restart=always
RestartSec=2
Nice=-5

[Install]
WantedBy=multi-user.target
```

> 注意：若识别程序正在跑，这个服务**必然起不来**（EBUSY）——请用 §8 的合进程方案。

---

## 6. 方案 A'：MJPEG 直通 over HTTP/TCP（抗丢包、可浏览器）

当系缆有丢包、或单帧超过 64 KB、或希望**浏览器直接看**时用这条。TCP 会重传，画面不会花，代价是丢包时延迟尖峰。

### 6.1 板端（【已本地验证】：HTTP 头与帧流）

```bash
ffmpeg -hide_banner -loglevel warning \
  -f v4l2 -input_format mjpeg -video_size 1280x720 -framerate 30 -i /dev/video0 \
  -c:v copy -flush_packets 1 -f mpjpeg -listen 1 "http://0.0.0.0:8080/"
```

`-listen 1` 让 ffmpeg 自己当 mini HTTP 服务器，**单客户端**连接（断了会自动等下一个）。

### 6.2 PC 端

```bash
# 最低延迟（推荐）：仍是 ffplay，走 http
ffplay -hide_banner -fflags nobuffer -flags low_delay -framedrop http://192.168.137.10:8080/
```

**浏览器要看的话有一个坑（实测）**：ffmpeg 返回的响应头是

```
HTTP/1.1 200 OK
Content-Type: application/octet-stream      ← 浏览器不认，不会当 MJPEG 流渲染
Transfer-Encoding: chunked
--ffmpeg
Content-type: image/jpeg
```

所以要浏览器 `<img src="http://127.0.0.1:8090/">` 直接出画面，需要一个**只改响应头、不动画面数据**的小代理（跑在 PC 上，零板端开销）：

```python
#!/usr/bin/env python3
# 水面 PC 侧：MJPEG 头改写代理 —— 板端 http://<board>:8080/ → 本机 8090 给浏览器
# 用法: python3 mjpeg_browser_proxy.py http://192.168.137.10:8080/ 8090
import socket, sys, urllib.request

UP = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.137.10:8080/"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8090

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("0.0.0.0", PORT)); srv.listen(4)
print("浏览器打开 http://127.0.0.1:%d/   （Ctrl-C 退出）" % PORT)
while True:
    c, _ = srv.accept()
    try:
        r = urllib.request.urlopen(UP, timeout=10)          # urllib 自动解 chunked
        c.sendall(b"HTTP/1.1 200 OK\r\n"
                  b"Content-Type: multipart/x-mixed-replace; boundary=ffmpeg\r\n"
                  b"Cache-Control: no-cache\r\nConnection: close\r\n\r\n")
        while True:
            buf = r.read(65536)
            if not buf:
                break
            c.sendall(buf)                                   # 原样透传，不重编码
    except Exception as e:
        print("client done:", e)
    finally:
        c.close()
```

（这段是**参考实现，未在板上联调**；边界串 `ffmpeg` 与 ffmpeg 默认 `boundary_tag` 一致。）

---

## 7. 方案 B：H.264（带宽紧张 / 无线 / 要录像时）

### 7.1 B1 软编（现在就能用，验过）

板端 ffmpeg 带 `libx264`（`libavcodec.so.58` 依赖 `libx264.so.163`）。低延迟参数组合：

```bash
ffmpeg -hide_banner -loglevel warning \
  -f v4l2 -input_format mjpeg -video_size 1280x720 -framerate 30 -i /dev/video0 \
  -c:v libx264 -preset ultrafast -tune zerolatency \
  -profile:v baseline -bf 0 -g 30 -keyint_min 30 -sc_threshold 0 \
  -b:v 4M -maxrate 4M -bufsize 1M -x264-params "rc-lookahead=0:sync-lookahead=0:sliced-threads=1" \
  -f rtp "rtp://192.168.137.1:5002?pkt_size=1400"
```

PC 端用同一个 SDP 播放（【已本地验证】：接收端解出 77 帧）：

```sdp
v=0
o=- 0 0 IN IP4 0.0.0.0
s=AUV front camera
c=IN IP4 0.0.0.0
t=0 0
m=video 5002 RTP/AVP 96
a=rtpmap:96 H264/90000
a=fmtp:96 packetization-mode=1
```

```bash
ffplay -hide_banner -protocol_whitelist file,udp,rtp \
  -fflags nobuffer -flags low_delay -framedrop auv_h264.sdp
```

`-preset ultrafast -tune zerolatency -bf 0 -rc-lookahead 0` 是“无 B 帧、无前瞻”的组合，等于把编码延迟压到 ~1 帧；
代价是 CPU：X5 的 8×A55 上 720p30 软编大致吃掉 **1~2 个核**，建议 `taskset` 绑到识别不用的核，并从 720p 起测；不够就降 `960x540`。

### 7.2 B2 硬编（推荐长期部署：CPU≈0，码率 2~4 Mbps）

X5 的 ENCODER API 官方支持 **H264/H265/MJPEG**，两条落地路径：

1. **TROS `hobot_codec`（最快落地）** —— 板上已安装：
   ```bash
   source /opt/tros/humble/setup.bash
   # 1) 取流：USB 相机以 mjpeg 输出（注意：hobot_usb_cam 会独占相机）
   ros2 launch hobot_usb_cam hobot_usb_cam.launch.py usb_video_device:=/dev/video0 usb_pixel_format:=mjpeg
   # 2) 硬编：ros → h264
   ros2 launch hobot_codec hobot_codec_encode.launch.py \
     codec_in_mode:=ros codec_in_format:=jpeg codec_out_mode:=ros \
     codec_sub_topic:=/image codec_pub_topic:=/image_h264 codec_out_format:=h264
   # 3) 还需一段自己的“ROS → RTP”桥（板端没有现成 RTSP/RTP 服务）
   ```
   这条路的短板是 **ROS 话题 + 自研桥**会引入额外拷贝与延迟，且 `hobot_usb_cam` 同样独占相机。
2. **C++ 直接用 `sp_codec`（`libspcdev.so`，`usr/include/sp_codec.h`）**：`sp_start_encode(SP_ENCODER_H264,…)` 拿码流，
   自己接 `rtph264pay` 或裸 UDP。注意 **X5 要求帧 16 位对齐**、输入必须是 **NV12**，且板上**没有 `h264parse`**，
   `rtph264pay` 需要显式 caps：`video/x-h264,stream-format=byte-stream,alignment=au`（并用 `config-interval=1` 周期重发 SPS/PPS）。
   这是“性能最好但工程量最大”的选项，建议放到 P3 阶段。

> 结论：**先做方案 A 拿到 60~90 ms；只有当带宽/无线成为瓶颈时，才上 B2。**

---

## 8. 方案 C：与识别/录像共存（**已落地**）

> ✅ **已实现**：`base/camera.py` 里 `UsbCamera` 打开相机时顺带把原始 JPEG 交给推流器（`stream_server.MjpegPusher`），
> 开关是 `cfg/vision.yaml` 的 `stream.enable`。因此 `main.py`（识别）与 `manual/recorder.py`（录像）**各自跑时都会自动推流**，
> 不额外打开相机、不抢设备。板端实测：录像 150 帧落盘的同时推送 139 帧，无冲突（§17.2）。
> 下面保留设计推导与备选实现。

### 8.1 为什么必须共进程

§2.3 已证明：第二个进程打开 `/dev/video0` 会在 `S_FMT/REQBUFS` 拿到 `EBUSY`。可选架构只有三种：

| 架构 | 做法 | 评价 |
|:---|:---|:---|
| ① 独立推流进程 | 推流独占相机，识别停跑 | 只适合调试/观察阶段（P0） |
| ② 进程内推流（**推荐**） | 识别进程持有相机，同一份帧既送推理又送推流 | 零额外 USB 带宽、零独占冲突、延迟最低 |
| ③ 串行分时 | 识别跑一会儿、推流跑一会儿 | 起流要 1~2 s，画面断续，**不要用** |

### 8.2 进程内两种实现（按改造量排序）

**(a) 最小改造：保留 `cv2.VideoCapture`，加一个“推流线程”**
识别循环里已经有 BGR 帧；把每帧（可降采样/降质量）交给线程，由线程做 JPEG 编码 + UDP 发出：

```python
# 参考实现（未在板上联调）：与 cv2.VideoCapture 共用相机，不额外开设备
import socket, threading, queue, time, cv2
_q = queue.Queue(maxsize=1)          # 只保留最新帧：满了就丢旧帧，杜绝延迟堆积
_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
PC = ("192.168.137.1", 5000)

def _worker():
    while True:
        frame = _q.get()
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            # 自行按 MTU 切片发出；接收端按 JPEG 边界跨数据报重组（见下）
            for i in range(0, len(jpg), 1316):
                _sock.sendto(jpg[i:i+1316], PC)

threading.Thread(target=_worker, daemon=True).start()

# 识别主循环里：
#   frame = cap.read()
#   if _q.full():
#       _q.get_nowait()     # 丢旧帧
#   _q.put_nowait(frame)
```

- 代价：每帧一次 JPEG 编码（720p 约 10~20 ms CPU/帧）。想省 CPU：**降采样到 960x540 或 640x360** 再发；
- 接收端可**直接复用方案 A 的 `ffplay -f mjpeg udp://@:5000`**：本机实测过「发送端按 1316 字节切片 → 接收端跨数据报还原」，
  46 KB 与 73 KB 单帧都能 74/74 完整解码，所以不需要自定义帧头协议。

**(b) 推荐：GStreamer 单管线 `tee`（零转码，识别拿 BGR，推流拿原始 JPEG）**
板上没有 `gst-launch`，但 `gi` + `Gst-1.0.typelib` 在，用 Python 搭：

```
v4l2src device=/dev/video0 io-mode=2 !
  image/jpeg,width=1280,height=720,framerate=30/1 !
  tee name=t
    t. ! queue leaky=downstream max-size-buffers=1 ! jpegdec ! videoconvert ! video/x-raw,format=BGR !
         appsink name=vision max-buffers=1 drop=true sync=false          # → 识别（替代 cv2.VideoCapture）
    t. ! queue leaky=downstream max-size-buffers=1 ! <推流分支>            # → 推流
```

推流分支（按可靠度从高到低）：

1. **§6 的 HTTP/TCP**：`multipartmux ! tcpserversink host=0.0.0.0 port=8080`
   —— 注意 GStreamer 的 `tcpserversink` **不会自动加 HTTP 响应头**，浏览器/ffplay 需要 §6.2 的小代理配合；
2. **H.264 硬编分支**：`hobot_codec`/`sp_codec` 编完再 `rtph264pay ! udpsink`；
3. **不要用 `rtpjpegpay`** —— 理由见 §8.4。

`tee` 的要点：两个分支都必须带 `queue leaky=downstream max-size-buffers=1`，
**slowest branch 不能拖累 fast branch**；识别分支用 `appsink drop=true max-buffers=1 sync=false`，永远只取最新帧。

### 8.3 关于“用 OpenCV 自带 GStreamer 后端省事”

板端 `libopencv_videoio.so.4.5d` **链接了 `libgstreamer-1.0`/`libgstapp`**，所以可以：

```python
cap = cv2.VideoCapture("v4l2src device=/dev/video0 io-mode=2 ! image/jpeg,width=1280,height=720,framerate=30/1 ! "
                       "jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1 sync=false",
                       cv2.CAP_GSTREAMER)
```

这样**相机只被打开一次**，且拿到的仍是 BGR 帧；但 OpenCV 侧拿不回“原始 JPEG”，推流仍需重编码（回到 8.2(a) 的代价）。
所以：**要零转码就上 8.2(b) 的 `tee`；要改造最小就 8.2(a)。**

### 8.4 已实测的失败路径：`v4l2src ! rtpjpegpay`

主机（GStreamer **1.20.3**，与板端同版本）实测：

```
$ gst-launch-1.0 videotestsrc is-live=true num-buffers=10 ! video/x-raw,width=320,height=240,framerate=15/1 \
    ! jpegenc quality=80 ! rtpjpegpay pt=26 ! udpsink host=127.0.0.1 port=5019 sync=false async=false
警告：来自组件 …/GstRtpJPEGPay:rtpjpegpay0：Invalid component
额外的调试信息: ../gst/rtp/gstrtpjpegpay.c(627): gst_rtp_jpeg_pay_read_sof () …
→ UDP 侧收到 0 个包（用 Python 监听端口核实）
```

这不是偶发：上游早就有 **Bug 684502（jpegenc 产出的 JPEG 子采样模式 rtpjpegpay 处理不了）** 与
**Bug 676134（rtpjpegpay 处理不了罗技 C920 的 MJPG）**。UVC 相机的 MJPEG 常见 4:2:2，
正好命中这个坑。**所以在板上不要走 `rtpjpegpay` 传 UVC MJPEG**；真要 RTP，就先 `jpegdec ! jpegenc` 强制成 4:2:0（多一次编解码），
或者按 §7 走 H.264。

---

## 9. 方案 D：TROS websocket（能看，但不低延迟）

```bash
source /opt/tros/humble/setup.bash
ros2 launch hobot_usb_cam hobot_usb_cam.launch.py usb_video_device:=/dev/video0 usb_pixel_format:=mjpeg
# 另开终端
ros2 launch websocket websocket.launch.py websocket_image_topic:=/image websocket_only_show_image:=true
# PC 浏览器打开 http://<板IP>:8000 → 左上角“Web 显示”
```

延迟 200~500 ms（JPEG/图像话题 + websocket + 浏览器缓冲），且**独占相机**。
定位：**验收硬件是否正常、给不装软件的人看一眼**，不要当作比赛图传。

---

## 10. 明确排除 / 反模式

| 反模式 | 为什么不要 |
|:---|:---|
| 两个进程各开 `/dev/video0` | 内核 UVC 单 privileged handle → 第二个 `EBUSY` |
| `v4l2src ! rtpjpegpay` | 1.20.3 实测 `Invalid component`，UDP 零包（§8.4） |
| 用 WiFi 传画面 | 2.4G 干扰 + 5~30 ms 抖动 + 丢包重传，延迟不可控 |
| 720p YUYV（未压缩） | 442 Mbps，USB2.0 上取不到帧率 |
| 推流前做 `jpegdec ! videoconvert ! jpegenc` | +10~30 ms/帧 + 满核 CPU，纯浪费 |
| 接收端不设 `nobuffer/framedrop` | 缓冲会累积到几百 ms 甚至秒级 |
| 用 ROS 图像话题当中转（非 D 方案） | 每次序列化/反序列化都在加延迟 |

---

## 11. 调优清单（按收益从高到低）

1. **相机侧**：`-input_format mjpeg`，分辨率 720p（帧大小超 64 KB 就降到 960x540）；帧率 30（60 可再省 16 ms）。
2. **传输**：**不用任何转码**；有线优先；`pkt_size=1316`；UDP 接收缓冲先 `sysctl -w net.core.rmem_max=8388608`。
3. **接收端**：`-fflags nobuffer -flags low_delay -framedrop`；VLC `--network-caching=0`；GStreamer `rtpjitterbuffer latency=0 drop-on-latency=true`（无线/丢包链路改 `latency=30`）。
4. **板端系统**：
   ```bash
   sudo sysctl -w net.core.rmem_max=8388608 net.core.wmem_max=8388608 net.core.netdev_max_backlog=2000
   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor   # 关调频
   # 关 USB autosuspend（避免相机被挂起后起流慢）
   echo -1 | sudo tee /sys/bus/usb/devices/*/power/autosuspend 2>/dev/null
   ```
5. **CPU 隔离**：`taskset -c 6,7` 把推流/编码钉到识别不用的核；识别进程 `taskset -c 0-5`。
6. **网卡/链路**：确认协商到 1000 Mbps（`ethtool eth0 | grep Speed`）；系缆/滑环接触不良会丢包，观察 `ip -s link show eth0` 的 errors/dropped。
7. **显示端**：关掉窗口动画/合成器特效；全屏 `ffplay` 比窗口化稳。

---

## 12. 延迟实测方法（上板必做）

**方法一：手机拍屏法（最简单、最可信）**

1. PC 上打开一个带**毫秒**的大号秒表（网页/本地计时器）；
2. 把前视相机对准这块屏幕（保持画面里始终有秒表读数）；
3. 推流回到 PC 并全屏播放；
4. 用手机把「PC 上的秒表**原始显示**」与「推流**画面里的秒表**」拍进同一张照片；
5. 两个读数之差 = **端到端延迟**。连拍 5 次取中位数（注意两屏刷新混叠，读数取相邻格子更保守）。

**方法二：帧内时间戳法（可连续统计）**

板端在推流前给每帧叠加时间戳（会引入一次转码，仅用于测量）：

```bash
ffmpeg -f v4l2 -input_format mjpeg -video_size 1280x720 -framerate 30 -i /dev/video0 \
  -vf "drawtext=text='%{pts\:hms}':fontsize=48:fontcolor=white:box=1:boxcolor=black@0.5" \
  -c:v libx264 -preset ultrafast -tune zerolatency -f rtp "rtp://192.168.137.1:5002?pkt_size=1400"
```

PC 端录 10 秒流，逐帧读画里的时间戳与本地时钟做差 → 得到延迟分布（均值/P95）。
**注意**：此法因叠加与编码本身会引入 30~80 ms，只用来做“不同方案之间”的相对比较，绝对值以方法一为准。

---

## 13. 验收 checklist

- [ ] `v4l2-ctl -d /dev/video0 --list-formats-ext` 能列出 `Motion-JPEG 1280x720 (30Hz)`
- [ ] 10 秒录制无丢帧报错：`ffprobe` 显示 `30 fps`、帧数 ≈ 300
- [ ] 单帧最大字节数已测（建议 ≲ 64 KB；超出则降分辨率或改 HTTP/TCP）
- [ ] 方案 A 起流后，PC `ffplay` 画面 **< 1 s** 出图，且连续跑 30 min 不卡死
- [ ] 手机拍屏法实测端到端延迟 **< 100 ms**（方案 A）
- [ ] 板端 `top`：推流进程 CPU **< 10%**（单核占比）且识别帧率不掉
- [ ] 识别 + 推流同时运行时，无 `EBUSY`（说明按 §8 共用了同一采集进程）
- [ ] 拔插相机后能自动恢复（systemd `Restart=always` 或进程内重连）
- [ ] 断网 10 s 后推流恢复，主任务不受影响

---

## 14. 落地步骤（分阶段）

| 阶段 | 做什么 | 产出 |
|:---|:---|:---|
| **P0 打通（半天）** | 识别停跑 → 按 §5 起 `ffmpeg -c:v copy` → PC `ffplay` 出图 → 手机拍屏测延迟 | 一条 60~90 ms 的可用链路 + 延迟基线 |
| **P1 加固（1 天）** | 按 §11 调 sysctl/governor/亲和；按 §5.4 做成 systemd；按 §6 补一条 HTTP/TCP 备链 | 稳定、可自启、有两条链路 |
| **P2 合进程（1~2 天）** | 按 §8.2 把推流并进识别进程（先 (a) 线程版，再评估 (b) tee 版） | 识别与推流同时运行，延迟不退化 |
| **P3 固化（按需）** | 若带宽/无线有压力 → §7.2 上硬件 H.264；把脚本/服务打进镜像（参考 `docs/01-开发工程项目.md` 的 deb 集成方式） | 长期部署形态 |

---

## 15. 风险与回退

| 风险 | 触发条件 | 回退动作 |
|:---|:---|:---|
| 单帧过大（> ~64 KB），丢包即花一帧 | 720p 复杂纹理、相机 MJPEG 质量高 | 降到 `960x540`，或改 §6 HTTP/TCP（TCP 重传） |
| UDP 丢包导致画面局部花屏 | 系缆磨损、滑环、PoE 长缆 | 改 §6（TCP 重传）或 §7（H.264 + 小包 RTP） |
| 两个进程抢相机（EBUSY） | 误把推流做成独立服务 | 停掉独立服务，按 §8 合进程 |
| 软编 H.264 抢占 CPU | §7.1 参数没绑核/分辨率过高 | `taskset` 绑核或降分辨率；实在不行改 §5 直通 |
| 浏览器看不到画面 | §6 的 `Content-Type` 坑 | 用 §6.2 代理，或直接 `ffplay` 看 |
| 相机掉线 | USB 供电/线缆 | systemd `Restart=always`；识别侧 `camera.fallback_sim` 已有软回退 |

---

## 16. 附录：命令速查与验证状态

```bash
# 相机能力
v4l2-ctl -d /dev/video0 --list-formats-ext

# A. MJPEG 直通 → UDP（板端；PC: ffplay -f mjpeg udp://@:5000）
ffmpeg -f v4l2 -input_format mjpeg -video_size 1280x720 -framerate 30 -i /dev/video0 \
  -c:v copy -flush_packets 1 -muxdelay 0 -muxpreload 0 -f mjpeg "udp://192.168.137.1:5000?pkt_size=1316"

# A'. MJPEG 直通 → HTTP/TCP（板端；PC: ffplay http://<板IP>:8080/）
ffmpeg -f v4l2 -input_format mjpeg -video_size 1280x720 -framerate 30 -i /dev/video0 \
  -c:v copy -flush_packets 1 -f mpjpeg -listen 1 "http://0.0.0.0:8080/"

# PC 端最低延迟接收
ffplay -fflags nobuffer -flags low_delay -framedrop -probesize 32 -analyzeduration 0 -f mjpeg "udp://@:5000?overrun_nonfatal=1&fifo_size=1000000"

# B1. 软编 H.264 → RTP（PC 用同端口 SDP + ffplay）
ffmpeg -f v4l2 -input_format mjpeg -video_size 1280x720 -framerate 30 -i /dev/video0 \
  -c:v libx264 -preset ultrafast -tune zerolatency -bf 0 -g 30 -x264-params "rc-lookahead=0" \
  -f rtp "rtp://192.168.137.1:5002?pkt_size=1400"

# 单帧大小体检
ffprobe -v error -select_streams v -show_entries packet=size -of csv=p=0 probe.mjpeg | sort -n | tail -3
```

| 命令/结论 | 验证状态 |
|:---|:---|
| `ffmpeg -c:v copy -f mjpeg udp://…` → `ffmpeg/ffplay -f mjpeg udp://@…` 解码出帧 | ✅ 主机实测（74~99 帧） |
| 单帧 46 KB / 73 KB 经 `pkt_size=1316` 切片后仍完整还原 | ✅ 主机实测（各 74/74 帧） |
| `ffmpeg -f mpjpeg -listen 1 http://…` 可被 HTTP 客户端取流 | ✅ 主机实测（HTTP 200 + multipart 帧） |
| 其响应头为 `application/octet-stream`（浏览器需代理） | ✅ 主机实测（curl -i） |
| `ffmpeg -c:v libx264 -tune zerolatency -f rtp` + 手写 SDP → ffplay | ✅ 主机实测（77 帧） |
| `rtpjpegpay` 处理 UVC/常见 MJPEG（4:2:2）失败 | ✅ 主机实测 + 上游 Bug 684502 / 676134 |
| GStreamer `udpsrc buffer-size` 超 `net.core.rmem_max` 直接失败 | ✅ 主机实测（默认 212992）：`Could not create a buffer of requested N bytes`；ffmpeg `fifo_size` 同受该上限约束 |
| `-c:v copy -f rtp`（ffmpeg MJPEG 走 RTP）不可用 | ✅ 主机实测：`Unsupported pixel format` |
| UVC 单 privileged handle（第二个进程 EBUSY） | ✅ 内核源码 `uvc_v4l2.c` 认定 |
| 板上 ffmpeg 4.4.2 / cv2+GStreamer / GStreamer 1.20.3 / TROS 组件 | ✅ 镜像 rootfs 认定 |
| X5 ENCODER API 支持 H264/H265/MJPEG | 📄 官方文档（见下），**板端待实测** |
| 相机规格 / 帧大小 / 推流帧率 / 共存 | ✅ **已板端实测**，见 §17 |

参考：

- RDK X5 硬件接口（USB 3.0×4、千兆 PoE、默认 IP）：<https://d-robotics.github.io/rdk_x_doc/en/Quick_start/hardware_introduction/rdk_x5>
- RDK X5 ENCODER API（H264/H265/MJPEG 编码、NV12 输入、16 位对齐）：<https://d-robotics.github.io/rdk_doc/en/Basic_Application/multi_media_sp_dev_api/RDK_X5/cdev_multimedia_api_x5/encoder_api/>
- TROS 数据采集（`hobot_usb_cam` 参数与支持的像素格式）：<https://d-robotics.github.io/rdk_doc/en/Robot_development/quick_demo/demo_sensor/>
- GStreamer Bug 684502：<https://lists.freedesktop.org/archives/gstreamer-bugs/2012-September/094989.html>
- GStreamer Bug 676134（rtpjpegpay 与 UVC MJPG）：<https://lists.freedesktop.org/archives/gstreamer-bugs/2012-May/090725.html>

---

## 17. 板端实测与落地记录（2026-09-10）

### 17.1 实测环境

| 项 | 值 |
|:---|:---|
| 板端 | RDK X5，`3.5.0`，内核 6.1.83，`ssh sunrise@192.168.137.10`，工作区 `~/AUV` |
| 相机 | `0c45:6368 Microdia USB 2.0 Camera`（Bus 001，USB 2.0），`/dev/video0` |
| 网络 | `eth0 1000Mb/s Full`，PC `192.168.137.2`，RTT 0.16 ms |
| PC 侧 | 本机 Ubuntu 22.04 + cv2（`stream_view.py`）/ ffplay |

相机能力（`v4l2-ctl -d /dev/video0 --list-formats-ext`）：

| 格式 | 分辨率 | 帧率 |
|:---|:---|:---|
| **MJPG** | 1920x1080 / **1280x720(60)** / 1280x1024 / 1024x768 / 800x600 / 640x480(120) / 320x240 | 高 |
| YUYV | 1920x1080(6) / **1280x720(9)** / 1024x768(6) … | 低 |

> **结论：720p 只有 MJPG 能用（60fps）；YUYV 720p 只有 9fps。**

### 17.2 实测数据

| 测量项 | 结果 | 说明 |
|:---|:---|:---|
| OpenCV 默认协商（未改前） | **YUYV 1280x720 @ 9.1 fps** | `cv2.VideoCapture(0)` + 设宽高 —— 识别链路实际只有 9fps |
| 强制 MJPG 后（`base/camera.py` 已修） | FOURCC=MJPG、`negotiated_fps=60`，读取 **48.8 fps** | `CONVERT_RGB=0` 拿原始 JPEG，识别侧再 `imdecode`（约 18.6 ms/帧） |
| ffmpeg `-c:v copy` 直通录制 | 594 帧/10s ≈ **59.4 fps** | 单帧 **中位 145 KB / 峰 148 KB**；约 **69 Mbps** |
| 推流（30fps 限速，原始 JPEG 转发） | **24.8 fps / 28~29 Mbps**，丢积压 0 | 相机 60fps 出帧 → 每 2 帧推 1 帧 |
| 板端 CPU（推流进程） | **≈ 11% 单核** | 零转码；系统 `sy 4.4%` 主要是 socket 发送 |
| PC 端接收 | 帧数**与发送端计数完全一致**（149/149、272 帧），**解码失败 0** | 140 KB/帧，`pkt=8000` 切片跨数据报重组正常 |
| UDP 切片 | 逐帧按 `pkt` 切片后接收端完整还原，**无 64KB 上限问题** | 对比：不分片依赖 IP 分片也能还原 |
| **第二个进程开相机** | `ffmpeg … /dev/video0: Device or resource busy` | 独占性在真机复现（§2.3 结论成立） |
| **录像 + 推流同时** | `recorder.py` 落盘 150 帧（1280x720 mp4）**同时**推送 139 帧（11.2 fps / 13.4 Mbps），无冲突 | 同一进程共用一路采集；推流帧率跟随录像循环（见 17.5） |

### 17.3 已落地的实现

1. **`base/camera.py`（识别/录像的取流入口，已改）**
   - 强制 `MJPG`、`CONVERT_RGB=0`（拿原始 JPEG），`read()` 仍返回 BGR（调用方无感）；
   - 首帧探测不是 JPEG 时自动回退 BGR（兼容不支持 raw 输出的相机）；
   - 每次 `read()` 顺手把原始 JPEG 交给推流器（`get_stream_pusher()`，非阻塞、只保留最新帧），
     **推流不额外打开相机 → 与识别/录像天然不冲突**；
   - 记录 `negotiated_fps`（本机为 60），`self.fps` 保持配置值以免影响 `recorder.py` 落盘帧率语义。
2. **`cfg/vision.yaml` 新增顶层 `stream:` 段**（`enable/host/port/stream_fps/pkt`，板端已置 `enable: true`、`host: 192.168.137.2`）。
3. **`manual/stream.py`（推流/接收库）**：`MjpegPusher`（UDP 零转码）、`MjpegHttpServer`（浏览器 `<img>` 可看，标准
   `multipart/x-mixed-replace` 头）、`UdpReceiver`/`HttpReceiver`/`Sink`（PC 端接收/显示/录制/统计）；
   命令行 `python3 -m manual.stream push|view`。
4. **`manual.sh`（根目录唯一入口脚本）**：手动模式一键 = 串口遥控桥(`udp_server.py`) + 录像(`manual/recorder.py`) + 推流
   （camera 钩子顺带完成）；参数 `--host/--port/--stream-fps/--out/--seconds/--sim/--no-udp/--no-record`；
   Ctrl-C 先停录像写完文件，再停遥控桥。
5. **板端工作区整理**：`~/AUV` 已与本机 `auv_vision/auv_vision` 对齐为包结构 —— 扁平旧模块
   （`camera.py`/`settings.py`/`uart.py`/`tasks.py`/`detector.py`/`preprocess.py`/`return_by_memory.py` 等）
   已归档到 `~/AUV/bak/flat_legacy_<日期>/`；入口脚本 `udp_server.py`、`run_ball_return*.sh` 保留原位但导入改为包路径
   （`base.settings`/`base.uart`/`task1_2/return_by_memory.py`）；板端专属的 `base/settings.py`（`SIM_MODE=False`）**未改动**；
   整目录备份 `~/auv_workspace_backup_<日期>.tgz`。

### 17.4 用法（当前板端已可直接用）

**录像在 PC 端做**（板端只推流：CPU≈0、帧率不被写盘拖累）。

```bash
# 板端
./manual.sh                                  # 遥控桥 + 推流（默认 192.168.137.2:5000）
./manual.sh --sim                            # 无串口硬件
./manual.sh --record --seconds 60 --out rec.mp4   # 例外：板端本地录像
python3 main.py --task ball                  # 识别任务：同样会自动同时推流

# 水面 PC —— 总调度脚本 pc/pc.sh（键盘遥控 + 本地录像到 pc/record/）
cd auv_vision/pc
./pc.sh                    # 键盘遥控 + 录像（默认直接出 mp4，零转码）
./pc.sh --show             # 边看边录（结束自动封 mp4）   ./pc.sh --raw  # 只留原始 .mjpeg
./pc.sh --view             # 只看画面
# 只看画面也可以直接：
ffplay -fflags nobuffer -flags low_delay -framedrop -f mjpeg "udp://@:5000"
```

同一 UDP 端口同时只能被一个进程接收 → **既看又录请用 ②**。关推流：`cfg/vision.yaml` 里 `stream.enable: false`。

### 17.5 后续可优化

| 项 | 现状 | 建议 |
|:---|:---|:---|
| 板端本地录像会拖慢推流 | `manual/recorder.py` 单线程“读帧 → mp4 编码”，实测推流掉到 ~11 fps | **已改：录像挪到 PC 端**（`ffmpeg -c copy` 或 `view --record-raw`，零转码不丢帧）；仅 `--record` 时才有此现象 |
| PC 端 cv2 写 mp4 会丢帧 | `view --record`（cv2 mp4v 编码）实测只到 ~12 fps | 改用 ①`ffmpeg -c copy` 或 ②`--record-raw`（实测 20/20 帧全落盘，实测帧率写回文件名提示） |
| 想要 60 fps 推流 | 默认 `stream_fps: 30` | 改 `stream_fps: 0`（不限速），注意 140 KB×60 ≈ 69 Mbps |
| 带宽紧张/无线 | UDP 28 Mbps | 走 §6 HTTP（TCP 重传）或 §7 H.264 硬编（2~4 Mbps） |
| 端到端延迟绝对值 | 采集/传输环节实测完整，但“玻璃到玻璃”需人工测 | 按 §12 手机拍屏法；预期 60~90 ms 量级 |

