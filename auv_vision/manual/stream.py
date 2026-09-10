# -*- coding: utf-8 -*-
"""stream.py — 画面推流/接收库（统一入口：根目录 manual.sh（或 python3 -m manual.stream））

板端发送（零转码）：相机输出 MJPEG，本模块把**原始 JPEG 字节**原样转发，不解码不重编码。
PC 端接收：按 JPEG 边界跨 UDP 数据报重组，显示/录制/统计；或直接拉 HTTP MJPEG。

给 `base/camera.py` 用的接口：
    pusher = stream.MjpegPusher(host, port, pkt=8000, stream_fps=30)
    pusher.offer(jpeg_bytes)        # 非阻塞，只保留最新帧

给 `manual.py` 用的接口：
    open_raw_camera(...), CameraSource, SimSource,
    MjpegPusher, MjpegHttpServer,          # 发送
    UdpReceiver, HttpReceiver, Stats, Sink # 接收/显示/录制

完整方案见 doc/前视USB相机低延迟推流方案.md。
"""
import os as _os, sys as _sys
if __package__ in (None, ""):        # 支持直接 python3 manual/xxx.py 运行
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import select
import socket
import sys
import threading
import time
import urllib.request

try:
    import cv2
    import numpy as np
except ImportError:                      # 纯发送端（板端）只需要 cv2 由调用方保证
    cv2 = None
    np = None

JPEG_SOI = b"\xff\xd8\xff"
JPEG_EOI = b"\xff\xd9"


def _as_bytes(buf):
    """把相机 read() 的返回值统一成 JPEG bytes。"""
    if isinstance(buf, (bytes, bytearray)):
        return bytes(buf)
    if np is not None and isinstance(buf, np.ndarray):
        return buf.reshape(-1).tobytes()
    raise TypeError("需要 bytes 或 numpy 数组，得到 %r" % type(buf))


# ===========================================================================
# 发送端
# ===========================================================================
def open_raw_camera(dev=0, width=1280, height=720, fps=60):
    """以 MJPEG 打开 UVC 相机，尽量拿到**原始 JPEG**。

    返回 (cap, mode)：
      mode == "raw" → read() 返回 (1,N) uint8 原始 JPEG（零解码，可直接转发）
      mode == "bgr" → 相机不支持原始 JPEG 输出，read() 返回 BGR（推流需重编码）

    注意：OpenCV 的 V4L2 后端默认协商 **YUYV**（本相机 720p YUYV 只有 9fps），
    必须显式 FOURCC=MJPG + CONVERT_RGB=0 才能到 60fps 档。
    """
    if cv2 is None:
        raise RuntimeError("需要 opencv-python（cv2）")
    cap = cv2.VideoCapture(dev, getattr(cv2, "CAP_V4L2", 0))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("无法打开相机 %s（可能被其它进程占用）" % dev)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    mode = "bgr"
    try:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        ok, buf = cap.read()
        if ok and buf is not None and _as_bytes(buf)[:3] == JPEG_SOI:
            mode = "raw"
    except Exception:
        mode = "bgr"
    if mode != "raw":
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    return cap, mode


class MjpegPusher(object):
    """后台线程把最新一帧 JPEG 通过 UDP 发出（单槽：只发最新帧，天然丢弃积压）。"""

    def __init__(self, host, port=5000, pkt=8000, stream_fps=30.0,
                 stats_interval=2.0, verbose=True):
        self.host = host
        self.port = int(port)
        self.pkt = int(pkt)
        self.interval = (1.0 / float(stream_fps)) if stream_fps and stream_fps > 0 else 0.0
        self.stats_interval = stats_interval
        self.verbose = verbose

        self.frames_sent = 0
        self.bytes_sent = 0
        self.frames_dropped = 0        # 上一帧还没发出去就被新帧顶掉
        self.frames_skipped = 0        # 被 stream_fps 限速主动跳过

        self._slot = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._last_send = 0.0
        self._last_stats = time.time()
        self._last_bytes = 0
        self._t0 = time.time()

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        except OSError:
            pass

        self._thread = threading.Thread(target=self._run, name="mjpeg-pusher", daemon=True)
        self._thread.start()

    def offer(self, jpeg):
        """投递一帧（numpy (1,N) uint8 或 bytes）。立刻返回，永不阻塞采集循环。"""
        data = _as_bytes(jpeg)          # 拷贝约 140KB ≈ 数十微秒，换取对相机缓冲复用的免疫
        with self._lock:
            if self._slot is not None:
                self.frames_dropped += 1
            self._slot = data

    def _run(self):
        while not self._stop.is_set():
            with self._lock:
                data, self._slot = self._slot, None
            if data is None:
                time.sleep(0.0005)
                continue
            if self.interval:
                now = time.time()
                if now - self._last_send < self.interval:
                    self.frames_skipped += 1
                    continue
                self._last_send = now
            n = len(data)
            try:
                for i in range(0, n, self.pkt):
                    self._sock.sendto(data[i:i + self.pkt], (self.host, self.port))
            except OSError as e:                     # 网络抖动：丢这一帧，别拖累采集
                if self.verbose:
                    print("[PUSH] 发送失败（丢帧）: %s" % e)
                continue
            self.frames_sent += 1
            self.bytes_sent += n
            self._maybe_stats()

    def _maybe_stats(self):
        now = time.time()
        if not self.verbose or now - self._last_stats < self.stats_interval:
            return
        dt = now - self._last_stats
        db = self.bytes_sent - self._last_bytes
        self._last_stats = now
        self._last_bytes = self.bytes_sent
        print("[PUSH] %d 帧/%.0fs = %.1f fps | %.1f Mbps | 丢积压 %d | 限速跳过 %d | 目标 %s:%d"
              % (self.frames_sent, now - self._t0, self.frames_sent / max(now - self._t0, 1e-6),
                 db * 8 / 1e6 / dt if dt else 0, self.frames_dropped,
                 self.frames_skipped, self.host, self.port))

    def stats(self):
        dt = max(time.time() - self._t0, 1e-6)
        return {"frames_sent": self.frames_sent, "fps": self.frames_sent / dt,
                "mbps": self.bytes_sent * 8 / 1e6 / dt,
                "dropped": self.frames_dropped, "skipped": self.frames_skipped}

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self._sock.close()
        except OSError:
            pass


class MjpegHttpServer(object):
    """multipart/x-mixed-replace MJPEG 服务（浏览器 <img src="http://<板IP>:8080/"> 直接看）。"""

    def __init__(self, port=8080, boundary="auvframe", stream_fps=30.0,
                 stats_interval=2.0, verbose=True):
        self.port = int(port)
        self.boundary = boundary.encode()
        self.verbose = verbose
        self.stats_interval = stats_interval
        self.interval = (1.0 / float(stream_fps)) if stream_fps and stream_fps > 0 else 0.0
        self.frames_sent = 0
        self.bytes_sent = 0
        self.clients_served = 0
        self._seq = 0
        self._slot = None
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._httpd = None
        self._thread = None

    def offer(self, jpeg):
        data = _as_bytes(jpeg)
        with self._cv:
            self._seq += 1
            self._slot = (self._seq, data)
            self._cv.notify_all()

    def _wait_frame(self, last_seq, timeout=2.0):
        with self._cv:
            if self._slot is not None and self._slot[0] > last_seq:
                return self._slot
            self._cv.wait(timeout)
            if self._slot is not None and self._slot[0] > last_seq:
                return self._slot
        return None

    def start(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        server = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"        # 不用 chunked，靠连接关闭界定流

            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=%s"
                                 % server.boundary.decode())
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                server.clients_served += 1
                last = 0
                next_ok = 0.0
                try:
                    while not server._stop.is_set():
                        item = server._wait_frame(last)
                        if item is None:
                            continue
                        last, data = item
                        if server.interval:                  # 限速：到点再发，且发最新帧
                            now = time.time()
                            if now < next_ok:
                                time.sleep(next_ok - now)
                                latest = server._wait_frame(last)
                                if latest is not None:
                                    last, data = latest
                            next_ok = time.time() + server.interval
                        self.wfile.write(b"--" + server.boundary + b"\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(b"Content-Length: %d\r\n\r\n" % len(data))
                        self.wfile.write(data)
                        self.wfile.write(b"\r\n")
                        server.frames_sent += 1
                        server.bytes_sent += len(data)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        class _Server(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        self._httpd = _Server(("0.0.0.0", self.port), _Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        name="mjpeg-http", daemon=True)
        self._thread.start()
        if self.verbose:
            print("[HTTP] http://<本机IP>:%d/  （浏览器 <img> / ffplay 均可）" % self.port)

    def close(self):
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()


class CameraSource(object):
    """真机帧源：把 read() 结果统一成 JPEG（raw 模式零转码）。"""

    def __init__(self, cap, mode):
        self.cap = cap
        self.mode = mode
        self.frames = 0

    def read_jpeg(self):
        ok, buf = self.cap.read()
        if not ok or buf is None:
            return None
        self.frames += 1
        if self.mode == "raw":
            return buf
        ok, enc = cv2.imencode(".jpg", buf, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return enc if ok else None

    def close(self):
        self.cap.release()


class SimSource(object):
    """无相机自测：合成画面，按 fps 走真实节奏。"""

    def __init__(self, width=1280, height=720, fps=30):
        if cv2 is None:
            raise RuntimeError("--camera-sim 需要 opencv-python")
        self.w, self.h, self.fps = width, height, max(int(fps), 1)
        self.frames = 0
        self._next = time.time()

    def read_jpeg(self):
        if self._next > time.time():
            time.sleep(self._next - time.time())
        self._next = max(self._next + 1.0 / self.fps, time.time())
        f = np.zeros((self.h, self.w, 3), np.uint8)
        f[:, :, 0] = 40
        x = int((self.frames * 6) % max(self.w - 200, 1))
        cv2.rectangle(f, (x, self.h // 3), (x + 160, self.h // 3 + 160), (0, 200, 255), -1)
        cv2.putText(f, "AUV SIM F%06d" % self.frames, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(f, time.strftime("%H:%M:%S"), (20, self.h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        self.frames += 1
        ok, enc = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return enc if ok else None

    def close(self):
        pass


# ===========================================================================
# 接收端（水面 PC）
# ===========================================================================
class JpegReassembler(object):
    """把跨 UDP 数据报的 JPEG 字节流还原成整帧（按 SOI/EOI 标记切分）。"""

    def __init__(self, max_buffer=8 * 1024 * 1024):
        self.buf = bytearray()
        self.max_buffer = max_buffer
        self.bad = 0

    def feed(self, data):
        self.buf += data
        frames = []
        while True:
            start = self.buf.find(JPEG_SOI)
            if start < 0:
                self.buf.clear()
                break
            if start > 0:
                del self.buf[:start]
            end = self.buf.find(JPEG_EOI, 3)
            if end < 0:
                if len(self.buf) > self.max_buffer:
                    self.bad += 1
                    del self.buf[:3]
                break
            frames.append(bytes(self.buf[:end + 2]))
            del self.buf[:end + 2]
        return frames


class Stats(object):
    def __init__(self):
        self.frames = 0
        self.bytes = 0
        self.size_sum = 0
        self.size_max = 0
        self.t0 = time.time()
        self.last_t = self.t0
        self.max_gap = 0.0
        self.decode_fail = 0

    def add(self, nbytes):
        now = time.time()
        gap = now - self.last_t
        self.last_t = now
        if self.frames:
            self.max_gap = max(self.max_gap, gap)
        self.frames += 1
        self.bytes += nbytes
        self.size_sum += nbytes
        self.size_max = max(self.size_max, nbytes)

    def report(self):
        dt = max(time.time() - self.t0, 1e-6)
        return ("%d 帧 / %.1fs = %.1f fps | %.1f Mbps | 帧 %.1f KB(均) / %.1f KB(峰) | 最大间隔 %.1f ms"
                % (self.frames, dt, self.frames / dt, self.bytes * 8 / 1e6 / dt,
                   self.size_sum / max(self.frames, 1) / 1024, self.size_max / 1024,
                   self.max_gap * 1000))


class Sink(object):
    """显示/录制端：永远只用最新帧，来不及就丢。

    save_fps=0 时按前若干帧实测帧率再建 writer（避免"30fps 写 25fps 流"造成的时长失真）。
    """

    def __init__(self, show=True, scale=1.0, save=None, save_fps=0.0, probe_frames=60,
                 save_raw=None):
        if cv2 is None and (show or save):
            raise RuntimeError("接收端需要 opencv-python/numpy")
        self.show = show
        self.scale = scale
        self.save_path = save
        self.save_raw = save_raw          # 原始 MJPEG 直存（零解码/零编码，不丢帧）
        self.save_fps = save_fps
        self.probe_frames = max(int(probe_frames), 1)
        self.writer = None
        self._buf = []
        self._t_first = None
        self._raw_f = None
        self._raw_n = 0
        self._raw_t0 = None
        self._raw_t1 = None
        if save_raw:
            self._raw_f = open(save_raw, "wb")
            print("[VIEW] 原始 MJPEG 直存 → %s（不丢帧；结束时会给出转 mp4 命令）" % save_raw)

    def _open_writer(self, shape, fps):
        h, w = shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*("mp4v" if self.save_path.endswith(".mp4") else "MJPG"))
        self.writer = cv2.VideoWriter(self.save_path, fourcc, fps, (w, h))
        print("[VIEW] 录制 → %s (%dx%d@%.1ffps)" % (self.save_path, w, h, fps))

    def push(self, jpg, stats):
        if self._raw_f is not None:                  # 直存：不解码，最快
            self._raw_f.write(jpg)
            self._raw_n += 1
            now = time.time()
            self._raw_t0 = now if self._raw_t0 is None else self._raw_t0
            self._raw_t1 = now
        if not self.show and self.writer is None and self.save_path is None:
            stats.add(len(jpg))                      # 不显示也不转码：只统计，省 CPU
            return
        if cv2 is None:
            stats.add(len(jpg))
            return
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            stats.decode_fail += 1
            return
        stats.add(len(jpg))
        if self.show:
            disp = img if self.scale == 1.0 else cv2.resize(img, None, fx=self.scale, fy=self.scale)
            cv2.imshow("AUV front camera", disp)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                raise KeyboardInterrupt
        if not self.save_path:
            return
        if self.writer is not None:
            self.writer.write(img)
            return
        # 前 probe_frames 帧先缓存，用来量真实帧率
        if self._t_first is None:
            self._t_first = time.time()
        self._buf.append(img)
        elapsed = time.time() - self._t_first
        if self.save_fps:
            fps = self.save_fps
        elif len(self._buf) >= self.probe_frames and elapsed > 0:
            fps = len(self._buf) / elapsed
        else:
            return
        fps = max(1.0, min(float(fps), 120.0))
        self._open_writer(img.shape, fps)
        for f in self._buf:
            self.writer.write(f)
        self._buf = []

    def close(self):
        if self._raw_f is not None:
            self._raw_f.close()
            dt = max((self._raw_t1 or 0) - (self._raw_t0 or 0), 1e-6)
            fps = self._raw_n / dt if self._raw_n > 1 else 0.0
            print("[VIEW] 原始 MJPEG 已保存：%d 帧 / %.1fs ≈ %.1f fps → %s"
                  % (self._raw_n, dt, fps, self.save_raw))
            if fps > 0:
                out = self.save_raw.rsplit(".", 1)[0] + ".mp4"
                print("[VIEW] 转 mp4（零转码）: ffmpeg -f mjpeg -r %.2f -i %s -c:v copy %s"
                      % (fps, self.save_raw, out))
        if self.save_path and self.writer is None and self._buf:      # 太短：按实测/默认建 writer
            fps = self.save_fps or (len(self._buf) / max(time.time() - self._t_first, 1e-6) if self._t_first else 30.0)
            self._open_writer(self._buf[0].shape, max(1.0, min(float(fps), 120.0)))
            for f in self._buf:
                self.writer.write(f)
            self._buf = []
        if self.writer is not None:
            self.writer.release()
            print("[VIEW] 录像文件已保存")
        if self.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


class UdpReceiver(object):
    """UDP 原始 MJPEG 接收：先把内核里积压的数据报取干，只解码最新帧 → 不累积延迟。"""

    def __init__(self, port=5000, bind="0.0.0.0", rcvbuf=4 * 1024 * 1024):
        self.port = int(port)
        self.bind = bind
        self.rcvbuf = rcvbuf

    def run(self, sink, stats, duration=0.0, max_frames=0, stats_interval=2.0, quiet=False):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        want = self.rcvbuf
        while want >= 212992:
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, want)
                break
            except OSError:
                want //= 2                                  # 超 net.core.rmem_max，逐步降级
        sock.bind((self.bind, self.port))
        sock.setblocking(False)
        if not quiet:
            print("[VIEW] 监听 udp://%s:%d（接收缓冲 %d 字节%s）"
                  % (self.bind, self.port, want, "，已降级" if want != self.rcvbuf else ""))
        reasm = JpegReassembler()
        t0 = time.time()
        last_report = t0
        warned = False
        while True:
            readable, _, _ = select.select([sock], [], [], 0.05)
            while readable:
                try:
                    data, _ = sock.recvfrom(65536)
                except BlockingIOError:
                    break
                if not data:
                    break
                for jpg in reasm.feed(data):
                    sink.push(jpg, stats)
                readable, _, _ = select.select([sock], [], [], 0)
            now = time.time()
            if duration and now - t0 >= duration:
                break
            if max_frames and stats.frames >= max_frames:
                break
            if now - last_report >= stats_interval:
                last_report = now
                print("[VIEW] %s" % stats.report())
                if stats.frames == 0 and not warned and now - t0 > 5:
                    warned = True
                    print("[VIEW] 还没收到帧：确认板端在推、PC 防火墙放行 UDP %d、IP/网段一致" % self.port)
        sock.close()


class HttpReceiver(object):
    """HTTP multipart/x-mixed-replace 接收（浏览器同款协议）。"""

    def __init__(self, url):
        self.url = url

    def run(self, sink, stats, duration=0.0, max_frames=0, stats_interval=2.0, quiet=False):
        if not quiet:
            print("[VIEW] 拉流 %s" % self.url)
        t0 = time.time()
        last_report = t0
        with urllib.request.urlopen(self.url, timeout=10) as r:
            ctype = r.headers.get("Content-Type", "")
            boundary = b"--" + (ctype.split("boundary=")[1].strip().strip('"').encode()
                                if "boundary=" in ctype else b"auvframe")
            if not quiet:
                print("[VIEW] Content-Type: %s" % ctype)
            buf = bytearray()
            while True:
                chunk = r.read(65536)
                if not chunk:
                    break
                buf += chunk
                while True:
                    i = buf.find(boundary)
                    if i < 0:
                        break
                    j = buf.find(b"Content-Length: ", i)
                    if j < 0:
                        break
                    k = buf.find(b"\r\n", j)
                    length = int(bytes(buf[j + 16:k]).strip())
                    hdr_end = buf.find(b"\r\n\r\n", k)
                    if hdr_end < 0:
                        break
                    start = hdr_end + 4
                    if len(buf) < start + length:
                        break
                    jpg = bytes(buf[start:start + length])
                    del buf[:start + length]
                    sink.push(jpg, stats)
                now = time.time()
                if duration and now - t0 >= duration:
                    break
                if max_frames and stats.frames >= max_frames:
                    break
                if now - last_report >= stats_interval:
                    last_report = now
                    print("[VIEW] %s" % stats.report())


# ===========================================================================
# 命令行：python3 -m manual.stream push|view
#   push — 无录像时单独推流（会独占相机；手动模式一般由 manual.sh 走 camera 钩子）
#   view — 水面 PC 端看流/录制
# ===========================================================================
def _cli_push(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="stream.py push", description="单独推流（独占相机）")
    ap.add_argument("--host", default=None, help="UDP 目标 PC IP")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--pkt", type=int, default=8000)
    ap.add_argument("--stream-fps", type=float, default=30.0, help="0=不限速")
    ap.add_argument("--http", type=int, default=0, help="同时开 HTTP MJPEG 端口（0=关）")
    ap.add_argument("--dev", default="/dev/video0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--camera-sim", action="store_true", help="无相机自测")
    ap.add_argument("--duration", type=float, default=0.0)
    a = ap.parse_args(argv)
    if not a.host and not a.http:
        print("至少给 --host 或 --http", file=sys.stderr)
        return 2
    if a.camera_sim:
        src, mode = SimSource(a.width, a.height, a.fps), "bgr"
    else:
        cap, mode = open_raw_camera(a.dev, a.width, a.height, a.fps)
        src = CameraSource(cap, mode)
    pusher = MjpegPusher(a.host, a.port, pkt=a.pkt, stream_fps=a.stream_fps) if a.host else None
    http = MjpegHttpServer(a.http, stream_fps=a.stream_fps) if a.http else None
    if http:
        http.start()
    print("[PUSH] 源=%s 模式=%s → %s" % ("sim" if a.camera_sim else a.dev, mode,
                                         ("udp://%s:%d" % (a.host, a.port)) if pusher else "")
          + (" http://<本机IP>:%d/" % a.http if http else ""))
    t0 = time.time()
    try:
        while True:
            jpg = src.read_jpeg()
            if jpg is None:
                time.sleep(0.005)
                continue
            if pusher:
                pusher.offer(jpg)
            if http:
                http.offer(jpg)
            if a.duration and time.time() - t0 >= a.duration:
                break
    except KeyboardInterrupt:
        print("\n[PUSH] 停止")
    finally:
        if pusher:
            s = pusher.stats()
            print("[PUSH] 结束：%d 帧（%.1f fps / %.1f Mbps）" % (s["frames_sent"], s["fps"], s["mbps"]))
            pusher.close()
        if http:
            http.close()
        src.close()
    return 0


def _cli_view(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="stream.py view", description="PC 端看流/录制")
    ap.add_argument("--port", type=int, default=5000, help="UDP 端口")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--http", default=None, help="改拉 HTTP MJPEG，如 http://192.168.137.10:8080/")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--save", default=None, help="保存视频到指定路径（.mp4/.avi）")
    ap.add_argument("--record", action="store_true",
                    help="在 PC 端录成 mp4 到 --out-dir（默认 record/）；PC 慢时会丢帧")
    ap.add_argument("--record-raw", action="store_true",
                    help="在 PC 端直存原始 MJPEG 到 --out-dir（零转码、不丢帧，事后一行转 mp4）")
    ap.add_argument("--out-dir", default="record", help="--record 的保存目录（默认 record）")
    ap.add_argument("--save-fps", type=float, default=0.0, help="录像帧率（0=按实测帧率）")
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--stats-interval", type=float, default=2.0)
    a = ap.parse_args(argv)
    if cv2 is None:
        print("需要 opencv-python/numpy；或用 ffplay：\n"
              "  ffplay -fflags nobuffer -flags low_delay -framedrop -f mjpeg \"udp://@:%d\"" % a.port,
              file=sys.stderr)
        return 2
    save_path = a.save
    save_raw = None
    if (a.record or a.record_raw) and not save_path:
        os.makedirs(a.out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        if a.record_raw:
            save_raw = os.path.join(a.out_dir, "auv_%s.mjpeg" % ts)
        else:
            save_path = os.path.join(a.out_dir, "auv_%s.mp4" % ts)
    stats = Stats()
    sink = Sink(show=not a.no_display, scale=a.scale, save=save_path, save_fps=a.save_fps,
                save_raw=save_raw)
    try:
        if a.http:
            HttpReceiver(a.http).run(sink, stats, a.duration, a.max_frames, a.stats_interval)
        else:
            UdpReceiver(a.port, a.bind).run(sink, stats, a.duration, a.max_frames, a.stats_interval)
    except KeyboardInterrupt:
        print("\n[VIEW] 手动停止")
    finally:
        sink.close()
    print("[VIEW] 结束：%s | 解码失败 %d" % (stats.report(), stats.decode_fail))
    return 0


def main(argv=None):
    import sys as _sys
    argv = list(_sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("用法:\n"
              "  python3 -m manual.stream push --host <PC_IP> [--port 5000] [--http 8080]\n"
              "  python3 -m manual.stream view [--port 5000] [--http URL] [--no-display]\n"
              "      [--record | --record-raw] [--out-dir record]\n"
              "（手动模式请直接用根目录 manual.sh）")
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "push":
        return _cli_push(rest)
    if cmd == "view":
        return _cli_view(rest)
    print("未知子命令 %r（可选 push / view）" % cmd, file=sys.stderr)
    return 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
