# -*- coding: utf-8 -*-
"""pc_recorder.py - PC-side recorder for the AUV front-camera stream.

Runs on the water-surface PC (not on the RDK). It receives the MJPEG stream that
the RDK pushes and saves it into a local folder (default `record/`).

Two recording paths:

  raw (default)  zero-transcode: incoming JPEG frames are appended to a `.mjpeg`
                 file (no decode / no encode -> no frame drops, ~0 CPU) plus a
                 `.timestamps` sidecar, and at the end it is packaged into an
                 `.mp4` with `ffmpeg -c:v copy` (the `.mjpeg` is removed unless
                 `--keep-raw`). Also supports `--show` to watch live in the same
                 process (a UDP port can only be received by one process, so this
                 is the way to watch *and* record at once).

  --mp4          delegate to ffmpeg (`-c:v copy`, zero-transcode) and write a
                 real `.mp4` directly. No live window in this mode (ffmpeg owns
                 the socket).

Examples:
    python pc_recorder.py                                  # 录到 record/auv_<ts>.mp4
    python pc_recorder.py --show                           # 边看边录
    python pc_recorder.py --mp4 --duration 60              # 直接用 ffmpeg 录 mp4
    python pc_recorder.py --http http://192.168.137.10:8080/
    python pc_recorder.py --remux record/auv_20260910_165344.mjpeg

Companion scripts in this folder:
    pc_keyboard_client.py   keyboard remote (PC -> RDK udp_server.py:9000)
"""
import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

JPEG_SOI = b"\xff\xd8\xff"
JPEG_EOI = b"\xff\xd9"

try:
    import cv2
    import numpy as np
except ImportError:                      # raw 录制/统计不需要 cv2
    cv2 = None
    np = None


# ---------------------------------------------------------------------------
# JPEG 重组（跨 UDP 数据报）
# ---------------------------------------------------------------------------
class JpegReassembler(object):
    def __init__(self, max_buffer=16 * 1024 * 1024):
        self.buf = bytearray()
        self.max_buffer = max_buffer

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
                if len(self.buf) > self.max_buffer:          # 病态数据：重新同步
                    del self.buf[:3]
                break
            frames.append(bytes(self.buf[:end + 2]))
            del self.buf[:end + 2]
        return frames


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
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
        return ("%d frames / %.1fs = %.1f fps | %.1f Mbps | frame %.1f KB(avg) / %.1f KB(max) | max gap %.1f ms"
                % (self.frames, dt, self.frames / dt, self.bytes * 8 / 1e6 / dt,
                   self.size_sum / max(self.frames, 1) / 1024, self.size_max / 1024,
                   self.max_gap * 1000))

    def fps(self):
        dt = max(time.time() - self.t0, 1e-6)
        return self.frames / dt


# ---------------------------------------------------------------------------
# 录像器
# ---------------------------------------------------------------------------
class Recorder(object):
    def __init__(self, args, out_path):
        self.args = args
        self.out_path = out_path
        self.stats = Stats()
        self.raw = None
        self.ts = None
        self.writer = None
        self.n = 0
        self.t_first = None
        self.t_last = None
        self.show = bool(args.show)

    # -- 原始 MJPEG 直存 ---------------------------------------------------
    def open_raw(self):
        if self.args.discard:
            self.raw = None
            self.ts = None
            print("[REC] discard mode: no file will be written")
        else:
            self.raw = open(self.out_path, "wb")
            self.ts = open(self.out_path + ".timestamps", "w")
            self.ts.write("# frame_index\tarrival_unix\tdelta_ms\n")
            print("[REC] raw MJPEG -> %s  (zero-transcode, no frame drops)" % self.out_path)
        if self.show and cv2 is None:
            print("[REC] --show needs opencv-python; recording only")
            self.show = False

    def write_raw(self, jpg):
        now = time.time()
        if self.raw is not None:
            self.raw.write(jpg)
            self.ts.write("%d\t%.6f\t%.2f\n"
                          % (self.n, now, (now - self.t_last) * 1000 if self.t_last else 0.0))
        self.t_first = now if self.t_first is None else self.t_first
        self.t_last = now
        self.n += 1
        if self.show:
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                cv2.imshow("AUV front camera (recording)", img)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    raise KeyboardInterrupt

    def close_raw(self):
        if self.raw is not None:
            self.raw.close()
            self.ts.close()
        dt = max((self.t_last or 0) - (self.t_first or 0), 1e-6)
        fps = (self.n - 1) / dt if self.n > 1 else 0.0
        if self.raw is None:
            if self.args.discard:
                print("[REC] discard mode done: %d frames / %.1fs ~= %.2f fps" % (self.n, dt, fps))
            # 绑定失败等提前中止：不打印、不提示 remux
        else:
            print("[REC] raw saved %d frames / %.1fs ~= %.2f fps -> %s" % (self.n, dt, fps, self.out_path))
            self._remux_after_record(fps)
        if self.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    def _remux_after_record(self, fps):
        """把刚落盘的 .mjpeg 零转码封成 .mp4（默认产物），中间文件默认删除。"""
        mp4 = os.path.splitext(self.out_path)[0] + ".mp4"
        ff = shutil.which("ffmpeg")
        if self.args.no_remux or not ff or fps <= 0:
            print("[REC] 录像文件（原始 MJPEG）: %s" % self.out_path)
            print("[REC] 想看 mp4 就执行: ffmpeg -f mjpeg -r %.2f -i %s -c:v copy %s"
                  % (fps or 25.0, self.out_path, mp4))
            return
        cmd = [ff, "-hide_banner", "-loglevel", "warning", "-y", "-f", "mjpeg",
               "-r", "%.4f" % fps, "-i", self.out_path, "-c:v", "copy", mp4]
        rc = subprocess.call(cmd)
        if rc != 0:
            print("[REC] remux 失败，保留原始文件: %s" % self.out_path)
            return
        print("[REC] 录像文件（mp4，零转码）: %s" % mp4)
        if self.args.keep_raw:
            print("[REC] 原始 MJPEG 保留: %s" % self.out_path)
        else:
            try:
                os.remove(self.out_path)
                print("[REC] 已删除中间 .mjpeg（--keep-raw 可保留）")
            except OSError:
                pass

    # -- cv2 直接写 mp4（PC 慢时会丢帧，仅在没有 ffmpeg 时用） --------------
    def write_mp4_cv2(self, jpg):
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            self.stats.decode_fail += 1
            return
        if self.writer is None:
            h, w = img.shape[:2]
            fps = self.args.save_fps or max(1.0, self.stats.fps() * 2)   # 先高估，随后按实测重开不现实；简单起见用给定值
            fps = self.args.save_fps or 25.0
            self.writer = cv2.VideoWriter(self.out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                          fps, (w, h))
            print("[REC] cv2 mp4 -> %s (%.0f fps)" % (self.out_path, fps))
        self.writer.write(img)

    def close_mp4_cv2(self):
        if self.writer is not None:
            self.writer.release()
            print("[REC] saved %d frames -> %s" % (self.n, self.out_path))

    # -- 统计 --------------------------------------------------------------
    def tick_stats(self, interval):
        now = time.time()
        if now - getattr(self, "_last_report", 0) >= interval:
            self._last_report = now
            print("[REC] %s" % self.stats.report())


# ---------------------------------------------------------------------------
# 三种输入：UDP raw MJPEG / HTTP MJPEG / ffmpeg 代理
# ---------------------------------------------------------------------------
def run_udp(args, rec):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    want = args.rcvbuf
    while want >= 212992:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, want)
            break
        except OSError:
            want //= 2
    try:
        sock.bind((args.bind, args.port))
    except OSError as e:
        print("[REC] 绑定 UDP %s:%d 失败: %s" % (args.bind, args.port, e), file=sys.stderr)
        print("[REC] 端口被占用（多半是上一次录制/查看没退干净）。处理：", file=sys.stderr)
        print("      Linux/macOS: ss -lunp | grep ':%d'  → kill <PID>   或  pkill -f pc_recorder.py" % args.port,
              file=sys.stderr)
        print("      Windows    : netstat -ano | findstr :%d  → taskkill /PID <PID> /F" % args.port,
              file=sys.stderr)
        print("      或换端口   : ./pc.sh --port 5010   （板端同时 ./manual.sh --port 5010）", file=sys.stderr)
        return 3
    sock.setblocking(False)
    if args.mode == "raw":                                   # 绑定成功后才建文件，失败不留空文件
        rec.open_raw()
    print("[REC] listening udp://%s:%d (rcvbuf %d)" % (args.bind, args.port, want))
    import select
    reasm = JpegReassembler()
    t0 = time.time()
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
                rec.stats.add(len(jpg))
                if args.mode == "raw":
                    rec.write_raw(jpg)
                else:
                    rec.write_mp4_cv2(jpg)
            readable, _, _ = select.select([sock], [], [], 0)
        rec.tick_stats(args.stats_interval)
        now = time.time()
        if args.duration and now - t0 >= args.duration:
            break
        if not warned and rec.stats.frames == 0 and now - t0 > 5:
            warned = True
            print("[REC] no frame yet: check the RDK is pushing, Windows firewall allows UDP %d, same subnet" % args.port)
    sock.close()


def run_http(args, rec):
    print("[REC] pulling %s" % args.http)
    t0 = time.time()
    with urllib.request.urlopen(args.http, timeout=10) as r:
        if args.mode == "raw":
            rec.open_raw()
        ctype = r.headers.get("Content-Type", "")
        boundary = b"--" + (ctype.split("boundary=")[1].strip().strip('"').encode()
                            if "boundary=" in ctype else b"auvframe")
        print("[REC] Content-Type: %s" % ctype)
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
                rec.stats.add(len(jpg))
                if args.mode == "raw":
                    rec.write_raw(jpg)
                else:
                    rec.write_mp4_cv2(jpg)
            rec.tick_stats(args.stats_interval)
            if args.duration and time.time() - t0 >= args.duration:
                break


def run_ffmpeg(args, out_path):
    ff = shutil.which("ffmpeg")
    if not ff:
        print("[REC] --mp4 needs ffmpeg in PATH (or use the default raw mode)", file=sys.stderr)
        return 2
    if args.http:
        src = ["-f", "mpjpeg", "-i", args.http]
    else:
        src = ["-f", "mjpeg", "-use_wallclock_as_timestamps", "1",
               "-i", "udp://@:%d?overrun_nonfatal=1&fifo_size=2000000" % args.port]
    cmd = [ff, "-hide_banner", "-loglevel", "info", "-y"] + src + \
          ["-c:v", "copy", "-flush_packets", "1"]
    if args.duration:
        cmd += ["-t", str(args.duration)]
    cmd += [out_path]
    print("[REC] ffmpeg (zero-transcode copy) -> %s" % out_path)
    print("      " + " ".join(cmd))
    return subprocess.call(cmd)


def remux(args):
    ff = shutil.which("ffmpeg")
    if not ff:
        print("[REC] --remux needs ffmpeg in PATH", file=sys.stderr)
        return 2
    src = args.remux
    dst = args.out if args.out else os.path.splitext(src)[0] + ".mp4"
    fps = args.save_fps
    if not fps:
        ts = src + ".timestamps"
        if os.path.exists(ts):
            times = []
            with open(ts) as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        times.append(float(parts[1]))
            if len(times) > 1:
                fps = (len(times) - 1) / max(times[-1] - times[0], 1e-6)
    if not fps:
        fps = 25.0
    cmd = [ff, "-hide_banner", "-loglevel", "warning", "-y", "-f", "mjpeg",
           "-r", "%.4f" % fps, "-i", src, "-c:v", "copy", dst]
    print("[REC] remux %.2f fps: %s" % (fps, " ".join(cmd)))
    rc = subprocess.call(cmd)
    if rc == 0:
        print("[REC] -> %s" % dst)
    return rc


# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="PC-side recorder for the AUV front-camera MJPEG stream",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=5000, help="UDP port (default 5000)")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--rcvbuf", type=int, default=4 * 1024 * 1024,
                    help="UDP receive buffer (auto-degrades above net.core.rmem_max)")
    ap.add_argument("--http", default=None, help="pull HTTP MJPEG instead, e.g. http://<rdk>:8080/")
    ap.add_argument("--out-dir", default="record", help="output folder (default: record)")
    ap.add_argument("--name", default=None, help="file name without extension (default: auv_<timestamp>)")
    ap.add_argument("--mode", choices=["raw", "mp4"], default="raw",
                    help="raw = zero-transcode .mjpeg (default); mp4 = cv2 mp4 (may drop frames)")
    ap.add_argument("--mp4", dest="mp4_ffmpeg", action="store_true",
                    help="record a real .mp4 via ffmpeg -c:v copy (zero-transcode, no live view)")
    ap.add_argument("--show", action="store_true", help="raw mode: also show the live picture")
    ap.add_argument("--discard", action="store_true",
                    help="do not write any file (view/monitor only; use with --show)")
    ap.add_argument("--no-remux", action="store_true",
                    help="raw mode: keep .mjpeg only, do not package an .mp4 at the end")
    ap.add_argument("--keep-raw", action="store_true",
                    help="raw mode: keep the intermediate .mjpeg as well as the .mp4")
    ap.add_argument("--save-fps", type=float, default=0.0, help="writer fps for mp4 mode (0=25)")
    ap.add_argument("--duration", type=float, default=0.0, help="record N seconds (0=until Ctrl-C)")
    ap.add_argument("--stats-interval", type=float, default=2.0)
    ap.add_argument("--remux", metavar="MJPEG_FILE", help="convert an existing .mjpeg to .mp4 (zero-transcode)")
    ap.add_argument("--out", default=None, help="--remux output path (default: same name .mp4)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.remux:
        return remux(args)

    os.makedirs(args.out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    stem = args.name or ("auv_%s" % ts)

    if args.mp4_ffmpeg:
        return run_ffmpeg(args, os.path.join(args.out_dir, stem + ".mp4"))

    if args.mode == "mp4" and cv2 is None:
        print("[REC] mp4 mode needs opencv-python; use --mp4 (ffmpeg) or raw mode", file=sys.stderr)
        return 2

    out_path = os.path.join(args.out_dir, stem + (".mjpeg" if args.mode == "raw" else ".mp4"))
    rec = Recorder(args, out_path)
    try:
        if args.http:
            run_http(args, rec)
        else:
            run_udp(args, rec)
    except KeyboardInterrupt:
        print("\n[REC] stopped by user")
    finally:
        if args.mode == "raw":
            rec.close_raw()
        else:
            rec.close_mp4_cv2()
    print("[REC] %s" % rec.stats.report())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
