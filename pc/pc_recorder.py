# -*- coding: utf-8 -*-
"""pc_recorder.py - PC-side recorder for the AUV front-camera stream.

Runs on the water-surface PC (not on the RDK). It receives the MJPEG stream that
the RDK pushes.

Layout (2026-09-23 起): **`record/` 只是 tmp（录制期间的中转）**，录完自动归档到
`--save-dir`（默认 `<工程根>/RDKX5-YOLOv11n-/raw-data`）下的**时间戳子目录**
`<save-dir>/<录像名>/`，视频流与它的 `.timestamps` **一起**搬过去。

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
import re
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
# 归档：record/ 当 tmp（临时中转），raw-data/<时间戳子目录>/ 当保存区
#   每段录像的产物（视频流 + 对应 .timestamps）**一起**放进一个以时间命名的子目录，
#   子目录名 = 录像名（如 auv_20260923_000228）⇒ 一眼能对上，也不会和别的段混在一起。
# ---------------------------------------------------------------------------
def default_save_dir():
    """保存区默认位置：<工程根>/RDKX5-YOLOv11n-/raw-data（与 pc/ 同级）。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "RDKX5-YOLOv11n-", "raw-data")


def stem_of(path):
    """去掉扩展名与 .timestamps 后缀，得到"录像名"。"""
    base = os.path.basename(path)
    for ext in (".mjpeg.timestamps", ".timestamps", ".mjpeg", ".mp4"):
        if base.endswith(ext):
            return base[:-len(ext)]
    return os.path.splitext(base)[0]


def archive_run(tmp_dir, save_dir, stem, also=()):
    """把一段录像的所有产物搬进 `save_dir/<stem>/`，返回搬过去的路径列表。

    视频（.mjpeg / .mp4）与它的 `.timestamps` 永远一起走 —— 这是本函数的唯一目的。
    0 字节文件不搬（启动即失败的残file 留在 tmp 自生自灭）。
    """
    vsz = 0
    for ext in (".mjpeg", ".mp4"):
        q = os.path.join(tmp_dir, stem + ext)
        if os.path.exists(q):
            vsz = max(vsz, os.path.getsize(q))
    if vsz <= 0:
        print("[REC] 没录到数据（0 字节视频）→ 不归档，残file 留在 tmp：%s" % stem)
        return []
    sub = os.path.join(save_dir, stem)
    os.makedirs(sub, exist_ok=True)
    moved = []
    for ext in (".mjpeg", ".mjpeg.timestamps", ".mp4", ".timestamps"):
        src = os.path.join(tmp_dir, stem + ext)
        if os.path.exists(src) and os.path.getsize(src) > 0:
            dst = os.path.join(sub, os.path.basename(src))
            shutil.move(src, dst)
            moved.append(dst)
    for p in also:
        if p and os.path.exists(p):
            dst = os.path.join(sub, os.path.basename(p))
            try:
                shutil.copy2(p, dst)
                moved.append(dst)
            except OSError:
                pass
    return moved


def adopt_tmp(tmp_dir, save_dir, min_age_s=5.0, quiet=False):
    """把 tmp 里**上一次**遗留的录像搬进保存区。

    为什么需要：进程被 kill / 崩溃时来不及归档，文件会留在 tmp；而 tmp 是"可清"的目录，
    不搬就等于把素材放在垃圾桶里（2026-09-22 那次 rm 事故就是这么丢的）。
    `min_age_s` 内的新文件不动，避免误搬正在写的那个。
    """
    if not os.path.isdir(tmp_dir):
        return []
    now = time.time()
    groups = {}
    for fn in sorted(os.listdir(tmp_dir)):
        if not fn.startswith("auv_"):
            continue
        p = os.path.join(tmp_dir, fn)
        if os.path.isfile(p):
            groups.setdefault(stem_of(fn), []).append(p)
    out = []
    for stem, paths in groups.items():
        vids = [q for q in paths if q.endswith((".mjpeg", ".mp4")) and os.path.getsize(q) > 0]
        if not vids:
            continue                        # 没有非空视频（只有 timestamps / 0 字节残file）→ 不管
        # "还在不在写"只看**视频**的新鲜度：sidecar 的 mtime 可能因为写入顺序略有差别，
        # 按组判断才不会把视频搬走却把它的 timestamps 落下（那正是要防的事）。
        try:
            if now - max(os.path.getmtime(q) for q in vids) < min_age_s:
                continue
        except OSError:
            continue
        sub = os.path.join(save_dir, stem)
        os.makedirs(sub, exist_ok=True)
        for q in paths:
            dst = os.path.join(sub, os.path.basename(q))
            shutil.move(q, dst)
            out.append(dst)
        if not quiet:
            print("[REC] tmp 里发现上次遗留的录像 → 已归档：%s（%d 个文件）→ %s/"
                  % (stem, len(paths), sub))
    return out


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
            # ⚠️ buffering=1（行缓冲）：`.timestamps` 是**每帧一行**写的，进程被 kill 时
            #    块缓冲会丢掉尾部（2026-09-22 实测：真值只剩 882/1094 帧，静默少 8.5 s）。
            #    行缓冲 = 每帧 flush，代价可忽略（25 fps × 一行文本）。
            self.ts = open(self.out_path + ".timestamps", "w", buffering=1)
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

    def archive(self):
        """把本次产物搬进保存区（--discard / --no-archive 时不动）。"""
        if getattr(self.args, "discard", False) or getattr(self.args, "no_archive", False):
            return []
        tmp_dir = os.path.dirname(os.path.abspath(self.out_path)) or "."
        moved = archive_run(tmp_dir, self.args.save_dir, stem_of(self.out_path))
        if moved:
            print("[REC] 已归档 → %s/（%d 个文件：视频 + timestamps 一起）"
                  % (os.path.dirname(moved[0]), len(moved)))
        return moved

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
            self.archive()
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
            self.archive()

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
    rc = subprocess.call(cmd)
    if rc == 0 and not getattr(args, "no_archive", False):
        moved = archive_run(os.path.dirname(os.path.abspath(out_path)), args.save_dir,
                            stem_of(out_path))
        if moved:
            print("[REC] 已归档 → %s/" % os.path.dirname(moved[0]))
    return rc


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
    ap.add_argument("--tmp-dir", "--out-dir", dest="tmp_dir", default="record",
                    help="中转目录（tmp）：录制期间写这里，结束后归档搬走（默认 record/；"
                         "--out-dir 是它的旧名）")
    ap.add_argument("--save-dir", default=default_save_dir(),
                    help="保存目录：每段录像进 <save-dir>/<录像名>/ 子目录（默认 "
                         "RDKX5-YOLOv11n-/raw-data）")
    ap.add_argument("--no-archive", action="store_true",
                    help="关掉归档：产物就留在 tmp（老行为）")
    ap.add_argument("--no-adopt", action="store_true",
                    help="不接管 tmp 里上次遗留的录像（默认会替它归档）")
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

    os.makedirs(args.tmp_dir, exist_ok=True)
    if not args.no_archive:
        os.makedirs(args.save_dir, exist_ok=True)
        if not args.no_adopt and not args.discard:
            adopt_tmp(args.tmp_dir, args.save_dir)        # 上次崩溃/被杀留下的，先搬去保存区
    ts = time.strftime("%Y%m%d_%H%M%S")
    stem = args.name or ("auv_%s" % ts)

    if args.mp4_ffmpeg:
        return run_ffmpeg(args, os.path.join(args.tmp_dir, stem + ".mp4"))

    if args.mode == "mp4" and cv2 is None:
        print("[REC] mp4 mode needs opencv-python; use --mp4 (ffmpeg) or raw mode", file=sys.stderr)
        return 2

    out_path = os.path.join(args.tmp_dir, stem + (".mjpeg" if args.mode == "raw" else ".mp4"))
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
