#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""carve_mjpeg.py — 从磁盘空闲区把**被误删的 MJPEG 录像**雕回来（只读扫描，绝不写设备）

2026-09-22 事故：`pc.sh` 里那行 `rm -f "${OUT_DIR}"/auv_*` 把 `pc/record/` 下 13 个录像
（≈2.9 GB / ≈15k 帧）全删了。`rm` 只删目录项 + 把数据块标记为空闲，**数据本身还在盘上**，
直到被覆盖或被 TRIM。本机 `/` 是 ext4 且挂载选项里**没有 discard**、`fstrim` 上次运行在删除
之前 ⇒ 数据大概率完整。

为什么能雕：MJPEG 就是**一串首尾相接的完整 JPEG**。所以不看 inode，直接在原始设备上找
`FFD8FF … FFD9`，并要求**下一帧的 `FFD8FF` 紧接在上一帧 EOI 之后（零间隔）**——
满足这个条件的连续帧链就是一段录像。链断了（有间隔 / 帧大得离谱）就收尾。

两阶段（**先列清单再取数据**，把"边扫边写"的自噬风险降到最低）：

    # ① 只扫一遍，写一个几十 KB 的清单 JSONL（这一步几乎不写盘）
    sudo python3 pc/carve_mjpeg.py --dev /dev/nvme0n1p5 --phase list \\
         --manifest /tmp/carve_20260922.jsonl

    # ② 按清单把每一段读出来落盘
    sudo python3 pc/carve_mjpeg.py --dev /dev/nvme0n1p5 --phase dump \\
         --manifest /tmp/carve_20260922.jsonl --out <输出目录>

⚠️ **输出目录若和被扫的设备在同一个文件系统上**（默认的 `raw-data/` 就是），写出来的数据
   有概率盖掉还没读到的候选块。脚本会检测并警告；想最保险就先写到**别的盘**（外接 U 盘 /
   另一块 NVMe 上的 NTFS 分区），事后再拷进 `RDKX5-YOLOv11n-/raw-data/`。

安全边界（刻意做成不能出事）：
  · 设备只以 `rb` 打开，全程只 `read`/`seek`，没有任何写设备/写 inode 的路径；
  · `--phase list` 不产出任何大文件；`--phase dump` 每个候选先写 `.part`，凑够 `--min-frames`
    才改名，不够就删 `.part`；
  · 输出文件名自带起始偏移，重复跑不会覆盖（同偏移会追加 `_2`）。
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import stat as statmod
import struct
import sys
import time

SOI = b"\xff\xd8\xff"
EOI = b"\xff\xd9"

try:                       # 输出被 `>` 重定向时默认是块缓冲：日志会"卡住"看不到进度，
                           # 万一进程被杀还会整块丢掉（2026-09-23 就吃过这个亏）→ 强制行缓冲
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

#: Linux BLKGETSIZE64：块设备的字节数（块设备的 st_size 是 0，必须走这个 ioctl）
_BLKGETSIZE64 = 0x80081272


def dev_size(path):
    """设备/镜像的字节数（块设备用 ioctl 问，普通文件用 st_size）。"""
    st = os.stat(path)
    if statmod.S_ISBLK(st.st_mode):
        with open(path, "rb") as f:
            return struct.unpack("Q", fcntl.ioctl(f.fileno(), _BLKGETSIZE64, b"\x00" * 8))[0]
    return st.st_size


# ---------------------------------------------------------------------------
# 只读窗口：顺序扫描大设备时只驻留一个 chunk，不把整盘读进内存
# ---------------------------------------------------------------------------
class Window(object):
    def __init__(self, path, start, end, chunk=16 << 20):
        self.path = path
        self.f = open(path, "rb", buffering=0)
        size = dev_size(path)
        self.start = int(start)
        self.base = int(start)
        self.end = int(min(end if end and end > 0 else size, size))
        self.chunk = int(chunk)
        self.buf = b""
        self.f.seek(self.base)

    def close(self):
        try:
            self.f.close()
        except OSError:
            pass

    def _fill_to(self, off_end):
        while self.base + len(self.buf) < off_end and self.base + len(self.buf) < self.end:
            want = min(self.chunk, self.end - (self.base + len(self.buf)))
            data = self.f.read(want)
            if not data:
                break
            self.buf += data

    def find(self, pat, off):
        """在 [off, end) 里找 pat，返回绝对偏移或 -1（跨窗口边界安全）。"""
        off = max(off, self.base)
        while True:
            i = self.buf.find(pat, max(0, off - self.base))
            if i >= 0:
                return self.base + i
            keep = len(pat) - 1
            tail = self.base + len(self.buf)
            if tail >= self.end:
                return -1
            newbase = max(self.base, tail - keep)
            self.buf = self.buf[newbase - self.base:]
            self.base = newbase
            before = len(self.buf)
            self._fill_to(newbase + self.chunk)
            if len(self.buf) == before:
                return -1

    def seek(self, off):
        """跳到 off（丢弃缓冲、直接 seek）。

        ⚠️ 没有这个函数时，取稀疏候选会**从当前位置顺序读到目标**（2026-09-23 实测：
        第一个候选在 483 GB 处 ⇒ 每批等于重扫一遍全盘，8 批要 4 小时）。有它以后
        只读候选本身（25 GB ⇒ 几分钟）。
        """
        self.base = max(self.start, int(off))
        self.buf = b""
        self.f.seek(self.base)

    def read(self, off, n):
        """读 [off, off+n)；窗口覆盖不到就 os.pread 直接读（帧很小，代价可忽略）。

        ⚠️ 不要"取不到就返回 b''"：那会在窗口边界**静默丢帧**（拼出来的文件比统计的短）。
        """
        if self.base <= off and (off - self.base + n) <= len(self.buf):
            return self.buf[off - self.base:off - self.base + n]
        return os.pread(self.f.fileno(), n, off)

    def scanned(self):
        return self.base + len(self.buf)


# ---------------------------------------------------------------------------
# 帧链识别
# ---------------------------------------------------------------------------
def scan_chain(w, off, max_frame, on_frame=None, gap=0):
    """从 off 处的 SOI 开始，数"首尾相接的完整 JPEG"链。返回 (帧数, 总字节, 结束偏移)。

    `gap`：允许帧之间最多有这么多字节的间隔（默认 0 = 严格首尾相接，用于 raw .mjpeg；
    MP4 里 JPEG 常带一个 4 字节长度前缀，那时用 `--gap 8` 才拼得起来）。
    """
    frames = 0
    total = 0
    pos = off
    while True:
        e = w.find(EOI, pos + 3)
        if e < 0:
            break
        flen = e + 2 - pos
        if flen > max_frame or flen < 256:          # 不像一帧 → 断链
            break
        if on_frame is not None:
            on_frame(pos, flen)
        frames += 1
        total += flen
        nxt = pos + flen
        head = w.read(nxt, 3 + gap)
        if head[:3] == SOI:                          # 紧挨着（零间隔）
            pos = nxt
        elif gap > 0:
            k = head.find(SOI, 1)                    # 允许 ≤gap 字节的间隔（跳过前缀）
            if k < 0:
                return frames, total, nxt
            pos = nxt + k
        else:
            return frames, total, nxt
    return frames, total, pos


def nearest_label(frames, expected):
    """把帧数对到"预期清单"里最接近的一项（±3% 内才算命中）。"""
    best = None
    for name, n in expected:
        if n <= 0:
            continue
        rel = abs(frames - n) / float(n)
        if rel <= 0.03 and (best is None or rel < best[2]):
            best = (name, n, rel)
    if best is None:
        return ""
    return "%s（%d 帧，命中 %.1f%%）" % (best[0], best[1], 100.0 * frames / best[1])


def phase_list(args):
    expected = parse_expect(args.expect)
    w = Window(args.dev, args.start, args.end, args.chunk)
    out = open(args.manifest, "w")
    off = w.find(SOI, max(args.start, 0))
    t0 = time.time()
    tlast = t0
    n = 0
    got = 0
    bytes_hit = 0
    try:
        while off >= 0:
            frames, total, endpos = scan_chain(w, off, args.max_frame, gap=args.gap)
            if frames >= args.min_frames and (args.max_frames <= 0 or frames <= args.max_frames):
                rec = {"offset": off, "end": endpos, "frames": frames, "bytes": total,
                       "gap": args.gap,
                       "avg_frame": round(total / max(frames, 1), 1),
                       "label": nearest_label(frames, expected)}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()          # 每条即落盘：进程被杀也不会丢已扫出的候选
                got += 1
                bytes_hit += total
                print("[CARVE] #%-3d 偏移 %-14d 帧 %5d 共 %7.1f MB 均帧 %6.1f KB %s"
                      % (got, off, frames, total / 1048576.0, total / max(frames, 1) / 1024.0,
                         rec["label"]))
            n += 1
            nxt = endpos if endpos > off else off + 3
            off = w.find(SOI, nxt)
            now = time.time()
            if now - tlast >= args.progress:
                tlast = now
                el = max(now - t0, 1e-6)
                print("[CARVE] 进度 %.1f/%.1f GB（%.0f MB/s）已命中 %d 段 / %.2f GB"
                      % (w.scanned() / 1e9, w.end / 1e9,
                         (w.scanned() - args.start) / 1e6 / el, got, bytes_hit / 1e9),
                      file=sys.stderr)
    finally:
        w.close()
        out.close()
    el = time.time() - t0
    print("[CARVE] 扫完：候选 %d 段，累计 %.2f GB，用时 %.1f 分钟 → 清单 %s"
          % (got, bytes_hit / 1e9, el / 60.0, args.manifest))
    if got == 0:
        print("[CARVE] ⚠️ 一段都没找到：可能已被覆盖/TRIM，或 --min-frames 太大，或扫的区间不对")


def phase_dump(args):
    if not args.manifest or not os.path.exists(args.manifest):
        print("!! --phase dump 需要 --manifest（先跑 --phase list）", file=sys.stderr)
        return 2
    os.makedirs(args.out, exist_ok=True)
    warn_same_fs(args.dev, args.out)
    recs = [json.loads(l) for l in open(args.manifest) if l.strip()]
    w = Window(args.dev, args.start, args.end, args.chunk)
    done = 0
    if args.hold_in_ram:
        total = sum(int(r["bytes"]) for r in recs)
        print("[CARVE] --hold-in-ram：先把 %d 段 / %.2f GB 全读进内存，再统一落盘"
              "（写盘时不会污染未读区域）" % (len(recs), total / 1e9))
        held = []
        for i, rec in enumerate(recs, 1):
            chunks = []
            off, frames = int(rec["offset"]), int(rec["frames"])
            w.seek(off)                                    # ← 直接跳过去，别顺序读
            scan_chain(w, off, args.max_frame, on_frame=lambda p, n: chunks.append(w.read(p, n)),
                       gap=int(rec.get("gap", args.gap)))
            held.append((rec, b"".join(chunks)))
            print("[CARVE] 读入 %d/%d：偏移 %d %d 帧 %.1f MB"
                  % (i, len(recs), off, frames, len(held[-1][1]) / 1048576.0))
        w.close()
        for rec, blob in held:
            frames = int(rec["frames"])
            if frames < args.min_frames:
                continue
            path = uniq_path(args.out, "rec_%d_%df.mjpeg" % (int(rec["offset"]), frames))
            with open(path, "wb") as fo:
                fo.write(blob)
            done += 1
            print("[CARVE] %-58s 帧 %5d 共 %7.1f MB %s"
                  % (os.path.basename(path), frames, len(blob) / 1048576.0, rec.get("label", "")))
        print("[CARVE] 落盘 %d 段 → %s" % (done, args.out))
        remux_hint(args.out)
        return 0
    try:
        for i, rec in enumerate(recs, 1):
            off, frames = int(rec["offset"]), int(rec["frames"])
            base = os.path.join(args.out, "rec_%d_%df" % (off, frames))
            path = base + ".mjpeg"
            k = 2
            while os.path.exists(path):
                path = "%s_%d.mjpeg" % (base, k)
                k += 1
            part = path + ".part"
            fout = open(part, "wb")
            cnt = [0]
            total = [0]

            def on_frame(pos, flen):
                fout.write(w.read(pos, flen))
                cnt[0] += 1
                total[0] += flen

            w.seek(off)                                    # ← 同上
            got, gotb, _ = scan_chain(w, off, args.max_frame, on_frame=on_frame,
                                      gap=int(rec.get("gap", args.gap)))
            fout.close()
            written = os.path.getsize(part)
            if written != gotb:
                print("[CARVE] ⚠️ 写出的字节数 %d ≠ 统计 %d（偏移 %d）：磁盘读取可能出错"
                      % (written, gotb, off), file=sys.stderr)
            if got >= args.min_frames:
                os.rename(part, path)
                done += 1
                print("[CARVE] %-58s 帧 %5d 共 %7.1f MB %s"
                      % (os.path.basename(path), got, gotb / 1048576.0, rec.get("label", "")))
            else:
                os.remove(part)
                print("[CARVE] 跳过（只拼出 %d 帧 < min-frames）：偏移 %d" % (got, off))
    finally:
        w.close()
    print("[CARVE] 落盘 %d 段 → %s" % (done, args.out))
    remux_hint(args.out)
    return 0


def uniq_path(out_dir, name):
    """输出文件名重复就加 _2/_3（重复跑不会覆盖已有的恢复结果）。"""
    path = os.path.join(out_dir, name)
    base, ext = os.path.splitext(path)
    k = 2
    while os.path.exists(path):
        path = "%s_%d%s" % (base, k, ext)
        k += 1
    return path


def remux_hint(out_dir):
    print("[CARVE] 想转 mp4（零转码，帧率按 25）："
          "for f in %s/*.mjpeg; do ffmpeg -f mjpeg -r 25 -i \"$f\" -c:v copy \"${f%%.mjpeg}.mp4\"; done"
          % out_dir)


def warn_same_fs(dev, out):
    try:
        a = os.stat(dev).st_dev
        b = os.stat(out).st_dev
    except OSError:
        return
    if a == b:
        print("[CARVE] ⚠️⚠️ 输出目录和被扫设备在**同一个文件系统**上：写出去的数据有概率盖掉"
              "还没读到的候选块。\n"
              "          最保险：把 --out 指到**别的盘**（外接 U 盘 / 另一块 NVMe），"
              "跑完再拷进 RDKX5-YOLOv11n-/raw-data/。\n"
              "          已按你的要求继续；若只想先试水，加 --max-frames 200 扫一小段。", file=sys.stderr)


def parse_expect(items):
    """--expect NAME:FRAMES（可多次）→ [(name, frames)]"""
    out = []
    for it in items or []:
        if ":" not in it:
            continue
        name, n = it.rsplit(":", 1)
        try:
            out.append((name, int(n)))
        except ValueError:
            pass
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", required=True,
                    help="要扫的设备或镜像（如 /dev/nvme0n1p5；也可以是 dd 出来的镜像文件）")
    ap.add_argument("--phase", choices=["list", "dump", "both"], default="list")
    ap.add_argument("--manifest", default="/tmp/carve_mjpeg.jsonl", help="清单文件（阶段①产物）")
    ap.add_argument("--out", default=None, help="阶段②输出目录（默认 RDKX5-YOLOv11n-/raw-data/recover_<日期>）")
    ap.add_argument("--start", type=int, default=0, help="起始偏移（字节）")
    ap.add_argument("--end", type=int, default=0, help="结束偏移（0=到设备末尾）")
    ap.add_argument("--min-frames", type=int, default=30, help="少于此帧数的链不算一段录像（默认 30）")
    ap.add_argument("--max-frames", type=int, default=0, help="多于此帧数的不算（0=不限；可用它排除盘上还在的整段）")
    ap.add_argument("--max-frame", type=int, default=4 << 20, help="单帧字节上限（防噪声，默认 4 MB）")
    ap.add_argument("--gap", type=int, default=0,
                    help="帧之间允许的间隔字节数（0=严格首尾相接，raw .mjpeg 用这个；"
                         "MP4 里的 JPEG 带长度前缀，用 8 试试）")
    ap.add_argument("--chunk", type=int, default=16 << 20, help="读窗口大小")
    ap.add_argument("--progress", type=float, default=5.0, help="进度打印间隔（秒）")
    ap.add_argument("--hold-in-ram", action="store_true",
                    help="dump 阶段先把所有候选读进内存再落盘：输出目录与被扫设备同盘也安全"
                         "（候选总量 GB 级，先看 free -g 够不够）")
    ap.add_argument("--expect", action="append", default=[],
                    help="预期清单项，可多次：--expect auv_20260922_231021.mjpeg:4396")
    a = ap.parse_args(argv)

    if not a.out:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        a.out = os.path.join(root, "RDKX5-YOLOv11n-", "raw-data",
                             "recover_" + time.strftime("%m%d"))
    if a.max_frames and a.max_frames < a.min_frames:
        print("!! --max-frames < --min-frames", file=sys.stderr)
        return 2
    if not os.path.exists(a.dev):
        print("!! 找不到设备/镜像：%s" % a.dev, file=sys.stderr)
        return 2
    print("[CARVE] 设备 %s（%.1f GB）｜区间 [%d, %s]｜min-frames %d｜输出 %s"
          % (a.dev, dev_size(a.dev) / 1e9, a.start,
             a.end or "末尾", a.min_frames, a.out))
    if a.phase in ("list", "both"):
        phase_list(a)
    if a.phase in ("dump", "both"):
        return phase_dump(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
