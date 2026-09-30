# -*- coding: utf-8 -*-
"""telemetry.py — 下位机**遥测上行**帧（14B：0xAA55 + 5×int16 + 校验和）

下位机(STM32)按下面的格式持续回传**深度**等状态；`base/hw/uart.py` 在每次发帧时顺带
读空串口接收缓冲，把最新深度交给"不得浮出水面"的限深保护（`comm.depth_guard`）。

帧格式（与下位机固件、`manual/udp_server.py` 的旧解析保持一致，**小端**）：

    byte0..1   : 0xAA 0x55            帧头
    byte2      : 保留（不参与语义，但计入校验和）
    byte3..4   : depth_cm   int16     当前深度（cm，正=水面以下）→ /100 = m
    byte5..6   : target_cm  int16     下位机目标深度（cm）→ /100 = m
    byte7..8   : roll_cd    int16     横滚（0.01°）→ /100 = °
    byte9..10  : pitch_cd   int16     俯仰（0.01°）→ /100 = °
    byte11..12 : yaw_cd     int16     航向（0.01°）→ /100 = °
    byte13     : checksum = sum(byte0..12) & 0xFF

用法：
    rx = TelemetryReceiver()
    rx.feed(ser.read(ser.in_waiting))      # 原始字节（可任意切分）
    rx.depth_m                             # 最新深度(m)；None = 还没收到过
    rx.fresh(stale_ms=500)                 # 数据是否新鲜（限深保护据此决定是否生效）
    build_telemetry_frame(0.42)            # 造帧（测试/台架模拟下位机用）
"""
import json
import struct
import time

TEL_LEN = 14                     # 整帧长度
TEL_HEADER = b"\xAA\x55"         # 帧头
TEL_PAYLOAD_OFF = 3              # 5×int16 起始偏移（byte2 保留）
TEL_FMT = "<hhhhh"               # 小端 5×int16：depth, target, roll, pitch, yaw
TEL_SCALE = 0.01                 # 原始值 → 物理量（cm→m / 0.01°→°）
TEL_CHECK_OFF = TEL_LEN - 1      # 校验字节位置（=13）


def _now_ms():
    return int(time.time() * 1000)


def build_telemetry_frame(depth_m=0.0, target_m=0.0, roll_deg=0.0,
                          pitch_deg=0.0, yaw_deg=0.0):
    """按协议打包一帧遥测（测试、台架模拟下位机、上位机自检用）。"""
    payload = struct.pack(TEL_FMT,
                          int(round(depth_m / TEL_SCALE)),
                          int(round(target_m / TEL_SCALE)),
                          int(round(roll_deg / TEL_SCALE)),
                          int(round(pitch_deg / TEL_SCALE)),
                          int(round(yaw_deg / TEL_SCALE)))
    body = bytearray(TEL_HEADER) + b"\x00" + payload     # 2+1+10 = 13B
    body.append(sum(body) & 0xFF)
    return bytes(body)


def check_telemetry(frame):
    """帧头 + 校验和是否合法（不合法即丢弃，绝不拿脏数据去限深）。"""
    return (len(frame) >= TEL_LEN and bytes(frame[:2]) == TEL_HEADER
            and (sum(frame[:TEL_CHECK_OFF]) & 0xFF) == frame[TEL_CHECK_OFF])


def decode_telemetry(frame):
    """14B 整帧 → (depth_m, target_m, roll_deg, pitch_deg, yaw_deg)。"""
    d, t, r, p, y = struct.unpack_from(TEL_FMT, frame, TEL_PAYLOAD_OFF)
    return (d * TEL_SCALE, t * TEL_SCALE, r * TEL_SCALE,
            p * TEL_SCALE, y * TEL_SCALE)


def parse_telemetry_frames(buf, stats=None):
    """从字节缓冲 buf 解析出所有完整遥测帧（**就地消费** buf）。

    返回 [(depth_m, target_m, roll_deg, pitch_deg, yaw_deg), ...]。
    stats（可选 dict）：累计统计 `ok`（解析成功帧数）/ `bad`（丢弃的坏帧头/校验错次数）。

    同步策略：逐字节搜 0xAA55；校验错只丢 1 字节继续搜（不整段丢），
    错位/半帧后能自己重新对齐；末尾可能是半个帧头的 1 字节(0xAA)会留下等下一批。
    """
    out = []
    while True:
        pos = buf.find(TEL_HEADER)
        if pos < 0:
            # 没有完整帧头：只留末尾那半个帧头(0xAA)，其余垃圾丢干净（缓冲不会无限涨）
            keep = 1 if buf[-1:] == TEL_HEADER[:1] else 0
            if len(buf) > keep:
                del buf[:len(buf) - keep]
            break
        if pos > 0:
            del buf[:pos]
        if len(buf) < TEL_LEN:
            break
        if not check_telemetry(buf):
            del buf[0]                          # 校验错：丢 1 字节继续对齐
            if stats is not None:
                stats["bad"] = stats.get("bad", 0) + 1
            continue
        out.append(decode_telemetry(bytes(buf[:TEL_LEN])))
        del buf[:TEL_LEN]
    if stats is not None and out:
        stats["ok"] = stats.get("ok", 0) + len(out)
    return out


class TelPlayback(object):
    """把一段**录制的遥测**（task jsonl 里的 `tdep/tyaw/trol/tpit`）按**真实时间**回放。
    用途（离线复现闭环）：视觉来自视频（`AUV_CAM_VIDEO=<clip>`），航向/深度来自**当时的真实遥测**
    启用：`AUV_SIM_TEL_JSONL=<当时的 task.jsonl>` + `AUV_SIM_MODE=1`。
    时间轴：以 jsonl 第一条的 `t` 为 0 点，按真实经过时间线性插值；超出末尾就夹在末值。"""

    def __init__(self, path):
        self.path = str(path)
        self.rows = []
        with open(self.path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                if o.get("t") is None:
                    continue
                self.rows.append((float(o["t"]), o.get("tdep"), o.get("tyaw"),
                                  o.get("trol"), o.get("tpit")))
        self.rows.sort(key=lambda r: r[0])
        self._t0 = None
        print("[TEL] 遥测回放：%s（%d 条，跨度 %.1fs）"
              % (self.path, len(self.rows),
                 (self.rows[-1][0] - self.rows[0][0]) if self.rows else 0.0))

    def pump(self, tel, now_ms=None):
        """按真实时间把当前时刻的遥测值写进 `tel`（无数据时返回 0，不改动）。"""
        if not self.rows:
            return 0
        now_s = (now_ms if now_ms is not None else _now_ms()) / 1000.0
        if self._t0 is None:
            self._t0 = now_s
        tt = self.rows[0][0] + (now_s - self._t0)
        lo = self.rows[0]
        hi = self.rows[-1]
        for i, r in enumerate(self.rows):
            if r[0] <= tt:
                lo = r
                hi = self.rows[i + 1] if i + 1 < len(self.rows) else r
            else:
                break

        def lerp(idx):
            a, b = lo[idx], hi[idx]
            if a is None:
                return b
            if b is None:
                return a
            span = (hi[0] - lo[0])
            k = 0.0 if span <= 1e-9 else max(0.0, min(1.0, (tt - lo[0]) / span))
            return a + (b - a) * k

        tel.depth_m = lerp(1)
        tel.yaw_deg = lerp(2)
        tel.roll_deg = lerp(3)
        tel.pitch_deg = lerp(4)
        tel.last_ms = _now_ms() if now_ms is None else now_ms
        return 1


class TelemetryReceiver(object):
    """串口字节流 → 最新一帧遥测（深度等）+ 新鲜度判定。

    只在 `UartController` 的发送路径里被调用（单线程），无需加锁。
    """

    def __init__(self):
        self.buf = bytearray()
        self.frames = 0            # 累计有效帧数
        self.bad = 0               # 帧头/校验异常丢弃次数
        self.depth_m = None        # 最新深度(m)；None = 还没收到过有效帧
        self.target_m = None
        self.roll_deg = None
        self.pitch_deg = None
        self.yaw_deg = None
        self.last_ms = 0           # 最后一帧的接收时刻(ms)

    def feed(self, data, now_ms=None):
        """喂入原始字节 → 解析并更新最新值；返回本次解析出的有效帧数。"""
        if data:
            self.buf.extend(data)
        stats = {"ok": 0, "bad": 0}
        rows = parse_telemetry_frames(self.buf, stats)
        # 姿态符号归一（`comm.telemetry.yaw_sign`，默认 +1 = 不改行为）。
        #   "物理转向→遥测符号"之积 g·s；g·s=-1 时闭环**仍然稳定**（都收敛在遥测上），
        #   反而不平行）。归一后探向应报 `imag_sign=+1`、`[HDG] 转向后 |psi| 应变小`。
        try:
            import base.cfg.settings as _S
            sign = float(_S.get("comm.telemetry.yaw_sign", 1.0) or 1.0)
        except Exception:
            sign = 1.0
        for depth, target, roll, pitch, yaw in rows:
            self.depth_m = depth
            self.target_m = target
            self.roll_deg = roll
            self.pitch_deg = pitch
            self.yaw_deg = None if yaw is None else sign * yaw
            self.last_ms = now_ms if now_ms is not None else _now_ms()
        self.frames += stats["ok"]
        self.bad += stats["bad"]
        return len(rows)

    def age_ms(self, now_ms=None):
        """距最后一帧多久(ms)；从未收到过返回 None。"""
        if not self.last_ms:
            return None
        return max(0, (now_ms if now_ms is not None else _now_ms()) - self.last_ms)

    def fresh(self, stale_ms, now_ms=None):
        """深度数据是否可用：收到过且不超过 stale_ms（stale_ms<=0 = 不判超时）。"""
        if self.depth_m is None:
            return False
        if stale_ms and stale_ms > 0 and self.age_ms(now_ms) > stale_ms:
            return False
        return True

    def reset(self):
        """丢缓冲与历史值（重开串口/重新入水时用）。"""
        del self.buf[:]
        self.frames = self.bad = 0
        self.depth_m = self.target_m = None
        self.roll_deg = self.pitch_deg = self.yaw_deg = None
        self.last_ms = 0
