# -*- coding: utf-8 -*-
"""camera.py — 相机：前视 / 下视分离，各自可 sim 或真机，真机失败可软回退 sim

配置：cfg/vision.yaml camera.front / camera.down（各自 type/device/宽高/fps/标定）。
调用：create_camera("front" | "down")；返回对象仅实现 read()。

**视频回放**（离线复现/回归；默认行为不变，只在设了环境变量时生效）：
    AUV_SIM_MODE=1 AUV_CAM_VIDEO=/path/clip.mp4 [AUV_CAM_VIDEO_FPS=30] \
        AUV_TASK_LOG=log/replay.jsonl python3 main.py --task gate
  · `AUV_CAM_VIDEO`：把前视相机源换成该视频文件（cfg 一个字段都不动）；
  · **必须**同时 `AUV_SIM_MODE=1`（只打印、不发串口），否则回放会真的驱动船；
  · 主循环没有帧率节流 ⇒ 回放按视频原生 fps（或 AUV_CAM_VIDEO_FPS）自己节拍，
    保证时间基与真机一致（HDG 的 PID/超时都按真实时间算）。
"""
import os
import time

import numpy as np

import base.settings as S

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class Camera(object):
    def __init__(self, width, height, fps):
        self.width, self.height, self.fps = width, height, fps

    def read(self):
        raise NotImplementedError

    def close(self):
        """释放相机资源（子类按需覆写）。"""
        pass


# ---------------------------------------------------------------------------
# 视频文件回放（离线复现：把一段录制视频当成前视相机）
# ---------------------------------------------------------------------------
class VideoFileCamera(Camera):
    """把视频文件当相机用（见模块头「视频回放」）。**只用于离线复现/回归**。

    要点：
      · 打开视频**不指定后端**（`cv2.VideoCapture(path)`）⇒ 文件与设备都能开；
      · 主循环没有节流 ⇒ 这里按 `play_fps` 自己睡，避免"以解码速度飞快跑完"导致时间基被压缩；
      · 读完返回 None（只打印一次），主循环空转直到任务自身超时/过门结束。
    """

    def __init__(self, cfg, path, fps=None):
        if not HAS_CV2:
            raise RuntimeError("需要 opencv-python(cv2) 才能读视频")
        super().__init__(cfg.width, cfg.height, cfg.fps)
        self.path = str(path)
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError("无法打开视频 %s" % self.path)
        try:
            native = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        except Exception:
            native = 0.0
        want = float(fps or 0.0)
        self.play_fps = want if want > 0 else (native if 5.0 <= native <= 120.0 else float(self.fps))
        self.frames = 0            # 已消费到的帧号（含被丢掉的）
        self.dropped = 0           # 因消费者慢而跳过的帧数（真机会自然丢帧，这里显式做）
        self._t0 = None
        self._eof_logged = False
        print("[CAM] 视频回放：%s（原生 %.1f fps ⇒ 按 %.1f fps 节拍）"
              % (self.path, native, self.play_fps))

    def read(self):
        """**Real-time 语义**（与真机相机一致）：消费者慢 ⇒ 丢帧给你最新的一帧；消费者快 ⇒ 等到该帧的时刻。

        为什么必须这样：主循环是"读一帧→处理一帧"，真机上处理慢就自然丢帧（时间基仍是真实时间）。
        若顺序回放不丢帧，19.8s 的片子会被拉成几十秒 ⇒ HDG 的 PID/超时/循环全变味，复现就失真了。
        """
        if self._t0 is None:
            self._t0 = time.monotonic()
        elapsed = time.monotonic() - self._t0
        want = int(elapsed * self.play_fps)          # 此刻"直播"应该播到第几帧
        while self.frames < want:                    # 落后 ⇒ 丢帧（grab 不解码，便宜）
            if not self._cap.grab():
                break
            self.frames += 1
            self.dropped += 1
        due = self._t0 + (self.frames + 1) / max(1e-6, self.play_fps)
        wait = due - time.monotonic()
        if wait > 0:                                 # 超前 ⇒ 等到该帧的时刻
            time.sleep(min(wait, 1.0))
        ok, buf = self._cap.read()
        if not ok:
            if not self._eof_logged:
                print("[CAM] 视频回放结束：读出 %d 帧、跳过 %d 帧" % (self.frames, self.dropped))
                self._eof_logged = True
            time.sleep(0.05)
            return None
        self.frames += 1
        if buf.shape[1] != self.width or buf.shape[0] != self.height:
            buf = cv2.resize(buf, (int(self.width), int(self.height)))
        return buf

    def close(self):
        try:
            self._cap.release()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 虚拟相机（前视/下视各自仿真；下视带红色标示线供 back 真实颜色逻辑测试）
# ---------------------------------------------------------------------------
class SimCamera(Camera):
    def __init__(self, which):
        cfg = S.vision.camera[which]
        super().__init__(cfg.width, cfg.height, cfg.fps)
        self.which = which
        self._rng = np.random.default_rng(20260904)
        self._frame = 0

    def read(self):
        h, w = self.height, self.width
        base = self._rng.integers(0, S.vision.sim.noise + 1,
                                  (h, w, 3), dtype=np.uint8)
        if self.which == "down":
            line = S.vision.sim.down.red_line
            if line.enable:
                lh = max(1, line.width // 2)
                y0 = h - lh
                band_w = max(8, int(w * 0.6))
                off = max(-1.0, min(1.0, float(line.dx_offset)))
                cx = int(w * (0.5 + 0.5 * off))
                x0 = max(0, min(w - band_w, cx - band_w // 2))
                base[y0:, x0:x0 + band_w, 2] = 220
                base[y0:, x0:x0 + band_w, 0] = 20
                base[y0:, x0:x0 + band_w, 1] = 20
        self._frame += 1
        pusher = get_stream_pusher()          # 仿真也能推（无硬件时验证全链路）
        if pusher is not None:
            ok, enc = cv2.imencode(".jpg", base, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                pusher.offer(enc)
        return base


# ---------------------------------------------------------------------------
# 推流（可选）：把相机原始 MJPEG 帧零转码转发给水面 PC
#   开关：cfg/vision.yaml 顶层 stream.*（手动模式由根目录 manual.sh 用环境变量覆盖）
#   实现：manual/stream.py
#   关键点：推流不另外开相机（UVC 只允许一个进程取流），而是"谁在用相机谁顺带推"，
#   因此识别（main.py）与录像（recorder.py）各自跑时都能同时推流，互不冲突。
#   环境变量覆盖：AUV_STREAM=1/0、AUV_STREAM_HOST、AUV_STREAM_PORT、
#                AUV_STREAM_FPS、AUV_STREAM_PKT
# ---------------------------------------------------------------------------
_stream_pusher = None


def get_stream_pusher():
    """惰性创建推流器；未开启或缺少依赖时返回 None（不影响主流程）。"""
    global _stream_pusher
    if _stream_pusher is not None:
        return _stream_pusher
    try:
        cfg = S.vision.stream
    except AttributeError:
        cfg = {}
    env = os.environ.get
    enable = bool(getattr(cfg, "enable", False))
    if env("AUV_STREAM") == "1":
        enable = True
    elif env("AUV_STREAM") == "0":
        enable = False
    if not enable:
        return None
    host = env("AUV_STREAM_HOST") or getattr(cfg, "host", None)
    if not host:
        print("[CAM] 推流已开启但没配 host，跳过")
        return None
    port = int(env("AUV_STREAM_PORT") or getattr(cfg, "port", 5000))
    pkt = int(env("AUV_STREAM_PKT") or getattr(cfg, "pkt", 8000))
    fps = float(env("AUV_STREAM_FPS") or getattr(cfg, "stream_fps", 30))
    try:
        from manual import stream as stream_mod        # 手动模式三件套之一（推流库）
    except ImportError:
        try:
            import stream as stream_mod                # 兼容：stream.py 与 camera.py 同目录
        except ImportError:
            print("[CAM] 找不到 manual/stream.py，跳过推流")
            return None
    try:
        _stream_pusher = stream_mod.MjpegPusher(host, port, pkt=pkt, stream_fps=fps)
        print("[CAM] 推流已开启 → %s:%d（原始 MJPEG 零转码，上限 %.0f fps）" % (host, port, fps))
    except Exception as e:
        print("[CAM] 推流开启失败（不影响识别/录像）：%s" % e)
        _stream_pusher = None
    return _stream_pusher


# ---------------------------------------------------------------------------
# 真机后端
# ---------------------------------------------------------------------------
class UsbCamera(Camera):
    def __init__(self, cfg):
        if not HAS_CV2:
            raise RuntimeError("需要 opencv-python(cv2) 打开 USB 相机")
        super().__init__(cfg.width, cfg.height, cfg.fps)
        self.dev = cfg.device
        self._cap = cv2.VideoCapture(self.dev, getattr(cv2, "CAP_V4L2", 0))
        if not self._cap.isOpened():
            raise RuntimeError("无法打开相机 %s" % self.dev)
        # OpenCV 的 V4L2 后端默认协商 YUYV：本相机 720p YUYV 只有 **9 fps**，
        # MJPG 有 60 fps。所以显式选 MJPG，并用 CONVERT_RGB=0 直接拿原始 JPEG
        # （退流零转码；识别侧再 cv2.imdecode 成 BGR，代价与让 OpenCV 内部解码相同）。
        self._raw = False
        try:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        try:
            self._cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            ok, buf = self._cap.read()
            if ok and buf is not None and bytes(buf.reshape(-1)[:3]) == b"\xff\xd8\xff":
                self._raw = True
        except Exception:
            self._raw = False
        if not self._raw:
            self._cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        # 以驱动实际协商结果为准（本相机 720p MJPG 只有 60fps 档，写 30 也按 60 出）；
        # self.fps 保持配置值不变，避免影响 recorder.py 的落盘帧率语义。
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.negotiated_fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 0.0

    def read(self):
        ok, buf = self._cap.read()
        if not ok or buf is None:
            return None
        pusher = get_stream_pusher()
        if self._raw:
            if pusher is not None:
                pusher.offer(buf)                        # 原始 JPEG：零转码（CPU≈0）
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)   # 识别/录像用 BGR
        if pusher is not None:                           # 相机不支持原始 JPEG：回退重编码
            ok2, enc = cv2.imencode(".jpg", buf, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok2:
                pusher.offer(enc)
        return buf

    def close(self):
        """释放 V4L2 采集句柄（否则进程异常退出后相机可能被占用）。"""
        cap = getattr(self, "_cap", None)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            self._cap = None


class MipiCamera(Camera):
    """下视 IMX415（MIPI/CSI）占位：板端接入后实现（可先经 V4L2 映射）。"""

    def __init__(self, cfg):
        super().__init__(cfg.width, cfg.height, cfg.fps)
        raise NotImplementedError(
            "下视 MIPI 相机尚未接入：确认 CSI 驱动节点后实现本类")


# ---------------------------------------------------------------------------
# 工厂：front/down 各自按 yaml type 实例化；真机失败按 fallback_sim 软回退
# ---------------------------------------------------------------------------
def create_camera(which, fallback_sim=None):
    cfg = S.vision.camera[which]
    typ = cfg.type
    fallback = S.vision.camera.fallback_sim if fallback_sim is None \
        else fallback_sim
    vid = os.environ.get("AUV_CAM_VIDEO")
    if vid and which == "front":
        # 回放：显式环境变量优先于 cfg（cfg 不动；默认不设 = 行为完全不变）
        return VideoFileCamera(cfg, vid, fps=os.environ.get("AUV_CAM_VIDEO_FPS"))
    try:
        if typ == "sim":
            return SimCamera(which)
        if typ == "usb":
            return UsbCamera(cfg)
        if typ == "mipi":
            return MipiCamera(cfg)
        raise RuntimeError("未知相机类型 %s.%s" % (which, typ))
    except Exception as e:
        if fallback:
            print("[CAM] %s 相机(%s)不可用：%s；软回退 sim" % (which, typ, e))
            dev = str(getattr(cfg, "device", "") or "")
            hold = _camera_holders(dev)
            if hold:
                print("[CAM] !! %s 正被占用：%s" % (dev or "相机", hold))
                print("[CAM]    相机是独占设备：手动模式(manual/udp_server.py + manual.stream)"
                      " 与预览/任务不能同时开。")
                print("[CAM]    先停手动模式（在它的终端 Ctrl-C，或 kill 上面这些 PID），"
                      "再跑预览/任务；测完要用手动模式时重新 bash manual.sh。")
            elif dev:
                print("[CAM]    %s 无进程占用 -> 检查线缆/USB 供电，或换 device 索引"
                      "（cfg/vision.yaml -> camera.front.device）" % dev)
            return SimCamera(which)
        raise

def _camera_holders(dev):
    """谁占着这个视频设备（相机是独占的，手动推流会抢走它）。

    返回 ["PID(命令)", ...]；`dev` 支持 "0" 与 "/dev/video0" 两种写法；没装 fuser 时返回 []。
    """
    if not dev:
        return []
    path = dev if str(dev).startswith("/dev/") else "/dev/video%s" % dev
    try:
        import subprocess
        r = subprocess.run(["fuser", path], capture_output=True, text=True, timeout=2)
        pids = [p for p in r.stdout.split() if p.isdigit()]
    except Exception:
        return []
    out = []
    for pid in pids:
        try:
            with open("/proc/%s/cmdline" % pid, "rb") as fh:
                cmd = fh.read().replace(b"\x00", b" ").decode(errors="replace").strip()
        except Exception:
            cmd = "?"
        out.append("%s(%s)" % (pid, cmd[:60]))
    return out
