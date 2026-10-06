# -*- coding: utf-8 -*-
"""manual/cam_switch.py — 手动模式的**主视/下视选路推流**
⚠️ **同一时刻只开一路相机**（2026-10-04 实测）：两块 720p 相机挂在同一个 USB 2.0 总线上
（详细用法、判据与实测见 doc/注释历史.md）"""
from __future__ import annotations

import threading
import time


def parse_camera_packet(data):
    """把 UDP 文本包翻译成相机切换指令。非相机指令返回 None。"""
    try:
        t = data.decode("utf-8", errors="ignore").strip().lower().replace(" ", "")
    except Exception:
        return None
    if not t.startswith("cam"):
        return None
    if t in ("cam", "cam:toggle", "cam:t"):
        return "toggle"
    if t in ("cam:front", "cam:f"):
        return "front"
    if t in ("cam:down", "cam:d"):
        return "down"
    return "bad"


class CamPusher(object):
    """后台线程：只开**当前选中**的那一路相机，转 JPEG 推给 pusher。"""

    def __init__(self, host, port=5000, pkt=8000, stream_fps=30.0, start="front"):
        self.which = "front" if start not in ("front", "down") else start
        self._host = host
        self._stop = threading.Event()
        self._pusher = None
        self._cap = None                 # 当前唯一打开的那路相机
        self._lock = threading.Lock()    # 保护 _cap / which（切换线程 vs 推流线程）
        # 相机放在线程里懒开：串口已经可能失败，相机再失败就明确报错而不是崩。
        self._cam_host = host
        self._cam_port = int(port)
        self._cam_pkt = int(pkt)
        self._cam_fps = float(stream_fps)
        self._thread = threading.Thread(target=self._run, name="cam-pusher",
                                        daemon=True)
        self._thread.start()

    # ---------------------------------------------------------------- 选路
    def _open_locked(self, which):
        """（**调用方需持 self._lock**）关掉当前相机，开指定那一路。"""
        if self._cap is not None:
            try:
                self._cap.close()
            except Exception:
                pass
            self._cap = None
        from base.hw.camera import create_camera
        self._cap = create_camera(which, stream=False)
        self.which = which

    def switch(self, which):
        """切到指定那一路（先关旧再开新）。返回是否成功。"""
        if which not in ("front", "down"):
            return False
        with self._lock:
            if which == self.which and self._cap is not None:
                return True
            try:
                self._open_locked(which)
            except Exception as exc:
                print("[CAM] 切换相机失败：%s" % exc)
                return False
        print("[CAM] 已切到 %s" % which)
        return True

    def set_host(self, host, port=None):
        """改推流目标地址（不重开相机）。板端 UDP 遥控桥收到 PC 包后调它，自动跟上 PC 的 IP。"""
        if not host:
            return False
        self._cam_host = str(host)
        if port:
            self._cam_port = int(port)
        if self._pusher is not None:
            try:
                self._pusher.set_host(self._cam_host, self._cam_port)
            except Exception:
                pass
        return True

    def toggle(self):
        self.switch("down" if self.which == "front" else "front")
        return self.which

    # ---------------------------------------------------------------- 推流
    def _run(self):
        if not self._cam_host:
            return                                  # 不推流 ⇒ 不开相机（纯遥控 / 测试）
        import cv2
        try:
            with self._lock:
                self._open_locked(self.which)
        except Exception as exc:
            print("[CAM] 打开相机失败：%s" % exc)
            return
        print("[CAM] 推流相机 = %s（同一时刻只开这一路，避免 USB 带宽超限）" % self.which)
        try:
            from manual.stream import MjpegPusher
            self._pusher = MjpegPusher(self._cam_host, self._cam_port,
                                       pkt=self._cam_pkt, stream_fps=self._cam_fps)
        except Exception as exc:
            print("[CAM] 推流初始化失败：%s" % exc)
        while not self._stop.is_set():
            with self._lock:
                cap = self._cap
            if cap is None:
                time.sleep(0.05)
                continue
            try:
                frame = cap.read()
            except Exception:
                time.sleep(0.05)
                continue
            if frame is not None and self._pusher is not None:
                ok, enc = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    self._pusher.offer(enc)
            time.sleep(0.005)

    def close(self):
        self._stop.set()
        if self._pusher is not None:
            try:
                self._pusher.close()
            except Exception:
                pass
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.close()
                except Exception:
                    pass
                self._cap = None
