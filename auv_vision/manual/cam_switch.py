# -*- coding: utf-8 -*-
"""manual/cam_switch.py — 手动模式的**主视/下视切换推流**

手动挡下，操作手在 PC 端发 `cam:front` / `cam:down` / `cam`(切换) 到遥控桥，
遥控桥就把推流画面在**前视**与**下视**相机之间切。
用 `base.hw.camera.create_camera("front"|"down")` 开相机（走 cfg/vision.yaml 的 camera.* 配置，
无硬件时 fallback_sim），`manual.stream.MjpegPusher` 零转码 UDP 推流（与 manual.sh 共用同一套）。
"""
from __future__ import annotations

import threading
import time


def parse_camera_packet(data):
    """把 UDP 文本包翻译成相机切换指令。非相机指令返回 None。

    接受：`cam:front` / `cam:f` → "front"；`cam:down` / `cam:d` → "down"；
          `cam` / `cam:toggle` / `cam:t` → "toggle"；其余 `cam:*` → "bad"。
    """
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
    """后台线程：开前视+下视两台相机，把**当前选中**的那台转 JPEG 推给 pusher。

    `switch("front"|"down")` / `toggle()` 只改一个标志，推流线程下一帧就切过去。
    """

    def __init__(self, host, port=5000, pkt=8000, stream_fps=30.0, start="front"):
        self.which = "front" if start not in ("front", "down") else start
        self._host = host
        self._stop = threading.Event()
        self._pusher = None
        self._cams = {}
        # 相机放在线程里懒开：串口已经可能失败，相机再失败就明确报错而不是崩。
        self._cam_host = host
        self._cam_port = int(port)
        self._cam_pkt = int(pkt)
        self._cam_fps = float(stream_fps)
        self._thread = threading.Thread(target=self._run, name="cam-pusher",
                                        daemon=True)
        self._thread.start()

    def switch(self, which):
        if which in ("front", "down"):
            self.which = which
            return True
        return False

    def toggle(self):
        self.which = "down" if self.which == "front" else "front"
        return self.which

    def _run(self):
        if not self._cam_host:
            return                                  # 不推流 ⇒ 不开相机（纯遥控 / 测试）
        import cv2
        try:
            from base.hw.camera import create_camera
            # stream=False：CamPusher 用自己的 MjpegPusher 推选中那一路；别让相机 read 再推全局流
            self._cams = {"front": create_camera("front", stream=False),
                          "down": create_camera("down", stream=False)}
        except Exception as exc:
            print("[CAM] 打开相机失败：%s" % exc)
            return
        if self._cam_host:
            try:
                from manual.stream import MjpegPusher
                self._pusher = MjpegPusher(self._cam_host, self._cam_port,
                                           pkt=self._cam_pkt, stream_fps=self._cam_fps)
            except Exception as exc:
                print("[CAM] 推流初始化失败：%s" % exc)
        while not self._stop.is_set():
            try:
                frame = self._cams[self.which].read()
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
        for c in self._cams.values():
            try:
                c.close()
            except Exception:
                pass
