# -*- coding: utf-8 -*-
"""main.py — 任务主程序（状态机调度；可选本次要执行的任务）

状态流：IDLE → [选定任务依次] → DONE；任意时刻 Ctrl-C/SIGTERM → 急停。

任务分工（按目录分区）：
  - 任务一 撞球          task1_2/ball.py（BallTask）—— 前视相机
  - 记忆返回（撞球后）   task1_2/return_by_memory.py + run_ball_return.sh 编排
                         （不依赖视觉：待机/下潜/前进 → 撞球记轨迹 → 反向回放 → 回退）
  - 任务三 过门          gate/（keypoint 四角 + PnP，相位机见 gate/gate_task.py）

用法：
    python3 main.py --task all          # 按 comm.yaml tasks.enabled 顺序执行
    python3 main.py --task ball         # 本次只执行撞球（gate 同理）
"""
import argparse
import os
import sys
import time

import base.settings as S

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
from base.camera import create_camera
from base.uart import UartController, install_signal_handlers
from common.detector import DetectorHub
from task1_2.ball import BallTask
from task1_2.ball_forward import BallForwardTask
from gate.gate_task import GateTask
from gate.gate_detector import build_gate_backend

TASK_CLASS = {"ball": BallTask, "ball_fwd": BallForwardTask, "gate": GateTask}
TASK_CAM = {"ball": "front", "ball_fwd": "front", "gate": "front"}
STATE_TASK = {S.STATE_BALL: "ball", S.STATE_BALL_FWD: "ball_fwd",
              S.STATE_GATE: "gate"}
TASK_STATE = {"ball": S.STATE_BALL, "ball_fwd": S.STATE_BALL_FWD,
              "gate": S.STATE_GATE}


class AppController(object):
    def __init__(self, tasks=None):
        self.uart = UartController()
        self.hub = DetectorHub()
        tasks = list(tasks or S.comm.tasks.enabled)
        for name in tasks:                     # 装配层：gate 专用后端（mock/真实）
            if name == "gate":
                self.hub.register("gate", build_gate_backend())
        self.cams = {"front": create_camera("front")}
        self.tasks = {}
        for name in tasks:
            cam = self.cams[TASK_CAM[name]]
            self.tasks[name] = TASK_CLASS[name](self.uart, self.hub,
                                                cam.width, cam.height)
        self.state = S.STATE_IDLE
        self.queue = [TASK_STATE[n] for n in (tasks or S.comm.tasks.enabled)]
        self._state_start = None
        self.frames = 0
        self._log_t = time.time()
        self._video_on = self._init_video()

    # ------------------------------------------------------------------ 画面监测
    _KIND_COLOR = {"red_ball": (60, 60, 255), "blue_ball": (255, 150, 30),
                   "gate": (60, 255, 60), "unknown": (200, 200, 200)}

    def _init_video(self):
        if os.environ.get("AUV_SHOW", "1") == "0":
            return False
        if not HAS_CV2:
            return False
        if not os.environ.get("DISPLAY") and not os.environ.get("AUV_FORCE_SHOW"):
            print("[MAIN] 无 DISPLAY，跳过画面窗口（需要时 export DISPLAY=:0 或 "
                  "AUV_FORCE_SHOW=1）")
            return False
        try:
            cv2.namedWindow("AUV")
            return True
        except Exception as e:
            print("[MAIN] 画面窗口不可用：%s" % e)
            return False

    def _draw(self, frame, dets):
        """叠加 识别框/类别/中心线 + 状态信息 后显示。"""
        img = frame.copy()
        h, w = img.shape[:2]
        fs = max(0.45, w / 900.0)
        cv2.line(img, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
        cv2.line(img, (0, h // 2), (w, h // 2), (255, 255, 255), 1)
        for d in dets:
            color = self._KIND_COLOR.get(d.kind, self._KIND_COLOR["unknown"])
            cv2.rectangle(img, (d.x, d.y), (d.x + d.w, d.y + d.h),
                          color, 2)
            lab = "%s %.2f" % (d.kind, d.score)
            ty = d.y - 8 if d.y > 24 else d.y + d.h + 20
            cv2.putText(img, lab, (d.x, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        fs * 0.7, color, 1, cv2.LINE_AA)
        lines = ["state=%s task=%s frame=%d"
                 % (self.state, list(self.tasks), self.frames)]
        if self.state in STATE_TASK:
            name = STATE_TASK[self.state]
            info = self.tasks[name].last_info
            parts = ["%s=%s" % (k, v) for k, v in info.items()
                     if isinstance(v, (int, float, str)) and
                     k in ("action", "ratio", "growth", "dx", "dy", "sway",
                           "heave", "surge", "phase", "substate", "mode", "z",
                           "area", "pass", "kpt", "reason")]
            lines.append(" ".join(parts))
        y = 20
        for ln in lines:
            cv2.putText(img, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        fs * 0.7, (0, 255, 0), 1, cv2.LINE_AA)
            y += int(24 * fs)
        try:
            cv2.imshow("AUV", img)
            cv2.waitKey(1)
        except Exception as e:
            if S.DEBUG:
                print("[MAIN] 显示异常：%s" % e)

    # ------------------------------------------------------------------ 主循环
    def step(self, now_ms):
        """单步推进（真实/虚拟时间均可；供 run 与 tests 复用）。"""
        self.frames += 1
        if self._state_start is None:
            self._state_start = now_ms

        if self.state == S.STATE_ESTOP or self.uart.estop_active:
            self.state = S.STATE_ESTOP
            self.uart.set_motion("stop")
            return self.state

        if self.state == S.STATE_IDLE:
            self.uart.set_motion("stop")
            if now_ms - self._state_start >= S.comm.tasks.idle_ms:
                self._advance(now_ms, "idle_done")
        elif self.state in STATE_TASK:
            task = self.tasks.get(STATE_TASK[self.state])
            if task is None or not task.ready:
                if not task and S.comm.tasks.fault_action == "exit":
                    self.uart.estop()
                    self.state = S.STATE_ESTOP
                    return self.state
                self._advance(now_ms, "skip(%s)" % self.state)
            else:
                frame = self.cams[TASK_CAM[STATE_TASK[self.state]]].read()
                if frame is not None:
                    task.process(frame, now_ms)
                    if self._video_on:
                        self._draw(frame, self.hub.detect_all(frame))
                    if task.last_info["status"] == S.STATUS_DONE:
                        self._advance(now_ms, "%s_done(%s)"
                                      % (self.state, task.last_info["reason"]))
        elif self.state == S.STATE_DONE:
            self.uart.set_motion("stop")

        if self.frames % 60 == 0 and S.LOG_FPS and S.DEBUG:
            fps = 60.0 / max(time.time() - self._log_t, 1e-6)
            self._log_t = time.time()
            print("[MAIN] fps=%.1f state=%s frame=%d" % (fps, self.state,
                                                         self.frames))
        return self.state

    def _advance(self, now_ms, reason):
        nxt = self.queue[0] if self.queue else S.STATE_DONE
        print("[MAIN] %s -> %s (%s)" % (self.state, nxt, reason))
        if self.queue:
            self.state = self.queue.pop(0)
        else:
            self.state = S.STATE_DONE
        self._state_start = now_ms

    def close(self):
        """收尾：关闭推流 + 相机 + 串口，避免残留占用（相机/串口被占会导致下次像“锁死”）。"""
        try:
            from base.camera import get_stream_pusher
            pusher = get_stream_pusher()
            if pusher is not None:
                pusher.close()
        except Exception:
            pass
        for cam in getattr(self, "cams", {}).values():
            fn = getattr(cam, "close", None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
        try:
            self.uart.close()
        except Exception:
            pass

    def run(self):
        install_signal_handlers(self.uart)
        print("===== %s | 任务=%s | camera front=%s | detector=%s ====="
              % (S.PROJECT_NAME, [k for k in self.tasks],
                 S.vision.camera.front.type, S.vision.model.mode))
        period = 1.0 / max(S.vision.camera.front.fps, 5)
        try:
            while self.state != S.STATE_DONE:
                now = int(time.time() * 1000)
                self.step(now)
                if self.state == S.STATE_ESTOP:
                    print("[MAIN] E-STOP，退出")
                    break
                time.sleep(period)
        except KeyboardInterrupt:
            print("[MAIN] 手动中断")
        finally:
            if not self.uart.estop_active:
                self.uart.neutral()
            self.close()
            print("[MAIN] 已停止")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="all",
                    help="本次任务: all|ball|ball_fwd|gate（默认 all=comm.yaml enabled）")
    args = ap.parse_args()
    tasks = list(S.comm.tasks.enabled) if args.task == "all" else [args.task]
    for t in tasks:
        if t not in TASK_CLASS:
            print("未知任务: %s（可选 all|ball|ball_fwd|gate）" % t)
            sys.exit(2)
    AppController(tasks).run()


if __name__ == "__main__":
    main()
