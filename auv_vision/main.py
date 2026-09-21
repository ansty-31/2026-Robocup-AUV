# -*- coding: utf-8 -*-
"""main.py — 任务主程序（状态机调度；可选本次要执行的任务）

状态流：IDLE → [选定任务依次] → DONE；任意时刻 Ctrl-C/SIGTERM → 急停。

任务分工（按目录分区）：
  - 任务一 撞球          task1_2/ball.py（BallTask）—— 前视相机
  - 任务三 过门          gate/（keypoint 四角 + PnP，相位机见 gate/gate_task.py）

用法：
    python3 main.py --task all          # 按 comm.yaml tasks.enabled 顺序执行
    python3 main.py --task gate         # 本次只执行过门（ball 同理）
"""
import argparse
import json
import os

import numpy as np
import sys
import time

import base.settings as S

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
from base.camera import create_camera
from base.uart import UartController, install_signal_handlers, _D_MIN_DEPTH_M
from common.detector import DetectorHub
from task1_2.ball import BallTask
from gate.gate_task import GateTask
from gate.gate_detector import build_gate_backend

TASK_CLASS = {"ball": BallTask, "gate": GateTask}
TASK_CAM = {"ball": "front", "gate": "front"}
STATE_TASK = {S.STATE_BALL: "ball", S.STATE_GATE: "gate"}
TASK_STATE = {"ball": S.STATE_BALL, "gate": S.STATE_GATE}


class AppController(object):
    def __init__(self, tasks=None):
        self.uart = UartController()
        self.hub = DetectorHub()
        tasks = list(tasks or S.comm.tasks.enabled)
        for name in tasks:                     # 装配层：gate 专用后端（mock/真实）
            if name == "gate":
                backend = build_gate_backend()
                self.hub.register("gate", backend)
                if backend is None:
                    print("[MAIN] ⚠️ gate 后端不可用：权重缺失/路径不对")
                    print("       先跑 python3 preview_detect.py --gate-kpt 自检权重")
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
        self._frame_seq = 0          # 采集序号（逐帧日志用）
        self._log_t = time.time()
        self._video_on = self._init_video()
        # 任务逐帧日志（AUV_TASK_LOG=<path>）：把 last_info 每帧存一行 JSON，
        # 供 tools/analyze/analyze_task_log.py 离线判读（phase/action/z/dx/dy/kpt/ratio/pass…）；
        # 默认不开，不影响运行。
        self._task_log_path = os.environ.get("AUV_TASK_LOG")
        self._task_log_fh = None

    def check_ready(self, tasks):
        """返回不可用任务名列表（下水前自检：避免设备已入水却空跑）。"""
        bad = []
        for name in tasks:
            task = self.tasks.get(name)
            if task is None or not task.ready:
                bad.append(name)
        return bad

    # ------------------------------------------------------------------ 画面监测
    _KIND_COLOR = {"red_ball": (60, 60, 255), "blue_ball": (255, 150, 30),
                   "gate": (60, 255, 60), "unknown": (200, 200, 200)}

    def _init_video(self):
        if os.environ.get("AUV_SHOW", "1") == "0":
            return False
        if not HAS_CV2:
            return False
        disp = os.environ.get("DISPLAY")
        if not disp and not os.environ.get("AUV_FORCE_SHOW"):
            print("[MAIN] 无 DISPLAY，跳过画面窗口（需要时 export DISPLAY=:0 或 "
                  "AUV_FORCE_SHOW=1）")
            return False
        # 板端 cv2 是 Qt 构建：X 不可用时 namedWindow **直接 abort 进程**（不是异常，
        # try/except 拦不住）→ 先自己确认本地 X socket 存在，别让浮窗把整船任务搞崩
        if disp and not os.environ.get("AUV_FORCE_SHOW"):
            n = disp.split(":")[-1].split(".")[0]
            if not os.path.exists("/tmp/.X11-unix/X%s" % n):
                print("[MAIN] DISPLAY=%s 无对应 X socket，跳过画面窗口"
                      "（需要强制时 AUV_FORCE_SHOW=1）" % disp)
                return False
        try:
            cv2.namedWindow("AUV")
            return True
        except Exception as e:
            print("[MAIN] 画面窗口不可用：%s" % e)
            return False

    def _uart_status(self):
        """画面监控行：下位机深度遥测 + 限深保护（无遥测显示 n/a）。

        深度来自下位机 14B 遥测帧（base/telemetry.py）；`guard` 用 comm.depth_guard：
        开启且当前深度 ≤ min_depth_m 时禁止上浮（base/uart.py::_apply_depth_guard）。
        """
        d = getattr(self.uart, "depth_m", None)
        # 兜底用 base/uart.py 里那个**与 cfg 同值**的常量（别再写 0.3：那会让画面
        # 显示的限深阈值与真正生效的保护阈值不一致，现场会被自己的叠加骗）
        lim = float(S.get("comm.depth_guard.min_depth_m", _D_MIN_DEPTH_M) or 0.0)
        on = bool(S.get("comm.depth_guard.enable", True))
        return ("depth=%s guard=%s min=%.2fm%s"
                % ("n/a" if d is None else "%.2fm" % d,
                   "on" if on else "off", lim,
                   " [限深保护]" if getattr(self.uart, "guard_active", False)
                   else ""))

    def _draw(self, frame, dets):
        """叠加 识别框/角点/中心线 + 状态信息 后显示。

        dets 用**当前任务本帧已算出的检测结果**（见各 task.last_dets），
        不再调 hub.detect_all：否则 gate 任务会额外跑一遍 ball 权重，帧率腰斩。
        """
        img = frame.copy()
        h, w = img.shape[:2]
        fs = max(0.45, w / 900.0)
        cv2.line(img, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
        cv2.line(img, (0, h // 2), (w, h // 2), (255, 255, 255), 1)
        kpt_r = max(3, int(w / 360.0))
        for d in dets:
            color = self._KIND_COLOR.get(d.kind, self._KIND_COLOR["unknown"])
            if d.w > 0 and d.h > 0:
                cv2.rectangle(img, (d.x, d.y), (d.x + d.w, d.y + d.h),
                              color, 2)
                lab = "%s %.2f" % (d.kind, d.score)
                ty = d.y - 8 if d.y > 24 else d.y + d.h + 20
                cv2.putText(img, lab, (d.x, ty), cv2.FONT_HERSHEY_SIMPLEX,
                            fs * 0.7, color, 1, cv2.LINE_AA)
            kpts = getattr(d, "kpts", None)
            if kpts is None:
                continue
            kc = getattr(d, "kpt_conf", None)
            for i in range(len(kpts)):
                c = float(kc[i]) if kc is not None else 1.0
                if c <= 0.0:
                    continue
                px, py = int(kpts[i][0]), int(kpts[i][1])
                pc = color if c >= 0.5 else (150, 150, 150)
                cv2.circle(img, (px, py), kpt_r, pc, -1)
                cv2.putText(img, "%d" % i, (px + kpt_r + 2, py - kpt_r),
                            cv2.FONT_HERSHEY_SIMPLEX, fs * 0.5, pc, 1,
                            cv2.LINE_AA)
        lines = ["state=%s task=%s frame=%d"
                 % (self.state, list(self.tasks), self.frames)]
        if self.state in STATE_TASK:
            name = STATE_TASK[self.state]
            info = self.tasks[name].last_info
            # 只显示"当前任务真的会产出"的键（两个任务的并集，见各自 last_info）
            parts = ["%s=%s" % (k, v) for k, v in info.items()
                     if isinstance(v, (int, float, str)) and
                     k in ("action", "ratio", "growth", "dx", "dy", "sway",
                           "heave", "surge", "phase", "substate", "mode", "z",
                           "pass", "kpt", "kpt_raw",
                           # yaw 恒 0 是**正常**（居中只用 sway）；它只由正航向 ALIGN.HDG 产生
                           # → hdg/hdg_i 一起显示便于现场核对
                           "yaw", "hdg", "hdg_i", "reason")]
            lines.append(" ".join(parts))
        lines.append(self._uart_status())
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
                    self._frame_seq += 1          # 采集序号（只用于日志/叠加）
                    task.process(frame, now_ms)
                    self._log_task_frame(task, now_ms)
                    if self._video_on:
                        self._draw(frame, getattr(task, "last_dets", None) or [])
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

    def _log_task_frame(self, task, now_ms):
        """把本帧 last_info（phase/action/z/dx/dy/kpt/ratio/pass/…）写一行 JSON。"""
        if not self._task_log_path:
            return
        try:
            if self._task_log_fh is None:
                self._task_log_fh = open(self._task_log_path, "w",
                                         encoding="utf-8", buffering=1)
                print("[MAIN] 任务日志 -> %s" % self._task_log_path)
            info = task.last_info
            rec = {"t": round(now_ms / 1000.0, 3), "frame": self._frame_seq,
                   "state": self.state, "task": task.name}
            for k, v in info.items():
                if isinstance(v, bool) or v is None or isinstance(v, (int, float, str)):
                    rec[k] = v
            self._task_log_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            print("[MAIN] 任务日志写入失败：%s" % e)
            self._task_log_path = None

    def _advance(self, now_ms, reason):
        nxt = self.queue[0] if self.queue else S.STATE_DONE
        print("[MAIN] %s -> %s (%s)" % (self.state, nxt, reason))
        if self.queue:
            self.state = self.queue.pop(0)
        else:
            self.state = S.STATE_DONE
        self._state_start = now_ms

    def close(self):
        """收尾：关闭推流 + 相机 + 串口 + 日志文件（残留占用会让下次启动像"锁死"）。"""
        if getattr(self, "_task_log_fh", None) is not None:
            try:
                self._task_log_fh.close()
            except Exception:
                pass
            self._task_log_fh = None
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

    def _model_desc(self):
        """本次运行实际会加载的权重（一眼确认没走错模型）。"""
        parts = []
        if "ball" in self.tasks:
            parts.append("ball=%s" % os.path.basename(str(S.vision.model.path)))
        if "gate" in self.tasks:
            gc = S.get("vision.model.task_models.gate", None) or {}
            parts.append("gate=%s" % os.path.basename(str(gc.get("path", "?"))))
        return " ".join(parts) or "-"

    def run(self):
        install_signal_handlers(self.uart)
        print("===== %s | 任务=%s | camera front=%s | detector=%s ====="
              % (S.PROJECT_NAME, [k for k in self.tasks],
                 S.vision.camera.front.type, S.vision.model.mode))
        print("===== 权重: %s =====" % self._model_desc())
        period = 1.0 / max(S.vision.camera.front.fps, 5)
        try:
            while self.state != S.STATE_DONE:
                now = int(time.time() * 1000)
                self.step(now)
                if self.state == S.STATE_ESTOP:
                    print("[MAIN] E-STOP，退出")
                    break
                time.sleep(period)
            # 任务全部结束：再持续发 stop 保持 done_hold_ms，确保"稳定保持停止"后才退出
            # （撞球命中后不会刚停就关串口；DASH 后的 STOP 相位已在任务内保持）
            hold_ms = int(S.get("comm.tasks.done_hold_ms", 0) or 0)
            if self.state == S.STATE_DONE and hold_ms > 0 \
                    and not self.uart.estop_active:
                print("[MAIN] DONE：保持停止 %.0f ms" % hold_ms)
                t_end = time.time() + hold_ms / 1000.0
                while time.time() < t_end:
                    self.uart.set_motion("stop")
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
                    help="本次任务: all|ball|gate（默认 all=comm.yaml enabled）")
    args = ap.parse_args()
    tasks = list(S.comm.tasks.enabled) if args.task == "all" else [args.task]
    for t in tasks:
        if t not in TASK_CLASS:
            print("未知任务: %s（可选 all|ball|gate）" % t)
            sys.exit(2)
    ctrl = AppController(tasks)
    # 单任务运行（下水前）自检：后端/权重不可用就直接拒绝启动，避免入水后空跑
    bad = ctrl.check_ready(tasks) if len(tasks) == 1 else []
    if bad:
        print("[MAIN] ✗ 任务 %s 的后端不可用，拒绝启动：%s" % (bad[0], ctrl._model_desc()))
        if "gate" in bad:
            print("       检查 cfg/vision.yaml → model.task_models.gate.path 是否存在，"
                  "并先跑 preview_detect.py --gate-kpt 自检")
        ctrl.close()
        sys.exit(3)
    ctrl.run()


if __name__ == "__main__":
    main()
