# -*- coding: utf-8 -*-
"""main.py — 任务主程序（状态机调度；可选本次要执行的任务）
用法：
（详细用法、判据与实测见 doc/注释历史.md）"""
import argparse
import json
import os

import numpy as np
import sys
import threading
import time

import base.cfg.settings as S

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
from base.hw.camera import create_camera
from base.hw.uart import UartController, install_signal_handlers, _D_MIN_DEPTH_M
from common.vision.detector import DetectorHub
from task.ball import BallTask
from gate.gate_task import GateTask
from handling.handling_task import HandlingTask
from handling.percept.grab_detector import build_grab_backend
from gate.percept.gate_detector import build_gate_backend

TASK_CLASS = {"ball": BallTask, "gate": GateTask}
# 夹取/放置是**同一个调度**的两种模式（`handling/`）：名字 → mode
TASK_MODE = {"grab": "grab", "place": "place", "handling": "full"}
# hub 注册/查询用的键 = `HandlingTask.name`（**与三种 CLI 名无关**）
HANDLING_NAME = HandlingTask.name
# CLI 允许的任务名（`--task` 的校验与提示都用它）
TASK_NAMES = tuple(sorted(set(TASK_CLASS) | set(TASK_MODE)))


def build_task(name, uart, hub, w, h):
    """按名字造任务：夹取/放置走同一个 `HandlingTask`，只差 mode。"""
    if name in TASK_MODE:
        return HandlingTask(uart, hub, w, h, mode=TASK_MODE[name])
    return TASK_CLASS[name](uart, hub, w, h)
TASK_CAM = {"ball": "front", "gate": "front",
            "grab": "down", "handling": "down", "place": "front"}
STATE_TASK = {S.STATE_BALL: "ball", S.STATE_GATE: "gate",
              S.STATE_GRAB: "grab", S.STATE_PLACE: "place"}
TASK_STATE = {"ball": S.STATE_BALL, "gate": S.STATE_GATE,
              "grab": S.STATE_GRAB, "place": S.STATE_PLACE,
              "handling": S.STATE_GRAB}      # full：先占夹取段，交接后由任务自己推进

class DownCamFeeder(threading.Thread):
    """后台读下视相机、只存**最新一帧**；主循环绝不阻塞在下视 read 上。"""

    def __init__(self):
        super().__init__(daemon=True)
        self._lock = threading.Lock()
        self._frame = None
        self._want = False
        self._cam = None
        self._stop = False

    def set_want(self, want):
        self._want = bool(want)

    def latest(self):
        with self._lock:
            return self._frame

    def run(self):
        while not self._stop:
            if not self._want:
                if self._cam is not None:
                    try:
                        self._cam.close()
                    except Exception:
                        pass
                    self._cam = None
                with self._lock:
                    self._frame = None
                time.sleep(0.1)
                continue
            if self._cam is None:
                try:
                    self._cam = create_camera("down", stream=False)   # 下视不推流，只做主视的辅助
                except Exception:
                    self._cam = None
                if self._cam is None:
                    time.sleep(0.5)
                    continue
            try:
                f = self._cam.read()
            except Exception:
                f = None
            if f is not None:
                with self._lock:
                    self._frame = f
            time.sleep(0.005)

    def stop(self):
        self._stop = True
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception:
                pass


def psi_line(info, tol_deg=8.0):
    """HUD 的「偏转角」一行（**纯函数**，便于用例）。返回 `(文本, BGR 颜色)`。"""
    psi = info.get("hdg")
    st = info.get("hdg_state") or ""
    it = info.get("hdg_i")
    skip = info.get("hdg_skip")
    if psi is None:
        txt = "PSI  --   (tol %.1f)  %s" % (float(tol_deg), st or "no-full-frame")
        col = (165, 165, 165)
    else:
        psi = float(psi)
        txt = "PSI %+6.1f deg  (tol %.1f)  %s" % (psi, float(tol_deg), st or "-")
        if it:
            txt += "  it=%d" % int(it)
        col = (0, 220, 0) if abs(psi) <= float(tol_deg) + 1e-9 else (0, 190, 255)
    if skip:
        txt += "   | SKIP: %s" % skip
        col = (0, 140, 255)
    return txt, col

class _ThroughLog(object):
    """★ 2026-10-07 用户定：**单独一份 through 日志** —— 只记"冲刺过门"那一段。

    为什么单独一份：任务日志几千帧，冲刺只占其中十几~几十帧；把每一次冲刺**单独成段**
    （`enter` / 逐帧 `frame` / `exit` 各一行），复盘时直接 `grep '"evt":"enter"'` 就能
    把每一次穿门拎出来对比（哪次是 `z≤cross` 触发的、哪次是门口超时兜底的、那次用了多久）。

    开关（都不设 = 不记）：
      · `AUV_THROUGH_LOG=<路径>.jsonl`  显式指定
      · 设了 `AUV_TASK_LOG` 时**自动派生**：`<任务日志去掉后缀>_through.jsonl`
        （即"记任务日志就顺带记 through"，不用额外记参数）
    """

    def __init__(self, path=None):
        self.path = path
        self._fh = None
        self._in = False          # 本帧是否处于 THROUGH
        self._t0 = None
        self._n = 0

    @staticmethod
    def _is_through(info):
        """THROUGH 的两种形态都算：正常冲刺 `through` 与门口兜底 `creep_through`。"""
        return (str(info.get("phase", "")) == "THROUGH"
                or str(info.get("action", "")) in ("through", "creep_through"))

    def write(self, info, now_ms, frame_seq, task_name):
        if self.path is None:
            return
        try:
            cur = self._is_through(info)
            if not cur and not self._in:
                return                                    # 非冲刺段：一个字都不写
            if self._fh is None:
                d = os.path.dirname(self.path)
                if d:
                    os.makedirs(d, exist_ok=True)
                self._fh = open(self.path, "w", encoding="utf-8", buffering=1)
                print("[MAIN] through 日志 -> %s" % self.path)
            evt = "frame"
            if cur and not self._in:
                evt = "enter"
                self._t0 = now_ms
                self._n = 0
            elif self._in and not cur:
                evt = "exit"
            if cur:
                self._n += 1
            rec = {"evt": evt, "t": round(now_ms / 1000.0, 3), "frame": frame_seq,
                   "task": task_name, "frames_in_through": self._n}
            if evt == "exit" and self._t0 is not None:
                rec["dur_ms"] = int(now_ms - self._t0)
                rec["pass"] = info.get("pass")
            for k, v in info.items():
                if isinstance(v, bool) or v is None or isinstance(v, (int, float, str)):
                    rec[k] = v
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._in = cur
        except Exception as e:
            print("[MAIN] through 日志写入失败：%s ⇒ 关闭它，任务继续" % e)
            self.close()

    def close(self):
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None


class _TaskRecorder(object):
    """**任务过程中的录像**（2026-10-06 用户要求）。

    把任务主循环每帧（有叠加就用叠加后的画面）写成 **.mjpeg** —— 零转码，`ffplay`/`ffmpeg` 直接看，
    和 pc 端 `pc_recorder.py` 的产物同格式。

    开关（环境变量，任选一个；都不设 = 不录）：
      · `AUV_RECORD=<路径>.mjpeg`  —— 显式指定文件
      · `AUV_RECORD_DIR=<目录>`    —— 自动命名 `<目录>/auv_<任务>_<时间戳>.mjpeg`
    画质：`AUV_RECORD_Q`（默认 85，0~100）。

    ⚠️ 逐帧同步写盘 ⇒ 会占 I/O。默认画质 85、720p 下约 60~90 KB/帧，**跑一趟 10 分钟 ≈ 数 GB**，
       上板前先确认 eMMC/SD 余量。
    """

    def __init__(self, path=None, quality=85):
        self.path = path
        self.quality = int(quality)
        self._fh = None
        self.frames = 0
        self._warned = False

    def write(self, frame):
        if self.path is None or frame is None:
            return
        try:
            if self._fh is None:
                d = os.path.dirname(self.path)
                if d:
                    os.makedirs(d, exist_ok=True)
                self._fh = open(self.path, "wb")
                print("[REC] 任务录像开始 → %s" % self.path)
            ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if ok:
                self._fh.write(enc.tobytes())
                self.frames += 1
        except Exception as e:                     # 录像绝不许拖垮任务
            if not self._warned:
                self._warned = True
                print("[REC] ⚠️ 录像写盘失败：%s ⇒ 关闭录像，任务继续" % e)
            self.close()

    def close(self):
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                print("[REC] 任务录像结束：%s（%d 帧）" % (self.path, self.frames))
                self._fh = None


# 显式"关"的写法（大小写不敏感）：让外层脚本/wrapper 设了也能这一趟不录
_OFF_WORDS = ("0", "off", "no", "false", "none", "-")


def _is_off(v):
    return str(v or "").strip().lower() in _OFF_WORDS


def _make_recorder(task_name="task"):
    """按环境变量建录像器。

    **默认不启动**（三个变量都不设 ⇒ 空壳，一帧都不写、不建文件）。
    开关（优先级从高到低）：
      · `AUV_RECORD=0|off|no|false|none|-` ⇒ **显式关闭**（即使 `AUV_RECORD_DIR` 也设了）
      · `AUV_RECORD=<路径>.mjpeg`         ⇒ 录到该文件
      · `AUV_RECORD_DIR=<目录>`           ⇒ 录到 `<目录>/auv_<任务>_<时间戳>.mjpeg`
    画质 `AUV_RECORD_Q`（默认 85）。
    """
    q = int(os.environ.get("AUV_RECORD_Q", "85") or 85)
    raw = os.environ.get("AUV_RECORD")
    if _is_off(raw):
        return _TaskRecorder(None, q)          # ★ 显式关闭：不再看 AUV_RECORD_DIR
    path = (raw or "").strip() or None
    if path is None:
        d = os.environ.get("AUV_RECORD_DIR")
        if _is_off(d):
            return _TaskRecorder(None, q)
        d = (d or "").strip()
        if d:
            path = os.path.join(d, "auv_%s_%s.mjpeg"
                                % (task_name, time.strftime("%Y%m%d_%H%M%S")))
    return _TaskRecorder(path, q)


class AppController(object):
    def __init__(self, tasks=None):
        self.uart = UartController()
        self.hub = DetectorHub()
        tasks = list(tasks or S.comm.tasks.enabled)
        for name in tasks:                     # 装配层：任务专用后端（mock/真实）
            if name == "gate":
                backend = build_gate_backend()
                self.hub.register("gate", backend)
                if backend is None:
                    print("[MAIN] ⚠️ gate 后端不可用：权重缺失/路径不对")
                    print("       先跑 python3 preview_detect.py --gate-kpt 自检权重")
            if name in TASK_MODE:
                # ⚠️ hub 的键是**任务实例的 `name`**（`HandlingTask.name = "handling"`，
                #    三种 CLI 名 grab/place/handling 都构造成同一个类）——**不是 CLI 名**。
                #    注册错了会拿到 None ⇒ `detect_all` 静默退回 legacy（= 夹取用撞球那版 YOLO），
                #    正是本项目明确否掉的路（BPU 留给门 keypoint）。`tests/tasks/handling` 钉着这点。
                backend = build_grab_backend()
                self.hub.register(HANDLING_NAME, backend)
                if backend is None:
                    print("[MAIN] ⚠️ 夹取/放置后端未注册（%s.detect.mode = mock）⇒ "
                          "感知会退回 legacy 模型" % "vision.grab")
                    print("       要跑纯 CV 红球检测：把 vision.yaml 的 grab.detect.mode 设回 cv")
        # ⚠️ **只开本次任务真正要用的相机**：USB 相机是**独占 + 吃带宽**的设备，
        #    板端同时开前视+下视两路 720p 时，**后开那路 read() 会返回 None**
        #    （2026-10-06 实测：`--task grab` 开了前视 ⇒ 下视读不到帧 ⇒ `task.process()`
        #     一帧都没被调用、状态机卡在 GRAB 空转、逐帧日志也不生成）。
        #    所以按 `TASK_CAM` 反推"需要哪几路"，不需要的**一个都不开**。
        need_cams = {TASK_CAM[t] for t in tasks}
        self.cams = {}
        for which in ("front", "down"):
            if which in need_cams:
                self.cams[which] = create_camera(which)
        print("[MAIN] 本次开相机：%s（按 TASK_CAM 反推，不用的不开）"
              % (sorted(self.cams) or "无"))
        # 下视是**独占设备**：已经直接开了就不再起 `DownCamFeeder`（否则同设备两个 handle）。
        self._need_down = "down" in self.cams
        self._down_feeder = DownCamFeeder() if ("gate" in tasks and not self._need_down) else None
        if self._down_feeder is not None:
            self._down_feeder.start()
        elif "gate" in tasks and self._need_down:
            print("[MAIN] ⚠️ 下视已被本任务直接占用 ⇒ gate 的 wants_down 注入不可用"
                  "（gate 的 down 默认关闭；要用得错开跑）")
        self.tasks = {}
        for name in tasks:
            cam = self.cams[TASK_CAM[name]]
            self.tasks[name] = build_task(name, self.uart, self.hub,
                                                cam.width, cam.height)
        self.state = S.STATE_IDLE
        self.queue = [TASK_STATE[n] for n in (tasks or S.comm.tasks.enabled)]
        self._state_start = None
        self.frames = 0
        self._frame_seq = 0          # 采集序号（逐帧日志用）
        self._log_t = time.time()
        self._video_on = self._init_video()
        self._rec = _make_recorder()          # ★ 任务过程录像（AUV_RECORD / AUV_RECORD_DIR）
        self._task_log_path = os.environ.get("AUV_TASK_LOG")
        self._task_log_fh = None
        # ★ through 专表：显式给了用它，否则跟着任务日志自动派生（记任务日志就顺带记冲刺）
        _tp = os.environ.get("AUV_THROUGH_LOG")
        if _is_off(_tp):                     # ★ AUV_THROUGH_LOG=0/off ⇒ 不记冲刺专表
            _tp = None
            self._task_log_path = self._task_log_path   # 任务日志照旧
        elif not _tp and self._task_log_path:
            _tp = (self._task_log_path[:-6] + "_through.jsonl"
                   if self._task_log_path.endswith(".jsonl")
                   else self._task_log_path + "_through.jsonl")
        self._through_log = _ThroughLog(_tp)
        # 相机连续无帧计数（超阈值 E-STOP；见任务分支里的注释）
        self._noframe = 0
        self._noframe_estop = int(os.environ.get("AUV_NOFRAME_ESTOP", "150") or 150)

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

    def _feed_down(self, task):
        """gate 需要下视时（coarse/width）才让后台 feeder 开/读下视；不需要就释放。"""
        if not (hasattr(task, "set_down_frame") and hasattr(task, "wants_down")):
            return
        if self._down_feeder is None:
            return
        self._down_feeder.set_want(task.wants_down())
        task.set_down_frame(self._down_feeder.latest())

    def _uart_status(self):
        """画面监控行：下位机深度遥测 + 限深保护（无遥测显示 n/a）。"""
        d = getattr(self.uart, "depth_m", None)
        lim = float(S.get("comm.depth_guard.min_depth_m", _D_MIN_DEPTH_M) or 0.0)
        on = bool(S.get("comm.depth_guard.enable", True))
        return ("depth=%s guard=%s min=%.2fm%s"
                % ("n/a" if d is None else "%.2fm" % d,
                   "on" if on else "off", lim,
                   " [限深保护]" if getattr(self.uart, "guard_active", False)
                   else ""))

    def _draw(self, frame, dets):
        """叠加 识别框/角点/中心线 + 状态信息 后显示。"""
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
        psi_row = None
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
                           "yaw", "hdg", "hdg_i", "hdg_state", "reason")]
            lines.append(" ".join(parts))
            if name == "gate":
                psi_row = psi_line(
                    info, float(S.get("comm.gate.hdg.tol_deg", 8.0) or 8.0))
        lines.append(self._uart_status())
        y = 20
        for ln in lines:
            cv2.putText(img, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        fs * 0.7, (0, 255, 0), 1, cv2.LINE_AA)
            y += int(24 * fs)
        if psi_row is not None:
            txt, col = psi_row
            # 深色描边：水面/倒影上白底文字容易糊
            cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        fs * 0.95, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        fs * 0.95, col, 2, cv2.LINE_AA)
            # 右下角再放一份（画面大时左上容易看漏）
            tw = int(len(txt) * fs * 11)
            cv2.putText(img, txt, (max(10, w - tw - 10), h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, (max(10, w - tw - 10), h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, col, 2, cv2.LINE_AA)
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
                which = TASK_CAM[STATE_TASK[self.state]]
                frame = self.cams[which].read()
                if frame is None:
                    # ⚠️ 相机没帧 ⇒ 本帧**连 `process()` 都不调用** ⇒ 任务自己的超时/丢目标逻辑
                    #    整条走不到，状态机会在这里**永久空转**（2026-10-06 板端真踩：两路 USB 抢带宽）。
                    #    所以这里自己兜底：先节流告警，连续太久就 E-STOP，绝不空转到天亮。
                    self._noframe += 1
                    if self._noframe % 30 == 1:
                        print("[MAIN] ⚠️ 相机 %s 连续 %d 帧没帧（任务本帧没跑）"
                              % (which, self._noframe))
                    if self._noframe >= self._noframe_estop:
                        print("[MAIN] ✗ 相机 %s 连续 %d 帧无帧 ⇒ 停手（E-STOP）"
                              "；查相机占用/线缆/带宽（两路 USB 720p 会互抢）"
                              % (which, self._noframe))
                        self.uart.estop()
                        self.state = S.STATE_ESTOP
                        return self.state
                else:
                    if self._noframe:
                        print("[MAIN] 相机 %s 恢复出帧（此前断了 %d 帧）" % (which, self._noframe))
                    self._noframe = 0
                    self._frame_seq += 1          # 采集序号（只用于日志/叠加）
                    self._feed_down(task)
                    task.process(frame, now_ms)
                    self._log_task_frame(task, now_ms)
                    if self._video_on:
                        _img = self._draw(frame, getattr(task, "last_dets", None) or [])
                    else:
                        _img = None
                    # ★ 任务过程录像（2026-10-06 用户要求）：有叠加就录叠加后的，否则录原帧
                    self._rec.write(_img if _img is not None else frame)
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
        # ★ through 专表**独立于任务日志**：只设了 AUV_THROUGH_LOG 也要能写 ⇒ 放在早退之前
        self._through_log.write(task.last_info, now_ms, self._frame_seq, task.name)
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
            try:
                tel = getattr(getattr(task, "uart", None), "telemetry", None)
                if tel is not None:
                    if tel.yaw_deg is not None:
                        rec["tyaw"] = round(float(tel.yaw_deg), 2)
                    if tel.depth_m is not None:
                        rec["tdep"] = round(float(tel.depth_m), 3)
                    if tel.roll_deg is not None:
                        rec["trol"] = round(float(tel.roll_deg), 2)
                    if tel.pitch_deg is not None:
                        rec["tpit"] = round(float(tel.pitch_deg), 2)
                    age = tel.age_ms(now_ms)
                    rec["ttel"] = None if age is None else int(age)
            except Exception:
                pass
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
            from base.hw.camera import get_stream_pusher
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

        # ★ 任务过程录像收尾（把缓冲落盘、打印帧数）
        try:
            self._rec.close()
        except Exception:
            pass
        # ★ through 专表收尾
        try:
            self._through_log.close()
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
                    help="本次任务: all|ball|gate|grab|place|handling（默认 all=comm.yaml enabled）；grab/place/handling=同一调度三种模式")
    args = ap.parse_args()
    tasks = list(S.comm.tasks.enabled) if args.task == "all" else [args.task]
    for t in tasks:
        if t not in TASK_NAMES:
            print("未知任务: %s（可选 all|%s）" % (t, "|".join(TASK_NAMES)))
            sys.exit(2)
    ctrl = AppController(tasks)
    # 单任务运行（下水前）自检：后端/权重不可用就直接拒绝启动，避免入水后空跑
    bad = ctrl.check_ready(tasks) if len(tasks) == 1 else []
    if bad:
        print("[MAIN] ✗ 任务 %s 不可用，拒绝启动：%s" % (bad[0], ctrl._model_desc()))
        task = ctrl.tasks.get(bad[0])
        if task is not None and hasattr(task, "calibrated") and not task.calibrated:
            seg = "place" if getattr(task, "mode", "") == "place" else "grab"
            print("       原因：comm.%s.calibrated = false —— 占位值没标定完，"
                  "机械闸拒绝下水（标定完再翻 true）" % seg)
        if "gate" in bad:
            print("       检查 cfg/vision.yaml → model.task_models.gate.path 是否存在，"
                  "并先跑 preview_detect.py --gate-kpt 自检")
        ctrl.close()
        sys.exit(3)
    ctrl.run()

if __name__ == "__main__":
    main()
