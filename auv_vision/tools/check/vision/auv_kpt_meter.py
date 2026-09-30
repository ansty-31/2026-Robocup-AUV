# -*- coding: utf-8 -*-
"""tools/check/vision/auv_kpt_meter.py — 距离-响应阶梯实测表（只读相机 + 只跑推理；**不碰串口、不驱动船**）
用途：**换水 / 换光 / 换门之后的第一件事** —— 量出"这个权重在这种环境下、几米内还能出角点"，
用法（板端，**工程根下**，先确认没有别的 preview/main.py 占着相机）：
每 ~4.5s 打印一行：
改判据请用 `tools/check/vision/check_kpt_decode.py`（它连原始 logit 一起报）。"""
import os
import sys
import time

# 工程根 = 向上第一个含 `cfg/` 的目录（**别写死层级**：脚本搬过位置，写死会静默指错）
_ROOT = os.path.dirname(os.path.abspath(__file__))
while _ROOT != os.path.dirname(_ROOT) and not os.path.isdir(os.path.join(_ROOT, "cfg")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
import numpy as np                                              # noqa: E402
import cv2                                                      # noqa: E402

import base.cfg.settings as S                                       # noqa: E402

S.vision.model.score_threshold = 0.01        # 暴露真实响应（见文件头注释）
from base.hw.camera import create_camera                           # noqa: E402
from gate.percept.gate_detector import build_gate_backend               # noqa: E402

DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 75.0
b = build_gate_backend()
if b is None:
    print("✗ 门角点后端不可用（权重缺失/路径不对）→ 先跑 tools/check/pipeline/check_paths.py")
    raise SystemExit(3)
cam = create_camera("front")
lo1, hi1 = np.array([0, 90, 70]), np.array([12, 255, 255])
lo2, hi2 = np.array([168, 90, 70]), np.array([180, 255, 255])
print("  t   | 最高分 | 角点置信度                      | 红块面积(占比) | 红块 bbox | >=0.5 角点数")
t0 = time.time()
while time.time() - t0 < DUR:
    f = None
    while f is None:
        f = cam.read()
        time.sleep(0.02)
    dets = b.detect(f)
    sc = [float(d.score) for d in dets]
    kc, npt = "—", "-"
    if sc:
        d = dets[int(np.argmax(sc))]
        if d.kpt_conf is not None:
            kc = str(np.round(np.asarray(d.kpt_conf), 2).tolist())
            npt = "%d" % int(np.sum(np.asarray(d.kpt_conf) >= 0.5))
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    area, box = 0, "-"
    if n > 1:
        k = 1 + int(np.argmax(st[1:, 4]))
        area = int(st[k, 4])
        if area > 300:
            box = "%dx%d@(%d,%d)" % (st[k, 2], st[k, 3], st[k, 0], st[k, 1])
    print("%5.0fs | %5.3f | %-31s | %7d (%.2f%%) | %-16s | %s"
          % (time.time() - t0, max(sc) if sc else -1, kc[:31],
             area, 100.0 * area / 1280 / 720, box, npt))
    time.sleep(4.5)
