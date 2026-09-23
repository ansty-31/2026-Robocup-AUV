# -*- coding: utf-8 -*-
"""tools/check/auv_kpt_meter.py — 距离-响应阶梯实测表（只读相机 + 只跑推理；**不碰串口、不驱动船**）

用途：**换水 / 换光 / 换门之后的第一件事** —— 量出"这个权重在这种环境下、几米内还能出角点"。
2026-09-22 就是因为跳过这一步，在新（更亮更清）水质里直接跑任务，看到的只是"门不识别"，
分不清是"模型弱响应（能救）"还是"完全没响应（要重训）"。

用法（板端，**工程根下**，先确认没有别的 preview/main.py 占着相机）：
    cd ~/Desktop/AUV_New && python3 tools/check/auv_kpt_meter.py 75      # 秒数，默认 75

每 ~4.5s 打印一行：
    t | 模型最高分（**门槛压到 0.01，故意暴露真实响应**） | 最高分那框的 4 角点置信度
      | 画面里最大红色连通块（门框）面积占比与 bbox | >=0.5 角点数

判据：要能解 PnP 需要 >=3 个角点 conf >= `vision.gate.keypoint.conf_thr`（**该阈值以 cfg 为准**，
      板端 2026-09-22 现为 0.6）。
⚠️ 这里把 `model.score_threshold` 压到 0.01 **只为暴露响应**，不代表能当阈值用；
   改判据请用 `tools/check/check_kpt_decode.py`（它连原始 logit 一起报）。

> 2026-09-22 从 `/tmp/auv_kpt_meter.py` 归档进仓库：之前只存在 /tmp，重启或清理就没了，
> 而 runbook §B1 一直在引用它。
"""
import sys
import time

sys.path.insert(0, ".")
import numpy as np                                              # noqa: E402
import cv2                                                      # noqa: E402

import base.settings as S                                       # noqa: E402

S.vision.model.score_threshold = 0.01        # 暴露真实响应（见文件头注释）
from base.camera import create_camera                           # noqa: E402
from gate.gate_detector import build_gate_backend               # noqa: E402

DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 75.0
b = build_gate_backend()
if b is None:
    print("✗ 门角点后端不可用（权重缺失/路径不对）→ 先跑 tools/check/check_paths.py")
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
