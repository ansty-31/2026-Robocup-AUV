# -*- coding: utf-8 -*-
"""preview_detect.py — 真机实时识别预览（**只画识别框，不发任何运动指令**）

在板子上实测某个 .bin 权重（默认 cfg vision.model.path，即 auv_multi.bin）
对指定类别（默认 gate）的识别效果：
  - 本脚本 **不创建 UartController**，不打开串口 → 船不会动；
  - 每帧做一次推理，把识别框画在实时画面上。

显示方式（可任选/组合）：
  --show            本机窗口 cv2.imshow（需要 DISPLAY；板子桌面终端可用）
  --stream          把“带框画面”编码成 MJPEG，用 UDP 推给 PC 观看
                    PC 端： ffplay -fflags nobuffer -flags low_delay -framedrop -f mjpeg "udp://@:5000"
  --save DIR        每 N 帧存一张带框图到 DIR（默认每 30 帧）

用法示例（板端，工程根目录）：
  python3 preview_detect.py --classes gate                       # 只打印统计(无显示)
  python3 preview_detect.py --classes gate --stream              # 推给 PC 看
  DISPLAY=:0 python3 preview_detect.py --classes gate --show     # 板子桌面窗口
  python3 preview_detect.py --classes all --save /tmp/pv --duration 20
  python3 preview_detect.py --classes gate --model models/auv_multi.bin --conf 0.4

门角点(关键点)模式 —— 用 gate 任务模型 gate_kpt_bayese_640x640_nv12.bin，
画 4 个角点 + 四边形，并打印每点坐标/置信度与四角组合判定：
  python3 preview_detect.py --gate-kpt --stream                  # 门角点预览
  DISPLAY=:0 python3 preview_detect.py --gate-kpt --show         # 板子桌面窗口
  python3 preview_detect.py --gate-kpt --conf 0.3 --duration 20  # 降低角点置信度门槛

按 Ctrl-C 或窗口里按 q/Esc 退出；结束打印各类别累计帧数。
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                   # noqa: E402
import base.settings as S                            # noqa: E402
from base.camera import create_camera                # noqa: E402
from common.detector import DetectorHub              # noqa: E402

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

COLORS = {"red_ball": (60, 60, 255), "blue_ball": (255, 150, 30),
          "gate": (60, 255, 60), "unknown": (200, 200, 200)}

# 门角点显示：名字取 cfg 的 kpt_order，画点门槛 KPT_VIS_THR
_GATE_CFG = S.get("vision.model.task_models.gate", None) or {}
KPT_NAMES = tuple(_GATE_CFG.get("kpt_order") or ("TL", "TR", "BR", "BL"))
KPT_VIS_THR = 0.30


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="front", choices=["front", "down"])
    ap.add_argument("--classes", default="gate",
                    help="要显示的类别(逗号分隔)；all=全部")
    ap.add_argument("--model", default=None, help="覆盖模型路径(默认 cfg 里的)")
    ap.add_argument("--gate-kpt", action="store_true",
                    help="门角点模式: 用 vision.model.task_models.gate 的 keypoint 权重，"
                         "画 4 角点+四边形并打印坐标/置信度")
    ap.add_argument("--conf", type=float, default=None, help="覆盖 score_threshold")
    ap.add_argument("--show", action="store_true", help="本机窗口显示")
    ap.add_argument("--stream", action="store_true", help="UDP 推流带框画面给 PC")
    ap.add_argument("--host", default=os.environ.get("AUV_STREAM_HOST", "192.168.137.2"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("AUV_STREAM_PORT", "5000")))
    ap.add_argument("--stream-fps", type=float, default=20.0)
    ap.add_argument("--save", default=None, help="保存带框图目录")
    ap.add_argument("--save-every", type=int, default=30)
    ap.add_argument("--duration", type=float, default=0.0, help="运行秒数; 0=一直跑")
    a = ap.parse_args()

    if not HAS_CV2:
        print("需要 opencv-python(cv2)")
        return 2
    if a.model:
        S.vision.model.path = a.model
    if a.conf is not None:
        S.vision.model.score_threshold = float(a.conf)
    want = None if a.classes.strip().lower() == "all" else \
        {c.strip() for c in a.classes.split(",") if c.strip()}

    # 本脚本自己画框/推流，关掉 camera.py 的“原始 MJPEG 推流钩子”避免重复
    os.environ["AUV_STREAM"] = "0"

    backend = None
    model_desc = S.vision.model.path
    if a.gate_kpt:
        try:
            from gate.gate_detector import build_gate_backend
        except ImportError:
            from auv_vision.gate.gate_detector import build_gate_backend
        backend = build_gate_backend()
        if backend is None:
            print("[PV] 门角点后端不可用(权重缺失)；检查 vision.model.task_models.gate.path")
            return 3
        want = None
        gc = S.get("vision.model.task_models.gate", None) or {}
        model_desc = "%s (%s)" % (gc.get("path", "?"), gc.get("kind", "keypoint"))

    print("=" * 64)
    print(" 真机识别预览(无运动) | 模型: %s" % model_desc)
    if a.gate_kpt:
        print(" 模式: 门角点(keypoint) | conf=%.2f | 相机: %s | 显示: show=%s stream=%s save=%s"
              % (S.vision.model.score_threshold, a.camera,
                 a.show, a.stream, a.save or "-"))
    else:
        print(" 类别: %s | conf=%.2f | 相机: %s | 显示: show=%s stream=%s save=%s"
              % ("ALL" if want is None else sorted(want),
                 S.vision.model.score_threshold, a.camera,
                 a.show, a.stream, a.save or "-"))
    print("=" * 64)

    cam = create_camera(a.camera)
    hub = DetectorHub()
    parse_kpt_mode = None
    if backend is not None:
        hub.register("gate", backend)         # 复用门任务后端(独立权重)
        try:
            from gate.gate_frontend import parse_kpt_mode
        except ImportError:
            from auv_vision.gate.gate_frontend import parse_kpt_mode

    pusher = None
    if a.stream:
        try:
            from manual import stream as stream_mod
        except Exception:
            import stream as stream_mod
        pusher = stream_mod.MjpegPusher(a.host, a.port, pkt=8000,
                                        stream_fps=a.stream_fps)
        print("[PV] 推流 → udp://%s:%d ；PC: ffplay -fflags nobuffer -flags "
              "low_delay -framedrop -f mjpeg \"udp://@:%d\""
              % (a.host, a.port, a.port))
    if a.save:
        os.makedirs(a.save, exist_ok=True)
    window = a.show
    if window:
        try:
            cv2.namedWindow("preview")
        except Exception as e:
            print("[PV] 窗口不可用(%s)，改用 --stream/--save" % e)
            window = False

    t0 = time.time()
    n = 0
    counts = {}
    last_log = t0
    try:
        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue
            if a.gate_kpt:
                dets = hub.detect_list("gate", frame)     # 门角点专用权重
            else:
                dets = hub.detect_all(frame)             # 一次前向，全类别
                if want is not None:
                    dets = [d for d in dets if d.kind in want]
            for d in dets:
                counts[d.kind] = counts.get(d.kind, 0) + 1

            img = frame.copy()
            for d in dets:
                col = COLORS.get(d.kind, COLORS["unknown"])
                if d.w > 0 and d.h > 0:
                    cv2.rectangle(img, (d.x, d.y), (d.x + d.w, d.y + d.h), col, 2)
                    cv2.putText(img, "%s %.2f" % (d.kind, d.score),
                                (d.x, max(14, d.y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, col, 2)
                if a.gate_kpt and d.kpts is not None:
                    quad = []
                    for i in range(min(len(d.kpts), len(KPT_NAMES))):
                        c = float(d.kpt_conf[i]) if d.kpt_conf is not None else 1.0
                        if c <= 0.0:
                            continue
                        px, py = int(round(float(d.kpts[i][0]))), \
                            int(round(float(d.kpts[i][1])))
                        pcol = (60, 255, 60) if c >= KPT_VIS_THR else (150, 150, 150)
                        cv2.circle(img, (px, py), 5, pcol, -1)
                        cv2.circle(img, (px, py), 7, (0, 0, 0), 1)
                        cv2.putText(img, "%s %.2f" % (KPT_NAMES[i], c),
                                    (px + 8, py - 6), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.45, pcol, 1)
                        quad.append((px, py))
                    if len(quad) >= 3:
                        cv2.polylines(img, [np.array(quad, dtype=np.int32).reshape(-1, 1, 2)],
                                      len(quad) == 4, (60, 255, 60), 2)
            fps = n / max(time.time() - t0, 1e-6)
            cv2.putText(img, "%s det=%d %.1ffps" % (S.vision.model.mode,
                                                    len(dets), fps),
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            n += 1
            if pusher is not None:
                ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    pusher.offer(enc)
            if a.save and (n % max(1, a.save_every) == 0):
                cv2.imwrite(os.path.join(a.save, "pv_%06d.jpg" % n), img)
            if window:
                cv2.imshow("preview", img)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord('q')):
                    break

            if time.time() - last_log >= 2.0:
                print("[PV] %.1f fps | 本帧 det=%d | 累计 %s"
                      % (fps, len(dets), counts))
                if a.gate_kpt:
                    if not dets:
                        print("     (本帧未检出门)")
                    for k, d in enumerate(dets[:3]):
                        if d.kpts is None:
                            print("     [%d] 无角点输出 (score=%.2f)" % (k, d.score))
                            continue
                        kc = d.kpt_conf if d.kpt_conf is not None else \
                            np.ones(len(d.kpts), np.float32)
                        n_ok = int(np.sum(np.asarray(kc) >= KPT_VIS_THR))
                        print("     [%d] score=%.2f 有效角点(>=%.2f)=%d/%d"
                              % (k, d.score, KPT_VIS_THR, n_ok, len(d.kpts)))
                        print("         kpts: %s"
                              % " ".join("%s(%.0f,%.0f)" % (KPT_NAMES[i], d.kpts[i][0],
                                                            d.kpts[i][1])
                                         for i in range(min(len(d.kpts), len(KPT_NAMES)))))
                        print("         conf: %s"
                              % " ".join("%s=%.2f" % (KPT_NAMES[i], kc[i])
                                         for i in range(min(len(kc), len(KPT_NAMES)))))
                        if parse_kpt_mode is not None:
                            try:
                                mode, ids = parse_kpt_mode(d.kpts, d.kpt_conf)
                                print("         判定: mode=%s ids=%s" % (mode, list(ids)))
                            except Exception as e:
                                print("         判定失败: %s" % e)
                last_log = time.time()
            if a.duration and (time.time() - t0) >= a.duration:
                break
    except KeyboardInterrupt:
        print("\n[PV] 手动中断")
    finally:
        if pusher is not None:
            pusher.close()
        if window:
            cv2.destroyAllWindows()
        print("[PV] 结束 | 共 %d 帧 | 类别累计帧数: %s" % (n, counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
