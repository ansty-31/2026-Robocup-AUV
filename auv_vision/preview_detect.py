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
    ap.add_argument("--fuse", action="store_true",
                    help="叠加显示 kpt_memory 融合后的角点(实心)+原始(空心)，并打印抖动对比"
                         "（强制开启 kpt_mem，与 enable 默认值/AUV_GATE_KPT_MEM 无关）")
    ap.add_argument("--dump", default=None,
                    help="逐帧存 JSONL（角点/置信度/框/时间戳），供离线标定 kpt_mem/q_lo 等")
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
        from gate.gate_detector import build_gate_backend
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
        from gate.gate_frontend import parse_kpt_mode

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

    # --fuse 是"显式要求对比融合" → force=True：无视 enable 默认值与 AUV_GATE_KPT_MEM，
    # 一律构造出来对比（这正是它存在的意义：看开/关差别，不用改配置）
    mem, fuse_stat = None, None
    if a.fuse and a.gate_kpt:
        from gate.kpt_memory import build_kpt_memory
        km = S.vision.gate.kpt_mem
        mem = build_kpt_memory(km, n_kpt=4, force=True)
        if mem is None:
            print("[PV] kpt_mem 构造失败（参数非法），无法对比融合")
        else:
            fuse_stat = {"raw": [], "fused": [], "prev": None}
            print("[PV] 融合对比 ON（--fuse 强制开启，与 enable 默认值无关）："
                  "实心=融合后，空心=原始；每 2s 打印抖动对比")
            print("[PV] kpt_mem: alpha=%s beta=%s k_sigma=%s r_min_px=%s r_max_px=%s"
                  " recall_conf=%s"
                  % (km.get("alpha"), km.get("beta"), km.get("k_sigma"),
                     km.get("r_min_px"), km.get("r_max_px"), km.get("recall_conf")))
    dump_fh = open(a.dump, "w", encoding="utf-8") if a.dump else None
    if dump_fh is not None:
        print("[PV] 逐帧 dump → %s（每行一条 JSON：角点/置信度/框/mode/时间戳）" % a.dump)
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
                if a.gate_kpt and d.kpts is not None and mem is not None:
                  try:
                    # 只对"选中的那个门"(角点最全)做融合，和 GateTask 口径一致
                    best = max(dets, key=lambda x: float(np.sum(
                        np.asarray(x.kpt_conf, float))) if x.kpt_conf is not None else -1)
                    if d is best:
                        fk, fc = mem.update(d.kpts, d.kpt_conf, time.monotonic() * 1000)
                        for i in range(min(len(d.kpts), 4)):
                            rx, ry = int(d.kpts[i][0]), int(d.kpts[i][1])
                            cv2.circle(img, (rx, ry), 4, (160, 160, 160), 1)   # 原始=空心灰
                            fx, fy = int(fk[i][0]), int(fk[i][1])
                            cv2.circle(img, (fx, fy), 4, (60, 255, 60), -1)    # 融合=实心绿
                        if fuse_stat is not None:
                            if fuse_stat["prev"] is not None:
                                pr, pf = fuse_stat["prev"]
                                fuse_stat["raw"].append(
                                    float(np.nanmedian(np.linalg.norm(
                                        np.asarray(d.kpts[:4], float) - pr, axis=1))))
                                fuse_stat["fused"].append(
                                    float(np.nanmedian(np.linalg.norm(
                                        np.asarray(fk[:4], float) - pf, axis=1))))
                                for k in ("raw", "fused"):
                                    if len(fuse_stat[k]) > 60:
                                        fuse_stat[k].pop(0)
                            fuse_stat["prev"] = (np.asarray(d.kpts[:4], float).copy(),
                                                 np.asarray(fk[:4], float).copy())
                  except Exception as _e:
                    print("[PV] 融合异常 -> 关闭 --fuse（识别/显示不受影响）：%s" % _e)
                    mem = None
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
            if dump_fh is not None:
                import json as _json
                rec = {"t": round(time.time() - t0, 3), "frame": n,
                       "n_det": len(dets), "fps": round(fps, 2), "dets": []}
                for d in dets:
                    item = {"kind": d.kind, "score": round(float(d.score), 4),
                            "bbox": [int(d.x), int(d.y), int(d.w), int(d.h)]}
                    if getattr(d, "kpts", None) is not None:
                        item["kpts"] = [[round(float(x), 2), round(float(y), 2)]
                                        for x, y in d.kpts]
                        kc = d.kpt_conf if d.kpt_conf is not None else None
                        item["kpt_conf"] = ([round(float(c), 4) for c in kc]
                                            if kc is not None else None)
                        if a.gate_kpt and parse_kpt_mode is not None:
                            try:
                                _m, _ids = parse_kpt_mode(d.kpts, d.kpt_conf)
                                item["mode"], item["ids"] = str(_m), [int(i) for i in _ids]
                            except Exception:
                                pass
                    rec["dets"].append(item)
                dump_fh.write(_json.dumps(rec, ensure_ascii=False) + "\n")
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
                    if mem is not None and fuse_stat is not None and fuse_stat["raw"]:
                        r = float(np.median(fuse_stat["raw"]))
                        f = float(np.median(fuse_stat["fused"]))
                        red = (1.0 - f / r) * 100.0 if r > 1e-6 else 0.0
                        print("     [融合] 原始抖动 %.1f px → 融合后 %.1f px（降 %.0f%%）"
                              " | 各点权重 %s | 半径 %s"
                              % (r, f, red,
                                 np.round(mem.weights, 2).tolist(),
                                 np.round(mem.radius, 1).tolist()))
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
        if dump_fh is not None:
            dump_fh.close()
            print("[PV] dump 已保存: %s（%d 行，JSONL 逐帧可离线回看）" % (a.dump, n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
