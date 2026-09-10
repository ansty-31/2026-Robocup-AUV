# -*- coding: utf-8 -*-
"""recorder.py — AUV 相机录像脚本（调试用，不参与镜像封装）

采集指定相机（sim/usb/mipi 占位）并保存视频：
  后端 1（推荐，需 cv2）：mp4/avi（mp4v），带可选时间戳叠加
  后端 2（无 cv2 时）：逐帧 .npy 落盘 + manifest.json，之后用
       python3 recorder.py --assemble <帧目录> [-o out.mp4]  补出视频

示例（在工程根目录执行；手动模式一般直接用 ./manual.sh --record，本文件默认不参与）：
  python3 manual/recorder.py --camera front --seconds 10 --out rec.mp4
  python3 manual/recorder.py --camera front --frames 60 --out rec/rec.mp4 --overlay
  python3 manual/recorder.py --assemble frames_dir --out rec.mp4
"""
import os as _os, sys as _sys
if __package__ in (None, ""):        # 支持直接 python3 manual/xxx.py 运行
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import os
import sys
import time

import numpy as np

import base.settings as S
from base.camera import create_camera

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def open_camera(which, dev=None):
    try:
        if dev is not None and which == "front":
            S.vision.camera.front.device = dev
        return create_camera(which)      # sim/usb/mipi 由 vision.yaml camera.* 决定
    except Exception as e:
        print("[REC] %s 相机打开失败：%s" % (which, e))
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# cv2 后端
# ---------------------------------------------------------------------------
class CvWriter(object):
    def __init__(self, path, fps, size):
        self._path = path
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._w = cv2.VideoWriter(path, fourcc, fps, size)
        if not self._w.isOpened():
            raise RuntimeError("无法创建视频文件: %s（检查目录是否存在/可写；父目录会自动创建，"
                               "但只读介质或权限不足仍会失败）" % path)

    def write(self, frame_bgr, t_text):
        if t_text and HAS_CV2:
            cv2.putText(frame_bgr, t_text, (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        self._w.write(frame_bgr)

    def finish(self, count):
        self._w.release()
        print("[REC] 已写入 %d 帧 → %s" % (count, self._path))


# ---------------------------------------------------------------------------
# 无 cv2：逐帧 npy 后端
# ---------------------------------------------------------------------------
class NpyBackend(object):
    def __init__(self, out_dir, fps, size):
        os.makedirs(out_dir, exist_ok=True)
        self._dir = out_dir
        self._fps = fps
        self._count = 0
        self._meta = {"w": size[0], "h": size[1], "fps": fps,
                      "backend": "npy", "frames": 0}

    def write(self, frame_bgr, t_text):
        np.save(os.path.join(self._dir, "frame_%06d.npy" % self._count),
                frame_bgr)
        self._count += 1

    def finish(self, count):
        self._meta["frames"] = count
        with open(os.path.join(self._dir, "manifest.json"), "w") as f:
            json.dump(self._meta, f, indent=2)
        print("[REC] 已落盘 %d 帧 → %s（无 cv2；可稍后 --assemble 补视频）"
              % (count, self._dir))


def assemble_npy(frames_dir, out_path):
    """把 npy 帧目录合成视频（需 cv2）。"""
    if not HAS_CV2:
        print("[REC] --assemble 需要 opencv（pip install opencv-python）")
        return 1
    manifest = {}
    mpath = os.path.join(frames_dir, "manifest.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            manifest = json.load(f)
    names = sorted(n for n in os.listdir(frames_dir) if n.endswith(".npy"))
    if not names:
        print("[REC] 帧目录为空: %s" % frames_dir)
        return 1
    first = np.load(os.path.join(frames_dir, names[0]))
    h, w = first.shape[:2]
    fps = manifest.get("fps", 20)
    wr = CvWriter(out_path, fps, (w, h))
    for n in names:
        wr.write(np.load(os.path.join(frames_dir, n)), None)
    wr.finish(len(names))
    return 0


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def record(args):
    cam = open_camera(args.camera, args.dev)
    fps = args.fps if args.fps else cam.fps   # 相机对象自带 fps（front/down 各自配置）
    size = (cam.width, cam.height)
    total = args.frames or int(args.seconds * fps)

    if os.path.dirname(os.path.abspath(args.out)):
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)   # 目录不存在就建
    video_like = args.out.lower().endswith((".mp4", ".avi", ".mkv"))
    if video_like and HAS_CV2:
        backend = CvWriter(args.out, fps, size)
    else:
        if video_like:
            print("[REC] 未安装 cv2，改为逐帧落盘到 %s.frames/" % args.out)
            args.out += ".frames"
        backend = NpyBackend(args.out, fps, size)

    print("[REC] camera=%s %dx%d@%d fps, %d 帧 | 后端=%s"
          % (args.camera, size[0], size[1], fps, total, type(backend).__name__))
    t0 = time.time()
    n = 0
    try:
        while n < total:
            frame = cam.read()
            if frame is None:
                time.sleep(1.0 / max(fps, 1))
                continue
            t_text = ""
            if args.overlay:
                t_text = "%s F%05d" % (time.strftime("%H:%M:%S"), n)
            backend.write(frame, t_text)
            n += 1
            if n % 30 == 0:
                el = time.time() - t0
                print("[REC] %d/%d 帧，%d 秒" % (n, total, int(el)))
    except KeyboardInterrupt:
        print("[REC] 手动停止")
    finally:
        backend.finish(n)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="front", choices=["front", "down"],
                    help="录制哪路相机（前视/下视；类型由 cfg/vision.yaml camera.* 决定）")
    ap.add_argument("--dev", default=None, help="覆盖 front 相机设备号/路径")
    ap.add_argument("--out", default="record_out",
                    help="输出：.mp4/.avi 或帧目录（默认 record_out）")
    ap.add_argument("--frames", type=int, default=0, help="录制帧数")
    ap.add_argument("--seconds", type=float, default=0, help="录制秒数")
    ap.add_argument("--fps", type=int, default=0, help="视频 fps")
    ap.add_argument("--overlay", action="store_true", help="叠加时间戳")
    ap.add_argument("--assemble", metavar="FRAMES_DIR",
                    help="把 npy 帧目录合成视频（-o 指定输出）")
    args = ap.parse_args()
    if args.assemble:
        out = args.out if args.out != "record_out" else "record_out.mp4"
        if not out.endswith((".mp4", ".avi")):
            out += ".mp4"
        return assemble_npy(args.assemble, out)
    if args.frames == 0 and args.seconds == 0:
        args.seconds = 5
    return record(args)


if __name__ == "__main__":
    sys.exit(main())