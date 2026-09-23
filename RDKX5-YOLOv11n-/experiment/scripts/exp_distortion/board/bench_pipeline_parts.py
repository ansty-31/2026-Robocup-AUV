# -*- coding: utf-8 -*-
import sys, os, glob, time
ROOT="/home/sunrise/Desktop/AUV_New"; sys.path.insert(0,ROOT); os.chdir(ROOT)
import cv2, numpy as np
import base.settings as S
print("fast_nv12 =", S.get("vision.model.fast_nv12", False))
from gate.gate_detector import build_gate_backend
import gate.gate_decode as GD
be = build_gate_backend()
pre, model, key = be._pre, be._model, be._key
frames=[cv2.imread(f) for f in sorted(glob.glob("/tmp/bench_frames/*.jpg"))][:12]
def T(fn,n=3):
    for _ in range(2): fn()
    t=time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter()-t)/n*1000
sq = pre.process(frames[0])
print("pack 函数:", getattr(GD,'bgr_to_packed_nv12').__name__, "| 有 fast 版:", hasattr(GD,'bgr_to_packed_nv12_fast'))
t_pre  = T(lambda: pre.process(frames[0]))
t_pack = T(lambda: GD.bgr_to_packed_nv12(sq, 640, 640))
try:
    t_fast = T(lambda: GD.bgr_to_packed_nv12_fast(sq, 640, 640))
except Exception as e:
    t_fast = float('nan'); print("fast 失败:", e)
nv = GD.bgr_to_packed_nv12(sq, 640, 640)
def run(): return model.run({key: nv})
t_inf = T(run)
outs = run()
t_dec = T(lambda: GD.decode_yolo11_kpt(outs, ["gate"], 1280, 720, input_w=640, input_h=640))
print(f"\n① 预处理 pre.process        {t_pre:7.2f} ms")
print(f"② NV12 打包 (slow)          {t_pack:7.2f} ms")
print(f"② NV12 打包 (fast)          {t_fast:7.2f} ms")
print(f"③ BPU model.run             {t_inf:7.2f} ms")
print(f"④ decode_yolo11_kpt         {t_dec:7.2f} ms")
print(f"   合计(用 slow pack)        {t_pre+t_pack+t_inf+t_dec:7.2f} ms  → {1000/(t_pre+t_pack+t_inf+t_dec):.1f} FPS")
print(f"   合计(用 fast pack)        {t_pre+t_fast+t_inf+t_dec:7.2f} ms  → {1000/(t_pre+t_fast+t_inf+t_dec):.1f} FPS")
