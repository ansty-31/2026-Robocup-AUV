# -*- coding: utf-8 -*-
"""板端预处理变体测速：全部走板端自己的 common/preprocess.ModelPreprocessor"""
import sys, time, glob, cv2, numpy as np
ROOT = "/home/sunrise/Desktop/AUV_New"
sys.path.insert(0, ROOT)
import os
os.chdir(ROOT)
from common.preprocess import ModelPreprocessor

frames = [cv2.imread(f) for f in sorted(glob.glob("/tmp/bench_frames/*.jpg"))]
frames = [f for f in frames if f is not None]
print("帧数", len(frames), "尺寸", frames[0].shape[:2])
CAL = "cfg/front_camera.yaml"
V = {
 "P1 全enhance (当前cfg: clip0.5)": dict(undistort=True, gains=[1.0,1.05,1.15], clip=0.5, gamma=0.85),
 "P1 去CLAHE (clip=0)":            dict(undistort=True, gains=[1.0,1.05,1.15], clip=0.0, gamma=0.85),
 "P1 去enhance (全中性)":           dict(undistort=True, gains=[1.,1.,1.], clip=0.0, gamma=1.0),
 "P4 不去畸变+全enhance":           dict(undistort=False, gains=[1.0,1.05,1.15], clip=0.5, gamma=0.85),
 "P4 不去畸变+去enhance":           dict(undistort=False, gains=[1.,1.,1.], clip=0.0, gamma=1.0),
}
res = {}
for name, kw in V.items():
    pre = ModelPreprocessor(size=640, calib_path=CAL, **kw)
    for f in frames[:4]:
        pre.process(f)
    t = time.perf_counter()
    for _ in range(3):
        for f in frames:
            pre.process(f)
    ms = (time.perf_counter() - t) / (3 * len(frames)) * 1000
    res[name] = ms
    print(f"{name:<34} {ms:7.2f} ms/帧")
base = res["P1 全enhance (当前cfg: clip0.5)"]
print("\n相对当前链路(P1全enhance)的节省：")
for k, v in res.items():
    d = base - v
    if d > 0.5:
        print(f"  {k:<34} 省 {d:5.2f} ms  → 预处理 {v:5.2f} ms（{1000/v:.0f} FPS 上限）")
# NV12 打包
try:
    from common.preprocess import bgr_to_packed_nv12
    sq = ModelPreprocessor(size=640, calib_path=CAL, **V["P1 全enhance (当前cfg: clip0.5)"]).process(frames[0])
    for _ in range(5): bgr_to_packed_nv12(sq, 640, 640)
    t=time.perf_counter()
    for _ in range(20): bgr_to_packed_nv12(sq, 640, 640)
    print(f"  NV12 打包                              {(time.perf_counter()-t)/20*1000:5.2f} ms")
except Exception as e:
    print("  NV12 打包：", e)
