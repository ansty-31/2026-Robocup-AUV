# -*- coding: utf-8 -*-
import sys, os, glob, time
ROOT="/home/sunrise/Desktop/AUV_New"; sys.path.insert(0,ROOT); os.chdir(ROOT)
import cv2, numpy as np
import base.settings as S
from gate.gate_detector import build_gate_backend
frames=[cv2.imread(f) for f in sorted(glob.glob("/tmp/bench2/*.jpg"))]
frames=[f for f in frames if f is not None]
print("已打补丁（LUT 白平衡）；帧数",len(frames),"| 当前 clahe_clip =",S.vision.image.clahe_clip)
def run(clip):
    S.vision.image.clahe_clip=clip
    be=build_gate_backend()
    for f in frames[:3]: be.detect(f)
    t=time.perf_counter(); allk=[]
    for f in frames:
        r=be.detect(f)
        if isinstance(r,tuple): r=r[0]
        allk.append(np.asarray(max(r,key=lambda d:d.score).kpts) if r else None)
    ms=(time.perf_counter()-t)/len(frames)*1000
    return ms, allk
ms5,k5=run(0.5); ms0,k0=run(0.0)
print(f"  补丁+LUT-WB, clahe 0.5 : {ms5:6.2f} ms/帧 → {1000/ms5:5.1f} FPS")
print(f"  补丁+LUT-WB, clahe 0   : {ms0:6.2f} ms/帧 → {1000/ms0:5.1f} FPS")
d=[np.linalg.norm(a-b,axis=1).mean() for a,b in zip(k5,k0) if a is not None and b is not None]
d=np.array(d)
print(f"  关 CLAHE 前后的角点差（原始帧空间，1280x720）：中位 {np.median(d):.3f} px  p90 {np.percentile(d,90):.3f}  最大 {d.max():.3f}")
print(f"  ⇒ 换算到 640 输入空间中位 {np.median(d)/2:.3f} px")
