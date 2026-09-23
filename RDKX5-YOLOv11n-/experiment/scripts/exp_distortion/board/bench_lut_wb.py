# -*- coding: utf-8 -*-
"""板端：① 证明 LUT 版白平衡与原 float 版逐像素相同 ② 实测提速 ③ BPU 推理耗时"""
import sys, os, time, glob
ROOT="/home/sunrise/Desktop/AUV_New"; sys.path.insert(0,ROOT); os.chdir(ROOT)
import cv2, numpy as np
import common.preprocess as P

frames=[cv2.imread(f) for f in sorted(glob.glob("/tmp/bench_frames/*.jpg"))]
frames=[f for f in frames if f is not None]
GAINS=[1.0,1.05,1.15]; CLIP=0.5; GAMMA=0.85

_WB={}
def _wb_luts(gains):
    k=tuple(gains); v=_WB.get(k)
    if v is None:
        idx=np.arange(256,dtype=np.float32)
        v=[np.clip(idx*g,0,255).astype(np.uint8) for g in gains]
        _WB[k]=v
    return v

def enhance_lut(frame, gains, clip, gamma):
    """与 P.enhance 等价（逐像素相同）但白平衡走 256 项 LUT"""
    w=_wb_luts(gains)
    b,g,r=cv2.split(frame)
    f=cv2.merge((cv2.LUT(b,w[0]),cv2.LUT(g,w[1]),cv2.LUT(r,w[2])))
    if clip>0:
        lab=cv2.cvtColor(f,cv2.COLOR_BGR2LAB); l,a,bb=cv2.split(lab)
        l=P._clahe(clip).apply(l)
        f=cv2.cvtColor(cv2.merge((l,a,bb)),cv2.COLOR_LAB2BGR)
    return cv2.LUT(f,P._gamma_lut(gamma))

sq=[cv2.resize(f,(640,640),interpolation=cv2.INTER_LINEAR) for f in frames]
a=[P.enhance(x.copy(),GAINS,CLIP,GAMMA) for x in sq]
b=[enhance_lut(x.copy(),GAINS,CLIP,GAMMA) for x in sq]
diff=max(int(np.abs(x.astype(int)-y.astype(int)).max()) for x,y in zip(a,b))
nequal=sum(1 for x,y in zip(a,b) if np.array_equal(x,y))
print(f"① LUT 版 vs 原 float 版：最大像素差 {diff}，逐像素完全相同 {nequal}/{len(a)} 张")

def bench(fn,n=3):
    for x in sq[:4]: fn(x)
    t=time.perf_counter()
    for _ in range(n):
        for x in sq: fn(x)
    return (time.perf_counter()-t)/(n*len(sq))*1000
t0=bench(lambda x:P.enhance(x.copy(),GAINS,CLIP,GAMMA))
t1=bench(lambda x:enhance_lut(x.copy(),GAINS,CLIP,GAMMA))
print(f"② enhance 本体：float 版 {t0:.2f} ms  →  LUT 版 {t1:.2f} ms  （省 {t0-t1:.2f} ms）")

CAL="cfg/front_camera.yaml"
for tag,fn in (("原版",P.enhance),("LUT版",enhance_lut)):
    P.enhance=fn
    pre=P.ModelPreprocessor(size=640,calib_path=CAL,undistort=True,gains=GAINS,clip=CLIP,gamma=GAMMA)
    for x in frames[:4]: pre.process(x)
    t=time.perf_counter()
    for _ in range(3):
        for x in frames: pre.process(x)
    ms=(time.perf_counter()-t)/(3*len(frames))*1000
    print(f"③ 整条 P1 预处理（{tag} enhance）：{ms:.2f} ms/帧")
P.enhance=enhance_lut
pre=P.ModelPreprocessor(size=640,calib_path=CAL,undistort=True,gains=GAINS,clip=0.0,gamma=GAMMA)
t=time.perf_counter()
for _ in range(3):
    for x in frames: pre.process(x)
print(f"   P1 + LUT版enhance + clip=0（再去 CLAHE）：{(time.perf_counter()-t)/(3*len(frames))*1000:.2f} ms/帧")
