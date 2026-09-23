# -*- coding: utf-8 -*-
"""端到端 detect() 三档对比（串帧、每档同一批 28 帧，可比）"""
import sys, os, glob, time
ROOT="/home/sunrise/Desktop/AUV_New"; sys.path.insert(0,ROOT); os.chdir(ROOT)
import cv2, numpy as np
import common.preprocess as P
import base.settings as S

_WB={}
def _wb(gains):
    k=tuple(gains); v=_WB.get(k)
    if v is None:
        idx=np.arange(256,dtype=np.float32); v=[np.clip(idx*g,0,255).astype(np.uint8) for g in gains]; _WB[k]=v
    return v
_orig=P.enhance
def enhance_lut(frame,gains,clip,gamma):
    w=_wb(gains); b,g,r=cv2.split(frame)
    f=cv2.merge((cv2.LUT(b,w[0]),cv2.LUT(g,w[1]),cv2.LUT(r,w[2])))
    if clip>0:
        lab=cv2.cvtColor(f,cv2.COLOR_BGR2LAB); l,a,bb=cv2.split(lab)
        l=P._clahe(clip).apply(l); f=cv2.cvtColor(cv2.merge((l,a,bb)),cv2.COLOR_LAB2BGR)
    return cv2.LUT(f,P._gamma_lut(gamma))

frames=[cv2.imread(f) for f in sorted(glob.glob("/tmp/bench2/*.jpg"))]
frames=[f for f in frames if f is not None]
from gate.gate_detector import build_gate_backend
def run_once(tag):
    be=build_gate_backend()
    for f in frames[:3]: be.detect(f)       # 预热
    t=time.perf_counter()
    for f in frames: be.detect(f)
    ms=(time.perf_counter()-t)/len(frames)*1000
    print(f"{tag:<44} {ms:7.2f} ms/帧   → {1000/ms:5.1f} FPS")
    return ms
a=run_once("A 现状（float WB + CLAHE 0.5）")
P.enhance=enhance_lut
b=run_once("B LUT-WB（CLAHE 仍 0.5）")
S.vision.image.clahe_clip=0.0
c=run_once("C LUT-WB + clip=0（去 CLAHE）")
print(f"\nB 相对 A 省 {a-b:.1f} ms；C 相对 A 省 {a-c:.1f} ms（提速 {a/c:.2f}×）")
