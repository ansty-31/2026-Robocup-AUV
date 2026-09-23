# -*- coding: utf-8 -*-
import sys, os, glob, time
ROOT="/home/sunrise/Desktop/AUV_New"; sys.path.insert(0,ROOT); os.chdir(ROOT)
import cv2, numpy as np
import common.preprocess as P
print("已加载补丁后的 common.preprocess ✓")

def enhance_orig(frame,gains,clip,gamma):      # 原实现（float 白平衡）
    f=np.clip(frame.astype(np.float32)*np.array(gains,np.float32),0,255).astype(np.uint8)
    if clip>0:
        lab=cv2.cvtColor(f,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab)
        l=P._clahe(clip).apply(l); f=cv2.cvtColor(cv2.merge((l,a,b)),cv2.COLOR_LAB2BGR)
    return cv2.LUT(f,P._gamma_lut(gamma))

frames=[cv2.imread(f) for f in sorted(glob.glob("/tmp/bench2/*.jpg"))]
frames=[f for f in frames if f is not None]
G=[1.0,1.05,1.15]; GA=0.85
for clip in (0.5, 0.0):
    eq=0; md=0
    for f in frames:
        a=enhance_orig(f.copy(),G,clip,GA); b=P.enhance(f.copy(),G,clip,GA)
        md=max(md,int(np.abs(a.astype(int)-b.astype(int)).max())); eq+=int(np.array_equal(a,b))
    verdict="✅ 逐像素完全相同" if eq==len(frames) and md==0 else "❌ 有差异"
    print(f"  clip={clip}: {eq}/{len(frames)} 张完全相同，最大差 {md}  {verdict}")
