# -*- coding: utf-8 -*-
"""换标定前后：同一批帧、同一批 kpt，走板端 gate_pose 比较门距"""
import sys, os, glob
ROOT="/home/sunrise/Desktop/AUV_New"; sys.path.insert(0,ROOT); os.chdir(ROOT)
import cv2, numpy as np
import base.settings as S
from gate.gate_detector import build_gate_backend
from gate.geometry import CameraModel, object_points, gate_pose
be=build_gate_backend()
frames=[cv2.imread(f) for f in sorted(glob.glob("/tmp/bench2/*.jpg"))]
op=object_points()
NEW=CameraModel.from_yaml("cfg/front_camera.yaml", rectified=True)                 # 现在生效(C)
OLD=CameraModel.from_yaml("cfg/front_camera.yaml.bak.A_fx782_20260923", rectified=True)  # 换之前(A)
print(f"nk720 fx：旧(A) {OLD.fx:.1f}  →  新(C) {NEW.fx:.1f}   比值 {NEW.fx/OLD.fx:.3f}")
dn,do=[],[]
for f in frames:
    r=be.detect(f)
    if isinstance(r,tuple): r=r[0]
    if not r: continue
    d=max(r,key=lambda x:x.score)
    k=np.asarray(d.kpts)
    for cam,acc in ((NEW,dn),(OLD,do)):
        res=gate_pose(cam, op, k)
        if res is not None: acc.append(float(np.linalg.norm(np.asarray(res[1]).ravel())))
print(f"门距中位：旧(A) {np.median(do):.3f} m  →  新(C) {np.median(dn):.3f} m   （n={len(dn)}）")
print(f"实测比值 {np.median(dn)/np.median(do):.3f}（预期 = nk720 比值 1.308）")
print(f"⇒ 换标定前：读数 = 真值 × {1/ (np.median(dn)/np.median(do)) if np.median(dn)>0 else 0:.3f}" if False else "")
