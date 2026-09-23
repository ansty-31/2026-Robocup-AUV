# -*- coding: utf-8 -*-
"""板端：用部署的 gate .bin 跑 24 帧，dump 4 角点（原始帧坐标）与 conf"""
import sys, os, glob, json, time
ROOT="/home/sunrise/Desktop/AUV_New"; sys.path.insert(0,ROOT); os.chdir(ROOT)
import cv2, numpy as np
import base.settings as S
from gate.gate_detector import build_gate_backend
be = build_gate_backend()
print("backend:", type(be).__name__, "| input_size:", getattr(be,"input_size",None))
frames=sorted(glob.glob("/tmp/bench2/*.jpg"))
out={}
t0=time.perf_counter()
for f in frames:
    im=cv2.imread(f)
    r=be.detect(im)
    if isinstance(r,tuple): r=r[0]
    dets=[]
    for d in (r or []):
        dets.append(dict(kpts=np.asarray(d.kpts).tolist(), conf=np.asarray(d.kpt_conf).tolist(),
                         score=float(d.score)))
    out[os.path.basename(f)]=dict(shape=list(im.shape[:2]), dets=dets)
dt=(time.perf_counter()-t0)/len(frames)*1000
print(f"端到端 detect()（含预处理+NV12+BPU+解码）：{dt:.2f} ms/帧")
json.dump(out, open("/tmp/board_kpts2.json","w"))
n=sum(1 for v in out.values() if v["dets"])
print(f"有检出帧 {n}/{len(out)}")
