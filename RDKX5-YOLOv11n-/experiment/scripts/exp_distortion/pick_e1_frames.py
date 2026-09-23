"""E1 备料 v3：判据 = 四角关键点在画面内且可信；ρ 取四角的**最大**径向距离"""
import cv2, numpy as np, sys, glob, random, csv, shutil, collections
from pathlib import Path
sys.path.insert(0,"scripts/1_prepare")
C="configs/front_camera.yaml"
fs=cv2.FileStorage(C,cv2.FILE_STORAGE_READ); K=fs.getNode("camera_matrix").mat(); fs.release()
cx,cy=K[0,2],K[1,2]; half=np.hypot(640,360)
from ultralytics import YOLO
m=YOLO("weights/yolo11n-pose.pt")
DIVES=[("AUV_1",["data/AUV_1/rec_front_frames"],400),("AUV_2",["data/AUV_2/auv_20260910_172208_frames"],400),
       ("AUV_3",["data/AUV_3/auv_20260911_201754_frames"],400),("AUV_4",["data/AUV_4/auv_4_frames"],400),
       ("AUV_5",sorted(glob.glob("data/AUV_5/raw-data/gate-*")),900)]
random.seed(0)
cands=collections.defaultdict(list); stat=collections.Counter()
for tag,pats,NS in DIVES:
    fl=[]
    for p in pats: fl+=sorted(glob.glob(p+"/*.jpg"))
    fl=random.sample(fl,min(NS,len(fl)))
    imgs=[]; ok=[]
    for f in fl:
        r=cv2.imread(f)
        if r is not None and r.shape[:2]==(720,1280):
            imgs.append(cv2.resize(r,(640,640),interpolation=cv2.INTER_LINEAR)); ok.append(f)
    for i in range(0,len(imgs),32):
        for r,f in zip(m.predict(imgs[i:i+32],imgsz=640,device=0,conf=0.25,verbose=False), ok[i:i+32]):
            stat[tag+":帧"]+=1
            if r.keypoints is None or r.keypoints.xy is None or r.keypoints.xy.shape[0]==0:
                stat[tag+":无kpt"]+=1; continue
            vv=(r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None)
            best=None
            for j in range(r.keypoints.xy.shape[0]):
                if vv is not None and not np.all(vv[j]>=0.25): continue
                kk=r.keypoints.xy.cpu().numpy()[j]
                if kk.shape[0]!=4: continue
                if not np.all((kk>0.01*640)&(kk<0.99*640)): continue
                # b640 -> 原始像素
                kr=kk.copy(); kr[:,0]*=2.0; kr[:,1]*=640/720
                rho=float(np.max(np.hypot(kr[:,0]-cx,kr[:,1]-cy)/half))
                if best is None or rho>best[0]: best=(rho,f,kk,kk.min(0),kk.max(0))
            if best is None: stat[tag+":四角不齐"]+=1; continue
            cands[tag].append(best)
    print(f"{tag}: 合格 {len(cands[tag])}/{len(fl)}  ρ(角点max) 中位 "
          f"{np.median([c[0] for c in cands[tag]]) if cands[tag] else -1:.3f}", flush=True)
import json as _j; _j.dump({k:[(c[0],c[1]) for c in v] for k,v in cands.items()}, open("/tmp/e1_cands_v3.json","w"))
print("\n=== 漏斗 ===")
for tag,_,_ in DIVES:
    print(f"  {tag}: 帧 {stat[tag+':帧']}  无kpt {stat[tag+':无kpt']}  四角不齐 {stat[tag+':四角不齐']}  合格 {len(cands[tag])}")
# 数据驱动的三分位分层（用"角点最大 ρ"的联合分布）
allrho=sorted(c[0] for v in cands.values() for c in v)
q1,q2=np.percentile(allrho,[33.3,66.7])
print(f"\n角点最大 ρ 联合分布: n={len(allrho)} p33={q1:.3f} p67={q2:.3f} min={allrho[0]:.3f} max={allrho[-1]:.3f}")
STRATA=[("内",0.0,q1),("中",q1,q2),("外",q2,1.01)]
WATER={"浊水(AUV_2_3)":["AUV_2","AUV_3"],"中(AUV_1_4)":["AUV_1","AUV_4"],"清水(AUV_5)":["AUV_5"]}
PER=6; sel=[]
print("\n=== 分层挑选（每格尽量取满 6）===")
for wname,ds in WATER.items():
    for sname,lo,hi in STRATA:
        pool=sorted([c for d in ds for c in cands.get(d,[]) if lo<=c[0]<hi], key=lambda c:c[0])
        if not pool: print(f"  {wname:<14}{sname:<3}[{lo:.2f},{hi:.2f})  ⚠️ 无候选"); continue
        idx=sorted(set(np.linspace(0,len(pool)-1,min(PER,len(pool))).astype(int).tolist()))
        for i in idx:
            rho,f,kk,a,b=pool[i]; sel.append((wname,sname,round(rho,3),f))
        print(f"  {wname:<14}{sname:<3}[{lo:.2f},{hi:.2f})  候选 {len(pool):>3} 取 {len(idx)}")
out=Path("experiment/runs/exp_distortion/e1_picked")
if out.exists(): shutil.rmtree(out)
out.mkdir(parents=True)
for w,s,rho,f in sel: shutil.copy2(f,out/f"{Path(f).parent.name}__{Path(f).name}")
with open("experiment/runs/exp_distortion/e1_manifest.csv","w",newline="") as fh:
    wr=csv.writer(fh); wr.writerow(["水质组","ρ档","ρ(角点max)","原路径","文件名"])
    for x in sel: wr.writerow([x[0],x[1],x[2],x[3],f"{Path(x[3]).parent.name}__{Path(x[3]).name}"])
print(f"\n✅ 挑出 {len(sel)} 帧")
