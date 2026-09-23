"""② 终版：按 GT 框 IoU 匹配检测（避免"锁错门"），两域都映射到 b640 再比"""
import cv2, numpy as np, sys, json, collections
from pathlib import Path
sys.path.insert(0,"scripts/1_prepare"); sys.path.insert(0,"experiment/scripts/exp_distortion")
from distortion_geometry import Geometry, GROUP_A, GROUP_B
geo=Geometry("configs/front_camera.yaml",size=640,alpha=0.0); g=geo.for_frame(1280,720)
K640,dist,newK640=g.K_sq,g.dist,g.newK640
obj=np.array([[-0.35,-0.25,0],[0.35,-0.25,0],[0.35,0.25,0],[-0.35,0.25,0]],np.float64)
def pnp(p,K,D):
    ok,r,t=cv2.solvePnP(obj,np.asarray(p,np.float64),K,D,flags=cv2.SOLVEPNP_IPPE); return (r,t) if ok else (None,None)
def side(p): return float(np.mean([np.linalg.norm(p[i]-p[(i+1)%4]) for i in range(4)]))
def box(p): return np.array([p[:,0].min(),p[:,1].min(),p[:,0].max(),p[:,1].max()])
def iou(a,b):
    x0,y0=max(a[0],b[0]),max(a[1],b[1]); x1,y1=min(a[2],b[2]),min(a[3],b[3])
    i=max(0,x1-x0)*max(0,y1-y0); u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i
    return i/u if u>0 else 0
from ultralytics import YOLO
m=YOLO("weights/yolo11n-pose.pt")
data=[d for d in json.load(open("/tmp/e1_data.json")) if d["ok"]]
B=Path("experiment/runs/exp_distortion/e1/B_no_undistort"); A=Path("experiment/runs/exp_distortion/e1/A_undistort_first")
def infer(paths):
    out=[]
    for i in range(0,len(paths),32):
        for r in m.predict([str(p) for p in paths[i:i+32]],imgsz=640,device=0,conf=0.25,verbose=False,stream=True):
            lst=[]
            if r.keypoints is not None and r.keypoints.xy is not None and r.keypoints.xy.shape[0]:
                vv=r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
                for j,kk in enumerate(r.keypoints.xy.cpu().numpy()):
                    if kk.shape[0]==4: lst.append((kk, float(vv[j].min()) if vv is not None else 1.0))
            out.append(lst)
    return out
rA=infer([A/d["name"] for d in data]); rB=infer([B/d["name"] for d in data])
rows=[]; nomatch=0
for d,la,lb in zip(data,rA,rB):
    if not la or not lb: nomatch+=1; continue
    gtB=np.array(d["gtB"]); gtA=np.array(d["gtA"]); gtb=box(gtB)
    bA=max(la,key=lambda z:iou(box(z[0]),gtb)) if False else max(
        ((z,iou(box(z[0]),gtb)) for z in la), key=lambda t:t[1])
    bB=max(((z,iou(box(z[0]),gtb)) for z in lb), key=lambda t:t[1])
    if bA[1]<0.5 or bB[1]<0.5: nomatch+=1; continue
    mA,okA=g.transform(bA[0][0],GROUP_A,GROUP_B)
    if not np.all(okA): nomatch+=1; continue
    eA=float(np.mean(np.linalg.norm(np.asarray(mA)-gtB,axis=1)))
    eB=float(np.mean(np.linalg.norm(np.asarray(bB[0][0])-gtB,axis=1)))
    _,tg=pnp(gtB,K640,dist); _,tA=pnp(mA,K640,dist); _,tB=pnp(np.asarray(bB[0][0]),K640,dist)
    if tg is None or tA is None or tB is None: nomatch+=1; continue
    dzA=abs(tA.ravel()[2]-tg.ravel()[2])/tg.ravel()[2]*100
    dzB=abs(tB.ravel()[2]-tg.ravel()[2])/tg.ravel()[2]*100
    rows.append((d["stratum"],d["rho"],d["water"],side(gtB),eA,eB,dzA,dzB,bA[1],bB[1]))
print(f"有效配对 {len(rows)}/{len(data)}（剔除 {nomatch}：无检出/IoU<0.5/映射失败）")
if not rows: sys.exit("无样本")
print(f"\n{'域':<14}{'角点误差中位(px)':>18}{'p90':>9}{'均值':>9}{'深度误差中位(%)':>16}")
for tag,ie,idz in [("A(去畸变)",4,6),("B(畸变)",5,7)]:
    e=np.array([r[ie] for r in rows]); dz=np.array([r[idz] for r in rows])
    print(f"{tag:<14}{np.median(e):>18.2f}{np.percentile(e,90):>9.2f}{e.mean():>9.2f}{np.median(dz):>16.2f}")
print(f"\n按 ρ 档（角点误差中位 px）")
for s in ["内","中","外"]:
    sub=[r for r in rows if r[0]==s]
    if sub: print(f"  {s:<3}(n={len(sub):>2})  A {np.median([r[4] for r in sub]):6.2f}   B {np.median([r[5] for r in sub]):6.2f}")
print(f"\n按水质组")
for w in sorted(set(r[2] for r in rows)):
    sub=[r for r in rows if r[2]==w]
    print(f"  {w:<16}(n={len(sub):>2})  A {np.median([r[4] for r in sub]):6.2f}   B {np.median([r[5] for r in sub]):6.2f}")
print(f"\n配对 IoU 中位: A {np.median([r[8] for r in rows]):.3f}  B {np.median([r[9] for r in rows]):.3f}")
print(f"门框边长中位: {np.median([r[3] for r in rows]):.0f}px")
