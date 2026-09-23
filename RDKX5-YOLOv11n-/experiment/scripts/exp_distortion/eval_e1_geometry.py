"""E1 评测：① 几何 N1 vs N2（人工标注）② 现有模型在 B 域 vs A 域的角点/位姿误差"""
import cv2, numpy as np, sys, json, collections
from pathlib import Path
sys.path.insert(0,"scripts/1_prepare"); sys.path.insert(0,"experiment/scripts/exp_distortion")
from distortion_geometry import Geometry, GROUP_A, GROUP_B
C="configs/front_camera.yaml"
geo=Geometry(C,size=640,alpha=0.0); g=geo.for_frame(1280,720)
K640,dist,newK640=g.K_sq,g.dist,g.newK640
W,H=0.70,0.50
obj=np.array([[-W/2,-H/2,0],[W/2,-H/2,0],[W/2,H/2,0],[-W/2,H/2,0]],np.float64)
def pnp(pts,K,D):
    ok,r,t=cv2.solvePnP(obj,np.asarray(pts,np.float64),K,D,flags=cv2.SOLVEPNP_IPPE)
    if not ok: return None
    rms=float(np.sqrt(np.mean(np.sum((cv2.projectPoints(obj,r,t,K,D)[0].reshape(-1,2)-np.asarray(pts))**2,1))))
    return r,t,rms
# ---- 读标注
recs=[json.loads(l) for l in Path("experiment/runs/exp_distortion/e1_labels_B.jsonl").read_text().splitlines() if l.strip()]
manifest={r["文件名"]:r for r in __import__("csv").DictReader(open("experiment/runs/exp_distortion/e1_manifest.csv"))}
data=[]
for r in recs:
    name=Path(r.get("src_path") or r["src"]).name
    gtB=np.array(r["dets"][0]["kpts"],np.float32)
    m=next((v for k,v in manifest.items() if name.endswith(k)), None)
    if m is None: continue
    gtA,ok=g.transform(gtB,GROUP_B,GROUP_A)
    data.append({"name":name,"gtB":gtB,"gtA":gtA if np.all(ok) else None,"ok":bool(np.all(ok)),
                 "rho":float(m["ρ(角点max)"]),"water":m["水质组"],"stratum":m["ρ档"]})
print(f"可用标注 {len(data)} 条（映射可逆 {sum(1 for d in data if d['ok'])}）")

print("\n" + "="*78)
print("① 几何：同一条人工标注，N1(不去畸变) vs N2(去畸变) 解出的位姿")
print("="*78)
rows=[]
for d in data:
    if not d["ok"]: continue
    a=pnp(d["gtB"],K640,dist); b=pnp(d["gtA"],newK640,np.zeros(5))
    if a is None or b is None: continue
    dz=abs(b[1].ravel()[2]-a[1].ravel()[2])/a[1].ravel()[2]*100
    dr=float(np.degrees(np.linalg.norm(a[0]-b[0])))
    rows.append((d["stratum"],d["rho"],dz,dr,a[1].ravel()[2],a[2],b[2]))
if not rows: sys.exit("无可用样本")
a=np.array([(r[2],r[3]) for r in rows])
print(f"  样本 {len(rows)} 条")
print(f"  |Δ深度|/深度:  中位 {np.median(a[:,0]):.4f}%   p90 {np.percentile(a[:,0],90):.4f}%   最大 {a[:,0].max():.4f}%")
print(f"  |Δ旋转|:        中位 {np.median(a[:,1]):.4f}°  p90 {np.percentile(a[:,1],90):.4f}°  最大 {a[:,1].max():.4f}°")
print("  按 ρ 档:")
for s in ["内","中","外"]:
    sub=[r for r in rows if r[0]==s]
    if sub: print(f"    {s:<3}(n={len(sub):>2})  Δ深度中位 {np.median([x[2] for x in sub]):.4f}%   深度中位 {np.median([x[4] for x in sub]):.2f} m")
json.dump([{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in d.items()} for d in data],
          open("/tmp/e1_data.json","w"))
np.save("/tmp/e1_geo.npy", np.array(rows,dtype=object), allow_pickle=True)
