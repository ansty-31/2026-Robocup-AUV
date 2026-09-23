#!/usr/bin/env python3
"""
域方案评估：GT 角点误差 + PnP 门框物理长宽估算误差

跨域可比性的关键
----------------
不同域的图像坐标不能直接比：B 域是「resize 后的畸变像素」，C 域是「720p 去畸变后
缩放」，D 域是「640 上去畸变」。要让两臂的角点误差可比，必须都换算到**同一物理空间**
——裸流 1280x720 畸变像素（相机原始成像面）：

    B   : pts_B  * (1280/640, 720/640)                       ← 纯缩放
    C   : pts_C  * (2, 1.125) → 用 C 的 720p 映射表反查回 raw
    D   : pts_D  → 用 C 的 640 映射表反查回 畸变640 → ×(2, 1.125)
    AOLD: pts_A  * (2, 1.125) → 用 A 的 720p 映射表反查回 raw

PnP 则各自用**自己域的正确相机模型**（物理上等价，E1 ① 已证）：
    B   用 (K, dist)          —— 畸变点直接吃畸变模型
    C   用 (nk720, 0)
    D   用 (nk640, 0)
    AOLD 用 (nk720_A, 0)

门框长宽估算（你要的判据）
--------------------------
`solvePnP(矩形 obj 长 0.77 / 高 0.56)` 拿到位姿后，把 4 个观测角点做成从光心出发的射线，
与解出的门平面求交，得到门坐标系下 4 个 3D 点，再量出实际的长和宽：

    W_est = mean(|X_TR-X_TL|, |X_BR-X_BL|)     H_est = mean(|Y_BL-Y_TL|, |Y_BR-Y_TR|)
    误差  = |W_est - 0.77| / 0.77            （H 同理）

几何上一致的角点 → W/H 回到真值；角点被"拉歪"（畸变模型不匹配的典型症状）→ W/H 偏。
同时报 reproj RMS 与解出的深度（跨臂相对比较用）。

用法：
    python experiment/scripts/exp_distortion/eval_domain_arms.py \
        --arm B:experiment/data/pose_B:weights/domain_B.pt \
        --arm C:experiment/data/pose_C:weights/domain_C.pt \
        --split test --out experiment/runs/domain/EVAL.md
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "1_prepare"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "1_prepare" / "pose"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from map_pose_dataset import S, SIZE, RAW_W, RAW_H, bilinear, load_calib  # noqa: E402

GATE_W, GATE_H = 0.77, 0.56          # auv_vision/gate/geometry.py: GATE_FRAME_W/H
CALIB_C = "configs/front_camera.yaml"
CALIB_A = "configs/backup/front_camera_AUV1_water_fx782.yaml"


class Cam:
    """一个域的相机模型：本域像素 ↔ 裸流畸变像素"""

    def __init__(self, name: str, calib: str | None):
        self.name = name
        self.calib = calib
        if calib:
            self.K, self.dist = load_calib(calib)
            self.nk720, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist,
                                                          (RAW_W, RAW_H), 0.0, (RAW_W, RAW_H))
            self.nk640 = S @ self.nk720
            self.K640 = S @ self.K
            self.m720f = cv2.initUndistortRectifyMap(self.K, self.dist, None, self.nk720,
                                                     (RAW_W, RAW_H), cv2.CV_32FC1)
            self.m640f = cv2.initUndistortRectifyMap(self.K640, self.dist, None, self.nk640,
                                                     (SIZE, SIZE), cv2.CV_32FC1)

    def to_raw(self, pts):
        """本域 640 像素 → 裸流畸变像素"""
        p = np.asarray(pts, np.float64)
        if self.name == "B":
            return p * np.array([RAW_W / SIZE, RAW_H / SIZE])
        if self.name == "AOLD":
            q = p * np.array([RAW_W / SIZE, RAW_H / SIZE])
            return np.stack([bilinear(self.m720f[0], q[:, 0], q[:, 1]),
                             bilinear(self.m720f[1], q[:, 0], q[:, 1])], 1)
        if self.name == "C":
            q = p * np.array([RAW_W / SIZE, RAW_H / SIZE])
            return np.stack([bilinear(self.m720f[0], q[:, 0], q[:, 1]),
                             bilinear(self.m720f[1], q[:, 0], q[:, 1])], 1)
        if self.name == "D":
            q = np.stack([bilinear(self.m640f[0], p[:, 0], p[:, 1]),
                          bilinear(self.m640f[1], p[:, 0], p[:, 1])], 1)
            return q * np.array([RAW_W / SIZE, RAW_H / SIZE])
        raise ValueError(self.name)

    def pnp_pts(self, pts):
        """本域像素 → PnP 用的 (点, Kx, Dx)"""
        p = np.asarray(pts, np.float64)
        if self.name == "B":
            # B 域的点在 640 像素上，必须用 640 尺度的 K（= S@K），
            # 否则 fx/cx 差 2 倍，PnP 深度和门框长宽全错
            return p, self.K640, self.dist
        if self.name == "C":
            return p * np.array([RAW_W / SIZE, RAW_H / SIZE]), self.nk720, np.zeros(5)
        if self.name == "D":
            return p, self.nk640, np.zeros(5)
        if self.name == "AOLD":
            return p * np.array([RAW_W / SIZE, RAW_H / SIZE]), self.nk720, np.zeros(5)
        raise ValueError(self.name)


class CamAoldC(Cam):
    """点来自 **A-old 域**（部署中模型的输出），但用 **标定 C 的几何** 评估。

    为什么需要：标定 A 与 C 对同一相机给出的门距差 59%（2026-09-23 实测，
    两者各自内部自洽）。要让"部署中模型"与 B/C 新臂在同一个几何基准上比，
    就把 A 域预测点经 A 的映射表映回裸流像素，再用 C 的 (K, dist) 做 PnP。
    """

    def __init__(self):
        super().__init__("AOLD", CALIB_A)
        self._c = Cam("B", CALIB_C)

    def pnp_pts(self, pts):
        return self.to_raw(pts), self._c.K, self._c.dist


def obj_points():
    return np.array([[-GATE_W / 2, -GATE_H / 2, 0], [GATE_W / 2, -GATE_H / 2, 0],
                     [GATE_W / 2, GATE_H / 2, 0], [-GATE_W / 2, GATE_H / 2, 0]], np.float64)


def implied_wh(pts, Kx, Dx, obj):
    """PnP 后射线-门平面求交，量出实际门宽/门高；返回 (W,H,reproj_rms,depth)"""
    if Dx is not None and np.any(np.asarray(Dx) != 0):
        norm = cv2.undistortPoints(pts.reshape(-1, 1, 2), Kx, Dx).reshape(-1, 2)
    else:
        norm = cv2.undistortPoints(pts.reshape(-1, 1, 2), Kx, None).reshape(-1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj, pts, Kx, Dx if Dx is not None else None,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.ravel()
    n = R[:, 2]                                   # 门平面法向（相机系）
    pts3 = []
    for d in norm:
        d3 = np.array([d[0], d[1], 1.0])
        denom = float(n @ d3)
        if abs(denom) < 1e-9:
            return None
        s = float(n @ t) / denom
        X = s * d3
        pts3.append(R.T @ (X - t))
    p3 = np.array(pts3)
    W = 0.5 * (np.linalg.norm(p3[1] - p3[0]) + np.linalg.norm(p3[2] - p3[3]))
    H = 0.5 * (np.linalg.norm(p3[3] - p3[0]) + np.linalg.norm(p3[2] - p3[1]))
    uv, _ = cv2.projectPoints(obj, rvec, tvec, Kx, Dx if Dx is not None else None)
    rms = float(np.sqrt(np.mean((uv.reshape(-1, 2) - pts) ** 2)))
    return float(W), float(H), rms, float(np.linalg.norm(t))


def pkey(img: str) -> str:
    """配对键：去掉重投影数据集命名时追加的 _<8位hex> 后缀，实现跨数据集按物理帧配对"""
    return re.sub(r"_[0-9a-f]{8}\.jpg$", ".jpg", img)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def gt_of(label_txt: Path, size=SIZE):
    """→ (kpts640 (4,2), box640, vis (4,))；vis 来自标签第 3 通道（0=未标/不可见）"""
    f = [float(x) for x in label_txt.read_text().split()]
    if len(f) < 17:                      # 空标签/截断标签（老数据集里有）
        return None
    cx, cy, w, h = f[1:5]
    k = np.array(f[5:17], np.float64).reshape(4, 3)
    box = np.array([(cx - w / 2) * size, (cy - h / 2) * size,
                    (cx + w / 2) * size, (cy + h / 2) * size])
    return k[:, :2] * size, box, k[:, 2]


WATER_FROM_SRC = {"AUV_1": "中水", "AUV_4dir": "中水", "AUV_4": "中水",
                  "AUV_2": "浊水", "AUV_3": "浊水", "AUV_5gate": "清水"}


def load_manifest(ds: Path) -> dict:
    """分层信息：优先 manifest.csv；否则用溯源总表 + 标签现算（让老数据集也能分层）"""
    mp = ds / "manifest.csv"
    if mp.exists():
        return {r["name"] + ".jpg": r for r in csv.DictReader(open(mp))}
    prov = PROJECT_ROOT / "runs/prov/provenance_final_PNP.kpt4.yolov8.csv"
    if not prov.exists():
        return {}
    pv = {r["img"]: r for r in csv.DictReader(open(prov)) if r.get("accept") == "1"}
    out = {}
    for p in ds.glob("*/images/*.jpg"):
        r = pv.get(p.name)
        if not r:
            continue
        rho = float("nan")
        lf = p.parent.parent / "labels" / (p.stem + ".txt")
        if lf.exists():
            f = [float(x) for x in lf.read_text().split()]
            if len(f) >= 17:                     # 空标签/格式不符则跳过 rho
                k = np.array(f[5:17], np.float64).reshape(4, 3)[:, :2] * SIZE
                rho = float(np.max(np.linalg.norm((k - SIZE / 2) / (SIZE / 2), axis=1)))
        out[p.name] = dict(water=WATER_FROM_SRC.get(r["src"], "?"), rho=rho,
                           dive=r["src"])
    return out


def eval_arm(tag, ds, weights, split, cam, conf=0.25, max_n=0, kpt_conf=0.5):
    from ultralytics import YOLO
    ds = PROJECT_ROOT / ds
    imgs = sorted((ds / split / "images").glob("*.jpg"))
    if max_n:
        imgs = imgs[:max_n]
    man = load_manifest(ds)
    model = YOLO(str(PROJECT_ROOT / weights))
    rec = []
    for i in range(0, len(imgs), 32):
        batch = imgs[i:i + 32]
        res = model.predict([str(p) for p in batch], imgsz=SIZE, conf=conf,
                            verbose=False, device=0)
        for p, r in zip(batch, res):
            mrow = man.get(p.name, {})
            meta = dict(water=mrow.get("water", "?"),
                        rho=float(mrow["rho"]) if mrow.get("rho") else float("nan"),
                        dive=mrow.get("dive", ""))
            gt = gt_of(ds / split / "labels" / (p.stem + ".txt"))
            if gt is None:
                rec.append(dict(img=p.name, hit=0, skip="bad_label", **meta))
                continue
            gk, gbox, gvis = gt
            if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
                rec.append(dict(img=p.name, hit=0, **meta))
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            best = max(range(len(boxes)), key=lambda j: iou(boxes[j], gbox))
            if iou(boxes[best], gbox) < 0.5:
                rec.append(dict(img=p.name, hit=0, **meta))
                continue
            pk = r.keypoints.xy.cpu().numpy()[best]
            try:
                pc = r.keypoints.conf.cpu().numpy()[best]
            except Exception:
                pc = np.ones(4)
            # 有效角点 = GT 标了可见(v>0) 且模型有把握(conf>=阈值，且不是 (0,0) 占位)
            valid = (gvis > 0) & (pc >= kpt_conf) & ~((pk[:, 0] < 0.5) & (pk[:, 1] < 0.5))
            d = dict(img=p.name, hit=1, iou=round(iou(boxes[best], gbox), 3),
                     conf=float(r.boxes.conf.cpu().numpy()[best]),
                     n_valid=int(valid.sum()), n_gt_vis=int((gvis > 0).sum()),
                     all4=int(valid.sum() == 4), **meta)
            if valid.any():
                cerr = np.linalg.norm(cam.to_raw(pk[valid]) - cam.to_raw(gk[valid]), axis=1)
                d.update(cerr_med=float(np.median(cerr)), cerr_max=float(cerr.max()))
            if valid.sum() == 4:                      # PnP 必须要四角
                pnp = implied_wh(*cam.pnp_pts(pk), obj_points())
                if pnp:
                    W, H, rms, dep = pnp
                    d.update(W=W, H=H, rms=rms, depth=dep,
                             dW=abs(W - GATE_W) / GATE_W, dH=abs(H - GATE_H) / GATE_H)
            if (gvis > 0).sum() == 4:
                gt_pnp = implied_wh(*cam.pnp_pts(gk), obj_points())
                if gt_pnp:
                    d["gt_depth"] = gt_pnp[3]
            rec.append(d)
    return rec


def summarize(tag, rec):
    hit = [r for r in rec if r.get("hit")]
    n = len(rec)
    if not hit:
        return dict(tag=tag, n=n, hit=0)
    def med(k):
        v = [r[k] for r in hit if k in r and r[k] is not None]
        return float(np.median(v)) if v else float("nan")
    gt4 = [r for r in hit if r.get("n_gt_vis") == 4]      # GT 四角齐全的帧
    n4 = int(sum(r.get("all4", 0) for r in gt4))
    return dict(tag=tag, n=n, hit=len(hit), rate=len(hit) / n,
                n_gt4=len(gt4), all4_n=n4,
                # 归一化口径：只看「GT 四角都可见」的帧里，模型给出 4 个角的比例
                all4=float(n4 / len(gt4)) if gt4 else float("nan"),
                cerr_med=med("cerr_med"), cerr_max_med=med("cerr_max"),
                dW=med("dW"), dH=med("dH"), rms=med("rms"),
                depth=med("depth"), gt_depth=med("gt_depth"), conf=med("conf"))


def strata(rec):
    """分层：ρ 三分位（畸变是径向的，不分层会打平）+ 水况。返回 [(层名, 子集)]"""
    out = [("全体", rec)]
    rho = np.array([r.get("rho", float("nan")) for r in rec], float)
    ok = np.isfinite(rho)
    if ok.sum() >= 30:
        p33, p67 = np.percentile(rho[ok], [33, 67])
        out += [(f"ρ内(≤{p33:.2f})", [r for r, m in zip(rec, ok & (rho <= p33)) if m]),
                (f"ρ中", [r for r, m in zip(rec, ok & (rho > p33) & (rho <= p67)) if m]),
                (f"ρ外(>{p67:.2f})", [r for r, m in zip(rec, ok & (rho > p67)) if m])]
    for w in ("中水", "浊水"):
        sub = [r for r in rec if r.get("water") == w]
        if len(sub) >= 10:
            out.append((w, sub))
    return out


def line_of(tag, layer, sub):
    r = summarize(tag, sub)
    if not r.get("hit"):
        return (f"{tag:<12}{layer:<12}{'0/%d' % r['n']:>9}  —— 无有效检出",
                f"| {tag} | {layer} | 0/{r['n']} | — | — | — | — | — |")
    a4 = r.get("all4", float("nan"))
    a4s = f"{a4*100:.1f}%" if np.isfinite(a4) else "—"
    txt = (f"{tag:<12}{layer:<12}{'%d/%d' % (r['hit'], r['n']):>9}{a4s:>10}"
           f"{r['cerr_med']:>15.2f}{r['dW']*100:>9.2f}%{r['dH']*100:>9.2f}%"
           f"{r['rms']:>11.2f}{r['depth']:>9.3f}")
    md = (f"| {tag} | {layer} | {r['hit']}/{r['n']} | {a4s} ({r.get('all4_n',0)}/{r.get('n_gt4',0)}) | {r['cerr_med']:.2f} | "
          f"{r['dW']*100:.2f}% | {r['dH']*100:.2f}% | {r['rms']:.2f} | {r['depth']:.3f} |")
    return txt, md


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                    help="tag:数据集目录:权重:相机模型(B|C|D|AOLD|AOLD_C)[:split]"
                         "（AOLD_C = A-old 域的点 + 标定 C 的几何，用于与 B/C 臂同基准）")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-n", type=int, default=0)
    ap.add_argument("--kpt-conf", type=float, default=0.5,
                    help="关键点置信阈值：低于此值视为未预测出该角")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pair", action="append", default=[],
                    help="配对比较：tagA:tagB（两臂须跑同一批同名图）")
    a = ap.parse_args()

    detail, allrows = {}, []
    for spec in a.arm:
        parts = spec.split(":")
        tag, ds, wt = parts[0], parts[1], parts[2]
        camname = parts[3] if len(parts) > 3 else tag
        split = parts[4] if len(parts) > 4 else a.split
        if camname == "AOLD_C":
            cam = CamAoldC()
        else:
            cam = Cam(camname, CALIB_A if camname == "AOLD" else CALIB_C)
        if camname == "B" and cam.calib is None:
            cam = Cam("B", CALIB_C)
        print(f"评估 {tag}（相机模型 {camname}, split={split}）…", flush=True)
        rec = eval_arm(tag, ds, wt, split, cam, a.conf, a.max_n, a.kpt_conf)
        detail[tag] = rec
        for layer, sub in strata(rec):
            allrows.append(line_of(tag, layer, sub))

    pair_lines = []
    if a.pair:
        head = "\n########## 配对比较（同帧逐张）##########"
        print(head)
        pair_lines.append(head)
        for spec in a.pair:
            ta, tb = spec.split(":")
            A = {pkey(r["img"]): r for r in detail.get(ta, [])}
            B = {pkey(r["img"]): r for r in detail.get(tb, [])}
            com = sorted(set(A) & set(B))
            if not com:
                s = f"  {ta} vs {tb}: 无同名图可比"
                print(s)
                pair_lines.append(s)
                continue
            d, only_a, only_b = [], 0, 0
            for k in com:
                a4a, a4b = A[k].get("all4", 0), B[k].get("all4", 0)
                if a4a and not a4b:
                    only_a += 1
                elif a4b and not a4a:
                    only_b += 1
                ca, cb = A[k].get("cerr_med"), B[k].get("cerr_med")
                if ca is not None and cb is not None:
                    d.append(ca - cb)
            d = np.array(d)
            if d.size < 2:      # 两臂共同有角点误差的帧太少（如清水集）→ 跳过配对统计
                s1 = (f"  {ta} − {tb}  n={len(com)}  可比帧不足"
                      f"（仅 {d.size} 帧两臂都有角点误差）→ 跳过配对统计")
                s2 = f"    四角齐全：仅 {ta} 成功 {only_a} 帧，仅 {tb} 成功 {only_b} 帧"
                print(s1); print(s2); pair_lines += [s1, s2]
                continue
            s1 = (f"  {ta} − {tb}  n={len(com)}  角点误差差(负=A更好): 中位 {np.median(d):+.3f} px  "
                  f"p25 {np.percentile(d,25):+.2f}  p75 {np.percentile(d,75):+.2f}  "
                  f"A更好 {(d<0).mean()*100:.1f}% / B更好 {(d>0).mean()*100:.1f}%")
            s2 = f"    四角齐全：仅 {ta} 成功 {only_a} 帧，仅 {tb} 成功 {only_b} 帧"
            print(s1)
            print(s2)
            pair_lines += [s1, s2]
            try:
                from scipy.stats import wilcoxon
                if len(d) >= 10:
                    st = wilcoxon(d)
                    s3 = (f"    Wilcoxon 符号秩检验 p = {st.pvalue:.4g}"
                          f"{'  ← 显著(p<0.05)' if st.pvalue < 0.05 else '  ← 不显著'}")
                    print(s3)
                    pair_lines.append(s3)
            except Exception as e:
                s3 = f"    （跳过显著性检验：{e}）"
                print(s3)
                pair_lines.append(s3)

    hdr = (f"{'方案':<12}{'分层':<12}{'检出':>9}{'四角齐全(gt4内)':>15}{'角点误差中位(px)':>15}"
           f"{'长W误差':>9}{'宽H误差':>9}{'reprojRMS':>11}{'门距(m)':>9}")
    print("\n" + hdr)
    print("-" * 150)
    for txt, _ in allrows:
        print(txt)
    if a.out:
        p = PROJECT_ROOT / a.out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            f"# 域方案评估（split={a.split}, conf={a.conf}）\n\n"
            "角点误差在各域都换算到**裸流 1280x720 畸变像素**后比较；长W/宽H 由 "
            "PnP 位姿 + 射线-门平面求交量出（真值 0.77 / 0.56 m）。\n\n"
            "| 方案 | 分层 | 检出 | 四角齐全率(GT四角可见帧内) | 角点误差中位(px) | 长W误差 | 宽H误差 | reprojRMS | 门距(m) |\n"
            "|---|---|---|---|---|---|---|---|---|\n" + "\n".join(md for _, md in allrows) + "\n"
            + ("\n```\n" + "\n".join(pair_lines) + "\n```\n" if pair_lines else ""))
        try:
            shown = p.relative_to(PROJECT_ROOT)
        except ValueError:                     # --out 给的是项目外的绝对路径
            shown = p
        print(f"\n→ {shown}")
    (PROJECT_ROOT / "experiment/runs/domain").mkdir(parents=True, exist_ok=True)
    stem = Path(a.out).stem if a.out else "eval"
    json.dump(detail, open(PROJECT_ROOT / f"experiment/runs/domain/eval_detail_{stem}.json", "w"),
              indent=1)


if __name__ == "__main__":
    main()
