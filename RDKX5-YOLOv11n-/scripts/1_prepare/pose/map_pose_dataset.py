#!/usr/bin/env python3
"""
把已标注的 pose 数据集重投影到指定「去畸变域」（Route B 的产物落地）

为什么可以这样做
----------------
`runs/prov/provenance_final_*.csv` 给出了每张标注图对应的**裸流原始帧**
（`src` + `num_src`）以及当年生成它时用的链路（`chain`：`aold`=标定A去畸变 /
`nound`=不去畸变）。因此：

    标注图上的角点(640 像素)  --链路反变换-->  裸流像素  --新链路-->  新域角点(640 像素)
    裸流原始帧                --新链路------>  新域图像

不需要重新标注，也不损失任何信息（唯一损失是新域图像本身的 JPEG 重编码）。

三个域（与 PROTOCOL_undistort_scheme.md 的 P1/P2/P4 对齐）
----------------------------------------------------------
    B 域 = P4   resize(640)                → enhance?          [不去畸变]
    C 域 = P1   remap@720p → resize(640)   → enhance?          [720p 上去畸变]
    D 域 = P2   resize(640) → enhance?     → remap@640         [640 上去畸变]

⚠️ D 域**不能**用 `prepare_frames.calibration_maps(calib, 640, 640)`：那会用
未缩放的 K（fx≈1208，按 1280 宽标定）去处理 640 图，几何是错的。
P2 的正确定义（与 experiment/scripts/exp_distortion/bench_undistort_placement.py 一致）：

    S     = diag(640/1280, 640/720, 1)
    nk720 = getOptimalNewCameraMatrix(K, dist, (1280,720), 0, (1280,720))
    nk640 = S @ nk720
    m640  = initUndistortRectifyMap(S @ K, dist, None, nk640, (640,640))

用法：
    python scripts/1_prepare/pose/map_pose_dataset.py --domain B --enhance on  --out experiment/data/pose_B
    python scripts/1_prepare/pose/map_pose_dataset.py --domain C --enhance on  --out experiment/data/pose_C
    python scripts/1_prepare/pose/map_pose_dataset.py --domain D --enhance on  --out experiment/data/pose_D
    python scripts/1_prepare/pose/map_pose_dataset.py --domain B --enhance off --out experiment/data/pose_B_noenh
    # 自检：角点 环回误差（标注→raw→同域应回到原点）
    python scripts/1_prepare/pose/map_pose_dataset.py --domain C --selfcheck
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mmap
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "1_prepare"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import prepare_frames as pf  # noqa: E402

RAW_W, RAW_H, SIZE = 1280, 720, 640
S = np.diag([SIZE / RAW_W, SIZE / RAW_H, 1.0])
SOI = b"\xff\xd8\xff"
WB, CLAHE, GAMMA = [1.0, 1.05, 1.15], 0.5, 0.85

SRC_PATH = {                     # provenance 的 src → 取帧方式
    "AUV_4dir": ("dir", "data/AUV_4/auv_4_frames"),
    "AUV_4":    ("mjpeg", "data/AUV_4/auv_4.mjpeg"),
    "AUV_3":    ("mjpeg", "data/AUV_3/auv_20260911_201754.mjpeg"),
    "AUV_2":    ("mjpeg", "data/AUV_2/auv_20260910_172208.mjpeg"),
    "AUV_1":    ("video", "data/AUV_1/rec_front.mp4"),
    "AUV_1board": ("dir", "data/AUV_1/board"),
}
WATER = {"AUV_1": "中水", "AUV_4": "中水", "AUV_4dir": "中水",
         "AUV_2": "浊水", "AUV_3": "浊水", "AUV_5": "清水"}
_AOLD = {}


# ------------------------------------------------------------------ 标定与映射
def load_calib(path) -> tuple[np.ndarray, np.ndarray]:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("distortion_coefficients").mat()
    fs.release()
    return K, dist


def aold_maps():
    """A-old 链路的映射表（float32），用于 去畸变640 → raw 的反查"""
    if "m" not in _AOLD:
        K, dist = load_calib("configs/backup/front_camera_AUV1_water_fx782.yaml")
        nk, _ = cv2.getOptimalNewCameraMatrix(K, dist, (RAW_W, RAW_H), 0.0, (RAW_W, RAW_H))
        _AOLD["m"] = cv2.initUndistortRectifyMap(K, dist, None, nk, (RAW_W, RAW_H),
                                                cv2.CV_32FC1)
    return _AOLD["m"]


def bilinear(arr, x, y):
    """在映射表上做双线性采样（x,y 为浮点像素坐标）"""
    h, w = arr.shape
    x0 = np.clip(np.floor(x).astype(int), 0, w - 2)
    y0 = np.clip(np.floor(y).astype(int), 0, h - 2)
    fx = np.clip(x - x0, 0, 1)
    fy = np.clip(y - y0, 0, 1)
    return (arr[y0, x0] * (1 - fx) * (1 - fy) + arr[y0, x0 + 1] * fx * (1 - fy) +
            arr[y0 + 1, x0] * (1 - fx) * fy + arr[y0 + 1, x0 + 1] * fx * fy)


class Domain:
    """一个目标域的 点变换 与 图像链路"""

    def __init__(self, name: str, calib: str | None, enhance):
        """enhance: True/"on"=WB+CLAHE+gamma；"wb"=只 WB+gamma（无 CLAHE）；False/"off"=不做
        （板端 2026-09-23 起用 "wb"，配 LUT 白平衡实现）"""
        self.name = name
        self.enhance = enhance
        self.calib = calib
        if calib:
            self.K, self.dist = load_calib(calib)
            self.nk720, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist,
                                                          (RAW_W, RAW_H), 0.0, (RAW_W, RAW_H))
            self.nk640 = S @ self.nk720
            self.K640 = S @ self.K
            self.m720 = cv2.initUndistortRectifyMap(self.K, self.dist, None, self.nk720,
                                                    (RAW_W, RAW_H), cv2.CV_16SC2)
            self.m640 = cv2.initUndistortRectifyMap(self.K640, self.dist, None, self.nk640,
                                                    (SIZE, SIZE), cv2.CV_16SC2)

    def _enh(self, f):
        """按档位做画面补偿：'on' → WB+CLAHE+gamma；'wb' → 只 WB+gamma；'off' → 不做"""
        e = self.enhance
        if e in (False, "off", None):
            return f
        if e == "wb":
            return pf.enhance(f, WB, 0.0, GAMMA)     # clip=0 → 跳过 CLAHE
        return pf.enhance(f, WB, CLAHE, GAMMA)

    # --- 点：raw(1280x720 像素) → 本域 640 像素
    def raw_to_domain(self, pts: np.ndarray) -> np.ndarray:
        p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        sc = np.array([SIZE / RAW_W, SIZE / RAW_H])
        if self.name == "B":
            # raw 像素 → squish 到 640
            return p.reshape(-1, 2) * sc
        if self.name == "C":
            # raw → 720p 去畸变 → squish 到 640（末尾缩放只对 720p 结果成立）
            q = cv2.undistortPoints(p, self.K, self.dist, P=self.nk720).reshape(-1, 2)
            return q * sc
        if self.name == "D":
            # raw → squish 到 640（畸变域）→ 640 上去畸变；P=nk640 已直接给出 640 像素，
            # 绝不能再乘一次 sc（2026-09-23 踩过：标签整体缩一半，x 差 233 px）
            return cv2.undistortPoints(p * sc, self.K640, self.dist,
                                       P=self.nk640).reshape(-1, 2)
        raise ValueError(self.name)

    # --- 图像：raw → 本域 640 图
    def render(self, raw: np.ndarray) -> np.ndarray:
        if self.name == "C":
            f = cv2.remap(raw, *self.m720, cv2.INTER_LINEAR)
            f = cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
            f = self._enh(f)
        elif self.name == "B":
            f = cv2.resize(raw, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
            f = self._enh(f)
        elif self.name == "D":
            f = cv2.resize(raw, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
            f = self._enh(f)
            f = cv2.remap(f, *self.m640, cv2.INTER_LINEAR)
        else:
            raise ValueError(self.name)
        return f


_D640 = {}


def d640_maps():
    """D 域(640 上去畸变)的 float 映射表：dest(去畸变640) → src(畸变640)"""
    if "m" not in _D640:
        K, dist = load_calib("configs/front_camera.yaml")
        nk720, _ = cv2.getOptimalNewCameraMatrix(K, dist, (RAW_W, RAW_H), 0.0, (RAW_W, RAW_H))
        nk = S @ nk720
        _D640["m"] = cv2.initUndistortRectifyMap(S @ K, dist, None, nk, (SIZE, SIZE),
                                                 cv2.CV_32FC1)
    return _D640["m"]


def labeled_to_raw(pts640: np.ndarray, chain: str) -> np.ndarray:
    """标注图上的 640 像素 → 裸流像素。

    chain:  aold  = 标定A 去畸变（720p remap→resize）
            nound = 不去畸变（只 resize）           ← B 域标注
            d640  = 640 上去畸变（resize→remap@640） ← **D 域标注（板端新方案）**
    """
    p = np.asarray(pts640, np.float64)
    p720 = p * np.array([RAW_W / SIZE, RAW_H / SIZE])      # 640 → 720p（squish 逆）
    if chain.startswith("nound"):
        return p720
    if chain.startswith("d640"):
        m1, m2 = d640_maps()
        d = np.stack([bilinear(m1, p[:, 0], p[:, 1]),
                      bilinear(m2, p[:, 0], p[:, 1])], axis=1)
        return d * np.array([RAW_W / SIZE, RAW_H / SIZE])
    m1, m2 = aold_maps()
    return np.stack([bilinear(m1, p720[:, 0], p720[:, 1]),
                     bilinear(m2, p720[:, 0], p720[:, 1])], axis=1)


# ------------------------------------------------------------------ 取原始帧
_OFF = {}


def stream_offsets(path: str) -> list[int]:
    if path not in _OFF:
        with open(path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            off, pos = [], 0
            while (i := mm.find(SOI, pos)) >= 0:
                off.append(i)
                pos = i + 3
            mm.close()
        _OFF[path] = off
    return _OFF[path]


CACHE = PROJECT_ROOT / "data/mapped/raw"


def read_raw(src: str, idx: int):
    cp = CACHE / src / f"frame_{idx:06d}.jpg"
    if cp.exists():                       # 优先用落地缓存（materialize_raw.py）
        return cv2.imread(str(cp), cv2.IMREAD_COLOR)
    kind, path = SRC_PATH[src]
    if kind == "dir":
        p = Path(path) / f"frame_{idx:06d}.jpg"
        return cv2.imread(str(p), cv2.IMREAD_COLOR) if p.exists() else None
    if kind == "mjpeg":
        off = stream_offsets(path)
        if idx < 1 or idx > len(off):
            return None
        end = off[idx] if idx < len(off) else Path(path).stat().st_size
        with open(path, "rb") as f:
            f.seek(off[idx - 1])
            buf = np.frombuffer(f.read(end - off[idx - 1]), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    # video
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None


# ------------------------------------------------------------------ 标签读写
def read_pose_label(txt: Path):
    """YOLO pose 标签 → (kpts640 (4,2), bbox640 (x1,y1,x2,y2), v (4,))"""
    lines = [l.split() for l in txt.read_text().splitlines() if l.strip()]
    if not lines:
        return None
    f = [float(x) for x in lines[0]]
    cx, cy, w, h = f[1:5]
    k = np.array(f[5:17], np.float64).reshape(4, 3)
    bbox = np.array([(cx - w / 2) * SIZE, (cy - h / 2) * SIZE,
                     (cx + w / 2) * SIZE, (cy + h / 2) * SIZE])
    return k[:, :2] * SIZE, bbox, k[:, 2]


def write_pose_label(path: Path, kpts640: np.ndarray, bbox640: np.ndarray, vis: np.ndarray):
    x1, y1, x2, y2 = bbox640
    cx, cy = (x1 + x2) / 2 / SIZE, (y1 + y2) / 2 / SIZE
    w, h = (x2 - x1) / SIZE, (y2 - y1) / SIZE
    parts = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
    for (x, y), v in zip(kpts640, vis):
        parts += [f"{x/SIZE:.6f}", f"{y/SIZE:.6f}", str(int(v))]
    path.write_text(" ".join(parts) + "\n")


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prov", default="runs/prov/provenance_final_PNP.kpt4.yolov8.csv")
    ap.add_argument("--dataset", default="data/AUV_4/PNP.kpt4.yolov8")
    ap.add_argument("--domain", required=True, choices=["B", "C", "D"])
    ap.add_argument("--calibration", default="configs/front_camera.yaml")
    ap.add_argument("--enhance", choices=["on", "wb", "off"], default="on",
                    help="on=WB+CLAHE+gamma（旧）；wb=只 WB+gamma 无 CLAHE（板端新方案）；off=不做")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--selfcheck", action="store_true",
                    help="只做角点环回自检（标注→raw→同域），不写图")
    a = ap.parse_args()

    dom = Domain(a.domain, a.calibration if a.domain in ("C", "D") else None,
                 {"on": True, "wb": "wb", "off": False}[a.enhance])
    ds = PROJECT_ROOT / a.dataset
    rows = [r for r in csv.DictReader(open(PROJECT_ROOT / a.prov)) if r["accept"] == "1"]
    print(f"已命中 {len(rows)} 张，域={a.domain} enhance={a.enhance}")

    if a.selfcheck:
        # 环回自检：标注 --(自身链路的反变换)--> raw --(自身链路)--> 应回到原点
        d_aold = Domain("C", "configs/backup/front_camera_AUV1_water_fx782.yaml", True)
        d_nound = Domain("B", None, {"on": True, "wb": "wb", "off": False}[a.enhance])
        d_d640 = Domain("D", a.calibration, {"on": True, "wb": "wb", "off": False}[a.enhance])
        errs = []
        for r in rows:
            lab = read_pose_label(ds / r["split"] / "labels" / (Path(r["img"]).stem + ".txt"))
            if lab is None:
                continue
            raw_pts = labeled_to_raw(lab[0], r["chain"])
            if r["chain"].startswith("nound"):
                back = d_nound.raw_to_domain(raw_pts)
            elif r["chain"].startswith("d640"):
                back = d_d640.raw_to_domain(raw_pts)
            else:
                back = d_aold.raw_to_domain(raw_pts)
            errs.append(np.abs(back - lab[0]).max())
        e = np.array(errs)
        print(f"环回自检 n={len(e)}：中位 {np.median(e):.4f} px  "
              f"p90 {np.percentile(e,90):.4f}  最大 {e.max():.4f} px")
        return

    # 默认输出目录按增强档位命名，与生成时的约定一致（定稿生产集 = data/mapped/pose_D_wb）
    _suffix = {"on": "", "off": "_noenh", "wb": "_wb"}[a.enhance]
    out = a.out or PROJECT_ROOT / f"data/mapped/pose_{a.domain}{_suffix}"
    manifest, stats = [], dict(n_img=0, n_fail=0, n_out=0, out_kpt=0,
                               dive={}, water={})
    for r in rows:
        split = r["split"]
        stem = Path(r["img"]).stem
        txt = ds / split / "labels" / (stem + ".txt")
        if not txt.exists():
            continue
        lab = read_pose_label(txt)
        if lab is None:
            continue
        kpts640, bbox640, vis = lab
        raw = read_raw(r["src"], int(r["num_src"]))
        if raw is None or raw.shape[:2] != (RAW_H, RAW_W):
            stats["n_fail"] += 1
            continue
        # 角点：标注 → raw → 新域
        raw_pts = labeled_to_raw(kpts640, r["chain"])
        new_k = dom.raw_to_domain(raw_pts)
        raw_bbox = labeled_to_raw(np.array([[bbox640[0], bbox640[1]],
                                            [bbox640[2], bbox640[3]]]), r["chain"])
        nb = dom.raw_to_domain(raw_bbox)
        img = dom.render(raw)
        # bbox = 映射后原框 AABB ∪ 角点 AABB，再外扩 2%
        allp = np.vstack([nb, new_k])
        x1, y1 = allp.min(0)
        x2, y2 = allp.max(0)
        pad = 0.02 * max(x2 - x1, y2 - y1, 1.0)
        x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
        x1, y1 = max(x1, 0.0), max(y1, 0.0)
        x2, y2 = min(x2, SIZE - 1.0), min(y2, SIZE - 1.0)
        if x2 - x1 < 4 or y2 - y1 < 4:
            stats["n_fail"] += 1
            continue
        vis = np.asarray(vis, np.float64).copy()
        inside = (new_k[:, 0] >= 0) & (new_k[:, 0] < SIZE) & \
                 (new_k[:, 1] >= 0) & (new_k[:, 1] < SIZE)
        n_out = int((~inside).sum())
        stats["out_kpt"] += n_out
        if n_out:
            stats["n_out"] += 1
        vis[~inside] = 0
        new_k = np.clip(new_k, 0.0, SIZE - 0.01)
        h8 = hashlib.md5(r["img"].encode()).hexdigest()[:8]
        name = f"{re.sub(r'_jpg$', '', stem)}_{h8}"
        d = out / split
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "labels").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "images" / f"{name}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        write_pose_label(d / "labels" / f"{name}.txt", new_k,
                         np.array([x1, y1, x2, y2]), vis)
        rg = float(np.max(np.linalg.norm((new_k - SIZE / 2) / (SIZE / 2), axis=1)))
        manifest.append(dict(name=name, split=split, src=r["src"],
                             num_src=r["num_src"], chain=r["chain"],
                             num_label=r["num_label"], dive=r["src"].replace("dir", ""),
                             water=WATER.get(r["src"], "?"),
                             n_visible=int((vis > 0).sum()), rho=round(rg, 4),
                             kpt=" ".join(f"{v:.2f}" for v in new_k.ravel())))
        stats["n_img"] += 1
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)
    (out / "data.yaml").write_text(
        f"train: train/images\nval: valid/images\ntest: test/images\n"
        f"kpt_shape:\n- 4\n- 3\nflip_idx:\n- 0\n- 1\n- 2\n- 3\nnc: 1\nnames:\n- gate\n")
    meta = dict(domain=a.domain, enhance=a.enhance, calibration=a.calibration,
                prov=a.prov, dataset=a.dataset,
                n_img=stats["n_img"], n_fail=stats["n_fail"],
                frames_with_kpt_out_of_view=stats["n_out"],
                kpt_out_of_view_total=stats["out_kpt"])
    (out / "domain_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"写出 {stats['n_img']} 张（失败 {stats['n_fail']}）→ {out}")
    print(f"  角点出画：{stats['n_out']} 帧 / {stats['out_kpt']} 个点（v 置 0）")
    import collections
    print("  来源:", dict(collections.Counter(m["src"] for m in manifest)))
    print("  水况:", dict(collections.Counter(m["water"] for m in manifest)))
    print("  划分:", dict(collections.Counter(m["split"] for m in manifest)))


if __name__ == "__main__":
    main()
