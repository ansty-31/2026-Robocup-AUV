#!/usr/bin/env python3
"""select_gate_frames.py — 从预处理好的帧里挑"打标素材"：单门为主、多门少许、优先正对的门。

做法：用当前权重跑一遍 → 每帧取"板端会选中的那个门"（最近 = 框最大）算两个几何量
  angle  = 四角与 90° 的平均偏差（越小越正对）
  sym    = 对边长度不对称度（|上-下|/均值 + |左-右|/均值）
正门度 = angle + 20*sym（越小越正）；按正门度排序取前 N，并强制同视频内**原始帧间隔 ≥ gap**（去近似重复）。

用法：
  python scripts/1_prepare/select_gate_frames.py --dir data/derived/xxx_prep \
      --out data/derived/xxx_selected_750 --single 500 --multi 250 \
      --weights weights/new/yolo11n-pose.stage1_v3.pt --gap 15

产物：out/ 下硬链接（名字 <视频>__<帧>.jpg）+ out/manifest.csv + out/README.md
"""
from __future__ import annotations
import argparse, csv, os, pathlib, re
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]


def angle_dev(k: np.ndarray) -> float:
    """四角**内角**与 90° 的平均偏差（度）。k: (4,2) TL,TR,BR,BL（640 输入空间）。"""
    d = []
    for i in range(4):
        v1, v2 = k[(i - 1) % 4] - k[i], k[(i + 1) % 4] - k[i]
        c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        d.append(abs(np.degrees(np.arccos(np.clip(c, -1, 1))) - 90.0))
    return float(np.mean(d))


def edge_sym(k: np.ndarray) -> float:
    """对边长度不对称度（上/下、左/右）。"""
    top = np.linalg.norm(k[1] - k[0]); bot = np.linalg.norm(k[2] - k[3])
    lef = np.linalg.norm(k[3] - k[0]); rig = np.linalg.norm(k[2] - k[1])
    return float(abs(top - bot) / (0.5 * (top + bot) + 1e-9) + abs(lef - rig) / (0.5 * (lef + rig) + 1e-9))


def flat_blocks(path, bs: int = 16, thr: float = 0.5) -> float:
    """完全平坦块(16×16, std<thr)占比：源 MJPEG 残帧会出现大片灰块，正常帧≈0。"""
    import cv2
    g = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape
    h -= h % bs; w -= w % bs
    b = g[:h, :w].reshape(h // bs, bs, w // bs, bs).transpose(0, 2, 1, 3).reshape(-1, bs * bs)
    return float((b.std(axis=1) < thr).mean())


def frame_idx(name: str) -> int:
    m = re.search(r"(\d+)(?=\.[A-Za-z]+$)", name)
    return int(m.group(1)) if m else -1


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dir", required=True, help="预处理帧目录（可含视频子目录）")
    ap.add_argument("--out", required=True, help="输出批次目录")
    ap.add_argument("--single", type=int, default=500, help="单门（n_det==1）要多少张")
    ap.add_argument("--multi", type=int, default=250, help="多门（n_det>=2）要多少张")
    ap.add_argument("--gap", type=int, default=15, help="同视频内最小原始帧间隔（去近似重复）")
    ap.add_argument("--max-flat", type=float, default=0.08, help="平坦块占比上限，超过判为源残帧并剔除")
    ap.add_argument("--stats", default=None, help="每帧统计 CSV：存在则直接读入，省一次推理")
    ap.add_argument("--weights", default="weights/new/yolo11n-pose.stage1_v3.pt")
    ap.add_argument("--conf", type=float, default=0.6)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    src = pathlib.Path(a.dir) if pathlib.Path(a.dir).is_absolute() else REPO / a.dir
    out = pathlib.Path(a.out) if pathlib.Path(a.out).is_absolute() else REPO / a.out
    files = sorted(src.rglob("*.jpg"))
    if not files:
        raise SystemExit(f"❌ {src} 里没有图片")

    stats = pathlib.Path(a.stats) if a.stats else out.with_suffix(".stats.csv")
    if stats.exists():
        I, F = ("idx", "n_det", "n_hi"), ("score", "angle", "sym", "conf", "area")
        rows = [{k: (int(v) if k in I else float(v) if k in F else v)
                 for k, v in r.items()} for r in csv.DictReader(open(stats, encoding="utf-8"))]
        print(f"复用统计 {stats}（{len(rows)} 帧）")
        return pick(rows, a, src, out)

    from ultralytics import YOLO
    lst = pathlib.Path("/tmp/_select_gate_list.txt")
    lst.write_text("\n".join(str(p.absolute()) for p in files), encoding="utf-8")
    model = YOLO(a.weights)
    rows = []
    n_bad = 0
    for res in model.predict(source=str(lst), imgsz=a.imgsz, conf=a.conf, device=a.device,
                             batch=a.batch, max_det=10, stream=True, verbose=False):
        p = pathlib.Path(res.path)
        video = p.parent.name
        if flat_blocks(p) > a.max_flat:               # 源 MJPEG 残帧：不入候选
            n_bad += 1
            rows.append(dict(video=video, name=p.name, idx=frame_idx(p.name), n_det=-1,
                             score=1e9, angle=0.0, sym=0.0, conf=0.0, n_hi=0, area=0.0))
            continue
        k = 0 if res.boxes is None else len(res.boxes)
        if not k or res.keypoints is None or len(res.keypoints) != k:
            rows.append(dict(video=video, name=p.name, idx=frame_idx(p.name), n_det=k,
                             score=1e9, angle=0.0, sym=0.0, conf=0.0, n_hi=0, area=0.0))
            continue
        kd = res.keypoints.data.cpu().numpy(); bx = res.boxes.xyxy.cpu().numpy(); cf = res.boxes.conf.cpu().numpy()
        areas = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])
        i = int(np.argmax(areas))                     # 板端口径：选最近（框最大）的那个
        v = kd[i, :, 2]
        ang, sym = angle_dev(kd[i, :, :2]), edge_sym(kd[i, :, :2])
        rows.append(dict(video=video, name=p.name, idx=frame_idx(p.name), n_det=k,
                         score=ang + 20.0 * sym, angle=round(ang, 2), sym=round(sym, 4),
                         conf=round(float(cf[i]), 3), n_hi=int((v >= 0.5).sum()), area=float(areas[i])))
    print(f"剔除源残帧 {n_bad} 张（平坦块 > {a.max_flat}）")
    print(f"跑完 {len(rows)} 帧：单门 {sum(1 for r in rows if r['n_det'] == 1)}、"
          f"多门 {sum(1 for r in rows if r['n_det'] >= 2)}、无检出 {sum(1 for r in rows if r['n_det'] == 0)}")
    with open(stats, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"统计已存 {stats}")
    return pick(rows, a, src, out)


def pick(rows, a, src, out):
    picked, used = [], {v: [] for v in {r["video"] for r in rows}}
    for want, tag in ((a.single, "single"), (a.multi, "multi")):
        pool = [r for r in rows if (r["n_det"] == 1 if tag == "single" else r["n_det"] >= 2)]
        pool.sort(key=lambda r: r["score"])            # 正门度优先
        n = 0
        for r in pool:
            if n >= want:
                break
            if any(abs(r["idx"] - j) < a.gap for j in used[r["video"]]):
                continue
            r["cls"] = tag; used[r["video"]].append(r["idx"]); picked.append(r); n += 1
        print(f"{tag}: 目标 {want}，实取 {n}（候选 {len(pool)}）")

    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.jpg"):
        f.unlink()
    for r in picked:
        srcp = src / r["video"] / r["name"] if (src / r["video"]).exists() else src / r["name"]
        dst = out / f"{r['video']}__{r['name']}"
        if not dst.exists():
            os.link(srcp, dst)
        r["file"] = dst.name
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "video", "name", "idx", "cls", "n_det", "score",
                                           "angle", "sym", "conf", "n_hi", "area"])
        w.writeheader()
        for r in sorted(picked, key=lambda r: (r["cls"], r["video"], r["idx"])):
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    per = {}
    for r in picked:
        per.setdefault(r["video"], [0, 0])
        per[r["video"]][0 if r["cls"] == "single" else 1] += 1
    print(f"✅ {out}：共 {len(picked)} 张")
    for v, (s, m) in sorted(per.items()):
        print(f"   {v:<28} 单门 {s:>4}  多门 {m:>4}")


if __name__ == "__main__":
    main()
