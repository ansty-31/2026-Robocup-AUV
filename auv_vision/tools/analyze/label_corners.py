# -*- coding: utf-8 -*-
"""tools/analyze/label_corners.py — 手工标注门框 4 角 → 与 `preview_detect --dump` **同格式**的 JSONL

为什么需要它
------------
PnP / 深度标定要的是「**像素角点 + 卷尺真值**」，检测器只是自动产出角点的手段。
当现场环境让 keypoint 权重不响应时（2026-09-22 实测：岸上工作间空气域最高分 **0.074**、
角点全 0；同一权重在水下旧录制帧上是 0.87~0.94），仍然可以在岸上把几何标定做完 —— 手工点 4 个角。

坐标域（**最容易错的一点**）
----------------------------
标注坐标必须与 PnP 域一致。`cfg/vision.yaml → image.undistort: true`（当前）⇒
PnP 域 = **去畸变域**（`gate.gate_detector.board_camera()` 用 rectified K）。所以：

  · 用 `--capture` 采的图**已经过去畸变**（本工具按同一套标定做 remap，与运行时同源）；
  · 用 `--images` 指自己拍的**原图**时，要加 `--undistort` 让工具先 remap 再显示/标注。

域弄错 ⇒ 深度整体缩放错（现象像"标尺不对"，很容易误判成门宽标错）。

用法
----
① 采图（每档距离多采几帧；图里**不要**有叠加框，所以别用 `preview_detect --save`）：
      python3 tools/analyze/label_corners.py --capture log/pnp_0922/z150 --n 8
② 标注（顺序必须是 **TL → TR → BR → BL**，与 `gate.geometry.object_points` 同序）：
      python3 tools/analyze/label_corners.py --images 'log/pnp_0922/z150/*.jpg' \\
              --out log/pnp_0922/pnp_z150.jsonl
   键盘：左键落点 ｜ u 撤销 ｜ r 重来 ｜ s 保存并下一张 ｜ n/空格 跳过 ｜ q 退出(保存已标)
          m 放大镜开关 ｜ + / - 放大镜倍率
          
   ⚠️ **放大镜默认关闭**（2026-09-23 改）：它画在右上角、边长 `140*zoom`，默认 zoom=4 时
   是 560px —— 在 640 的画面上会盖掉约 87%，把图挡死。现在按 `m` 才显示，且以 0.65
   不透明度叠加（底下的画面仍可见）。`--loupe` 可让它启动即开。
③ 标完喂标定工具（文件名里的真值照旧生效）：
      python3 tools/analyze/pnp_calib.py --report log/pnp_report.md log/pnp_0922/pnp_z*.jsonl

离线/脚本模式（不起窗口，便于批量与用例）：`--coords "x,y;x,y;x,y;x,y"`

**点在哪一条线上（2026-09-22 补，直接决定标定口径）**
点**红管的中线**，不是管子的外沿/内沿。理由：门框外缘 0.77×0.56 m 对应的是"管子外轮廓的角"，
而 `snap_to_red` 只在 **±6 px** 邻域里找最红的像素 —— 近距档管子宽（2 m 处约 28 px），
窗口完全落在管内，红度平台一片 ⇒ 吸附**不会**把点拉到管中线，只保证"点落在管子上"。
所以冷启动那一刻的点击位置就是最终答案：必要时按 `m` 开放大镜把十字压在中线上，
四角都偏"同一侧"会变成一个 3% 量级的**系统性宽度偏差**（正对时 = 3% 深度偏差）。
判据：标完看 `Δu/Δv`，应落在 0.77/0.56 = **1.375** 附近（正对档）；明显偏离就先查是不是点偏了。

**也可以拿来量卷尺刻度（§A0b 内参靶子法）**：卷尺刻度没有红色 ⇒ 吸附不生效，
点 0 与 100 cm 两个刻度，读控制台打印的 `TL = (x, y)` / `TR = (x, y)`，取 `Δu = |x₂−x₁|`，
然后按 **q** 退出（**不要**存 jsonl —— 那不是门角点，喂给 `pnp_calib` 只会得到垃圾位姿）。

⚠️ 标注点置信度一律给 **1.0**（这是「干净标注」，不是检测结果）——
比检测结果更可信，正是标定 PnP 想要的输入。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np                                              # noqa: E402

import base.settings as S                                       # noqa: E402

KPT_NAMES = ("TL", "TR", "BR", "BL")           # 与 object_points 同序
WIN = "label_corners"


# ------------------------------------------------------------------ 坐标域
def _calib_path():
    """与运行时同源的标定 yaml 路径（缺失则返回 None）。"""
    p = S.get("vision.camera.front.calibration", None)
    for c in (p, os.path.join(_ROOT, "cfg", os.path.basename(p or ""))):
        if c and os.path.exists(c):
            return c
    return None


def rectify(frame, calib=None):
    """按 cfg（`vision.image.undistort`）把原图 remap 到 PnP 域；不需要则原样返回。"""
    if not bool(S.get("vision.image.undistort", False)):
        return frame
    path = calib or _calib_path()
    if not path:
        print("⚠️ undistort=true 但找不到标定文件 → 按原图处理（坐标域会与 PnP 不一致！）")
        return frame
    import cv2
    from common.preprocess import calibration_maps
    h, w = frame.shape[:2]
    maps = calibration_maps(path, w, h)
    return cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)


# ------------------------------------------------------------------ 纯函数（可测）
def snap_to_red(img, x, y, r=6, min_gain=25.0):
    """把点击吸到附近「最红」的像素，返回 (x, y)。

    红度 = R − max(G, B)（BGR 输入）。邻域内最大红度比点击处高出 `min_gain` 才吸附，
    否则保持原点击坐标（避免在非红区域乱跳）。

    ⚠️ 它**只保证"点落在管子上"，不保证落在管中线上**（2026-09-22 修正了旧注释里
    "误差从 ±5px 降到 ±1px" 的过度承诺）：搜索半径只有 `r=6` px，而近距档红管宽约 28 px
    ⇒ 窗口整片落在管内、红度是平台，去平台后仍然由**点击位置**决定。所以：
    想标准，自己把十字压到管中线（配合放大镜）。四角都偏同一侧 = 系统性宽度偏差。
    """
    h, w = img.shape[:2]
    x0, x1 = max(0, int(x) - r), min(w, int(x) + r + 1)
    y0, y1 = max(0, int(y) - r), min(h, int(y) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return float(x), float(y)
    patch = img[y0:y1, x0:x1].astype(np.float32)
    red = patch[:, :, 2] - np.maximum(patch[:, :, 1], patch[:, :, 0])
    rmax = float(red.max())
    cx, cy = int(x) - x0, int(y) - y0
    gain = rmax - float(red[cy, cx])
    if rmax <= 0 or gain < min_gain:
        return float(x), float(y)
    # ⚠️ 红管是一条"脊"：同红度的像素成排出现，直接 argmax 会固定吸到**窗口左上角**那一个
    #    （方向性偏差）。所以取"红度接近最大"的那一批，再选**离点击最近**的 —— 无偏。
    yy, xx = np.nonzero(red >= rmax - 2.0)
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    k = int(np.argmin(d2))
    return float(x0 + xx[k]), float(y0 + yy[k])


def bbox_of(kpts):
    a = np.asarray(kpts, np.float64)
    x0, y0 = float(a[:, 0].min()), float(a[:, 1].min())
    x1, y1 = float(a[:, 0].max()), float(a[:, 1].max())
    return [int(round(x0)), int(round(y0)),
            int(max(1, round(x1 - x0))), int(max(1, round(y1 - y0)))]


def make_record(idx, kpts, src=None, t=None, score=1.0):
    """生成**与 `preview_detect --dump` 同 schema** 的一行记录（额外字段对下游无害）。"""
    kpts = [[round(float(a), 2), round(float(b), 2)] for a, b in kpts]
    return {"t": (round(float(t), 3) if t is not None else 0.0),
            "frame": int(idx), "n_det": 1, "fps": None,
            "dets": [{"kind": "gate", "score": float(score), "bbox": bbox_of(kpts),
                      "kpts": kpts, "kpt_conf": [1.0] * 4}],
            "src": os.path.basename(src) if src else None,
            "src_path": (src or None),
            "label": "manual"}


# ------------------------------------------------------- 卷尺靶子（§A0b/§B4，纯函数可测）
def span_stats(pts):
    """3 个**刻度点**（依次：起点刻度 / 中间刻度 / 终点刻度）→ 跨度统计。

    **任意方向都成立**（横着、竖着、斜着量都行）—— 这是 2026-09-22 现场踩出来的：
      · 靶面只要**与光轴垂直**，`z=const` 平面到图像就是**均匀缩放** `f/z`（小孔模型下严格成立），
        所以两端点的**欧氏像素距离** `du = |P₃−P₁| = (f/z)·L`，与靶线在画面里怎么转无关
        —— 用坐标轴投影会白丢一个 `cosθ`（竖着量更是直接得 0）。
      · 三个刻度点等距 ⇒ `P₂` 落在 `P₁P₃` 的**中点**。
    `skew = (后半 − 前半)/du`：靶面绕竖轴/横轴偏了就会偏（近侧半段更长）。
    `off_mid`：中点离 `P₁P₃` 连线的**垂距(px)** —— 点错刻度/卷尺有折角时会变大。

    返回 dict：`du`(px, 欧氏)、`du_l/du_r`(px)、`skew`、`mid_frac`(应 ≈0.5)、
    `off_mid`(px)、`theta_deg`（靶线在画面里的倾角，0=水平 ⇒ 量到的是 fx）。
    """
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if p.shape[0] != 3:
        raise ValueError("span_stats 需要 3 个点（起/中/终刻度），收到 %d" % p.shape[0])
    d = p[2] - p[0]
    du = float(np.hypot(d[0], d[1]))
    if du < 1e-6:
        raise ValueError("起点与终点刻度重合（du≈0）—— 是不是点重了？")
    u = d / du
    rel = p[1] - p[0]
    t_mid = float(rel[0] * u[0] + rel[1] * u[1])          # 中点沿靶线的投影长度
    off_mid = float(abs(rel[0] * u[1] - rel[1] * u[0]))   # 到靶线的垂距
    du_l, du_r = t_mid, du - t_mid
    return {"x1": float(p[0, 0]), "x2": float(p[1, 0]), "x3": float(p[2, 0]),
            "v": [float(p[0, 1]), float(p[1, 1]), float(p[2, 1])],
            "du_l": du_l, "du_r": du_r, "du": du,
            "skew": (du_r - du_l) / du,
            "mid_frac": t_mid / du,
            "off_mid": off_mid,
            "theta_deg": float(np.degrees(np.arctan2(d[1], d[0])))}


def make_ruler_record(src, pts, z_tape=None, span_m=1.0, ticks=None):
    """一行「靶子读数」记录（**不是**门角点 schema，别喂给 `pnp_calib`）。"""
    st = span_stats(pts)
    rec = {"src": os.path.basename(src) if src else None,
           "src_path": (src or None), "ticks": ticks,
           "kind": "ruler", "span_m": float(span_m), "z_tape": z_tape,
           "p": [[round(float(a), 2), round(float(b), 2)] for a, b in
                 np.asarray(pts, np.float64).reshape(-1, 2)]}
    rec.update({k: round(st[k], 3) for k in
                ("du_l", "du_r", "du", "skew", "mid_frac", "off_mid", "theta_deg")})
    rec["x1"], rec["x2"], rec["x3"] = (round(st["x1"], 2), round(st["x2"], 2), round(st["x3"], 2))
    return rec


def _z_from_name(path):
    """从图名/档名里抠 z 真值（`...z150...` → 1.50 m）。

    ⚠️ 靶子图长这样：`log/pnp_0922/ruler_z200/cap_001.jpg` —— 真值在**目录名**上，
    所以只看 `basename` 会漏掉（2026-09-22 踩过）。这里按"最后两段路径"扫：
    既覆盖 `pnp_z150.jsonl`，也覆盖 `ruler_z200/cap_001.jpg`。
    """
    import re
    parts = [p for p in str(path or "").replace("\\", "/").split("/") if p]
    tail = "/".join(parts[-2:]) if len(parts) >= 2 else (parts[0] if parts else "")
    m = re.search(r"z(\d{2,4})", tail)
    return (int(m.group(1)) / 100.0) if m else None


def load_done(out_path):
    """已测/已标的**去重键**集合（用于 `--resume`）。

    ⚠️ 2026-09-22 踩过：早期版本按 `basename` 去重，而采图目录里的文件名永远是
    `cap_001.jpg`… ⇒ **不同距离档之间互相覆盖**（`ruler_z100` 的记录被 `ruler_z300`
    整行删掉，`--resume` 更会直接跳过整档）。现在优先用 `src_path`（完整路径），
    只有老记录（没有 `src_path`）才退回 basename。
    """
    paths, legacy = set(), set()
    if not out_path or not os.path.exists(out_path):
        return {"paths": paths, "legacy": legacy}
    with open(out_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("src_path"):
                paths.add(rec["src_path"])
            elif rec.get("src"):
                legacy.add(rec["src"])
            else:
                legacy.add("#%s" % rec.get("frame"))
    return {"paths": paths, "legacy": legacy}


def is_done(done, key, src=None):
    """这条图是否已经处理过（`--resume` 用）。`key` = 完整路径。"""
    if key in done["paths"]:
        return True
    return bool(src) and src in done["legacy"] and key not in done["paths"]


def remove_src(out_path, src):
    """删掉某个图的旧记录（重标/重测时用，避免同一张图出现两行）。

    `src` 传**完整路径**优先按 `src_path` 匹配；老记录（只有 basename）按名字匹配。
    """
    if not out_path or not os.path.exists(out_path) or not src:
        return 0
    base = os.path.basename(src)
    keep, gone = [], 0
    with open(out_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            key = rec.get("src_path") or rec.get("src")
            if key == src or (rec.get("src_path") is None and rec.get("src") == base):
                gone += 1
                continue
            keep.append(line if line.endswith("\n") else line + "\n")
    if gone:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.writelines(keep)
    return gone


def append_records(out_path, records):
    with open(out_path, "a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_coords(s):
    pts = []
    for part in s.split(";"):
        if not part.strip():
            continue
        a, b = part.split(",")
        pts.append((float(a), float(b)))
    if len(pts) != 4:
        raise SystemExit("--coords 需要 4 组 'x,y'，用分号分隔（顺序 TL;TR;BR;BL）")
    return pts


# ------------------------------------------------------------------ 采图
def capture(outdir, n=8, warmup=5, every=1):
    """从相机抓 n 张（去畸变后）存成 jpg：**不加任何叠加**，专供手工标注。"""
    import cv2
    from base.camera import create_camera
    os.makedirs(outdir, exist_ok=True)
    cam = create_camera("front")
    for _ in range(max(0, warmup)):               # 让自动曝光稳定
        cam.read()
    got = 0
    for i in range(n * every + 40):
        if got >= n:
            break
        f = cam.read()
        if f is None:
            continue
        if i % every:
            continue
        g = rectify(f)
        p = os.path.join(outdir, "cap_%03d.jpg" % (got + 1))
        cv2.imwrite(p, g, [cv2.IMWRITE_JPEG_QUALITY, 95])
        got += 1
    print("采图完成：%d 张 → %s（已按 cfg 去畸变：%s）"
          % (got, outdir, bool(S.get("vision.image.undistort", False))))
    return got


# ------------------------------------------------------------------ 标注（GUI）
def label(paths, out_path, do_undistort=False, snap=True, resume=False, zoom=4,
          loupe=False):
    import cv2
    done = load_done(out_path)          # 去重键 = 完整路径（basename 会跨目录撞车）
    n_ok = 0
    print("\n【键盘】→ / 回车 / s = 保存并下一张 ｜ n / 空格 = 跳过这张 ｜ ← = 回上一张(重标)"
          "\n         u = 撤销刚点的角 ｜ r = 本张重来 ｜ +/- = 放大镜倍率 ｜ q / ESC = 退出(保留已标)"
          "\n        ⚠️ 先用鼠标点一下窗口让它聚焦，否则按键不生效"
          "\n        ⚠️ 四角都点在**红管的中线**上（别点管沿）—— 吸附只在 ±6px 内找最红点，"
          "近距管子比窗口宽，吸附不会帮你居中（详见文件头）\n")
    i = 0
    try:
        while i < len(paths):
            p = paths[i]
            name = os.path.basename(p)
            if resume and is_done(done, p, name):
                print("跳过（已标）：%s" % p)
                i += 1
                continue
            print("[%d/%d] %s" % (i + 1, len(paths), name))
            img = cv2.imread(p)
            if img is None:
                print("读不到：%s" % p)
                continue
            if do_undistort:
                img = rectify(img)
            pts, cur = [], [0, 0]
            mag = [zoom]
            show_mag = [bool(loupe)]      # 放大镜默认【关】：默认倍率 4 时它是 (70*2*4)=560px
                                          # 的正方形，在 640 画面上会盖掉 87%，把图挡死。
            action = "next"

            def on_mouse(ev, x, y, flags, param):
                if ev == cv2.EVENT_MOUSEMOVE:
                    cur[0], cur[1] = x, y
                elif ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
                    px, py = (snap_to_red(img, x, y) if snap else (x, y))
                    pts.append((px, py))
                    print("  %s = (%.0f, %.0f)%s"
                          % (KPT_NAMES[len(pts) - 1], px, py,
                             "（已吸附到红管）" if (px, py) != (x, y) else ""))

            cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WIN, on_mouse)
            while True:
                disp = img.copy()
                if len(pts) >= 2:
                    cv2.polylines(disp, [np.array(pts, np.int32).reshape(-1, 1, 2)],
                                  len(pts) == 4, (60, 255, 60), 2)
                for k, (px, py) in enumerate(pts):
                    cv2.circle(disp, (int(px), int(py)), 5, (60, 255, 60), -1)
                    cv2.putText(disp, KPT_NAMES[k], (int(px) + 8, int(py) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 255, 60), 2)
                # 放大镜（跟手）：**默认关**（按 m 开）。开启时以 0.65 不透明度叠加，
                # 仍能看见底下的画面；默认倍率 4 下它 560px，纯不透明会盖掉 87% 的图。
                m = int(mag[0]); r = 70
                if show_mag[0]:
                    h, w = img.shape[:2]
                    x0, y0 = max(0, cur[0] - r), max(0, cur[1] - r)
                    x1, y1 = min(w, cur[0] + r), min(h, cur[1] + r)
                    if x1 > x0 and y1 > y0:
                        patch = cv2.resize(img[y0:y1, x0:x1], None, fx=m, fy=m,
                                           interpolation=cv2.INTER_NEAREST)
                        lh, lw = patch.shape[:2]
                        if lh < disp.shape[0] and lw < disp.shape[1]:
                            roi = disp[0:lh, disp.shape[1] - lw:disp.shape[1]]
                            cv2.addWeighted(roi, 0.35, patch, 0.65, 0, roi)
                            cv2.rectangle(disp, (disp.shape[1] - lw, 0),
                                          (disp.shape[1] - 1, lh - 1), (0, 255, 255), 1)
                # ⚠️ cv2.putText 的 Hershey 字体**只画 ASCII**，中文会变成一串方块 —— 所以
                #    屏上提示一律英文（中文提示走控制台，见下面的 print）。
                tip = ("[%d/%d] %s | pts %d/4  order TL,TR,BR,BL  click the TUBE CENTERLINE | "
                       "Enter/s=save  n/space=skip  Left=prev  u=undo  r=redo  "
                       "m=loupe:%s  +/-=zoom %dx  q=quit"
                       % (i + 1, len(paths), name, len(pts),
                          "on" if show_mag[0] else "off", m))
                cv2.putText(disp, tip, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                            (0, 255, 255), 2)
                cv2.imshow(WIN, disp)
                k = cv2.waitKeyEx(20)
                KEY_ESC, KEY_ENTER, KEY_RIGHT, KEY_LEFT = 27, 13, 65363, 65361
                if k in (KEY_ESC,) or (k & 0xFF) == ord("q"):
                    cv2.destroyAllWindows()
                    print("退出标注：本次保存 %d 张" % n_ok)
                    return n_ok
                if (k & 0xFF) == ord("u") and pts:
                    pts.pop()
                elif (k & 0xFF) == ord("r"):
                    pts = []
                elif (k & 0xFF) in (ord("n"), ord(" ")):
                    print("  ↷ 跳过 %s" % name)
                    action = "next"
                    break
                elif k in (KEY_ENTER, KEY_RIGHT) or (k & 0xFF) == ord("s"):
                    if len(pts) == 4:
                        gone = remove_src(out_path, p)         # 重标同一张：先删旧行（按路径）
                        append_records(out_path, [make_record(n_ok + 1, pts, src=p)])
                        n_ok += 1
                        print("  ✓ 已保存 %s（4 角完整）%s"
                              % (name, "（替换了旧记录）" if gone else ""))
                        action = "next"
                        break
                    print("  ⚠️ 只点了 %d 个角，还不能保存 —— 4 个都点完再按 → / 回车 / s"
                          % len(pts))
                elif k == KEY_LEFT:
                    if i > 0:
                        print("  ← 回上一张（旧记录会在保存时被替换）")
                        action = "prev"
                        break
                    print("  （已经是第一张）")
                elif (k & 0xFF) == ord("m"):
                    show_mag[0] = not show_mag[0]
                    print("  放大镜：%s（倍率 %dx）" % ("开" if show_mag[0] else "关", int(mag[0])))
                elif (k & 0xFF) in (ord("+"), ord("=")):
                    mag[0] = min(8, mag[0] + 1)
                elif (k & 0xFF) in (ord("-"), ord("_")):
                    mag[0] = max(2, mag[0] - 1)
            i += -1 if action == "prev" else 1
        cv2.destroyAllWindows()
    except Exception as e:                        # 无 GUI 环境（ssh -X 没开等）
        print("✗ 标注窗口不可用（%s）" % e)
        print("  → 在没有显示器的环境请在板端桌面里跑，或用 --coords 离线模式")
        return n_ok
    print("标注完成：本次保存 %d 张 → %s" % (n_ok, out_path))
    return n_ok


# ------------------------------------------------- 卷尺靶子测量（§A0b / §B4）
def measure(paths, out_path, do_undistort=False, resume=False, zoom=4, span_m=1.0,
            ticks=None, loupe=False):
    """点 3 个刻度（0 / 50 / 100 cm）→ 自动算 `Δu`/半跨/斜视诊断 → 追加一行 JSONL。

    与 `label()` 同一套坐标域处理（cfg 的 `undistort` 决定），所以量出来的 `fx` 与 PnP 同域。
    记录 schema 见 `make_ruler_record`（`kind: "ruler"`，**别**喂给 `pnp_calib`）。
    """
    import cv2
    done = load_done(out_path)          # 去重键 = 完整路径（basename 会跨目录撞车）
    ticks_tag = ticks
    n_ok = 0
    print("\n【卷尺靶子模式】左键依次点 **起点刻度 / 中间刻度 / 终点刻度**（例如 10 / 50 / 90 cm）"
          "\n  跨度 L=%.2f m；z 真值从文件名解析（如 ruler_z200 → 2.00 m）"
          "\n  ⚠️ 三个点要点在同一条刻度线上（同一水平高度最好），放大镜 +/- 调倍率"
          "\n  按键：→/回车/s = 保存并下一张 ｜ n/空格 = 跳过 ｜ ← = 上一张 ｜ u = 撤销 ｜ q/ESC = 退出\n"
          % span_m)
    i = 0
    try:
        while i < len(paths):
            p = paths[i]
            name = os.path.basename(p)
            if resume and is_done(done, p, name):
                print("跳过（已测）：%s" % p)
                i += 1
                continue
            print("[%d/%d] %s   z_tape=%s m" % (i + 1, len(paths), name, _z_from_name(p)))
            img = cv2.imread(p)
            if img is None:
                print("  读不到：%s" % p)
                i += 1
                continue
            if do_undistort or bool(S.get("vision.image.undistort", False)):
                img = rectify(img)
            pts, cur = [], [0, 0]
            mag = [zoom]
            show_mag = [bool(loupe)]      # 同 label()：默认关，按 m 开
            action = "next"

            def on_mouse(ev, x, y, flags, param):
                if ev == cv2.EVENT_MOUSEMOVE:
                    cur[0], cur[1] = x, y
                elif ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 3:
                    pts.append((x, y))
                    if len(pts) == 3:
                        st = span_stats(pts)
                        flag = []
                        if abs(st["skew"]) > 0.02:
                            flag.append("⚠️ 靶面没正对（skew 超 2%）：转船/转靶重拍")
                        if st["off_mid"] > 3.0:
                            flag.append("⚠️ 中点偏出靶线 %.1f px：是不是点错了刻度？" % st["off_mid"])
                        if 10.0 < abs(st["theta_deg"]) < 80.0:
                            flag.append("⚠️ 靶线斜着（%.0f°）：量到的是 fx/fy 的混合，尽量横平或竖直"
                                        % st["theta_deg"])
                        print("  Δu = %.1f px（前半 %.1f / 后半 %.1f，占 %.1f%%）  skew = %+.2f%%  %s"
                              % (st["du"], st["du_l"], st["du_r"], st["mid_frac"] * 100.0,
                                 st["skew"] * 100.0, "  ".join(flag) if flag else "✓"))
                    else:
                        print("  P%d = (%.0f, %.0f)" % (len(pts), x, y))

            cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WIN, on_mouse)
            while True:
                disp = img.copy()
                for k, (px, py) in enumerate(pts):
                    cv2.circle(disp, (int(px), int(py)), 5, (60, 255, 60), -1)
                    cv2.putText(disp, ("L", "M", "R")[k], (int(px) + 8, int(py) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 255, 60), 2)
                if len(pts) >= 2:
                    cv2.polylines(disp, [np.array(pts, np.int32).reshape(-1, 1, 2)],
                                  len(pts) == 3, (60, 255, 60), 2)
                m = int(mag[0])
                r, h, w = 70, img.shape[0], img.shape[1]
                if show_mag[0]:
                    x0, y0 = max(0, cur[0] - r), max(0, cur[1] - r)
                    x1, y1 = min(w, cur[0] + r), min(h, cur[1] + r)
                    if x1 > x0 and y1 > y0:
                        patch = cv2.resize(img[y0:y1, x0:x1], None, fx=m, fy=m,
                                           interpolation=cv2.INTER_NEAREST)
                        lh, lw = patch.shape[:2]
                        if lh < disp.shape[0] and lw < disp.shape[1]:
                            roi = disp[0:lh, disp.shape[1] - lw:disp.shape[1]]
                            cv2.addWeighted(roi, 0.35, patch, 0.65, 0, roi)
                            cv2.rectangle(disp, (disp.shape[1] - lw, 0),
                                          (disp.shape[1] - 1, lh - 1), (0, 255, 255), 1)
                tip = ("[%d/%d] %s | pts %d/3  click tape ticks L,M,R | "
                       "Enter/s=save  n/space=skip  Left=prev  u=undo  "
                       "m=loupe:%s  +/-=zoom %dx  q=quit"
                       % (i + 1, len(paths), name, len(pts),
                          "on" if show_mag[0] else "off", m))
                cv2.putText(disp, tip, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                            (0, 255, 255), 2)
                cv2.imshow(WIN, disp)
                k = cv2.waitKeyEx(20)
                KEY_ESC, KEY_ENTER, KEY_RIGHT, KEY_LEFT = 27, 13, 65363, 65361
                if k in (KEY_ESC,) or (k & 0xFF) == ord("q"):
                    cv2.destroyAllWindows()
                    print("退出测量：本次保存 %d 张" % n_ok)
                    return n_ok
                if (k & 0xFF) == ord("u"):
                    if pts:
                        pts.pop()
                elif (k & 0xFF) == ord("r"):
                    pts = []
                elif (k & 0xFF) in (ord("n"), ord(" ")):
                    print("  ↷ 跳过 %s" % name)
                    action = "next"
                    break
                elif k in (KEY_ENTER, KEY_RIGHT) or (k & 0xFF) == ord("s"):
                    if len(pts) == 3:
                        gone = remove_src(out_path, p)
                        append_records(out_path, [make_ruler_record(
                            p, pts, z_tape=_z_from_name(p), span_m=span_m, ticks=ticks_tag)])
                        n_ok += 1
                        print("  ✓ 已保存 %s（span=%.2f m, ticks=%s）%s"
                              % (name, span_m, ticks_tag or "未给",
                                 "（替换旧记录）" if gone else ""))
                        action = "next"
                        break
                    print("  ⚠️ 只点了 %d/3 个刻度，还不能保存" % len(pts))
                elif k == KEY_LEFT:
                    if i > 0:
                        action = "prev"
                        break
                    print("  （已经是第一张）")
                elif (k & 0xFF) == ord("m"):
                    show_mag[0] = not show_mag[0]
                    print("  放大镜：%s（倍率 %dx）" % ("开" if show_mag[0] else "关", int(mag[0])))
                elif (k & 0xFF) in (ord("+"), ord("=")):
                    mag[0] = min(8, mag[0] + 1)
                elif (k & 0xFF) in (ord("-"), ord("_")):
                    mag[0] = max(2, mag[0] - 1)
            i += -1 if action == "prev" else 1
        cv2.destroyAllWindows()
    except Exception as e:
        print("✗ 测量窗口不可用（%s）" % e)
        return n_ok
    print("测量完成：本次保存 %d 张 → %s" % (n_ok, out_path))
    return n_ok


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="手工标注门框 4 角 → pnp_calib 可吃的 dump")
    ap.add_argument("--capture", default=None, help="采图模式：输出目录（去畸变后，无叠加）")
    ap.add_argument("--n", type=int, default=8, help="--capture 抓几张（默认 8）")
    ap.add_argument("--images", default=None, help="标注模式：图片 glob/目录")
    ap.add_argument("--out", default=None, help="输出 JSONL（追加写；已存在则续写）")
    ap.add_argument("--undistort", action="store_true",
                    help="输入图是**原图**时加它：先 remap 到 PnP 域再标注")
    ap.add_argument("--no-snap", action="store_true", help="关闭「吸附到红管」")
    ap.add_argument("--resume", action="store_true", help="跳过已标注过的图（按文件名）")
    ap.add_argument("--zoom", type=int, default=4,
                    help="放大镜倍率（2~8，默认 4；仅在 --loupe 或按 m 打开时生效）")
    ap.add_argument("--loupe", action="store_true",
                    help="启动即打开放大镜（默认【关】：默认倍率下它 560px，会盖住 640 画面约 87%%）")
    ap.add_argument("--coords", default=None,
                    help="离线模式：直接给 'x,y;x,y;x,y;x,y'（TL;TR;BR;BL），不起窗口")
    ap.add_argument("--src", default=None, help="离线模式的图片名（写进记录的 src 字段）")
    ap.add_argument("--measure", action="store_true",
                    help="卷尺靶子模式（点 3 个刻度 → 算 Δu/斜视，写 kind=ruler 的 JSONL）")
    ap.add_argument("--span-m", type=float, default=1.0,
                    help="靶子刻度跨度(m)，默认 1.00（0→100 cm）")
    ap.add_argument("--ticks", default=None,
                    help="你点的三个刻度读数(cm)，如 10,50,90：自动算出 span 并留档"
                         "（推荐 —— 避免 span 填错）")
    a = ap.parse_args(argv)
    if getattr(a, "ticks", None):
        try:
            tv = [float(x) for x in str(a.ticks).replace("，", ",").split(",") if x.strip()]
            if len(tv) != 3:
                raise ValueError
            a.span_m = abs(tv[2] - tv[0]) / 100.0
            print("按 --ticks %s 推得 span = %.3f m（覆盖 --span-m）" % (a.ticks, a.span_m))
        except ValueError:
            print("✗ --ticks 要写成 '起,中,终' 三个数(cm)，例如 10,50,90")
            return 2

    if a.capture:
        capture(a.capture, n=a.n)
        return 0

    if a.coords:
        if not a.out:
            print("--coords 需要 --out")
            return 2
        pts = parse_coords(a.coords)
        d = load_done(a.out)
        rec = make_record(1 + len(d["paths"]) + len(d["legacy"]), pts, src=a.src or "coords")
        append_records(a.out, [rec])
        print("已写 1 条记录 → %s：%s" % (a.out, rec["dets"][0]["kpts"]))
        # 真值是写在**输出文件名**上的（pnp_calib 按文件名解析），不是 src
        from tools.analyze.pnp_calib import parse_gt_from_name
        g = parse_gt_from_name(a.out)
        print("（真值从输出文件名解析：%s）"
              % (g if g else "⚠️ 输出文件名里没有 z<厘米> → pnp_calib 会跳过这个文件！"
                             "例如命名为 pnp_z150.jsonl"))
        return 0

    if a.measure:
        if not a.images or not a.out:
            print("--measure 需要 --images 与 --out（靶子读数 JSONL，如 log/pnp_0922/ruler.jsonl）")
            return 2
        paths = sorted(glob.glob(a.images))
        if os.path.isdir(a.images):
            paths = sorted(glob.glob(os.path.join(a.images, "*.jpg")) +
                           glob.glob(os.path.join(a.images, "*.png")))
        if not paths:
            print("没找到图片：%s" % a.images)
            return 2
        print("待测 %d 张；坐标域 = %s" % (len(paths),
              "去畸变域（PnP 域，正确）" if bool(S.get("vision.image.undistort", False))
              else "⚠️ 原图域"))
        measure(paths, a.out, do_undistort=a.undistort, resume=a.resume,
                zoom=a.zoom, span_m=a.span_m, ticks=a.ticks, loupe=a.loupe)
        return 0

    if not a.images or not a.out:
        print("标注模式需要 --images 与 --out（或改用 --capture / --coords）")
        return 2
    paths = sorted(glob.glob(a.images))
    if os.path.isdir(a.images):
        paths = sorted(glob.glob(os.path.join(a.images, "*.jpg")) +
                       glob.glob(os.path.join(a.images, "*.png")))
    if not paths:
        print("没找到图片：%s" % a.images)
        return 2
    print("待标 %d 张；坐标域 = %s" % (len(paths),
          "去畸变域（PnP 域，正确）" if (bool(S.get("vision.image.undistort", False))
                                   or a.undistort)
          else "⚠️ 原图域（与 PnP 不一致！若这是原图请加 --undistort）"))
    label(paths, a.out, do_undistort=a.undistort, snap=not a.no_snap,
          resume=a.resume, zoom=a.zoom, loupe=a.loupe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
