#!/usr/bin/env python3
"""本地手工标「门 4 角点 + tag」的小工具（一张图可标多个门）。

已停用（现役打标走 Roboflow，本工具保留备查，见 scripts/README.md）。

每个门 = 一个框 + 4 个角点槽位：先在图上拖框，再在框内点 4 下。
  · 自动模式（默认）：按「离框的哪个角最近」判定该点属于 TL/TR/BR/BL；
  · 键位模式（`t` 切换）：先按 `1/2/3/4` 指定槽位再点（或先点再按）。
按下左键后移动超过 6 px 算拖框，几乎没动算打点。

约定：
  · v=2 可见（参与坐标损失），v=0 不可见（ultralytics 用 kpt_mask 屏蔽其坐标损失）。
    角点出画/被挡时按 `x` 写 `0 0 v=0`，不要为凑满 4 点去点一个假位置。
  · 标签一行一个门：`0 cx cy w h [x y v]×4`（归一化，槽位顺序 TL,TR,BR,BL）。
  · 框由手拖（把整根管子框进去），运行中 `[` `]` 统一外扩（每次 0.02，默认 0）。

键位
    左键拖          画/重画当前门的框（当前门已有点则自动开新门）
    左键点          框内打点（自动模式=按位置判定；键位模式=待指定）
    右键 / u        撤销上一步（点、归属、不可见、框，都可撤）
    1 / 2 / 3 / 4   指定角点槽位 TL / TR / BR / BL
    x               把当前武装槽位或第一个空槽标成不可见（v=0）
    t               切换 自动判定 / 键位指定
    g               手动开一个新门
    c               清空当前门的 4 个点（保留框）
    [ / ]           框统一外扩 -/+（每次 0.02）
    + / -           缩放显示（1x~4x）
    5 / 6 / 7       水质 tag：清水 / 中水 / 浊水（按文件名预填，可改）
    8 / 9 / 0       完整性 tag：g_full4 / g_part / g_tube  ← 保存前必选
    z               循环场景 tag（s_multi → s_close → s_none → 全关）
    s / 回车        保存本张并下一张
    n               跳过（不保存）下一张
    b               上一张
    p               删除本张已有标签（重新标）
    q / ESC         退出（已保存的都已落盘）

用法
    /home/ansty/anaconda3/envs/yolov8/bin/python scripts/1_prepare/label_pose_local.py \
        output/preview/auv5_pose/tables/worklist_local846.csv \
        --images-dir data/derived/AUV_5_selected_3000_Dwb \
        --out data/derived/AUV_5_labeled_local

    # 自检：不开窗口，验证「框 + 4 槽位」写成标签是否正确（含多门/不可见角点）
    ... label_pose_local.py <csv> --selftest

输出：<out>/labels/*.txt、<out>/images/（硬链接）、<out>/tags.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
NM = ["TL", "TR", "BR", "BL"]
NC = [(0, 0, 255), (0, 200, 0), (255, 128, 0), (0, 220, 220)]
BOX_COL = (255, 255, 255)
CUR_BOX_COL = (0, 255, 255)
WATER_BY_DIVE = {"AUV_5": "w_clear", "AUV_1": "w_mid", "AUV_4": "w_mid",
                 "AUV_2": "w_turbid", "AUV_3": "w_turbid"}
WATER_KEYS = {"5": "w_clear", "6": "w_mid", "7": "w_turbid"}
COMPL_KEYS = {"8": "g_full4", "9": "g_part", "0": "g_tube"}
SCENE_CYCLE = ["s_multi", "s_close", "s_none"]
V_VIS, V_INV = 2, 0
CLICK_TOL = 6.0          # 按下-抬起位移小于此值（原图像素）算「打点」，否则算「拖框」
LEGEND1 = ("drag=box  click=point  1/2/3/4=slot TL/TR/BR/BL  x=invisible  t=auto/manual slot  "
           "u=undo  c=clear pts  g=new gate")
LEGEND2 = ("[ ]=box pad  +/-=zoom | 5/6/7 water  8/9/0 completeness  z=scene | "
           "s=save+next  n=skip  b=prev  p=del  q=quit")


def new_gate() -> dict:
    """一个门：box=(x1,y1,x2,y2) 或 None；slots=[None 或 (x,y,v)] × 4"""
    return dict(box=None, slots=[None, None, None, None])


def gate_done(g: dict) -> bool:
    return g["box"] is not None and all(s is not None for s in g["slots"])


def label_line(g: dict, img_shape, pad_frac: float, min_box: float = 0.02) -> str:
    """把一个门（框 + 4 槽位）写成一行 YOLO pose 标签"""
    h, w = img_shape[:2]
    x1, y1, x2, y2 = g["box"]
    if pad_frac > 0:
        px, py = pad_frac * (x2 - x1), pad_frac * (y2 - y1)
        x1, y1, x2, y2 = x1 - px, y1 - py, x2 + px, y2 + py
    mw, mh = min_box * w, min_box * h
    if x2 - x1 < mw:
        c = (x1 + x2) / 2; x1, x2 = c - mw / 2, c + mw / 2
    if y2 - y1 < mh:
        c = (y1 + y2) / 2; y1, y2 = c - mh / 2, c + mh / 2
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(w), x2), min(float(h), y2)
    parts = ["0", f"{(x1+x2)/2/w:.6f}", f"{(y1+y2)/2/h:.6f}",
             f"{(x2-x1)/w:.6f}", f"{(y2-y1)/h:.6f}"]
    for i in range(4):
        s = g["slots"][i]
        if s is None:
            parts += ["0.000000", "0.000000", str(V_INV)]
        elif s[2] == V_VIS:
            parts += [f"{s[0]/w:.6f}", f"{s[1]/h:.6f}", str(V_VIS)]
        else:
            parts += ["0.000000", "0.000000", str(V_INV)]
    return " ".join(parts)


def nearest_free_slot(g: dict, x: float, y: float):
    """自动模式：点 (x,y) 离框的哪个角最近 → 归给那个槽位（只考虑空槽）"""
    x1, y1, x2, y2 = g["box"]
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]      # TL TR BR BL
    best, bd = None, 1e18
    for i in range(4):
        if g["slots"][i] is not None:
            continue
        d = (corners[i][0] - x) ** 2 + (corners[i][1] - y) ** 2
        if d < bd:
            best, bd = i, d
    return best


def status_line(name: str, gates: list, cur: int, pending, armed, tags: set,
                pad: float, zoom: float, auto: bool, idx: int, total: int, msg: str = "") -> str:
    water = next((t for t in tags if t.startswith("w_")), "?")
    compl = next((t for t in tags if t.startswith("g_")), "?")
    scene = ",".join(sorted(t for t in tags if t.startswith("s_"))) or "-"
    done = sum(1 for g in gates if gate_done(g))
    g = gates[cur] if gates else new_gate()
    have = "".join(NM[i][0] if g["slots"][i] is not None else "." for i in range(4))
    return (f"[{idx+1}/{total}] {name} gates={done} cur=G{cur+1}[{have}] "
            f"box={'Y' if g['box'] else 'N'} {'AUTO' if auto else 'KEY'} "
            f"armed={NM[armed] if armed is not None else '-'} pending={'Y' if pending else '-'} "
            f"water={water} compl={compl} scene={scene} pad={pad:.2f} zoom={zoom:.2f}  {msg}")


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("images", help="待标清单 CSV（含 name 列）或图片目录")
    ap.add_argument("--images-dir", default="data/derived/AUV_5_selected_3000_Dwb")
    ap.add_argument("--out", default="data/derived/AUV_5_labeled_local")
    ap.add_argument("--pad-frac", type=float, default=0.0,
                    help="框统一外扩比例（默认 0 = 完全按你拖的框）")
    ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--max-side", type=int, default=980)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    src = Path(a.images)
    if src.is_dir():
        names = sorted(p.name for p in src.iterdir()
                       if p.suffix.lower() in (".jpg", ".png", ".jpeg"))
    else:
        names = [r["name"] for r in csv.DictReader(open(src, encoding="utf-8"))]
    root = Path(a.images_dir)
    out = REPO / a.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    tags_path = out / "tags.csv"
    done = {}
    if tags_path.exists() and not a.redo:
        done = {r["name"]: r["tags"] for r in csv.DictReader(open(tags_path, encoding="utf-8"))}
    todo = [n for n in names
            if n not in done or not (out / "labels" / (Path(n).stem + ".txt")).exists()]
    print(f"共 {len(names)} 张；已完成 {len(names)-len(todo)}；待标 {len(todo)}")
    print(f"输出：{out}（labels/ + images/ + tags.csv）")
    print("交互：先在图上**拖一个框**，再在框里点 4 个角点；多门就多拖几个框。")

    if a.selftest:
        n = (todo or names)[0]
        img = cv2.imread(str(root / n))
        if img is None:
            sys.exit(f"❌ 读不到图 {root/n}")
        h, w = img.shape[:2]
        g1 = dict(box=(90, 110, 545, 515),
                  slots=[(105, 125, V_VIS), (520, 135, V_VIS), (535, 495, V_VIS), (110, 490, V_VIS)])
        g2 = dict(box=(40, 40, 260, 130),
                  slots=[(60, 60, V_VIS), (230, 62, V_VIS), None, None])
        g3 = dict(box=(300, 560, 420, 630), slots=[None, None, None, None])
        print(f"\n自检图 {n} ({w}x{h})：一张图 3 个门")
        for gi, g in enumerate((g1, g2, g3), 1):
            line = label_line(g, img.shape, a.pad_frac)
            v = line.split()
            coords = [float(x) for i, x in enumerate(v) if i >= 1 and (i - 5) % 3 != 2]
            print(f"  门{gi}: {line}")
            print(f"      列数={len(v)}（应 17）坐标 max={max(coords):.3f}（应≤1）"
                  f" v列={[int(v[7+3*i]) for i in range(4)]} 框wh={[round(float(x),4) for x in v[3:5]]}")
        print("\n（保存时这三行写进同一个 .txt；门2 的 BR/BL 是空槽 → 记成 0 0 v=0）")
        print("自动判归属自检：门2 里点 (240,70) 应归给 TR，点 (60,120) 应归给 BL ——")
        g2b = dict(box=(40, 40, 260, 130), slots=[(60, 60, V_VIS), None, None, None])
        print(f"   (240,70) → 槽位 {NM[nearest_free_slot(g2b, 240, 70)]}"
              f" / (60,120) → 槽位 {NM[nearest_free_slot(g2b, 60, 120)]}")
        return

    sys.path.insert(0, str(REPO / "output" / "preview" / "auv5_pose" / "tools"))
    from triage_review import ascii_only

    st = dict(gates=[new_gate()], cur=0, pending=None, armed=None, auto=True,
              drag=None, drag_start=None, pad=a.pad_frac, zoom=a.zoom, tags=set(),
              msg="", dirty=True, hist=[])
    idx = [0]

    def snap():
        st["hist"].append([(g["box"], list(g["slots"])) for g in st["gates"]] + [st["cur"]])
        if len(st["hist"]) > 50:
            st["hist"].pop(0)

    def scale_now() -> float:
        s = st["zoom"]
        return a.max_side / 640 if 640 * s > a.max_side else s

    def draw(view, s):
        for gi, g in enumerate(st["gates"]):
            is_cur = gi == st["cur"]
            if g["box"] is not None:
                b = g["box"]
                cv2.rectangle(view, (int(b[0] * s), int(b[1] * s)), (int(b[2] * s), int(b[3] * s)),
                              CUR_BOX_COL if is_cur else BOX_COL, 2 if is_cur else 1, cv2.LINE_AA)
                cv2.putText(view, f"G{gi+1}", (int(b[0] * s) + 3, int(b[1] * s) + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, CUR_BOX_COL if is_cur else BOX_COL,
                            1, cv2.LINE_AA)
            for i, sl in enumerate(g["slots"]):
                if sl is None:
                    continue
                q = (int(sl[0] * s), int(sl[1] * s))
                if sl[2] == V_VIS:
                    cv2.circle(view, q, 5, NC[i], -1, cv2.LINE_AA)
                else:
                    cv2.circle(view, q, 6, (150, 150, 150), 1, cv2.LINE_AA)
                col = NC[i] if sl[2] == V_VIS else (150, 150, 150)
                cv2.putText(view, NM[i], (q[0] + 6, q[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, col, 1, cv2.LINE_AA)
            pts = [s_ for s_ in g["slots"] if s_ is not None and s_[2] == V_VIS]
            if len(pts) == 4:
                for i in range(4):
                    a_, b_ = g["slots"][i], g["slots"][(i + 1) % 4]
                    if a_[2] == V_VIS and b_[2] == V_VIS:
                        cv2.line(view, (int(a_[0] * s), int(a_[1] * s)),
                                 (int(b_[0] * s), int(b_[1] * s)), NC[i], 2, cv2.LINE_AA)
        if st["drag"]:
            d = st["drag"]
            cv2.rectangle(view, (int(d[0] * s), int(d[1] * s)), (int(d[2] * s), int(d[3] * s)),
                          (60, 60, 255), 1, cv2.LINE_AA)
        if st["pending"]:
            p = st["pending"]
            cv2.drawMarker(view, (int(p[0] * s), int(p[1] * s)), (255, 255, 255),
                           cv2.MARKER_CROSS, 14, 2)
        if st["armed"] is not None:
            b = st["gates"][st["cur"]]["box"]
            if b:
                corner = [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])][st["armed"]]
                cv2.circle(view, (int(corner[0] * s), int(corner[1] * s)), 10, NC[st["armed"]], 2)

    def redraw(img0):
        if not st["dirty"]:
            return None
        s = scale_now()
        view = cv2.resize(img0, (int(img0.shape[1] * s), int(img0.shape[0] * s)),
                          interpolation=cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA)
        draw(view, s)
        bar = np.full((48, view.shape[1], 3), 24, np.uint8)
        cv2.putText(bar, ascii_only(status_line(names[min(idx[0], len(names)-1)], st["gates"],
                                               st["cur"], st["pending"], st["armed"], st["tags"],
                                               st["pad"], st["zoom"], st["auto"], idx[0], len(todo),
                                               st["msg"])),
                    (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, ascii_only(LEGEND1), (8, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (210, 210, 210), 1, cv2.LINE_AA)
        cv2.putText(bar, ascii_only(LEGEND2), (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (210, 210, 210), 1, cv2.LINE_AA)
        st["dirty"] = False
        st["msg"] = ""
        return np.vstack([view, bar])

    def assign(slot: int, x: float, y: float, v: int):
        st["gates"][st["cur"]]["slots"][slot] = (x, y, v)

    def on_mouse(event, mx, my, flags, param):
        s = scale_now()
        x, y = mx / s, my / s
        if event == cv2.EVENT_LBUTTONDOWN:
            snap()
            st["drag_start"] = (x, y)
            g = st["gates"][st["cur"]]
            if g["box"] is None or any(v is not None for v in g["slots"]):
                st["gates"].append(new_gate())          # 当前门已开动 → 新拖框属于新门
                st["cur"] = len(st["gates"]) - 1
            st["drag"] = [x, y, x, y]
            st["dirty"] = True
        elif event == cv2.EVENT_MOUSEMOVE and st.get("drag"):
            st["drag"][2], st["drag"][3] = x, y
            st["dirty"] = True
        elif event == cv2.EVENT_LBUTTONUP and st.get("drag"):
            x0, y0 = st["drag_start"]
            d = st["drag"]
            st["drag"] = None
            if abs(d[2] - x0) < CLICK_TOL and abs(d[3] - y0) < CLICK_TOL:
                # 视作「打一个点」
                g = st["gates"][st["cur"]]
                if g["box"] is None:
                    st["msg"] = "!! drag a box first"
                elif st["armed"] is not None:
                    assign(st["armed"], x0, y0, V_VIS)
                    st["armed"] = None
                elif st["auto"]:
                    sl = nearest_free_slot(g, x0, y0)
                    if sl is None:
                        st["msg"] = "!! all 4 slots taken (use c to clear)"
                    else:
                        assign(sl, x0, y0, V_VIS)
                else:
                    st["pending"] = (x0, y0)
                    st["msg"] = "press 1/2/3/4 to choose slot"
            else:
                st["gates"][st["cur"]]["box"] = (min(d[0], d[2]), min(d[1], d[3]),
                                                 max(d[0], d[2]), max(d[1], d[3]))
            st["dirty"] = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            undo()

    def undo():
        if not st["hist"]:
            return
        snap_state = st["hist"].pop()
        st["gates"] = [dict(box=b, slots=list(s)) for b, s in snap_state[:-1]]
        st["cur"] = snap_state[-1]
        if not st["gates"]:
            st["gates"], st["cur"] = [new_gate()], 0
        st["pending"] = None
        st["dirty"] = True

    def save(name: str) -> int:
        if not any(t.startswith("g_") for t in st["tags"]):
            st["msg"] = "!! need completeness tag (8=full4 9=part 0=tube)"
            st["dirty"] = True
            return -1
        full = [g for g in st["gates"] if gate_done(g)]
        if not full:
            st["msg"] = "!! need >=1 gate with box + 4 slots"
            st["dirty"] = True
            return -1
        img = cv2.imread(str(root / name))
        lines = [label_line(g, img.shape, st["pad"]) for g in full]
        (out / "labels" / (Path(name).stem + ".txt")).write_text("\n".join(lines) + "\n",
                                                                 encoding="utf-8")
        dst = out / "images" / name
        if not dst.exists():
            try:
                os.link(root / name, dst)
            except OSError:
                shutil.copy2(root / name, dst)
        tags = set(st["tags"])
        if len(full) >= 2:
            tags.discard("s_none")
            if "s_multi" not in tags:
                st["msg"] = "note: 2+ gates saved; consider scene tag s_multi (z)"
        rows = {}
        if tags_path.exists():
            rows = {r["name"]: r["tags"]
                    for r in csv.DictReader(open(tags_path, encoding="utf-8"))}
        rows[name] = "|".join(sorted(tags))
        with open(tags_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name", "tags"])
            for k in sorted(rows):
                w.writerow([k, rows[k]])
        return len(full)

    win = "label"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    sc_i = 0

    while 0 <= idx[0] < len(todo):
        name = todo[idx[0]]
        img0 = cv2.imread(str(root / name))
        if img0 is None:
            idx[0] += 1
            continue
        if not st["tags"]:
            dive = "_".join(name.split("_")[:2])
            st["tags"] = {WATER_BY_DIVE.get(dive, "w_clear")}
        st["dirty"] = True
        view = redraw(img0)
        if view is not None:
            cv2.imshow(win, view)
        k = cv2.waitKey(20)
        if k == -1:
            continue
        ch = chr(k) if 0 <= k < 256 else ""
        if ch == "q" or k == 27:
            break
        elif ch == "u":
            undo()
        elif ch in "1234":
            i = "1234".index(ch)
            g = st["gates"][st["cur"]]
            if st["pending"] is not None:
                assign(i, st["pending"][0], st["pending"][1], V_VIS)
                st["pending"] = None
            elif g["box"] is not None:
                st["armed"] = i
                st["msg"] = f"armed {NM[i]}: next click goes here"
            st["dirty"] = True
        elif ch == "x":
            g = st["gates"][st["cur"]]
            i = st["armed"] if st["armed"] is not None else next(
                (j for j in range(4) if g["slots"][j] is None), None)
            if i is None:
                st["msg"] = "!! all slots set"
            else:
                snap()
                assign(i, 0.0, 0.0, V_INV)
                st["armed"] = None
                st["msg"] = f"{NM[i]} = invisible (v=0)"
            st["dirty"] = True
        elif ch == "t":
            st["auto"] = not st["auto"]
            st["msg"] = f"slot mode = {'AUTO (by position)' if st['auto'] else 'KEY (press 1-4)'}"
            st["dirty"] = True
        elif ch == "g":
            snap()
            st["gates"].append(new_gate())
            st["cur"] = len(st["gates"]) - 1
            st["dirty"] = True
        elif ch == "c":
            snap()
            st["gates"][st["cur"]]["slots"] = [None, None, None, None]
            st["dirty"] = True
        elif ch == "[":
            st["pad"] = max(0.0, st["pad"] - 0.02); st["dirty"] = True
        elif ch == "]":
            st["pad"] = min(0.5, st["pad"] + 0.02); st["dirty"] = True
        elif ch in ("+", "="):
            st["zoom"] = min(4.0, st["zoom"] * 1.25); st["dirty"] = True
        elif ch in ("-", "_"):
            st["zoom"] = max(0.5, st["zoom"] / 1.25); st["dirty"] = True
        elif ch in WATER_KEYS:
            st["tags"] = {t for t in st["tags"] if not t.startswith("w_")} | {WATER_KEYS[ch]}
            st["dirty"] = True
        elif ch in COMPL_KEYS:
            st["tags"] = {t for t in st["tags"] if not t.startswith("g_")} | {COMPL_KEYS[ch]}
            st["dirty"] = True
        elif ch == "z":
            cur_scene = [t for t in SCENE_CYCLE if t in st["tags"]]
            st["tags"] = {t for t in st["tags"] if t not in SCENE_CYCLE}
            if not cur_scene:
                st["tags"].add(SCENE_CYCLE[0])
            elif cur_scene[0] != SCENE_CYCLE[-1]:
                st["tags"].add(SCENE_CYCLE[SCENE_CYCLE.index(cur_scene[0]) + 1])
            st["dirty"] = True
        elif ch in ("s", "\r", "\n"):
            ng = save(name)
            if ng > 0:
                idx[0] += 1
                st.update(gates=[new_gate()], cur=0, pending=None, armed=None,
                          hist=[], tags=set(), msg=st["msg"] or f"saved {ng} gate(s)")
            st["dirty"] = True
        elif ch == "n":
            idx[0] += 1
            st.update(gates=[new_gate()], cur=0, pending=None, armed=None, hist=[],
                      tags=set(), msg="skipped")
            st["dirty"] = True
        elif ch == "b":
            idx[0] = max(0, idx[0] - 1)
            st.update(gates=[new_gate()], cur=0, pending=None, armed=None, hist=[], tags=set())
            st["dirty"] = True
        elif ch == "p":
            for p in (out / "labels" / (Path(name).stem + ".txt"), out / "images" / name):
                if p.exists():
                    p.unlink()
            if tags_path.exists():
                rows = {r["name"]: r["tags"] for r in
                        csv.DictReader(open(tags_path, encoding="utf-8")) if r["name"] != name}
                with open(tags_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f); w.writerow(["name", "tags"])
                    for kk in sorted(rows):
                        w.writerow([kk, rows[kk]])
            st.update(gates=[new_gate()], cur=0, pending=None, armed=None, hist=[], tags=set(),
                      msg="deleted")
            st["dirty"] = True
    cv2.destroyAllWindows()
    n_done = len(list(csv.DictReader(open(tags_path, encoding="utf-8")))) if tags_path.exists() else 0
    print(f"\n已标注 {n_done} 张 → {out}")
    print("下一步：python scripts/2_train/build_stage_dataset.py --stage 1 "
          "--base data/datasets/AUV_5_gate-pose.yolov8 --add "
          f"{a.out} --negatives output/preview/auv5_pose/judgment/negatives_log.csv")


if __name__ == "__main__":
    main()
