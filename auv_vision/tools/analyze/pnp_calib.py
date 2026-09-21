# -*- coding: utf-8 -*-
"""tools/analyze/pnp_calib.py — 过门 PnP 位姿 / 深度估计的**离线标定与报告**工具。

它回答一个具体问题：**我们解出来的 z（和位姿）到底准不准、该把哪个参数改成多少。**

输入
----
`preview_detect.py --gate-kpt --dump` 产出的逐帧 JSONL（含 4 角点 + 置信度 + bbox）。
真值从**文件名**里读，不用手抄：

    pnp_z150.jsonl          正对门，卷尺量到 z = 1.50 m
    pnp_z150_lat+25.jsonl   同上，且相机光轴相对门中心横向偏 +25 cm（门在光轴右侧）
    pnp_z150_up+10.jsonl    同上，且门中心在光轴**上方** 10 cm
    pnp_z150_yaw+20.jsonl   同上，且船体相对门右转 20°
    pnp_z075_yaw-30.jsonl   组合随便，段名可任意顺序

    约定：lat/up/yaw 的**符号**都按"门相对相机"描述：
      lat+ = 门在光轴右侧（相机系 t_x 应为正）
      up+  = 门在光轴上方（相机系 t_y 应为**负**，因为图像 y 向下）
      yaw+ = 船体右转（门法向相对光轴偏左 → psi = gate_normal_angles_deg 应为负）

    文件名不带真值时，用 `--gt gt.csv`（表头：file,z_m,lat_m,up_m,yaw_deg）。

输出
----
1. 控制台：每档一行（z 真值 / z 实测 p50 / 相对误差 / 可用率 / mode 分布）；
2. `--report out.md`：完整报告（含参数校正建议、可直接粘贴的 cfg 片段）；
3. `--csv out.csv`：每档一行（喂 Excel / 贴进 runbook 记录表）；
4. `--frames-csv out.csv`：每帧一行（tz/tx/ty/psi/rms/mode，做散点图用）。

它做的四件事
------------
① **深度标尺反演**：正对门时 `z_meas = z_true · (W_cfg / W_true)`（4 角 PnP 的深度由
   门框**尺寸标尺**决定）→ 用实测比值直接反解**真实门宽**，再 1D 扫 `frame_h/frame_w`
   比值（正对时深度对宽高比只有弱依赖）拿最优 `frame_h`。
② **横向/竖向**：`t_x`（米）与 lat 真值、`t_y` 与 up 真值对比（校核尺度与符号）。
③ **航向**：`psi`（`gate_normal_angles_deg` 的 yaw 分量）与 yaw 真值对比 →
   安装角偏置（均值）与噪声（std）——这是 ALIGN.HDG 能不能收敛的直接依据。
④ **选参**：`conf_thr × reproj_px` 扫描（可用率 vs 深度误差 p90）、p3p vs full 对比、
   可选 kpt_mem 开/关 A/B。全部**离线复算**，不碰相机、不碰串口。

用法
----
    # 单档
    python3 tools/analyze/pnp_calib.py log/pnp_z150.jsonl
    # 整组（通配）+ 出报告
    python3 tools/analyze/pnp_calib.py --report log/pnp_report.md --csv log/pnp_summary.csv \\
        log/pnp_z*.jsonl
    # 只看 p3p/full 对比与选参扫描
    python3 tools/analyze/pnp_calib.py --sweep log/pnp_z*.jsonl

⚠️ 本工具**只读**：不改 cfg、不发串口。它给出的建议值要人工确认后再写进 cfg。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics as st
import sys

# 工程根 = tools/<类>/x.py 往上**三**级（分类重整后本脚本深了一层）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np                                              # noqa: E402

import base.settings as S                                       # noqa: E402
from gate.geometry import (object_points, gate_pose, reproj_rms,   # noqa: E402
                           gate_normal_angles_deg, CameraModel)
from gate.gate_frontend import parse_kpt_mode                   # noqa: E402

MODE_FULL = "full"
MODE_P3P = "p3p"
MODE_WIDTH = "width"
MODE_COARSE = "coarse"

# 真值文件名解析：pnp_z150 / pnp_z150_lat+25 / pnp_z150_up-10 / pnp_z150_yaw+20
_RE_Z = re.compile(r"z(\d{2,4})(?![\d])")
_RE_LAT = re.compile(r"lat([+-]?\d+)")
_RE_UP = re.compile(r"up([+-]?\d+)")
_RE_YAW = re.compile(r"yaw([+-]?\d+)")


# ---------------------------------------------------------------- 真值
class Gt(object):
    """一档的真值（米 / 度）。`src` 记录它从哪来（文件名 or csv）。"""

    def __init__(self, z_m, lat_m=0.0, up_m=0.0, yaw_deg=0.0, src="name"):
        self.z_m = float(z_m)
        self.lat_m = float(lat_m)
        self.up_m = float(up_m)
        self.yaw_deg = float(yaw_deg)
        self.src = src

    @property
    def fronto(self):
        """是否"正对门"档（只有这些档能用来反演尺寸标尺）。"""
        return abs(self.lat_m) < 1e-9 and abs(self.up_m) < 1e-9 and abs(self.yaw_deg) < 1e-9

    def __repr__(self):
        return "z=%.2f lat=%+.2f up=%+.2f yaw=%+.0f" % (
            self.z_m, self.lat_m, self.up_m, self.yaw_deg)


def parse_gt_from_name(path):
    """从文件名解析真值；解析不到 z → None（调用方可用 --gt 补）。"""
    base = os.path.basename(path)
    m = _RE_Z.search(base)
    if not m:
        return None
    z_cm = float(m.group(1))
    lat = float(_RE_LAT.search(base).group(1)) / 100.0 if _RE_LAT.search(base) else 0.0
    up = float(_RE_UP.search(base).group(1)) / 100.0 if _RE_UP.search(base) else 0.0
    yaw = float(_RE_YAW.search(base).group(1)) if _RE_YAW.search(base) else 0.0
    return Gt(z_cm / 100.0, lat, up, yaw, src="name")


def load_gt_csv(path):
    """CSV：file,z_m,lat_m,up_m,yaw_deg（相对路径按 CSV 所在目录解析）。"""
    out = {}
    d = os.path.dirname(os.path.abspath(path))
    with open(path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            f = (row.get("file") or "").strip()
            if not f:
                continue
            key = f if os.path.isabs(f) else os.path.join(d, f)
            out[os.path.basename(key)] = Gt(
                row["z_m"],
                row.get("lat_m") or 0.0, row.get("up_m") or 0.0,
                row.get("yaw_deg") or 0.0, src="csv")
    return out


# ---------------------------------------------------------------- 取检测
def _det_key(item, conf_thr):
    """与 `gate_task._pick_gate` 相同的选门规则：有效角点数 > 置信度和 > score。"""
    kc = item.get("kpt_conf")
    if not kc:
        return (-1, 0.0, float(item.get("score") or 0.0))
    kc = np.asarray(kc, np.float64)
    return (int((kc >= conf_thr).sum()), float(kc.sum()), float(item.get("score") or 0.0))


def read_frames(path, conf_thr):
    """读一个 dump 文件 → [(t, kpts(4,2)|None, kconf(4,)|None, bbox, fps, mode, ids)]。"""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            items = [d for d in (rec.get("dets") or []) if d.get("kind") == "gate"]
            if not items:
                out.append((rec.get("t") or 0.0, None, None, None, rec.get("fps"), "", []))
                continue
            best = max(items, key=lambda d: _det_key(d, conf_thr))
            kpts = best.get("kpts")
            kc = best.get("kpt_conf")
            kpts = np.asarray(kpts, np.float64) if kpts else None
            kc = np.asarray(kc, np.float64) if kc else None
            mode, ids = "", []
            if kpts is not None and kc is not None:
                mode, ids = parse_kpt_mode(kpts, kc, conf_thr)
            out.append((rec.get("t") or 0.0, kpts, kc, best.get("bbox"),
                        rec.get("fps"), mode, [int(i) for i in ids]))
    return out


# ---------------------------------------------------------------- 逐帧复算
def pose_frames(frames, camera, W, H, conf_thr, reproj_px, refine=True,
                use_kpt_mem=False, km_cfg=None):
    """对一档的帧序列复算位姿（`prev` 按帧序串联，与在线行为一致）。

    返回 dict：逐帧量 + 汇总（含 mode 分布与位姿可用率）。
    """
    obj_w = object_points(W, H)
    km = None
    if use_kpt_mem:
        from gate.kpt_memory import build_kpt_memory
        cfg = km_cfg if km_cfg is not None else (S.vision.gate.get("kpt_mem", None) or {})
        if not bool((cfg or {}).get("enable", False)):
            cfg = dict(cfg or {})
            cfg["enable"] = True            # A/B 时强制开（不改 cfg 文件）
        km = build_kpt_memory(cfg, n_kpt=4)

    rows = []
    prev = None
    n_mode = {MODE_FULL: 0, MODE_P3P: 0, MODE_WIDTH: 0, MODE_COARSE: 0}
    n_det = 0
    for (t, kpts, kc, bbox, fps, _mode, _ids) in frames:
        if kpts is None or kc is None:
            prev = None                     # 丢帧 → 与在线一致：解除上帧消歧
            continue
        n_det += 1
        if km is not None:
            # 时间轴用 dump 里的相对秒 → 毫秒，否则 kpt_memory 的"短消失回忆/valid_ms"
            # 会以为全程都在同一瞬间，A/B 结果失真
            kpts, kc = km.update(kpts, kc, int(round(float(t or 0.0) * 1000.0)))
            kpts = np.asarray(kpts, np.float64)
            kc = np.asarray(kc, np.float64)
        mode, ids = parse_kpt_mode(kpts, kc, conf_thr)
        n_mode[mode if mode in n_mode else MODE_COARSE] += 1
        row = {"t": t, "mode": mode, "n_kpt": len(ids), "fps": fps,
               "ratio": (float(bbox[2]) / camera.width) if bbox else None,
               "tz": None, "tx": None, "ty": None, "psi": None, "pitch": None,
               "rms": None}
        if mode in (MODE_FULL, MODE_P3P) and len(ids) >= 3:
            img2 = kpts[ids]
            res = gate_pose(camera, obj_w[ids], img2, prev=prev,
                            reproj_thr=reproj_px, refine=refine)
            if res is not None:
                rvec, tvec = res
                t3 = np.asarray(tvec, np.float64).reshape(3)
                row["tx"], row["ty"], row["tz"] = float(t3[0]), float(t3[1]), float(t3[2])
                row["rms"] = float(reproj_rms(camera, obj_w[ids], img2, rvec, tvec))
                psi, pit = gate_normal_angles_deg(rvec, tvec)
                row["psi"], row["pitch"] = float(psi), float(pit)
                prev = (rvec, tvec)
            else:
                prev = None                 # 解失败：别把坏位姿喂给下一帧消歧
        else:
            prev = None
        rows.append(row)
    return {"rows": rows, "n_det": n_det, "n_mode": n_mode, "n_frames": len(frames)}


def _vals(rows, key, mode=None):
    return [r[key] for r in rows
            if r.get(key) is not None and (mode is None or r["mode"] == mode)]


def _p(v, q):
    if not v:
        return None
    v = sorted(v)
    i = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
    return v[i]


def summarize(res, gt, mode=MODE_FULL):
    """一档的汇总（默认只统计 full 帧；p3p 单独看）。"""
    rows = res["rows"]
    z = _vals(rows, "tz", mode)
    out = {"gt": gt, "n_frames": res["n_frames"], "n_det": res["n_det"],
           "n_mode": dict(res["n_mode"]),
           "usable": (len(z) / float(res["n_det"])) if res["n_det"] else 0.0,
           "mode": mode, "z_n": len(z),      # z_n=0 = 这一档"有检测但一个 full 位姿都没解出来"
           # 不限档位的"真的解出位姿"帧数（full+p3p 合计）—— 判"有没有位姿"用它
           "z_any_n": len(_vals(rows, "tz")),
           "usable_any": (len(_vals(rows, "tz")) / float(res["n_det"]))
           if res["n_det"] else 0.0}
    if z:
        out.update(z_p50=st.median(z), z_p10=_p(z, 0.1), z_p90=_p(z, 0.9),
                   z_rel_p50=(st.median(z) - gt.z_m) / gt.z_m,
                   z_abs_p50=abs(st.median(z) - gt.z_m),
                   z_abs_p90=max(abs(_p(z, 0.1) - gt.z_m), abs(_p(z, 0.9) - gt.z_m)),
                   rms_p50=_p(_vals(rows, "rms", mode), 0.5),
                   ratio_p50=_p([r["ratio"] for r in rows if r.get("ratio")], 0.5),
                   tx_p50=(st.median(_vals(rows, "tx", mode))
                           if _vals(rows, "tx", mode) else None),
                   ty_p50=(st.median(_vals(rows, "ty", mode))
                           if _vals(rows, "ty", mode) else None),
                   psi_p50=(st.median(_vals(rows, "psi", mode))
                            if _vals(rows, "psi", mode) else None),
                   psi_std=(st.pstdev(_vals(rows, "psi", mode))
                            if len(_vals(rows, "psi", mode)) > 1 else None))
    return out


# ---------------------------------------------------------------- 标尺反演
def fit_depth(z_true, z_meas):
    """最小二乘 `z_meas = a·z_true + b`；返回 (a, b, r2, n)。"""
    if len(z_true) < 2:
        return None
    x = np.asarray(z_true, np.float64)
    y = np.asarray(z_meas, np.float64)
    A = np.vstack([x, np.ones_like(x)]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = float(sol[0]), float(sol[1])
    pred = a * x + b
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return a, b, r2, len(z_true)


def invert_scale(per_file, camera, conf_thr, reproj_px, W0, H0,
                 max_frames=300, refine=True, slope=None):
    """反演门框尺寸标尺。

    ① 用"正对档"的**线性拟合斜率** `a`（`z_meas = a·z_true + b`）反解真实门宽：
       `W* = W0 / a`。用斜率而不是"比值的直接中位数"是关键 —— 中位数会把截距 `b`
       （真值口径差、畸变残差）混进比例里，近距离档尤其容易被带偏（实测差 ~2%）。
       拟合不可用时退回"比值中位数"。
    ② 固定 `W*`，1D 扫宽高比 `frame_h/frame_w`（正对时深度对宽高比只有弱依赖，
       但比例错会让残余变大 → 近距被拒）→ 取 |相对误差| 中位数最小者。
    返回 dict（含扫描表，便于看平坦度）。
    """
    fronto = [f for f in per_file if f["gt"].fronto and f.get("z_p50")]
    if not fronto:
        return None
    ratios = [f["z_p50"] / f["gt"].z_m for f in fronto]
    ratio_med = st.median(ratios)
    if slope is None:
        ft = fit_depth([f["gt"].z_m for f in fronto], [f["z_p50"] for f in fronto])
        slope = ft[0] if ft else ratio_med
    W_star = W0 / slope if slope > 1e-6 else W0
    # 1D 扫宽高比
    table = []
    for ar in [0.45 + 0.02 * i for i in range(31)]:        # 0.45 ~ 1.05
        H = W_star * ar
        errs = []
        for f in fronto:
            frames = f["_frames"][:max_frames]
            r = pose_frames(frames, camera, W_star, H, conf_thr, reproj_px, refine=refine)
            z = _vals(r["rows"], "tz", MODE_FULL)
            if z:
                errs.append(abs(st.median(z) - f["gt"].z_m) / f["gt"].z_m)
        if errs:
            table.append((ar, H, float(np.median(errs)), len(errs)))
    return {"ratio": ratio_med, "slope": slope, "W_star": W_star, "H0": H0,
            "W_now": W0, "ar_table": table}


def _fmt(v, nd=3, dash="—"):
    if v is None:
        return dash
    if isinstance(v, float):
        return ("%." + str(nd) + "f") % v
    return str(v)


# ---------------------------------------------------------------- 报告
def render_md(per_file, results, camera, W0, H0, inv, sweeps, group_name=""):
    L = []
    A = L.append
    A("# 过门 PnP 位姿 / 深度标定报告")
    A("")
    A("> 由 `tools/analyze/pnp_calib.py` 自动生成。真值来自文件名（或 `--gt` CSV）；")
    A("> 位姿为**离线复算**（同一份 dump、同一套 cfg 参数），不涉及相机与推进器。")
    A("")
    A("## 0. 本次数据")
    A("")
    A("| 项 | 值 |")
    A("|---|---|")
    A("| dump 文件 | %d 档 |" % len(per_file))
    A("| 相机模型 | %s（%dx%d, fx=%.1f, fy=%.1f, cx=%.1f, cy=%.1f, rectified=%s） |"
      % (os.path.basename(str(S.vision.camera.front.calibration)),
         camera.width, camera.height, camera.fx, camera.fy, camera.cx, camera.cy,
         camera.rectified))
    A("| 检测坐标域 | %s（`vision.image.undistort=%s`，与 dump 同域） |"
      % ("去畸变域" if camera.rectified else "原始域", S.vision.image.undistort))
    A("| cfg 门框尺寸 | frame_w=%.3f m, frame_h=%.3f m |" % (W0, H0))
    A("| conf_thr / reproj_px | %s / %s |" % (results["conf_thr"], results["reproj_px"]))
    A("")

    A("## 1. 深度（`tvec.z`）逐档实测（只统计 full 4 角帧）")
    A("")
    A("| 真值 z (m) | 档名 | z 实测 p50 | 相对误差 | |误差| p90 | 可用率(full) | 帧数(full) | 有位姿帧(全档) | RMS p50 | 框占比 p50 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for f in per_file:
        gt = f["gt"]
        A("| %.2f | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            gt.z_m, os.path.basename(f["path"]),
            _fmt(f.get("z_p50")), _fmt(f.get("z_rel_p50")),
            _fmt(f.get("z_abs_p90")), "%.0f%%" % (100 * f.get("usable", 0.0)),
            f.get("z_n"), f.get("z_any_n"),
            _fmt(f.get("rms_p50"), 2), _fmt(f.get("ratio_p50"))))
    A("")
    no_pose = [os.path.basename(f["path"]) for f in per_file
               if f["n_det"] and not f.get("z_any_n")]
    if no_pose:
        A("> ⚠️ **这些档一帧位姿都没解出来**（有检测、但全被拒或全非 full/p3p）：")
        A("> `%s`。" % "`, `".join(no_pose))
        A("> 先看它们的 `RMS p50` 与 mode 分布：残差 > `reproj_px` 说明**门框尺寸/比例不对**"
          "（近距离最容易暴露），而不是「角点太差」。")
        A("")

    zt = [f["gt"].z_m for f in per_file if f["gt"].fronto and f.get("z_p50")]
    zm = [f["z_p50"] for f in per_file if f["gt"].fronto and f.get("z_p50")]
    fit = results.get("fit") or (fit_depth(zt, zm) if len(zt) >= 2 else None)
    A("### 1.1 正对档的线性拟合 `z_meas = a·z_true + b`")
    A("")
    if fit:
        a, b, r2, n = fit
        A("```")
        A("a = %.4f      （1.000 = 标尺正确；>1 = 解出的 z 偏大 = cfg 门宽比真实的大）" % a)
        A("b = %+.4f m   （应 ≈0；明显非 0 说明有系统性偏置/畸变残差）" % b)
        A("R² = %.4f     n = %d 档" % (r2, n))
        A("```")
        A("")
        A("**尺度关系**（正对门、4 角 PnP）：`z_meas = z_true · (W_cfg / W_true)`")
        A("⇒ 真实门宽 `W_true = W_cfg / a = %.3f / %.4f = **%.3f m**`"
          % (W0, a, W0 / a if a > 1e-6 else float("nan")))
    else:
        A("（可用档不足 2 个，无法拟合）")
    A("")

    if inv:
        A("### 1.2 门框尺寸标尺反演（可直接回填 `cfg/vision.yaml`）")
        A("")
        A("① 正对档的拟合斜率 `a = %.4f`（首选：它把截距 `b` 分离出去了）"
          "⇒ `frame_w* = %.3f / %.4f = **%.3f m**；"
          % (inv["slope"], inv["W_now"], inv["slope"], inv["W_star"]))
        A("   （对照：比值的直接中位数 = %.4f —— 它会把截距混进比例，故不作为首选）"
          % inv["ratio"])
        A("")
        A("② 固定 `frame_w*`，扫宽高比 `frame_h/frame_w`（正对时深度对宽高比只有弱依赖）：")
        A("")
        A("| frame_h/frame_w | frame_h (m) | 深度相对误差中位数 | 档数 |")
        A("|---|---|---|---|")
        best = min(inv["ar_table"], key=lambda t: t[2]) if inv["ar_table"] else None
        for ar, H, err, n in inv["ar_table"]:
            mark = " ← **最优**" if best and abs(ar - best[0]) < 1e-9 else ""
            A("| %.2f | %.3f | %.1f%% | %d%s |" % (ar, H, 100 * err, n, mark))
        A("")
        if best:
            A("⇒ 建议：`geometry.frame_w: %.3f`, `geometry.frame_h: %.3f`"
              % (inv["W_star"], best[1]))
            A("")
            A("```yaml")
            A("gate:")
            A("  geometry: {frame_w: %.3f, frame_h: %.3f}   # 由 pnp_calib 反演（原 %.3f/%.3f）"
              % (inv["W_star"], best[1], W0, H0))
            A("```")
        A("")
        A("> ⚠️ 回填前先看 ①b：若 `b` 明显非 0，或各档相对误差**随距离单调变化**"
          "（不是纯比例），说明问题不只是尺寸标尺（还有畸变/角点系统性外扩），"
          "此时**不要**用单点比例去校正，先把畸变与角点口径查清。")
        A("")

    A("## 2. 横向 / 竖向（`t_x` / `t_y`）")
    A("")
    A("| 档名 | lat 真值 (m) | t_x 实测 p50 | 偏差 | up 真值 (m) | t_y 实测 p50 | 偏差 |")
    A("|---|---|---|---|---|---|---|")
    for f in per_file:
        gt = f["gt"]
        if gt.lat_m == 0.0 and gt.up_m == 0.0 and f.get("tx_p50") is None:
            continue
        A("| %s | %+.2f | %s | %s | %+.2f | %s | %s |" % (
            os.path.basename(f["path"]), gt.lat_m, _fmt(f.get("tx_p50")),
            _fmt((f["tx_p50"] - gt.lat_m) if f.get("tx_p50") is not None else None),
            gt.up_m, _fmt(f.get("ty_p50")),
            _fmt((f["ty_p50"] + gt.up_m) if f.get("ty_p50") is not None else None)))
    A("")
    A("> 符号：图像 x 向右 ⇒ `t_x` 正 = 门在光轴**右**侧（与 lat+ 同号）；")
    A("> 图像 y 向下 ⇒ 门在光轴**上方**时 `t_y` 为**负**（所以偏差列写成 `t_y + up`）。")
    dx = (camera.cx - camera.width / 2.0) / camera.fx
    dy = (camera.cy - camera.height / 2.0) / camera.fy
    A(">")
    A("> ⚠️ **主点不在画面中心**（本相机 `cx=%.1f` vs 中心 %.1f、`cy=%.1f` vs 中心 %.1f）："
      % (camera.cx, camera.width / 2.0, camera.cy, camera.height / 2.0))
    A("> 把门摆在**画面正中**时，它相对**光轴**其实偏了，固有值 ≈ "
      "`t_x = %+.4f·z`、`t_y = %+.4f·z` 米。" % (dx, -dy))
    A("> 判 Q3 之前先把这一项从偏差里减掉（或干脆按光轴而不是画面中心去标定真值），"
      "**不要**为了让它归零去改代码里的坐标。")
    A("")

    A("## 3. 航向（`psi` = 门法向相对光轴的水平角）")
    A("")
    A("| 档名 | yaw 真值 (°) | psi 实测 p50 | psi + yaw | psi std | 帧数 |")
    A("|---|---|---|---|---|---|")
    yaw_rows = [f for f in per_file if f.get("psi_p50") is not None]
    for f in yaw_rows:
        gt = f["gt"]
        A("| %s | %+.0f | %s | %s | %s | %s |" % (
            os.path.basename(f["path"]), gt.yaw_deg, _fmt(f.get("psi_p50"), 2),
            _fmt(f.get("psi_p50") + gt.yaw_deg, 2), _fmt(f.get("psi_std"), 2),
            f.get("z_n")))
    A("")
    if yaw_rows:
        offs = [f["psi_p50"] + f["gt"].yaw_deg for f in yaw_rows]
        stds = [f["psi_std"] for f in yaw_rows if f.get("psi_std")]
        A("⇒ **安装角偏置** ≈ %+.2f°（各档 `psi + yaw` 的中位数；船体右转时 psi 应为负，"
          "故两者相加应≈0）；**帧内噪声** ≈ %.2f°（各档 std 的中位数）。"
          % (st.median(offs), st.median(stds) if stds else float("nan")))
        A("")
        A("> 用法：`comm.gate.hdg.tol_deg` 必须显著大于这个噪声（当前 8°，实测噪声若 >4° 就别再收紧）。")
        A("> ⚠️ 这个偏置**不是** `gate.geometry.body_center_offset`（那是**位置**偏置，单位米），"
          "别拿它去抵角度偏置。")
        A("")

    if sweeps:
        A("## 4. 选参扫描（离线复算；同一份数据换参数重解）")
        A("")
        A("### 4.1 `keypoint.conf_thr` × `pnp.reproj_px`（正对档）")
        A("")
        A("| conf_thr | reproj_px | 位姿可用率 | 深度相对误差中位数 | 误差 p90 |")
        A("|---|---|---|---|---|")
        for (ct, rj, usable, errm, errp) in sweeps["grid"]:
            A("| %.2f | %.0f | %.0f%% | %.1f%% | %.1f%% |"
              % (ct, rj, 100 * usable, 100 * errm, 100 * errp))
        A("")
        A("**怎么选**：先要**可用率**（低帧率下位姿太少 → 只能退回 coarse 档），")
        A("但在可用率接近时优先选**误差 p90 更小**的那一格；`reproj_px` 放大到误差开始变差为止。")
        A("")
        A("### 4.2 p3p（3 角）vs full（4 角）")
        A("")
        A("| 档名 | mode | z p50 | 相对误差 | psi p50 | psi std | 帧数 |")
        A("|---|---|---|---|---|---|---|")
        for r in sweeps["p3p"]:
            A("| %s | %s | %s | %s | %s | %s | %s |" % (
                r["file"], r["mode"], _fmt(r.get("z_p50")), _fmt(r.get("z_rel")),
                _fmt(r.get("psi_p50"), 2), _fmt(r.get("psi_std"), 2), r.get("n")))
        A("")
        A("> 结论模板：若 p3p 的 |相对误差| 明显大于 full（历史实测 ≈ −28%）")
        A("> ⇒ **p3p 只能用来「继续对准」，不能喂米制阈值**（`z.cross`/`near_lost_m`/`slow_max`）。")
        A("")

    A("## 5. 结论与待办（**人工填写**）")
    A("")
    A("| 项 | 本次结论 | 依据（哪张表/哪一行） |")
    A("|---|---|---|")
    A("| 深度是否可信 | | §1/§1.1 |")
    A("| `frame_w/frame_h` 改成多少 | | §1.2 |")
    A("| `reproj_px` / `conf_thr` 改成多少 | | §4.1 |")
    A("| p3p 能不能给米制阈值 | | §4.2 |")
    A("| `hdg.tol_deg` 定多少 | | §3 |")
    A("| 下一步要补什么数据 | | |")
    A("")
    A("> 回填 cfg 后请重跑同一份 dump 复算，确认表 1 的相对误差确实变小；")
    A("> 再用 `bash tools/deploy/check_board_parity.sh --board` 确认板端一致。")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- 扫描
def sweep_grid(files_frames, camera, W, H, conf_thrs, reprojs, max_frames=200):
    """conf_thr × reproj_px 扫描（只看正对档的可用率与深度误差）。"""
    out = []
    for ct in conf_thrs:
        for rj in reprojs:
            usable, errs = [], []
            for frames, gt in files_frames:
                r = pose_frames(frames[:max_frames], camera, W, H, ct, rj)
                z = _vals(r["rows"], "tz", MODE_FULL)
                if r["n_det"]:
                    usable.append(len(z) / float(r["n_det"]))
                if z:
                    errs.append(abs(st.median(z) - gt.z_m) / gt.z_m)
            if not usable:
                continue
            out.append((ct, rj, st.mean(usable),
                        st.median(errs) if errs else float("nan"),
                        max(errs) if errs else float("nan")))
    return out


def p3p_compare(files_frames, camera, W, H, conf_thr, reproj_px, max_frames=400):
    """p3p vs full 的深度/航向对比。`files_frames` = [(frames, gt, 档名)]。"""
    rows = []
    for frames, gt, name in files_frames:
        r = pose_frames(frames[:max_frames], camera, W, H, conf_thr, reproj_px)
        for mode in (MODE_FULL, MODE_P3P):
            z = _vals(r["rows"], "tz", mode)
            psi = _vals(r["rows"], "psi", mode)
            if not z:
                continue
            rows.append({"file": name, "mode": mode,
                         "z_p50": st.median(z),
                         "z_rel": (st.median(z) - gt.z_m) / gt.z_m,
                         "psi_p50": st.median(psi) if psi else None,
                         "psi_std": (st.pstdev(psi) if len(psi) > 1 else None),
                         "n": len(z)})
    return rows


# ---------------------------------------------------------------- 主流程
def run(paths, camera=None, gt_csv=None, conf_thr=None, reproj_px=None,
        W=None, H=None, max_frames=400, sweep=False, do_invert=True,
        conf_thrs=(0.5, 0.6, 0.7, 0.8), reprojs=(8.0, 12.0, 16.0, 20.0, 25.0),
        use_kpt_mem=False, verbose=True):
    """主流程：返回 (per_file, results)。所有阈值默认取当前 cfg。"""
    from gate.gate_detector import board_camera
    cam = camera if camera is not None else board_camera()
    V = S.vision.gate
    G = V.get("geometry", None) or {}
    W = float(W if W is not None else G.get("frame_w", 0.70))
    H = float(H if H is not None else G.get("frame_h", 0.50))
    P = V.get("pnp", None) or {}
    conf_thr = float(conf_thr if conf_thr is not None
                     else (V.get("keypoint", None) or {}).get("conf_thr", 0.7))
    reproj_px = float(reproj_px if reproj_px is not None
                      else P.get("reproj_px", 20.0))
    ref = bool(P.get("refine", True))

    csv_gt = load_gt_csv(gt_csv) if gt_csv else {}
    per_file = []
    for p in paths:
        gt = csv_gt.get(os.path.basename(p)) or parse_gt_from_name(p)
        if gt is None:
            print("⚠️ 跳过（文件名里没有 z 真值，也没在 --gt 里给出）：%s" % p)
            continue
        frames = read_frames(p, conf_thr)
        res = pose_frames(frames, cam, W, H, conf_thr, reproj_px, refine=ref,
                          use_kpt_mem=use_kpt_mem)
        s = summarize(res, gt, MODE_FULL)
        s["path"] = p
        s["rows"] = res["rows"]
        s["_frames"] = frames
        per_file.append(s)
        if verbose:
            print("  %-28s 真值 z=%.2f | 实测 p50=%s | 相对误差=%s | full 帧=%s | 可用率=%.0f%%"
                  % (os.path.basename(p), gt.z_m, _fmt(s.get("z_p50")),
                     _fmt(s.get("z_rel_p50")), s.get("z_n"), 100 * s.get("usable", 0.0)))
    per_file.sort(key=lambda f: (f["gt"].z_m, f["gt"].lat_m, f["gt"].up_m, f["gt"].yaw_deg))

    # 正对档的深度拟合（只取真解析出 z 的档；没解出位姿的档不进拟合）
    fronto = [f for f in per_file if f["gt"].fronto and f.get("z_p50") is not None]
    fit = fit_depth([f["gt"].z_m for f in fronto], [f["z_p50"] for f in fronto]) \
        if len(fronto) >= 2 else None

    inv = None
    if do_invert:
        inv = invert_scale(per_file, cam, conf_thr, reproj_px, W, H, refine=ref,
                           slope=(fit[0] if fit else None))

    sweeps = None
    if sweep:
        fronto_s = [(f["_frames"], f["gt"]) for f in per_file if f["gt"].fronto]
        allf = [(f["_frames"], f["gt"], os.path.basename(f["path"])) for f in per_file]
        sweeps = {"grid": sweep_grid(fronto_s, cam, W, H, conf_thrs, reprojs),
                  "p3p": p3p_compare(allf, cam, W, H, conf_thr, reproj_px)}
    return per_file, {"cam": cam, "W": W, "H": H, "conf_thr": conf_thr,
                      "reproj_px": reproj_px, "inv": inv, "sweeps": sweeps,
                      "fit": fit, "use_kpt_mem": use_kpt_mem}


def write_summary_csv(path, per_file):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "z_true_m", "lat_true_m", "up_true_m", "yaw_true_deg",
                    "z_meas_p50_m", "z_rel_err", "z_abs_p90_m", "usable_rate",
                    "n_full", "mode_full", "mode_p3p", "mode_width", "mode_coarse",
                    "rms_p50_px", "ratio_p50", "tx_p50_m", "ty_p50_m",
                    "psi_p50_deg", "psi_std_deg", "gt_src"])
        for f in per_file:
            gt = f["gt"]
            w.writerow([os.path.basename(f["path"]), gt.z_m, gt.lat_m, gt.up_m, gt.yaw_deg,
                        _fmt(f.get("z_p50"), 4), _fmt(f.get("z_rel_p50"), 4),
                        _fmt(f.get("z_abs_p90"), 4), _fmt(f.get("usable"), 4),
                        f.get("z_n"), f["n_mode"].get(MODE_FULL),
                        f["n_mode"].get(MODE_P3P), f["n_mode"].get(MODE_WIDTH),
                        f["n_mode"].get(MODE_COARSE), _fmt(f.get("rms_p50"), 3),
                        _fmt(f.get("ratio_p50"), 4), _fmt(f.get("tx_p50"), 4),
                        _fmt(f.get("ty_p50"), 4), _fmt(f.get("psi_p50"), 3),
                        _fmt(f.get("psi_std"), 3), gt.src])


def write_frames_csv(path, per_file):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "z_true_m", "yaw_true_deg", "t", "mode", "n_kpt",
                    "tz_m", "tx_m", "ty_m", "psi_deg", "pitch_deg", "rms_px",
                    "ratio", "fps"])
        for f in per_file:
            gt = f["gt"]
            for r in f["rows"]:
                w.writerow([os.path.basename(f["path"]), gt.z_m, gt.yaw_deg,
                            r.get("t"), r.get("mode"), r.get("n_kpt"),
                            _fmt(r.get("tz"), 4), _fmt(r.get("tx"), 4),
                            _fmt(r.get("ty"), 4), _fmt(r.get("psi"), 3),
                            _fmt(r.get("pitch"), 3), _fmt(r.get("rms"), 3),
                            _fmt(r.get("ratio"), 4), r.get("fps")])


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="过门 PnP 位姿/深度标定：真值 dump → 误差表 + 参数反演 + 报告",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="+", help="dump jsonl（支持通配，shell 会展开）")
    ap.add_argument("--gt", default=None, help="真值 CSV（文件名没编码真值时用）")
    ap.add_argument("--report", default=None, help="把 markdown 报告写到这个文件")
    ap.add_argument("--csv", default=None, help="每档一行的汇总 CSV")
    ap.add_argument("--frames-csv", default=None, help="每帧一行的明细 CSV")
    ap.add_argument("--conf-thr", type=float, default=None, help="覆盖 vision.gate.keypoint.conf_thr")
    ap.add_argument("--reproj-px", type=float, default=None, help="覆盖 vision.gate.pnp.reproj_px")
    ap.add_argument("--frame-w", type=float, default=None, help="覆盖 geometry.frame_w (m)")
    ap.add_argument("--frame-h", type=float, default=None, help="覆盖 geometry.frame_h (m)")
    ap.add_argument("--max-frames", type=int, default=400, help="每档参与统计的最大帧数")
    ap.add_argument("--sweep", action="store_true", help="附加 conf_thr×reproj_px 扫描与 p3p 对比")
    ap.add_argument("--no-invert", action="store_true", help="跳过门框尺寸反演（省时间）")
    ap.add_argument("--kpt-mem", action="store_true", help="离线开启角点融合做 A/B")
    a = ap.parse_args(argv)

    paths = []
    for p in a.dumps:
        paths.extend(sorted(glob.glob(p)) or [p])
    if not paths:
        print("没有可分析的 dump")
        return 2
    print("== pnp_calib：%d 个 dump ==" % len(paths))
    per_file, res = run(paths, gt_csv=a.gt, conf_thr=a.conf_thr, reproj_px=a.reproj_px,
                        W=a.frame_w, H=a.frame_h, max_frames=a.max_frames,
                        sweep=a.sweep, do_invert=not a.no_invert,
                        use_kpt_mem=a.kpt_mem)
    if not per_file:
        print("✗ 没有一档解析出真值：文件名要含 z<厘米>（如 pnp_z150.jsonl）或用 --gt")
        return 2
    if a.csv:
        write_summary_csv(a.csv, per_file)
        print("汇总 CSV → %s" % a.csv)
    if a.frames_csv:
        write_frames_csv(a.frames_csv, per_file)
        print("逐帧 CSV → %s" % a.frames_csv)
    if a.report:
        md = render_md(per_file, res, res["cam"], res["W"], res["H"],
                       res["inv"], res["sweeps"])
        open(a.report, "w", encoding="utf-8").write(md)
        print("报告 → %s" % a.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
