# -*- coding: utf-8 -*-
"""tools/analyze/ruler_calib.py — 「卷尺刻度靶子」解算器：从靶子读数求**等效焦距 fx 与距离口径 c**

要解决的问题
------------
PnP 的深度 `z_meas = fx_cfg·W_cfg/Δu` 里有两个未知量在只看门框时**简并**：
  · 相机/介质带来的有效焦距 `fx_med`（罩外是空气还是水，差 ~1.4×）；
  · 门框的真实口径 `W_true`（外缘 0.77 还是管子中线 0.72）。
用**已知长度的卷尺刻度**当靶子就与门无关了 —— 但还需要一个距离口径 `c`：

    Δu = fx·L / (z_tape + c)      L = 刻度跨度(m)，z_tape = 从固定零点量到的读数
    ⇒ z_tape + c = fx·L/Δu        ← 对 (fx, c) **线性**，两条以上即可最小二乘

`c` 的物理含义：**光心相对你那个零点标记的偏置**（镜头/入瞳在零点前方则 c>0）。
⚠️ 换零点只是把 `c` 变成另一个固定常数，**不会让它归零** —— 只能"零点固定成制度 + 实测一次"。

输入
----
`tools/analyze/label_corners.py --measure` 产出的 JSONL（`kind: "ruler"`，一行一个档位）。
`z_tape` 优先取记录字段，缺了就从 `src`/文件名里解析（`ruler_z200` → 2.00 m）。

用法
----
    python3 tools/analyze/ruler_calib.py log/pnp_0922/ruler.jsonl
    python3 tools/analyze/ruler_calib.py --report log/pnp_0922/ruler_air.md log/pnp_0922/ruler.jsonl
    python3 tools/analyze/ruler_calib.py --skip 300 log/pnp_0922/ruler_wet.jsonl   # 丢掉最远档

判据（runbook §A0b / §B4）
-------------------------
  · 每次解算至少 **2 档**（3 档可做留一校验）；`skew` 超过 ±2% 的档**先别用**（靶面没正对）；
  · `fx` 的用途：与 `vision.camera.front.calibration` 里的 fx 比 ⇒ `k_med = fx_cfg/fx_med`；
    水下解出 `fx_water ≈ 782.5 ± 5%` ⇒ 现用标定在水里仍有效，跑船的米制阈值不用改；
  · `c` 的用途：之后所有卷尺读数都按 `z_true = z_tape + c` 换算。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np                                              # noqa: E402


# ------------------------------------------------------------------ 读入
def z_from_name(s):
    """`...z150...` → 1.50 m。

    ⚠️ 靶子图是 `log/pnp_0922/ruler_z200/cap_001.jpg` —— 真值在**目录名**上（`cap_001.jpg` 里没有），
    所以按"最后两段路径"扫，别只看 basename。
    """
    parts = [p for p in str(s or "").replace("\\", "/").split("/") if p]
    tail = "/".join(parts[-2:]) if len(parts) >= 2 else (parts[0] if parts else "")
    m = re.search(r"z(\d{2,4})", tail)
    return (int(m.group(1)) / 100.0) if m else None


def du_from_points(p):
    """由三个刻度点算跨度(px)：**欧氏距离**，任意方向都成立。

    ⚠️ 早期版本用 `p[2][0]-p[0][0]`（x 差）—— 卷尺竖着放时它直接得 0
    （2026-09-22 现场实测：竖向 379 px 的真实跨度被算成 2 px）。靶面与光轴垂直时
    `z=const` 平面到图像是均匀缩放，所以欧氏距离 = `(f/z)·L`，与靶线在画面里的方向无关。
    """
    q = np.asarray(p, dtype=np.float64).reshape(-1, 2)
    if q.shape[0] < 2:
        return None
    return float(np.hypot(*(q[-1] - q[0])))


def axis_of(rec):
    """这条读数量到的是 `fx` 还是 `fy`。

    靶线水平 ⇒ 量到 fx；竖直 ⇒ fy；斜着 ⇒ 两者混合（45° 附近最糟）。
    判据用 `label_corners --measure` 记下的 `theta_deg`（画面里的倾角）。
    没有 theta（手写记录）时按 `axis` 键，再不行默认 x。
    """
    th = rec.get("theta_deg")
    if th is not None:
        th = abs(float(th)) % 180.0
        if th > 90.0:
            th = 180.0 - th
        return "y" if th >= 45.0 else "x"
    ax = str(rec.get("axis") or "x").lower()
    return "y" if ax.startswith("y") else "x"


def load_records(paths):
    """读所有靶子读数；返回 [{src, z, du, span, skew}]（跳过非 ruler 行与缺 z 的行）。"""
    out, skipped = [], 0
    for pat in paths:
        for p in sorted(glob.glob(pat)) or [pat]:
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except ValueError:
                        skipped += 1
                        continue
                    if r.get("kind") not in (None, "ruler") and "du" not in r:
                        skipped += 1
                        continue
                    du = r.get("du")
                    # ⚠️ **优先用 `p` 重算**：`du` 可能是早期版本（把竖向跨度算成 x 差）写下的，
                    #    三点才是原始观测。两者不一致时以 `p` 为准，这样老记录自动被治好。
                    if r.get("p"):
                        du = du_from_points(r["p"])            # 欧氏距离（任意方向）
                    if du is None:
                        du = r.get("du")
                    z = r.get("z_tape")
                    if z is None:
                        z = z_from_name(r.get("src") or p)
                    span = float(r.get("span_m") or 1.0)
                    if du is None or z is None or du <= 0:
                        skipped += 1
                        continue
                    rec = {"src": r.get("src") or os.path.basename(p),
                           "z": float(z), "du": float(du), "span": span,
                           "skew": float(r.get("skew") or 0.0),
                           "mid_frac": r.get("mid_frac"), "off_mid": r.get("off_mid"),
                           "theta_deg": r.get("theta_deg")}
                    # ⚠️ 同理：三点在场就**重算**斜视/中点偏离。老版本在 du≈0 时把 skew
                    #    算成 3.0（300%），留在记录里会污染报告。
                    if r.get("p"):
                        try:
                            from tools.analyze.label_corners import span_stats
                            st = span_stats(r["p"])
                            rec.update({"skew": st["skew"], "mid_frac": st["mid_frac"],
                                        "off_mid": st["off_mid"], "theta_deg": st["theta_deg"]})
                            rec["du"] = st["du"]
                        except Exception:
                            pass
                    rec["axis"] = axis_of(rec)          # ⚠️ 放在自愈之后：竖着量的记录要判成 y
                    out.append(rec)
    return out, skipped


# ------------------------------------------------------------------ 解算
def _span(r):
    """跨度(m)：`load_records` 产出的键是 `span`，原始记录里是 `span_m` —— 两个都认。"""
    return float(r.get("span", r.get("span_m", 1.0)) or 1.0)


def _z(r):
    """真值距离：`load_records` 归一成 `z`；手写记录可能是 `z_tape`。"""
    v = r.get("z", r.get("z_tape"))
    return float(v) if v is not None else None


def _du(r):
    """像素跨度：优先 `du`，否则由 `p`（三个刻度点）算。"""
    d = du_from_points(r.get("p") or [])                       # 有点就用点（欧氏，任意方向）
    if d is not None:
        return d
    return float(r["du"]) if r.get("du") is not None else None


def solve(recs):
    """最小二乘解 (fx, c)。模型 `z + c = fx·L/Δu` 对 (fx, c) 线性。

    `recs` 里的每条记录兼容 `load_records` 的产物（`z/du/span/skew`）与手写形式
    （`z_tape/du(|p)/span_m`）。返回 dict：fx, c, rms_m, rms_px, 逐档预测, 条件数。
    """
    norm = []
    for r in recs:
        z, du = _z(r), _du(r)
        if z is None or du is None or du <= 0:
            raise ValueError("记录缺 z 真值或缺 Δu：%s" % (r.get("src") or r))
        norm.append({"src": r.get("src"), "z": z, "du": du, "span": _span(r),
                     "axis": axis_of(r), "skew": float(r.get("skew") or 0.0),
                     "theta_deg": r.get("theta_deg")})
    recs = norm
    if len(recs) < 2:
        raise ValueError("至少要 2 档（拿到 %d 档）：两条方程才解得开 (fx, c)" % len(recs))
    if len({round(r["z"], 4) for r in recs}) < 2:
        raise ValueError(
            "所有档位的 z_tape 都是 %.2f m —— 同一个距离量再多遍也解不开 (fx, c)（两条方程退化成一条）。"
            "\n  → 再量一个**不同距离**的档；或加 --gate 用门框数据联立（见 --gate 说明）。"
            % recs[0]["z"])
    axes = [ax for ax in ("x", "y") if any(r["axis"] == ax for r in recs)]
    # 未知量：[各轴的 f...] + [c]；靶线水平→fx，竖直→fy（混着量就同时解两个）
    A = np.array([[r["span"] / r["du"] if r["axis"] == ax else 0.0 for ax in axes] + [-1.0]
                  for r in recs])
    if np.linalg.matrix_rank(A) < A.shape[1]:
        raise ValueError(
            "设计矩阵秩不足（%d 档 / %d 个未知量：%s 与 c）—— 档位太集中，解不开。"
            "\n  → 加一个**距离明显不同**的档（档位跨度越大越稳；同一距离量多遍没用）。"
            % (len(recs), A.shape[1], "、".join("f%s" % ax for ax in axes)))
    b = np.array([r["z"] for r in recs])
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    focal = {ax: float(sol[i]) for i, ax in enumerate(axes)}
    c = float(sol[-1])
    fx = focal.get("x", float("nan"))
    rows, res = [], []
    for r in recs:
        L = r["span"]
        pred_du = focal[r["axis"]] * L / (r["z"] + c)
        resid_px = r["du"] - pred_du
        pred_z = focal[r["axis"]] * L / r["du"] - c    # 用实测 Δu 反推的距离
        res.append(r["z"] - pred_z)
        rows.append({"src": r["src"], "z": r["z"], "du": r["du"], "skew": r["skew"],
                     "span": L, "axis": r["axis"],
                     "du_pred": pred_du, "resid_px": resid_px,
                     "rel_du": (resid_px / pred_du) if pred_du else 0.0,
                     "z_fit": pred_z, "resid_m": r["z"] - pred_z})
    cond = float(np.linalg.cond(A)) if len(recs) >= 2 else float("nan")
    # 留一（≥3 档才有意义）：看 (fx, c) 稳不稳
    loo = []
    if len(recs) >= 3:
        for i in range(len(recs)):
            sub = [r for j, r in enumerate(recs) if j != i]
            # ⚠️ 子集必须**保留每个轴的档**，否则那个焦距在子集里根本没有约束
            #    （2026-09-22 自检：丢掉某轴唯一的档后，留一会给出 fx=-223 这种垃圾）。
            if {r["axis"] for r in sub} != set(axes):
                continue
            try:
                s = solve(sub)
                loo.append(("#%d %s z=%.2f(%s)"
                            % (i + 1, recs[i]["src"], recs[i]["z"], recs[i]["axis"]),
                            s["focal"], s["c"]))
            except ValueError:
                pass
    return {"fx": fx, "fy": focal.get("y", float("nan")), "focal": focal, "axes": axes,
            "c": float(c),
            "rms_m": float(np.sqrt(np.mean(np.square(res)))),
            "rms_px": float(np.sqrt(np.mean(np.square([r["resid_px"] for r in rows])))),
            "rows": rows, "cond": cond, "loo": loo, "n": len(recs)}


def cfg_focal(default=None):
    """cfg 里的 `(fx, fy)`（读不到就返回 default，不抛）。"""
    try:
        import base.settings as S
        p = S.get("vision.camera.front.calibration", None)
        for cand in (p, os.path.join(_ROOT, "cfg", os.path.basename(p or ""))):
            if cand and os.path.exists(cand):
                import cv2
                fs = cv2.FileStorage(cand, cv2.FILE_STORAGE_READ)
                v = fs.getNode("camera_matrix").mat()
                fs.release()
                if v is not None:
                    return float(v[0, 0]), float(v[1, 1])
    except Exception:
        pass
    return default if isinstance(default, tuple) else (default, default)


def fx_cfg(default=None):
    """cfg 里的 fx（兼容旧调用）。"""
    return cfg_focal(default)[0]


# ---------------------------------------------- 门框 dump（联立用）
def load_gate(paths):
    """读门框 dump：文件名 `...z<厘米>...` 即真值；每档取**门宽像素中位数**。

    复用 `pnp_calib.parse_gt_from_name` 的命名约定（`pnp_z150.jsonl` → 1.50 m）。
    返回 [{src, z, du, n}]，`du` = 每帧 `|x_TR − x_TL|` 的中位数（门框横向跨度）。
    """
    from tools.analyze.pnp_calib import parse_gt_from_name
    out = []
    for pat in paths:
        for p in sorted(glob.glob(pat)) or [pat]:
            if not os.path.exists(p):
                continue
            gt = parse_gt_from_name(p)                  # 返回 pnp_calib.Gt（含 z_m/lat/up/yaw）
            if gt is None:
                continue
            if not gt.fronto:                           # 只用"正对门"的档（斜/偏档会改 Δu）
                continue
            z = gt.z_m
            w = []
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    for d in (r.get("dets") or []):
                        k = d.get("kpts")
                        if k and len(k) == 4:
                            w.append(abs(float(k[1][0]) - float(k[0][0])))
            if w:
                out.append({"src": os.path.basename(p), "z": float(z),
                            "du": float(np.median(w)), "n": len(w)})
    return out


def solve_joint(ruler, gate, ratio=1.0, c_grid=None):
    """靶子 + 门框**联立**解 `(fx, fy, c, W_eff)`。

    靶子 `Δu = f_轴·L/(z+c)`（水平靶线给 `fx`、竖直给 `fy`）；门 `Δu = fx·W/(z+c)`。
    固定 `c` 时**线性** ⇒ 对 `c` 一维扫描，每个 `c` 闭式解其余参数，取相对残差最小者。

    未知量按手头数据自动决定：
      · 靶子只有竖向档 → `(fy, fx·W)`，并用 `fx = ratio·fy` 联结（`ratio` = cfg 的 fx/fy）；
      · 靶子有横向档 → 直接解 `fx`，**不需要**任何等比假设。

    ⚠️ 为什么必须联立：**单档靶子只有 1 个观测**，`(f, c)` 是整条曲线（简并）；
    门框的几档提供额外的"比例"信息，把 `c` 与 `W_eff` 一起压出来。
    """
    if not gate:
        raise ValueError("联立解需要门框 dump（--gate）")
    if not ruler:
        raise ValueError("联立解需要靶子读数（位置参数）")
    if c_grid is None:
        c_grid = np.arange(-0.40, 0.401, 0.01)
    ruler = [dict(r, axis=axis_of(r)) for r in ruler]   # 手写记录可能没有 axis 键
    has_x = any(r["axis"] == "x" for r in ruler)
    has_y = any(r["axis"] == "y" for r in ruler)
    # 列：[fx?] [fy?] [fx·W]；没有横向靶子时把 fx 用 ratio 挂在 fy 上
    cols = (["fx"] if has_x else []) + (["fy"] if has_y else []) + ["fxW"]
    rows = []
    for c in c_grid:
        A, b = [], []
        for r in ruler:
            row = dict.fromkeys(cols, 0.0)
            k = r["span"] / (r["z"] + c)
            if r["axis"] == "x":
                row["fx"] = k
            else:
                row["fy"] = k
            A.append([row[cn] for cn in cols])
            b.append(r["du"])
        for g in gate:
            # 门框观测给出的是乘积 `fx·W`（不是单独某个焦距）⇒ 系数恒为 1/(z+c)；
            # `fx = ratio·fy` 只在**报告**里用来把 W 拆出来（不需要它也能拟合）。
            row = dict.fromkeys(cols, 0.0)
            row["fxW"] = 1.0 / (g["z"] + c)
            A.append([row[cn] for cn in cols])
            b.append(g["du"])
        A, b = np.asarray(A), np.asarray(b)
        if np.linalg.matrix_rank(A) < A.shape[1]:
            raise ValueError("联立的设计矩阵秩不足：靶子与门框的档位距离太集中，解不开 (f, c, W)。")
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        d = {cn: float(sol[i]) for i, cn in enumerate(cols)}
        fy = d.get("fy", d.get("fx", 0.0) * (1.0 / ratio if ratio else 1.0))
        fx = d.get("fx", fy * ratio)
        fxW = d["fxW"]
        res = A @ sol - b
        wgt = np.array([1.0 / r["du"] for r in ruler] + [1.0 / g["du"] for g in gate])
        rows.append({"c": float(c), "fy": fy, "fx": fx,
                     "W": (fxW / fx) if abs(fx) > 1e-9 else float("nan"),
                     "rms_px": float(np.sqrt(np.mean(res ** 2))),
                     "rms_rel": float(np.sqrt(np.mean((res * wgt) ** 2))),
                     "res": res})
    best = dict(min(rows, key=lambda r: r["rms_rel"]))
    best["profile"] = rows
    return best


def report_joint(ruler, gate, best, fx_cfg_value=None, title="靶子+门框 联立解"):
    A = ["# %s（`tools/analyze/ruler_calib.py --gate`）\n" % title]
    A.append("模型：靶子 `Δu = fy·L/(z+c)`；门框 `Δu = fx·W_eff/(z+c)`，`fx/fy` 取自 cfg。")
    A.append("固定 `c` 时对 `(fy, fx·W_eff)` 线性 ⇒ 一维扫描 `c`，取相对残差最小者。\n")
    A.append("| 观测 | z_tape (m) | 长度/宽度 | Δu 实测 (px) | Δu 预测 (px) | 残差 (px) |")
    A.append("|---|---|---|---|---|---|")
    for r in ruler:
        pred = best["fy"] * r["span"] / (r["z"] + best["c"])
        A.append("| 靶子 %s | %.2f | L=%.2f m | %.1f | %.1f | %+.1f |"
                 % (r["src"], r["z"], r["span"], r["du"], pred, pred - r["du"]))
    for g in gate:
        pred = best["fx"] * best["W"] / (g["z"] + best["c"])
        A.append("| 门框 %s（%d 帧） | %.2f | W_eff=%.3f m | %.1f | %.1f | %+.1f |"
                 % (g["src"], g["n"], g["z"], best["W"], g["du"], pred, pred - g["du"]))
    A.append("")
    A.append("## 联立结果\n")
    A.append("| 量 | 值 | 说明 |")
    A.append("|---|---|---|")
    A.append("| `fy` | **%.0f px** | 该介质的等效 fy（靶子是竖向量的就是它） |" % best["fy"])
    A.append("| `fx` | **%.0f px** | = `fy`×cfg 的 fx/fy 比 |" % best["fx"])
    A.append("| `c` | **%+.3f m** | 光心相对零点标记的偏置 |" % best["c"])
    A.append("| `W_eff` | **%.3f m** | **模型眼里的门宽**（卷尺量的外缘是 0.77） |" % best["W"])
    if fx_cfg_value:
        A.append("| `k_air = fx_cfg/fx` | **%.3f** | PnP 深度在**空气**里的系统性比例 |"
                 % (fx_cfg_value / best["fx"]))
    A.append("| rms | %.2f px（相对 %.2f%%） | 全部观测 / 3 未知量 ⇒ 自由度有限 |"
             % (best["rms_px"], best["rms_rel"] * 100))
    A.append("")
    A.append("## `c` 的简并剖面（**本结果最大的不确定度来源**）\n")
    A.append("| c (m) | fy | fx | W_eff (m) | k_air | 相对 rms |")
    A.append("|---|---|---|---|---|---|")
    for r in best["profile"]:
        if abs((round(r["c"] * 100) % 5)) != 0:
            continue
        A.append("| %+.2f | %.0f | %.0f | %.3f | %.3f | %.2f%% |"
                 % (r["c"], r["fy"], r["fx"], r["W"],
                    (fx_cfg_value / r["fx"]) if fx_cfg_value else float("nan"),
                    r["rms_rel"] * 100))
    A.append("")
    A.append("⇒ 表里 `k_air` 的跨度就是 `c` 简并带来的不确定度。**再量一个不同距离的靶子档**"
             "（把卷尺**横过来**放 —— 画面宽度 1280 px 够 1.5 m 处的 1.00 m 跨度）就能把 `c` 钉死。")
    return "\n".join(A) + "\n"


# ------------------------------------------------------------------ 报告
def report(recs, skipped, res, fx_cfg_value, title="卷尺靶子解算", fy_cfg_value=None):
    A = []
    A.append("# %s（`tools/analyze/ruler_calib.py`）\n" % title)
    A.append("模型：`Δu = f·L/(z_tape + c)` ⇒ `z_tape + c = f·L/Δu`（对 `f, c` 线性，最小二乘）。")
    A.append("**水平靶线量到 `fx`、竖直靶线量到 `fy`** —— 混着量就同时解两个，"
             "顺便验证「罩是否各向异性」（不再需要假设 `fx/fy` 等比）。\n")
    A.append("| src | 轴 | z_tape (m) | L (m) | Δu (px) | skew | Δu 预测 (px) | 残差 (px) | 相对 |")
    A.append("|---|---|---|---|---|---|---|---|---|")
    for r in res["rows"]:
        A.append("| %s | %s | %.2f | %.2f | %.1f | %+.2f%% | %.1f | %+.1f | %+.2f%% |"
                 % (r["src"], r["axis"], r["z"], r["span"], r["du"], r["skew"] * 100,
                    r["du_pred"], r["resid_px"], r["rel_du"] * 100))
    A.append("")
    A.append("## 解算结果\n")
    A.append("| 量 | 值 | 说明 |")
    A.append("|---|---|---|")
    for ax in res["axes"]:
        A.append("| **f%s** | **%.1f px** | 由**%s向**靶线解出的等效焦距 |"
                 % (ax, res["focal"][ax], "水平" if ax == "x" else "竖直"))
    A.append("| c | **%+.3f m** | 光心相对零点标记的偏置（`z_true = z_tape + c`） |" % res["c"])
    A.append("| rms | %.1f px / %.3f m | 拟合残差 |" % (res["rms_px"], res["rms_m"]))
    A.append("| 档数 | %d | 设计矩阵条件数 %.1f（越小越稳） |" % (res["n"], res["cond"]))
    if fx_cfg_value:
        for ax in res["axes"]:
            fcfg = fx_cfg_value if ax == "x" else fy_cfg_value
            if fcfg:
                A.append("| 对比 cfg（%s） | f%s_cfg = %.1f px ⇒ **k = %.3f** | "
                         "PnP 深度在**该介质**里的系统性比例 |"
                         % (ax, ax, fcfg, fcfg / res["focal"][ax]))
    if not res["loo"] and res["n"] >= 3:
        A.append("")
        A.append("（留一校验跳过：档数只够恰好定住 %d 个未知量，没有多余自由度。"
                 "想开留一就再加一档不同距离的读数。）" % (len(res["axes"]) + 1))
    if res["loo"]:
        A.append("")
        A.append("## 留一校验（每次丢一档重解；只对「各轴都还有档」的子集做）\n")
        A.append("| 丢掉 | " + " | ".join("f%s" % ax for ax in res["axes"]) + " | c (m) |")
        A.append("|---" * (len(res["axes"]) + 2) + "|")
        for src, focal, c in res["loo"]:
            A.append("| %s | " % src
                     + " | ".join("%.1f" % focal[ax] for ax in res["axes"])
                     + " | %+.3f |" % c)
        c_s = [x[2] for x in res["loo"]] + [res["c"]]
        A.append("")
        spread = []
        for ax in res["axes"]:
            v = [x[1][ax] for x in res["loo"]] + [res["focal"][ax]]
            spread.append("f%s ±%.1f px（%.1f%%）"
                          % (ax, (max(v) - min(v)) / 2,
                             100 * (max(v) - min(v)) / 2 / res["focal"][ax]))
        A.append("⇒ " + "、".join(spread) + "、c ±%.3f m —— 超过判据（f ±2%%，c ±3 cm）"
                 "说明档位分布不够开或标注有系统偏差。" % ((max(c_s) - min(c_s)) / 2))
    # ⚠️ 斜视告警按**全部输入读数**判（包括被你 `--skip` 或被过滤掉的那些档），
    #    否则"歪掉的那一档"正好会从报告里消失 —— 那是最需要看见的信息。
    bad = [r for r in recs if abs(float(r.get("skew") or 0.0)) > 0.02]
    if bad:
        A.append("")
        A.append("⚠️ **skew > 2%% 的档**（靶面没正对，先重拍再用；它们可能没进上面的解算）：%s"
                 % "、".join("%s(%+.2f%%)" % (r.get("src"), float(r.get("skew") or 0.0) * 100)
                             for r in bad))
    if skipped:
        A.append("\n（跳过 %d 行：缺 z 真值或缺 Δu）" % skipped)
    return "\n".join(A) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="卷尺靶子 → 解算等效焦距 fx 与距离口径 c")
    ap.add_argument("jsonl", nargs="+", help="靶子读数 JSONL（label_corners --measure 产出）")
    ap.add_argument("--report", default=None, help="把 markdown 报告写到文件")
    ap.add_argument("--skip", type=float, nargs="*", default=[],
                    help="丢弃这些 z_tape 档（m），如 --skip 3.0")
    ap.add_argument("--fx-cfg", type=float, default=None,
                    help="覆盖对比用的 cfg fx（默认从 vision.camera.front.calibration 读）")
    ap.add_argument("--gate", nargs="*", default=None,
                    help="门框 dump（`...z<厘米>...jsonl`）：与靶子**联立**解 (f, c, W_eff)。"
                         "单档靶子本来解不开 (f,c)，加这个才有解")
    ap.add_argument("--focal-ratio", type=float, default=None,
                    help="联立时假定 fx/fy（默认取 cfg 的比值；给 1.0 表示假设两轴等比）")
    ap.add_argument("--title", default="卷尺靶子解算")
    a = ap.parse_args(argv)

    recs, skipped = load_records(a.jsonl)
    if a.skip:
        keep = [r for r in recs if not any(abs(r["z"] - s) < 1e-6 for s in a.skip)]
        skipped += len(recs) - len(keep)
        recs = keep
    if not recs:
        print("没读到可用的靶子读数（需要 kind=ruler 且有 z 真值的 JSONL）")
        return 2
    print("读到 %d 档靶子：%s" % (len(recs), "、".join("z=%.2f→Δu=%.1f" % (r["z"], r["du"])
                                                     for r in recs)))
    fxc, fyc = cfg_focal()
    if a.fx_cfg is not None:
        fxc = a.fx_cfg
    ratio = (a.focal_ratio if a.focal_ratio is not None
             else ((fxc / fyc) if fyc else 1.0))

    if a.gate:
        gate = load_gate(a.gate)
        if not gate:
            print("✗ --gate 里没有能解析出真值的门框 dump（文件名要带 z<厘米>，如 A_pnp_z150.jsonl）")
            return 2
        print("读到 %d 档门框：%s" % (len(gate), "、".join("z=%.2f→Δu=%.1f(%d帧)"
                                                          % (g["z"], g["du"], g["n"]) for g in gate)))
        try:
            best = solve_joint(recs, gate, ratio=ratio)
        except ValueError as e:
            print("✗ %s" % e)
            return 2
        txt = report_joint(recs, gate, best, fx_cfg_value=fxc, title=a.title)
    else:
        try:
            res = solve(recs)
        except ValueError as e:
            print("✗ %s" % e)
            return 2
        txt = report(recs, skipped, res, fxc, title=a.title, fy_cfg_value=fyc)
    print(txt)
    if a.report:
        os.makedirs(os.path.dirname(os.path.abspath(a.report)) or ".", exist_ok=True)
        with open(a.report, "w", encoding="utf-8") as fh:
            fh.write(txt)
        print("→ 报告已写：%s" % a.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
