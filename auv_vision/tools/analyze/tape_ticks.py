# -*- coding: utf-8 -*-
"""tools/analyze/tape_ticks.py — 用**卷尺自己的刻度**测像素比例（不点鼠标、亚像素）

为什么需要它
------------
`label_corners --measure` 让操作者点 3 个刻度，靠人眼定位 ⇒ 误差 ~±1.5 px/点。
而卷尺上**每隔 1 cm 就有一条刻度线**，一整条尺上有几十上百条 ⇒
把「1 cm 对应多少像素」当成一个**周期估计**问题，精度可以做到 ±0.5%，
而且**完全不用点鼠标**。2026-09-22 实测：1 m 档三张图给出的周期 10.679/10.763/10.833 px
（离散 1.4%），与门框/A1 那套独立证据互证。

模型（与 `ruler_calib` 完全一致，所以输出可以直接喂它）
----------------------------------------------------
    Δu = f·L/(z + c)     这里 L = 一个刻度间隔（标准卷尺 = 0.01 m）
    ⇒ f = period_px · (z + c) / L

⚠️ 前两个前提，错了结果就错：
  1. **刻度间隔必须是 1 cm**（=0.01 m）。若这把尺细刻度是 5 mm，`f` 会**差 2 倍**。
     自检办法：拿两个已知距离各测一次，比值应约等于距离比（与 1 cm 假设无关）；
     或与门框/`--gate` 的结果对一下量级。
  2. **靶线要水平**（量到的是 `fx`）。竖直放量到的是 `fy`。
  3. 距太远就测不出来：刻度周期 ≈ `f·0.01/z`。`f≈1080` 时 3 m 处只有 3.6 px，
     而 JPEG 的 8×8 块效应正好落在这个尺度上（2026-09-22 实测在 3 m 处检出的是块效应，
     周期 7.9 px）⇒ **只信 1.0~2.0 m 的结果**；3 m 用点击法。

用法
----
    python3 tools/analyze/tape_ticks.py --z 1.00 log/pnp_0922/ruler_z100/*.jpg
    python3 tools/analyze/tape_ticks.py --z 1.00 --out log/pnp_0922/ruler_ticks.jsonl \
            --band 118,140 --x 220,950 cap_001.jpg
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


# ------------------------------------------------------------------ 纯函数（可测）
def find_ticks(pr, min_prom=0.6, min_gap=2):
    """亮度剖面 → 刻度（暗线）位置，抛物线插值到亚像素。

    `pr` 是沿尺方向的平均亮度（亮尺 + 暗刻度）。返回 x 索引数组（相对 `pr` 起点）。
    """
    sm = np.convolve(pr, np.ones(3) / 3, mode="same")
    base = np.convolve(pr, np.ones(15) / 15, mode="same")
    d = sm - base
    out = []
    for i in range(2, len(d) - 2):
        if d[i] < d[i - 1] and d[i] <= d[i + 1] and d[i] < -min_prom:
            a, b, c = d[i - 1], d[i], d[i + 1]
            den = a - 2 * b + c
            off = 0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0
            x = i + off
            if out and x - out[-1] < min_gap:      # 同一条刻度被检两次
                continue
            out.append(x)
    return np.asarray(out, float)


def drop_extra_ticks(x, ratio=1.45, iters=200):
    """删掉"多检"的杂线：若 `(a,b,c)` 三点里 `(b−a)+(c−b) < ratio × 中位间距`，
    说明这三格挤成了两格 ⇒ **中间那个是多余的**（印刷数字的竖笔画、污点、
    或尺上比目标格更细的刻度）。

    ⚠️ 这一步不能靠"拟合后剔离群点"代替：多检一格会让**它后面所有序号整体平移**，
    残差因此呈锯齿状、MAD 阈值刚好放它过去（2026-09-22 自检实测：rms 2.26 px 剔不掉）。
    """
    x = list(np.asarray(x, float))
    for _ in range(iters):
        if len(x) < 8:
            break
        d = np.diff(x)
        med = float(np.median(d))
        if med <= 0:
            break
        hit = -1
        for i in range(1, len(x) - 1):
            if d[i - 1] + d[i] < ratio * med:
                hit = i
                break
        if hit < 0:
            break
        del x[hit]
    return np.asarray(x, float)


def fft_period(pr, lo=2.0, hi=25.0, nbins=8000):
    """整条尺的**全局**周期（FFT 主峰 + 抛物线插值）。

    它对噪声稳，但会被"靶面没正对"带来的比例梯度糊掉（整条尺各处周期不同）；
    `period_at` 则相反。**两者必须交叉校验**——只信一个会出事：
    2026-09-22 实测 3 m 档（刻度周期只有 3.6 px，与 JPEG 8×8 块效应同量级），
    刻度拟合锁到了杂线、自信地给出 f=4252 px，而 FFT 给的是 3.58 px（正确）。
    """
    d = np.asarray(pr, float)
    if len(d) < 16 or d.std() < 1e-6:
        return None
    # ⚠️ 必须先**高通**（减滑动均值）：否则尺子的亮度包络/印刷数字这些低频成分
    #    会把真正的刻度周期压掉（3 m 档实测：不高通 → FFT 给 11.6 px；高通后 → 3.6 px，正确）
    d = d - np.convolve(d, np.ones(9) / 9.0, mode="same")
    d = d - d.mean()
    sp = np.abs(np.fft.rfft(d * np.hanning(len(d)), 1 << 16))
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), nbins))
    amp = sp[np.clip((1.0 / grid * (1 << 16)).astype(int), 0, len(sp) - 1)]
    # ⚠️ 亚谐波改正：**窄脉冲串的频谱是梳状**，2 倍频（周期一半）的峰可能比基频还高
    #    （合成的锐利 1px 刻度就会这样：真周期 11 px，直接取最大峰得到 5.5 px）。
    #    若 2×/3× 周期处也有 ≥60% 的峰，取**最大的**那个（真周期）。
    k = int(np.argmax(amp))
    a_max = float(amp[k])
    for mul in (3, 2):
        per_try = float(grid[k]) * mul
        if per_try > hi:
            continue
        j = int(np.argmin(np.abs(grid - per_try)))
        if float(amp[j]) >= 0.6 * a_max:
            k = j
            a_max = float(amp[j])
    per = float(grid[k])
    if 0 < k < nbins - 1:
        a, b, c = float(amp[k - 1]), float(amp[k]), float(amp[k + 1])
        den = a - 2 * b + c
        if abs(den) > 1e-12:
            per = float(grid[k] + 0.5 * (a - c) / den * (grid[k + 1] - grid[k]))
    return per


def robust_polyfit(k, x, deg=2, floor=0.6, k_mad=3.0, iters=5):
    """稳健多项式拟合：按 **MAD** 剔离群点（用 std 会被离群点自己撑大，剔不掉）。

    返回 `(co, keep_mask)`。
    """
    k = np.asarray(k, float)
    x = np.asarray(x, float)
    keep = np.ones(len(k), bool)
    co = np.polyfit(k, x, min(deg, max(1, len(k) - 1)))
    for _ in range(iters):
        resid = x - np.polyval(co, k)
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med))) * 1.4826
        thr = max(floor, k_mad * mad)
        new_keep = np.abs(resid - med) <= thr
        if new_keep.sum() < 8:
            break
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
        co = np.polyfit(k[keep], x[keep], min(deg, max(1, keep.sum() - 1)))
    return co, keep


def fit_index(x, deg=2, iters=8):
    """给刻度编序号 k 并拟合 `x(k)`（容忍漏检、多检与透视梯度）。

    三步，2026-09-22 自检逐条逼出来的：
      1. **增量编号**：用最近几格的中位间距决定这一步跨几格（漏检 → 一次跳 2 格）；
      2. **稳健拟合**（按 MAD 剔离群点）—— 用 std 会被离群点撑大阈值，剔不掉；
      3. **反解序号并合并重复**：用拟合曲线反解每点的序号，同号只留离拟合最近的那个
         （多检的杂线会落到同一号上被合并）；再回到 2，直到序号不再变。

    ⚠️ 只做"常数周期划线"不行（靶面没正对时尺上比例本来就在变，实测跨度内差 6%）。

    返回 dict：`co`(多项式系数)、`k`、`x`、`rms`、`n`、`period_med`。
    """
    x = drop_extra_ticks(np.asarray(x, float))
    if len(x) < 8:
        return None
    if float(np.median(np.diff(x))) <= 0:
        return None
    k = [0.0]
    for i in range(1, len(x)):
        p_loc = float(np.median(np.diff(x[max(0, i - 6):i + 1])))
        step = max(1, int(round((x[i] - x[i - 1]) / max(p_loc, 1e-6))))
        k.append(k[-1] + step)
    k = np.asarray(k, float)

    co = None
    for _ in range(iters):
        co, keep = robust_polyfit(k, x, deg)
        if keep.sum() < 8:
            return None
        xk, kk = x[keep], k[keep]
        kn = np.empty_like(xk)
        for j, xv in enumerate(xk):
            lo, hi = kk.min() - 2.0, kk.max() + 2.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if np.polyval(co, mid) < xv:
                    lo = mid
                else:
                    hi = mid
            kn[j] = round(0.5 * (lo + hi))
        # 同一序号只留离拟合最近的那个
        order = np.argsort(kn, kind="stable")
        keep2, taken = [], set()
        for j in order:
            kj = int(kn[j])
            if kj in taken:
                continue
            taken.add(kj)
            keep2.append(j)
        keep2 = np.asarray(sorted(keep2))
        if len(keep2) == len(xk) and np.array_equal(kn[keep2], kk):
            x, k = xk, kk
            break
        x, k = xk[keep2], kn[keep2]

    co, keep = robust_polyfit(k, x, deg)
    x, k = x[keep], k[keep]
    if len(k) < 8:
        return None
    co = np.polyfit(k, x, min(deg, len(k) - 1))
    resid = x - np.polyval(co, k)
    return {"co": co, "k": k, "x": x, "n": len(k),
            "rms": float(np.sqrt(np.mean(resid ** 2))),
            "period_med": float(np.median(np.diff(x)))}


def period_at(fit, x_target):
    """`x(k) = x_target` 处的 `dx/dk`（= 该处每格多少 px）。

    ⚠️ 取**光轴处**（`x = cx`）而不是整条尺的平均：靶面没正对时，尺上各处的
    像素/厘米 本来就不同，而我们量的距离是到**尺中心**的 —— 中心处的局部比例才对。
    """
    co = np.asarray(fit["co"], float)
    lo, hi = float(fit["k"].min()) - 10.0, float(fit["k"].max()) + 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if np.polyval(co, mid) < x_target:
            lo = mid
        else:
            hi = mid
    k0 = 0.5 * (lo + hi)
    return float(np.polyval(np.polyder(co), k0)), k0


def f_from_period(period_px, z_m, c_m=0.0, grad_m=0.01):
    """`f = 周期 × (z + c) / 刻度间隔`。"""
    return float(period_px) * (float(z_m) + float(c_m)) / float(grad_m)


def _runs(mask, min_len):
    out, st = [], None
    for i, v in enumerate(list(mask) + [False]):
        if v and st is None:
            st = i
        elif not v and st is not None:
            if i - st >= min_len:
                out.append((st, i))
            st = None
    return out


def tickiness(pr):
    """一段剖面的「刻度纹理强度」= 高通后的标准差（门板/墙这些平滑区几乎没有）。"""
    d = np.asarray(pr, float)
    if len(d) < 12:
        return 0.0
    return float((d - np.convolve(d, np.ones(9) / 9.0, mode="same")).std())


def detect_span(gray, y0, y1, x0, x1, min_len=60, margins=(25.0, 12.0, 6.0),
                min_frac=0.08):
    """在给定的行带里找**卷尺的横向范围**。

    ⚠️ 光靠"亮 + 长"挑不出来：实测 3 m 档门板的高光比尺子更亮更长。
    判据用**刻度纹理性**（`tickiness`，高通标准差）—— 尺子有周期性刻度，门板没有。
    """
    band = gray[y0:y1 + 1, x0:x1].mean(0)
    xm = float(band.mean())
    # ⚠️ 先在**平滑**后的剖面上找连续段：刻度本身会把亮尺切成一段段 ~1 cm 的碎块，
    #    不平滑就永远拼不出一条长段（实测 1 m 档：最长段只有 11 px）。
    k = 21
    sm = np.convolve(band, np.ones(k) / k, mode="same")
    cands = []
    for margin in margins:
        m = sm > xm + 0.6 * margin
        cands += _runs(m, max(min_len, int(min_frac * (x1 - x0))))
        if cands:
            break
    best, best_t = None, -1.0
    for a, b in cands:
        t = tickiness(band[a:b])
        if t > best_t:
            best, best_t = (a, b), t
    if best is None:
        return None
    return (x0 + best[0], x0 + best[1])


def detect_band(gray, x0, x1, min_len=40):
    """自动找卷尺所在行带：在 x 范围内找"连续亮像素最长"的那一行，再向上下扩展。

    返回 `(y0, y1)`；找不到返回 `None`。
    """
    h = gray.shape[0]
    best = (0, None)
    # ⚠️ 阈值必须用**全图**均值：若亮带占满整行，用行均值当阈值会把自己排除掉
    #    （合成图自检直接踩到：整行 210，阈值 225，一个像素都不算"亮"）。
    # ⚠️ 阈值要用**全图**统计：亮带占满整行时，用行均值当阈值会把整条带自己排除掉。
    #    带子偏暗（远距档）时就逐步放宽重试。
    xm = float(gray[:, x0:x1].mean())
    for margin in (25.0, 12.0, 6.0):
        gthr = xm + margin
        best = (0, None)
        for y in range(h):
            cnt = int((gray[y, x0:x1] > gthr).sum())
            if cnt > best[0]:
                best = (cnt, y)
        if best[1] is not None and best[0] >= min_len:
            break
    if best[1] is None or best[0] < min_len:
        return None
    gthr = xm + margin
    y_c = best[1]
    y0 = y1 = y_c
    thr = max(gthr, float(gray[y_c, x0:x1].mean()) + 6.0)
    while y0 > 0 and float(gray[y0 - 1, x0:x1].mean()) > thr - 25:
        y0 -= 1
    while y1 < h - 1 and float(gray[y1 + 1, x0:x1].mean()) > thr - 25:
        y1 += 1
    return (y0, y1)


# ------------------------------------------------------------------ 图像 → 记录
def analyze_x(gray, band, xr, cx, z_m, c_m, grad_m, min_prom):
    """在一个给定的 x 范围里做一次完整估计；不可信/失败返回 None。"""
    y0, y1 = band
    pr = gray[y0:y1 + 1, xr[0]:xr[1]].mean(0)
    ticks = find_ticks(pr, min_prom=min_prom)
    fit = fit_index(ticks)
    if fit is None:
        return None
    p_fit, _k0 = period_at(fit, cx - xr[0])
    lo_fft = max(1.5, 300.0 * grad_m / max(abs(z_m), 1e-6))
    hi_fft = min((xr[1] - xr[0]) / 8.0, 3500.0 * grad_m / max(abs(z_m), 1e-6), 40.0)
    p_fft = fft_period(pr, lo=lo_fft, hi=hi_fft) if hi_fft > lo_fft * 1.5 else None
    dis = (abs(p_fit - p_fft) / p_fft) if p_fft else 9.99
    return {"x": [int(xr[0]), int(xr[1])], "n_ticks": int(len(ticks)),
            "n_used": int(fit["n"]), "rms_px": round(float(fit["rms"]), 3),
            "period_fit_px": round(float(p_fit), 4),
            "period_fft_px": (round(float(p_fft), 4) if p_fft else None),
            "disagree": float(dis),
            "period_med_px": round(float(fit["period_med"]), 4)}


def measure_one(path, z_m, c_m=0.0, grad_m=0.01, band=None, xr=None, cx=None,
                min_prom=0.6):
    """一张图 → dict（周期 / f / 质量），失败返回 None。

    横向范围试**两个候选**：用户/默认的整段，和自动收窄到尺子的那段。
    ⚠️ 谁更好用「刻度拟合与全局 FFT 是否一致 + 用到的刻度数」判，不能一刀切：
    实测 1 m 档整段更好（尺子几乎占满整幅），3 m 档自动收窄才对（门板纹理混进来）。
    """
    import cv2
    img = cv2.imread(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    fixed_x = xr is not None
    if xr is None:
        xr = (int(0.12 * w), int(0.78 * w))
    if band is None:
        band = detect_band(gray, xr[0], xr[1])
        if band is None:
            return None
    if cx is None:
        cx = w / 2.0

    cands = [tuple(xr)]
    if not fixed_x:
        sp = detect_span(gray, band[0], band[1], xr[0], xr[1])
        if sp is not None and sp[1] - sp[0] > 0.25 * (xr[1] - xr[0]):
            pad = int(0.03 * (sp[1] - sp[0]))
            c2 = (sp[0] + pad, sp[1] - pad)
            if c2[0] < cx < c2[1] and c2 != tuple(xr):
                cands.append(c2)

    results = [r for r in (analyze_x(gray, band, c, cx, z_m, c_m, grad_m, min_prom)
                           for c in cands) if r]
    if not results:
        return None
    # 首选"两个估计量一致"的；再比用到的刻度数（越多越稳）
    results.sort(key=lambda r: (round(r["disagree"], 2), -r["n_used"]))
    best = results[0]
    period = (best["period_fit_px"] if best["disagree"] <= 0.20
              else float(best["period_fft_px"] or best["period_fit_px"]))
    flag, quality = None, "ok"
    if best["disagree"] > 0.20:
        # 两个独立估计量都不一致 ⇒ **不报数**（宁可没有，也不要一个看着像样的错数）：
        # 实测 3 m 档会给出 f=1823（真值 ~1080）。要救这一档只能显式 --band/--x。
        quality = "reject"
        flag = ("刻度拟合(%.2f px) 与全局 FFT(%.2f px) 差 %.0f%% —— 刻度拟合多半锁到了杂线/"
                "块效应，改用 FFT 值（本档只能当参考；要更准就 --band/--x 显式给出尺子范围）"
                % (best["period_fit_px"], best["period_fft_px"],
                   100 * best["disagree"]))
    f_px = f_from_period(period, z_m, c_m, grad_m)
    if not (300.0 <= f_px <= 3500.0):
        quality = "reject"
        flag = (flag + "；" if flag else "") + "f=%.0f px 不在合理区间（300~3500）" % f_px
    return {"src": os.path.basename(path), "src_path": path, "cx": float(cx),
            "band": [int(band[0]), int(band[1])], "x": best["x"],
            "tried_x": [[int(c[0]), int(c[1])] for c in cands],
            "n_ticks": best["n_ticks"], "n_used": best["n_used"],
            "rms_px": best["rms_px"], "period_px": round(float(period), 4),
            "period_fit_px": best["period_fit_px"],
            "period_fft_px": best["period_fft_px"],
            "period_med_px": best["period_med_px"],
            "z_tape": float(z_m), "grade_m": float(grad_m),
            "f_px": round(float(f_px), 1), "flag": flag, "quality": quality}


def main(argv=None):
    ap = argparse.ArgumentParser(description="用卷尺刻度测等效焦距（不点鼠标）")
    ap.add_argument("images", nargs="+", help="靶子图（glob/目录/文件）")
    ap.add_argument("--z", type=float, required=True, help="该档卷尺读数(m)")
    ap.add_argument("--c", type=float, default=0.0, help="已知口径偏置(m)，默认 0")
    ap.add_argument("--grade-cm", type=float, default=1.0,
                    help="细刻度间隔(cm)，标准卷尺 = 1.0（改它会按比例缩放 f）")
    ap.add_argument("--band", default=None, help="强制行带 'y0,y1'（自动检测失败时用）")
    ap.add_argument("--x", default=None, help="强制 x 范围 'x0,x1'")
    ap.add_argument("--cx", type=float, default=None, help="光轴 x（默认图宽/2）")
    ap.add_argument("--out", default=None, help="把结果写成 ruler 记录 JSONL（可喂 ruler_calib）")
    ap.add_argument("--min-prom", type=float, default=0.6, help="刻度对比度门限（模糊图调小）")
    ap.add_argument("--force", action="store_true",
                    help="即使刻度拟合与 FFT 不一致也采用（默认跳过这些图）")
    a = ap.parse_args(argv)

    paths = []
    for pat in a.images:
        if os.path.isdir(pat):
            paths += sorted(glob.glob(os.path.join(pat, "*.jpg")) +
                            glob.glob(os.path.join(pat, "*.png")))
        else:
            paths += sorted(glob.glob(pat)) or [pat]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print("没找到图片")
        return 2
    band = tuple(int(v) for v in a.band.split(",")) if a.band else None
    xr = tuple(int(v) for v in a.x.split(",")) if a.x else None
    recs, fs = [], []
    print("z=%.2f m, 刻度=%.1f cm, %d 张" % (a.z, a.grade_cm, len(paths)))
    for p in paths:
        r = measure_one(p, a.z, c_m=a.c, grad_m=a.grade_cm / 100.0,
                        band=band, xr=xr, cx=a.cx, min_prom=a.min_prom)
        if r is None:
            print("  ✗ %s：找不到卷尺/刻度太少（试 --band / --x / 调小 --min-prom）"
                  % os.path.basename(p))
            continue
        if r["quality"] in ("fft_only", "reject"):
            print("  %s %s  ⚠️ %s" % ("✗" if r["quality"] == "reject" else "!", r["src"], r["flag"]))
            if r["quality"] == "reject" and not a.force:
                print("     （已跳过；确要采用请加 --force）")
                continue
        fs.append(r["f_px"])
        print("  %s  带 y=%s x=%s  刻度 %d/%d  rms=%.2f px  周期=%.3f px"
              "（FFT %.3f）  ⇒ f=%.0f px%s"
              % (r["src"], r["band"], r["x"], r["n_used"], r["n_ticks"], r["rms_px"],
                 r["period_fit_px"], r["period_fft_px"] or float("nan"), r["f_px"],
                 "  ⚠️ 已强制采用" if (r["flag"] and a.force) else ""))
        recs.append({"kind": "ruler", "method": "tick_period", "src": r["src"],
                     "src_path": r["src_path"], "span_m": a.grade_cm / 100.0,
                     "z_tape": a.z, "du": r["period_px"], "theta_deg": 0.0, "skew": 0.0,
                     "ticks": "%.0fcm刻度周期" % a.grade_cm,
                     "quality": r["quality"],
                     "note": "tape_ticks: 带%s rms=%.2f n=%d 拟合%.2f FFT%.2f"
                             % (r["band"], r["rms_px"], r["n_used"],
                                r["period_fit_px"], r["period_fft_px"] or float("nan"))})
    if not fs:
        return 2
    fs = np.asarray(fs, float)
    print("⇒ f = %.0f ± %.0f px（%.1f%%）" % (fs.mean(), fs.std(), 100 * fs.std() / fs.mean()))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("→ 已写 %d 条 ruler 记录：%s（可直接喂 ruler_calib.py）" % (len(recs), a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
