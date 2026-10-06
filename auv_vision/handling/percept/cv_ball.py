# -*- coding: utf-8 -*-
"""cv_ball.py — 夹取小球（任务三）的**纯 CV 红球检测**（不走 BPU）。
（详细用法、判据与实测见 doc/注释历史.md）"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

try:            # 允许脱离工程单独 import（PC 侧离线验证 / 单测）
    import base.cfg.settings as S
except Exception:                                     # pragma: no cover
    S = None


def _cfg(path, default):
    """读 `vision.<path>`（如 `grab.cv.dom_min`）；缺配置或缺 settings 用时用代码默认值。

    ⚠️ **必须带 `vision.` 前缀**：`S.get()` 只认 `vision.`/`comm.` 开头的路径，漏了前缀会
    **静默返回默认值**。本文件与 `grab_detector.py` 原来就漏了 ⇒ `cfg/vision.yaml` 里整个
    `grab.*` 段（红球阈值/跟踪参数/kind/mode）**一直没生效**，只因 yaml 值与代码默认值恰巧
    相同才没暴露（2026-10-06 修；`tests/tasks/handling/test_grab_targets.py` 有用例钉住）。
    """
    if S is None:
        return default
    p = path if path.startswith(("vision.", "comm.")) else "vision." + path
    v = S.get(p, None)
    return default if v is None else v


# 目标色 → cfg 前缀：红球沿用 `grab.cv`（历史工作点，单一来源，别搬）
_TARGET_ALIAS = {"red": "grab.cv."}
# 非红目标色必须显式写出这几个键（缺一个就不认，见 `Params.from_target`）
_TARGET_REQUIRED = ("dom_min", "s_min", "v_min")


@dataclass
class Params:
    """全部可调项。默认值 = PC 侧在 700 帧上扫参 + 板端复核后的工作点。"""

    # —— 颜色 ——
    dom_min: int = 30          # R - max(G,B) 下限（绝对红占优；对衰减最鲁棒）
    use_rel: bool = True       # 叠加"相对红占优"通道（救暗/远的球）
    rel_min: float = 0.30      # (R-max(G,B))/(R+max(G,B)+1) 下限
    s_min: int = 40            # HSV 饱和度下限
    v_min: int = 60            # HSV 亮度下限
    use_hue: bool = True       # 色相门限（挡橙色导轨）
    h_lo: int = 158            # 红带 1（高段）H ∈ [h_lo, h_hi]
    h_hi: int = 179
    h_hi2: int = 10            # 红带 2（低段）H ∈ [0, h_hi2]
    # —— 形态学 ——
    close_k: int = 9           # 闭运算核（并小碎块）
    open_k: int = 5            # 开运算核（去椒盐）
    fill_hole_frac: float = 0.65   # 只填比父轮廓小这么多的孔（高光斑）
    # —— 几何过滤 ——
    min_area: int = 30         # 掩膜面积下限（px²）
    max_area_frac: float = 0.75
    circ_min: float = 0.30     # 4πA/P² 圆度下限
    hull_fill_min: float = 0.55   # A / hull_area 下限（凸性）
    cov_min: float = 0.35      # 拟合圆被红色掩膜覆盖的比例下限
    min_arc_deg: float = 150.0  # 拟合边界点的最小张角（防"细长条拟出大圆"）
    r_min: float = 25.0        # 半径范围（px）；<30px 的检出实测是池底图案红点
    r_max: float = 900.0
    score_min: float = 0.0     # 分数下限（默认关：它是 circ×cov×arc 的乘积，会重复惩罚遮挡球）
    # —— 鲁棒圆拟合 ——
    fit_iters: int = 6
    fit_sigma: float = 1.8
    # —— 去重 ——
    merge_iou: float = 0.6
    # —— 可选 Hough 精修 ——
    refine: bool = False
    hough_param2: float = 28.0
    hough_r_tol: float = 1.45
    # —— 实现选择 ——
    fast_mask: bool = True     # True=uint8 SIMD 快路径（板端必需）；False=numpy 参考实现

    @classmethod
    def from_cfg(cls, prefix: str = "grab.cv.") -> "Params":
        """从 `cfg/vision.yaml` 取 `prefix*` 下的键（缺项用代码默认 = **红球的工作点**）。

        ⚠️ 默认值就是红的 ⇒ 拿它去读**别的颜色**的节点（如 `grab.targets.pink.`）会把红阈值
        当兜底用，等于悄悄用红判据认粉球。别的颜色**一律走 `from_target()`**（那里有闸）。
        """
        p = cls()
        for name in p.__dataclass_fields__:
            setattr(p, name, _cfg(prefix + name, getattr(p, name)))
        return p

    @classmethod
    def from_target(cls, name) -> "Params | None":
        """按目标色名取判据：`"red"` → `grab.cv`（700 帧已标定）；其它 → `grab.targets.<name>`。

        **未 `enable` / 未 `calibrated` / 关键键没写全 → None**（当成"这个颜色还不存在"）。
        这是"未知不许编"的机械保障：粉球/黄球的阈值标出来之前，它们**不会**参与检测，
        更不会拿红球的默认值冒充。
        """
        nm = str(name).strip().lower()
        if nm in _TARGET_ALIAS:
            return cls.from_cfg(_TARGET_ALIAS[nm])
        node = _cfg("grab.targets.%s" % nm, None)
        if not isinstance(node, dict) or not node.get("enable") or not node.get("calibrated"):
            return None
        for k in _TARGET_REQUIRED:
            if node.get(k) is None:
                return None                      # 关键键没写全 = 还没标定，别猜
        return cls.from_cfg("grab.targets.%s." % nm)


@dataclass
class Detection:
    """一个球的检出：圆心 + 半径 + 质量指标（原图像素坐标）。"""

    cx: float
    cy: float
    r: float
    score: float
    area: int
    circ: float
    cov: float
    arc_deg: float = 0.0
    border_frac: float = 0.0     # 拟合圆落在画面外的面积比（>0 = 贴边/出画）
    bbox: tuple = field(default_factory=tuple)   # (x1,y1,x2,y2) 外接方框

    @property
    def center(self):
        return (self.cx, self.cy)

    def as_dict(self):
        d = asdict(self)
        d["bbox"] = [float(v) for v in self.bbox]
        return d


_KERN_CACHE: dict = {}
_REL_LUT_CACHE: dict = {}


def _KERNEL(k: int):
    kern = _KERN_CACHE.get(k)
    if kern is None:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        _KERN_CACHE[k] = kern
    return kern


def _rel_lut(rel_min: float) -> np.ndarray:
    """相对红占优判据的 **256 项 LUT**：按 mx 查出 dom 的**严格下限**。"""
    key = round(float(rel_min), 6)
    lut = _REL_LUT_CACHE.get(key)
    if lut is None:
        from fractions import Fraction
        fr = Fraction(float(rel_min)).limit_denominator(1000)   # 0.30 -> 3/10
        a, b = fr.numerator, fr.denominator
        mx = np.arange(256, dtype=np.int64)
        num = a * (2 * mx + 1)          # t·(2mx+1)（两边同乘 b）
        den = b - a                     # 1 - t
        need = (num + den - 1) // den if den > 0 else np.full(256, 256)
        lut = np.clip(need - 1, 0, 255).astype(np.uint8)
        _REL_LUT_CACHE[key] = lut
    return lut


def active_targets():
    """`cfg grab.targets.active` 里**真正可用**的目标色名（未标定的直接丢弃）。

    现在默认只有 `["red"]`（用户 2026-10-06：先用红球测逻辑与运动，"红球外的可以都毙了"）。
    比赛用粉球/黄球 ⇒ 把它们的阈值标出来、把 `enable/calibrated` 翻真，再改 `active`。
    """
    names = _cfg("grab.targets.active", ["red"])
    if isinstance(names, str):
        names = [names]
    out = []
    for n in (names or []):
        if Params.from_target(n) is not None:
            out.append(str(n).strip().lower())
    return out


def red_mask_fast(img, p):
    """uint8 SIMD 快路径（板端用）：判据与 `red_mask_ref` 逐格等价（单测守着）。"""
    b, g, r = cv2.split(img)                       # uint8，无 astype
    mx = cv2.max(g, b)
    dom = cv2.subtract(r, mx)                      # 饱和减 = max(0, r-max(g,b))
    redness = dom
    m = cv2.compare(dom, p.dom_min, cv2.CMP_GE)    # 0/255
    if p.use_rel:
        thr = cv2.LUT(mx, _rel_lut(p.rel_min))     # 每个 mx 对应的 dom 严格下限
        m = cv2.bitwise_or(m, cv2.compare(dom, thr, cv2.CMP_GT))

    if p.use_hue or p.s_min > 0 or p.v_min > 0:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if p.use_hue:
            band = cv2.bitwise_or(cv2.inRange(hsv[:, :, 0], p.h_lo, p.h_hi),
                                  cv2.inRange(hsv[:, :, 0], 0, p.h_hi2))
            m = cv2.bitwise_and(m, band)
        if p.s_min > 0:
            m = cv2.bitwise_and(m, cv2.inRange(hsv[:, :, 1], p.s_min, 255))
        if p.v_min > 0:
            m = cv2.bitwise_and(m, cv2.inRange(hsv[:, :, 2], p.v_min, 255))

    return _morph_roi(m, p.open_k, p.close_k), redness


def red_mask_ref(img, p):
    """numpy **参考实现**（PC 上调参与 A/B 用；A55 上 68.6 ms/帧，别拿去板端跑）。"""
    b, g, r = cv2.split(img.astype(np.int16))
    mx = np.maximum(g, b)
    dom = r - mx
    redness = np.clip(dom, 0, 255).astype(np.uint8)

    m = dom >= p.dom_min
    if p.use_rel:
        rel = dom.astype(np.float32) / (r + mx + 1)
        m |= rel >= p.rel_min
    if p.s_min > 0 or p.v_min > 0 or p.use_hue:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if p.use_hue:
            band = ((h >= p.h_lo) & (h <= p.h_hi)) | (h <= p.h_hi2)
            m &= band
        if p.s_min > 0:
            m &= s >= p.s_min
        if p.v_min > 0:
            m &= v >= p.v_min
    mask = (m.astype(np.uint8)) * 255

    if p.open_k > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL(p.open_k))
    if p.close_k > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL(p.close_k))
    return mask, redness


def red_mask(img, p):
    """返回 (mask 0/255 uint8, redness)。`p.fast_mask` 决定走哪条实现。"""
    if p.fast_mask:
        return red_mask_fast(img, p)
    return red_mask_ref(img, p)


def _morph_roi(mask: np.ndarray, open_k: int, close_k: int) -> np.ndarray:
    """形态学只在"掩膜 bbox + margin"的 ROI 里算：结果与全图**逐位相同**，便宜 ~30×。"""
    if open_k <= 0 and close_k <= 0:
        return mask
    h, w = mask.shape
    k = max(open_k, close_k)
    small = cv2.resize(mask, (max(1, w // 8), max(1, h // 8)),
                       interpolation=cv2.INTER_AREA)
    ys, xs = np.nonzero(small)
    if ys.size == 0:
        return mask
    mg = k + 2
    x0 = max(0, int(xs.min()) * 8 - mg); x1 = min(w, (int(xs.max()) + 1) * 8 + mg)
    y0 = max(0, int(ys.min()) * 8 - mg); y1 = min(h, (int(ys.max()) + 1) * 8 + mg)
    roi = mask[y0:y1, x0:x1]
    if open_k > 0:
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, _KERNEL(open_k))
    if close_k > 0:
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, _KERNEL(close_k))
    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = roi
    return out


def fill_highlights(mask: np.ndarray, frac: float) -> np.ndarray:
    """按 contour 层级填"高光孔"：只填面积 < frac × 父轮廓 的子轮廓。"""
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return mask
    hier = hier[0]
    out = mask.copy()
    for i, c in enumerate(cnts):
        if hier[i][3] == -1:               # 只看孔（有父轮廓的）
            continue
        a, ap = cv2.contourArea(c), cv2.contourArea(cnts[hier[i][3]])
        if ap > 0 and a < frac * ap:
            cv2.drawContours(out, [c], -1, 255, -1)
    return out


def fit_circle_robust(pts: np.ndarray, iters: int, sigma: float):
    """Kasa 代数拟合 + 迭代剔离群点。返回 (cx, cy, r, n_inlier, rms, inlier_pts) 或 None。"""
    if len(pts) < 5:
        return None
    P = pts.astype(np.float64)
    best = None
    for it in range(max(1, iters)):
        x, y = P[:, 0], P[:, 1]
        A = np.c_[2 * x, 2 * y, np.ones(len(P))]
        bb = x ** 2 + y ** 2
        try:
            sol, *_ = np.linalg.lstsq(A, bb, rcond=None)
        except np.linalg.LinAlgError:
            return best
        cx, cy = sol[0], sol[1]
        r2 = sol[2] + cx ** 2 + cy ** 2
        if r2 <= 0:
            return best
        r = float(np.sqrt(r2))
        res = np.abs(np.hypot(x - cx, y - cy) - r)
        rms = float(np.sqrt((res ** 2).mean()))
        best = (float(cx), float(cy), r, len(P), rms, P.copy())
        if it == iters - 1:
            break
        thr = max(sigma * rms, 1.0)
        keep = res <= thr
        if keep.sum() < 6 or keep.all():
            break
        P = P[keep]
    return best


def arc_span_deg(pts: np.ndarray, cx: float, cy: float) -> float:
    """边界点相对圆心的角张角（度）：360 - 最大角度空档。"""
    if len(pts) < 3:
        return 0.0
    ang = np.sort(np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)) % 360.0)
    gaps = np.diff(np.r_[ang, ang[0] + 360.0])
    return float(360.0 - gaps.max())


def border_frac(cx: float, cy: float, r: float, w: int, h: int, n: int = 720) -> float:
    """拟合圆落在画面外的面积比（球贴边/半个球出画时 >0）。"""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False).reshape(1, -1)
    rr = np.linspace(0.02, 1.0, 40).reshape(-1, 1)
    xs = cx + (rr * r) * np.cos(a)
    ys = cy + (rr * r) * np.sin(a)
    return float(((xs < 0) | (xs >= w) | (ys < 0) | (ys >= h)).mean())


def circle_coverage(mask: np.ndarray, cx: float, cy: float, r: float) -> float:
    """拟合圆内被红色掩膜覆盖的比例（全盘≈0.95+，月牙≈0.5，乱拟合很低）。"""
    h, w = mask.shape
    x1, y1 = max(0, int(cx - r) - 1), max(0, int(cy - r) - 1)
    x2, y2 = min(w, int(cx + r) + 2), min(h, int(cy + r) + 2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    sub = mask[y1:y2, x1:x2]
    yy, xx = np.ogrid[y1:y2, x1:x2]
    disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    n = int(disc.sum())
    if n == 0:
        return 0.0
    return float((sub[disc] > 0).sum()) / n


def _circ_iou(a: Detection, b: Detection) -> float:
    d = np.hypot(a.cx - b.cx, a.cy - b.cy)
    if d >= a.r + b.r:
        return 0.0
    if d <= abs(a.r - b.r):
        return (min(a.r, b.r) ** 2) / (max(a.r, b.r) ** 2)
    r1, r2 = a.r, b.r
    a1 = r1 ** 2 * np.arccos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
    a2 = r2 ** 2 * np.arccos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
    a3 = 0.5 * np.sqrt(max(0.0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)))
    inter = a1 + a2 - a3
    union = np.pi * (r1 ** 2 + r2 ** 2) - inter
    return float(inter / union) if union > 0 else 0.0


def merge_overlaps(dets: list, iou_thr: float) -> list:
    """同一球被遮挡切成几块时会出多个圆 → 按 score 贪心合并（保留高分者）。"""
    dets = sorted(dets, key=lambda d: -d.score)
    keep: list = []
    for d in dets:
        if any(_circ_iou(d, k) > iou_thr for k in keep):
            continue
        keep.append(d)
    return keep


def detect(img: np.ndarray, p: Params) -> list:
    """整图（或任意 ROI）红球检测 → `[Detection]`（按 score 降序，已去重）。"""
    mask, redness = red_mask(img, p)
    filled = fill_highlights(mask, p.fill_hole_frac)
    H, W = img.shape[:2]
    n, lab, stats, _ = cv2.connectedComponentsWithStats(filled, 8)
    out: list = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < p.min_area or area > p.max_area_frac * H * W:
            continue
        x0, y0 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        comp = (lab[y0:y0 + bh, x0:x0 + bw] == i).astype(np.uint8)

        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        per = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (per ** 2) if per > 0 else 0.0
        hull_a = cv2.contourArea(cv2.convexHull(c))
        hull_fill = area / hull_a if hull_a > 0 else 0.0
        if circ < p.circ_min or hull_fill < p.hull_fill_min:
            continue

        pts = c[:, 0, :].astype(np.float64) + [x0, y0]     # 轮廓点搬到全图坐标
        if len(pts) > 400:                                  # 控点数，省时间
            pts = pts[:: int(np.ceil(len(pts) / 400))]
        fit = fit_circle_robust(pts, p.fit_iters, p.fit_sigma)
        if fit is None:
            continue
        cx, cy, r, _n_in, _rms, in_pts = fit
        if not (p.r_min <= r <= p.r_max):
            continue
        arc = arc_span_deg(in_pts, cx, cy)
        if arc < p.min_arc_deg:
            continue
        cov = circle_coverage(mask, cx, cy, r)
        if cov < p.cov_min:
            continue
        bfrac = border_frac(cx, cy, r, W, H)

        if p.refine:                                        # 可选：ROI 内 HoughCircles 精修
            rr = int(max(4, r * p.hough_r_tol))
            sub = redness[max(0, int(cy) - rr):int(cy) + rr,
                          max(0, int(cx) - rr):int(cx) + rr]
            if sub.size:
                sub = cv2.GaussianBlur(sub, (5, 5), 0)
                cir = cv2.HoughCircles(sub, cv2.HOUGH_GRADIENT, dp=1.2, minDist=rr,
                                       param1=100, param2=p.hough_param2,
                                       minRadius=int(max(4, r / p.hough_r_tol)),
                                       maxRadius=int(r * p.hough_r_tol))
                if cir is not None and len(cir[0]):
                    hx, hy, hr = cir[0][0]
                    hx += max(0, int(cx) - rr); hy += max(0, int(cy) - rr)
                    if circle_coverage(mask, hx, hy, hr) >= p.cov_min:
                        cx, cy, r = float(hx), float(hy), float(hr)
                        bfrac = border_frac(cx, cy, r, W, H)

        score = float(np.clip(circ, 0, 1) * np.clip(cov, 0, 1) *
                      np.clip(arc / 280.0, 0, 1) * min(1.0, 0.35 + area / 2500.0))
        if score < p.score_min:
            continue
        out.append(Detection(cx=cx, cy=cy, r=r, score=score, area=area,
                             circ=float(circ), cov=float(cov), arc_deg=arc,
                             border_frac=bfrac,
                             bbox=(float(x0), float(y0), float(x0 + bw), float(y0 + bh))))
    return merge_overlaps(out, p.merge_iou)


def best(dets: list):
    """取最可信的一个（先按 score 排序；平手时取大的）。"""
    if not dets:
        return None
    return max(dets, key=lambda d: (d.score, d.r))


def debug_maps(img: np.ndarray, p: Params):
    """中间量可视化（离线核对阈值用）：mask / filled / redness / vis。"""
    mask, redness = red_mask(img, p)
    filled = fill_highlights(mask, p.fill_hole_frac)
    vis = img.copy()
    for d in detect(img, p):
        cv2.circle(vis, (int(round(d.cx)), int(round(d.cy))), int(round(d.r)), (0, 255, 0), 2)
        cv2.drawMarker(vis, (int(round(d.cx)), int(round(d.cy))), (0, 255, 255),
                       cv2.MARKER_CROSS, 14, 2)
        cv2.putText(vis, "r=%.0f s=%.2f" % (d.r, d.score),
                    (int(d.cx - d.r), int(d.cy - d.r - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return {"mask": mask, "filled": filled, "redness": redness, "vis": vis}
