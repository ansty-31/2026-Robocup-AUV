#!/usr/bin/env python3
"""夹取小球 · 方案 A：传统 CV 红色小球检测（可调参、可解释）。

**它不用任何模型**：靠"红色占优 + 圆 + 质心/半径几何一致"检出小球，输出圆（cx, cy, r, score）。
水下场景的难点与对应设计（每条都对应代码里的一段）：

| 难点 | 现象（本数据集实测） | 对策 |
|---|---|---|
| 水下红光衰减 | 小/远球的 S、V 都掉，纯色阈值会漏 | 主判据用**绝对红占优** `R - max(G,B)`，H/S/V 只作辅助门限 |
| 橙色框架干扰 | 画面里那圈橙色导轨 H≈20~30 | 色相门限卡在 `[h_lo,h_hi]∪[h_lo2,h_hi2]`（实测球心 H≈176），把橙挡在外面 |
| 镜面高光打穿球心 | 大红球中间一块亮斑把红色区域切成环 | 按 contour 层级只填"面积 < fill_hole_frac × 父轮廓"的孔（高光），保留真孔 |
| 被夹爪/黑条遮挡成月牙 | 画面里小球常常只剩一条红边 | **对掩膜边界做鲁棒圆拟合**（迭代剔残差大的点）；月牙的弧仍是真圆的一段，直边被当离群点剔掉 |
| 水面焦散噪声 | 背景出现随机红斑（实测背景红色像素 0.8%） | 面积/圆度/填充率下限 + 可选跨帧一致性 |

对外只暴露两个函数：`Params`（全部可调）与 `detect(img, p)` → `[Detection]`。
`debug_maps(img, p)` 返回掩膜等中间图，用来肉眼核对阈值（不要靠猜）。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# 参数
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    # —— 颜色 ——
    dom_min: int = 55          # R - max(G,B) 下限（绝对红占优；对衰减最鲁棒）
    fast_mask: bool = True     # True=用 cv2 SIMD 快路径（板端必需）；False=numpy 参考实现
    use_rel: bool = True       # 叠加"相对红占优"通道（救暗/远的球）
    rel_min: float = 0.30      # (R-max(G,B))/(R+max(G,B)) 下限
    s_min: int = 100           # HSV 饱和度下限
    v_min: int = 40            # HSV 亮度下限
    use_hue: bool = True       # 是否启用色相门限（挡橙色框架）
    h_hi: int = 179            # 红带 1（高段）H ∈ [h_lo, h_hi]
    h_lo: int = 158
    h_hi2: int = 10            # 红带 2（低段）H ∈ [0, h_hi2]
    # —— 形态学 ——
    close_k: int = 5           # 闭运算核（并小碎块）
    open_k: int = 3            # 开运算核（去椒盐）
    fill_hole_frac: float = 0.65   # 只填比父轮廓小这么多的孔（高光斑），防填掉真背景
    # —— 几何过滤 ——
    min_area: int = 50         # 掩膜面积下限（px²）
    max_area_frac: float = 0.75  # 掩膜面积 / 图面积 上限
    circ_min: float = 0.40     # 4πA/P² 圆度下限
    hull_fill_min: float = 0.55   # A / hull_area 下限（凸性）
    cov_min: float = 0.42      # 拟合圆被红色掩膜覆盖的比例下限（月牙≈0.5，全盘≈0.95）
    min_arc_deg: float = 150.0  # 拟合所用边界点的最小张角（防"细长条拟合出巨大的圆"）
    r_min: float = 25.0        # 半径下限（px）。实测 <30px 的检出是池底图案红点/瓦缝红光，见 sheet
    r_max: float = 900.0
    score_min: float = 0.25    # 分数下限。实测那批假小球的 score ≤0.31 且 r<30，双管齐下
    # —— 鲁棒圆拟合 ——
    fit_iters: int = 6
    fit_sigma: float = 1.8
    # —— 去重 ——
    merge_iou: float = 0.6
    # —— 精修（可选，用 ROI 内 HoughCircles 再找一次圆）——
    refine: bool = False
    hough_param2: float = 28.0
    hough_r_tol: float = 1.45   # Hough 半径搜索范围 = 初估 r × [1/tol, tol]


@dataclass
class Detection:
    cx: float
    cy: float
    r: float
    score: float
    area: int
    circ: float
    cov: float
    arc_deg: float = 0.0
    border_frac: float = 0.0     # 拟合圆落在画面外的面积比例（>0 说明被边缘截断）
    bbox: tuple = field(default_factory=tuple)   # x1,y1,x2,y2（原图坐标）

    def as_dict(self):
        d = asdict(self)
        d["bbox"] = [float(v) for v in self.bbox]
        return d


# --------------------------------------------------------------------------- #
# 步骤 1：红色掩膜
# --------------------------------------------------------------------------- #
def red_mask(img: np.ndarray, p: Params) -> tuple[np.ndarray, np.ndarray]:
    """返回 (mask, redness)。redness = R - max(G,B) 的 uint8 可视化，便于精修/调试。

    `p.fast_mask=True` 走 `red_mask_fast()`（板端用，等价判据、cv2 SIMD）；
    False 走下面这段 numpy 参考实现（PC 上调参时的原始版本，留着做 A/B 对照）。"""
    if p.fast_mask:
        return red_mask_fast(img, p)
    return red_mask_ref(img, p)


def red_mask_ref(img: np.ndarray, p: Params) -> tuple[np.ndarray, np.ndarray]:
    """numpy 参考实现（原版）。在 A55 上 68.6 ms/帧，**不要拿去板端跑**。

    红色判据 = 绝对红占优(dom ≥ dom_min) **或** 相对红占优(rel ≥ rel_min)，
    后者专治水下衰减：球的 R 只有 40 但 G/B 更低时绝对差会掉到阈值以下，
    比值却仍然很高（实测暗球 rel≈0.3~0.6，背景蓝绿 rel<0）。"""
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
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.open_k,) * 2))
    if p.close_k > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.close_k,) * 2))
    return mask, redness


_REL_LUT_CACHE: dict = {}


def _rel_lut(rel_min: float) -> np.ndarray:
    """相对红占优判据的 **256 项 LUT**（按 mx 查出所需的最小 r）。

    要判的是 `(r-mx)/(r+mx+1) >= t`，即 `(1-t)r >= (1+t)mx + t`，整理成 `r >= f(mx)`：
        f(mx) = ceil( ((1+t)*mx + t) / (1-t) )
    mx 是 uint8（0..255）=> f 只有 256 种取值 => 查表 + 一次整数比较即可，**与除法版逐位等价**
    （r 是整数：`7r >= 13mx+3` <=> `r >= ceil((13mx+3)/7)`）。t=0.30 时 f 在 mx>137 后饱和到 255
    => 那些像素直接判否（r<=255 永不满足），与除法版一致。

    为什么绕这一步：板端 A55 上 `cv2.multiply(..., dtype=CV_16U)` 是**标量循环**，
    实测"2 乘 + 1 加 + 1 比较"要 **30.3 ms**；换成 LUT+比较（都 uint8 SIMD）约 **2 ms**。
    """
    key = round(float(rel_min), 6)
    lut = _REL_LUT_CACHE.get(key)
    if lut is None:
        mx = np.arange(256, dtype=np.float64)
        t = float(rel_min)
        need = np.ceil(((1.0 + t) * mx + t) / (1.0 - t))
        lut = np.clip(need, 0, 255).astype(np.uint8)
        _REL_LUT_CACHE[key] = lut
    return lut


def _morph_roi(mask: np.ndarray, open_k: int, close_k: int) -> np.ndarray:
    """形态学只在"掩膜 bbox + margin"的 ROI 里算：结果与全图**逐位相同**，但便宜 ~30x。

    红掩膜通常只占全图 <2%，而全图 open(5)+close(9) 在 A55 上要 **35.8 ms**（快路径最大单项）。
    正确性：open/close 的输出都落在"输入 bbox 外 k/2 之内"（两者都是 dilate 的子集），
    取 margin = k 必然覆盖；ROI 边界外的邻域在全图里同样是 0，故边界处结果也一致。
    bbox 由 1/8 缩略图求 nonzero 得到（`INTER_AREA` 是块均值，块内任一像素非零 => 均值>0，
    所以缩略图 bbox 是真实 bbox 的超集），在 ~14k 像素上求 nonzero 很便宜。
    """
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


def red_mask_fast(img: np.ndarray, p: Params) -> tuple[np.ndarray, np.ndarray]:
    """**板端快路径**：判据与 `red_mask_ref()` 逐位等价，但全部走 uint8 SIMD。

    板端实测（A55，1280x720）：

    | 步骤 | numpy 参考版 | 本函数 |
    |---|---|---|
    | `astype(int16)` + split + dom | 21.0 ms | `cv2.split/max/subtract/compare`（uint8） **6.8 ms** |
    | 相对红占优（float 除法） | 8.6 ms | `cv2.LUT` 查表 + `cv2.compare` **~2 ms** |
    | HSV 三通道全图比较 | 8.0 ms | `cv2.inRange`（SIMD） **~4 ms** |
    | 全图 open(5)+close(9) | 35.8 ms | ROI 内做 **~1-2 ms** |
    | red_mask 合计 | 68.6 ms | **~15 ms** |

    ⚠️ 曾试过"uint16 整数等价式 + np.nonzero 稀疏 HSV 索引"，在 A55 上**更慢（97.6 ms）**：
    `cv2.multiply(dtype=CV_16U)`、`np.nonzero`、花式索引在 ARM 上都是标量实现。
    教训：**ARM 上先用 uint8，别用 16 位、别用稀疏索引。**
    """
    b, g, r = cv2.split(img)                       # uint8，无 astype
    mx = cv2.max(g, b)
    dom = cv2.subtract(r, mx)                      # 饱和减 = max(0, r-max(g,b))
    redness = dom
    m = cv2.compare(dom, p.dom_min, cv2.CMP_GE)    # 0/255
    if p.use_rel:
        thr = cv2.LUT(mx, _rel_lut(p.rel_min))     # 每个 mx 对应的最小 r
        m = cv2.bitwise_or(m, cv2.compare(r, thr, cv2.CMP_GE))

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


_KERN_CACHE: dict = {}


def _KERNEL(k: int):
    kern = _KERN_CACHE.get(k)
    if kern is None:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        _KERN_CACHE[k] = kern
    return kern


def fill_highlights(mask: np.ndarray, frac: float) -> np.ndarray:
    """按 contour 层级填"高光孔"：只填面积 < frac × 父轮廓 的子轮廓（真背景孔比球还大，不会被填）。

    这一步是水下小球的关键：镜面高光把红球打成红环，不填的话圆度/质心全错。"""
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return mask
    hier = hier[0]
    out = mask.copy()
    for i, c in enumerate(cnts):
        if hier[i][3] == -1:          # 只看孔（有父轮廓的）
            continue
        parent = hier[i][3]
        a, ap = cv2.contourArea(c), cv2.contourArea(cnts[parent])
        if ap > 0 and a < frac * ap:
            cv2.drawContours(out, [c], -1, 255, -1)
    return out


# --------------------------------------------------------------------------- #
# 步骤 2：鲁棒圆拟合（对月牙/部分遮挡有效）
# --------------------------------------------------------------------------- #
def fit_circle_robust(pts: np.ndarray, iters: int, sigma: float):
    """Kasa 代数拟合 + 迭代剔除离群点。pts: (N,2) float。
    返回 (cx, cy, r, n_inlier, rms, inlier_pts) 或 None。"""
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
    """边界点相对圆心的角度分布：返回"最大空档之外"的角张角（度）。

    月牙/被遮挡球的有效边界只是一段弧 —— 弧本身一定是真圆的一段，所以拟合成立；
    但一条**细长条**的边界也能被最小二乘拟合成一个巨大的圆（残差还很小），
    这时张角很小。用它当门槛，专门拦这类"解不唯一"的退化拟合。"""
    if len(pts) < 3:
        return 0.0
    ang = np.sort(np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)) % 360.0)
    gaps = np.diff(np.r_[ang, ang[0] + 360.0])
    return float(360.0 - gaps.max())


def border_frac(cx: float, cy: float, r: float, w: int, h: int, n: int = 720) -> float:
    """拟合圆有多少比例落在画面外（球贴边/半个球出画时 >0）。用极坐标网格估算，够准且快。"""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False).reshape(1, -1)
    rr = np.linspace(0.02, 1.0, 40).reshape(-1, 1)
    xs = cx + (rr * r) * np.cos(a)
    ys = cy + (rr * r) * np.sin(a)
    outside = (xs < 0) | (xs >= w) | (ys < 0) | (ys >= h)
    return float(outside.mean())


def circle_coverage(mask: np.ndarray, cx: float, cy: float, r: float) -> float:
    """拟合圆内被红色掩膜覆盖的比例。全盘≈0.95+，月牙≈0.5，乱拟合会很低。"""
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


# --------------------------------------------------------------------------- #
# 步骤 3：检测
# --------------------------------------------------------------------------- #
def detect(img: np.ndarray, p: Params) -> list[Detection]:
    mask, redness = red_mask(img, p)
    filled = fill_highlights(mask, p.fill_hole_frac)
    H, W = img.shape[:2]
    n, lab, stats, cent = cv2.connectedComponentsWithStats(filled, 8)
    out: list[Detection] = []
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

        # 边界点（按轮廓采样，控制点数以省时间）
        pts = c[:, 0, :].astype(np.float64)
        pts = pts + [x0, y0]
        if len(pts) > 400:
            pts = pts[:: int(np.ceil(len(pts) / 400))]
        fit = fit_circle_robust(pts, p.fit_iters, p.fit_sigma)
        if fit is None:
            continue
        cx, cy, r, n_in, rms, in_pts = fit
        if not (p.r_min <= r <= p.r_max):
            continue
        arc = arc_span_deg(in_pts, cx, cy)
        if arc < p.min_arc_deg:
            continue
        cov = circle_coverage(mask, cx, cy, r)
        if cov < p.cov_min:
            continue
        bfrac = border_frac(cx, cy, r, W, H)

        if p.refine:
            rr = int(max(4, r * p.hough_r_tol))
            sub = redness[max(0, int(cy) - rr):int(cy) + rr, max(0, int(cx) - rr):int(cx) + rr]
            if sub.size:
                sub = cv2.GaussianBlur(sub, (5, 5), 0)
                cir = cv2.HoughCircles(sub, cv2.HOUGH_GRADIENT, dp=1.2,
                                       minDist=rr, param1=100, param2=p.hough_param2,
                                       minRadius=int(max(4, r / p.hough_r_tol)),
                                       maxRadius=int(r * p.hough_r_tol))
                if cir is not None and len(cir[0]):
                    hx, hy, hr = cir[0][0]
                    hx += max(0, int(cx) - rr); hy += max(0, int(cy) - rr)
                    if circle_coverage(mask, hx, hy, hr) >= p.cov_min:
                        cx, cy, r = float(hx), float(hy), float(hr)
                        bfrac = border_frac(cx, cy, r, W, H)

        # score：圆度 × 覆盖 × 弧张角 × 面积权重（小目标不因面积小被过分压低）
        score = float(np.clip(circ, 0, 1) * np.clip(cov, 0, 1) *
                      np.clip(arc / 280.0, 0, 1) *
                      min(1.0, 0.35 + area / 2500.0))
        if score < p.score_min:
            continue
        out.append(Detection(cx=cx, cy=cy, r=r, score=score, area=area,
                             circ=float(circ), cov=float(cov), arc_deg=arc,
                             border_frac=bfrac,
                             bbox=(float(x0), float(y0), float(x0 + bw), float(y0 + bh))))

    return merge_overlaps(out, p.merge_iou)


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


def merge_overlaps(dets: list[Detection], iou_thr: float) -> list[Detection]:
    """同一球被遮挡切成几块时会出多个圆 → 按 score 贪心合并重叠圆（保留高分者）。"""
    dets = sorted(dets, key=lambda d: -d.score)
    keep: list[Detection] = []
    for d in dets:
        if any(_circ_iou(d, k) > iou_thr for k in keep):
            continue
        keep.append(d)
    return keep


# --------------------------------------------------------------------------- #
# 调试用中间图
# --------------------------------------------------------------------------- #
def debug_maps(img: np.ndarray, p: Params):
    mask, redness = red_mask(img, p)
    filled = fill_highlights(mask, p.fill_hole_frac)
    dets = detect(img, p)
    vis = img.copy()
    for d in dets:
        cv2.circle(vis, (int(round(d.cx)), int(round(d.cy))), int(round(d.r)), (0, 255, 0), 2)
        cv2.drawMarker(vis, (int(round(d.cx)), int(round(d.cy))), (0, 255, 255),
                       cv2.MARKER_CROSS, 14, 2)
        cv2.putText(vis, f"r={d.r:.0f} s={d.score:.2f}", (int(d.cx - d.r), int(d.cy - d.r - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return {"mask": mask, "filled": filled, "redness": redness, "vis": vis}
