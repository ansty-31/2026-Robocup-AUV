#!/usr/bin/env python3
"""
去畸变实验（畸变对水下门框特征点识别的影响）—— 共享几何模块

本模块只做一件事：把「同一台相机、同一帧原始画面」在**三个坐标域**之间的
映射写得唯一、可核对，供预处理脚本与标注换算脚本共用。任何一处各写一份
remap，实验结论就不可信，所以全部走这里。

## 三个组 / 坐标域

| 组 | 目录 | 图像链路 | 模型看到的域 | 输出坐标所在域 |
|---|---|---|---|---|
| **A** 先去畸变 | `A_undistort_first/` | `remap@720p` → `resize640` → `enhance` | 去畸变域 | `a640` |
| **B** 不去畸变 | `B_no_undistort/`   | `resize640` → `enhance`               | 畸变域     | `b640` |
| **C** 最后去畸变（链路末端） | `C_undistort_last/` | `resize640` → `enhance` → `remap@640` | 去畸变域 | `c640` |
| **C′** 最后去畸变（坐标后映射） | 复用 B 的图 | 检测在 B 上进行，输出点再换算 | 畸变域 | `a640` |

- `raw`  : 原始 720p 畸变域像素（= 标定时的分辨率）
- `b640` : 640 畸变域 = `raw × S`，`S = diag(size/w, size/h)`
- `rect720` / `a640` : 720p 去畸变域 / 其 `× S`
- `c640` : 640 去畸变域（`K_sq = S·K`，`dist` 不变）

**板端是各向异性缩放**：`cv2.resize((1280,720) → (640,640))` 的
`S = diag(0.5, 0.8889)`，不是等比。所以 640 域的等效内参是 `K_sq = S·K`
（`dist` 不随缩放变化，它作用在归一化坐标上）。

## 一条推论（写实验结论时要用）

去畸变是**逐像素坐标映射**、`S` 是线性映射，两者在坐标上可交换：

    S ∘ undistort(K, newK720)  ==  undistort(S·K, S·newK720) ∘ S

`getOptimalNewCameraMatrix` 在 `alpha=0` 时取「去畸变后有效区域」的外接矩形，
该区域在此仿射变换下同样只被 `S` 缩放，于是 `newK640 == S · newK720`。
**结论：A 组与 C 组的几何是同一套**，差别只在**重采样顺序**（先在 720p 上插值
再降采样 → 细节保留多；先降到 640 再插值 → 在粗网格上插值）。所以 C 组回答的是
「去畸变放链路哪一段」，不是「去畸变算得对不对」。`check_equivalence()` 会打印偏差。

## ⚠️ 为什么不用 cv2.undistortPoints

`cv2.undistortPoints` 内部是固定 5 次的不动点迭代，在强畸变下会**不收敛并静默
返回垃圾**（实测本工程 `configs/front_camera.yaml`：原始像素 (0,360) 返回它自己，
(80,360) 却跳到 -525px）。本模块改用**解析畸变模型 + 向量化牛顿法求逆**：

    r_d = r_u · (1 + k1·r_u² + k2·r_u⁴ + k3·r_u⁶)     （径向，含切向项）

径向函数在 `r_u = r_turn` 处取极大 `r_d_max`；**`r_d > r_d_max` 的点在这个畸变
模型下没有逆解**（不是代码问题，是模型本身覆盖不到）。这类点返回 `valid=False`，
调用方必须显式处理，不许当成坐标用。`mappable_stats()` 会给出画面里可逆像素的占比。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # scripts/<stage>/<pkg>/x.py

GROUP_A = "A_undistort_first"
GROUP_B = "B_no_undistort"
GROUP_C = "C_undistort_last"
GROUP_CPRIME = "Cprime_map_after"
ALL_GROUPS = (GROUP_A, GROUP_B, GROUP_C)

GROUP_DESC = {
    GROUP_A: "先去畸变：remap@720p -> resize640 -> enhance（板端当前链路）",
    GROUP_B: "不去畸变：resize640 -> enhance",
    GROUP_C: "最后去畸变：resize640 -> enhance -> remap@640",
    GROUP_CPRIME: "最后去畸变（坐标后映射）：B 的图 -> 检测 -> 畸变域坐标换算到 a640",
}


def resolve_group(name: str) -> str:
    """命令行里允许用短名 A/B/C（大小写不敏感），也可以写全名。

    别名表：A = 先去畸变，B = 不去畸变，C = 最后去畸变（链路末端），
    C' = 最后去畸变（坐标后映射，只在评估脚本里作为一个"虚拟组"使用）。
    """
    s = str(name).strip()
    if s in GROUP_DESC:
        return s
    alias = {"A": GROUP_A, "B": GROUP_B, "C": GROUP_C, "C'": GROUP_CPRIME,
             "CPRIME": GROUP_CPRIME, "C+": GROUP_CPRIME}
    key = s.upper() if s.upper() != "C'" else s
    if key in alias:
        return alias[key]
    raise ValueError(f"未知组名 {name!r}，可用: A / B / C / C' 或全名 "
                     f"{list(GROUP_DESC)}")


# ------------------------------------------------------------------ 标定文件

def load_calibration(path: str | Path):
    """读 OpenCV FileStorage 标定 yaml，返回 (K, dist, (w, h) | None)"""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"标定文件无法打开: {path}")
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("distortion_coefficients").mat()
    size = None
    try:
        nw, nh = fs.getNode("image_width"), fs.getNode("image_height")
        if not nw.empty() and not nh.empty():
            size = (int(nw.real()), int(nh.real()))
    except Exception:  # noqa: BLE001 —— 老 yaml 可能没有这两个字段
        size = None
    fs.release()
    if K is None or dist is None:
        raise ValueError(f"标定文件缺少 camera_matrix / distortion_coefficients: {path}")
    return (np.asarray(K, np.float64),
            np.asarray(dist, np.float64).reshape(-1), size)


def _scale_K(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """各向异性缩放的等效内参：K' = diag(sx, sy, 1) · K"""
    return np.diag([sx, sy, 1.0]) @ np.asarray(K, np.float64)


# ------------------------------------------------------------------ 畸变模型

def _radial_terms(dist: np.ndarray):
    """拆分 5 / 8 / 12 参数畸变模型 → (k1,k2,k3,k4,k5,k6)

    参数顺序遵循 OpenCV：`(k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4]]])`。
    8 参数（`CALIB_RATIONAL_MODEL`）多出的 k4~k6 是**分母**，
    12 参数还多 4 个薄棱镜项（本模块不处理，只忽略并使用径向部分）。
    """
    d = np.asarray(dist, np.float64).reshape(-1)
    pad = np.zeros(6)
    pad[:min(6, d.size)] = d[:6]
    k1, k2, p1, p2, k3, k4 = pad
    k5 = d[6] if d.size > 6 else 0.0
    k6 = d[7] if d.size > 7 else 0.0
    return k1, k2, p1, p2, k3, k4, k5, k6


def distort_normalized(xu: np.ndarray, yu: np.ndarray, dist: np.ndarray):
    """归一化无畸变坐标 -> 归一化畸变坐标（OpenCV 标准模型，5/8 参数都支持）

    与 `cv2.initUndistortRectifyMap` / `cv2.projectPoints` 用的是同一套公式；
    `FrameGeometry.self_test()` 会与 `cv2.projectPoints(K=I)` 对拍验证。
    """
    k1, k2, p1, p2, k3, k4, k5, k6 = _radial_terms(dist)
    r2 = xu * xu + yu * yu
    num = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    den = 1.0 + r2 * (k4 + r2 * (k5 + r2 * k6))
    radial = num / den
    xd = xu * radial + 2.0 * p1 * xu * yu + p2 * (r2 + 2.0 * xu * xu)
    yd = yu * radial + p1 * (r2 + 2.0 * yu * yu) + 2.0 * p2 * xu * yu
    return xd, yd


def radial_curve(dist: np.ndarray, r_max: float = 3.0, n: int = 200001):
    """径向函数 g(r_u) 的采样，用于找极大（可逆半径上限）"""
    k1, k2, _, _, k3, k4, k5, k6 = _radial_terms(dist)
    r = np.linspace(0.0, r_max, n)
    r2 = r * r
    num = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    den = 1.0 + r2 * (k4 + r2 * (k5 + r2 * k6))
    return r, r * num / den


def invertible_limit(dist: np.ndarray, r_scan: float = 3.0):
    """返回 (r_turn, r_d_max)：径向函数取极大处的 r_u 与其值。

    `r_d <= r_d_max` 的畸变半径有逆解（取较小的那个根，即画面内那一支）。
    """
    r, g = radial_curve(dist, r_scan)
    i = int(np.argmax(g))
    return float(r[i]), float(g[i])


def undistort_normalized(xd: np.ndarray, yd: np.ndarray, dist: np.ndarray,
                         iters: int = 30, tol: float = 1e-12):
    """归一化畸变坐标 -> 归一化无畸变坐标（向量化牛顿法）

    返回 (xu, yu, valid)。`valid=False` 表示该点的畸变半径超出模型可逆范围，
    此时返回的坐标**没有意义**，调用方必须丢弃或另行处理。
    """
    xd = np.asarray(xd, np.float64).reshape(-1)
    yd = np.asarray(yd, np.float64).reshape(-1)
    n = xd.size
    if n == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0, bool)

    r_turn, r_d_max = invertible_limit(dist)
    rd = np.hypot(xd, yd)
    valid = rd <= r_d_max

    # 初值：先解径向方程（二分，取 [0, r_turn] 上较小的一支），方向沿用畸变方向
    r, g = radial_curve(dist, r_turn * 1.0000001, 400001)
    ru0 = np.interp(np.clip(rd, 0, g.max()), g, r)
    scale = np.where(rd > 1e-12, ru0 / np.maximum(rd, 1e-12), 1.0)
    xu, yu = xd * scale, yd * scale

    # 牛顿迭代（有限差分雅可比，向量化）
    h = 1e-6
    for _ in range(iters):
        fx, fy = distort_normalized(xu, yu, dist)
        ex, ey = fx - xd, fy - yd
        if float(np.max(np.hypot(ex, ey))) < tol:
            break
        j11, j21 = distort_normalized(xu + h, yu, dist)
        j12, j22 = distort_normalized(xu, yu + h, dist)
        a11, a21 = (j11 - fx) / h, (j21 - fy) / h
        a12, a22 = (j12 - fx) / h, (j22 - fy) / h
        det = a11 * a22 - a12 * a21
        bad = np.abs(det) < 1e-14
        det = np.where(bad, 1.0, det)
        dx = (a22 * ex - a12 * ey) / det
        dy = (-a21 * ex + a11 * ey) / det
        xu, yu = xu - dx, yu - dy
        valid &= ~bad

    fx, fy = distort_normalized(xu, yu, dist)
    valid &= np.hypot(fx - xd, fy - yd) < 1e-6
    return xu, yu, valid


def undistort_pixels(pts, K, dist, newK):
    """畸变域像素 -> 去畸变域像素。返回 (pts_rect, valid)"""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    if pts.size == 0:
        return np.zeros((0, 2)), np.zeros(0, bool)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xu = (pts[:, 0] - cx) / fx
    yu = (pts[:, 1] - cy) / fy
    xu, yu, valid = undistort_normalized(xu, yu, dist)
    x = newK[0, 0] * xu + newK[0, 1] * yu + newK[0, 2]
    y = newK[1, 0] * xu + newK[1, 1] * yu + newK[1, 2]
    return np.stack([x, y], axis=1), valid


# ------------------------------------------------------------------ 单分辨率几何

@dataclass
class FrameGeometry:
    """某个「原始分辨率 + 输出 size」下的全部映射与 remap 表。坐标均为 (N, 2) 像素。"""

    raw_w: int
    raw_h: int
    size: int
    alpha: float
    K: np.ndarray
    dist: np.ndarray
    K_sq: np.ndarray
    newK720: np.ndarray
    newK640: np.ndarray
    sx: float
    sy: float

    _maps_720_i16: tuple | None = None
    _maps_720_f32: tuple | None = None
    _maps_640_i16: tuple | None = None
    _maps_640_f32: tuple | None = None

    # ---- 图像 remap 表（CV_16SC2，与板端 common/preprocess.py 同款）----
    @property
    def maps_720_i16(self):
        if self._maps_720_i16 is None:
            self._maps_720_i16 = cv2.initUndistortRectifyMap(
                self.K, self.dist, None, self.newK720,
                (self.raw_w, self.raw_h), cv2.CV_16SC2)
        return self._maps_720_i16

    @property
    def maps_640_i16(self):
        if self._maps_640_i16 is None:
            self._maps_640_i16 = cv2.initUndistortRectifyMap(
                self.K_sq, self.dist, None, self.newK640,
                (self.size, self.size), cv2.CV_16SC2)
        return self._maps_640_i16

    # ---- 浮点 remap 表（用于「去畸变域 -> 畸变域」的点采样，即 remap 的逆）----
    @property
    def maps_720_f32(self):
        if self._maps_720_f32 is None:
            self._maps_720_f32 = cv2.initUndistortRectifyMap(
                self.K, self.dist, None, self.newK720,
                (self.raw_w, self.raw_h), cv2.CV_32FC1)
        return self._maps_720_f32

    @property
    def maps_640_f32(self):
        if self._maps_640_f32 is None:
            self._maps_640_f32 = cv2.initUndistortRectifyMap(
                self.K_sq, self.dist, None, self.newK640,
                (self.size, self.size), cv2.CV_32FC1)
        return self._maps_640_f32

    # ---------------- 坐标域换算 ----------------
    #
    # 变换有**两类**失效情形，都会返回 valid=False，调用方必须显式处理：
    #   1) 畸变模型不可逆（rd > r_d_max）：B/C 域的原始画面外圈没有去畸变解；
    #   2) 目标点落在目标画面之外：A/C 的整流图只覆盖原始画面的中央区域
    #      （本工程实测 A 只用到 raw x∈[147,1137]、y∈[34,689]），
    #      所以 B 域里"可逆"的点仍可能落到 A 画面外。
    # 只判 (1) 不判 (2)，往返误差会假到几百 px。

    def b640_to_raw(self, pts):
        p = np.asarray(pts, np.float64).reshape(-1, 2)
        return p / np.array([self.sx, self.sy], np.float64)

    def raw_to_b640(self, pts):
        p = np.asarray(pts, np.float64).reshape(-1, 2)
        return p * np.array([self.sx, self.sy], np.float64)

    def rect720_to_a640(self, pts):
        p = np.asarray(pts, np.float64).reshape(-1, 2)
        return p * np.array([self.sx, self.sy], np.float64)

    def inside(self, pts) -> np.ndarray:
        """点是否落在本分辨率下 640 画面的有效像素范围内"""
        p = np.asarray(pts, np.float64).reshape(-1, 2)
        return ((p[:, 0] >= 0) & (p[:, 0] <= self.size - 1) &
                (p[:, 1] >= 0) & (p[:, 1] <= self.size - 1))

    def b640_to_a640(self, pts):
        """B 域点 -> a640 域点（畸变域 -> 去畸变域）。返回 (pts, valid)

        ⚠️ `undistort_pixels` 出来的是 **720p 整流域**（newK720 域）像素，
        必须再乘 `S` 才是 a640 —— 漏乘会导致「是否在画面内」判错，
        进而把可逆点误判成无效（本模块早期版本正是如此）。
        """
        rect720, ok = undistort_pixels(self.b640_to_raw(pts), self.K, self.dist,
                                       self.newK720)
        a = self.rect720_to_a640(rect720)
        return a, ok & self.inside(a)

    def a640_to_b640(self, pts):
        """a640 域点 -> B 域点：用 remap 表反查（与 remap 定义严格互逆）"""
        p = np.asarray(pts, np.float64).reshape(-1, 2)
        ok = self.inside(p)
        rect720 = p / np.array([self.sx, self.sy], np.float64)
        raw, ok_map = self._sample(self.maps_720_f32, rect720)
        ok &= ok_map
        return self.raw_to_b640(raw), ok

    def b640_to_c640(self, pts):
        """B 域点 -> c640 域点（C 组图上直接量到的坐标）。返回 (pts, valid)"""
        out, ok = undistort_pixels(pts, self.K_sq, self.dist, self.newK640)
        return out, ok & self.inside(out)

    def c640_to_b640(self, pts):
        """c640 域点 -> B 域点：用 640 域 remap 表反查。返回 (pts, valid)"""
        return self._sample(self.maps_640_f32,
                            np.asarray(pts, np.float64).reshape(-1, 2))

    def a640_to_c640(self, pts):
        return self.transform(pts, GROUP_A, GROUP_C)

    def c640_to_a640(self, pts):
        return self.transform(pts, GROUP_C, GROUP_A)

    # ---------------- 通用跨组换算（以 B 域为枢纽，逐段传递有效性）----------------

    def to_b640(self, pts, group: str):
        if group == GROUP_B:
            p = np.asarray(pts, np.float64).reshape(-1, 2)
            return p, self.inside(p)
        if group == GROUP_A:
            return self.a640_to_b640(pts)
        if group == GROUP_C:
            return self.c640_to_b640(pts)
        raise ValueError(f"未知组: {group}")

    def from_b640(self, pts, group: str):
        if group == GROUP_B:
            p = np.asarray(pts, np.float64).reshape(-1, 2)
            return p, self.inside(p)
        if group == GROUP_A:
            return self.b640_to_a640(pts)
        if group == GROUP_C:
            return self.b640_to_c640(pts)
        raise ValueError(f"未知组: {group}")

    def transform(self, pts, src_group: str, dst_group: str):
        """任意两组之间换算 640 域像素坐标。返回 (pts, valid)

        `valid=False` 的点不得使用：见本类顶部「两类失效情形」。
        对应不上时返回的坐标是中间量，调用方必须一起丢弃。
        """
        if src_group == dst_group:
            p = np.asarray(pts, np.float64).reshape(-1, 2)
            return p, self.inside(p)
        b, ok1 = self.to_b640(pts, src_group)
        out, ok2 = self.from_b640(b, dst_group)
        return out, ok1 & ok2

    # ---------------- 工具 ----------------
    def _sample(self, maps_f32, pts):
        """在浮点 remap 表上双线性采样：去畸变域点 -> 畸变域点

        maps_f32 = (map_x, map_y)，索引是去畸变域像素、值是畸变域像素，
        所以「采样」就是 remap 的逆。返回 (pts_raw, valid)，
        `valid` 同时要求「输入点在表内」且「查到的源点在原始画面内」。
        """
        map_x, map_y = maps_f32
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        if pts.size == 0:
            return np.zeros((0, 2), np.float64), np.zeros(0, bool)
        h, w = map_x.shape[:2]
        in_range = ((pts[:, 0] >= 0) & (pts[:, 0] <= w - 1) &
                    (pts[:, 1] >= 0) & (pts[:, 1] <= h - 1))
        pf32 = pts.astype(np.float32)
        mx = pf32[:, 0].reshape(1, -1, 1)
        my = pf32[:, 1].reshape(1, -1, 1)
        vx = cv2.remap(map_x, mx, my, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        vy = cv2.remap(map_y, mx, my, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        src = np.stack([vx.ravel(), vy.ravel()], axis=1).astype(np.float64)
        src_ok = ((src[:, 0] >= 0) & (src[:, 0] <= self.raw_w - 1) &
                  (src[:, 1] >= 0) & (src[:, 1] <= self.raw_h - 1))
        return src, in_range & src_ok

    # ---------------- 诊断 ----------------
    def invertible_limit(self):
        """(r_turn, r_d_max)：归一化可逆半径上限"""
        return invertible_limit(self.dist)

    def mappable_stats(self, n: int = 400) -> dict:
        """B 域（= 原始画面）里可逆像素占比，以及半径上限。

        「可逆」= 该像素在标定畸变模型下存在去畸变解**且**结果落在 A/C 画面内。
        A/C 的整流图只覆盖原始画面中央（实测本标定 A 只用到
        raw x∈[147,1137]、y∈[34,689]），所以两块约束都要算上。
        不可逆区域的点在 A/C 域**没有对应坐标**，跨组标签换算与 C′ 组必须跳过。
        """
        xs = np.linspace(0, self.size - 1, n)
        gx, gy = np.meshgrid(xs, xs)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        _, ok = self.b640_to_a640(pts)
        r_turn, rd_max = self.invertible_limit()
        fx, fy = self.K[0, 0], self.K[1, 1]
        return {"b640_mappable_frac": float(ok.mean()),
                "raw_radius_limit_px": rd_max * fx,
                "b640_radius_limit_px": rd_max * fx * self.sx,
                "r_turn": r_turn, "r_d_max": rd_max,
                "frame_corner_radius_px": float(np.hypot(
                    max(self.K[0, 2], self.raw_w - self.K[0, 2]),
                    max(self.K[1, 2], self.raw_h - self.K[1, 2])))}

    def magnification_profile(self, n: int = 121, h: float = 0.5,
                              bins=(0, 50, 100, 150, 200, 230, 260, 290, 330, 400)):
        """去畸变的局部放大倍率 |d(a640)/d(b640)|（最大奇异值）随半径的分布。

        这是判断「这份标定能不能用」的**实操判据**：去畸变在外圈被过度放大时，
        门框会被拉伸变形（`|d a/d b|` 大），既不是"没校正"也不是"校正好了"，
        而是**把几何搞坏**。倍率 ≈1 的区域去畸变基本是恒等变换（能算，但没意义）；
        `≤1.3` 通常可以认为"校正量温和、可信"。
        """
        xs = np.linspace(2, self.size - 3, n)
        X, Y = np.meshgrid(xs, xs)
        P = np.stack([X.ravel(), Y.ravel()], axis=1)

        def a(p):
            out, ok = self.b640_to_a640(p)
            return out, ok
        p0, ok0 = a(P)
        px, okx = a(P + [h, 0.0])
        py, oky = a(P + [0.0, h])
        J = np.stack([(px - p0) / h, (py - p0) / h], axis=2)
        sv = np.linalg.svd(J, compute_uv=False)[:, 0]
        r = np.hypot(P[:, 0] - self.K[0, 2] * self.sx,
                     P[:, 1] - self.K[1, 2] * self.sy)
        valid = ok0 & okx & oky

        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = valid & (r >= lo) & (r < hi)
            if m.sum() < 5:
                continue
            rows.append({"r_lo": lo, "r_hi": hi, "frac": float(m.mean()),
                         "mag_median": float(np.median(sv[m])),
                         "mag_p90": float(np.percentile(sv[m], 90)),
                         "mag_max": float(sv[m].max())})
        safe = {}
        for thr in (1.2, 1.3, 1.5, 2.0):
            m = valid & (sv <= thr)
            safe[f"mag<= {thr}"] = {"frac_of_b_frame": float(m.mean()),
                                    "max_radius_b640_px": float(r[m].max())
                                    if m.any() else 0.0}
        return {"bins": rows, "safe": safe}

    def check_equivalence(self) -> dict:
        """A/C 同域性 + 与 remap 表互逆性 + 往返误差（px）"""
        S = np.diag([self.sx, self.sy, 1.0])
        rng = np.random.default_rng(0)
        grid = np.stack(np.meshgrid(np.linspace(0.05, 0.95, 12) * self.size,
                                    np.linspace(0.05, 0.95, 12) * self.size),
                        axis=-1).reshape(-1, 2)
        pts_b = np.vstack([grid, rng.uniform(0, self.size, (200, 2))])

        # 1) a -> b -> a（b->a 走解析逆解，a->b 走 remap 表）：检验两者互逆
        #    ⚠️ 只在「两个方向都有解且都在画面内」的点上统计：超出标定可逆半径、
        #    或落到 A 画面之外的点本来就没有对应坐标，算进误差会得到几百 px 的假数字
        pts_a, ok = self.b640_to_a640(pts_b)
        back, ok_b = self.a640_to_b640(pts_a[ok])
        both1 = ok_b
        rt_a = float(np.max(np.linalg.norm(back[both1] - pts_b[ok][both1], axis=1)))
        rt_frac = float(ok.mean())

        # 2) 与 remap 表对拍：rect 点经表查到 raw，再由解析逆解回 rect
        xs = np.linspace(0.02, 0.98, 24) * self.raw_w
        ys = np.linspace(0.02, 0.98, 24) * self.raw_h
        gx, gy = np.meshgrid(xs, ys)
        rect = np.stack([gx.ravel(), gy.ravel()], axis=1)
        raw, ok_map = self._sample(self.maps_720_f32, rect)
        rect2, ok2 = undistort_pixels(raw, self.K, self.dist, self.newK720)
        good = ok_map & ok2
        err = np.linalg.norm(rect2[good] - rect[good], axis=1)
        map_err = float(err.max()) if err.size else float("nan")

        # 3) A 域 vs C 域（同样只统计有解的点）
        pts_c, ok_c = self.a640_to_c640(pts_a)
        both = ok & ok_c
        ac = (float(np.max(np.linalg.norm(pts_c[both] - pts_a[both], axis=1)))
              if both.any() else float("nan"))

        return {"newK640_vs_S_newK720_max": float(np.max(np.abs(
                    self.newK640 - S @ self.newK720))),
                "map_vs_newton_inverse_px": map_err,
                "a_to_b_to_a_px": rt_a,
                "domain_a_vs_c_px": ac,
                "valid_frac_grid_b": rt_frac,
                "valid_frac_map_grid": float(ok2.mean())}

    def self_test(self) -> dict:
        """解析畸变模型 vs cv2.projectPoints（K=I）对拍，确认公式一致"""
        rng = np.random.default_rng(1)
        xu = rng.uniform(-1.1, 1.1, 300)
        yu = rng.uniform(-1.1, 1.1, 300)
        obj = np.stack([xu, yu, np.ones_like(xu)], axis=1).reshape(-1, 1, 3)
        ref = cv2.projectPoints(obj, np.zeros(3), np.zeros(3),
                                np.eye(3), self.dist.reshape(1, -1))[0].reshape(-1, 2)
        mine = np.stack(distort_normalized(xu, yu, self.dist), axis=1)
        return {"distort_model_vs_cv2_max": float(np.max(np.abs(ref - mine)))}


class Geometry:
    """按标定文件 + 目标尺寸构造；对每种原始分辨率缓存一份 FrameGeometry。"""

    def __init__(self, calib_path: str | Path, size: int = 640,
                 alpha: float = 0.0, raw_size: tuple[int, int] | None = None):
        self.calib_path = Path(calib_path)
        self.K, self.dist, self.calib_size = load_calibration(self.calib_path)
        self.size = int(size)
        self.alpha = float(alpha)
        self.raw_size = raw_size
        self._cache: dict[tuple[int, int], FrameGeometry] = {}

    def for_frame(self, w: int, h: int) -> FrameGeometry:
        key = (int(w), int(h))
        if key not in self._cache:
            self._cache[key] = self._build(w, h)
        return self._cache[key]

    def _build(self, w: int, h: int) -> FrameGeometry:
        sx, sy = self.size / w, self.size / h
        newK720, _ = cv2.getOptimalNewCameraMatrix(
            self.K, self.dist, (w, h), self.alpha, (w, h))
        K_sq = _scale_K(self.K, sx, sy)
        newK640, _ = cv2.getOptimalNewCameraMatrix(
            K_sq, self.dist, (self.size, self.size), self.alpha,
            (self.size, self.size))
        return FrameGeometry(raw_w=w, raw_h=h, size=self.size, alpha=self.alpha,
                             K=self.K, dist=self.dist, K_sq=K_sq,
                             newK720=np.asarray(newK720, np.float64),
                             newK640=np.asarray(newK640, np.float64),
                             sx=sx, sy=sy)

    def expected_raw_size(self) -> tuple[int, int] | None:
        return self.raw_size or self.calib_size

    def describe(self) -> str:
        res = self.expected_raw_size()
        lines = [f"标定文件 : {self.calib_path}",
                 f"K        : {np.array2string(self.K, precision=3)}",
                 f"dist     : {np.array2string(self.dist, precision=5)}",
                 f"标定分辨率: {self.calib_size}   输出: {self.size}   alpha={self.alpha}"]
        if res:
            g = self.for_frame(*res)
            lines += [
                f"S        : sx={g.sx:.6f} sy={g.sy:.6f}（板端各向异性缩放）",
                f"newK720  : {np.array2string(g.newK720, precision=3)}",
                f"newK640  : {np.array2string(g.newK640, precision=3)}"]
        return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="几何自检（不含图像 I/O）")
    ap.add_argument("--calibration", type=Path,
                    default=PROJECT_ROOT / "configs" / "front_camera.yaml")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--raw-size", type=int, nargs=2, default=None, metavar=("W", "H"))
    a = ap.parse_args()
    geo = Geometry(a.calibration, a.size, a.alpha,
                   tuple(a.raw_size) if a.raw_size else None)
    print(geo.describe())
    res = geo.expected_raw_size() or (1280, 720)
    g = geo.for_frame(*res)
    print(f"\n--- {res[0]}x{res[1]} 自检 ---")
    for k, v in {**g.self_test(), **g.check_equivalence(), **g.mappable_stats()}.items():
        print(f"  {k:32s} {v:.6g}" if isinstance(v, float) else f"  {k:32s} {v}")
