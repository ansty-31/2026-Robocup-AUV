# -*- coding: utf-8 -*-
"""gate/data/kpt_memory.py — 门角点"逐点软融合"（抗水面倒影：小漂移/短消失/偶发鬼点）

**设计原则**（按使用要求收敛）：
  * **不硬判无效**：每个角点始终输出"融合后的状态"（位置 + 连续置信度），
    把"信不信这一帧"变成权重的连续变化，而不是"有效/无效"的二值切换 ——
    这样状态机不会因为单点抖动而翻来覆去（少掉 full↔p3p↔coarse 的抖动），
    也不至于把信息量本来就不多的 4 个点滤没了。
  * **逐点自身信息融合**：各角点独立维护 位置/速度/置信度/残差尺度，
    只利用该点自己的历史（不动识别后的 mode/PnP 判定逻辑）。
  * **范围圆半径来自该点历史分布**：R_i = clip(k_sigma × σ_i, r_min, r_max)，
    σ_i 是该点残差分布的尺度（残差慢跟踪，会自适应：点安静→半径小；
    门在快速逼近、帧间位移变大→半径自动放大；世界真变了→σ 慢慢变大从而跟上）。
  * **软几何权重**：w_geo = 1 / (1 + (d/R)²)（柯西权重，越小越不可信，连续无跳变）；
    一帧鬼点（d ≫ R）权重≈0 → 对融合几乎无影响，但**该点仍然有效**、状态不丢。
  * **不影响速度**：α-β 时间加权平均带匀速补偿（稳态滞后≈0），不做窗口算术平均，
    不引入额外确认延迟。
  * **短消失回忆 / 长丢失衰减**：短时没打到 → 用融合状态外推并保持 recall_conf；
    连续缺失超限或超 valid_ms → 置信度按 conf_decay 衰减，最终自然低于 keypoint.conf_thr，
    让 mode 照常降级（不瞎编）。

**质量分的第一个去向（v1.4）**：本层接受两路"证据质量"，让 Q 直接作用在融合权重上，
而不是只拿去调速度（质量分的第二个去向在 `gate/data/quality.py → QualityTracker.posterior`）：
  * `w_prior[i]`：**逐点先验权重**（原始角点置信度 × 帧新鲜度，由 quality.prior_weights 给）
    —— 传入时**取代**原来的 `conf_i` 因子（w = w_prior × w_geo，不重复相乘）；
    帧老了/点本来就虚 → 这一帧少拉一点，但**照样输出融合状态**，不做硬判。
  * `q`：**帧级证据质量**（历史位姿链的 EMA，`QualityTracker.q_mem`）→ 增益
    `gain = q_gain_floor + (1-q_gain_floor) × q`，同时乘到 α 与 β 上：
    位姿链最近一直不可信 → 更信历史（融合更平滑），有下限 q_gain_floor，不会停死。
  两路都缺省（`w_prior=None, q=None`）时与升级前**逐位一致**（gain=1、w=conf×w_geo）。

全部参数可调（`vision.yaml → gate.kpt_mem`），`enable=false` 即恢复"单帧直用"。
"""
from __future__ import annotations

import numpy as np


def _as_kpts(kpts, n_kpt):
    if kpts is None:
        return None
    a = np.asarray(kpts, np.float64)
    if a.ndim != 2 or a.shape[0] < n_kpt or a.shape[1] < 2:
        return None
    return a[:n_kpt, :2]


class KptMemory(object):
    """逐点软融合：置信度加权 + 历史分布定半径 + α-β 平均（零滞后）。"""

    def __init__(self, n_kpt=4, alpha=0.4, beta=0.1, valid_ms=800.0,
                 k_sigma=3.0, r_min_px=10.0, r_max_px=80.0,
                 sigma_init_px=8.0, sigma_lambda=0.15, sigma_floor_px=0.5,
                 w_min=0.0, conf_floor=0.05, recall_conf=0.55,
                 conf_decay=0.35, max_missing_frames=5, min_frames=2,
                 q_gain_floor=0.35):
        self.n = int(n_kpt)
        self.alpha = float(alpha)              # 融合增益（大=快/小=稳）
        self.beta = float(beta)                # 匀速补偿增益
        self.valid_ms = float(valid_ms)        # 多久没见到就不再算"近期见过"
        self.k_sigma = float(k_sigma)          # 半径 = k_sigma × 残差尺度σ
        self.r_min = float(r_min_px)
        self.r_max = float(r_max_px)           # <=0 表示不设几何权重（纯 α-β）
        self.sigma_init = float(sigma_init_px)
        self.sigma_lambda = float(sigma_lambda)  # σ 慢跟踪速率（历史分布窗口）
        self.sigma_floor = float(sigma_floor_px)
        self.w_min = float(w_min)              # >0 时把权重低于它的帧彻底忽略(默认0=不硬关)
        self.conf_floor = float(conf_floor)    # 低于此置信度视为"本帧没打到"
        self.recall_conf = float(recall_conf)  # 短消失回忆时给的置信度
        self.conf_decay = float(conf_decay)    # 长丢失时置信度衰减率
        self.max_missing = int(max_missing_frames)
        self.min_frames = int(min_frames)
        self.q_gain_floor = float(q_gain_floor)  # 帧级质量=0 时保留的融合增益比例
        self.reset()

    # ------------------------------------------------------------ 装配
    @classmethod
    def from_settings(cls, km, n_kpt=4):
        """`km` = `vision.yaml → gate.kpt_mem` 的字典；`enable=false` → None（单帧直用）。"""
        km = km or {}
        if not km.get("enable", True):
            return None
        names = ("alpha", "beta", "valid_ms", "k_sigma", "r_min_px", "r_max_px",
                 "sigma_init_px", "sigma_lambda", "sigma_floor_px", "w_min",
                 "conf_floor", "recall_conf", "conf_decay",
                 "max_missing_frames", "min_frames", "q_gain_floor")
        kw = {k: km[k] for k in names if k in km}
        return cls(n_kpt=int(n_kpt), **kw)


    def reset(self):
        self._pos = None
        self._vel = None
        self._conf = None
        self._sigma = None
        self._seen_n = np.zeros(self.n, np.int32)
        self._last_seen = np.full(self.n, -1e9, np.float64)
        self._miss = np.zeros(self.n, np.int32)
        self._t = None
        self._n_frames = 0
        self._last_w = np.zeros(self.n, np.float64)      # 诊断：上帧各点权重
        self._last_d = np.zeros(self.n, np.float64)      # 诊断：上帧各点残差
        self._last_gain = 1.0                            # 诊断：上帧帧级增益（质量分→本层）
        self._last_q = 1.0                               # 诊断：上帧帧级证据质量

    # ------------------------------------------------------------------ 诊断
    @property
    def radius(self):
        if self._sigma is None:
            return np.zeros(self.n, np.float64)
        return np.array([self._radius(i) for i in range(self.n)], np.float64)

    @property
    def weights(self):
        return self._last_w.copy()

    @property
    def residuals(self):
        return self._last_d.copy()

    @property
    def gain(self):
        """上帧帧级增益 = q_gain_floor + (1-q_gain_floor)×q（1.0=质量分未介入）。"""
        return float(self._last_gain)

    def _radius(self, i):
        """该点的范围圆半径：由它自身历史残差分布 σ 决定（可关）。"""
        if self.k_sigma <= 0 or self.r_max <= 0:
            return float("inf")
        r = self.k_sigma * float(self._sigma[i])
        return float(min(max(r, self.r_min), self.r_max))

    # ------------------------------------------------------------------ 对外
    def update(self, kpts, conf, now_ms, w_prior=None, q=None):
        """当帧 (kpts, kpt_conf) → 融合后的 (kpts, kpt_conf)。

        始终返回 4 个点的融合状态；置信度是连续的（低到一定程度后由
        parse_kpt_mode 的 conf_thr 自然剔除，不需要这里硬判）。

        `w_prior`：质量分**先验**的逐点权重（见 quality.prior_weights）——
                   给出时取代原始 conf 作为权重因子（不重复相乘）。
        `q`      ：质量分**帧级**证据质量（0..1，通常=QualityTracker.q_mem）——
                   只缩放融合增益 α/β（低质量→更信历史），下限 q_gain_floor。
        两者都不给 → 与升级前逐位一致。
        """
        cur = _as_kpts(kpts, self.n)
        cconf = np.zeros(self.n, np.float64)
        if cur is not None and conf is not None:
            c = np.asarray(conf, np.float64).reshape(-1)
            cconf[:min(self.n, c.size)] = c[:self.n]
        w_in = self._prior_vector(w_prior, cconf)
        gain = self._gain(q)
        alpha, beta = self.alpha * gain, self.beta * gain

        if self._pos is None:                       # 首帧：直接建状态
            self._pos = np.zeros((self.n, 2), np.float64) if cur is None \
                else cur.copy()
            self._vel = np.zeros((self.n, 2), np.float64)
            self._conf = cconf.copy()
            self._sigma = np.full(self.n, self.sigma_init, np.float64)
            self._t = float(now_ms)
            self._n_frames = 1
            seen = (cconf >= self.conf_floor) if cur is not None \
                else np.zeros(self.n, bool)
            self._seen_n = seen.astype(np.int32)
            self._last_seen = np.where(seen, float(now_ms), -1e9)
            self._miss = (~seen).astype(np.int32)
            self._last_w = w_in
            self._note_quality(gain, q)
            return (self._pos.astype(np.float32),
                    np.where(seen, self._conf, 0.0).astype(np.float32))

        dt = max(1.0, min(500.0, float(now_ms) - self._t))
        self._t = float(now_ms)
        self._n_frames += 1
        out_k = np.zeros((self.n, 2), np.float64)
        out_c = np.zeros(self.n, np.float64)

        for i in range(self.n):
            pred = self._pos[i] + self._vel[i] * dt          # 匀速外推（补滞后）
            seen = cur is not None and cconf[i] >= self.conf_floor
            r = self._radius(i)
            d = float(np.linalg.norm(cur[i] - pred)) if seen else 0.0
            # 软几何权重：越偏离"该点历史/预测位置"越不可信（连续、无跳变）；
            # 帧/点级证据质量（w_in）在这里进入融合权重
            w_geo = 1.0 if not np.isfinite(r) else 1.0 / (1.0 + (d / r) ** 2)
            w = float(np.clip(w_in[i] * w_geo, 0.0, 1.0)) if seen else 0.0
            if self.w_min > 0 and w < self.w_min:
                w = 0.0

            if seen:
                a = alpha * w                                # 置信度加权融合
                self._pos[i] = pred + a * (cur[i] - pred)
                self._vel[i] = self._vel[i] * (1 - beta * w) + \
                    beta * w * (self._pos[i] - pred) / dt
                self._vel[i] = np.clip(self._vel[i], -5.0, 5.0)     # px/ms
                self._conf[i] = (1 - a) * self._conf[i] + a * cconf[i]
                # σ 慢跟踪该点残差分布（含离群帧）→ 半径自适应、可自愈
                self._sigma[i] = (1 - self.sigma_lambda) * self._sigma[i] + \
                    self.sigma_lambda * max(d, self.sigma_floor)
                self._seen_n[i] += 1
                self._last_seen[i] = float(now_ms)
                self._miss[i] = 0
            else:
                # 没打到：用融合状态外推；短期"回忆"（保持可用的置信度），
                # 长期则按 conf_decay 衰减，最终自然低于 conf_thr
                self._miss[i] += 1
                self._pos[i] = pred
                recent = self._seen_n[i] >= self.min_frames and \
                    (float(now_ms) - self._last_seen[i]) <= self.valid_ms and \
                    self._miss[i] <= self.max_missing
                if recent:
                    self._conf[i] = max(self._conf[i], self.recall_conf)
                else:
                    self._conf[i] = self._conf[i] * (1 - self.conf_decay)

            out_k[i] = self._pos[i]
            out_c[i] = self._conf[i]
            self._last_w[i] = w
            self._last_d[i] = d
        self._note_quality(gain, q)
        return out_k.astype(np.float32), out_c.astype(np.float32)

    # ------------------------------------------------------------------ 质量分入口
    def _prior_vector(self, w_prior, cconf):
        """先验逐点权重：给了就用它（质量分先验已含 conf 与帧新鲜度），否则用原始 conf。"""
        if w_prior is None:
            return cconf.copy()
        w = np.asarray(w_prior, np.float64).reshape(-1)
        out = cconf.copy()
        out[:min(self.n, w.size)] = np.clip(w[:self.n], 0.0, 1.0)
        return out

    def _gain(self, q):
        """帧级增益：q 越低越信历史；q=None（旧调用）→ 1.0，行为不变。"""
        if q is None:
            return 1.0
        return float(self.q_gain_floor +
                     (1.0 - self.q_gain_floor) * float(np.clip(q, 0.0, 1.0)))

    def _note_quality(self, gain, q):
        """记诊断量。q=None 表示本帧没有质量分（沿用上一帧值）。"""
        self._last_gain = float(gain)
        if q is not None:
            self._last_q = float(np.clip(q, 0.0, 1.0))
