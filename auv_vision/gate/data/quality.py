# -*- coding: utf-8 -*-
"""gate/data/quality.py — 质量分 Q 的两条去向（v1.4）

Q 不是"塞给任务的一个数"，而是**一帧的证据质量**。它按发生时刻分两半，
各自去该去的地方，任务只负责读结果：

  ① **先验 prior（看位姿之前就能算）**：`prior_weights()` / `QualityTracker.prior()`
     只看"本帧**原始**角点置信度 + 帧新鲜度"（age），不看位姿。
     去向：`gate/data/kpt_memory.py` —— 逐点权重 `w_prior[i]`（这一帧这个点值多少）
     + 帧级增益（`QualityTracker.q_mem`，历史位姿证据的 EMA）。
     质量低 → 该帧少拉、更信历史，但**不硬判无效**（点仍有效、状态不丢）。

  ② **后验 posterior（融合+解算之后）**：`compute_quality()` / `QualityTracker.posterior()`
     融合后有了重投影 RMS 与"位姿投影框 vs 检测框"IoU，才是"本帧位姿可不可信"。
     去向：运动谨慎度 c（速度缩放 / 死区缩放）；同时把总量记进 `q_mem`，
     成为**下一帧**先验的历史证据（闭环，但只用过去帧，不看未来）。

因果顺序（改这里时别破坏）：
    原始conf/帧龄 ──prior──► kpt_memory 权重+增益 ──► 融合角点 ──► PnP/RMS
                                                        └─posterior─► 谨慎度 c
                                                                    └─► q_mem ─► 下一帧 prior

分量（均可加权/关闭，`cfg/comm.yaml → gate.data.quality:`）：
  q_kpt  融合后角点的可见数 + 平均置信度
  q_pnp  重投影质量 1/(1+(rms/rms_ref)²)；无位姿 → q_no_pose
  q_box  位姿投影框 vs 检测框 IoU（**位姿链↔线索链互相自检的那一条**：
         倒影把框拉高/拼框时 IoU 掉，位姿自动降权）
  q_time 帧新鲜度（age → 下限 q_time_floor）

Q **不参与过门判定本身**（安全判定仍由位姿链的确认决定）；`confirm_max_scale=1.0`
时它也不改确认时长，只改"给多少权重/开多快"。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gate.data.cues import quality_agreement


@dataclass
class QualityCfg:
    w_kpt: float = 1.0
    w_pnp: float = 1.0
    w_box: float = 0.8
    w_time: float = 0.5
    rms_ref_px: float = 5.0
    q_no_pose: float = 0.25
    box_iou_floor: float = 0.4
    q_lo: float = 0.35
    q_hi: float = 0.75
    speed_min_scale: float = 0.4
    confirm_max_scale: float = 1.0   # 1.0=质量分不影响确认时长(只调速度/死区)
    deadzone_max_scale: float = 1.6
    # --- 先验（质量分→kpt_memory 权重）---
    prior_conf_pow: float = 1.0      # 先验里原始置信度的幂（<1 更宽容）
    q_time_floor: float = 0.2        # 帧新鲜度下限（过期帧仍参与，但权重低）
    mem_lambda: float = 0.3          # q_mem（历史证据）EMA 速率，喂下一帧先验

    @classmethod
    def from_settings(cls, S):
        cfg = cls()
        data = S.get("comm.gate.quality") or {}
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


@dataclass(frozen=True)
class Quality:
    total: float
    q_kpt: float
    q_pnp: float
    q_box: float
    q_time: float


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * \
        max(0.0, min(ay2, by2) - max(ay1, by1))
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / ua) if ua > 0 else 0.0


def _project_bbox(camera, obj3, rvec, tvec, image_size):
    uv = camera.project(obj3, rvec, tvec)
    w, h = float(image_size[0]), float(image_size[1])
    return (max(0.0, float(uv[:, 0].min())), max(0.0, float(uv[:, 1].min())),
            min(w, float(uv[:, 0].max())), min(h, float(uv[:, 1].max())))


def time_quality(cfg, age, max_age_s=0.9):
    """帧新鲜度 → [q_time_floor, 1]：过期帧照常估计，只是权重低。"""
    return float(np.clip(1.0 - float(age) / max(float(max_age_s), 1e-6),
                         float(cfg.q_time_floor), 1.0))


def prior_weights(cfg, raw_conf, age=0.0, max_age_s=0.9):
    """**先验**逐点权重（喂 kpt_memory）+ 帧新鲜度 q_time。

    只用"这一帧自己看得见的东西"（原始角点置信度）与"这帧多旧"，不看位姿结果
    —— 因此可以**在融合之前**算出来，真的能左右这一帧的融合权重。
    """
    c = np.asarray(raw_conf, float).reshape(-1) if raw_conf is not None \
        else np.zeros(0)
    if c.size < 4:
        c = np.zeros(4)
    q_t = time_quality(cfg, age, max_age_s)
    w = np.clip(c[:4], 0.0, 1.0) ** float(cfg.prior_conf_pow) * q_t
    return np.clip(w, 0.0, 1.0), q_t


def compute_quality(cfg, camera, obj3, fused_conf, bbox_xyxy, image_size,
                    conf_thr=0.5, pose=None, rms=None, age=0.0,
                    max_age_s=0.9):
    """**后验**质量分：融合后角点 + 位姿(RMS/框一致性) + 帧龄 → Q 及其分量。"""
    conf = np.asarray(fused_conf, float).reshape(-1)
    if conf.size >= 4:
        n_ok = float(np.sum(conf[:4] >= conf_thr))
        mean_conf = float(np.mean(np.clip(conf[:4], 0.0, 1.0)))
    else:
        n_ok, mean_conf = 0.0, 0.0
    q_kpt = 0.5 * (n_ok / 4.0) + 0.5 * mean_conf

    if pose is not None and rms is not None:
        q_pnp = 1.0 / (1.0 + (float(rms) / max(cfg.rms_ref_px, 1e-6)) ** 2)
        if bbox_xyxy is not None:
            proj = _project_bbox(camera, obj3, pose[0], pose[1], image_size)
            iou = bbox_iou(proj, bbox_xyxy)
        else:
            iou = 0.0
        if iou <= cfg.box_iou_floor:
            q_box = 0.0
        elif iou >= 0.9:
            q_box = 1.0
        else:
            q_box = (iou - cfg.box_iou_floor) / max(1e-6, 0.9 - cfg.box_iou_floor)
    else:
        q_pnp = float(cfg.q_no_pose)
        q_box = 0.7
    q_time = time_quality(cfg, age, max_age_s)

    parts = ((cfg.w_kpt, q_kpt), (cfg.w_pnp, q_pnp), (cfg.w_box, q_box),
             (cfg.w_time, q_time))
    wsum = sum(w for w, _ in parts if w > 0)
    total = sum(w * v for w, v in parts if w > 0) / wsum if wsum > 0 else 0.5
    return Quality(float(np.clip(total, 0.0, 1.0)), q_kpt, q_pnp, q_box, q_time)


def caution_from_q(cfg, q_total):
    """Q → 谨慎度 c ∈ [0,1]（0=最谨慎，1=正常）。"""
    lo, hi = float(cfg.q_lo), float(cfg.q_hi)
    if hi <= lo:
        return 1.0
    return float(np.clip((float(q_total) - lo) / (hi - lo), 0.0, 1.0))


def speed_scale(cfg, caution):
    return float(cfg.speed_min_scale + (1.0 - cfg.speed_min_scale) * float(caution))


def confirm_scale(cfg, caution):
    return float(1.0 + (cfg.confirm_max_scale - 1.0) * (1.0 - float(caution)))


def deadzone_scale(cfg, caution):
    return float(1.0 + (cfg.deadzone_max_scale - 1.0) * (1.0 - float(caution)))


class QualityTracker(object):
    """有状态的质量分：把 ① 先验（→kpt_memory）与 ② 后验（→谨慎度/q_mem）串起来。

    一帧的典型调用（顺序不能反，先验必须早于融合）：
        w_prior, q_time = trk.prior(det.kpt_conf, age)
        kpts, kconf = kpt_mem.update(det.kpts, det.kpt_conf, now_ms,
                                     w_prior=w_prior, q=trk.q_mem)
        ... PnP ...
        q, agree = trk.posterior(kconf, bbox, pose=pose, rms=rms, age=age,
                                 cue=cue, z=z)
        caution = caution_from_q(cfg, q.total) → speed_scale/deadzone_scale

    `q_mem` 只在**有可信位姿**的帧上更新（无位姿帧保持，不惩罚：位姿链一时解不出来
    不代表观测变差），因此它表示"位姿链最近好不好"，用作下一帧的融合增益。
    """

    def __init__(self, cfg, camera, obj3, image_size, conf_thr=0.5,
                 max_age_s=0.9, cue_cfg=None):
        self.cfg = cfg
        self.camera = camera
        self.obj3 = obj3
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.conf_thr = float(conf_thr)
        self.max_age_s = float(max_age_s)
        self.cue_cfg = cue_cfg
        self.reset()

    @classmethod
    def from_settings(cls, S, camera, obj3, image_size, conf_thr=0.5,
                      max_age_s=0.9):
        from gate.data.cues import CueCfg
        return cls(QualityCfg.from_settings(S), camera, obj3, image_size,
                   conf_thr=conf_thr, max_age_s=max_age_s,
                   cue_cfg=CueCfg.from_settings(S))

    def reset(self):
        self._q = None                 # 上帧后验质量分（诊断）
        self._q_mem = 1.0              # 历史位姿证据（喂下一帧先验的融合增益）
        self._q_time = 1.0
        self._w_prior = np.ones(4, np.float64)
        self._agree = None

    # ------------------------------------------------------------ 诊断
    @property
    def q(self):
        return self._q

    @property
    def q_mem(self):
        return float(self._q_mem)

    @property
    def q_time(self):
        return float(self._q_time)

    @property
    def w_prior(self):
        return self._w_prior.copy()

    @property
    def agree(self):
        return self._agree

    # ------------------------------------------------------------ ① 先验
    def prior(self, raw_conf, age=0.0):
        """看位姿之前：返回 (逐点先验权重, 帧新鲜度)；帧级增益读 `q_mem`。"""
        w, q_t = prior_weights(self.cfg, raw_conf, age, self.max_age_s)
        self._w_prior, self._q_time = w, q_t
        return w, q_t

    # ------------------------------------------------------------ ② 后验
    def posterior(self, fused_conf, bbox_xyxy, pose=None, rms=None, age=0.0,
                  cue=None, z=None):
        """融合+解算之后：返回 (Quality, agree|None)，并更新 `q_mem`。"""
        q = compute_quality(self.cfg, self.camera, self.obj3, fused_conf,
                            bbox_xyxy, self.image_size, conf_thr=self.conf_thr,
                            pose=pose, rms=rms, age=age,
                            max_age_s=self.max_age_s)
        self._agree = None
        if cue is not None and self.cue_cfg is not None:
            # 线索链 ↔ 位姿链 互相自检：一致加成 / 冲突打折（不驱动运动）
            total, agree = quality_agreement(cue, pose is not None, z,
                                             q.total, self.cue_cfg)
            self._agree = agree
            if total != q.total:
                q = Quality(total, q.q_kpt, q.q_pnp, q.q_box, q.q_time)
        self._q = q
        if pose is not None:                     # 只让"有位姿证据"的帧进历史
            m = float(np.clip(self.cfg.mem_lambda, 0.0, 1.0))
            self._q_mem = (1.0 - m) * self._q_mem + m * float(q.total)
        return q, self._agree
