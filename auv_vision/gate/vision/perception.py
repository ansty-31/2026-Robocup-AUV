# -*- coding: utf-8 -*-
"""gate/vision/perception.py — 感知半场（GateTask 里"看"的那一半）

一帧的感知链，顺序固定（质量分先验必须在融合之前）：

    选目标 → 质量**先验**(原始 conf + 帧龄) → 逐点软融合(kpt_memory，吃先验权重/增益)
    → 前端 mode → 可见性掩码 + 恢复线索 → 位姿 PnP / 重投影 RMS（含 z 跳变保护）
    → 质量**后验**(RMS + 投影框 IoU，含线索↔位姿互检) → 谨慎度(速度/死区缩放)

输出一个 `Sighting`：本帧"看到了什么 + 这一帧有多少证据"。相位机（gate_task.py）
只吃 Sighting，不再碰原始检测/角点/PnP —— 这样"看"和"决策"各自独立可测。

不做的事：不推进相位、不下发 DOF、不改任务计数器（z 跳变保护的**状态**仍在
GateTask 里，本层只按传入的 prev_z 判这一帧，并把结论放在 Sighting.dropped_z）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import base.settings as S
from gate.data.kpt_memory import KptMemory
from gate.vision.gate_frontend import parse_kpt_mode, MODE_FULL, MODE_P3P, MODE_COARSE
from gate.vision.geometry import gate_pose, reproj_rms
from gate.data.quality import (QualityCfg, QualityTracker, caution_from_q,
                          speed_scale, deadzone_scale)
from gate.data.cues import CueCfg, recovery_cue, visible_mask
from common.frame_stamp import StampCfg          # 共用件：帧龄上限（先验要用）


@dataclass
class Sighting(object):
    """本帧感知结果（纯数据，不含任何 GateTask 状态）。"""
    det: object = None
    bbox: tuple = (0.0, 0.0, 0.0, 0.0)
    kpts: object = None                # 融合后角点（供 width 档/叠加复用）
    kconf: object = None               # 融合后置信度
    mask: tuple = (False, False, False, False)
    mode: str = MODE_COARSE
    ids: list = field(default_factory=list)
    n: int = 0
    pose: tuple = None                 # (rvec, tvec) 或 None
    rms: float = None
    z: float = None
    cue: str = None
    cue_reason: str = ""
    quality: object = None             # Quality
    caution: float = 1.0
    sp: float = 1.0                    # 速度缩放（质量分→运动谨慎度）
    dz: float = 1.0                    # 死区/对准阈值缩放
    agree: object = None               # 线索↔位姿一致（True/False/None）
    dropped_z: bool = False            # 上一帧有基准但本帧 z 突变 → 弃帧
    kpt_raw: int = 0                   # 原始有效角点数（诊断：倒影污染程度）


def pick_gate(dets, conf_thr=0.5):
    """选目标门：**优先角点更全、更好的框**，其次才比 score。

    水中倒影常让角点落到倒影上（有时形成第二个门框）：只按 score 选可能把
    倒影当真门 → 位姿错、甚至 z≤cross 直接冲门。角点数量/置信度和更能反映
    "哪个是真门框"。（再配合上帧位姿消歧 + z 跳变保护一起用。）
    """
    best, best_key = None, None
    for d in dets or []:
        if getattr(d, "kind", None) != "gate":
            continue
        if d.kpt_conf is None:
            key = (-1, 0.0, float(d.score))     # 无角点信息 → 排最后
        else:
            kc = np.asarray(d.kpt_conf, np.float64)
            key = (int((kc >= conf_thr).sum()), float(kc.sum()), float(d.score))
        if best_key is None or key > best_key:
            best, best_key = d, key
    return best


class GatePerception(object):
    """门感知链（可选配 kpt_memory；`enable=false` 时退化为单帧直用）。"""

    def __init__(self, camera, obj3, image_size, conf_thr=0.5,
                 kpt_mem=None, tracker=None, cue_cfg=None, pnp_cfg=None,
                 stamp_cfg=None):
        self.camera = camera
        self.obj3 = obj3
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.conf_thr = float(conf_thr)
        self.kpt_mem = kpt_mem
        self.tracker = tracker
        self.cue_cfg = cue_cfg
        self.pnp = pnp_cfg or {}
        self.stamp = stamp_cfg
        self.reset()

    @classmethod
    def from_settings(cls, S_, camera, obj3, image_size, n_kpt=4):
        """按当前配置装配（vision.yaml → gate.keypoint/kpt_mem/pnp；
        comm.yaml → gate.data.quality/cues/stamp）。"""
        V = S_.vision.gate
        conf_thr = V.keypoint.get("conf_thr", 0.5)
        stamp_cfg = StampCfg.from_settings(S_)
        max_age = float(stamp_cfg.max_age_s)
        return cls(camera, obj3, image_size, conf_thr=conf_thr,
                   kpt_mem=KptMemory.from_settings(V.get("kpt_mem", None),
                                                   n_kpt=n_kpt),
                   tracker=QualityTracker.from_settings(
                       S_, camera, obj3, image_size, conf_thr=conf_thr,
                       max_age_s=max_age),
                   cue_cfg=CueCfg.from_settings(S_),
                   pnp_cfg=V.get("pnp", None),
                   stamp_cfg=stamp_cfg)

    def reset(self):
        if self.kpt_mem is not None:
            self.kpt_mem.reset()
        if self.tracker is not None:
            self.tracker.reset()

    # ------------------------------------------------------------------ 主入口
    def observe(self, det, now_ms, age=0.0, prev_pose=None, z_guard=False,
                prev_z=None):
        """一帧检测 → Sighting（不做相位/运动决策）。"""
        conf_thr = self.conf_thr
        bbox = (det.x, det.y, det.x + det.w, det.y + det.h)
        kpt_raw = int((np.asarray(det.kpt_conf) >= conf_thr).sum()) \
            if det.kpt_conf is not None else 0

        # ① 质量先验（只看原始 conf + 帧龄）→ 逐点权重 + 帧级增益
        w_prior, q_time = (None, 1.0)
        q_frame = None
        if self.tracker is not None:
            w_prior, q_time = self.tracker.prior(det.kpt_conf, age)
            q_frame = self.tracker.q_mem

        # ② 逐点软融合（质量分在这里真正进入权重与增益）
        kpts, kconf = det.kpts, det.kpt_conf
        if self.kpt_mem is not None:
            kpts, kconf = self.kpt_mem.update(det.kpts, det.kpt_conf, now_ms,
                                             w_prior=w_prior, q=q_frame)

        # ③ 前端 → mode（用记忆后的角点）
        if kpts is not None and np.asarray(kpts).shape[0] >= 4 and kconf is not None:
            mode, ids = parse_kpt_mode(np.asarray(kpts), kconf, conf_thr)
            n = len(ids)
        else:
            mode, ids, n = MODE_COARSE, [], 0

        # ④ 可见性 + 恢复线索（与融合层共用同一套"融合后"状态）
        mask = visible_mask(kconf, conf_thr)
        cue, cue_reason = (None, "")
        if self.cue_cfg is not None and self.cue_cfg.enable:
            cue, cue_reason = recovery_cue(mask, bbox, self.image_size,
                                           self.cue_cfg)

        # ⑤ 位姿（full/p3p 且 ≥3 点）；z 跳变保护只判这一帧
        pose, rms, z, dropped = None, None, None, False
        if mode in (MODE_FULL, MODE_P3P) and n >= 3:
            obj3s = self.obj3[ids]
            img2s = np.asarray(kpts)[ids]
            pnp = self.pnp or {}
            res = gate_pose(self.camera, obj3s, img2s, prev=prev_pose,
                            reproj_thr=pnp.get("reproj_px", 8.0),
                            z_bounds=(pnp.get("z_min", 0.2),
                                      pnp.get("z_max", 15.0)),
                            refine=pnp.get("refine", True))
            if res is not None:
                max_jump = float(pnp.get("max_z_jump_m", 0.8))
                if z_guard and prev_z is not None and max_jump > 0:
                    z_new = float(res[1].ravel()[2])
                    if abs(z_new - prev_z) > max_jump:
                        if S.DEBUG:
                            print("[GATE] 弃帧: z 跳变 %.2f→%.2f m (>%.2f)"
                                  % (prev_z, z_new, max_jump))
                        res, dropped = None, True
            if res is not None:
                pose = res
                z = float(res[1].ravel()[2])
                rms = reproj_rms(self.camera, obj3s, img2s, *res)
            else:
                mode = MODE_COARSE          # 位姿校验失败/跳变 → 退化 coarse

        # ⑥ 质量后验（谨慎度 + q_mem；线索↔位姿互检在这里）
        if self.tracker is not None:
            q, agree = self.tracker.posterior(kconf, bbox, pose=pose, rms=rms,
                                              age=age, cue=cue, z=z)
        else:
            q, agree = None, None
        qcfg = self.tracker.cfg if self.tracker is not None else QualityCfg()
        caution = caution_from_q(qcfg, q.total) if q is not None else 1.0
        return Sighting(det=det, bbox=bbox, kpts=kpts, kconf=kconf, mask=mask,
                        mode=mode, ids=ids, n=n, pose=pose, rms=rms, z=z,
                        cue=cue, cue_reason=cue_reason, quality=q,
                        caution=caution, sp=speed_scale(qcfg, caution),
                        dz=deadzone_scale(qcfg, caution), agree=agree,
                        dropped_z=dropped, kpt_raw=kpt_raw)
