#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_domain.py — 「识别方案」**代次指纹自检（单域 D+wb）**（本地/板端都能跑，不入运行时）

2026-10-01 起本项目**只有一套识别方案**（A 域 / `vision.image.chain` 已退役归档，见
`bak/retired/Adomain_20261001/`），所以这里守的不再是"两域别混"，而是**同代配对**：

    链路 D（resize640 → enhance → remap@640）＋ 白平衡 wb `[1.0, 1.05, 1.15]`
    ＋ CLAHE 关（`clahe_clip 0`）＋ gamma 0.85 ＋ **阶段一 v3 门权重** ＋ 水下重标的内参

这四样**必须同代**：换了权重没换标定（或反过来把老 AUV_4 权重配到新链上）都会跨域掉点，
只看单张图是看不出来的 —— 这就是本脚本存在的理由。

退出码：0 = 指纹自洽；1 = 自相矛盾（终端里会打印该改哪一项）。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
while _ROOT != os.path.dirname(_ROOT) and not os.path.isdir(os.path.join(_ROOT, "cfg")):
    _ROOT = os.path.dirname(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

#: 本代（D+wb）的指纹：值改了就必须同步改这里（以及 tools/README.md §6 的代次表）
_GT_W_MD5 = "ca5ba84f"          # 阶段一 v3（新识别方案）门权重；上一代 g240_i16（ef52c18b）在 bak/
_WB_GAINS = [1.0, 1.05, 1.15]   # 白平衡通道增益 = 「D+wb」里的 wb
_GAMMA = 0.85
_FX = 1207.6                    # 水下重标内参（老 A 域那份是 782.5，已归档）
_FX_TOL = 1.0


def md5_of(path):
    if not path or not os.path.exists(path):
        return "-"
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fx_of(calib):
    """标定 yaml 的 fx（cv2 FileStorage；读不到返回 None）。"""
    try:
        import cv2
    except Exception:
        return None
    fs = cv2.FileStorage(calib, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        return None
    K = fs.getNode("camera_matrix").mat()
    fs.release()
    try:
        return float(K[0][0])
    except Exception:
        return None


def fingerprint():
    """打印本仓库的代次指纹，返回 (是否自相矛盾, 失败项列表)。"""
    import base.cfg.settings as S

    im = S.vision.image
    clahe = im.clahe_clip
    gains = list(im.white_balance_bgr)
    gamma = im.gamma
    calib = S.vision.camera.front.calibration
    gate = S.get("vision.model.task_models.gate.path")
    ball = S.vision.model.path
    gm = md5_of(gate)
    fx = _fx_of(calib) if calib and os.path.exists(calib) else None

    print("==== 代次指纹（%s，单域 D+wb）====" % _ROOT)
    print("  image.undistort      = %s" % im.undistort)
    print("  image.clahe_clip     = %s      （本代应 0）" % clahe)
    print("  image.white_balance  = %s   （本代 = wb 增益）" % gains)
    print("  image.gamma          = %s" % gamma)
    print("  标定                  = %s  md5=%s  fx=%s" % (calib, md5_of(calib)[:8], fx))
    print("  门权重                = %s" % gate)
    print("                       md5=%s  size=%s" % (
        gm, os.path.getsize(gate) if gate and os.path.exists(gate) else "-"))
    print("  ball 权重             = %s  md5=%s" % (ball, md5_of(ball)[:8]))
    print("  model.mode           = %s" % S.vision.model.mode)
    print("  SIM_MODE             = %s" % getattr(S, "SIM_MODE", "?"))

    bad = []
    # ① 退役的域开关不许复活（配置里有它 = 有人又把两套方案加回来了）
    if S.get("vision.image.chain", None) is not None:
        bad.append("vision.image.chain=%r 已退役（A 域 2026-10-01 归档）⇒ 从 cfg 里删掉这条键"
                   % S.get("vision.image.chain"))
    # ② 链路本身
    if not bool(im.undistort):
        bad.append("image.undistort=%s 但本代链路要求 true（remap@640 是 D 链的一部分）" % im.undistort)
    # ③ D+wb 的画面补偿参数
    if abs(float(clahe) - 0.0) > 1e-9:
        bad.append("image.clahe_clip=%s 应为 0（本代关 CLAHE；老 A 域才是 0.5）" % clahe)
    if [round(float(g), 3) for g in gains] != _WB_GAINS:
        bad.append("image.white_balance_bgr=%s 应为 %s（D+wb 的 wb 增益）" % (gains, _WB_GAINS))
    if abs(float(gamma) - _GAMMA) > 1e-9:
        bad.append("image.gamma=%s 应为 %s" % (gamma, _GAMMA))
    # ④ 权重代次
    if gm != "-" and not gm.startswith(_GT_W_MD5):
        bad.append("门权重 md5=%s… 不是本代 %s…（阶段一 v3）⇒ 换权重请同步标定/参数与这张指纹表"
                   % (gm[:8], _GT_W_MD5))
    # ⑤ 标定代次
    if fx is not None and abs(fx - _FX) > _FX_TOL:
        bad.append("前视内参 fx=%.1f 不是本代 %.1f（水下重标）⇒ 老 A 域内参 782.5 已归档，别混用"
                   % (fx, _FX))
    if bad:
        print("  ✗ 代次指纹自相矛盾：")
        for b in bad:
            print("      · %s" % b)
    else:
        print("  ✓ 代次指纹自洽（链路 / wb / CLAHE / gamma / 权重 / 内参 同代）")
    return bool(bad), bad


def measure(frames_dir, limit=None):
    """样本帧上的角点置信度（v）分布 —— 用来看本代权重在该链路上的工作点。"""
    import cv2
    import numpy as np
    from gate.percept.gate_detector import build_gate_backend

    files = []
    for d in (frames_dir,) if isinstance(frames_dir, str) else frames_dir:
        if os.path.isdir(d):
            files += [os.path.join(d, f) for f in sorted(os.listdir(d))
                      if f.lower().endswith((".jpg", ".png"))]
    if limit:
        files = files[:limit]
    b = build_gate_backend()
    if b is None:
        print("  ✗ 门后端没建起来（权重缺失 / mode=mock / 未配置）⇒ 测不了")
        return 1
    b._vis_thr = 0.0                      # 关掉解码期硬门限，看模型原始 v
    print("==== 角点置信度分布（%d 帧，vis_thr=0 原始值）====" % len(files))
    rows = []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        for det in (b.detect(img) or []):
            kc = np.asarray(getattr(det, "kpt_conf"), dtype=float).ravel()
            rows.append(kc)
            print("  %-24s %s" % (os.path.basename(f), np.round(kc, 3).tolist()))
    if not rows:
        print("  （没有检出任何门实例）")
        return 0
    a = np.array(rows)
    print("  实例 %d 个；全体 v   min/中位/max = %.3f / %.3f / %.3f"
          % (len(a), a.min(), np.median(a), a.max()))
    print("  每实例最差角点(min of 4) 中位 = %.3f" % np.median(a.min(axis=1)))
    for thr in (0.5, 0.7, 0.8, 0.9):
        print("    四角全 >= %.1f 的实例占比 = %3.0f%%"
              % (thr, 100.0 * (a.min(axis=1) >= thr).mean()))
    return 0


def main():
    ap = argparse.ArgumentParser(description="识别方案代次指纹自检（单域 D+wb，只读）")
    ap.add_argument("--frames", default=None,
                    help="样本帧目录：测本代权重在该链路上的角点置信度分布")
    ap.add_argument("--limit", type=int, default=None, help="最多用几帧")
    a = ap.parse_args()

    bad, _items = fingerprint()
    rc = 1 if bad else 0
    if a.frames:
        rc = max(rc, measure(a.frames, a.limit))
    return rc


if __name__ == "__main__":
    sys.exit(main())
