#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_domain.py — 「识别方案（域）」指纹与核对（本地/板端都能跑，不入运行时）

工程里同时存在**两套识别方案**，它们必须与门权重成对出现（跨域掉点）：

    | 域 | 链路（`vision.image.chain`） | 门权重 | clahe_clip |
    |---|---|---|---|
    | A（原方案） | remap@720p → resize640 → enhance | AUV_4 `d38b803e…` | 0.5 |
    | D（新方案） | resize640 → enhance → remap@640 | 阶段一 `g240_i16` `ef52c18b…` | 0 |

板端两个仓库的分工见 `tools/README.md` §6。本脚本回答三个问题：

    python3 tools/check/check_domain.py                 # ① 我这个仓库现在是哪一域？
    python3 tools/check/check_domain.py --frames DIR    # ② 本域权重在样本帧上的角点置信度分布
    python3 tools/check/check_domain.py \
        --equiv-ref bak/Adomain_ref/preprocess.py       # ③ A 域改造前后是否逐像素一致

③ 的用法：`--equiv-ref` 给一份**参考实现**（例如改造前旧仓库的 `common/preprocess.py`）。
本脚本会用**相同参数**（size/gains/clip/gamma/标定）跑"参考实现 / 本仓库 A 序 / 本仓库 D 序"，
输出三者的逐像素差 —— 参考 vs A 序应当是 0（证明"原识别方案"没被改动），
A vs D 的差则是**跨域差**的量级（拿它判断"两个仓库结果不可直接比较"）。

退出码：0=一致/无异常；1=域指纹自相矛盾（如 chain=D 却配 AUV_4 权重）或等价性超差。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys

import numpy as np

_ROOT_UP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _project_root():
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        if os.path.isdir(os.path.join(here, "cfg")):
            return here
        here = os.path.dirname(here)
    return _ROOT_UP


_ROOT = _project_root()
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

#: 代次指纹：域 ↔ 门权重 md5 前缀（改动权重文件时**必须**同步这里与 tools/README.md §6）
DOMAIN_WEIGHTS = {
    "A": ("d38b803e", "AUV_4（原识别方案）"),
    "D": ("ef52c18b", "阶段一 g240_i16（新识别方案）"),
}
#: 每个域的链路与配套参数（写入 cfg 的期望值）
DOMAIN_EXPECT = {
    "A": {"chain": "A", "clahe_clip": 0.5},
    "D": {"chain": "D", "clahe_clip": 0.0},
}


def md5_of(path):
    if not path or not os.path.exists(path):
        return "-"
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fingerprint():
    """打印本仓库的域指纹，返回 (chain, gate_path, gate_md5, 是否自相矛盾)。"""
    import base.settings as S

    chain = str(S.get("vision.image.chain", "D")).strip().upper()
    clahe = S.vision.image.clahe_clip
    calib = S.vision.camera.front.calibration
    gate = S.get("vision.model.task_models.gate.path")
    ball = S.vision.model.path

    print("==== 域指纹（%s）====" % _ROOT)
    print("  chain（识别方案）  = %s   [%s]" % (
        chain, DOMAIN_EXPECT.get(chain, {}).get("chain", "?")))
    print("  clahe_clip         = %s   （A 域应 0.5 / D 域应 0）" % clahe)
    print("  undistort          = %s" % S.vision.image.undistort)
    print("  model.mode         = %s" % S.vision.model.mode)
    print("  标定                = %s  md5=%s" % (calib, md5_of(calib)[:8]))
    print("  门权重              = %s" % gate)
    gm = md5_of(gate)
    print("                     md5=%s  size=%s" % (
        gm, os.path.getsize(gate) if gate and os.path.exists(gate) else "-"))
    print("  ball 权重           = %s  md5=%s" % (ball, md5_of(ball)[:8]))
    print("  SIM_MODE           = %s" % getattr(S, "SIM_MODE", "?"))

    bad = []
    exp = DOMAIN_EXPECT.get(chain)
    if exp is None:
        bad.append("chain=%r 不是 A/D" % chain)
    elif abs(float(clahe) - exp["clahe_clip"]) > 1e-9:
        bad.append("chain=%s 但 clahe_clip=%s（应为 %s）" % (chain, clahe, exp["clahe_clip"]))
    if gm != "-" and chain in DOMAIN_WEIGHTS:
        want_prefix, want_name = DOMAIN_WEIGHTS[chain]
        if not gm.startswith(want_prefix):
            bad.append("chain=%s 应对应 %s（md5 %s…），实际门权重 md5=%s ⇒ **跨域配对**"
                       % (chain, want_name, want_prefix, gm[:8]))
    if bad:
        print("  ✗ 域自相矛盾：")
        for b in bad:
            print("      · %s" % b)
    else:
        print("  ✓ 域指纹自洽（chain / clahe / 权重三代次对得上）")
    return chain, gate, gm, bool(bad)


def measure(frames_dir, limit=None):
    """样本帧上的角点置信度（v）分布 —— 用来判断该域权重的工作点。"""
    import cv2
    from gate.gate_detector import build_gate_backend

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


def equiv(ref_path, frames_dir, limit=4):
    """参考实现 vs 本仓库 A 序 vs 本仓库 D 序：逐像素差。"""
    import cv2
    from common.preprocess import ModelPreprocessor
    import base.settings as S

    ref_mod = load_module(ref_path, "preprocess_ref")
    calib = S.vision.camera.front.calibration
    size = S.vision.model.input_size
    gains = S.vision.image.white_balance_bgr
    gamma = S.vision.image.gamma
    clip = DOMAIN_EXPECT["A"]["clahe_clip"]        # 固定成 A 域参数，隔离"顺序"这一个变量
    kw = dict(undistort=True, gains=list(gains), gamma=gamma, size=size,
              calib_path=calib)

    ref = ref_mod.ModelPreprocessor(clip=clip, **kw)
    new_a = ModelPreprocessor(chain="A", clip=clip, **kw)
    new_d = ModelPreprocessor(chain="D", clip=clip, **kw)

    files = [os.path.join(frames_dir, f) for f in sorted(os.listdir(frames_dir))
             if f.lower().endswith((".jpg", ".png"))][:limit]
    print("==== 链路等价性（%d 帧；参数固定 clip=%.2f，只比「顺序」这一个变量） ===="
          % (len(files), clip))
    print("  参考 = %s" % ref_path)
    worst_ref, worst_ad = 0, 0
    for f in files:
        raw = cv2.imread(f)
        if raw is None:
            continue
        o_ref, o_a, o_d = ref.process(raw), new_a.process(raw), new_d.process(raw)
        d_ref = np.abs(o_ref.astype(np.int16) - o_a.astype(np.int16))
        d_ad = np.abs(o_a.astype(np.int16) - o_d.astype(np.int16))
        worst_ref = max(worst_ref, int(d_ref.max()))
        worst_ad = max(worst_ad, int(d_ad.max()))
        print("  %-22s 参考vsA: max=%2d 均值=%.4f 不同像素=%5.2f%%   |  A vs D: max=%2d 均值=%.3f"
              % (os.path.basename(f), d_ref.max(), d_ref.mean(),
                 100.0 * (d_ref > 0).mean(), d_ad.max(), d_ad.mean()))
    print("  → 参考 vs A 序：全帧最大差 = %d %s"
          % (worst_ref, "✓ 逐像素一致（原识别方案未被改动）" if worst_ref == 0 else "⚠️ 有差"))
    print("  → A vs D 跨域差：全帧最大差 = %d（两仓库结果不可直接逐帧比较）" % worst_ad)
    return 0 if worst_ref <= 1 else 1


def main():
    ap = argparse.ArgumentParser(description="识别方案（域）指纹与核对")
    ap.add_argument("--frames", default=None,
                    help="样本帧目录：测本域权重在该域的角点置信度分布")
    ap.add_argument("--limit", type=int, default=None, help="最多用几帧")
    ap.add_argument("--equiv-ref", default=None,
                    help="参考 preprocess.py 路径：与它的 A 序逐像素比对")
    ap.add_argument("--equiv-frames", default=None, help="等价性比对用的帧目录")
    a = ap.parse_args()

    chain, gate, gm, bad = fingerprint()
    rc = 1 if bad else 0

    if a.equiv_ref:
        frames = a.equiv_frames or a.frames
        if not frames or not os.path.isdir(frames):
            print("  ⚠️ --equiv-ref 需要 --equiv-frames（或 --frames）指向一个帧目录")
            rc = 1
        else:
            rc = max(rc, equiv(a.equiv_ref, frames, a.limit or 4))

    if a.frames:
        rc = max(rc, measure(a.frames, a.limit))
    return rc


if __name__ == "__main__":
    sys.exit(main())
