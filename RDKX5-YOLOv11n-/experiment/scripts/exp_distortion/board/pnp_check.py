#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pnp_check.py — 标定档「合格性」快速判据（板端 /tmp 用，只读 dump，不碰相机与串口）

为什么需要它：`pnp_calib.py` 给的是单档 p50，但**单档重复性 ≈ ±3–5%**，
与我们要测的 4% 系统偏差同量级 —— 采集时船一漂，这档数字就废了，而离线是看不出来的。
本脚本在**每次抓完 dump 后立刻**判这档能不能用于标定：

    python3 /tmp/pnp_check.py log/pnp_0923/pnp_z177.jsonl

判据（三项全过才算合格）：
  ① 四角齐全率（min kpt_conf ≥ 0.9 的帧占比） ≥ 60%
  ② 最稳 30 帧窗口的「四边形水平跨度」相对 std ≤ 2%（船/门基本静止）
  ③ 门中心偏离光轴 ≤ 10% 画面宽（≈128 px）—— 太大时"距离"的口径歧义会吃掉几个百分点

输出：每项 PASS/FAIL + 该档的建议 z（最稳窗口的 fx·W/跨度 与全段中位），
不合格时给出**具体该怎么做**（摆正/静止/缩短采集）。
"""
from __future__ import annotations

import json
import sys

import numpy as np

CX, FX = 702.266, 1023.764      # 板端 front_camera.yaml（C 标定）光心与 fx
W = 0.770                        # geometry.frame_w（外缘）


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    rows = [json.loads(l) for l in open(path)]
    good = []
    for r in rows:
        for d in r.get("dets", []):
            k = d.get("kpts")
            if not k or len(k) < 4:
                continue
            kc = d.get("kpt_conf") or []
            if kc and min(kc) < 0.9:
                continue
            q = np.asarray(k, float)
            good.append((q[:, 0].ptp(), q.mean(0)[0], q.mean(0)[1]))
    if not good:
        print(f"✗ {path}: 没有任何四角齐全帧（上不了标定）")
        return 1

    A = np.array(good)
    span, cx_, cy_ = A[:, 0], A[:, 1], A[:, 2]
    full_rate = len(good) / max(1, len(rows))

    win = min(30, len(A))
    best = min(((span[i:i + win].std(), i) for i in range(len(A) - win + 1)),
               default=(float("nan"), 0))
    std_abs, i0 = best
    std_rel = std_abs / max(1e-9, np.median(span[i0:i0 + win]))
    off_u = abs(np.median(cx_) - CX)

    z_win = FX * W / np.median(span[i0:i0 + win])
    z_all = FX * W / np.median(span)

    ok1, ok2, ok3 = full_rate >= 0.60, std_rel <= 0.02, off_u <= 128
    print(f"档：{path}")
    print(f"  帧数 {len(rows)}｜四角齐全帧 {len(good)}（{full_rate*100:.0f}%）")
    print(f"  [{'PASS' if ok1 else 'FAIL'}] ① 齐全率 ≥60%")
    print(f"  [{'PASS' if ok2 else 'FAIL'}] ② 最稳 {win} 帧窗口 跨度相对 std = {std_rel*100:.1f}%（≤2%，第 {i0}–{i0+win} 帧）")
    print(f"  [{'PASS' if ok3 else 'FAIL'}] ③ 门中心偏光轴 |Δu| = {off_u:.0f} px（≤128 px）")
    print(f"  → 建议 z：最稳窗口 {z_win:.3f} m｜全段中位 {z_all:.3f} m（真值见文件名）")
    if ok1 and ok2 and ok3:
        print("  ✅ 合格：可进 pnp_calib 的多档拟合")
        return 0
    print("  ❌ 不合格，不要拿这档去拟合。怎么改：")
    if not ok2:
        print("     · 船/门**完全静止**再抓（系住船或顶住池边），采集 3–5 s 即可，别抓 12 s；")
    if not ok3:
        print("     · 把门摆到**画面中心**（横向移动船或转船），让 |Δu| 尽量 <50 px；")
    if not ok1:
        print("     · 让门在画面里完整且大一些（靠近些/别贴边），或降低 kpt conf 门槛重抓；")
    return 1


if __name__ == "__main__":
    sys.exit(main())
