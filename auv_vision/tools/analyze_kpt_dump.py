# -*- coding: utf-8 -*-
"""tools/analyze_kpt_dump.py — 从 preview_detect --dump 的 JSONL 看角点可得率与 PnP 命中率。

用途：回答"放宽哪个阈值能真正提高位姿可用率"（conf_thr / vis_thr / reproj_px）。
用法：python3 tools/analyze_kpt_dump.py <dump.jsonl> [more.jsonl ...]
"""
from __future__ import annotations

import collections
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                              # noqa: E402
import base.settings as S                                       # noqa: E402
from gate.geometry import (object_points, _iter_candidates,     # noqa: E402
                           reproj_rms, gate_pose)
from gate.gate_detector import board_camera                     # noqa: E402


def main():
    files = sys.argv[1:]
    if not files:
        print(__doc__)
        return 2
    cam = board_camera()
    obj3 = object_points()
    print("相机 fx=%.1f cx=%.1f cy=%.1f  门框 %.2fx%.2f m"
          % (cam.fx, cam.cx, cam.cy,
             float(S.get("vision.gate.geometry.frame_w", 0.7)),
             float(S.get("vision.gate.geometry.frame_h", 0.5))))
    for path in files:
        if not os.path.exists(path):
            print("跳过（不存在）: %s" % path)
            continue
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        kc = [d.get("kpt_conf") for r in rows for d in r.get("dets", [])
              if d.get("kpts")]
        kp = [np.asarray(d["kpts"], np.float64) for r in rows
              for d in r.get("dets", []) if d.get("kpts")]
        if not kc:
            print("\n%s：无角点数据" % os.path.basename(path))
            continue
        print("\n=== %s （%d 帧带角点）===" % (os.path.basename(path), len(kc)))
        print("  角点数分布 / 可解位姿(≥3) 随 conf_thr 变化：")
        for thr in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            c = collections.Counter(sum(1 for x in (k or []) if x >= thr)
                                    for k in kc)
            ge3 = sum(v for k, v in c.items() if k >= 3)
            print("    conf_thr=%.2f  角点数%s  → ≥3 的帧 %d/%d = %3.0f%%"
                  % (thr, dict(sorted(c.items())), ge3, len(kc),
                     100.0 * ge3 / len(kc)))
        print("  PnP 命中率随 reproj_px 变化（用该 conf_thr 下的有效角点）：")
        for thr in (0.6, 0.8):
            for rp in (8.0, 12.0, 16.0, 20.0, 25.0):
                ok = tot = 0
                prev = None
                for k, c in zip(kp, kc):
                    ids = [i for i in range(4) if (c or [0] * 4)[i] >= thr]
                    if len(ids) < 3:
                        continue
                    tot += 1
                    res = gate_pose(cam, obj3[ids], k[ids], prev=prev,
                                    reproj_thr=rp,
                                    z_bounds=(float(S.get("vision.gate.pnp.z_min", 0.2)),
                                              float(S.get("vision.gate.pnp.z_max", 15.0))))
                    if res is not None:
                        ok += 1
                        prev = res
                    else:
                        prev = None
                print("    conf_thr=%.2f reproj_px=%5.1f → %3d/%3d = %3.0f%%"
                      % (thr, rp, ok, tot, 100.0 * ok / max(tot, 1)))
        # 有效候选的 RMS 统计（看门槛该给多少）
        rms_all = []
        for k, c in zip(kp, kc):
            ids = [i for i in range(4) if (c or [0] * 4)[i] >= 0.6]
            if len(ids) < 3:
                continue
            best = None
            for r_, t_ in _iter_candidates(cam, obj3[ids], k[ids]):
                try:
                    rms = reproj_rms(cam, obj3[ids], k[ids], r_, t_)
                except Exception:
                    continue
                if best is None or rms < best:
                    best = rms
            if best is not None:
                rms_all.append(best)
        if rms_all:
            v = sorted(rms_all)
            print("  最佳候选 RMS(conf≥0.6)：p10=%.1f p50=%.1f p90=%.1f max=%.1f px"
                  % (v[int(0.1 * len(v))], st.median(v), v[int(0.9 * len(v)) - 1], v[-1]))

        # 帧间 z 跳变分布 → 判断 max_z_jump_m(默认 0.8) 是否过严
        for thr, rp in ((0.7, 20.0), (0.8, 16.0)):
            zs, dz = [], []
            prev = None
            for k, c in zip(kp, kc):
                ids = [i for i in range(4) if (c or [0] * 4)[i] >= thr]
                if len(ids) < 3:
                    prev = None
                    continue
                res = gate_pose(cam, obj3[ids], k[ids], prev=prev, reproj_thr=rp,
                                z_bounds=(float(S.get("vision.gate.pnp.z_min", 0.2)),
                                          float(S.get("vision.gate.pnp.z_max", 15.0))))
                if res is None:
                    prev = None
                    continue
                z = float(np.asarray(res[1]).ravel()[2])
                if prev is not None:
                    dz.append(abs(z - float(np.asarray(prev[1]).ravel()[2])))
                zs.append(z)
                prev = res
            if zs:
                d = sorted(dz)
                mx = float(S.get("vision.gate.pnp.max_z_jump_m", 0.8))
                over = sum(1 for x in d if x > mx)
                print("  接受位姿 conf≥%.1f reproj≤%.0fpx：z p50=%.2f ｜帧间|Δz| "
                      "p50=%s p90=%s max=%s ｜ 超 max_z_jump_m(%.2f) 的帧 %d/%d"
                      % (thr, rp, st.median(zs),
                         "%.2f" % st.median(d) if d else "-",
                         "%.2f" % d[int(0.9 * len(d)) - 1] if d else "-",
                         "%.2f" % d[-1] if d else "-", mx, over, len(d)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
