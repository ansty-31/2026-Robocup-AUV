# -*- coding: utf-8 -*-
"""tools/analyze/log/analyze_pnp_center.py — 位姿档"居中误差"到底可不可信？（p3p vs full）
用法：python3 tools/analyze/log/analyze_pnp_center.py <dump.jsonl> [...]
（详细用法、判据与实测见 doc/注释历史.md）"""
from __future__ import annotations

import json
import os
import sys

# 工程根 = 向上第一个含 `cfg/` 的目录（**别写死层级**：脚本搬过位置，写死会静默指错）
_ROOT = os.path.dirname(os.path.abspath(__file__))
while _ROOT != os.path.dirname(_ROOT) and not os.path.isdir(os.path.join(_ROOT, "cfg")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)

import numpy as np                                              # noqa: E402
import base.cfg.settings as S                                       # noqa: E402
from gate.percept.geometry import (object_points, gate_pose,            # noqa: E402
                           reproj_rms, gate_normal_angles_deg)
from gate.percept.gate_detector import board_camera                     # noqa: E402

CONF_THR = float(S.get("vision.gate.keypoint.conf_thr", 0.7) or 0.7)
REPROJ = float(S.get("vision.gate.pnp.reproj_px", 20.0) or 20.0)
Z_MIN = float(S.get("vision.gate.pnp.z_min", 0.2) or 0.2)
Z_MAX = float(S.get("vision.gate.pnp.z_max", 15.0) or 15.0)

def _pct(v, p):
    return float(np.percentile(v, p)) if len(v) else float("nan")

def main():
    files = sys.argv[1:]
    if not files:
        print(__doc__)
        return 2
    cam = board_camera()
    obj3 = object_points()
    W, H = float(cam.width), float(cam.height)
    print("门槛 conf=%.2f reproj≤%.0fpx；判据：位姿误差可信 ⇔ |中位偏差| 小 且 符号相反少"
          % (CONF_THR, REPROJ))
    for path in files:
        if not os.path.exists(path):
            print("\n跳过（不存在）: %s" % path)
            continue
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        buckets = {4: {"d": [], "opp": 0, "z": [], "rms": [], "hdg": []},
                   3: {"d": [], "opp": 0, "z": [], "rms": [], "hdg": []}}
        prev = None
        n_tot = 0
        for r in rows:
            for det in r.get("dets", []):
                if not det.get("kpts") or not det.get("bbox"):
                    continue
                k = np.asarray(det["kpts"], np.float64)
                c = np.asarray(det.get("kpt_conf") or np.zeros(4), np.float64)
                ids = [i for i in range(4) if c[i] >= CONF_THR]
                if len(ids) < 3:
                    prev = None
                    continue
                n_tot += 1
                res = gate_pose(cam, obj3[ids], k[ids], prev=prev,
                                reproj_thr=REPROJ, z_bounds=(Z_MIN, Z_MAX))
                if res is None:
                    prev = None
                    continue
                prev = res
                rv, tv = res
                uv = cam.project(np.zeros((1, 3), np.float32), rv, tv)[0]
                dxn_pose = (float(uv[0]) - W / 2.0) / (W / 2.0)
                x, y, bw, bh = [float(v) for v in det["bbox"]]
                dxn_bbox = ((x + bw / 2.0) - W / 2.0) / (W / 2.0)
                b = buckets[len(ids)]
                b["d"].append(dxn_pose - dxn_bbox)
                if dxn_pose * dxn_bbox < 0 and abs(dxn_bbox) > 0.05:
                    b["opp"] += 1
                b["z"].append(float(np.asarray(tv).ravel()[2]))
                b["rms"].append(reproj_rms(cam, obj3[ids], k[ids], rv, tv))
                b["hdg"].append(gate_normal_angles_deg(rv, tv)[0])
        print("\n=== %s （%d 帧 ≥3 角）===" % (os.path.basename(path), n_tot))
        for n in (4, 3):
            b = buckets[n]
            d = np.asarray(b["d"], np.float64)
            if len(d) < 3:
                print("  %d 角：样本不足（%d）" % (n, len(d)))
                continue
            print("  %d 角（%s）%d 帧：偏差(pose-bbox) 中位=%+.3f 均值=%+.3f "
                  "p10=%+.3f p90=%+.3f std=%.3f  ｜符号相反 %d 帧(%.0f%%)"
                  % (n, "full" if n == 4 else "p3p", len(d),
                     float(np.median(d)), float(d.mean()), _pct(d, 10), _pct(d, 90),
                     float(d.std()), b["opp"], 100.0 * b["opp"] / len(d)))
            print("       z p50=%.2f ｜RMS p50=%.1f p90=%.1f px ｜航向 p50=%+.1f° std=%.1f°"
                  % (float(np.median(b["z"])), float(np.median(b["rms"])),
                     _pct(b["rms"], 90), float(np.median(b["hdg"])),
                     float(np.std(b["hdg"]))))
    return 0

if __name__ == "__main__":
    sys.exit(main())
