# -*- coding: utf-8 -*-
"""tools/check_gate_pose.py — 查"为什么位姿被拒"（只读相机，不驱动推进器）。

打印每帧：角点置信度 / mode / 候选解数量 / 最佳候选的 RMS 与 tz / gate_pose 最终结果。
用法（板端）：python3 tools/check_gate_pose.py [秒数]
"""
import os
import sys
import time

for _p in (os.getcwd(), os.path.dirname(os.path.abspath(__file__))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                     # noqa: E402
import base.settings as S                              # noqa: E402
from base.camera import create_camera                  # noqa: E402
from gate.gate_detector import build_gate_backend, board_camera   # noqa: E402
from gate.gate_frontend import parse_kpt_mode          # noqa: E402
from gate.geometry import object_points, _iter_candidates, reproj_rms, gate_pose   # noqa: E402
from gate.kpt_memory import build_kpt_memory           # noqa: E402

DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
cam = board_camera()
obj3 = object_points()
V = S.vision.gate
conf_thr = float(S.get("vision.gate.keypoint.conf_thr", 0.5))
pnp = V.get("pnp", {}) or {}
reproj_thr = float(pnp.get("reproj_px", 8.0))
zb = (float(pnp.get("z_min", 0.2)), float(pnp.get("z_max", 15.0)))

backend = build_gate_backend()                 # 板端 cfg 的 vis_thr(0.5) 已能看到原始置信度
if backend is None:
    print("后端不可用")
    sys.exit(3)
km = build_kpt_memory(V.get("kpt_mem", {}) or {}, n_kpt=4)
camera = create_camera("front")
print("conf_thr=%.2f reproj_thr=%.1fpx z_bounds=%s kpt_mem=%s"
      % (conf_thr, reproj_thr, zb, km is not None))
print("%-5s %-6s %-4s %-4s %-22s %-6s %-7s %-6s %s"
      % ("f", "mode", "raw", "fus", "kpt_conf(raw)", "cand", "rms_min", "tz", "gate_pose"))

prev = None
n_ok = n_rej_rms = n_rej_z = n_nocand = n_ok_all = 0
t0 = time.time()
f = 0
while time.time() - t0 < DUR:
    frame = camera.read()
    if frame is None:
        time.sleep(0.02)
        continue
    f += 1
    dets = [d for d in backend.detect(frame) if d.kind == "gate"]
    if not dets:
        continue
    d = max(dets, key=lambda x: float(np.sum(x.kpt_conf)) if x.kpt_conf is not None else -1)
    kc = np.asarray(d.kpt_conf, np.float32)
    n_raw = int((kc >= conf_thr).sum())
    fk, fc = (km.update(d.kpts, d.kpt_conf, time.monotonic() * 1000)
              if km is not None else (d.kpts, d.kpt_conf))
    mode, ids = parse_kpt_mode(np.asarray(fk, np.float32),
                               np.asarray(fc, np.float32), conf_thr)
    n_fus = len(ids)
    rms_min = tz = None
    n_cand = 0
    if n_fus >= 3:
        cands = _iter_candidates(cam, obj3[ids], np.asarray(fk)[ids], prev=prev)
        n_cand = len(cands)
        best = None
        for r, t in cands:
            try:
                rms = reproj_rms(cam, obj3[ids], np.asarray(fk)[ids], r, t)
            except Exception:
                continue
            z_ = float(np.asarray(t).ravel()[2])
            if best is None or rms < best[0]:
                best = (rms, z_)
        if best:
            rms_min, tz = best
        res = gate_pose(cam, obj3[ids], np.asarray(fk)[ids], prev=prev,
                        reproj_thr=reproj_thr, z_bounds=zb,
                        refine=bool(pnp.get("refine", True)))
        if res is not None:
            n_ok += 1
            prev = res
        elif n_cand == 0:
            n_nocand += 1
        elif rms_min is not None and rms_min > reproj_thr:
            n_rej_rms += 1
        elif tz is not None and not (zb[0] <= tz <= zb[1]):
            n_rej_z += 1
        else:
            n_ok_all += 1     # 候选被别的条件滤掉（如 refine 后超界）
        if rms_min is not None and rms_min <= reproj_thr:
            pass
    else:
        res = None
    print("%-5d %-6s %-4d %-4d %-22s %-6d %-7s %-6s %s"
          % (f, mode, n_raw, n_fus,
             " ".join("%.2f" % c for c in kc[:4]),
             n_cand, "-" if rms_min is None else "%.2f" % rms_min,
             "-" if tz is None else "%.2f" % tz,
             "None" if res is None else "OK z=%.2f dx=%+.2f dy=%+.2f"
             % (float(np.asarray(res[1]).ravel()[2]),
                float(np.asarray(res[1]).ravel()[0]),
                float(np.asarray(res[1]).ravel()[1]))))

print("---- 汇总：帧 %d，gate_pose 成功 %d，被拒 %d（RMS>%.1fpx: %d ; z 越界: %d ; 无候选: %d ; 其它: %d）"
      % (f, n_ok, f - n_ok, reproj_thr, n_rej_rms, n_rej_z, n_nocand, n_ok_all))
