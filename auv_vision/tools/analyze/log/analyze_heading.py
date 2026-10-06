# -*- coding: utf-8 -*-
"""tools/analyze/log/analyze_heading.py — 从角点 dump 估「机身正不正」能不能测（航向估计的噪声底）。
用法：python3 tools/analyze/log/analyze_heading.py <dump.jsonl> [...]
（详细用法、判据与实测见 doc/注释历史.md）"""
from __future__ import annotations

import json
import os
import statistics as st
import sys

# 工程根 = 向上第一个含 `cfg/` 的目录（**别写死层级**：脚本搬过位置，写死会静默指错）
_ROOT = os.path.dirname(os.path.abspath(__file__))
while _ROOT != os.path.dirname(_ROOT) and not os.path.isdir(os.path.join(_ROOT, "cfg")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)

import numpy as np                                              # noqa: E402
import base.cfg.settings as S                                       # noqa: E402
from gate.percept.geometry import (object_points, gate_pose,            # noqa: E402
                           plane_from_pose)
from gate.percept.gate_detector import board_camera                     # noqa: E402

CONF_THR = float(S.get("vision.gate.keypoint.conf_thr", 0.7) or 0.7)
REPROJ = float(S.get("vision.gate.pnp.reproj_px", 20.0) or 20.0)
Z_MIN = float(S.get("vision.gate.pnp.z_min", 0.2) or 0.2)
Z_MAX = float(S.get("vision.gate.pnp.z_max", 15.0) or 15.0)

def _ema(xs, k):
    """因果 EMA（等价于一阶低通），返回滤波后的序列。"""
    a = 2.0 / (k + 1.0)
    out, s = [], None
    for x in xs:
        s = x if s is None else (1 - a) * s + a * x
        out.append(s)
    return out

def _deg(v):
    return float(np.degrees(v))

def main():
    files = sys.argv[1:]
    if not files:
        print(__doc__)
        return 2
    cam = board_camera()
    obj3 = object_points()
    print("相机 fx=%.1f fy=%.1f cx=%.1f cy=%.1f（标定 %s）"
          % (cam.fx, cam.fy, cam.cx, cam.cy,
             "去畸变域" if getattr(cam, "rectified", False) else "原始域"))
    print("门槛：conf_thr=%.2f  reproj≤%.0fpx  z∈[%.1f,%.1f]"
          % (CONF_THR, REPROJ, Z_MIN, Z_MAX))
    print("参考：门 %.2fx%.2f m；10° 航向误差在 z=1.5m 处 = 横向 %.2f m"
          % (float(S.get("vision.gate.percept.geometry.frame_w", 0.7)),
             float(S.get("vision.gate.percept.geometry.frame_h", 0.5)),
             1.5 * np.tan(np.radians(10.0))))

    for path in files:
        if not os.path.exists(path):
            print("\n跳过（不存在）: %s" % path)
            continue
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        yaws, pitches, rolls, zs, rmss, dxs = [], [], [], [], [], []
        prev = None
        n_frames = 0
        for r in rows:
            for d in r.get("dets", []):
                if not d.get("kpts"):
                    continue
                k = np.asarray(d["kpts"], np.float64)
                c = np.asarray(d.get("kpt_conf") or np.zeros(4), np.float64)
                ids = [i for i in range(4) if c[i] >= CONF_THR]
                if len(ids) < 3:
                    continue
                n_frames += 1
                res = gate_pose(cam, obj3[ids], k[ids], prev=prev,
                                reproj_thr=REPROJ, z_bounds=(Z_MIN, Z_MAX))
                if res is None:
                    prev = None
                    continue
                prev = res
                rv, tv = res
                n_vec, _rho = plane_from_pose(rv, tv)
                # 航向/俯仰：门法向相对光轴的角度（机身正对门时为 0）
                yaws.append(_deg(np.arctan2(n_vec[0], n_vec[2])))
                pitches.append(_deg(np.arctan2(n_vec[1], n_vec[2])))
                # 门内滚转：门 x 轴（图像里"向右"那条）相对水平的角度
                R = np.asarray(__import__("cv2").Rodrigues(
                    np.asarray(rv, np.float64))[0], np.float64)
                rolls.append(_deg(np.arctan2(R[1, 0], R[0, 0])))
                zs.append(float(np.asarray(tv).ravel()[2]))
                dxs.append(float(np.asarray(tv).ravel()[0]))
                from gate.percept.geometry import reproj_rms
                rmss.append(reproj_rms(cam, obj3[ids], k[ids], rv, tv))
        print("\n=== %s （%d 帧有角点，%d 帧解出位姿）==="
              % (os.path.basename(path), n_frames, len(yaws)))
        if len(yaws) < 5:
            print("  可用位姿太少，跳过统计")
            continue

        def stat(name, xs, unit="°"):
            xs = np.asarray(xs, np.float64)
            print("  %-12s p10=%+7.2f p50=%+7.2f p90=%+7.2f  std=%5.2f  "
                  "范围=[%+.2f,%+.2f] %s"
                  % (name, np.percentile(xs, 10), np.median(xs),
                     np.percentile(xs, 90), xs.std(), xs.min(), xs.max(), unit))
            return xs

        y = stat("航向 θ_yaw", yaws)
        stat("俯仰 θ_pitch", pitches)
        stat("滚转 θ_roll", rolls)
        stat("深度 z", zs, "m")
        stat("重投影 RMS", rmss, "px")

        # 帧间抖动 = 噪声底（静态场景下就是纯噪声）
        d = np.abs(np.diff(y))
        print("  帧间 |Δθ_yaw|：p50=%.2f° p90=%.2f° max=%.2f°（静态场景≈纯噪声）"
              % (np.median(d), np.percentile(d, 90), d.max()))
        print("  滤波后噪声（因果 EMA，静态场景 std 应随之下降）：")
        for k in (2, 4, 8, 16):
            f = np.asarray(_ema(list(y), k), np.float64)
            half = f[len(f) // 2:]
            print("    EMA k=%-3d → std=%5.2f°（后半段）" % (k, half.std()))
        # 与横向偏移的相关性：怀疑"航向估计其实是被横向偏移带出来的"
        if len(dxs) == len(y) and np.std(dxs) > 1e-6:
            cc = float(np.corrcoef(dxs, y)[0, 1])
            print("  corr(tvec.x, θ_yaw) = %+.2f（明显非 0 → 两者在当前几何下耦合）" % cc)
        print("  ⇒ 判据：噪声底 <2° 才能拿它做 yaw 闭环；3~8° 只能做『慢慢校』；"
              ">8° 等于测不出来")
    return 0

if __name__ == "__main__":
    sys.exit(main())
