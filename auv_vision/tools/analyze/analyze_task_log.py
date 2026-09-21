# -*- coding: utf-8 -*-
"""tools/analyze/analyze_task_log.py — 分析 main.py 的逐帧任务日志（AUV_TASK_LOG 产出的 JSONL）

用途：下水/台架跑完后，判断"到底是识别、位姿、还是决策在卡"。
不依赖硬件，纯离线：读 JSONL + 用 cfg 里的当前阈值反推判据。

用法：
    python3 tools/analyze/analyze_task_log.py <log.jsonl> [--task gate]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as st
import sys

# 工程根 = tools/<类>/x.py 往上**三**级（分类重整后本脚本深了一层）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import base.settings as S                                    # noqa: E402


def _g(r, k, d=0.0):
    v = r.get(k, d)
    return d if v is None else v


def _rng(name, vals):
    v = [x for x in vals if x is not None]
    if not v:
        print("   %-8s 无数据" % name)
        return
    vs = sorted(v)
    print("   %-8s min=%+.3f  p10=%+.3f  p50=%+.3f  p90=%+.3f  max=%+.3f"
          % (name, vs[0], vs[int(0.1 * len(vs))], st.median(vs),
             vs[int(0.9 * len(vs)) - 1], vs[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--no-lazy", action="store_true",
                    help="不用 gate 相机模型（跳过 z 交叉核对）")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.log, encoding="utf-8") if l.strip()]
    if not rows:
        print("空日志")
        return 2
    n = len(rows)
    dur = _g(rows[-1], "t") - _g(rows[0], "t")
    print("=" * 72)
    print("帧数=%d  时长=%.1fs  fps≈%.1f  任务=%s"
          % (n, dur, n / max(dur, 1e-9), rows[-1].get("task", "?")))

    # ---- ① mode：coarse 里 kpt≥3 的 = PnP 被拒（位姿失败降级） ----
    print("\n① mode 分布（**kpt≥3 却是 coarse = PnP 失败被降级**）")
    for k, v in collections.Counter(r.get("mode", "") for r in rows).most_common():
        print("   %-8s %4d (%.0f%%)" % (k, v, 100.0 * v / n))
    bad = [r for r in rows if r.get("mode") == "coarse" and int(_g(r, "kpt")) >= 3]
    print("   → kpt≥3 却 coarse：%d 帧 (%.0f%%)  ← 位姿被拒"
          % (len(bad), 100.0 * len(bad) / n))

    # ---- ② 相位 ----
    print("\n② phase / substate / action")
    print("   phase   :", dict(collections.Counter(r.get("phase", "") for r in rows)))
    print("   substate:", dict(collections.Counter(r.get("substate", "") for r in rows)))
    print("   action  :", dict(collections.Counter(r.get("action", "") for r in rows)))

    # ---- ③ 角点 ----
    print("\n③ 每帧有效角点数")
    print("   kpt(融合后):", dict(collections.Counter(int(_g(r, "kpt")) for r in rows)))
    print("   kpt_raw    :", dict(collections.Counter(int(_g(r, "kpt_raw")) for r in rows)))

    # ---- ④ 误差 / 命令 ----
    print("\n④ 误差与命令分布")
    for k in ("dx", "dy", "z", "ratio", "sway", "heave", "yaw", "surge"):
        if k in rows[0]:
            _rng(k, [_g(r, k) for r in rows])

    # ---- ⑤ heave 健康度（结合下潜放大与执行器死区） ----
    dc = S.get("comm.dof_comp", None) or {}
    ds = abs(float(dc.get("dive_scale", 1.0) or 1.0))
    dz = float(S.get("comm.gate.dz_comp.act_deadzone", 0.138) or 0.138)
    dz = float(S.get("comm.dof_comp.act_deadzone", dz) or dz)
    h = [_g(r, "heave") for r in rows]
    eff = [max(-1.0, x * ds) if x < 0 else x for x in h]
    print("\n⑤ heave 健康度（执行器死区 %.3f，dive_scale=%.2f）" % (dz, ds))
    for label, cond in (("|heave|≥0.95 顶满", lambda x: abs(x) >= 0.95),
                        ("0<|heave|<死区（原无效）", lambda x: 0 < abs(x) < dz),
                        ("heave==0（不动）", lambda x: abs(x) < 1e-9)):
        c = sum(1 for x in h if cond(x))
        print("   %-22s %4d 帧 (%.0f%%)" % (label, c, 100.0 * c / n))
    c = sum(1 for x in eff if 0 < abs(x) < dz)
    print("   %-22s %4d 帧 (%.0f%%)   ← 放大后仍进不了死区"
          % ("放大后仍<死区（无效）", c, 100.0 * c / n))

    # ---- ⑥ 用当前阈值反推"居中成功" ----
    al = S.get("comm.gate.align", None) or {}
    co = S.get("comm.gate.coarse", None) or {}
    px = float(al.get("px_x", 0.10))
    py = float(al.get("px_y", 0.10))
    cx = float(co.get("align_x", 0.10))
    cy = float(co.get("align_y", 0.18))
    print("\n⑥ 居中成功（aligned）反推 —— **全档位按像素**（位姿档也一样，dx/dy 即归一化像素偏差）")
    print("   |dx|≤%.2f 且 |dy|≤%.2f : %d 帧 (%.0f%%)"
          % (px, py, sum(1 for r in rows
                         if abs(_g(r, "dx")) <= px and abs(_g(r, "dy")) <= py),
             100.0 * sum(1 for r in rows
                         if abs(_g(r, "dx")) <= px and abs(_g(r, "dy")) <= py) / n))
    print("   coarse |dx|≤%.2f,|dy|≤%.2f : %d 帧"
          % (cx, cy, sum(1 for r in rows if r.get("mode") == "coarse"
                         and abs(_g(r, "dx")) <= cx and abs(_g(r, "dy")) <= cy)))
    hold = [r for r in rows if r.get("action") == "hold"]
    if hold:
        dx = sorted(abs(_g(r, "dx")) for r in hold)
        dy = sorted(abs(_g(r, "dy")) for r in hold)
        print("   HOLD 帧（%d）误差: |dx| p50=%.3f p90=%.3f | |dy| p50=%.3f p90=%.3f"
              % (len(hold), st.median(dx), dx[int(0.9 * len(dx)) - 1],
                 st.median(dy), dy[int(0.9 * len(dy)) - 1]))

    # ---- ⑦ z 与"框宽测距"交叉核对 ----
    if not a.no_lazy:
        try:
            from gate.gate_detector import board_camera
            cam = board_camera()
            fw = float(S.get("vision.gate.geometry.frame_w", 0.70))
            d = []
            for r in rows:
                ratio, z = _g(r, "ratio"), _g(r, "z")
                if ratio > 0.05 and z > 0.1:
                    zw = cam.fx * fw / (ratio * cam.width)
                    d.append((z, zw, (z - zw) / zw))
            if d:
                rel = sorted(x[2] for x in d)
                print("\n⑦ z(PnP) vs z(框宽粗估，假设门宽 %.2fm、正对)" % fw)
                print("   样本 %d：z p50=%.2f / z_w p50=%.2f → 相对偏差 p10=%+.0f%% p50=%+.0f%% p90=%+.0f%%"
                      % (len(d), st.median([x[0] for x in d]), st.median([x[1] for x in d]),
                         100 * rel[int(0.1 * len(d))], 100 * st.median(rel),
                         100 * rel[int(0.9 * len(d)) - 1]))
                print("   偏差大 → 位姿尺度可疑 / 门宽假设不对 / 框含倒影或余量（需卷尺实测定论）")
        except Exception as e:
            print("\n⑦ 跳过 z 交叉核对：%s" % e)

    # ---- ⑦b 航向（hdg）：机身正不正 / bias 标定值 ----
    hd = [(r.get("mode"), _g(r, "hdg")) for r in rows if r.get("hdg") is not None]
    print("\n⑦b 航向误差 hdg（度；0 = 机身正对门。**只有 full 的位姿可信**，p3p 航向 std 64~69°）")
    if not hd:
        print("   无 hdg 数据（旧日志没有该字段 / 全程没出过位姿档）")
    else:
        full_h = sorted(v for m, v in hd if m == "full")
        p3p_h = sorted(v for m, v in hd if m == "p3p")
        for nm, v in (("full", full_h), ("p3p", p3p_h)):
            if len(v) < 3:
                print("   %-5s 样本不足（%d）" % (nm, len(v)))
                continue
            print("   %-5s %4d 帧: p10=%+6.1f p50=%+6.1f p90=%+6.1f std=%5.1f°"
                  % (nm, len(v), v[int(0.1 * len(v))], st.median(v),
                     v[int(0.9 * len(v)) - 1], st.pstdev(v)))
        if len(full_h) >= 5:
            print("   => 这个中位数就是**正航向要消掉的姿态偏置**（相机安装偏置已包含在内）："
                  "%+.1f°" % st.median(full_h))
            print("      收敛阈值见 comm.gate.hdg.tol_deg（用户定 8~10°）；"
                  "若中位数明显偏一个常数，先怀疑相机—机身安装偏置"
                  "（`gate.geometry.body_center_offset`，尚未标定），不要用别的手段去「补偿」它。")
            print("   注：full 的 std = 此刻机身实际在晃多少，不是噪声底；"
                  "噪声底用静态 dump 测（tools/analyze/analyze_heading.py）")

    # ---- ⑦c 水平通道"能动力"：为什么调不动 ----
    # ⚠️ 2026-09-20 起**居中只有 sway 通道**（`comm.gate.align_yaw` 已删除；
    #     gate 里唯一的 yaw 来源是 ALIGN.HDG 正航向 → 增益 comm.motion.turn_pid）。
    yaw_out = float((S.get("comm.motion.turn_pid", None) or {}).get("out_max", 0.45) or 0.45)
    sway_out = float((S.get("comm.gate.pid_sway", None) or {}).get(
        "out_max", (S.get("comm.motion.pid_sway", None) or {}).get("out_max", 0.45)) or 0.45)
    sw = [_g(r, "sway") for r in rows]
    yw = [_g(r, "yaw") for r in rows]
    print("\n⑦c 水平通道能动力（执行器死区 %.3f；sway 上限 %.2f=25%% 推力）" % (dz, sway_out))
    print("    注：yaw 若为 0 属**正常**（居中不用转向；只有正航向 ALIGN.HDG 才发 yaw，上限 %.2f）"
          % yaw_out)
    for nm, v, om in (("sway", sw, sway_out), ("yaw", yw, yaw_out)):
        dead = sum(1 for x in v if 0 < abs(x) < dz)
        top = sum(1 for x in v if abs(x) >= om - 1e-9)
        zero = sum(1 for x in v if abs(x) < 1e-9)
        print("   %-5s 顶满 %3d帧(%.0f%%) ｜死区内(无效) %3d帧(%.0f%%) ｜=0 %3d帧(%.0f%%)"
              % (nm, top, 100.0 * top / n, dead, 100.0 * dead / n,
                 zero, 100.0 * zero / n))
    print("   => sway 顶满多 = 横移已到天花板（要更强：comm.gate.pid_sway.out_max 0.45→0.8）")

    # ---- ⑦d 冲刺瞬间偏了多少（蹭杆的直接证据） ----
    th = []
    for i in range(1, len(rows)):
        if rows[i - 1].get("phase") != "THROUGH" and rows[i].get("phase") == "THROUGH":
            r0 = rows[i - 1]
            th.append((_g(r0, "dx"), _g(r0, "dy"), _g(r0, "z"), r0.get("mode", ""),
                       _g(r0, "ratio")))
    print("\n⑦d 进入 THROUGH 前的最后一帧（= 冲刺瞬间的偏心；蹭不蹭杆看这里）")
    if not th:
        print("   没有进入过 THROUGH")
    for dx, dy, z, md, rt in th[:10]:
        if md == "coarse":
            # coarse 档没有测距：日志里的 z 是上一次 width/位姿留下的陈旧值（可能是垃圾），
            # 拿它换算 cm 会得到一个假数（现场见过 9.25 → "偏心 20cm"）。
            print("   dx=%+.3f dy=%+.3f 框占比=%.2f mode=coarse → **该档无测距，别用 z 换算 cm**"
                  "（z=%s 是陈旧值）" % (dx, dy, rt, "%.2f" % z))
        else:
            print("   dx=%+.3f dy=%+.3f z=%.2f mode=%-7s → 该距离处偏心 ≈ %.1fcm / %.1fcm"
                  % (dx, dy, z, md, abs(dx) * 0.818 * z * 100, abs(dy) * 0.462 * z * 100))

    # ---- ⑦e 在门口徘徊：到了门边却还在慢爬多久 ----
    near_r = float((S.get("comm.gate.z", None) or {}).get("near_lost_ratio", 0.60) or 0.60)
    creep_surge = float((S.get("comm.gate.surge", None) or {}).get("creep", 0.20) or 0.20)
    thr = float((S.get("comm.gate.surge", None) or {}).get("through", 0.5) or 0.5)
    at_door = [r for r in rows if _g(r, "ratio") >= near_r]
    print("\n⑦e 在门口徘徊（框占比 ≥ %.2f 视为「已在门口」）" % near_r)
    if at_door:
        act = collections.Counter(r.get("action") for r in at_door)
        print("   %d 帧 (%.0f%%) 已在门口；其中 action: %s"
              % (len(at_door), 100.0 * len(at_door) / n, dict(act)))
        # 连续"已在门口但只用 creep/非冲刺速度"的最长时间
        worst = 0.0
        run_start = None
        for r in rows:
            slow = (_g(r, "ratio") >= near_r) and abs(_g(r, "surge")) < thr - 1e-9
            if slow and run_start is None:
                run_start = _g(r, "t")
            elif not slow and run_start is not None:
                worst = max(worst, _g(r, "t") - run_start)
                run_start = None
        if run_start is not None:
            worst = max(worst, _g(rows[-1], "t") - run_start)
        print("   最长连续「已在门口却没用冲刺速度」时长 = %.1fs  ← 这段就是白等/蹭门风险" % worst)
        print("   => 若这段很长：出口③「在门口超时兜底」没触发？看 loiter.timeout_ms / dx_max / dy_max")
    else:
        print("   从未到过门口（占比始终 < %.2f）" % near_r)

    # ---- ⑧ 时间线（相位/子状态切换点） ----
    print("\n⑧ 时间线（切换点，最多 25 条）")
    last = None
    shown = 0
    for r in rows:
        key = (r.get("phase"), r.get("substate"), r.get("mode"), r.get("action"))
        if key != last and shown < 25:
            print("   %7.2fs f=%-4d %-9s %-9s %-7s %-10s z=%5.2f dx=%+.2f dy=%+.2f kpt=%d"
                  % (_g(r, "t") - _g(rows[0], "t"), int(_g(r, "frame")),
                     r.get("phase", ""), r.get("substate", ""), r.get("mode", ""),
                     r.get("action", ""), _g(r, "z"), _g(r, "dx"), _g(r, "dy"),
                     int(_g(r, "kpt"))))
            last = key
            shown += 1
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
