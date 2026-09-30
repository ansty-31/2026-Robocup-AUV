# -*- coding: utf-8 -*-
"""feature_coverage.py — 从逐帧任务日志整理「哪些功能在最近的下水测试里根本没被用到」。

用途：下水几轮之后，回答「我调的那些开关/分支，到底跑过没跑过」。
用法：python3 tools/analyze/feature_coverage.py log/*.jsonl [--json]

判定分三类（**只看日志与配置，不做推测**）：
  · 用到了   = 日志里出现了该分支特有的字段取值（phase/substate/action/mode/命令非零…）
  · 没用到   = 该轮日志里**一次都没出现**
  · 未部署   = 代码/配置里存在，但该轮日志时间之后才加（由 --after-ms 或人工判断）

注意：本工具只统计「日志能证明的东西」。像「深度保护是否真的压掉了上浮」这类
需要下位机告警行/遥测细节的，标记为「需控制台日志」，不在这里下结论。
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

# 工程根 = tools/<类>/x.py 往上**三**级（分类重整后本脚本深了一层）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _f(r, k, d=0.0):
    v = r.get(k, d)
    try:
        return float(v if v is not None else d)
    except (TypeError, ValueError):
        return float(d)


def scan(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if "frame" in r and "phase" in r]
    if not rows:
        return None
    ev = {
        "n": len(rows),
        "phase": collections.Counter(str(r.get("phase")) for r in rows),
        "substate": collections.Counter(str(r.get("substate")) for r in rows),
        "action": collections.Counter(str(r.get("action")) for r in rows),
        "mode": collections.Counter(str(r.get("mode")) for r in rows),
        "yaw_nz": sum(1 for r in rows if abs(_f(r, "yaw")) > 1e-9),
        "hdg_v": sum(1 for r in rows if r.get("hdg") is not None),
        "hdg_i": sum(1 for r in rows if int(_f(r, "hdg_i")) > 0),
        "surge_nz": sum(1 for r in rows if abs(_f(r, "surge")) > 1e-9),
        "surge_neg": sum(1 for r in rows if _f(r, "surge") < -1e-9),
        "pass": int(_f(rows[-1], "pass")),
        "reason": rows[-1].get("reason") or "",
        "ratio_max": max(_f(r, "ratio") for r in rows),
        "z_max": max(_f(r, "z") for r in rows),
        "t_span": _f(rows[-1], "t") - _f(rows[0], "t"),
    }
    # SEARCH 的横移波形（2026-09-20 起 SEARCH = 左右平移扫视，不许旋转）
    se = [r for r in rows if str(r.get("action")) == "search"
          or str(r.get("phase")) == "SEARCH"]
    ev["search_n"] = len(se)
    ev["search_right"] = any(_f(r, "sway") > 1e-9 for r in se)
    ev["search_left"] = any(_f(r, "sway") < -1e-9 for r in se)
    ev["search_yaw_nz"] = sum(1 for r in se if abs(_f(r, "yaw")) > 1e-9)
    return ev


def report(path, ev):
    a = ev["action"]
    ph = ev["phase"]
    md = ev["mode"]
    st = ev["substate"]
    print("\n=== %s ===" % os.path.basename(path))
    print("  帧数=%d 时长=%.1fs | phase=%s" % (ev["n"], ev["t_span"], dict(ph)))
    print("  mode=%s" % dict(md))
    print("  substate=%s" % {k: v for k, v in st.items() if k != "None"})
    print("  action=%s" % dict(a))
    print("  yaw≠0 帧=%d | hdg 有值帧=%d | hdg_i>0 帧=%d | surge≠0 帧=%d | surge<0 帧=%d | pass=%d %s"
          % (ev["yaw_nz"], ev["hdg_v"], ev["hdg_i"], ev["surge_nz"], ev["surge_neg"],
             ev["pass"], ("(" + ev["reason"] + ")") if ev["reason"] else ""))
    print("  SEARCH 帧=%d（右扫 %s / 左扫 %s / 其中 yaw≠0 %d）"
          % (ev["search_n"],
             "有" if ev["search_right"] else "无", "有" if ev["search_left"] else "无",
             ev["search_yaw_nz"]))

    def mark(used, why):
        return ("✔ 用过" if used else "✘ **没用过**") + ("" if used else "  ← " + why)

    print("  ---- 逐项覆盖度 ----")
    checks = [
        ("SEARCH 左右平移扫视（2026-09-20 起不许旋转）",
         ph.get("SEARCH", 0) > 0 or a.get("search", 0) > 0, "没进过 SEARCH"),
        ("SEARCH 真的在平移（sway 正负都出现过）", ev["search_right"] and ev["search_left"],
         "SEARCH 期间 sway 只出现一个方向（扫视没走完一个来回 / 只扫了一侧）"),
        ("SEARCH 期间不旋转（yaw=0）", ev["search_n"] > 0 and ev["search_yaw_nz"] == 0,
         "没进过 SEARCH（无从判）" if ev["search_n"] == 0 else
         "SEARCH 出现了 yaw —— 旋转搜索应已删除（见 cfg comm.gate.search）"),
        ("ALIGN.GOLDEN（位姿档居中）", st.get("GOLDEN", 0) > 0, "没进过 GOLDEN"),
        ("ALIGN.CREEP（慢速靠近）", st.get("CREEP", 0) > 0 or a.get("creep", 0) > 0, "没进过 CREEP"),
        ("ALIGN.HOLD（原地保持）", st.get("HOLD", 0) > 0 or a.get("hold", 0) > 0, "没进过 HOLD"),
        ("ALIGN.REACQUIRE（后退重取）", st.get("REACQUIRE", 0) > 0 or a.get("reacquire", 0) > 0,
         "没进过 REACQUIRE"),
        ("ALIGN.HDG（正航向：测→转→停稳→再测）", st.get("HDG", 0) > 0 or a.get("hdg", 0) > 0,
         "正航向没跑过（今天才加 / 未部署 / 未满足条件）"),
        ("APPROACH（进近）", ph.get("APPROACH", 0) > 0 or a.get("forward_slow", 0) > 0
         or a.get("forward_fast", 0) > 0, "没进过 APPROACH"),
        ("THROUGH（冲刺穿门）", ph.get("THROUGH", 0) > 0 or a.get("through", 0) > 0,
         "整轮没穿过门（出口一次都没触发）"),
        ("mode=full（4 角位姿）", md.get("full", 0) > 0, "没出现过 full"),
        ("mode=p3p（3 角位姿）", md.get("p3p", 0) > 0, "没出现过 p3p"),
        ("mode=width（对向 2 角）", md.get("width", 0) > 0, "没出现过 width"),
        ("mode=coarse（仅整框）", md.get("coarse", 0) > 0, "没出现过 coarse"),
        ("后退指令（surge<0）", ev["surge_neg"] > 0, "整轮没发过后退（REACQUIRE/lost_backward 没动过）"),
        ("yaw 指令（任何 yaw≠0）", ev["yaw_nz"] > 0,
         "整轮 yaw 恒 0 —— 居中已不用 yaw（2026-09-20 删除），只有正航向 ALIGN.HDG 会发 yaw；"
         "恒 0 = 正航向没跑过（未部署 / 未满足条件 / 没遥测）"),
        ("航向测量（日志 hdg 有值）", ev["hdg_v"] > 0, "没有一帧测到航向（full 帧都没有？）"),
        ("正航向迭代（hdg_i>0）", ev["hdg_i"] > 0, "正航向一次迭代都没发生"),
        ("过门计数（pass≥1）", ev["pass"] >= 1, "整轮没判过一次过门"),
        ("多门（pass≥2，pass_target≥2 才可能）", ev["pass"] >= 2, "只过了一个门/没过门"),
    ]
    for name, used, why in checks:
        print("    %-46s %s" % (name, mark(used, why)))
    print("  ⚠️ 需控制台日志才能判的项：限深保护是否真的压掉上浮、推流是否在跑、"
          "PnP 是否被拒（[GATE] 弃帧）、近距丢门判过门的 [GATE] 打印。")


def main():
    ap = argparse.ArgumentParser(description="从任务日志整理功能覆盖度（哪些没用过）")
    ap.add_argument("logs", nargs="+", help="JSONL 日志路径（支持通配）")
    args = ap.parse_args()
    files = []
    for pat in args.logs:
        files.extend(sorted(glob.glob(pat)) or [pat])
    n = 0
    for f in files:
        if not os.path.exists(f):
            print("跳过（不存在）: %s" % f)
            continue
        ev = scan(f)
        if ev is None:
            print("跳过（不是逐帧任务日志）: %s" % f)
            continue
        report(f, ev)
        n += 1
    print("\n共分析 %d 份日志。" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
