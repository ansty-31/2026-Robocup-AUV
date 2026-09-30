#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""板端自检：**转完反向平移（post_sway）会不会把 HDG 永久锁死**。
python3 tools/check/boat/check_hdg_lockup.py --root /home/sunrise/AUV
退出码 0 = 全部通过；1 = 有 FAIL（终端里会打印要改的那一行）。"""
import argparse
import os
import re
import sys

FAIL = []
WARN = []
OK = []


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def _methods(src):
    """切出 (方法名, 起始行号, 方法体) —— 只按 4 空格缩进的 def 切，够用且不会误伤内部函数。"""
    out, cur, name, ln = [], [], None, 0
    for i, line in enumerate(src.splitlines(), 1):
        m = re.match(r"    def (\w+)\(", line)
        if m:
            if name:
                out.append((name, ln, "\n".join(cur)))
            name, ln, cur = m.group(1), i, [line]
        elif name is not None:
            if line and not line.startswith((" ", "\t")) and not line.startswith("#"):
                out.append((name, ln, "\n".join(cur)))
                name, cur = None, []
            else:
                cur.append(line)
    if name:
        out.append((name, ln, "\n".join(cur)))
    return out


# 会**消费**这个窗口的方法（它们拿到窗口就 `return`，所以必须有过期路径）
_CONSUMERS = ("_step", "_on_pose", "_on_width", "_on_coarse", "_tick_lost", "_tick_through")


def check_expiry(task_src):
    """① 每个**消费** post_sway 窗口的方法都必须有「过期 ⇒ 收手」的路径。"""
    n_guard = n_ok = 0
    for name, ln, body in _methods(task_src):
        if name not in _CONSUMERS or "_post_sway_until_ms is not None" not in body:
            continue
        if not re.search(r"if .*_post_sway_until_ms is not None", body):
            continue
        n_guard += 1
        if re.search(r"now_ms\s*(>=|<)\s*self\._post_sway_until_ms", body):
            n_ok += 1
            OK.append("%s：窗口有「过期 ⇒ 收手」路径（判据在 %d 行起）" % (name, ln))
        else:
            FAIL.append("%s（%d 行起）**消费 post_sway 窗口但没有过期判据** ⇒ "
                        "窗口过期后仍会 return，HDG 永久锁死；给它加 "
                        "`now_ms >= self._post_sway_until_ms` 的无条件收手分支" % (name, ln))
    if not n_guard:
        WARN.append("没有任何方法消费 post_sway 窗口（这一层可能已整体删除 —— 另一种可接受状态）")
    elif n_ok == 0:
        FAIL.append("**没有任何过期判据** ⇒ 只要有一帧把窗口内的门判成「远门」，HDG 就再也不动了")


def check_same_gate(task_src):
    """② 「是不是刚转丢的那个门」必须**先用 z**（转向不改距离），占比只是退路且阈值 ≤ 0.5。"""
    body = None
    for name, ln, b in _methods(task_src):
        if name == "_post_sway_same_gate":
            body = b
    if body is None:
        WARN.append("没有 `_post_sway_same_gate`（没有「同一门」判据 ⇒ 远端门一出现就顶掉刚丢的门）")
        return
    if "_post_sway_z_ref" in body and "post_sway_same_z_m" in body:
        OK.append("`_post_sway_same_gate` 用 z 做主判据（z 不受转向影响）")
    else:
        FAIL.append("`_post_sway_same_gate` 没把 z（`post_sway_same_z_m`）当判据 ⇒ "
                    "转向后同一个门会被判成「远门」（框 ∝cosψ）")
    if "self._z_last" in body and "z=None" not in body and "z=None" not in body.replace("z=None", ""):
        WARN.append("`_post_sway_same_gate` 里出现 `self._z_last`（**上一帧**的 z）⇒ "
                    "转向前后 z 一样，判据失效（应把本帧位姿的 z 传进来）")


def check_cfg(cfg_src):
    """③ cfg 阈值：`post_sway_far_ratio` ≤ 0.5；`max_turns` = 4（每门转向次数上限）。"""
    if cfg_src is None:
        WARN.append("读不到 cfg/comm.yaml")
        return
    m = re.search(r"post_sway_far_ratio:\s*([0-9.]+)", cfg_src)
    if m:
        v = float(m.group(1))
        (OK if v <= 0.5 else FAIL).append(
            "cfg gate.hdg.post_sway_far_ratio=%g%s" % (
                v, "" if v <= 0.5 else "  → 必须 ≤ 0.5（取 0.7 会把同一个门判成远门）"))
    else:
        WARN.append("cfg 里没有 post_sway_far_ratio（会用代码默认 0.5，可接受）")
    m = re.search(r"max_turns:\s*([0-9]+)", cfg_src)
    if m:
        v = int(m.group(1))
        (OK if v == 4 else WARN).append("cfg gate.hdg.max_turns=%d（用户定 4）" % v)
    else:
        WARN.append("cfg 里没有 max_turns（会用代码默认 4，可接受）")


def check_cap_wired(task_src, align_src):
    """④ 每门转向次数上限必须真的通电：计数 + 判据 + 兜底表里都有这个键。"""
    if "_hdg_turns_full" not in task_src:
        FAIL.append("gate_task.py 没有 `_hdg_turns_full`（每门转向上限没实现 ⇒ 转向无效时会自转）")
    if "_hdg_turns +=" not in task_src and "_hdg_turns+=" not in task_src:
        FAIL.append("gate_task.py 没有 `_hdg_turns += 1`（转完不计数 ⇒ 上限永远不触发）")
    if "_hdg_turns_full" in task_src and "_hdg_turns_full()" not in task_src.split("def _hdg_turns_full")[0]:
        FAIL.append("`_hdg_turns_full()` 没有被调用（只在定义处出现 ⇒ 上限没接到起转判据上）")
    if align_src is not None and "max_turns" not in align_src:
        FAIL.append("gate/motion/heading_align.py 的 `_D_HDG` 里没有 `max_turns` ⇒ "
                    "`hdg_cfg()` 会把 cfg 里的 `max_turns` **静默丢掉**（cfg 看起来生效、其实没读）")
    elif align_src is not None:
        OK.append("heading_align._D_HDG 里有 max_turns（cfg 的键真的会被读到）")


def main():
    ap = argparse.ArgumentParser(description="post_sway / HDG 锁死自检（只读）")
    ap.add_argument("--root", default=".", help="工程根（默认当前目录）")
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    task = _read(os.path.join(root, "gate", "gate_task.py"))
    align = _read(os.path.join(root, "gate", "heading_align.py"))
    cfg = _read(os.path.join(root, "cfg", "comm.yaml"))
    if task is None:
        print("❌ 读不到 %s/gate/motion/gate_task.py" % root)
        return 1

    check_expiry(task)
    check_same_gate(task)
    check_cfg(cfg)
    check_cap_wired(task, align)

    print("=" * 78)
    print("post_sway / HDG 锁死自检 —— 工程根 %s" % root)
    print("=" * 78)
    for s in OK:
        print("  ✅ %s" % s)
    for s in WARN:
        print("  ⚠️  %s" % s)
    for s in FAIL:
        print("  ❌ %s" % s)
    print("-" * 78)
    print("结论：%s（OK %d / WARN %d / FAIL %d）"
          % ("通过" if not FAIL else "**有必须修的问题**", len(OK), len(WARN), len(FAIL)))
    if FAIL:
        print("""
提示：本机（开发机）已修好的版本在 auv_vision/gate/motion/gate_task.py 与 gate/motion/heading_align.py，
      `gate/motion/gate_task.py`、`gate/motion/heading_align.py` **不在** tools/deploy/board_forks_Adomain.txt
      的跳过清单里 ⇒ 按项目既有同步约定，它们本来就该与开发机一致，可直接覆盖过去。""")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
