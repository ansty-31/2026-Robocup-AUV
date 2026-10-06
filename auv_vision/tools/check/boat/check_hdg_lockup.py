#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""板端自检：**转完反向平移（post_sway）会不会把 HDG 永久锁死** + 每门转向上限有没有通电。
用法：
退出码 0 = 全部通过（或只剩"按设计如此"的 NOTE）；1 = 有 FAIL（终端里会打印要改的那一行）。
（详细用法、判据与实测见 doc/注释历史.md）"""
import argparse
import os
import re
import sys

FAIL = []
WARN = []
OK = []
NOTE = []


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def _read_union(root, rels):
    """把若干相对路径的源码拼起来（跳过不存在的），返回 (源码, 实际读到的文件)。"""
    got, parts = [], []
    for rel in rels:
        s = _read(os.path.join(root, *rel.split("/")))
        if s is not None:
            got.append(rel)
            parts.append(s)
    return "\n".join(parts), got


def _read_dir(root, reldir, skip=()):
    d = os.path.join(root, *reldir.split("/"))
    got, parts = [], []
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".py") and f not in skip and f != "__init__.py":
                s = _read(os.path.join(d, f))
                if s is not None:
                    got.append("%s/%s" % (reldir, f))
                    parts.append(s)
    return "\n".join(parts), got


def _task_src(root):
    """相位机的全部代码（门面 + 方法簇；兼容旧布局的单文件）。"""
    front, f1 = _read_union(root, ("gate/gate_task.py", "gate/motion/gate_task.py"))
    clusters, f2 = _read_dir(root, "gate/motion", skip=("gate_task.py",))
    return front + "\n" + clusters, f1 + f2


def _align_src(root):
    """正航向 + 兜底表所在文件（合并后的 hdg.py、合并前的 heading_align.py、以及 params.py）。"""
    a, f1 = _read_union(root, ("gate/motion/hdg.py", "gate/motion/heading_align.py"))
    b, f2 = _read_union(root, ("gate/motion/params.py",))
    return a + "\n" + b, f1 + f2


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


def _dict_block(src, name):
    """取 `name = dict(...)` 的整块文本（跨行；用于确认某个键真的在兜底表里）。"""
    i = src.find("%s = dict(" % name)
    if i < 0:
        i = src.find("%s = {" % name)
        if i < 0:
            return None
        close = "}"
    else:
        close = ")"
    depth = 0
    for j in range(i, len(src)):
        if src[j] in "([{":
            depth += 1
        elif src[j] in ")]}":
            depth -= 1
            if depth == 0:
                return src[i:j + 1]
    return src[i:]


# 会**消费**这个窗口的方法（它们拿到窗口就 `return`，所以必须有过期路径）。
#   `_set_info` = 本机的**唯一下发口**，窗口就是在这里被消费/推进的（2026-09-30 拆分后）。
_CONSUMERS = ("_set_info", "_step", "_on_pose", "_on_width", "_on_coarse",
              "_tick_lost", "_tick_through")


def check_expiry(task_src):
    """① 每个**消费** post_sway 窗口的方法都必须有「过期 ⇒ 收手」的路径。"""
    n_guard = n_ok = 0
    for name, ln, body in _methods(task_src):
        if name not in _CONSUMERS or "_post_sway_until_ms is not None" not in body:
            continue
        if not re.search(r"if .*_post_sway_until_ms is not None", body):
            continue
        n_guard += 1
        # 过期判据（本机用局部 `_now`，旧版用 `now_ms`）
        if re.search(r"(_now|now_ms)\s*(>=|<)\s*self\._post_sway_until_ms", body):
            n_ok += 1
            OK.append("%s：窗口有「过期 ⇒ 收手」路径（判据在 %d 行起）" % (name, ln))
        else:
            FAIL.append("%s（%d 行起）**消费 post_sway 窗口但没有过期判据** ⇒ "
                        "窗口过期后仍会 return，HDG 永久锁死；给它加 "
                        "`_now >= self._post_sway_until_ms` 的无条件收手分支" % (name, ln))
    if not n_guard:
        WARN.append("没有任何方法消费 post_sway 窗口（这一层可能已整体删除 —— 另一种可接受状态）")
    elif n_ok == 0:
        FAIL.append("**没有任何过期判据** ⇒ 只要有一帧把窗口内的门判成「远门」，HDG 就再也不动了")


def check_same_gate(task_src):
    """② 「是不是刚转丢的那个门」：**本机已按用户决定删除该判据**（2026-09-30）。"""
    body = None
    for name, ln, b in _methods(task_src):
        if name == "_post_sway_same_gate":
            body = b
    if body is None:
        NOTE.append("本机没有 `_post_sway_same_gate` —— **按用户决定删除**（2026-09-30）："
                    "postsway 退出只看「当前门角点数 ≥ post_sway_kpt_min」+ 超时兜底，"
                    "不再判「是不是同一个门」")
        return
    if "_post_sway_z_ref" in body and "post_sway_same_z_m" in body:
        OK.append("`_post_sway_same_gate` 用 z 做主判据（z 不受转向影响）")
    else:
        FAIL.append("`_post_sway_same_gate` 没把 z（`post_sway_same_z_m`）当判据 ⇒ "
                    "转向后同一个门会被判成「远门」（框 ∝cosψ）")
    if "self._z_last" in body:
        WARN.append("`_post_sway_same_gate` 里出现 `self._z_last`（**上一帧**的 z）⇒ "
                    "转向前后 z 一样，判据失效（应把本帧位姿的 z 传进来）")


def check_cfg(cfg_src):
    """③ cfg 阈值：`max_turns` 必须存在且 = 4（每门转向次数上限）。"""
    if cfg_src is None:
        WARN.append("读不到 cfg/comm.yaml")
        return
    if re.search(r"^\s*(post_sway_far_ratio|post_sway_same_z_m|entry_psi_first|psi_first_)", cfg_src, re.M):
        WARN.append("cfg 里还留着已废弃的键（post_sway_far_ratio / post_sway_same_z_m / "
                    "entry_psi_first / psi_first_*）⇒ 代码不读它们，删掉免得误导")
    else:
        NOTE.append("cfg 里没有已废弃的 post_sway_far_ratio / psi_first_* 键（按 2026-09-30 决定删除）")
    m = re.search(r"max_turns:\s*([0-9]+)", cfg_src)
    if m:
        v = int(m.group(1))
        (OK if v >= 1 else WARN).append(
            "cfg gate.hdg.max_turns=%d（≥1 即可：2026-10-02 起降级为兜底，主闸是"
            "「一门只转一次」；值以 cfg 为准）" % v)
    else:
        FAIL.append("cfg gate.hdg 里没有 `max_turns` ⇒ 每门转向上限没有可调旋钮"
                    "（旧版靠它治「疯狂旋转」，别丢）")


def check_cap_wired(task_src, align_src):
    """④ 每门转向次数上限必须真的通电：计数 + 判据 + 调用点 + 兜底表里都有这个键。"""
    if "_hdg_turns_full" not in task_src:
        FAIL.append("没有 `_hdg_turns_full`（每门转向上限没实现 ⇒ 转向无效时会自转）")
    if "_hdg_turns +=" not in task_src and "_hdg_turns+=" not in task_src:
        FAIL.append("没有 `_hdg_turns += 1`（转完不计数 ⇒ 上限永远不触发）")
    if "_hdg_turns_full" in task_src:
        cap = "_hdg_turns_full()" in task_src.split("def _hdg_turns_full")[0] or \
              "_hdg_turns_full()" in task_src.split("def _hdg_turns_full")[-1].split("\n", 1)[-1]
        if "def _hdg_turns_full" in task_src and not cap:
            FAIL.append("`_hdg_turns_full()` 没有被调用（只在定义处出现 ⇒ 上限没接到起转判据上）")
        elif "def _hdg_turns_full" in task_src:
            OK.append("`_hdg_turns_full()` 已接到起转判据上（`_hdg_ready`）")
    blk = _dict_block(align_src, "_D_HDG")
    if blk is None:
        WARN.append("找不到 `_D_HDG` 兜底表（读到的文件见结尾清单）")
    elif "max_turns" not in blk:
        FAIL.append("`_D_HDG` 里没有 `max_turns` ⇒ `hdg_cfg()` 会把 cfg 里的 `max_turns` "
                    "**静默丢掉**（cfg 看起来生效、其实没读）")
    else:
        OK.append("`_D_HDG` 里有 max_turns（cfg 的键真的会被读到）")


def check_kpt_exit(task_src, cfg_src):
    """⑤ 新设计的退出判据：只看**当前门**的可信角点数（+ 超时兜底）。"""
    if "_post_sway_kpt_count" not in task_src:
        WARN.append("没有 `_post_sway_kpt_count`（本机设计里 postsway 的退出只看当前门角点数）")
        return
    if not re.search(r"self\._post_sway_kpt_count\(\)", task_src):
        FAIL.append("`_post_sway_kpt_count()` 定义了却没被调用 ⇒ 角点数判据没通电")
    elif cfg_src is not None and "post_sway_kpt_min" not in cfg_src:
        WARN.append("cfg 里没有 post_sway_kpt_min（会用兜底表的值；角点数门限就没法现场调）")
    else:
        OK.append("postsway 退出 = 当前门可信角点数 ≥ `post_sway_kpt_min`（+ 超时兜底）")


def main():
    ap = argparse.ArgumentParser(description="post_sway / HDG 锁死自检（只读）")
    ap.add_argument("--root", default=".", help="工程根（默认当前目录）")
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    task_src, task_files = _task_src(root)
    align_src, align_files = _align_src(root)
    cfg = _read(os.path.join(root, "cfg", "comm.yaml"))
    if not task_files:
        print("❌ 读不到相位机代码（试过 gate/gate_task.py 与 gate/motion/gate_task.py）")
        return 1

    check_expiry(task_src)
    check_same_gate(task_src)
    check_cfg(cfg)
    check_cap_wired(task_src, align_src)
    check_kpt_exit(task_src, cfg)

    print("=" * 78)
    print("post_sway / HDG 锁死自检 —— 工程根 %s" % root)
    print("=" * 78)
    for s in OK:
        print("  ✅ %s" % s)
    for s in NOTE:
        print("  📝 %s" % s)
    for s in WARN:
        print("  ⚠️  %s" % s)
    for s in FAIL:
        print("  ❌ %s" % s)
    print("-" * 78)
    print("实际扫描：%s" % "、".join(task_files + align_files))
    print("结论：%s（OK %d / NOTE %d / WARN %d / FAIL %d）"
          % ("通过" if not FAIL else "**有必须修的问题**", len(OK), len(NOTE), len(WARN), len(FAIL)))
    if FAIL:
        print("""
提示：本机（开发机）是修好的一侧，代码在 `gate/gate_task.py` + `gate/motion/*.py`；
      这些路径**不在** tools/deploy/board_forks.txt 的跳过清单里
      ⇒ 按项目既有同步约定，它们本来就该与开发机一致，可直接覆盖过去。""")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
