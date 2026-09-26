#!/usr/bin/env python3
"""把已有的 YOLO 数据集重新划分 train/val/test（序列感知，避免相邻帧泄漏）。

先把帧按「同一来源流内帧号相差 <= --group-gap」聚成连续段，再整段随机分配到各
split，所以同一片段的相邻帧不会跨 split。来源流 = 文件名 `_frame_` 之前的部分
（旧命名 `frame_000123…` 为空串）；不同来源流的帧号互不相干，不能跨流比较。

各 split 目标数 = --ratios × 总数（最大余数法取整，每个 split 至少 1）；在「段不跨
split」约束下多轮贪婪重启 + 模拟退火，按优先级权衡：数量 → 类别均衡 → 时间泄漏
（帧号相差 <= --leak-window 却跨 split 的帧对）→ 稀有类保护（全量实例少于
--rare-threshold 的类，含该类的段强制留在 train；--no-rare-to-train 关闭）。

用法：
    python scripts/1_prepare/resplit_dataset.py data/datasets/AUV_2_AUV.yolov11 --dry-run   # 只看方案，不动文件
    python scripts/1_prepare/resplit_dataset.py data/datasets/AUV_2_AUV.yolov11             # 执行（先备份）
    python scripts/1_prepare/resplit_dataset.py <数据集> --ratios 0.7 0.2 0.1 --seed 7
    python scripts/1_prepare/resplit_dataset.py <数据集> --group-gap -1                     # 关闭聚合 = 纯随机
    python scripts/1_prepare/resplit_dataset.py <数据集> --leak-window 10 --leak-weight 0.02

输出：就地重排 <dataset>/{train,valid,test}/{images,labels}/；
      <dataset>/split_manifest.csv（每张图 old/new split、段号、帧号、标注框数）；
      备份 <dataset>_backup_<时间戳>/（--no-backup 关闭）。
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
# 两种命名都要认：
#   frame_000292_jpg.rf.<hash>                 → stream="",                 num=292
#   AUV_5_gate-11_frame_000001_jpg.rf.<hash>   → stream="AUV_5_gate-11",    num=1
FRAME_RE = re.compile(r"^(?P<stream>.*?)_?frame_(?P<num>\d+)", re.IGNORECASE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="序列感知地把 YOLO 数据集重新划分为 train/val/test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("dataset", type=Path, help="数据集根目录（含 train/valid/test 与 data.yaml）")
    p.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1),
                   metavar=("TRAIN", "VAL", "TEST"), help="三个 split 的比例")
    p.add_argument("--names", nargs=3, default=("train", "valid", "test"),
                   metavar=("TRAIN", "VAL", "TEST"), help="三个 split 的目录名")
    p.add_argument("--group-gap", type=int, default=2,
                   help="帧号差 <= 该值的帧视为同一连续段；-1 表示关闭聚合（纯随机）")
    p.add_argument("--seed", type=int, default=0, help="随机种子（可复现）")
    p.add_argument("--restarts", type=int, default=1000, help="贪心随机重启次数")
    p.add_argument("--refine-rounds", type=int, default=24, help="退火微调的随机起点数")
    p.add_argument("--refine-iters", type=int, default=200000, help="每次退火的交换尝试次数")
    p.add_argument("--anneal-t0", type=float, default=50.0, help="退火初始温度（允许恶化的幅度）")
    p.add_argument("--anneal-t1", type=float, default=0.02, help="退火终止温度")
    p.add_argument("--balance-weight", type=float, default=1.0,
                   help="类别实例均衡在打分中的权重（0 = 只看数量）")
    p.add_argument("--leak-weight", type=float, default=0.005,
                   help="跨 split 相邻帧对数（时间泄漏）在打分中的权重")
    p.add_argument("--leak-window", type=int, default=5,
                   help="把帧号相差 <= 该值的帧对视为近似重复，纳入泄漏代价")
    p.add_argument("--rare-threshold", type=int, default=5,
                   help="全量实例数少于该值的类别视为稀有类，含稀有类的段强制留在 train")
    p.add_argument("--no-rare-to-train", dest="rare_to_train", action="store_false",
                   default=True, help="关闭稀有类保护（允许稀有类只出现在 val/test）")
    p.add_argument("--backup", dest="backup", action="store_true", default=True,
                   help="执行前把现有划分复制到 <dataset>_backup_<时间戳>")
    p.add_argument("--no-backup", dest="backup", action="store_false", help="不备份")
    p.add_argument("--dry-run", action="store_true", help="只打印方案，不移动文件")
    p.add_argument("--force", action="store_true", help="方案偏差过大时仍然执行")
    return p.parse_args(argv)


def frame_key(stem: str) -> tuple[str, int | None]:
    """文件名 → (来源流, 帧号)。来源流 = `_frame_` 之前的部分（旧命名为空串）；
    相邻关系只在同一来源流内成立。"""
    m = FRAME_RE.match(stem)
    if not m:
        return "", None
    return (m.group("stream") or "").strip("_"), int(m.group("num"))


def scan(dataset: Path, names: tuple[str, str, str]) -> list[dict]:
    """收集三个 split 下的所有图片，校验 label 配对，返回记录列表。"""
    records: list[dict] = []
    for split in names:
        img_dir = dataset / split / "images"
        lbl_dir = dataset / split / "labels"
        if not img_dir.is_dir():
            print(f"[warn] 缺少目录：{img_dir}", file=sys.stderr)
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lbl = lbl_dir / f"{img.stem}.txt"
            _st, _num = frame_key(img.stem)
            records.append({
                "src_split": split,
                "img": img,
                "lbl": lbl,
                "has_label": lbl.is_file(),
                "stream": _st,
                "frame": _num,
            })
    if not records:
        sys.exit(f"[error] {dataset} 下没有找到任何图片")
    return records


def group_records(records: list[dict], gap: int) -> list[list[dict]]:
    """按「同一来源流内帧号相邻」聚成连续段；无帧号的图各自成段。
    gap < 0 时每张图各自成段（= 纯随机划分）。"""
    if gap < 0:
        return [[r] for r in records]
    by_stream: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    no_num: list[dict] = []
    for r in records:
        if r["frame"] is None:
            no_num.append(r)
        else:
            by_stream[r.get("stream", "")][r["frame"]].append(r)
    groups: list[list[dict]] = []
    for st, by_num in by_stream.items():
        nums = sorted(by_num)
        cur = [nums[0]]
        for a, b in zip(nums, nums[1:]):
            if b - a <= max(gap, 1):  # gap=0：同帧号（跨前缀变体）并在一起
                cur.append(b)
            else:
                groups.append([r for n in cur for r in by_num[n]])
                cur = [b]
        groups.append([r for n in cur for r in by_num[n]])
    if no_num:
        print(f"[warn] {len(no_num)} 张图没有可解析的帧号（各自成段）；"
              f"请确认命名形如 frame_000123… 或 <来源>_frame_000123…", file=sys.stderr)
    groups.extend([r] for r in no_num)
    return groups


def round_targets(total: int, ratios: list[float]) -> list[int]:
    """最大余数法取整，总和恰好等于 total，且每个 split 至少 1。"""
    raw = [total * r / sum(ratios) for r in ratios]
    base = [int(math.floor(x)) for x in raw]
    for i in sorted(range(3), key=lambda i: raw[i] - base[i], reverse=True)[: total - sum(base)]:
        base[i] += 1
    while min(base) < 1:  # 极端比例下保底
        big = base.index(max(base))
        if base[big] <= 1:
            break
        base[big] -= 1
        base[base.index(0)] += 1
    return base


def label_classes(path: Path) -> Counter:
    c: Counter = Counter()
    if not path.is_file():
        return c
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if parts:
            c[parts[0]] += 1
    return c


COUNT_WEIGHT = 50.0  # 每偏离目标 1 张图的代价，远大于类别/泄漏项，确保数量优先


def build_context(groups: list[list[dict]], window: int):
    """预计算每段的类别计数，以及帧号相距 <= window 的跨段相邻对数（邻接表）。"""
    gcls = []
    for g in groups:
        c: Counter = Counter()
        for r in g:
            c.update(r["classes"])
        gcls.append(c)
    num2g: dict[tuple[str, int], int] = {}
    for gi, g in enumerate(groups):
        for r in g:
            if r["frame"] is not None:
                num2g[(r.get("stream", ""), r["frame"])] = gi
    adj: list[dict[int, int]] = [dict() for _ in groups]
    ks = sorted(num2g)
    for idx, a in enumerate(ks):
        for b in ks[idx + 1:]:
            if b[0] != a[0] or b[1] - a[1] > window:   # 只比同一来源流内相邻的帧
                break
            ga, gb = num2g[a], num2g[b]
            if ga != gb:
                adj[ga][gb] = adj[ga].get(gb, 0) + 1
                adj[gb][ga] = adj[gb].get(ga, 0) + 1
    return gcls, adj


def balance_term(cls: list[Counter], cls_tot: Counter, targets: list[int]) -> float:
    """各类别实例在各 split 的占比与目标占比的相对偏差之和。"""
    s = 0.0
    tot_t = sum(targets)
    for c, n in cls_tot.items():
        if n < 3:  # 实例太少的类无法分层，跳过
            continue
        for i in range(3):
            s += abs(cls[i][c] - n * targets[i] / tot_t) / n
    return s


def combined(counts: list[int], cls: list[Counter], leak: int, cls_tot: Counter,
             targets: list[int], w: float, leak_w: float) -> float:
    return (COUNT_WEIGHT * sum(abs(counts[i] - targets[i]) for i in range(3))
            + w * balance_term(cls, cls_tot, targets) + leak_w * leak)


def leak_count(assign: list[int], adj: list[dict[int, int]]) -> int:
    return sum(w for gi, nb in enumerate(adj) for gj, w in nb.items()
               if gj > gi and assign[gj] != assign[gi])


def state_of(assign: list[int], groups: list[list[dict]], gcls: list[Counter],
             adj: list[dict[int, int]]):
    counts = [0, 0, 0]
    cls: list[Counter] = [Counter() for _ in range(3)]
    for gi, g in enumerate(groups):
        counts[assign[gi]] += len(g)
        cls[assign[gi]].update(gcls[gi])
    return counts, cls, leak_count(assign, adj)


def greedy_once(groups: list[list[dict]], targets: list[int], rng: random.Random,
                protected: frozenset[int] = frozenset()) -> list[int] | None:
    """大段优先，放进「缺口最大且装得下」的 split；稀有类所在的段先钉进 train。"""
    seq = sorted(range(len(groups)), key=lambda i: (-len(groups[i]), rng.random()))
    counts = [0, 0, 0]
    assign = [-1] * len(groups)
    for gi in sorted(protected, key=lambda i: -len(groups[i])):
        if counts[0] + len(groups[gi]) > targets[0]:
            return None  # 稀有类段放不进 train，交由调用方决定是否放弃保护
        assign[gi] = 0
        counts[0] += len(groups[gi])
    for gi in seq:
        if assign[gi] >= 0:
            continue
        size = len(groups[gi])
        cands = [i for i in range(3) if counts[i] + size <= targets[i]]
        if not cands:
            return None
        pick = max(cands, key=lambda i: ((targets[i] - counts[i]) / max(targets[i], 1), rng.random()))
        assign[gi] = pick
        counts[pick] += size
    return assign


def refine(assign: list[int], groups: list[list[dict]], gcls: list[Counter], adj: list[dict[int, int]],
           targets: list[int], cls_tot: Counter, w: float, leak_w: float,
           rng: random.Random, iters: int,
           protected: frozenset[int] = frozenset(),
           t0: float = 50.0, t1: float = 0.02) -> tuple[list[int], list[int], list[Counter], int]:
    """模拟退火：随机「交换两个段」或「搬动一个段」，增量维护数量/类别/泄漏代价，
    best 记录全局最优快照。"""
    assign = assign[:]
    n = len(groups)
    sizes = [len(g) for g in groups]
    counts, cls, leak = state_of(assign, groups, gcls, adj)
    cur = combined(counts, cls, leak, cls_tot, targets, w, leak_w)
    best = (cur, assign[:], counts[:], [c.copy() for c in cls], leak)
    if n < 2:
        return best[1], best[2], best[3], best[4]

    temp = t0
    cooling = (t1 / t0) ** (1.0 / max(iters, 1)) if t0 > 0 else 0.0
    for _ in range(iters):
        temp *= cooling
        if rng.random() < 0.15:  # 搬动：容忍数量暂时偏离，帮助跳出局部最优
            i = rng.randrange(n)
            if i in protected:
                continue
            a = assign[i]
            b = rng.randrange(3)
            if a == b:
                continue
            sa = sizes[i]
            old = (assign[i], counts[a], counts[b], cls[a], cls[b], leak)
            assign[i] = b
            counts[a] -= sa
            counts[b] += sa
            cls[a] = cls[a] - gcls[i]
            cls[b] = cls[b] + gcls[i]
            leak += (sum(c for k, c in adj[i].items() if assign[k] != b)
                     - sum(c for k, c in adj[i].items() if assign[k] != a))
            new = combined(counts, cls, leak, cls_tot, targets, w, leak_w)
            if new <= cur or (temp > 0 and rng.random() < math.exp(-(new - cur) / max(temp, 1e-9))):
                cur = new
                if new < best[0]:
                    best = (new, assign[:], counts[:], [c.copy() for c in cls], leak)
            else:
                assign[i], counts[a], counts[b], cls[a], cls[b], leak = old
        else:  # 交换两个不同 split 的段
            i, j = rng.randrange(n), rng.randrange(n)
            a, b = assign[i], assign[j]
            if i == j or a == b or i in protected or j in protected:
                continue
            old = (assign[i], assign[j], counts[a], counts[b], cls[a], cls[b], leak)
            assign[i], assign[j] = b, a
            counts[a] += sizes[j] - sizes[i]
            counts[b] += sizes[i] - sizes[j]
            cls[a] = cls[a] - gcls[i] + gcls[j]
            cls[b] = cls[b] - gcls[j] + gcls[i]
            leak += (sum(c for k, c in adj[i].items() if k != j and assign[k] != b)
                     - sum(c for k, c in adj[i].items() if k != j and assign[k] != a)
                     + sum(c for k, c in adj[j].items() if k != i and assign[k] != a)
                     - sum(c for k, c in adj[j].items() if k != i and assign[k] != b))
            new = combined(counts, cls, leak, cls_tot, targets, w, leak_w)
            if new <= cur or (temp > 0 and rng.random() < math.exp(-(new - cur) / max(temp, 1e-9))):
                cur = new
                if new < best[0]:
                    best = (new, assign[:], counts[:], [c.copy() for c in cls], leak)
            else:
                (assign[i], assign[j], counts[a], counts[b], cls[a], cls[b], leak) = old
    return best[1], best[2], best[3], best[4]


def allocate(groups: list[list[dict]], targets: list[int], args, cls_tot: Counter):
    """多轮贪婪随机重启 + 模拟退火微调，取综合代价最优的方案。"""
    rng = random.Random(args.seed)
    gcls, adj = build_context(groups, args.leak_window)
    w, lw = args.balance_weight, args.leak_weight
    protected: frozenset[int] = frozenset()
    if args.rare_to_train:
        rare = {c for c, n in cls_tot.items() if n < args.rare_threshold}
        protected = frozenset(gi for gi, c in enumerate(gcls) if any(c[k] for k in rare))
    cands: list[list[int]] = []
    for _ in range(max(args.restarts, 1)):
        assign = greedy_once(groups, targets, rng, protected)
        if assign is not None:
            cands.append(assign)
    if not cands and protected:
        print("[warn] 稀有类所在的段放不进 train，已放弃稀有类保护", file=sys.stderr)
        protected = frozenset()
        for _ in range(max(args.restarts, 1)):
            assign = greedy_once(groups, targets, rng)
            if assign is not None:
                cands.append(assign)
    if not cands:
        return None, float("inf")

    def key(a: list[int]) -> float:
        counts, cls, leak = state_of(a, groups, gcls, adj)
        return combined(counts, cls, leak, cls_tot, targets, w, lw)

    ranked = sorted(cands, key=key)
    rounds = max(args.refine_rounds, 1)
    # 取最优的一批，并穿插排名靠后的候选，增加退火起点多样性
    if len(ranked) > rounds:
        starts = (ranked[: max(1, rounds // 2)]
                  + ranked[max(1, rounds // 2):: max(1, len(ranked) // rounds)])
    else:
        starts = ranked
    starts = starts[:rounds]
    best_assign, best_score, best_state = None, float("inf"), None
    for idx, start in enumerate(starts):
        a = start if idx == 0 else start[:]
        a, counts, cls, leak = refine(a, groups, gcls, adj, targets, cls_tot, w, lw,
                                      rng, args.refine_iters, protected,
                                      args.anneal_t0, args.anneal_t1)
        sc = combined(counts, cls, leak, cls_tot, targets, w, lw)
        if sc < best_score:
            best_assign, best_score, best_state = a, sc, (counts, cls, leak, gcls, adj)
    return {"assign": best_assign, "counts": best_state[0], "cls": best_state[1],
            "leak": best_state[2]}, best_score


def do_backup(dataset: Path, names: tuple[str, str, str]) -> Path | None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = dataset.parent / f"{dataset.name}_backup_{ts}"
    dest.mkdir(parents=True, exist_ok=False)
    for n in names:
        if (dataset / n).is_dir():
            shutil.copytree(dataset / n, dest / n)
    for extra in ("data.yaml",):
        if (dataset / extra).is_file():
            shutil.copy2(dataset / extra, dest / extra)
    return dest


def leakage(assign: list[int], groups: list[list[dict]], window: int = 5) -> tuple[int, int]:
    """统计帧号相差 <= window 却落在不同 split 的相邻帧对（时间泄漏指标）。"""
    num: dict[tuple[str, int], int] = {}
    for gi, g in enumerate(groups):
        for r in g:
            if r["frame"] is not None:
                num[(r.get("stream", ""), r["frame"])] = assign[gi]
    ks = sorted(num)
    bad = pair = 0
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            if b[0] != a[0] or b[1] - a[1] > window:
                break
            pair += 1
            bad += num[a] != num[b]
    return bad, pair


def report(records: list[dict], groups: list[list[dict]], state: dict,
           targets: list[int], names: tuple[str, str, str], gap: int, window: int) -> None:
    counts, cls = state["counts"], state["cls"]
    empty = [0, 0, 0]
    for gi, g in enumerate(groups):
        for r in g:
            if sum(r["classes"].values()) == 0:
                empty[state["assign"][gi]] += 1
    print("\n===== 划分结果 =====")
    print(f"总图数 {len(records)}，连续段 {len(groups)} 个（帧号差 <= {gap} 归为同一段）")
    print(f"{'split':<8}{'图片':>6}{'目标':>6}{'段数':>6}{'无标注框':>10}   类别实例")
    for i, n in enumerate(names):
        gs = sum(1 for gi in range(len(groups)) if state["assign"][gi] == i)
        inst = ", ".join(f"cls{k}:{v}" for k, v in sorted(cls[i].items())) or "-"
        print(f"{n:<8}{counts[i]:>6}{targets[i]:>6}{gs:>6}{empty[i]:>10}   {inst}")
    tot = sum(counts)
    print("实际占比 " + " / ".join(f"{names[i]} {counts[i] / tot:.1%}" for i in range(3)))
    dev = sum(abs(counts[i] - targets[i]) for i in range(3))
    print(f"与目标的数量偏差：{dev} 张")
    for wd in sorted({3, window, 10}):
        bad, pair = leakage(state["assign"], groups, wd)
        print(f"时间泄漏：帧号相差 <= {wd:>2} 的相邻帧对中，跨 split 的有 {bad:>3}/{pair} 对"
              f"（纯随机划分约 {pair * 2 / 3:.0f} 对）")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset: Path = args.dataset.resolve()
    if not dataset.is_dir():
        sys.exit(f"[error] 数据集目录不存在：{dataset}")
    names = tuple(args.names)

    records = scan(dataset, names)
    for r in records:
        r["classes"] = label_classes(r["lbl"]) if r["has_label"] else Counter()
    missing = [r["img"].name for r in records if not r["has_label"]]
    if missing:
        print(f"[warn] {len(missing)} 张图没有对应 label（将一并划分）：{missing[:5]}", file=sys.stderr)
    dup = [n for n, c in Counter(r["img"].name for r in records).items() if c > 1]
    if dup:
        sys.exit(f"[error] 图片名在多个 split 中重复，无法合并重排：{dup[:5]}")

    gap = args.group_gap
    groups = group_records(records, gap)
    targets = round_targets(len(records), list(args.ratios))
    print(f"数据集：{dataset}")
    print(f"比例 {args.ratios} -> 目标数量 {dict(zip(names, targets))}（共 {len(records)} 张，自动取整）")

    cls_tot: Counter = Counter()
    for r in records:
        cls_tot.update(r["classes"])
    print("全量类别实例：" + (", ".join(f"cls{k}:{v}" for k, v in sorted(cls_tot.items())) or "-"))

    rare = {c for c, n in cls_tot.items() if n < args.rare_threshold}
    if args.rare_to_train and rare:
        print(f"稀有类保护：cls{sorted(rare)} 全量实例少于 {args.rare_threshold}，"
              f"含这些类的段强制留在 train（避免训练集完全没有该类）")

    best, best_score = allocate(groups, targets, args, cls_tot)
    if best is None:
        sys.exit("[error] 无法在「段不跨 split」约束下满足目标数量，请放宽比例或减小 --group-gap")
    report(records, groups, best, targets, names, gap, args.leak_window)

    dev = sum(abs(best["counts"][i] - targets[i]) for i in range(3))
    allowed = max(2, math.ceil(0.02 * len(records)))
    if dev > allowed and not args.force and not args.dry_run:
        print(f"[warn] 数量偏差 {dev} 张 > 允许值 {allowed}，段大小差异过大；"
              f"确认无误可加 --force 强制执行", file=sys.stderr)

    if args.dry_run:
        print("\n[dry-run] 未移动任何文件。")
        return 0

    if args.backup:
        bk = do_backup(dataset, names)
        print(f"\n已备份原划分 -> {bk}")

    # 移动文件（先收集再落位，源目录同名文件直接覆盖式写入）
    moved = 0
    for i, n in enumerate(names):
        (dataset / n / "images").mkdir(parents=True, exist_ok=True)
        (dataset / n / "labels").mkdir(parents=True, exist_ok=True)
    manifest = []
    for gi, g in enumerate(groups):
        for r in g:
            dst_split = names[best["assign"][gi]]
            manifest.append({
                "file": r["img"].name,
                "old_split": r["src_split"],
                "new_split": dst_split,
                "group_id": gi,
                "frame": "" if r["frame"] is None else r["frame"],
                "has_label": int(r["has_label"]),
                "n_boxes": sum(r["classes"].values()),
            })
            dst_img = dataset / dst_split / "images" / r["img"].name
            dst_lbl = dataset / dst_split / "labels" / r["lbl"].name
            if dst_img != r["img"]:
                shutil.move(str(r["img"]), str(dst_img))
                moved += 1
            if r["lbl"].is_file():
                if dst_lbl != r["lbl"]:
                    shutil.move(str(r["lbl"]), str(dst_lbl))
    print(f"已移动 {moved} 张图片（其余本就在目标 split）")

    # 清理 split 目录里可能残留的孤儿文件
    keep = {m["file"] for m in manifest}
    stale = []
    for n in names:
        for sub in ("images", "labels"):
            d = dataset / n / sub
            for f in sorted(d.iterdir()):
                stem_ok = f.stem in {Path(k).stem for k in keep}
                if not stem_ok:
                    stale.append(f)
                    f.unlink()
    if stale:
        print(f"清理 {len(stale)} 个残留文件：{[f.name for f in stale[:5]]}")

    man_path = dataset / "split_manifest.csv"
    with man_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(sorted(manifest, key=lambda m: (m["new_split"], m["file"])))
    print(f"已写出划分清单 -> {man_path}")

    # 落盘后复核
    bad = 0
    placed = Counter()
    for n in names:
        imgs = {f.stem for f in (dataset / n / "images").iterdir() if f.suffix.lower() in IMG_EXTS}
        lbls = {f.stem for f in (dataset / n / "labels").iterdir() if f.suffix == ".txt"}
        placed[n] = len(imgs)
        if imgs - lbls and len(imgs - lbls) > len([m for m in manifest if not m["has_label"] and m["new_split"] == n]):
            bad += 1
        for extra in lbls - imgs:
            bad += 1
            print(f"[warn] {n}/labels 有孤儿标注：{extra}.txt", file=sys.stderr)
    print("复核：" + ", ".join(f"{n}={placed[n]}" for n in names) +
          f"，合计 {sum(placed.values())}" + (f"，异常 {bad}" if bad else "，无异常"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
