#!/usr/bin/env python3
"""内容去重天花板测算（实验支持脚本，不进常规流水线）。

问题：`--min-gap 25`（帧号间隔）在不同素材上语义不一致 —— 25fps 全率池的 25 帧 = 1 s，
而 25 步抽样后的 `*_frames` 池里相邻编号已经差 1 s。且帧号只在**同一段连续录像内**可比。

这里改用**内容去重的 32x32 灰度签名**：按时间顺序贪心，只有当与"上一张保留帧"的
平均绝对差 >= 阈值时才保留。它自带速度自适应（动得快→单位时间保住更多张），
也直接对应"尽量不相邻/不重复"的诉求。

用法：
    /home/ansty/anaconda3/envs/yolov8/bin/python experiment/scripts/dedup_ceiling.py
输出：每个来源在各阈值下能保住的张数（天花板），用于决定目标配额是否现实。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "1_prepare"))

from make_mixed_gate_set import SOURCES, WATER  # noqa: E402

THRESHOLDS = (1.5, 2.0, 3.0, 4.0, 6.0)


def sig(path: Path) -> np.ndarray | None:
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    return cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)


def load_sigs(paths: list[Path]) -> np.ndarray:
    """一次读盘，返回 (N,32,32) 签名栈（读一次给所有阈值复用）。"""
    out = []
    for p in paths:
        s = sig(p)
        if s is not None:
            out.append(s)
    return np.stack(out) if out else np.zeros((0, 32, 32), np.float32)


def ceiling(sigs: np.ndarray, thr: float) -> int:
    """按时间顺序贪心：与上一张保留帧的 mad >= thr 才保留。"""
    last, n = None, 0
    for s in sigs:
        if last is None or float(np.abs(s - last).mean()) >= thr:
            last, n = s, n + 1
    return n


def main() -> None:
    print(f"{'来源':<8}{'水质':<5}{'连续段':>4}{'池':>7}" + "".join(f"{'≥' + str(t):>8}" for t in THRESHOLDS))
    grand = {t: 0 for t in THRESHOLDS}
    for dive, specs in SOURCES.items():
        for pat, quota, _thr in specs:
            dirs = sorted(p for p in PROJECT_ROOT.glob(pat) if p.is_dir())
            files = sorted(p for p in PROJECT_ROOT.glob(pat) if p.is_file())
            groups = [(d.name, sorted(d.glob("*.jpg"))) for d in dirs] if dirs else [("(flat)", files)]
            pool = sum(len(g[1]) for g in groups)
            # 逐段算天花板再相加（段间互不相关，只有段内相邻才算泄漏风险）
            sigs = [load_sigs(g[1]) for g in groups]
            cnt = {t: sum(ceiling(s, t) for s in sigs) for t in THRESHOLDS}
            for t in THRESHOLDS:
                grand[t] += cnt[t]
            print(f"{dive:<8}{WATER[dive]:<5}{len(groups):>4}{pool:>7}"
                  + "".join(f"{cnt[t]:>8}" for t in THRESHOLDS), flush=True)
    print("-" * (24 + 8 * len(THRESHOLDS)))
    print(f"{'合计':<17}{'':>5}" + "".join(f"{grand[t]:>8}" for t in THRESHOLDS))
    print("\n注：阈值单位是 32x32 灰度平均绝对差（0-255）。≥2 大致相当于画面有可见变化，"
          "≥4 相当于明显位移。要凑 3000 张需看合计行是否够。")


if __name__ == "__main__":
    # 这个脚本没有命令行参数：直接跑就是全量测算（几分钟）。
    # 加个守卫，免得 `--help` 变成"静默跑全量"。
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__)
        raise SystemExit(0)
    main()
