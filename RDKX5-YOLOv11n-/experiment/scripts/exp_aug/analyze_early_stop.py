#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_early_stop.py — 训练**提前结束**的归因（早停 / 崩溃 / 被打断）

用途：训练没跑满预设 epoch 时，先别往下走（导出/量化），先回答"为什么停"。
本脚本只读 `results.csv` 与训练日志，输出**基于曲线证据**的判断与改救策略。

用法：
    python experiment/scripts/exp_aug/analyze_early_stop.py runs/auv5/train/pose_fromcoco_844
    python experiment/scripts/exp_aug/analyze_early_stop.py <run_dir> --log <logfile> --epochs 300

判据（不是凭感觉，按下面顺序排除）：
  A. 日志里有 EarlyStopping 行 → 早停：看「best epoch」与「之后 val 是否真的没再涨」
  B. 训练 loss（box/cls/dfl/pose/kobj）在后 30% 的**线性斜率**：
       明显<0（仍在降） + val 平  → **过拟合 / 数据不足**（小数据集典型）
       全部≈0（都平）            → **优化饱和**（已收敛或 LR 太低）
       只有 pose/kobj 平在高位    → **关键点分支欠拟合**（从 coco 起步时 kpt 头被重初始化）
       有 NaN/Inf 或 loss 跳变     → 数值不稳（LR 过高 / 数据坏样本）
  C. val 指标在后 100 epoch 的**标准差**大 → 验证集太小/标签噪声 → 别只看 mAP，用面积比判据
  D. 相反：loss 仍在降、val 也在涨，只是慢 → 单纯"还没训够"，延长/关早停即可

⚠️ 关于"梯度消失"：YOLO11n 用 SiLU + 残差 C3k2 + C2PSA 注意力，出现**梯度消失**的概率很低；
   若 loss 仍在下降，就说明梯度没消失。真要实测，需要给某个卷积层注册 backward hook 打印
   梯度范数（本脚本给出命令模板，但不默认跑）。
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np


def slope(x, y):
    """最小二乘斜率（每 epoch 变化量）"""
    if len(x) < 3:
        return float("nan")
    return float(np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="ultralytics 训练目录（含 results.csv）")
    ap.add_argument("--log", default=None, help="训练日志（用于找 EarlyStopping 行）")
    ap.add_argument("--epochs", type=int, default=300, help="原本计划的 epoch 数")
    ap.add_argument("--tail-frac", type=float, default=0.3, help="用尾部多少比例判断趋势")
    ap.add_argument("--skip-last", type=int, default=10,
                    help="趋势窗口排除最后 N 个 epoch（默认 10 = close_mosaic 尾巴：最后 10 个 epoch 关掉 mosaic 会让 loss 整体跳升，混进去会把趋势判反）")
    a = ap.parse_args()

    run = Path(a.run_dir)
    csv_path = run / "results.csv"
    if not csv_path.exists():
        print(f"✗ 找不到 {csv_path}")
        return 2
    rows = list(csv.DictReader(open(csv_path)))
    cols = {c.strip(): c for c in rows[0].keys()}
    ep = np.array([int(float(r[cols["epoch"]])) for r in rows])

    def col(name):
        for k, v in cols.items():
            if name in k:
                return np.array([float(r[v] or "nan") for r in rows])
        return None

    print("=" * 78)
    print(f"训练提前结束归因：{run}")
    print("=" * 78)
    print(f"已完成 epoch：{ep.min()} → {ep.max()}（共 {len(ep)}），计划 {a.epochs}"
          f"{'  ← 未跑满' if ep.max() < a.epochs else '  ← 跑满'}")

    # ---- 日志里的早停/崩溃证据 ----
    log_ev = []
    if a.log and Path(a.log).exists():
        txt = Path(a.log).read_text(errors="ignore")
        txt = txt.replace("\r", "\n")
        for pat in ("EarlyStopping", "best model saved", "epochs completed", "Traceback",
                    "CUDA error", "out of memory", "KeyboardInterrupt"):
            for line in txt.split("\n"):
                if pat in line:
                    log_ev.append(line.strip()[:150])
        log_ev = log_ev[:6]
    if log_ev:
        print("\n[日志证据]")
        for l in log_ev:
            print("   " + re.sub(r"\x1b\[[0-9;]*m", "", l))

    # ---- loss 趋势 ----
    candidates = [k for k in cols if k.startswith("train/")]
    hi = len(ep) - max(0, a.skip_last)
    n_tail = max(5, int(len(ep[:hi]) * a.tail_frac))
    idx = slice(max(0, hi - n_tail), hi)
    print(f"\n[训练 loss 趋势：epoch {ep[idx][0]}–{ep[idx][-1]} 的线性斜率（每 epoch）"
          + (f"；已排除最后 {a.skip_last} 个 epoch（close_mosaic 尾巴）" if a.skip_last else "") + "]")
    verdicts = []
    for k in candidates:
        y = col(k.split("/")[1])
        if y is None or not np.isfinite(y[idx]).all():
            continue
        s = slope(ep[idx], y[idx])
        rel = s / (abs(np.mean(y[idx])) + 1e-9)          # 相对变化率
        state = "仍在下降" if rel < -0.0015 else ("≈平台" if abs(rel) <= 0.0015 else "在上升")
        print(f"   {k:<26} 末值 {y[-1]:8.4f}  斜率 {s:+.5f}（相对 {rel*100:+.2f}%/ep） → {state}")
        verdicts.append((k, rel))
    if a.skip_last and len(ep) > a.skip_last:
        print(f"\n   [close_mosaic 尾巴（最后 {a.skip_last} 个 epoch，关掉 mosaic）的 loss 跳变]")
        for k in candidates:
            y = col(k.split("/")[1])
            if y is None:
                continue
            print(f"   {k:<24} 前 {y[hi-3:hi].mean():8.4f} → 后 {y[hi:].mean():8.4f}"
                  f"（{y[hi:].mean() - y[hi-3:hi].mean():+.4f}）")
    nan_hit = [k for k in candidates if (y := col(k.split("/")[1])) is not None
               and not np.isfinite(y).all()]
    if nan_hit:
        print(f"   ⚠️ 出现 NaN/Inf 的 loss：{nan_hit}")

    # ---- val 指标趋势 ----
    val_m = col("mAP50-95")
    print("\n[val 指标]")
    if val_m is not None:
        best_i = int(np.nanargmax(val_m))
        last100 = val_m[max(0, best_i - 0):]
        print(f"   Pose mAP50-95 最好 {val_m[best_i]:.4f} @epoch {ep[best_i]}；"
              f"其后 {len(val_m)-1-best_i} 个 epoch 的最高值 {np.nanmax(val_m[best_i:]):.4f}")
        print(f"   尾部 {min(100,len(val_m))} 个 epoch 的 std = {np.nanstd(val_m[-100:]):.4f}"
              f"（大 = 抖动 → 验证集偏小/标签噪声）")
    lr = col("lr/pg0")
    if lr is not None:
        print(f"   lr：起 {lr[0]:.2e} → 末 {lr[-1]:.2e}")

    # ---- 规则化结论 ----
    print("\n" + "=" * 78)
    print("结论与改救策略")
    print("=" * 78)
    down = [k for k, rel in verdicts if rel < -0.0015]
    flat = [k for k, rel in verdicts if abs(rel) <= 0.0015]
    kpt = [k for k, rel in verdicts if ("pose" in k or "kobj" in k)]
    kpt_flat = [k for k, _ in verdicts if ("pose" in k or "kobj" in k) and
                abs(dict(verdicts)[k]) <= 0.0015]
    if nan_hit:
        print("· 数值不稳：loss 出现 NaN/Inf → 先查坏样本与 lr0（可先 --no-amp、降 lr0、开 clip）")
    elif val_m is not None and down and (np.nanmax(val_m[int(np.nanargmax(val_m)):]) <= val_m[int(np.nanargmax(val_m))] + 1e-4):
        print("· 训练 loss 仍在降、val 已停涨 ⇒ **过拟合 / 数据不足**（当前 train 675 张，"
              "远小于同类任务的常规量）")
        print("  改救：① 优先补标（worklist_remaining.csv 还有 2065 张，中/浊 1198）；"
              "② 加正则（weight_decay↑、label_smoothing>0、保留 mosaic）；"
              "③ 用小 val 做模型选择时改用面积比判据。")
    elif kpt_flat and len(kpt_flat) == len(kpt):
        print("· **关键点分支欠拟合**：pose/kobj loss 早已平台且未降到位（从 coco 起步时 "
              "kpt 头是重新初始化的：17 点 51 通道 → 4 点 12 通道）")
        print("  改救：① `freeze=10` 先冻主干 30–50 ep 只热身头部，再解冻继续；"
              "② 关早停 `patience=0` 训满；③ 适度提高 lr0 或延长 warmup_epochs。")
    elif flat and len(flat) >= max(1, len(verdicts) - 1):
        print("· **优化饱和**：所有 loss 都已平台 ⇒ 要么已收敛（那就看面积比/mAP 是否达标），"
              "要么 LR 太小/调度过早衰减。")
        print("  改救：延长训练或提高 lr0 / 用 cos_lr，并确认 close_mosaic 的末段增益已吃到。")
    else:
        print("· 趋势不单一：逐条看上面的斜率表；若 loss 与 val 都还在涨只是慢 ⇒ **单纯没训够**，"
              "可 `patience=0` 训满或直接延长 epochs。")
    print("\n· 关于「梯度消失」：本架构（SiLU + 残差 + C2PSA 注意力）出现概率低；"
          "若 loss 仍在下降即可判定梯度未消失。真要实测，可在训练脚本里挂 hook 打印梯度范数：")
    print("     python -c \"import torch; m=model.model; h=m.model[9].register_full_backward_hook("
          "lambda mod,gi,go: print(float(go[0].norm()))); m.train()\"   # 需自行接到训练循环")
    print("\n[后续流程] 按约定：**提前结束即暂停导出/量化**，先把上表与原因确认后再决定。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
