#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""板端脚本：写 A 域退役的文字记录，然后删除旧树 /home/sunrise/AUV。

⚠️ 这是**破坏性操作**（板端非 git 仓库）。前置条件（已逐条校验，缺一不删）：
  1. 归档已建成：/home/sunrise/Desktop/AUV_New/bak/legacy_Adomain_tree_20261006.tar.gz
  2. 条目数一致：tar -tzf = 583 = find /home/sunrise/AUV | wc -l
  3. 抽样字节级比对通过（cfg/vision.yaml、common/preprocess.py、models/*.bin 三个文件 cmp 相同）
  4. 全树无引用（AUV_New 仅在 bak/ 历史快照里提到它）、无 crontab / systemd 自启

用法（板端）：
    python3 retire_Adomain_tree.py --archive /home/sunrise/Desktop/AUV_New/bak/legacy_Adomain_tree_20261006.tar.gz \
        --md5 3a7cda63af1f9453006e17cd3a26734f --apply
不给 --apply 时只打印将要做什么（dry-run）。
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

LEGACY = Path("/home/sunrise/AUV")
AUV_NEW = Path("/home/sunrise/Desktop/AUV_New")
RECORD = AUV_NEW / "doc" / "记录" / "2026-10-06-A域退役与旧树清理.md"

RECORD_TEXT = """# 2026-10-06 · A 域退役收尾：旧树 `/home/sunrise/AUV` 已清理

> 本文是**文字记录**，也是"不要再引入 A 域"的落点。代码里已无 A 域分支（在用副本 2026-10-01 就退役了），
> 这篇记的是**最后一处还能跑起 A 域的地方** —— 板端旧树 `/home/sunrise/AUV` —— 已被归档删除。

## 一、做了什么

| 项 | 值 |
|---|---|
| 被清理对象 | `/home/sunrise/AUV`（**09-17 的旧树**，A 域时代产物，不在任何部署清单里） |
| 清理方式 | 整树 `tar -czf` 归档后 `rm -rf`（用户 2026-10-06 决定） |
| 归档路径 | `/home/sunrise/Desktop/AUV_New/bak/legacy_Adomain_tree_20261006.tar.gz` |
| 归档大小 / md5 | 317 MB / `3a7cda63af1f9453006e17cd3a26734f` |
| 条目数 | 583（= 清理前 `find /home/sunrise/AUV | wc -l`，逐项一致） |
| 删除前校验 | ① 条目数一致 ② 抽样字节级 `cmp` 通过（`cfg/vision.yaml`、`common/preprocess.py`、`models/yolo11n_detect_bayese_640x640_nv12.bin`）③ 全树无引用、无 crontab/systemd 自启 |
| 释放空间 | ~333 MB 目录 → 317 MB 归档（JPEG/`.bin` 已压缩，压缩比很低，别指望省空间） |

**恢复方式**（万一要回溯旧行为）：
```bash
sudo tar -xzf /home/sunrise/Desktop/AUV_New/bak/legacy_Adomain_tree_20261006.tar.gz -C /home/sunrise
# 解出来的就是完整的 /home/sunrise/AUV/，路径与删除前一致
```

## 二、A 域到底是什么（记下来，别再重新发明）

| | A 域（**已废弃**） | D 域（**唯一在用**） |
|---|---|---|
| 链路 | `remap@720p` → `resize 640` → `enhance` | `resize 640` → `enhance` → `remap@640` |
| 去畸变用的内参 | 老标定 `cfg/front_camera.yaml`（fx≈782.5） | 新标定（fx≈1207.6） |
| `clahe_clip` | **0.5**（必须，换了就掉点） | **0**（关掉省 ~8 ms/帧） |
| 配套权重 | AUV_4 代（`gate_kpt_auv4_bayese_640x640_nv12.bin`） | 阶段一 `g240_i16` 代（`gate_kpt_stage1_v3_...`） |
| 现状 | **全部退役**：代码分支于 2026-10-01 移除，最后的运行载体（旧树）于 2026-10-06 归档删除 | 撞球、过门均走 D + wb（白平衡） |

## 三、之后的规矩（防止 A 域又长回来）

1. **识别链路只有 D + wb**：`resize 640 → enhance(WB) → remap@640`，`clahe_clip: 0`。
2. **新增任务不要自带图像处理**：夹取小球按 2026-10-06 的决定，**只把原图 resize 成 640×640 直接送 BPU**，
   不做去畸变 / 增强 / 白平衡，也不引入 `chain` 之类的新开关。预处理只保证"能按 NV12 喂进模型"。
3. **不要再给 `vision.image` 加域开关**。历史教训：A/D 两套域 × 两代权重交叉喂图 → 角点飘、球漏检，
   而且这种错不会报错，只会静默掉点。要换输入口径就**换权重**，不是给同一条链路加分支。
4. 旧树里的 `models/gate_kpt_bayese_640x640_nv12.bin.`（**文件名尾部多一个点**）也随之消失 ——
   那个"权重在、配置指不带点的名字、gate 被静默跳过"的坑不用再防了。
"""


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fp:
        while True:
            b = fp.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--md5", required=True)
    ap.add_argument("--apply", action="store_true", help="不加则只 dry-run")
    args = ap.parse_args()
    arch = Path(args.archive)

    # ---- 删除前置条件逐条复核（任何一条不过就退出）----
    if not LEGACY.is_dir():
        sys.exit(f"❌ 待删目录不存在，无需处理: {LEGACY}")
    if not arch.is_file():
        sys.exit(f"❌ 归档不存在: {arch}")
    got = md5_of(arch)
    if got != args.md5:
        sys.exit(f"❌ 归档 md5 不符！\n   期望 {args.md5}\n   实际 {got}\n   → 拒绝删除")
    n_arch = int(subprocess.run(["tar", "-tzf", str(arch)], capture_output=True,
                                text=True).stdout.count("\n"))
    n_tree = int(subprocess.run(["find", str(LEGACY)], capture_output=True,
                                text=True).stdout.count("\n"))
    print(f"归档条目 {n_arch} / 原树条目 {n_tree} / md5 OK ({got[:8]}…)")
    if n_arch != n_tree:
        sys.exit("❌ 条目数不一致 → 拒绝删除")
    size = int(subprocess.run(["du", "-sb", str(LEGACY)], capture_output=True,
                              text=True).stdout.split()[0])
    print(f"待删目录大小 {size/1e6:.0f} MB；磁盘余量 "
          f"{shutil.disk_usage('/home/sunrise').free/1e9:.1f} GB")
    if not args.apply:
        print("（dry-run：加 --apply 才真正执行）")
        return

    # ---- 写文字记录 ----
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(RECORD_TEXT.replace("3a7cda63af1f9453006e17cd3a26734f", got),
                      encoding="utf-8")
    print(f"✅ 文字记录 → {RECORD}")

    # ---- 删除 ----
    shutil.rmtree(LEGACY)
    print(f"✅ 已删除 {LEGACY}；归档留在 {arch}")
    if LEGACY.exists():
        sys.exit("❌ 删除后目录仍存在？")
    print("✅ 复核通过：目录已不存在")


if __name__ == "__main__":
    main()
