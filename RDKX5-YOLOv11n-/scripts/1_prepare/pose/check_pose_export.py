#!/usr/bin/env python3
"""把 YOLO-pose 标签里 v=1 的角点改成 v=0。

约定：v=2 可见，v=0 不可见，v=1 不用。

用法：
    python scripts/1_prepare/pose/check_pose_export.py <导出目录> --out <新目录>
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SPLITS = ("train", "valid", "test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="Roboflow 导出的 YOLO-pose 目录")
    ap.add_argument("--out", required=True, help="输出目录（不存在才写）")
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    if out.exists():
        raise SystemExit(f"输出目录已存在：{out}")

    fixed = 0
    for sp in SPLITS:
        lab = src / sp / "labels"
        if lab.is_dir():
            for f in sorted(lab.glob("*.txt")):
                lines = []
                for line in f.read_text(encoding="utf-8").splitlines():
                    t = line.split()
                    if len(t) >= 5 and (len(t) - 5) % 3 == 0:
                        for i in range((len(t) - 5) // 3):
                            if int(float(t[7 + 3 * i])) == 1:
                                t[7 + 3 * i] = "0"
                                fixed += 1
                    lines.append(" ".join(t))
                dst = out / f.relative_to(src)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
        imgs = src / sp / "images"
        if imgs.is_dir():
            shutil.copytree(imgs, out / sp / "images", dirs_exist_ok=True)

    if (src / "data.yaml").exists():
        shutil.copy2(src / "data.yaml", out / "data.yaml")

    print(f"v=1 → v=0：{fixed} 个角点 → {out}")


if __name__ == "__main__":
    main()
