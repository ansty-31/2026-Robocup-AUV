# -*- coding: utf-8 -*-
"""tools/check/check_paths.py — 路径依赖自检（**换拷贝/搬目录后第一件事**）

为什么需要它（2026-09-22 板端踩过的两类坑）
------------------------------------------
1. **cfg 里写绝对路径** ⇒ 把配置钉死在某一份拷贝上。板端同时存在
   `/home/sunrise/AUV`（旧副本）与 `/home/sunrise/Desktop/AUV_New`（在用的工作副本），
   标定/权重指错一份就静默用了另一个文件；那份被删/被搬 ⇒ **相机初始化直接崩**。
   规则：cfg 里**一律写仓库内相对路径**（`cfg/front_camera.yaml`、`models/x.bin`），
   运行时由 `base.settings.resolve_path()` 按工程根解析。
2. **相机标定 yaml 的第一个字节**：板端 cv2 4.11 靠文件开头认格式 ⇒ 前面有注释行、
   **甚至空行**都会 `Input file is invalid`（SystemError），而 `calibration_maps`
   **不兜底** ⇒ 崩。注释只能放在 `---` **之后**。

检查项
------
  A 工程根自证：`main.py` / `cfg/` / `models/` 都在同一个根下
  B `cfg/*.yaml` 里**没有绝对路径**（扫文本，不看注释以外的东西）
  C 所有"路径类键"能解析且**文件存在**（用 `base.settings` 的实际解析结果）
  D 相机标定 yaml：第一个字节是 `%YAML:1.0`，且 cv2 能读、fx 合理
  E 提示同名拷贝（`/home/sunrise/AUV` 之类）存在时的混用风险

用法（板端/本地都行）
--------------------
    python3 tools/check/check_paths.py
    python3 tools/check/check_paths.py --quiet      # 只打印失败项
退出码：0=全部通过 ｜ 1=有失败项（可直接进 CI/部署前检查）
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

#: 路径类键（与 base.settings._PATH_KEYS 同一份语义；这里列出来只为打印）
PATH_KEYS = (
    "vision.camera.front.calibration",
    "vision.camera.ball.calibration",
    "vision.model.path",
    "vision.model.task_models.gate.path",
)

_RE_ABS = re.compile(r"(/home/|/Users/|[A-Za-z]:\\\\)")


def scan_cfg_abs_paths(cfg_dir):
    """返回 [(文件, 行号, 行内容)] —— cfg 里出现的绝对路径（注释行也算，注释里也别写）。"""
    out = []
    for p in sorted(glob.glob(os.path.join(cfg_dir, "*.yaml"))):
        with open(p, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh, 1):
                if _RE_ABS.search(line):
                    out.append((os.path.basename(p), i, line.rstrip()))
    return out


def check_yaml_head(path):
    """(ok, 说明)：相机标定 yaml 的第一个字节必须是 %YAML:1.0。"""
    try:
        raw = open(path, "rb").read(16)
    except OSError as e:
        return False, "读不到：%s" % e
    if raw.startswith(b"%YAML:1.0"):
        return True, "首字节 OK"
    if raw[:1] in (b"#", b"\n", b"\r", b" ", b"\t"):
        return False, ("首字节是注释/空白（board cv2 4.11 会判 Input file is invalid）"
                       "—— 把注释挪到 '---' 之后")
    return False, "开头不是 %%YAML:1.0，而是 %r" % raw[:16]


def main(argv=None):
    ap = argparse.ArgumentParser(description="路径依赖自检（cfg 相对路径 / 标定文件头）")
    ap.add_argument("--quiet", action="store_true", help="只打印失败项")
    a = ap.parse_args(argv)

    fails, warns = [], []

    def ok(msg):
        if not a.quiet:
            print("  ✓ %s" % msg)

    def bad(msg):
        fails.append(msg)
        print("  ✗ %s" % msg)

    def warn(msg):
        warns.append(msg)
        print("  ! %s" % msg)

    print("路径依赖自检（工程根 = %s）" % _ROOT)

    # ---- A 工程根自证
    print("\nA 工程根")
    for rel in ("main.py", "cfg/vision.yaml", "cfg/comm.yaml", "models", "gate"):
        full = os.path.join(_ROOT, rel)
        (ok if os.path.exists(full) else bad)("%s 存在（%s）" % (rel, full))

    # ---- B cfg 里不许有绝对路径
    print("\nB cfg 里的绝对路径（应为 0）")
    hits = scan_cfg_abs_paths(os.path.join(_ROOT, "cfg"))
    if hits:
        for f, i, line in hits:
            bad("%s:%d  %s" % (f, i, line.strip()[:100]))
        print("    → 改成仓库内相对路径（`cfg/xxx.yaml`、`models/xxx.bin`），"
              "运行时由 base.settings.resolve_path() 解析")
    else:
        ok("cfg/*.yaml 里没有绝对路径")

    # ---- C 路径类键解析 + 存在
    print("\nC 路径类键（解析后必须存在）")
    try:
        import base.settings as S
    except Exception as e:
        bad("import base.settings 失败：%s" % e)
        S = None
    if S is not None:
        if getattr(S, "resolve_path", None) is None:
            bad("base.settings 没有 resolve_path()（板端 settings.py 没打上补丁？）")
        for k in PATH_KEYS:
            v = S.get(k)
            if not v:
                ok("%s = %s（空值，跳过）" % (k, v))
                continue
            if not os.path.isabs(str(v)):
                bad("%s 解析后仍是相对路径：%s" % (k, v))
            elif not os.path.exists(str(v)):
                bad("%s 指向的文件不存在：%s" % (k, v))
            else:
                ok("%s → %s" % (k, os.path.basename(str(v))))

    # ---- D 相机标定 yaml 的文件头 + 可读性
    print("\nD 相机标定 yaml（文件头 + cv2 可读）")
    for p in sorted(glob.glob(os.path.join(_ROOT, "cfg", "front_camera*.yaml"))):
        good, why = check_yaml_head(p)
        (ok if good else bad)("%s：%s" % (os.path.basename(p), why))
        if not good:
            continue
        try:
            import cv2
            fs = cv2.FileStorage(p, cv2.FILE_STORAGE_READ)
            K = fs.getNode("camera_matrix").mat()
            fs.release()
            if K is None:
                bad("%s：cv2 读得到文件但取不到 camera_matrix" % os.path.basename(p))
            else:
                ok("%s：cv2 读到 fx=%.1f fy=%.1f" % (os.path.basename(p), K[0, 0], K[1, 1]))
        except Exception as e:
            bad("%s：cv2 读取异常 %s: %s" % (os.path.basename(p), type(e).__name__, e))

    # ---- E 同名拷贝提示
    print("\nE 同项目其它拷贝（混用风险）")
    others = [d for d in ("/home/sunrise/AUV",) if os.path.isdir(d)]
    if others:
        for d in others:
            warn("还存在另一份拷贝 %s —— 部署/核对时确认你在改哪一份"
                 "（默认已指向 /home/sunrise/Desktop/AUV_New）" % d)
    else:
        ok("没发现可疑的其它拷贝")

    print("\n结果：%d 项失败，%d 项提醒" % (len(fails), len(warns)))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
