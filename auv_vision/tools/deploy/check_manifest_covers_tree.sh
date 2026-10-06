#!/bin/bash
# check_manifest_covers_tree.sh — **本地有、清单没有**的代码文件自检（部署前必跑）。
#
# 为什么需要它（2026-10-06 连踩两次）：`deploy_to_board.sh` 只推 `board_parity.md5` 里
# **列出的**文件 ⇒ 新增模块若没进清单，板端就是 `ModuleNotFoundError`（板端自检 → collection error），
# 而 deploy 的输出只会说「新 0」——看着像没事。
#
# 用法（工程根）：bash tools/deploy/check_manifest_covers_tree.sh
set -u
cd "$(dirname "$0")/../.."
python3 - <<'PYEOF'
import io, os, sys
mf = "tools/deploy/board_parity.md5"
paths = set()
for ln in io.open(mf, encoding="utf-8"):
    s = ln.strip()
    if not s or s.startswith("#"):
        continue
    t = s.strip("[]").strip().split()
    if len(t) >= 2:
        paths.add(t[-1])
missing = []
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    rel = os.path.relpath(root, ".")
    top = rel.split(os.sep)[0]
    if top in ("bak", "log", "models", "doc", ".git"):
        continue
    # pytest 缓存 / 预设 / 部署脚本本身：不上板端，也不算"缺清单"
    if "pytest_cache" in rel or rel.startswith(("tools/presets", "tools/deploy")):
        continue
    for f in files:
        if not f.endswith((".py", ".sh", ".yaml", ".md")):
            continue
        p = os.path.normpath(os.path.join(rel, f)).lstrip("./")
        if p not in paths:
            missing.append(p)
if missing:
    print("✗ 有 %d 个文件**不在清单里** ⇒ deploy 不会推它们（板端会 ImportError / 缺文件）：" % len(missing))
    for x in sorted(missing):
        print("   %s" % x)
    print("  修法：按「<md5>  <相对路径>」**无方括号**追加到 tools/deploy/board_parity.md5")
    sys.exit(1)
print("✓ 清单覆盖完整（本地代码文件都在 board_parity.md5 里）")
PYEOF
