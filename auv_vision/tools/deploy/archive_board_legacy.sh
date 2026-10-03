#!/bin/bash
# tools/deploy/archive_board_legacy.sh — 把**板端旧布局**的文件先备份、再删除；**不上传任何东西**。
#
#   bash tools/deploy/archive_board_legacy.sh --dry-run   # 只列出板端存在、将被归档的旧路径
#   bash tools/deploy/archive_board_legacy.sh             # 备份 → 校验 → 删除 → 打包归档
#   KEEP_TAR=0 bash tools/deploy/archive_board_legacy.sh  # 不额外打整包（默认打）
#
# 为什么单独一个脚本，而不是直接跑 deploy：
#   `deploy_to_board.sh` 会**先上传本地文件**。而板端当前带着**本地仓库没有的新功能**
#   （下位机旋转执行协议 + 冲刺前航向确认锁存 `_hdg_ok`，见 doc/记录/），全量 deploy 会把它们冲掉。
#   归档旧布局不需要上传，所以这里只做"备份 + 删除"。
#
# 旧路径清单的**唯一来源**是 `deploy_to_board.sh` 里的 `DELETED=(...)`：本脚本解析它、不另抄一份
#   （抄两份必然漂移：deploy 加了、这边没加 = 永远清不干净）。
#
# 板端布局（<板端> = AUV_BOARD_DIR，默认 /home/sunrise/Desktop/AUV_New）：
#   <板端>/bak/legacy_layout_<本地时间戳>/   # 原样保留的旧文件（cp -p，可直接拷回去）
#   <板端>/bak/archive/legacy_layout_<时间戳>.tar.gz   # 上面那份的压缩包
#   <板端>/bak/board_state_<时间戳>.tar.gz             # 顺手存的"归档前板端代码全貌"（排除 bak/log/models）
#
# 原则：**先备份、再删除、后校验**；只碰 DELETED 清单里的路径与 <板端>/bak/，不动其它代码。
set -u
TOOLS_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPLOY="$TOOLS_DIR/deploy_to_board.sh"
BOARD_HOST="${AUV_BOARD_HOST:-sunrise@192.168.137.10}"
BOARD="${AUV_BOARD_DIR:-/home/sunrise/Desktop/AUV_New}"
HELP_DIR="${AUV_HELP_DIR:-/home/ansty/RDKX5}"
SSH="${AUV_SSH:-$HELP_DIR/.ssh_x5.sh}"
STREAM="${AUV_STREAM:-$HELP_DIR/.ssh_x5_stream.sh}"
# 注：$SSH 带 `-n`（不吃 stdin），所以**凡是往板端喂脚本/tar 的地方一律用 $STREAM**。
KEEP_TAR="${KEEP_TAR:-1}"

DRY=0
for a in "$@"; do [ "$a" = "--dry-run" ] && DRY=1; done

[ -f "$DEPLOY" ] || { echo "✗ 找不到 $DEPLOY（旧路径清单的来源）"; exit 1; }
[ -x "$SSH" ] || { echo "✗ 找不到 SSH 包装器 $SSH（见 deploy_to_board.sh 顶部说明）"; exit 1; }
[ -x "$STREAM" ] || { echo "✗ 找不到 STREAM 包装器 $STREAM"; exit 1; }

# ---- 旧路径清单：解析 deploy 脚本的 DELETED 数组（唯一来源）----
mapfile -t LEGACY < <(sed -n '/^DELETED=(/,/^)/p' "$DEPLOY" \
  | grep -vE '^[[:space:]]*(DELETED=\(|\)[[:space:]]*$|#)' \
  | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/[[:space:]]\+#.*$//' \
  | grep -v '^$')
[ "${#LEGACY[@]}" -gt 0 ] || { echo "✗ 没解析出 DELETED 清单"; exit 1; }
echo "[legacy] 旧路径清单 ${#LEGACY[@]} 条（来自 $(basename "$DEPLOY") 的 DELETED）"

# ---- 护栏：本地存在同名文件 = **活文件**，绝不在板端删它 ----
#   （2026-10-02 踩过：DELETED 里误列 `gate/gate_task.py`，照单执行把板端生效的编排文件删了，
#     靠 bak/legacy_layout_*/ 才恢复。DELETED 是人写的清单，会漂；这里的判断来自文件系统。）
LOCAL_ROOT="$(cd "$TOOLS_DIR/../.." && pwd)"
GUARD=(); SAFE=()
for f in "${LEGACY[@]}"; do
  if [ -e "$LOCAL_ROOT/$f" ] && [ "${FORCE:-0}" != "1" ]; then GUARD+=("$f"); else SAFE+=("$f"); fi
done
if [ "${#GUARD[@]}" -gt 0 ]; then
  echo "[legacy] ⛔ 跳过 ${#GUARD[@]} 条**本地仍是活文件**的清单条目（删了板端就少模块）："
  printf '        %s\n' "${GUARD[@]}"
  echo "        要强行归档请显式 FORCE=1（一般说明 DELETED 清单该修了）"
fi
LEGACY=("${SAFE[@]}")
[ "${#LEGACY[@]}" -gt 0 ] || { echo "[legacy] 没有可归档的路径（全被护栏拦住）"; exit 0; }

STAMP="$(date +%m%d_%H%M%S)"
TODAY="$(date +%Y-%m-%d)"
DEST="bak/legacy_layout_$STAMP"

t0=$(date +%s)
step() { echo "[legacy +$(( $(date +%s) - t0 ))s] $*"; }

# ---- 1) 板端可达性 + 哪些旧路径真的存在 ----
timeout 12 "$SSH" "echo BOARD-UP" </dev/null 2>/dev/null | grep -q BOARD-UP \
  || { echo "✗ 板子不可达（$BOARD_HOST）"; exit 1; }
step "板端可达"

{ echo "cd '$BOARD' || exit 1"
  for f in "${LEGACY[@]}"; do
    echo "[ -e '$f' ] && echo 'EXIST $f'"
  done
} > /tmp/_legacy_probe.sh
EXIST=$(timeout 60 "$STREAM" 'bash -s' < /tmp/_legacy_probe.sh 2>/dev/null | sed -n 's/^EXIST //p')
N=$(printf '%s\n' "$EXIST" | grep -c . || true)
echo "[legacy] 板端实际存在 ${N} 个旧路径："
printf '%s\n' "$EXIST" | sed 's/^/        /'

if [ "$DRY" = "1" ]; then
  echo "[legacy] --dry-run：不动板端。去掉 --dry-run 即执行「备份 → 校验 → 删除 → 打包」。"
  exit 0
fi
[ "$N" -gt 0 ] || { echo "[legacy] 板端没有旧路径了，无需归档"; exit 0; }

# ---- 2) 归档前先存一份"板端代码全貌"（保险：板端有本地没有的新功能）----
if [ "$KEEP_TAR" = "1" ] && [ -x "$STREAM" ]; then
  timeout 120 "$STREAM" "cd '$BOARD' && mkdir -p bak && tar czf bak/board_state_$STAMP.tar.gz \
      --exclude=./bak --exclude=./log --exclude=./models --exclude=__pycache__ --exclude=.pytest_cache . \
      && echo SNAP-OK && ls -l bak/board_state_$STAMP.tar.gz" </dev/null 2>/dev/null | tail -2 | sed 's/^/        /'
  step "已存 bak/board_state_$STAMP.tar.gz（归档前的板端代码全貌）"
fi

# ---- 3) 备份 + 删除（一次 SSH；cp -p 保权限/时间）----
{ echo "cd '$BOARD' || exit 1"
  echo "mkdir -p '$DEST'"
  for f in "${LEGACY[@]}"; do
    echo "[ -e '$f' ] && { mkdir -p '$DEST'/\"\$(dirname '$f')\"; cp -p '$f' '$DEST/$f' && rm -f '$f' && echo \"[archived] $f\"; }"
  done
} > /tmp/_legacy_do.sh
timeout 300 "$STREAM" 'bash -s' < /tmp/_legacy_do.sh 2>/dev/null | sed -n 's/^\[archived\] /        /p' > /tmp/_legacy_done.txt
DONE=$(grep -c . /tmp/_legacy_done.txt || true)
step "已备份并删除 $DONE 个旧路径"

# ---- 4) 校验：备份目录里的文件数 == 删掉的数，且原位置确实没了 ----
{ echo "cd '$BOARD' || exit 1"
  echo "echo BACKUP_N=\$(find '$DEST' -type f | wc -l)"
  for f in "${LEGACY[@]}"; do echo "[ -e '$f' ] && echo \"STILL $f\""; done
  echo "echo ---"
  if [ "$KEEP_TAR" = "1" ]; then
    echo "mkdir -p bak/archive && tar czf bak/archive/legacy_layout_$STAMP.tar.gz '$DEST' && echo TAR-OK"
  fi
  echo "du -sh '$DEST' 2>/dev/null"
} > /tmp/_legacy_verify.sh
timeout 300 "$STREAM" 'bash -s' < /tmp/_legacy_verify.sh 2>/dev/null | sed 's/^/        /'
step "校验完成（BACKUP_N 应等于 $DONE；没有任何 STILL 行 = 旧路径已清空）"
echo "[legacy] 归档位置：$BOARD/$DEST   （压缩包 bak/archive/legacy_layout_$STAMP.tar.gz）"
echo "[legacy] 回滚方法：cd $BOARD && cp -rp $DEST/. ."
echo "[legacy] 板端 bak/ 进一步瘦身：bash tools/deploy/tidy_board_bak.sh"
