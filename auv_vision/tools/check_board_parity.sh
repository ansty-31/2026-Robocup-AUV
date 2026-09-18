#!/bin/bash
# check_board_parity.sh — 检查"本地代码 == 最后移植到板端的代码"
#
# 用法（在 tools/ 下或任意位置都行；工程根自动定位到上一级）：
#   bash tools/check_board_parity.sh              # 本地 vs tools/board_parity.md5（无需板子，毫秒级）
#   AUV_SSH=/home/ansty/RDKX5/.ssh_x5.sh \
#     bash check_board_parity.sh --board          # 再与板端比对（**1 次 SSH 批量取 md5**，秒级）
#   ... --board --only-verified                   # 只查有板端实测证据的那批
#   ... --board --slow                            # 退回"一文件一次 SSH"的老路径（SSH 不稳时用）
#   ... --board --write                           # 顺带把 tools/board_parity.md5 刷成当前实测状态
#
# 说明：manifest 记录的是"最后一次与板端逐文件 md5 比对一致"时的字节；
#   ① 本地全绿 = 本地没被改过；
#   ② --board 也全绿 = 板端确实装的就是这份代码。
#   只读操作：不写板端、不改本地。
set -u
# 本脚本在 tools/ 下，工程根 = 上一级；清单随工具一起放在 tools/
TOOLS_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$TOOLS_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

MANIFEST="$TOOLS_DIR/board_parity.md5"
BOARD_DIR="${AUV_BOARD_DIR:-/home/sunrise/AUV}"
AUV_SSH="${AUV_SSH:-ssh}"
ONLY_VERIFIED=0
SLOW=0
BOARD=0
WRITE=0
for a in "$@"; do
  [ "$a" = "--only-verified" ] && ONLY_VERIFIED=1
  [ "$a" = "--slow" ] && SLOW=1
  [ "$a" = "--board" ] && BOARD=1
  [ "$a" = "--write" ] && WRITE=1
done
[ "$WRITE" = "1" ] && BOARD=1

[ -f "$MANIFEST" ] || { echo "缺少 $MANIFEST"; exit 2; }

# 解析一行 → "md5<TAB>路径<TAB>是否板端实测(1/0)"；注释/空行不输出
parse() {
  awk '
    /^[[:space:]]*#/ { next }
    NF < 2 { next }
    {
      v = 0; p = 1;
      if ($1 ~ /^\[/) { v = 1; p = 2 }
      md5 = $p; path = $(p+1);
      printf "%s\t%s\t%d\n", md5, path, v
    }' "$MANIFEST"
}

# 1) 本地 vs manifest
echo "==== [1/2] 本地 vs $MANIFEST ===="
local_bad=0; n=0
while IFS=$'\t' read -r md5 f verified; do
  [ "$ONLY_VERIFIED" = "1" ] && [ "$verified" != "1" ] && continue
  n=$((n+1))
  if [ ! -f "$f" ]; then
    echo "  [缺失] $f"; local_bad=$((local_bad+1)); continue
  fi
  now=$(md5sum "$f" | cut -d' ' -f1)
  if [ "$now" != "$md5" ]; then
    echo "  [改动] $f"
    echo "         清单=$md5"
    echo "         现在=$now"
    local_bad=$((local_bad+1))
  fi
done < <(parse)
echo "  检查本地文件 $n 个，不一致 $local_bad 个"
if [ "$local_bad" = "0" ]; then
  echo "  ✓ 本地与\"最后同步到板端的状态\"逐字节一致"
else
  echo "  ✗ 本地有改动/缺失（板端未必同步过）"
fi

# 2) 可选：与板端实机比对
if [ "$BOARD" = "1" ]; then
  echo "==== [2/2] 板端 $BOARD_DIR vs 本地 ===="
  if ! timeout 12 $AUV_SSH "echo BOARD-UP" </dev/null 2>/dev/null | grep -q BOARD-UP; then
    echo "  ✗ 板子不可达（确认网线/供电，或设置 AUV_SSH 指向你的 ssh 包装脚本）"
    exit 1
  fi
  T0=$(date +%s)
  TMPL=$(mktemp); TMPR=$(mktemp); TMPC=$(mktemp)
  trap 'rm -f "$TMPL" "$TMPR" "$TMPC"' EXIT
  parse > "$TMPL"

  remote_bad=0; rn=0
  if [ "$SLOW" = "1" ]; then
    # 老路径：每个文件一次 SSH（几十秒~几分钟）
    while IFS=$'\t' read -r md5 f verified; do
      [ "$ONLY_VERIFIED" = "1" ] && [ "$verified" != "1" ] && continue
      rn=$((rn+1))
      r=$(timeout 20 $AUV_SSH "md5sum $BOARD_DIR/$f 2>/dev/null | cut -d' ' -f1" 2>/dev/null </dev/null | tr -d '\r\n')
      if [ -z "$r" ]; then
        echo "  [板端缺失] $f"; remote_bad=$((remote_bad+1)); continue
      fi
      [ "$r" = "$md5" ] || { echo "  [板端不同] $f  本地=$md5 板端=$r"; remote_bad=$((remote_bad+1)); }
    done < "$TMPL"
  else
    # 快路径：1 次 SSH 批量取 md5（清单里的文件名无空格，逐个加引号拼接）
    {
      echo "cd $BOARD_DIR 2>/dev/null || exit 1"
      printf 'md5sum'
      while IFS=$'\t' read -r _md5 f v; do
        [ "$ONLY_VERIFIED" = "1" ] && [ "$v" != "1" ] && continue
        printf " '%s'" "$f"
      done < "$TMPL"
      echo
    } > "$TMPC"
    timeout 60 $AUV_SSH "$(cat "$TMPC")" </dev/null 2>/dev/null \
      | grep -E '^[0-9a-f]{32} ' > "$TMPR"
    declare -A RM=()
    while read -r m p; do RM["$p"]="$m"; done < "$TMPR"
    while IFS=$'\t' read -r md5 f verified; do
      [ "$ONLY_VERIFIED" = "1" ] && [ "$verified" != "1" ] && continue
      rn=$((rn+1))
      r="${RM[$f]:-}"
      if [ -z "$r" ]; then
        echo "  [板端缺失] $f"; remote_bad=$((remote_bad+1)); continue
      fi
      [ "$r" = "$md5" ] || { echo "  [板端不同] $f  本地=$md5 板端=$r"; remote_bad=$((remote_bad+1)); }
    done < "$TMPL"
    echo "  （批量模式：1 次 SSH，用时 $(( $(date +%s) - T0 ))s）"
  fi
  echo "  比对板端文件 $rn 个，不一致 $remote_bad 个"
  [ "$remote_bad" = "0" ] && echo "  ✓ 板端就是这份代码" || echo "  ✗ 板端与本地有差异"

  # 2b) --write：把清单刷成"当前实测"（匹配 → [ md5 path ]；不匹配/缺失 → 无方括号）
  if [ "$WRITE" = "1" ]; then
    LIST=$(mktemp)
    # ① 现有清单里仍存在于本地的条目（这样非源码产物也能继续被跟踪）
    while IFS=$'\t' read -r _m f _v; do
      [ -f "$f" ] && echo "$f"
    done < "$TMPL" > "$LIST"
    # ② 本地源码树里新增的文件
    #    -not -path './.*' 把**点目录**整个排除（.pytest_cache/、.git/、.vscode/…）：
    #    它们里面的 README.md 文件名不以点开头，只靠 `-not -name '.*'` 拦不住，
    #    一旦进清单就会被 deploy 当源码传上板。
    find . -type f \( -name '*.py' -o -name '*.sh' -o -name '*.md' -o -name '*.yaml' \
         -o -name '*.txt' \) \
      -not -path './.*' \
      -not -path './bak/*' -not -path './log/*' -not -path './rec/*' \
      -not -path './models/*' -not -path '*/__pycache__/*' -not -name '*.pyc' \
      -not -name 'check_board_parity.sh' -not -name 'board_parity.md5' \
      -not -name 'deploy_to_board.sh' -not -name 'tidy_board_bak.sh' \
      -not -name 'ssh_x5*.sh' -not -name 'askpass*.sh' \
      -not -name '.*' | sed 's|^\./||' >> "$LIST"
    sort -u "$LIST" -o "$LIST"
    {
      echo "# board_parity.md5 — 板端($BOARD_DIR) vs 本地(auv_vision/) 逐文件一致性清单"
      echo "#"
      echo "# 用法：bash tools/check_board_parity.sh               # 本地 vs 本清单（不用板子）"
      echo "#       AUV_SSH=/home/ansty/RDKX5/.ssh_x5.sh bash check_board_parity.sh --board"
      echo "#       AUV_SSH=/home/ansty/RDKX5/.ssh_x5.sh bash check_board_parity.sh --board --write  # 刷新本清单"
      echo "#"
      echo "# 格式：[ md5  文件 ] = 已与板端逐字节核对一致（板端实测证据）；无方括号 = 仅本地记录"
      echo "# 注意：tools/ 下的 check_board_parity.sh / deploy_to_board.sh / board_parity.md5 /"
      echo "#       tidy_board_bak.sh 是\"本地驱动\"工具，不入清单（板端不需要）；archive_baks.sh 入清单（板端维护用）"
      echo "# cfg/vision.yaml **与 comm.yaml 一样整份同步**（本地是唯一参数源：下水前要 red 就改本地）；"
      echo "#   板端手改过该文件的话，下次 deploy 会被本地覆盖（覆盖前会备份到 bak/deploy_<stamp>/）"
      echo "# 更新：$(date +%m%d_%H%M%S)（--write 实测刷新：1 次 SSH 批量核对）"
      echo "#"
      echo "# md5                               文件"
      n_w=0
      while read -r f; do
        [ -z "$f" ] && continue
        lm=$(md5sum "$f" 2>/dev/null | cut -d' ' -f1)
        [ -z "$lm" ] && continue
        if [ "${RM[$f]:-}" = "$lm" ]; then
          echo "[ $lm  $f ]"; n_w=$((n_w+1))
        else
          echo "$lm  $f"
        fi
      done < "$LIST" > "$MANIFEST.new"
      mv "$MANIFEST.new" "$MANIFEST"
      rm -f "$LIST"
      echo "  [write] 已刷新 $MANIFEST（板端实测一致 $n_w 项）"
    }
  fi
  [ "$remote_bad" = "0" ] || exit 1
fi
[ "$local_bad" = "0" ] || exit 1
exit 0
