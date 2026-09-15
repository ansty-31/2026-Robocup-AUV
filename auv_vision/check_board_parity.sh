#!/bin/bash
# check_board_parity.sh — 检查“本地代码 == 最后移植到板端的代码”
#
# 用法：
#   bash check_board_parity.sh                    # 本地 vs doc/board_parity.md5（无需板子）
#   AUV_SSH=/home/ansty/RDKX5/.ssh_x5.sh \
#     bash check_board_parity.sh --board          # 再逐文件与板端比对（需要板子在线）
#   bash check_board_parity.sh --board --only-verified   # 只查有板端实测证据的那批
#
# 说明：manifest 记录的是“最后一次与板端逐文件 md5 比对一致”时的字节；
#   ① 本地全绿 = 本地没被改过；
#   ② --board 也全绿 = 板端确实装的就是这份代码。
#   只读操作：不写板端、不改本地。
set -u
cd "$(dirname "$0")"

MANIFEST="doc/board_parity.md5"
BOARD_DIR="${AUV_BOARD_DIR:-/home/sunrise/AUV}"
AUV_SSH="${AUV_SSH:-ssh}"
ONLY_VERIFIED=0
for a in "$@"; do [ "$a" = "--only-verified" ] && ONLY_VERIFIED=1; done

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
  echo "  ✓ 本地与“最后同步到板端的状态”逐字节一致"
else
  echo "  ✗ 本地有改动/缺失（板端未必同步过）"
fi

# 2) 可选：与板端实机比对
if [ "${1:-}" = "--board" ] || [ "${2:-}" = "--board" ]; then
  echo "==== [2/2] 板端 $BOARD_DIR vs 本地 ===="
  if ! timeout 12 $AUV_SSH "echo BOARD-UP" </dev/null 2>/dev/null </dev/null | grep -q BOARD-UP; then
    echo "  ✗ 板子不可达（确认网线/供电，或设置 AUV_SSH 指向你的 ssh 包装脚本）"
    exit 1
  fi
  remote_bad=0; rn=0
  while IFS=$'\t' read -r md5 f verified; do
    [ "$ONLY_VERIFIED" = "1" ] && [ "$verified" != "1" ] && continue
    rn=$((rn+1))
    r=$(timeout 20 $AUV_SSH "md5sum $BOARD_DIR/$f 2>/dev/null | cut -d' ' -f1" 2>/dev/null </dev/null | tr -d '\r\n')
    if [ -z "$r" ]; then
      echo "  [板端缺失] $f"; remote_bad=$((remote_bad+1)); continue
    fi
    [ "$r" = "$md5" ] || { echo "  [板端不同] $f  本地=$md5 板端=$r"; remote_bad=$((remote_bad+1)); }
  done < <(parse)
  echo "  比对板端文件 $rn 个，不一致 $remote_bad 个"
  [ "$remote_bad" = "0" ] && echo "  ✓ 板端就是这份代码" || echo "  ✗ 板端与本地有差异"
  [ "$remote_bad" = "0" ] || exit 1
fi
[ "$local_bad" = "0" ] || exit 1
exit 0
