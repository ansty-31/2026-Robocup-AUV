#!/bin/bash
# tools/tidy_board_bak.sh — 整理**板端** bak/：按时间归档压缩 + 去冗余，只留最近 K 个快照可直接回滚
#
#   bash tools/tidy_board_bak.sh --dry-run     # 只打印将做什么（不动板端）
#   bash tools/tidy_board_bak.sh               # 执行
#   KEEP=5 bash tools/tidy_board_bak.sh        # 保留最近 5 个快照不解压（默认 3）
#   BIG_FILES=10 bash tools/tidy_board_bak.sh  # 文件数 ≥ 它的算"大改动"，也保留（默认 10）
#
# 整理后的板端布局（<板端> = AUV_BOARD_DIR，默认 /home/sunrise/AUV）：
#   <板端>/bak/
#   ├── README.md                      # 自动生成的索引（含每个归档的内容 + 恢复方法）
#   ├── rollback/deploy_MMDD_HHMMSS/   # 值得留着回滚的**原样**快照：cp -p 回去即可
#   │                                  #   = 最近 KEEP 个 ∪ 文件数 ≥ BIG_FILES 的大改动
#   └── archive/
#       ├── deploy_YYYY-MM-DD.tar.gz   # 该天其余 deploy_* 快照（按目录名里的 MMDD 归类）
#       ├── legacy_YYYY-MM-DD.tar.gz   # 散落的 *.bak_* 文件 + bak/cfg（archive_baks.sh 的产物）
#       └── *.tgz                      # 板端原有归档，原样移入
#
# 原则：
#   * **先打包再删**：除"空目录"外，任何内容都不会被丢掉（要彻底释放空间就手工删 archive/）；
#   * 日期一律用**本地时间**：板端 RTC 常年不准（`deploy_*` 目录名是本地时间生成的，
#     板端 `date`/mtime 却是 2000-01-01 那一类），所以归档名 = 目录名 MMDD + 本地年份；
#   * 幂等：可随时重跑；新的 `bak/deploy_<stamp>/`（deploy_to_board.sh 每次同步前生成）会被
#     自动归入 rollback/ 或 archive/；
#   * 只操作板端 `<板端>/bak/`，不动代码目录。
#
# SSH 走与 deploy_to_board.sh 相同的包装器（默认 /home/ansty/RDKX5/.ssh_x5*.sh）：
#   AUV_SSH / AUV_STREAM / AUV_BOARD_DIR / AUV_HELP_DIR
set -u
TOOLS_DIR="$(cd "$(dirname "$0")" && pwd)"
HELP_DIR="${AUV_HELP_DIR:-/home/ansty/RDKX5}"
SSH="${AUV_SSH:-$HELP_DIR/.ssh_x5.sh}"                 # 探板子用（不喂 stdin）
STREAM="${AUV_STREAM:-$HELP_DIR/.ssh_x5_stream.sh}"    # 传脚本用（保留 stdin）
BOARD="${AUV_BOARD_DIR:-/home/sunrise/AUV}"
BOARD_BAK="$BOARD/bak"
KEEP="${KEEP:-3}"
BIG_FILES="${BIG_FILES:-10}"
TODAY="$(date +%Y-%m-%d)"      # 本地日期：板端 RTC 不可信，归档名一律用它
NOW="$(date '+%Y-%m-%d %H:%M:%S')"
YEAR="$(date +%Y)"
DRY=0
for a in "$@"; do
  case "$a" in
    --dry-run|-n) DRY=1 ;;
    --help|-h) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "未知参数：$a（可用 --dry-run | --help）" >&2; exit 2 ;;
  esac
done
[ -x "$SSH" ] || { echo "缺少 SSH 包装器 $SSH（见 tools/README.md）"; exit 2; }
[ -x "$STREAM" ] || STREAM="$SSH"

TMPD=$(mktemp -d); trap 'rm -rf "$TMPD"' EXIT

cat > "$TMPD/tidy.sh" <<'EOS'
set -u
BAK="__BAK__"
BOARD="__BOARD__"
KEEP=__KEEP__
BIG_FILES=__BIG_FILES__
TODAY="__TODAY__"          # 本地日期（板端 RTC 不可信）
NOW="__NOW__"              # 本地时间
YEAR="__YEAR__"
DRY=__DRY__
cd "$BAK" 2>/dev/null || { echo "[tidy] ✗ 板端没有 $BAK"; exit 1; }

echo "[tidy] 目标：$BAK（保留 最近 $KEEP 个 ∪ 文件数≥$BIG_FILES 的快照；DRY=$DRY）"
echo "[tidy] 板端时间：$(date '+%Y-%m-%d %H:%M:%S')（RTC 不可信属正常，归档用本地时间 $TODAY）"
echo "[tidy] 整理前：$(du -sh "$BAK" | cut -f1)，$(find "$BAK" -type f | wc -l) 个文件，$(find "$BAK" -maxdepth 1 -type d -name 'deploy_*' | wc -l) 个 deploy_* 目录"

# ---- 0) 空备份目录：里面没有任何文件，直接删（唯一"无归档即删"的情形）----
for d in deploy_*/ rollback/deploy_*/; do
  [ -d "$d" ] || continue
  if [ -z "$(find "$d" -type f -print -quit)" ]; then
    echo "[tidy] 删除空目录: $d"
    [ "$DRY" = 1 ] || rm -rf -- "$d"
  fi
done

mkdir -p archive rollback

# ---- 1) 候选快照 = bak/deploy_* + bak/rollback/deploy_*，按时间戳倒序 ----
mapfile -t cand < <(
  { for d in deploy_*/ rollback/deploy_*/; do
      [ -d "$d" ] || continue
      [ -n "$(find "$d" -type f -print -quit)" ] || continue
      printf '%s\t%s\n' "$(basename "$d")" "${d%/}"
    done; } | sort -r -k1,1 | cut -f2
)
echo "[tidy] 非空快照共 ${#cand[@]} 个"

rank=0
declare -A day_dirs=()          # 待归档：YYYY-MM-DD -> "dir1 dir2 ..."（同一天合成**一个** tar.gz）
for p in "${cand[@]:-}"; do
  [ -n "$p" ] || continue
  rank=$((rank + 1))
  base="$(basename "$p")"
  nfiles="$(find "$p" -type f | wc -l)"
  # 保留判据：最近 KEEP 个 **或** 文件数 ≥ BIG_FILES（大改动最值得回滚）
  if [ "$rank" -le "$KEEP" ] || [ "$nfiles" -ge "$BIG_FILES" ]; then
    if [ "$rank" -le "$KEEP" ]; then why="最近 $KEEP"; else why="大改动($nfiles 文件)"; fi
    case "$p" in
      rollback/*) echo "[tidy] 保留回滚点[$why]: $p" ;;
      *) echo "[tidy] 保留回滚点[$why]: $p  ->  rollback/$base"
         [ "$DRY" = 1 ] || mv -- "$p" rollback/ ;;
    esac
    continue
  fi
  mmdd="${base#deploy_}"; mmdd="${mmdd%%_*}"
  ym="$YEAR-${mmdd:0:2}-${mmdd:2:2}"
  day_dirs[$ym]="${day_dirs[$ym]:-} $p"     # 目录名里没有空格，用空白分隔足够
done

# ---- 2) 按天归档：一天一个 archive/deploy_YYYY-MM-DD.tar.gz ----
for ym in $(printf '%s\n' "${!day_dirs[@]}" | sort); do
  dirs="${day_dirs[$ym]}"
  cnt=$(printf '%s\n' $dirs | wc -l)
  tgz="archive/deploy_${ym}.tar.gz"
  i=1; while [ -e "$tgz" ]; do tgz="archive/deploy_${ym}.$i.tar.gz"; i=$((i + 1)); done
  echo "[tidy] 归档 $cnt 个快照 -> $tgz"
  [ "$DRY" = 1 ] || { tar czf "$tgz" $dirs && rm -rf -- $dirs; }
done

# ---- 3) 板端原有压缩包：原样移进 archive/（已经是压缩态，不重复压）----
for f in *.tgz *.tar.gz; do
  [ -e "$f" ] || continue
  echo "[tidy] 移入 archive/: $f"
  [ "$DRY" = 1 ] || mv -- "$f" archive/
done

# ---- 4) 散落的 *.bak_*（archive_baks.sh 的产物）+ bak/cfg → 一个按日期的 legacy 包 ----
legacy=()
for f in *.bak_* *.bak *.bak.*; do
  [ -e "$f" ] && legacy+=("$f")
done
[ -d cfg ] && legacy+=(cfg)
if [ "${#legacy[@]}" -gt 0 ]; then
  tgz="archive/legacy_${TODAY}.tar.gz"
  i=1; while [ -e "$tgz" ]; do tgz="archive/legacy_${TODAY}.$i.tar.gz"; i=$((i + 1)); done
  echo "[tidy] 归档散落备份 ${#legacy[@]} 项（$(du -ch "${legacy[@]}" 2>/dev/null | tail -1 | cut -f1)）-> $tgz"
  [ "$DRY" = 1 ] || { tar czf "$tgz" "${legacy[@]}" && rm -rf -- "${legacy[@]}"; }
else
  echo "[tidy] 无散落 *.bak_* 需要归档"
fi

# ---- 5) 生成索引 bak/README.md（板端本地文件，不入同步清单：清单排除 ./bak/*）----
if [ "$DRY" = 1 ]; then
  echo "[tidy] (dry-run) 跳过生成 README.md"
else
  {
    echo "# 板端备份区（$BOARD/bak）"
    echo
    echo "> 本目录由 \`tools/tidy_board_bak.sh\` 整理：**先打包再删**，除空目录外不丢内容。"
    echo "> 整理时间（本地）：$NOW；保留 **最近 $KEEP 个 ∪ 文件数≥$BIG_FILES 的大改动** 快照不解压。"
    echo "> ⚠️ 板端 RTC 不准（当前板端时间 $(date '+%Y-%m-%d %H:%M:%S')），目录名/mtime 可能显示 2000 年，属正常。"
    echo
    echo "## rollback/ —— 可直接回滚（同步前的原样快照）"
    echo
    if compgen -G "rollback/deploy_*" > /dev/null; then
      for d in rollback/deploy_*/; do
        [ -d "$d" ] || continue
        echo "- \`${d%/}\` — $(find "$d" -type f | wc -l) 个文件，$(du -sh "$d" | cut -f1)，$(find "$d" -type f | head -3 | sed "s|^$d||" | tr '\n' ' ')"
      done
    else
      echo "（空）"
    fi
    echo
    echo "## archive/ —— 按时间归档（tar.gz，要空间就直接删这里）"
    echo
    for f in archive/*; do
      [ -e "$f" ] || continue
      echo "- \`$(basename "$f")\` — $(du -sh "$f" | cut -f1)"
    done
    echo
    echo "## 回滚 / 取文件"
    echo
    echo '```bash'
    echo "cd $BOARD"
    echo "# 整份回滚到某次同步前（例：改动最多的那次 = 大同步前）"
    biggest="$(for d in rollback/deploy_*/; do [ -d "$d" ] || continue
                 printf '%s\t%s\n' "$(find "$d" -type f | wc -l)" "${d%/}"; done \
               | sort -rn | head -1 | cut -f2)"
    echo "cp -rp bak/${biggest:-rollback/deploy_MMDD_HHMMSS}/. ."
    echo "# 看归档里有什么 / 只取某个文件"
    echo "tar tzf bak/archive/deploy_${YEAR}-*.tar.gz | head"
    echo "tar xzf bak/archive/deploy_${YEAR}-09-16.tar.gz -C /tmp && cp -p /tmp/deploy_0916_185910/cfg/comm.yaml cfg/"
    echo '```'
    echo
    echo "> \`bak/deploy_<stamp>/\` 是 \`deploy_to_board.sh\` 在每次覆盖/删除**之前**生成的快照；"
    echo "> 定时重跑 \`tools/tidy_board_bak.sh\` 即可把它们归类到本目录结构里。"
  } > README.md
  echo "[tidy] 已写索引 $BAK/README.md"
fi

echo "[tidy] 整理后：$(du -sh "$BAK" | cut -f1)，$(find "$BAK" -type f | wc -l) 个文件"
echo "[tidy] 布局："
find . -maxdepth 2 -mindepth 1 \( -type d -o -type f \) | sort | head -40 | sed 's/^/         /'
echo TIDY-OK
EOS

sed -i "s|__BAK__|$BOARD_BAK|; s|__BOARD__|$BOARD|; s|__KEEP__|$KEEP|; \
        s|__BIG_FILES__|$BIG_FILES|; s|__TODAY__|$TODAY|; s|__NOW__|$NOW|; s|__YEAR__|$YEAR|; \
        s|__DRY__|$DRY|" "$TMPD/tidy.sh"
grep -q '__' "$TMPD/tidy.sh" && { echo "[tidy] ✗ 变量替换不完整："; grep -n '__[A-Z_]*__' "$TMPD/tidy.sh"; exit 2; }

# 探板子（快速失败，避免 SSH 卡住）
timeout 12 "$SSH" "echo BOARD-UP" </dev/null 2>/dev/null | grep -q BOARD-UP \
  || { echo "[tidy] ✗ 板子未接入"; exit 1; }

timeout 300 "$STREAM" "bash -s" < "$TMPD/tidy.sh" 2>&1 | grep -v "Warning: Permanently"
