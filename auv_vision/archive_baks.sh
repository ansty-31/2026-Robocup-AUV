#!/bin/bash
# archive_baks.sh — 把工程内散落的备份文件统一收进 bak/
#   约定：cfg/ 下的备份 → bak/cfg/ ；其余 → bak/（平铺）
#   同名冲突自动加序号，不覆盖已归档文件
#
# 用法（在工程根目录执行）：
#   ./archive_baks.sh              # 归档（移动 *.bak* 到 bak/）
#   ./archive_baks.sh --dry-run    # 只列出将要移动的文件，不动
#   ./archive_baks.sh --list       # 查看 bak/ 现有内容
#   ./archive_baks.sh --help
set -u
cd "$(dirname "$0")"

BAK="bak"
SRC_EXCLUDES=(-not -path "./$BAK/*" -not -path "./.git/*"
              -not -path "*/__pycache__/*" -not -path "./rec/*"
              -not -path "./log/*" -not -path "./models/*")

case "${1:-}" in
  --help|-h) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  --list|-l) echo "bak/ 现有备份："; find "$BAK" -type f 2>/dev/null | sort
             echo "共 $(find "$BAK" -type f 2>/dev/null | wc -l) 个"; exit 0 ;;
  --dry-run|-n) DRY=1 ;;
  "") DRY=0 ;;
  *) echo "未知参数：$1（可用 --dry-run | --list | --help）" >&2; exit 2 ;;
esac

mkdir -p "$BAK"

# 先取快照再移动（避免边遍历边改目录）
mapfile -t files < <(find . -type f -name "*.bak*" "${SRC_EXCLUDES[@]}" | sort)

if [ "${#files[@]}" -eq 0 ]; then
  echo "没有需要归档的备份文件（bak/ 之外无 *.bak*）"
  exit 0
fi

moved=0
for f in "${files[@]}"; do
  rel="${f#./}"                 # 去掉 ./ 前缀
  dir="$(dirname "$rel")"       # 原所在目录
  base="$(basename "$rel")"
  if [ "$dir" = "cfg" ]; then   # 约定：cfg 的备份进 bak/cfg/
    dst_dir="$BAK/cfg"
  else
    dst_dir="$BAK"
  fi
  mkdir -p "$dst_dir"
  dst="$dst_dir/$base"
  if [ -e "$dst" ]; then        # 同名冲突 → 加序号
    i=1
    while [ -e "${dst}.$i" ]; do i=$((i + 1)); done
    dst="${dst}.$i"
  fi
  if [ "${DRY:-0}" = "1" ]; then
    echo "[dry] $rel  ->  $dst"
  else
    mv -- "$rel" "$dst" && { echo "moved: $rel  ->  $dst"; moved=$((moved + 1)); }
  fi
done

if [ "${DRY:-0}" = "1" ]; then
  echo "(dry-run：以上 ${#files[@]} 个将归档，未移动)"
else
  echo "完成：本次归档 $moved 个文件到 $BAK/"
fi
