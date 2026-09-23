#!/usr/bin/env bash
# 融合 3000 张待标图的选图 runner（带锁：重复拉起是空操作，不会 rm 掉正在写的结果）
#
# 用法（后台）：
#   setsid nohup bash experiment/scripts/run_mix3000.sh > experiment/logs/mix3000.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=/home/ansty/anaconda3/envs/yolov8/bin/python
OUT=data/AUV_5/selected_3000_Dwb
LOCK=experiment/logs/.mix3000.lock

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "[$(date +%H:%M:%S)] 已有同名任务在跑，本次退出（不触碰输出目录）"
    exit 0
fi

# 幂等：已完成过就跳过（防止后台命令被重复执行时把成果删掉重跑）。--force 可强制重跑。
if [ "${1:-}" != "--force" ] && [ -s "$OUT/manifest.csv" ]; then
    n=$(find "$OUT" -name '*.jpg' | wc -l)
    if [ "$n" -ge 2500 ]; then
        echo "[$(date +%H:%M:%S)] $OUT 已有 $n 张 + manifest，跳过（--force 可重跑）"
        exit 0
    fi
fi

echo "[$(date +%H:%M:%S)] 开始选图 → $OUT"
rm -rf "$OUT"
"$PY" scripts/1_prepare/pose/make_mixed_gate_set.py \
    --total 3000 --dedup-thr 3.0 --oversample 3.0 --out-dir "$OUT"
echo "[$(date +%H:%M:%S)] 完成"
