#!/bin/bash
# ============================================================
# 量化一致性校验（hb_mapper checker）
#   在 OpenExplorer 容器里跑量化模拟，逐层比对「量化后」与「float ONNX」的
#   余弦相似度，用来判断 PTQ 有没有把某一层量化坏。
#
# 用法: ./scripts/3_export/check_quant.sh [config.yaml]
#   默认配置: configs/gate_kpt_config.yaml
#   ONNX / march 都从配置里读，和 quantize.sh 用同一份，避免两处不一致。
#
# 判定口径（本项目的验收线）: 最小余弦相似度 >= 0.95
#   不达标时优先按顺序调：
#     1) 校准集换更贴近部署分布、更多样的图（本项目用 D+wb 链路的 640 图）
#     2) 校准集张数加密（100 → 200/300）
#     3) calibration_type: default → max
#     4) 把掉点的层（看日志里的 Node 名）用 node_info 指定到 BPU/int16
# ============================================================
set -e

cd "$(dirname "$0")/../.."

DOCKER_IMAGE="openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310"
CONFIG=${1:-configs/gate_kpt_config.yaml}
THRESH=${THRESH:-0.95}

if ! docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
    echo "❌ 未找到Docker镜像: $DOCKER_IMAGE"
    exit 1
fi
[ -f "$CONFIG" ] || { echo "❌ 配置文件不存在: $CONFIG"; exit 1; }

# 与 quantize.sh 完全相同的解析方式
ONNX_IN_CONTAINER=$(sed -n "s|^[[:space:]]*onnx_model:[[:space:]]*['\"]\{0,1\}\([^'\"]*\)['\"]\{0,1\}[[:space:]]*$|\1|p" "$CONFIG" | head -1)
ONNX_HOST="${ONNX_IN_CONTAINER#/data/}"
MARCH=$(sed -n "s|^[[:space:]]*march:[[:space:]]*['\"]\{0,1\}\([^'\"]*\)['\"]\{0,1\}[[:space:]]*$|\1|p" "$CONFIG" | head -1)
INPUT_NAME=${INPUT_NAME:-images}
INPUT_SHAPE=${INPUT_SHAPE:-1x3x640x640}

[ -n "$ONNX_IN_CONTAINER" ] || { echo "❌ 无法从配置解析 onnx_model: $CONFIG"; exit 1; }
[ -f "$ONNX_HOST" ] || { echo "❌ ONNX 不存在: $ONNX_HOST（配置里是 $ONNX_IN_CONTAINER）"; exit 1; }

echo "🔎 hb_mapper checker"
echo "   配置: $CONFIG"
echo "   ONNX: $ONNX_HOST"
echo "   march: ${MARCH:-bayes-e}   输入: $INPUT_NAME $INPUT_SHAPE"
echo ""

docker run --rm \
    -v "$(pwd)":/data \
    -w /data \
    --name oe_toolchain_check \
    "$DOCKER_IMAGE" \
    hb_mapper checker --model-type onnx --march "${MARCH:-bayes-e}" \
        --model "$ONNX_IN_CONTAINER" \
        --input-shape "$INPUT_NAME" "$INPUT_SHAPE"

mkdir -p output
LOG=$(ls -t hb_mapper_checker_*.log 2>/dev/null | head -1)
if [ -z "$LOG" ]; then
    echo "⚠️  没找到 hb_mapper_checker_*.log，请手动查看容器输出"
    exit 1
fi
mv "$LOG" "output/${LOG}"
echo ""
echo "📄 日志: output/${LOG}"
echo "----------------------------------------"
python3 - "$LOG" "$THRESH" <<'PY'
import re, sys
from pathlib import Path
log, thresh = Path(sys.argv[1]), float(sys.argv[2])
txt = log.read_text(errors="ignore")
# checker 的日志是一张表：Node 名 + 若干数值列（含 Cosine Similarity）
rows = []
for line in txt.splitlines():
    m = re.match(r"\s*([\w./:-]+)\s+((?:-?\d+\.\d+\s+)+)", line)
    if m:
        vals = [float(v) for v in m.group(2).split()]
        rows.append((m.group(1), vals))
if not rows:
    print("（没解析出数值表，直接看日志尾部）")
    print("\n".join(txt.splitlines()[-25:]))
    sys.exit(0)
# 余弦相似度通常是表里最后一列（1.0 = 完全一致）
cos = [(n, v[-1]) for n, v in rows]
bad = [(n, c) for n, c in cos if c < thresh]
print(f"解析到 {len(cos)} 行；余弦相似度 min = {min(c for _, c in cos):.4f}  "
      f"（阈值 {thresh}）")
worst = sorted(cos, key=lambda x: x[1])[:8]
print("最差 8 层：")
for n, c in worst:
    print(f"   {c:.4f}  {n}")
print()
if bad:
    print(f"❌ 有 {len(bad)} 层低于 {thresh} —— 需要调校准集/量化参数后重跑（最多再来一轮）")
else:
    print(f"✅ 全部 >= {thresh}，量化一致性通过")
PY
