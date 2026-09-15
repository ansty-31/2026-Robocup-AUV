#!/bin/bash
# ============================================================
# PTQ量化脚本（在 OpenExplorer Docker 容器内执行 hb_mapper）
# 用法: ./scripts/3_export/quantize.sh [config.yaml]
#   默认配置: configs/yolo11n_config.yaml
# ============================================================
set -e

# 切到项目根运行（容器内 /data = 项目根，configs/weights/calibration_data 相对路径才正确）
cd "$(dirname "$0")/../.."

DOCKER_IMAGE="openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310"
CONFIG=${1:-configs/yolo11n_config.yaml}

if ! docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
    echo "❌ 未找到Docker镜像: $DOCKER_IMAGE"
    echo "   请先加载镜像: docker load < docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "❌ 配置文件不存在: $CONFIG"
    exit 1
fi

# 从配置里读 onnx 路径与输出前缀（detect/pose 各用各的，不再硬编码）
#   onnx_model: '/data/weights/yolo11n-pose.onnx'   → 容器内 /data = 项目根
#   output_model_file_prefix: 'gate_kpt_bayese_640x640_nv12'
ONNX_IN_CONTAINER=$(sed -n "s|^[[:space:]]*onnx_model:[[:space:]]*['\"]\{0,1\}\([^'\"]*\)['\"]\{0,1\}[[:space:]]*$|\1|p" "$CONFIG" | head -1)
ONNX_HOST="${ONNX_IN_CONTAINER#/data/}"
PREFIX=$(sed -n "s|^[[:space:]]*output_model_file_prefix:[[:space:]]*['\"]\{0,1\}\([^'\"]*\)['\"]\{0,1\}[[:space:]]*$|\1|p" "$CONFIG" | head -1)

if [ -z "$ONNX_IN_CONTAINER" ]; then
    echo "❌ 无法从配置中解析 onnx_model: $CONFIG"
    exit 1
fi

if [ ! -f "$ONNX_HOST" ]; then
    echo "❌ ONNX模型不存在: $ONNX_HOST（配置里写的是 $ONNX_IN_CONTAINER）"
    if [[ "$CONFIG" == *gate_kpt* || "$CONFIG" == *pose* ]]; then
        echo "   请先运行: python scripts/3_export/modify_ultralytics.py --task pose"
        echo "             python scripts/3_export/export_onnx.py --task pose"
    else
        echo "   请先运行: python scripts/3_export/modify_ultralytics.py --task detect"
        echo "             python scripts/3_export/export_onnx.py"
    fi
    exit 1
fi

# 校准目录同样从配置里读：detect / pose 各用各的，避免两套校准数据互相覆盖
CAL_DIR_IN_CONTAINER=$(sed -n "s|^[[:space:]]*cal_data_dir:[[:space:]]*['\"]\{0,1\}\([^'\"]*\)['\"]\{0,1\}[[:space:]]*$|\1|p" "$CONFIG" | head -1)
CAL_DIR_HOST="${CAL_DIR_IN_CONTAINER#/data/}"
CAL_DIR_HOST="${CAL_DIR_HOST:-calibration_data}"

if [ ! -d "$CAL_DIR_HOST" ] || [ -z "$(ls -A "$CAL_DIR_HOST" 2>/dev/null)" ]; then
    echo "❌ 校准数据目录不存在或为空: $CAL_DIR_HOST（配置里写的是 ${CAL_DIR_IN_CONTAINER:-未设置}）"
    echo "   请先运行: python scripts/3_export/prepare_calibration.py \\"
    echo "               --coco-path <与该任务同分布的代表性图片目录> --output-dir $CAL_DIR_HOST"
    exit 1
fi

mkdir -p output

echo "🚀 开始PTQ量化..."
echo "   配置: $CONFIG"
echo "   ONNX: $ONNX_HOST"
echo "   校准集: $CAL_DIR_HOST ($(ls -A "$CAL_DIR_HOST" | wc -l) 个文件)"
echo "   输出前缀: $PREFIX"
echo "   Docker镜像: $DOCKER_IMAGE"
echo ""

docker run --rm \
    -v "$(pwd)":/data \
    --name oe_toolchain_quantize \
    "$DOCKER_IMAGE" \
    hb_mapper makertbin --model-type onnx --config "/data/$CONFIG"

echo ""
BIN="output/${PREFIX}.bin"
if [ -f "$BIN" ]; then
    echo "✅ 量化完成！"
    echo "   模型文件: $BIN  ($(du -h "$BIN" | cut -f1))"
else
    echo "⚠️  量化命令已结束，但未找到预期的 $BIN"
    echo "   请检查 output/ 目录内容"
    exit 1
fi
