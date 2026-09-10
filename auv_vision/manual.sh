#!/bin/bash
# manual.sh — 手动模式一键启动（板端）：串口遥控桥 + 画面推流
#   录像默认放在**水面 PC**（板端只推流，省板端 CPU、不受写盘拖累）：
#       PC 端: python3 -m manual.stream view --record      # 存到 PC 的 record/ 目录
#   确需板端本地录像时再加 --record（例如 PC 不在场）。
#   三件套在 manual/：manual/udp_server.py（遥控桥）· manual/recorder.py（录像）· manual/stream.py（推流/接收库）
#
# 推流是"零转码"的：base/camera.py 打开相机时顺手把原始 MJPEG 交给 manual/stream.py 发出，
# 所以**推流与录像共用同一路采集，互不抢相机**（UVC 只允许一个进程取流）。
#
# 用法（在工程根目录执行）：
#   ./manual.sh                          # 遥控桥 + 推流（录像在 PC 端做）
#   ./manual.sh --sim                    # 无串口硬件时：遥控桥只打印
#   ./manual.sh --no-udp                 # 只推流
#   ./manual.sh --record --seconds 60 --out rec.mp4    # 额外做板端本地录像
#   ./manual.sh --host 192.168.137.2 --port 5000 --stream-fps 30
#   ./manual.sh --help
#
# 水面 PC 端看画面（任选）：
#   ffplay -fflags nobuffer -flags low_delay -framedrop -f mjpeg "udp://@:5000"
#   python3 -m manual.stream view --record # 本仓库自带：看画面 + 存到 PC 的 record/ 目录
#   浏览器（板端另开 HTTP 时）：python3 -m manual.stream push --host <PC> --http 8080
set -u
cd "$(dirname "$0")"

HOST="${AUV_STREAM_HOST:-192.168.137.2}"
PORT="${AUV_STREAM_PORT:-5000}"
STREAM_FPS="${AUV_STREAM_FPS:-30}"
PKT="${AUV_STREAM_PKT:-8000}"
CAMERA="front"
OUT=""
SECONDS_REC=0
USE_UDP=1
USE_REC=0
UART_MODE="--real"
CTRL_PORT=9000
CAM_SIM=0

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --host)        HOST="$2"; shift 2 ;;
    --port)        PORT="$2"; shift 2 ;;
    --stream-fps)  STREAM_FPS="$2"; shift 2 ;;
    --pkt)         PKT="$2"; shift 2 ;;
    --camera)      CAMERA="$2"; shift 2 ;;
    --out)         OUT="$2"; shift 2 ;;
    --seconds)     SECONDS_REC="$2"; shift 2 ;;
    --ctrl-port)   CTRL_PORT="$2"; shift 2 ;;
    --sim)         UART_MODE="--sim"; shift ;;
    --real)        UART_MODE="--real"; shift ;;
    --no-udp)      USE_UDP=0; shift ;;
    --record)      USE_REC=1; shift ;;   # 板端本地录像（默认关闭，录像在 PC 端）
    --no-record)   USE_REC=0; shift ;;
    --camera-sim)  CAM_SIM=1; shift ;;   # 无相机自测（仅 --no-record 纯推流路径）
    -h|--help)     usage; exit 0 ;;
    *) echo "未知参数：$1（--help 看用法）" >&2; exit 2 ;;
  esac
done

[ -n "${OUT}" ] || OUT="/tmp/auv_manual_$(date +%m%d_%H%M%S).mp4"
mkdir -p "$(dirname "${OUT}")" 2>/dev/null || true      # 录像目录不存在就建（否则 cv2 会直接报无法创建文件）
PY="${PYTHON:-python3}"

# 让 camera.py 的推流钩子按本次手动模式的参数工作（不改 cfg/vision.yaml）
export AUV_STREAM=1
export AUV_STREAM_HOST="${HOST}"
export AUV_STREAM_PORT="${PORT}"
export AUV_STREAM_FPS="${STREAM_FPS}"
export AUV_STREAM_PKT="${PKT}"

UDP_PID=""
FG_PID=""
CLEANED=0
cleanup() {
  [ "${CLEANED}" = "1" ] && return
  CLEANED=1
  echo
  echo "==== 收尾 ===="
  # 先把两个子进程都发 INT（录像会写完文件），再各自有限等待，避免互相阻塞
  for pid in "${FG_PID}" "${UDP_PID}"; do
    [ -n "${pid}" ] && kill -INT "${pid}" 2>/dev/null
  done
  for pid in "${FG_PID}" "${UDP_PID}"; do
    [ -z "${pid}" ] && continue
    kill -0 "${pid}" 2>/dev/null || continue
    i=0
    while kill -0 "${pid}" 2>/dev/null && [ "${i}" -lt 6 ]; do sleep 0.5; i=$((i + 1)); done
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null
      sleep 0.5
      kill -0 "${pid}" 2>/dev/null && kill -KILL "${pid}" 2>/dev/null
    fi
    wait "${pid}" 2>/dev/null
    echo "  已停止 pid ${pid}"
  done
  echo "==== 收尾完成 ===="
}
trap cleanup EXIT INT TERM HUP

echo "=================================================================="
echo " 手动模式：遥控桥$( [ "${USE_UDP}" = "1" ] && echo "(udp_server ${UART_MODE}, 端口 ${CTRL_PORT})" || echo "(关闭)" )"
echo "           录像$( [ "${USE_REC}" = "1" ] && echo "(板端本地 ${CAMERA} → ${OUT})" || echo "(在 PC 端: python3 -m manual.stream view --record)" )"
echo "           推流${STREAM_FPS}fps → udp://${HOST}:${PORT}（零转码，随采集进程一起跑）"
echo "  PC 端：ffplay -fflags nobuffer -flags low_delay -framedrop -f mjpeg \"udp://@:${PORT}\""
echo "  停止：Ctrl-C（会先停录像并写完文件，再停遥控桥）"
echo "=================================================================="

# 1) 串口遥控桥（后台）
if [ "${USE_UDP}" = "1" ]; then
  "${PY}" -u manual/udp_server.py ${UART_MODE} --port "${CTRL_PORT}" &
  UDP_PID=$!
  sleep 1
  if ! kill -0 "${UDP_PID}" 2>/dev/null; then
    echo "!! udp_server 启动失败（串口被占用/权限不足？试 --sim 或 --no-udp）；" >&2
    echo "!! 遥控桥不可用，但**继续录像/推流**" >&2
    UDP_PID=""
  fi
fi

# 2) 录像（前台；推流由 camera.py 钩子顺带完成）或纯推流
#    注意：这里不用 exec —— 要让本脚本的 EXIT trap 有机会停掉后台的 udp_server
RC=0
if [ "${USE_REC}" = "1" ]; then
  set -- --camera "${CAMERA}" --out "${OUT}"
  [ "${SECONDS_REC}" != "0" ] && set -- "$@" --seconds "${SECONDS_REC}"
  "${PY}" -u manual/recorder.py "$@" &     # 后台+wait：Ctrl-C/SIGTERM 都能转发到子进程
  FG_PID=$!
  wait "${FG_PID}" || RC=$?
  echo "录像文件：$(cd "$(dirname "${OUT}")" && pwd)/$(basename "${OUT}")"
else
  set -- push --host "${HOST}" --port "${PORT}" --stream-fps "${STREAM_FPS}" --pkt "${PKT}"
  [ "${CAM_SIM}" = "1" ] && set -- "$@" --camera-sim
  "${PY}" -u -m manual.stream "$@" &
  FG_PID=$!
  wait "${FG_PID}" || RC=$?
fi
exit "${RC}"
