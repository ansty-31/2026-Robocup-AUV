#!/bin/bash
# pc.sh — PC 端（水面电脑）总调度：键盘遥控 + 本地录像
#   画面推流由板端负责（板端 ./manual.sh）；PC 端只接收、录像、键盘遥控。
#
# 用法（在 pc/ 目录执行）：
#   ./pc.sh                              # 键盘遥控 + 录像（默认直接出 mp4，零转码）
#   ./pc.sh --show                       # 边看边录（结束后自动封 mp4）
#   ./pc.sh --raw                        # 只留原始 .mjpeg（不封 mp4）
#   ./pc.sh --view                       # 只看画面（不录、不控）
#   ./pc.sh --no-key                     # 只录像（不用键盘遥控）
#   ./pc.sh --no-record                  # 只键盘遥控
#   ./pc.sh --duration 60                # 录 60 秒自动结束
#   ./pc.sh --host 192.168.137.10 --port 5000 --ctrl-port 9000 --out-dir record
#   ./pc.sh --kill-stale --show          # 先清掉上次残留的录像进程（释放 UDP 端口）
#   ./pc.sh --dry-run                    # 只打印将执行的命令
#
# 录像产物（默认在 pc/record/）：
#   auv_<时间戳>.mp4      ← 默认产物（ffmpeg -c:v copy 零转码；--show 时结束后自动封装）
#   auv_<时间戳>.mjpeg     ← 仅 --raw / --keep-raw 时保留（原始流）
#   auv_<时间戳>.mjpeg.timestamps  ← 逐帧到达时间轴
#
# 顺序建议：先起板端 ./manual.sh，再起本脚本；结束先 Ctrl-C 本脚本，再停板端。
set -u
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
HOST="192.168.137.10"      # 板端 IP（键盘遥控目标）
PORT=5000                  # 板端推流端口（本机接收）
CTRL_PORT=9000             # 板端遥控桥端口（本机发送）
OUT_DIR="record"
DURATION=0
USE_KEY=1
USE_REC=1
SHOW=0
MP4=1                      # 默认走 ffmpeg 直录 mp4（无窗口）
RAW=0
VIEW=0
HTTP=""
SPEED=""
DRY=0
KILL_STALE=0

usage() { sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --host)       HOST="$2"; shift 2 ;;
    --port)       PORT="$2"; shift 2 ;;
    --ctrl-port)  CTRL_PORT="$2"; shift 2 ;;
    --out-dir)    OUT_DIR="$2"; shift 2 ;;
    --duration)   DURATION="$2"; shift 2 ;;
    --http)       HTTP="$2"; shift 2 ;;
    --speed)      SPEED="$2"; shift 2 ;;
    --show)       SHOW=1; shift ;;
    --mp4)        MP4=1; RAW=0; shift ;;
    --raw)        RAW=1; MP4=0; shift ;;
    --view)       VIEW=1; USE_KEY=0; shift ;;
    --no-key)     USE_KEY=0; shift ;;
    --no-record)  USE_REC=0; shift ;;
    --kill-stale) KILL_STALE=1; shift ;;
    --dry-run)    DRY=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
done

mkdir -p "${OUT_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${OUT_DIR}/pc_${TS}.log"

# ---- 录像命令 ----
REC_CMD=()
if [ "${USE_REC}" = "1" ]; then
  REC_CMD=("${PY}" -u pc_recorder.py --out-dir "${OUT_DIR}" --port "${PORT}")
  [ -n "${HTTP}" ] && REC_CMD+=(--http "${HTTP}")
  [ "${DURATION}" != "0" ] && REC_CMD+=(--duration "${DURATION}")
  if [ "${VIEW}" = "1" ]; then
    REC_CMD+=(--show --discard)
  elif [ "${SHOW}" = "1" ]; then
    REC_CMD+=(--show)                       # 进程内直存 + 显示，结束时自动封 mp4
    [ "${RAW}" = "1" ] && REC_CMD+=(--no-remux)
  elif [ "${MP4}" = "1" ]; then
    REC_CMD+=(--mp4)                        # ffmpeg 直录 mp4（零转码，无窗口）
  elif [ "${RAW}" = "1" ]; then
    REC_CMD+=(--no-remux)                   # 只留原始 mjpeg
  fi
fi

# ---- 键盘遥控命令 ----
KEY_CMD=()
if [ "${USE_KEY}" = "1" ]; then
  KEY_CMD=("${PY}" -u pc_keyboard_client.py --host "${HOST}" --port "${CTRL_PORT}")
  [ -n "${SPEED}" ] && KEY_CMD+=(--speed "${SPEED}")
fi

if [ "${DRY}" = "1" ]; then
  echo "[pc.sh] recorder: ${REC_CMD[*]:-(off)}"
  echo "[pc.sh] keyboard: ${KEY_CMD[*]:-(off)}"
  echo "[pc.sh] log: ${LOG}"
  exit 0
fi

# 本机 UDP 端口是否已被占用
port_in_use() {
  if command -v ss >/dev/null 2>&1; then
    ss -lun 2>/dev/null | grep -qE "[:.]${PORT}[[:space:]]"
  elif command -v netstat >/dev/null 2>&1; then
    netstat -lun 2>/dev/null | grep -qE "[:.]${PORT}[[:space:]]"
  else
    return 1                                # 检测不了就交给录像进程自己报错
  fi
}

# 0) 可选：清理残留录像进程（上次异常退出会一直占着 UDP 端口）
if [ "${KILL_STALE}" = "1" ]; then
  killed=0
  for pat in "pc_recorder.py" "manual.stream view" "udp://@:${PORT}"; do
    if pkill -f "${pat}" 2>/dev/null; then killed=1; echo "[pc.sh] 已清理残留进程: ${pat}"; fi
  done
  if [ "${killed}" = "1" ]; then sleep 1; else echo "[pc.sh] 没有发现残留进程"; fi
fi

# 1) 端口占用预检：与其启动后报错，不如现在就给处置办法
if [ "${USE_REC}" = "1" ] && [ -z "${HTTP}" ] && port_in_use; then
  cat >&2 <<EOF
!! 本机 UDP ${PORT} 已被占用，无法接收板端推流（多半是上一次录制/查看的进程没退干净）。
   任选一种处理：
     1) 找到并结束它： ss -lunp | grep ':${PORT}'       → kill <PID>
        （Windows: netstat -ano | findstr :${PORT}     → taskkill /PID <PID> /F）
     2) 让本脚本自动清： ./pc.sh --kill-stale$([ "${SHOW}" = "1" ] && echo " --show")
     3) 换一个端口：    ./pc.sh --port 5010$([ "${SHOW}" = "1" ] && echo " --show")   （板端也要 ./manual.sh --port 5010）
EOF
  exit 2
fi

REC_PID=""
CLEANED=0
cleanup() {
  [ "${CLEANED}" = "1" ] && return
  CLEANED=1
  echo
  echo "==== 收尾 ===="
  if [ -n "${REC_PID}" ] && kill -0 "${REC_PID}" 2>/dev/null; then
    kill -INT "${REC_PID}" 2>/dev/null
    i=0
    while kill -0 "${REC_PID}" 2>/dev/null && [ "${i}" -lt 10 ]; do sleep 0.5; i=$((i + 1)); done
    kill -0 "${REC_PID}" 2>/dev/null && kill -TERM "${REC_PID}" 2>/dev/null
    wait "${REC_PID}" 2>/dev/null
    echo "  录像已停止"
  fi
  echo "==== 收尾完成 ===="
  ls -1t "${OUT_DIR}"/auv_* 2>/dev/null | head -3 | sed 's/^/  产物: /'
  if [ -s "${LOG}" ]; then
    echo "--- 录像日志尾部 (${LOG}) ---"
    tail -n 6 "${LOG}"
  fi
}
trap cleanup EXIT INT TERM HUP

echo "=================================================================="
echo " PC 端总调度"
echo "   键盘遥控 $( [ "${USE_KEY}" = "1" ] && echo "${HOST}:${CTRL_PORT}（窗口需保持焦点；W/S/A/D、方向键、空格停、Esc 退出）" || echo "(关闭)" )"
if [ "${VIEW}" = "1" ]; then
  echo "   画面    只看不录（--show --discard）"
elif [ "${USE_REC}" = "1" ]; then
  echo "   录像    $(
      if [ "${SHOW}" = "1" ] || [ "${VIEW}" = "1" ]; then echo "进程内接收（结束自动封 mp4）";
      elif [ "${RAW}" = "1" ]; then echo "原始 MJPEG 直存";
      else echo "ffmpeg 直录 mp4（零转码，无窗口）"; fi
    ) → ${OUT_DIR}/"
else
  echo "   录像    (关闭)"
fi
echo "   接收    $( [ -n "${HTTP}" ] && echo "${HTTP}" || echo "udp://0.0.0.0:${PORT}" )"
echo "   日志    ${LOG}"
echo "   停止    Ctrl-C（会先停录像并给出转 mp4 命令）"
echo "=================================================================="

# 2) 录像（后台，日志落盘；--show/--view 时同时显示到终端）
if [ "${USE_REC}" = "1" ]; then
  if [ "${VIEW}" = "1" ] || [ "${SHOW}" = "1" ]; then
    # 用进程替换而不是 `| tee`：管道时 $! 是 tee 的 PID，Ctrl-C 只会杀掉 tee，
    # python 录像进程会变成孤儿继续占着 UDP 端口（曾因此无法再次绑定 5000）。
    "${REC_CMD[@]}" > >(tee "${LOG}") 2>&1 &
  else
    "${REC_CMD[@]}" > "${LOG}" 2>&1 &
  fi
  REC_PID=$!
  sleep 1
  if ! kill -0 "${REC_PID}" 2>/dev/null; then
    echo "!! 录像启动失败，见 ${LOG}" >&2
    tail -n 8 "${LOG}" 2>/dev/null
    rm -f "${OUT_DIR}"/auv_* 2>/dev/null      # 启动即失败：清掉本次的 0 字节残留
    REC_PID=""
    exit 1
  fi
fi

# 3) 键盘遥控（前台）
if [ "${USE_KEY}" = "1" ]; then
  "${KEY_CMD[@]}"
  RC=$?
  echo "[pc.sh] 键盘遥控退出 (rc=${RC})"
elif [ "${USE_REC}" = "1" ]; then
  wait "${REC_PID}" 2>/dev/null               # 只录像：等它自然结束（--duration）或 Ctrl-C
fi
exit 0
