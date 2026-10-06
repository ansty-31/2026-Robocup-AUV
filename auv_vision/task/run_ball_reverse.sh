#!/bin/bash
# run_ball_reverse.sh — 待机 → 下潜 → 前进 → 撞球 → 回退 → 前进 → (转角度 → 过门)×N → 转角度
#
# "四个门当一个门四次过"：每次"过门"都是**单门**（pass_target=1，见 cfg/comm.yaml），
set -u
cd "$(dirname "$0")/.."

export AUV_SIM_MODE="${AUV_SIM_MODE:-0}"

WAIT_S="${AUV_WAIT_S:-10}"
DESCEND_S="${AUV_DESCEND_S:-0}"
FWD_S="${AUV_FWD_S:-0}"
FWD_SURGE="${AUV_FWD_SURGE:-0.35}"
REV_S="${AUV_REV_S:-5}"
REV_SURGE="${AUV_REV_SURGE:--0.5}"
POST_FWD_S="${AUV_POST_FWD_S:-2}"
POST_FWD_SURGE="${AUV_POST_FWD_SURGE:-0.35}"
TURN_ANGLES=(
   90   # 第 1 次：第 1 扇门前转角（正=左转）
   90   # 第 2 次
   90   # 第 3 次
   90   # 第 4 次
)
# 也可用环境变量覆盖（空格分隔）：AUV_TURN_ANGLES="90 -45 90 30"
[ -n "${AUV_TURN_ANGLES:-}" ] && read -r -a TURN_ANGLES <<< "${AUV_TURN_ANGLES}"
# ⚠️ 角度**带符号**：正数=左转，负数=右转（turn_deg.py 的 --deg 会被 abs() 吃符号，
#   （细节与实测见 doc/注释历史.md）
TURN_TIMEOUT="${AUV_TURN_TIMEOUT:-20}"
BALL_SKIP="${AUV_BALL_SKIP:-0}"
REV_ON_FAIL="${AUV_REV_ON_FAIL:-0}"
TAG="${AUV_LOG_TAG:-run}"
BALL_LOG="${AUV_BALL_LOG:-log/${TAG}ball.jsonl}"
GATE_LOG_PREFIX="${AUV_GATE_LOG_PREFIX:-log/${TAG}gate}"
LOG="${AUV_DOF_LOG:-/tmp/${TAG}path.csv}"

# 清理上次残留：孤儿 main.py 会占住相机/串口，下次启动像"锁死"
if pkill -f "[p]ython3 main.py --task" 2>/dev/null; then
  echo "[clean] 已清理上一次残留的 main.py 进程"
  sleep 1
fi

echo "=================================================================="
echo " 待机 → 下潜 → 前进 → 撞球 → 回退 → 前进 → (转角度→过门)×${#TURN_ANGLES[@]}"
echo "   待机 ${WAIT_S}s | 下潜 ${DESCEND_S}s | 前进 ${FWD_S}s | 回退 ${REV_S}s @ ${REV_SURGE}"
echo "   回退后前进 ${POST_FWD_S}s @ ${POST_FWD_SURGE} | 过门 ${#TURN_ANGLES[@]} 次(单门)"
echo "   转角序列（现场代填，正=左转/负=右转）：${TURN_ANGLES[*]}"
[ "${AUV_SIM_MODE}" = "1" ] && echo " ⚠️ AUV_SIM_MODE=1：只打印指令，不会真正驱动电机！"
echo " 日志：撞球 ${BALL_LOG} ｜ 过门 ${GATE_LOG_PREFIX}_<N>.jsonl ｜ 轨迹 ${LOG}"
echo "=================================================================="

# ---- 通用：定时发一组 DOF（结束时硬停回中位）--------------------------------
timed_dof() {   # $1=秒 $2=surge $3=sway $4=heave $5=yaw $6=标签
  python3 - "$1" "$2" "$3" "$4" "$5" "$6" <<'PYEOF'
from base.hw.uart import UartController
import sys, time
dur, surge, sway, heave, yaw = (float(sys.argv[1]), float(sys.argv[2]),
                               float(sys.argv[3]), float(sys.argv[4]),
                               float(sys.argv[5]))
label = sys.argv[6]
if dur <= 0:
    print("[%s] 跳过（时长 0）" % label)
    raise SystemExit(0)
u = UartController()
if u.sim:
    print("!! 串口处于 SIM（只打印）：[%s] 不会真正驱动电机" % label)
print("[%s] 发送 %.1fs dof=(surge %.2f, sway %.2f, heave %.2f, yaw %.2f) ..."
      % (label, dur, surge, sway, heave, yaw))
try:
    t0 = time.time()
    while time.time() - t0 < dur:
        u.send_dof(surge, sway, heave, yaw)
        time.sleep(0.05)
finally:
    u.stop_hard(verify=False)        # 连发中性帧走完 ramp（下位机无无帧超时停车）
    u.close()
print("[%s] 完成，已回中位（硬停）" % label)
PYEOF
}

# ---- 转角度（**带符号**：正=左转 / 负=右转；手动指令 turn_deg.py，失败只报不中断）----
turn_once() {   # $1=带符号角度，如 90 / -45
  local sdeg="$1" mag="$1" dir="left" txt="左转"
  case "${sdeg}" in
    -*) dir="right"; txt="右转"; mag="${sdeg#-}" ;;
  esac
  if [ "${mag}" = "0" ] || [ -z "${mag}" ]; then
    echo "-- 角度 0，跳过转向 --"
    return 0
  fi
  python3 common/motion/turn_deg.py --deg "${mag}" --dir "${dir}" --timeout "${TURN_TIMEOUT}"
  echo "-- ${txt} ${mag}° 转向退出码 $?（0=完成 / 3=等完成超时 / 5=下发失败）--"
  return 0
}

# ---- 过门一次（单门 pass_target=1）----------------------------------------
gate_once() {   # $1=第几次
  local idx="$1" glog="${GATE_LOG_PREFIX}_${idx}.jsonl"
  echo "==== 过门 ${idx}/${#TURN_ANGLES[@]}（单门）→ ${glog} ===="
  AUV_TASK_LOG="${glog}" python3 main.py --task gate
  echo "-- 过门 ${idx} 退出码 $? --"
}

# ============================== 流程 ==============================
echo "==== [1] 待机 ${WAIT_S} 秒 ===="
sleep "${WAIT_S}"

echo "==== [2] 下潜 ${DESCEND_S} 秒 ===="
timed_dof "${DESCEND_S}" 0 0 -1 0 descend

echo "==== [3] 前进 ${FWD_S} 秒 (surge=${FWD_SURGE}) ===="
timed_dof "${FWD_S}" "${FWD_SURGE}" 0 0 0 pre_forward

echo "==== [4] 撞球任务 main.py --task ball（轨迹 → ${LOG}，逐帧 → ${BALL_LOG}）===="
rc=0
if [ "${BALL_SKIP}" = "1" ]; then
  echo "-- 已按 AUV_BALL_SKIP=1 跳过撞球任务（rc 视为 0）--"
else
  AUV_DOF_LOG="${LOG}" AUV_TASK_LOG="${BALL_LOG}" python3 main.py --task ball
  rc=$?
  echo "-- 撞球退出码 ${rc}（0=正常，3=后端不可用拒绝启动）--"
fi

echo "==== [5] 回退 ${REV_S} 秒 (surge=${REV_SURGE}) ===="
if [ "${rc}" = "0" ] || [ "${REV_ON_FAIL}" = "1" ]; then
  timed_dof "${REV_S}" "${REV_SURGE}" 0 0 0 backoff
else
  echo "!! 撞球未正常结束（rc=${rc}），按 AUV_REV_ON_FAIL=${REV_ON_FAIL} 跳过回退"
fi

echo "==== [6] 回退后前进 ${POST_FWD_S} 秒 (surge=${POST_FWD_SURGE}) ===="
timed_dof "${POST_FWD_S}" "${POST_FWD_SURGE}" 0 0 0 post_forward

echo "==== [7] (转角度 → 过门) × ${#TURN_ANGLES[@]} ===="
idx=0
for deg in "${TURN_ANGLES[@]}"; do
  idx=$((idx + 1))
  echo "---- [7.${idx}] 转 ${deg}°（$([ "${deg:0:1}" = "-" ] && echo 右转 || echo 左转)）----"
  turn_once "${deg}"
  gate_once "${idx}"
done

echo "==== 完成 ===="
