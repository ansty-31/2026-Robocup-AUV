#!/bin/bash
# run_gate.sh — 过门单任务：待机 → (可选下潜/前进) → (转指定角度 → 过门)×N → 转指定角度
#
# 只跑 gate、不撞球；每次"过门"都是单门（pass_target=1，见 cfg/comm.yaml），
# 转角由脚本机械控制、按固定序列执行 —— "四个门当一个门四次过"。
#
# 用法：  ./task1_2/run_gate.sh
# 可调（环境变量，默认值在括号里）：
#   AUV_WAIT_S(10)        待机秒数
#   AUV_DESCEND_S(0)      下潜秒数（0=不下潜）
#   AUV_FWD_S(0)          下潜后前进秒数（0=不前进）
#   AUV_FWD_SURGE(0.35)   前进速度
#   AUV_TURN_ANGLES       转角序列（空格分隔，带符号：正=左转 / 负=右转）
#                         如 AUV_TURN_ANGLES="90 -45 90 30"；个数=过门次数
#   AUV_TURN_TIMEOUT(20)  转向等完成反馈超时（秒）
#   AUV_LOG_TAG(run)      日志前缀（区分轮次，别用 date）
#                         → log/<tag>gate_<N>.jsonl
set -u
cd "$(dirname "$0")/.."

export AUV_SIM_MODE="${AUV_SIM_MODE:-0}"

WAIT_S="${AUV_WAIT_S:-0}"
DESCEND_S="${AUV_DESCEND_S:-0}"
FWD_S="${AUV_FWD_S:-0}"
FWD_SURGE="${AUV_FWD_SURGE:-0.35}"
# ============ ★ 现场代填：每一次"转角度→过门"的转角（度）============
# 数组长度 = 过门次数；四扇门可以填四个不同的角度。
TURN_ANGLES=(
   0   # 第 1 次：第 1 扇门前转角（正=左转）
   0   # 第 2 次
   20   # 第 3 次
   -30   # 第 4 次
)
[ -n "${AUV_TURN_ANGLES:-}" ] && read -r -a TURN_ANGLES <<< "${AUV_TURN_ANGLES}"
# ⚠️ 角度**带符号**：正数=左转，负数=右转
TURN_TIMEOUT="${AUV_TURN_TIMEOUT:-20}"
TAG="${AUV_LOG_TAG:-run}"
GATE_LOG_PREFIX="${AUV_GATE_LOG_PREFIX:-log/${TAG}gate}"

# 清理上次残留
if pkill -f "[p]ython3 main.py --task" 2>/dev/null; then
  echo "[clean] 已清理上一次残留的 main.py 进程"
  sleep 1
fi

echo "=================================================================="
echo " 过门单任务：待机 ${WAIT_S}s → 下潜 ${DESCEND_S}s → 前进 ${FWD_S}s → (转角度→过门)×${#TURN_ANGLES[@]}"
echo "   过门 ${#TURN_ANGLES[@]} 次(单门 pass_target=1)"
echo "   转角序列（现场代填，正=左转/负=右转）：${TURN_ANGLES[*]}"
[ "${AUV_SIM_MODE}" = "1" ] && echo " ⚠️ AUV_SIM_MODE=1：只打印指令，不会真正驱动电机！"
echo " 日志：过门 ${GATE_LOG_PREFIX}_<N>.jsonl"
echo "=================================================================="

# ---- 权重自检（不加载模型，只看文件在不在；缺失立刻退出，别白下水）----
GATE_BIN="$(python3 - <<'PYEOF'
import sys
sys.path.insert(0, ".")
import base.cfg.settings as S
cfg = S.get("vision.model.task_models.gate", None) or {}
print(cfg.get("path", "") or "")
PYEOF
)"
if [ -z "${GATE_BIN}" ] || [ ! -f "${GATE_BIN}" ]; then
  echo "!! gate 权重不存在：${GATE_BIN:-<空>}"
  echo "!! 检查 cfg/vision.yaml → model.task_models.gate.path"
  exit 1
fi
echo "==== [0] 权重自检 ok：$(ls -l "${GATE_BIN}" | awk '{print $5" bytes"}') ===="

# ---- 定时 DOF ----
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
    u.stop_hard(verify=False)
    u.close()
print("[%s] 完成，已回中位（硬停）" % label)
PYEOF
}

# ---- 转角度（**带符号**：正=左转 / 负=右转）----
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

# ---- 过门一次（单门）----
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

echo "==== [4] (转角度 → 过门) × ${#TURN_ANGLES[@]} ===="
idx=0
for deg in "${TURN_ANGLES[@]}"; do
  idx=$((idx + 1))
  echo "---- [4.${idx}] 转 ${deg}°（$([ "${deg:0:1}" = "-" ] && echo 右转 || echo 左转)）----"
  turn_once "${deg}"
  gate_once "${idx}"
done

echo "==== 完成 ===="
