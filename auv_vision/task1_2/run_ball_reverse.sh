#!/bin/bash
# run_ball_reverse.sh — 撞球 → **直接倒车回来**（直线后退，不做记忆回放、不依赖视觉）
#
# 撞球 → **直线倒车** REV_S 秒返回（简单、不依赖轨迹/视觉；不做记忆回放）
#
# 流程：待机 W → (可选)下潜 → (可选)前进 → main.py --task ball → 直接倒车 → 停
#
# 用法：  ./run_ball_reverse.sh
# 可调：  AUV_WAIT_S(30)        待机秒数
#         AUV_DESCEND_S(3)      下潜秒数（0=不下潜）
#         AUV_FWD_S(3)          撞球前前进秒数（0=不前进）
#         AUV_FWD_SURGE(0.35)   撞球前前进速度
#         AUV_REV_S(5)          倒车秒数
#         AUV_REV_SURGE(-0.35)  倒车速度（负=后退）
#         AUV_DOF_LOG(/tmp/path.csv)  撞球轨迹（本脚本不用它返回，留着可选复用）
#         AUV_REV_ON_FAIL(0)    撞球任务非 0 退出时是否仍然倒车（1=倒）
#         AUV_SIM_MODE(0)       0=真发串口（默认） / 1=只打印（台架）
#
# 注意：默认 AUV_SIM_MODE=0，**真驱动**；台架只想看打印时用 AUV_SIM_MODE=1。
set -u
cd "$(dirname "$0")/.."

# 默认真驱动：避免"串口只打印、船不动"那种最难查的状态（要台架只打印就显式 AUV_SIM_MODE=1）
export AUV_SIM_MODE="${AUV_SIM_MODE:-0}"

WAIT_S="${AUV_WAIT_S:-30}"
DESCEND_S="${AUV_DESCEND_S:-3}"
FWD_S="${AUV_FWD_S:-3}"
FWD_SURGE="${AUV_FWD_SURGE:-0.35}"
REV_S="${AUV_REV_S:-5}"
REV_SURGE="${AUV_REV_SURGE:--0.35}"
LOG="${AUV_DOF_LOG:-/tmp/path.csv}"
REV_ON_FAIL="${AUV_REV_ON_FAIL:-0}"

# 清理上次残留：孤儿 main.py 会占住相机/串口，下次启动像"锁死"
if pkill -f "[p]ython3 main.py --task" 2>/dev/null; then
  echo "[clean] 已清理上一次残留的 main.py 进程"
  sleep 1
fi

echo "=================================================================="
echo " 撞球 → 直接倒车（待机 ${WAIT_S}s | 下潜 ${DESCEND_S}s | 前进 ${FWD_S}s | 倒车 ${REV_S}s @ ${REV_SURGE}）"
[ "${AUV_SIM_MODE}" = "1" ] && echo " ⚠️ AUV_SIM_MODE=1：只打印指令，不会真正驱动电机！"
echo "=================================================================="

# ---- 通用：定时发一组 DOF（结束时回中位；Ctrl-C 也会回中位）----------------
timed_dof() {   # $1=秒 $2=surge $3=sway $4=heave $5=yaw $6=标签
  python3 - "$1" "$2" "$3" "$4" "$5" "$6" <<'PYEOF'
from base.uart import UartController
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
    u.neutral()
    u.close()
print("[%s] 完成，已回中位" % label)
PYEOF
}

echo "==== [1/5] 待机 ${WAIT_S} 秒 ===="
sleep "${WAIT_S}"

echo "==== [2/5] 下潜 ${DESCEND_S} 秒 ===="
timed_dof "${DESCEND_S}" 0 0 -1 0 descend      # heave=-1 → 下潜

echo "==== [3/5] 前进 ${FWD_S} 秒 (surge=${FWD_SURGE}) ===="
timed_dof "${FWD_S}" "${FWD_SURGE}" 0 0 0 pre_forward

echo "==== [4/5] 撞球任务 main.py --task ball（轨迹 → ${LOG}）===="
AUV_DOF_LOG="${LOG}" python3 main.py --task ball
rc=$?
echo "main.py 退出码 ${rc}（0=正常，3=后端不可用拒绝启动）"

echo "==== [5/5] 直接倒车 ${REV_S} 秒 (surge=${REV_SURGE}) ===="
if [ "${rc}" = "0" ] || [ "${REV_ON_FAIL}" = "1" ]; then
  timed_dof "${REV_S}" "${REV_SURGE}" 0 0 0 backoff
else
  echo "!! 撞球任务未正常结束（退出码 ${rc}），按 AUV_REV_ON_FAIL=${REV_ON_FAIL} **跳过倒车**"
  echo "   （确认船在安全位置后，仍想倒车就: AUV_REV_ON_FAIL=1 ./run_ball_reverse.sh）"
fi

echo "==== 完成 ===="
