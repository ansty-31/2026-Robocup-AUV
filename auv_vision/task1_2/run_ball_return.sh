#!/bin/bash
# run_ball_return.sh — 当前唯一编排版（原备份版取代主流程并更名）：
#   待机45s → 下潜3s → 前进3s → 撞球(记录轨迹) → 按记忆反向回放返回 → 回退3s
#
# 用法：  ./run_ball_return.sh
# 可调：  AUV_TASK（默认 ball）、AUV_DOF_LOG（轨迹路径，默认 /tmp/path.csv）、
#         AUV_AUTO_RETURN（默认 1 = 无人值守自动确认回放；0 = 回放前回车确认）、
#         AUV_WAIT_S（待机秒数，默认 45）、AUV_DESCEND_S（下潜秒数，默认 3）、
#         AUV_FWD_S（前进秒数，默认 3）、AUV_FWD_SURGE（前进速度，默认 0.35）、
#         AUV_REV_S（回退秒数，默认 3）、AUV_REV_SURGE（回退速度，默认 -0.35）
set -u
cd "$(dirname "$0")/.."

TASK="${AUV_TASK:-ball}"
LOG="${AUV_DOF_LOG:-/tmp/path.csv}"
WAIT_S="${AUV_WAIT_S:-45}"
DESCEND_S="${AUV_DESCEND_S:-3}"
FWD_S="${AUV_FWD_S:-3}"
FWD_SURGE="${AUV_FWD_SURGE:-0.35}"
REV_S="${AUV_REV_S:-3}"
REV_SURGE="${AUV_REV_SURGE:--0.35}"

echo "=================================================================="
echo " 撞球(${TASK}) + 按记忆返回出发点 + 回退 ${REV_S}s（下潜 ${DESCEND_S}s，待机 ${WAIT_S}s）"
echo "   待机 ${WAIT_S}s → 下潜 ${DESCEND_S}s → 前进 ${FWD_S}s → main.py --task ${TASK} (记轨迹 ${LOG}) → 反向回放 → 回退 ${REV_S}s"
echo "=================================================================="

echo "==== [1/7] 待机 ${WAIT_S} 秒 ===="
sleep "${WAIT_S}"

echo "==== [2/7] 串口下潜 ${DESCEND_S} 秒 ===="
python3 - "$DESCEND_S" <<'PYEOF'
from base.uart import UartController
import sys, time
dur = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
u = UartController()
if u.sim:
    print("!! 串口未打开(sim)，下潜仅模拟")
print("[descend] 串口发送下潜指令 %.1fs ..." % dur)
t0 = time.time()
while time.time() - t0 < dur:
    u.send_dof(0.0, 0.0, -1.0, 0.0)   # heave=-1 → 下潜
    time.sleep(0.05)
u.neutral()
u.close()
print("[descend] 完成，已回中位")
PYEOF

echo "==== [3/7] 串口前进 ${FWD_S} 秒 ===="
python3 - "$FWD_S" "$FWD_SURGE" <<'PYEOF'
from base.uart import UartController
import sys, time
dur = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
surge = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35
u = UartController()
if u.sim:
    print("!! 串口未打开(sim)，前进仅模拟")
print("[fwd] 串口发送前进指令 %.1fs (surge=%.2f) ..." % (dur, surge))
t0 = time.time()
while time.time() - t0 < dur:
    u.send_dof(surge, 0.0, 0.0, 0.0)   # surge 前进
    time.sleep(0.05)
u.neutral()
u.close()
print("[fwd] 完成，已回中位")
PYEOF

echo "==== [4/7] 执行 ${TASK} 任务并记录轨迹 => ${LOG} ===="
AUV_DOF_LOG="${LOG}" python3 main.py --task "${TASK}"
ret=$?
echo "main.py 退出码 ${ret}"

if [ ! -s "${LOG}" ]; then
  echo "!! 轨迹文件 ${LOG} 为空/不存在，跳过返回回放"
  echo "   请先确认撞球过程正常记录轨迹（串口需为真机）。"
  exit 1
fi

echo "==== [5/7] 按记忆反向回放返回出发点 ===="
if [ "${AUV_AUTO_RETURN:-1}" = "1" ]; then
  printf '\n' | python3 task1_2/return_by_memory.py --log "${LOG}"
else
  python3 task1_2/return_by_memory.py --log "${LOG}"
fi
ret=$?
echo "return_by_memory 退出码 ${ret}"

echo "==== [6/7] 记忆返回后回退 ${REV_S} 秒 ===="
python3 - "$REV_S" "$REV_SURGE" <<'PYEOF'
from base.uart import UartController
import sys, time
dur = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
surge = float(sys.argv[2]) if len(sys.argv) > 2 else -0.35
u = UartController()
if u.sim:
    print("!! 串口未打开(sim)，回退仅模拟")
print("[rev] 串口发送回退指令 %.1fs (surge=%.2f) ..." % (dur, surge))
t0 = time.time()
while time.time() - t0 < dur:
    u.send_dof(surge, 0.0, 0.0, 0.0)   # surge 负 = 后退
    time.sleep(0.05)
u.neutral()
u.close()
print("[rev] 完成，已回中位")
PYEOF

echo "==== [7/7] 完成 ===="
