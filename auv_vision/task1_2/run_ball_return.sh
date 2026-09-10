#!/bin/bash
# run_ball_return.sh — 撞球编排（返回策略可切换）：
#   待机45s → 下潜3s → 前进3s → 撞球(记录轨迹) → 返回出发区 → 完成
#
# 返回策略（AUV_RETURN）：
#   memory   按记忆反向回放返回 + 回退3s（默认，不依赖视觉）
#   handover 策略B：后退1s → 原地yaw居中 → 换帧头(0xAA)交下位机接管 → 待命等标志
#            → 收权后仅sway平移居中 → 后退3s（视觉+握手；固件待确认）
#
# 用法：  ./run_ball_return.sh
# 可调：  AUV_RETURN（默认 memory）、AUV_TASK（默认 ball）、AUV_DOF_LOG（默认 /tmp/path.csv）、
#         AUV_AUTO_RETURN（memory策略: 1=无人值守自动确认回放，0=回车确认）、
#         AUV_WAIT_S（待机秒数，默认 45）、AUV_DESCEND_S（下潜秒数，默认 3）、
#         AUV_FWD_S（前进秒数，默认 3）、AUV_FWD_SURGE（前进速度，默认 0.35）、
#         AUV_REV_S（memory策略回退秒数，默认 3）、AUV_REV_SURGE（回退速度，默认 -0.35）
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
RETURN_MODE="${AUV_RETURN:-memory}"

echo "=================================================================="
echo " 撞球(${TASK}) + 返回出发区(策略: ${RETURN_MODE})（下潜 ${DESCEND_S}s，待机 ${WAIT_S}s）"
echo "   待机 ${WAIT_S}s → 下潜 ${DESCEND_S}s → 前进 ${FWD_S}s → main.py --task ${TASK} (记轨迹 ${LOG}) → 返回"
echo "=================================================================="

echo "==== [1/6] 待机 ${WAIT_S} 秒 ===="
sleep "${WAIT_S}"

echo "==== [2/6] 串口下潜 ${DESCEND_S} 秒 ===="
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

echo "==== [3/6] 串口前进 ${FWD_S} 秒 ===="
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

echo "==== [4/6] 执行 ${TASK} 任务并记录轨迹 => ${LOG} ===="
AUV_DOF_LOG="${LOG}" python3 main.py --task "${TASK}"
ret=$?
echo "main.py 退出码 ${ret}"

echo "==== [5/6] 返回出发区（策略: ${RETURN_MODE}） ===="
if [ "${RETURN_MODE}" = "handover" ]; then
  # 策略B：自带 后退1s + yaw居中 + 交接待命 + sway居中 + 后退3s
  python3 task1_2/return_handover.py --yes
  echo "return_handover 退出码 $?"
else
  if [ ! -s "${LOG}" ]; then
    echo "!! 轨迹文件 ${LOG} 为空/不存在，跳过按记忆返回"
    echo "   请先确认撞球过程正常记录轨迹（串口需为真机）。"
    exit 1
  fi
  if [ "${AUV_AUTO_RETURN:-1}" = "1" ]; then
    printf '\n' | python3 task1_2/return_by_memory.py --log "${LOG}"
  else
    python3 task1_2/return_by_memory.py --log "${LOG}"
  fi
  echo "return_by_memory 退出码 $?"

  echo "==== [6/6] 记忆返回后回退 ${REV_S} 秒 ===="
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
fi

echo "==== 完成 ===="
