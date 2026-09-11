#!/bin/bash
# run_ball_forward.sh — 撞球“简化版”编排：待机 → 下潜 → 前进 → 简化撞球(搜索/居中/恒速前进, 累计前进10s停)
#
# 与 run_ball_return.sh 的区别：
#   - 跑的是简化任务 main.py --task ball_fwd（**不做命中识别**，速度恒定）；
#   - 任务在“累计前进时间”达到 forward_total_s（cfg/comm.yaml ball_forward.forward_total_s，默认10s）后自行停止；
#   - **没有返回出发点**步骤。
#
# 用法：  ./run_ball_forward.sh
# 可调：  AUV_TASK（默认 ball_fwd）、AUV_WAIT_S（待机秒数，默认30）、
#         AUV_DESCEND_S（下潜秒数，默认3）、AUV_FWD_S（前进秒数，默认3）、AUV_FWD_SURGE（前进速度，默认0.35）
set -u
cd "$(dirname "$0")/.."

TASK="${AUV_TASK:-ball_fwd}"
WAIT_S="${AUV_WAIT_S:-30}"
DESCEND_S="${AUV_DESCEND_S:-3}"
FWD_S="${AUV_FWD_S:-3}"
FWD_SURGE="${AUV_FWD_SURGE:-0.35}"

# 清理上次残留：孤儿 main.py 会占住相机/串口，导致下次启动像“锁死”
if pkill -f "python3 main.py --task" 2>/dev/null; then
  echo "[clean] 已清理上一次残留的 main.py 进程"
  sleep 1
fi

echo "=================================================================="
echo " 撞球简化版(${TASK})：搜索→居中→恒速前进，累计前进即停（待机 ${WAIT_S}s，下潜 ${DESCEND_S}s）"
echo "   待机 ${WAIT_S}s → 下潜 ${DESCEND_S}s → 前进 ${FWD_S}s → main.py --task ${TASK}"
echo "=================================================================="

echo "==== [1/4] 待机 ${WAIT_S} 秒 ===="
sleep "${WAIT_S}"

echo "==== [2/4] 串口下潜 ${DESCEND_S} 秒 ===="
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

echo "==== [3/4] 串口前进 ${FWD_S} 秒 ===="
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

echo "==== [4/4] 简化撞球任务（搜索/居中/恒速前进）===="
python3 main.py --task "${TASK}"
echo "main.py 退出码 $?"

echo "==== 完成 ===="
