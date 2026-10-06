#!/bin/bash
# run_grab.sh — **任务三 单任务（夹取小球）**：只跑 `handling` 的 grab 模式，不撞球、不过门。
#
#   待机 → (可选下潜) → 夹取一次（main.py 内部的相位机自己走完）
#   **逐帧日志默认就开**（AUV_TASK_LOG → log/run_grab_<时间戳>.jsonl），收尾打印路径与判读命令。
#   感知**全程下视**（`cfg/vision.yaml camera.down`，USB）⇒ 现场先确认下视没被别的程序占着。
#
# 用法（工程根）：
#   bash task/run_grab.sh                      # 真机下水
#   AUV_SIM_MODE=1 bash task/run_grab.sh       # 只打印、不驱动电机（台架/PC 预演）
#   AUV_GRAB_DESCEND_S=3 bash task/run_grab.sh # 先下潜 3s 再开始夹取
#
# ⚠️ 未标定前本脚本会被 main.py 的机械闸拒绝启动（`comm.grab.calibrated: false`），
#    那是**预期行为**：占位值没标完不下水。标定顺序见 handling/README.md。
set -u
cd "$(dirname "$0")/.."

export AUV_SIM_MODE="${AUV_SIM_MODE:-0}"
WAIT_S="${AUV_WAIT_S:-0}"
DESCEND_S="${AUV_GRAB_DESCEND_S:-0}"
TAG="${AUV_LOG_TAG:-run}"
mkdir -p log
# 逐帧日志（main.py 的 AUV_TASK_LOG）：带时间戳，一次下水一个文件，不会被下一趟覆盖
#   命名：一次下水一个文件；**防撞**（板端 RTC 不可靠时会撞名 → 覆盖上一趟的日志）
if [ -n "${AUV_GRAB_LOG:-}" ]; then
  GRAB_LOG="${AUV_GRAB_LOG}"
else
  _base="log/${TAG}grab_$(date +%m%d_%H%M%S)"
  GRAB_LOG="${_base}.jsonl"; _n=1
  while [ -e "${GRAB_LOG}" ]; do _n=$((_n+1)); GRAB_LOG="${_base}_${_n}.jsonl"; done
fi
: > "${GRAB_LOG}"        # 先建空文件：即使被机械闸拒启动，也能一眼看出"跑过、但一行都没写"

# 下视是独占设备：先把上次残留的 main.py 清掉，否则相机被占 → 静默回退 sim（追着假球开船）
if pkill -f "[p]ython3 main.py --task grab" 2>/dev/null; then
  echo "[clean] 已清理上一次残留的 main.py --task grab 进程"
  sleep 1
fi

echo "=================================================================="
echo " 任务三 夹取小球（单任务）"
echo "   待机 ${WAIT_S}s → 下潜 ${DESCEND_S}s → 夹取一次"
echo "   感知：下视（$(python3 - <<'PYEOF'
import sys; sys.path.insert(0, ".")
import base.cfg.settings as S
c = S.vision.camera.down
print("%s %s %sx%s@%s" % (c.type, getattr(c, "device", "-"), c.width, c.height, c.fps))
PYEOF
)）"
echo "   逐帧日志：${GRAB_LOG}"
echo "   无帧兜底：连续 ${AUV_NOFRAME_ESTOP:-150} 帧读不到帧 → E-STOP（AUV_NOFRAME_ESTOP 可调）"
[ "${AUV_SIM_MODE}" = "1" ] && echo " ⚠️ AUV_SIM_MODE=1：只打印指令，不会真正驱动电机！"

# ---- 标定闸自检（不建串口）：没标定就别白下水 ----
python3 - <<'PYEOF'
import sys
sys.path.insert(0, ".")
import base.cfg.settings as S
cal = bool(S.get("comm.grab.calibrated", False))
flr = S.get("comm.grab.depth_floor_m", None)
mode = str(S.get("vision.grab.detect.mode", "cv"))
print("   comm.grab.calibrated = %s（false ⇒ main.py 会拒绝启动）" % cal)
print("   任务级限深下限 = %s m（全局 min_depth_m = %s m，**不动**）"
      % (flr, S.get("comm.depth_guard.min_depth_m", None)))
print("   感知后端 mode = %s（cv = 纯 CV 红球，不走 BPU）" % mode)
act = S.get("vision.grab.targets.active", None)
print("   目标色 active = %s（现在红球；比赛粉/黄见 vision.yaml 的 targets）" % act)
PYEOF
echo "=================================================================="

echo "==== [1] 待机 ${WAIT_S} 秒 ===="
sleep "${WAIT_S}"

if [ "${DESCEND_S}" != "0" ]; then
  echo "==== [2] 先下潜 ${DESCEND_S} 秒（入水后到工作深度）===="
  python3 - "${DESCEND_S}" <<'PYEOF'
from base.hw.uart import UartController
import sys, time
dur = float(sys.argv[1])
u = UartController()
if u.sim:
    print("!! 串口 SIM：只打印")
print("[descend] 下潜 %.1fs ..." % dur)
try:
    t0 = time.time()
    while time.time() - t0 < dur:
        u.send_dof(0.0, 0.0, -1.0, 0.0)
        time.sleep(0.05)
finally:
    u.stop_hard(verify=False)
    u.close()
print("[descend] 完成，已回中位")
PYEOF
fi

echo "==== [3] 夹取一次（相位机自动走完）→ ${GRAB_LOG} ===="
AUV_TASK_LOG="${GRAB_LOG}" python3 main.py --task grab
RC=$?
echo "-- 夹取退出码 ${RC}（0=正常结束 / 2=任务名非法 / 3=自检拒绝启动(未标定/后端不可用)）--"
if [ "${RC}" = "3" ]; then
  echo "-- 若是'未标定'：先把 comm.grab.calibrated 翻 true（标定顺序见 handling/README.md）--"
fi
echo "==== [4] 日志 ===="
LINES=$(wc -l < "${GRAB_LOG}" 2>/dev/null || echo 0)
echo "   逐帧日志：${GRAB_LOG}（${LINES} 行）"
if [ "${LINES}" = "0" ]; then
  echo "   ⚠️ 一行都没有 ⇒ 任务状态机没跑起来（多半是自检拒绝启动，或相机没出帧）"
fi
echo "   判读（相位/出口/通道能动力/哪些功能没被用到）："
echo "     python3 tools/analyze/log/analyze_task_log.py ${GRAB_LOG}"
echo "     python3 tools/analyze/log/feature_coverage.py ${GRAB_LOG}"
echo "==== 完成 ===="
