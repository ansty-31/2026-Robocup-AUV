#!/bin/bash
# run_ball_reverse.sh — 撞球 → 直接倒车回来 → 前进 2s → 左转 90° → **接过门任务**
#
# 撞球 → **直线倒车** REV_S 秒返回（简单、不依赖轨迹/视觉；不做记忆回放）
#      → 前进 POST_FWD_S 秒（重新摆位）
#      → 左转 TURN_DEG 度（**下位机执行相对角 + 等完成反馈**，见 common/motion/turn_deg.py）
#      → main.py --task gate（过门）
#
# 流程：待机 → (可选)下潜 → (可选)前进 → main.py --task ball → 直接倒车
#       → 前进 2s → 左转 90° → main.py --task gate
#
# 用法：  ./task1_2/run_ball_reverse.sh
# 可调：  AUV_WAIT_S(30)        待机秒数
#         AUV_DESCEND_S(3)      下潜秒数（0=不下潜）
#         AUV_FWD_S(3)          撞球前前进秒数（0=不前进）
#         AUV_FWD_SURGE(0.35)   撞球前前进速度
#         AUV_REV_S(5)          倒车秒数
#         AUV_REV_SURGE(-0.5)   倒车速度（负=后退）
#         AUV_LOG_TAG(run)        日志文件名前缀（**用前缀区分轮次，别用 date**：板子时钟不准）
#                                 → log/<tag>ball.jsonl / log/<tag>gate.jsonl / /tmp/<tag>path.csv
#         AUV_BALL_LOG / AUV_GATE_LOG / AUV_DOF_LOG   单独覆盖这三个路径
#                                 逐帧日志（过门那份）用 tools/analyze/log/analyze_task_log.py 判读
#         AUV_REV_ON_FAIL(0)    撞球任务非 0 退出时是否仍然倒车（1=倒）
#         AUV_POST_FWD_S(2)       倒车后**前进秒数**（0=跳过）
#         AUV_POST_FWD_SURGE(0.35) 该前进的速度
#         AUV_TURN_DEG(90)        转角大小（度）
#         AUV_TURN_LEFT(1)        1=左转（默认） 0=右转
set -u
cd "$(dirname "$0")/.."

# 默认真驱动：避免"串口只打印、船不动"那种最难查的状态（要台架只打印就显式 AUV_SIM_MODE=1）
export AUV_SIM_MODE="${AUV_SIM_MODE:-0}"

WAIT_S="${AUV_WAIT_S:-10}"
DESCEND_S="${AUV_DESCEND_S:-0}"
FWD_S="${AUV_FWD_S:-0}"
FWD_SURGE="${AUV_FWD_SURGE:-0.35}"
REV_S="${AUV_REV_S:-5}"
REV_SURGE="${AUV_REV_SURGE:--0.5}"
TAG="${AUV_LOG_TAG:-run}"
BALL_LOG="${AUV_BALL_LOG:-log/${TAG}ball.jsonl}"
GATE_LOG="${AUV_GATE_LOG:-log/${TAG}gate.jsonl}"
LOG="${AUV_DOF_LOG:-/tmp/${TAG}path.csv}"
REV_ON_FAIL="${AUV_REV_ON_FAIL:-0}"
# 倒车后三步（前进 / 转角 / 过门）
POST_FWD_S="${AUV_POST_FWD_S:-2}"
POST_FWD_SURGE="${AUV_POST_FWD_SURGE:-0.35}"
TURN_DEG="${AUV_TURN_DEG:-90}"
TURN_LEFT="${AUV_TURN_LEFT:-1}"
TURN_YAW=""
TURN_KP=""
TURN_KD=""
TURN_NORM=""
TURN_TIMEOUT="${AUV_TURN_TIMEOUT:-20}"
GATE_AFTER="${AUV_GATE_AFTER:-1}"
BALL_SKIP="${AUV_BALL_SKIP:-0}"

# 清理上次残留：孤儿 main.py 会占住相机/串口，下次启动像"锁死"
if pkill -f "[p]ython3 main.py --task" 2>/dev/null; then
  echo "[clean] 已清理上一次残留的 main.py 进程"
  sleep 1
fi

TURN_DIR_TXT="左转"; [ "${TURN_LEFT}" = "0" ] && TURN_DIR_TXT="右转"
echo "=================================================================="
echo " 撞球 → 倒车 → 前进 → ${TURN_DIR_TXT} → 过门"
echo "   待机 ${WAIT_S}s | 下潜 ${DESCEND_S}s | 前进 ${FWD_S}s | 倒车 ${REV_S}s @ ${REV_SURGE}"
echo "   倒车后：前进 ${POST_FWD_S}s @ ${POST_FWD_SURGE} → ${TURN_DIR_TXT} ${TURN_DEG}° (遥测闭环)"
echo "   最后：$([ "${GATE_AFTER}" = "1" ] && echo "接过门任务 main.py --task gate" || echo "不过门(AUV_GATE_AFTER=0)")"
[ "${AUV_SIM_MODE}" = "1" ] && echo " ⚠️ AUV_SIM_MODE=1：只打印指令，不会真正驱动电机！"
echo " 日志：撞球 ${BALL_LOG} ｜ 过门 ${GATE_LOG} ｜ 轨迹 ${LOG}"
echo "       （前缀用 AUV_LOG_TAG=xxx 区分轮次；板子时钟不准，别用 date 命名）"
echo "=================================================================="

# ---- 通用：定时发一组 DOF（结束时回中位；Ctrl-C 也会回中位）----------------
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
    # ⚠️ **必须用 stop_hard**：neutral() 只发一帧，而 _ramp_step 是字节级平滑
    #    （force 只绕过心跳节流、不绕过 ramp）⇒ 单帧 neutral 时轴字节还在半路；
    #    下位机没有"无帧超时停车"，一关串口船就锁在那个还在动的值上。
    u.stop_hard(verify=False)        # 连发中性帧走完 ramp（yaw 验证交给转向那一步）
    u.close()
print("[%s] 完成，已回中位（硬停）" % label)
PYEOF
}

echo "==== [1/8] 待机 ${WAIT_S} 秒 ===="
sleep "${WAIT_S}"

echo "==== [2/8] 下潜 ${DESCEND_S} 秒 ===="
timed_dof "${DESCEND_S}" 0 0 -1 0 descend      # heave=-1 → 下潜

echo "==== [3/8] 前进 ${FWD_S} 秒 (surge=${FWD_SURGE}) ===="
timed_dof "${FWD_S}" "${FWD_SURGE}" 0 0 0 pre_forward

echo "==== [4/8] 撞球任务 main.py --task ball（轨迹 → ${LOG}，逐帧 → ${BALL_LOG}）===="
if [ "${BALL_SKIP}" = "1" ]; then
  echo "-- 已按 AUV_BALL_SKIP=1 跳过撞球任务（rc 视为 0）--"
  rc=0
else
  AUV_DOF_LOG="${LOG}" AUV_TASK_LOG="${BALL_LOG}" python3 main.py --task ball
  rc=$?
  echo "main.py 退出码 ${rc}（0=正常，3=后端不可用拒绝启动）"
fi

echo "==== [5/8] 直接倒车 ${REV_S} 秒 (surge=${REV_SURGE}) ===="
if [ "${rc}" = "0" ] || [ "${REV_ON_FAIL}" = "1" ]; then
  timed_dof "${REV_S}" "${REV_SURGE}" 0 0 0 backoff
else
  echo "!! 撞球任务未正常结束（退出码 ${rc}），按 AUV_REV_ON_FAIL=${REV_ON_FAIL} **跳过倒车**"
  echo "   （确认船在安全位置后，仍想倒车就: AUV_REV_ON_FAIL=1 ./task1_2/run_ball_reverse.sh）"
fi

echo "==== [6/8] 倒车后前进 ${POST_FWD_S} 秒 (surge=${POST_FWD_SURGE}) ===="
timed_dof "${POST_FWD_S}" "${POST_FWD_SURGE}" 0 0 0 post_forward

  echo "==== [7/8] ${TURN_DIR_TXT} ${TURN_DEG}°（手动给角度；下位机执行 + 等完成反馈）===="
if [ "${TURN_DEG}" = "0" ]; then
  echo "-- 角度为 0，跳过转向 --"
else
  turn_args=(--deg "${TURN_DEG}" --timeout "${TURN_TIMEOUT}")
  [ "${TURN_LEFT}" = "0" ] && turn_args+=(--dir right) || turn_args+=(--dir left)
  python3 common/motion/turn_deg.py "${turn_args[@]}"
  trc=$?
  case "${trc}" in
    0) echo "-- 转向到达（退出码 0）--" ;;
    3) echo "!! 转向**没等到完成反馈**（退出码 3）：下位机没回 turn_done"
       echo "   查 15B 上行帧（byte2=turn_id / byte13=turn_done）是否在回、编号是否对得上；"
       echo "   若船其实转了但没回报 ⇒ 先查下位机固件这一侧（上位机已不再闭环）"
         echo "   实测角度很小(<5°) = 舵效不足；一直朝一个方向转到停不下来 = 查 dof_map.yaw.sign / 接线" ;;
    5) echo "!! **没拿到遥测 yaw → 已拒绝转向**（退出码 5）。查："
       echo "   控制台有没有 [UART←] depth=… yaw=… 这类上行行（没有=下位机没回传/串口不对）" ;;
    6) echo "!! （旧模型遗留的退出码 6；新模型不会出现：方向自证已随上位机闭环删除）"
       echo "   comm.telemetry.yaw_sign / 推进器接线（转向已停，不会一路转到底）" ;;
    *) echo "!! 转向异常退出码 ${trc}" ;;
  esac
fi

echo "==== [8/8] 接过门任务 main.py --task gate（日志 → ${GATE_LOG}）===="
if [ "${GATE_AFTER}" = "1" ]; then
  AUV_TASK_LOG="${GATE_LOG}" python3 main.py --task gate
  grc=$?
  echo "过门任务退出码 ${grc}（0=正常结束，3=后端不可用拒绝启动）"
  echo "逐帧日志：${GATE_LOG}（拉回来用 python3 tools/analyze/log/analyze_task_log.py 判读）"
else
  echo "-- 已按 AUV_GATE_AFTER=0 跳过过门任务 --"
fi

echo "==== 完成 ===="
