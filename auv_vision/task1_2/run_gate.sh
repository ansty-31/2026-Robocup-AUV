#!/bin/bash
# run_gate.sh — 过门(gate)单任务下水编排：权重自检 → 待机 → (可选下潜/前进) → main.py --task gate
#
# 与 run_ball_forward.sh / run_ball_return.sh 的关系：这三个脚本都是**单任务**编排，
# 互不干扰。本脚本只跑 gate，不会加载 ball 权重（DetectorHub 惰性加载）。
#
# 关键点（为什么能单独测 gate）：
#   - main.py --task gate 只装配 gate 专用后端（vision.model.task_models.gate.path），
#     GateTask 走 hub.detect_list("gate")，用的就是 gate_kpt_*.bin 角点权重；
#   - 启动时若 gate 后端不可用会**直接拒绝启动**(退出码 3)，不会入水后空跑；
#   - 本脚本在待机前先做权重自检，权重缺失当场退出，省得白等/白下水。
#
# 用法：  ./run_gate.sh
# 可调：  AUV_WAIT_S（待机秒数，默认30）、AUV_DESCEND_S（下潜秒数，默认0=不下潜）、
#         AUV_FWD_S（下潜后前进秒数，默认0=不前进）、AUV_FWD_SURGE（前进速度，默认0.35）
#
# 下水前建议先单独验视觉（船不动）：
#   python3 preview_detect.py --gate-kpt --stream     # 门框 + 4 角点 + 置信度
set -u
cd "$(dirname "$0")/.."

WAIT_S="${AUV_WAIT_S:-30}"
DESCEND_S="${AUV_DESCEND_S:-0}"
FWD_S="${AUV_FWD_S:-0}"
FWD_SURGE="${AUV_FWD_SURGE:-0.35}"

echo "=================================================================="
echo " 过门任务(gate)单任务下水：待机 ${WAIT_S}s → 下潜 ${DESCEND_S}s → 前进 ${FWD_S}s"
echo "                           → main.py --task gate"
echo "=================================================================="

# ---- [0/5] 残留进程清理：孤儿 main.py 会占住相机/串口，导致下次启动像“锁死”
if pkill -f "python3 main.py --task" 2>/dev/null; then
  echo "[clean] 已清理上一次残留的 main.py 进程"
  sleep 1
fi

# ---- [1/5] 权重自检（不加载模型，只看文件在不在；缺失立刻退出，别白下水）
GATE_BIN="$(python3 - <<'PYEOF'
import sys
sys.path.insert(0, ".")
import base.settings as S
cfg = S.get("vision.model.task_models.gate", None) or {}
print(cfg.get("path", "") or "")
PYEOF
)"
echo "==== [1/5] 权重自检：${GATE_BIN:-<未配置>} ===="
if [ -z "${GATE_BIN}" ] || [ ! -f "${GATE_BIN}" ]; then
  echo "!! gate 权重不存在：${GATE_BIN:-<空>}"
  echo "!! 检查 cfg/vision.yaml → model.task_models.gate.path"
  exit 1
fi
echo "[ok] $(ls -l "${GATE_BIN}" | awk '{print $5" bytes"}')"

# ---- [2/5] 待机（放船/对门）
echo "==== [2/5] 待机 ${WAIT_S} 秒 ===="
sleep "${WAIT_S}"

# ---- [3/5] 可选下潜
if [ "${DESCEND_S}" != "0" ]; then
  echo "==== [3/5] 串口下潜 ${DESCEND_S} 秒 ===="
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
else
  echo "==== [3/5] 跳下潜（AUV_DESCEND_S=0）===="
fi

# ---- [4/5] 可选前进（离壁/进入门前方位）
if [ "${FWD_S}" != "0" ]; then
  echo "==== [4/5] 串口前进 ${FWD_S} 秒 (surge=${FWD_SURGE}) ===="
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
else
  echo "==== [4/5] 跳前进（AUV_FWD_S=0）===="
fi

# ---- [5/5] gate 任务本体
echo "==== [5/5] 过门任务 main.py --task gate ===="
echo "     (板端看画面: export DISPLAY=:0 后运行; 推流: 默认 vision.stream.enable)"
python3 main.py --task gate
rc=$?
echo "main.py 退出码 ${rc}（0=正常结束，3=gate 后端不可用拒绝启动，130=Ctrl-C）"

echo "==== 完成 ===="
exit "${rc}"
