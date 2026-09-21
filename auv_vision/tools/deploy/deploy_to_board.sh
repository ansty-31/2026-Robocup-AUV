#!/bin/bash
# tools/deploy/deploy_to_board.sh — 按 tools/deploy/board_parity.md5 **批量**增量同步到板端 + 板端自检
#
#   bash tools/deploy/deploy_to_board.sh              # 全流程（同步+删除+编译+pytest+终检）
#   bash tools/deploy/deploy_to_board.sh --no-test    # 跳过板端 pytest（最快，秒级）
#   bash tools/deploy/deploy_to_board.sh --dry-run    # 只报告将上传/删除哪些文件
#
# 退出码：0 = 同步完成且板端自检通过；1 = 同步完成但**板端 pytest 未全绿**（末尾 ⚠️ 会说明）。
#   ⚠️ 板端"多出来的旧文件"（本地已删/改名的用例等）不会自动消失，全量 pytest 会因
#      import 已删除模块而收集报错 → 把这类文件补进下面的 DELETED 列表（会先备份再删）。
#
# SSH 包装器默认在 /home/ansty/RDKX5/（含密码的 askpass 不进仓库），可用环境变量改：
#   AUV_SSH / AUV_SCP / AUV_STREAM / AUV_ASKPASS
#
# 为什么快：① 板端 md5 **一次 SSH 批量取**；② 只把变化的文件打成一个 tar 传过去；
#           ③ 备份/删除各一次 SSH。之前是"一文件一次 SSH×2"，70+ 文件要几分钟。
# 每一步都打印累计耗时，慢在哪一眼能看到。
#
# 注：md5 只比对内容、**看不见文件权限**；而 tar 只上传"内容变化"的文件，
#     所以内容没变但权限丢了的（如 scp 覆盖掉 manual.sh 的 +x）永远不会被修。
#     这里在收尾步骤顺手对板端 *.sh 统一 chmod +x，杜绝 "./manual.sh: Permission denied"。
set -u
TOOLS_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL="$(cd "$TOOLS_DIR/../.." && pwd)"       # 工程根 = tools/deploy 的上两级（含 cfg/、main.py）
BOARD_HOST="${AUV_BOARD_HOST:-sunrise@192.168.137.10}"
BOARD="${AUV_BOARD_DIR:-/home/sunrise/AUV}"
HELP_DIR="${AUV_HELP_DIR:-/home/ansty/RDKX5}"
SSH="${AUV_SSH:-$HELP_DIR/.ssh_x5.sh}"
STREAM="${AUV_STREAM:-$HELP_DIR/.ssh_x5_stream.sh}"   # 同 SSH 但保留 stdin（tar/管道用）
SCP="${AUV_SCP:-$HELP_DIR/.scp_x5.sh}"
MANIFEST="$TOOLS_DIR/board_parity.md5"
BOARD_WAIT_S="${BOARD_WAIT_S:-60}"

NO_TEST=0; DRY=0
for a in "$@"; do
  [ "$a" = "--no-test" ] && NO_TEST=1
  [ "$a" = "--dry-run" ] && DRY=1
done

# 板端已废弃、需要删掉的文件（先备份到 bak/deploy_<stamp>/）
DELETED=(
  task1_2/ball_forward.py
  task1_2/run_ball_forward.sh
  tests/test_ball_forward.py
  # 2026-09-19：按角度原地转的实现迁到 common/turn_deg.py（脚本与 gate 正航向共用），
  # 板端旧的 task1_2/turn_deg.py 必须删掉，否则"两个同名脚本、行为不同"必然踩坑。
  task1_2/turn_deg.py
  # 工程根旧副本：这些工具已迁到 tools/ 或已删除，板端根目录的同名旧文件要删
  check_pipeline_identity.py
  archive_baks.sh
  # 已废弃：gate_pnp 参考实现的交付凭据（本地已删除，运行时不需要）
  work/gate_pnp-before-guidance.zip
  work/verification-before-guidance.json
  work/verify_and_package.py
  # v1.3/v1.4 分层版 gate/ **已确定不需要、已删除**：
  # 板端把分层目录删掉，回到扁平 gate/（扁平那 8 个文件在清单里，会自动上传）
  gate/README.md
  gate/vision/gate_decode.py
  gate/vision/gate_detector.py
  gate/vision/gate_frontend.py
  gate/vision/geometry.py
  gate/vision/perception.py
  gate/vision/__init__.py
  gate/data/cues.py
  gate/data/kpt_memory.py
  gate/data/quality.py
  gate/data/__init__.py
  gate/motion/phase_degrade.py
  gate/motion/phase_recover.py
  gate/motion/phases.py
  gate/motion/__init__.py
  # 分层版专用工具/测试/文档（板端旧副本一并删）
  tools/watch_quality.py
  tools/check_cue_geometry.py
  tools/analyze_quality_dump.py
  tests/test_gate_layers.py
  tests/test_gate_enhance.py
  tests/test_gate_cue_lead.py
  doc/算法说明-gate-v1.4-分层与质量分.md
  # 2026-09-21 分类重整：tests/tools 下移到子目录后，**旧扁平路径必须删掉**
  # （否则板端会同时存在新旧两份：pytest 收集到重名模块会 import-mismatch 报错）
  tools/analyze_heading.py
  tools/analyze_kpt_dump.py
  tools/analyze_pnp_center.py
  tools/analyze_task_log.py
  tools/feature_coverage.py
  tools/pnp_calib.py
  tools/check_gate_pose.py
  tools/check_kpt_decode.py
  tools/check_pipeline_identity.py
  tools/archive_baks.sh
  tests/test_base.py
  tests/test_common.py
  tests/test_ball.py
  tests/test_gate_vision.py
  tests/test_gate_flow.py
  tests/test_motion.py
  tests/test_pnp_calib.py
  # 板端遗留的**旧版/已合并**用例（2026-09-20 合并成 6 个文件时的旧名）
  # 板端这些更早的拆分文件不在清单里、会 import 已删除模块（gate.vision/gate.data/
  # common.ramp/build_frame_with_header…）→ 板端 pytest 收集即报错、把全量自检搞红。
  # 先备份到 bak/deploy_<stamp>/ 再删，保证"板端 == 清单"。
  tests/test_gate_dash.py
  tests/test_gate_decode.py
  tests/test_preprocess_rt.py
  tests/test_heading_align.py
  tests/test_turn_deg.py
  tests/test_gate.py
  tests/test_ball_hit.py
  tests/test_ball_port.py
  tests/test_ball_search.py
  tests/test_detector.py
  # ⚠️ 反面教材（别再犯）：`tests/test_gate_flow.py` 曾经既是"被删的分层版用例"、又是
  #    "活文件"的同名路径 —— 那种情况下把它列进 DELETED 会把刚上传的文件再删掉。
  #    2026-09-21 分类重整后活文件是 `tests/tasks/test_gate_flow.py`，扁平路径 `tests/test_gate_flow.py`
  #    才是要清的旧位置（见上面的"分类重整"段）；两者同名不同路径，**加 DELETED 时看清层级**。
  tests/test_gate_geometry.py
  tests/test_gate_kpt_memory.py
  tests/test_gate_standalone.py
  tests/test_logic.py
  tests/test_pid.py
  tests/test_preprocess.py
  tests/test_ramp.py
  tests/test_return_by_memory.py
  tests/test_return_handover.py
  tests/test_settings.py
  tests/test_uart.py
)

T0=$(date +%s)
step() { echo "[$(printf '%4ds' $(( $(date +%s) - T0 )))] $*"; }
TMPD=$(mktemp -d); trap 'rm -rf "$TMPD"' EXIT
CHANGED="$TMPD/changed"; ALL="$TMPD/all"
STAMP=$(date +%m%d_%H%M%S)

# ---- 0) helper 自检（stream 包装器缺失则就地生成，内容与 SSH 包装器同，只是不加 -n）----
if [ ! -x "$SSH" ]; then
  cat <<'EOS'
[deploy] ✗ 缺少 SSH 包装器（含密码的 askpass 不进仓库）。一次性建好：
  cat > /home/ansty/RDKX5/.tmp_askpass.sh <<'EOF'
  #!/bin/bash
  echo 'sunrise'          # 板子密码（chmod 600，别提交）
  EOF
  chmod 600 /home/ansty/RDKX5/.tmp_askpass.sh
  cat > "$AUV_HELP_DIR/.ssh_x5.sh" <<'EOF'
  #!/bin/bash
  export SSH_ASKPASS=/home/ansty/RDKX5/.tmp_askpass.sh
  export SSH_ASKPASS_REQUIRE=force
  export DISPLAY=:0
  exec setsid -w ssh -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=8 -o ServerAliveInterval=5 sunrise@192.168.137.10 "$@"
  EOF
  chmod +x /home/ansty/RDKX5/.ssh_x5.sh
（scp 版同理，把 ssh 换成 scp 并去掉 -n；STREAM 版**不要** -n，本脚本会按需自动生成）
EOS
  exit 2
fi
if [ ! -x "$STREAM" ]; then
  sed 's/exec setsid -w ssh -n /exec setsid -w ssh /' "$SSH" > "$STREAM"
  chmod +x "$STREAM"
  step "已生成 $STREAM（tar 管道需要 stdin）"
fi

# ---- 1) 探板子 ----
t=0
until timeout 12 "$SSH" "echo BOARD-UP" </dev/null 2>/dev/null | grep -q BOARD-UP; do
  t=$((t+10)); [ "$t" -ge "$BOARD_WAIT_S" ] && { echo "[deploy] ✗ 板子未接入（等了 ${BOARD_WAIT_S}s）；本地已就绪，稍后重跑"; exit 1; }
  sleep 10
done
step "板子在线"

# ---- 2) 一次 SSH 批量取板端 md5，本地算差异 ----
awk '!/^[[:space:]]*#/ && NF>=2 { v=0; p=1; if ($1 ~ /^\[/) { v=1; p=2 } printf "%s\t%s\t%d\n", $p, $(p+1), v }' "$MANIFEST" > "$ALL"
{
  echo "cd $BOARD 2>/dev/null || exit 1"
  printf 'md5sum'
  while IFS=$'\t' read -r _m f _v; do [ -f "$LOCAL/$f" ] && printf " '%s'" "$f"; done < "$ALL"
  echo
} > "$TMPD/rcmd"
timeout 60 "$SSH" "$(cat "$TMPD/rcmd")" </dev/null 2>/dev/null | grep -E '^[0-9a-f]{32} ' > "$TMPD/remote"
declare -A RM=()
while read -r m p; do RM["$p"]="$m"; done < "$TMPD/remote"

n_same=0; n_new=0; n_diff=0; n_miss_local=0
: > "$CHANGED"
while IFS=$'\t' read -r _m f _v; do
  [ -f "$LOCAL/$f" ] || { n_miss_local=$((n_miss_local+1)); continue; }
  lm=$(md5sum "$LOCAL/$f" | cut -d' ' -f1)
  r="${RM[$f]:-}"
  if [ "$r" = "$lm" ]; then n_same=$((n_same+1)); continue; fi
  if [ -z "$r" ]; then n_new=$((n_new+1)); echo "[new ] $f"; else n_diff=$((n_diff+1)); echo "[diff] $f"; fi
  echo "$f" >> "$CHANGED"
done < "$ALL"
step "差异：上传 $(wc -l < "$CHANGED") 个（新 $n_new / 改 $n_diff）· 已一致 $n_same · 本地缺 $n_miss_local"

if [ "$DRY" = "1" ]; then
  step "--dry-run：不动板端。将删除："
  for f in "${DELETED[@]}"; do [ -n "${RM[$f]:-}" ] && echo "        $f"; done
  exit 0
fi

# ---- 3) 备份 + 上传（各一次 SSH；上传只打包变化的文件）----
if [ -s "$CHANGED" ]; then
  { echo "cd $BOARD"; echo "mkdir -p bak/deploy_$STAMP";
    while read -r f; do
      [ -n "${RM[$f]:-}" ] && echo "mkdir -p bak/deploy_$STAMP/$(dirname "$f") && cp -p '$f' 'bak/deploy_$STAMP/$f'"
    done < "$CHANGED"
    echo "echo BAK-OK"; } > "$TMPD/bak.sh"
  timeout 120 "$STREAM" "bash -s" < "$TMPD/bak.sh" 2>/dev/null | grep -q BAK-OK \
    && step "板端已备份到 bak/deploy_$STAMP/" || step "⚠️ 备份步骤异常（继续上传）"
  tar czf - -C "$LOCAL" -T "$CHANGED" 2>/dev/null \
    | timeout 300 "$STREAM" "cd $BOARD && tar xzf - && echo UNPACK-OK" 2>/dev/null | grep -q UNPACK-OK \
    && step "上传完成（$(du -sh "$TMPD" >/dev/null; echo "$(wc -l < "$CHANGED") 个文件）")" || { echo "[deploy] ✗ 上传失败"; exit 1; }
  # 逐文件 md5 复核（还是只一次 SSH）
  { echo "cd $BOARD"; printf 'md5sum'; while read -r f; do printf " '%s'" "$f"; done < "$CHANGED"; echo; } > "$TMPD/vcmd"
  timeout 60 "$SSH" "$(cat "$TMPD/vcmd")" </dev/null 2>/dev/null | grep -E '^[0-9a-f]{32} ' > "$TMPD/vrf"
  bad=0
  while read -r f; do
    lm=$(md5sum "$LOCAL/$f" | cut -d' ' -f1)
    r=$(awk -v k="$f" '$2==k{print $1}' "$TMPD/vrf")
    [ "$r" = "$lm" ] || { echo "[FAIL] $f 本地=$lm 板端=${r:-缺失}"; bad=$((bad+1)); }
  done < "$CHANGED"
  [ "$bad" = "0" ] && step "上传校验：全部 md5 一致" || { echo "[deploy] ✗ $bad 个文件不一致"; exit 1; }
else
  step "无需上传（板端已是最新）"
fi

# ---- 4) 删除已废弃文件（一次 SSH，先备份）----
{ echo "cd $BOARD"; echo "mkdir -p bak/deploy_$STAMP";
  for f in "${DELETED[@]}"; do
    echo "[ -f '$f' ] && { mkdir -p bak/deploy_$STAMP/$(dirname "$f"); cp -p '$f' 'bak/deploy_$STAMP/$f'; rm -f '$f'; echo \"[removed+bak] $f\"; }"
  done
  echo "rmdir work 2>/dev/null"
  echo "rmdir gate/vision 2>/dev/null; rmdir gate/data 2>/dev/null; rmdir gate/motion 2>/dev/null"
  echo "find . -name '*.sh' -not -path './bak/*' -exec chmod +x {} + 2>/dev/null; \
        find . -name __pycache__ -type d -not -path './bak/*' -prune -exec rm -rf {} + 2>/dev/null; echo CLEAN-OK"; } > "$TMPD/del.sh"
timeout 120 "$STREAM" "bash -s" < "$TMPD/del.sh" 2>/dev/null | grep -vE "^\s*$" | sed 's/^/        /'
step "废弃文件处理完成"

# ---- 5) 板端自检 ----
{
  echo "cd $BOARD"
  echo "python3 -m py_compile main.py preview_detect.py base/settings.py base/camera.py base/uart.py base/telemetry.py common/detector.py common/PID.py common/preprocess.py manual/recorder.py manual/stream.py manual/udp_server.py task1_2/ball.py gate/__init__.py gate/gate_task.py gate/gate_detector.py gate/gate_decode.py gate/gate_frontend.py gate/geometry.py gate/kpt_memory.py gate/mock.py && echo COMPILE-OK"
  [ "$NO_TEST" = "1" ] || echo "echo '--- 全量 pytest ---'; timeout 600 python3 -m pytest tests/ -q > /tmp/auv_pytest.log 2>&1; tail -3 /tmp/auv_pytest.log; if grep -qE '[0-9]+ passed' /tmp/auv_pytest.log && ! grep -qE '[0-9]+ (failed|error)' /tmp/auv_pytest.log; then echo PYTEST-OK; else echo PYTEST-FAIL; fi"
  cat <<'PYEOF'
echo '--- 配置快照 ---'
python3 - <<'PY'
import sys; sys.path.insert(0,'.')
import base.settings as S
from gate.kpt_memory import ENV_ENABLE, kpt_mem_enabled
print('ball : timeout_ms=%s dash_ratio=%s stop_hold_s=%s' % (S.comm.ball.timeout_ms, S.comm.ball.dash_ratio, S.comm.ball.stop_hold_s))
print('gate : near_ratio=%s hold=%s reacquire=%s' % (S.comm.gate.coarse.near_ratio, S.comm.gate.hold.max_frames, dict(S.comm.gate.reacquire)))
print('motion(共用) : sway_kp=%s heave_kp=%s surge_fast=%s surge_slow=%s turn_kp=%s' % (
    S.comm.motion.pid_sway.kp, S.comm.motion.pid_heave.kp,
    S.comm.motion.surge_fast, S.comm.motion.surge_slow, S.comm.motion.turn_pid.kp))
print('kptm : enable=%s（%s=%s）alpha=%s beta=%s recall_conf=%s' % (
    kpt_mem_enabled(S.vision.gate.kpt_mem), ENV_ENABLE,
    __import__('os').environ.get(ENV_ENABLE, '<未设>'),
    S.vision.gate.kpt_mem.alpha, S.vision.gate.kpt_mem.beta,
    S.vision.gate.kpt_mem.recall_conf))
print('depth: guard=%s min=%sm stale=%sms（下位机 0xAA55 遥测；≤min 禁止上浮）' % (
    S.get('comm.depth_guard.enable', True), S.get('comm.depth_guard.min_depth_m', 0.3),
    S.get('comm.depth_guard.stale_ms', 500)))
print('model: gate=%s' % S.vision.model.task_models.gate.path.split('/')[-1])
print('SIM_MODE=%s  目标色=%s' % (S.SIM_MODE, S.vision.mission.target_color))
PY
PYEOF
} > "$TMPD/check.sh"
timeout 900 "$STREAM" "bash -s" < "$TMPD/check.sh" 2>&1 | grep -v "Warning: Permanently" | tee "$TMPD/selfcheck.log"
step "板端自检完成"

# ---- 6) 终检 + 刷新清单（一次 SSH 批量 md5）----
AUV_SSH="$SSH" bash "$TOOLS_DIR/check_board_parity.sh" --board --write 2>&1 | tail -4
step "完成：上传 ${n_new}/${n_diff} 新/改，已一致 ${n_same}"

# 板端 pytest **未全绿要吼出来**（以前是静默放过：红着也照样"完成"，很容易误判）
# 清单已按 md5 刷新（那是"板端字节 == 本地"，与测试无关）；这里只把退出码拉红。
if [ "$NO_TEST" != "1" ] && grep -q PYTEST-FAIL "$TMPD/selfcheck.log" 2>/dev/null; then
  echo "[deploy] ⚠️ 板端 pytest 未全绿（见上面 tail -3）：代码已同步，但**板端自检不算通过**。"
  echo "         常见原因：板端还留着旧用例（import 已删除模块）→ 加进本脚本 DELETED 列表后重跑。"
  exit 1
fi
