#!/bin/bash
# tools/deploy/deploy_to_board.sh — 按 tools/deploy/board_parity.md5 **批量**增量同步到板端 + 板端自检
#
#   bash tools/deploy/deploy_to_board.sh              # 全流程（同步+删除+编译+pytest+终检）
set -u
TOOLS_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL="$(cd "$TOOLS_DIR/../.." && pwd)"       # 工程根 = tools/deploy 的上两级（含 cfg/、main.py）
BOARD_HOST="${AUV_BOARD_HOST:-sunrise@192.168.137.10}"
BOARD="${AUV_BOARD_DIR:-/home/sunrise/Desktop/AUV_New}"
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
  # 2026-10-06 测试目录 tests/tasks/{grab,place}/ → tests/tasks/handling/
  tests/tasks/grab/__init__.py
  tests/tasks/grab/test_cv_ball.py
  tests/tasks/grab/test_drop.py
  tests/tasks/grab/test_grab_flow.py
  tests/tasks/grab/test_grab_motion.py
  tests/tasks/grab/test_grab_targets.py
  tests/tasks/grab/test_handling_modes.py
  tests/tasks/place/__init__.py
  tests/tasks/place/test_place_flow.py
  # 2026-10-06 `grab/`+`place/` 合并为 `handling/`：板端旧包要清
  grab/README.md
  grab/__init__.py
  grab/percept/__init__.py
  grab/percept/ball_tracker.py
  grab/percept/cv_ball.py
  grab/percept/grab_detector.py
  # 2026-10-02 `task1_2/` → `task/` 改名：板端旧目录要清
  task1_2/__init__.py
  task1_2/run_ball_reverse.sh
  task1_2/run_gate.sh
  # 2026-09-30 gate_task 上移 + hdg/heading_align 合并：旧路径在板端要删
  gate/motion/gate_task.py
  gate/motion/heading_align.py
  task1_2/ball_forward.py
  task1_2/run_ball_forward.sh
  # 2026-10-06 结构迁移：task1_2/ → task/（ball.py 旧路径要删，否则板端同时有两份 ball）
  task1_2/ball.py
  tests/test_ball_forward.py
  # 板端旧的 task1_2/turn_deg.py 必须删掉，否则"两个同名脚本、行为不同"必然踩坑。
#   （细节与实测见 doc/注释历史.md）
  common/turn_log.py
  task1_2/turn_deg.py
  # 工程根旧副本：这些工具已迁到 tools/ 或已删除，板端根目录的同名旧文件要删
  check_pipeline_identity.py
  archive_baks.sh
  # 已废弃：gate_pnp 参考实现的交付凭据（本地已删除，运行时不需要）
  work/gate_pnp-before-guidance.zip
  work/verification-before-guidance.json
  work/verify_and_package.py
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
  tools/watch_quality.py
  tools/check_cue_geometry.py
  tools/analyze_quality_dump.py
  tests/test_gate_layers.py
  tests/test_gate_enhance.py
  tests/test_gate_cue_lead.py
  doc/算法说明-gate-v1.4-分层与质量分.md
  # 分类重整：tests/tools 下移到子目录后，**旧扁平路径必须删掉**（否则板端新旧两份并存 → pytest 收集到重名模块会 import-mismatch）
#   （细节与实测见 doc/注释历史.md）
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
  # ⚠️ 反面教材（别再犯）：`tests/test_gate_flow.py` 曾经既是"被删的分层版用例"、又是 "活文件"的同名路径 ⇒ 列进 DELETED 会把**刚上传的**文件再删掉。
#   （细节与实测见 doc/注释历史.md）
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
  doc/2026-09-22-改动记录-review.md
  doc/2026-09-23-改动记录-review.md
  doc/2026-09-26-改动记录-review.md
  doc/psi测量步骤_20260923.txt
  # gate/过门-状态机与参数.md 已归位到 doc/记录/过门-状态机与参数.md（来源见 doc/记录/README.md）
  gate/过门-状态机与参数.md
  doc/psi_测量步骤_20260923.txt
  base/settings.py
  base/camera.py
  base/uart.py
  base/telemetry.py
  base/turn_log.py
  common/cfgnode.py
  common/detector.py
  common/preprocess.py
  common/PID.py
  common/turn_deg.py
  gate/gate_decode.py
  gate/gate_detector.py
  gate/gate_frontend.py
  gate/gate_postproc.py
  gate/geometry.py
  gate/kpt_memory.py
  gate/mock.py
    # [REMOVED 2026-10-04] gate/gate_task.py  ← 是活文件（本地存在+在清单里），列在 DELETED 会被误删
  gate/heading_align.py
  tools/analyze/analyze_task_log.py
  tools/analyze/feature_coverage.py
  tools/analyze/analyze_kpt_dump.py
  tools/analyze/analyze_heading.py
  tools/analyze/analyze_pnp_center.py
  tools/analyze/pnp_calib.py
  tools/analyze/ruler_calib.py
  tools/analyze/tape_ticks.py
  tools/analyze/label_corners.py
  tools/check/check_kpt_decode.py
  tools/check/check_gate_pose.py
  tools/check/auv_kpt_meter.py
  tools/check/check_dof_sign.py
  tools/check/check_hdg_lockup.py
  tools/check/check_domain.py
  tools/check/check_paths.py
  tools/check/check_pipeline_identity.py
  tests/test_base.py
  tests/test_common.py
  tests/test_paths.py
  tests/test_hud.py
  tests/tasks/test_gate_flow.py
  tests/tasks/test_gate_vision.py
  tests/tasks/test_gate_postproc.py
  doc/算法说明.md
  doc/gate_pose_decode_spec.md
  doc/现场卡-靶子法.md
  doc/待研究-缺角位姿先验（三维信息复用）.md
  doc/算法说明-gate-角点逐点融合滤波.md
  doc/前视USB相机低延迟推流方案.md
  doc/实验待测-runbook.md
  doc/算法说明-gate-PnP移植方案.md
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

n_same=0; n_new=0; n_diff=0; n_miss_local=0; n_skip=0
: > "$CHANGED"
#   ⚠️ 别把"cfg 不要整份推板端"只写在文档里、靠人工记着排除：漏一次就把现场值冲掉（那些现场值没有本地副本）
#   （细节与实测见 doc/注释历史.md）
FORK_FILE="${AUV_FORKS_FILE:-$TOOLS_DIR/board_forks.txt}"
SKIP_FILES=()
if [ -f "$FORK_FILE" ]; then
  while read -r f; do [ -n "$f" ] && SKIP_FILES+=("$f"); done \
    < <(sed -e 's/#.*//' "$FORK_FILE" | awk 'NF{print $1}')
fi
# 分叉行里带 `[sync+patch]` 标记的 = 「**照常上传**，上传后在板端打补丁」：
#   这类文件本地与板端**必须不同**（如 settings.py 的 SIM_MODE），但整份跳过会让板端
#   拿不到文件里的其它更新（2026-10-06 真踩到：板端缺 STATE_GRAB/STATE_PLACE）。
PATCH_FILES=()
if [ -f "$FORK_FILE" ]; then
  while read -r f; do [ -n "$f" ] && PATCH_FILES+=("$f"); done \
    < <(grep -F '[sync+patch]' "$FORK_FILE" | sed -e 's/#.*//' | awk 'NF{print $1}')
fi
is_patch() { local x="$1" s; for s in "${PATCH_FILES[@]}"; do [ "$s" = "$x" ] && return 0; done; return 1; }
# ⚠️ 带 [sync+patch] 的**不算 skip**（要照常上传，步骤 3b 再打补丁）
is_skip() {
  local x="$1" s
  is_patch "$x" && return 1
  for s in "${SKIP_FILES[@]}"; do [ "$s" = "$x" ] && return 0; done
  return 1
}
while IFS=$'\t' read -r _m f _v; do
  [ -f "$LOCAL/$f" ] || { n_miss_local=$((n_miss_local+1)); continue; }
  lm=$(md5sum "$LOCAL/$f" | cut -d' ' -f1)
  r="${RM[$f]:-}"
  if [ "$r" = "$lm" ]; then n_same=$((n_same+1)); continue; fi
  if is_skip "$f"; then n_skip=$((n_skip+1)); echo "[skip] $f（本地≠板端是**故意的**，不自动上传）"; continue; fi
  if [ -z "$r" ]; then n_new=$((n_new+1)); echo "[new ] $f"; else n_diff=$((n_diff+1)); echo "[diff] $f"; fi
  echo "$f" >> "$CHANGED"
done < "$ALL"
step "差异：上传 $(wc -l < "$CHANGED") 个（新 $n_new / 改 $n_diff）· 已一致 $n_same · 本地缺 $n_miss_local · 故意跳过 $n_skip"

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

# ---- 3b) 同步后补丁（[sync+patch] 的文件；必须在 md5 复核**之后**）----
if [ "${#PATCH_FILES[@]}" -gt 0 ]; then
  cat > "$TMPD/patch_settings.py" <<'PATCHPY'
import sys

p = 'base/cfg/settings.py'
lines = open(p, encoding='utf-8').read().split('\n')
hits = [i for i, l in enumerate(lines) if l.startswith('SIM_MODE = ')]
if len(hits) != 1:
    print('  x SIM_MODE 锚点 %d 个 -> 放弃' % len(hits)); sys.exit(2)
i = hits[0]
if lines[i].startswith('SIM_MODE = False'):
    print('  = 板端已是 SIM_MODE=False（无需改）')
else:
    old = lines[i]
    lines[i] = 'SIM_MODE = False' + old[len('SIM_MODE = True'):]
    open(p, 'w', encoding='utf-8').write('\n'.join(lines))
    print('  ok 板端 SIM_MODE: True -> False（这一行之外不动）')
sys.path.insert(0, '.')
import base.cfg.settings as S
print('  - 运行期读到的 SIM_MODE =', S.SIM_MODE)
PATCHPY
  for f in "${PATCH_FILES[@]}"; do
    case "$f" in
      base/cfg/settings.py)
        base64 -w0 "$TMPD/patch_settings.py" \
          | timeout 120 "$STREAM" "cd $BOARD && base64 -d | python3 -" 2>&1 \
          | grep -vE "Permanently added" | sed 's/^/        /'
        step "补丁完成：$f（板端 SIM_MODE=False，代码其余部分跟本地一致）"
        ;;
      *) echo "[deploy] ⚠️ 未知的补丁目标 $f（[sync+patch] 目前只为 base/cfg/settings.py 实现）" ;;
    esac
  done
fi

# ---- 4) 删除已废弃文件（一次 SSH，先备份）----
{ echo "cd $BOARD"; echo "mkdir -p bak/deploy_$STAMP";
  for f in "${DELETED[@]}"; do
    if awk -v p="$f" '{ sub(/[[:space:]]*\]$/, ""); n=split($0, a, /[[:space:]]+/); if (a[n]==p) { found=1; exit } } END { exit !found }' "$MANIFEST"; then
      echo "[skip-live] $f（在清单里 = 活文件，不删）"
      continue
    fi
    echo "[ -f '$f' ] && { mkdir -p bak/deploy_$STAMP/$(dirname "$f"); cp -p '$f' 'bak/deploy_$STAMP/$f'; rm -f '$f'; echo \"[removed+bak] $f\"; }"
  done
  echo "rmdir work 2>/dev/null"
  echo "rmdir task1_2 2>/dev/null   # 2026-10-06 结构迁移后只剩空目录"
  echo "rmdir gate/vision 2>/dev/null; rmdir gate/data 2>/dev/null; rmdir gate/motion 2>/dev/null"
  echo "find . -name '*.sh' -not -path './bak/*' -exec chmod +x {} + 2>/dev/null; \
        find . -name __pycache__ -type d -not -path './bak/*' -prune -exec rm -rf {} + 2>/dev/null; echo CLEAN-OK"; } > "$TMPD/del.sh"
timeout 120 "$STREAM" "bash -s" < "$TMPD/del.sh" 2>/dev/null | grep -vE "^\s*$" | sed 's/^/        /'
step "废弃文件处理完成"

# ---- 5) 板端自检 ----
{
  echo "cd $BOARD"
  echo "python3 -m py_compile main.py preview_detect.py base/__init__.py base/hw/__init__.py base/hw/camera.py base/hw/telemetry.py base/hw/uart.py base/cfg/__init__.py base/cfg/settings.py base/log/__init__.py base/log/turn_log.py common/__init__.py common/motion/PID.py common/motion/__init__.py common/motion/axis.py common/motion/drop.py common/motion/depth_hold.py common/motion/search_sweep.py common/motion/search_scan.py common/motion/turn_deg.py common/vision/__init__.py common/vision/detector.py common/vision/preprocess.py common/cfg/__init__.py common/cfg/cfgnode.py gate/__init__.py gate/gate_task.py gate/motion/__init__.py gate/motion/channels.py gate/motion/exits.py gate/motion/hdg.py gate/motion/modes.py gate/motion/params.py gate/percept/__init__.py gate/percept/down_view.py gate/percept/gate_decode.py gate/percept/gate_detector.py gate/percept/gate_frontend.py gate/percept/gate_postproc.py gate/percept/geometry.py gate/percept/kpt_memory.py gate/percept/mock.py task/__init__.py task/ball.py handling/handling_task.py handling/motion/__init__.py handling/motion/actions.py handling/motion/params.py handling/motion/phases.py handling/percept/__init__.py handling/percept/ball_tracker.py handling/percept/cage_color.py handling/percept/cv_ball.py handling/percept/grab_detector.py manual/__init__.py manual/cam_switch.py manual/recorder.py manual/stream.py manual/udp_server.py && echo COMPILE-OK"
  [ "$NO_TEST" = "1" ] || echo "echo '--- 全量 pytest ---'; timeout 600 python3 -m pytest tests/ -q > /tmp/auv_pytest.log 2>&1; tail -3 /tmp/auv_pytest.log; if grep -qE '[0-9]+ passed' /tmp/auv_pytest.log && ! grep -qE '[0-9]+ (failed|error)' /tmp/auv_pytest.log; then echo PYTEST-OK; else echo PYTEST-FAIL; fi"
  cat <<'PYEOF'
echo '--- 配置快照 ---'
python3 - <<'PY'
import sys; sys.path.insert(0,'.')
import base.cfg.settings as S
from gate.percept.kpt_memory import ENV_ENABLE, kpt_mem_enabled
print('ball : timeout_ms=%s dash_ratio=%s stop_hold_s=%s' % (S.comm.ball.timeout_ms, S.comm.ball.dash_ratio, S.comm.ball.stop_hold_s))
print('gate : near_ratio=%s hold=%s reacquire=%s' % (S.comm.gate.coarse.near_ratio, S.comm.gate.hold.max_frames, dict(S.comm.gate.reacquire)))
print('motion(共用) : sway_kp=%s heave_kp=%s surge_fast=%s surge_slow=%s' % (
    S.comm.motion.pid_sway.kp, S.comm.motion.pid_heave.kp,
    S.comm.motion.surge_fast, S.comm.motion.surge_slow))
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

# ⚠️ `board_parity.md5` **只对应默认目标（在用副本 AUV_New）**。
#   （细节与实测见 doc/注释历史.md）
if [ "${AUV_PARITY_WRITE:-1}" = "1" ]; then
  AUV_SSH="$SSH" bash "$TOOLS_DIR/check_board_parity.sh" --board --write 2>&1 | tail -4
else
  step "跳过清单刷新（AUV_PARITY_WRITE=0：目标 $BOARD 不是清单对应的仓库）"
fi
step "完成：上传 ${n_new}/${n_diff} 新/改，已一致 ${n_same}"

if [ "$NO_TEST" != "1" ] && grep -q PYTEST-FAIL "$TMPD/selfcheck.log" 2>/dev/null; then
  echo "[deploy] ⚠️ 板端 pytest 未全绿（见上面 tail -3）：代码已同步，但**板端自检不算通过**。"
  echo "         常见原因：板端还留着旧用例（import 已删除模块）→ 加进本脚本 DELETED 列表后重跑。"
  exit 1
fi
