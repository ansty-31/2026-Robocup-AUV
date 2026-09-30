#!/bin/bash
# tools/deploy/sync_Adomain_repo.sh — 板端**旧仓库（A 域 / 原识别方案）**同步与自检
#
# 板端两个仓库：**代码相同（都是最新），参数不同（两套识别方案）**
#
#   | 仓库 | 路径 | 识别方案 | 链路 | 门权重 | 内参 | 阈值 |
#   |---|---|---|---|---|---|---|
#   | 旧仓库（本脚本管） | /home/sunrise/AUV | **A 域（原方案）** | remap@720p→resize640→enhance | AUV_4 `d38b803e…` | 老 `fx=782.5` | 老 |
#   | 在用副本 | ~/Desktop/AUV_New | **D 域（新方案）** | resize640→enhance→remap@640 | 阶段一 `ef52c18b…` | 新 `fx=1207.6` | 新 |
#
# 口径：A 域那套**老功能的参数一个都不改**，只放两个**新功能**进来（① 转角度 ② 解码后约束），
# 它们自己的键取新值。口径与逐键映射见
# `tools/deploy/Adomain_cfg/README.md`（A 域参数的 canonical 副本就在那个目录）。
#
# 用法：
#   bash tools/deploy/sync_Adomain_repo.sh            # 备份 → 推 A 域 cfg → 同步最新代码 → 自检
#   bash tools/deploy/sync_Adomain_repo.sh --dry-run  # 只看差异，不动板端
# 退出码：0=全部通过；1=板端 pytest 未全绿 / 下游与在用副本不一致 / 域指纹不自洽 / 老参数没落对。
set -u
TOOLS_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL="$(cd "$TOOLS_DIR/../.." && pwd)"
BOARD_HOST="${AUV_BOARD_HOST:-sunrise@192.168.137.10}"
A_DIR="${AUV_ADOMAIN_DIR:-/home/sunrise/AUV}"               # 旧仓库（A 域）
NEW_DIR="${AUV_BOARD_DIR:-/home/sunrise/Desktop/AUV_New}"   # 在用副本（D 域），只用来核对下游一致
HELP_DIR="${AUV_HELP_DIR:-/home/ansty/RDKX5}"
SSH="${AUV_SSH:-$HELP_DIR/.ssh_x5.sh}"
STREAM="${AUV_STREAM:-$HELP_DIR/.ssh_x5_stream.sh}"
SCP="${AUV_SCP:-$HELP_DIR/.scp_x5.sh}"
LOCAL_BAK="${AUV_LOCAL_BAK:-$LOCAL/bak}"
CFG_DIR="$TOOLS_DIR/Adomain_cfg"
REF_REL="tools/deploy/Adomain_cfg/preprocess_A_ref_20260917.py"
STAMP=$(date +%m%d_%H%M%S)
DRY=0; for a in "$@"; do [ "$a" = "--dry-run" ] && DRY=1; done
T0=$(date +%s); step() { echo "[$(printf '%4ds' $(( $(date +%s) - T0 )))] $*"; }

# ---- 0) 前置检查 ----
for f in "$CFG_DIR/vision.yaml" "$CFG_DIR/comm.yaml" "$CFG_DIR/front_camera.yaml" "$CFG_DIR/preprocess_A_ref_20260917.py"; do
  [ -f "$f" ] || { echo "[Adomain] ✗ 缺 canonical 文件 $f"; exit 2; }
done
{ [ -x "$SSH" ] && [ -x "$STREAM" ] && [ -x "$SCP" ]; } \
  || { echo "[Adomain] ✗ 缺 SSH/SCP 包装器（见 deploy_to_board.sh 头部）"; exit 2; }
[ "$A_DIR" != "$NEW_DIR" ] || { echo "[Adomain] ✗ AUV_ADOMAIN_DIR 不能等于在用副本"; exit 2; }

# ---- 1) 探板 + 目标校验 ----
t=0
until timeout 12 "$SSH" "echo BOARD-UP" </dev/null 2>/dev/null | grep -q BOARD-UP; do
  t=$((t+10)); [ "$t" -ge 60 ] && { echo "[Adomain] ✗ 板子未接入（等了 60s）"; exit 1; }; sleep 10
done
step "板子在线"
timeout 20 "$SSH" "test -d '$A_DIR' && test -f '$A_DIR/models/gate_kpt_auv4_bayese_640x640_nv12.bin' && echo A-OK" \
  </dev/null 2>/dev/null | grep -q A-OK \
  || { echo "[Adomain] ✗ $A_DIR 不存在或缺 AUV_4 权重（先按 tools/README.md §6.1 传权重）"; exit 1; }
step "目标 = $A_DIR（旧仓库 / A 域）"

if [ "$DRY" = "1" ]; then
  step "--dry-run：将推送的 A 域 cfg（canonical → 板端 cfg/）"
  for f in vision comm front_camera; do
    printf '        %-20s %s\n' "cfg/$f.yaml" "$(md5sum "$CFG_DIR/$f.yaml" | cut -d' ' -f1)"
  done
  step "--dry-run：代码侧差异（cfg 三项 + 阶段一权重列入分叉，不会覆盖）"
  AUV_BOARD_DIR="$A_DIR" AUV_FORKS_FILE="$TOOLS_DIR/board_forks_Adomain.txt" AUV_PARITY_WRITE=0 \
    bash "$TOOLS_DIR/deploy_to_board.sh" --dry-run 2>&1 | tail -20
  exit 0
fi

# ---- 2) 备份旧仓库（先备份再动）----
step "备份 $A_DIR → 板端 bak/ 与本地 $LOCAL_BAK/"
timeout 300 "$STREAM" "set -e; cd '$A_DIR'; tar czf /tmp/repo_Adomain_$STAMP.tar.gz \
    --exclude=./log --exclude=./rec --exclude='*/__pycache__' --exclude=./__pycache__ \
    --exclude=./.pytest_cache --exclude='./bak/*' . ; mkdir -p bak && mv /tmp/repo_Adomain_$STAMP.tar.gz bak/ \
  && ls -l bak/repo_Adomain_$STAMP.tar.gz && echo SNAP-OK" 2>/dev/null | grep -q SNAP-OK \
  || { echo "[Adomain] ✗ 备份失败（**别继续**）"; exit 1; }
mkdir -p "$LOCAL_BAK"
timeout 300 "$STREAM" "cat '$A_DIR/bak/repo_Adomain_$STAMP.tar.gz'" > "$LOCAL_BAK/board_legacy_Adomain_$STAMP.tar.gz" 2>/dev/null
step "备份完成：板端 bak/repo_Adomain_$STAMP.tar.gz ＋ 本地 board_legacy_Adomain_$STAMP.tar.gz"
# 注：跑过 tools/deploy/tidy_board_bak.sh 之后，板端快照会被移进 bak/archive/（仍可 tar xzf 取回）

# ---- 3) 推 A 域 cfg（canonical → 板端 cfg/）----
for f in vision comm front_camera; do
  timeout 120 "$SCP" "$CFG_DIR/$f.yaml" "$BOARD_HOST:$A_DIR/cfg/$f.yaml" 2>/dev/null \
    || { echo "[Adomain] ✗ 推送 cfg/$f.yaml 失败"; exit 1; }
done
step "A 域 cfg 已推送（chain A / clahe 0.5 / 老内参 782.5 / 老阈值 + 两个新功能的键）"

# ---- 4) 同步最新**代码**（cfg 三项与阶段一权重按分叉跳过）----
step "同步最新代码（AUV_FORKS_FILE=board_forks_Adomain.txt，AUV_PARITY_WRITE=0）"
AUV_BOARD_DIR="$A_DIR" AUV_FORKS_FILE="$TOOLS_DIR/board_forks_Adomain.txt" AUV_PARITY_WRITE=0 \
  bash "$TOOLS_DIR/deploy_to_board.sh" --no-test 2>&1 | sed 's/^/        /'
step "代码同步完成"

# ---- 5) base/cfg/settings.py：板端 = 真发串口（SIM_MODE=False）----
cat > /tmp/Adomain_settings.py <<'PYEOF'
import sys
p = 'base/cfg/settings.py'
L = open(p, encoding='utf-8').read().split('\n')
hits = [i for i, l in enumerate(L) if l.startswith('SIM_MODE = ')]
if len(hits) != 1:
    print('  x SIM_MODE 锚点 %d 个 -> 中止' % len(hits)); sys.exit(2)
i = hits[0]
if 'sync_Adomain_repo.sh' in '\n'.join(L[i:i + 4]):
    print('  = base/cfg/settings.py 已是 A 域状态（跳过）')
else:
    block = [
        'SIM_MODE = False         # 无硬件调试：串口仅打印（板上置 False）',
        '# 由 tools/deploy/sync_Adomain_repo.sh 落成 False：本仓库 = 旧仓库（A 域），默认真发串口。',
        '#   在用副本 AUV_New 的这份文件是"本地分叉"（本地 True），见 tools/deploy/board_forks.txt。',
    ]
    j = i + 1
    while j < len(L) and L[j].startswith('#') and \
            any(k in L[j] for k in ('分叉', 'board_forks', '台架只想打印')):
        j += 1
    L[i:j] = block
    open(p, 'w', encoding='utf-8').write('\n'.join(L))
    print('  ok base/cfg/settings.py: SIM_MODE -> False')
PYEOF
base64 -w0 /tmp/Adomain_settings.py | timeout 120 "$STREAM" "cd '$A_DIR' && base64 -d | python3 -" 2>&1 \
  | grep -vE "Permanently added" | sed 's/^/        /'
rm -f /tmp/Adomain_settings.py

# ---- 5b) 清掉误入的阶段一权重（属 D 域）；A 序参考实现放上板端供 ② 核验 ----
{
  echo "cd '$A_DIR'"
  echo "mkdir -p 'bak/Adomain_ref_$STAMP'"
  echo "[ -f models/gate_kpt_stage1_g240_i16_bayese_640x640_nv12.bin ] && { cp -p models/gate_kpt_stage1_g240_i16_bayese_640x640_nv12.bin 'bak/Adomain_ref_$STAMP/'; rm -f models/gate_kpt_stage1_g240_i16_bayese_640x640_nv12.bin; echo '  [removed] 阶段一权重（属 D 域，旧仓库不用）'; }"
  # 转向调用日志 `turn_log` 已从 common/ 挪到 base/（用户 2026-09-27 定）→ 旧位置删掉，先备份
  echo "[ -f common/turn_log.py ] && { cp -p common/turn_log.py 'bak/Adomain_ref_$STAMP/turn_log.py.old'; rm -f common/turn_log.py; echo '  [moved] common/turn_log.py → base/log/turn_log.py（旧位置已删，备份在 bak/Adomain_ref_$STAMP/）'; }"
  echo "find . -name __pycache__ -type d -not -path './bak/*' -prune -exec rm -rf {} + 2>/dev/null; echo CLEAN-OK"
} > /tmp/Adomain_clean.sh
timeout 120 "$STREAM" "bash -s" < /tmp/Adomain_clean.sh 2>/dev/null | grep -vE "^\s*$" | sed 's/^/        /'
rm -f /tmp/Adomain_clean.sh
REF_BOARD="bak/Adomain_ref_$STAMP/preprocess_A_ref.py"
timeout 60 "$STREAM" "cd '$A_DIR' && cat > '$REF_BOARD'" < "$LOCAL/$REF_REL" 2>/dev/null

# ---- 6) 板端自检 ----
{
  echo "cd '$A_DIR'"
  echo "python3 -m py_compile main.py preview_detect.py base/cfg/settings.py common/vision/preprocess.py common/cfg/cfgnode.py common/motion/turn_deg.py gate/percept/gate_decode.py gate/percept/gate_postproc.py gate/motion/gate_task.py gate/percept/gate_detector.py gate/motion/heading_align.py gate/percept/kpt_memory.py gate/percept/geometry.py && echo COMPILE-OK"
  echo "echo '--- (1) 域指纹 + (2) A 链序 vs 参考实现逐像素 + (3) AUV_4 角点分布 ---'"
  echo "timeout 900 python3 tools/check/pipeline/check_domain.py --equiv-ref '$REF_BOARD' --equiv-frames log/frames --limit 4 --frames log/frames --limit 4; echo DOMAIN-RC=\$?"
  echo "echo '--- (4) 过门下游与在用副本逐字节一致 ---'"
  echo "for f in gate/percept/gate_postproc.py gate/percept/kpt_memory.py gate/percept/geometry.py gate/motion/gate_task.py gate/motion/heading_align.py common/motion/turn_deg.py common/cfg/cfgnode.py common/vision/preprocess.py; do a=\$(md5sum \$f 2>/dev/null | cut -d' ' -f1); b=\$(md5sum '$NEW_DIR'/\$f 2>/dev/null | cut -d' ' -f1); if [ \"\$a\" = \"\$b\" ]; then echo \"  [same] \$f\"; else echo \"  [DIFF] \$f  A=\$a  new=\$b\"; fi; done"
  echo "echo '  —— 以下两个文件板端**就地改过**（接入域映射，属预期的不同）——'"
  echo "for f in gate/percept/gate_decode.py gate/percept/gate_detector.py; do echo \"  [domain-map] \$f  \$(md5sum \$f | cut -c1-8)（在用副本 \$(md5sum '$NEW_DIR'/\$f | cut -c1-8)）\"; done"
  echo "echo '--- (4b) 检测域→PnP域 几何映射核对（合成往返 + 位姿回收）---'"
  echo "timeout 900 python3 tools/check/check_domain_map.py --frames log/frames; echo DOMAINMAP-RC=\$?"
  echo "echo '--- (5) A 域口径的老参数 + 两个新功能的键 ---'"
  cat <<'PYEOF'
python3 - <<'PY'
import base.cfg.settings as S
want = [('vision.image.chain', 'A'), ('vision.image.clahe_clip', 0.5),
        ('vision.model.score_threshold', 0.5), ('vision.gate.det.conf', 0.5),
        ('vision.gate.keypoint.conf_thr', 0.7), ('vision.gate.percept.geometry.frame_w', 0.70),
        ('comm.gate.timeout_ms', 180000), ('comm.gate.align.confirm_frames', 4),
        ('comm.gate.align.px_x', 0.20), ('comm.gate.z.cross', 0.7),
        ('comm.gate.z.near_lost_m', 1.0), ('comm.gate.z.near_lost_ratio', 0.60),
        ('vision.gate.postproc.edge_tol', 0.10),
        ('comm.gate.hdg.enable', True), ('comm.gate.through.require_align_deg', 8.0),
        ('comm.motion.turn_pid.kd', 0.0), ('comm.motion.turn_pid.period', 0.05),
        ('comm.motion.pid_sway.kp', 8.0), ('comm.motion.pid_heave.kp', 1.0)]
bad = 0
for k, v in want:
    got = S.get(k, '<none>')
    ok = got == v
    bad += 0 if ok else 1
    print('  %s %-40s = %s' % ('ok ' if ok else 'BAD', k, got))
print('PARAM-RC=%d' % (1 if bad else 0))
PY
PYEOF
  echo "echo '--- (6) 全量 pytest ---'"
  echo "timeout 900 python3 -m pytest tests/ -q > /tmp/Adomain_pytest.log 2>&1; tail -3 /tmp/Adomain_pytest.log; if grep -qE '[0-9]+ passed' /tmp/Adomain_pytest.log && ! grep -qE '[0-9]+ (failed|error)' /tmp/Adomain_pytest.log; then echo PYTEST-OK; else echo PYTEST-FAIL; fi"
} > /tmp/Adomain_check.sh
timeout 1800 "$STREAM" "bash -s" < /tmp/Adomain_check.sh 2>&1 \
  | grep -vE "Permanently added|^\[BPU|^\[HBRT|^\[DNN|^\[A\]\[DNN" | tee /tmp/Adomain_check.log
rm -f /tmp/Adomain_check.sh
step "板端自检完成"

rc=0
grep -q "PYTEST-FAIL" /tmp/Adomain_check.log && { echo "[Adomain] !! 板端 pytest 未全绿（见 tail -3）"; rc=1; }
grep -q "\[DIFF\]" /tmp/Adomain_check.log && { echo "[Adomain] !! 过门下游与在用副本不一致（见 [DIFF]）"; rc=1; }
grep -q "DOMAIN-RC=1" /tmp/Adomain_check.log && { echo "[Adomain] !! 域自检不通过（域指纹/等价性）"; rc=1; }
grep -q "PARAM-RC=1" /tmp/Adomain_check.log && { echo "[Adomain] !! A 域老参数没落对（见 (5)）"; rc=1; }
grep -q "DOMAINMAP-RC=1" /tmp/Adomain_check.log && { echo "[Adomain] !! 检测域→PnP域 几何映射核对不通过（见 (4b)）"; rc=1; }
if [ "$rc" = "0" ]; then
  step "完成：$A_DIR = 最新代码 + A 域（老内参/老阈值 + 两个新功能）"
else
  echo "[Adomain] 有项目未通过，详见上面 (1)~(6)"
fi
exit $rc
