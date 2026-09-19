# 实验待测 Runbook（人机协同：现场执行 / 后台分析）

> 目标：把 `gate/README.md §5` 的待测项按"先台架、后水池、先视觉、后运动"的顺序跑完，
> 每一步都有**明确命令、明确判据、明确要汇报的量**。执行的原始数据一律落盘 → 拉回来分析 → 再改参数。
>
> 记录表在文末，边测边填。相关工具见 `tools/README.md`。

## 安全前提（每次下水前逐条确认）

| 检查 | 命令/动作 | 通过标准 |
|---|---|---|
| 串口模式 | 启动横幅 / `AUV_SIM_MODE=0` 显式指定 | 要动船必须 `实发`；台架用 `仅打印` |
| 权重 | `python3 tools/check_pipeline_identity.py` + 启动横幅 `权重: gate=gate_kpt_bayese_640x640_nv12.bin` | 域自证通过 + 权重名正确 |
| 代码版本 | `bash tools/check_board_parity.sh` | 本地 == 清单（板端 == 清单 更好） |
| 相机占用 | `ps -ef \| grep "[m]ain.py"` | 无孤儿进程（run_gate.sh 会自动清） |
| 电池/遥控 | 手动模式可随时接管 | 遥控器在手且能急停 |

## 测试阶梯

### S0 台架自检（不接水、船不动）
```bash
cd /home/sunrise/AUV
python3 tools/check_pipeline_identity.py            # 域自证
python3 preview_detect.py --gate-kpt --duration 5   # 权重能跑、能看到门框/角点
```
**汇报**：域自证是否通过；`det=` 数量；能否看到 4 个角点。失败→先解决，别下水。

### S1 视觉验收 + 数据落盘（船不动，镜头对准门）
```bash
python3 preview_detect.py --gate-kpt --dump log/kpt_s1_static.jsonl --duration 30 --show
```
**要看的**：4 个角点是否稳定出现、有没有点落到倒影上、`有效角点 n/4`、画面里的 `mode/ids`。
**汇报**：`det` 数量、`n/4`、有没有明显鬼点（点跳到水面倒影上）。
**产出**：`log/kpt_s1_static.jsonl` → `bash tools/fetch_logs.sh` → 分析。

### S2 倒影专项（船不动，镜头慢慢靠近水面线）
```bash
python3 preview_detect.py --gate-kpt --dump log/kpt_s2_reflect.jsonl --duration 30
```
**要看的**：角点跳到倒影上的频率；框是否被"拉高"（宽高比变小）。
**判据**：分析脚本会报"宽高比 p10" 与 ">30px 跳变次数"。

### S3 线索方向核对（`record_only: true`，船**不动**）
```bash
# 板端：cfg/comm.yaml → gate.cues.record_only: true（先只记录不驱动）
python3 preview_detect.py --gate-kpt --duration 30       # 看 visible 掩码
# 或全程跑任务但不下发线索动作
AUV_SIM_MODE=1 python3 main.py --task gate
```
**判据**：只看到上边(TL,TR)→应 `DESCEND`；下边→`ASCEND`；左列→`TURN_RIGHT`；右列→`TURN_LEFT`。
方向反了 ⇒ 改 `cues.*` 开关或 `motion/phase_recover.py::_CUE_DOF` 符号。
**汇报**：每种可见性组合 → 观察到的 cue 动作 → 与预期是否一致。

### S4 低风险运动（下水，慢速）
```bash
cd task1_2 && AUV_SIM_MODE=0 AUV_DESCEND_S=0 AUV_FWD_S=0 ./run_gate.sh
```
**要看的**：`phase/substate/mode/ratio/kpt/kpt_raw/Q/caution/cue_w/cue_mode/z/pass`。
**汇报**：完整终端日志（我拉回来逐帧对）。重点：有没有"点不全就后退"、REACQUIRE 次数、
`z` 抖不抖、`cue_w` 有没有乱升。

### S5 主导权三档对比（同一条轨迹跑 3 遍）
```yaml
gate.cues.mode: pose   # ① 位姿主导
gate.cues.mode: auto   # ② 依据质量自动分配（默认）
gate.cues.mode: cue    # ③ 线索主导
```
**判据**：倒影期 `cue_w` 应按 ①=0 → ② 自动升高 → ③=1；比较过门成功率与用时。

### S6 参数标定（用 S1/S2 的 dump）
```bash
bash tools/fetch_logs.sh
python3 tools/analyze_kpt_dump.py log/board_*/kpt_*.jsonl
```
把建议值写进 `cfg/vision.yaml`（kpt_mem / keypoint.conf_thr）与 `cfg/comm.yaml`（quality），
`bash tools/deploy_to_board.sh --no-test` 同步后复测 S1，比较 `pose_flips` 是否下降。

### S7 帧率与预处理
```bash
python3 preview_detect.py --gate-kpt --duration 20     # 记录 fps（带/不带 --stream 各一次）
```
**判据**：带推流 7.5 fps（实测）；目标把预处理 54ms 压下来 → 10 fps+。

## 记录表

| 步骤 | 日期 | 条件（船/门/水） | 关键数字 | 结论/改动 | 数据文件 |
|---|---|---|---|---|---|
| S0 | | | | | |
| S1 | | | | | |
| S2 | | | | | |
| S3 | | | | | |
| S4 | | | | | |
| S5 | | | | | |
| S6 | | | | | |
| S7 | | | | | |
