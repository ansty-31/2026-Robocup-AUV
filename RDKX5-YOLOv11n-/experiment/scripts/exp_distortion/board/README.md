# 板端实测脚本（2026-09-23 那一晚用的）

这些脚本只在**板上**跑（RDK X5，`/home/sunrise/Desktop/AUV_New`）。当晚它们只存在于
板子的 `/tmp`，所以数字不可复现；现已归档到这里。

> ⚠️ 脚本里的路径写的是板端绝对路径（`/home/sunrise/Desktop/AUV_New`、
> 测试帧目录 `--frames-dir` 默认 `/tmp/bench2`）。跑到板上之前先把测试帧 scp 过去。

| 脚本 | 用途 | 对应结论 |
|---|---|---|
| `bench_preprocess.py` | 预处理各变体（P1/P1-noclahe/P1-noenh/P4…）单测 | §18.2 分项 |
| `bench_pipeline_parts.py` | pre / NV12(slow,fast) / BPU / decode 拆开计时 | §18.2（NV12 慢路径 37.6 vs 快 1.1 ms） |
| `bench_lut_wb.py` | 证明 LUT 白平衡与原 float 版**逐像素相同** + 提速 | §18.2 / §5.5（已落地） |
| `bench_end2end.py` | 端到端 `detect()` 三档对比（现状 / LUT / LUT+clip0） | §18.2（72.1 → 45.45 → 37.21 ms） |
| `bench_end2end_and_kpt_delta.py` | 端到端 + **关 CLAHE 前后角点偏移** | §4.4④（int8 模型偏移 5.83 px@640） |
| `dump_board_kpts.py` | 用部署 `.bin` 跑测试帧、dump 角点 → 供 PC 侧比精度 | §4.6 / §18.3 |
| `verify_lut_patch.py` | 校验补丁后的 `enhance` 与原实现逐像素一致 | §18.2（28/28 张 max diff=0） |
| `verify_calibration_swap.py` | 换标定前后门距对比（走板端 `gate_pose`） | §5.4 / §18.1（×1.401） |
| `preprocess_lut_wb.deployed.py` | **板端已部署**的 `common/preprocess.py`（LUT 白平衡） | 归档，便于与板端 md5 对照 |
| `preprocess_original.py` | 改动前的原文件 | 供 diff / 回退参考 |
| `frames_run1.txt` | 第一次板端测速用的 24 帧（`data/mapped/raw/AUV_4dir`） | 复现用 |
| `frames_run2_testsplit.txt` | 精度测试用的 28 帧（test split、4 角可见） | §4.6 / §18.3 |

板端原始测量输出：`experiment/runs/domain/board/board_kpts_run{1,2}*.json`；
板端当前状态（md5 / 备份名）见 `experiment/runs/domain/EXPERIMENT_DESIGN.md` §18.5。
