# 已知坑与诊断顺序

> 每条都尽量带上"症状 → 第一嫌疑 → 怎么证实"。踩过的坑按"会让人误判"的程度排前面。

## A. 会让人**误判结论**的坑（最贵）

### A1 把离线数字当成水中效果
- 症状：拿"整帧 78.4% / ROI 25.6 ms"说"夹取没问题了"。
- 事实：这些数字是**离线逐帧喂图**测的，不含相机 read、显示、曝光变化、水面折射与倒影。`r_min=32` 这类阈值是在**特定池子、特定相机高度**的素材上扫的（2026-10-07 已由 25 抬到 32，
  因为新素材里 25–32 px 那段全是瓦条端头深红反光）。
- 怎么证实：③ 板端预览 → ④ 板端实时指标（见 `verify-and-deploy.md`），并明确写"离线 vs 板端"两组数字。

### A2 把"顺手复用"当成"YOLO 不行"
- 症状：看到远距段 33–42% 命中就断言"BPU 在这条任务上没用"。
- 事实：那版权重**本来就不是给夹取训的**（`grab/README.md` 明确写了这句）。结论只成立到"顺手复用不可行"。
- 怎么证实：要下"YOLO 行不行"的结论，必须先补远距样本、标注、量化、板端实测。

### A3 拿 `TrackInfo` 当控制判据
- 症状：用 `TrackInfo.mode` / `lost` 直接决定"该搜索还是该夹"。
- 事实：代码注释写着"**别拿它做控制判据**"（它是给日志/调试的）。控制判据要自己从 `Detection` 与历史里算。
- 怎么证实：`grep -n "别拿它做控制判据" handling/percept/ball_tracker.py`。

### A4 数据来源不明就写进结论
- 症状：把 `grab/README.md` 的数字当"现在代码跑出来的"。
- 事实：README 是 2026-10-06 的快照，代码可能已变（尤其是 `Params` 默认值）。
- 怎么证实：任何要引用的数字，同轮用 `pytest` 或②离线脚本**重跑一遍**再引用；不能重跑的明确标"来自 README 快照"。

## B. 会把**性能搞坏**的坑（板端实测过）

### B1 ARM 上用 16 位 / 稀疏索引
`cv2.multiply(dtype=CV_16U)`、`np.nonzero` + 花式索引在 A55 上是**标量实现**：整体 97.6 ms，比不优化还慢。
→ 全 uint8 + `cv2.LUT` + `cv2.inRange`。详见 `perf-and-params.md` §2 坑 1。

### B2 ROI 尺寸写错公式
`side = 4r + 80` ⇒ 球半径 188 时 ROI 占 69% 全图 ⇒ **ROI 模式反而比全图慢**。
→ 正确公式 `side = 2(r + margin)`（`margin` 默认 64 ≈ 1.6× 实测最大帧间位移 40 px）。

### B3 归一化改纵横比
把截成 427×360 的 ROI 硬 resize 成 256×256 ⇒ 圆心回算飘（单测抓到过 `cy` 差 52 px）。
→ `cv2.resize(sub, None, fx=sc, fy=sc)`，**同一比例缩两轴**。

## C. 换目标（小环）时会踩的坑

### C1 把"填空高光"套到空心目标上
红球的"填高光孔"是为了修"球被高光打成红环"。小环**本身就是环**，填空等于把环心当孔填掉 ⇒ 圆心/半径全错。
→ 空心目标要换判据（找环带/内外边界），见 `algorithm.md` §2 与 §6。

### C2 用 IoU 去重处理同心圆
`merge_iou=0.6` 会把环的**内圈与外圈**判成同一目标的两个候选，只留一个 ⇒ 半径错一半。
→ 同心目标要按"圆心接近 + 半径差显著"区分，不能只看 IoU。

### C3 直接复用 `cv_ball.Params`
`cov_min(0.35)`、`min_arc_deg(150)`、`fill_hole_frac(0.65)`、`circ_min/hull_fill_min` 全部建立在"**实心、偏红、不透光**的圆"这个先验上。
→ 小环要独立的参数段（如 `grab.cv_ring`）与独立 detector；**可以**复用 `Detection` 结构与 `BallTracker` 的跟踪骨架（跟踪只依赖圆心+半径+位移，与语义无关）。

### C4 环被高光/水花打碎后仍按整圆面积判据
→ 与红球"月牙"问题同类：用张角/覆盖率这类"部分可见也成立"的判据，别用"完整圆度"。

## D. 工程与流程坑

### D1 新增源文件没进部署清单
`tools/deploy/board_parity.md5` 里的路径**就是部署契约**；不在清单里的新文件（例如 `grab/motion/*.py`）不会上板。
→ 新增文件后更新清单，再用 `check_board_parity.sh --board` 双向核对。

### D2 `SIM_MODE` 忘了切
`base/cfg/settings.py` 本地默认 `SIM_MODE = True`（只打印、不驱动），且该文件 `deploy` **整份跳过**（板端与本地各留一份）。
→ 真驱动必须显式 `AUV_SIM_MODE=0 python3 main.py --task ...`；上板后要单独确认板端那一份的值。测试环境 `conftest.py` 强制 SIM，所以**单测通过 ≠ 会动**。

### D3 11B 帧通道已经被占
`frame.axis_count=7`（byte1–7 七轴），byte8/9 被**相对转角/取消**占用（`0x80 | 高7位`，`build_turn_frame`），byte10 是旋转编号。
→ 夹爪要新增通道，先确认下位机固件留没留位；新增后要保证普通运动帧与转帧互不干扰。

### D4 把夹取塞进状态机太早
`grab/README.md` §4 与人格都写了：**决策层没设计完之前不要占状态机**（`TASK_CLASS` / `TASK_CAM` / `TASK_STATE` 三处要一起加）。
→ 顺序是：感知能自证 → 决策层纸面设计 → 台架闭环 → 才接 `main.py`。

### D5 `ball/comm` 共用参数被顺手改
`comm.motion.surge_fast/surge_slow`、两套 PID 等是 **ball 与 gate 共用**的（`common/cfg/cfgnode.py` 是唯一入口）。
→ 夹取要速度档，优先新增 `comm.grab.*` 自有键，不要动共用值（会静默改变撞球与过门行为）。

### D6 跨工作区写坏了训练侧
`../RDKX5-YOLOv11n-/` 只有**测试脚本**可以共享/写入，其他文件（训练/导出/量化脚本与配置）**一律只读**——它们会静默影响 `.bin`。

### D7 子包被 eager import
`grab/__init__.py` 刻意不 eager import 子模块（为了 `grab.percept.cv_ball` 能脱离工程单独 import 做离线验证，与 `gate/__init__.py` 同做法）。
→ 别为"方便"加 import；依赖规则：`grab/` 只 import `base` / `common` 与同级任务模块。

### D8 工作区被并发改动
用户或另一个会话可能同时改文件（换权重、改 cfg、删素材）。
→ 依赖的文件定期用 mtime + 内容复核；发现漂移先停下来核对，不要把别人的改动当成自己的，也不要覆盖它。

## E. 诊断顺序（出问题时按这个顺序查，别跳）

```
1. 现在在哪一代？      git log/status + md5 关键文件（cfg/权重/素材）—— 先排除"跑的不是我以为的代码"
2. 感知行不行？        pytest tests/tasks/handling → 离线喂图（分桶命中率/耗时）
3. 链路对不对？        preview_detect.py --grab --show（真相机、不发运动）
4. 实时够不够快？      板端 p50/p95 耗时 + fallback_rate
5. 决策层逻辑对不对？  AUV_SIM_MODE=1 台架闭环 + analyze_task_log.py + feature_coverage.py
6. 才轮到实船          AUV_SIM_MODE=0（⚠️ 先用户拍板）
```
