---
name: auv-grab-engineering
description: Use when the task concerns the RoboCup AUV 夹取 (grab) task on the RDK X5 project — the two independent stages 先夹小球、再夹小环 — building or changing the grab perception layer (纯 CV 圆检测 / ROI 跟踪 vs BPU), designing the grab phase machine (搜索→对准→接近→夹取→退出), deciding the gripper channel and its timing, calibrating grab parameters against real frames, or running the grab verification ladder on PC and on the board. 触发词：夹取、夹球、夹小球、夹小环、小环、圆环、机械爪、夹爪、grab、grab_task、cv_ball、ball_tracker、红球、ROI 跟踪、对准、夹取判据。
---

# AUV 夹取工程 playbook

夹取这件事的失败模式不是"检测不准"，而是**在只有感知层的地基上，把没标定过的判据和没验证过的夹爪时序直接当成能跑的东西**。水下的代价不可逆：撞池壁、撞坏夹爪、把目标顶飞、露出水面终止比赛。所以这份 playbook 只服务三个目标——**现状有据**、**未知的明确标成未知**、**每一步都有可核对的产物**。

## 0. 开工前：建立事实基线（不要跳过）

**这份 playbook 是地图，不是事实。** 代码会变、参数会被调、README 会滞后。任何要落进代码的判断，先用命令读一遍现在的样子：

```bash
cd /home/ansty/RDKX5/auv_vision/auv_vision

# 夹取这条线现在到底有什么（预期：只有 percept/ 四个文件）
find grab -name '*.py' | sort
ls handling/motion handling/handling_task.py 2>&1            # 预期：都不存在

# 装配层有没有接 grab（预期：只有 ball / gate）
grep -n "TASK_CLASS\|TASK_CAM\|TASK_STATE\|register(" main.py

# 配置面（预期：vision.yaml 只有 grab.cv / grab.track；comm.yaml 没有 grab 段）
sed -n '/^grab:/,/^[a-z]/p' cfg/vision.yaml
grep -n "grab" cfg/comm.yaml

# 现有测试与它们实际断言什么（不要只数条数）
python3 -m pytest tests/tasks/grab -q
python3 -m pytest tests/ -q | tail -3

# 串口通道占用（夹爪要落在哪里，先看清谁占了什么）
grep -n "frame:" -A8 cfg/comm.yaml
grep -n "def build_frame_from_dof\|def build_turn_frame\|btn_values" base/hw/uart.py
```

再看一眼**当前的数据与产物**：有没有板端录的夹取素材、`grab_preview/` 里存了什么、`models/` 里有没有为夹取准备的权重。**素材决定你能标定什么**。

## 参考文件（按需读，不要一次全读）

| 文件 | 内容 |
|---|---|
| `reference/task-and-rules.md` | 任务口径（用户确认的"先夹小球、再夹小环"）、每个阶段要回答的问题、**未验证项与待用户确认清单**、判分与安全边界 |
| `reference/code-map.md` | 代码地图与数据流：`handling/percept/` 逐个模块的职责与 API、装配层怎么接、参数落在哪个 yaml、测试在哪 |
| `reference/algorithm.md` | 纯 CV 圆检测六步与"为什么这么做"、为什么夹取不用 BPU、**小环检测该怎么做/不该怎么做**、夹取对准与接近的判据选择、相位机候选形态 |
| `reference/perf-and-params.md` | 板端实测基线表（整帧/ROI/按半径分档）与两个性能坑、全部参数的含义与标定顺序 |
| `reference/verify-and-deploy.md` | 验证阶梯：PC 单测 → 离线喂图扫参 → `preview_detect.py --grab` → 板端实时 → 下水；部署与一致性检查；日志判读 |
| `reference/pitfalls.md` | 已知坑清单与诊断顺序（含项目级纪律与跨工作区只读约束） |

## 1. 四条铁律

1. **只有感知层，别假装有决策层**。`grab/motion/`、`handling/handling_task.py`、夹爪指令、小环感知**都不存在**。写代码前先说清"我在新建哪一层"，并让 `main.py` 保持"决策层没设计完就不占状态机"的现状，直到链路能自证。
2. **不准动 BPU**。夹取走纯 CV；要复用或新训 YOLO 权重是另一个需要用户拍板的决策（成本 + 收益 + 板端实测一起给）。
3. **未知不许编**。小环的颜色/材质/直径/夹取点、夹爪的行程与指令通道、夹取阶段的时间预算——这些当前都是未知。要么去问用户（带选项），要么标成假设并给出标定方案，**不要写一个看起来合理的默认值就往下走**。
4. **参数进 cfg，缺键不崩**。视觉进 `cfg/vision.yaml` 的 `grab.*`，运动/时序进 `cfg/comm.yaml`；新键写注释与默认值，删掉也要靠代码兜底跑通（现有 `Params.from_cfg` / `_cfg` 就是这个模式）。

## 2. 起手式：夹取任务的第一步永远是这个顺序

```
① 证实现状      find/grep/sed + pytest     → 说清"有什么/没有什么"
② 问清未知      ask_user_question（带选项）→ 小环外观/尺寸、夹爪机制、时间预算
③ 拿到素材      板端录一段真实帧           → 没有素材就没有可标定的判据
④ 定判据        圆心/半径/占比/像素误差    → 写死"多近算够近""偏多少算居中"
⑤ 才写检测器    照 reference/algorithm.md 的骨架，先只做检测+可视
⑥ 离线标定      pytest + 逐帧喂图扫参      → 出命中率/耗时表，不达基线不往下走
⑦ 再写决策层    相位机 + 运动链（纸面先过一遍失效模式）
⑧ 板端实测      实时链路 + 日志判读        → 没有板端数据就没有"已验证"
```

**第 ③ 步不能跳**：`grab/README.md` 里的所有数字都来自板端 700 帧真图；换水质、换相机高度、换镜头都要重标。空手写参数等于编数字。

## 3. 当前能力边界（截至 2026-10-06，改代码前复核）

| 层面 | 状态 | 说明 |
|---|---|---|
| 红球纯 CV 检测 | ✅ 有 | `handling/percept/cv_ball.py`：掩膜→填高光→几何过滤→鲁棒圆拟合→覆盖校验→去重 |
| 流式 ROI 跟踪 | ✅ 有 | `handling/percept/ball_tracker.py`：板端把整帧 64 ms 降到 ROI ~14–31 ms |
| 工程接口 | ✅ 有 | `handling/percept/grab_detector.py`：`GrabBallDetector` / `build_grab_backend()` / `circle_to_ratio()` |
| 装配层接入 | ❌ 没接 | `main.py` 只注册 `gate`；`TASK_CLASS` 只有 `ball`/`gate` |
| 夹取决策层 | ❌ 没有 | 相位机、运动链、时限管理全部待建 |
| 小环感知 | ❌ 没有 | 工程与文档里**零记录**，第一步是找用户要实物信息与素材 |
| 夹爪指令 | ❌ 没有 | 11B 帧的 byte8/9 已被转角/取消占用、byte10 是旋转编号；夹爪通道待定义 |
| 测试 | ⚠️ 仅感知 | `tests/tasks/grab` 18 例（全工程 191 例）——决策层一行测试都没有 |

**红线**：不要把这张表里的 ❌ 当成"顺手补一下"就动手。每一项都要先有判据、再有代码。

## 4. 交付与沟通

- 结论先给，依据在后；"已验证"与"推测/未验证"在文字上必须分开写，并给出复现命令。
- 交付物留在工作区，不擅自提交；长跑（逐帧喂图、板端 pytest、录制）用 `nohup ... > log 2>&1 &` 起再轮询日志。
- 破坏性操作（覆盖 cfg 段、删素材、改 `uart.py` 帧结构、动 `main.py` 装配）先讲影响面与回退方式。
- 工作区可能被并发改动：定期用 mtime 与内容复核依赖文件，发现漂移先停下来核对。
