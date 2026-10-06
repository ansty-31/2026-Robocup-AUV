# handling/ — 任务「夹取 + 放置」（**同一个总调度，两种模式**）

> 2026-10-06 整合：原 `grab/`（任务三 夹取）与 `place/`（任务四 放置）合并到本目录。
> 结构（与 `gate/` 同规）：`handling_task.py` 总调度 + `motion/{params,actions,phases}.py` + `percept/`。
> 模式：`HandlingTask(mode="grab"|"place"|"full")`；`full` = 夹取完成后**同一实例交接**给放置。

## 原 grab/README.md

# `grab/` — 任务三「夹取小球」分区

> **现状（2026-10-06 晚）**：感知层 + **决策层都已在**，但**未标定、未装配、未上板**。
> * `percept/`：纯 CV 红球检测（`cv_ball.py`）+ 流式 ROI 跟踪（`ball_tracker.py`）+ 工程接口
>   （`grab_detector.py`）+ 验色读数（`cage_color.py`：下视笼区 ROI 的**目标色占比 percent**）。
> * `motion/axis.py`：指定角度轴动作（抬头/回水平/横倾）+ 下潜到位 + 定时推力。
> * `grab_task.py`：相位机（抬头→扫描→yaw 居中→对准前进→回水平→轻微对准→下压→上升→验色→倒球重来）。
> * **安全闸 `comm.grab.calibrated: false`** ⇒ `GrabTask.ready` 为假 ⇒ 装配层跳过它 ——
>   「决策层没标定完就不占状态机」是机械保障，不是口头约定。**标定完再翻 true。**
>
> 结构照 `gate/` 的规矩：`percept/`（感知）+ 决策层（`grab_task.py` + `motion/`）。
> **感知源是下视相机**（用户 2026-10-06 定：本任务全程用下视），不是前视。
> * 下视**走 USB**（`cfg/vision.yaml camera.down`：usb `/dev/video0`）；文档里"MIPI IMX415"的
  说法已过时，以 cfg 为准。
> * 下视**不做去畸变**（用户 2026-10-06 定：畸变对小球的居中影响不大）⇒ `calibration: null`
  是**有意的**，不是欠账。
> * 任务三**不需要主视** ⇒ 主视挂了与本任务无关；但**下视是本任务唯一来源**，装配时仍要挡
  「下视相机不可用 → 静默回退 sim」那条路（`camera.fallback_sim`）。
> * 目标色：**现在用红球测逻辑与运动**（"红球外的可以都毙了" ⇒ 红掩膜天然只认红）；比赛用
  **粉球 / 黄球** ⇒ 阈值槽位、闸与量法见 §7。

## 0. 决策层速览（口径来自用户 2026-10-06 的口令，逐条落在代码里）

| # | 相位 | 动作 / 通道 | 关键判据 |
|---|---|---|---|
| 1 | `INIT` | 抬**任务级**限深下限到 `grab.depth_floor_m`(1.0m) | 只抬不降；全局 `min_depth_m`(0.55) 不动 |
| 2 | `PITCH_UP` | 先下潜到位，再 `pitch +30°`（byte[7]=2） | 深度够深才抬头；到不了就放弃 |
| 3 | `SEARCH` | `search_scan`（yaw 慢扫，复用公共件） | 检到球 |
| 4 | `CENTER` | **只 yaw**（不前进/不横移/不升降） | `\|dx\| ≤ eps` 连续 N 帧 |
| 5 | `APPROACH` | surge 分级 + sway 修 dx（**同撞球**） | 圆面积占比 ≥ `dip_ratio` |
| 6 | `LEVEL` | `pitch` 回水平：相对角 = `wrap180(抬头前遥测 − 当前遥测)` | 完成标志；**算不出就停手** |
| 7 | `ALIGN` | dx→sway、dy→surge，**两通道都轻微**（抗水波）、**不用 yaw** | `\|dx\|`、`\|dy\|` 各自容差 |
| 8 | `DIP`→`RISE` | heave 下压 → 上浮（**被动收球笼，无执行器通道**） | 时间 |
| 9 | `VERIFY` | 下视笼区 ROI 读目标色 **percent** | 达标 → `grab_ok` |
| 9b | `DUMP` | 右移→横倾倒出→回正→左移→后退→停稳→**重新抬头** | 第 2 次尝试默认对（`retry`） |
| 10 | `EXIT` | **若还仰着头，先放平再结束** | 放平带超时兜底 |

**倒球/放球的区别**：倒球（错球）在本任务内；**放球（横倾 30° + 断动力 3s）属任务四「放置」**
（运输/放球，TBD）—— 本任务验色通过即结束，交接看 `GrabTask.holds_ball`。

## 1. 为什么这条线不用 BPU（2026-10-06 板端实测）

在 RDK X5 上拿撞球那版权重（`yolo11n_detect_bayese_640x640_nv12.bin`，与板端那颗 md5 完全相同）
跑夹取场景的 700 帧，**按球大小分桶看命中率**：

| 球半径 | 帧数 | YOLO 命中 | YOLO conf 中位 |
|---|---|---|---|
| 25–40 px | 46 | **33%** | 0.00 |
| 40–70 px | 120 | **42%** | 0.10 |
| 70–120 px | 6 | 100% | 0.88 |
| 120–200 px | 161 | **99%** | 0.84 |
| 200–300 px | 209 | 90% | 0.49 |

⇒ **远距段（夹取任务一开始就在的那段）那版权重只有 33–42%**。
同一个球用纯 CV：整帧命中 **549/700 = 78.4%**，且**每帧都不占 BPU** ⇒
把 BPU 让给门 keypoint（过门才是真需要神经网络的）。

⚠️ 前提说明：**那版权重本来就不是给夹取任务训的**，这段对比说明的是"能不能顺手复用"，不是
"YOLO 不行"。真要为夹取训一版权重是另一件事（要补远距样本 + 标注 + 量化）。

## 2. 算法一览（代码在 `percept/cv_ball.py`，参数在 `cfg/vision.yaml` 的 `grab.cv`）

| 步骤 | 做什么 | 为什么 |
|---|---|---|
| 红色掩膜 | `R-max(G,B) ≥ dom_min` **或** `(R-max(G,B))/(R+max(G,B)+1) ≥ rel_min`，再叠色相/饱和度/亮度门限 | 水下红光衰减 ⇒ 暗球绝对差掉到阈值以下、比值却仍高，两条判据取并；色相门限把橙色导轨（H≈20–30）挡在外面 |
| 填高光孔 | 只填"面积 < 0.65×父轮廓"的孔 | 镜面高光把红球打成红环，不填则圆度/质心全错 |
| 几何过滤 | 面积 / 圆度 / 凸性 下限 | 滤掉瓦缝红光、池底斑点 |
| 鲁棒圆拟合 | Kasa 拟合 + 迭代剔离群点，边界**张角 ≥150°** | 球被黑瓦条挡成月牙时，月牙的弧仍是真圆的一段；细长条能拟出任意大圆，用张角拦住 |
| 覆盖校验 | 拟合圆内红掩膜占比 ≥0.35 | 全盘≈0.95、被挡月牙≈0.5 |
| 去重 | 圆 IoU>0.6 只留高分 | 同一球被切成几块时会出多个圆 |

## 3. 板端性能与两个坑（都在代码注释里留了因）

**A55 上每一次全图（1280×720）uint8 pass ≈ 2.6 ms（内存带宽受限）**，所以成本 ≈ pass 数 × 2.6 ms：

| 口径 | 板端单帧 | 帧率 |
|---|---|---|
| numpy 参考实现（`fast_mask=False`）整帧 | 87 ms | 11.5 FPS |
| **快掩膜**（`fast_mask=True`）整帧 | 64 ms | 15.6 FPS |
| **流式 ROI 跟踪**（`grab.track.enable: true`）跟踪稳态 | 25.6 ms | 39 FPS |
| 流式整体（含全图搜索/回退帧） | 31.1 ms | 32 FPS |
| 对照：YOLO 只 resize 640 送 BPU | 21.4 ms | 47 FPS |

**按球半径分段**（流式 ROI，板端）：`r≤70 → 14 ms / 71 FPS`；`r 120–200 → 25 ms / 40 FPS`；
`r 200–300 → 31 ms / 32 FPS`。CV 的耗时随球变大而升，YOLO 恒定 —— 这是两者本质差别。

### 坑 1：ARM 上别用 16 位 / 稀疏索引
先写的"uint16 整数等价式 + `np.nonzero` 稀疏 HSV 索引"在 A55 上 **97.6 ms，比不优化还慢**
（`cv2.multiply(dtype=CV_16U)`、`np.nonzero`、花式索引都是标量实现）。
正解：全 uint8 + `cv2.LUT`（相对红占优判据的 LUT **与除法判据逐格等价**，见单测）+ `cv2.inRange`。

### 坑 2：ROI 尺寸公式与纵横比
* ROI 曾写成 `side = 4r+80`（把 gain 乘在半径上又乘 2）⇒ 球半径 188 时 ROI = 832² = 69% 全图，
  实测"ROI 模式反而比全图慢"。现在 `side = 2(r+margin)`，`margin` 默认 64（≈1.6× 实测帧间最大位移 40px）。
* 尺度归一化（`proc_side>0`）必须**按同一比例缩两轴**：ROI 贴画面边缘时会被截成非正方形
  （例 427×360），硬拉成正方形等于改纵横比、圆心回算就飘（单测抓到过 cy 差 52 px）。
  归一化代价实测：中心偏差 p50 0.71 px / p95 1.94 px、半径比 0.998（占球半径 0.38%），夹取足够。

## 4. 怎么接进工程（装配层唯一改动点）

```python
# main.py · AppController.__init__（与 gate 同形，2 行）
if name == "grab":
    self.hub.register("grab", build_grab_backend())      # grab/percept/grab_detector.py
# 之后：hub.detect_list("grab", frame) -> [Det]；要圆心/半径用 detector.circles(frame)
```

**决策层写好之后仍然没动 `main.py`**（`calibrated: false` 就是那道闸）。要接的时候是这几处：
```python
TASK_CLASS["grab"] = GrabTask; TASK_CAM["grab"] = "down"
self.cams["down"] = create_camera("down")           # 或复用 DownCamFeeder + wants_down()
hub.register("grab", build_grab_backend())
TASK_STATE["grab"] = S.STATE_...                    # 见 README §4 的两行
```
⚠️ **下视是 USB 1.1（/dev/video0, 1280×720@20fps）**：主循环直接 read 可能拖慢帧率 ⇒
优先走已有的 `DownCamFeeder` 后台注入（`set_down_frame`/`wants_down`，gate 已用同一机制）。
⚠️ `cfg/vision.yaml camera.down.calibration: null`（**没去畸变标定**）⇒ 圆心/半径带径向误差；
要么补标定，要么把工作点收在画面中心附近（畸变最小）。另外 `main._uart_status` 的 HUD 仍显示
全局 `min_depth_m`；任务级下限接上后应改显示 `uart.effective_min_depth_m`。

**肉眼看效果**（板端，工程根，不发运动指令）：
```bash
python3 preview_detect.py --grab --show                  # 桌面终端
python3 preview_detect.py --grab --stream --save grab_preview/   # 或推流给 PC + 存图
```

**PC 侧离线验证**（无需相机/权重）：
```bash
python3 -m pytest tests/tasks/grab -q                    # 45 例（感知 18 + 运动原语 10 + 相位机 17）
python3 - <<'PY'
import cv2, sys; sys.path.insert(0, '.')
from handling.percept.cv_ball import Params, detect, debug_maps
img = cv2.imread('some.jpg')
print(detect(img, Params.from_cfg()))
PY
```

## 5. 参数（`cfg/vision.yaml` 的 `grab.*`）

| 键 | 默认 | 含义 |
|---|---|---|
| `grab.detect.mode` | `cv` | `mock` = 不注册后端（交回 legacy/mock） |
| `grab.kind` | `red_ball` | 返回的 `Det.kind`，与 `model.labels` 对齐 |
| `grab.cv.dom_min` / `rel_min` | 30 / 0.30 | 绝对 / 相对红占优下限 |
| `grab.cv.s_min` / `v_min` | 40 / 60 | HSV 门限（挡非红亮斑） |
| `grab.cv.close_k` / `open_k` | 9 / 5 | 形态学核（填高光孔 / 去椒盐） |
| `grab.cv.min_area` / `circ_min` / `cov_min` | 30 / 0.30 / 0.35 | 几何与覆盖下限 |
| `grab.cv.min_arc_deg` | 150 | 拟合边界最小张角 |
| `grab.cv.r_min` | 25 | 半径下限（<30px 的检出实测是池底图案红点） |
| `grab.track.enable` | `true` | 关掉 = 每帧全图（15 FPS，便于对照/调试） |
| `grab.track.margin` / `proc_side` / `max_lost` | 64 / 256 / 3 | ROI 余量 / 归一化边长（0=关）/ 放弃 ROI 的丢帧数 |

**删掉任一 key 都用代码同值默认**（`Params.from_cfg()` / `_cfg()` 兜底）⇒ 不改 cfg 也能跑。

## 6. 还没做 / 未验证

- **决策层已写、但全部未标定**：见上面 §0 的相位表；`comm.grab` 里标「占位·未标定」的值（下压/上升的
  秒数与幅值、surge/sway 增益、`dip_ratio`、验色 ROI 与 `red_percent_min`）都要现场实测后才上水。
- **`up_sign` / `dump_sign` 未验证**：pitch/roll 的**物理方向**（抬头/低头、向左倾/向右倾）协议正值
  不保证。上板第一件事：`python3 common/motion/turn_deg.py --axis pitch --deg 5` 看机头往哪边走。
- **`读取 percent` 的口径需确认**：本实现理解为「下视笼区 ROI 里目标色像素占比」；若原意是别的量
  （如某个遥测字段），只需改 `grab/percept/cage_color.py` 一处。
- **粉球/黄球的阈值还没量**（用户 2026-10-06：比赛才是粉/黄；现在用红球测逻辑与运动）：
  槽位与闸已就位（`vision.grab.targets` + `Params.from_target`：未标定 ⇒ 不参与检测，
  **不会拿红球的默认值冒充**），**数值等实拍素材**；⚠️ 黄球的判据形式要重定
  （`R−max(G,B)` 对黄色恒 ≈0 ⇒ 色相带 或 min(R,G)−B，实测后再选）。
- **任务四「放置」已建骨架**（`place/place_task.py`）：运输保持 roll 平衡 → 到点 → 横倾放球
  → 停动力 3s；**「到点」的真实判据待定**（见 `place/README.md`）。放球序列与任务三倒错球
  共用 `common/motion/drop.py`。
- **运动原语不在本包**：轴动作/下潜到位/定时推力 = `common/motion/axis.py`；横倾放球/倒球 =
  `common/motion/drop.py`（任务四共用同一套 —— 依赖规则不许 `place` import `grab`）。
- **修掉一个历史坑（2026-10-06）**：`cv_ball._cfg` / `grab_detector._cfg` 原来传的是
  `"grab.cv.dom_min"`，而 `S.get()` 只认 `vision.`/`comm.` 开头 ⇒ **`cfg/vision.yaml` 的整个
  `grab.*` 段从未生效**（只因 yaml 值与代码默认值相同才没暴露）。现在会真读了，
  `tests/tasks/grab/test_grab_targets.py` 有用例钉住。
- **接相机实时跑未验证**：本文所有秒数都是离线逐帧喂图；实时链路还要算上相机 read 与显示开销。
- **换水质/相机高度要重标**：`dom_min`、`s_min`、`v_min` 对水质敏感（当前工作点是在 AUV 夹球池子这批素材上扫的）。
- **CV 的短板**：球被黑瓦条挡到只剩 1–3 px 宽的红弧时，CV 与 YOLO 都漏（700 帧里约 43–55 帧）——
  这部分要"瓦条边缘 + 轨迹预测"，不是调阈值能解决的。
- **旧源码位置**：本模块的算法与调参过程在另一个仓库留档：
  `RDKX5-YOLOv11n-/experiment/scripts/small_ball/`（`cv_red_ball.py` / `cv_ball_tracker.py`）
  与 `RDKX5-YOLOv11n-/output/preview/small_ball/README.md`（PC+板端完整实测记录）。
  两份代码在 700 帧真图上**逐帧检出完全一致**（命中 549/549、0 差异）——同步时核对过。

## 原 place/README.md

# `place/` — 任务四「放置」分区

> **现状（2026-10-06 晚）**：相位机骨架已写（`place_task.py`），**未标定、未装配、未上板**。
> 安全闸 `comm.place.calibrated: false` ⇒ `ready` 为假 ⇒ 装配层跳过它。

## 1. 它是什么（用户口径）

任务三把球夹起来之后**交棒**给任务四：**运输 → 到点 → 放球**。
* **运输路上保持 roll 平衡**，避免球从被动收球笼里滚出去；
* 到点后：**横倾 30° → 球自动滚出 → 停掉所有动力 3s** → 放置结束。

## 2. 与任务三的关系（结构纪律）

两个任务**并列**，谁也不 import 谁：

```
grab/   = 任务三（感知 percept/ + 决策 grab_task.py）
place/  = 任务四（只有决策 place_task.py；无感知）
     ↓ 两边都只 import ↓
common/motion/axis.py   指定角度轴动作 / 下潜到位 / 定时推力
common/motion/drop.py   横倾放球/倒球序列（任务三倒错球、任务四放球**同一套代码**）
```

依据是 README 的依赖规则：「任务代码只 import `base`/`common` 与同级任务模块；
`main.py` 是唯一装配点」。所以**跨任务共用的动作必须在 `common/`** —— 这也是
`common/motion/drop.py` 从 `grab/motion/` 搬走的原因（`tests/tasks/place` 有一条用例钉着这点）。

交接面：任务三 `GrabTask.holds_ball` → 任务四 `PlaceTask.holds_ball(True)`。

## 3. 相位

| 相位 | 动作 | 判据/出口 |
|---|---|---|
| `INIT` | 抬**任务级**限深下限（只抬不降）+ **回正 roll** | 回正完成 → 进运输 |
| `TRANSPORT` | 前进 + 盯着 roll 偏差，超容差就重新回正（`relevel_min_s` 节流） | **到点**（见 §4）|
| `RELEASE` | `common.motion.drop.BallDropSequence("release")`：横倾 30° → 断动力 3s | 序列 `done` |
| `STOP` | 全 0 保持 → 结束 | — |

## 4. ⚠️ 未定项（不编，等你定）

1. **「到点」怎么判** —— 现在只有两个入口：`comm.place.transport_s` 计时（占位·未标定），
   或装配层/上位机判定后调 `task.arrived()`。**真实判据（视觉？定深？区域？）待定。**
2. **运输用什么通道/速度/要不要感知** —— `transport.surge: 0.25` 只是占位。
3. **「IMU roll 0 = 物理水平」未验证** —— 「保持 roll 平衡」目前按 IMU roll 目标 0° 算
   （`roll_target_deg`）。协议 §2.2 明说 pitch/roll 正负与物理方向需实测 ⇒ 上板先单轴实测
   （`python3 common/motion/turn_deg.py --axis roll --deg 5`）。
4. **横倾方向** `dump_sign` 未验证（与任务三同一个物理方向）。
5. 放球后**要不要自动回正/后退** —— 用户口径是"结束"，所以**不回正不后退**（`release` 步骤表只有两步）；
   要改就改 `comm.motion.drop.release.steps` 或在 `comm.place.drop.release` 覆盖。

## 5. 单独测这个动作（不占状态机）

```bash
python3 common/motion/drop.py --what release --prefix place   # 放球序列（台架/水池）
python3 common/motion/drop.py --what dump    --prefix grab    # 任务三倒错球
python3 -m pytest tests/tasks/place -q                        # 8 例（离线）
```
