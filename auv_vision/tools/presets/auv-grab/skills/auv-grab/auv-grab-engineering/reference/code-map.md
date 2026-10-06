# 代码地图、API 与数据流

> 行号与本文件写作时（2026-10-06）一致；读之前先 `grep -n` 复核，代码是事实。

## 1. 分区结构（`grab/` 照 `gate/` 的规矩分两层）

```
auv_vision/auv_vision/
├── grab/                          # 夹取任务专属代码（**当前只有感知层**）
│   ├── __init__.py                # 刻意不 eager import 子模块 → 便于脱离工程做 PC 侧离线验证
│   ├── README.md                  # 口径 / 板端实测 / 两个性能坑 / 接入方式 / 参数表（改之前先看）
│   ├── percept/                   # 感知层（✅ 已有）
│   │   ├── cv_ball.py             # 纯 CV 红球检测：掩膜→填高光→几何过滤→鲁棒圆拟合→覆盖校验→去重
│   │   ├── ball_tracker.py        # 流式 ROI 跟踪：只跟一个目标，全图搜索与 ROI 检测自动切换
│   │   └── grab_detector.py       # 接工程接口：GrabBallDetector / build_grab_backend() / circle_to_ratio()
│   └── motion/                    # ❌ 待建：运动链与相位机（届时 handling/handling_task.py + grab/motion/）
├── main.py                        # 唯一装配层（TASK_CLASS/TASK_CAM/TASK_STATE，**尚未含 grab**）
├── cfg/vision.yaml                # 视觉参数（`grab:` 段 = detect/kind/cv/track）
├── cfg/comm.yaml                  # 运动/时序参数（`tasks.enabled: [ball]`、`frame:` 11B 帧、`depth_guard`）
├── base/cfg/settings.py           # YAML → S.vision.* / S.comm.*（`S.get(path, default)` + 点号访问）
├── base/hw/uart.py                # 11B 帧组帧与下发；夹爪通道**不存在**，要新增先改这里 + cfg
├── common/vision/detector.py      # Det / DetectorBase / DetectorHub（register / detect_list / detect）
└── tests/tasks/grab/test_cv_ball.py   # 18 例：感知层唯一的测试
```

## 2. 数据流（现在能跑通的那一段）

```
camera (front, 前视)  →  frame(np.ndarray, BGR)
        │
        ├─ hub.detect_list("grab", frame)   # 需要装配层先 register("grab", backend)
        │        └─ backend.detect(frame) → [Det]        # Det.kind = grab.kind（默认 red_ball）
        └─ backend.circles(frame)           # → [Detection]（夹取真正要的：圆心 + 半径）
                 └─ GrabBallDetector.step(frame) → (Detection|None, TrackInfo)
                          └─ BallTracker.step(frame)
                                   ├─ ROI 内 detect()（有上一帧且未跟丢）
                                   ├─ ROI 无可信结果 → 本帧全图 detect()（mode='fallback'）
                                   └─ 没有上一帧 → 全图 detect()（mode='full'）
        ↓
   决策层（❌ 待建）：圆心/半径/占比 → 对准与接近判据 → DOF 指令 / 夹爪指令
        ↓
   uart.send_dof(surge, sway, heave, yaw) → 11B 0xA5 帧 → STM32
```

**注意两套返回值的用途不同**：`Det` 是与工程其它任务对齐的通用形状（bbox + `center`），`Detection` 是夹取算法自己的形状（`cx, cy, r, score, area, circ, cov, arc_deg, border_frac, bbox`）。要半径就走 `Detection`，别用 `Det` 的 bbox 反推。

## 3. 感知层 API（写决策层时要用的）

### `handling/percept/cv_ball.py`

| 名字 | 签名 | 干什么 |
|---|---|---|
| `Params` | `@dataclass`，字段见 §4 | 全部可调项；`Params.from_cfg(prefix="grab.cv.")` 从 cfg 取值（缺项用代码默认） |
| `Detection` | `@dataclass` | `cx, cy, r, score, area, circ, cov, arc_deg, border_frac, bbox`；`.center` → `(cx, cy)`；`.as_dict()` |
| `detect` | `detect(img, p) -> list[Detection]` | 整图**或任意 ROI** 检测；按 `score` 降序，已去重 |
| `best` | `best(dets)` | 取最可信的一个（先 `score`，平手取大的 `r`） |
| `red_mask` | `red_mask(img, p)` | 返回 `(mask, redness)`；`fast_mask=True` 走 uint8 LUT 快路径 |
| `debug_maps` | `debug_maps(img, p)` | 调参可视化用的中间图 |
| `circle_coverage` / `arc_span_deg` / `border_frac` | — | 质量量，调参/诊断时单独用 |
| `fit_circle_robust` | `fit_circle_robust(pts, iters, sigma)` | Kasa 拟合 + 迭代剔离群点 → `(cx, cy, r, n_in, rms, in_pts)` 或 `None` |
| `merge_overlaps` | `merge_overlaps(dets, iou_thr)` | 圆 IoU > 阈值只留高分 |

### `handling/percept/ball_tracker.py`

| 名字 | 签名 | 干什么 |
|---|---|---|
| `BallTracker` | `BallTracker(params, margin=64.0, proc_side=256, max_lost=3, edge_margin=6.0, min_roi=128)` | 单目标 ROI 跟踪器 |
| `step` | `step(img) -> (Detection|None, TrackInfo)` | 喂一帧；`TrackInfo.mode ∈ {'roi','full','fallback'}` |
| `reset` / `stats` | — | 丢轨迹 / 回退率计数（`n, roi, full, fallback, fallback_rate`） |

跟踪的"选谁"判据（`_pick`，决策层不要重复实现）：
1. 丢掉贴着 ROI 边界的候选（`edge < edge_margin + 1/scale`）—— 贴边说明圆被 ROI 切了，圆心与半径都不可信；
2. 在剩下的里面按"相对上一帧的跳动"`jump = |Δ中心| / 上一帧 r` 取最小；优先取 `jump ≤ 2.0` 的；
3. `max_lost` 帧都没有 → `prev = None`，下一帧回全图。

### `handling/percept/grab_detector.py`

| 名字 | 签名 | 干什么 |
|---|---|---|
| `GrabBallDetector` | `GrabBallDetector(params=None)` | `DetectorBase` 子类；`kind` 取 `grab.kind`，`track_enable` 取 `grab.track.enable` |
| `.detect` | `detect(frame) -> list[Det]` | 工程接口（bbox 形状） |
| `.circles` | `circles(frame) -> list[Detection]` | 夹取要的圆心/半径 |
| `.step` / `.last` / `.reset` / `.stats` | — | 流式接口：一帧一次，拿最优目标与跟踪状态 |
| `build_grab_backend` | `build_grab_backend()` | 装配层用；`grab.detect.mode: mock` 时返回 `None`（交回 mock/legacy） |
| `circle_to_ratio` | `circle_to_ratio(d, fw, fh) -> float` | 按**圆**算面积占比（与撞球 bbox 占比同量纲，但不会高估 4/π） |

## 4. 参数落在哪里（改参数前先看这里）

`cfg/vision.yaml` 的 `grab:` 段（当前只有这些键；代码里 `Params` 有更多字段，cfg 没写就用代码默认）：

| 段 | 键 | 代码默认 |
|---|---|---|
| `grab.detect` | `mode: cv \| mock` | `cv` |
| `grab` | `kind`（返回的 `Det.kind`，与 `model.labels` 对齐） | `red_ball` |
| `grab.cv` | `dom_min / rel_min / s_min / v_min` | 30 / 0.30 / 40 / 60 |
| `grab.cv` | `close_k / open_k` | 9 / 5 |
| `grab.cv` | `min_area / circ_min / cov_min` | 30 / 0.30 / 0.35 |
| `grab.cv` | `min_arc_deg / r_min` | 150.0 / 25.0 |
| `grab.track` | `enable / margin / proc_side / max_lost` | true / 64 / 256 / 3 |

`Params` 里**cfg 没暴露但代码存在**的字段（要调就加进 cfg，`from_cfg` 会自动取）：`use_rel, use_hue, h_lo, h_hi, h_hi2, fill_hole_frac(0.65), max_area_frac(0.75), hull_fill_min(0.55), r_max(900), score_min(0), fit_iters(6), fit_sigma(1.8), merge_iou(0.6), refine(False), hough_param2(28), hough_r_tol(1.45), fast_mask(True)`。`BallTracker` 的 `edge_margin(6.0)` / `min_roi(128)` 目前**只能**从代码传参，cfg 里没有对应键。

## 5. 装配层怎么接（`grab/README.md` §4，两行）

```python
# main.py · AppController.__init__（与 gate 同形）
if name == "grab":
    backend = build_grab_backend()          # from handling.percept.grab_detector import build_grab_backend
    self.hub.register("grab", backend)
    if backend is None:
        print("[MAIN] ⚠️ grab 后端未启用（grab.detect.mode: mock）")
```

要真正作为任务跑，还需要一起加：`TASK_CLASS["grab"] = GrabTask`、`TASK_CAM["grab"] = "front"`、`TASK_STATE["grab"] = S.STATE_*`（`main.py:35-38` 那三行）。**但状态机接入的前提是决策层已经设计好并通过离线验证**——现在还不满足。

## 6. 相关公共件（改动前想清楚影响面）

| 文件 | 与本任务的关系 |
|---|---|
| `common/vision/detector.py` | `Det`（`kind/score/x/y/w/h` + `.center`）、`DetectorBase`、`DetectorHub`（`register` / `detect_list` / `has_extra` / `ready`）；同帧缓存按帧对象去重，别绕过 hub 自己推理 |
| `base/cfg/settings.py` | `S.get(path, default)` + `S.vision.*` / `S.comm.*`；工程根自动定位 |
| `base/hw/uart.py` | `send_dof(surge, sway, heave, yaw, force=False)`、`build_frame_from_dof`、`build_neutral_frame`、`build_turn_frame(angle, id)`；`frame.axis_count=7`、`frame.aux_axis: {4: 165, 5: 1, 6: 1}`、`frame.btn_values: [0, 0, 0]` |
| `base/hw/telemetry.py` | 下位机 14B 遥测（深度/姿态）→ 限深保护；夹取若要用深度/姿态做判据，从这里取 |
| `task/ball.py` | 撞球相位机（`SEARCH→CENTER→APPROACH→DASH→STOP`）——**可参考结构、不可照抄动作**（它是撞不是夹） |
| `gate/gate_task.py` + `gate/motion/` | 本项目最成熟的"相位机按方法簇拆分"范例（params / channels / modes / exits），新建 `grab/motion/` 时值得照着分文件 |
| `tools/analyze/log/analyze_task_log.py` | 逐帧日志（`AUV_TASK_LOG=<path>`）的离线判读工具 |

## 7. 测试与自检入口

```bash
cd /home/ansty/RDKX5/auv_vision/auv_vision

python3 -m pytest tests/tasks/grab -q          # 18 例（感知层：检测 + 跟踪 + cfg 兜底）
python3 -m pytest tests/ -q                    # 全工程 191 例

# 板端看效果（不发运动指令）
python3 preview_detect.py --grab --show                    # 桌面终端
python3 preview_detect.py --grab --stream --save grab_preview/   # 推流给 PC + 存图

# PC 侧离线验证（无需相机/权重）
python3 - <<'PY'
import cv2, sys; sys.path.insert(0, '.')
from handling.percept.cv_ball import Params, detect, debug_maps
img = cv2.imread('some.jpg')
print(detect(img, Params.from_cfg()))
PY
```
