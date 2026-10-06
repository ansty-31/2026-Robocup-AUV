# 验证阶梯、上线与日志判读

> 原则：**能用离线证据回答的问题，不要拿船去试**。每一级都要有明确的通过判据与产物；跨不过去就停在这一级。
> ⚠️ 本文件描述的"上板/下水"步骤属于**实船动作**，执行前必须让用户拍板。

## 1. 六级验证阶梯（从便宜到昂贵，逐级通过）

### ① PC 单测（秒级，无相机无权重）

```bash
cd /home/ansty/RDKX5/auv_vision/auv_vision
python3 -m pytest tests/tasks/handling -q      # 感知层 18 例
python3 -m pytest tests/ -q                # 全工程回归（当前 191 例）
```

- `tests/conftest.py` 用 `setdefault("AUV_SIM_MODE", "1")` ⇒ 测试里 `S.SIM_MODE=True`，**串口只打印，不会真驱动电机**。决策层的测试同样应该在这个前提下写（运动指令用 mock/记录型 uart 断言）。
- **通过判据**：全绿；新增功能的用例必须"会咬"——在内存里改一个 cfg 值能让它变红。

### ② 离线逐帧喂图（分钟级，无相机）

```bash
python3 - <<'PY'
import cv2, sys, time, glob; sys.path.insert(0, '.')
from handling.percept.cv_ball import Params, detect
p = Params.from_cfg()
files = sorted(glob.glob('素材目录/*.jpg'))
ok = t = 0
for f in files:
    img = cv2.imread(f)
    t0 = time.perf_counter(); dets = detect(img, p); t += time.perf_counter() - t0
    ok += bool(dets)
print("命中 %d/%d = %.1f%%  |  平均 %.1f ms/帧" % (ok, len(files), 100.0*ok/max(1,len(files)), 1000*t/max(1,len(files))))
PY
```

- **按目标大小分桶统计**（远距段才是难点），不要只看总数（见 `perf-and-params.md` §5）。
- **通过判据**：与 `grab/README.md` 记录的基线同量级（整帧 78.4%、快掩膜 15 ms 级），或明确说明为什么不同（换了素材/水质/目标）。

### ③ 板端预览（不发运动指令，最安全的上板第一步）

```bash
# 板端，工程根
python3 preview_detect.py --grab --show                       # 桌面终端看圆 + 圆心 + r/score
python3 preview_detect.py --grab --stream --save grab_preview/ # 推流给 PC 并存图
```

- 这一条**不发任何 DOF**，是"检测在真相机链路上到底行不行"的第一手证据。
- **通过判据**：肉眼 + `r/score` 与离线一致；记下实时链路的实际帧率（离线 25.6 ms 不含相机 read 与显示）。

### ④ 板端实时指标（仍不发运动）

在板端逐帧记录 `step()` 的 `TrackInfo.mode` 与 `stats()['fallback_rate']`、耗时 p50/p95。用后台跑 + 轮询日志：

```bash
nohup python3 your_bench.py > /tmp/grab_bench.log 2>&1 &
tail -n 40 /tmp/grab_bench.log ; grep -nE "fallback|p95|FPS" /tmp/grab_bench.log
```

- **通过判据**：实时帧率满足决策层的控制周期要求；`fallback_rate` 不长期偏高。

### ⑤ 台架闭环（`AUV_SIM_MODE=1`，只打印不驱动）

决策层写完之后的第一轮集成：完整跑一遍相位机，**串口只打印**，看状态迁移、时限、丢目标兜底是否按设计走。

```bash
AUV_SIM_MODE=1 python3 main.py --task <grab 任务名>
```

- 若要看逐帧日志：`AUV_TASK_LOG=<path> python3 main.py --task ...`，然后 `python3 tools/analyze/log/analyze_task_log.py <path>` 离线判读。
- **通过判据**：相位迁移与设计一致；超时/丢目标/夹空各种异常路径都走得到且收尾到全 0 停。

### ⑥ 真实驱动（`AUV_SIM_MODE=0`，⚠️ **必须用户拍板**）

```bash
AUV_SIM_MODE=0 python3 main.py --task <grab 任务名>
```

下水前必查：
- `SIM_MODE` 当前值到底是 True 还是 False（`base/cfg/settings.py` 本地默认 `True`，即只打印；**不加 `AUV_SIM_MODE=0` 就不会动**）。板端这份文件 `deploy` 会**整份跳过**，两边不互相覆盖 ⇒ 上板后要单独确认。
- `comm.depth_guard.enable: true` 且 `min_depth_m: 0.55`（**现场定死，不准改**）。
- 夹取阶段的时间预算已设，且到点有确定收尾动作。
- 夹爪机制已确认（能发指令 / 何时闭合 / 行程耗时），见 `task-and-rules.md` §2.2。
- 手动兜底可用（`manual.sh` 遥控桥），出问题能接管。

## 2. 部署与一致性（板端同步）

```bash
bash tools/deploy/deploy_to_board.sh          # 本机 → 板端：按清单增量同步 + 板端自检
bash tools/deploy/check_board_parity.sh       # 本地 vs 清单 vs 板端逐文件 md5
bash tools/deploy/check_board_parity.sh --board   # 含板端实际文件的双向比对
```

- **`tools/deploy/board_parity.md5` 里的路径就是部署契约** —— 新增源文件（比如 `grab/motion/*.py`）如果不在清单里，**不会**被同步上板。新增文件后先更新清单。
- `cfg/*.yaml` 里仍有台架分叉：**不要整份推板端**（README 明确写过这条），改参数时想清楚推哪一份。
- `base/cfg/settings.py` 是**整份跳过**的：板端与本地各留自己的 `SIM_MODE`。

## 3. 日志与判读

| 入口 | 用途 |
|---|---|
| `AUV_TASK_LOG=<path> python3 main.py --task ...` | 逐帧 JSON 日志（相位/动作/量测） |
| `python3 tools/analyze/log/analyze_task_log.py <path>` | 逐帧判读①~⑧ |
| `python3 tools/analyze/log/feature_coverage.py` | 这轮哪些功能/相位根本没被用到（✔✘ 清单）—— 写新相位机后特别有用 |
| `BallTracker.stats()` | `n / roi / full / fallback / fallback_rate`：ROI 策略是否健康 |
| `TrackInfo.mode` | `roi / full / fallback`：这一帧走的哪条路 |
| `tools/analyze/calib/*` | 标定（`pnp_calib.py` / `ruler_calib.py` / `tape_ticks.py`）：夹取的"像素 ↔ 距离"标定可以在这里落 |

## 4. 交付前清单（每次改动收尾都过一遍）

- [ ] `python3 -m pytest tests/ -q` 全绿；新增行为有"会咬"的用例。
- [ ] 新参数进了 cfg（`vision.yaml` 的 `grab.*` / `comm.yaml`），删掉该键仍能跑（代码兜底）。
- [ ] 新增的源文件已加进 `tools/deploy/board_parity.md5` 清单（否则上不了板）。
- [ ] 结论里明确区分"已验证（附命令与数字）"与"未验证/假设（附标定或实测方案）"。
- [ ] 破坏性或不可逆动作（改 `uart.py` 帧结构、动 `main.py` 装配、改 `settings.py`）已在答复里写清影响面与回退方式。
- [ ] 预览图/录制短片/标定报告这类交付物已 `present` 给用户（若产生了）。
