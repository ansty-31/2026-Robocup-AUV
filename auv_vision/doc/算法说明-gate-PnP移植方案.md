# gate 任务 YOLO-Keypoint(门框四角) + PnP/反投影位姿策略 — 移植方案 v1.1

> 目标：把当前 gate 策略（"整门 bbox 面积估算距离 + bbox 框心对中 + 大框消失计数"）
> 升级为 BumblebeeAS（NUS 水下机器人）思路：
> **YOLO keypoint 识别门框四角 → 已知门几何角点 → PnP 解 6-DoF 位姿 → 门平面反投影得到 3D 瞄准/深度 → 位姿闭环过门**。
> 参考仓库：
> - https://github.com/BumblebeeAS/pose_estimator （算法源码）
> - https://github.com/BumblebeeAS/release （技术文档，RoboSub2025 各任务说明）
> 本文档 = 移植指导书：素材/打标 → 算法 → 代码框架 → 部署。**核心是算法移植。**

## 0. 已锁定的决策

| 项 | 决策 |
|---|---|
| 识别模型 | **ball 与 gate 权重分开**：ball 沿用现有 bbox 权重；gate 用 **keypoint 模型（门 + 4 角点）** 重训 |
| gate 模型输出 | 每实例一个"门"检测，带 **4 个 keypoint = 门框外轮廓四角**（已定，§2.2），顺序固定 TL,TR,BR,BL（与 §3 的 3D 表同序）；标注**允许只标可见角**（缺角 visibility=0） |
| OBB / 部件 bbox | **均不采用**：OBB 角点是外接矩形角、非物理点，不能做 PnP 对应；四角 keypoint 是更直接的对应前端 |
| 特征匹配 | **暂不实现**（备而不启）：思路已记录为文档备忘（§2.4）；将来做有纹理目标（动物图靶板等）再启用 |
| 算法载体 | Python `auv_vision`（cv2 可用）；C++ `auv_vision_cpp` 后续同步 |
| 门物理尺寸 | 已知/可实测 → 填 §3 参数表 |
| 推理后端 | RDK X5 OE `.bin`（hbm_runtime）；解码需扩展 keypoint 头（§5.1） |

## 1. 移植的五个核心算法点（Bumblebee → 本项目）

| # | Bumblebee 的做法（源码文件） | 移植到本项目后的形态 |
|---|---|---|
| A | 位姿估计 = PnP：`solvePnPRansac(SQPNP)` + `solvePnPRefineVVS`（`utils/pose_estimator.py`） | 平面四点用 `cv2.solvePnP(flags=SOLVEPNP_IPPE)` + `solvePnPRefineLM/VVS`；3 点/混合时用 P3P/SQPNP |
| B | 相机对象 `PinholeCamera`（K+畸变 / rectified K 双模式）（`utils/PinholeCamera.py`） | `CameraModel`：读 `front_camera.yaml`；按"原始域/去畸变域"给出 K/D（§4.2，**易错点**） |
| C | 2D↔3D 对应：掩码→best-fit 四边形→`match_polygon_points`（`utils/detections.py`、`config/object_points.py`） | **keypoint 自带语义与顺序，零匹配**：4 角点 = 训练时固定的 TL,TR,BR,BL，直接与 3D 表同序配对 |
| D | 反投影：`backproject_pixel` + "物在已知平面上→取平面深度"（trash 借桌位姿定深） | 门位姿 (R,t) → 门平面方程 → 瞄准像素反投影到门平面得 3D 点 → 位置误差闭环 |
| E | 深度=距离的用法：PnP 的 `tvec.z` 即深度 | 前进量由 `Z=t.z` 分档（替代 bbox 面积占比）；穿门 = 过 Z 阈值后目标消失 |

辅助移植项（可选）：位姿 EMA / 协方差（`estimate_covariance`、`pose_weighted_average`）、
单应滤波（`filter_by_homography`）、front/back 前后侧参数（换 object_points 符号）。

## 2. 素材与打标准备

### 2.1 采集
- 板端前视 USB 相机实拍（**与最终部署同款镜头/安装位/1280×720**）：0.5~8 m 分距、不同偏航/俯仰、逆光/顺光、气泡/雾感；
- **数据集与推理必须同域**：标注/训练帧必须经过与板上推理**同一条**链路——
  `undistort(remap) → enhance(WB→CLAHE→gamma) → 640 缩放`；该链路与 ball 共用、已对齐，gate 不新增；
  录帧后离线过同一链路存 PNG 再标注即可（keypoint 对域偏移比 bbox 更敏感，别跳步）；
- 记录与预处理代码：`recorder.py` / `preprocess.py` **均不改**（§7.1）。

### 2.2 keypoint 标注（已定：标"外轮廓角"；允许只标可见角）
- **角的物理定义（已定）**：标门框**外轮廓角点** = 两相邻管外沿交点处的像素（labeler 最一致、
  训练最稳）；3D 表用对应**外矩形实测值**（§3 `frame_w/frame_h`）。若门框管径左右/上下对称，
  外矩形中心 = 开口中心 → 瞄准/穿门数学不变（管径不对称时见 §3 备注补偏移）；
- **4 个 keypoint 顺序固定 TL,TR,BR,BL**（左上→右上→右下→左下，y 向下），训练配置与 §3 表同序声明；
- **允许部分标注**：某一角被挡/出画/太糊时，**只标可见角**，不可见角标 `visibility=0`
  （ultralytics keypoint 标签 `x y visibility`：0=缺席 / 2=可见）。理由：
  ① 训练让模型学会"缺角是正常输入"，与运行时降级链（§4.3/§4.6：1~3 角各有模式）对齐；
  ② 不要硬猜被挡角的位置——猜错会污染角点回归；
  ③ 若一帧里一个角都不可靠，则该帧只当 `gate` 框用或弃标（coarse 档够用）；
- 每点自带可见度/置信度 → 推理时按 §4.6 仲裁降级，不强行外推。

### 2.3 数量与工具
- 实例数建议 **≥500~1000**（每实例 4 角）；难例：大偏航、部分遮挡、近距角点出画、弱对比；
- 工具：CVAT / LabelImg（keypoint 导出格式与训练工程 RDKX5-YOLOv11n 兼容）；
- 评估：按"接近过程连续 ≥3 角可用帧率"与"四角重投影对齐度"看，别只看 mAP。

### 2.4 前端选型结论（防止反复）
```
keypoint(四角) —— 主前端：素色门框也稳、BPU 单前向、顺序即对应 → PnP
特征匹配(XFeat) —— 暂不实现（备而不启）：门素色无纹理点、水下光照不稳，做主前端风险高
OBB            —— 不采用：外接矩形角 ≠ 物理角，无法做稳定 3D 对应
```

**特征匹配思路备忘（暂不实现，仅记录）**：不训练任何 YOLO，用"模板图 ↔ 实时帧"的局部特征匹配
直接产生点对应，喂同一个 PnP 内核（Bumblebee torpedo/bin 路线）：
1. 准备**模板图**：正对目标拍一张（或 CAD 渲染），图上预先标注/标定每个特征点对应的 **3D 点**；
2. 提取器（XFeat/SIFT）从模板与实时帧各提取关键点+描述子，最近邻匹配（比例测试去歧义）；
3. 每个匹配对 = 1 个 2D(实时帧)↔3D(模板) 对应 → 通常**几百个点** → RANSAC 的
   `solvePnPRansac(SQPNP)` + 精调 = 抗遮挡、抗误匹配；
4. 优：零训练、泛化到任意目标（换模板即可）、点数多 → PnP 天然稳；
   劣：要求目标**有纹理**（素色 PVC 门特征稀疏）、对光照/视角/尺度变化敏感、X5 CPU 跑特征提取延迟要预算；
   多模板时还要"模板选择"（Bumblebee 用 toggle_template 服务、bin 用旋转/不旋转模板计数消歧）。

## 3. 门几何参数表（object_points，单位 mm，填实测值）

约定：原点 = **门框矩形中心**（对称时 = 开口中心），x 向右、y 向下、z 向 AUV（门平面 = z=0，
法向 +z 朝 AUV）。角点顺序 TL,TR,BR,BL —— **与 keypoint 训练顺序一致**。

```yaml
# cfg/vision.yaml 新增段（结构：闭合矩形门框 = 左右立柱 + 上下横梁，悬空不触地）
gate:
  geometry:
    frame_w: 0.70     # 外轮廓宽（已定 70 cm）
    frame_h: 0.50     # 外轮廓高（已定 50 cm）
    bar_width: 0.0    # 管/梁外宽（可选：开口净宽/中心换算与目视核对用）
    sym_bars: true    # 四边管径/位置对称? false 时外矩形中心≠开口中心, 控制侧补偏移
    # 前后完全对称、无朝向/对准要求 → 不做 from_front 切换(恒正面语义, 见 §3 备注)
    body_center_offset: 0.0   # 已定 = 0.0：相机体轴位置≈机身中心(穿越判据退化为相机平面穿越即过门);
                              # 若实机相机在机头(>0)再改, 语义=前视相机→机身中心沿体轴距离
  # 两扇门(高低门)同尺寸、仅中心离底不同：矮门 0.45 m、高门 0.65 m → 底梁离底 ≈ 0.20/0.40 m
  # 竖直对准无需外部高度表：位姿的 t_y 直接给"门中心相对相机轴"的竖直偏差 → heave 闭环
```

```python
def object_points(W_m, H_m, from_front=True):
    """门框矩形四角 3D 点 (N,3)，z=0 平面，顺序 = keypoint 顺序。"""
    s = 1.0 if from_front else -1.0
    pts = [(-s*W_m/2, -H_m/2, 0.), (s*W_m/2, -H_m/2, 0.),
           ( s*W_m/2,  H_m/2, 0.), (-s*W_m/2,  H_m/2, 0.)]   # TL,TR,BR,BL
    return np.asarray(pts, dtype=np.float32)
```
> 备注：结构已定 = **闭合矩形门框（左右立柱 + 上下横梁），悬空不触地** —— 四角都是真实管外角，
> 直接标注/量测，**不做任何地面/相机高度假设**（无"底梁触地"、无 floor 反推）。
> 已定标"外轮廓角"，3D 表按外矩形实测（`frame_w × frame_h`，外沿）。若四边管径/位置不对称
> （外矩形中心 ≠ 开口中心），控制侧补"开口中心相对外矩形中心"偏移（`sym_bars: false`）。
> 四角必须在同一平面（门平面），否则 §4.5 平面反投影失效——建表后做 §6 的往返自检。
> **无朝向/对准要求**：门前后完全对称（平面矩形框），正面/背面视图与位姿（平面+中心）相同 →
> 不需要 `from_front` 前后侧区分（`object_points` 参数保留、恒 true）；任务完成 = **机身穿过开口**，
> 完成判据见 §4.5/§5.3（用 `body_center_offset` 判定机身中心越过门平面）。

## 4. 算法规划（移植核心）

### 4.1 总体数据流

```
前视帧 ──(undistort+enhance)──► [gate keypoint 模型] ──► 门检测 + 4 角点(px + conf)
                                                                 │
                       §4.3 keypoint 解析: 同域全尺寸坐标, conf 门限, 排序
                                                                 ▼
                    img2 (N,2)  ↔  obj3 (N,3)  (§3 同序查表)
                                                                 ▼
        §4.4 PnP(4点 IPPE 主 / 3点 P3P 备) → rvec,tvec → R,t
                                                                 ▼
        §4.5 门平面方程 → 反投影瞄准/位置误差 →(sway,heave,yaw)
              深度 Z=tvec.z → surge 分档(替代面积占比)
                                                                 ▼
       相位机 SEARCH→RANGE_ALIGN→APPROACH→THROUGH(PASS 计数)→下一门/DONE
```

### 4.2 坐标系与相机模型约定（易错，先定死）
- 检测/keypoint 坐标在"喂给 preprocessor 的整帧"域（模型输入解码后按 `(fw/640, fh/640)` 还原）。
  `image.undistort=true` 时该域是 **remap 后无畸变帧**（尺寸仍 1280×720）；否则是原始畸变帧；
- **PnP/反投影与检测坐标同域，K 取对应域**：
  - `undistort=true` → rectified K（`getOptimalNewCameraMatrix(..., alpha=0)` 的 new_K），D=0；
  - `undistort=false` → 原始 K（fx=782.54, fy=779.79, cx=633.80, cy=386.91 @1280×720）+ 原始 D；
- 实现为 `CameraModel`（对应 Bumblebee `PinholeCamera`）：`camera_matrix()/dist_coeffs()/rectified 工厂`，
  由 `image.undistort` 决定加载哪套；
- 图像 y 向下与世界 y 向下一致（§3 约定，OpenCV 惯例）。

### 4.3 前端：keypoint → img_pts（单层解析，无多边形/匹配代码）

```python
def keypoints_to_img_pts(det, w, h, kpt_conf_thr, order=("TL","TR","BR","BL")):
    """det: 门检测(含 kpts (4,2) 全尺寸像素, conf (4,)) → 通过门限的点对。"""
    ok = det.kpt_conf >= kpt_conf_thr
    idx = np.where(ok)[0]
    if idx.size >= 4:
        return det.kpts[idx[:4]], "full"       # 顺序即 §3 索引
    if idx.size == 3:
        return det.kpts[idx[:3]], "p3p"        # 缺一角 → P3P(消歧见下)
    if idx.size == 2 and _opposite(idx):        # 对向角(TL,TR 或 BL,BR)
        return det.kpts[idx[:2]], "width_range"  # 宽度测距(§4.6)
    return None, "coarse"                       # 角不足 → 用检测框粗对准
```
- **顺序由训练固定**，运行时不需要 match；只做一次"四角围成凸四边形"合理性校验（防回归乱序）；
- keypoint 坐标从模型域还原到全尺寸帧时与 bbox 同一缩放（§4.2）；
- 建议质量闸门：`四角重投影面积/检测框面积` 合理性 + `score` 门限。

### 4.4 位姿解算

```python
def gate_pose(camera, obj3, img2, mode, prev=None):
    K, D = camera.camera_matrix(), camera.dist_coeffs()
    if mode == "full":            # 4 共面点：封闭解(快稳), 两解取 t_z>0 & 重投影小者
        ok, rvec, tvec = cv2.solvePnP(obj3, img2, K, D, flags=cv2.SOLVEPNP_IPPE)
    elif mode == "p3p":           # 3 点歧义大 → 用上一帧姿态选最近解; 无 prev 则弃帧
        ok, rvec, tvec, _ = cv2.solvePnPRansac(obj3, img2, K, D,
            iterationsCount=300, reprojectionError=cfg.pnp.reproj_px,
            flags=cv2.SOLVEPNP_P3P)
    else:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj3, img2, K, D, rvec, tvec)
    return rvec, tvec
```
- 质量检查：重投影 RMS < 阈值（初给 5 px）、`Z=t[2]∈[0.2,15]m`；不合格弃帧保持上一帧输出；
- 门**前后完全对称且无朝向/对准要求** → `from_front` 不做切换（参数恒 true、代码零分支，
  同 Bumblebee `GATE_FRONT/BACK` 思路但此处不需要）；
- 平面四点 + 对称矩形的"正视/背视"镜像歧义：IPPE 两解仍按 `t_z>0` + 与上一帧姿态最接近选，
  保证门平面始终在机前（左右镜像因前后对称而不存在任务差异）。

### 4.5 门平面方程与反投影（瞄准/定深的真正"反投影"）

```
R = cv2.Rodrigues(rvec)[0]
门平面法向(相机系)  n_c = R·[0,0,1]^T          # 门平面 z=0, 法向朝 AUV
平面方程            n_c·p_c = n_c·t =: ρ
像素反投影到门平面  p_c = λ·K⁻¹[u,v,1]^T,  λ = ρ/(n_c·K⁻¹[u,v,1]^T)
开口中心 3D(相机系) = t                          # 原点在门矩形中心
```
用处：
- **对准误差**：`sway_err=t[0]`、`heave_err=t[1]`（符号按实车 dof_map 复核）→ PID → DOF sway/heave；
- **偏航误差**：法向与光轴夹角（符号按实车定）→ yaw 修正，正对开口；
- **深度**：`Z=t[2]` → surge 分档（远快近慢）；
- **穿过判据（机身过门）**：门无朝向要求，穿越判定用**几何穿越 + 消失确认**双保险：
  ① 主判据：相机平面穿越后继续直行，到 `Z ≤ -body_center_offset`（机身中心越过门平面）→ 判过门；
  ② 兜底判据：`Z` 已过 `Z_pass` 后连续 K 帧 keypoint 失效/消失（原"大框消失"），同样判过门；
  任一触发 → 计数+1，进入下一门/SEARCH。

### 4.6 降级链（角点不足/目标不完整时仍能干活）

| 可用输入 | 模式 | 深度/朝向来源 |
|---|---|---|
| 4 角全可见 | **full**：IPPE | Z=t.z；yaw 由 R |
| 3 角可见 | **p3p**：P3P + 上帧消歧 | 同上（角缺失侧误差大，进近后期慎用） |
| 仅对向 2 角(TL,TR / BL,BR) | **width_range**：角距测距 | Z≈fx·W/|u_TR−u_TL|；瞄准点=两角中点；yaw 不可估 |
| 仅 `gate` 框(角不足) | **coarse**：框中心对中（运动按"缺角仲裁"分档，非一律前进） | 无深度；前进仅限远距小框（见下） |
| 无目标 | **search**：旋转搜索 | — |

主循环（对应 Bumblebee `gate_pose_estimator_node.detections_callback`）：
```
每帧:
  det = 门检测(带4 kpt); 若无 → search
  img2, mode = keypoints_to_img_pts(det, ...)
  if mode in (full, p3p):
     rvec,tvec = gate_pose(...); 质量校验通过 → EMA(t_x,t_y,Z)
     瞄准误差→PID→send_dof(surge_by_Z, sway, heave, yaw)
     Z<Z_pass 起算穿过窗口
  elif mode == width_range: 测距 + 中点对中 + 慢速前进
  else: coarse → 按"缺角仲裁"分档运动（见下）
  穿过计数 ≥ pass_target → DONE
```
> 帧间连续：本帧失败/丢目标用上帧位姿保持 N 帧（防抖），N 超限才回 search。

**缺角时的运动仲裁（对准阶段禁止盲目前进）**：只有两件事值得"前进"——① 远距小框，需要
靠近让角点可辨（靠近本身=信息增益）；② `Z < Z_pass`，已进入穿门窗口。其余一律不前进；
缺角是"信息不足"，正确反应是**用不盲目的动作换信息**：

```
角不足(非 full/p3p)时，按 门框占屏比 与 是否有过好位姿 分档:
┌ 远(框占比小, 从未出好位姿):  慢速 creep + 框心对中(surge 限量)
│     → 靠近使门变大, keypoint 可辨 → 进入 golden window
├ 中(占比中等):                surge=0; 用 框心/两角中点/上帧位姿 持续对中
│     → 连续 ≥K 帧仍缺角(≥2角/整框不可信) → 进 REACQUIRE
├ 近(占比大, 角出画/画面装不下): surge=0 → 直接 REACQUIRE(前进只会更糟)
└ search(无整框):            旋转扫描(既有逻辑)

REACQUIRE(信息重取): 短时后退(surge<0, 限幅+限时) 或 朝"可见侧"小幅转向
  · 目的: 拉大距离 → 整门重进画面 → 4 角重现 → 回 RANGE_ALIGN
  · 若只能看到"一侧柱/对向 2 角": 转向应朝向可见侧(把它拉回画面中心)，
    不要无信息旋转(那会把仅有的目标转丢)
  · 超时/后退超限 → 回 search; 全程保持心跳与急停语义

穿过例外: 一旦 Z < Z_pass(已进入穿门窗口), 上述一切让位于"保持姿态直行穿越"。
```

### 4.7 少于 4 个点时，Bumblebee 到底怎么做？（本节是移植要点）

源码事实（`points_pose_estimator_node.py`、两个 trash 节点）：

1. **点数不足 → 不发位姿，绝不硬解**。`points_pose_estimator` 对 `<4` 个对应直接
   `warn + return`；`get_object_pose` 对 `<4` 点直接 `ValueError`。他们从不拿 2~3 点硬凑 6-DoF。
2. **把自由度拆开估，不追求一次 PnP 出全量**。trash 里每个物体只有"质心 1 点"：
   `object_depth = 桌位姿.z`（桌子由 4 角 PnP 求得）→ `get_detection_centroid_position`
   （= `backproject_pixel(质心, Z)`）给出 **位置**；**yaw 从图像 OBB 角度 / 桌 yaw ±90° 推算**；
   pitch/roll 置 0（结构假设）→ 照样发布完整 TF。
   ⇒ 即"**1 个像素 + 参考面/参考系深度先验 → 3D 位置；朝向另找图像/参考系来源**"。
3. **刚性/共面先验就是约束**。门是刚体矩形：少 1 角 ≠ 丢一个未知数，而是"第 4 角由矩形几何决定"
   → 3 角可走 P3P（≤4 解），用"上帧姿态最近 + 第 4 角重投影落回检测框内"消歧；
   少到只剩**对向 2 角**（TL,TR 或 BL,BR）时，已知实宽 W → `Z≈fx·W/Δu`
   （进近段 yaw≈0，近似好）——这正是 README"平面目标靠几何把点压稳"的思路。
4. **RANSAC 的两种用法别搞混**（README 两 case + 代码注释）：
   - 平面目标（门就是）：**homography 能滤就先滤**，RANSAC 只给初值；不按 RANSAC inliers 严格过滤
     ——点少时 RANSAC 太苛刻 → 剩余点太少 → 位姿抖（`points_pose_estimator` 注释原话）；
   - 非平面/单应失效：才退化为"只用 RANSAC inliers"。
5. **上游多给点，胜过下游少点硬补**。他们特征匹配一次几百点、gate 把左/右/中柱合成**一个** PnP；
   对本项目 keypoint 门：黄金窗口期应保证 4 角同帧，角点不足多发在**远/近两端** →
   远用 coarse 粗对准、近用"对向 2 角测距 + 中点瞄准"，中间 4 角窗口才做精对准与穿越判定。

**朝向来源小结（本项目无 odom / 桌位姿这类"已知参考系朝向"）**：trash "yaw 借参考系"的招数对 gate 不可用。
朝向只在 **4 角全可见（full，IPPE 解出 R）或 3 角（p3p）**时由 PnP 直接给出——含正视/背视双解消歧
（按 `t_z>0` + 与上帧姿态最近选，**不需要罗盘/参考系**）；**width_range / coarse 档不做 yaw 修正**，
只做位置对中（中点像素瞄准 / 测距），利用"进近段赛题几何近似正对"的假设；若在 golden window 后刚丢角，
可短时沿用上一帧 yaw 保持直行穿越，超时回 search。

keypoint 前端的对应关系见 §4.6 降级表；随着前端（训练/推理）成熟，若发现"某模式长期用不到/某模式常触发"，
只改 `keypoints_to_img_pts` 这一个 adapter 的判据，几何内核（§4.4/§4.5）与 GateTask 相位机不动。

## 5. 代码框架整理

### 5.0 代码架构原则（ball / gate 完全解耦，公共只留"地基"）
```
auv_vision/
├── settings.py / camera.py / uart.py / preprocess.py     # 公共地基（不掺任务逻辑）
├── detector.py        # 后端注册表 + 通用件（NMS/坐标缩放/帧缓存）；不写任务分支
├── common/PID.py      # PID 等任务公用小件（原 root tasks.py 迁移）
├── main.py            # 状态机装配：按 comm.tasks.enabled 注册 ball/gate 任务对象
├── ball/              # ball 专属：沿用现状，原则"gate 改动不碰 ball"
│   └── task1_2/ball.py  # 任务一撞球（现居 task1_2/ball.py）
├── gate/              # gate 专属（本次新建，全部新代码收在这里）
│   ├── gate_decode.py   # keypoint 头解码（§5.1）
│   ├── gate_frontend.py # keypoints_to_img_pts / mode 判定（§4.3，adapter）
│   ├── geometry.py      # CameraModel / object_points / gate_pose / 平面·反投影（§4.2/4.4/4.5）
│   └── gate_task.py     # GateTask 相位机（已落地）
└── tests/
    ├── test_geometry.py   # 合成往返、反投影、from_front（§6）
    └── test_gate_logic.py # mock 场景状态机
```
- 依赖规则：`ball/*` 看不见 `gate/*` 任何函数；`gate/*` 只 import `detector` 公共件 +
  `geometry`；两任务唯一共享是地基与 `DetectorHub` 注册表（装配层，无任务逻辑）；
- 好处：ball 权重/解码/任务互不污染 gate；gate 的 keypoint 解码写坏不会连累 ball 回归；
  gate 可独立验证：`tests/test_gate_flow.py` 或 `python3 main.py --task gate`（mock 模式）。

### 5.1 detector.py：后端注册表（球门不互见）+ keypoint 解码收在 gate/ 下
```yaml
# cfg/vision.yaml model 段
model:
  mode: hbm_runtime
  input_size: 640
  task_models:
    ball: {path: models/auv_multi.bin, labels: [blue_ball, red_ball]}   # 沿用旧权重(bbox)
    gate: {path: models/gate_kpt.bin,  kind: keypoint,                  # 门 + 4 角点
           labels: [gate], kpt_order: [TL,TR,BR,BL], kpt_conf_thr: 0.5}
```
- `DetectorHub._models: {task: 实例}`（注册表只负责"任务→后端"，无任务逻辑）；
  `_cache` 按 task 分开（同帧两模型各一次前向；任务顺序执行，同一时刻只有一个模型热）；
- **keypoint 解码放 `gate/gate_decode.py`**（§5.0 结构）：X5 OE 的 pose 头输出张量与 reg/cls
  分裂头不同，新增 `decode_yolo11_kpt`：按检测框聚合 keypoint、缩放到全尺寸帧、附每点置信度；
  `ball` 路径不引用它，球侧零回归；模型导出前核对 kpt 张量坐标基准（相对输入/相对框/归一化方式），
  用 §6 合成测试反查；
- 检测结果对象扩展：`Det.kpts (4,2)`、`Det.kpt_conf (4,)`（bbox 任务不填，向后兼容）；
- Mock：`MockGateDetector`（脚本：远门框 → 4 角变大 → 近距缺角 → 消失），供 SIM/测试（放 `gate/`）。

### 5.2 新模块 geometry.py（纯函数，前端无关，可单测）
```
CameraModel               # §4.2
keypoints_to_img_pts()    # §4.3 前端适配层(adapter #1)
object_points()           # §3 表驱动
gate_pose()               # §4.4 (IPPE/P3P + RefineLM)
plane_from_pose(R,t)      # n_c, rho
backproject_to_plane(u,v,cam,n,rho)   # §4.5 (对应 Bumblebee backproject_pixel 的门平面版)
project_corners(cam,R,t,obj3)         # 叠加调试用(前向投影四角/中心)
ema_pose()                # 可选(平移 EMA + 四元数平均, Bumblebee pose_weighted_average)
```
> 内核只吃 `(img2, obj3)` 点对 → 未来加特征匹配只需新增 adapter #2（§2.4）。

### 5.3 gate/gate_task.py — GateTask（已落地：相位机 + 位姿闭环 + 缺角仲裁）

```
SEARCH ─(见门整框)──► RANGE_ALIGN ──(4/3角 golden window 对准达标, Z 进档)──► APPROACH
   ▲                      │  ▲                                                    │
   │                      │  │ (恢复 full/p3p/width_range)                        │ Z<Z_pass 穿门
   │                      ▼  │                                                    ▼
   │            REACQUIRE ──┘                                              THROUGH(直行穿越, 计数+1)
   │            (短后退/朝可见侧转)                                              │
   │                ▲                                                           ▼
   └──(丢目标超N帧 / REACQUIRE超限)◄───────────────────────────── 复位 → SEARCH / 下一门
```

RANGE_ALIGN 内部按"前端 mode"分 4 个子状态（状态 = 档位，逐帧由 §4.3/§4.6 判定）：

| 子状态 | 触发 mode | 运动输出 | 目标 |
|---|---|---|---|
| **GOLDEN** | full / p3p | sway/heave/yaw PID 对准（按位姿误差）；**surge=0** | 偏差与偏航进死区、连续 N 帧 → APPROACH |
| **COARSE_FAR** | coarse 且框占比小 | 限量 creep（慢速 surge）+ 框心对中 | 靠近让 4 角可辨 → GOLDEN |
| **COARSE_HOLD** | coarse 中/近 / width_range | surge=0；框心/两角中点/上帧位姿保持对中 | 连续 ≥K 帧无进展 → REACQUIRE |
| **REACQUIRE** | 见左 | 短后退(surge<0, 限幅限时)或朝可见侧小幅转向 | 拉距重取整门 → 恢复 full/p3p/width_range 即回 GOLDEN/对应档 |

- 门无朝向/对准要求：GOLDEN/APPROACH 里 yaw 修正可放宽（大死区/低速），核心 = 对准开口中心后直穿；
- APPROACH：按 `Z=t.z` 分档前进 + 姿态保持（对准误差持续 PID）；`Z < Z_pass` → THROUGH；
- THROUGH（机身过门，穿过例外）：**忽略角丢失**（缺角=已过门正常现象），保持直行到
  `Z ≤ -body_center_offset` 判机身过门（无 odom 时用"穿越事件 + 消失确认"双保险，§4.5），计数+1 后复位；
- 帧间防抖：任何档位丢目标/位姿校验失败，用上帧姿态保持 ≤N 帧，超限才回 SEARCH/REACQUIRE；
- 复用现有 `PID`；yaw 通道走 `uart.send_dof(surge, sway, heave, yaw)`（接口已支持）；
- `last_info` 增加 `z / yaw / phase / substate / mode / reproj_rms / kpt_ok` 便于调参；
- 计数沿用 `pass_target` 语义（每轮一门）。

### 5.4 参数（cfg/comm.yaml + cfg/vision.yaml）
```
comm.gate:    相位/距离阈值(Z_align/Z_pass/Z_min)、PID(sway/heave/yaw)、
              pass_confirm_frames、timeout、降级开关
vision.gate:  geometry(§3)、keypoint(kpt_conf_thr)、pnp(reproj_px, z_bounds)
vision.model: task_models(§5.1)
```

## 6. 测试与验证（先无硬件闭环）

| 层 | 内容 | 验收 |
|---|---|---|
| 合成往返测试 | 随机 6-DoF → `CameraModel` 投影 §3 角点（±0.5px 噪声/缺角）→ `gate_pose` 反解 | 无噪声：位置≈0（<1cm）、角度≈0（<0.01rad）；0.5pxσ 噪声：位置误差 ∝ noise·z²/(f·W)，1.4~6m 实测包络 <4cm（按距离缩放容差）；**平面倾角在远距+小目标时噪声放大（病态）——任务无朝向要求、不依赖姿态角，只用门中心/穿越**；3 角必须带上一帧（§4.7） |
| keypoint 解码自检 | 用导出后的 `.bin` 在合成图上比对 kpt 坐标基准 | 无系统性偏移(否则改解码缩放) |
| geometry 单测 | 反投影往返、from_front 正反一致 | 往返 <0.1 px |
| SIM 场景 | `MockGateDetector` 出序列跑 GateTask | 迁移/计数/超时/E_STOP |
| 录制回放 | 真机过门视频离线逐帧跑 geometry+GateTask(不开串口) | 相位合理、无跳变、计数正确 |

先做"合成往返 + 录制回放"再上池测：PnP 数学、keypoint 解码、前端解析三者分开验证。

## 7. 部署落实

### 7.1 预处理：不改，沿用 ball 已对齐的同域链路
- **结论**：本地训练与板上推理的预处理（去畸变 → 画面补偿 → 640 缩放）在 ball 阶段已协调好，
  gate 沿用同一链路，**`preprocess.py` / `recorder.py` 均不改**，也不新增数据集导出工具/函数；
- 唯一要求：gate 标注帧取自**同一条链路的输出**（录帧后离线批处理一遍即可，纯跑批，无新代码）；
- 工程习惯（可选非必需）：批处理时把白平衡/gamma/CLIP/标定版本写个小 json 记档，便于日后调参追溯；
- 换 gate 权重时复测 §7.2 第 0 条（缩放策略拉伸 vs letterbox 一致——ball 已验证过，gate 复测一次）。

### 7.2 训练/编译/上板
0. **先核对训练/推理缩放策略一致（否则后续全偏）**：本地训练数据加载器的最后一步（缩放 640 的方式）
   必须与板上 `ModelPreprocessor.process` 一致——现有 `decode_yolo11_*` 按**纯拉伸**（sx,sy 逐轴缩放）还原坐标，
   若训练端用的是 letterbox（等比+补边），检测框/角点坐标会带系统性偏移；
   （ball 权重已验证一致；gate 换新权重/新导出时复测一次即可）
1. PC 工程 RDKX5-YOLOv11n 框架内训 keypoint（门 + 4 角）→ `.pt`；
2. `.pt → ONNX`（核对 kpt 张量结构）→ OE `hb_mapper makertbin`（march `bayes-e`）→ `gate_kpt.bin`；
   校准集用池测同域帧（重点保远距/低分），PTQ 对 keypoint 回归精度影响要实测；
3. 板端验证清单：
   - `hrt_model_exec model_info --model_file gate_kpt.bin` 张量正常；
   - 叠加调试：把四角 keypoint 与 §3 角点**前向投影到画面**目视对齐（最快校准几何/相位符号）；
   - 测距对照：卷尺 0.5~6 m 多点看 `Z=t.z` 误差曲线；
   - 延迟测量；`sway/heave/yaw` 方向复核（改 `comm.yaml dof_map` sign）；
4. 通过后打 arm64 deb（现有 `pack_image.sh` 流程挂 `/app/auv_vision`）→ 池测 → C++ 同步移植。

## 8. 未决项（需输入后收敛）

✅ 已定：① 四角 keypoint 标**门框外轮廓角**，允许只标可见角（缺角 visibility=0）（§2.2/§3）；
② 门 = **闭合矩形门框（红色 PVC 管）、平面、悬空不触地** → 四角皆真实外角，无地面/相机高度假设（§3）；
③ 门**前后完全对称、无朝向/对准要求**，任务 = **机身穿过开口** → 不做 `from_front` 切换
（§3/§4.4），完成判据 = 机身中心越过门平面（§4.5/§5.3，`body_center_offset`）；
④ **尺寸已定**：外轮廓 0.70 m（宽）× 0.50 m（高）；两扇门（矮/高）同尺寸，中心离底 0.45/0.65 m；
⑤ **`body_center_offset` 已定 = 0.0**（相机≈机身中心；穿越判据=相机过平面即判过门；相机若在机头再改）。

1. 特征匹配（纹理目标）暂不实现，仅留 §2.4 备忘。
