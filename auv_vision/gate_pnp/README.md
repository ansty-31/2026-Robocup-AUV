# 水下门框 PnP

独立的 Python + OpenCV 模块。输入 YOLO Pose 检测到的门框四角，输出**相机在门框坐标系中的位置与姿态**。

已增加**部分门框视野恢复 → 对准 → 前进过门**决策模块，详见 [行为决策与接入说明](GUIDANCE.md)。仅 TL/TR 可用时输出 `DESCEND`（下降），仅 BR/BL 可用时输出 `ASCEND`（上升），均需连续确认至少0.2秒且3个不同有效帧；门框过大且上下贴边时后退优先。原有 PnP 接口保持兼容。新模块仅输出行为，不连接推进器或串口；对准期间丢点可以重新找门，开始前进过门后不再补视野、升降、后退或扫描。

当前交付包含 PnP 核心、Ultralytics 结果适配器、相机几何配置和测试。**不加载模型、不打开相机、不提供运行时模拟角点**。训练完成后，在现有推理循环中调用适配器即可。

## 1. 现在如何运行

在本目录打开终端，用现有 Python 环境执行：

```powershell
python -m gate_pnp
python -m gate_pnp --guidance
python -m unittest discover -s tests -v
```

第一条命令只检查配置，正常输出 `status: NO_INPUT`、`pose: null`，退出码为 0。配置无效时输出 `CONFIG_ERROR`，退出码为 2。它不会伪造一个门框姿态。

`--guidance` 检查新行为配置，无输入时输出 `phase: ACQUIRE`、`action: HOLD`。中文目录中的标定文件现在由 Python 读取文本后交给 OpenCV 内存解析。

本次验证的解释器：`C:/Users/ASUS/miniconda3/envs/PythonProject/python.exe`。核心只依赖 OpenCV、NumPy、PyYAML。没有安装 Ultralytics 也能运行上面两条命令；当前环境未安装 Ultralytics。

`requirements.txt` 为依赖范围，`requirements-tested.txt` 记录本次实际验证版本。若使用已有训练环境，先运行测试，再确认**图片预处理与 PnP 使用相同的 OpenCV 版本**。不要仅为加载本包覆盖既有训练环境的依赖。

## 2. 训练完成后的接入位置

模型应输出门框类别 `gate`，四个关键点按 **TL、TR、BR、BL** 标注。在你的推理程序中初始化一次相机几何与配置：

```python
from pathlib import Path
from gate_pnp import (
    PIPELINE_ID,
    PnPConfig,
    geometry_from_files,
    estimate_from_ultralytics,
)

project = Path.cwd()  # 在交付目录运行；集成到其他工程时设为交付目录的实际路径
geometry = geometry_from_files(
    project / "config/front_camera.yaml",
    project / "config/vision.yaml",
)
settings = PnPConfig.from_file(project / "config/pnp.json")
```

随后把已有 `model.predict(...)` 返回列表中的**每一个 Results 对象**交给下面的函数。这里不创建模型、不读取图片，也不需要手工输入角点：

```python
def consume_pose_result(result, frame_id=None, timestamp=None):
    return estimate_from_ultralytics(
        result,
        geometry,
        settings,
        preprocessing_id=PIPELINE_ID,
        frame_id=frame_id,
        timestamp=timestamp,
    )

# 你的模型训练好并完成推理后：
# for result in model.predict(processed_frame_640):
#     for outcome in consume_pose_result(result):
#         if outcome.ok:
#             position_m = outcome.pose.camera_position_gate
#             rotation = outcome.pose.camera_rotation_gate
#             quaternion_xyzw = outcome.pose.camera_quaternion_xyzw
#         else:
#             print(outcome.status.value, outcome.reason)
```

`processed_frame_640` 必须是本次三个附件定义的已去畸变、直接缩放到 640×640 的图像。不要再次对这些图像去畸变。`Results.keypoints.xy` 已经是相对于传给模型的图像的像素坐标，**不要再次反算 YOLO 内部缩放或补边，不使用归一化的 `xyn`**。

如果模型类别名不同，修改 `config/pnp.json` 的 `target_class_names`。若模型关键点顺序不同，修改 `keypoint_indices`，使取出的四点依次对应 TL、TR、BR、BL；例如模型顺序为 TL/TR/BL/BR 时设为 `[0,1,3,2]`。适配器不会根据像素的左右上下重新排序。

没有检测结果时可以传 `None`，得到包含一个 `NO_INPUT` 的列表。没有目标得到一个 `NO_TARGET`。检测到多个门框时逐个返回结果，并保留原始检测索引 `target_index`；不自动选导航目标。某个目标低置信度不会影响其他完整目标。

## 3. 相机几何必须与预处理匹配

本次以附件实际代码为准：

1. 原始分辨率 **1280×720**，读取 `front_camera.yaml` 的 K 和五项畸变系数。
2. `getOptimalNewCameraMatrix(K, D, (1280,720), 0, (1280,720))`，使用默认 `centerPrincipalPoint=False`。
3. 用新内参生成 remap 去畸变映射；返回的 ROI **未被用于裁剪**。
4. 白平衡、CLAHE、gamma 只影响色彩亮度。
5. `cv2.resize(..., (640,640), INTER_LINEAR)`，**直接非等比例缩放，没有裁剪或补边**。

因此 PnP 使用 `K_final = A @ K_undistorted`，畸变系数为全零。OpenCV 像素中心缩放约定为：

```text
u_final = (u_undistorted + 0.5) × sx − 0.5
v_final = (v_undistorted + 0.5) × sy − 0.5
sx = 640/1280 = 0.5
sy = 640/720

A = [[sx,  0, (sx−1)/2],
     [ 0, sy, (sy−1)/2],
     [ 0,  0,        1]]
```

`config/camera_geometry.json` 记录本次 OpenCV 5.0.0 下计算的矩阵、处理参数、来源 SHA-256 和版本，**不是对既有训练图片所用 OpenCV 版本的确认**。本次计算结果：

```text
K_final ≈ [[391.26832822,   0,          316.64756694],
           [  0,          693.14491322, 343.86685847],
           [  0,            0,           1         ]]
D_final = [0, 0, 0, 0, 0]
```

代码每次初始化按当前运行环境计算，不把上述数值硬编码为所有环境通用。`CameraGeometry.from_file(...)` 会拒绝与当前 OpenCV 版本不同的导出文件。若更换环境，应先确认与图像预处理所用版本一致，再重新导出：

```powershell
python -m gate_pnp --export-geometry config/camera_geometry.json
```

如果图像实际使用了其他原始分辨率、alpha、主点设置、裁剪或 letterbox，需要另建匹配的几何配置；本版本遇到已声明的尺寸/处理方式不一致会拒绝求解。`preprocessing_id` 是调用方对实际流水线的声明，**仅靠四个坐标无法检测图片是否被错误处理或错误标记**。导出几何时校验的附件参数不包含命令行私自覆盖或上游另外处理过的图像。

## 4. 坐标与输出含义

门框宽 0.70 m、高 0.50 m，四点之间的物理尺寸已由用户确认。原点为门框中心，从指定正面看 X 向右、Y 向下、Z 穿过门框向前，构成右手系：

| 点 | 门框坐标 / m |
|---|---|
| TL | (−0.35, −0.25, 0) |
| TR | (+0.35, −0.25, 0) |
| BR | (+0.35, +0.25, 0) |
| BL | (−0.35, +0.25, 0) |

相机坐标使用 OpenCV 约定：X 向图像右、Y 向图像下、Z 沿光轴向前。四角名称是**固定物理身份**，不能因相机翻滚或观察方向变化自动交换。

| 字段 | 含义 |
|---|---|
| `T_camera_from_gate` | `X_camera = R @ X_gate + t`，4×4 齐次变换 |
| `T_gate_from_camera` | 上述变换的逆变换 |
| `camera_position_gate` | 相机原点在门框坐标中的位置 `−R.T @ t`，米 |
| `camera_rotation_gate` | 相机坐标转换到门框坐标的旋转 `R.T` |
| `camera_quaternion_xyzw` | 同一旋转的单位四元数，顺序 x/y/z/w |
| `rvec`、`tvec` | OpenCV 原始方向：门框到相机；rvec 是旋转向量，不是欧拉角 |
| `rms_error_px` | 四点二维欧氏误差平方均值的平方根，单位为640×640图像像素 |
| `per_point_error_px` | 四个角点各自的二维欧氏重投影误差 |

例如，相机正对门框中心、两者坐标轴平行，相距 2 m 时，`tvec` 约为 `(0,0,+2)`，而**相机在门框坐标中的位置约为 `(0,0,−2)`**。负 Z 是坐标约定的结果。

输出不是机器人机体位姿或水下世界定位。机体位姿还需要相机安装外参；本模块不包含该变换。

## 5. 求解、阈值与状态

核心入口 `estimate_gate_pose(observation, camera_geometry, config=None)` 接收 `Observation`：`points` 为 `(4,2)`，`confidence` 为 `(4,)`，`image_size` 为 `(宽,高)`；还可带 `preprocessing_id`、`detection_confidence`、`target_index`、`frame_id` 和 `timestamp`。适配器负责构建该对象。

算法使用普通 **IPPE** 返回两个平面候选，再分别执行 LM 细化；非有限解或任一点位于相机后方的解被剔除，细化变差或失败则保留原始有效候选。相同姿态按旋转差 ≤0.1° 且相机位置差 ≤1 mm 合并，避免两条初值收敛到同一解后误报歧义。这是可配置的数值去重容差。

默认检测框和四个关键点置信度均要求 ≥0.5；重投影 RMS 上限 3 px。两个不同候选的误差差距 <0.2 px 时返回歧义，包括次优候选恰好超过 3 px 的边界；若最佳候选本身已超过 3 px，先返回误差过大。这些是可配置的初始工程阈值，尚未经真实水下门框数据调优。

| 状态 | 调用方处理 |
|---|---|
| `SUCCESS` | `ok=True`，可读取 `pose` |
| `NO_INPUT` | 未提供输入，不计算 |
| `NO_TARGET` | 未检测到配置的门框类别 |
| `CONFIG_ERROR` | 校准、尺寸、处理声明或配置不一致 |
| `INVALID_KEYPOINTS` | 缺点、低置信度、越界、重复、形状退化或结构错误 |
| `SOLVE_FAILED` | 求解失败或没有四点正深度的有效解 |
| `HIGH_REPROJECTION_ERROR` | 最优候选误差超过上限 |
| `AMBIGUOUS` | 存在近似同优的不同姿态，保留候选但不选择唯一姿态 |

只有 `SUCCESS` 的 `pose` 非空。其他状态可能包含诊断用 `candidates`，不能当成已通过的姿态。结果可用 `to_dict()` 转为可序列化字典。每次调用无状态，不复用上帧姿态。

四点没有可靠剔除一个错点所需的冗余，不采用四点 RANSAC 宣称解决错点。矩形对称、语义标注交换和部分错误对应可能仍产生低重投影误差，必须在数据标注和输入映射环节确保正确；代码不能从对称轮廓恢复未提供的物理角点身份。

## 6. 文件与验证范围

- `gate_pnp/camera.py`：按附件计算最终相机几何，导出与读取版本化配置。
- `gate_pnp/solver.py`：输入检查、平面求解、细化、歧义判定和坐标逆变换。
- `gate_pnp/adapter.py`：读取单个 Ultralytics Results，不依赖其安装。
- `gate_pnp/models.py`：公开数据结构、状态与阈值。
- `config/`：原始两份 YAML 的逐字节副本、PnP 阈值和派生相机几何。
- `reference/prepare_frames(1).py`：本次预处理附件的逐字节副本，仅供核对，不由模块自动运行。
- `tests/`：合成投影、边界与适配器测试；所有合成角点都仅存在于测试。
- `verification.json`、`test-results.txt`：本次实跑验证记录。

已验证算法与接口结构，**未验证真实 Ultralytics 模型推理、GPU/板端速度、水下测距及姿态精度**。标定 RMS 0.705098 px、棋盘内角 11×8、格长20 mm，以及水下带防水罩的标定条件记录在几何元数据中；它们不等同于门框测距精度。接入真实数据后，应在已知距离和角度下比较输出与实测值，再设置应用阈值。

## 参考

实现独立编写，参考其平面多解处理思路，不复制 ROS 节点或引入 ROS 依赖：

- [BumblebeeAS 平面 PnP 参考实现](https://github.com/BumblebeeAS/pose_estimator/blob/main/pose_estimator/dock_pnp_pose_estimator_node.py)
- [OpenCV PnP 与变换方向](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)
- [Ultralytics Results / Keypoints 接口](https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Keypoints)
