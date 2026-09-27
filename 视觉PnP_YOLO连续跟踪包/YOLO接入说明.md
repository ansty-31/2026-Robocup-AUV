# YOLO 接入说明：110 mm 门框连续跟踪

这个文件夹只包含连续跟踪部分，原来的红色检测程序不会被修改。核心文件是 temporal_quad_tracker.py，它接收 YOLO 每次推理得到的四个角点，并完成：

- 前几帧位移平均得到下一帧预测位置；
- 用预测位置初始化 LK 光流；
- 四角前后向光流误差、边界、凸四边形、位移和单应性检查；
- 预测 ROI 裁剪并放大后重新调用 YOLO；
- 目标丢失时回退到全画面 YOLO；
- 角点亚像素精修和随速度变化的平滑；
- 可选的 110 mm 正方形 IPPE PnP 与重投影 RMS 检查。

## 1. YOLO 模型要求

推荐训练 YOLO Pose，把门框四角标为四个关键点，并固定顺序：

~~~text
0 = TL 左上
1 = TR 右上
2 = BR 右下
3 = BL 左下
~~~

普通 YOLO Detect 只有外接框，不能直接提供可靠的四角 PnP 输入。YOLO Segmentation 也可以使用，但需要先从 mask 拟合四条边和四个角点，再交给跟踪器。

训练图片要覆盖远近距离、快速移动、运动模糊、不同光照、倾斜视角、局部遮挡和目标部分出画面。按视频或场景划分训练集和测试集，避免相邻帧同时出现在两边。

## 2. 最小接入代码

把 temporal_quad_tracker.py 放在你的 YOLO 程序旁边，然后把 YOLO 输出包装成 detect_corners(image)。这个函数每次接收一张 BGR 图像，返回 shape=(4, 2) 的 TL、TR、BR、BL 角点；检测不到时返回 None。

~~~python
import cv2
import numpy as np
from ultralytics import YOLO

from temporal_quad_tracker import (
    SquarePnPValidator,
    TemporalQuadTracker,
    TrackerConfig,
)


model = YOLO("door_frame_pose.pt")


def _numpy(value):
    return value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)


def detect_corners(image: np.ndarray):
    result = model.predict(
        image,
        conf=0.35,
        imgsz=1280,
        verbose=False,
    )[0]
    if result.keypoints is None or len(result.keypoints.xy) == 0:
        return None

    box_conf = _numpy(result.boxes.conf).reshape(-1)
    index = int(np.argmax(box_conf)) if box_conf.size else 0
    points = _numpy(result.keypoints.xy[index])[:4]
    if points.shape != (4, 2):
        return None

    # 如果训练时设置了关键点置信度，可在这里拒绝低置信角点。
    if result.keypoints.conf is not None:
        keypoint_conf = _numpy(result.keypoints.conf[index])[:4]
        if np.any(keypoint_conf < 0.25):
            return None
    return points.astype(np.float32)


with np.load("camera_calibration.npz") as calibration:
    pnp = SquarePnPValidator(
        calibration["camera_matrix"],
        calibration["dist_coeffs"],
        side_mm=110.0,
    )

tracker = TemporalQuadTracker(
    detector=detect_corners,
    pnp_validator=pnp,
    config=TrackerConfig(
        roi_padding_ratio=0.75,
        roi_upscale=2.0,
        max_tracking_misses=3,
    ),
)

capture = cv2.VideoCapture(0)
while True:
    ok, frame = capture.read()
    if not ok:
        break

    result = tracker.update(frame)
    if result.valid:
        corners = np.rint(result.corners).astype(np.int32)
        cv2.polylines(
            frame,
            [corners.reshape(-1, 1, 2)],
            True,
            (0, 255, 0) if result.mode == "detected" else (255, 0, 0),
            2,
        )
        cv2.putText(
            frame,
            f"{result.mode} RMS={result.pnp_rms_px:.2f}px",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    cv2.imshow("PnP", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

capture.release()
cv2.destroyAllWindows()
~~~

ROI 被放大后，detect_corners 得到的是 ROI 坐标；跟踪器会自动映射回原图。全画面重新检测时，坐标直接使用原图坐标。

当前标定合同是 1920×1080。如果把整帧缩放后再做 PnP，需要同步缩放相机内参；只放大 ROI 不需要修改相机内参。

## 3. 关键参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| velocity_history | 3 | 用最近几帧角点位移估计速度 |
| max_tracking_misses | 3 | 最多允许连续多少帧只用光流 |
| roi_padding_ratio | 0.75 | 预测框四周的 ROI 扩展比例 |
| roi_upscale | 2.0 | ROI 放大倍数，远距离目标可提高 |
| lk_window_size | (41, 41) | 光流搜索窗口，快速运动可增大 |
| lk_max_level | 5 | 光流金字塔层数，尺度变化或快速运动可增大 |
| max_backtrack_error_px | 2.0 | 前后向光流最大往返误差 |
| max_homography_error_px | 3.0 | 四角单应性重投影误差 |
| max_jump_ratio | 1.5 | 相对上一门框边长的最大角点跳变 |
| min_area_fraction | 0.005 | 四边形最小面积占画面比例 |
| smooth_alpha_slow | 0.35 | 慢速移动时的新角点权重 |
| smooth_alpha_fast | 0.85 | 快速移动时的新角点权重 |
| smooth_speed_ratio | 0.25 | 速度达到该比例时逐渐切换到快速权重 |
| max_pnp_rms_px | 5.0 | PnP 重投影 RMS 上限 |

调参顺序建议：远距离先提高 roi_upscale 和 YOLO 的 imgsz；快速移动再提高 lk_window_size、lk_max_level，最后才放宽 max_jump_ratio。放宽位移阈值会增加错误目标被持续跟踪的风险。

## 4. 运行自检

本目录的 smoke_test.py 使用合成矩形验证：首次检测、连续 3 帧光流跟踪，以及超过跟踪上限后清除状态。

~~~powershell
python smoke_test.py
~~~

依赖：

~~~powershell
pip install -r requirements.txt
~~~

自检通过只说明模块接口和合成帧逻辑可运行，真实摄像头仍需用包含快速移动、远距离和遮挡的视频验证。
