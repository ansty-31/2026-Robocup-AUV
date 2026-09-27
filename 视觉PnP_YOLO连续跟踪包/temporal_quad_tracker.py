from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np


CornersDetector = Callable[[np.ndarray], Optional[np.ndarray]]
PnPValidator = Callable[[np.ndarray], Optional[float]]


@dataclass(frozen=True)
class TrackerConfig:
    """Parameters that control prediction, optical flow, ROI and smoothing."""

    roi_padding_ratio: float = 0.75
    roi_upscale: float = 2.0
    velocity_history: int = 3
    max_tracking_misses: int = 3
    lk_window_size: tuple[int, int] = (41, 41)
    lk_max_level: int = 5
    max_backtrack_error_px: float = 2.0
    max_homography_error_px: float = 3.0
    min_area_px: float = 500.0
    min_area_fraction: float = 0.005
    min_reference_side_px: float = 50.0
    max_jump_ratio: float = 1.5
    smooth_alpha_slow: float = 0.35
    smooth_alpha_fast: float = 0.85
    smooth_speed_ratio: float = 0.25
    max_pnp_rms_px: float = 5.0
    refine_corners: bool = True


@dataclass(frozen=True)
class TrackResult:
    """Output for one frame."""

    valid: bool
    corners: Optional[np.ndarray]
    mode: str  # "detected", "tracked" or "invalid"
    misses: int
    pnp_rms_px: Optional[float]
    roi: Optional[tuple[int, int, int, int]]
    message: str


def _invalid(message: str, misses: int = 0) -> TrackResult:
    return TrackResult(
        valid=False,
        corners=None,
        mode="invalid",
        misses=misses,
        pnp_rms_px=None,
        roi=None,
        message=message,
    )


def _align_cyclic(corners: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, float]:
    current = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    previous = np.asarray(reference, dtype=np.float32).reshape(4, 2)
    candidates = [(float(np.mean(np.linalg.norm(np.roll(current, n, axis=0) - previous, axis=1))),
                   np.roll(current, n, axis=0)) for n in range(4)]
    distance, aligned = min(candidates, key=lambda item: item[0])
    return aligned.astype(np.float32), distance


def _jump_limit(corners: np.ndarray, config: TrackerConfig) -> float:
    points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    side_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    return max(config.min_reference_side_px, float(np.median(side_lengths)) * config.max_jump_ratio)


def _homography_ok(
    previous: np.ndarray,
    current: np.ndarray,
    config: TrackerConfig,
) -> bool:
    previous = np.asarray(previous, dtype=np.float32).reshape(4, 1, 2)
    current = np.asarray(current, dtype=np.float32).reshape(4, 1, 2)
    if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(current)):
        return False
    try:
        homography, inliers = cv2.findHomography(
            previous,
            current,
            cv2.RANSAC,
            config.max_homography_error_px,
        )
    except cv2.error:
        return False
    if homography is None or inliers is None or not np.all(inliers):
        return False
    projected = cv2.perspectiveTransform(previous, homography)
    error = np.linalg.norm(projected - current, axis=2).reshape(-1)
    if not np.all(np.isfinite(error)) or np.max(error) > config.max_homography_error_px:
        return False

    old_lengths = np.linalg.norm(
        np.roll(previous.reshape(4, 2), -1, axis=0) - previous.reshape(4, 2),
        axis=1,
    )
    new_lengths = np.linalg.norm(
        np.roll(current.reshape(4, 2), -1, axis=0) - current.reshape(4, 2),
        axis=1,
    )
    if np.any(old_lengths <= 1e-6) or np.any(new_lengths <= 1e-6):
        return False
    scale = new_lengths / old_lengths
    median_scale = float(np.median(scale))
    return bool(
        np.isfinite(median_scale)
        and median_scale > 0
        and np.all(scale / median_scale <= 2.5)
        and np.all(median_scale / scale <= 2.5)
    )


def _refine(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
    try:
        refined = cv2.cornerSubPix(
            gray,
            points.reshape(4, 1, 2),
            (5, 5),
            (-1, -1),
            criteria,
        )
    except cv2.error:
        return points.copy()
    if refined is None or not np.all(np.isfinite(refined)):
        return points.copy()
    aligned, distance = _align_cyclic(refined.reshape(4, 2), points)
    return aligned if distance <= 4.0 else points.copy()


def _edge_image(gray: np.ndarray) -> np.ndarray:
    return cv2.Canny(gray, 20, 80)


def _track_with_flow(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    previous_corners: np.ndarray,
    predicted_corners: np.ndarray,
    config: TrackerConfig,
) -> Optional[np.ndarray]:
    """Track four corners with predicted LK initialization and forward/backward checks."""

    previous = np.asarray(previous_corners, dtype=np.float32).reshape(4, 1, 2)
    predicted = np.asarray(predicted_corners, dtype=np.float32).reshape(4, 1, 2)
    old_edges = _edge_image(previous_gray)
    new_edges = _edge_image(current_gray)
    try:
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            old_edges,
            new_edges,
            previous,
            predicted.copy(),
            winSize=config.lk_window_size,
            maxLevel=config.lk_max_level,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        )
        if next_points is None or status is None or not np.all(status.reshape(-1)):
            return None
        back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
            new_edges,
            old_edges,
            next_points,
            previous.copy(),
            winSize=config.lk_window_size,
            maxLevel=config.lk_max_level,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        )
    except cv2.error:
        return None

    if back_points is None or back_status is None or not np.all(back_status.reshape(-1)):
        return None
    back_error = np.linalg.norm(
        back_points.reshape(4, 2) - previous.reshape(4, 2),
        axis=1,
    )
    if not np.all(np.isfinite(back_error)) or np.max(back_error) > config.max_backtrack_error_px:
        return None

    current = next_points.reshape(4, 2).astype(np.float32)
    height, width = current_gray.shape[:2]
    if np.any(current < 0) or np.any(current[:, 0] >= width) or np.any(current[:, 1] >= height):
        return None
    contour = current.reshape(4, 1, 2)
    if not cv2.isContourConvex(contour):
        return None
    if cv2.contourArea(contour) < max(config.min_area_px, width * height * config.min_area_fraction):
        return None

    aligned, distance = _align_cyclic(current, predicted)
    if distance > _jump_limit(predicted, config):
        return None
    if not _homography_ok(previous.reshape(4, 2), aligned, config):
        return None
    return aligned


class SquarePnPValidator:
    """Return the best positive-depth PnP RMS for a 110 mm square."""

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        side_mm: float = 110.0,
    ) -> None:
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)
        self.side_mm = float(side_mm)
        if self.camera_matrix.shape != (3, 3) or self.side_mm <= 0:
            raise ValueError("camera_matrix 必须为 3×3，side_mm 必须为正数")
        half = self.side_mm / 2.0
        self.object_points = np.array(
            [[-half, -half, 0.0], [half, -half, 0.0],
             [half, half, 0.0], [-half, half, 0.0]],
            dtype=np.float64,
        )

    def __call__(self, corners: np.ndarray) -> Optional[float]:
        points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        try:
            result = cv2.solvePnPGeneric(
                self.object_points,
                points.reshape(4, 1, 2),
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE,
            )
        except cv2.error:
            return None
        if len(result) < 3 or not result[0]:
            return None

        best: Optional[float] = None
        for rvec_raw, tvec_raw in zip(result[1], result[2]):
            rvec = np.asarray(rvec_raw, dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(tvec_raw, dtype=np.float64).reshape(3, 1)
            try:
                rotation, _ = cv2.Rodrigues(rvec)
                camera_points = (rotation @ self.object_points.T + tvec).T
                if np.any(camera_points[:, 2] <= 0):
                    continue
                projected, _ = cv2.projectPoints(
                    self.object_points,
                    rvec,
                    tvec,
                    self.camera_matrix,
                    self.dist_coeffs,
                )
            except cv2.error:
                continue
            delta = projected.reshape(4, 2) - points
            rms = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
            if np.isfinite(rms) and (best is None or rms < best):
                best = rms
        return best


class TemporalQuadTracker:
    """YOLO corner detector + ROI/flow/homography/PnP temporal tracker.

    The detector must return four points in the fixed order TL, TR, BR, BL.
    It receives a BGR image and may return None when the target is missing.
    """

    def __init__(
        self,
        detector: CornersDetector,
        pnp_validator: Optional[PnPValidator] = None,
        config: Optional[TrackerConfig] = None,
    ) -> None:
        self.detector = detector
        self.pnp_validator = pnp_validator
        self.config = config or TrackerConfig()
        self.reset()

    def reset(self) -> None:
        self.previous_gray: Optional[np.ndarray] = None
        self.raw_history: deque[np.ndarray] = deque(
            maxlen=max(2, self.config.velocity_history)
        )
        self.smoothed_corners: Optional[np.ndarray] = None
        self.misses = 0

    def _predict(self) -> Optional[np.ndarray]:
        if not self.raw_history:
            return None
        prediction = self.raw_history[-1].copy()
        if len(self.raw_history) >= 2:
            history = np.stack(tuple(self.raw_history), axis=0)
            deltas = np.diff(history, axis=0)
            prediction += np.mean(deltas[-2:], axis=0)
        return prediction.astype(np.float32)

    def _roi_bounds(
        self,
        frame: np.ndarray,
        prediction: np.ndarray,
    ) -> Optional[tuple[int, int, int, int]]:
        height, width = frame.shape[:2]
        points = np.asarray(prediction, dtype=np.float32).reshape(4, 2)
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        span = max(float(x_max - x_min), float(y_max - y_min), 32.0)
        padding = max(32.0, span * self.config.roi_padding_ratio)
        left = max(0, int(np.floor(x_min - padding)))
        top = max(0, int(np.floor(y_min - padding)))
        right = min(width, int(np.ceil(x_max + padding)))
        bottom = min(height, int(np.ceil(y_max + padding)))
        if right - left < 16 or bottom - top < 16:
            return None
        return left, top, right, bottom

    def _detect_roi(
        self,
        frame: np.ndarray,
        prediction: np.ndarray,
    ) -> tuple[Optional[np.ndarray], Optional[tuple[int, int, int, int]]]:
        bounds = self._roi_bounds(frame, prediction)
        if bounds is None:
            return None, None
        left, top, right, bottom = bounds
        crop = frame[top:bottom, left:right]
        scale = float(self.config.roi_upscale)
        scaled = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        detected = self.detector(scaled)
        if detected is None:
            return None, bounds
        points = np.asarray(detected, dtype=np.float32).reshape(4, 2)
        points = points / scale + np.array([left, top], dtype=np.float32)
        return points, bounds

    def _geometry_ok(
        self,
        corners: np.ndarray,
        frame_shape: tuple[int, ...],
        prediction: Optional[np.ndarray],
    ) -> tuple[bool, Optional[np.ndarray], str]:
        points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        if not np.all(np.isfinite(points)):
            return False, None, "角点包含非有限值"
        height, width = frame_shape[:2]
        if np.any(points < 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height):
            return False, None, "角点超出画面边界"
        contour = points.reshape(4, 1, 2)
        if not cv2.isContourConvex(contour):
            return False, None, "四角不是凸四边形"
        if cv2.contourArea(contour) < max(
            self.config.min_area_px,
            width * height * self.config.min_area_fraction,
        ):
            return False, None, "四边形面积过小"
        if self.raw_history:
            previous = self.raw_history[-1]
            aligned, distance = _align_cyclic(points, previous)
            reference_limit = _jump_limit(previous, self.config)
            predicted_distance = (
                _align_cyclic(points, prediction)[1]
                if prediction is not None
                else float("inf")
            )
            if min(distance, predicted_distance) > reference_limit:
                return False, None, "角点位移超限"
            if not _homography_ok(previous, aligned, self.config):
                return False, None, "单应性一致性失败"
            points = aligned
        return True, points, "OK"

    def _pnp_ok(self, corners: np.ndarray) -> tuple[bool, Optional[float]]:
        if self.pnp_validator is None:
            return True, None
        rms = self.pnp_validator(corners)
        if rms is None or not np.isfinite(rms):
            return False, rms
        return bool(rms <= self.config.max_pnp_rms_px), float(rms)

    def _smooth(self, raw: np.ndarray) -> np.ndarray:
        if self.smoothed_corners is None:
            return raw.copy()
        previous = self.smoothed_corners
        speed = float(np.mean(np.linalg.norm(raw - previous, axis=1)))
        side = max(self.config.min_reference_side_px, float(np.median(
            np.linalg.norm(np.roll(previous, -1, axis=0) - previous, axis=1)
        )))
        ratio = min(1.0, speed / max(1e-6, side * self.config.smooth_speed_ratio))
        alpha = self.config.smooth_alpha_slow + (
            self.config.smooth_alpha_fast - self.config.smooth_alpha_slow
        ) * ratio
        return ((1.0 - alpha) * previous + alpha * raw).astype(np.float32)

    def update(self, frame: np.ndarray) -> TrackResult:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame 必须是 BGR 三通道图像")

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prediction = self._predict()
        corners: Optional[np.ndarray] = None
        mode = "detected"
        roi: Optional[tuple[int, int, int, int]] = None
        misses = 0

        if prediction is None:
            corners = self.detector(frame)
        else:
            corners, roi = self._detect_roi(frame, prediction)
            if corners is None and self.previous_gray is not None and self.misses < self.config.max_tracking_misses:
                corners = _track_with_flow(
                    self.previous_gray,
                    current_gray,
                    self.raw_history[-1],
                    prediction,
                    self.config,
                )
                if corners is not None:
                    mode = "tracked"
                    misses = self.misses + 1
            if corners is None:
                # 急停、反向或目标进入 ROI 外时，重新尝试全画面 YOLO。
                corners = self.detector(frame)
                mode = "detected"
                misses = 0

        if corners is None:
            self.reset()
            return _invalid("检测、ROI 和光流均未得到四角")

        if mode == "detected" and self.config.refine_corners:
            corners = _refine(current_gray, corners)
        geometry_ok, corners, reason = self._geometry_ok(corners, frame.shape, prediction)
        if not geometry_ok or corners is None:
            self.reset()
            return _invalid(reason)

        pnp_ok, rms = self._pnp_ok(corners)
        if not pnp_ok:
            self.reset()
            return _invalid(
                f"PnP RMS 无效或超过 {self.config.max_pnp_rms_px:.2f} px"
            )

        smoothed = self._smooth(corners)
        smooth_ok, smooth_rms = self._pnp_ok(smoothed)
        if not smooth_ok:
            smoothed = corners
            message = "原始角点有效，平滑结果被 PnP 拒绝"
        else:
            message = "OK"

        self.raw_history.append(corners.copy())
        self.smoothed_corners = smoothed.copy()
        self.previous_gray = current_gray
        self.misses = misses
        return TrackResult(
            valid=True,
            corners=smoothed.copy(),
            mode=mode,
            misses=misses,
            pnp_rms_px=smooth_rms if smooth_rms is not None else rms,
            roi=roi,
            message=message,
        )
