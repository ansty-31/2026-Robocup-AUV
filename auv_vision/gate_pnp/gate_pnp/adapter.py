"""Adapt one Ultralytics Results object without importing its optional runtime.

The caller loops over ``model.predict(...)`` results and passes the geometry of
the exact image supplied to the model. Only pixel ``keypoints.xy`` is consumed.
"""
from __future__ import annotations

import numpy as np

from .camera import CameraGeometry
from .models import PIPELINE_ID, Observation, PnPConfig, PoseResult, Status
from .solver import estimate_gate_pose


def _array(value, name):
    """Copy tensor-like values onto CPU before NumPy conversion."""
    if value is None:
        raise ValueError(f"Missing {name}; confidence is never inferred")
    try:
        for method in ("detach", "cpu", "numpy"):
            convert = getattr(value, method, None)
            if callable(convert):
                value = convert()
        return np.asarray(value, dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"Cannot read {name}: {exc}") from exc


def estimate_from_ultralytics(
    result,
    camera_geometry: CameraGeometry,
    config: PnPConfig | None = None,
    *,
    preprocessing_id=PIPELINE_ID,
    frame_id=None,
    timestamp=None,
) -> list[PoseResult]:
    """Estimate every configured target, preserving original detection indices.

    A single malformed target produces its own failure result. Missing model
    input, configuration/coordinate mismatches, and malformed class metadata
    produce a single frame-level failure. Input must be ONE Results object.
    """
    def failure(status, reason, index=None):
        return PoseResult(status=status, reason=reason, target_index=index,
                          frame_id=frame_id, timestamp=timestamp)

    if result is None:
        return [failure(Status.NO_INPUT, "No Ultralytics result was supplied")]

    config = PnPConfig() if config is None else config
    try:
        config.validate()
        camera_geometry.validate()
        if preprocessing_id != camera_geometry.preprocessing_id or preprocessing_id != PIPELINE_ID:
            raise ValueError("Preprocessing identity does not match calibrated image geometry")
        if not hasattr(result, "boxes") or not hasattr(result, "orig_shape"):
            raise ValueError("Expected one Ultralytics Results object with boxes and orig_shape")
        shape = _array(result.orig_shape, "orig_shape")
        if shape.shape != (2,) or not np.all(np.isfinite(shape)) or np.any(shape <= 0) or np.any(shape != np.floor(shape)):
            raise ValueError("orig_shape must be the positive integer pair (height, width)")
        if tuple(int(v) for v in shape[::-1]) != tuple(camera_geometry.image_size):
            raise ValueError("Model image dimensions do not match calibrated preprocessing output")
    except Exception as exc:
        return [failure(Status.CONFIG_ERROR, str(exc))]

    keypoints = getattr(result, "keypoints", None)

    def no_boxes():
        if keypoints is not None:
            try:
                xy = _array(getattr(keypoints, "xy", None), "keypoints.xy")
                if xy.size:
                    return [failure(Status.INVALID_KEYPOINTS, "Keypoints exist without detection boxes")]
            except ValueError as exc:
                return [failure(Status.INVALID_KEYPOINTS, str(exc))]
        return [failure(Status.NO_TARGET, "No detections in this image")]

    boxes = result.boxes
    if boxes is None:
        return no_boxes()
    try:
        classes = _array(getattr(boxes, "cls", None), "boxes.cls")
        if classes.ndim != 1 or not np.all(np.isfinite(classes)) or np.any(classes < 0) or np.any(classes != np.floor(classes)):
            raise ValueError("boxes.cls must contain finite nonnegative integer class indices")
        if classes.size == 0:
            return no_boxes()
        names = getattr(result, "names", None)
        if not isinstance(names, (dict, list, tuple)):
            raise ValueError("Result names must map model class indices to class names")
        targets = []
        for index, value in enumerate(classes):
            class_id = int(value)
            try:
                class_name = names[class_id]
            except (KeyError, IndexError) as exc:
                raise ValueError(f"Class index {class_id} is absent from result.names") from exc
            if not isinstance(class_name, str) or not class_name:
                raise ValueError(f"Class index {class_id} has no valid class name")
            if class_name in config.target_class_names:
                targets.append(index)
    except Exception as exc:
        return [failure(Status.CONFIG_ERROR, str(exc))]

    if not targets:
        return [failure(Status.NO_TARGET, "No configured target class was detected")]

    try:
        detection_confidence = _array(getattr(boxes, "conf", None), "boxes.conf")
        if detection_confidence.shape != classes.shape:
            raise ValueError("boxes.conf must contain one confidence per detection")
        xy = _array(getattr(keypoints, "xy", None), "keypoints.xy")
        confidence = _array(getattr(keypoints, "conf", None), "keypoints.conf")
        if xy.ndim != 3 or xy.shape[0] != len(classes) or xy.shape[2] != 2:
            raise ValueError("keypoints.xy must have shape (detections, keypoints, 2)")
        if confidence.shape != xy.shape[:2]:
            raise ValueError("keypoints.conf must have shape (detections, keypoints)")
        indices = np.asarray(config.keypoint_indices, dtype=int)
        if max(indices) >= xy.shape[1]:
            raise ValueError("Configured semantic keypoint indices are absent from model output")
    except Exception as exc:
        return [failure(Status.INVALID_KEYPOINTS, str(exc), index) for index in targets]

    outcomes = []
    for index in targets:
        score = float(detection_confidence[index])
        if not np.isfinite(score) or not 0 <= score <= 1 or score < config.detection_confidence:
            outcomes.append(failure(Status.INVALID_KEYPOINTS, "Detection confidence is missing, invalid, or below the configured threshold", index))
            continue
        # Preserve model semantics: TL, TR, BR, BL. Never sort points by position.
        observation = Observation(
            points=xy[index, indices].copy(),
            confidence=confidence[index, indices].copy(),
            image_size=tuple(camera_geometry.image_size),
            preprocessing_id=preprocessing_id,
            detection_confidence=score,
            target_index=index,
            frame_id=frame_id,
            timestamp=timestamp,
        )
        outcomes.append(estimate_gate_pose(observation, camera_geometry, config))
    return outcomes
