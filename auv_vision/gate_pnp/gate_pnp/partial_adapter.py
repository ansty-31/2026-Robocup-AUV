"""Extract partial gate observations without running a model or solving PnP.

Pixel keypoints retain their semantic order and original 640x640 coordinates.
Zero, off-image, and nonfinite keypoints remain available for the downstream
visibility classifier. Detection indices identify this frame only; ``target_id``
is the caller's goal identity and is never inferred from a detection index.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .adapter import _array
from .camera import CameraGeometry
from .guidance_types import GateObservation, ObservationKind
from .models import PIPELINE_ID, PnPConfig


def extract_gate_observations(
    result,
    camera_geometry: CameraGeometry,
    config: PnPConfig | None = None,
    *,
    frame_id: int,
    timestamp: float,
    target_id: str = 'default',
    preprocessing_id: str = PIPELINE_ID,
) -> list[GateObservation]:
    """Extract configured targets from ONE Ultralytics Results object.

    Missing input is ERROR; a valid empty frame is NO_TARGET. Structural errors
    produce ERROR for every matching detection when class indices are known.
    Invalid confidence or bounding boxes affect only their own target. Low
    scores are preserved, with no thresholding or fabricated confidence.
    Frame ordering, freshness, and goal identity are validated by the controller.
    """
    image_size = (640, 640)

    def observation(kind, error='', index=None, **data):
        return GateObservation(
            frame_id=frame_id, timestamp=timestamp, image_size=image_size,
            target_id=target_id, kind=kind, target_index=index,
            preprocessing_id=preprocessing_id, error=error, **data,
        )

    def failure(error, index=None):
        return observation(ObservationKind.ERROR, str(error), index)

    if result is None:
        return [failure('No Ultralytics result was supplied')]

    config = PnPConfig() if config is None else config
    try:
        config.validate()
        camera_geometry.validate()
        image_size = tuple(camera_geometry.image_size)
        if preprocessing_id != camera_geometry.preprocessing_id or preprocessing_id != PIPELINE_ID:
            raise ValueError('Preprocessing identity does not match calibrated image geometry')
        if not hasattr(result, 'boxes') or not hasattr(result, 'orig_shape'):
            raise ValueError('Expected one Ultralytics Results object with boxes and orig_shape')
        shape = _array(result.orig_shape, 'orig_shape')
        if (shape.shape != (2,) or not np.isfinite(shape).all()
                or np.any(shape <= 0) or np.any(shape != np.floor(shape))):
            raise ValueError('orig_shape must be the positive integer pair (height, width)')
        if tuple(int(v) for v in shape[::-1]) != image_size:
            raise ValueError('Model image dimensions do not match calibrated preprocessing output')
    except Exception as exc:
        return [failure(exc)]

    keypoints = getattr(result, 'keypoints', None)

    def empty_frame():
        # Ultralytics may have no keypoint object for a detection-free frame.
        if keypoints is not None:
            try:
                xy = _array(getattr(keypoints, 'xy', None), 'keypoints.xy')
                if xy.ndim != 3 or xy.shape[0] != 0 or xy.shape[2] != 2:
                    raise ValueError('Empty detection frame requires empty (0, keypoints, 2) xy')
                confidence = _array(getattr(keypoints, 'conf', None), 'keypoints.conf')
                if confidence.shape != xy.shape[:2]:
                    raise ValueError('Empty keypoints.conf must match empty keypoints.xy')
            except Exception as exc:
                return [failure(exc)]
        return [observation(ObservationKind.NO_TARGET)]

    boxes = result.boxes
    if boxes is None:
        return empty_frame()
    try:
        classes = _array(getattr(boxes, 'cls', None), 'boxes.cls')
        if (classes.ndim != 1 or not np.isfinite(classes).all() or np.any(classes < 0)
                or np.any(classes != np.floor(classes))):
            raise ValueError('boxes.cls must contain finite nonnegative integer class indices')
        if classes.size == 0:
            scores = _array(getattr(boxes, 'conf', None), 'boxes.conf')
            bounds = _array(getattr(boxes, 'xyxy', None), 'boxes.xyxy')
            if scores.shape != (0,) or bounds.shape != (0, 4):
                raise ValueError('Empty boxes require conf shape (0,) and xyxy shape (0,4)')
            return empty_frame()
        names = getattr(result, 'names', None)
        if not isinstance(names, (dict, list, tuple)):
            raise ValueError('Result names must map class indices to class names')
        targets = []
        for index, value in enumerate(classes):
            class_id = int(value)
            try:
                class_name = names[class_id]
            except (KeyError, IndexError) as exc:
                raise ValueError(f'Class index {class_id} is absent from result.names') from exc
            if not isinstance(class_name, str) or not class_name:
                raise ValueError(f'Class index {class_id} has no valid class name')
            if class_name in config.target_class_names:
                targets.append(index)
    except Exception as exc:
        return [failure(exc)]

    try:
        scores = _array(getattr(boxes, 'conf', None), 'boxes.conf')
        bounds = _array(getattr(boxes, 'xyxy', None), 'boxes.xyxy')
        xy = _array(getattr(keypoints, 'xy', None), 'keypoints.xy')
        confidence = _array(getattr(keypoints, 'conf', None), 'keypoints.conf')
        if scores.shape != classes.shape:
            raise ValueError('boxes.conf must have shape (detections,)')
        if bounds.shape != (len(classes), 4):
            raise ValueError('boxes.xyxy must have shape (detections, 4)')
        if xy.ndim != 3 or xy.shape[0] != len(classes) or xy.shape[2] != 2:
            raise ValueError('keypoints.xy must have shape (detections, keypoints, 2)')
        if confidence.shape != xy.shape[:2]:
            raise ValueError('keypoints.conf must have shape (detections, keypoints)')
        indices = np.asarray(config.keypoint_indices, dtype=int)
        if max(indices) >= xy.shape[1]:
            raise ValueError('Configured semantic keypoint indices are absent from model output')
    except Exception as exc:
        return [failure(exc, index) for index in targets] if targets else [failure(exc)]

    width, height = image_size

    def detection_error(index):
        score = float(scores[index])
        if not np.isfinite(score) or not 0 <= score <= 1:
            return 'Detection confidence must be finite and in [0,1]'
        if (not np.isfinite(confidence[index]).all()
                or np.any(confidence[index] < 0) or np.any(confidence[index] > 1)):
            return 'Keypoint confidence must be finite and in [0,1]'
        x1, y1, x2, y2 = bounds[index]
        if (not np.isfinite(bounds[index]).all()
                or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height)):
            return 'Bounding box must be finite, ordered and within the image'
        return ''

    if not targets:
        # A damaged model result is never evidence that this frame has no gate.
        for index in range(len(classes)):
            error = detection_error(index)
            if error:
                return [failure(error)]
        return [observation(ObservationKind.NO_TARGET)]

    outcomes = []
    for index in targets:
        error = detection_error(index)
        if error:
            outcomes.append(failure(error, index))
            continue
        outcomes.append(observation(
            ObservationKind.TARGET, index=index,
            points=xy[index, indices].copy(), confidence=confidence[index, indices].copy(),
            bbox_xyxy=bounds[index].copy(), detection_confidence=float(scores[index]),
        ))
    return outcomes


def select_gate_observation(
    observations: list[GateObservation], target_index: int | None = None,
) -> GateObservation:
    """Require an explicit original detection index whenever candidates compete.

    Explicit selection also returns a malformed target's ERROR, never a different
    target. An empty list raises ValueError because it carries no frame metadata.
    Pass observations from exactly one extraction call (one frame and goal).
    """
    if not observations:
        raise ValueError('Cannot select from an empty observation list')

    def failure(kind, error):
        return replace(observations[0], kind=kind, error=error,
                       target_index=target_index, points=None, confidence=None,
                       bbox_xyxy=None, detection_confidence=None)

    if target_index is not None:
        if type(target_index) is not int or target_index < 0:
            return failure(ObservationKind.ERROR, 'target_index must be a nonnegative integer')
        selected = [o for o in observations if o.target_index == target_index]
        if len(selected) != 1:
            return failure(ObservationKind.ERROR, 'Selected detection index is absent or duplicated')
        return selected[0]
    if len(observations) == 1:
        return observations[0]
    return failure(ObservationKind.MULTIPLE_TARGETS,
                   'Multiple candidate targets require an explicit detection index')
