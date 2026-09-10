"""Public data contracts. Transforms are named destination_from_source."""
from dataclasses import asdict, dataclass, fields
from enum import Enum
import json
from pathlib import Path
import numpy as np

PIPELINE_ID = 'undistort_alpha0_resize640_pixel_centers_v1'


class Status(str, Enum):
    NO_INPUT = 'NO_INPUT'
    NO_TARGET = 'NO_TARGET'
    CONFIG_ERROR = 'CONFIG_ERROR'
    INVALID_KEYPOINTS = 'INVALID_KEYPOINTS'
    SOLVE_FAILED = 'SOLVE_FAILED'
    HIGH_REPROJECTION_ERROR = 'HIGH_REPROJECTION_ERROR'
    AMBIGUOUS = 'AMBIGUOUS'
    SUCCESS = 'SUCCESS'


@dataclass(frozen=True)
class PnPConfig:
    width_m: float = .70
    height_m: float = .50
    target_class_names: tuple[str, ...] = ('gate',)
    keypoint_indices: tuple[int, ...] = (0, 1, 2, 3)
    detection_confidence: float = .5
    keypoint_confidence: float = .5
    max_rms_error_px: float = 3.0
    ambiguity_gap_px: float = .2
    same_pose_rotation_deg: float = .1
    same_pose_translation_m: float = .001
    min_quad_area_px2: float = 4.0
    min_edge_px: float = 1.0
    min_area_edge_ratio: float = 1e-4

    def validate(self):
        positives = ('width_m', 'height_m', 'max_rms_error_px', 'same_pose_rotation_deg',
                     'same_pose_translation_m', 'min_quad_area_px2', 'min_edge_px',
                     'min_area_edge_ratio')
        for name in positives:
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('detection_confidence', 'keypoint_confidence'):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be in [0,1]')
        if not np.isfinite(self.ambiguity_gap_px) or self.ambiguity_gap_px < 0:
            raise ValueError('ambiguity_gap_px must be finite and nonnegative')
        if (not isinstance(self.keypoint_indices, (tuple,list)) or len(self.keypoint_indices) != 4
                or any(type(i) is not int or i < 0 for i in self.keypoint_indices)
                or len(set(self.keypoint_indices)) != 4):
            raise ValueError('keypoint_indices must contain four distinct nonnegative integers')
        if (not isinstance(self.target_class_names, (tuple,list)) or not self.target_class_names
                or any(not isinstance(n,str) or not n for n in self.target_class_names)):
            raise ValueError('target_class_names must contain nonempty class names')

    @classmethod
    def from_file(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        allowed = {f.name for f in fields(cls)}
        if not isinstance(data,dict) or set(data) - allowed:
            raise ValueError('Unknown PnP configuration fields')
        for key in ('keypoint_indices', 'target_class_names'):
            if key in data:
                data[key] = tuple(data[key])
        config = cls(**data)
        config.validate()
        return config


@dataclass(frozen=True)
class Observation:
    points: np.ndarray
    confidence: np.ndarray | None
    image_size: tuple[int, int]
    preprocessing_id: str = PIPELINE_ID
    detection_confidence: float = 1.0
    target_index: int | None = None
    frame_id: str | None = None
    timestamp: float | None = None


@dataclass(frozen=True)
class PoseCandidate:
    T_camera_from_gate: np.ndarray
    T_gate_from_camera: np.ndarray
    camera_position_gate: np.ndarray
    camera_rotation_gate: np.ndarray
    camera_quaternion_xyzw: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    rms_error_px: float
    per_point_error_px: np.ndarray
    refined: bool


def _plain(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _plain(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [_plain(v) for v in value]
    return value


@dataclass(frozen=True)
class PoseResult:
    status: Status
    reason: str
    target_index: int | None = None
    frame_id: str | None = None
    timestamp: float | None = None
    rms_error_px: float | None = None
    candidates: tuple[PoseCandidate, ...] = ()
    pose: PoseCandidate | None = None

    @property
    def ok(self):
        return self.status == Status.SUCCESS and self.pose is not None

    def to_dict(self):
        return _plain(asdict(self))
