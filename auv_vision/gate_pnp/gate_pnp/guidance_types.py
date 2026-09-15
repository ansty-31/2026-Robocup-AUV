"""Per-frame observations and discrete guidance outputs, not actuator commands."""
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
import json
import math
import numpy as np
from .models import PIPELINE_ID, PoseResult, _plain


class ObservationKind(str, Enum):
    TARGET='TARGET'
    NO_TARGET='NO_TARGET'
    ERROR='ERROR'
    MULTIPLE_TARGETS='MULTIPLE_TARGETS'


class Phase(str, Enum):
    ACQUIRE='ACQUIRE'
    ALIGN='ALIGN'
    PASSING='PASSING'
    COMPLETE='COMPLETE'
    CANCELLED='CANCELLED'


class Action(str, Enum):
    HOLD='HOLD'
    TURN_RIGHT='TURN_RIGHT'
    TURN_LEFT='TURN_LEFT'
    DESCEND='DESCEND'
    ASCEND='ASCEND'
    BACKWARD='BACKWARD'
    SCAN='SCAN'
    ALIGN_TO_GATE='ALIGN_TO_GATE'
    START_PASS='START_PASS'
    CONTINUE_PASS='CONTINUE_PASS'


@dataclass(frozen=True)
class GateObservation:
    frame_id: int
    timestamp: float
    image_size: tuple[int,int]
    target_id: str = 'default'
    kind: ObservationKind = ObservationKind.TARGET
    points: np.ndarray | None = None
    confidence: np.ndarray | None = None
    bbox_xyxy: np.ndarray | None = None
    detection_confidence: float | None = None
    target_index: int | None = None
    preprocessing_id: str = PIPELINE_ID
    error: str = ''


@dataclass(frozen=True)
class GuidanceConfig:
    visible_confidence: float = .5
    too_close_height_ratio: float = .9
    boundary_margin_ratio: float = .02
    action_confirm_s: float = .2
    action_min_frames: int = 3
    no_target_confirm_s: float = .5
    scan_half_period_s: float = 3.
    pnp_hold_s: float = .3
    pnp_min_frames: int = 3
    alignment_x_tolerance_m: float = .03
    alignment_y_tolerance_m: float = .03
    alignment_axis_tolerance_deg: float = 5.
    alignment_hold_s: float = .5
    alignment_min_frames: int = 5
    max_frame_age_s: float = .5
    max_frame_gap_s: float = .25

    def validate(self):
        for field in fields(self):
            value=getattr(self,field.name)
            if field.name.endswith('_frames'):
                if type(value) is not int or value<1:
                    raise ValueError(f'{field.name} must be a positive integer')
            elif isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                raise ValueError(f'{field.name} must be finite and positive')
        if self.visible_confidence>1 or self.too_close_height_ratio>1:
            raise ValueError('Confidence and image ratios must not exceed one')
        if self.boundary_margin_ratio>=.5:
            raise ValueError('Boundary margin must be smaller than half the frame')
        if self.alignment_axis_tolerance_deg>=90:
            raise ValueError('Alignment direction tolerance must be smaller than 90 degrees')

    @classmethod
    def from_file(cls,path):
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(data,dict) or set(data)-{f.name for f in fields(cls)}:
            raise ValueError('Unknown guidance configuration fields')
        config=cls(**data)
        config.validate()
        return config


@dataclass(frozen=True)
class AlignmentError:
    horizontal_m: float
    vertical_m: float
    optical_axis_error_deg: float
    optical_axis_gate: np.ndarray
    desired_offset_gate_xy_m: np.ndarray


@dataclass(frozen=True)
class GuidanceDecision:
    phase: Phase
    action: Action
    reason: str
    target_id: str
    frame_id: int | None = None
    timestamp: float | None = None
    target_index: int | None = None
    visible_points: tuple[str,...] = ()
    pnp_result: PoseResult | None = None
    alignment_error: AlignmentError | None = None
    scan_direction: str | None = None
    speed_profile: str | None = None

    def to_dict(self):
        return _plain(asdict(self))
