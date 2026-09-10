"""Independent underwater gate PnP. Importing never loads a model or image."""
from .models import PIPELINE_ID, Observation, PnPConfig, PoseCandidate, PoseResult, Status
from .camera import CameraGeometry, geometry_from_files
from .solver import estimate_gate_pose
from .guidance_types import (Action, Phase, ObservationKind, GateObservation,
                             GuidanceConfig, GuidanceDecision, AlignmentError)
from .guidance import GateGuidance
from .partial_adapter import extract_gate_observations, select_gate_observation


def estimate_from_ultralytics(result, camera_geometry, config=None, **kwargs):
    """Lazy adapter import: no Ultralytics or Torch dependency in the core."""
    from .adapter import estimate_from_ultralytics as adapt
    return adapt(result,camera_geometry,config,**kwargs)


__all__=['PIPELINE_ID','Observation','PnPConfig','PoseCandidate','PoseResult','Status',
         'CameraGeometry','geometry_from_files','estimate_gate_pose','estimate_from_ultralytics',
         'Action','Phase','ObservationKind','GateObservation','GuidanceConfig',
         'GuidanceDecision','AlignmentError','GateGuidance',
         'extract_gate_observations','select_gate_observation']
