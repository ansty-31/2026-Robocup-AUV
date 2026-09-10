"""Recover full gate view, align, then commit to passing; one selected target.

Only this decision object is stateful. It never sends commands or caches a pose
for future motion. Call update once per new captured frame, using a monotonic
capture timestamp and now in the SAME time domain (seconds).
"""
import math
import numbers
import numpy as np
from .models import Observation, PnPConfig
from .solver import estimate_gate_pose
from .guidance_types import (Action, AlignmentError, GateObservation, GuidanceConfig,
                             GuidanceDecision, ObservationKind, Phase)


class _Streak:
    def __init__(self):
        self.reset()

    def reset(self):
        self.key=None
        self.start=None
        self.count=0

    def add(self,key,timestamp):
        if key!=self.key:
            self.key=key
            self.start=timestamp
            self.count=0
        self.count+=1

    def ready(self,timestamp,seconds,frames=1):
        return self.count>=frames and self.start is not None and timestamp-self.start+1e-9>=seconds


class GateGuidance:
    def __init__(self,camera_geometry,config=None,pnp_config=None):
        self.camera_geometry=camera_geometry
        self.config=GuidanceConfig() if config is None else config
        self.pnp_config=PnPConfig() if pnp_config is None else pnp_config
        self.config.validate()
        self.pnp_config.validate()
        self.camera_geometry.validate()
        self._motion=_Streak()
        self._pnp=_Streak()
        self._aligned=_Streak()
        self._absent=_Streak()
        self.start_target('default')

    def _clear_streaks(self):
        for streak in (self._motion,self._pnp,self._aligned,self._absent):
            streak.reset()

    def start_target(self,target_id):
        """Explicit new target/session; detection indices are not target IDs."""
        if not isinstance(target_id,str) or not target_id.strip():
            raise ValueError('target_id must be a nonempty string')
        self.target_id=target_id
        self.phase=Phase.ACQUIRE
        self._last_frame=None
        self._last_capture=None
        self._last_now=None
        self._clear_streaks()

    def complete_pass(self):
        """External confirmation only; disappearing gate is never completion."""
        if self.phase!=Phase.PASSING:
            raise ValueError('Can only complete a committed passing phase')
        self.phase=Phase.COMPLETE
        self._clear_streaks()

    def cancel(self):
        self.phase=Phase.CANCELLED
        self._clear_streaks()

    def _decision(self,action,reason,obs=None,visible=(),**kwargs):
        return GuidanceDecision(self.phase,action,reason,self.target_id,
                                frame_id=obs.frame_id if obs else None,
                                timestamp=obs.timestamp if obs else None,
                                target_index=obs.target_index if obs else None,
                                visible_points=visible,**kwargs)

    def _hold_fault(self,reason,obs=None):
        self._clear_streaks()
        return self._decision(Action.HOLD,reason,obs)

    def _validate_frame(self,obs,now):
        if isinstance(now,bool) or not isinstance(now,numbers.Real) or not math.isfinite(now):
            raise ValueError('now must be finite monotonic seconds')
        if self._last_now is not None and now<self._last_now:
            raise ValueError('Controller time moved backwards')
        self._last_now=float(now)
        if not isinstance(obs,GateObservation):
            raise ValueError('No new observation frame')
        if isinstance(obs.frame_id,bool) or not isinstance(obs.frame_id,numbers.Integral) or obs.frame_id<0:
            raise ValueError('frame_id must be a nonnegative increasing integer')
        if isinstance(obs.timestamp,bool) or not isinstance(obs.timestamp,numbers.Real) or not math.isfinite(obs.timestamp):
            raise ValueError('Capture timestamp must be finite seconds')
        if self._last_frame is not None and obs.frame_id<=self._last_frame:
            raise ValueError('Repeated or out-of-order frame rejected')
        if self._last_capture is not None and obs.timestamp<=self._last_capture:
            raise ValueError('Repeated or out-of-order capture timestamp rejected')
        age=now-obs.timestamp
        if age<0 or age>self.config.max_frame_age_s+1e-9:
            raise ValueError('Future or expired frame rejected')
        # Record consumed frame freshness even when later semantic validation fails.
        if self._last_capture is not None and obs.timestamp-self._last_capture>self.config.max_frame_gap_s+1e-9:
            self._clear_streaks()
        self._last_frame=int(obs.frame_id)
        self._last_capture=float(obs.timestamp)
        if obs.target_id!=self.target_id:
            raise ValueError('Target changed without explicit start_target/reset')
        if obs.target_index is not None and (isinstance(obs.target_index,bool)
                or not isinstance(obs.target_index,numbers.Integral) or obs.target_index<0):
            raise ValueError('target_index must be None or a nonnegative integer')
        if tuple(obs.image_size)!=self.camera_geometry.image_size or obs.preprocessing_id!=self.camera_geometry.preprocessing_id:
            raise ValueError('Image size or preprocessing identity mismatch')
        if obs.kind not in (ObservationKind.TARGET,ObservationKind.NO_TARGET):
            raise ValueError(obs.error or 'Invalid input or multiple targets require selection')

    def _target_data(self,obs):
        points=np.asarray(obs.points,float)
        conf=np.asarray(obs.confidence,float)
        box=np.asarray(obs.bbox_xyxy,float)
        if points.shape!=(4,2) or conf.shape!=(4,) or box.shape!=(4,):
            raise ValueError('Target needs four pixel pairs, four confidences and one xyxy box')
        if not np.isfinite(conf).all() or np.any((conf<0)|(conf>1)):
            raise ValueError('Keypoint confidences must be finite and in [0,1]')
        if not np.isfinite(box).all():
            raise ValueError('Bounding box must be finite')
        w,h=obs.image_size
        if not (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h):
            raise ValueError('Bounding box must be within the image with positive area')
        score=obs.detection_confidence
        if not isinstance(score,numbers.Real) or not math.isfinite(score) or not 0<=score<=1:
            raise ValueError('Detection confidence must be finite and in [0,1]')
        visible=(conf>=self.config.visible_confidence)&np.isfinite(points).all(axis=1)
        visible&=np.all(points>=0,axis=1)&np.all(points<np.array([w,h]),axis=1)
        return points,conf,box,visible

    def update(self,observation,now):
        """Consume one selected frame; errors HOLD and clear continuity, not phase."""
        obs=observation
        if self.phase in (Phase.COMPLETE,Phase.CANCELLED):
            return self._decision(Action.HOLD,'Task ended; call start_target for a new gate')
        try:
            self.config.validate()
            self.pnp_config.validate()
            self.camera_geometry.validate()
            self._validate_frame(obs,now)
            data=self._target_data(obs) if obs.kind==ObservationKind.TARGET else None
        except (ValueError,TypeError,AttributeError,OverflowError,KeyError) as exc:
            # Do not echo malformed timestamps/IDs into a JSON motion decision.
            return self._hold_fault(str(exc))
        visible=() if data is None else tuple(n for n,v in zip(('TL','TR','BR','BL'),data[3]) if v)
        if self.phase==Phase.PASSING:
            return self._decision(Action.CONTINUE_PASS,'Passing committed; view recovery disabled',obs,visible)
        if data is None or obs.detection_confidence<self.pnp_config.detection_confidence:
            if self.phase==Phase.ALIGN:
                self.phase=Phase.ACQUIRE
                self._clear_streaks()
            return self._no_target(obs)
        points,confidence,box,mask=data
        if self.phase==Phase.ALIGN:
            if mask.all():
                pnp=self._solve(obs,points,confidence)
                if pnp.ok:
                    return self._align(obs,pnp,visible)
            else:
                pnp=None
            self.phase=Phase.ACQUIRE
            self._clear_streaks()
            return self._acquire(obs,data,visible,pnp)
        return self._acquire(obs,data,visible)

    def _solve(self,obs,points,confidence):
        return estimate_gate_pose(Observation(points,confidence,obs.image_size,obs.preprocessing_id,
                                  obs.detection_confidence,obs.target_index,str(obs.frame_id),obs.timestamp),
                                  self.camera_geometry,self.pnp_config)

    def _no_target(self,obs):
        self._motion.reset();self._pnp.reset();self._aligned.reset()
        self._absent.add('absent',obs.timestamp)
        if not self._absent.ready(obs.timestamp,self.config.no_target_confirm_s):
            return self._decision(Action.HOLD,'Confirming absence of a reliable gate',obs)
        scan_age=max(0.,obs.timestamp-self._absent.start-self.config.no_target_confirm_s)
        direction='RIGHT' if int((scan_age+1e-9)//self.config.scan_half_period_s)%2==0 else 'LEFT'
        return self._decision(Action.SCAN,'No reliable gate; slow alternating scan',obs,
                              scan_direction=direction,speed_profile='SLOW')

    def _acquire(self,obs,data,visible,pnp=None):
        points,confidence,box,mask=data
        self._absent.reset();self._aligned.reset()
        h=obs.image_size[1];margin=h*self.config.boundary_margin_ratio
        too_close=((box[3]-box[1])/h>=self.config.too_close_height_ratio
                   and box[1]<=margin and box[3]>=h-margin)
        action=None
        if too_close:
            action=Action.BACKWARD
        elif tuple(mask)==(True,False,False,True):
            action=Action.TURN_RIGHT
        elif tuple(mask)==(False,True,True,False):
            action=Action.TURN_LEFT
        if action:
            self._pnp.reset()
            self._motion.add(action,obs.timestamp)
            if self._motion.ready(obs.timestamp,self.config.action_confirm_s,self.config.action_min_frames):
                return self._decision(action,'Confirmed partial/too-close gate geometry',obs,visible)
            return self._decision(Action.HOLD,'Waiting for consistent recovery condition',obs,visible)
        self._motion.reset()
        if not mask.all():
            self._pnp.reset()
            return self._decision(Action.HOLD,'Partial gate has no unambiguous recovery direction',obs,visible)
        pnp=self._solve(obs,points,confidence) if pnp is None else pnp
        if not pnp.ok:
            self._pnp.reset()
            return self._decision(Action.HOLD,'Full gate PnP is not reliable: '+pnp.reason,obs,visible,pnp_result=pnp)
        self._pnp.add('pnp',obs.timestamp)
        if self._pnp.ready(obs.timestamp,self.config.pnp_hold_s,self.config.pnp_min_frames):
            self.phase=Phase.ALIGN
            self._clear_streaks()
            return self._align(obs,pnp,visible)
        return self._decision(Action.HOLD,'Confirming continuous full-gate PnP',obs,visible,pnp_result=pnp)

    def _align(self,obs,pnp,visible):
        pose=pnp.pose
        position=pose.camera_position_gate
        axis=pose.camera_rotation_gate[:,2]
        angle=float(np.degrees(np.arccos(np.clip(axis[2],-1.,1.))))
        error=AlignmentError(float(position[0]),float(position[1]),angle,axis.copy(),-position[:2].copy())
        aligned=(abs(error.horizontal_m)<=self.config.alignment_x_tolerance_m
                 and abs(error.vertical_m)<=self.config.alignment_y_tolerance_m
                 and angle<=self.config.alignment_axis_tolerance_deg)
        if aligned:
            self._aligned.add('aligned',obs.timestamp)
        else:
            self._aligned.reset()
        if self._aligned.ready(obs.timestamp,self.config.alignment_hold_s,self.config.alignment_min_frames):
            self.phase=Phase.PASSING
            self._clear_streaks()
            return self._decision(Action.START_PASS,'Stable centered position and forward optical axis',obs,visible,
                                  pnp_result=pnp,alignment_error=error)
        return self._decision(Action.ALIGN_TO_GATE,'Align gate-frame x/y and optical axis; no longitudinal target',
                              obs,visible,pnp_result=pnp,alignment_error=error)
