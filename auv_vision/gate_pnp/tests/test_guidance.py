"""Temporal behavior tests use synthetic observations only; no physical commands."""
import importlib.util
from dataclasses import replace
from pathlib import Path
import unittest
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


class GuidanceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('gate_pnp.guidance'),
                             'Gate guidance module must be implemented')
        from gate_pnp.guidance import GateGuidance
        from gate_pnp.guidance_types import GateObservation, GuidanceConfig, ObservationKind, Action, Phase
        from gate_pnp import geometry_from_files
        self.g=geometry_from_files(ROOT/'config/front_camera.yaml',ROOT/'config/vision.yaml')
        self.controller=GateGuidance(self.g)
        self.Obs,self.Config,self.Kind,self.Action,self.Phase=GateObservation,GuidanceConfig,ObservationKind,Action,Phase
        self.frame=0

    def obs(self,t,visible=(True,True,True,True),bbox=(150,100,490,540),r=(0,0,0),trans=(0,0,1.8),**kwargs):
        self.frame+=1
        xyz=np.array([[-.35,-.25,0],[.35,-.25,0],[.35,.25,0],[-.35,.25,0]],float)
        pixels=cv2.projectPoints(xyz,np.array(r,float),np.array(trans,float),self.g.K,self.g.distortion)[0].reshape(4,2)
        return self.Obs(self.frame,t,(640,640),points=pixels,confidence=np.array(visible,float),
                        bbox_xyxy=np.array(bbox,float),detection_confidence=.9,**kwargs)

    def step(self,t,**kwargs):
        return self.controller.update(self.obs(t,**kwargs),t)

    def no_gate(self,t):
        self.frame+=1
        return self.Obs(self.frame,t,(640,640),kind=self.Kind.NO_TARGET)

    def advance_to_align(self):
        for t in [0,.1,.2,.3]:
            result=self.step(t,trans=(.08,0,1.8))
        self.assertEqual(result.phase,self.Phase.ALIGN)
        self.assertEqual(result.action,self.Action.ALIGN_TO_GATE)
        return result

    def advance_to_pass(self):
        for t in [0,.1,.2,.3,.4,.5,.6,.7,.8]:
            result=self.step(t)
        self.assertEqual(result.action,self.Action.START_PASS)
        self.assertEqual(result.phase,self.Phase.PASSING)
        return result

    def test_half_gate_requires_three_frames_and_duration(self):
        for vis,expected in [((1,0,0,1),self.Action.TURN_RIGHT),((0,1,1,0),self.Action.TURN_LEFT)]:
            self.controller.start_target('default')
            self.assertEqual(self.step(0,visible=vis).action,self.Action.HOLD)
            self.assertEqual(self.step(.1,visible=vis).action,self.Action.HOLD)
            self.assertEqual(self.step(.2,visible=vis).action,expected)

    def test_too_close_overrides_half_gate_only_when_both_edges(self):
        for t in [0,.1,.2]:
            result=self.step(t,visible=(1,0,0,1),bbox=(100,0,500,640))
        self.assertEqual(result.action,self.Action.BACKWARD)
        self.controller.start_target('default')
        for t in [1,1.1,1.2]:
            result=self.step(t,visible=(1,0,0,1),bbox=(100,0,500,570))
        self.assertEqual(result.action,self.Action.TURN_RIGHT)

    def test_partial_and_ambiguous_patterns_hold_without_fabricated_pose(self):
        for visible in [(1,0,0,0),(1,0,1,0),(1,1,0,0),(0,0,0,0)]:
            self.controller.start_target('default')
            for t in [0,.1,.2,.3]:
                result=self.step(t,visible=visible)
            self.assertEqual(result.action,self.Action.HOLD)
            self.assertIsNone(result.pnp_result)

    def test_no_target_scan_right_then_left_but_no_input_holds(self):
        for t in np.round(np.arange(0,3.61,.1),6):
            result=self.controller.update(self.no_gate(float(t)),float(t))
            if t==.4:
                self.assertEqual(result.action,self.Action.HOLD)
            if t==.5:
                self.assertEqual((result.action,result.scan_direction,result.speed_profile),
                                 (self.Action.SCAN,'RIGHT','SLOW'))
            if t==3.5:
                self.assertEqual(result.scan_direction,'LEFT')
        result=self.controller.update(None,3.7)
        self.assertEqual(result.action,self.Action.HOLD)
        self.assertIsNone(result.scan_direction)

    def test_condition_changes_and_duplicate_frames_break_stability(self):
        self.step(0,visible=(1,0,0,1))
        self.step(.1,visible=(0,1,1,0))
        result=self.step(.2,visible=(1,0,0,1))
        self.assertEqual(result.action,self.Action.HOLD)
        obs=self.obs(.3,visible=(1,0,0,1))
        self.controller.update(obs,.3)
        self.assertEqual(self.controller.update(obs,.4).action,self.Action.HOLD)
        self.assertEqual(self.step(.5,visible=(1,0,0,1)).action,self.Action.HOLD)

    def test_out_of_order_stale_future_and_long_gap_do_not_accumulate(self):
        first=self.obs(0); self.controller.update(first,0)
        for bad,now in [(replace(first,frame_id=0),.1),
                        (self.obs(.1),.8),(self.obs(2),1)]:
            self.assertEqual(self.controller.update(bad,now).action,self.Action.HOLD)
        self.controller.start_target('default')
        self.step(0);self.step(.1)
        result=self.step(.5)
        self.assertEqual(result.phase,self.Phase.ACQUIRE)
        self.assertEqual(result.action,self.Action.HOLD)

    def test_align_needs_center_and_angle_not_just_pnp_success(self):
        result=self.advance_to_align()
        self.assertAlmostEqual(result.alignment_error.horizontal_m,-.08,places=5)
        for t in [.4,.5,.6,.7,.8,.9]:
            result=self.step(t,trans=(.08,0,1.8))
            self.assertEqual(result.action,self.Action.ALIGN_TO_GATE)
        # Camera centered in gate x/y but optical axis tilted 10 degrees.
        r=np.array([0,np.deg2rad(10),0]); R=cv2.Rodrigues(r)[0]
        tvec=-R@np.array([0.,0.,-1.8])
        for t in [1,1.1,1.2,1.3,1.4,1.5]:
            result=self.step(t,r=r,trans=tvec)
            self.assertEqual(result.action,self.Action.ALIGN_TO_GATE)
        self.assertGreater(result.alignment_error.optical_axis_error_deg,5)

    def test_lost_points_in_align_reacquire_and_can_reverse(self):
        self.advance_to_align()
        for t in [.4,.5,.6]:
            result=self.step(t,visible=(1,0,0,1),bbox=(100,0,500,640))
        self.assertEqual(result.phase,self.Phase.ACQUIRE)
        self.assertEqual(result.action,self.Action.BACKWARD)
        for t in [.7,.8,.9]:
            self.assertEqual(self.step(t).phase,self.Phase.ACQUIRE)
        self.assertEqual(self.step(1).phase,self.Phase.ALIGN)

    def test_pose_ambiguity_in_align_returns_to_acquire(self):
        from unittest.mock import patch
        from gate_pnp import PoseResult,Status
        self.advance_to_align()
        with patch('gate_pnp.guidance.estimate_gate_pose',return_value=PoseResult(Status.AMBIGUOUS,'test')):
            result=self.step(.4)
        self.assertEqual(result.phase,self.Phase.ACQUIRE)
        self.assertEqual(result.action,self.Action.HOLD)
        self.assertEqual(result.pnp_result.status,Status.AMBIGUOUS)

    def test_input_gap_keeps_align_but_clears_aligned_streak(self):
        for t in [0,.1,.2,.3,.4,.5,.6]:self.step(t)
        result=self.controller.update(None,.7)
        self.assertEqual(result.phase,self.Phase.ALIGN)
        self.assertEqual(result.action,self.Action.HOLD)
        self.assertIsNone(result.pnp_result)
        for t in [.8,.9,1,1.1,1.2]:
            self.assertEqual(self.step(t).action,self.Action.ALIGN_TO_GATE)
        self.assertEqual(self.step(1.3).action,self.Action.START_PASS)

    def test_start_pass_only_once_and_never_recover_view_afterwards(self):
        self.advance_to_pass()
        for t,vis,bbox in [(.9,(1,0,0,1),(0,0,640,640)),
                          (1,(0,1,1,0),(100,0,500,640)),
                          (1.1,(1,1,1,1),(0,0,640,640))]:
            self.assertEqual(self.step(t,visible=vis,bbox=bbox).action,self.Action.CONTINUE_PASS)
        result=self.controller.update(self.no_gate(1.2),1.2)
        self.assertEqual(result.action,self.Action.CONTINUE_PASS)
        result=self.controller.update(None,1.3)
        self.assertEqual((result.action,result.phase),(self.Action.HOLD,self.Phase.PASSING))
        self.assertEqual(self.step(1.4,visible=(1,0,0,1)).action,self.Action.CONTINUE_PASS)

    def test_external_completion_cancel_and_next_target_reset(self):
        self.advance_to_pass()
        self.controller.complete_pass()
        self.assertEqual(self.step(.9).phase,self.Phase.COMPLETE)
        self.assertEqual(self.step(1).action,self.Action.HOLD)
        self.controller.start_target('gate-2')
        result=self.step(1.1)  # observation still belongs to default, not gate-2
        self.assertEqual(result.action,self.Action.HOLD)
        self.assertEqual(result.phase,self.Phase.ACQUIRE)
        self.controller.cancel()
        self.assertEqual(self.step(1.2,target_id='gate-2').phase,self.Phase.CANCELLED)
        self.controller.start_target('default')
        for t in [0,.1,.2]:result=self.step(t,visible=(1,0,0,1))
        self.assertEqual(result.action,self.Action.TURN_RIGHT)

    def test_error_or_unselected_multiple_targets_never_trigger_search(self):
        for kind in [self.Kind.ERROR,self.Kind.MULTIPLE_TARGETS]:
            for t in [0,.1,.2,.3,.4,.5,.6]:
                obs=replace(self.obs(t),kind=kind,error='unavailable')
                result=self.controller.update(obs,t)
                self.assertEqual(result.action,self.Action.HOLD)
            self.controller.start_target('default')

    def test_configuration_validation_and_json_roundtrip(self):
        from gate_pnp.guidance import GateGuidance
        for name,value in [('action_confirm_s',-1),('action_min_frames',0),
                           ('visible_confidence',1.1),('max_frame_age_s',float('nan'))]:
            with self.assertRaises(ValueError):
                GateGuidance(self.g,replace(self.Config(),**{name:value}))
        config=self.Config.from_file(ROOT/'config/guidance.json')
        self.assertEqual(config.alignment_hold_s,.5)
        import json
        result=self.advance_to_pass()
        self.assertEqual(json.loads(json.dumps(result.to_dict(),allow_nan=False))['action'],'START_PASS')

    def test_invalid_target_index_is_input_fault_not_motion(self):
        import json
        for invalid in [float('nan'),-1,1.5,True,'0']:
            self.controller.start_target('default')
            for t in [0,.1,.2]:
                obs=replace(self.obs(t,visible=(1,0,0,1)),target_index=invalid)
                result=self.controller.update(obs,t)
            self.assertEqual(result.action,self.Action.HOLD)
            json.dumps(result.to_dict(),allow_nan=False)
