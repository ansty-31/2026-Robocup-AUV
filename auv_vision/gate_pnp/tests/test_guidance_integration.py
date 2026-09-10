import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest
import cv2
import numpy as np
from gate_pnp import (GateGuidance, Action, Phase, geometry_from_files,
                       extract_gate_observations, select_gate_observation)

ROOT=Path(__file__).resolve().parents[1]


class GuidanceIntegrationTests(unittest.TestCase):
    def test_partial_results_to_alignment_and_pass_with_real_pnp(self):
        geometry=geometry_from_files(ROOT/'config/front_camera.yaml',ROOT/'config/vision.yaml')
        controller=GateGuidance(geometry)
        xyz=np.array([[-.35,-.25,0],[.35,-.25,0],[.35,.25,0],[-.35,.25,0]],float)
        xy=cv2.projectPoints(xyz,np.zeros(3),np.array([0.,0.,1.8]),geometry.K,geometry.distortion)[0].reshape(4,2)
        starts=0
        for frame_id in range(15):
            t=round(frame_id*.1,6)
            conf=[1,0,0,1] if frame_id<3 else [1,1,1,1]
            result=NS(orig_shape=(640,640),names={0:'gate'},
                      boxes=NS(cls=np.array([0]),conf=np.array([.9]),xyxy=np.array([[150,100,490,540]])),
                      keypoints=NS(xy=xy[None],conf=np.array([conf],float)))
            observations=extract_gate_observations(result,geometry,frame_id=frame_id,timestamp=t)
            decision=controller.update(select_gate_observation(observations),now=t)
            starts+=decision.action==Action.START_PASS
            if frame_id<2:
                self.assertEqual(decision.action,Action.HOLD)
                self.assertIsNone(decision.pnp_result)
            if frame_id==2:
                self.assertEqual(decision.action,Action.TURN_RIGHT)
            if frame_id==6:
                self.assertEqual(decision.phase,Phase.ALIGN)
        self.assertEqual(starts,1)
        self.assertEqual(decision.action,Action.CONTINUE_PASS)

    def test_cli_guidance_mode_has_no_model_and_no_motion_input(self):
        result=subprocess.run([sys.executable,'-B','-m','gate_pnp','--guidance'],cwd=ROOT,
                              capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(result.returncode,0,result.stderr)
        output=json.loads(result.stdout)
        self.assertEqual(output['phase'],'ACQUIRE')
        self.assertEqual(output['action'],'HOLD')
        self.assertIsNone(output['pnp_result'])
