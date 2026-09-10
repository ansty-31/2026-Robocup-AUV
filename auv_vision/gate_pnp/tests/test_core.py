"""Synthetic correspondences are confined to tests; no model or camera required."""
import importlib.util
import unittest
from pathlib import Path
from dataclasses import replace
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('gate_pnp'),
                             'The standalone gate_pnp package must exist')
        from gate_pnp import geometry_from_files, PnPConfig, Observation, Status
        from gate_pnp import estimate_gate_pose
        self.g = geometry_from_files(ROOT / 'config/front_camera.yaml', ROOT / 'config/vision.yaml')
        self.config = PnPConfig()
        self.Observation, self.Status = Observation, Status
        self.estimate = estimate_gate_pose
        self.xyz = np.array([[-.35,-.25,0],[.35,-.25,0],[.35,.25,0],[-.35,.25,0]], np.float64)

    def observation(self, r=(.25,-.35,.08), t=(.05,-.03,1.8)):
        xy, _ = cv2.projectPoints(self.xyz, np.array(r, float), np.array(t, float),
                                  self.g.K, self.g.distortion)
        return self.Observation(xy.reshape(4,2), np.ones(4), (640,640),
                                target_index=7, frame_id='frame-1', timestamp=1.25)

    def test_recovers_known_poses_and_inverse_not_tvec(self):
        for r,t in [((0,0,0),(0,0,1.5)), ((.25,-.35,.08),(.05,-.03,1.8)),
                    ((-.4,.3,-.1),(-.1,.05,1.2)), ((.2,.4,.15),(.2,-.1,3.0))]:
            with self.subTest(r=r,t=t):
                result = self.estimate(self.observation(r,t), self.g)
                self.assertEqual(result.status, self.Status.SUCCESS, result.reason)
                p = result.pose
                R,_ = cv2.Rodrigues(np.array(r,float))
                np.testing.assert_allclose(p.T_camera_from_gate[:3,:3],R,atol=1e-5)
                np.testing.assert_allclose(p.T_camera_from_gate[:3,3],t,atol=1e-5)
                np.testing.assert_allclose(p.camera_position_gate,-R.T@np.array(t),atol=1e-5)
                np.testing.assert_allclose(p.camera_rotation_gate,R.T,atol=1e-5)
                np.testing.assert_allclose(p.T_gate_from_camera@p.T_camera_from_gate,np.eye(4),atol=1e-10)
                self.assertAlmostEqual(np.linalg.norm(p.camera_quaternion_xyzw),1,places=10)
                x,y,z,w=p.camera_quaternion_xyzw
                qr=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                             [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                             [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
                np.testing.assert_allclose(qr,R.T,atol=1e-5)
                self.assertLess(result.rms_error_px,1e-5)
                self.assertEqual(result.target_index,7)
                self.assertEqual(result.frame_id,'frame-1')

    def test_no_input_never_reuses_previous_pose(self):
        self.assertTrue(self.estimate(self.observation(),self.g).ok)
        result=self.estimate(None,self.g)
        self.assertEqual(result.status,self.Status.NO_INPUT)
        self.assertIsNone(result.pose)

    def test_invalid_points_do_not_produce_pose(self):
        good=self.observation()
        bad=[np.zeros((3,2)), np.zeros((4,2)), np.array([[1,1],[2,2],[3,3],[4,4]]),
             np.array([[20,20],[40,40],[40,20],[20,40]]),
             np.array([[-1,20],[40,20],[40,40],[20,40]]),
             np.array([[np.nan,20],[40,20],[40,40],[20,40]])]
        for points in bad:
            with self.subTest(points=points):
                result=self.estimate(replace(good,points=points),self.g)
                self.assertEqual(result.status,self.Status.INVALID_KEYPOINTS)
                self.assertIsNone(result.pose)

    def test_confidence_required_and_thresholds_enforced(self):
        good=self.observation()
        for confidence in [None,[1,1,.49,1],[1,1,np.nan,1],[1,1,2,1],[1,1,1]]:
            self.assertEqual(self.estimate(replace(good,confidence=confidence),self.g).status,
                             self.Status.INVALID_KEYPOINTS)
        self.assertEqual(self.estimate(replace(good,detection_confidence=.49),self.g).status,
                         self.Status.INVALID_KEYPOINTS)

    def test_mismatched_geometry_or_declared_double_undistortion_rejected(self):
        good=self.observation()
        for obs in [replace(good,image_size=(1280,720)),
                    replace(good,preprocessing_id='raw_distorted'),
                    replace(good,preprocessing_id='undistorted_twice')]:
            self.assertEqual(self.estimate(obs,self.g).status,self.Status.CONFIG_ERROR)
        bad=replace(self.g,distortion=np.ones(5))
        self.assertEqual(self.estimate(good,bad).status,self.Status.CONFIG_ERROR)
        bad=replace(self.g,K=np.eye(3))
        self.assertEqual(self.estimate(good,bad).status,self.Status.CONFIG_ERROR)

    def test_invalid_config_is_reported(self):
        for config in [replace(self.config,max_rms_error_px=-1),
                       replace(self.config,keypoint_indices=(0,0,1,2)),
                       replace(self.config,keypoint_confidence=float('nan'))]:
            self.assertEqual(self.estimate(self.observation(),self.g,config).status,
                             self.Status.CONFIG_ERROR)

    def test_noisy_non_rectangle_triggers_error_gate(self):
        obs=self.observation()
        points=obs.points.copy(); points[0]+=[15,-9]
        result=self.estimate(replace(obs,points=points),self.g,
                             replace(self.config,max_rms_error_px=.05))
        self.assertEqual(result.status,self.Status.HIGH_REPROJECTION_ERROR)
        self.assertIsNone(result.pose)
        self.assertGreater(result.rms_error_px,.05)

    def test_resize_matches_opencv_pixel_center_sampling(self):
        # OpenCV INTER_LINEAR samples src=(dst+.5)/scale-.5. Linear ramps expose
        # the mapping independently from our intrinsic-matrix computation.
        new=np.array(self.g.metadata['new_camera_matrix'])
        ramp=np.tile(np.arange(1280,dtype=np.float32),(720,1))
        resized=cv2.resize(ramp,(640,640),interpolation=cv2.INTER_LINEAR)
        p=self.g.K@np.linalg.inv(new)@np.array([resized[200,150],225.0625,1.0])
        np.testing.assert_allclose(p[:2],[150,200],atol=1e-4)
        self.assertTrue(np.all(self.g.distortion==0))

    def test_matches_attachment_undistortion_projection(self):
        meta=self.g.metadata
        original=np.array(meta['original_camera_matrix']); dist=np.array(meta['original_distortion'])
        new=np.array(meta['new_camera_matrix'])
        xyz=self.xyz.copy(); r=np.array([.2,-.2,.05]); t=np.array([.01,.02,1.4])
        original_pixels,_=cv2.projectPoints(xyz,r,t,original,dist)
        criteria=(cv2.TERM_CRITERIA_COUNT+cv2.TERM_CRITERIA_EPS,100,1e-12)
        if hasattr(cv2,'undistortPointsIter'):
            undist=cv2.undistortPointsIter(original_pixels,original,dist,None,new,criteria)
        else:
            undist=cv2.undistortPoints(original_pixels,original,dist,P=new,criteria=criteria)
        undist=undist.reshape(4,2)
        expected=(undist+.5)*np.array([.5,640/720])-.5
        projected,_=cv2.projectPoints(xyz,r,t,self.g.K,np.zeros(5))
        np.testing.assert_allclose(projected.reshape(4,2),expected,atol=1e-5)

    def test_json_serialization_has_no_nan_or_arrays(self):
        import json
        result=self.estimate(self.observation(),self.g)
        encoded=json.dumps(result.to_dict(),allow_nan=False)
        self.assertEqual(json.loads(encoded)['status'],'SUCCESS')

    def test_real_planar_ambiguity_keeps_both_without_selected_pose(self):
        # Near-frontal, far gate with fixed subpixel perturbations. This is not
        # a mocked solver: both IPPE seeds retain separate local minima after LM.
        pixels=np.array([[281.9111024887708,299.93579755051894],
                         [350.9324867384873,301.10997765974827],
                         [351.1435290788366,387.5469087559445],
                         [282.4784230714664,386.6740240295408]])
        # Literal fixture pixels use a known virtual camera, independent of the
        # platform-specific getOptimalNewCameraMatrix result.
        fixture_k=np.array([[391.26832821931316,0,316.64756694306493],
                            [0,693.1449132246055,343.86685847342064],[0,0,1]])
        normalized=(pixels-fixture_k[:2,2])/np.array([fixture_k[0,0],fixture_k[1,1]])
        pixels=normalized*np.array([self.g.K[0,0],self.g.K[1,1]])+self.g.K[:2,2]
        result=self.estimate(replace(self.observation(),points=pixels),self.g)
        self.assertEqual(result.status,self.Status.AMBIGUOUS)
        self.assertEqual(len(result.candidates),2)
        self.assertIsNone(result.pose)
        self.assertFalse(result.ok)
        self.assertLess(abs(result.candidates[1].rms_error_px-result.candidates[0].rms_error_px),.2)

    def test_solver_failure_and_negative_depth_do_not_leak_pose(self):
        from unittest.mock import patch
        with patch('gate_pnp.solver.cv2.solvePnPGeneric',return_value=(0,(),(),None)):
            self.assertEqual(self.estimate(self.observation(),self.g).status,self.Status.SOLVE_FAILED)

        with patch('gate_pnp.solver.cv2.solvePnPGeneric',side_effect=cv2.error('test failure')):
            self.assertEqual(self.estimate(self.observation(),self.g).status,self.Status.SOLVE_FAILED)
        r=np.zeros((3,1)); t=np.array([[0.],[0.],[-1.]])
        with patch('gate_pnp.solver.cv2.solvePnPGeneric',return_value=(1,[r],[t],None)):
            result=self.estimate(self.observation(),self.g)
        self.assertEqual(result.status,self.Status.SOLVE_FAILED)
        self.assertIsNone(result.pose)
        # A positive center is insufficient when one or more corners are behind.
        r=np.array([[0.],[1.3],[0.]])
        t=np.array([[0.],[0.],[.1]])
        with patch('gate_pnp.solver.cv2.solvePnPGeneric',return_value=(1,[r],[t],None)):
            self.assertEqual(self.estimate(self.observation(),self.g).status,self.Status.SOLVE_FAILED)

    def test_ambiguity_gap_is_checked_across_error_limit_boundary(self):
        pixels=np.array([[297.98965225911036,309.7779557801343],
                         [342.12643952458325,323.3412250481195],
                         [340.3087066909564,387.39228208931496],
                         [289.1422829034561,363.4933023881339]])
        result=self.estimate(replace(self.observation(),points=pixels),self.g)
        self.assertEqual(result.status,self.Status.AMBIGUOUS)
        self.assertIsNone(result.pose)

    def test_failed_or_worse_refinement_preserves_initial_solution(self):
        from unittest.mock import patch
        with patch('gate_pnp.solver.cv2.solvePnPRefineLM',side_effect=cv2.error('LM failure')):
            result=self.estimate(self.observation(),self.g)
        self.assertEqual(result.status,self.Status.SUCCESS)
        self.assertLess(result.rms_error_px,1e-5)
        self.assertFalse(result.pose.refined)
        wrong=(np.zeros((3,1)),np.array([[0.],[0.],[10.]]))
        with patch('gate_pnp.solver.cv2.solvePnPRefineLM',return_value=wrong):
            result=self.estimate(self.observation(),self.g)
        self.assertEqual(result.status,self.Status.SUCCESS)
        self.assertLess(result.rms_error_px,1e-5)

    def test_quaternion_remains_valid_near_180_degree_rotation(self):
        r=np.array([0.,0.,3.13]); t=np.array([0.,0.,1.8])
        result=self.estimate(self.observation(r,t),self.g)
        self.assertEqual(result.status,self.Status.SUCCESS)
        np.testing.assert_allclose(result.pose.camera_quaternion_xyzw,
                                  [0,0,-np.sin(3.13/2),np.cos(3.13/2)],atol=1e-5)

    def test_geometry_roundtrip_config_and_version_mismatch(self):
        import tempfile,json
        from gate_pnp import CameraGeometry,PnPConfig,geometry_from_files
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'camera.json'
            self.g.save(path)
            loaded=CameraGeometry.from_file(path)
            self.assertTrue(self.estimate(self.observation(),loaded).ok)
            data=json.loads(path.read_text(encoding='utf-8'))
            data['metadata']['opencv_version']='different'
            path.write_text(json.dumps(data),encoding='utf-8')
            with self.assertRaises(ValueError):
                CameraGeometry.from_file(path)
            settings=PnPConfig.from_file(ROOT/'config/pnp.json')
            self.assertTrue(self.estimate(self.observation(),loaded,settings).ok)
            path.write_text('{"typo_threshold":0.5}',encoding='utf-8')
            with self.assertRaises(ValueError):
                PnPConfig.from_file(path)
            import yaml
            config=yaml.safe_load((ROOT/'config/vision.yaml').read_text(encoding='utf-8'))
            config['image']['resize_mode']='letterbox'
            path.write_text(yaml.safe_dump(config),encoding='utf-8')
            with self.assertRaises(ValueError):
                geometry_from_files(ROOT/'config/front_camera.yaml',path)

    def test_cli_without_data_returns_no_input_and_can_export_geometry(self):
        import subprocess,sys,json,tempfile
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'geometry.json'
            proc=subprocess.run([sys.executable,'-m','gate_pnp','--export-geometry',str(target)],
                                cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
            self.assertEqual(proc.returncode,0,proc.stderr)
            data=json.loads(proc.stdout)
            self.assertEqual(data['status'],'NO_INPUT')
            self.assertIsNone(data['pose'])
            self.assertTrue(target.is_file())

    def test_core_import_and_no_input_without_optional_model_packages(self):
        import subprocess,sys
        code='''import sys, importlib.abc
class DenyModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('ultralytics','torch'):
            raise ImportError('Model libraries deliberately blocked')
sys.meta_path.insert(0,DenyModels())
from gate_pnp import estimate_gate_pose, estimate_from_ultralytics, Status
assert estimate_gate_pose(None,None).status == Status.NO_INPUT
assert estimate_from_ultralytics(None,None)[0].status == Status.NO_INPUT
'''
        proc=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(proc.returncode,0,proc.stderr)


if __name__=='__main__':
    unittest.main()
