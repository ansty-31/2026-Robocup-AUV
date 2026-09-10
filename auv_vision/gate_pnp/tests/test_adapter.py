"""Ultralytics-shaped test doubles; these tests never load a neural model."""
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


class TensorDouble:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.calls = []

    def detach(self):
        self.calls.append("detach")
        return self

    def cpu(self):
        self.calls.append("cpu")
        return self

    def numpy(self):
        self.calls.append("numpy")
        return self.values


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gate_pnp.adapter import estimate_from_ultralytics
        from gate_pnp.camera import geometry_from_files
        from gate_pnp.models import PnPConfig, Status, PIPELINE_ID
        cls.adapt = staticmethod(estimate_from_ultralytics)
        cls.Config, cls.Status, cls.pipeline_id = PnPConfig, Status, PIPELINE_ID
        root = Path(__file__).resolve().parents[1]
        cls.geometry = geometry_from_files(root / "config/front_camera.yaml", root / "config/vision.yaml")
        object_points = np.array([[-.35, -.25, 0], [.35, -.25, 0], [.35, .25, 0], [-.35, .25, 0]], dtype=np.float64)
        cls.points, _ = cv2.projectPoints(object_points, np.array([.2, -.3, .05]), np.array([.04, -.02, 1.8]), cls.geometry.K, cls.geometry.distortion)
        cls.points = cls.points.reshape(4, 2)

    def result(self, classes=(7,), confidences=None, points=None, names=None):
        count = len(classes)
        return SimpleNamespace(
            orig_shape=self.geometry.image_size[::-1],
            names={7: "gate", 2: "fish"} if names is None else names,
            boxes=SimpleNamespace(cls=np.asarray(classes), conf=np.ones(count) if confidences is None else confidences),
            keypoints=SimpleNamespace(xy=np.repeat(self.points[None], count, axis=0) if points is None else points, conf=np.ones((count, 4))),
        )

    def assert_status(self, result, status, **kwargs):
        actual = self.adapt(result, self.geometry, **kwargs)
        self.assertEqual(len(actual), 1)
        self.assertEqual(actual[0].status, status, actual[0].reason)
        self.assertTrue(actual[0].reason)
        return actual[0]

    def test_none_and_empty_and_non_target(self):
        self.assert_status(None, self.Status.NO_INPUT)
        self.assert_status(self.result(classes=()), self.Status.NO_TARGET)
        self.assert_status(self.result(classes=(2,)), self.Status.NO_TARGET)
        absent = self.result(classes=())
        absent.boxes = None
        absent.keypoints = None
        self.assert_status(absent, self.Status.NO_TARGET)

    def test_multiple_matching_targets_keep_original_indices_and_metadata(self):
        actual = self.adapt(self.result(classes=(2, 7, 7)), self.geometry, frame_id="frame-12", timestamp=12.5)
        self.assertEqual([p.target_index for p in actual], [1, 2])
        for pose in actual:
            self.assertIn(pose.status, (self.Status.SUCCESS, self.Status.AMBIGUOUS), pose.reason)
            self.assertEqual(pose.frame_id, "frame-12")
            self.assertEqual(pose.timestamp, 12.5)
            self.assertTrue(np.isfinite(pose.rms_error_px))

    def test_class_names_are_configurable_and_support_lists(self):
        config = self.Config(target_class_names=("portal",))
        actual = self.adapt(self.result(classes=(1,), names=["fish", "portal"]), self.geometry, config)
        self.assertIn(actual[0].status, (self.Status.SUCCESS, self.Status.AMBIGUOUS), actual[0].reason)

    def test_low_detection_confidence_is_reported_per_target(self):
        actual = self.adapt(self.result(classes=(7, 7), confidences=[.1, .99]), self.geometry)
        self.assertEqual([p.target_index for p in actual], [0, 1])
        self.assertEqual(actual[0].status, self.Status.INVALID_KEYPOINTS)
        self.assertIn(actual[1].status, (self.Status.SUCCESS, self.Status.AMBIGUOUS))

    def test_bad_target_points_do_not_discard_another_target(self):
        result = self.result(classes=(7, 7))
        result.keypoints.xy[0, 0, 0] = np.nan
        actual = self.adapt(result, self.geometry)
        self.assertEqual([p.target_index for p in actual], [0, 1])
        self.assertEqual(actual[0].status, self.Status.INVALID_KEYPOINTS)
        self.assertIn(actual[1].status, (self.Status.SUCCESS, self.Status.AMBIGUOUS))

    def test_missing_or_nonfinite_detection_confidence_is_invalid(self):
        for confidence in (None, [np.nan], [np.inf], [[.99]], [], [1.2]):
            with self.subTest(confidence=confidence):
                result = self.result()
                result.boxes.conf = confidence
                self.assert_status(result, self.Status.INVALID_KEYPOINTS)

    def test_missing_keypoints_or_confidence_does_not_invent_values(self):
        result = self.result()
        result.keypoints = None
        self.assert_status(result, self.Status.INVALID_KEYPOINTS)
        result = self.result()
        result.keypoints.conf = None
        self.assert_status(result, self.Status.INVALID_KEYPOINTS)
        result = self.result()
        result.keypoints.conf[0, 2] = .1
        self.assert_status(result, self.Status.INVALID_KEYPOINTS)

    def test_explicit_keypoint_mapping_is_honored(self):
        result = self.result(points=self.points[[2, 0, 3, 1]][None])
        config = self.Config(keypoint_indices=(1, 3, 0, 2))
        actual = self.adapt(result, self.geometry, config)[0]
        self.assertIn(actual.status, (self.Status.SUCCESS, self.Status.AMBIGUOUS), actual.reason)
        self.assertLess(actual.rms_error_px, 1e-5)

    def test_semantic_corner_order_is_preserved_for_rotated_gate(self):
        object_points = np.array([[-.35, -.25, 0], [.35, -.25, 0], [.35, .25, 0], [-.35, .25, 0]], dtype=np.float64)
        rvec, tvec = np.array([.2, -.3, 2.0]), np.array([.04, -.02, 1.8])
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, self.geometry.K, self.geometry.distortion)
        actual = self.adapt(self.result(points=projected.reshape(1, 4, 2)), self.geometry)[0]
        self.assertEqual(actual.status, self.Status.SUCCESS, actual.reason)
        expected_rotation, _ = cv2.Rodrigues(rvec)
        np.testing.assert_allclose(actual.pose.T_camera_from_gate[:3, :3], expected_rotation, atol=1e-5)
        np.testing.assert_allclose(actual.pose.T_camera_from_gate[:3, 3], tvec, atol=1e-5)

    def test_mismatched_shape_or_coordinate_metadata_is_config_error(self):
        for shape in ((1080, 1920), (640,), (640, 640, 3), (np.nan, 640), (640.5, 640), None):
            with self.subTest(shape=shape):
                result = self.result()
                result.orig_shape = shape
                self.assert_status(result, self.Status.CONFIG_ERROR)
        self.assert_status(self.result(), self.Status.CONFIG_ERROR, preprocessing_id="raw-camera")
        self.assert_status(self.result(), self.Status.CONFIG_ERROR, config=self.Config(keypoint_indices=(0, 0, 2, 3)))

    def test_unsupported_result_and_bad_class_arrays_are_reported(self):
        self.assert_status([], self.Status.CONFIG_ERROR)
        self.assert_status(object(), self.Status.CONFIG_ERROR)
        for classes in ([np.nan], [7.5], [[7]], [float("inf")], [99]):
            with self.subTest(classes=classes):
                result = self.result()
                result.boxes.cls = np.asarray(classes)
                self.assert_status(result, self.Status.CONFIG_ERROR)

    def test_malformed_point_arrays_are_reported(self):
        for points in (self.points, np.zeros((1, 3, 2)), np.zeros((1, 4, 3)), np.zeros((2, 4, 2)), [[["bad", 0]] * 4]):
            with self.subTest(shape=np.shape(points)):
                self.assert_status(self.result(points=points), self.Status.INVALID_KEYPOINTS)
        for conf in (np.ones(4), np.ones((2, 4)), np.ones((1, 3))):
            result = self.result()
            result.keypoints.conf = conf
            self.assert_status(result, self.Status.INVALID_KEYPOINTS)

    def test_keypoints_without_boxes_are_reported(self):
        result = self.result()
        result.boxes = None
        self.assert_status(result, self.Status.INVALID_KEYPOINTS)
        result.boxes = SimpleNamespace(cls=np.array([]), conf=np.array([]))
        self.assert_status(result, self.Status.INVALID_KEYPOINTS)

    def test_tensor_conversion_uses_detach_cpu_numpy_and_pixel_xy(self):
        result = self.result()
        arrays = [TensorDouble(result.boxes.cls), TensorDouble(result.boxes.conf), TensorDouble(result.keypoints.xy), TensorDouble(result.keypoints.conf)]
        result.boxes.cls, result.boxes.conf, result.keypoints.xy, result.keypoints.conf = arrays
        result.keypoints.xyn = np.zeros((1, 4, 2))
        actual = self.adapt(result, self.geometry)[0]
        self.assertIn(actual.status, (self.Status.SUCCESS, self.Status.AMBIGUOUS), actual.reason)
        for tensor in arrays:
            self.assertEqual(tensor.calls, ["detach", "cpu", "numpy"])


if __name__ == "__main__":
    unittest.main()
