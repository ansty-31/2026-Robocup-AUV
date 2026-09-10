"""Partial model observations are preserved; selection never changes targets."""
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class TensorDouble:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class PartialAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gate_pnp.partial_adapter import extract_gate_observations, select_gate_observation
        from gate_pnp.guidance_types import ObservationKind
        from gate_pnp.camera import geometry_from_files
        from gate_pnp.models import PnPConfig, PIPELINE_ID
        cls.extract = staticmethod(extract_gate_observations)
        cls.select = staticmethod(select_gate_observation)
        cls.Kind, cls.Config, cls.pipeline = ObservationKind, PnPConfig, PIPELINE_ID
        root = Path(__file__).resolve().parents[1]
        cls.geometry = geometry_from_files(root / 'config/front_camera.yaml', root / 'config/vision.yaml')

    def result(self, classes=(7,)):
        count = len(classes)
        points = np.array([[100., 150.], [500., 150.], [500., 450.], [100., 450.]])
        return SimpleNamespace(
            orig_shape=(640, 640), names={2: 'fish', 7: 'gate'},
            boxes=SimpleNamespace(cls=np.asarray(classes), conf=np.full(count, .9),
                                  xyxy=np.tile([90., 140., 510., 460.], (count, 1))),
            keypoints=SimpleNamespace(xy=np.repeat(points[None], count, axis=0),
                                      conf=np.ones((count, 4))),
        )

    def adapt(self, result, **kwargs):
        return self.extract(result, self.geometry, frame_id=42, timestamp=3.5,
                            target_id='mission-gate', **kwargs)

    def assert_error(self, result, **kwargs):
        actual = self.adapt(result, **kwargs)
        self.assertTrue(actual)
        for observation in actual:
            self.assertEqual(observation.kind, self.Kind.ERROR)
            self.assertTrue(observation.error)
        return actual

    def test_half_visible_is_preserved_in_physical_order(self):
        result = self.result()
        result.keypoints.xy[0, 2:] = [[0., 0.], [np.nan, 900.]]
        result.keypoints.conf[0] = [.95, .9, .01, .2]
        actual = self.adapt(result)[0]
        self.assertEqual(actual.kind, self.Kind.TARGET)
        np.testing.assert_equal(actual.points, result.keypoints.xy[0])
        np.testing.assert_equal(actual.confidence, result.keypoints.conf[0])
        self.assertEqual((actual.frame_id, actual.timestamp, actual.target_id),
                         (42, 3.5, 'mission-gate'))
        self.assertEqual(actual.image_size, (640, 640))
        self.assertEqual(actual.preprocessing_id, self.pipeline)
        result.keypoints.xy[:] = 123
        self.assertEqual(actual.points[0, 0], 100.)

    def test_low_detection_confidence_remains_a_target(self):
        result = self.result()
        result.boxes.conf[0] = .01
        actual = self.adapt(result)[0]
        self.assertEqual(actual.kind, self.Kind.TARGET)
        self.assertEqual(actual.detection_confidence, .01)

    def test_all_targets_retain_original_detection_index(self):
        observations = self.adapt(self.result((2, 7, 7)))
        self.assertEqual([o.target_index for o in observations], [1, 2])
        self.assertEqual([o.target_id for o in observations], ['mission-gate'] * 2)
        self.assertEqual(self.select(observations).kind, self.Kind.MULTIPLE_TARGETS)
        self.assertIs(self.select(observations, target_index=2), observations[1])

    def test_selection_never_falls_back_after_index_mismatch(self):
        observations = self.adapt(self.result((2, 7, 7)))
        for index in (0, 9, -1, True, 1.5):
            with self.subTest(index=index):
                self.assertEqual(self.select(observations, index).kind, self.Kind.ERROR)
        self.assertEqual(self.select(observations[:1], 2).kind, self.Kind.ERROR)
        self.assertEqual(self.select([observations[0]] * 2, 1).kind, self.Kind.ERROR)
        self.assertIs(self.select(observations[:1]), observations[0])

    def test_explicit_selection_preserves_target_error(self):
        result = self.result((7, 7))
        result.boxes.xyxy[0, 0] = -10
        observations = self.adapt(result)
        self.assertEqual(self.select(observations, 0).kind, self.Kind.ERROR)
        self.assertEqual(self.select(observations, 1).kind, self.Kind.TARGET)
        self.assertEqual(self.select(observations).kind, self.Kind.MULTIPLE_TARGETS)

    def test_empty_selection_is_developer_error(self):
        with self.assertRaises(ValueError):
            self.select([])

    def test_empty_frame_is_distinct_from_absent_input(self):
        empty = self.adapt(self.result(()))[0]
        self.assertEqual(empty.kind, self.Kind.NO_TARGET)
        self.assertEqual((empty.frame_id, empty.timestamp), (42, 3.5))
        self.assert_error(None)
        result = self.result(())
        result.boxes = result.keypoints = None
        self.assertEqual(self.adapt(result)[0].kind, self.Kind.NO_TARGET)
        self.assertEqual(self.adapt(self.result((2,)))[0].kind, self.Kind.NO_TARGET)

    def test_keypoints_without_boxes_are_an_error(self):
        result = self.result()
        result.boxes = None
        self.assert_error(result)

    def test_missing_fields_and_bad_shapes_report_each_target(self):
        cases = [('keypoints', 'conf', None), ('keypoints', 'conf', np.ones((3, 3))),
                 ('keypoints', 'xy', np.zeros((3, 4, 3))),
                 ('keypoints', 'xy', np.zeros((2, 4, 2))),
                 ('boxes', 'xyxy', None), ('boxes', 'xyxy', np.zeros((3, 5))),
                 ('boxes', 'conf', None), ('boxes', 'conf', np.ones((3, 1)))]
        for owner, field, value in cases:
            with self.subTest(owner=owner, field=field, value=value):
                result = self.result((2, 7, 7))
                setattr(getattr(result, owner), field, value)
                actual = self.assert_error(result)
                self.assertEqual([o.target_index for o in actual], [1, 2])

    def test_malformed_non_targets_are_errors_not_empty_scene(self):
        cases = [('boxes', 'conf', None), ('boxes', 'xyxy', None),
                 ('keypoints', 'xy', None), ('keypoints', 'conf', None),
                 ('boxes', 'conf', np.ones((1, 1))),
                 ('boxes', 'xyxy', np.ones((1, 3))),
                 ('keypoints', 'xy', np.ones((1, 4, 3))),
                 ('keypoints', 'conf', np.ones((1, 3))),
                 ('boxes', 'conf', [np.nan]), ('boxes', 'conf', [1.1]),
                 ('keypoints', 'conf', [[np.inf, 1, 1, 1]]),
                 ('keypoints', 'conf', [[-.1, 1, 1, 1]]),
                 ('boxes', 'xyxy', [[-1, 10, 100, 100]]),
                 ('boxes', 'xyxy', [[10, 10, 10, 100]]),
                 ('boxes', 'xyxy', [[10, 10, 100, 641]]),
                 ('boxes', 'xyxy', [[np.nan, 10, 100, 100]])]
        for owner, field, value in cases:
            with self.subTest(owner=owner, field=field, value=value):
                result = self.result((2,))
                setattr(getattr(result, owner), field, value)
                actual = self.assert_error(result)
                self.assertEqual(len(actual), 1)
                self.assertIsNone(actual[0].target_index)
        result = self.result((2,))
        result.boxes.conf = result.boxes.xyxy = result.keypoints = None
        self.assertEqual(len(self.assert_error(result)), 1)
        self.assertEqual(self.adapt(self.result((2,)))[0].kind, self.Kind.NO_TARGET)

    def test_invalid_target_confidence_does_not_discard_other_target(self):
        for owner in ('boxes', 'keypoints'):
            for value in (np.nan, np.inf, -.01, 1.01):
                with self.subTest(owner=owner, value=value):
                    result = self.result((7, 7))
                    getattr(result, owner).conf[0] = value
                    actual = self.adapt(result)
                    self.assertEqual([o.kind for o in actual], [self.Kind.ERROR, self.Kind.TARGET])

    def test_invalid_bboxes_do_not_discard_other_target(self):
        for box in ([-1, 1, 10, 10], [1, -1, 10, 10], [10, 1, 10, 10],
                    [1, 10, 10, 10], [1, 1, 641, 10], [1, 1, 10, 641],
                    [np.nan, 1, 10, 10], [1, 1, np.inf, 10]):
            with self.subTest(box=box):
                result = self.result((7, 7))
                result.boxes.xyxy[0] = box
                actual = self.adapt(result)
                self.assertEqual([o.kind for o in actual], [self.Kind.ERROR, self.Kind.TARGET])

    def test_dimension_preprocessing_and_config_errors(self):
        for shape in ((720, 1280), (640,), (640., 640.1), None, [np.nan, 640]):
            result = self.result()
            result.orig_shape = shape
            self.assert_error(result)
        self.assert_error(self.result(), preprocessing_id='raw-camera')
        for indices in ((0, 1, 2, 9), (0, 0, 2, 3), (-1, 1, 2, 3)):
            self.assert_error(self.result(), config=self.Config(keypoint_indices=indices))
        self.assert_error(self.result(), config=self.Config(target_class_names=()))

    def test_class_mapping_and_explicit_semantic_mapping(self):
        result = self.result((1,))
        result.names = ['fish', 'portal']
        result.keypoints.xy[0] = result.keypoints.xy[0, [2, 0, 3, 1]]
        config = self.Config(target_class_names=('portal',), keypoint_indices=(1, 3, 0, 2))
        actual = self.adapt(result, config=config)[0]
        self.assertEqual(actual.kind, self.Kind.TARGET)
        np.testing.assert_equal(actual.points, self.result().keypoints.xy[0])
        for names in ({}, {1: ''}, {1: None}, None):
            result.names = names
            self.assert_error(result, config=config)
        for classes in ([np.nan], [7.5], [-1], [[7]], [99]):
            result = self.result()
            result.boxes.cls = np.asarray(classes)
            self.assert_error(result)

    def test_tensor_pixels_are_consumed_without_rescaling(self):
        result = self.result()
        expected = result.keypoints.xy[0].copy()
        result.keypoints.xyn = np.zeros((1, 4, 2))
        for owner, field in (('boxes', 'cls'), ('boxes', 'conf'), ('boxes', 'xyxy'),
                             ('keypoints', 'xy'), ('keypoints', 'conf')):
            container = getattr(result, owner)
            setattr(container, field, TensorDouble(getattr(container, field)))
        actual = self.adapt(result)[0]
        self.assertEqual(actual.kind, self.Kind.TARGET)
        np.testing.assert_equal(actual.points, expected)


if __name__ == '__main__':
    unittest.main()
