import tempfile
import unittest
from pathlib import Path
import numpy as np
from gate_pnp import geometry_from_files


class UnicodePathTests(unittest.TestCase):
    def test_calibration_loads_from_chinese_path_without_modification(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/'中文 标定目录'
            folder.mkdir()
            source=(root/'config/front_camera.yaml').read_bytes()
            calibration=folder/'前置相机.yaml'
            calibration.write_bytes(source)
            geometry=geometry_from_files(calibration,root/'config/vision.yaml')
            self.assertEqual(geometry.image_size,(640,640))
            self.assertTrue(np.isfinite(geometry.K).all())
            self.assertEqual(calibration.read_bytes(),source)
