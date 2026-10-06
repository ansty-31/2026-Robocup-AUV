# -*- coding: utf-8 -*-
"""tests/tooling/test_cfg_yaml_head.py — 守住"标定 yaml 的第一个字节必须是 %YAML:1.0"
（详细用法、判据与实测见 doc/注释历史.md）"""
from __future__ import annotations

import glob
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CAM_YAML = sorted(glob.glob(os.path.join(ROOT, "cfg", "front_camera*.yaml")))


def test_some_camera_yaml_exists():
    assert CAM_YAML, "cfg/ 下应该有 front_camera*.yaml"


@pytest.mark.parametrize("path", CAM_YAML or ["cfg/front_camera.yaml"])
def test_camera_yaml_starts_with_marker(path):
    raw = open(path, "rb").read()
    assert raw[:1] != b"#", "%s 以注释开头 → 板端 cv2 4.11 会认不出格式" % path
    assert not raw[:1].isspace(), "%s 以空白/空行开头 → 同上" % path
    assert raw.startswith(b"%YAML:1.0"), "%s 必须从 %%YAML:1.0 开始" % path


@pytest.mark.parametrize("path", CAM_YAML or ["cfg/front_camera.yaml"])
def test_camera_yaml_loads_and_has_sane_intrinsics(path):
    from gate.percept.geometry import CameraModel
    for rect in (False, True):
        c = CameraModel.from_yaml(path, rectified=rect)
        assert 200.0 < c.fx < 4000.0, "%s: fx=%.1f 不像个内参" % (path, c.fx)
        assert 200.0 < c.fy < 4000.0
        assert 0.0 < c.cx < c.width and 0.0 < c.cy < c.height
    assert len(CameraModel.from_yaml(path, rectified=False).d) == 5


def test_water_and_air_differ_by_dome_factor():
    """水下标定的等效焦距应比空气大（罩+折射把视场收窄 ⇒ fx 变大）。"""
    from gate.percept.geometry import CameraModel
    w = CameraModel.from_yaml(os.path.join(ROOT, "cfg", "front_camera.yaml"))
    a = CameraModel.from_yaml(os.path.join(ROOT, "cfg", "front_camera_air.yaml"))
    ratio = w.fx / a.fx
    assert 1.05 < ratio < 1.20, "水/空气 焦距比 = %.3f，不像罩+折射的介质差" % ratio
