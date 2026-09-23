# -*- coding: utf-8 -*-
"""tests/tooling/test_cfg_yaml_head.py — 守住"标定 yaml 的第一个字节必须是 %YAML:1.0"

2026-09-22 现场踩到（板端 cv2 4.11 实测，见 `cfg/front_camera.yaml` 里的说明）：
`cv2.FileStorage` 靠**文件开头**判断格式 ⇒ 前面只要有一个注释行**或一个空行**，
它就 `Input file is invalid` → 抛 SystemError；而 `common.preprocess.calibration_maps`
**不做兜底**（不返回恒等映射）⇒ **相机初始化直接崩**。
本地的 cv2 5.0 更宽容，所以"本地能读"**证明不了**板端能读 —— 这条用例因此检查**字节**。

另外：`base/settings.py` 用的 YAML 是项目自己的解析器（UTF-8 中文没问题），
所以这条规则只适用于**由 cv2 读的相机标定文件**（`cfg/front_camera*.yaml`）。
"""
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
    from gate.geometry import CameraModel
    for rect in (False, True):
        c = CameraModel.from_yaml(path, rectified=rect)
        assert 200.0 < c.fx < 4000.0, "%s: fx=%.1f 不像个内参" % (path, c.fx)
        assert 200.0 < c.fy < 4000.0
        assert 0.0 < c.cx < c.width and 0.0 < c.cy < c.height
    assert len(CameraModel.from_yaml(path, rectified=False).d) == 5


def test_water_and_air_differ_by_dome_factor():
    """空气侧等效焦距必须明显大于水下标定值（罩外折射本领 0.49 vs 0.16）。"""
    from gate.geometry import CameraModel
    w = CameraModel.from_yaml(os.path.join(ROOT, "cfg", "front_camera.yaml"))
    a = CameraModel.from_yaml(os.path.join(ROOT, "cfg", "front_camera_air.yaml"))
    ratio = a.fx / w.fx
    assert 1.25 < ratio < 1.55, "空气/水 焦距比 = %.3f，不像罩介质差" % ratio
