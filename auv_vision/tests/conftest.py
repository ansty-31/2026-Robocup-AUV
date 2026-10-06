# -*- coding: utf-8 -*-
"""tests/conftest.py — 公共装置：工程根入 sys.path + 无硬件 SIM 默认值。
（详细用法、判据与实测见 doc/注释历史.md）"""
import os
import sys

os.environ.setdefault("AUV_SIM_MODE", "1")
os.environ["AUV_STREAM"] = "0"      # 测试绝不开推流 socket（强制，外部 AUV_STREAM=1 也压掉）

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest  # noqa: E402  (必须放在环境变量设置之后，语义上从属于本文件的装置)


class _FakeTelemetry(object):
    """假遥测。`yaw_deg=None` = **没有遥测**（与加这个属性之前的行为一致 ⇒ 不影响既有用例）。"""

    def __init__(self):
        self.yaw_deg = None
        self.rate_deg_s = 90.0


class FakeUart(object):
    """记录下发的 DOF 帧，代替真串口（GateTask / BallTask 测试用）。"""

    def __init__(self):
        self.frames = []          # 每帧一条 (surge, sway, heave, yaw)
        self.motions = []         # set_motion 名称
        self.neutral_calls = 0
        self.telemetry = _FakeTelemetry()

    def send_dof(self, surge=0.0, sway=0.0, heave=0.0, yaw=0.0):
        self.frames.append((float(surge), float(sway), float(heave), float(yaw)))
        if self.telemetry.yaw_deg is not None:
            # +yaw = 右转 ⇒ 回传 yaw 减小（σ=-1）；dt 固定 0.1s（用例时基一律 100ms）
            self.telemetry.yaw_deg -= float(yaw) * self.telemetry.rate_deg_s * 0.1

    def set_motion(self, name, force=False):
        self.motions.append(name)

    def neutral(self):
        self.neutral_calls += 1


@pytest.fixture
def fake_uart():
    return FakeUart()
