# -*- coding: utf-8 -*-
"""tests/conftest.py — 公共装置：工程根入 sys.path + 无硬件 SIM 默认值。

必须在任何测试模块 import base.settings **之前** 执行：
  * AUV_SIM_MODE=1 → S.SIM_MODE=True（UartController 默认只打印，不发串口）
  * AUV_STREAM=0   → SimCamera.read() 不新建 UDP 推流器（测试不碰网络）

模块以顶层包互相引用（import base.settings as S），所以工程根必须在 sys.path 上。
"""
import os
import sys

os.environ.setdefault("AUV_SIM_MODE", "1")
os.environ["AUV_STREAM"] = "0"      # 测试绝不开推流 socket（强制，外部 AUV_STREAM=1 也压掉）

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest  # noqa: E402  (必须放在环境变量设置之后，语义上从属于本文件的装置)


class FakeUart(object):
    """记录下发的 DOF 帧，代替真串口（GateTask / BallTask 测试用）。"""

    def __init__(self):
        self.frames = []          # 每帧一条 (surge, sway, heave, yaw)
        self.motions = []         # set_motion 名称
        self.neutral_calls = 0

    def send_dof(self, surge=0.0, sway=0.0, heave=0.0, yaw=0.0):
        self.frames.append((float(surge), float(sway), float(heave), float(yaw)))

    def set_motion(self, name, force=False):
        self.motions.append(name)

    def neutral(self):
        self.neutral_calls += 1


@pytest.fixture
def fake_uart():
    return FakeUart()
