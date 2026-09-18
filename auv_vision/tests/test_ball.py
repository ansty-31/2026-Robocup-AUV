# -*- coding: utf-8 -*-
"""tests/test_ball.py — task1_2.ball：BallTask 相位机（命中序列 / 搜索超时）。

用脚本化 hub 直接喂 Det，不加载模型、不开相机；now_ms 由测试推进（不真 sleep）。
"""
import numpy as np

import base.settings as S
from common.detector import Det
from task1_2.ball import (PH_APPROACH, PH_CENTER, PH_DASH, PH_SEARCH, PH_STOP,
                          BallTask)

W, H = 640, 480


class _BallHub(object):
    """脚本化 ball 后端：frame_fn(n) -> Det 或 None，n = 已调用帧号。"""

    def __init__(self, frame_fn):
        self._frame_fn = frame_fn
        self.calls = 0

    def ready(self, task):
        return True

    def detect(self, task, frame):
        d = self._frame_fn(self.calls)
        self.calls += 1
        return d

    def detect_all(self, frame):
        return []


# 故意偏右 40px（dx≈0.125 < center_eps 0.25）：CENTER 必须真的收敛进死区才转进近，
# 而不是因为目标恰好压在画面中心而"白拿"确认。
_DX_PX = 40


def _growing_ball(n):
    """前 3 帧无目标，之后边长 60→ 递增的球（面积占比升过 dash 阈值）。"""
    if n < 3:
        return None
    s = 60 + 8 * (n - 3)
    cx = W // 2 + _DX_PX
    return Det("blue_ball", 0.9, cx - s // 2, H // 2 - s // 2, s, s)


def test_ball_growing_detection_hits(fake_uart):
    """SEARCH→CENTER→APPROACH→DASH→STOP→DONE(hit)，且各段 DOF 约束成立。"""
    task = BallTask(fake_uart, _BallHub(_growing_ball), W, H)
    assert task.ready is True
    frame = np.zeros((H, W, 3), np.uint8)

    rows = []          # (phase, action, 本帧下发的 DOF)
    status = S.STATUS_RUNNING
    for i in range(400):
        status = task.process(frame, 33 * i)
        info = task.last_info
        dof = fake_uart.frames[-1] if fake_uart.frames else (0.0,) * 4
        rows.append((info["phase"], info["action"], dof))
        if status == S.STATUS_DONE:
            break

    phases = []
    for p, _a, _d in rows:
        if not phases or phases[-1] != p:
            phases.append(p)
    assert phases == [PH_SEARCH, PH_CENTER, PH_APPROACH, PH_DASH, PH_STOP]
    assert status == S.STATUS_DONE
    assert task.last_info["reason"] == "hit"

    # CENTER 只 yaw+heave：绝不前进、绝不横移；且确实朝偏心目标转向
    center_yaws = []
    for p, _a, (surge, sway, _heave, yaw) in rows:
        if p == PH_CENTER:
            assert surge == 0.0 and sway == 0.0
            center_yaws.append(yaw)
    assert center_yaws and any(y != 0.0 for y in center_yaws)
    # APPROACH 只用 sway 修水平：yaw=0、heave=0
    for p, _a, (_s, _sw, heave, yaw) in rows:
        if p == PH_APPROACH:
            assert heave == 0.0 and yaw == 0.0
    # DASH = 配置的最高速纯前进
    dash_surges = [d[0] for p, _a, d in rows if p == PH_DASH]
    assert dash_surges
    assert set(dash_surges) == {S.comm.ball.surge_fast}
    # 命中后明确回中位
    assert fake_uart.neutral_calls >= 1


def test_ball_without_detections_times_out(fake_uart):
    """全程无目标：SEARCH 保持，跑满 comm.ball.timeout_ms 后 DONE(timeout)。"""
    task = BallTask(fake_uart, _BallHub(lambda n: None), W, H)
    frame = np.zeros((H, W, 3), np.uint8)

    status = S.STATUS_RUNNING
    for i in range(400):
        status = task.process(frame, 100 * i)
        if status == S.STATUS_DONE:
            break

    assert status == S.STATUS_DONE
    assert task.last_info["reason"] == "timeout"
    assert task.last_info["phase"] == PH_SEARCH          # 从未见到球
    assert 100 * i >= S.comm.ball.timeout_ms             # 确实跑满时限才结束
    assert fake_uart.neutral_calls >= 1

    # 搜索期确实在"脉冲旋转 + 周期性前进探测"，不是原地不动
    surges = [f[0] for f in fake_uart.frames]
    yaws = [f[3] for f in fake_uart.frames]
    assert S.comm.ball.search_advance_surge in surges
    assert S.comm.motion.search_yaw in yaws
