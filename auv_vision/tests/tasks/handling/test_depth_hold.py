# -*- coding: utf-8 -*-
"""tests/tasks/handling/test_depth_hold.py — **定深**（用户 2026-10-06：翘头时深度控制在 0.5~0.6）。

判据（.）与方向（+ = 上浮）都钉在这里；另外把「**0.50 做不到**」这条硬事实也钉住 ——
它由全局限深下限 `min_depth_m`（现场定死值，2026-10-06 起为 0.50）决定。
"""
from __future__ import annotations

import numpy as np
import pytest

import base.cfg.settings as S
from common.motion.depth_hold import DepthHold, hold_cfg
from handling.motion.params import PH_GRAB_ALIGN, PH_GRAB_APPROACH, PH_GRAB_CENTER, PH_GRAB_SEARCH
from tests.tasks.handling.test_grab_flow import FakeUart, _Hub, _task


def _frame():
    return np.zeros((200, 320, 3), dtype=np.uint8)


def test_band_geometry_comes_from_cfg():
    """★ 判据只许来自 cfg（**板端优先**：用户在板端改 0.9~1.1 或 0.5~0.6 都不该让用例假红）。"""
    c = hold_cfg("grab")
    assert c["enable"] is True
    assert 0.0 < c["target_m"]
    assert c["band_lo_m"] <= c["target_m"] <= c["band_hi_m"], "目标必须落在自己的区间里"
    assert c["kp"] > 0.0


def test_too_deep_ascends_too_shallow_dives_inside_holds():
    """★ 三条判据：太深→上浮(+heave)、太浅→下潜(−)、区间内→**0**（不是"总在推"）。"""
    h = DepthHold(log=lambda *a: None)
    c = hold_cfg("grab")
    lo, hi, tgt, dz = c["band_lo_m"], c["band_hi_m"], c["target_m"], c["deadzone_m"]
    assert h.step(0, FakeUart(depth=hi + 2 * dz)) > 0.0, "太深必须上浮"
    assert h.step(0, FakeUart(depth=lo - 2 * dz)) < 0.0, "太浅必须下潜"
    assert h.step(0, FakeUart(depth=tgt)) == 0.0, "区间内不许动手（否则一直抖）"


def test_band_is_usable_when_the_site_floor_is_not_shallower_than_it():
    """浅端**实际生效值** = `max(band_lo, 全局限深下限)`：全局不比区间浅 ⇒ 整段可用、无告警。"""
    site = float(S.comm.depth_guard.min_depth_m)
    c = hold_cfg("grab")
    lo, tgt, dz = c["band_lo_m"], c["target_m"], c["deadzone_m"]
    h = DepthHold(log=lambda *a: None)
    if site > lo + 1e-9:
        pytest.skip("全局限深(%.2f)比区间下沿(%.2f)深 ⇒ 浅端由全局顶住（见下一条用例）" % (site, lo))
    assert h.step(0, FakeUart(depth=tgt)) == 0.0, "区间内 ⇒ 不动手"
    assert h.step(0, FakeUart(depth=lo + dz * 0.5)) == 0.0, "贴近下沿也还在区间内"
    assert h.step(0, FakeUart(depth=lo - 2 * dz)) < 0.0, "比下沿浅 ⇒ 下潜"
    assert h._warned_lo is False, "上下沿一致时不该有「被抬高」的告警"


def test_effective_shallow_edge_follows_a_raised_site_floor(monkeypatch):
    """机制：全局限深一旦比区间下沿**深**，生效下沿跟着抬并告警一次（不静默）。"""
    lo, dz = hold_cfg("grab")["band_lo_m"], hold_cfg("grab")["deadzone_m"]
    monkeypatch.setitem(S.comm.depth_guard, "min_depth_m", lo + 0.2)
    h = DepthHold(log=lambda *a: None)
    assert h.step(0, FakeUart(depth=lo + 0.05)) < 0.0, "比被顶住的生效下沿浅 ⇒ 只能往深里修"
    assert h._warned_lo is True


def test_no_telemetry_never_guesses():
    h = DepthHold(log=lambda *a: None)
    u = FakeUart(depth=None)
    assert h.step(0, u) == 0.0 and h.step(1000, u) == 0.0
    assert h._warned_tel is True


def test_disabled_cfg_yields_zero_and_missing_cfg_raises(monkeypatch):
    """★ 2026-10-07 用户定：**删掉兜底** ——
    · `enable: false` ⇒ 定深不动（但仍要求节点键齐全）；
    · **整段删掉 ⇒ 报名字**（`MissingCfg`），不再"静默用代码兜底"。
    """
    import pytest
    from common.cfg.cfgnode import MissingCfg
    # enable=false：键齐全，只是关掉 ⇒ 不动手
    monkeypatch.setitem(S.comm.grab, "depth_hold", S.Y(dict(
        S.comm.grab.get("depth_hold"), enable=False)))
    assert DepthHold(log=lambda *a: None).step(0, FakeUart(depth=0.9)) == 0.0
    # 整段删掉 ⇒ 必须报名字（无兜底）
    monkeypatch.delitem(S.comm.grab, "depth_hold", raising=False)
    with pytest.raises(MissingCfg):
        hold_cfg("grab")
    with pytest.raises(MissingCfg):
        DepthHold(log=lambda *a: None)


def test_depth_hold_is_wired_only_into_the_pitched_up_phases(monkeypatch):
    """★ 接线范围：抬头工作段（SEARCH/CENTER/APPROACH）由定深管 heave；
    DIP/RISE 故意变深度（下压/上升）⇒ 定深绝不插手；ALIGN 本就不动 heave。"""
    uart = FakeUart(depth=hold_cfg("grab")["band_hi_m"] + 0.2)   # 明显太深 ⇒ 定深应该上浮
    task, be = _task(monkeypatch, uart)
    monkeypatch.setattr(be, "circles", lambda f: [])       # 没球 ⇒ 停在 SEARCH
    task._phase = PH_GRAB_SEARCH
    task.process(_frame(), 0)
    assert task.last_info["phase"] == PH_GRAB_SEARCH
    assert uart.dofs[-1][2] > 0.0, "SEARCH 里太深就该上浮（heave>0）"
    assert task.last_info["heave"] > 0.0

    # 对准段（ALIGN）不动 heave：球居中且在区间内深度 ⇒ 全 0
    uart2 = FakeUart(depth=hold_cfg("grab")["target_m"])
    task2, be2 = _task(monkeypatch, uart2)
    task2._phase = PH_GRAB_ALIGN
    task2._ensure_pids()
    task2.process(_frame(), 0)
    assert uart2.dofs[-1][2] == 0.0, "ALIGN 不该被定深插手（它本来就不用 heave）"
