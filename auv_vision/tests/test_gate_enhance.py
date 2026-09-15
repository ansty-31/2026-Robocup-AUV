# -*- coding: utf-8 -*-
"""
test_gate_enhance.py — gate v1.3 增强件测试（帧有效性 / 双条件确认 / 质量分 / 恢复线索）

运行：cd auv_vision/auv_vision && python3 tests/test_gate_enhance.py

覆盖：
  1. Streak：时间+帧双条件、key 变化重置
  2. FrameValidator（common/ 共用件）：唯一帧、过期帧"可估计不计确认"、断点、微未来容差(回归)
  3. cues：可见性表 + 贴边后退 + 位姿/线索冲突自检
  4. quality：好位姿 Q 高、rms 大 Q 降、无位姿 q_no_pose、谨慎度单调、缩放有下限
  5. GateTask：重复帧不推进确认（沿用上一帧指令）、四参调用兼容
  6. GateTask：仅上边可见 → 恢复线索接管（source=cue / cue_descend）
  7. GateTask：Mock 端到端 DONE(pass)，带 Q/caution/visible 诊断量
  8. check_pipeline_identity.run()==0（全项目域自证脚本，位于工程根）
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import base.settings as S                                       # noqa: E402
from common.detector import Det                                 # noqa: E402
from gate.vision.gate_detector import board_camera                     # noqa: E402
from gate.vision.geometry import object_points, gate_pose, reproj_rms  # noqa: E402
from gate.mock import MockGateBackend                           # noqa: E402
from gate.gate_task import GateTask                             # noqa: E402
from common.streak import Streak                                # noqa: E402
from common.frame_stamp import FrameStamp, FrameValidator, StampCfg  # noqa: E402
from gate.data.cues import (CueCfg, recovery_cue, visible_mask, too_close,  # noqa: E402
                       cue_conflicts, TURN_RIGHT, TURN_LEFT, DESCEND,
                       ASCEND, BACKWARD)
from gate.data.quality import (QualityCfg, compute_quality, caution_from_q,  # noqa: E402
                          speed_scale, confirm_scale)
import check_pipeline_identity as CPI                            # noqa: E402

PASS = []


def check(name, cond, extra=""):
    if not cond:
        raise AssertionError("FAIL: %s %s" % (name, extra))
    PASS.append(name)
    print("PASS: %s %s" % (name, extra))


class FakeUart(object):
    def __init__(self):
        self.estop_active = False
        self.log = []

    def send_dof(self, surge, sway, heave, yaw):
        self.log.append((surge, sway, heave, yaw))

    def set_motion(self, name):
        pass

    def neutral(self):
        self.log.append((0.0, 0.0, 0.0, 0.0))

    def estop(self):
        self.estop_active = True


class StubHub(object):
    def __init__(self, backend):
        self.backend = backend

    def has_extra(self, task):
        return True

    def detect_list(self, task, frame):
        return self.backend.detect(frame)


def make_task(backend, cam):
    return GateTask(FakeUart(), StubHub(backend), cam.width, cam.height)


# ---------------------------------------------------------------- 1) Streak
def test_streak():
    print("[1] Streak（时间+帧双条件）")
    s = Streak()
    s.add("a", 0.0)
    check("1 帧不达标", not s.ready(0.0, 0.5, 3))
    s.add("a", 0.2)
    s.add("a", 0.4)
    check("帧够但时间不够 → 不达标", not s.ready(0.4, 0.5, 3))
    check("帧够且时间够 → 达标", s.ready(0.5, 0.5, 3))
    s.add("b", 0.6)
    check("key 变化 → 重新计数", s.count == 1 and not s.ready(0.6, 0.5, 3))
    print()


# ---------------------------------------------------------------- 2) 帧有效性
def test_frame_stamp():
    print("[2] FrameValidator（唯一帧 / 有效期 / 微未来容差）")
    v = FrameValidator(StampCfg(max_age_s=0.9, max_gap_s=0.35, adaptive_gap=False))
    check("正常帧", v.validate(FrameStamp(1, 0.0), 0.1).ok)
    check("正常帧2", v.validate(FrameStamp(2, 0.1), 0.2).ok)
    r = v.validate(FrameStamp(2, 0.1), 0.3)
    check("重复帧：不可用", (not r.ok) and r.kind == "duplicate" and not r.usable)
    r = v.validate(FrameStamp(3, 0.2), 2.0)
    check("过期帧：可估计 / 不计确认", (not r.ok) and r.kind == "stale"
          and r.usable and not r.countable)
    v2 = FrameValidator(StampCfg(max_age_s=0.9, max_gap_s=0.35, adaptive_gap=False))
    v2.validate(FrameStamp(1, 0.0), 0.05)
    r = v2.validate(FrameStamp(2, 0.5), 0.55)
    check("采集断点被标记", r.ok and r.discontinuity)
    # 回归：now_ms 取整导致 captured_at 比 now 大 ~1ms（未来容差内 → 仍算新鲜）
    v3 = FrameValidator(StampCfg())
    r = v3.validate(FrameStamp(1, 0.3005), 0.300)
    check("微未来(≤future_tol)不算 future",
          r.ok and r.kind == "ok" and abs(r.age) < 1e-9, r)
    v4 = FrameValidator(StampCfg())
    r = v4.validate(FrameStamp(1, 5.0), 1.0)
    check("明显未来帧被拒", (not r.ok) and r.kind == "future")
    print()


# ---------------------------------------------------------------- 3) 线索表
def test_cues():
    print("[3] 恢复线索表 + 冲突自检")
    cfg = CueCfg(enable=True)
    box, img = (100.0, 100.0, 400.0, 300.0), (640, 640)
    cases = {
        (True, False, False, True): TURN_RIGHT,
        (False, True, True, False): TURN_LEFT,
        (True, True, False, False): DESCEND,
        (False, False, True, True): ASCEND,
        (True, False, False, False): None,      # 单点不猜
        (True, False, True, False): None,       # 对角不猜
        (True, True, True, False): None,        # 3 点不猜
        (True, True, True, True): None,         # 四点交给位姿链
    }
    for mask, want in cases.items():
        got, _ = recovery_cue(mask, box, img, cfg)
        check("mask=%s → %s" % (mask, want), got == want, "got=%s" % got)
    got, _ = recovery_cue((True, True, True, True), (0, 0, 640, 640), img, cfg)
    check("贴边大框 → BACKWARD", got == BACKWARD)
    check("too_close 判定", too_close((0, 5, 640, 635), img, cfg))
    check("visible_mask 阈值", visible_mask([.9, .2, .6, 0], 0.5)
          == (True, False, True, False))
    check("线索BACKWARD vs 位姿 z 大 → 冲突",
          cue_conflicts(BACKWARD, True, 3.0, cfg) != "")
    check("位姿 z 小 → 不冲突", cue_conflicts(BACKWARD, True, 0.9, cfg) == "")
    print()


# ---------------------------------------------------------------- 4) 质量分
def test_quality():
    print("[4] 质量分 Q → 谨慎度")
    cam = board_camera()
    obj3 = object_points()
    rvec = np.array([0.05, -0.03, 0.02], float)
    tvec = np.array([0.02, 0.01, 3.0], float).reshape(3, 1)
    uv = cam.project(obj3, rvec, tvec)
    res = gate_pose(cam, obj3, uv)
    check("合成本征位姿可解", res is not None)
    rms = reproj_rms(cam, obj3, uv, *res)
    box = (uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max())
    cfg = QualityCfg()
    q_ok = compute_quality(cfg, cam, obj3, np.full(4, 0.95), box,
                           (cam.width, cam.height), pose=res, rms=rms, age=0.05)
    q_bad = compute_quality(cfg, cam, obj3, np.full(4, 0.95), box,
                            (cam.width, cam.height), pose=res, rms=30.0, age=0.05)
    q_none = compute_quality(cfg, cam, obj3, np.full(4, 0.2), box,
                             (cam.width, cam.height), pose=None, rms=None, age=0.05)
    check("好位姿 Q 高", q_ok.total > 0.75, "Q=%.3f" % q_ok.total)
    check("rms 大 → Q 降", q_bad.total < q_ok.total and q_bad.q_pnp < q_ok.q_pnp)
    check("无位姿 → q_pnp=q_no_pose", abs(q_none.q_pnp - cfg.q_no_pose) < 1e-9)
    check("谨慎度单调", caution_from_q(cfg, 0.9) > caution_from_q(cfg, 0.1))
    check("速度缩放有下限", speed_scale(cfg, 0.0) >= cfg.speed_min_scale - 1e-9)
    check("确认默认不被质量分放大", abs(confirm_scale(cfg, 0.0) - 1.0) < 1e-9)
    print()


# ---------------------------------------------------------------- 5) 唯一帧
def test_task_unique_frame():
    print("[5] GateTask：重复帧不推进确认")
    cam = board_camera()
    task = make_task(MockGateBackend(camera=cam), cam)
    frame = object()
    now = 0
    for fid in range(1, 8):                     # 先跑几帧建立对齐确认
        now += 100
        task.process(frame, now, fid, now / 1000.0)
    cnt = task._center_cnt                       # 原对齐计数器（底层运动未改）
    kind = task.last_info["frame_kind"]
    now += 100
    task.process(frame, now, 7, now / 1000.0)   # 同一 frame_id 再来一次
    check("重复帧被识别", task.last_info["frame_kind"] == "duplicate",
          task.last_info["frame_kind"])
    check("重复帧不推进原计数器", task._center_cnt == cnt,
          "before=%d after=%d" % (cnt, task._center_cnt))
    check("重复帧沿用上一帧指令",
          str(task.last_info["action"]).startswith("hold_frame"))
    check("正常帧未被误判(frame_kind=%s)" % kind, kind in ("ok", "stale"))
    # 两分法：过期帧仍用于估计(frame_kind=stale)，但不计入确认。
    # 构造：控制时刻前进 1.0s，采集时刻只早 0.95s（age>0.9 且采集递增、间隔正常）
    cnt2 = task._center_cnt
    now += 1000
    task.process(frame, now, 8, now / 1000.0 - 0.95)
    check("过期帧被识别为 stale", task.last_info["frame_kind"] == "stale",
          task.last_info["frame_kind"])
    check("过期帧不推进确认计数", task._center_cnt == cnt2,
          "before=%d after=%d" % (cnt2, task._center_cnt))
    # 四参调用兼容：不传后两参（旧调用方式）也应正常推进
    t2 = make_task(MockGateBackend(camera=cam), cam)
    now = 0
    ok = True
    for _ in range(5):
        now += 33
        t2.process(object(), now)
    check("旧两参调用兼容", ok and t2.frames == 5)
    print()


# ---------------------------------------------------------------- 6) 线索接管
class TopEdgeBackend(object):
    """只给上边两点(TL,TR)可见 → 位姿不可用，线索应为 DESCEND。"""

    def __init__(self, camera):
        self.camera = camera

    def detect(self, frame):
        uv = self.camera.project(object_points(), np.zeros(3),
                                 np.array([0.0, 0.0, 3.0], float).reshape(3, 1))
        conf = np.array([0.9, 0.9, 0.0, 0.0], np.float32)
        x0, y0 = int(uv[:, 0].min()), int(uv[:, 1].min())
        x1, y1 = int(uv[:, 0].max()), int(uv[:, 1].max())
        return [Det("gate", 0.9, x0, y0, x1 - x0, y1 - y0, kpts=uv, kpt_conf=conf)]


def test_cue_takeover():
    print("[6] GateTask：仅上边可见 → 线索接管")
    cam = board_camera()
    task = make_task(TopEdgeBackend(cam), cam)
    now, seen_cue, seen_act = 0, False, False
    for fid in range(1, 40):
        now += 100
        task.process(object(), now, fid, now / 1000.0)
        seen_cue |= (task.last_info["source"] == "cue")
        seen_act |= (task.last_info["action"] == "cue_descend")
    check("出现线索接管(source=cue)", seen_cue, task.last_info)
    check("动作为 cue_descend（低速 heave<0）", seen_act)
    check("可见性记录 1100", task.last_info["visible"] == "1100",
          task.last_info["visible"])
    print()


# ---------------------------------------------------------------- 7) 端到端
def test_task_pass_mock():
    print("[7] GateTask：Mock 端到端 → DONE(pass)")
    cam = board_camera()
    task = make_task(MockGateBackend(camera=cam), cam)
    now = 0
    for fid in range(1, 500):
        now += 100
        task.process(object(), now, fid, now / 1000.0)
        if task.last_info["status"] == S.STATUS_DONE:
            break
    info = task.last_info
    check("DONE(pass)", info["reason"] == "pass", info)
    check("pass 计数", int(info["pass"]) >= 1)
    check("诊断量齐全", all(k in info for k in ("Q", "caution", "visible",
                                                "source", "cue", "frame_kind")))
    print("     轨迹: frames=%d Q=%.2f caution=%.2f source=%s"
          % (task.frames, info["Q"], info["caution"], info["source"]))
    print()


# ---------------------------------------------------------------- 8) 域自证
def test_pipeline_identity():
    print("[8] 域自证脚本")
    check("check_pipeline_identity.run()==0",
          CPI.run(os.path.join(_ROOT, "cfg", "vision.yaml")) == 0)
    print()


def main():
    test_streak()
    test_frame_stamp()
    test_cues()
    test_quality()
    test_task_unique_frame()
    test_cue_takeover()
    test_task_pass_mock()
    test_pipeline_identity()
    print("\n全部通过 %d 项" % len(PASS))


if __name__ == "__main__":
    main()
