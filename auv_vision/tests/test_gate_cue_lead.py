# -*- coding: utf-8 -*-
"""test_gate_cue_lead.py — 主导权三档位自检（v1.5）

三档（`cfg/comm.yaml → gate.cues.mode`，实现见 `gate/data/cues.py`）：

| 档 | 含义 | 位姿在位时的线索权重 wc |
|---|---|---|
| `pose` | **位姿主导** | 0（线索只在位姿不可用时兜底；冲突也不参与仲裁） |
| `auto` | **依据质量自动分配（默认）** | `Q≥lead_q→0`，`Q≤lead_full_q→1`，中间线性；冲突抬到 `conflict_w` |
| `cue`  | **线索主导** | `1-cue_pose_share`（不看 Q） |

覆盖：
  [1] `cue_mode()` 归一化（三档 + 旧名别名 + 未知值）与默认档 = auto
  [2] 三档权重对照表（同一组输入，逐档断言）
  [3] `cue_dof` 方向表 + **安全不变量：线索动作永不含前进**
  [4] `blend_dof` 端点/中点/逐通道/±1 截断（主导权平滑交接）
  [5] `pose` 档：位姿在位时线索不参与（含冲突帧）＝ v1.4 行为
  [6] `auto` 档（默认）：好帧位姿主导；中等帧混合；极差帧交给线索；冲突帧抬权
  [7] `cue` 档：与 Q 无关地主导；`cue_pose_share` 给位姿留份额
  [8] `allow_approach` 控制 APPROACH 相位；主导档同样要过动作确认
  [9] 主导档速度不被谨慎度拖慢；`pose` 档与 v1.3 一致
  [10] 配置面：yaml 默认 auto、三档键齐全

跑法：python3 tests/test_gate_cue_lead.py
"""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import base.settings as S                                       # noqa: E402
from common.detector import Det                                 # noqa: E402
from gate.vision.gate_detector import board_camera              # noqa: E402
from gate.vision.geometry import object_points                  # noqa: E402
from gate.data.cues import (CueCfg, cue_weight, cue_mode, is_lead, cue_dof,  # noqa: E402
                            blend_dof, quality_agreement, MODE_POSE,
                            MODE_AUTO, MODE_CUE, DESCEND, ASCEND,
                            BACKWARD, TURN_LEFT, TURN_RIGHT)
from gate.data.quality import Quality                          # noqa: E402
from gate.vision.perception import Sighting                     # noqa: E402
from gate.motion.phases import PH_ALIGN, PH_APPROACH            # noqa: E402
from gate.gate_task import GateTask                             # noqa: E402

PASS = []


def check(name, cond, extra=""):
    print("  [%s] %s %s" % ("PASS" if cond else "FAIL", name, extra))
    if cond:
        PASS.append(name)
    else:
        raise AssertionError("FAIL: %s %s" % (name, extra))


# ---------------------------------------------------------------- 1) 档位归一化
def test_mode_names():
    print("[1] 档位归一化与默认档")
    check("默认档 = auto（依据质量自动分配）",
          cue_mode(CueCfg()) == MODE_AUTO, cue_mode(CueCfg()))
    check("yaml 默认也是 auto",
          cue_mode(CueCfg.from_settings(S)) == MODE_AUTO,
          cue_mode(CueCfg.from_settings(S)))
    for m in (MODE_POSE, MODE_AUTO, MODE_CUE):
        check("三档原样可用：%s" % m, cue_mode(CueCfg(mode=m)) == m)
    for old, new in (("pose_first", MODE_POSE), ("blend", MODE_AUTO),
                     ("cue_first", MODE_CUE), (" CUE ", MODE_CUE)):
        check("旧名/大小写别名 %s → %s" % (old, new),
              cue_mode(CueCfg(mode=old)) == new)
    check("未知值 → auto（不静默变成 cue）",
          cue_mode(CueCfg(mode="whatever")) == MODE_AUTO)
    check("is_lead：pose 不为真，auto/cue 为真",
          (not is_lead(CueCfg(mode=MODE_POSE)))
          and is_lead(CueCfg(mode=MODE_AUTO)) and is_lead(CueCfg(mode=MODE_CUE)))
    print()


# ---------------------------------------------------------------- 2) 三档对照
def test_three_tiers():
    print("[2] 三档权重对照表（Q 高 / 中 / 极低 / 冲突）")
    qs = (0.90, 0.30, 0.15)
    got = {}
    for m in (MODE_POSE, MODE_AUTO, MODE_CUE):
        c = CueCfg(mode=m, lead_q=0.35, lead_full_q=0.20)
        got[m] = ([round(cue_weight(c, q, DESCEND, True, 3.0), 3) for q in qs],
                  round(cue_weight(c, 0.90, BACKWARD, True, 3.0), 3))
    check("pose：位姿在位恒 0（含冲突）",
          got[MODE_POSE][0] == [0.0, 0.0, 0.0] and got[MODE_POSE][1] == 0.0,
          str(got[MODE_POSE]))
    check("auto：随 Q 连续交接 0 → 0.333 → 1",
          got[MODE_AUTO][0] == [0.0, 0.333, 1.0], str(got[MODE_AUTO][0]))
    check("auto：冲突时抬到 conflict_w（默认 1.0）", got[MODE_AUTO][1] == 1.0)
    check("cue：与 Q 无关，恒 1（冲突也是 1）",
          got[MODE_CUE][0] == [1.0, 1.0, 1.0] and got[MODE_CUE][1] == 1.0,
          str(got[MODE_CUE]))
    c = CueCfg(mode=MODE_AUTO)
    check("auto 单调不增（Q 越低越听线索）",
          cue_weight(c, 0.20, DESCEND, True, 3.0) >=
          cue_weight(c, 0.30, DESCEND, True, 3.0) >=
          cue_weight(c, 0.34, DESCEND, True, 3.0) >= 0.0)
    check("三档一致：位姿不可用 → 1（兜底接管）",
          all(cue_weight(CueCfg(mode=m), 0.9, DESCEND, False) == 1.0
              for m in (MODE_POSE, MODE_AUTO, MODE_CUE)))
    check("record_only / enable=false → 0",
          cue_weight(CueCfg(mode=MODE_CUE, record_only=True), 0.1, DESCEND,
                     True, 3.0) == 0.0
          and cue_weight(CueCfg(mode=MODE_CUE, enable=False), 0.1, DESCEND,
                         True, 3.0) == 0.0)
    check("conflict_w=0 → 冲突不抬权（auto）",
          cue_weight(CueCfg(mode=MODE_AUTO, conflict_w=0.0),
                     0.95, BACKWARD, True, 3.0) == 0.0)
    check("conflict_w=0.5 → 冲突各让一半",
          cue_weight(CueCfg(mode=MODE_AUTO, conflict_w=0.5),
                     0.95, BACKWARD, True, 3.0) == 0.5)
    check("冲突判定依赖 backward_agree_z_m（z 小 = 一致）",
          cue_weight(CueCfg(mode=MODE_AUTO), 0.95, BACKWARD, True, 0.5) == 0.0)
    print("     Q=0.9/0.30/0.15 → pose=%s auto=%s cue=%s"
          % (got[MODE_POSE][0], got[MODE_AUTO][0], got[MODE_CUE][0]))
    print()


# ---------------------------------------------------------------- 3) cue_dof
def test_cue_dof():
    print("[3] cue_dof：方向表 + 安全不变量")
    d = cue_dof(DESCEND, 0.3)
    check("DESCEND → heave<0", d["heave"] < 0 and abs(d["heave"] + 0.3) < 1e-9)
    check("ASCEND → heave>0", cue_dof(ASCEND, 0.3)["heave"] > 0)
    check("TURN_LEFT/RIGHT → ±yaw",
          cue_dof(TURN_LEFT, 0.3)["yaw"] < 0 < cue_dof(TURN_RIGHT, 0.3)["yaw"])
    check("BACKWARD → surge<0", cue_dof(BACKWARD, 0.3)["surge"] < 0)
    check("None → 全 0", all(v == 0.0 for v in cue_dof(None, 0.3).values()))
    forward = [a for a in (DESCEND, ASCEND, BACKWARD, TURN_LEFT, TURN_RIGHT)
               if cue_dof(a, 0.3)["surge"] > 0]
    check("安全不变量：线索永不命令前进(surge>0)", not forward, str(forward))
    print()


# ---------------------------------------------------------------- 4) blend_dof
def test_blend():
    print("[4] blend_dof：主导权平滑交接")
    pose = {"surge": 0.4, "sway": -0.2, "heave": 0.0, "yaw": 0.0}
    cue = cue_dof(DESCEND, 0.3)                     # heave=-0.3
    check("wc=0 → 纯位姿", blend_dof(pose, cue, 0.0)["surge"] == 0.4
          and blend_dof(pose, cue, 0.0)["heave"] == 0.0)
    check("wc=1 → 纯线索", abs(blend_dof(pose, cue, 1.0)["heave"] + 0.3) < 1e-9
          and blend_dof(pose, cue, 1.0)["surge"] == 0.0)
    mid = blend_dof(pose, cue, 0.5)
    check("wc=0.5 → 逐通道中点",
          abs(mid["surge"] - 0.2) < 1e-9 and abs(mid["heave"] + 0.15) < 1e-9)
    check("未涉及通道保持位姿值", abs(mid["sway"] + 0.1) < 1e-9)
    check("结果截断在 ±1",
          abs(blend_dof({"surge": 1.0}, {"surge": 1.0}, 1.0)["surge"]) <= 1.0 + 1e-9)
    print()


# ---------------------------------------------------------------- 5) pose 档
def test_pose_tier():
    print("[5] pose 档（位姿主导）：线索不参与")
    cam = board_camera()
    data = S.comm.gate["cues"]
    old = dict(data)
    try:
        for tier in (MODE_POSE, "pose_only", "pose_first", "POSE"):
            data["mode"] = tier
            task = _task(ConflictBackend(cam), cam)
            ws, surges = set(), []
            for fid in range(1, 12):
                task.process(object(), fid * 100, fid, fid * 0.1)
                ws.add(float(task.last_info["cue_w"]))
                surges.append(float(task.last_info["surge"]))
            check("pose 档（%s）cue_w 恒 0" % tier, ws == {0.0}, str(ws))
            check("pose 档（%s）不出现后退" % tier, min(surges) >= 0.0, str(min(surges)))
            if tier == MODE_POSE:
                check("动作仍为位姿档",
                      str(task.last_info["action"]).startswith(("center", "forward")),
                      task.last_info["action"])
        data["mode"] = "auto"
        task = _task(ConflictBackend(cam), cam)
        task._cuecfg.cue_confirm_frames, task._cuecfg.cue_confirm_s = 1, 0.0
        task._st_cue.reset()
        _, wc = task._cue_lead(_sight(DESCEND, q=0.10), 1000)
        check("对照：auto 档同条件会给权重（别名没有把档位判错）", wc > 0.0, "wc=%.2f" % wc)
    finally:
        data.clear()
        data.update(old)
    print()


# ---------------------------------------------------------------- 6) auto 档
def test_auto_tier():
    print("[6] auto 档（默认）：好帧位姿主导，劣化帧自动交给线索")
    cam = board_camera()
    task = _task(ConflictBackend(cam), cam)
    cfg = task._cuecfg
    check("默认档就是 auto", cue_mode(cfg) == MODE_AUTO)
    cfg.cue_confirm_frames, cfg.cue_confirm_s = 1, 0.0
    # ① 位姿好（Q 高）→ 线索不掺和
    p = task._pose_dof(_sight(DESCEND, q=0.95), 1000, surge=0.3, sway=0.0, heave=0.0)
    check("好帧：位姿主导（cue_w=0）", task.last_info["cue_w"] == 0.0)
    check("好帧：输出等于纯位姿", p["surge"] == 0.3 and p["heave"] == 0.0)
    # ② 位姿中等（Q 落在交接区间）→ 混合
    task._st_cue.reset()
    p = task._pose_dof(_sight(DESCEND, q=0.30), 2000, surge=0.3, sway=0.2, heave=0.1)
    wc = task.last_info["cue_w"]
    check("中等帧：自动分配 0<wc<1", 0.0 < wc < 1.0, "wc=%.3f" % wc)
    wc_true = ((cfg.lead_q - 0.30) / (cfg.lead_q - cfg.lead_full_q))
    check("wc 与 Q 插值一致（诊断值四舍五入到 3 位）",
          abs(wc - wc_true) <= 1e-3, "wc=%.3f 期望=%.3f" % (wc, wc_true))
    check("输出是位姿与线索的混合",
          abs(p["surge"] - 0.3 * (1 - wc_true)) < 1e-6 and p["heave"] < 0.1,
          str({k: round(v, 3) for k, v in p.items()}))
    # ③ 位姿极差（Q 低于 lead_full_q）→ 线索完全主导
    task._st_cue.reset()
    p = task._pose_dof(_sight(DESCEND, q=0.10), 3000, surge=0.3, sway=0.2, heave=0.1)
    check("位姿极差：wc=1，整帧听线索", task.last_info["cue_w"] == 1.0)
    check("输出等于纯线索（heave=-speed_lead、surge=0）",
          abs(p["heave"] + cfg.speed_lead) < 1e-9 and p["surge"] == 0.0,
          str({k: round(v, 3) for k, v in p.items()}))
    # ④ 冲突 → 抬权到 conflict_w（即使 Q 很高）
    task._st_cue.reset()
    p = task._pose_dof(_sight(BACKWARD, q=0.95, z=3.0), 4000,
                       surge=0.3, sway=0.0, heave=0.0)
    check("冲突：Q 高也把主导权给线索", task.last_info["cue_w"] == 1.0)
    check("冲突：下发后退而非前进", p["surge"] < 0.0, "%.3f" % p["surge"])
    print()


# ---------------------------------------------------------------- 7) cue 档
def test_cue_tier():
    print("[7] cue 档（线索主导）：不看 Q，只看线索")
    cam = board_camera()
    task = _task(ConflictBackend(cam), cam)
    cfg = task._cuecfg
    cfg.mode, cfg.cue_confirm_frames, cfg.cue_confirm_s = MODE_CUE, 1, 0.0
    for q in (0.95, 0.50, 0.10):
        task._st_cue.reset()
        task._pose_dof(_sight(DESCEND, q=q), 1000, surge=0.3, sway=0.0, heave=0.0)
        check("cue 档 Q=%.2f → wc=1（与 Q 无关）" % q,
              task.last_info["cue_w"] == 1.0)
    cfg.cue_pose_share = 0.2
    task._st_cue.reset()
    p = task._pose_dof(_sight(DESCEND, q=0.95), 2000, surge=0.4, sway=0.0, heave=0.0)
    check("cue_pose_share=0.2 → 给位姿留 20%",
          abs(task.last_info["cue_w"] - 0.8) < 1e-9
          and abs(p["surge"] - 0.4 * 0.2) < 1e-9,
          "wc=%.2f" % task.last_info["cue_w"])
    cfg.cue_pose_share, cfg.mode = 0.0, MODE_AUTO
    print()


# ---------------------------------------------------------------- 8) 相位/确认
def test_phase_and_confirm():
    print("[8] 相位控制与动作确认（三档共用）")
    cam = board_camera()
    task = _task(ConflictBackend(cam), cam)
    cfg = task._cuecfg
    cfg.mode, cfg.cue_confirm_frames, cfg.cue_confirm_s = MODE_CUE, 3, 0.10
    cfg.allow_approach = False
    task.phase = PH_APPROACH
    check("allow_approach=false → APPROACH 不掺和",
          task._cue_lead(_sight(DESCEND, q=0.1), 1000)[1] == 0.0)
    cfg.allow_approach = True
    ws = [task._cue_lead(_sight(DESCEND, q=0.1), 1000 + i * 100)[1]
          for i in range(4)]
    check("主导档同样要过动作确认（前两帧不出力）",
          ws[0] == 0.0 and ws[1] == 0.0 and ws[3] > 0.0, str(ws))
    task.phase = PH_ALIGN
    task._st_cue.reset()
    ws2 = [task._cue_lead(_sight(DESCEND, q=0.1), 3000 + i * 100)[1]
           for i in range(4)]
    check("ALIGN 恒可掺和（同样受确认约束）", ws2[0] == 0.0 and ws2[3] > 0.0,
          str(ws2))
    cfg.allow_approach, cfg.mode = False, MODE_AUTO
    print()


# ---------------------------------------------------------------- 9) 速度
def test_speed():
    print("[9] 速度：主导档不乘谨慎度，pose 档与 v1.3 一致")
    cam = board_camera()
    task = _task(ConflictBackend(cam), cam)
    cfg = task._cuecfg
    task._sp = 0.5
    cfg.mode, cfg.speed, cfg.speed_lead = MODE_AUTO, 0.15, 0.30
    cfg.speed_uses_caution = False
    check("auto 档：speed_lead、不乘 sp", abs(task._cue_speed() - 0.30) < 1e-9,
          "%.3f" % task._cue_speed())
    cfg.speed_uses_caution = True
    check("auto 档开开关后乘 sp", abs(task._cue_speed() - 0.15) < 1e-9)
    cfg.mode, cfg.speed_uses_caution = MODE_CUE, False
    check("cue 档：speed_lead", abs(task._cue_speed() - 0.30) < 1e-9)
    cfg.mode = MODE_POSE
    check("pose 档：speed×sp（与 v1.3 逐位一致）",
          abs(task._cue_speed() - 0.15 * 0.5) < 1e-9, "%.4f" % task._cue_speed())
    cfg.mode = MODE_AUTO
    print()


# ---------------------------------------------------------------- 10) 配置面
def test_config():
    print("[10] 配置面")
    raw = open(os.path.join(_ROOT, "cfg", "comm.yaml"), encoding="utf-8").read()
    check("yaml 里 mode 默认 auto", "mode: auto" in raw)
    cfg = CueCfg.from_settings(S)
    check("from_settings → auto", cue_mode(cfg) == MODE_AUTO)
    for k in ("mode", "allow_approach", "cue_pose_share", "speed_lead",
              "speed_uses_caution", "lead_q", "lead_full_q", "conflict_w",
              "lead_min_w"):
        check("CueCfg 有 %s" % k, hasattr(cfg, k))
    check("lead_full_q < lead_q（区间有效）", cfg.lead_full_q < cfg.lead_q)
    check("cue_pose_share ≤ 0.5", cfg.cue_pose_share <= 0.5)
    check("互检仍在：冲突给 Q 打折",
          quality_agreement(BACKWARD, True, 3.0, 0.8, CueCfg())[0] == 0.4)
    print()


# ---------------------------------------------------------------- 工具
class ConflictBackend(object):
    """位姿可信（4 角齐全、z=3m）但检测框贴满上下边 → 线索判 BACKWARD（与位姿冲突）。"""

    def __init__(self, camera):
        self.camera = camera
        self.w, self.h = camera.width, camera.height

    def detect(self, frame):
        uv = self.camera.project(object_points(), np.zeros(3),
                                 np.array([0.0, 0.0, 3.0], float).reshape(3, 1))
        return [Det("gate", 0.9, 0, 0, self.w, self.h,
                    kpts=uv, kpt_conf=np.full(4, 0.9, np.float32))]


def _sight(cue, q=0.10, z=3.0):
    """合成"有位姿、质量为 q"的 Sighting（只测仲裁与混合，不走感知链）。"""
    return Sighting(cue=cue, z=z,
                    pose=(np.zeros(3), np.array([0, 0, z], float).reshape(3, 1)),
                    quality=Quality(q, q, q, q, q), caution=q, sp=1.0, dz=1.0)


def _task(backend, cam):
    class _Hub(object):
        def __init__(self, be):
            self.be = be

        def has_extra(self, name):
            return True

        def detect_list(self, name, frame):
            return self.be.detect(frame)

    class _Uart(object):
        def __init__(self):
            self.dofs = []

        def send_dof(self, *dof):
            self.dofs.append(dof)

        def neutral(self):
            pass

    return GateTask(_Uart(), _Hub(backend), cam.width, cam.height)


def main():
    test_mode_names()
    test_three_tiers()
    test_cue_dof()
    test_blend()
    test_pose_tier()
    test_auto_tier()
    test_cue_tier()
    test_phase_and_confirm()
    test_speed()
    test_config()
    print("\n全部通过 %d 项" % len(PASS))


if __name__ == "__main__":
    main()
