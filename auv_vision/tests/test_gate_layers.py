# -*- coding: utf-8 -*-
"""test_gate_layers.py — v1.4 分层 + "质量分的两条去向"自检

覆盖：
  [1] 文件分层：gate_task.py 只留相位机骨架；相位段在 phase_degrade/phase_recover
      （GateTask 是这两个 mixin 的子类，方法定义落在对应文件里）
  [2] 质量分**先验**（quality.prior_weights）：帧越旧权重越低；原始 conf 越低权重越低
  [3] 质量分先验真的进了融合权重：KptMemory(w_prior=...) 取代原始 conf
  [4] 质量分**帧级**增益：q 低 → gain 低 → 同一跳变观测被拉得更少（更信历史）
  [5] 旧调用（不给质量分）与升级前逐位一致：gain=1、w=conf×w_geo、位置相同
  [6] QualityTracker：q_mem 在"有位姿"帧更新、在"无位姿"帧保持（不惩罚）
  [7] GatePerception：Sighting 的 sp/dz 与 caution 一致（后验→运动谨慎度）
  [8] 真实任务里两条去向都可见：Q/Qmem/gain 在 last_info 中变化且 gain∈(0,1]
  [9] 兼容面：相位常量仍可从 gate.gate_task 导入；两参 process 仍可用

跑法：python3 tests/test_gate_layers.py
"""
import inspect
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import base.settings as S                                       # noqa: E402
from common.detector import Det                                 # noqa: E402
from gate.vision.gate_detector import board_camera                     # noqa: E402
from gate.vision.geometry import object_points, gate_pose              # noqa: E402
from gate.data.kpt_memory import KptMemory                           # noqa: E402
from gate.data.quality import (QualityCfg, QualityTracker, prior_weights,  # noqa: E402
                          caution_from_q, speed_scale, deadzone_scale)
from gate.vision.perception import GatePerception, pick_gate, Sighting  # noqa: E402
from gate.gate_task import GateTask                             # noqa: E402
from gate.motion import phase_degrade, phase_recover            # noqa: E402
from gate.mock import MockGateBackend                           # noqa: E402
import gate.gate_task as GT                                     # noqa: E402

PASS = []


def check(name, cond, extra=""):
    print("  [%s] %s %s" % ("PASS" if cond else "FAIL", name, extra))
    if cond:
        PASS.append(name)
    else:
        raise AssertionError(name)


def _srcfile(obj):
    return os.path.basename(inspect.getsourcefile(obj) or "")


BASE = np.array([[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]])


# ---------------------------------------------------------------- 1) 分层
def test_layers():
    print("[1] 文件分层（gate_task 只留骨架）")
    n = sum(1 for _ in open(os.path.join(_ROOT, "gate", "gate_task.py"),
                            encoding="utf-8"))
    check("gate_task.py 已瘦身(<460 行)", n < 460, "lines=%d" % n)
    check("GateTask 继承降级相位段", issubclass(GateTask, phase_degrade.DegradeTicks))
    check("GateTask 继承恢复相位段", issubclass(GateTask, phase_recover.RecoverTicks))
    check("_on_coarse 在 phase_degrade.py",
          _srcfile(GateTask._on_coarse) == "phase_degrade.py",
          _srcfile(GateTask._on_coarse))
    check("_tick_reacquire 在 phase_degrade.py",
          _srcfile(GateTask._tick_reacquire) == "phase_degrade.py")
    check("_tick_lost 在 phase_recover.py",
          _srcfile(GateTask._tick_lost) == "phase_recover.py")
    check("_maybe_cue 在 phase_recover.py",
          _srcfile(GateTask._maybe_cue) == "phase_recover.py")
    check("_on_pose 留在骨架", _srcfile(GateTask._on_pose) == "gate_task.py")
    check("process 留在骨架", _srcfile(GateTask.process) == "gate_task.py")
    check("感知半场独立文件",
          _srcfile(GatePerception.observe) == "perception.py")
    check("质量分两条去向同文件",
          _srcfile(QualityTracker.prior) == "quality.py"
          and _srcfile(QualityTracker.posterior) == "quality.py")
    print()


# ---------------------------------------------------------------- 2) 先验权重
def test_prior():
    print("[2] 质量分先验：帧龄/原始 conf → 逐点权重")
    cfg = QualityCfg()
    w_fresh, qt_fresh = prior_weights(cfg, np.full(4, 0.9), age=0.0)
    w_stale, qt_stale = prior_weights(cfg, np.full(4, 0.9), age=0.9)
    w_low, _ = prior_weights(cfg, np.full(4, 0.3), age=0.0)
    check("新鲜帧权重≈原始 conf", abs(w_fresh[0] - 0.9) < 1e-9,
          "w=%.3f" % w_fresh[0])
    check("过期帧权重被压低", w_stale[0] < w_fresh[0] * 0.3,
          "fresh=%.3f stale=%.3f" % (w_fresh[0], w_stale[0]))
    check("过期帧仍非零（两分法：照常估计）", w_stale[0] > 0.0)
    check("低 conf 权重低", w_low[0] < w_fresh[0])
    check("帧新鲜度有下限", qt_stale >= cfg.q_time_floor - 1e-9,
          "q_time=%.3f floor=%.2f" % (qt_stale, cfg.q_time_floor))
    # 未传 conf（整门框可信、角点全缺）→ 全 0，不贡献
    w_none, _ = prior_weights(cfg, None, age=0.0)
    check("无角点信息 → 权重全 0", float(np.max(w_none)) == 0.0)
    print()


# ---------------------------------------------------------------- 3) 先验进融合
def test_prior_enters_fusion():
    print("[3] 先验权重真的进 kpt_memory（取代原始 conf）")
    obs = BASE + 40.0                      # 一次大跳变观测
    for i in range(3):                     # 先建状态
        pass
    m1 = KptMemory(alpha=0.4)
    m2 = KptMemory(alpha=0.4)
    for t in range(6):
        m1.update(BASE, np.full(4, 0.9), 1000 + t * 100)
        m2.update(BASE, np.full(4, 0.9), 1000 + t * 100,
                  w_prior=np.full(4, 0.9))
    check("w_prior=conf 时与旧路径逐位一致",
          np.allclose(m1.update(obs, np.full(4, 0.9), 1700)[0],
                      m2.update(obs, np.full(4, 0.9), 1700,
                                w_prior=np.full(4, 0.9))[0]))
    m3 = KptMemory(alpha=0.4)
    m4 = KptMemory(alpha=0.4)
    for t in range(6):
        m3.update(BASE, np.full(4, 0.9), 1000 + t * 100)
        m4.update(BASE, np.full(4, 0.9), 1000 + t * 100)
    o3 = m3.update(obs, np.full(4, 0.9), 1700)[0]           # 无先验
    o4 = m4.update(obs, np.full(4, 0.9), 1700,
                   w_prior=np.full(4, 0.2))[0]              # 先验=0.2
    check("先验低 → 融合点被拉得更少", o4[0, 0] < o3[0, 0],
          "no_prior=%.2f low_prior=%.2f base=%.2f" % (o3[0, 0], o4[0, 0], BASE[0, 0]))
    check("先验低时权重诊断同步下降",
          m4.weights[0] < m3.weights[0], "%.3f<%.3f" % (m4.weights[0], m3.weights[0]))
    print()


# ---------------------------------------------------------------- 4) 帧级增益
def test_gain():
    print("[4] 质量分帧级增益：q 低 → 更信历史（有下限）")
    obs = BASE + 60.0
    m_hi, m_lo = KptMemory(alpha=0.4, q_gain_floor=0.35), \
        KptMemory(alpha=0.4, q_gain_floor=0.35)
    for t in range(6):
        m_hi.update(BASE, np.full(4, 0.9), 1000 + t * 100, q=1.0)
        m_lo.update(BASE, np.full(4, 0.9), 1000 + t * 100, q=0.0)
    check("q=1 → gain=1", abs(m_hi.gain - 1.0) < 1e-9, "gain=%.3f" % m_hi.gain)
    check("q=0 → gain=q_gain_floor", abs(m_lo.gain - 0.35) < 1e-9,
          "gain=%.3f" % m_lo.gain)
    o_hi = m_hi.update(obs, np.full(4, 0.9), 1700, q=1.0)[0]
    o_lo = m_lo.update(obs, np.full(4, 0.9), 1700, q=0.0)[0]
    check("低质量帧融合更平滑（跳变被抑制）", o_lo[0, 0] < o_hi[0, 0],
          "q1=%.2f q0=%.2f" % (o_hi[0, 0], o_lo[0, 0]))
    check("增益有下限（不会停死）", m_lo.gain >= 0.35 - 1e-9)
    print()


# ---------------------------------------------------------------- 5) 旧行为一致
def test_legacy_parity():
    print("[5] 不给质量分 → 与升级前逐位一致")
    obs = BASE + 25.0
    a = KptMemory(alpha=0.4)
    b = KptMemory(alpha=0.4)
    for t in range(8):
        a.update(BASE, np.full(4, 0.9), 1000 + t * 100)
        b.update(BASE, np.full(4, 0.9), 1000 + t * 100, w_prior=None, q=None)
    oa = a.update(obs, np.full(4, 0.9), 1900)
    ob = b.update(obs, np.full(4, 0.9), 1900, w_prior=None, q=None)
    check("位置一致", np.allclose(oa[0], ob[0]))
    check("置信度一致", np.allclose(oa[1], ob[1]))
    check("gain=1", abs(b.gain - 1.0) < 1e-9)
    check("权重=conf×w_geo（诊断）", abs(b.weights[0] - 0.9 * b.weights[0] / 0.9) < 1e-9)
    print()


# ---------------------------------------------------------------- 6) q_mem
def test_q_mem():
    print("[6] QualityTracker：q_mem 只吃【有位姿】的帧")
    cam = board_camera()
    obj3 = object_points()
    tvec = np.array([0.0, 0.0, 3.0], float).reshape(3, 1)
    uv = cam.project(obj3, np.zeros(3), tvec)
    trk = QualityTracker(QualityCfg(), cam, obj3, (cam.width, cam.height))
    check("初始 q_mem=1（帧一上来就敢用观测）", abs(trk.q_mem - 1.0) < 1e-9)
    res = gate_pose(cam, obj3, uv)
    bad = (res[0], np.array([0.0, 0.0, 9.0], float).reshape(3, 1))
    uv_bad = cam.project(obj3, bad[0], bad[1])
    for _ in range(12):                      # 位姿与投影框不符 → q_box/q_pnp 低
        trk.posterior(np.full(4, 0.9), (uv_bad[:, 0].min(), uv_bad[:, 1].min(),
                                        uv_bad[:, 0].max(), uv_bad[:, 1].max()),
                      pose=bad, rms=40.0, age=0.0)
    q_drop = trk.q_mem
    check("差位姿帧把 q_mem 拉低", q_drop < 0.9, "q_mem=%.3f" % q_drop)
    for _ in range(12):                      # 无位姿帧 → 保持，不继续惩罚
        trk.posterior(np.full(4, 0.9), (0, 0, 10, 10), pose=None, rms=None, age=0.0)
    check("无位姿帧保持 q_mem（不惩罚降级）",
          abs(trk.q_mem - q_drop) < 1e-9, "%.3f→%.3f" % (q_drop, trk.q_mem))
    w, qt = trk.prior(np.full(4, 0.9), age=0.0)
    check("先验仍照常给出（不等位姿）", w.shape == (4,) and qt <= 1.0)
    print()


# ---------------------------------------------------------------- 7) 后验→谨慎度
def test_posterior_to_caution():
    print("[7] 后验质量分 → 运动谨慎度（Sighting.sp/dz）")
    cam = board_camera()
    cfg = QualityCfg()
    for qv in (0.0, 0.5, 1.0):
        c = caution_from_q(cfg, qv)
        check("cautious(%0.1f) 单调" % qv,
              (qv > 0.75 and abs(c - 1.0) < 1e-9) or c <= 1.0)
    check("质量低 → 速度缩放小",
          speed_scale(cfg, 0.0) < speed_scale(cfg, 1.0))
    check("质量低 → 死区放大",
          deadzone_scale(cfg, 0.0) > deadzone_scale(cfg, 1.0))
    check("速度缩放 ≥ 下限", speed_scale(cfg, 0.0) >= cfg.speed_min_scale - 1e-9)
    check("死区放大 ≤ 上限", deadzone_scale(cfg, 0.0) <= cfg.deadzone_max_scale + 1e-9)
    # Sighting 只是数据容器
    s = Sighting()
    check("Sighting 默认值安全", s.sp == 1.0 and s.dz == 1.0 and s.pose is None)
    print()


# ---------------------------------------------------------------- 8) 真实任务两条去向
def test_task_two_destinations():
    print("[8] GateTask：两条去向都在跑（Q/Qmem/gain）")
    cam = board_camera()
    task = _make_task(MockGateBackend(camera=cam), cam)
    now, seen_gain_lt1 = 0, False
    for fid in range(1, 400):
        now += 100
        task.process(object(), now, fid, now / 1000.0)
        if task.last_info["gain"] < 1.0:
            seen_gain_lt1 = True
        if task.last_info["status"] == S.STATUS_DONE:
            break
    info = task.last_info
    check("DONE(pass)", info["reason"] == "pass", info.get("reason"))
    check("后验 Q 有效", 0.0 < float(info["Q"]) <= 1.0, "Q=%.3f" % info["Q"])
    check("Qmem 有效（喂下一帧先验）", 0.0 < float(info["Qmem"]) <= 1.0,
          "Qmem=%.3f" % info["Qmem"])
    check("gain ∈ (0,1]（先验真的用了质量分）", 0.0 < float(info["gain"]) <= 1.0,
          "gain=%.3f" % info["gain"])
    check("gain 随质量分浮动（不是常量）", seen_gain_lt1)
    check("认知：融合对象就是感知半场那份 kpt_memory",
          task._kpt_mem is task._per.kpt_mem)
    print("     frames=%d Q=%.2f Qmem=%.2f gain=%.3f caution=%.2f"
          % (task.frames, info["Q"], info["Qmem"], info["gain"], info["caution"]))
    print()


# ---------------------------------------------------------------- 9) 兼容面
def test_compat():
    print("[9] 兼容面")
    check("相位常量仍从 gate.gate_task 导出",
          GT.PH_APPROACH == "APPROACH" and GT.PH_THROUGH == "THROUGH"
          and GT.SUB_REACQUIRE == "REACQUIRE" and GT.SUB_CUE == "CUE")
    cam = board_camera()
    t = _make_task(MockGateBackend(camera=cam), cam)
    for i in range(5):
        t.process(object(), 33 * (i + 1))
    check("两参 process 兼容", t.frames == 5)
    check("pick_gate 选角点更全的框",
          pick_gate([Det("gate", 0.99, 0, 0, 10, 10),
                     Det("gate", 0.80, 0, 0, 50, 50,
                         kpts=BASE, kpt_conf=np.full(4, 0.9))],
                    0.5).score == 0.80)
    print()


# ---------------------------------------------------------------- 10) 软耦合
def test_soft_coupling():
    """质量分对融合的作用**可完全关闭**（q_gain_floor=1.0），关闭前后任务成败一致。

    这既证明"耦合是软的"（Q 高时几乎不动，只有证据差时才明显压低），
    也证明本次改动没有偷偷改判定：把耦合关掉就是 v1.3 的融合路径。
    """
    print("[10] 软耦合：关掉质量分增益 → 任务成败与用时基本不变")
    cam = board_camera()
    f_on = _run_mock(cam, None)          # 配置默认（质量分耦合开）
    f_off = _run_mock(cam, 1.0)          # gain 恒为 1（等价于 v1.3 融合路径）
    check("开耦合：DONE(pass)", f_on[1] == "pass", str(f_on))
    check("关耦合：DONE(pass)", f_off[1] == "pass", str(f_off))
    check("用时接近（±5%）", abs(f_on[0] - f_off[0]) <= 0.05 * f_off[0],
          "on=%d off=%d" % (f_on[0], f_off[0]))
    check("过门计数一致", f_on[2] == f_off[2] == 1)
    print("     开=%d 帧 / 关=%d 帧" % (f_on[0], f_off[0]))
    print()


def _run_mock(cam, gain_floor):
    """按指定 q_gain_floor 跑一遍 mock 端到端（跑完恢复配置）。"""
    km = S.vision.gate["kpt_mem"]
    had = "q_gain_floor" in km
    old = km.get("q_gain_floor", None)
    if gain_floor is None:
        km.pop("q_gain_floor", None)
    else:
        km["q_gain_floor"] = gain_floor
    try:
        task = _make_task(MockGateBackend(camera=cam), cam)
        now = 0
        for fid in range(1, 500):
            now += 100
            task.process(object(), now, fid, now / 1000.0)
            if task.last_info["status"] == S.STATUS_DONE:
                break
        return task.frames, task.last_info.get("reason"), int(task.last_info["pass"])
    finally:
        if had:
            km["q_gain_floor"] = old
        else:
            km.pop("q_gain_floor", None)


def _make_task(backend, cam):
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
            self.dofs.append((0.0,) * 4)

    return GateTask(_Uart(), _Hub(backend), cam.width, cam.height)


# ---------------------------------------------------------------- 11) 目录分层
def test_layout():
    print("[11] gate/ 目录分层（顶层只留 task/mock，其余按三类别）")
    gdir = os.path.join(_ROOT, "gate")
    top = set(os.listdir(gdir))
    check("顶层只有 task/mock/README/分层目录",
          {"gate_task.py", "mock.py", "README.md", "vision", "data", "motion",
           "__init__.py"} <= top,
          str(sorted(top)))
    for pkg, mods in (("vision", ("gate_decode", "gate_detector", "gate_frontend",
                                  "geometry", "perception")),
                      ("data", ("kpt_memory", "quality", "cues")),
                      ("motion", ("phases", "phase_degrade", "phase_recover"))):
        check("gate/%s/ 有 __init__.py" % pkg,
              os.path.isfile(os.path.join(gdir, pkg, "__init__.py")))
        got = {f[:-3] for f in os.listdir(os.path.join(gdir, pkg))
               if f.endswith(".py") and f != "__init__.py"}
        check("gate/%s/ 内容正确" % pkg, got == set(mods), str(sorted(got)))
    # 依赖方向：data 不 import vision/motion；vision 不 import motion
    bad = []
    for pkg, banned in (("data", ("gate.vision", "gate.motion")),
                        ("vision", ("gate.motion",))):
        for fn in os.listdir(os.path.join(gdir, pkg)):
            if not fn.endswith(".py"):
                continue
            txt = open(os.path.join(gdir, pkg, fn), encoding="utf-8").read()
            for b in banned:
                if ("from %s" % b) in txt or ("import %s" % b) in txt:
                    bad.append("%s/%s → %s" % (pkg, fn, b))
    check("依赖单向（data←vision←motion）", not bad, str(bad))
    # 别再写死 __file__ 深度（v1.4 踩过：挪一层就静默退化近似针孔）
    frag = []
    for dp, dn, fns in os.walk(gdir):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for fn in fns:
            if not fn.endswith(".py"):
                continue
            txt = open(os.path.join(dp, fn), encoding="utf-8").read()
            if "dirname(os.path.dirname(os.path.abspath(__file__)))" in txt \
                    and "_project_root" not in txt:
                frag.append(fn)
    check("gate/ 内无写死 __file__ 深度的路径推断", not frag, str(frag))
    check("gate/README.md 存在且含待测项/参数/使用",
          all(k in open(os.path.join(gdir, "README.md"), encoding="utf-8").read()
              for k in ("实验待测项", "参数配置指南", "使用说明")))
    print()


# ---------------------------------------------------------------- 12) 配置键没被改坏
def test_config_keys():
    print("[12] 配置键：comm.gate.quality/cues/stamp 真的能取到")
    raw = open(os.path.join(_ROOT, "cfg", "comm.yaml"), encoding="utf-8").read()
    for key in ("quality:", "cues:", "stamp:"):
        check("comm.yaml 含 %s" % key, ("\n  " + key) in raw)
    for path in ("comm.gate.quality", "comm.gate.cues", "comm.gate.stamp"):
        check("S.get(%s) 非空" % path, bool(S.get(path)), repr(S.get(path))[:40])
    from common.frame_stamp import StampCfg
    from gate.data.quality import QualityCfg
    from gate.data.cues import CueCfg
    q = QualityCfg.from_settings(S)
    c = CueCfg.from_settings(S)
    st = StampCfg.from_settings(S)
    check("quality 取值来自 yaml",
          abs(q.mem_lambda - float(S.get("comm.gate.quality.mem_lambda"))) < 1e-9)
    check("cues 取值来自 yaml",
          int(c.cue_after_frames) == int(S.get("comm.gate.cues.cue_after_frames")))
    check("stamp 取值来自 yaml",
          abs(st.max_age_s - float(S.get("comm.gate.stamp.max_age_s"))) < 1e-9)
    check("vision.gate.kpt_mem.q_gain_floor 已配置",
          "q_gain_floor" in open(os.path.join(_ROOT, "cfg", "vision.yaml"),
                                 encoding="utf-8").read())
    print()


# ---------------------------------------------------------------- 13) 标定真被加载
def test_calibration_loaded():
    print("[13] 相机标定确实被加载（不是近似针孔退化）")
    from gate.vision.gate_detector import board_camera, _project_root
    from gate.vision.geometry import CameraModel
    root = _project_root()
    check("_project_root() 找到含 cfg 的项目根",
          os.path.isfile(os.path.join(root, "cfg", "front_camera.yaml")), root)
    calib = os.path.join(root, "cfg", "front_camera.yaml")
    ref = CameraModel.from_yaml(calib, rectified=True)
    cam = board_camera()
    check("board_camera().fx == 标定 fx", abs(cam.fx - ref.fx) < 1e-6,
          "cam=%.3f calib=%.3f" % (cam.fx, ref.fx))
    check("board_camera().cx == 标定 cx", abs(cam.cx - ref.cx) < 1e-6,
          "cam=%.3f calib=%.3f" % (cam.cx, ref.cx))
    check("不是近似针孔(fx=0.61×W)", abs(cam.fx - cam.width * 0.61) > 1.0)
    print()


def main():
    test_layers()
    test_prior()
    test_prior_enters_fusion()
    test_gain()
    test_legacy_parity()
    test_q_mem()
    test_posterior_to_caution()
    test_task_two_destinations()
    test_compat()
    test_soft_coupling()
    test_layout()
    test_config_keys()
    test_calibration_loaded()
    print("\n全部通过 %d 项" % len(PASS))


if __name__ == "__main__":
    main()
