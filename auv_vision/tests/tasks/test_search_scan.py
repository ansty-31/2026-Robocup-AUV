# -*- coding: utf-8 -*-
"""tests/tasks/test_search_scan.py — 公共搜索扫描 `common/motion/search_scan.py`
（详细用法、判据与实测见 doc/注释历史.md）"""
import base.cfg.settings as S
from common.motion.search_scan import Scan, _K_SCAN, _K_SCAN_PID

DEADZONE = 0.138          # 执行器死区（与 tests/tasks/gate/test_gate_flow.py 同一常量）


class _YawSim(object):
    """假船：遥测 yaw 按指令积分。σ=-1（右转 ⇒ raw yaw 减小）。rate_deg_s = 每单位 DOF 的角速度。"""

    def __init__(self, yaw=0.0, sigma=-1.0, rate_deg_s=90.0, dt_ms=100):
        self.yaw = float(yaw)
        self.sigma = float(sigma)
        self.rate = float(rate_deg_s)
        self.dt = dt_ms / 1000.0

    def step(self, cmd):
        self.yaw += self.sigma * float(cmd) * self.rate * self.dt
        return self.yaw


def _run(scan, sim, n_legs, dt_ms=100, t0=1000):
    """跑到走完 n_legs 段，返回 (每帧 (t, cmd, yaw), 段目标列表)。"""
    trace, targets = [], []
    t = t0
    while scan.legs_done < n_legs and t < t0 + n_legs * 20000:
        cmd = scan.step(t, sim.yaw)
        if scan.target_psi is not None and (not targets or targets[-1] != scan.target_psi):
            targets.append(scan.target_psi)
        trace.append((t, cmd, sim.yaw))
        sim.step(cmd)
        t += dt_ms
    return trace, targets


def test_scan_sweeps_pm_span_and_alternates():
    """① 轨迹 = -span → +span → -span …（±span 往复）。"""
    sc = Scan()
    assert sc.span_total_deg == 2.0 * sc.span > 0
    sim = _YawSim(yaw=0.0, sigma=sc.sigma)
    _trace, targets = _run(sc, sim, n_legs=4)
    assert len(targets) >= 4, "至少该走出 4 段，实际 %d" % len(targets)
    span = sc.span
    # 入场朝向 psi0 = σ·0 = 0 ⇒ 段目标应是 -span, +span, -span, +span
    expect = [-span, span, -span, span]
    for got, want in zip(targets[:4], expect):
        assert abs(got - want) < 1e-6, "段目标错：得到 %.2f 期望 %.2f" % (got, want)
    # 改 span_deg 就改扫幅（可配）
    sc2 = Scan(node=S.Y(dict(S.comm.motion.search_scan, span_deg=30.0)))
    assert sc2.span_total_deg == 60.0


def test_scan_pauses_after_each_leg():
    """② 每段到位后停 pause_ms（输出 0）。"""
    sc = Scan()
    sim = _YawSim(yaw=0.0, sigma=sc.sigma)
    trace, targets = _run(sc, sim, n_legs=2)
    pause_frames = [c for (_t, c, _y) in trace if c == 0.0]
    assert pause_frames, "整段没有停顿帧（pause 没生效）"
    # 停顿总时长应 ≥ 一段 pause_ms（两段之间至少一次完整停顿）
    n_zero = len(pause_frames)
    assert n_zero * 100 >= sc.pause_ms * 0.5, \
        "停顿帧太少（%d 帧 ≈ %.0fms，pause_ms=%.0f）" % (n_zero, n_zero * 100.0, sc.pause_ms)


def test_scan_does_not_turn_without_telemetry():
    """③ 没遥测 ⇒ 不转（闭环量缺失时不乱转）。"""
    sc = Scan()
    assert sc.step(1000, None) == 0.0
    assert sc.step(1100, None) == 0.0
    assert sc.psi0 is None, "没遥测就不该定入场朝向"


def test_scan_output_within_outmax_and_above_actuator_deadzone():
    """④ 出力 ≤ out_max；且非零出力不得落在执行器死区里（0<|c|<0.138 = 实际 0 推力）。"""
    sc = Scan()
    sim = _YawSim(yaw=0.0, sigma=sc.sigma)
    trace, _t = _run(sc, sim, n_legs=2)
    om = float(sc._pid.out_max)
    for _t_ms, c, _y in trace:
        assert abs(c) <= om + 1e-9, "出力 %.3f 超 out_max %.3f" % (c, om)
    assert max(abs(c) for (_t, c, _y) in trace) >= DEADZONE, \
        "最大出力都不过执行器死区 %.3f ⇒ 实际 0 推力，等于没扫" % DEADZONE
    # cfg 的 out_max 必须**明显高于**执行器死区，否则"还剩二十几度就没推力"（当年 turn_deg 的警告）。
    import base.cfg.settings as _S
    _om_cfg = float((_S.get("comm.motion.search_scan.pid", {}) or {}).get("out_max"))
    assert _om_cfg >= 1.05 * DEADZONE, \
        "cfg out_max=%.2f 已到执行器死区 %.3f 的下限（当年警告线就是 0.15）" % (_om_cfg, DEADZONE)


def test_scan_converges_with_correct_polarity():
    """★ 极性对（σ=-1，+yaw=右转 ⇒ 回传 yaw 减小）⇒ **慢慢收敛**，不转圈。"""
    sc = Scan()
    sim = _YawSim(yaw=0.0, sigma=sc.sigma)      # 与 Scan 的 σ 一致 = 极性正确
    trace, targets = _run(sc, sim, n_legs=6)
    span, tol = sc.span, sc.tol
    ps = [sc.sigma * y for (_t, _c, y) in trace]        # 各帧 psi
    assert max(ps) <= span + tol + 1e-6, "越过 +span（%.1f°）—— 扫描幅度失控" % max(ps)
    assert min(ps) >= -span - tol - 1e-6, "越过 -span（%.1f°）—— 扫描幅度失控" % min(ps)
    # 交替：-span, +span, -span, ...
    for k, want in enumerate([-span, span, -span, span]):
        assert abs(targets[k] - want) < 1e-6, "第 %d 段目标错：%.1f 期望 %.1f" % (k, targets[k], want)


def test_scan_does_not_spin_when_polarity_is_wrong():
    """★★ 极性反了（σ 算错 ⇒ 正反馈）时，**绝不能整圈转下去** —— 角度预算必须拦住。"""
    sc = Scan()
    sim = _YawSim(yaw=0.0, sigma=-sc.sigma)     # ★ 故意反极性：船按相反方向响应
    trace, _t = _run(sc, sim, n_legs=1, dt_ms=100)
    import base.cfg.settings as _S2
    budget = 2.0 * sc.span + float(_S2.get("comm.motion.search_scan.runaway_slack_deg"))
    turned = abs(sc.sigma * sim.yaw)            # 实际转过的角度
    assert turned <= budget + 5.0, \
        "反极性下转了 %.0f°（预算 %.0f°）⇒ 防转圈失效，船会整圈转下去" % (turned, budget)
    # 越预算那几帧必须**当帧停手**（该段不再发舵）；之后进停顿/新段才算正常
    zeros = sum(1 for (_t2, c, _y) in trace if c == 0.0)
    assert zeros > 0, "越预算后从没停手过 ⇒ 会一路转圈"


def test_search_scan_cfg_keys_are_all_present():
    """★ 2026-10-07 用户定：**删掉代码兜底** ⇒ 这里只做"必填检查"：
    `motion.search_scan` 的每个键（含 `pid.*`）都必须在 cfg 里，缺则 `Scan()` 直接报名字。"""
    import base.cfg.settings as S
    cfg = S.get("comm.motion.search_scan", {}) or {}
    bad = [k for k in _K_SCAN if k != "pid" and cfg.get(k) is None]
    pid = cfg.get("pid", {}) or {}
    bad += ["pid.%s" % k for k in _K_SCAN_PID if pid.get(k) is None]
    assert not bad, "cfg 缺 motion.search_scan 的键（代码已无兜底）：\n  " + "\n  ".join(bad)
    Scan()      # 能构造出来 = 必填项齐全
