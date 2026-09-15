# -*- coding: utf-8 -*-
"""test_gate_kpt_memory.py — 门角点"逐点数据处理"层（gate/data/kpt_memory.py）

识别不动，只融合每个特征点自身的信息。锁住这些性质：
  1. 抑抖动（置信度加权 α-β 平均）
  2. 零滞后（匀速补偿，不拖慢过门判据）
  3. **软抗鬼点**：一帧鬼点对融合点几乎无影响，但该点仍有效（置信度不被打掉）
     —— 这是"不硬判无效、保持状态机稳定"的关键
  4. 半径来自该点历史分布（安静→小，漂移大→自动放大）
  5. 短消失回忆（conf ≥ conf_thr，不触发降级）
  6. 长丢失衰减（conf 自然掉到 conf_thr 以下，不瞎编）
  7. 世界真变了能自愈跟上（不会永久锁在旧位置）
  8. 参数可调/可关
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                       # noqa: E402
import base.settings as S                # noqa: E402
from gate.data.kpt_memory import KptMemory    # noqa: E402

PASS = []
BASE = np.array([[200., 150.], [500., 150.], [500., 400.], [200., 400.]])


def check(name, cond, extra=""):
    if not cond:
        raise AssertionError("FAIL: %s %s" % (name, extra))
    PASS.append(name)
    print("PASS: %s %s" % (name, extra))


def run(mem, gen, n=140, dt=100):
    mem.reset()
    pts, cfs, rad = [], [], []
    for i in range(n):
        k, c = gen(i)
        o, oc = mem.update(k, c, 1000 + i * dt)
        pts.append(o[0, 0])
        cfs.append(oc[0])
        rad.append(mem.radius[0])
    return np.array(pts), np.array(cfs), np.array(rad)


def _mk(**kw):
    cfg = dict(alpha=0.4, beta=0.1, k_sigma=3.0, r_min_px=10, r_max_px=80,
               sigma_init_px=8.0, sigma_lambda=0.15, recall_conf=0.55,
               conf_decay=0.35, max_missing_frames=5, min_frames=2)
    cfg.update(kw)
    return KptMemory(**cfg)


def test_jitter_and_lag():
    print("[1] 抑抖动 + 零滞后")
    rng = np.random.default_rng(0)
    m = _mk()
    o, _, _ = run(m, lambda i: (BASE + rng.normal(0, 6, BASE.shape),
                               np.full(4, 0.9)))
    sig_in = 6.0
    sig_out = float(np.std(o[20:]))
    check("抖动被压小", sig_out < sig_in * 0.7,
          "输入σ≈%.2f → 输出σ=%.2f" % (sig_in, sig_out))
    m2 = _mk()
    o2, _, _ = run(m2, lambda i: (BASE + np.array([i * 0.6, 0.0]),
                                  np.full(4, 0.9)))
    lag = float(np.mean(o2[40:] - (BASE[0, 0] + np.arange(len(o2))[40:] * 0.6)))
    check("匀速下几乎无滞后", abs(lag) < 2.0, "稳态滞后=%.2f px" % lag)
    print()


def test_ghost_soft():
    print("[2] 软抗鬼点：几乎不影响融合，但该点不被判无效")
    m = _mk()
    rng = np.random.default_rng(1)
    # 前 30 帧正常 + 抖动
    for i in range(30):
        m.update(BASE + rng.normal(0, 3, BASE.shape), np.full(4, 0.9), 1000 + i * 100)
    ref = m.update(BASE, np.full(4, 0.9), 4000)[0][0, 0]
    # 一帧鬼点：角0 跳到 300px 外
    ghost = BASE.copy()
    ghost[0] = ghost[0] + np.array([300.0, 0.0])
    o, c = m.update(ghost, np.full(4, 0.95), 4100)
    shift = abs(float(o[0, 0]) - ref)
    check("鬼点对融合点影响很小", shift < 2.0, "位移=%.2f px" % shift)
    check("鬼点帧该点仍有效(conf≥阈值)", float(c[0]) >= 0.5,
          "conf=%.3f" % float(c[0]))
    check("权重诊断：鬼点权重≈0", m.weights[0] < 0.05,
          "w=%.4f d=%.1fpx R=%.1fpx" % (m.weights[0], m.residuals[0],
                                        m.radius[0]))
    # 恢复的下一帧：正常值立刻被采纳
    o2, c2 = m.update(BASE, np.full(4, 0.9), 4200)
    check("下一帧恢复正常", abs(float(o2[0, 0]) - ref) < 6.0,
          "偏差=%.2f px" % abs(float(o2[0, 0]) - ref))
    print()


def test_radius_from_history():
    print("[3] 半径来自该点历史分布（安静小 / 漂移大自动放大）")
    rng_c = np.random.default_rng(2)
    rng_w = np.random.default_rng(3)
    calm = _mk()
    run(calm, lambda i: (BASE + rng_c.normal(0, 1, BASE.shape),
                         np.full(4, 0.9)))
    r_calm = float(calm.radius[0])
    wild = _mk()
    run(wild, lambda i: (BASE + rng_w.normal(0, 25, BASE.shape),
                         np.full(4, 0.9)))
    r_wild = float(wild.radius[0])
    check("漂移大 → 半径自动放大", r_wild > r_calm * 1.5,
          "安静 R=%.1f px vs 漂移 R=%.1f px" % (r_calm, r_wild))
    check("半径被限幅在 [r_min,r_max]",
          S.vision.gate.kpt_mem.r_min_px - 1e-6 <= min(r_calm, r_wild)
          and max(r_calm, r_wild) <= S.vision.gate.kpt_mem.r_max_px + 1e-6,
          (r_calm, r_wild))
    print()


def test_recall_and_decay():
    print("[4] 短消失回忆 / 长丢失衰减")
    m = _mk()
    for i in range(10):
        m.update(BASE, np.full(4, 0.9), 1000 + i * 100)
    lost = np.zeros(4)
    c_short = None
    for i in range(3):                       # 短消失 3 帧
        _, c = m.update(BASE, lost, 2000 + i * 100)
        c_short = float(c[0])
    check("短消失仍给可用置信度", c_short >= 0.5, "conf=%.3f" % c_short)
    for i in range(20):                      # 长时间丢失
        _, c = m.update(BASE, lost, 2300 + i * 100)
    check("长丢失置信度衰减到阈值以下", float(c[0]) < 0.5,
          "conf=%.3f" % float(c[0]))
    print()


def test_self_heal():
    print("[5] 世界真变了能自愈跟上（不会永久锁死）")
    m = _mk()
    for i in range(30):
        m.update(BASE, np.full(4, 0.9), 1000 + i * 100)
    moved = BASE + np.array([120.0, 0.0])     # 角点整体真的平移了 120px
    o = None
    for i in range(60):
        o, _ = m.update(moved, np.full(4, 0.9), 4000 + i * 100)
    err = abs(float(o[0, 0]) - moved[0, 0])
    check("持续位移后能跟上", err < 10.0, "最终偏差=%.2f px" % err)
    print()


def test_knobs():
    print("[6] 参数可调/可关")
    off = _mk(k_sigma=0.0, r_max_px=0.0)      # 关掉几何权重 → 纯 α-β
    for i in range(30):
        off.update(BASE, np.full(4, 0.9), 1000 + i * 100)
    ref = off.update(BASE, np.full(4, 0.9), 4000)[0][0, 0]
    ghost = BASE.copy()
    ghost[0] = ghost[0] + np.array([300.0, 0.0])
    o, _ = off.update(ghost, np.full(4, 0.95), 4100)
    shift_off = abs(float(o[0, 0]) - ref)
    on = _mk()
    for i in range(30):
        on.update(BASE, np.full(4, 0.9), 1000 + i * 100)
    ref2 = on.update(BASE, np.full(4, 0.9), 4000)[0][0, 0]
    o2, _ = on.update(ghost, np.full(4, 0.95), 4100)
    shift_on = abs(float(o2[0, 0]) - ref2)
    check("关掉后鬼点影响明显更大(旋钮有效)", shift_off > shift_on * 3,
          "关=%.1fpx 开=%.1fpx" % (shift_off, shift_on))
    check("alpha 可调(越大越快)", _mk(alpha=0.8).alpha == 0.8)
    print()


def main():
    test_jitter_and_lag()
    test_ghost_soft()
    test_radius_from_history()
    test_recall_and_decay()
    test_self_heal()
    test_knobs()
    print("\n全部通过 %d 项" % len(PASS))


if __name__ == "__main__":
    main()
