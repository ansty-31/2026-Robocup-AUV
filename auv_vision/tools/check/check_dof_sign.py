# -*- coding: utf-8 -*-
"""tools/check/check_dof_sign.py — **DOF 通道物理符号自检**（板端真实串口 + 相机）

为什么需要它（2026-09-23 实战结论）：
  转向探向（`common/turn_deg.py` PROBE）只能测到「命令符号→物理转向」与
  「物理转向→遥测符号」之**积** `g·s`，且它**默认自己给的 DOF 方向就是物理方向** ⇒
  命令侧反了（g=-1）它发现不了，闭环照样"收敛"（收敛在遥测量上），船却朝反方向转。
  现场现象：命令"左转 13.7°"之后机身反而与门面不平行。sway 更糟：连探向都没有。

  本脚本用**视觉**当独立真值（相机是唯一不受 DOF/遥测符号影响的量）：
    命令 +yaw  ⇒ 若门在画面里**左移**(dx↓) 且 psi↓ ⇒ 符号正确；反之 ⇒ dof_map.yaw.sign: -1
    命令 +sway ⇒ 若门在画面里**左移**(dx↓)        ⇒ 符号正确；反之 ⇒ dof_map.sway.sign: -1
  同时打印遥测 yaw 的变化 ⇒ 顺带定出遥测侧符号 s（g·s 的那个 s）。

用法（板端、水里、真实串口；**必须显式 --go 才会发推力**）：

    AUV_SIM_MODE=0 python3 tools/check/check_dof_sign.py --dof sway --value 0.30 --hold 2.5
    AUV_SIM_MODE=0 python3 tools/check/check_dof_sign.py --dof yaw  --value 0.40 --hold 2.0
    AUV_SIM_MODE=0 python3 tools/check/check_dof_sign.py --dof sway --value 0.30 --dry   # 只看基线，不发推力

输出：每相位的 dx/psi/遥测 yaw 均值差 → 结论 + 该改哪一行 cfg。
安全：任何退出路径都会补发中性帧（try/finally + 信号处理）；hold 期间 50Hz 心跳。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np                                                    # noqa: E402

import base.settings as S                                             # noqa: E402
from base.camera import create_camera                                 # noqa: E402
from base.uart import UartController                                  # noqa: E402
from gate.gate_detector import board_camera, build_gate_backend       # noqa: E402
from gate.geometry import (gate_normal_angles_deg, gate_pose,         # noqa: E402
                           object_points)

KPT_MIN = 4          # 只有 4 角(full)才允许用来判方向（p3p 的 psi std 64~69° = 噪声）


class Rig(object):
    """串口 + 相机 + 门检测（一帧一读，不做任务状态机）。"""

    def __init__(self):
        self.uart = UartController()
        self.cam = create_camera("front")
        self.backend = build_gate_backend()
        if self.backend is None:
            raise SystemExit("gate 权重缺失 → 无法用视觉判方向（检查 vision.model.task_models.gate）")
        V = S.get("vision.gate", None) or {}
        geo = V.get("geometry", None) or {}
        self.obj3 = object_points(float(geo.get("frame_w", 0.77)),
                                  float(geo.get("frame_h", 0.56)))
        self.camera = board_camera()
        for _ in range(5):                       # 预热（自动曝光/白平衡收敛）
            f = self.cam.read()
            if f is None:
                raise SystemExit("相机无帧 → 检查 /dev/video*")
            time.sleep(0.05)
        self.w = None

    def measure(self):
        """返回 (dx, psi, kpt) 或 (None, None, kpt)。dx=门心像素归一化偏差(+ = 门在画面右)。"""
        frame = self.cam.read()
        if frame is None:
            return None, None, 0
        dets = self.backend.detect(frame)
        if not dets:
            return None, None, 0
        d = max(dets, key=lambda x: float(getattr(x, "score", 0.0)))
        kpts, kconf = d.kpts, d.kpt_conf
        if kpts is None or kconf is None:
            return None, None, 0
        n = int((np.asarray(kconf) >= 0.7).sum())
        h, w = frame.shape[:2]
        self.w = w
        if n < KPT_MIN:
            return None, None, n
        res = gate_pose(self.camera, self.obj3, np.asarray(kpts))
        if res is None:
            return None, None, n
        rvec, tvec = res
        c = self.camera.project(np.zeros((1, 3), np.float32), rvec, tvec)[0]
        dx = float((c[0] - w / 2.0) / (w / 2.0))
        psi, _ = gate_normal_angles_deg(rvec, tvec)
        return dx, float(psi), n

    def hold(self, dof, value, seconds, samples):
        """以 50Hz 发 dof=value，同时采样 (dx, psi, tyaw)。"""
        t_end = time.time() + seconds
        while time.time() < t_end:
            args = {"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}
            args[dof] = value
            self.uart.send_dof(args["surge"], args["sway"], args["heave"], args["yaw"])
            dx, psi, n = self.measure()
            tel = getattr(self.uart, "telemetry", None)
            ty = None if tel is None else tel.yaw_deg
            samples.append((dx, psi, ty, n))
            time.sleep(0.02)

    def neutral(self, seconds=1.5, samples=None):
        self.hold("surge", 0.0, seconds, samples if samples is not None else [])


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def phase_stats(samples):
    return (_mean([s[0] for s in samples]), _mean([s[1] for s in samples]),
            _mean([s[2] for s in samples]), len(samples),
            sum(1 for s in samples if s[0] is not None))


def main():
    ap = argparse.ArgumentParser(description="DOF 通道物理符号自检（相机当独立真值）")
    ap.add_argument("--dof", choices=["yaw", "sway"], default="sway")
    ap.add_argument("--value", type=float, default=0.30, help="测试幅度（归一化 DOF，>死区 0.138）")
    ap.add_argument("--hold", type=float, default=2.5, help="每个方向的保持时长(s)")
    ap.add_argument("--gap", type=float, default=1.5, help="方向之间的回中时长(s)")
    ap.add_argument("--dry", action="store_true", help="只测基线，不发任何推力")
    ap.add_argument("--go", action="store_true", help="确认下水/推进器可动，才会发推力")
    a = ap.parse_args()

    if S.SIM_MODE and not a.dry:
        print("⚠️ SIM_MODE=True（只打印不发串口）⇒ 测不到真实动作。用 AUV_SIM_MODE=0 重跑。")
    print("配置：dof_map.%s = %s ｜ telemetry.yaw_sign = %s ｜ SIM_MODE=%s"
          % (a.dof, dict(S.get("comm.dof_map.%s" % a.dof, {})),
             S.get("comm.telemetry.yaw_sign", 1.0), S.SIM_MODE))
    if not a.go and not a.dry:
        raise SystemExit("拒绝发推力：加 --go（并确认船在水里/有人看着）")

    rig = Rig()
    dof = a.dof
    try:
        base = []
        print("\n[相位 A] 回中 %.1fs（基线）" % max(a.gap, 1.5))
        rig.neutral(max(a.gap, 1.5), base)
        b_dx, b_psi, b_ty, n_tot, n_ok = phase_stats(base)
        print("  基线： dx=%s  psi=%s  tyaw=%s   (%d/%d 帧有 4 角)"
              % (_f(b_dx), _f(b_psi), _f(b_ty), n_ok, n_tot))

        if a.dry:
            print("\n--dry：只测基线，未发推力。基线可视 = 视觉链路通。")
            return 0

        pos = []
        print("\n[相位 B] %s = %+.2f × %.1fs" % (dof, a.value, a.hold))
        rig.hold(dof, a.value, a.hold, pos)
        p_dx, p_psi, p_ty, _, p_ok = phase_stats(pos)

        mid = []
        print("[相位 C] 回中 %.1fs" % a.gap)
        rig.neutral(a.gap, mid)
        m_dx, m_psi, m_ty, _, m_ok = phase_stats(mid)

        neg = []
        print("[相位 D] %s = %+.2f × %.1fs" % (dof, -a.value, a.hold))
        rig.hold(dof, -a.value, a.hold, neg)
        n_dx, n_psi, n_ty, _, n_ok2 = phase_stats(neg)
    finally:
        for _ in range(25):                       # 1s 中性收尾（下位机无帧超时停车）
            rig.uart.send_dof(0.0, 0.0, 0.0, 0.0)
            time.sleep(0.04)
        print("\n[安全] 已连续发送中性帧（~1s）")

    print("\n===== 结果 =====")
    print("  基线   : dx=%s psi=%s tyaw=%s" % (_f(b_dx), _f(b_psi), _f(b_ty)))
    print("  %s=%+.2f: dx=%s psi=%s tyaw=%s" % (dof, a.value, _f(p_dx), _f(p_psi), _f(p_ty)))
    print("  回中   : dx=%s psi=%s tyaw=%s" % (_f(m_dx), _f(m_psi), _f(m_ty)))
    print("  %s=%+.2f: dx=%s psi=%s tyaw=%s" % (dof, -a.value, _f(n_dx), _f(n_psi), _f(n_ty)))

    d_dx = _diff(p_dx, b_dx)
    d_psi = _diff(p_psi, b_psi)
    d_ty = _diff(p_ty, b_ty)
    print("\n  相对基线： Δdx=%s  Δpsi=%s  Δtyaw=%s" % (_f(d_dx), _f(d_psi), _f(d_ty)))

    print("\n===== 判读 =====")
    if dof == "sway":
        if d_dx is None:
            print("  判不出：测试窗口里没有 4 角（门没在画面里/太远/太近）⇒ 先把门摆进画面再跑")
        elif d_dx < -0.05:
            print("  ✅ +sway 让门在画面里**左移** ⇒ 船向右平移 ⇒ dof_map.sway.sign 正确")
        elif d_dx > 0.05:
            print("  ❌ +sway 让门在画面里**右移** ⇒ 船实际向左平移（=你说的'要右移却左移'）")
            print("     ⇒ 改 cfg/comm.yaml： dof_map: {sway: {axis: 3, sign: -1}}")
        else:
            print("  ⚠️ Δdx 太小（%s）：推力不够/时间太短/船没动 ⇒ 加大 --value/--hold 重测" % _f(d_dx))
    else:
        if d_dx is None or d_psi is None:
            print("  判不出：需要门在画面里且有 4 角 ⇒ 先把门摆进画面再跑")
        else:
            ok_dx = d_dx < -0.05
            ok_psi = d_psi < -0.5
            if ok_dx and ok_psi:
                print("  ✅ +yaw 让门**左移**且 psi 变小 ⇒ 命令侧符号正确（+yaw = 真右转）")
            elif (not ok_dx) and (not ok_psi):
                print("  ❌ +yaw 让门**右移**且 psi 变大 ⇒ 命令侧反了（=你看到的'左转后更歪'）")
                print("     ⇒ 改 cfg/comm.yaml： dof_map: {yaw: {axis: 0, sign: -1}}")
            else:
                print("  ⚠️ dx 与 psi 结论不一致（dx%s / psi%s）⇒ 潮/漂移干扰，重测或加大幅度"
                      % (_f(d_dx), _f(d_psi)))
            if d_ty is not None and abs(d_ty) > 0.5:
                s = 1.0 if (d_ty > 0) == (d_dx < 0) else -1.0
                print("  遥测侧：Δtyaw=%s ⇒ 与视觉转向%s ⇒ 遥测符号 s ≈ %+d"
                      % (_f(d_ty), "同向" if s > 0 else "**反向**", s))
                if s < 0:
                    print("     （探向 `img_sign` 会自动补偿这个反向；命令侧才是要害）")
    return 0


def _f(v):
    return "None" if v is None else "%+.3f" % v


def _diff(a, b):
    return None if (a is None or b is None) else a - b


if __name__ == "__main__":
    sys.exit(main())
