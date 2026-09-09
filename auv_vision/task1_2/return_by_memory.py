# -*- coding: utf-8 -*-
"""return_by_memory.py — 不依赖视觉的"按记忆返回出发点"

原理：撞球（或任意前进过程）运行时用环境变量记录实际发出的运动帧；
返回时把该段轨迹**反向回放**（时间对称、方向取反），即可大致回到起点。
（纯航位推算，无视觉/无磁罗盘；受水流/打滑影响会有偏差，可用 --extra-sec 补一点回程。）

使用流程：
  1) 记录（跑撞球/手动前推时执行，真实串口）：
     AUV_DOF_LOG=/tmp/path.csv python3 main.py --task ball
  2) 返回（撞完球后另开终端执行）：
     python3 return_by_memory.py --log /tmp/path.csv
可选：
  --sim           仅打印预览（不接串口）
  --dry-run       打印回放计划不执行
  --extra-sec 2   回放结束后再低速后退 2 秒（保险余量）
安全：执行前会要求回车确认；Ctrl-C 可随时中断（脚本会补发中位）。
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base.settings as S
from base.uart import UartController

AXIS_MID = S.comm.frame.axis_mid
AXIS_RNG = S.comm.frame.axis_range
BTN = list(S.comm.frame.btn_values)
HEADER = S.comm.frame.header


# ---------------------------------------------------------------------------
# 轨迹工具（纯函数，可测试）
# ---------------------------------------------------------------------------
def load_log(path):
    """读轨迹 csv → [(dt_s, a0,a1,a2,a3), ...]（aN=运动轴字节）。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("dt_s"):
                continue
            p = line.split(",")
            if len(p) != 5:
                continue
            dt = min(float(p[0]), 0.5)          # 防御异常大间隔
            axes = tuple(max(0, min(255, int(x))) for x in p[1:5])
            rows.append((dt, axes))
    return rows


def negate_axis(v):
    """运动轴取反（中位镜像）：128±d → 128∓d；钳制到 [0,255] 防 bytes() 越界。"""
    return max(0, min(255, AXIS_MID * 2 - v))


def negate_rows(rows):
    """反向回放序列：顺序颠倒 + 每帧运动轴镜像。"""
    return [(dt, tuple(negate_axis(v) for v in a)) for dt, a in reversed(rows)]


def estimate(rows):
    """累计估计：返回各通道等效时间积（用于打印，供人工判断）。"""
    sign = {}
    for name, ch in S.comm.dof_map.items():
        if isinstance(ch, dict) and "axis" in ch:
            sign[ch["axis"]] = ch.get("sign", 1)
    acc = {"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}
    names = list(acc)
    for dt, a in rows:
        for name, axis in zip(names, (1, 3, 2, 0)):   # surge,sway,heave,yaw 顺序
            acc[name] += (a[axis] - AXIS_MID) * sign.get(axis, 1) / AXIS_RNG * dt
    return acc


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------
def build_frame(axes4):
    body = list(axes4) + [S.comm.frame.aux_axis[i] for i in (4, 5, 6)]
    return bytes([HEADER] + body + BTN)


def replay(rows, uart, dry=False):
    n = len(rows)
    total_dt = sum(dt for dt, _ in rows)
    print("将反向回放 %d 帧（约 %.1f 秒）" % (n, total_dt))
    if dry:
        return
    t0 = time.time()
    for i, (dt, axes) in enumerate(rows):
        if (i + 1) % max(1, n // 10) == 0 or i == n - 1:
            print("[RET] %3d%%" % ((i + 1) * 100 // n))
        uart.send_frame_bytes(build_frame(axes), force=True)
        # 等待与录制等长（保证时间对称）；扣少量发送耗时，绝不小于 0
        time.sleep(max(0.0, dt - 0.005))
    print("[RET] 回放完成（%.1fs 实际）" % (time.time() - t0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, help="轨迹 csv（AUV_DOF_LOG 记录产物）")
    ap.add_argument("--sim", action="store_true", help="仅打印预览帧")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划不发送")
    ap.add_argument("--extra-sec", type=float, default=0.0,
                    help="回放结束后低速后退的保险时长(秒)")
    a = ap.parse_args()
    if not os.path.exists(a.log):
        print("轨迹文件不存在: %s" % a.log)
        return 2

    rows = load_log(a.log)
    if not rows:
        print("轨迹为空——请先带 AUV_DOF_LOG 跑一次前进/撞球")
        return 1
    back = negate_rows(rows)
    est = estimate(rows)
    print("记录 %d 帧（%.1fs） 累计≈ %s"
          % (len(rows), sum(dt for dt, _ in rows),
             "  ".join("%s%.1fs" % (k, v) for k, v in est.items())))
    print("准备反向运动返回出发点；周围安全后回车开始（Ctrl-C 中断自动停）")
    os.environ.pop("AUV_DOF_LOG", None)   # 回放本身不再续记
    if a.dry_run:
        replay(back, None, dry=True)
        return 0
    input("回车开始...")
    uart = UartController(sim=True if a.sim else None)
    if not a.sim and uart.sim:
        print("!! 串口未打开，无法发送——检查 comm.yaml serial")
        return 1
    try:
        replay(back, uart, dry=a.dry_run)
        if a.extra_sec > 0:
            print("[RET] 额外低速后退 %.1fs" % a.extra_sec)
            t0 = time.time()
            while time.time() - t0 < a.extra_sec:
                uart.send_dof(-0.4, 0.0, 0.0, 0.0)
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[RET] 手动中断")
    finally:
        uart.neutral()
        print("[RET] 已回中位停止")


if __name__ == "__main__":
    sys.exit(main())
