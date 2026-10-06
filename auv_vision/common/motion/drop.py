# -*- coding: utf-8 -*-
"""common/motion/drop.py — **横倾放球/倒球**序列（非阻塞状态机 + 可单独运行的脚本）。

两个用途，**同一套代码、不同 cfg 步骤表**：
* `dump`    —— 任务三：笼里是"错球" ⇒ 右移 → 横倾倒出 → 回正 → 左移 → 后退 → 停稳，再重来；
* `release` —— 任务四：到点放球 ⇒ 横倾 30° → 停动力 3s（用户 2026-10-06 定的放球口径）。

为什么单独成件：任务三和任务四都要"横倾把球弄出去"，这只是**动作**，别各写一份。
两个任务都只是调用方：任务三 `BallDropSequence("dump")`、任务四 `BallDropSequence("release")`；
每帧 `state, dof = seq.step(now_ms, uart)` 并下发 `dof`；
或直接把本文件当脚本跑（台架/水池单测这个动作）：
```bash
python3 common/motion/drop.py --what dump      # 或 --what release
```

两条纪律：
* **下发点唯一**：本类只返回 DOF，不自己 `send_dof`（同帧两次下发会把推力抹掉，踩过）。
* **横倾要收紧限深**：横倾 30° 会让机身一侧抬高（没有 heave 指令也会靠近水面），
  所以序列运行期间把**任务级有效限深下限**抬到 `comm.grab.drop.depth_floor_m`；
  结束**恢复成运行前的值**（不是恢复到 0 —— 任务三自己已经把下限抬到 1.0m 了，
  恢复成 0 等于把任务的收紧给撤了，那是"谁最后动谁说了算"的经典坑）。
"""
from __future__ import annotations

import sys

import base.cfg.settings as S

from common.motion.axis import (ZERO_DOF, AxisMove, TimedDof, axis_deg,
                               level_relative_deg, roll_dump_deg, task_node)
from common.cfg.cfgnode import sub, req, req_flag, req_node, MissingCfg

# ★ 2026-10-07 用户定：**删掉兜底步骤表** —— `comm.motion.drop.<name>.steps` 必须写全，
#   缺则报名字（`_steps_of()`）。改动作序列只需动 cfg 一处。
# `comm.grab.drop` 段的兜底
_K_DROP = (
    "depth_floor_m",
    "restore_floor"
)
# 步骤里允许的 DOF 键（拼错就当 0，别静默当成功）
_DOF_KEYS = ("surge", "sway", "heave", "yaw")


def _pick(paths, default=None):
    """按顺序取第一个**存在**的 cfg 值（"任务覆盖 → 共用默认"）。"""
    from base.cfg.settings import get as _get
    for p in paths:
        v = _get(p, None)
        if v is not None:
            return v
    return default


def _steps_of(which, prefix="grab"):
    """步骤表解析顺序（前者覆盖后者）：

    1. `comm.<prefix>.drop.<which>.steps` —— **任务覆盖**（只在"确实要单独一套"时写）
    2. `comm.motion.drop.<which>.steps`   —— **共用默认**（任务三/四共享这一份）
    3. 代码兜底表 `_D_STEPS`
    """
    steps = _pick(["comm.%s.drop.%s.steps" % (prefix, which),
                   "comm.motion.drop.%s.steps" % which])
    if not isinstance(steps, list) or not steps:
        raise MissingCfg("cfg 缺 comm.motion.drop.steps.%s（代码已无兜底）" % which)
    return [dict(s) for s in steps if isinstance(s, dict)]


class BallDropSequence(object):
    """横倾放球/倒球序列（非阻塞）。

    `step(now_ms, uart) -> (state, dof)`；`state ∈ idle|run|done|failed`，
    `dof` 是**本帧要下发**的四元组（唯一下发点在调用方）。
    """

    IDLE, RUN, DONE, FAILED = "idle", "run", "done", "failed"

    def __init__(self, which="dump", log=None, prefix="grab"):
        self.which = str(which)
        self.prefix = prefix
        self.log = log or (lambda *a: None)
        self.steps = _steps_of(self.which, prefix)
        d = sub(task_node(prefix), "drop")                    # 任务覆盖
        g = S.get("comm.motion.drop", None) or {}             # 共用默认
        floor = d.get("depth_floor_m")
        if floor is None:
            floor = req(g, "depth_floor_m")
        self.depth_floor_m = float(floor)
        self.restore_floor = bool(d.get("restore_floor",
                                       req_flag(g, "restore_floor")))
        self.state = self.IDLE
        self.why = ""
        self.i = 0
        self._timed = None
        self._axis = None
        self._roll_ref = None
        self._floor_prev = 0.0
        self._floor_done = False

    # ---------------------------------------------------------------- 内部
    def _floor_on(self, uart):
        setter = getattr(uart, "set_extra_min_depth", None)
        if not callable(setter) or self.depth_floor_m <= 0.0 or self._floor_done:
            return
        self._floor_prev = float(getattr(uart, "extra_min_depth_m", 0.0) or 0.0)
        setter(max(self._floor_prev, self.depth_floor_m))     # 只抬不降
        self._floor_done = True

    def _floor_off(self, uart):
        setter = getattr(uart, "set_extra_min_depth", None)
        if callable(setter) and self._floor_done and self.restore_floor:
            setter(self._floor_prev)                          # 恢复运行前的值（不是 0）
        self._floor_done = False

    def _open_step(self, step, uart):
        """开始一步：时间步 → `TimedDof`；`tilt`/`level` → 轴动作。

        `tilt` 时**先记下横倾前的 roll**，它就是后面 `level` 的回正基准（按遥测算，不写死）。
        """
        self._timed = self._axis = None
        if step.get("tilt") or step.get("level"):
            if step.get("tilt"):
                self._roll_ref = axis_deg(uart, "roll")
                self._axis = AxisMove("roll", roll_dump_deg(prefix=self.prefix),
                                      name="roll", log=self.log, prefix=self.prefix)
            else:
                rel = level_relative_deg(self._roll_ref, axis_deg(uart, "roll"))
                if rel is None:
                    self.why = "no_roll_telemetry"
                    self.log("[DROP] ⚠️ 拿不到 roll 遥测 ⇒ 回不了正，停手（横着开船不安全）")
                    self.state = self.FAILED
                    return
                self._axis = AxisMove("roll", rel, name="roll", log=self.log,
                                      prefix=self.prefix)
            return
        dof = dict((k, step.get(k, 0.0)) for k in _DOF_KEYS)
        self._timed = TimedDof(dur_s=float(step.get("s", 0.0)),
                               name=str(step.get("name", "step%d" % self.i)),
                               log=self.log, **dof)

    # ---------------------------------------------------------------- 主入口
    def step(self, now_ms, uart):
        if self.state in (self.DONE, self.FAILED):
            return self.state, ZERO_DOF
        if self.state == self.IDLE:
            self._floor_on(uart)
            self.log("[DROP] 开始「%s」序列：%d 步" % (self.which, len(self.steps)))
            self.state = self.RUN
        if self.i >= len(self.steps):
            self.state = self.DONE
            self._floor_off(uart)
            self.log("[DROP] 「%s」序列结束" % self.which)
            return self.state, ZERO_DOF
        if self._timed is None and self._axis is None:
            self._open_step(self.steps[self.i], uart)
            if self.state == self.FAILED:
                self._floor_off(uart)
                return self.state, ZERO_DOF
        if self._axis is not None:                    # 轴动作：自己走 request_turn，本帧不发 DOF
            st = self._axis.step(now_ms, uart)
            if st == AxisMove.DONE:
                self.i += 1
                self._axis = None
            elif st == AxisMove.FAILED:
                self.state, self.why = self.FAILED, self._axis.why or "axis_failed"
                self._floor_off(uart)
                self.log("[DROP] ⚠️ 步骤 %d（%s）失败：%s"
                         % (self.i, self.steps[self.i].get("name"), self.why))
            return self.state, ZERO_DOF
        st = self._timed.step(now_ms)
        if st == TimedDof.DONE:
            self.i += 1
            self._timed = None
            return self.state, ZERO_DOF
        return self.state, self._timed.dof

    @property
    def finished(self):
        return self.state in (self.DONE, self.FAILED)


# --------------------------------------------------------------------------- #
# 单独运行（台架/水池里只测这个动作；不发别的指令）
# --------------------------------------------------------------------------- #
def main():
    import argparse
    import time

    ap = argparse.ArgumentParser(description="横倾放球/倒球动作（下位机执行；收尾硬停）")
    ap.add_argument("--prefix", default="grab", help="任务段名：grab=任务三 / place=任务四")
    ap.add_argument("--what", choices=("dump", "release"), default="dump",
                    help="dump=任务三倒掉错球 / release=任务四放球")
    ap.add_argument("--timeout", type=float, default=30.0, help="整个序列的超时秒数")
    args = ap.parse_args()

    from base.hw.uart import UartController
    u = UartController()
    if u.sim:
        print("!! 串口处于 SIM（只打印）：不会真正驱动电机")
    seq = BallDropSequence(args.what, log=print, prefix=args.prefix)
    t0 = time.time()
    try:
        while not seq.finished:
            st, dof = seq.step(int(time.time() * 1000), u)
            u.send_dof(*dof)
            if time.time() - t0 > args.timeout:
                print("[DROP] ⚠️ 超时 %.0fs → 中止" % args.timeout)
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[DROP] Ctrl-C → 硬停")
    finally:
        stop_hard = getattr(u, "stop_hard", None)
        if callable(stop_hard):
            stop_hard(verify=True)
        else:
            u.neutral()
        u.close()
    print("[DROP] 结束状态：%s%s" % (seq.state, ("（%s）" % seq.why) if seq.why else ""))
    return 0 if seq.state == BallDropSequence.DONE else 3


if __name__ == "__main__":
    sys.exit(main())
