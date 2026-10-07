# -*- coding: utf-8 -*-
"""common/motion/axis.py — **指定角度轴动作 / 下潜到位 / 定时推力**（非阻塞原语）（全部非阻塞：每帧 `step()` 一次）。

三个原语，任务三专用，不动 ball/gate：

* `AxisMove` —— 一次**指定角度轴**动作（抬头 / 回水平 / 倒球横倾 / 放球横倾）。内部就是
  `common.motion.turn_deg.TurnCore`（同一条"下位机执行 + 编号重发 + 等完成标志"的链），额外记下
  动作前的**遥测角**当基准 ⇒ 回水平按 `wrap180(ref − now)` 算相对角，不写死"−30°"（中途被扰动也能回正）。
* `EnsureDepth` —— 下潜到 `>= 有效限深下限` 的非阻塞状态机：抬头的**前置条件**，到不了就 `failed`，
  调用方据此**拒绝抬头**（"保证不出水面"只有这一条能兑现）。
* `TimedDof` —— 按固定 DOF 持续 T 毫秒（下压/上升/平移/停手/放球都用它）。

**限深下限的生命周期归任务独占**（`GrabTask` 开头 `set_extra_min_depth`、结束撤回）：
原语只读 `uart.effective_min_depth_m`，不各自改它 —— 避免"谁最后动谁说了算"。

极性：`pitch.up_sign` 已**现场实测 = -1.0**（负相对角 = 抬头）；`roll.dump_sign` **仍未验证**。
两者都只是"相对角的正负"（协议 §2.2 明说正值
不保证等于物理抬头/左倾 ⇒ 上板第一件事是单轴小幅实测：
`python3 common/motion/turn_deg.py --axis pitch --deg 5`，看清机头往哪边动再填 cfg。
**本模块未上板、未下水。**
"""
from __future__ import annotations

import base.cfg.settings as S
from common.cfg.cfgnode import merge, num, sub, req, req_flag, req_node, MissingCfg
from common.motion.turn_deg import (TurnCore, clamp_tilt_deg, max_tilt_deg,
                                    wrap180)

# 兜底表（取值 = 当前 cfg/comm.yaml 的 grab 段；删掉任一键都用这里的同值）
_K_GRAB = (
    "timeout_ms",
    "depth_floor_m",
    "ensure_depth",
    "ensure_depth_s",
    "descend_heave",
    "depth_tol_m"
)
_K_MOVE = (
    "timeout_s",
    "settle_ms"
)
# 推力动作各一张完整默认表：**调用方读到的每个键都必须在这里**，否则删 cfg 键就 KeyError
#（这正是本模块单测抓到过的错：`thrust_cfg` 原来只补一部分键，读 `dump.hold_s` 直接崩）
_K_DIP = (
    "heave",
    "dur_s"
)
_K_RISE = (
    "heave",
    "dur_s"
)
_K_DUMP = (
    "sway",
    "dur_s",
    "hold_s",
    "back_dur_s",
    "back_surge",
    "settle_s"
)


ZERO_DOF = (0.0, 0.0, 0.0, 0.0)

# 姿态超限的判定余量（度）：指令侧已在 ±max_tilt 夹紧，这里给一点余量再判实测超标
_TILT_MARGIN_DEG = 2.0


def task_node(prefix="grab"):
    """任务段节点 `comm.<prefix>`：**必填**（★ 2026-10-07 删掉兜底 ⇒ 缺段报名字）。

    `prefix` 是**任务段名**：任务三 `"grab"`、任务四 `"place"` —— 原语不绑任务，才能被两个
    任务共用（跨任务代码必须放 `common/`，见 README 依赖规则）。
    """
    n = S.get("comm.%s" % prefix, None)
    if not isinstance(n, dict) or not n:
        raise MissingCfg("cfg 缺 comm.%s（代码已无兜底）" % prefix)
    return n


def grab_node():                 # 兼容旧名（= task_node("grab")）
    return task_node("grab")


def grab_cfg(prefix="grab"):
    """任务级参数 `comm.<prefix>`：**必填**（缺段 ⇒ `MissingCfg`）。"""
    return task_node(prefix)


def move_cfg(name, prefix="grab"):
    """某个轴动作的参数：`comm.<prefix>.<name>` 覆盖 `_K_MOVE`。"""
    return req_node(task_node(prefix), name)


def action_cfg(name, prefix="grab"):
    """某个推力动作的参数：**直接取自 `comm.<prefix>.<name>`，必填**（缺则报名字）。

    ★ 2026-10-07 用户定：**删掉兜底默认表** —— `comm.<prefix>.<name>` 必须写全。
    """
    return req_node(task_node(prefix), name)


def axis_deg(uart, axis):
    """本帧遥测里该轴的角度；拿不到 → None（回水平/回正就算不出相对角 ⇒ 调用方必须停手）。"""
    tel = getattr(uart, "telemetry", None)
    if tel is None:
        return None
    key = {"pitch": "pitch_deg", "roll": "roll_deg", "yaw": "yaw_deg"}.get(str(axis))
    if key is None:
        return None
    v = getattr(tel, key, None)
    return None if v is None else float(v)


def telemetry_axis_deg(uart, axis):
    """本帧遥测里该轴的角度（模块级，供不需要实例的调用方用）；拿不到 → None。"""
    return axis_deg(uart, axis)


class AxisMove(object):
    """一次指定角度轴动作（非阻塞）。

    用法：`m = AxisMove("pitch", pitch_up_deg()); st = m.step(now_ms, uart)`。
    `st` ∈ `"idle" | "run" | "done" | "failed"`；只有 `done/failed` 是终态。
    `m.ref_deg` = 动作前的遥测角（回水平的基准，由调用方取走）。
    """

    IDLE, RUN, DONE, FAILED = "idle", "run", "done", "failed"

    def __init__(self, axis, relative_deg, name=None, log=None, prefix="grab"):
        self.axis = axis
        self.prefix = prefix
        # ★ pitch/roll 安全夹紧（±max_tilt，默认 50°）：与 `TurnCore` 同一判据，这里先夹一遍
        #   是为了日志里显示的是**真正会下发**的角度。yaw 不受限。
        self.deg, self.clamped = clamp_tilt_deg(self.axis, float(relative_deg), prefix, log)
        self.max_tilt = max_tilt_deg(prefix)
        self.over_cnt = 0
        self.name = name or str(axis)
        self.log = log or (lambda *a: None)
        c = move_cfg(self.name, prefix)
        self.timeout_s = float(c["timeout_s"])
        self.settle_ms = float(c["settle_ms"])
        self.state = self.IDLE
        self.why = ""
        self.ref_deg = None            # 动作前的遥测角（回水平算相对角用）
        self._core = None

    def _check_tilt(self, uart):
        """★ 姿态侧兜底：pitch/roll **实测**超限（含余量）就撤指令并判失败，让调用方安全收尾。

        为什么两处都判：`TurnCore` 夹的是**指令幅度**（静态），这一层看的是**下位机真的把船转成
        什么样了**（动态）。下位机若越界/失控，指令侧夹紧是拦不住的 —— 而「pitch/roll 不准超过
        50°」这条保护必须以**实测姿态**为准。超限即 `cancel_turn`（撤掉目标，下位机锁当前姿态），
        再由调用方走各自的 bail（停手/放平）；看门狗是更外面那层（停船 + 强制退出）。
        """
        if str(self.axis) not in ("pitch", "roll") or self.max_tilt <= 0.0:
            return False
        v = telemetry_axis_deg(uart, self.axis)
        if v is None:
            return False
        if abs(float(v)) <= self.max_tilt + _TILT_MARGIN_DEG:
            return False
        self.over_cnt += 1
        self.state = self.FAILED
        self.why = "tilt_over_limit"
        try:
            self._core.abort("tilt_over_limit")
        except Exception:
            pass
        self.log("[GRAB] ⛔ %s 实测 %.1f° 超安全限 ±%.0f° ⇒ 撤指令、停手（%s）"
                 % (self.axis, float(v), self.max_tilt, self.name))
        return True

    def telemetry_axis_deg(self, uart):
        """本帧遥测的该轴角度；拿不到 → None（回水平就算不出相对角）。"""
        return axis_deg(uart, self.axis)

    @property
    def finished(self):
        return self.state in (self.DONE, self.FAILED)

    def step(self, now_ms, uart):
        if self.finished:
            return self.state
        if self.state == self.IDLE:
            self.ref_deg = self.telemetry_axis_deg(uart)
            self._core = TurnCore(deg=abs(self.deg), left=(self.deg < 0.0),
                                  timeout=self.timeout_s, log=self.log, axis=self.axis)
            self.state = self.RUN
            self.log("[GRAB] 轴动作 %s：相对角 %+.2f°（动作前遥测 %s=%s）"
                     % (self.name, self.deg, self.axis,
                        "n/a" if self.ref_deg is None else "%.2f" % self.ref_deg))
        if self._check_tilt(uart):
            return self.state
        st, _ = self._core.step(now_ms, uart)
        if st == TurnCore.DONE:
            self.state = self.DONE
            self.log("[GRAB] 轴动作 %s 完成（下位机回报）" % self.name)
        elif self._core.finished():
            self.state = self.FAILED
            self.why = self._core.why or "turn_aborted"
            self.log("[GRAB] ⚠️ 轴动作 %s 失败：%s" % (self.name, self.why))
        return self.state


def level_relative_deg(ref_deg, now_deg):
    """回水平/回零要发的相对角 = `wrap180(ref − now)`（协议 §2.2：目标 = 当时测量 + 相对角）。

    任一侧拿不到（没遥测）→ None ⇒ 调用方**不许**瞎发"−30°"，应退回保守处理。
    """
    if ref_deg is None or now_deg is None:
        return None
    return wrap180(float(ref_deg) - float(now_deg))


class EnsureDepth(object):
    """下潜到 `>= 有效限深下限` 的非阻塞状态机（抬头前的前置条件）。

    下限取 `uart.effective_min_depth_m`（任务级已抬高的那条），拿不到就退回 `comm.grab.depth_floor_m`。
    到位 = "深到抬头也不会露出水面"。`failed` 的三种由因：无深度遥测 / 超时 / 不允许下潜。
    """

    IDLE, RUN, DONE, FAILED = "idle", "run", "done", "failed"

    def __init__(self, log=None, cfg=None, floor_m=None, prefix="grab"):
        c = cfg or grab_cfg(prefix)
        self.floor = float(c["depth_floor_m"] if floor_m is None else floor_m)
        self.tol = float(c["depth_tol_m"])
        self.heave = float(c["descend_heave"])
        self.max_s = float(c["ensure_depth_s"])
        self.enable = bool(c["ensure_depth"])
        self.log = log or (lambda *a: None)
        self.state = self.IDLE if self.enable else self.DONE
        self.why = ""
        self._t0 = None

    def _floor(self, uart):
        v = getattr(uart, "effective_min_depth_m", None)
        return self.floor if v is None else max(self.floor, float(v))

    def step(self, now_ms, uart):
        """返回 `(状态, 本帧 DOF)` —— **本类不发帧**，下发点由调用方独占。"""
        if self.state in (self.DONE, self.FAILED):
            return self.state, ZERO_DOF
        if self._t0 is None:
            self._t0 = now_ms
        floor = self._floor(uart)
        d = getattr(uart, "depth_m", None)
        if d is None:
            self.state, self.why = self.FAILED, "no_depth_telemetry"
            self.log("[GRAB] ⚠️ 没有深度遥测，无法保证抬头不出水面 → 放弃")
            return self.state, ZERO_DOF
        # ★ 到位判据是**严格** `>= floor`：下限是硬边界，不能拿 `tol` 去啃它
        #   （原来写的是 `floor - tol` ⇒ 下限 0.55 时会在 0.50 就算"到位"，
        #    而 0.50 已经在下限之下、上浮又被禁 ⇒ 状态与保护自相矛盾）。
        if float(d) >= floor:
            self.state = self.DONE
            self.log("[GRAB] 深度 %.2fm ≥ 下限 %.2fm，可以抬头" % (float(d), floor))
            return self.state, ZERO_DOF
        if float(d) >= floor - self.tol:
            self.log("[GRAB] 深度 %.2fm 差下限 %.2fm 一点（容差 %.2f），继续压" 
                     % (float(d), floor, self.tol))
        if now_ms - self._t0 > self.max_s * 1000.0:
            self.state, self.why = self.FAILED, "depth_timeout"
            self.log("[GRAB] ⚠️ 下潜超时：深度 %.2fm 仍 < 下限 %.2fm → 放弃抬头"
                     % (float(d), floor))
            return self.state, ZERO_DOF
        self.state = self.RUN
        return self.state, (0.0, 0.0, self.heave, 0.0)


class TimedDof(object):
    """按固定 DOF 持续 T 毫秒（非阻塞）。**自己不发帧**：`step()` 只推进状态，
    调用方在 `state == RUN` 时发 `self.dof`，`DONE` 时发新相位的值/零。

    为什么改成这样：原来"子动作发一帧 + 主循环又补发一帧全零"，同帧两次 `send_dof`
    —— 实际只靠 UART 的 50ms 节流侥幸没把推力抹掉，但 `_dof_target` 已被清零。
    下发点唯一化后这类"隐形覆盖"不可能再发生。
    """

    IDLE, RUN, DONE = "idle", "run", "done"

    def __init__(self, dur_s=0.0, name="", log=None, **dof):
        self.dof = (float(dof.get("surge", 0.0)), float(dof.get("sway", 0.0)),
                    float(dof.get("heave", 0.0)), float(dof.get("yaw", 0.0)))
        self.dur_ms = max(0.0, float(dur_s) * 1000.0)
        self.name = name or "timed"
        self.log = log or (lambda *a: None)
        self.state = self.IDLE
        self._t0 = None

    @property
    def finished(self):
        return self.state == self.DONE

    def step(self, now_ms):
        if self.state == self.DONE:
            return self.state
        if self._t0 is None:
            self._t0 = now_ms
            self.state = self.RUN
            self.log("[GRAB] %s：surge=%.2f sway=%.2f heave=%.2f 持续 %.2fs"
                     % ((self.name,) + self.dof[:3] + (self.dur_ms / 1000.0,)))
        # `dur_ms <= 0` = 跳过这一步（第一步就结束）。**注意别写成 `dur_ms > 0 and ...`**：
        # 那样 `s: 0` 的步骤永远不结束 ⇒ 整个序列死等（这是本模块单测抓到的坑）。
        if now_ms - self._t0 >= self.dur_ms:
            self.state = self.DONE
        return self.state


def pitch_up_deg(cfg=None, prefix="grab"):
    """抬头的相对角：`up_sign × up_deg`（`up_sign` = 现场实测 -1.0；缺键时兜底同值）。"""
    p = sub(task_node(prefix), "pitch")
    return req(p, "up_sign") * abs(req(p, "up_deg"))


def roll_dump_deg(cfg=None, prefix="grab"):
    """横倾的相对角（含**未验证**的 `dump_sign`）：倒球（任务三）与放球（任务四）共用。"""
    r = sub(task_node(prefix), "roll")
    return req(r, "dump_sign") * abs(req(r, "dump_deg"))
