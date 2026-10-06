# -*- coding: utf-8 -*-
"""common/motion/search_sweep.py — **纯左右平移扫视**（搜索用；任务三/四共用）。

用户 2026-10-06 定：任务三/四的 search **不用** `common/motion/search_scan`（那是 yaw 慢扫，
靠遥测 yaw 闭环），改成**只左右平移**。形状照 gate 里那套（`gate/motion/channels.py::_search_sweep`：
右 → 停 → 左 → 停，**每轮时长按 2 的次幂递增**，带总时长上限）—— 门后来换了方案，那套留档。

为什么平移而不是转 yaw（这条要记住）：
* 抬着头（pitch 30°）时 yaw 一转，画面里的目标会**绕圈跑**，而平移只会让目标**横移** ⇒ 居中
  与"够近了"的判据更容易保持一致；
* 平移**不依赖 yaw 遥测**（`search_scan` 没遥测就一步不转），少一个失效点；
* 代价（与门那套一样，**未验证**）：开环、无横向位置反馈 ⇒ 一趟来回后**可能有净漂移**，
  现场要实测"往左扫完再往右扫，船有没有整体漂走"。

非阻塞：每帧 `step(now_ms)` → 返回 sway DOF（`+` = 右移，见 `comm.dof_map.sway`）。
参数：`comm.motion.search_sweep`（共用默认），任务段 `comm.<prefix>.search_sweep` 可覆盖。
"""
from __future__ import annotations

import base.cfg.settings as S
from common.cfg.cfgnode import merge, num, sub, req, req_flag, req_node, MissingCfg

# ★ 2026-10-07 用户定：**删掉兜底表**。这里是"cfg 必须有哪些键"的**清单**，值一律从 cfg 读。
_K_SWEEP = (
    "sweep_s", "sweep_max_s", "pause_s", "sway", "duty", "duty_window_ms", "max_ms",
)


def sweep_cfg(prefix=None):
    """`motion.search_sweep` + 任务覆盖 `comm.<prefix>.search_sweep`。"""
    base = sub(S.get("comm.motion", None) or {}, "search_sweep")
    if prefix:
        base = merge(sub(S.get("comm.%s" % prefix, None) or {}, "search_sweep"), base)
    if not base:
        raise MissingCfg("cfg 缺 comm.motion.search_sweep（代码已无兜底）")
    return base


class Sweep(object):
    """左右平移扫视状态机（每帧 `step()` 一次）。"""

    def __init__(self, prefix=None, log=None):
        self.prefix = prefix
        self.log = log or (lambda *a: None)
        c = sweep_cfg(prefix)
        # ★ 无兜底：逐项必填（缺则 `MissingCfg` 报名字）
        self.sweep_s = max(0.0, req(c, "sweep_s"))
        self.sweep_max_s = max(self.sweep_s, req(c, "sweep_max_s"))
        self.pause_ms = max(0.0, req(c, "pause_s") * 1000.0)
        self.sway = req(c, "sway")
        # ★ 占空比：**幅值不动**（不掉死区），只把时间切成"推一小段 + 停一下"
        #   ⇒ 平均速度更慢、节奏更缓，也顺带给检测留出静止帧
        self.duty = max(0.0, min(1.0, req(c, "duty")))
        self.duty_window_ms = max(1.0, req(c, "duty_window_ms"))
        self.max_ms = max(0.0, req(c, "max_ms"))
        self.reset()

    def reset(self, now_ms=None):
        self._t0 = now_ms
        self._logged = False
        self.rounds = 0

    @property
    def done(self):
        return self._t0 is not None and self.max_ms > 0 and \
            (self._t_ms - self._t0) > self.max_ms

    def _sweep_len(self, k):
        """第 k 轮（0 起）的单程时长(ms)：base × 2^k，封顶 `sweep_max_s`。"""
        return min(self.sweep_s * 1000.0 * (2.0 ** k), self.sweep_max_s * 1000.0)

    def step(self, now_ms):
        """返回本帧的 sway DOF（段：右 → 停 → 左 → 停）。"""
        if self.sweep_s <= 0.0:
            return 0.0
        if self._t0 is None:
            self._t0 = now_ms
            self.log("[SWEEP] 开始左右平移扫视：首轮单程 %.1fs（每轮翻倍，上限 %.1fs），"
                     "幅值 %.2f，占空比 %.2f/%.0fms（推一小段停一下，不一次到位）"
                     % (self.sweep_s, self.sweep_max_s, self.sway, self.duty,
                        self.duty_window_ms))
        self._t_ms = now_ms
        t = now_ms - self._t0
        if self.max_ms > 0 and t > self.max_ms:
            return 0.0                       # 到点就不再扫（保持静止，等目标自己出现）
        k, t0 = 0, 0.0
        sweep = self._sweep_len(0)
        while k < 24:                        # 定位当前落在第几轮（每轮 = 右+停+左+停）
            cyc = 2.0 * (sweep + self.pause_ms)
            if t < t0 + cyc:
                break
            t0 += cyc
            k += 1
            sweep = self._sweep_len(k)
        if k != self.rounds:
            self.rounds = k
            self.log("[SWEEP] 第 %d 轮：单程 %.1fs" % (k + 1, sweep / 1000.0))
        ph = t - t0
        if ph < sweep:                       # 段0：向右
            return self._duty(self.sway, now_ms)
        if ph < sweep + self.pause_ms:       # 段1：停
            return 0.0
        if ph < 2.0 * sweep + self.pause_ms:  # 段2：向左
            return self._duty(-self.sway, now_ms)
        return 0.0                           # 段3：停

    def _duty(self, v, now_ms):
        """占空比：窗口内前 `duty` 比例推 `v`，其余发 0（= 同一方向**不一次到位**）。"""
        if self.duty >= 1.0:
            return v
        if self.duty <= 0.0:
            return 0.0
        ph = (now_ms % self.duty_window_ms) / self.duty_window_ms
        return v if ph < self.duty else 0.0
