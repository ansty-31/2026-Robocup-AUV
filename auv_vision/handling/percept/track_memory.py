# -*- coding: utf-8 -*-
"""handling/percept/track_memory.py — **历史位置推演**（用户 2026-10-07 定）。

"同时对比前几帧之间位置，历史位置推演，防止球丢失找不到。"

做什么：把最近若干帧的**检测结果**（归一化偏差 dx/dy + 半径）记住，用**匀速模型**外推
"现在球大概在哪"，于是：
* 短暂丢检测（水波/反光/掩膜闪断）时**接着按预测位置修**，不用一丢就回盲扫；
* 真丢了也**知道往哪边找**（扫视从"上次看到的那一侧"先扫）；
* 给日志留下轨迹，方便复盘。

不做什么（安全边界，别越界）：
* **不发明速度**：样本不足（< `min_samples`）就不预测，退化成"最后位置"；
* **不让预测无限外推**：`horizon_s` 到点即视为失效（`None`），交给搜索流程；
* **不碰 heave**（深度归定深）、**不碰 yaw**（夹取不许旋转）。
"""
from __future__ import annotations

from common.cfg.cfgnode import num, sub

#: 代码兜底（cfg 缺键/删段都不崩；要现场调就在 `comm.grab.track_memory` 里覆盖）
_D_MEM = dict(
    history=6,          # 记住最近多少帧
    min_samples=3,      # 少于这么多帧不估速度
    horizon_s=1.6,      # 预测有效期（超时 = 失效，交给搜索）
    #   ⚠️ 必须**覆盖** 丢目标的「宽限期 + 保持窗口」(comm.grab.lost.grace_frames/hold_s，
    #      默认 15 帧@20fps=0.75s + 0.8s ≈ 1.55s)，否则推演还没轮到就被判失效 —— 形同没有。
    #      幅值另有时效折减（decay × (1-age)），所以外推久了自己就趋近 0，不会乱推。
    v_max=2.0,          # 速度限幅（归一化偏差/秒）：防一帧跳变把预测甩飞
    decay=0.5,          # 预测输出的幅值折减（比"看得见"时更保守）
)


def _node(prefix="grab"):
    import base.cfg.settings as S
    g = S.get("comm.%s" % prefix, None) or {}
    return sub(g, "track_memory")


def mem_cfg(prefix="grab"):
    n = _node(prefix)
    out = {}
    for k, v in _D_MEM.items():
        val = n.get(k, None) if isinstance(n, dict) else None
        try:
            out[k] = float(val) if val is not None else float(v)
        except (TypeError, ValueError):
            out[k] = float(v)
    return out


class TrackMemory(object):
    """最近检测的轨迹缓冲 + 匀速外推（**纯状态机，无 IO**，可单测）。"""

    def __init__(self, prefix="grab", log=None):
        c = mem_cfg(prefix)
        self.history = max(2, int(c["history"]))
        self.min_samples = max(2, int(c["min_samples"]))
        self.horizon_ms = float(c["horizon_s"]) * 1000.0
        self.v_max = abs(float(c["v_max"]))
        self.decay = min(1.0, abs(float(c["decay"])))
        self.log = log or (lambda *a: None)
        self._s = []                       # [(t_ms, dx, dy, r)]
        self.last_seen_ms = None
        self.last_dir = None               # 'left' | 'right'：最后一次看到球在哪一侧

    # ---- 记录 / 查询 ------------------------------------------------------ #
    def add(self, now_ms, dx, dy, radius=None):
        """一帧**检到了**就记一笔（顺带更新"最后看到在哪一侧"）。"""
        self._s.append((float(now_ms), float(dx), float(dy),
                        None if radius is None else float(radius)))
        if len(self._s) > self.history:
            self._s = self._s[-self.history:]
        self.last_seen_ms = float(now_ms)
        self.last_dir = "left" if float(dx) < 0.0 else "right"

    def clear(self):
        self._s = []
        self.last_seen_ms = None
        self.last_dir = None

    @property
    def seen(self):
        return len(self._s)

    def velocity(self):
        """最小二乘/端点差分的**匀速速度**（归一化/秒）；样本不足 ⇒ `(0, 0)`。"""
        if len(self._s) < self.min_samples:
            return 0.0, 0.0
        t0, x0, y0 = self._s[0][0], self._s[0][1], self._s[0][2]
        t1, x1, y1 = self._s[-1][0], self._s[-1][1], self._s[-1][2]
        dt = (t1 - t0) / 1000.0
        if dt <= 1e-6:
            return 0.0, 0.0

        def _clip(v):
            return max(-self.v_max, min(self.v_max, v))

        return _clip((x1 - x0) / dt), _clip((y1 - y0) / dt)

    def predict(self, now_ms):
        """外推 `(dx, dy, scale)`；**没有可信预测就返回 `None`**。

        `scale` = 幅值折减系数（越久没看到越小）——调用方按它缩幅值，别把预测当"确定值"。
        """
        if not self._s:
            return None
        if self.last_seen_ms is not None and now_ms - self.last_seen_ms > self.horizon_ms:
            return None                                     # 超有效期：交给搜索
        dt = (float(now_ms) - self._s[-1][0]) / 1000.0
        vx, vy = self.velocity()
        dx = self._s[-1][1] + vx * dt
        dy = self._s[-1][2] + vy * dt
        scale = self.decay
        if self.last_seen_ms is not None and self.horizon_ms > 0:
            age = (float(now_ms) - self.last_seen_ms) / self.horizon_ms
            scale *= max(0.0, 1.0 - age)
        return dx, dy, scale

    def side_hint(self):
        """真丢了以后"先往哪边扫"：最后一次看到球的那一侧（没有记录 ⇒ `None`）。"""
        return self.last_dir
