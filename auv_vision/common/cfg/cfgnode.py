# -*- coding: utf-8 -*-
"""cfgnode.py — 配置节点读取的公共件（gate_task / hdg / turn_deg 共用）。
（详细用法、判据与实测见 doc/注释历史.md）"""
from __future__ import annotations

_FALSE = ("0", "false", "no", "off", "")


def sub(node, key):
    """取子配置节点；节点缺失/类型不对 → {}（后续用默认值，不抛 KeyError）。"""
    v = node.get(key) if isinstance(node, dict) else None
    return v if isinstance(v, dict) else {}


def merge(node, defaults):
    """子节点 + 默认值合并成普通 dict：缺键/None → 默认值，其余原样（保类型）。"""
    out = dict(defaults)
    if not isinstance(node, dict):
        return out
    for k in defaults:
        v = node.get(k)
        if v is not None:
            out[k] = v
    return out


def num(node, key, default):
    """取数值参数：缺失/非数值 → float(default)。"""
    try:
        return float(node[key])
    except (KeyError, TypeError, ValueError):
        return float(default)


def nums(node, defaults):
    """把一个（可能不全的）配置节点规范化成**完整缺省表**：缺键用 defaults 补。"""
    return dict((k, num(node, k, defaults[k])) for k in defaults)


def flag(node, key, default=False):
    """取布尔开关：真 bool 原样；字符串按 1/0/true/false/yes/no/on/off 解析。"""
    if not isinstance(node, dict) or key not in node:
        return bool(default)
    v = node[key]
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v).strip().lower() not in _FALSE


def pid_kw(node, defaults):
    """按默认表造 PID 构造参数：`(kp, ki, kd, out_max, deadzone)`。"""
    d = merge(node, defaults)
    om = num(d, "out_max", defaults["out_max"])
    return dict(kp=num(d, "kp", defaults["kp"]), ki=num(d, "ki", defaults["ki"]),
                kd=num(d, "kd", defaults["kd"]), out_min=-om, out_max=om,
                deadzone=num(d, "deadzone", defaults["deadzone"]))


MOTION_DEFAULTS = dict(
    # 共用 PID（同一船/同一推进器；量纲都是"归一化偏差 ±1"）
    pid_sway=dict(kp=8.0, ki=0.01, kd=0.05, out_max=0.55, deadzone=0.05),
    pid_heave=dict(kp=1.0, ki=0.0, kd=0.15, out_max=1.0, deadzone=0.04),
    surge_fast=0.35,
    surge_slow=0.15,
    loss_inertia_surge=0.25,
    search_yaw=0.3,
)


def motion_node(key):
    """取共用运动参数里的**表**（如 `pid_sway`）；缺键 → MOTION_DEFAULTS[key]。"""
    raw = motion(key, None)
    return merge(raw, MOTION_DEFAULTS[key])


def motion_num(key):
    """取共用运动参数里的**标量**（如 `surge_fast`）；缺键 → MOTION_DEFAULTS[key]。"""
    return num({"v": motion(key, None)}, "v", MOTION_DEFAULTS[key])


def motion_pid(key):
    """取共用运动参数里的 **PID 参数**（如 `pid_sway`）；缺键 → MOTION_DEFAULTS[key]。"""
    return pid_kw(motion_node(key), MOTION_DEFAULTS[key])


def motion(path, default=None):
    """读 `comm.motion.<path>`（ball / gate / 转向脚本共用的那份）。"""
    import base.cfg.settings as S
    return S.get("comm.motion." + path, default)
