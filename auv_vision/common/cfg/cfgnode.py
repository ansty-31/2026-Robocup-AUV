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


def pid_kw(node, where=""):
    """按 **cfg 必填**造 PID 构造参数：`(kp, ki, kd, out_max, deadzone)`。

    ★ 2026-10-07 用户定：**彻底删掉兜底** —— 五个键一个都不能缺（缺则报名字）。
    `where` 只用于报错时指路（如 `comm.gate.motion.pid_sway`）。
    """
    om = req(node, "out_max")
    out = dict(kp=req(node, "kp"), ki=req(node, "ki"), kd=req(node, "kd"),
               out_min=-om, out_max=om, deadzone=req(node, "deadzone"),
               # `bias` = **执行器死区补偿**（可选；缺省 0 = 不补偿，行为与加它之前一致）
               bias=float(node.get("bias", 0.0) or 0.0))
    return out


# ★ 2026-10-07 用户定：**`MOTION_DEFAULTS` 已彻底删除**（兜底机制不再存在）。
#   共用运动参数一律**必填**，缺键报名字；键名清单见 `_K_MOTION_*`（各调用方各自校验）。


def motion_node(key):
    """取共用运动参数里的**表**（如 `pid_sway`）：**必填**，缺则报 `comm.motion.<key>`。"""
    raw = motion(key, None)
    if not isinstance(raw, dict):
        raise MissingCfg("cfg 缺 comm.motion.%s（代码已无兜底）" % key)
    return raw


def motion_num(key):
    """取共用运动参数里的**标量**（如 `surge_fast`）：**必填**，缺则报 `comm.motion.<key>`。"""
    v = motion(key, None)
    if v is None:
        raise MissingCfg("cfg 缺 comm.motion.%s（代码已无兜底）" % key)
    return float(v)


def motion_pid(key):
    """取共用运动参数里的 **PID 参数**（如 `pid_sway`）：**必填**。"""
    return pid_kw(motion_node(key), "comm.motion.%s" % key)


def motion(path, default=None):
    """读 `comm.motion.<path>`（ball / gate / 转向脚本共用的那份）。"""
    import base.cfg.settings as S
    return S.get("comm.motion." + path, default)


# ---------------------------------------------------------------- 「必填」访问器
# ★ 2026-10-07 用户定：**删掉代码里的兜底默认值**（改一个参数要同步两处，纯鸡肋）。
#   代之以"**必填**"语义：cfg 缺键 ⇒ **当场报出键名**，而不是静默用代码里的默认值。
#   为什么必须"报错"而不是"返回 None/0"：静默兜底会让 `z.cross` 缺键变成 0
#   ⇒ 永远不冲刺；`keypoint.conf_thr` 缺键变成 0 ⇒ 全角点可信。**没人报错，船照跑**。

class MissingCfg(KeyError):
    """cfg 必填项缺失/非法（代码已无兜底，必须在 cfg/*.yaml 里补上）。"""


def _where(node):
    """尽力给出"这个节点是谁"，便于按报错直接定位 yaml 行。"""
    for a in ("_path", "path", "name"):
        v = getattr(node, a, None) or (node.get(a) if isinstance(node, dict) else None)
        if v:
            return str(v)
    return "?"


def req(node, key):
    """**必填**数值：缺失/非数值 ⇒ 抛 `MissingCfg`（报出键名与位置）。"""
    if not isinstance(node, dict) or node.get(key) is None:
        raise MissingCfg("cfg 必填缺参数：%s.%s（代码已无兜底，请写进 cfg/*.yaml）"
                         % (_where(node), key))
    try:
        return float(node[key])
    except (TypeError, ValueError):
        raise MissingCfg("cfg 参数 %s.%s 不是数值：%r" % (_where(node), key, node[key]))


def req_flag(node, key):
    """**必填**布尔：缺失 ⇒ 抛 `MissingCfg`；有值则按 `flag` 的宽松规则解析。"""
    if not isinstance(node, dict) or key not in node:
        raise MissingCfg("cfg 必填开关缺失：%s.%s（代码已无兜底）" % (_where(node), key))
    return flag(node, key, False)


def req_node(node, key):
    """**必填**子节点：缺失/不是 dict ⇒ 抛 `MissingCfg`。"""
    if not isinstance(node, dict) or not isinstance(node.get(key), dict):
        raise MissingCfg("cfg 必填子节点缺失：%s.%s（代码已无兜底）" % (_where(node), key))
    return node[key]
