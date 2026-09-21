# -*- coding: utf-8 -*-
"""cfgnode.py — 配置节点读取的公共件（gate_task / heading_align / turn_deg 共用）。

cfg 是 YAML → dict，读参数的规矩只有一条：
**缺键 / 类型不对时用"等于当前 cfg 的默认值"兜底，绝不抛异常**
（允许 `AUV_CFG_DIR` 指向缺新键的旧配置，也允许用例在内存里改 cfg）。

通用读法：
    sub(node, key)              取子节点；缺失/非 dict → {}
    merge(node, defaults)       子节点 + 默认值 → 普通 dict（缺键/None 用默认值）
    num(node, key, default)     取数值；缺失/非数值 → float(default)
    nums(node, defaults)        把不全的节点规范化成完整表（缺键用 defaults 补）
    flag(node, key, default)    取布尔（认 bool / "1|0|true|false|yes|no|on|off"）
    pid_kw(node, defaults)      按 (kp,ki,kd,out_max,deadzone) 造 PID 参数（out_max 兼作 ±限幅）

**共用运动参数**（`comm.motion.*`：ball / gate / 转向脚本用的是同一份，任务段不再各抄一套）：
    MOTION_DEFAULTS             代码兜底表（取值等于当前 cfg/comm.yaml 的 motion 段）
    motion_node(key)            取共用表（dict）
    motion_num(key)             取共用标量（float）

分层：本模块**不 import 任何东西**（不读文件、不碰 yaml），所以
`common/turn_deg.py` 这类要能脱离 cfg 独立跑的脚本可以安全地在模块级 import 它。
"""
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
    """把一个（可能不全的）配置节点规范化成**完整缺省表**：缺键用 defaults 补。

    用途：`nums(显式节点, 实时取到的另一套参数)` —— 缺省值直接取那份实时值，
    而不是在代码里再抄一份数字（那样又会变成两套参数各自漂移）。
    """
    return dict((k, num(node, k, defaults[k])) for k in defaults)


def flag(node, key, default=False):
    """取布尔开关：真 bool 原样；字符串按 1/0/true/false/yes/no/on/off 解析。

    ⚠️ **不要用 `bool(node.get(k))`**：`bool("false")` 是 True —— 板端就因为这个
    把 `kpt_mem.enable` 的一个拼写错值当成了"开"。这里统一按字符串语义解析。
    """
    if not isinstance(node, dict) or key not in node:
        return bool(default)
    v = node[key]
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v).strip().lower() not in _FALSE


def pid_kw(node, defaults):
    """按默认表造 PID 构造参数：`(kp, ki, kd, out_max, deadzone)`。

    `out_max` 同时作为 ±限幅（`out_min=-out_max`）。`node` 里写了的键优先，
    没写的取 `defaults` —— 于是"任务级覆盖 → 共用默认"就是一次调用。
    """
    d = merge(node, defaults)
    om = num(d, "out_max", defaults["out_max"])
    return dict(kp=num(d, "kp", defaults["kp"]), ki=num(d, "ki", defaults["ki"]),
                kd=num(d, "kd", defaults["kd"]), out_min=-om, out_max=om,
                deadzone=num(d, "deadzone", defaults["deadzone"]))


# --------------------------------------------------------------- 共用运动参数
# **ball / gate 共用**的底层运动参数（cfg/comm.yaml → comm.motion）。
# 这里存的是"配置缺键/缺文件"时的代码兜底，取值**等于当前 cfg**（有用例钉住"兜底 == cfg"）。
# 任务段只放本任务特有的旋钮；确需单独一套时才在自己的段里写覆盖键（如 gate.pid_sway）。
MOTION_DEFAULTS = dict(
    # 共用 PID（同一船/同一推进器；量纲都是"归一化偏差 ±1"）
    pid_sway=dict(kp=8.0, ki=0.01, kd=0.05, out_max=0.45, deadzone=0.05),
    pid_heave=dict(kp=1.0, ki=0.0, kd=0.15, out_max=1.0, deadzone=0.04),
    # 共用速度档（归一化 DOF；⚠️ 任何档都不能落在执行器死区 (0, 0.138)）
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
    """读 `comm.motion.<path>`（ball / gate / 转向脚本共用的那份）。

    延迟 import base.settings：本模块要保持"不碰文件"的性质（turn_deg 会在模块级 import 它）。
    """
    import base.settings as S
    return S.get("comm.motion." + path, default)
