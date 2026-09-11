# -*- coding: utf-8 -*-
"""settings.py — YAML 参数加载（全部参数见 cfg/vision.yaml 与 cfg/comm.yaml）

用法（代码与 YAML 一一对应）：
    import base.settings as S
    S.vision.camera.mode            # → vision.yaml camera.mode
    S.comm.serial.baud              # → comm.yaml serial.baud
    S.comm.frame.aux_axis[4]        # 整数键用索引
运行期可改：S.vision.sim.red_line.dx_offset = 0.3
换配置路径：环境变量 AUV_CFG_DIR（目录内含同名 yaml）、
           AUV_VISION_CFG / AUV_COMM_CFG
"""
import os

# ---- 代码级常量（非参数；参数一律在 cfg/*.yaml）----
PROJECT_NAME = "AUV"
SIM_MODE = True          # 无硬件调试：串口仅打印（板上置 False）
DEBUG = True
LOG_FPS = True

STATE_IDLE = "IDLE"
STATE_BALL = "BALL"
STATE_BALL_FWD = "BALL_FWD"      # 撞球简化版（无命中识别，累计前进10s停）
STATE_BACK = "BACK"
STATE_GATE = "GATE"
STATE_DONE = "DONE"
STATE_ESTOP = "ESTOP"
STATUS_RUNNING = "RUNNING"
STATUS_DONE = "DONE"
STATUS_LOST = "LOST"


class Y(dict):
    """带属性访问的 dict（深包装，读写均可点号或下标）。"""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = _deep(value)


def _deep(v):
    if isinstance(v, dict):
        return Y({k: _deep(x) for k, x in v.items()})
    if isinstance(v, list):
        return [_deep(x) for x in v]
    return v


def _load(path):
    try:
        import yaml
    except ImportError:
        raise RuntimeError("缺少 PyYAML：pip install pyyaml"
                           "（板端: sudo apt install python3-yaml）")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError("配置文件格式错误: %s" % path)
    return _deep(data)


def _resolve(env_val, default):
    if os.environ.get("AUV_CFG_DIR"):
        cand = os.path.join(os.environ["AUV_CFG_DIR"], os.path.basename(default))
        if os.path.exists(cand):
            return cand
    if env_val and os.path.exists(env_val):
        return env_val
    return default


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # base/ 的上一级 = 工程根（cfg/ 所在）
_DEFAULT_VISION = os.path.join(_ROOT, "cfg", "vision.yaml")
_DEFAULT_COMM = os.path.join(_ROOT, "cfg", "comm.yaml")

vision = _load(_resolve(os.environ.get("AUV_VISION_CFG"), _DEFAULT_VISION))
comm = _load(_resolve(os.environ.get("AUV_COMM_CFG"), _DEFAULT_COMM))


def get(path, default=None):
    """按 'vision.a.b' 路径读取，缺任意一级返回 default（兼容旧 YAML）。"""
    parts = path.split(".")
    if not parts or parts[0] not in ("vision", "comm"):
        return default
    node = globals().get(parts[0])
    if node is None:
        return default
    for k in parts[1:]:
        try:
            node = node[k]
        except (KeyError, AttributeError, TypeError):
            return default
    return node


def reload():
    """从 YAML 重新加载（丢弃运行期改动）。"""
    global vision, comm
    vision = _load(_resolve(os.environ.get("AUV_VISION_CFG"), _DEFAULT_VISION))
    comm = _load(_resolve(os.environ.get("AUV_COMM_CFG"), _DEFAULT_COMM))
