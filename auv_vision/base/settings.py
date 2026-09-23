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
# 运行期可覆盖（默认行为不变）：AUV_SIM_MODE=0 真正发串口 / =1 只打印
#   AUV_SIM_MODE=0 python3 main.py --task gate      ← 下水真实驱动
#   AUV_SIM_MODE=1 python3 main.py --task gate      ← 台架只打印（默认）
if os.environ.get("AUV_SIM_MODE") is not None:
    SIM_MODE = os.environ["AUV_SIM_MODE"].strip().lower() not in \
        ("0", "false", "no", "off", "sim_off")
DEBUG = True
LOG_FPS = True

STATE_IDLE = "IDLE"
STATE_BALL = "BALL"
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


# --------------------------------------------------------------- 路径解析
# cfg 里的文件路径**一律写仓库内相对路径**（`cfg/front_camera.yaml`、`models/x.bin`），
# 运行时在这里统一按**工程根**解析成绝对路径。为什么这么做（2026-09-22 踩过）：
#   板端有两份拷贝（`/home/sunrise/AUV` 旧副本 与 `~/Desktop/AUV_New` 在用副本），
#   绝对路径会把标定/权重钉死在某一个拷贝上 —— 那份被删/被改/被搬，管线立刻崩。
#   写成相对路径后，**整棵树搬到哪都能跑**（本地 / AUV_New / 旧副本 / U 盘）。
# 绝对路径仍然接受（老配置不改也能跑），但不推荐。
def project_root():
    """工程根（含 `main.py`/`cfg/`/`models/` 的那一层）。"""
    return _ROOT


def resolve_path(p):
    """相对路径按工程根解析；绝对路径原样返回；空值原样返回。"""
    if not p or not isinstance(p, str):
        return p
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(_ROOT, p))


#: cfg 里"值是文件路径"的键（新增路径类参数时**记得加到这里**，否则不会被解析）
_PATH_KEYS = (
    "camera.front.calibration",
    "camera.ball.calibration",
    "model.path",
    "model.task_models.gate.path",
)


def _resolve_path_keys(node, keys):
    """就地把 cfg 里指定键的相对路径解析成绝对路径（键不存在就跳过）。"""
    for k in keys:
        parts = k.split(".")
        cur = node
        for part in parts[:-1]:
            cur = cur.get(part) if isinstance(cur, dict) else None
            if not isinstance(cur, dict):
                cur = None
                break
        if isinstance(cur, dict) and cur.get(parts[-1]):
            cur[parts[-1]] = resolve_path(cur[parts[-1]])


_resolve_path_keys(vision, _PATH_KEYS)


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
