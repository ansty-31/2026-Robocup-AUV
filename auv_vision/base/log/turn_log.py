# -*- coding: utf-8 -*-
"""base/log/turn_log.py — **转向调用日志**（只写字，不参与任何控制）。
· 谁调用的（src=cli/turn/gate/gate_loop）、进了几次（n 递增）、间隔多久；
用法：
from base.log.turn_log import turn_log"""
from __future__ import annotations

import json
import os
import threading
import time

_LOCK = threading.Lock()
_N = 0

def default_path():
    """没配 `AUV_TURN_LOG` 时的落盘路径：<工程根>/log/turn_calls.jsonl。"""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "log", "turn_calls.jsonl")

def log_path():
    """当前生效的日志路径；返回 None = 关掉（`AUV_TURN_LOG=` 空串）。"""
    env = os.environ.get("AUV_TURN_LOG")
    if env is not None and not env.strip():
        return None
    return (env or "").strip() or default_path()

def turn_log(event, src="turn", **fields):
    """追加一行转向调用日志。event ∈ {call, enter, anchor, exit, return, gate_loop, ...}。"""
    global _N
    try:
        p = log_path()
        if not p:
            return
        d = os.path.dirname(p)
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
        with _LOCK:
            _N += 1
            rec = {"t": round(time.time(), 3),          # 墙钟（板端 RTC 可能不准）
                   "mono": round(time.monotonic(), 3),  # 单调钟：**看前后顺序/间隔用这个**
                   "hms": time.strftime("%H:%M:%S"),
                   "n": _N, "pid": os.getpid(), "src": src, "event": event}
            for k, v in fields.items():
                if isinstance(v, bool) or v is None or isinstance(v, (int, str)):
                    rec[k] = v
                elif isinstance(v, float):
                    rec[k] = round(v, 4)
                else:
                    rec[k] = str(v)
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
