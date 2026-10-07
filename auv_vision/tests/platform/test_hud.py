# -*- coding: utf-8 -*-
"""tests/platform/test_hud.py — HUD 的「偏转角」显示行（`main.psi_line`）。
（详细用法、判据与实测见 doc/注释历史.md）"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import psi_line                                            # noqa: E402

GREEN = (0, 220, 0)
YELLOW = (0, 190, 255)
GRAY = (165, 165, 165)
ORANGE = (0, 140, 255)


def test_no_measurement_shows_dashes_not_zero():
    txt, col = psi_line({"hdg": None}, 8.0)
    assert "--" in txt and "0.0" not in txt, "没测到必须显示 --，不能显示 0：%r" % txt
    assert col == GRAY
    assert "no-full-frame" in txt


def test_within_tolerance_is_green():
    txt, col = psi_line({"hdg": -3.4, "hdg_state": "done"}, 8.0)
    assert col == GREEN and "-3.4" in txt and "done" in txt


def test_outside_tolerance_is_yellow():
    txt, col = psi_line({"hdg": 12.3, "hdg_state": "turn"}, 8.0)
    assert col == YELLOW and "+12.3" in txt and "turn" in txt


def test_exactly_on_tolerance_counts_as_converged():
    assert psi_line({"hdg": 8.0}, 8.0)[1] == GREEN
    assert psi_line({"hdg": 8.01}, 8.0)[1] == YELLOW


def test_iteration_shown_only_when_nonzero():
    assert "it=" not in psi_line({"hdg": 1.0, "hdg_i": 0}, 8.0)[0]
    assert "it=2" in psi_line({"hdg": 1.0, "hdg_i": 2}, 8.0)[0]


def test_skip_reason_is_highlighted():
    txt, col = psi_line({"hdg": 1.0, "hdg_state": "measure",
                         "hdg_skip": "mode=p3p(只有 full 帧的 psi 可用)"}, 8.0)
    assert col == ORANGE, "跳过正航向必须变色（否则会被当成航向正常）"
    assert "SKIP" in txt and "p3p" in txt


def test_tolerance_is_printed():
    assert "(tol 8.0)" in psi_line({"hdg": 0.0}, 8.0)[0]
    assert "(tol 10.0)" in psi_line({"hdg": 0.0}, 10.0)[0]


def test_missing_keys_do_not_crash():
    for info in ({}, {"hdg_state": None}, {"hdg": 0, "hdg_i": None}):
        txt, col = psi_line(info, 8.0)
        assert isinstance(txt, str) and len(col) == 3


# --------------------------------------------------------------------------- #
# through 专表（★ 2026-10-07 用户要求：**单独一份记录 through 的日志**）
# --------------------------------------------------------------------------- #
def test_through_log_records_only_the_dash_as_segments(tmp_path):
    """★ through 专表：**只记冲刺段**，每次冲刺自成一段（enter → 逐帧 → exit）。

    为什么单独一份：任务日志几千帧、冲刺只占十几~几十帧；单独成段才能 `grep '"evt":"enter"'`
    把每一次穿门拎出来对比（哪次走 `z≤cross`、哪次走门口超时兜底、各用了几帧多久）。
    """
    import json
    import main as M
    p = tmp_path / "t.jsonl"
    lg = M._ThroughLog(str(p))
    seq = [("ALIGN", "center", 0), ("APPROACH", "forward_creep", 0),
           ("THROUGH", "through", 0), ("THROUGH", "through", 0),
           ("THROUGH", "through", 1), ("ALIGN", "hold", 1)]
    for i, (ph, act, ps) in enumerate(seq):
        lg.write({"phase": ph, "action": act, "z": 1.0, "pass": ps}, 1000 + i * 100, i, "gate")
    lg.close()
    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert [r["evt"] for r in recs] == ["enter", "frame", "frame", "exit"], \
        "非冲刺帧不该写进来；冲刺段必须是 enter→frame…→exit"
    assert recs[0]["frames_in_through"] == 1 and recs[-2]["frames_in_through"] == 3
    assert recs[-1]["dur_ms"] == 300, "exit 要给出本段时长"
    assert recs[-1]["pass"] == 1, "exit 要带上 pass"
    # 没处于冲刺时：一个字都不写
    lg2 = M._ThroughLog(str(tmp_path / "t2.jsonl"))
    lg2.write({"phase": "ALIGN", "action": "hold"}, 0, 0, "gate")
    lg2.close()
    assert not (tmp_path / "t2.jsonl").exists(), "非冲刺段不该创建文件"


def test_through_log_counts_creep_through_too():
    """门口超时兜底的 `creep_through` 也算冲刺段（两种出口都要被记下来）。"""
    import main as M
    assert M._ThroughLog._is_through({"phase": "ALIGN", "action": "creep_through"}) is True
    assert M._ThroughLog._is_through({"phase": "THROUGH", "action": "through"}) is True
    assert M._ThroughLog._is_through({"phase": "APPROACH", "action": "forward_creep"}) is False


# --------------------------------------------------------------------------- #
# through 专表（★ 2026-10-07 用户要求：**单独一份记录 through 的日志**）
# --------------------------------------------------------------------------- #
def test_through_log_records_only_the_dash_as_segments(tmp_path):
    """★ through 专表：**只记冲刺段**，每次冲刺自成一段（enter → 逐帧 → exit）。

    为什么单独一份：任务日志几千帧、冲刺只占十几~几十帧；单独成段才能把每一次穿门拎出来
    对比（哪次走 `z<=cross`、哪次走门口超时兜底、各用了几帧多久）。
    """
    import json
    import main as M
    p = tmp_path / "t.jsonl"
    lg = M._ThroughLog(str(p))
    seq = [("ALIGN", "center", 0), ("APPROACH", "forward_creep", 0),
           ("THROUGH", "through", 0), ("THROUGH", "through", 0),
           ("THROUGH", "through", 1), ("ALIGN", "hold", 1)]
    for i, (ph, act, ps) in enumerate(seq):
        lg.write({"phase": ph, "action": act, "z": 1.0, "pass": ps}, 1000 + i * 100, i, "gate")
    lg.close()
    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert [r["evt"] for r in recs] == ["enter", "frame", "frame", "exit"], \
        "非冲刺帧不该写进来；冲刺段必须是 enter->frame...->exit"
    assert recs[0]["frames_in_through"] == 1 and recs[-2]["frames_in_through"] == 3
    assert recs[-1]["dur_ms"] == 300, "exit 要给出本段时长"
    assert recs[-1]["pass"] == 1, "exit 要带上 pass"
    lg2 = M._ThroughLog(str(tmp_path / "t2.jsonl"))
    lg2.write({"phase": "ALIGN", "action": "hold"}, 0, 0, "gate")
    lg2.close()
    assert not (tmp_path / "t2.jsonl").exists(), "非冲刺段不该创建文件"


def test_through_log_counts_creep_through_too():
    """门口超时兜底的 `creep_through` 也算冲刺段（两种出口都要被记下来）。"""
    import main as M
    assert M._ThroughLog._is_through({"phase": "ALIGN", "action": "creep_through"}) is True
    assert M._ThroughLog._is_through({"phase": "THROUGH", "action": "through"}) is True
    assert M._ThroughLog._is_through({"phase": "APPROACH", "action": "forward_creep"}) is False


def test_recorder_is_off_by_default_and_has_an_explicit_off_switch(monkeypatch):
    """★ 录像是**默认不启动**的；另外给一个**显式关闭**的写法（外层设了也能这一趟不录）。

    开关优先级：`AUV_RECORD=0|off|no|false|none|-` > `AUV_RECORD=<路径>` > `AUV_RECORD_DIR=<目录>`
    """
    import main as M
    for k in ("AUV_RECORD", "AUV_RECORD_DIR"):
        monkeypatch.delenv(k, raising=False)
    assert M._make_recorder().path is None, "默认必须是不录（不建文件、不写盘）"

    monkeypatch.setenv("AUV_RECORD_DIR", "/tmp/recdir")
    p1 = M._make_recorder().path
    assert p1 and p1.endswith(".mjpeg"), "设了 DIR 才自动命名录制"

    for word in ("0", "off", "no", "false", "none", "-", "OFF"):
        monkeypatch.setenv("AUV_RECORD", word)
        assert M._make_recorder().path is None, "AUV_RECORD=%s 必须显式关闭（即使 DIR 也设了）" % word

    monkeypatch.setenv("AUV_RECORD", "/tmp/x.mjpeg")
    monkeypatch.setenv("AUV_RECORD_DIR", "0")
    assert M._make_recorder().path == "/tmp/x.mjpeg", "显式路径优先于 DIR"


def test_no_run_script_turns_recording_on_by_default():
    """★ 三个 `task/run_*.sh` **都不该**替用户打开录像 —— 默认下水不录，要用就现场加变量。"""
    import io, os
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    for name in ("run_gate.sh", "run_ball_reverse.sh", "run_grab.sh"):
        p = os.path.join(root, "task", name)
        if not os.path.exists(p):
            continue
        src = io.open(p, encoding="utf-8").read()
        # 允许注释里提到，但不许有真正 export/前缀赋值
        for line in src.splitlines():
            ls = line.strip()
            if ls.startswith("#"):
                continue
            assert "AUV_RECORD=" not in ls, "%s 里替用户开了录像：%s" % (name, ls)
