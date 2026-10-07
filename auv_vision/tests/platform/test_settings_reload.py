# -*- coding: utf-8 -*-
"""tests/platform/test_settings_reload.py — `settings.reload()` 的正确性（真 bug 回归）。

2026-10-07 发现：`reload()` **漏了** import 时那一步"路径类键解析" ⇒ reload 之后
`camera.*.calibration` / `model.path` 会退回**相对路径**（`tests/platform/test_paths.py`
的用例一抓就红）。`AUV_CFG_DIR` / `AUV_VISION_CFG` 切换配置的路径都会踩到它。
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import base.cfg.settings as S          # noqa: E402
import importlib.util                       # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'check_paths', os.path.join(ROOT, 'tools', 'check', 'pipeline', 'check_paths.py'))
_cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cp)
PATH_KEYS = _cp.PATH_KEYS      # 与 base.cfg.settings._PATH_KEYS 同一份语义（那边是私有名）


def _path_keys_are_absolute():
    return [k for k in PATH_KEYS if S.get(k) and not os.path.isabs(str(S.get(k)))]


def test_reload_keeps_path_keys_absolute():
    assert not _path_keys_are_absolute(), "import 期就该解析成绝对路径"
    S.reload()
    try:
        assert not _path_keys_are_absolute(), \
            "reload() 之后路径类键必须是绝对路径（漏了 _resolve_path_keys 就会退回相对）"
    finally:
        S.reload()


def test_reload_follows_the_cfg_dir(monkeypatch, tmp_path):
    """`AUV_CFG_DIR` 里放一份 comm.yaml/vision.yaml ⇒ 解析到的就是那一份。"""
    (tmp_path / "comm.yaml").write_text("grab: {timeout_ms: 12345}\n", encoding="utf-8")
    (tmp_path / "vision.yaml").write_text("model: {path: models/x.bin}\n", encoding="utf-8")
    monkeypatch.setenv("AUV_CFG_DIR", str(tmp_path))
    S.reload()
    try:
        assert S.get("comm.grab.timeout_ms") == 12345, "换 cfg 目录后取的是新目录那份"
        # 路径类键按**工程根**解析（`resolve_path` 的语义），且 reload 后必须是绝对路径
        v = str(S.get("vision.model.path"))
        assert os.path.isabs(v) and v.endswith(os.path.join("models", "x.bin")), v
    finally:
        monkeypatch.delenv("AUV_CFG_DIR", raising=False)
        S.reload()
