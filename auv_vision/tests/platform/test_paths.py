# -*- coding: utf-8 -*-
"""tests/platform/test_paths.py — 路径依赖：cfg 相对路径 + 解析器 + 自检工具。

守的是 2026-09-22 板端踩过的两类坑：
  1. **cfg 里写绝对路径** ⇒ 把配置钉死在某一份拷贝上（板端有 `/home/sunrise/AUV` 与
     `~/Desktop/AUV_New` 两份），指错一份就静默用另一个文件、那份被删就直接崩；
     ⇒ 规则：cfg 一律写**仓库内相对路径**，运行时由 `base.settings.resolve_path()` 解析。
  2. **标定 yaml 的文件头**：板端 cv2 4.11 要求第一个字节是 `%YAML:1.0`。
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import base.settings as S                                            # noqa: E402
from tools.check import check_paths as CP                            # noqa: E402


# ------------------------------------------------------------------ 解析器
def test_resolve_path_keeps_absolute():
    assert S.resolve_path("/tmp/x.yaml") == "/tmp/x.yaml"


def test_resolve_path_joins_project_root():
    p = S.resolve_path("cfg/front_camera.yaml")
    assert os.path.isabs(p)
    assert p == os.path.join(S.project_root(), "cfg", "front_camera.yaml")


def test_resolve_path_passthrough_for_empty():
    assert S.resolve_path(None) is None
    assert S.resolve_path("") == ""
    assert S.resolve_path(123) == 123


def test_project_root_contains_main_and_cfg():
    r = S.project_root()
    assert os.path.exists(os.path.join(r, "main.py"))
    assert os.path.isdir(os.path.join(r, "cfg"))


# ------------------------------------------------------------------ 配置
def test_cfg_has_no_absolute_paths():
    """cfg/*.yaml 里出现 `/home/` 之类就说明又被钉死在某个拷贝上了。"""
    hits = CP.scan_cfg_abs_paths(os.path.join(ROOT, "cfg"))
    assert not hits, "cfg 里还有绝对路径：%s" % hits


def test_path_keys_resolve_to_existing_files():
    for k in CP.PATH_KEYS:
        v = S.get(k)
        if not v:
            continue
        assert os.path.isabs(str(v)), "%s 没被解析成绝对路径：%s" % (k, v)
        assert os.path.exists(str(v)), "%s 指向的文件不存在：%s" % (k, v)


def test_path_keys_list_matches_settings():
    """新增路径类参数时要同时加到 `base.settings._PATH_KEYS` 与工具里的清单。"""
    for k in CP.PATH_KEYS:
        short = k[len("vision."):] if k.startswith("vision.") else k
        assert short in S._PATH_KEYS, \
            "%s 不在 base.settings._PATH_KEYS 里 → 它的相对路径不会被解析" % short


# ------------------------------------------------------------------ 标定文件头
def test_yaml_head_helper_flags_comment_and_blank(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_bytes(b"# comment\n%YAML:1.0\n---\n")
    good, why = CP.check_yaml_head(str(p))
    assert not good and "注释" in why

    p.write_bytes(b"\n%YAML:1.0\n---\n")
    good, why = CP.check_yaml_head(str(p))
    assert not good, "前导空行也必须判失败（cv2 4.11 实测就是它）"

    p.write_bytes(b"%YAML:1.0\n---\nimage_width: 1280\n")
    good, _ = CP.check_yaml_head(str(p))
    assert good


def test_real_camera_yamls_pass_head_check():
    import glob
    files = sorted(glob.glob(os.path.join(ROOT, "cfg", "front_camera*.yaml")))
    assert files, "cfg/ 下应有 front_camera*.yaml"
    for p in files:
        good, why = CP.check_yaml_head(p)
        assert good, "%s: %s" % (os.path.basename(p), why)


# ------------------------------------------------------------------ 自检工具端到端
def test_check_paths_tool_passes_on_this_repo(capsys):
    """本仓库必须自检通过（否则说明有人又把绝对路径/坏文件头写回来了）。"""
    rc = CP.main(["--quiet"])
    out = capsys.readouterr().out
    assert rc == 0, "check_paths 失败：\n%s" % out


def test_deploy_scripts_default_to_the_live_copy():
    """部署/核对脚本的默认目录必须是**现场在用的工作副本** AUV_New。

    2026-09-22 之前默认是旧副本 `/home/sunrise/AUV` —— 照默认跑就会把文件推/比到错的拷贝上。
    """
    import glob
    import re
    for p in sorted(glob.glob(os.path.join(ROOT, "tools", "deploy", "*.sh"))):
        with open(p, encoding="utf-8") as fh:
            txt = fh.read()
        for m in re.finditer(r"AUV_BOARD_DIR:-([^}]*)}", txt):
            assert m.group(1) == "/home/sunrise/Desktop/AUV_New", \
                "%s 的默认板端目录是 %s（应为在用副本 AUV_New）" % (
                    os.path.basename(p), m.group(1))
