# -*- coding: utf-8 -*-
"""test_settings.py — YAML 参数加载/访问测试（config.py 已删除，全部走 YAML）
运行：cd auv_vision && python3 tests/test_settings.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base.settings as settings  # noqa: E402

PASS = []


def check(name, cond, extra=""):
    if not cond:
        raise AssertionError("FAIL: %s %s" % (name, extra))
    PASS.append(name)
    print("PASS: %s" % name)


def test_access():
    # 点号读取 = YAML 直读（vision / comm 两文件）
    labels = list(settings.vision.model.labels)
    wb = list(settings.vision.image.white_balance_bgr)
    check("vision_dot",
          settings.vision.camera.front.type in ("sim", "usb") and
          settings.vision.model.input_size == 640 and
          {"red_ball", "blue_ball", "gate"} <= set(labels) and
          len(wb) == 3 and all(isinstance(x, (int, float)) for x in wb))
    check("comm_dot",
          isinstance(settings.comm.serial.baud, int) and
          settings.comm.serial.baud > 0 and
          settings.comm.frame.header == 0xA5 and
          settings.comm.dof_map.surge.axis == 1)
    check("pid_dot", settings.comm.ball.pid.kp == 0.9 and
          settings.comm.ball.align_x is True)
    # 下标/整数键/嵌套列表
    check("index", settings.comm.frame.aux_axis[4] == 165 and
          "red_ball" in settings.vision.model.labels and
          settings.comm.motion.presets.stop[0] == 0.0)
    # 未知属性报错
    try:
        _ = settings.vision.not_exist_key
        check("unknown_raises", False)
    except AttributeError:
        check("unknown_raises", True)


def test_override_and_reload():
    # 运行期赋值（点号）→ 读取生效
    old = settings.vision.sim.down.red_line.dx_offset
    settings.vision.sim.down.red_line.dx_offset = 0.42
    check("override_dot", settings.vision.sim.down.red_line.dx_offset == 0.42)
    settings.vision.sim.down.red_line.dx_offset = old
    check("override_restore", settings.vision.sim.down.red_line.dx_offset == old)

    tmp = tempfile.mkdtemp(prefix="auv_cfg_")
    try:
        import pathlib
        base = pathlib.Path(__file__).resolve().parent.parent / "cfg"
        shutil.copy(base / "vision.yaml", tmp)
        text = (base / "comm.yaml").read_text(encoding="utf-8")
        text = text.replace("idle_ms: 1000", "idle_ms: 4242")
        (pathlib.Path(tmp) / "comm.yaml").write_text(text, encoding="utf-8")
        os.environ["AUV_CFG_DIR"] = tmp
        settings.reload()
        check("reload_env", settings.comm.tasks.idle_ms == 4242,
              settings.comm.tasks.idle_ms)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        os.environ.pop("AUV_CFG_DIR", None)
        settings.reload()
        check("reload_restore", settings.comm.tasks.idle_ms == 1000)


def main():
    test_access()
    test_override_and_reload()
    print("\n全部通过 %d 项" % len(PASS))


if __name__ == "__main__":
    main()