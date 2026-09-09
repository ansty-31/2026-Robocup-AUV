# -*- coding: utf-8 -*-
"""test_return_by_memory.py — 按记忆返回的轨迹工具测试"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task1_2 import return_by_memory as R  # noqa: E402

PASS = []


def check(name, cond, extra=""):
    if not cond:
        raise AssertionError("FAIL: %s %s" % (name, extra))
    PASS.append(name)
    print("PASS: %s" % name)


def test_tools():
    tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    tmp.write("dt_s,a0,a1,a2,a3\n0.10,128,255,128,128\n0.10,128,200,60,128\n")
    tmp.close()
    rows = R.load_log(tmp.name)
    os.unlink(tmp.name)
    check("load_rows", len(rows) == 2 and rows[0][1][1] == 255)
    back = R.negate_rows(rows)
    check("reverse_order", back[0][0] == rows[1][0] and back[0][1][2] ==
          R.negate_axis(60))
    check("negate_mid", R.negate_axis(128) == 128)
    check("negate_ff", R.negate_axis(255) == 1)
    est = R.estimate(rows)
    check("est_surge", abs(est["surge"] - (1.0 * 0.1 + 0.572 * 0.1)) < 0.05,
          est)


def main():
    test_tools()
    print("\n全部通过 %d 项" % len(PASS))


if __name__ == "__main__":
    main()
