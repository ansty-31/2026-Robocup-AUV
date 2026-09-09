# -*- coding: utf-8 -*-
"""
test_detector.py — 检测层测试（decode/NMS、NV12、真后端未就绪降级）
运行：cd auv_vision && python3 tests/test_detector.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                    # noqa: E402

import base.settings as S                         # noqa: E402
from common.detector import (DetectorHub, Det, pick_target,  # noqa: E402
                      bgr_to_packed_nv12, check_x5_model,
                      decode_yolov8)

PASS = []


def check(name, cond, extra=""):
    if not cond:
        raise AssertionError("FAIL: %s %s" % (name, extra))
    PASS.append(name)
    print("PASS: %s" % name)


def test_decode_multi_class():
    # 单权重三类别模型：[1, 4+3, N]（labels: blue_ball/red_ball/gate）
    labels = ["blue_ball", "red_ball", "gate"]
    anchors = np.array([
        [160, 160, 320, 320, 0.9, 0.1, 0.0],   # blue_ball 主目标
        [200, 190, 320, 320, 0.85, 0.0, 0.1],  # 与主目标高重叠 → NMS 去除
        [400, 400, 100, 100, 0.1, 0.1, 0.6],   # gate 独立框
        [300, 300, 40, 40, 0.2, 0.0, 0.0],     # 低置信 → 滤除
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.float32)
    out = anchors.T[np.newaxis, ...]           # [1,7,7]
    dets = decode_yolov8(out, labels, 1280, 720,
                         conf=0.5, iou=0.45, input_w=640, input_h=640)
    check("decode_count", len(dets) == 2, len(dets))
    kinds = sorted(d.kind for d in dets)
    check("decode_kinds", kinds == ["blue_ball", "gate"], kinds)
    blue = next(d for d in dets if d.kind == "blue_ball")
    gate = next(d for d in dets if d.kind == "gate")
    check("decode_blue", blue.x == 0 and blue.w == 640, blue)
    check("decode_gate", gate.x == 700 and gate.w == 200, gate)
    check("score_order", dets[0].score >= dets[1].score)
    # [1,N,4+nc] 布局（锚点数>特征数，不触发转置）
    anchors2 = np.vstack([anchors, np.zeros((1, 7), dtype=np.float32)])
    dets2 = decode_yolov8(anchors2[np.newaxis, ...], labels, 1280, 720,
                          conf=0.5, iou=0.45, input_w=640, input_h=640)
    check("decode_layout2", len(dets2) == 2, len(dets2))


def test_x5_model_file_check():
    try:
        check_x5_model("/models/auv_multi.hbm")
        check("x5_reject_hbm", False)
    except RuntimeError as e:
        check("x5_reject_hbm", "bayes" in str(e) or ".hbm" in str(e), e)
    try:
        check_x5_model("/models/auv_multi.bin")
        check("x5_missing_bin", False)
    except FileNotFoundError:
        check("x5_missing_bin", True)


def test_instruction_first():
    # 红球分更高，但指令要蓝球 → 必须选蓝球；红球不能替代目标
    dets = [Det("red_ball", 0.99, 0, 0, 10, 10),
            Det("blue_ball", 0.55, 0, 0, 20, 20),
            Det("gate", 0.90, 0, 0, 5, 5)]
    got = pick_target(dets, "blue_ball")
    check("instruction_color", got is not None and got.kind == "blue_ball")
    check("instruction_ignore_higher_other", got.score < 0.99, got.score)
    # 蓝球缺席 → None（不得自动换红球）
    absent = pick_target(dets, "red_ball")  # 存在红球场景测“指令色缺席”用另一组
    present_red = pick_target([Det("red_ball", 0.99, 0, 0, 10, 10),
                               Det("gate", 0.9, 0, 0, 5, 5)], "blue_ball")
    check("instruction_no_fallback", present_red is None)
    check("instruction_present", absent.kind == "red_ball")


def test_nv12():
    frame = np.full((64, 64, 3), 100, dtype=np.uint8)
    nv12 = bgr_to_packed_nv12(frame, 64, 64)
    check("nv12_len", len(nv12) == 64 * 64 * 3 // 2, len(nv12))
    y = nv12[:64 * 64]
    uv = nv12[64 * 64:]
    check("nv12_y", np.all(y == 100))
    check("nv12_uv", abs(int(uv.mean()) - 128) <= 2, uv.mean())


def test_real_mode_degrade():
    saved = (S.vision.model.mode, S.vision.model.path)
    try:
        S.vision.model.mode = "onnx"
        S.vision.model.path = "/nonexistent/model.onnx"
        hub = DetectorHub()
        check("onnx_not_ready", not hub.ready("ball"))
    finally:
        S.vision.model.mode, S.vision.model.path = saved


def main():
    test_decode_multi_class()
    test_x5_model_file_check()
    test_instruction_first()
    test_nv12()
    test_real_mode_degrade()
    print("\n全部通过 %d 项" % len(PASS))


if __name__ == "__main__":
    main()