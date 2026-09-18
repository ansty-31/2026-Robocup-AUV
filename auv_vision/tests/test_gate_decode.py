# -*- coding: utf-8 -*-
"""tests/test_gate_decode.py — gate keypoint 头解码（decode_yolo11_kpt）约定锁定

## 为什么必须有这组测试

`decode_yolo11_kpt` 是**训练侧 ONNX 与板端之间唯一的耦合面**，而它没有任何测试：
一旦张量约定被改（改 head 补丁、换导出方式、改 quantize 配置后的输出布局），
错误不会报错，只会让角点整体偏固定像素 —— 直接进 PnP，深度/姿态全偏。

三个已知的坑，每个都在这组测试里有对应的断言：

1. **-0.5 陷阱**：ONNX（`scripts/3_export/modify_ultralytics.py` 的 POSE_FORWARD）
   输出的 kpt x/y 是 `raw*2 + 网格索引`（cell 单位，已含网格项）。板端只需 `×stride`。
   早期版本多写一个 `-0.5`，造成 stride 8/16/32 上 4/8/16 px 的**固定偏移**。
   → `test_kpt_grid_term_is_plus_index`
2. **单位约定**：kpt 张量是「已含网格索引的 cell 坐标」，不是「相对网格的偏移」。
   → `test_kpt_channel_layout_and_units`
3. **letterbox vs squish**：板端预处理是 `cv2.resize((640,640))` 直接拉伸（非等比），
   所以回投到原帧必须用 `(frame_w/640, frame_h/640)` 这一对**不同**的系数。
   → `test_scale_back_to_frame_is_squish`

测试全部用**合成张量**，不需要权重/.bin/BPU，可离线跑：
    python3 -m pytest tests/test_gate_decode.py -q

## ⚠️ 工作区约定（务必遵守）

`auv_vision/auv_vision` 下**只有 `tests/` 里的测试脚本可以新增/修改**，
其余文件（`gate/`、`common/`、`base/`、`cfg/` …）在本工作区里**只读**：
可以读来核对约定，**不要改**。发现业务代码有问题，写一条会失败的测试把它钉住、
在报告里说明，由板端侧的人去改。

这条约定直接影响本文档的用法：**不要用「临时改 gate_decode.py 再跑测试」的方式
做负向对照**——那会写到只读文件上。本文件里的
`test_minus_half_trap_is_detectable` 用**临时目录里的副本**自动完成同样的验证。
"""
import numpy as np
import pytest

from gate.gate_decode import decode_yolo11_kpt

G = 80                                   # 用 stride=8 那一层做算术最直观
STRIDE = 640 / G                         # = 8.0
INPUT = 640
REG_MAX = 16
KPT_DIM = 4
LABELS = ["gate"]


def _blank_outputs(g=G, nc=1, kpt_dim=KPT_DIM, reg_max=REG_MAX):
    """一个尺度的空输出：{通道数: [1,g,g,C]}。

    cls 初值给 -10（sigmoid≈4.5e-5）而不是 0：sigmoid(0)=0.5 会全部越过 conf 阈值，
    合成用例里必须显式把背景压下去，否则解出几千个空框。
    """
    cls = np.full((1, g, g, nc), -10.0, np.float32)
    return {64: np.zeros((1, g, g, 4 * reg_max), np.float32),
            nc: cls,
            kpt_dim * 3: np.zeros((1, g, g, kpt_dim * 3), np.float32)}


def _set_dfl(out, gx, gy, dist=3):
    """把 (gx,gy) 格的 DFL 分布设成 one-hot 在 bin=dist → 解码距离恒为 dist 格。"""
    a = out[64][0, gy, gx]
    a[:] = 0.0
    for side in range(4):
        a[side * REG_MAX + dist] = 40.0          # softmax → 几乎 one-hot


def _set_score(out, gx, gy, logit=6.0):
    out[1][0, gy, gx, 0] = logit                 # sigmoid(6) ≈ 0.9975


def _set_kpt_cell(out, gx, gy, i, cell_x, cell_y, v_logit=6.0):
    """把第 i 个角点放在 **cell 坐标 (cell_x, cell_y)**。

    注意：ONNX 输出的 kpt x/y **已经是 cell 坐标**（= `raw*2 + 网格索引`，
    见 scripts/3_export/modify_ultralytics.py 的 POSE_FORWARD），
    所以这里直接写 cell 坐标；板端只做 `×stride`。
    """
    a = out[KPT_DIM * 3][0, gy, gx]
    a[3 * i + 0] = cell_x
    a[3 * i + 1] = cell_y
    a[3 * i + 2] = v_logit


# --------------------------------------------------------------- 1. 网格项

def test_kpt_grid_term_is_plus_index():
    """网格项必须是 **+索引**，不是 +索引-0.5。

    把角点编码在 cell 坐标 = 网格索引本身（raw=0）处：
        正确 → 像素 = gx * stride
        多写 -0.5 → 像素 = (gx-0.5) * stride，即 stride=8 上偏 4 px
    """
    gx, gy = 10, 20
    out = _blank_outputs()
    _set_score(out, gx, gy)
    _set_dfl(out, gx, gy, dist=3)
    for i in range(4):
        _set_kpt_cell(out, gx, gy, i, cell_x=gx, cell_y=gy)

    dets = decode_yolo11_kpt(out, LABELS, INPUT, INPUT,
                             conf=0.25, iou=0.45, input_w=INPUT, input_h=INPUT)
    assert len(dets) == 1, "合成张量应当解出 1 个目标"
    k = np.asarray(dets[0].kpts)

    # +索引 约定：cell 坐标 == 网格索引 → 像素 == 索引×stride
    np.testing.assert_allclose(k[:, 0], gx * STRIDE, atol=1e-3)
    np.testing.assert_allclose(k[:, 1], gy * STRIDE, atol=1e-3)

    # 显式反证：-0.5 约定会给出不同的值，且差正好半个 cell
    np.testing.assert_allclose(k[:, 0], (gx - 0.5) * STRIDE + 0.5 * STRIDE, atol=1e-3)
    assert abs(k[0, 0] - (gx - 0.5) * STRIDE) > 3.9, "若成立说明网格项被写成了 -0.5"


def test_kpt_cell_center_maps_to_half_cell_pixel():
    """cell 中心 (gx+0.5, gy+0.5) → 像素 ((gx+0.5)·stride, (gy+0.5)·stride)（与框心同基准）。"""
    gx, gy = 7, 33
    out = _blank_outputs()
    _set_score(out, gx, gy)
    _set_dfl(out, gx, gy, dist=2)
    for i in range(4):
        _set_kpt_cell(out, gx, gy, i, cell_x=gx + 0.5, cell_y=gy + 0.5)

    d = decode_yolo11_kpt(out, LABELS, INPUT, INPUT,
                          conf=0.25, iou=0.45, input_w=INPUT, input_h=INPUT)[0]
    np.testing.assert_allclose(np.asarray(d.kpts)[:, 0], (gx + 0.5) * STRIDE, atol=1e-3)
    np.testing.assert_allclose(np.asarray(d.kpts)[:, 1], (gy + 0.5) * STRIDE, atol=1e-3)


# --------------------------------------------------------------- 2. 单位/通道布局

def test_kpt_channel_layout_is_xyv_interleaved():
    """通道序必须是 [x,y,v]×K，且 K 个点各自独立（reshape(g*g, K, 3) 的语义）。"""
    gx, gy = 5, 6
    cells = [(gx + 0.5, gy + 0.5), (gx + 2.5, gy + 0.5),
             (gx + 2.5, gy + 2.5), (gx + 0.5, gy + 2.5)]
    out = _blank_outputs()
    _set_score(out, gx, gy)
    _set_dfl(out, gx, gy, dist=4)
    for i, (cx, cy) in enumerate(cells):
        _set_kpt_cell(out, gx, gy, i, cx, cy, v_logit=(6.0 if i < 3 else -6.0))

    d = decode_yolo11_kpt(out, LABELS, INPUT, INPUT, conf=0.25, iou=0.45,
                          input_w=INPUT, input_h=INPUT, vis_thr=0.5)[0]
    k = np.asarray(d.kpts)
    for i, (cx, cy) in enumerate(cells):
        np.testing.assert_allclose(k[i], [cx * STRIDE, cy * STRIDE], atol=1e-3)
    # v 经 sigmoid 后阈值化：第 4 点 logit=-6 → sigmoid≈0.0025 < 0.5 → conf 置 0
    c = np.asarray(d.kpt_conf)
    assert c[0] > 0.9 and c[3] == 0.0, "可见度应 sigmoid 后按 vis_thr 阈值化"


def test_box_decode_is_dfl_times_stride_around_grid_center():
    """框：dist 由 DFL 期望给出，中心 = (索引+0.5)×stride。"""
    gx, gy, dist = 12, 9, 5
    out = _blank_outputs()
    _set_score(out, gx, gy)
    _set_dfl(out, gx, gy, dist=dist)
    for i in range(4):
        _set_kpt_cell(out, gx, gy, i, cell_x=gx + 0.5, cell_y=gy + 0.5)

    d = decode_yolo11_kpt(out, LABELS, INPUT, INPUT,
                          conf=0.25, iou=0.45, input_w=INPUT, input_h=INPUT)[0]
    cx, cy = (gx + 0.5) * STRIDE, (gy + 0.5) * STRIDE
    # Det.x/y/w/h 是 int（见 common/detector.py:34），所以用 w/h 反推中心并放宽 1px
    got_cx = d.x + d.w / 2.0
    got_cy = d.y + d.h / 2.0
    assert abs(got_cx - cx) <= 1.0 and abs(got_cy - cy) <= 1.0
    assert abs(d.w - 2 * dist * STRIDE) <= 2 and abs(d.h - 2 * dist * STRIDE) <= 2


# --------------------------------------------------------------- 3. 回投原帧

def test_scale_back_to_frame_is_squish():
    """回投必须是 squish（x/y 用不同系数），因为板端预处理是直接 resize 到 640×640。"""
    gx, gy = 10, 20
    out = _blank_outputs()
    _set_score(out, gx, gy)
    _set_dfl(out, gx, gy, dist=3)
    for i in range(4):
        _set_kpt_cell(out, gx, gy, i, cell_x=gx + 0.5, cell_y=gy + 0.5)

    d = decode_yolo11_kpt(out, LABELS, 1280, 720,
                          conf=0.25, iou=0.45, input_w=640, input_h=640)[0]
    k = np.asarray(d.kpts)
    np.testing.assert_allclose(k[:, 0], (gx + 0.5) * STRIDE * (1280 / 640), atol=1e-3)
    np.testing.assert_allclose(k[:, 1], (gy + 0.5) * STRIDE * (720 / 640), atol=1e-3)
    # 若误用等比（两个方向都乘 2）→ y 会差 720/640=1.125 倍，这里显式反证
    assert abs(k[0, 1] - (gy + 0.5) * STRIDE * 2.0) > 1.0


# --------------------------------------------------------------- 4. 多尺度 / NMS

def test_multiscale_outputs_are_merged_and_nms_dedups():
    """3 个尺度的字典应合并，重叠目标被 NMS 压成一个。"""
    out = {}
    for g in (20, 40, 80):
        o = _blank_outputs(g=g)
        gx = gy = g // 4
        _set_score(o, gx, gy, logit=6.0)
        _set_dfl(o, gx, gy, dist=2)
        for i in range(4):
            _set_kpt_cell(o, gx, gy, i, cell_x=gx + 0.5, cell_y=gy + 0.5)
        out.update(o)

    dets = decode_yolo11_kpt(out, LABELS, INPUT, INPUT,
                             conf=0.25, iou=0.45, input_w=INPUT, input_h=INPUT)
    assert len(dets) == 1, f"多层同位置目标应被 NMS 合并，实际 {len(dets)}"
    assert dets[0].kind == "gate"
    assert dets[0].kpts.shape == (4, 2) and dets[0].kpt_conf.shape == (4,)


def test_low_score_cells_are_filtered_by_conf():
    """cls 低于 conf 的格子必须被丢掉（否则 NMS 前会爆炸）。"""
    out = _blank_outputs()
    _set_score(out, 3, 3, logit=-6.0)            # sigmoid ≈ 0.0025 < 0.25
    _set_dfl(out, 3, 3, dist=2)
    for i in range(4):
        _set_kpt_cell(out, 3, 3, i, cell_x=3.5, cell_y=3.5)
    assert decode_yolo11_kpt(out, LABELS, INPUT, INPUT, conf=0.25,
                             input_w=INPUT, input_h=INPUT) == []


# ------------------------------------------------- 5. 负向对照（测试本身有效吗）

def test_minus_half_trap_is_detectable(tmp_path):
    """负向对照：把 -0.5 bug 注入**副本**，上面的断言必须失败。

    「不能失败的测试等于没有测试」。这里不碰只读的 `gate/gate_decode.py`：
    把源码读成文本（只读操作）→ 在内存里改掉网格项 → 写到 pytest 的
    `tmp_path` 里作为一个独立模块加载 → 断言它解出的角点确实偏了 4 px（stride=8）。

    `base` / `common` / `gate.geometry` 仍从真工程导入（conftest 已把工程根
    放进 sys.path），所以副本与真模块的行为差异**只**来自那一处注入。
    """
    import importlib.util
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "gate" / "gate_decode.py"
    text = src.read_text(encoding="utf-8")                     # 只读
    anchor = ("        kx = kpt[..., 0] * stride * kpt_scale\n"
              "        ky = kpt[..., 1] * stride * kpt_scale")
    assert anchor in text, "gate_decode.py 的 kpt 解码锚点变了，请同步本测试"
    mutated = text.replace(
        anchor,
        "        kx = (kpt[..., 0] - 0.5) * stride * kpt_scale\n"
        "        ky = (kpt[..., 1] - 0.5) * stride * kpt_scale")

    bad_path = tmp_path / "gate_decode_minus_half.py"
    bad_path.write_text(mutated, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("gate_decode_minus_half", bad_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    gx, gy = 10, 20
    out = _blank_outputs()
    _set_score(out, gx, gy)
    _set_dfl(out, gx, gy, dist=3)
    for i in range(4):
        _set_kpt_cell(out, gx, gy, i, cell_x=gx, cell_y=gy)

    k_bad = np.asarray(mod.decode_yolo11_kpt(
        out, LABELS, INPUT, INPUT, conf=0.25, iou=0.45,
        input_w=INPUT, input_h=INPUT)[0].kpts)
    k_ok = np.asarray(decode_yolo11_kpt(
        out, LABELS, INPUT, INPUT, conf=0.25, iou=0.45,
        input_w=INPUT, input_h=INPUT)[0].kpts)

    # 正确实现在网格索引处；注入 -0.5 后整体偏半个 cell = 4 px（stride=8）
    np.testing.assert_allclose(k_ok[:, 0], gx * STRIDE, atol=1e-3)
    np.testing.assert_allclose(k_bad[:, 0], (gx - 0.5) * STRIDE, atol=1e-3)
    assert abs(k_bad[0, 0] - k_ok[0, 0]) == pytest.approx(0.5 * STRIDE, abs=1e-3)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
