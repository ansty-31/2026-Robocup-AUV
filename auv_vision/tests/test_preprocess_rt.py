# -*- coding: utf-8 -*-
"""tests/test_preprocess_rt.py — 模型**运行时输入链路**的一致性守卫

`decode_yolo11_kpt` 管的是「张量→角点」；这一组管的是它前面那半条链：
「BGR 帧 → 640×640 → NV12 → BPU → 张量」。这条链上任一处静默偏差，
表现都是「模型精度莫名下降」而不是报错，所以必须有断言钉住。

## 覆盖的两个坑

1. **NV12 打包的色度保真**：板端有两条打包实现，`vision.model.fast_nv12` 切换。
   `bgr_to_packed_nv12_fast`（cv2 / 标准 BT.601）是忠实的；
   `bgr_to_packed_nv12`（numpy）把 2×2 求和当平均，**色度被放大 4 倍**
   （该函数自己的 docstring 已注明）。BPU 会用这批 NV12 反解回 RGB 喂网络，
   所以选错路径 = 模型看到的颜色整体偏，精度掉但没有异常。
   → `test_selected_nv12_path_is_color_faithful`
2. **非等比缩放（squish）**：板端预处理是 `cv2.resize((640,640))` 直接拉伸，
   不是 letterbox。所以 `scale()` 给出的 x/y 系数**不相等**（1280×720 → 2.0 / 1.125）。
   若哪天改成 letterbox，角点回投就会出现与位置相关的偏差。
   → `test_preprocess_is_squish_not_letterbox`

跑法：python3 -m pytest tests/test_preprocess_rt.py -q

## ⚠️ 工作区约定（务必遵守）

`auv_vision/auv_vision` 下**只有 `tests/` 里的测试脚本可以新增/修改**，
其余文件（`gate/`、`common/`、`base/`、`cfg/` …）在本工作区里**只读**：
可以读来核对约定，**不要改**。本文件里的两条断言就是按这个原则写的 ——
它们只**读取** `bgr_to_packed_nv12*` 与配置项，发现问题时让测试失败并说明，
而不是去改 `common/detector.py` 或 `cfg/vision.yaml`。
"""
import numpy as np
import pytest

import base.settings as S
from common.detector import bgr_to_packed_nv12, bgr_to_packed_nv12_fast

H = W = 640


def _test_frame():
    """确定性合成帧：含平滑渐变 + 饱和色块，能同时压到亮度和色度通道。"""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.zeros((H, W, 3), np.float32)
    img[..., 0] = 40 + 120 * (xx / W)
    img[..., 1] = 30 + 150 * (yy / H)
    img[..., 2] = 200 - 100 * (xx / W)
    img[80:240, 80:240] = (30, 40, 220)      # 红块（BGR）
    img[400:560, 380:560] = (230, 200, 40)   # 青块
    return np.clip(img, 0, 255).astype(np.uint8)


def _nv12_to_bgr(packed):
    import cv2
    n = W * H
    y = packed[:n].reshape(H, W)
    uv = packed[n:]
    u = uv[0::2].reshape(H // 2, W // 2)
    v = uv[1::2].reshape(H // 2, W // 2)
    i420 = np.concatenate([y.reshape(-1), u.reshape(-1), v.reshape(-1)])
    return cv2.cvtColor(i420.reshape(H * 3 // 2, W), cv2.COLOR_YUV2BGR_I420)


@pytest.mark.parametrize("packer,name", [
    (bgr_to_packed_nv12_fast, "fast(cv2/BT.601)"),
    (bgr_to_packed_nv12, "numpy"),
])
def test_nv12_packer_roundtrip_error(packer, name):
    """两条打包路径各自的往返保真度（记录基线，便于定位是哪条路径退化了）。"""
    img = _test_frame()
    back = _nv12_to_bgr(packer(img, W, H))
    err = float(np.abs(back.astype(np.int32) - img.astype(np.int32)).mean())
    if name.startswith("fast"):
        assert err < 3.0, f"{name} 色度应当忠实，实测往返误差 {err:.2f} 灰阶"
    else:
        # numpy 版已知有 4× 色度偏差（见其 docstring）；这里只记录量级
        assert err > 5.0, ("numpy 版的色度偏差消失了？如果它被修好了，"
                           "请把 vision.model.fast_nv12 的取舍与本文档一起更新")


def test_selected_nv12_path_is_color_faithful():
    """**当前配置选中的**那条 NV12 路径必须是色度忠实的。

    `vision.model.fast_nv12=false` 会切到 numpy 版，而那一版的色度被放大 4 倍
    （实测往返误差 ~15 灰阶、最大 161）。BPU 用这批 NV12 反解 RGB 喂网络，
    模型看到的颜色会整体偏 —— 不报错，只掉精度。这条断言把取舍摆到测试里。
    """
    fast = bool(S.vision.model.fast_nv12)
    packer = bgr_to_packed_nv12_fast if fast else bgr_to_packed_nv12
    img = _test_frame()
    back = _nv12_to_bgr(packer(img, W, H))
    err = float(np.abs(back.astype(np.int32) - img.astype(np.int32)).mean())
    assert err < 3.0, (
        "vision.model.fast_nv12=%s 选中的 NV12 路径色度失真（往返误差 %.2f 灰阶）。"
        "这条路径会改变模型输入的颜色分布，请改用 fast 版或先修 numpy 版的色度系数。"
        % (fast, err))


def test_preprocess_is_squish_not_letterbox():
    """预处理必须是「直接拉伸到正方形」，回投系数 x/y 不相等。"""
    from common.preprocess import ModelPreprocessor
    pre = ModelPreprocessor(calib_path=None, undistort=False)
    out = pre.process(_test_frame())
    assert out.shape[:2] == (pre.size, pre.size), "预处理输出必须是 size×size"

    sx, sy = pre.scale(1280, 720)
    assert sx == pytest.approx(1280 / 640) and sy == pytest.approx(720 / 640)
    assert abs(sx - sy) > 1e-6, ("x/y 系数相等说明变成了等比缩放；"
                                 "板端 cfg/vision.yaml 与 prepare_frames.py 都是 squish，"
                                 "改 letterbox 必须两侧同时改并重训")
