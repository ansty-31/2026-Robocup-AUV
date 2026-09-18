#!/usr/bin/env python3
"""
test_decode_parity.py — 端到端校核「本地导出的 pose ONNX」与「板端 gate/gate_decode.py」是否对齐

## 为什么需要这个

`decode_yolo11_kpt` 是训练侧与板端**唯一**的耦合面，而它自身的 docstring 也写着
「坐标基准为约定值，导出 .bin 后必须校核」。这里有三处一旦错就会让 PnP 深度/角点**整体偏**：

1. **网格项（-0.5 陷阱）**：ultralytics 的 `kpts_decode` 是
   `(raw*2 + (anchors - 0.5)) * stride`，而 `make_anchors` 的 anchors 含 `+0.5`
   （`utils/tal.py:310-311`），两项相消后净为 `+index`。少写或多写 0.5 cell
   → stride 8/16/32 上就是 4/8/16 px 的固定偏移。
2. **letterbox vs squish**：训练图由 `prepare_frames.py` 用 `cv2.resize((640,640))`
   直接拉伸（squish），板端 `common/preprocess.py:118` 同样是 squish；
   而 ultralytics 的 `predict/val` 默认走 letterbox。三者不一致 → 几何分布漂移。
3. **单位约定**：kpt 张量里到底是「相对网格的 cell 偏移」还是「已含网格索引的 cell 坐标」，
   决定板端该不该再补索引项。

本脚本用真图把三条链在同一条数据上对拍：

    A. 训练侧预处理 vs 板端预处理（逐像素）
    B. 板端 decode(ONNX 输出) vs ultralytics 参考输出（逐角点，px）
    C. 网格项取 `+index` vs `+index-0.5` 各自与参考的误差（确认没有 0.5 cell 偏移）

## 用法

    # 先恢复原版 head.py（predict 需要；导出 ONNX 才要补丁版）
    cp <site-packages>/ultralytics/nn/modules/head.py.backup .../head.py

    python scripts/3_export/test_decode_parity.py --onnx weights/yolo11n-pose.onnx \
        --board /home/ansty/RDKX5/auv_vision/auv_vision --n 20

    # 跑完记得重新打补丁（导出 ONNX 用）
    python scripts/3_export/modify_ultralytics.py --task pose

## 验收标准

* A 段：板端与训练侧预处理逐像素最大差 ≈ 0（同一 JPEG 质量/同一链路时 0；
  经 JPEG 往返则个位数灰阶）
* B 段：角点平均误差 < 1 px、最大 < 3 px（同图同权重的两条解码路径应当几乎一致）
* C 段：`+index` 的误差显著小于 `+index-0.5`（否则说明约定反了，需改板端 decode）

## ⚠️ 与板端工作区的边界

本脚本会 `sys.path.insert` 板端工程根并 **import** `gate.gate_decode` /
`common.preprocess` —— 只读，不写。

`auv_vision/auv_vision` 下**只有 `tests/` 里的测试脚本可以新增/修改**，
其余文件在本工作区里一律**只读**。所以：

* 不要为了做对照实验去改板端的业务代码（历史上踩过：就地注入 -0.5 bug 验证测试有效性，
  虽然逐字节还原了，但按约定不该这么做）。要注入变体就在临时目录里改副本，
  参考板端 `tests/test_gate_decode.py::test_minus_half_trap_is_detectable` 的写法。
* 发现板端有问题 → 在板端 `tests/` 里加一条会失败的测试把它钉住，报告里说明，由板端侧修。

## 已知的未覆盖范围（别误读成"全链路已验"）

* 本脚本验的是 **float32 ONNX**。量化后的 `*.bin` 在 BPU 上跑出来的数值差异
  （校准集分布 → 量化阈值）是**另一条轴**，需要上机或用 OE 工具链验证。
* NV12 打包 / BPU 的 YUV→RGB 反解不在本脚本覆盖范围内，
  由板端 `tests/test_preprocess_rt.py` 守卫色度保真。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BOARD = Path("/home/ansty/RDKX5/auv_vision/auv_vision")


# --------------------------------------------------------------------- 工具

def iou_xyxy(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def local_decode(outputs, labels, frame_w, frame_h, input_w, input_h,
                 reg_max=16, kpt_dim=4, vis_thr=0.5, conf=0.25, iou=0.45,
                 grid_shift=0.0, kpt_scale=1.0):
    """`gate/gate_decode.py:decode_yolo11_kpt` 的等价实现，唯一区别是网格项可加
    `grid_shift`（用于 C 段的 -0.5 对照）。grid_shift=0 时应与板端函数逐位一致。"""
    from common.detector import Det, _nms
    nc = len(labels)
    kch = 3 * kpt_dim
    bins = np.arange(reg_max, dtype=np.float32)
    grids = {}
    for arr in outputs.values():
        a = np.asarray(arr)
        if a.ndim == 4 and a.shape[0] == 1:
            grids.setdefault(a.shape[1], {})[a.shape[3]] = a[0]
    allb, alls, allc, allk = [], [], [], []
    for g, tens in grids.items():
        if 64 not in tens or nc not in tens or kch not in tens:
            continue
        stride = float(input_w) / g
        reg = tens[64].reshape(g * g, 4, reg_max)
        cls = tens[nc].reshape(g * g, nc)
        pr = 1.0 / (1.0 + np.exp(-cls.astype(np.float64)))
        score = pr.max(axis=1)
        idx = np.where(score >= conf)[0]
        if not idx.size:
            continue
        r = reg[idx].astype(np.float64)
        e = np.exp(r - r.max(axis=2, keepdims=True))
        e /= e.sum(axis=2, keepdims=True)
        dist = (e * bins).sum(axis=2)
        ys, xs = np.divmod(idx, g)
        cx = (xs.astype(np.float32) + 0.5) * stride
        cy = (ys.astype(np.float32) + 0.5) * stride
        kpt = tens[kch].reshape(g * g, kpt_dim, 3)[idx]
        vis = 1.0 / (1.0 + np.exp(-kpt[..., 2].astype(np.float64)))
        kx = (kpt[..., 0] + grid_shift * 0.0 + 0.0) * stride
        # 网格项已烘进 ONNX 输出；grid_shift 以 cell 为单位整体平移该尺度
        kx = (kpt[..., 0] + grid_shift) * stride * kpt_scale
        ky = (kpt[..., 1] + grid_shift) * stride * kpt_scale
        allb.append(np.stack([cx - dist[:, 0] * stride, cy - dist[:, 1] * stride,
                              cx + dist[:, 2] * stride, cy + dist[:, 3] * stride], 1))
        alls.append(score[idx]); allc.append(pr[idx].argmax(1))
        allk.append(np.stack([kx, ky, vis], -1))
    if not allb:
        return []
    boxes = np.concatenate(allb); scores = np.concatenate(alls)
    clss = np.concatenate(allc); kpts = np.concatenate(allk)
    sx, sy = frame_w / input_w, frame_h / input_h
    boxes = np.clip(np.stack([boxes[:, 0] * sx, boxes[:, 1] * sy,
                              boxes[:, 2] * sx, boxes[:, 3] * sy], 1),
                    0, [frame_w, frame_h, frame_w, frame_h])
    keep = []
    for c in range(nc):
        b = np.where(clss == c)[0]
        if b.size:
            keep.extend(b[_nms(boxes[b], scores[b], iou)])
    out = []
    for i in keep:
        kpt_xy = kpts[i, :, :2] * np.array([sx, sy], np.float32)
        v = kpts[i, :, 2]
        out.append((boxes[i], float(scores[i]), kpt_xy,
                    np.where(v >= vis_thr, v, 0.0)))
    return out


# --------------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser(description="ONNX ↔ 板端 decode 端到端对拍")
    ap.add_argument("--onnx", default=str(PROJECT_ROOT / "weights/yolo11n-pose.onnx"))
    ap.add_argument("--weights", default=str(PROJECT_ROOT / "weights/yolo11n-pose.pt"),
                    help="ultralytics 参考模型（需原版 head.py）")
    ap.add_argument("--board", type=Path, default=DEFAULT_BOARD, help="板端工程根目录")
    ap.add_argument("--frames", type=Path,
                    default=PROJECT_ROOT / "data/AUV_4/auv_4_frames")
    ap.add_argument("--n", type=int, default=20, help="抽样帧数")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--vis-thr", type=float, default=0.5)
    a = ap.parse_args()

    sys.path.insert(0, str(a.board))
    import base.settings as S                                   # noqa: E402
    from common.preprocess import ModelPreprocessor             # noqa: E402
    from gate.gate_decode import decode_yolo11_kpt              # noqa: E402
    import onnxruntime as ort                                   # noqa: E402

    calib = str(a.board / "cfg" / "front_camera.yaml")
    pre = ModelPreprocessor(calib_path=calib)
    size = pre.size
    print(f"板端预处理: undistort={pre.undistort} gains={pre.gains} "
          f"clahe={pre.clip} gamma={pre.gamma} size={size}")

    # ---- 训练侧同链路（复刻 scripts/1_prepare/prepare_frames.py:197-233）----
    sys.path.insert(0, str(PROJECT_ROOT / "scripts/1_prepare"))
    import prepare_frames as pf                                 # noqa: E402
    gains, clahe, gamma = pre.gains, pre.clip, pre.gamma
    maps = pf.calibration_maps(calib, 1280, 720)

    def train_side(raw):
        f = cv2.remap(raw, maps[0], maps[1], cv2.INTER_LINEAR)
        f = cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR)
        return pf.enhance(f, gains, clahe, gamma)

    files = sorted(a.frames.glob("*.jpg"))
    if not files:
        sys.exit(f"❌ 没找到帧: {a.frames}")
    step = max(1, len(files) // a.n)
    files = files[::step][:a.n]
    print(f"对拍帧数: {len(files)}  (来源 {a.frames})\n")

    sess = ort.InferenceSession(str(a.onnx), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    onames = [o.name for o in sess.get_outputs()]

    from ultralytics import YOLO
    ref = YOLO(a.weights)

    A_diff, B_errs, C_shift_errs, B_vis = [], [], [], []
    for k, fp in enumerate(files):
        raw = cv2.imread(str(fp))
        if raw is None:
            continue
        sq_b = pre.process(raw)               # 板端
        sq_t = train_side(raw)                # 训练侧
        A_diff.append(np.abs(sq_b.astype(np.int16) - sq_t.astype(np.int16)))

        x = sq_b[..., ::-1].astype(np.float32) / 255.0
        outs = sess.run(None, {iname: x[None].transpose(0, 3, 1, 2)})
        od = dict(zip(onames, outs))

        dets = decode_yolo11_kpt(od, ["gate"], size, size,
                                 conf=a.conf, iou=0.45,
                                 input_w=size, input_h=size, vis_thr=a.vis_thr)
        locs = local_decode(od, ["gate"], size, size, size, size,
                            conf=a.conf, grid_shift=0.0, vis_thr=a.vis_thr)
        locs_m = local_decode(od, ["gate"], size, size, size, size,
                              conf=a.conf, grid_shift=-0.5, vis_thr=a.vis_thr)
        if dets and locs:
            d = np.abs(np.asarray(dets[0].kpts) - locs[0][2]).max()
            if d > 1e-4:
                print(f"  ⚠️ 本地复刻与板端 decode 不一致: max={d:.5f} px（用例失效）")

        # ultralytics 参考（同一张 640 图，imgsz=640 → letterbox 退化为恒等）
        r = ref.predict(sq_b, imgsz=640, conf=a.conf, device="cpu", verbose=False)[0]
        if r.keypoints is None or r.keypoints.data.shape[0] == 0 or not dets:
            continue
        Rk = r.keypoints.xy.cpu().numpy()[0]
        Rc = (r.keypoints.conf.cpu().numpy()[0] if r.keypoints.conf is not None
              else np.ones(len(Rk), np.float32))
        Rb = r.boxes.xyxy.cpu().numpy()[0]
        D = dets[0]
        db = np.array([D.x, D.y, D.x + D.w, D.y + D.h])
        if iou_xyxy(db, Rb) < 0.5:
            continue
        # ⚠️ ultralytics 会把低置信角点写成 (0,0)，板端则保留真实坐标、只把 conf 置 0。
        #    这是有意的行为差异（gate_frontend.parse_kpt_mode 靠 conf 判可见性），
        #    对拍时必须把参考里 (0,0)/低置信的点剔掉，否则误差是假的。
        ok = (Rc >= a.vis_thr) & ~((Rk[:, 0] == 0) & (Rk[:, 1] == 0))
        if ok.sum() < 2:
            continue
        Dk = np.asarray(D.kpts)
        B_errs.append(np.linalg.norm(Dk[ok] - Rk[ok], axis=1))
        B_vis.append(ok.sum())
        if locs and locs_m:
            E5 = np.linalg.norm(locs_m[0][2][ok] - Rk[ok], axis=1)
            C_shift_errs.append(E5)

    print("=" * 72)
    print("A. 预处理对拍（板端 ModelPreprocessor vs prepare_frames 链路，灰阶）")
    if A_diff:
        m = np.array([d.max() for d in A_diff])
        print(f"   最大差: 均值 {m.mean():.2f}  最大 {m.max()}   样本 {len(A_diff)}")
        print("   " + ("✅ 完全一致" if m.max() == 0 else "⚠️ 有差异，检查参数/顺序"))
    else:
        print("   （无样本）")

    print("\nB. 解码对拍（板端 decode vs ultralytics 同图预测，px@640）")
    if B_errs:
        Flat = np.concatenate(B_errs)
        print(f"   样本 {len(B_errs)} 张 / 参与对比角点 {Flat.size} 个")
        print(f"   整体 平均 {Flat.mean():.2f} px   最大 {Flat.max():.2f} px")
        print("   " + ("✅ 两条路径一致（<1px）" if Flat.mean() < 1.0 and Flat.max() < 3.0
                       else "⚠️ 存在系统性差异，需查网格项/单位约定"))
    else:
        print("   （没有可匹配的检测，检查 conf 或权重）")

    print("\nC. 网格项对照（-0.5 陷阱）：同一次 ONNX 输出，两种网格项与参考的误差")
    if C_shift_errs:
        F5 = np.concatenate(C_shift_errs)
        F0 = np.concatenate(B_errs)
        print(f"   +index-0.5 变体: 平均 {F5.mean():.2f} px  最大 {F5.max():.2f} px")
        print(f"   +index     变体: 平均 {F0.mean():.2f} px  最大 {F0.max():.2f} px")
        print(f"   → 差值 {F5.mean() - F0.mean():+.2f} px   "
              + ("✅ +index 正确" if F5.mean() > F0.mean() else "⚠️ 反而 -0.5 更优，约定反了"))
    else:
        print("   （无样本）")
    print("=" * 72)


if __name__ == "__main__":
    main()
