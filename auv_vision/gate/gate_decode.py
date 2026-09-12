# -*- coding: utf-8 -*-
"""gate/gate_decode.py — keypoint 模型(门+4角)后端与解码脚手架

状态说明（诚实标注）：
  真实 gate keypoint 权重尚未导出/量化，本模块提供：
  1) `decode_yolo11_kpt`：按 X5 split-head 惯例实现角点解码；
     坐标基准（相对输入 or 相对框、归一化、可见度 sigmoid 阈值）为约定值，
     导出 .bin 后必须用 §6"keypoint 解码自检"（合成图比对标定/量纲）校核修正；
  2) `GateKeypointBackend`：镜像 detector 真机后端骨架（hbm_runtime / pyeasy_dnn / onnx），
     输入 nv12(640)，输出按 labels 取 gate 检测并附 kpts。
依赖方向：gate → detector 公共件（符合 §5.0）。
"""
from __future__ import annotations

import numpy as np

import base.settings as S
from common.detector import (Det, bgr_to_packed_nv12, bgr_to_packed_nv12_fast,
                      decode_yolo11_split, _nms)  # noqa: F401
from gate.geometry import GATE_FRAME_W, GATE_FRAME_H

_KPT_PER_PT = 3                       # (x, y, visible)


def find_model_input(model, default="input"):
    """解析 hbm_runtime 模型的输入名。

    以模型自身元数据(input_names/input_name)为准；cfg 的 vision.model.input_name
    只有在确实命中模型输入名时才采用，否则忽略 —— 避免出现
    `Input name "input" is invalid for model ...` 这类硬编码错。
    """
    names = []
    for attr in ("input_names", "input_name"):
        v = getattr(model, attr, None)
        if isinstance(v, dict):                     # {model_name: [输入名...]}
            for val in v.values():
                if isinstance(val, (list, tuple)):
                    names.extend(str(x) for x in val)
                elif isinstance(val, str):
                    names.append(val)
        elif isinstance(v, (list, tuple)):
            names.extend(str(x) for x in v)
        elif isinstance(v, str):
            names.append(v)
    cfg = getattr(S.vision.model, "input_name", None)
    if cfg and names and str(cfg) in names:
        return str(cfg)
    if names:
        return names[0]
    return str(cfg) if cfg else default


def decode_yolo11_kpt(outputs, labels, frame_w, frame_h,
                      conf=None, iou=None, input_w=None, input_h=None,
                      reg_max=16, kpt_dim=4, vis_thr=0.5,
                      kpt_scale=1.0):
    """从 X5 split-head 输出还原 检测框+角点。

    约定（导出后需用 §6 自检校核）：
      - 每尺度：reg(64) + cls(nc) + kpt(3*kpt_dim)；
      - 角点坐标与框心同一网格单位(×stride)，可见度经 sigmoid 后阈值 vis_thr；
      - kpt_scale 校正"坐标单位=输入像素"的缩放差异（默认 1.0）。

    Returns: [Det(...)]（Det.kpts=(4,2) 像素、kpt_conf=(4,)）。
    """
    conf = conf if conf is not None else S.vision.model.score_threshold
    iou = iou if iou is not None else S.vision.model.nms_threshold
    input_w = input_w or S.vision.model.input_size
    input_h = input_h or S.vision.model.input_size
    nc = len(labels)
    kch = _KPT_PER_PT * kpt_dim
    bins = np.arange(reg_max, dtype=np.float32)
    grids = {}
    for arr in outputs.values():
        a = np.asarray(arr)
        if a.ndim != 4 or a.shape[0] != 1:
            continue
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
        x1 = cx - dist[:, 0] * stride
        y1 = cy - dist[:, 1] * stride
        x2 = cx + dist[:, 2] * stride
        y2 = cy + dist[:, 3] * stride
        kpt = tens[kch].reshape(g * g, kpt_dim, _KPT_PER_PT)[idx]
        vis = 1.0 / (1.0 + np.exp(-kpt[..., 2].astype(np.float64)))
        kx = kpt[..., 0] * stride * kpt_scale
        ky = kpt[..., 1] * stride * kpt_scale
        allb.append(np.stack([x1, y1, x2, y2], axis=1))
        alls.append(score[idx])
        allc.append(pr[idx].argmax(axis=1))
        allk.append(np.stack([kx, ky, vis], axis=-1))
    if not allb:
        return []
    boxes = np.concatenate(allb)
    scores = np.concatenate(alls)
    clss = np.concatenate(allc)
    kpts = np.concatenate(allk)                 # (M, kpt_dim, 3)
    sx, sy = frame_w / input_w, frame_h / input_h
    boxes = np.clip(np.stack([boxes[:, 0] * sx, boxes[:, 1] * sy,
                              boxes[:, 2] * sx, boxes[:, 3] * sy], axis=1),
                    0, [frame_w, frame_h, frame_w, frame_h])
    keep = []
    for c in range(nc):
        b = np.where(clss == c)[0]
        if b.size:
            keep.extend(b[_nms(boxes[b], scores[b], iou)])
    out = []
    for i in keep:
        kind = labels[int(clss[i])] if int(clss[i]) < nc else "unknown"
        kpt_xy = kpts[i, :, :2] * np.array([sx, sy], np.float32)
        kpt_v = kpts[i, :, 2]
        ok = kpt_v >= vis_thr
        out.append(Det(kind, scores[i], boxes[i][0], boxes[i][1],
                       boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1],
                       kpts=kpt_xy.astype(np.float32),
                       kpt_conf=np.where(ok, kpt_v, 0.0).astype(np.float32)))
    out.sort(key=lambda d: d.score, reverse=True)
    return out


class GateKeypointBackend(object):
    """真机 gate keypoint 后端骨架（hbm_runtime / pyeasy_dnn / onnx，同 detector 惯例）。"""

    def __init__(self, path, labels, kpt_order, camera,
                 kind="hbm_runtime", input_size=640):
        from gate.geometry import CameraModel
        assert isinstance(camera, CameraModel)
        self.labels = list(labels)
        self.kpt_order = list(kpt_order)          # ['TL','TR','BR','BL']
        self.camera = camera
        self.path = path
        self.input_size = int(input_size)
        self.kind = kind
        self._key = "input"                       # hbm_runtime 输入名(加载后解析)
        self._init_backend()

    # ------------------------------------------------------------ 后端加载
    def _init_backend(self):
        if self.kind == "hbm_runtime":
            import hbm_runtime
            self._model = hbm_runtime.HB_HBMRuntime(self.path)
            self._key = find_model_input(self._model)
        elif self.kind == "pyeasy_dnn":
            from hobot_dnn import pyeasy_dnn as dnn
            self._model = dnn.load(self.path)
        elif self.kind == "onnx":
            import onnxruntime
            self._sess = onnxruntime.InferenceSession(self.path)
            self._iname = self._sess.get_inputs()[0].name
            shape = self._sess.get_inputs()[0].shape
            if len(shape) == 4 and shape[2] and shape[3]:
                self.input_size = int(shape[2])
        else:
            raise ValueError("未知 kind: %s" % self.kind)

    # ------------------------------------------------------------ 推理
    def detect(self, frame):
        from common.preprocess import ModelPreprocessor
        pre = ModelPreprocessor(calib_path=S.vision.camera.front.calibration)
        h, w = frame.shape[:2]
        square = pre.process(frame)
        if S.get("vision.model.fast_nv12", False):
            nv12 = bgr_to_packed_nv12_fast(square, pre.size, pre.size)
        else:
            nv12 = bgr_to_packed_nv12(square, pre.size, pre.size)
        if self.kind == "hbm_runtime":
            outputs = self._model.run({self._key: nv12})
            if isinstance(outputs, dict) and len(outputs) == 1:
                outputs = next(iter(outputs.values()))
        elif self.kind == "pyeasy_dnn":
            m = self._model[0]
            try:
                outputs = m.forward([nv12])
            except TypeError:
                outputs = m.forward(nv12)
        else:
            rgb = square[..., ::-1].astype(np.float32) / 255.0
            outputs = self._sess.run(None, {self._iname: rgb[None].transpose(0, 3, 1, 2)})
        # X5 导出结构未定型：默认按 split-head 张量字典；若是列表则包成 dict 占位
        if not isinstance(outputs, dict):
            outputs = {("out%d" % i): o for i, o in enumerate(outputs)}
        return decode_yolo11_kpt(outputs, self.labels, w, h,
                                 input_w=pre.size, input_h=pre.size)
