# -*- coding: utf-8 -*-
"""detector.py — 目标识别（RDK X5，单权重多类别）

三目标（red_ball / blue_ball / gate）共用一个模型（cfg/vision.yaml model.path），
一次前向输出所有类别，由 DetectorHub 按任务取所需类别：
  - 撞球 ball  → mission.target_color 对应类别（red_ball / blue_ball）
  - 过门 gate  → gate
  - 返回 back 走颜色逻辑（不加载模型）

模型文件（RDK X5）：
  - 必须是 X5 OE 工具链（OpenExplorer，hb_mapper makertbin，march: bayes-e）产出的 **.bin**
  - 名称 hbm_runtime 只是包名传承：X5 上它加载的仍是 .bin；
    .hbm 属车载 J5/J6/S100(Nash-e/m) 路线，X5 无法解析 → 代码会直接拒绝并提示

DETECTOR 模式（vision.yaml model.mode）：
  mock        虚拟测试（无权重；按任务分别模拟）
  hbm_runtime RDK X5 BPU 推理：packed NV12 → model.run()
  onnx        CPU onnxruntime 调试（.onnx 不经过工具链，仅联调用）
"""
import os

import numpy as np

import base.settings as S
from common.preprocess import ModelPreprocessor


class Det(object):
    __slots__ = ("kind", "score", "x", "y", "w", "h", "kpts", "kpt_conf")

    def __init__(self, kind, score, x, y, w, h, kpts=None, kpt_conf=None):
        self.kind = kind
        self.score = float(score)
        self.x, self.y, self.w, self.h = int(x), int(y), int(w), int(h)
        # keypoint 扩展（gate 用；bbox 任务为 None，向后兼容）：
        #   kpts: (K,2) 像素坐标，顺序=训练顺序；kpt_conf: (K,) 每点置信度
        self.kpts = None if kpts is None else np.asarray(kpts, np.float32)
        self.kpt_conf = None if kpt_conf is None else np.asarray(kpt_conf, np.float32)

    @property
    def area(self):
        return float(self.w * self.h)

    def ratio(self, fw, fh):
        return self.area / float(fw * fh)

    @property
    def center(self):
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    def __repr__(self):
        return "Det(%s %.2f %d,%d %dx%d)" % (self.kind, self.score,
                                              self.x, self.y, self.w, self.h)


class DetectorBase(object):
    """所有检测器返回 Det 列表（可能为空）。"""

    def detect(self, frame):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 预处理 / 后处理
# ---------------------------------------------------------------------------
def bgr_to_packed_nv12(bgr, out_w, out_h):
    """BGR→packed NV12(Y+UV 拼接)。要求分辨率==模型输入。"""
    if bgr.shape[1] != out_w or bgr.shape[0] != out_h:
        raise RuntimeError("分辨率不符：需 %dx%d" % (out_w, out_h))
    f = bgr.astype(np.float32)
    y = np.clip(0.299 * f[..., 2] + 0.587 * f[..., 1] + 0.114 * f[..., 0],
                0, 255).astype(np.uint8)
    q = f[0::2, 0::2] + f[0::2, 1::2] + f[1::2, 0::2] + f[1::2, 1::2]
    u = np.clip(-0.169 * q[..., 2] - 0.331 * q[..., 1] + 0.5 * q[..., 0] + 128,
                0, 255).astype(np.uint8)
    v = np.clip(0.5 * q[..., 2] - 0.419 * q[..., 1] - 0.081 * q[..., 0] + 128,
                0, 255).astype(np.uint8)
    uv = np.empty((out_h // 2, out_w), dtype=np.uint8)
    uv[:, 0::2] = u
    uv[:, 1::2] = v
    return np.concatenate([y.reshape(-1), uv.reshape(-1)])


def bgr_to_packed_nv12_fast(bgr, out_w, out_h):
    """BGR→packed NV12（cv2 版：快 ~30x，色度为标准 BT.601）。

    ⚠️ 与 numpy 版 bgr_to_packed_nv12 的色度**不同**（numpy 版把 2x2 求和当平均，
    色度偏差被放大 4 倍）。切换会改变模型输入，请上机 A/B 验证后再定。
    """
    if bgr.shape[1] != out_w or bgr.shape[0] != out_h:
        raise RuntimeError("分辨率不符：需 %dx%d" % (out_w, out_h))
    import cv2
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420).reshape(-1)
    n = out_w * out_h
    uv = np.empty(n // 2, dtype=np.uint8)
    uv[0::2] = yuv[n:n + n // 4]           # U 平面 → 交织
    uv[1::2] = yuv[n + n // 4:n + n // 2]  # V 平面 → 交织
    return np.concatenate([yuv[:n], uv])


def _nms(boxes, scores, iou_th):
    keep = []
    order = scores.argsort()[::-1]
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        x1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        y1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        x2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        y2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        a1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a2 = ((boxes[order[1:], 2] - boxes[order[1:], 0]) *
              (boxes[order[1:], 3] - boxes[order[1:], 1]))
        iou = inter / (a1 + a2 - inter + 1e-9)
        order = order[1:][iou <= iou_th]
    return keep


def decode_yolov8(output, labels, frame_w, frame_h,
                  conf=None, iou=None, input_w=None, input_h=None):
    """YOLOv8 多类别解码：输出 [1,4+nc,N] 或 [1,N,4+nc] → Det 列表（按分降序）。
    labels：类别名列表，顺序须与训练/导出一致（决定 kind）。"""
    conf = conf if conf is not None else S.vision.model.score_threshold
    iou = iou if iou is not None else S.vision.model.nms_threshold
    input_w = input_w or S.vision.model.input_size
    input_h = input_h or S.vision.model.input_size
    arr = np.asarray(output)
    if arr.ndim == 3:
        arr = arr[0]
        if arr.shape[0] <= arr.shape[1]:   # 特征(4+nc) ≤ 锚点数 → 转置为 [N,4+nc]
            arr = arr.T
    else:
        arr = arr.reshape(-1, arr.shape[-1])
    nc = len(labels)
    boxes_xywh = arr[:, :4]
    cls = arr[:, 4:4 + nc].argmax(axis=1)
    score = arr[:, 4:4 + nc].max(axis=1)
    m = score >= conf
    if not m.any():
        return []
    boxes_xywh, cls, score = boxes_xywh[m], cls[m], score[m]
    sx, sy = frame_w / input_w, frame_h / input_h
    x1 = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2) * sx
    y1 = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2) * sy
    x2 = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2) * sx
    y2 = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2) * sy
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    # 按类别分别 NMS（不同类别可重叠，如红球/蓝球并排）
    keep = []
    for c in range(nc):
        idx = np.where(cls == c)[0]
        if idx.size:
            keep.extend(idx[_nms(boxes[idx], score[idx], iou)])
    out = []
    for i in keep:
        kind = labels[int(cls[i])] if int(cls[i]) < nc else "unknown"
        out.append(Det(kind, score[i], x1[i], y1[i],
                       x2[i] - x1[i], y2[i] - y1[i]))
    out.sort(key=lambda d: d.score, reverse=True)
    return out

def decode_yolo11_split(outputs, labels, frame_w, frame_h,
                        conf=None, iou=None, input_w=None, input_h=None,
                        reg_max=16):
    """YOLO11(X5 .bin split-head) 解码。

    X5 编译产物把检测头拆成每尺度两个 tensor：reg 64ch(DFL 4xreg_max, logits) +
    cls nc ch(logits)，concat/softmax/sigmoid/DFL 均未包含，需在此还原。
    outputs: {输出名: ndarray (1,g,g,C)}；返回 Det 列表（按分降序）。
    """
    conf = conf if conf is not None else S.vision.model.score_threshold
    iou = iou if iou is not None else S.vision.model.nms_threshold
    input_w = input_w or S.vision.model.input_size
    input_h = input_h or S.vision.model.input_size
    nc = len(labels)
    bins = np.arange(reg_max, dtype=np.float32)
    grids = {}
    for arr in outputs.values():
        a = np.asarray(arr)
        if a.ndim != 4 or a.shape[0] != 1:
            continue
        grids.setdefault(a.shape[1], {})[a.shape[3]] = a[0]
    allb, alls, allc = [], [], []
    for g, tens in grids.items():
        if 64 not in tens or nc not in tens:
            continue
        stride = float(input_w) / g
        reg = tens[64].reshape(g * g, 4, reg_max)     # (N,4,16) logits
        cls = tens[nc].reshape(g * g, nc)             # (N,nc) logits
        pr = 1.0 / (1.0 + np.exp(-cls.astype(np.float64)))   # sigmoid
        score = pr.max(axis=1)
        idx = np.where(score >= conf)[0]
        if not idx.size:
            continue
        r = reg[idx].astype(np.float64)
        e = np.exp(r - r.max(axis=2, keepdims=True))
        e /= e.sum(axis=2, keepdims=True)
        dist = (e * bins).sum(axis=2)                 # (M,4) ltrb grid 单位
        ys, xs = np.divmod(idx, g)
        cx = (xs.astype(np.float32) + 0.5) * stride
        cy = (ys.astype(np.float32) + 0.5) * stride
        x1 = cx - dist[:, 0] * stride
        y1 = cy - dist[:, 1] * stride
        x2 = cx + dist[:, 2] * stride
        y2 = cy + dist[:, 3] * stride
        allb.append(np.stack([x1, y1, x2, y2], axis=1))
        alls.append(score[idx])
        allc.append(pr[idx].argmax(axis=1))
    if not allb:
        return []
    boxes = np.concatenate(allb)
    scores = np.concatenate(alls)
    clss = np.concatenate(allc)
    sx, sy = frame_w / input_w, frame_h / input_h
    x1 = np.clip(boxes[:, 0] * sx, 0, frame_w)
    y1 = np.clip(boxes[:, 1] * sy, 0, frame_h)
    x2 = np.clip(boxes[:, 2] * sx, 0, frame_w)
    y2 = np.clip(boxes[:, 3] * sy, 0, frame_h)
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    keep = []
    for c in range(nc):
        b = np.where(clss == c)[0]
        if b.size:
            keep.extend(b[_nms(boxes[b], scores[b], iou)])
    out = []
    for i in keep:
        kind = labels[int(clss[i])] if int(clss[i]) < nc else "unknown"
        out.append(Det(kind, scores[i], x1[i], y1[i],
                       x2[i] - x1[i], y2[i] - y1[i]))
    out.sort(key=lambda d: d.score, reverse=True)
    return out


# ---------------------------------------------------------------------------
# X5 模型文件校验
# ---------------------------------------------------------------------------
def check_x5_model(path):
    """RDK X5 只认 OE 工具链 .bin；.hbm（车载 Nash）拒绝并给正确链路提示。"""
    if str(path).lower().endswith(".hbm"):
        if S.get("vision.model.strict_x5_bin", True):
            raise RuntimeError(
                ".hbm 是车载 J5/J6/S100(Nash) 格式，RDK X5(Bayes-e) 无法加载！"
                "请用同一份 ONNX 走 X5 OpenExplorer 工具链重编："
                "hb_mapper makertbin --model-type onnx ... (march: bayes-e) → .bin")
    if not os.path.exists(path):
        raise FileNotFoundError("权重不存在: %s（先放好 X5 工具链产出的 .bin）" % path)


# ---------------------------------------------------------------------------
# 真机后端（单权重多类别）
# ---------------------------------------------------------------------------
class _RealBase(DetectorBase):
    def __init__(self):
        path = S.vision.model.path
        check_x5_model(path)             # .hbm 拒绝 / 缺文件报错
        self.labels = list(S.vision.model.labels)
        self._pre = ModelPreprocessor(
            calib_path=S.vision.camera.front.calibration)
        self._iw = self._ih = self._pre.size
        print("[DET] 单权重模型 %s" % path)

    def detect(self, frame):
        return self._run(frame)

    def _run(self, frame):
        h, w = frame.shape[:2]
        square = self._pre.process(frame)
        return self._infer(square, h, w)


class HbmRuntimeDetector(_RealBase):
    """hbm_runtime（RDK X5 OS>=3.5.0）。"""

    def __init__(self):
        super().__init__()
        import hbm_runtime
        self._model = hbm_runtime.HB_HBMRuntime(S.vision.model.path)
        self._key = self._find_input(self._model)

    @staticmethod
    def _find_input(model):
        if S.vision.model.input_name:
            return S.vision.model.input_name
        for attr in ("input_names", "input_name"):
            v = getattr(model, attr, None)
            if isinstance(v, dict):
                # hbm_runtime: input_names -> {model_name: [input...]}
                for val in v.values():
                    if isinstance(val, (list, tuple)) and val:
                        return str(val[0])
                    if isinstance(val, str) and val:
                        return val
            elif isinstance(v, (list, tuple)) and v:
                return str(v[0])
            elif isinstance(v, str) and v:
                return v
        return "input"

    def _infer(self, square, h, w):
        if S.get("vision.model.fast_nv12", False):
            nv12 = bgr_to_packed_nv12_fast(square, self._iw, self._ih)
        else:
            nv12 = bgr_to_packed_nv12(square, self._iw, self._ih)
        outputs = self._model.run({self._key: nv12})
        # run() -> {模型名: {输出名: ndarray}}；单模型时解一层
        if isinstance(outputs, dict) and len(outputs) == 1:
            outputs = next(iter(outputs.values()))
        if isinstance(outputs, dict):            # split-head：每尺度 reg64 + cls nc
            return decode_yolo11_split(outputs, self.labels, w, h,
                                       input_w=self._iw, input_h=self._ih)
        arr = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        return decode_yolov8(arr, self.labels, w, h,
                             input_w=self._iw, input_h=self._ih)


class PyeasyDnnDetector(_RealBase):
    """pyeasy_dnn（X5 legacy）。"""

    def __init__(self):
        super().__init__()
        from hobot_dnn import pyeasy_dnn as dnn
        self._models = dnn.load(S.vision.model.path)

    def _infer(self, square, h, w):
        nv12 = bgr_to_packed_nv12(square, self._iw, self._ih)
        m = self._models[0]
        try:
            outputs = m.forward([nv12])
        except TypeError:
            outputs = m.forward(nv12)
        arr = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        return decode_yolov8(arr, self.labels, w, h,
                             input_w=self._iw, input_h=self._ih)


class OnnxDetector(_RealBase):
    """onnxruntime（CPU 真机闭环调试）。"""

    def __init__(self):
        super().__init__()
        import onnxruntime
        self._sess = onnxruntime.InferenceSession(S.vision.model.path)
        self._iname = self._sess.get_inputs()[0].name
        shape = self._sess.get_inputs()[0].shape
        if len(shape) == 4 and shape[2] and shape[3]:
            self._ih, self._iw = int(shape[2]), int(shape[3])

    def _infer(self, square, h, w):
        if (self._pre.size, self._pre.size) != (self._iw, self._ih):
            import cv2
            square = cv2.resize(square, (self._iw, self._ih),
                                interpolation=cv2.INTER_LINEAR)
        rgb = square[..., ::-1].astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[None]
        out = self._sess.run(None, {self._iname: blob})[0]
        return decode_yolov8(out, self.labels, w, h,
                             input_w=self._iw, input_h=self._ih)


# ---------------------------------------------------------------------------
# Mock（虚拟测试：按任务模拟单一类别序列）
# ---------------------------------------------------------------------------
class MockDetector(DetectorBase):
    """按 vision.yaml sim.ball 序列模拟 远→近→越球消失。"""

    def __init__(self, kind):
        self.kind = kind
        self._f = 0

    def _ratio(self):
        k = self._f - S.vision.sim.ball.appear_frame
        if k < 0:
            return None
        r = S.vision.sim.ball.ratio_start + S.vision.sim.ball.ratio_step * k
        if r >= S.vision.sim.ball.hit_ratio and \
                r >= S.vision.sim.ball.hit_ratio + \
                S.vision.sim.ball.ratio_step * 1.5:
            return None
        return min(r, S.vision.sim.ball.hit_ratio)

    def _side(self, r_des, w, h, cx, cy):
        r_des = min(r_des, 0.98)
        for s in range(1, int((w + h) * 1.5) + 2):
            x0 = max(0, cx - s // 2)
            x1 = min(w, cx + (s + 1) // 2)
            y0 = max(0, cy - s // 2)
            y1 = min(h, cy + (s + 1) // 2)
            if (x1 - x0) * (y1 - y0) >= r_des * w * h:
                return s
        return max(w, h)

    def detect(self, frame):
        self._f += 1
        h, w = frame.shape[:2]
        r = self._ratio()
        if r is None:
            return []
        cx = int(w * (0.5 + S.vision.sim.ball.center_jitter *
                      ((self._f % 7) - 3) / 3.0))
        cy = int(h * 0.45)
        s = self._side(r, w, h, cx, cy)
        x0 = max(0, cx - s // 2)
        x1 = min(w, cx + (s + 1) // 2)
        y0 = max(0, cy - s // 2)
        y1 = min(h, cy + (s + 1) // 2)
        if x1 <= x0:
            x1 = min(w, x0 + 1)
        if y1 <= y0:
            y1 = min(h, y0 + 1)
        return [Det(self.kind, 0.9, x0, y0, x1 - x0, y1 - y0)]


# ---------------------------------------------------------------------------
# 指令优先：按任务/目标类别挑选（绝不因“别的类分更高”换目标）
# ---------------------------------------------------------------------------
def pick_target(dets, want):
    """在 dets 中选 kind==want 且置信度最高的 Det；want 缺失或无目标返回 None。"""
    best = None
    for d in dets:
        if d.kind == want and (best is None or d.score > best.score):
            best = d
    return best


# ---------------------------------------------------------------------------
# 工厂：单权重共享后端；按任务取类别
# ---------------------------------------------------------------------------
def _want_label(task):
    """任务 → 所需类别名（labels 里没有则返回 None）。"""
    labels = list(S.vision.model.labels)
    if task == "ball":
        tgt = S.vision.mission.target_color
        name = {"red": "red_ball", "blue": "blue_ball"}.get(tgt, tgt)
        return name if name in labels else None
    if task == "gate":
        return "gate" if "gate" in labels else None
    return None


class DetectorHub(object):
    _REAL = {"hbm_runtime": HbmRuntimeDetector,
             "pyeasy_dnn": PyeasyDnnDetector,
             "onnx": OnnxDetector}

    def __init__(self):
        self.mode = S.vision.model.mode
        self._mocks = {}
        self._real = None
        self._extra = {}            # 装配层注册的任务专用后端（如 gate keypoint/mock）
        self._extra_cache = {}      # {task: (frame, [Det])} 同帧只跑一次
        self._cache = []            # 最近一帧全量检测（供画框，避免二次推理）
        self._cache_frame = None
        if self.mode == "mock":
            for task in ("ball", "gate"):
                if task == "gate":
                    # gate 走新 keypoint 前端：由装配层注册 Mock/真实后端，见 main/register
                    continue
                if task in S.comm.tasks.enabled and _want_label(task):
                    self._mocks[task] = MockDetector(_want_label(task))
        else:
            try:
                self._real = self._REAL[self.mode]()
            except Exception as e:
                print("[DET] 单权重模型未就绪(%s)：%s" % (self.mode, e))

    def _gather(self, frame):
        """单帧全量检测（带缓存：同一帧对象只推理一次）。"""
        if self._cache_frame is frame:
            return self._cache
        if self.mode == "mock":
            dets = []
            for m in self._mocks.values():
                dets.extend(m.detect(frame))
        else:
            dets = self._real.detect(frame)
        self._cache, self._cache_frame = list(dets), frame
        return self._cache

    def detect_all(self, frame):
        """整帧检测列表（调试/画框用）。"""
        return list(self._gather(frame))

    # ------------------------------------------------------------ 任务专用后端
    def register(self, task, backend):
        """装配层注册任务专用后端（如 gate keypoint / gate mock）。

        backend 需提供 detect(frame) -> [Det]；传 None 表示该任务无后端。"""
        self._extra[task] = backend

    def has_extra(self, task):
        return task in self._extra and self._extra[task] is not None

    def detect_list(self, task, frame):
        """任务专用后端整帧检测列表；未注册时退化为 legacy 单模型全量。"""
        if not self.has_extra(task):
            return self.detect_all(frame)
        cached = self._extra_cache.get(task)
        if cached is not None and cached[0] is frame:
            return list(cached[1])
        dets = list(self._extra[task].detect(frame))
        self._extra_cache[task] = (frame, dets)
        return dets

    def ready(self, task):
        if task in self._extra:
            return self._extra[task] is not None
        if self.mode == "mock":
            return task in self._mocks
        return self._real is not None and _want_label(task) is not None

    def detect(self, task, frame):
        """指令优先：只返回任务所需类别（want）中最高分的 Det。

        例如撞球按 mission.target_color 打蓝球：画面里红球分再高也不选，
        蓝球缺席则返回 None（继续搜索），绝不自动换目标。"""
        want = _want_label(task)
        if want is None:
            return None
        try:
            dets = self._gather(frame)
        except Exception as e:
            if S.DEBUG:
                print("[DET] 推理异常：%s" % e)
            return None
        return pick_target(dets, want)
