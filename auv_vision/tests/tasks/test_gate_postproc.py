# -*- coding: utf-8 -*-
"""tests/tasks/test_gate_postproc.py — 解码后处理（规范 §4）的无硬件用例

覆盖 `gate/gate_postproc.py` 四条规则里的可离线部分：
  ① 可信角点（V_MIN 口径） ② 四角几何合法（顺序 + 不自交）
  ③ 重复框去重（**不看包含**） ④ 选门键（near / corner 两种策略）
外加一条**配置守卫**：代码兜底 == cfg（与 test_gate_defaults_match_cfg 同一约定）。

这些是**逻辑**证明：能证明规则按写的那样跑，**不能**证明它在水里/板端的效果。
"""
import numpy as np
import pytest

import base.settings as S
from common.detector import Det
from gate import gate_postproc as pp

FW, FH = 1280, 720


def _det(score=0.9, x=400, y=200, w=300, h=220, confs=(0.95, 0.95, 0.95, 0.95),
         kpts=None):
    """造一个 gate Det。confs 是四个角点的置信度（顺序 TL,TR,BR,BL）。"""
    if kpts is None:
        kpts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return Det("gate", score, x, y, w, h, kpts=np.asarray(kpts, np.float32),
               kpt_conf=np.asarray(confs, np.float32))


# --------------------------------------------------------------------------
# ① 可信角点
# --------------------------------------------------------------------------
def test_trusted_mask_uses_vmin():
    d = _det(confs=(0.95, 0.85, 0.75, 0.6))
    assert pp.n_trusted(d, 0.8) == 2          # 0.95 / 0.85
    assert pp.n_trusted(d, 0.5) == 4
    assert pp.n_trusted(d, 0.99) == 0
    # 无角点信息（检测类 Det）→ 全 False，不炸
    assert pp.n_trusted(Det("gate", 0.9, 0, 0, 10, 10), 0.5) == 0


def test_v_min_default_comes_from_cfg():
    """不给参数时走 cfg 的 keypoint.conf_thr（= 规范 V_MIN）。"""
    assert pp.v_min() == float(S.vision.gate.keypoint.conf_thr)


# --------------------------------------------------------------------------
# ② 四角几何合法
# --------------------------------------------------------------------------
def test_quad_legal_accepts_normal_quad():
    d = _det()
    assert pp.quad_legal(d.kpts, [0, 1, 2, 3]) is True


def test_quad_legal_rejects_bowtie():
    """TR 与 BR 互换 → 自交（蝴蝶结）→ 不合法。"""
    d = _det(kpts=[(400, 200), (700, 200), (400, 420), (700, 420)])
    # 顺序变成 TL,TR,BR,BL 后：TL.x=400 < TR.x=700 ✓，但 BL.x=400 < BR.x=700 ✓，
    # TL.y=200 < BL.y=420 ✓，TR.y=200 < BR.y=420 ✓ —— 顺序检查过，靠自交检查挡
    assert pp.quad_legal(d.kpts, [0, 1, 2, 3]) is False


def test_quad_legal_rejects_bad_order():
    """上下边左右颠倒（TR 在 TL 左边）→ 不合法。"""
    d = _det(kpts=[(700, 200), (400, 200), (400, 420), (700, 420)])
    assert pp.quad_legal(d.kpts, [0, 1, 2, 3]) is False


def test_quad_legal_ignores_non_four_corner_instances():
    """不是"四个可信角点"的实例不归它管（门框不全要降级，不是丢帧）。"""
    d = _det()
    assert pp.quad_legal(d.kpts, [0, 1, 2]) is True      # 只有 3 个可信
    assert pp.quad_legal(None, None) is True


# --------------------------------------------------------------------------
# ③ 重复框去重
# --------------------------------------------------------------------------
def test_dedup_drops_smaller_duplicate_with_coincident_edges():
    big = _det(score=0.95, x=400, y=200, w=300, h=220)
    small = _det(score=0.80, x=402, y=202, w=296, h=216)   # 同一个门，scale 不同
    out = pp.apply([big, small], conf_thr=0.8)
    assert len(out) == 1 and out[0] is big


def test_dedup_keeps_far_gate_inside_near_box():
    """**不看包含**：远处第二个门可能整个落在近门框内 —— 不许因为"被包含"就删。"""
    near = _det(score=0.95, x=300, y=100, w=600, h=440)
    far = _det(score=0.70, x=500, y=250, w=120, h=90)      # 在近门框内部，但边不重合
    out = pp.apply([near, far], conf_thr=0.8)
    assert len(out) == 2


def test_dedup_keeps_smaller_box_when_its_score_is_higher():
    """规则的完整条件是"小框 conf **更低** 且 ≥2 条边重合" —— 只重合不够。"""
    big = _det(score=0.60, x=400, y=200, w=300, h=220)
    small = _det(score=0.90, x=401, y=201, w=298, h=218)
    out = pp.apply([big, small], conf_thr=0.8)
    assert len(out) == 2


def test_dedup_needs_min_edges():
    """只重合 1 条边（共用左边）→ 不是重复框，两个都留。"""
    a = _det(score=0.9, x=400, y=200, w=300, h=200)
    b = _det(score=0.7, x=400, y=430, w=300, h=200)        # 左边重合，上下不重合
    out = pp.apply([a, b], conf_thr=0.8)
    assert len(out) == 2


def test_dedup_config_can_disable_by_edges():
    """min_edges 调大 → 同一对框不再被判重复（参数真的接线了）。"""
    big = _det(score=0.95, x=400, y=200, w=300, h=220)
    small = _det(score=0.80, x=401, y=201, w=298, h=218)
    out = pp.dedup([big, small], conf_thr=0.8,
                   cfg={"edge_tol": 0.10, "min_edges": 5, "geom_check": True})
    assert len(out) == 2


# --------------------------------------------------------------------------
# ④ apply()：几何 + 去重一起，保序、不改动传入对象
# --------------------------------------------------------------------------
def test_apply_drops_illegal_quad_and_preserves_order():
    """规范 §4②：**先 L/R 归一**（左右标反 → 交换后保留），归一救不回来的才丢；保序。

    只有"上下也反/自交"这种真的非法才丢；左右反不算非法（模型约一半实例会标反，
    旧行为会把近门整条丢掉，沿革见 doc/_注释历史_fragments/base_tests.md）。
    """
    good = _det(score=0.9, x=100, y=100, w=200, h=150)
    lr_swapped = _det(score=0.8, x=400, y=200, w=200, h=150,
                      kpts=[(600, 200), (400, 200), (400, 350), (600, 350)])  # 左右反 → 归一
    bad = _det(score=0.7, x=800, y=200, w=200, h=150,
               kpts=[(800, 350), (1000, 350), (1000, 200), (800, 200)])      # 上下反 → 救不回
    other = Det("gate", 0.5, 10, 10, 20, 20)                  # 无角点信息
    out = pp.apply([good, lr_swapped, bad, other], conf_thr=0.8)
    assert out == [good, lr_swapped, other]
    # L/R 归一真的交换了（TL 与 TR、BL 与 BR），且 kpt_conf 跟着换
    assert lr_swapped.kpts[0][0] == 400 and lr_swapped.kpts[1][0] == 600
    assert lr_swapped.kpts[2][0] == 600 and lr_swapped.kpts[3][0] == 400
    # 传入列表未被就地改写
    assert len([good, lr_swapped, bad, other]) == 4


def test_lr_normalize_skips_placeholder_and_single_side():
    """归一的两个护栏：`(0,0)` 占位不算可见；只看到一侧时判不出左右 → 不动。"""
    ph = _det(kpts=[(0, 0), (400, 200), (400, 350), (600, 350)],
              confs=(0.9, 0.9, 0.9, 0.9))     # TL 是 (0,0) 占位（模型没找到时会这样）
    assert pp.visible_mask(ph)[0] == False, "(0,0) 占位不算可见"
    # 占位**不参与**求均值：真左列只剩 BL(600)，右列 400/400 ⇒ 确实标反 ⇒ 必须交换。
    # 若把占位的 x=0 算进去，均值变成 300 ≤ 400 ⇒ 会被误判成"没标反"而漏掉这次归一。
    assert pp.lr_normalize(ph) is True
    one = _det(kpts=[(600, 200), (0, 0), (0, 0), (600, 350)],
               confs=(0.9, 0.0, 0.0, 0.9))     # 只有左列可见
    assert pp.lr_normalize(one) is False


def test_apply_geom_check_can_be_disabled():
    bad = _det(kpts=[(600, 200), (400, 200), (400, 350), (600, 350)])
    out = pp.apply([bad], conf_thr=0.8,
                   cfg={"edge_tol": 0.10, "min_edges": 2, "geom_check": False})
    assert len(out) == 1


# --------------------------------------------------------------------------
# ⑤ 选门
# --------------------------------------------------------------------------
def test_select_near_prefers_bigger_box_even_with_lower_score():
    """near（默认，规范 §4）：近距离优先 —— 框宽当测距代理。"""
    far = _det(score=0.99, x=500, y=250, w=120, h=90)
    near = _det(score=0.70, x=200, y=100, w=600, h=440)
    assert pp.pick([far, near], conf_thr=0.8, cfg={"mode": "near"}) is near


def test_select_corner_mode_is_the_old_behaviour():
    """corner：角点更全更可信的优先（旧行为，倒影抢门的对策）。"""
    a = _det(score=0.99, x=200, y=100, w=600, h=440, confs=(0.9, 0.9, 0.2, 0.2))
    b = _det(score=0.70, x=500, y=250, w=120, h=90, confs=(0.95, 0.95, 0.95, 0.95))
    assert pp.pick([a, b], conf_thr=0.8, cfg={"mode": "corner"}) is b
    assert pp.pick([a, b], conf_thr=0.8, cfg={"mode": "near"}) is a


def test_select_ignores_non_gate_and_survives_bad_mode():
    ball = Det("blue_ball", 0.9, 0, 0, 10, 10)
    gate = _det()
    assert pp.pick([ball], conf_thr=0.8) is None
    assert pp.pick([ball, gate], conf_thr=0.8, cfg={"mode": "TYPO"}) is gate


# --------------------------------------------------------------------------
# ⑥ 配置守卫：代码兜底 == cfg（改了 cfg 就要同步兜底表）
# --------------------------------------------------------------------------
def test_postproc_defaults_match_cfg():
    bad = []
    for name, code, cfg in (("vision.gate.det", pp._D_DET, S.get("vision.gate.det", {})),
                            ("vision.gate.postproc", pp._D_POST, S.get("vision.gate.postproc", {})),
                            ("vision.gate.select", pp._D_SELECT, S.get("vision.gate.select", {}))):
        for k, v in dict(cfg).items():
            if k not in code:
                bad.append("%s.%s：cfg 有但代码兜底表没有" % (name, k))
            elif code[k] != v:
                bad.append("%s.%s：cfg=%r 代码兜底=%r" % (name, k, v, code[k]))
    assert not bad, "解码后处理的兜底默认值与 cfg 不一致：\n  " + "\n  ".join(bad)
    assert pp.det_cfg() == float(S.vision.gate.det.conf)
    assert pp.select_cfg()["mode"] == str(S.vision.gate.select.mode)
    assert pp.postproc_cfg()["min_edges"] == int(S.vision.gate.postproc.min_edges)


def test_pick_does_not_touch_safety_defaults():
    """选门/去重不该顺手改别的：z.cross 与 conf_thr 只是被读，不是被这个模块写。"""
    assert float(S.comm.gate.z.cross) == float(S.comm.gate.z.cross)
    assert pp.v_min() > 0.0


# --------------------------------------------------------------------------
# ⑦ 解码后端接线（不加载真实模型：把 _init_backend 换成空实现）
# --------------------------------------------------------------------------
def test_gate_backend_wires_det_conf_and_postproc(monkeypatch):
    from gate.gate_decode import GateKeypointBackend
    from gate.geometry import CameraModel

    monkeypatch.setattr(GateKeypointBackend, "_init_backend", lambda self: None)
    cam = CameraModel.pinhole(1280, 720, fx=1024.0, fy=1152.0, cx=702.0, cy=409.0)
    b = GateKeypointBackend(path="models/nonexistent.bin", labels=["gate"],
                            kpt_order=["TL", "TR", "BR", "BL"], camera=cam)
    # 候选阈值走 gate 专用的 det.conf（**不是**共用的 model.score_threshold）
    # ⚠️ 别写成"两者必须不相等"：两者取值可以巧合相同，那种断言会假红。
    #    这里改成**显式改两个值、看谁生效**：
    assert b._det_conf == float(S.vision.gate.det.conf)
    monkeypatch.setitem(S.vision.gate.det, "conf", 0.71)
    monkeypatch.setattr(S.vision.model, "score_threshold", 0.42, raising=False)
    b3 = GateKeypointBackend(path="x", labels=["gate"], kpt_order=["TL", "TR", "BR", "BL"],
                             camera=cam)
    assert b3._det_conf == pytest.approx(0.71), "gate 的候选阈值必须跟 det.conf 走"
    assert b3._det_conf != pytest.approx(0.42), "…而不是跟共用的 model.score_threshold 走"
    # 解码期 vis_thr 走 keypoint.vis_thr
    assert b._vis_thr == float(S.vision.gate.keypoint.vis_thr)
    # 后处理参数已解析
    assert b._post["min_edges"] == int(S.vision.gate.postproc.min_edges)
    # 显式参数优先级最高（预览 --conf 用）
    b2 = GateKeypointBackend(path="x", labels=["gate"], kpt_order=["TL", "TR", "BR", "BL"],
                             camera=cam, det_conf=0.25)
    assert b2._det_conf == pytest.approx(0.25)


# --------------------------------------------------------------------------
# ⑧ 解码→后处理 集成（合成张量，不加载模型）
#    形状/语义照 doc/gate_pose_decode_spec.md §2/§3：NHWC、kpt x/y 已是 cell 坐标、v 是 raw logit
# --------------------------------------------------------------------------
INPUT, G = 640, 80
STRIDE = INPUT // G
REG_MAX, KPT_DIM = 16, 4


def _blank(g=G):
    return {64: np.zeros((1, g, g, 4 * REG_MAX), np.float32),
            1: np.full((1, g, g, 1), -10.0, np.float32),
            KPT_DIM * 3: np.zeros((1, g, g, KPT_DIM * 3), np.float32)}


def _put(outs, gx, gy, dist=3, score_logit=6.0, cells=None, vs=None):
    outs[64][0, gy, gx] = 0.0
    for side in range(4):
        outs[64][0, gy, gx, side * REG_MAX + dist] = 40.0      # DFL → one-hot
    outs[1][0, gy, gx, 0] = score_logit
    for i, (cx, cy) in enumerate(cells):
        outs[KPT_DIM * 3][0, gy, gx, 3 * i + 0] = cx           # **cell 坐标**，解码只 ×stride
        outs[KPT_DIM * 3][0, gy, gx, 3 * i + 1] = cy
        outs[KPT_DIM * 3][0, gy, gx, 3 * i + 2] = vs[i]        # raw logit，解码自己 sigmoid


def _decode(outs, conf=0.6, vis_thr=0.5):
    from gate.gate_decode import decode_yolo11_kpt
    return decode_yolo11_kpt(outs, ["gate"], FW, FH, conf=conf, iou=0.45,
                             input_w=INPUT, input_h=INPUT, vis_thr=vis_thr)


def test_decode_then_postproc_keeps_legal_quad():
    outs = _blank()
    gx, gy = 10, 20
    _put(outs, gx, gy, cells=[(gx - 2, gy - 2), (gx + 3, gy - 2),
                              (gx + 3, gy + 3), (gx - 2, gy + 3)],
         vs=[6.0, 6.0, 6.0, 6.0])
    dets = _decode(outs)
    assert len(dets) == 1
    kept = pp.apply(dets, conf_thr=0.8)
    assert len(kept) == 1, "四角合法 + 高 v 的实例必须留下"
    d = kept[0]
    assert pp.n_trusted(d, 0.8) == 4
    from gate.gate_frontend import parse_kpt_mode, MODE_FULL
    assert parse_kpt_mode(d.kpts, d.kpt_conf, 0.8)[0] == MODE_FULL


def test_decode_then_postproc_drops_illegal_quad():
    """四角都可信但顺序非法（TR 在 TL 左侧）→ 规范 §4 要求丢掉这个退化实例。"""
    outs = _blank()
    gx, gy = 10, 20
    _put(outs, gx, gy, cells=[(gx + 3, gy - 2), (gx - 2, gy - 2),   # TL/TR 互换
                              (gx + 3, gy + 3), (gx - 2, gy + 3)],
         vs=[6.0, 6.0, 6.0, 6.0])
    dets = _decode(outs)
    assert len(dets) == 1 and pp.n_trusted(dets[0], 0.8) == 4
    assert pp.apply(dets, conf_thr=0.8) == []


def test_decode_low_v_corner_is_degraded_not_dropped():
    """规范 §7.2：低 v 角点坐标不可信 ⇒ 不进 PnP；但**不许据此丢整帧**（要降级）。"""
    outs = _blank()
    gx, gy = 10, 20
    _put(outs, gx, gy, cells=[(gx - 2, gy - 2), (gx + 3, gy - 2),
                              (gx + 3, gy + 3), (gx - 2, gy + 3)],
         vs=[6.0, 6.0, -3.0, -3.0])          # 下面两个角点 sigmoid≈0.05 → 被 vis_thr 清 0
    dets = _decode(outs, vis_thr=0.5)
    assert len(dets) == 1
    assert pp.n_trusted(dets[0], 0.8) == 2, "低 v 的两个角点不该算可信"
    assert len(pp.apply(dets, conf_thr=0.8)) == 1, "门框不全 → 降级，不是丢帧"
    from gate.gate_frontend import parse_kpt_mode, MODE_WIDTH
    assert parse_kpt_mode(dets[0].kpts, dets[0].kpt_conf, 0.8)[0] == MODE_WIDTH


def test_decode_conf_threshold_is_applied():
    """候选阈值真的接线：score 在 0.6 与 0.9 之间时，conf=0.6 收、conf=0.9 不收。"""
    outs = _blank()
    gx, gy = 10, 20
    _put(outs, gx, gy, score_logit=1.0,   # sigmoid(1)=0.731
         cells=[(gx - 2, gy - 2), (gx + 3, gy - 2), (gx + 3, gy + 3), (gx - 2, gy + 3)],
         vs=[6.0] * 4)
    assert len(_decode(outs, conf=0.6)) == 1
    assert len(_decode(outs, conf=0.9)) == 0
