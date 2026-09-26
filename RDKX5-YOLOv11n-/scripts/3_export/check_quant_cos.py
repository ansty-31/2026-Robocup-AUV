#!/usr/bin/env python3
"""量化一致性校验（余弦相似度）—— 在 OpenExplorer 容器内运行。

A. 输出层实测：同一条输入分别喂 `<prefix>_original_float_model.onnx`（float）与
   `<prefix>_calibrated_model.onnx`（量化模拟），逐输出张量算余弦相似度。
B. 逐节点自报：读 `<prefix>_quant_info.json`，取 OE 校准时记的每个节点
   `cosine_similarity`，给出最小值与最差节点（定位量化坏在哪一层）。

判定口径：A 的最小值 >= --thresh（默认 0.95）视为通过；B 作定位用。

用法（容器内，项目根挂成 /data；--prefix 取配置里的 output_model_file_prefix）：
    python3 /data/scripts/3_export/check_quant_cos.py \
        --prefix gate_kpt_bayese_640x640_nv12 \
        --out-dir /data/output --cal-dir /data/calibration_data --n 20 --thresh 0.95

退出码：0=通过，1=不通过。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def node_stats(quant_info: Path) -> tuple[list[tuple[str, float]], int]:
    d = json.loads(quant_info.read_text())
    rows = []
    for k, v in d.items():
        if isinstance(v, dict) and "cosine_similarity" in v:
            try:
                rows.append((k, float(v["cosine_similarity"])))
            except (TypeError, ValueError):
                pass
    return rows, len(d)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", required=True, help="output_model_file_prefix")
    ap.add_argument("--out-dir", default="/data/output")
    ap.add_argument("--cal-dir", default="/data/calibration_data")
    ap.add_argument("--n", type=int, default=20, help="抽几张校准图算输出层余弦")
    ap.add_argument("--thresh", type=float, default=0.95)
    a = ap.parse_args()

    from horizon_tc_ui import HB_ONNXRuntime

    out = Path(a.out_dir)
    f_onnx = out / f"{a.prefix}_original_float_model.onnx"
    c_onnx = out / f"{a.prefix}_calibrated_model.onnx"
    q_info = out / f"{a.prefix}_quant_info.json"
    for p in (f_onnx, c_onnx, q_info):
        if not p.exists():
            print(f"❌ 缺文件: {p}")
            return 1

    cal = sorted(Path(a.cal_dir).glob("*.rgb"))
    if not cal:
        print(f"❌ {a.cal_dir} 里没有 .rgb")
        return 1
    step = max(1, len(cal) // a.n)
    picks = cal[::step][:a.n]

    print("=" * 72)
    print("A. 输出层余弦（float vs 量化模拟 calibrated）")
    print("=" * 72)
    print(f"float : {f_onnx}")
    print(f"quant : {c_onnx}")
    print(f"校准图: {len(cal)} 张，抽 {len(picks)} 张\n")

    fs, cs = HB_ONNXRuntime(model_file=str(f_onnx)), HB_ONNXRuntime(model_file=str(c_onnx))
    in_name = fs.input_names[0]
    if len(fs.output_names) != len(cs.output_names):
        print("❌ 两模型输出个数不一致")
        return 1
    n_out = len(fs.output_names)
    cos_t = [[] for _ in range(n_out)]
    mad_t = [[] for _ in range(n_out)]
    for i, p in enumerate(picks):
        x = np.fromfile(str(p), dtype=np.float32).reshape(1, 3, 640, 640)
        fo, co = fs.run(None, {in_name: x}), cs.run(None, {in_name: x})
        for k in range(n_out):
            cos_t[k].append(cosine(fo[k], co[k]))
            mad_t[k].append(float(np.abs(np.asarray(fo[k], np.float64)
                                         - np.asarray(co[k], np.float64)).mean()))
        if (i + 1) % 5 == 0:
            print(f"  已算 {i + 1}/{len(picks)}")

    print(f"\n{'输出':<5}{'形状':<20}{'余弦中位':>11}{'余弦最小':>11}{'平均绝对差':>12}")
    print("-" * 60)
    mins = []
    for k in range(n_out):
        c, m = np.array(cos_t[k]), np.array(mad_t[k])
        mins.append(float(c.min()))
        print(f"{k:<5}{str(tuple(fo[k].shape)):<20}{np.median(c):>11.6f}{c.min():>11.6f}{m.mean():>12.4f}")
    worst = min(mins)
    print("-" * 60)
    print(f"输出层余弦最小值 = {worst:.6f}（阈值 {a.thresh}）")

    print()
    print("=" * 72)
    print("B. 逐节点余弦（OE 校准时自报，定位用）")
    print("=" * 72)
    rows, n_all = node_stats(q_info)
    if rows:
        c = np.array([v for _, v in rows])
        low = sorted(rows, key=lambda x: x[1])[:8]
        print(f"quant_info 节点 {n_all} 个，其中 {len(rows)} 个带余弦")
        print(f"  最小 {c.min():.6f} | p1 {np.percentile(c, 1):.6f} | "
              f"中位 {np.median(c):.6f} | 低于阈值 {(c < a.thresh).sum()} 个")
        print("  最差 8 个：")
        for k, v in low:
            print(f"    {v:.6f}  {k}")
    print()
    if worst >= a.thresh:
        print(f"✅ 通过：输出层余弦 >= {a.thresh}（量化未破坏模型输出）")
        return 0
    print(f"❌ 不通过：输出层最小余弦 {worst:.6f} < {a.thresh} —— 需调校准集/量化参数后重跑")
    return 1


if __name__ == "__main__":
    sys.exit(main())
