#!/usr/bin/env python3
"""
修改 ultralytics 输出头，适配 RDK X5 BPU（NHWC 分裂头）

    --task detect  Detect.forward → 6 个张量：3×bbox(C=4*reg_max) + 3×cls(C=nc)
    --task pose    Pose.forward   → 9 个张量：每尺度 bbox(C=64) + cls(C=nc) + kpt(C=12)
    --restore      从 head.py.backup 恢复原版（训练/推理必须用原版）

pose 约定（与板端 gate/gate_decode.py 的 decode_yolo11_kpt 对齐）：
    * 每个尺度 3 个张量，shape=[1, g, g, C]（NHWC）；C = 64(reg, DFL raw) / nc(cls, raw logits) / 12(kpt)
    * kpt 通道顺序 [x,y,v] × kpt_dim（本仓库 gate 为 4 点：TL,TR,BR,BL）
    * x,y = (raw*2 - 0.5 + 网格索引)，单位=cell（板端 ×stride 得输入像素，再 ×frame/640 得原图）
    * v   = raw logit（板端自行 sigmoid）

用法：
    python scripts/3_export/modify_ultralytics.py --task detect     # 导出检测模型前
    python scripts/3_export/modify_ultralytics.py --task pose       # 导出门关键点模型前
    python scripts/3_export/modify_ultralytics.py --restore         # 训练前/手动推理前
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # scripts/<stage>/x.py

MARKERS = {"detect": "*bboxes, *clses", "pose": "POSE_9OUT_MARKER"}

DETECT_FORWARD = '''    def forward(self, x):
        """【补丁】拆分 bbox/cls 为 6 个 NHWC 输出（RDK X5 BPU）
        cv2: bbox 分支 4*reg_max 通道；cv3: cls 分支 nc 通道
        """
        if self.end2end:
            return self.forward_end2end(x)
        bboxes = [self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous() for i in range(self.nl)]
        clses = [self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous() for i in range(self.nl)]
        return (*bboxes, *clses)
'''

POSE_FORWARD = '''    def forward(self, x):
        """【补丁】Pose 拆分头：每尺度 bbox(64)/cls(nc)/kpt(3*kpt_dim)，NHWC，供 RDK X5 BPU
        POSE_9OUT_MARKER
        约定见 scripts/README.md：kpt x/y = (raw*2 + 网格索引)（cell 单位，×stride 得像素），
        v = raw logit（板端 sigmoid）；与板端 gate/gate_decode.py 的 decode_yolo11_kpt 对齐。

        ⚠️ 网格项是 **+索引**，不是 "+索引-0.5"：ultralytics 的 kpts_decode 用
        `(raw*2 + (anchors - 0.5)) * stride`，而 `make_anchors` 的 anchors 已含
        +0.5（grid_cell_offset=0.5），两项相消后净为 **+索引**。早期版本多写了一个
        -0.5，导致角点系统性偏移 0.5 cell（stride 8/16/32 → 4/8/16 px）。

        实现注意（RDK X5 编译友好，务必保持）：
          1) **不用 5D view + 整数索引**：会导出 rank-5 的 Gather，hbdk 报
             "onnx_gather output rank should no greater than 4"，把模型切成
             7 个子图并留下大量 CPU 节点；改用 4D 上的步长切片（kpt[:, 0::3]
             → Slice）后为**单个 BPU 子图**。
          2) **不要先 permute 到 NHWC 再算**：那样 layout 变换会在 DDR 实体化，
             实测 DDR 18MB→48MB、延迟 6.8ms→11.5ms。保持"NCHW 上算完再 permute"。
          实测（hb_perf）：单 BPU 子图 6.83ms（≈146FPS），仅 9 个很小的
          Concat/Reshape/Transpose 在 CPU。
        """
        nk, ndim = self.kpt_shape
        outputs = []
        for i in range(self.nl):
            bbox = self.cv2[i](x[i])          # [B, 4*reg_max, g, g]
            cls = self.cv3[i](x[i])           # [B, nc, g, g]
            kpt = self.cv4[i](x[i])           # [B, nk*ndim, g, g]（通道序 x1,y1,v1,x2,...）
            bs, _, g, _ = bbox.shape
            idx = torch.arange(g, dtype=bbox.dtype, device=bbox.device)
            gy, gx = torch.meshgrid(idx, idx, indexing="ij")
            gx = gx.view(1, 1, g, g)
            gy = gy.view(1, 1, g, g)
            # 在 NCHW 上按步长取通道（Slice），做完算术再统一 permute：
            # 这样 BPU 只跑一次 6.8ms 的主图；若改成"先 permute 到 NHWC 再算"，
            # layout 变换会在 DDR 里实体化（实测 DDR 18→48MB、延迟 6.8→11.5ms）。
            kx = kpt[:, 0::ndim] * 2.0 + gx                        # [B, nk, g, g]
            ky = kpt[:, 1::ndim] * 2.0 + gy
            kv = kpt[:, 2::ndim]                                   # raw logit
            k = torch.stack((kx, ky, kv), dim=2).reshape(bs, nk * ndim, g, g)
            outputs += [bbox.permute(0, 2, 3, 1).contiguous(),
                        cls.permute(0, 2, 3, 1).contiguous(),
                        k.permute(0, 2, 3, 1).contiguous()]
        return tuple(outputs)
'''

PATCHES = {
    "detect": ("Detect", DETECT_FORWARD, "6 个张量（3×bbox + 3×cls）"),
    "pose": ("Pose", POSE_FORWARD, "9 个张量（3×(bbox + cls + kpt)）"),
}
_W = {"detect": "yolo11n.pt", "pose": "yolo11n-pose.pt"}
WEIGHTS = {k: str(PROJECT_ROOT / "weights" / v) for k, v in _W.items()}


def head_file() -> Path:
    try:
        import ultralytics
    except ImportError:
        sys.exit("❌ 未安装 ultralytics，请先: pip install -r requirements.txt")
    return Path(ultralytics.__file__).resolve().parent / "nn" / "modules" / "head.py"


def restore(head: Path) -> None:
    backup = head.with_name(head.name + ".backup")
    if not backup.exists():
        sys.exit("❌ 未找到 head.py.backup（从未打过补丁？）")
    if head.read_text(encoding="utf-8") == backup.read_text(encoding="utf-8"):
        print("ℹ️  head.py 已是原始版本")
        return
    shutil.copy2(backup, head)
    for pyc in (head.parent / "__pycache__").glob("head*.pyc"):
        pyc.unlink(missing_ok=True)
    print(f"✅ 已恢复原始 head.py（训练/手动推理模式）: {head}")


def patch(head: Path, task: str) -> bool:
    cls_name, new_forward, desc = PATCHES[task]
    backup = head.with_name(head.name + ".backup")
    if not backup.exists():
        shutil.copy2(head, backup)
        print(f"✅ 已备份原文件: {backup}")

    content = head.read_text(encoding="utf-8")
    if MARKERS[task] in content:
        print(f"ℹ️  {task} 补丁已存在，跳过")
        return True

    # 定位 "class <Name> ... def forward(self, x):" 并整体替换其 forward 方法
    pattern = (rf'(class {cls_name}.*?def forward\(self, x\):)'
               rf'(.*?)(?=\n    def |\nclass |\Z)')

    def _replace(m: re.Match) -> str:
        head_part = m.group(1)
        head_part = head_part[:head_part.rfind("\n")]      # 去掉签名行前的缩进残留
        return head_part + "\n" + new_forward

    new_content = re.sub(pattern, _replace, content, flags=re.DOTALL)
    if new_content == content:
        print(f"❌ 未能找到 {cls_name}.forward（ultralytics 版本不兼容？）")
        return False
    head.write_text(new_content, encoding="utf-8")
    print(f"✅ 已写入 {task} 补丁：{cls_name}.forward → {desc}")
    return True


def verify(task: str) -> bool:
    """独立子进程验证（避免同进程模块缓存误判），权重不存在则跳过"""
    weights = WEIGHTS[task]
    if not Path(weights).exists():
        print(f"ℹ️  未找到 {weights}，跳过补丁验证（导出时会再次校验）")
        return True
    expect = 6 if task == "detect" else 9
    code = textwrap.dedent(f'''
        import sys, torch
        from ultralytics import YOLO
        model = YOLO({weights!r})
        model.model.eval()
        with torch.no_grad():
            out = model.model(torch.randn(1, 3, 640, 640))
        if isinstance(out, tuple) and len(out) == {expect}:
            print("✅ 验证通过：输出", len(out), "个张量")
            for i, o in enumerate(out):
                print(f"   Output {{i}}: {{tuple(o.shape)}}")
            sys.exit(0)
        print(f"❌ 验证失败：输出 {{len(out) if isinstance(out, tuple) else type(out)}}（预期 {expect}）")
        sys.exit(1)
    ''')
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True)
    print((proc.stdout or "").rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip())
    return proc.returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="为 RDK X5 导出修改 ultralytics 输出头（detect 6 输出 / pose 9 输出）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--task", choices=["detect", "pose"], default="detect",
                    help="补丁类型（导出前按模型任务选择）")
    ap.add_argument("--restore", action="store_true",
                    help="恢复原始 head.py（训练/手动推理前）")
    a = ap.parse_args()

    head = head_file()
    print(f"📝 head.py: {head}")
    if a.restore:
        restore(head)
        return
    print(f"🔧 打补丁: task={a.task}")
    if not patch(head, a.task):
        sys.exit(1)
    if not verify(a.task):
        print("⚠️  验证未通过，但补丁已写入；请在导出 ONNX 时再确认")
    print("\n下一步：python scripts/3_export/export_onnx.py" +
          (" --task pose" if a.task == "pose" else ""))


if __name__ == "__main__":
    main()
