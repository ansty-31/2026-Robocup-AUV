#!/usr/bin/env python3
"""
train_yolo11n.py — 用自定义图片数据集训练/微调 YOLO11n

【在流程中的位置】衔接在 导出ONNX（scripts/3_export/export_onnx.py）之前：
    下载预训练权重 → 【本脚本：训练】 → modify_ultralytics(6输出头) → export_onnx → 量化

【脚本自动处理的关键问题】
  本项目的 modify_ultralytics.py 会把安装版 ultralytics 的 Detect.forward 改为
  直接输出 6 个 tensor（无 training 分支），若带着该补丁训练，loss 计算会出错。
  因此脚本流程为：
    1) 若 head.py 已打 6 输出补丁，先从 head.py.backup 恢复原始版本（训练用）
    2) 用 ultralytics 训练（默认基于项目根目录 yolo11n.pt 预训练权重微调；
       文件不存在时会自动下载）
    3) 将训练产物 best.pt 复制为项目根目录 yolo11n.pt（默认），
       使后续 export_onnx.py 无需任何参数即可衔接
    4) 重新执行 scripts/3_export/modify_ultralytics.py 打上 6 输出头补丁（导出 ONNX 需要）

【用法示例】
    # 检测（RoboFlow/Ultralytics 格式数据集）
    python scripts/2_train/train_yolo11n.py --data data/AUV_1/dataset/AUV.yolov11/data.yaml \
        --epochs 300 --batch 4 --device 0 --cache ram

    # 关键点（gate 4 角点，RoboFlow Pose 导出；data.yaml 需含 kpt_shape: [4, 3]）
    python scripts/2_train/train_yolo11n.py --task pose --data data/gate_kpt/data.yaml \
        --epochs 300 --batch 4 --device 0

    # 只训练不打补丁/不覆盖权重（例如只想先对比效果）
    python scripts/2_train/train_yolo11n.py --data /path/to/data.yaml --no-repatch \
        --output-pt runs/best_tmp.pt

【注意】
    - imgsz 默认 640，与后续 PTQ 量化配置（640x640 nv12）保持一致
    - 类别数不受限制：加载 80 类 COCO 预训练权重后，ultralytics 会自动适配自定义 nc
    - 训练结束后 head.py 处于"已打补丁"状态，属于预期行为（下一步就是导出 ONNX）
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # scripts/<stage>/x.py → 项目根
SCRIPTS_DIR = REPO_ROOT / "scripts"
WEIGHT_FILES = {"detect": "weights/yolo11n.pt", "pose": "weights/yolo11n-pose.pt"}


def find_head_file():
    """定位当前 Python 环境中 ultralytics 的 head.py

    注意：必须用 importlib.util.find_spec 定位，**不能 import ultralytics**——
    一旦导入，补丁版的 head 会进入同进程模块缓存，之后 restore 只改文件也无效，
    训练仍会用到补丁版 Pose/Detect（典型症状：pose loss 解包报错）。
    """
    import importlib.util
    spec = importlib.util.find_spec("ultralytics")
    if spec is None or not spec.origin:
        print("❌ 未安装 ultralytics，请先执行: pip install -r requirements.txt")
        sys.exit(1)
    return Path(spec.origin).resolve().parent / "nn" / "modules" / "head.py"


def restore_original_head():
    """把 head.py 恢复为原始版本（训练必需，因为 6 输出补丁会让训练/AMP 后处理崩溃）。
    用“内容与备份不同则恢复”，不依赖标记字符串，更鲁棒。"""
    head_file = find_head_file()
    backup_file = head_file.with_name(head_file.name + ".backup")

    if not backup_file.exists():
        print("⚠️  未找到 head.py.backup，无法校验/恢复 head.py；"
              "若 head.py 曾被 scripts/3_export/modify_ultralytics.py 改过，训练可能报错")
        return
    head_text = head_file.read_text(encoding="utf-8")
    backup_text = backup_file.read_text(encoding="utf-8")
    if head_text == backup_text:
        print("ℹ️  head.py 已是原始版本，无需恢复")
        return

    shutil.copy2(backup_file, head_file)
    # 清除字节码缓存，避免文件 mtime 粒度导致的旧 pyc 被复用
    for pyc in (head_file.parent / "__pycache__").glob("head*.pyc"):
        pyc.unlink(missing_ok=True)
    print(f"✅ 已从备份恢复原始 head.py（训练模式）: {head_file}")


def reapply_output_head_patch(task: str):
    """训练完成后按任务重打输出头补丁（detect=6 输出 / pose=9 输出）"""
    print(f"\n🔄 训练完成，重新应用 {task} 输出头补丁（供导出 ONNX 使用）...")
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "3_export" / "modify_ultralytics.py"), "--task", task],
        cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        print("⚠️  重新打补丁脚本返回非零状态，请手动检查 scripts/3_export/modify_ultralytics.py")
        return False
    return True


def resolve_data_yaml(data_path: Path):
    """
    返回一份可被 ultralytics 直接使用的 data.yaml
    ultralytics 会将 yaml 中相对/无效的 path 解析到 settings.datasets_dir
    （而它可能指向无关的其他目录），因此这里把 path 与各 split 全部改成
    【绝对路径】写入临时 yaml，彻底绕开该拼接逻辑（不修改用户原文件）
    """
    import yaml

    data_path = data_path.resolve()          # 关键：先绝对化，避免 str() 后仍是相对路径
    raw = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    raw_path = raw.get("path")
    splits = ["train", "val", "test"]

    candidates = []
    if raw_path:
        p = Path(raw_path)
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(REPO_ROOT / p)          # 相对项目根
            candidates.append(data_path.parent / p)   # 相对 yaml 所在目录
    else:
        # RoboFlow 等没有 path 字段：以 yaml 所在目录为数据根
        candidates.append(data_path.parent)
    # 若 path 字段本身失效，尝试 yaml 同目录下的同名数据集文件夹
    if raw_path:
        candidates.append(data_path.parent / Path(str(raw_path)).name)

    resolved = None
    for cand in candidates:
        if cand.is_dir():
            resolved = cand
            break
    if resolved is None:
        print(f"❌ 无法定位数据集目录（yaml 中 path={raw_path!r} 的候选均不存在）")
        sys.exit(1)
    resolved = resolved.resolve()             # 绝对化数据根

    # 校验并把各 split 也改为绝对路径
    missing = [s for s in splits if raw.get(s) and not (resolved / raw[s]).is_dir()]
    if missing:
        print(f"❌ 数据集目录 {resolved} 中缺少: {missing}")
        print(f"   data.yaml: {data_path}")
        sys.exit(1)

    raw["path"] = str(resolved)
    for s in splits:
        if raw.get(s):
            raw[s] = str((resolved / raw[s]).resolve())

    tmp_yaml = Path("/tmp") / f"{data_path.stem}_abs.yaml"
    tmp_yaml.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"📁 数据集目录: {resolved}")
    print(f"📄 使用修正后的配置: {tmp_yaml}（path 与 split 均为绝对路径；原文件未改动: {data_path}）")
    return str(tmp_yaml)


def main():
    parser = argparse.ArgumentParser(
        description="用自定义图片数据集训练/微调 YOLO11n（衔接导出ONNX）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True,
                        help="数据集配置 data.yaml（YOLO格式，含 train/val、nc、names）")
    parser.add_argument("--task", choices=["detect", "pose"], default="detect",
                        help="任务：detect=目标检测 / pose=关键点（gate 4 角点）")
    parser.add_argument("--weights", default=None,
                        help="预训练权重（默认 detect: yolo11n.pt / pose: yolo11n-pose.pt，"
                             "不存在则自动下载）")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="训练输入尺寸（与量化640x640保持一致）")
    parser.add_argument("--batch", type=int, default=4, help="批大小")
    parser.add_argument("--workers", type=int, default=2,
                        help="dataloader 进程数（本机出现过 8 个 worker 与 CUDA fork 死锁，"
                             "默认 2 更稳定）")
    parser.add_argument("--cache", type=str, default=None,
                        help="图像缓存: None/False=不缓存, 'ram'=缓存到内存, 'disk'=缓存到磁盘")
    parser.add_argument("--device", default="",
                        help="训练设备: ''=自动, '0'=GPU, 'cpu'=CPU")
    parser.add_argument("--project", default=None,
                        help="训练输出根目录（默认 runs/<task>）")
    parser.add_argument("--name", default=None,
                        help="本次运行名称（默认自动生成，避免覆盖旧运行）")
    parser.add_argument("--output-pt", default=None,
                        help="best.pt 复制到的路径（默认 detect: yolo11n.pt / "
                             "pose: yolo11n-pose.pt，衔接 export_onnx）")
    parser.add_argument("--no-repatch", action="store_true",
                        help="训练后不重新打输出头补丁（自行衔接导出时使用）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--amp", action="store_true", dest="amp",
                        help="启用自动混合精度（默认）")
    parser.add_argument("--no-amp", action="store_false", dest="amp",
                        help="关闭 AMP（若 AMP 检查崩溃/报错时使用）")
    parser.set_defaults(amp=True)
    args = parser.parse_args()
    # 按任务填充默认值（detect: yolo11n.pt / pose: yolo11n-pose.pt）
    args.weights = args.weights or str(REPO_ROOT / WEIGHT_FILES[args.task])
    args.project = args.project or str(REPO_ROOT / "runs" / args.task)
    args.output_pt = args.output_pt or str(REPO_ROOT / WEIGHT_FILES[args.task])

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ 数据集配置不存在: {args.data}")
        print("   请提供 YOLO 格式的 data.yaml，例如:")
        print("     data/auv/data.yaml")
        sys.exit(1)
    data_yaml = resolve_data_yaml(data_path)

    # ---------- 1. 训练前：恢复原始输出头 ----------
    print("=" * 70)
    print("YOLO11n 自定义数据集训练")
    print("=" * 70)
    restore_original_head()

    # ---------- 2. 训练 ----------
    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"📦 ultralytics 版本: {ultralytics.__version__}")
    except ImportError:
        print("❌ 未安装 ultralytics，请先执行: pip install -r requirements.txt")
        sys.exit(1)

    weights = args.weights
    if not Path(weights).exists():
        print(f"ℹ️  未找到预训练权重 {weights}，将由 ultralytics 自动下载 ...")
    model = YOLO(weights)

    print(f"📁 数据集: {data_path}")
    print(f"⚙️  参数: epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, device='{args.device}'")
    print("🚀 开始训练...\n")

    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        cache=args.cache,
        device=args.device or None,
        project=args.project,
        name=args.name,          # None → ultralytics 自动编号，不覆盖旧运行
        seed=args.seed,
        amp=args.amp,
        exist_ok=False,
        verbose=True,
    )

    # ---------- 3. 定位 best.pt 并衔接导出流程 ----------
    save_dir = Path(model.trainer.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    print(f"\n📦 训练结果目录: {save_dir}")
    if not best_pt.exists():
        print(f"❌ 未找到训练产物 {best_pt}，请检查训练日志")
        sys.exit(1)
    print(f"✅ 最佳权重: {best_pt}")

    output_pt = Path(args.output_pt)
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, output_pt)
    print(f"✅ 已复制为: {output_pt}  ← 供 scripts/3_export/export_onnx.py 直接使用")

    # ---------- 4. 训练后：重新打 6 输出头补丁 ----------
    if args.no_repatch:
        print("\n⚠️  --no-repatch 已指定，未打输出头补丁")
        print(f"   后续导出前请手动执行: python scripts/3_export/modify_ultralytics.py --task {args.task}")
    else:
        ok = reapply_output_head_patch(args.task)
        if not ok:
            sys.exit(1)

    # ---------- 5. 衔接提示 ----------
    print("\n" + "=" * 70)
    print("✅ 训练完成！下一步（导出 ONNX 并量化）：")
    print("=" * 70)
    print("   python scripts/3_export/export_onnx.py" +
          (" --task pose" if args.task == "pose" else ""))
    print("   python scripts/3_export/prepare_calibration.py --coco-path /path/to/coco/val2017")
    print("   ./scripts/3_export/quantize.sh")
    print()


if __name__ == "__main__":
    main()
