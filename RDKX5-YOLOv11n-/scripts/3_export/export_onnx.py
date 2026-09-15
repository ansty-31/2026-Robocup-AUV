#!/usr/bin/env python3
"""
导出YOLOv11n ONNX模型（6输出版本）

作者: RDKX5-YOLOv11n-项目
许可证: MIT
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # scripts/<stage>/x.py

def export_onnx(model_path='yolo11n.pt', output_name='yolo11n.onnx', imgsz=640,
                task='detect'):
    """
    导出ONNX模型

    Args:
        model_path: 预训练模型路径
        output_name: 期望的输出ONNX文件名（实际文件名由 ultralytics 按模型名生成）
        imgsz: 输入图像尺寸
        task: detect=6 输出（3×bbox + 3×cls）/ pose=9 输出（每尺度 bbox+cls+kpt）
    """
    
    try:
        from ultralytics import YOLO
        import onnx
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("   请先安装: pip install ultralytics onnx")
        return False
    
    print("=" * 60)
    print("导出YOLOv11n ONNX模型")
    print("=" * 60)
    print()
    
    # 检查模型文件
    if not Path(model_path).exists():
        print(f"❌ 模型文件不存在: {model_path}")
        print("   正在下载...")
        try:
            model = YOLO(model_path)
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            return False
    else:
        print(f"✅ 找到模型文件: {model_path}")
        model = YOLO(model_path)
    
    print()
    print("📝 导出配置:")
    print(f"  模型: {model_path}")
    print(f"  输出: {output_name}")
    print(f"  输入尺寸: {imgsz}x{imgsz}")
    print(f"  ONNX Opset: 11")
    print()
    
    # 导出ONNX
    print("🚀 开始导出...")
    try:
        success = model.export(
            format='onnx',
            imgsz=imgsz,
            opset=11,           # RDK X5支持opset 10/11
            simplify=False,     # 不简化，避免ir version问题
            dynamic=False,      # 静态shape（BPU不支持动态shape）
            half=False          # 使用float32
        )
        
        print()
        print(f"✅ ONNX导出成功: {success}")
        onnx_path = str(success) if success else output_name
        
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 验证ONNX模型
    print()
    print("🔍 验证ONNX模型...")
    try:
        onnx_model = onnx.load(onnx_path)

        print(f"  IR Version: {onnx_model.ir_version}")
        print(f"  Opset Version: {onnx_model.opset_import[0].version}")
        print(f"  Producer: {onnx_model.producer_name} {onnx_model.producer_version}")
        print()
        print(f"  输入数量: {len(onnx_model.graph.input)}")
        for i, inp in enumerate(onnx_model.graph.input):
            print(f"    Input {i}: {inp.name}")

        outs = onnx_model.graph.output
        dims = [[d.dim_value for d in o.type.tensor_type.shape.dim] for o in outs]
        print()
        print(f"  输出数量: {len(outs)}")
        for i, (o, d) in enumerate(zip(outs, dims)):
            print(f"    Output {i}: {o.name}  shape={d}")
        print()

        if task == "pose":
            chans = [d[-1] for d in dims]
            grid = sorted({d[1] for d in dims}, reverse=True)
            uniq = sorted(set(chans))
            # 9 个输出 = 3 尺度 × (reg 64 / cls nc / kpt 3*kpt_dim)，每类出现 3 次
            ok = (len(outs) == 9 and chans.count(64) == 3 and len(uniq) == 3
                  and all(chans.count(c) == 3 for c in uniq))
            if ok:
                kpt_ch = [c for c in uniq if c != 64 and c != 1] or [0]
                kpt_dim = (kpt_ch[0] // 3) if kpt_ch[0] else 4
                print("✅ 输出数量正确（9 个 = 3 尺度 × (bbox 64 + cls nc + kpt 3*kpt_dim)）")
                print(f"   每尺度 NHWC：网格 {grid}，kpt 通道 {kpt_ch[0]}（= {kpt_dim} 点 × (x,y,v)）")
                if kpt_ch[0] != 12:
                    print(f"   ⚠️ gate 需要 4 点（kpt 通道 12），当前模型为 {kpt_dim} 点；"
                          "请用 kpt_shape=(4,3) 的数据集训练")
            else:
                print(f"⚠️  输出结构异常（{len(outs)} 个，通道 {uniq}）")
                print("   预期 9 个：3×C=64(reg) + 3×C=nc(cls) + 3×C=3*kpt_dim(kpt)")
                print("   提示：python scripts/modify_ultralytics.py --task pose")
                return False
        else:
            if len(outs) == 6:
                print("✅ 输出数量正确（6个）")
                print("   - Output 0-2: BBox特征 (stride=8/16/32)")
                print("   - Output 3-5: Class分数 (stride=8/16/32)")
            else:
                print(f"⚠️  警告：输出数量为 {len(outs)}")
                print("   预期6个输出，请检查ultralytics是否正确修改")
                print("   提示：运行 python scripts/modify_ultralytics.py --task detect")
                return False

    except Exception as e:
        print(f"⚠️  验证过程出错: {e}")
    
    print()
    print("=" * 60)
    print("✅ ONNX导出完成！")
    print("=" * 60)
    print()
    print("下一步：")
    print("  1. 准备校准数据:")
    print("     python scripts/3_export/prepare_calibration.py --coco-path /path/to/coco")
    print()
    print("  2. PTQ量化:")
    print("     ./scripts/3_export/quantize.sh" +
          (" configs/gate_kpt_config.yaml" if task == 'pose' else ""))
    print()
    
    return True


def main():
    """主函数"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='导出YOLO11 ONNX模型（RDK X5 分裂头）')
    parser.add_argument('--task', choices=['detect', 'pose'], default='detect',
                        help='detect=6 输出 / pose=9 输出（关键点）')
    parser.add_argument('--model', type=str, default=None,
                        help='模型权重（默认 detect: yolo11n.pt / pose: yolo11n-pose.pt）')
    parser.add_argument('--output', type=str, default=None,
                        help='期望输出名（默认 <模型名>.onnx）')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像尺寸 (default: 640)')

    args = parser.parse_args()
    args.model = args.model or str(PROJECT_ROOT / 'weights' /
                                 ('yolo11n-pose.pt' if args.task == 'pose' else 'yolo11n.pt'))
    args.output = args.output or Path(args.model).with_suffix('.onnx').name

    success = export_onnx(
        model_path=args.model,
        output_name=args.output,
        imgsz=args.imgsz,
        task=args.task,
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
