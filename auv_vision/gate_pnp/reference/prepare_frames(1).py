#!/usr/bin/env python3
"""
图片集 → YOLO 训练图集（去畸变 + 画面补偿 + 640x640）

输入为【已经切好/分好类的原始图片】（配合 scripts/extract_frames.py 切帧、
人工分类后的目录），逐张复刻板端推理（rdk_bpu_worker.py）的预处理链路，
保证【进入 YOLO 训练的图片】与【进入 YOLO 推理的图片】像素级一致：

    1) 去畸变      cv2.remap(标定yaml生成的 remap 映射, INTER_LINEAR)
    2) 画面补偿    enhance(): 白平衡通道增益 → LAB空间L通道 CLAHE → gamma LUT
    3) 统一尺寸    cv2.resize((640, 640))

参数（白平衡增益/CLAHE/gamma/标定文件）以板端 auv_vision 工程的 cfg/vision.yaml
为唯一来源（镜像见 configs/vision.yaml）：image.undistort / white_balance_bgr /
clahe_clip / gamma / camera.front.calibration / model.input_size；
用 --config 直接读取即可，保证训练图与板端推理图逐参数一致。

用法示例：
    # 方式1：--config 读取板端同一份 vision.yaml（推荐，天然一致）
    python scripts/prepare_frames.py data/AUV_1/red_ball --config configs/vision.yaml

    # 方式2：显式给出与板端一致的参数
    python scripts/prepare_frames.py data/AUV_1/red_ball data/AUV_1/blue_ball data/AUV_1/gate \
        --calibration configs/front_camera.yaml \
        --wb-gains 1.15,1.0,0.9 --clahe 2.0 --gamma 1.0 \
        --out-dir data/AUV_1/processed_640

    # 通配符（shell 展开成多个目录，按来源目录分组建子目录输出）
    python scripts/prepare_frames.py data/AUV_1/frames/red_ball*

输出: <out_dir>/<来源目录名>/frame_000001.jpg ...（每张 640x640）
之后标注并组织为 YOLO 数据集（train/valid + data.yaml）即可开始训练：
    python scripts/train_yolo11n.py --data data/<名字>/data.yaml ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ---------- 与板端 rdk_bpu_worker.py 完全一致的函数 ----------

def calibration_maps(path: str, width: int, height: int):
    """由标定 yaml 生成去畸变 remap 映射（按分辨率缓存，只算一次）"""
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"camera calibration missing: {path}")
    k = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("distortion_coefficients").mat()
    fs.release()
    if k is None or dist is None:
        raise ValueError(f"invalid camera calibration: {path}")
    new_k, _ = cv2.getOptimalNewCameraMatrix(k, dist, (width, height), 0,
                                             (width, height))
    return cv2.initUndistortRectifyMap(k, dist, None, new_k,
                                       (width, height), cv2.CV_16SC2)


def enhance(frame: np.ndarray, gains: list[float], clip: float,
            gamma: float) -> np.ndarray:
    """画面补偿：白平衡通道增益 → (clip>0 时) LAB-L通道CLAHE → gamma LUT。
    与板端 auv_vision/preprocess.py 的 enhance 逐行一致。"""
    f = np.clip(frame.astype(np.float32) * np.array(gains, np.float32),
                0, 255).astype(np.uint8)
    if clip > 0:                              # 板端：clip=0 时不做 CLAHE
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l)
        f = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    lut = np.array([pow(i / 255.0, gamma) * 255 for i in range(256)],
                   dtype=np.uint8)
    return cv2.LUT(f, lut)


# ---------- 参数解析 ----------

def parse_gains(text: str) -> list[float]:
    try:
        gains = [float(x) for x in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"非法增益列表: {text!r}")
    if len(gains) != 3:
        raise argparse.ArgumentTypeError("需要 3 个值，顺序为 B,G,R")
    return gains


def load_board_config(config: Path):
    """从板端同一份 vision.yaml 读取图像处理参数（板端参数唯一来源）。
    兼容两种结构：
      - auv_vision 工程 vision.yaml：image.undistort + camera.front.calibration
      - 旧 mission.yaml 风格：image.white_balance_bgr + camera.front_calibration
    """
    try:
        import yaml
    except ImportError:
        sys.exit("❌ 缺少 PyYAML，请先: pip install -r requirements.txt")

    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    img = cfg.get("image") or {}
    cam = cfg.get("camera") or {}
    front = cam.get("front") or {}          # vision.yaml: camera.front.calibration
    model = cfg.get("model") or {}
    return {
        "undistort": bool(img.get("undistort", True)),
        "calibration": (front.get("calibration")
                        or cam.get("front_calibration") or None),
        "gains": (img.get("white_balance_bgr")
                  or img.get("white_balance") or None),
        "clahe": img.get("clahe_clip"),
        "gamma": img.get("gamma"),
        "size": model.get("input_size"),
    }


def localize_calibration(path: str | None, config: Path | None) -> str | None:
    """板端 yaml 里的标定路径常是 /app/... 绝对路径，PC 端不存在时回退：
    --config 同目录 或 项目 configs/ 下同名文件"""
    if not path:
        return None
    p = Path(path)
    if p.exists():
        return str(p)
    for base in ((config.parent if config else None), Path("configs")):
        if base is None:
            continue
        cand = base / p.name
        if cand.exists():
            print(f"📄 标定路径 {path} 本机不存在，已回退使用 {cand}")
            return str(cand)
    return str(p)   # 保持原样，交给 calibration_maps 给出明确报错


def collect_image_sets(inputs: list[str]):
    """
    把输入整理为 [(组名, [图片路径...]), ...]：
      - 目录  → 该目录下所有图片为一组（组名=目录名）
      - 文件  → 同父目录的文件合并为一组（组名=父目录名）
    """
    groups: dict[str, list[Path]] = {}
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files = sorted(
                f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS)
            if not files:
                print(f"⚠️  目录中没有图片，跳过: {p}")
                continue
            groups.setdefault(p.name, []).extend(files)
        elif p.is_file() and p.suffix.lower() in IMG_EXTS:
            groups.setdefault(p.parent.name, []).append(p)
        else:
            print(f"⚠️  无法识别（既非目录也非图片）: {p}")
    if not groups:
        sys.exit("❌ 没有可处理的图片输入")
    return list(groups.items())


# ---------- 主流程 ----------

def prepare(groups, out_dir: Path, calibration: str | None,
            gains: list[float], clahe: float, gamma: float,
            size: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    total_saved = 0
    maps_cache: dict[tuple[int, int], tuple] = {}

    for group, files in groups:
        sub = out_dir / group
        sub.mkdir(parents=True, exist_ok=True)
        saved = 0
        for img_path in files:
            raw = cv2.imread(str(img_path))
            if raw is None:
                print(f"⚠️  读取失败，跳过: {img_path}")
                continue
            h, w = raw.shape[:2]

            if calibration:                          # 1) 去畸变（按分辨率缓存映射）
                if (w, h) not in maps_cache:
                    maps_cache[(w, h)] = calibration_maps(calibration, w, h)
                    print(f"📐 生成去畸变映射 {w}x{h} <- {calibration}")
                m0, m1 = maps_cache[(w, h)]
                frame = cv2.remap(raw, m0, m1, cv2.INTER_LINEAR)
            else:                                    # 未标定则原图直出（告警一次）
                frame = raw
            frame = enhance(frame, gains, clahe, gamma)  # 2) 画面补偿
            frame = cv2.resize(frame, (size, size),   # 3) 统一 640x640
                               interpolation=cv2.INTER_LINEAR)
            dst = sub / f"frame_{saved + 1:06d}.jpg"
            cv2.imwrite(str(dst), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1

        print(f"✅ {group}: {saved}/{len(files)} 张 -> {sub}  ({size}x{size})")
        total_saved += saved
    return total_saved


def main() -> None:
    ap = argparse.ArgumentParser(
        description="图片集 → 去畸变+画面补偿+640x640 训练图集（与板端推理一致）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="图片目录或图片文件（可多个/通配符；目录名作为分组）")
    ap.add_argument("--out-dir", type=Path, default=Path("processed_640"),
                    help="输出根目录（每个输入组一个子目录）")
    ap.add_argument("--calibration", default=None,
                    help="标定 yaml（calibrate_camera_video.py 的输出）；不传则跳过去畸变")
    ap.add_argument("--size", type=int, default=640,
                    help="输出/模型输入尺寸（板端 model.input_size）")
    ap.add_argument("--wb-gains", type=parse_gains, default=[1.0, 1.0, 1.0],
                    help="白平衡通道增益 B,G,R（板端 image.white_balance_bgr）")
    ap.add_argument("--clahe", type=float, default=2.0,
                    help="CLAHE clipLimit（板端 image.clahe_clip）")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="gamma 值（板端 image.gamma）")
    ap.add_argument("--config", type=Path, default=None,
                    help="板端 vision.yaml/mission.yaml：自动读取图像参数与去畸变开关"
                         "（板端参数唯一来源，推荐）")
    a = ap.parse_args()

    board = None
    if a.config:
        if not a.config.exists():
            sys.exit(f"❌ 配置文件不存在: {a.config}")
        board = load_board_config(a.config)
        if board["gains"]:
            a.wb_gains = [float(x) for x in board["gains"]]
        if board["clahe"] is not None:
            a.clahe = float(board["clahe"])
        if board["gamma"] is not None:
            a.gamma = float(board["gamma"])
        if board["size"]:
            a.size = int(board["size"])
        print(f"📄 已从 {a.config} 读取参数: gains={a.wb_gains} "
              f"clahe={a.clahe} gamma={a.gamma} size={a.size} "
              f"undistort={board['undistort']}")

    # 去畸变开关语义与板端一致：
    #   显式 --calibration 视为强制开启；否则跟随 board 配置的 undistort
    if a.calibration:
        a.calibration = str(a.calibration)
    elif board and board["undistort"]:
        a.calibration = localize_calibration(board["calibration"], a.config)
    elif board:
        print("⚠️  板端配置 image.undistort=false：本次跳过去畸变（仅补偿+缩放），"
              "与板端推理一致；如需开启请显式传 --calibration 或改板端 vision.yaml")
    else:
        a.calibration = None

    if not a.calibration:
        print("⚠️  本次跳过去畸变（仅补偿+缩放）。需要去畸变时请传"
              "--calibration（或使用已开启 undistort 的板端配置）")

    groups = collect_image_sets(a.inputs)
    n = prepare(groups, a.out_dir, a.calibration, a.wb_gains, a.clahe,
                a.gamma, a.size)
    print(f"\n🎉 共生成 {n} 张训练图片（{a.size}x{a.size}），位于 {a.out_dir}/")
    print("下一步：标注后按 YOLO 格式组织（train/valid + data.yaml），再运行:")
    print("   python scripts/train_yolo11n.py --data data/<数据集>/data.yaml ...")


if __name__ == "__main__":
    main()
