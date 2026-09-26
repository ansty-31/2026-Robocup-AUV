#!/usr/bin/env python3
"""图片集 → YOLO 训练图集（与板端推理同链路）：resize(640) → 画面补偿 → 去畸变 remap@640。

链路逐行对齐板端 common/preprocess.py（顺序不可交换）：
    1) 统一尺寸  cv2.resize((size, size), INTER_LINEAR)
    2) 画面补偿  enhance(): 白平衡通道增益 → LAB-L CLAHE(clip>0 时) → gamma LUT
    3) 去畸变    cv2.remap(640 域映射, INTER_LINEAR)

参数（gains/clahe/gamma/标定/尺寸）由 --config 从板端同一份 vision.yaml 读取（推荐），
或用 --calibration + --wb-gains/--clahe/--gamma 显式给出；不给标定则跳过去畸变。

用法：
    python scripts/1_prepare/prepare_frames.py data/frames/AUV_1_red_ball --config configs/vision.yaml

    python scripts/1_prepare/prepare_frames.py data/frames/AUV_1_red_ball data/frames/AUV_1_gate \
        --calibration configs/front_camera.yaml \
        --wb-gains 1.15,1.0,0.9 --clahe 2.0 --gamma 1.0 \
        --out-dir data/frames/AUV_1_processed_640

输出：<out-dir>/<来源目录名>/frame_000001.jpg（每张 size×size，默认 640）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ---------- 与板端 common/preprocess.py 完全一致的函数 ----------

def calibration_maps_640(path: str, size: int = 640):
    """640 域去畸变映射：dest(去畸变 size×size) ← src(畸变 size×size)，与
    map_pose_dataset.py 的「D 域」同式：

        S = diag(size/RAW_W, size/RAW_H, 1)
        nk720 = getOptimalNewCameraMatrix(K, dist, (RAW_W,RAW_H), 0, (RAW_W,RAW_H))
        nk640 = S · nk720 ;  K640 = S · K
        initUndistortRectifyMap(K640, dist, None, nk640, (size,size))
    """
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"camera calibration missing: {path}")
    k = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("distortion_coefficients").mat()
    w = int(fs.getNode("image_width").real() or 1280)
    h = int(fs.getNode("image_height").real() or 720)
    fs.release()
    if k is None or dist is None:
        raise ValueError(f"invalid camera calibration: {path}")
    sc = np.diag([size / float(w), size / float(h), 1.0])
    new_k, _ = cv2.getOptimalNewCameraMatrix(k, dist, (w, h), 0, (w, h))
    return cv2.initUndistortRectifyMap(sc @ k, dist, None, sc @ new_k,
                                       (size, size), cv2.CV_16SC2)


def calibration_maps(path: str, width: int, height: int):
    """由标定 yaml 生成 (width, height) 分辨率下的去畸变 remap 映射"""
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


_CLAHE_CACHE: dict[float, object] = {}
_GAMMA_CACHE: dict[float, np.ndarray] = {}


def _clahe(clip: float):
    """CLAHE 对象缓存（与板端一致，避免每帧 createCLAHE）"""
    c = _CLAHE_CACHE.get(clip)
    if c is None:
        c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        _CLAHE_CACHE[clip] = c
    return c


def _gamma_lut(gamma: float) -> np.ndarray:
    """gamma LUT 缓存（与板端一致，避免每帧重建 256 项 LUT）"""
    lut = _GAMMA_CACHE.get(gamma)
    if lut is None:
        lut = np.array([pow(i / 255.0, gamma) * 255 for i in range(256)],
                       dtype=np.uint8)
        _GAMMA_CACHE[gamma] = lut
    return lut


def enhance(frame: np.ndarray, gains: list[float], clip: float,
            gamma: float) -> np.ndarray:
    """画面补偿：白平衡通道增益 → (clip>0 时) LAB-L CLAHE → gamma LUT。
    与板端 common/preprocess.py 的 enhance 逐行一致。"""
    f = np.clip(frame.astype(np.float32) * np.array(gains, np.float32),
                0, 255).astype(np.uint8)
    if clip > 0:                              # 板端：clip=0 时不做 CLAHE
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = _clahe(clip).apply(l)
        f = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return cv2.LUT(f, _gamma_lut(gamma))


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
    """读板端 vision.yaml 的图像参数：image.undistort/white_balance_bgr/clahe_clip/
    gamma + camera.front.calibration + model.input_size。
    兼容旧 mission.yaml 风格（image.white_balance + camera.front_calibration）。"""
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
    """板端 yaml 里的标定路径常是 /app/... 绝对路径：本机不存在时回退到
    --config 同目录 或 项目 configs/ 下的同名文件"""
    if not path:
        return None
    p = Path(path)
    if p.exists():
        return str(p)
    for base in ((config.parent if config else None), PROJECT_ROOT / "configs"):
        if base is None:
            continue
        cand = base / p.name
        if cand.exists():
            print(f"📄 标定路径 {path} 本机不存在，已回退使用 {cand}")
            return str(cand)
    return str(p)   # 保持原样，交给 calibration_maps 给出明确报错


def collect_image_sets(inputs: list[str]):
    """把输入整理为 [(组名, [图片路径...]), ...]：
    目录 → 该目录下所有图片为一组（组名=目录名）；
    文件 → 与其同父目录的文件合并为一组（组名=父目录名）。"""
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
    maps = None

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

            # D 域链路（与板端 common/preprocess.py 一致，顺序不可交换）：
            # resize(size) → enhance → remap@640
            frame = raw
            if (w, h) != (size, size):               # 1) 先统一尺寸
                frame = cv2.resize(frame, (size, size),
                                   interpolation=cv2.INTER_LINEAR)
            frame = enhance(frame, gains, clahe, gamma)      # 2) 在 size 上做补偿
            if calibration:                          # 3) 再去畸变
                if maps is None:
                    maps = calibration_maps_640(calibration, size)
                    print(f"📐 生成 640 域去畸变映射 {size}x{size} <- {calibration}")
                frame = cv2.remap(frame, *maps, cv2.INTER_LINEAR)
            dst = sub / f"frame_{saved + 1:06d}.jpg"
            cv2.imwrite(str(dst), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1

        print(f"✅ {group}: {saved}/{len(files)} 张 -> {sub}  ({size}x{size})")
        total_saved += saved
    return total_saved


def main() -> None:
    ap = argparse.ArgumentParser(
        description="图片集 → 去畸变+640x640+画面补偿 训练图集（与板端推理同链路同顺序）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="图片目录或图片文件（可多个/通配符；目录名作为分组）")
    ap.add_argument("--out-dir", type=Path, default=Path("processed_640"),
                    help="输出根目录（每个输入组一个子目录）")
    ap.add_argument("--calibration", default=None,
                    help="标定 yaml（calibrate_camera.py 的输出）；不传则跳过去畸变")
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

    if not a.calibration:
        print("⚠️  本次跳过去畸变（仅补偿+缩放）。需要去畸变时请传"
              "--calibration（或使用已开启 undistort 的板端配置）")

    groups = collect_image_sets(a.inputs)
    n = prepare(groups, a.out_dir, a.calibration, a.wb_gains, a.clahe,
                a.gamma, a.size)
    print(f"\n🎉 共生成 {n} 张训练图片（{a.size}x{a.size}），位于 {a.out_dir}/")


if __name__ == "__main__":
    main()
