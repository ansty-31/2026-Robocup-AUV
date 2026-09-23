#!/usr/bin/env python3
"""
bench_undistort_placement.py — E3：去畸变放哪里？四种方案的预处理耗时 / DDR 流量

## 四种方案（同一个门框场景，只换去畸变的位置）

    P1  remap@1280x720 → resize640 → enhance      ← 板端当前链路（ModelPreprocessor.process）
    P2  resize640 → enhance → remap@640           ← = 畸变实验的 C 组
    P3  resize640 → enhance →(检测)→ 只映射 4 个角点  ← C′，省掉整幅 remap
    P4  resize640 → enhance                       ← 完全不去畸变（PnP 直接吃 K+D）

P1/P2 是**整幅重采样**，P3/P4 把去畸变降级成「4 个点做坐标变换」。几何上 P1 与 P2 等价
（同一套映射，只差重采样次序），P3/P4 的几何误差必须由 E1（PnP 真值）来判。

## 为什么必须在板上跑

x86 上 cv2.remap 的绝对耗时不能代表 RDK X5 的 A55；而且板端真正的瓶颈是
**DDR 流量**（remap 在 720p 上做，每帧多约 3.3 MB 读写）与 NV12 打包。
本脚本同时报**耗时**与**估算流量**，两者一起看。

## 用法（在板子上）

    python3 bench_undistort_placement.py \
        --board-root /home/sunrise/AUV \
        --calibration /home/sunrise/AUV/cfg/front_camera.yaml \
        --frames /home/sunrise/AUV/frames --n 60

    # 没有素材时用合成帧（只测耗时，不反映真实内容）
    python3 bench_undistort_placement.py --board-root /home/sunrise/AUV --synthetic

输出：逐方案 ms/帧（中位/p90）、估算 DDR 流量、以及相对 P1 的加速比。
**不含模型推理** —— 那部分由板端 hbm_runtime 自己测。
"""
from __future__ import annotations
import argparse, glob, json, statistics, sys, time
from pathlib import Path
import cv2, numpy as np

RAW_W, RAW_H, SIZE = 1280, 720, 640


def load_calib(path):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("distortion_coefficients").mat()
    fs.release()
    return K, dist


def build_maps(K, dist, alpha=0.0):
    """720p 与 640 两套映射，与板端 / 畸变实验脚本同公式"""
    nk720, _ = cv2.getOptimalNewCameraMatrix(K, dist, (RAW_W, RAW_H), alpha, (RAW_W, RAW_H))
    m720 = cv2.initUndistortRectifyMap(K, dist, None, nk720, (RAW_W, RAW_H), cv2.CV_16SC2)
    S = np.diag([SIZE / RAW_W, SIZE / RAW_H, 1.0])
    nk640 = S @ nk720
    m640 = cv2.initUndistortRectifyMap(S @ K, dist, None, nk640, (SIZE, SIZE), cv2.CV_16SC2)
    return m720, m640, nk640


def make_enhance(board_root):
    """优先用板端自己的 common.preprocess.enhance（唯一权威实现）"""
    sys.path.insert(0, str(board_root))
    try:
        from common.preprocess import enhance as board_enhance
        import base.settings as S
        gains = list(S.vision.image.white_balance_bgr)
        clip = float(S.vision.image.clahe_clip)
        gamma = float(S.vision.image.gamma)
        return (lambda f: board_enhance(f, gains, clip, gamma)), "board", gains, clip, gamma
    except Exception as e:
        print(f"⚠️  无法导入板端 common.preprocess（{e}），退回本地实现")
        gains, clip, gamma = [1.0, 1.05, 1.15], 0.5, 0.85
        clahe_obj = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], np.uint8)

        def local_enhance(f):
            b, g, r = cv2.split(f.astype(np.float32))
            f = cv2.merge([np.clip(b * gains[0], 0, 255), np.clip(g * gains[1], 0, 255),
                           np.clip(r * gains[2], 0, 255)]).astype(np.uint8)
            lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
            lab[..., 0] = clahe_obj.apply(lab[..., 0])
            return cv2.LUT(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), lut)
        return local_enhance, "local", gains, clip, gamma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board-root", type=Path, default=Path("/home/sunrise/AUV"))
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--frames", type=Path, default=None, help="原始 720p 帧目录")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="结果 json")
    a = ap.parse_args()

    calib = a.calibration or (a.board_root / "cfg" / "front_camera.yaml")
    K, dist = load_calib(calib)
    m720, m640, nk640 = build_maps(K, dist)
    enhance, src, gains, clip, gamma = make_enhance(a.board_root)
    print(f"标定: {calib}  fx={K[0,0]:.2f}")
    print(f"enhance 来源: {src}  gains={gains} clahe={clip} gamma={gamma}")

    if a.synthetic or not a.frames:
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 255, (RAW_H, RAW_W, 3), dtype=np.uint8) for _ in range(max(8, a.n))]
        print(f"用合成帧 {len(frames)} 张（只测耗时）")
    else:
        files = sorted(glob.glob(str(Path(a.frames) / "*.jpg")))[: a.n]
        frames = [cv2.imread(f) for f in files]
        frames = [f for f in frames if f is not None and f.shape[:2] == (RAW_H, RAW_W)]
        print(f"用真实帧 {len(frames)} 张 <- {a.frames}")
    if not frames:
        sys.exit("❌ 没有可用帧（尺寸需为 1280x720）")

    pts = np.array([[100., 100.], [500., 120.], [520., 500.], [120., 520.]], np.float32)

    def p1(f):
        x = cv2.remap(f, *m720, cv2.INTER_LINEAR)
        x = cv2.resize(x, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        return enhance(x)

    def p2(f):
        x = cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        x = enhance(x)
        return cv2.remap(x, *m640, cv2.INTER_LINEAR)

    def p3(f):
        x = cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        x = enhance(x)
        cv2.undistortPoints(pts.reshape(-1, 1, 2), np.diag([SIZE / RAW_W, SIZE / RAW_H, 1.0]) @ K,
                            dist, P=nk640)          # 4 个点的坐标变换
        return x

    plans = [("P1 remap@720p→resize→enhance（板端当前）", p1),
             ("P2 resize→enhance→remap@640", p2),
             ("P3 resize→enhance→映射4点", p3),
             ("P4 resize→enhance（不去畸变）", p3)]

    def bench(fn):
        for _ in range(3):
            fn(frames[0])
        ts = []
        for _ in range(a.repeat):
            t = time.perf_counter()
            for f in frames:
                fn(f)
            ts.append((time.perf_counter() - t) / len(frames) * 1000)
        return statistics.median(ts)

    res = {}
    print("\n方案                                       中位 ms/帧   相对P1")
    base = None
    for name, fn in plans:
        ms = bench(fn)
        base = base if base is not None else ms
        res[name] = ms
        print(f"{name:<42} {ms:8.3f}   {base/ms:6.2f}x")

    b720, b640, nv12 = RAW_W * RAW_H * 3 / 1e6, SIZE * SIZE * 3 / 1e6, SIZE * SIZE * 1.5 / 1e6
    print("\n每帧 DDR 流量估算（BGR 720p=2.76 / 640=1.23 / NV12=0.61 MB）")
    for k, v in [("P1", b720 * 6 + b640 * 4 + nv12), ("P2", b720 + b640 * 6 + nv12),
                 ("P3", b720 + b640 * 4 + nv12), ("P4", b720 + b640 * 4 + nv12)]:
        print(f"  {k:<4} {v:5.1f} MB/帧   @30fps {v*30:6.0f} MB/s")
    if a.out:
        json.dump({"calibration": str(calib), "fx": float(K[0, 0]), "enhance_source": src,
                   "ms_per_frame": res, "n_frames": len(frames)}, open(a.out, "w"), indent=1)
        print(f"\n💾 {a.out}")
    print("\n⚠️  本脚本只测**预处理**；端到端时延还要加上 BPU 推理与后处理。")


if __name__ == "__main__":
    main()
