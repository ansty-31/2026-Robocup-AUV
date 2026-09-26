#!/usr/bin/env python3
"""按 tag 生成「某一期」的 YOLO pose 数据集（硬链接 + 过采样 + 负样本空标签）。

累加：第 N 期 = 前 N-1 期全部 ∪ 本期新增；难例硬链接复制成 `xxx__r2.jpg`；
负样本 = 空标签文件（ultralytics 视作背景图）；`eval_reserved.csv` 里的帧一律排除。
tag 来源：`pool_inventory.csv`（模型推断）+ `--human-tags`（人工覆盖自动值），水质来自 manifest。

用法：
    python scripts/2_train/build_stage_dataset.py --stage 1 --dry-run

    python scripts/2_train/build_stage_dataset.py --stage 2 \
        --base data/datasets/AUV_5_gate-pose.yolov8 \
        --add  data/datasets/AUV_5_labeled_stage1 data/datasets/AUV_5_labeled_stage2 \
        --negatives output/preview/auv5_pose/tables/pool/negatives_verified.csv

约定：`data.yaml` 的 `kpt_shape=[4,3]`、`flip_idx=[0,1,2,3]` 原样沿用基线，勿改。
过采样默认（可 --replicate 覆盖）：s_none=2, s_multi=2, g_part2=2, g_part3=2,
g_tube=2.5, w_mid=1.5, w_turbid=1.5；复制因子上限 3。
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
PREVIEW = REPO / "output" / "preview" / "auv5_pose"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")

DEFAULT_REPLICATE = {"s_none": 2, "s_multi": 2, "g_part2": 2, "g_part3": 2,
                     "g_tube": 2.5, "w_mid": 1.5, "w_turbid": 1.5}
MAX_REP = 3.0


def to_pool(name: str) -> str:
    """Roboflow 导出名 → 池子名；池子名原样返回。

    `AUV_5_gate-7_frame_001271_jpg.rf.TVM5NL.jpg` → `AUV_5_gate-7_frame_001271.jpg`
    """
    n = name[:-4] if name.endswith(".txt") else name
    n = re.sub(r"\.rf\.[A-Za-z0-9]+", "", n)          # 去 .rf.xxx
    n = re.sub(r"_jpg(\.[A-Za-z]+)$", r"\1", n)    # xxx_jpg.jpg → xxx.jpg
    return re.sub(r"_jpg$", "", n)                 # 已去 .txt 的标签名（无扩展名）


def link(src: Path, dst: Path) -> None:
    """硬链接（同盘零成本）；跨盘/不支持时退回复制"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def tags_of(name: str, inv: dict, human: dict) -> set[str]:
    """一帧的 tag 集合：人工 tag 覆盖自动推断"""
    if name in human:
        return set(human[name])
    r = inv.get(name)
    if not r:
        return set()
    t = {f"w_{ {'清水':'clear','中水':'mid','浊水':'turbid'}.get(r['water'],'unknown') }"}
    cls = r["cls"]
    t.add({"c4": "g_full4", "c3": "g_part3", "c2": "g_part2", "c1": "g_tube",
           "c0": "g_tube", "none": "s_none"}.get(cls, "g_unknown"))
    if int(r["n_det"]) == 0:
        t.add("s_none")
    if int(r["n_det"]) >= 2:
        t.add("s_multi")
    if r["sizebin"] == "close(>300)":
        t.add("s_close")
    return t


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2, 3], help="第几期")
    ap.add_argument("--base", default="data/datasets/AUV_5_gate-pose.yolov8",
                    help="已有标注数据集（基线，始终包含）")
    ap.add_argument("--add", nargs="*", default=[],
                    help="本期及前期新增的已标数据集目录（累加；多个用空格分隔）")
    ap.add_argument("--negatives", default=None,
                    help="人眼确认的无门帧 CSV（含 name 列）；对应图片从 --negatives-root 找")
    ap.add_argument("--negatives-root", default=None, help="负样本图片所在目录（默认取 CSV 里的 rendered 路径）")
    ap.add_argument("--inventory", default=str(PREVIEW / "tables" / "pool" / "pool_inventory.csv"))
    ap.add_argument("--human-tags", default=None,
                    help="人工 tag CSV：列 name,tags（tags 用 | 分隔），覆盖自动推断")
    ap.add_argument("--reserved", default=str(PREVIEW / "tables" / "pool" / "eval_reserved.csv"),
                    help="仅评测、必须排除的帧清单")
    ap.add_argument("--out", default=None,
                    help="输出根目录（默认 data/datasets/AUV_5_stage<N>）")
    ap.add_argument("--replicate", nargs="*", default=None, help="覆盖默认过采样，如 s_none=3 w_mid=2")
    ap.add_argument("--dry-run", action="store_true", help="只打印方案，不建数据集")
    a = ap.parse_args()

    # 默认产物落 data/datasets/AUV_5_stage<N>
    out = Path(a.out) if a.out else REPO / "data" / "datasets" / f"AUV_5_stage{a.stage}"
    repl = dict(DEFAULT_REPLICATE)
    for item in (a.replicate or []):
        k, _, v = item.partition("=")
        repl[k.strip()] = min(float(v), MAX_REP)

    inv = {r["name"]: r for r in csv.DictReader(open(a.inventory, encoding="utf-8"))}
    human: dict[str, list[str]] = {}
    if a.human_tags:
        for r in csv.DictReader(open(a.human_tags, encoding="utf-8")):
            human[r["name"]] = [t for t in r["tags"].replace(",", "|").split("|") if t]
    reserved = set()
    rp = Path(a.reserved)
    if rp.exists():
        reserved = {r["name"] for r in csv.DictReader(open(rp, encoding="utf-8"))}

    # ---------- 收集来源：(图片路径, 标签路径, tags, 来源名) ----------
    items: list[tuple[Path, Path | None, set[str], str]] = []

    def take_dataset(root: Path, label: str) -> None:
        n = 0
        for sp in ("train", "valid", "test"):
            for img in sorted((root / sp / "images").glob("*")):
                if img.suffix.lower() not in IMG_EXT:
                    continue
                lab = root / sp / "labels" / (img.stem + ".txt")
                # 沿用来源目录的 split（重划会引入相邻帧泄漏）
                items.append((img, lab if lab.exists() else None,
                              tags_of(to_pool(img.name), inv, human), label,
                              "valid" if sp == "valid" else "train"))
                n += 1
        print(f"  {label}: {n} 张（{root}）")

    print("📦 收集来源：")
    take_dataset(Path(a.base), "base(已有标注)")
    for d in a.add:
        take_dataset(Path(d), f"add({Path(d).name})")
    negs = []
    if a.negatives:
        # 两种格式都收：triage 日志（bucket=n）或手填表（human_no_gate=1）
        neg_rows = list(csv.DictReader(open(a.negatives, encoding="utf-8")))
        is_log = any("bucket" in r for r in neg_rows)
        default_root = PREVIEW / "renders" / "negatives"
        for r in neg_rows:
            if is_log:
                if r.get("bucket") != "n":
                    continue
            elif str(r.get("human_no_gate", "")).lower() not in ("1", "1.0", "yes", "y", "true"):
                continue
            name = r["name"]
            p = None
            for base in ([Path(a.negatives_root)] if a.negatives_root else []) + \
                        [default_root, None]:
                if base is None:
                    if r.get("rendered_path"):
                        cand = Path(r["rendered_path"])
                        p = cand if cand.exists() else None
                    break
                for cand_name in (name, to_pool(name)):
                    cand = base / cand_name
                    if cand.exists():
                        p = cand
                        break
                if p:
                    break
            if p and p.exists():
                negs.append(p)
        for p in negs:
            # 负样本按名字确定性分 10% 进 valid（可复现）
            sp = "valid" if (hash(p.name) % 10 == 0) else "train"
            items.append((p, None, {"s_none"}, "negative", sp))
        print(f"  负样本: {len(negs)} 张（空标签 = 背景图）")

    # ---------- 排除保留集 ----------
    before = len(items)
    items = [it for it in items if to_pool(it[0].name) not in reserved]
    if before != len(items):
        print(f"🚫 排除保留评测帧 {before - len(items)} 张（eval_reserved.csv）")

    # ---------- 过采样（硬链接复制） ----------
    expanded = []
    for img, lab, tags, src, sp in items:
        rep = 1.0
        for t in tags:
            if t in repl:
                rep = max(rep, repl[t])
        rep = min(rep, MAX_REP)
        n_copy = max(1, int(round(rep)))
        for k in range(1, n_copy + 1):
            expanded.append((img, lab, tags, src, sp, k))

    plan = Counter()
    per_tag = Counter()
    for img, lab, tags, src, sp, k in expanded:
        split = sp
        plan[f"{split}"] += 1
        for t in sorted(tags):
            per_tag[t] += 1

    print(f"\n📊 方案（stage {a.stage}）：共 {len(expanded)} 张"
          f"（唯一原图 {len({e[0].name for e in expanded})} 张，过采样展开后）")
    for k_, v in sorted(plan.items()):
        print(f"    {k_}: {v}")
    print("    tag 计数（含复制）：" + ", ".join(f"{t}={c}" for t, c in sorted(per_tag.items())))
    neg_share = per_tag.get("s_none", 0) / max(len(expanded), 1)
    print(f"    负样本占比 {100*neg_share:.1f}%（目标 10~15%）")

    if a.dry_run:
        print("\n（--dry-run：未写盘）")
        return

    # ---------- 落盘 ----------
    if out.exists():
        print(f"♻️  清空已存在的 {out}")
        shutil.rmtree(out)
    dst_rows = []
    for img, lab, tags, src, sp, k in expanded:
        split = sp
        stem = img.stem if k == 1 else f"{img.stem}__r{k}"
        d_img = out / split / "images" / (stem + img.suffix)
        d_lab = out / split / "labels" / (stem + ".txt")
        link(img, d_img)
        if lab is not None and lab.exists():
            link(lab, d_lab)
        else:
            d_lab.parent.mkdir(parents=True, exist_ok=True)
            d_lab.write_text("", encoding="utf-8")       # 空标签 = 背景图（负样本）
        dst_rows.append(dict(stem=stem, src_name=img.name, split=split, source=src,
                             rep=k, tags="|".join(sorted(tags))))

    # data.yaml：增广约定沿用基线（kpt_shape / flip_idx）
    base_yaml = Path(a.base) / "data.yaml"
    cfg = yaml.safe_load(open(base_yaml, encoding="utf-8"))
    cfg.update(train="train/images", val="valid/images",
               test="valid/images")
    (out / "data.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                                   encoding="utf-8")
    with open(out / "tags.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dst_rows[0].keys()))
        w.writeheader()
        w.writerows(dst_rows)

    print(f"\n✅ 已生成 {out}")
    print(f"   data.yaml: kpt_shape={cfg.get('kpt_shape')} flip_idx={cfg.get('flip_idx')} "
          f"nc={cfg.get('nc')} names={cfg.get('names')}  ← 已沿用基线，勿改")
    print(f"   tags.csv: {len(dst_rows)} 行（逐图 tag，供分层评测与下期加权）")
    try:
        shown = out.relative_to(REPO)
    except ValueError:                     # --out 指向仓库外
        shown = out
    print(f"\n下一步：\n"
          f"  /home/ansty/anaconda3/envs/yolov8/bin/python scripts/2_train/train_yolo11n.py \\\n"
          f"      --task pose --data {shown}/data.yaml \\\n"
          f"      --weights {'weights/yolo11n-pose.coco.pt' if a.stage == 1 else 'weights/yolo11n-pose.pt'} \\\n"
          f"      --epochs {300 if a.stage == 1 else 60} --batch 4 --device 0")


if __name__ == "__main__":
    main()
