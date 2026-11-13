"""按内容哈希去重后重建数据集（避免同一张图同时出现在 train/val 造成泄漏）。

划分规则: 用图片哈希做确定性切分（同一张图整组进同一个 split），约 80/20。

用法:
    python build_v2.py --labels relabel/auto/labels --out dataset_v2
"""
import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict

IMG_EXTS = (".jpg", ".jpeg", ".png")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"D:\bick\bike_dataset")
    ap.add_argument("--labels", default=r"D:\bick\relabel\auto\labels")
    ap.add_argument("--out", default=r"D:\bick\dataset_v2")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    args = ap.parse_args()

    groups = defaultdict(list)
    for split in ("train", "val"):
        idir = os.path.join(args.src, "images", split)
        for fn in sorted(os.listdir(idir)):
            if fn.lower().endswith(IMG_EXTS):
                groups[md5(os.path.join(idir, fn))].append((split, fn))

    stat = {"train": 0, "val": 0}
    dupes = 0
    for hv, members in sorted(groups.items()):
        # 优先保留 train 里的那份，命名沿用原文件名
        members_sorted = sorted(members, key=lambda m: (m[0] != "train", m[1]))
        split0, fname = members_sorted[0]
        dupes += len(members) - 1

        bucket = "val" if int(hv[:8], 16) % 100 < args.val_ratio * 100 else "train"
        stem = os.path.splitext(fname)[0]

        dst_img = os.path.join(args.out, "images", bucket, fname)
        dst_lab = os.path.join(args.out, "labels", bucket, stem + ".txt")
        os.makedirs(os.path.dirname(dst_img), exist_ok=True)
        os.makedirs(os.path.dirname(dst_lab), exist_ok=True)

        shutil.copy2(os.path.join(args.src, "images", split0, fname), dst_img)
        src_lab = os.path.join(args.labels, split0, stem + ".txt")
        with open(dst_lab, "w", encoding="utf-8") as f:
            if os.path.exists(src_lab):
                f.write(open(src_lab, encoding="utf-8-sig").read())
        stat[bucket] += 1

    yaml_path = os.path.join(args.out, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# 自行车检测：yolo11x 预训练模型自动标注（待人工复核）+ 按哈希去重\n")
        f.write("path: %s\n" % os.path.abspath(args.out).replace("\\", "/"))
        f.write("train: images/train\nval: images/val\n\nnc: 1\nnames: ['bicycle']\n")

    print("唯一图片 %d 张, 丢弃重复 %d 个" % (sum(stat.values()), dupes))
    print("train %d / val %d" % (stat["train"], stat["val"]))
    print("输出:", args.out)
    with open(os.path.join(args.out, "split.json"), "w", encoding="utf-8") as f:
        json.dump(stat, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
