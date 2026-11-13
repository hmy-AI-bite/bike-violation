"""训练前数据体检：检查 YOLO 数据集能不能直接拿去训练。

用法:
    python check_dataset.py --data D:\\bick\\dataset_school\\data.yaml
"""
import argparse
import os
import sys
from collections import Counter

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def read_yaml_lite(path):
    """不依赖 pyyaml，够读 ultralytics 的 data.yaml。"""
    out = {}
    for line in open(path, encoding="utf-8-sig"):
        s = line.strip()
        if not s or s.startswith("#") or ":" not in s:
            continue
        k, v = s.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()

    cfg = read_yaml_lite(args.data)
    root = cfg.get("path", "").strip("'\"")
    if root and not os.path.isabs(root):
        root = os.path.join(os.path.dirname(os.path.abspath(args.data)), root)
    if not root:
        root = os.path.dirname(os.path.abspath(args.data))
    root = root.replace("/", os.sep)

    nc = int(cfg.get("nc", "1").strip("'\""))
    names_raw = cfg.get("names", "")
    print("=" * 60)
    print("数据集:", args.data)
    print("根目录:", root)
    print("类别数:", nc, "| 类别名:", names_raw)
    print("=" * 60)

    problems, warns = [], []
    total_img = total_lab = total_box = 0
    cls_counter = Counter()
    sizes = []

    for split in ("train", "val"):
        rel = cfg.get(split, "").strip("'\"")
        if not rel:
            continue
        d = os.path.join(root, rel.replace("/", os.sep))
        if not os.path.isdir(d):
            problems.append("找不到 %s 目录: %s" % (split, d))
            continue

        images = [f for f in os.listdir(d) if f.lower().endswith(IMG_EXTS)]
        lab_dir = os.path.join(root, "labels", split)
        labels = ([f for f in os.listdir(lab_dir) if f.endswith(".txt")]
                  if os.path.isdir(lab_dir) else [])

        print("\n[%s] 图片 %d 张, 标注 %d 个" % (split, len(images), len(labels)))
        if not labels:
            problems.append("%s 没有标注目录，ultralytics 会报错" % split)
            continue

        img_stems = {os.path.splitext(f)[0] for f in images}
        lab_stems = {f[:-4] for f in labels}
        no_label = img_stems - lab_stems
        no_img = lab_stems - img_stems
        if no_label:
            warns.append("%s: %d 张图没有标注文件（会被当背景图）" % (split, len(no_label)))
        if no_img:
            problems.append("%s: %d 个标注没有对应图片: %s" %
                            (split, len(no_img), sorted(no_img)[:3]))

        empty = 0
        for f in labels:
            path = os.path.join(lab_dir, f)
            lines = [l for l in open(path, encoding="utf-8-sig") if l.strip()]
            if not lines:
                empty += 1
            for i, line in enumerate(lines, 1):
                q = line.split()
                if len(q) not in (5, 9) and len(q) < 5:
                    problems.append("%s/%s 第 %d 行字段数 %d（应为 5）" % (split, f, i, len(q)))
                    continue
                try:
                    cls = int(float(q[0]))
                    vals = [float(v) for v in q[1:5]]
                except ValueError:
                    problems.append("%s/%s 第 %d 行不是数字" % (split, f, i))
                    continue
                cls_counter[cls] += 1
                if cls < 0 or cls >= nc:
                    problems.append("%s/%s 类别 %d 超出 nc=%d" % (split, f, cls, nc))
                if any(v < 0 or v > 1 for v in vals):
                    problems.append("%s/%s 坐标越界: %s" % (split, f, [round(v, 3) for v in vals]))
                if vals[2] <= 0 or vals[3] <= 0:
                    problems.append("%s/%s 宽或高 <= 0" % (split, f))
                total_box += 1
        if empty:
            warns.append("%s: %d 个空标注文件（当负样本用，正常）" % (split, empty))
        total_img += len(images)
        total_lab += len(labels)

    print("\n" + "-" * 60)
    print("合计: 图片 %d, 标注 %d, 框 %d" % (total_img, total_lab, total_box))
    print("类别分布:", dict(sorted(cls_counter.items())))

    missing_cls = [c for c in range(nc) if cls_counter.get(c, 0) == 0]
    if missing_cls:
        warns.append("这些类别一个框都没有: %s（训练会得到 mAP=0）" % missing_cls)

    if total_img < 50:
        warns.append("图片只有 %d 张，属于极小数据集：mAP 波动会很大" % total_img)

    print()
    if problems:
        print("❌ 必须修的问题 %d 个（前 10 条）:" % len(problems))
        for p in problems[:10]:
            print("   -", p)
    else:
        print("✅ 结构检查通过，可以训练")

    if warns:
        print("\n⚠️ 提醒 %d 条:" % len(warns))
        for w in warns:
            print("   -", w)

    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
