"""把「图片 + 审阅后的标注」组装成可直接训练的 YOLO 数据集。

用法:
    python build_dataset.py --labels relabel/auto/labels --out dataset_v2
    python build_dataset.py --labels dataset_v2/labels --out dataset_v2   # 只重建 data.yaml
"""
import argparse
import os
import shutil

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"D:\bick\bike_dataset")
    ap.add_argument("--labels", default=r"D:\bick\relabel\auto\labels")
    ap.add_argument("--out", default=r"D:\bick\dataset_v2")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--skip-images", action="store_true")
    args = ap.parse_args()

    for split in args.splits.split(","):
        src_img = os.path.join(args.src, "images", split)
        src_lab = os.path.join(args.labels, split)
        dst_img = os.path.join(args.out, "images", split)
        dst_lab = os.path.join(args.out, "labels", split)
        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_lab, exist_ok=True)

        n_img = n_lab = n_box = 0
        for fn in sorted(os.listdir(src_lab)):
            if not fn.endswith(".txt"):
                continue
            stem = fn[:-4]
            src_txt = os.path.join(src_lab, fn)
            shutil.copy2(src_txt, os.path.join(dst_lab, fn))
            n_lab += 1
            with open(src_txt, encoding="utf-8-sig") as f:
                n_box += sum(1 for line in f if line.strip())

            if not args.skip_images:
                for ext in IMG_EXTS:
                    cand = os.path.join(src_img, stem + ext)
                    if os.path.exists(cand):
                        shutil.copy2(cand, os.path.join(dst_img, stem + ext))
                        n_img += 1
                        break
        print("%-6s 图片 %d, 标注 %d, 框 %d" % (split, n_img, n_lab, n_box))

    yaml_path = os.path.join(args.out, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# 自行车检测数据集（预训练模型重标 + 人工复核）\n")
        f.write("path: %s\n" % os.path.abspath(args.out).replace("\\", "/"))
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("nc: 1\nnames: ['bicycle']\n")
    print("\ndata.yaml ->", yaml_path)


if __name__ == "__main__":
    main()
