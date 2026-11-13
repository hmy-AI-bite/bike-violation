"""学校实拍数据集 -> 可训练的 YOLO 检测数据集。

原标注:
    0 cx cy w h              bicycle (bbox)
    1 x1 y1 x2 y2 x3 y3 x4 y4  parking_area (4 点多边形)

处理:
    * parking_area 多边形转外接矩形后按 bbox 训练（多边形的原始坐标另存一份，将来要做分割可直接用）
    * 按文件名排序，每 6 张取 1 张作验证集（帧连续，均匀抽）
    * 输出 dataset_school/ 结构

用法:
    python school_to_yolo.py
"""
import argparse
import json
import os
import shutil
from collections import Counter

SRC = r"J:\project-1-at-2026-11-13-17-44-a1d4a70f"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=r"D:\bick\dataset_school")
    ap.add_argument("--val-every", type=int, default=6)
    args = ap.parse_args()

    img_dir = os.path.join(args.src, "images")
    lab_dir = os.path.join(args.src, "labels")
    names = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))

    stat = Counter()
    polygons = {}

    for idx, fn in enumerate(names):
        stem = os.path.splitext(fn)[0]
        split = "val" if idx % args.val_every == args.val_every - 1 else "train"

        dst_img = os.path.join(args.out, "images", split, fn)
        dst_lab = os.path.join(args.out, "labels", split, stem + ".txt")
        os.makedirs(os.path.dirname(dst_img), exist_ok=True)
        os.makedirs(os.path.dirname(dst_lab), exist_ok=True)
        shutil.copy2(os.path.join(img_dir, fn), dst_img)

        out_lines, polys = [], []
        with open(os.path.join(lab_dir, stem + ".txt"), encoding="utf-8") as f:
            for line in f:
                q = line.split()
                if len(q) < 5:
                    continue
                c = int(float(q[0]))
                vals = [float(v) for v in q[1:]]
                if c == 0:
                    xc, yc, w, h = vals[:4]
                    out_lines.append("0 %.6f %.6f %.6f %.6f" % (xc, yc, w, h))
                    stat["bicycle"] += 1
                else:
                    pts = [(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)]
                    polys.append(pts)
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    x1, x2 = max(0.0, min(xs)), min(1.0, max(xs))
                    y1, y2 = max(0.0, min(ys)), min(1.0, max(ys))
                    out_lines.append("1 %.6f %.6f %.6f %.6f" %
                                     ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
                    stat["parking_area"] += 1
        with open(dst_lab, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")
        polygons[stem] = polys
        stat["images"] += 1
        stat["split_" + split] += 1

    with open(os.path.join(args.out, "polygons.json"), "w", encoding="utf-8") as f:
        json.dump(polygons, f, ensure_ascii=False, indent=1)

    with open(os.path.join(args.out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("# 学校自行车停放区检测\n")
        f.write("path: %s\n" % os.path.abspath(args.out).replace("\\", "/"))
        f.write("train: images/train\nval: images/val\n\n")
        f.write("nc: 2\nnames: ['bicycle', 'parking_area']\n")

    print("图片 %d (train %d / val %d)" % (stat["images"], stat["split_train"], stat["split_val"]))
    print("bicycle 框 %d, parking_area %d" % (stat["bicycle"], stat["parking_area"]))
    print("输出:", args.out)


if __name__ == "__main__":
    main()
