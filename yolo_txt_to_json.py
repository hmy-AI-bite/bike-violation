#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 YOLO 数据集里的 .txt 标注逐个转换成同名 .json（保留原 txt 不动）。

用法:
    python yolo_txt_to_json.py                 # 默认处理 D:\\bick\\bike_dataset
    python yolo_txt_to_json.py <dataset_dir>

输入: <dataset>/labels/{train,val}/image_xxxxxx.txt
输出: <dataset>/labels/{train,val}/image_xxxxxx.json
"""

import json
import os
import sys
from struct import unpack

ROOT = sys.argv[1] if len(sys.argv) > 1 else r"D:\bick\bike_dataset"
SPLITS = ["train", "val"]
INDENT = 2


def jpeg_size(path):
    """从 JPEG 文件头读出 (width, height)。"""
    with open(path, "rb") as f:
        data = f.read()
    i = 2
    n = len(data)
    while i < n - 9:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        # SOF0..SOF3, SOF5..SOF7, SOF9..SOF11, SOF13..SOF15
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                      0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h, w = unpack(">HH", data[i + 5:i + 9])
            return w, h
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seg_len = unpack(">H", data[i + 2:i + 4])[0]
        i += 2 + seg_len
    raise ValueError("无法解析 JPEG 尺寸: %s" % path)


def read_class_names(dataset_dir):
    """尽量从 data.yaml 的 names 里读出类别名。"""
    names = {}
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        return names
    text = open(yaml_path, encoding="utf-8", errors="replace").read()
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("names"):
            continue
        raw = line.split(":", 1)[1].strip()
        try:
            value = json.loads(raw.replace("'", '"'))
        except ValueError:
            continue
        if isinstance(value, list):
            names = {i: str(v) for i, v in enumerate(value)}
        elif isinstance(value, dict):
            names = {int(k): str(v) for k, v in value.items()}
    return names


def main():
    class_names = read_class_names(ROOT)
    total_files = 0
    total_boxes = 0
    skipped = []

    for split in SPLITS:
        label_dir = os.path.join(ROOT, "labels", split)
        image_dir = os.path.join(ROOT, "images", split)
        if not os.path.isdir(label_dir):
            continue

        for name in sorted(os.listdir(label_dir)):
            if not name.lower().endswith(".txt"):
                continue
            stem = os.path.splitext(name)[0]
            txt_path = os.path.join(label_dir, name)
            image_name = None
            width = height = None

            for ext in (".jpg", ".jpeg", ".png"):
                candidate = os.path.join(image_dir, stem + ext)
                if os.path.exists(candidate):
                    image_name = stem + ext
                    if ext in (".jpg", ".jpeg"):
                        width, height = jpeg_size(candidate)
                    break

            if image_name is None:
                skipped.append("%s/%s (找不到对应图片)" % (split, name))
                continue
            if width is None:
                skipped.append("%s/%s (无法读取图片尺寸)" % (split, name))
                continue

            objects = []
            with open(txt_path, encoding="utf-8-sig") as f:
                for line_no, line in enumerate(f, 1):
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        class_id = int(float(parts[0]))
                        xc, yc, bw, bh = (float(v) for v in parts[1:5])
                    except (ValueError, IndexError):
                        skipped.append("%s/%s 第 %d 行格式异常" % (split, name, line_no))
                        continue

                    x1 = (xc - bw / 2.0) * width
                    y1 = (yc - bh / 2.0) * height
                    x2 = (xc + bw / 2.0) * width
                    y2 = (yc + bh / 2.0) * height

                    objects.append({
                        "id": len(objects) + 1,
                        "class_id": class_id,
                        "class_name": class_names.get(class_id, str(class_id)),
                        "x_center": xc,
                        "y_center": yc,
                        "width": bw,
                        "height": bh,
                        "bbox_xyxy": [round(x1, 2), round(y1, 2),
                                      round(x2, 2), round(y2, 2)],
                        "out_of_bounds": not all(
                            -1e-9 <= v <= 1 + 1e-9 for v in (xc, yc, bw, bh)
                        ),
                    })

            payload = {
                "image": image_name,
                "split": split,
                "width": width,
                "height": height,
                "label_file": "labels/%s/%s" % (split, name),
                "class_names": class_names,
                "count": len(objects),
                "objects": objects,
            }

            out_path = os.path.join(label_dir, stem + ".json")
            with open(out_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(payload, f, ensure_ascii=False, indent=INDENT)
                f.write("\n")

            total_files += 1
            total_boxes += len(objects)

    print("转换完成: %d 个 json, 共 %d 个标注框" % (total_files, total_boxes))
    if skipped:
        print("跳过/异常 %d 条:" % len(skipped))
        for s in skipped:
            print("  -", s)


if __name__ == "__main__":
    main()
