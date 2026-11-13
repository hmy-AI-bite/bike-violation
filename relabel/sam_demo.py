"""YOLO 出框 + SAM2 出掩膜：拿到每辆自行车的精确轮廓。

用法: python sam_demo.py <image> [out.jpg]
"""
import os
import sys

sys.path.insert(0, r"D:\CV-Datatext-Pipeline\sam2-repo")
sys.path.insert(0, r"D:\bick\yolo-lib")

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

CKPT = r"D:\CV-Datatext-Pipeline\models\sam2\sam2.1_hiera_tiny.pt"
CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else \
        r"J:\project-1-at-2026-11-13-17-44-a1d4a70f\images\039f0796-A_013.jpg"
    out = sys.argv[2] if len(sys.argv) > 2 else r"D:\bick\relabel\report\sam_demo.jpg"

    from ultralytics import YOLO
    yolo = YOLO(r"D:\bick\relabel\weights\yolo11x.pt")
    im = Image.open(src).convert("RGB")
    W, H = im.size
    small = im.copy()
    small.thumbnail((1400, 1900))
    sw, sh = small.size

    res = yolo.predict(np.array(small), imgsz=1280, conf=0.35, iou=0.45,
                       classes=[1], device=0, verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy()
    print("检出自行车:", len(boxes))

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam = build_sam2(CFG, CKPT, device="cuda")
    pred = SAM2ImagePredictor(sam)
    pred.set_image(np.array(small))

    d = ImageDraw.Draw(small, "RGBA")
    arr = np.array(small).astype(np.float32)
    from scipy import ndimage
    for i, b in enumerate(boxes):
        masks, scores, _ = pred.predict(box=b, multimask_output=False)
        m = np.asarray(masks)[0]
        if m.ndim == 3:
            m = m[0]
        m = m > 0.0
        print("  框 %d: mask 得分 %.3f, 面积 %d px" % (i + 1, float(scores[0]), int(m.sum())))
        if m.sum() == 0:
            continue
        # 轮廓：腐蚀后的差集，画在图上（红色描边）
        edges = ndimage.binary_erosion(m) ^ m
        arr[edges] = arr[edges] * 0.2 + np.array([255.0, 40.0, 40.0]) * 0.8
        d.rectangle([b[0], b[1], b[2], b[3]], outline=(255, 255, 0, 160), width=2)

    small = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    small.save(out, quality=90)
    print("保存:", out)


if __name__ == "__main__":
    main()
