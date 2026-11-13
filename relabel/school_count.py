"""学校场景：检测自行车 + 按固定停放区统计数量。

停放区用标注里的多边形（相机固定，区域不需要模型去学），
只让模型负责"找出每一辆车"，工程上更稳。

用法:
    python school_count.py --weights runs/school_v3/weights/best.pt
输出:
    D:\\bick\\school_infer\\*.jpg  带框图
    D:\\bick\\school_infer\\counts.csv
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, r"D:\bick\yolo-lib")

SRC = r"J:\project-1-at-2026-11-13-17-44-a1d4a70f"
POLY = r"D:\bick\dataset_school\polygons.json"


def in_poly(x, y, pts):
    """射线法判断点是否在多边形内。"""
    inside = False
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < xin:
                inside = not inside
    return inside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=r"D:\bick\yolo-lib")
    ap.add_argument("--weights", default=r"D:\bick\runs\school_v3\weights\best.pt")
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=r"D:\bick\school_infer")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    sys.path.insert(0, args.lib)
    from PIL import Image, ImageDraw, ImageFont
    from ultralytics import YOLO

    polys = json.load(open(POLY, encoding="utf-8"))
    model = YOLO(args.weights)
    os.makedirs(args.out, exist_ok=True)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 30)
    except Exception:
        font = ImageFont.load_default()

    img_dir = os.path.join(args.src, "images")
    lab_dir = os.path.join(args.src, "labels")
    rows = []

    for fn in sorted(os.listdir(img_dir)):
        if not fn.lower().endswith(".jpg"):
            continue
        stem = os.path.splitext(fn)[0]
        path = os.path.join(img_dir, fn)

        # 真值
        gt_bikes, gt_inside = 0, 0
        gt_polys = polys.get(stem, [])
        for line in open(os.path.join(lab_dir, stem + ".txt"), encoding="utf-8"):
            q = line.split()
            if len(q) >= 5 and int(float(q[0])) == 0:
                gt_bikes += 1
                xc, yc = float(q[1]), float(q[2])
                if gt_polys and in_poly(xc, yc, gt_polys[0]):
                    gt_inside += 1

        res = model.predict(path, imgsz=args.imgsz, conf=args.conf, classes=[0],
                            device=args.device if hasattr(args, "device") else 0, verbose=False)[0]
        im = Image.open(path).convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)
        pred_bikes, pred_inside = 0, 0
        pts_px = [(x * W, y * H) for x, y in gt_polys[0]] if gt_polys else []
        if pts_px:
            d.line(pts_px + [pts_px[0]], fill=(255, 60, 60), width=6)
        for box in res.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = [float(v) for v in box]
            pred_bikes += 1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            inside = bool(pts_px) and in_poly(cx, cy, pts_px)
            pred_inside += int(inside)
            color = (0, 220, 90) if inside else (255, 190, 0)
            d.rectangle([x1, y1, x2, y2], outline=color, width=5)
        d.rectangle([0, 0, 620, 46], fill=(0, 0, 0))
        d.text((8, 8), "预测 %d 辆 (区内 %d) / 真值 %d 辆 (区内 %d)"
               % (pred_bikes, pred_inside, gt_bikes, gt_inside), fill=(255, 255, 0), font=font)
        im.save(os.path.join(args.out, fn), quality=90)

        rows.append(dict(file=fn, pred=pred_bikes, gt=gt_bikes,
                         pred_inside=pred_inside, gt_inside=gt_inside))
        print("%-26s 预测 %2d (区内 %2d) | 真值 %2d (区内 %2d)" %
              (fn, pred_bikes, pred_inside, gt_bikes, gt_inside))

    with open(os.path.join(args.out, "counts.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    tp = sum(r["pred"] for r in rows)
    tg = sum(r["gt"] for r in rows)
    print("\n合计: 预测 %d 辆 / 真值 %d 辆, 相差 %+d (%.1f%%)" %
          (tp, tg, tp - tg, 100 * (tp - tg) / tg))
    print("输出:", args.out)


if __name__ == "__main__":
    main()
