"""用 COCO 预训练 YOLO 对数据集重新推理，输出 YOLO 格式 txt（只保留 bicycle）。

COCO 里 bicycle 的类别 id 是 1，输出时重映射为本地类别 0。

用法:
    python autolabel.py --weights relabel/weights/yolo11x.pt \
                        --src bike_dataset \
                        --out relabel/auto \
                        --imgsz 1280 --conf 0.15 --device 0
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, r"D:\bick\yolo-lib")

COCO_BICYCLE = 1
COCO_MOTORCYCLE = 3
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def draw_preview(src, bikes, motos, out_dir, stem, scale=1.0):
    """把检出的框画到图上：绿=高置信自行车，黄=可疑，红=低置信，蓝=摩托车（提示别标成自行车）。"""
    from PIL import Image, ImageDraw, ImageFont

    os.makedirs(out_dir, exist_ok=True)
    img = Image.open(src).convert("RGB")
    W, H = img.size
    if scale != 1.0:
        img = img.resize((max(1, int(W * scale)), max(1, int(H * scale))))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 14)
    except Exception:
        font = ImageFont.load_default()

    for i, b in enumerate(bikes, 1):
        x1, y1, x2, y2 = b["box"]
        c = b["conf"]
        color = (0, 200, 80) if c >= 0.5 else ((255, 190, 0) if c >= 0.35 else (255, 50, 50))
        d.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = "#%d %.2f" % (i, c)
        d.text((x1 + 2, max(1, y1 + 1)), label, fill=color, font=font)

    for m in motos:
        x1, y1, x2, y2 = m["box"]
        d.rectangle([x1, y1, x2, y2], outline=(80, 160, 255), width=2)
        d.text((x1 + 2, max(1, y1 + 1)), "moto", fill=(80, 160, 255), font=font)

    d.rectangle([0, 0, 260, 20], fill=(0, 0, 0))
    d.text((3, 2), "%s  %d bike / %d moto" % (stem, len(bikes), len(motos)),
           fill=(255, 255, 0), font=font)
    img.save(os.path.join(out_dir, stem + ".jpg"), quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=r"D:\bick\yolo-lib")
    ap.add_argument("--weights", default=r"D:\bick\relabel\weights\yolo11x.pt")
    ap.add_argument("--src", default=r"D:\bick\bike_dataset")
    ap.add_argument("--out", default=r"D:\bick\relabel\auto")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-det", type=int, default=200)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--save-preview", action="store_true",
                    help="同时输出带框的可视化图到 <out>/preview/<split>/")
    ap.add_argument("--preview-scale", type=float, default=1.0,
                    help="预览图缩放比例（大图设 0.6 省空间）")
    args = ap.parse_args()

    sys.path.insert(0, args.lib)
    from ultralytics import YOLO

    model = YOLO(args.weights)
    print("模型:", os.path.basename(args.weights), "| imgsz:", args.imgsz,
          "| conf:", args.conf, "| device:", args.device)

    records = []
    t0 = time.time()
    done = 0
    total = 0
    for split in args.splits.split(","):
        img_dir = os.path.join(args.src, "images", split)
        if not os.path.isdir(img_dir):
            continue
        total += len([f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS)])

    for split in args.splits.split(","):
        img_dir = os.path.join(args.src, "images", split)
        lab_dir = os.path.join(args.out, "labels", split)
        os.makedirs(lab_dir, exist_ok=True)
        files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS))

        for fname in files:
            path = os.path.join(img_dir, fname)
            res = model.predict(path, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                                classes=[COCO_BICYCLE, COCO_MOTORCYCLE], device=args.device,
                                max_det=args.max_det, verbose=False, augment=False)[0]
            h, w = res.orig_shape
            bikes, motos = [], []
            if res.boxes is not None and len(res.boxes):
                for box, cls, cf in zip(res.boxes.xyxy.cpu().numpy(),
                                        res.boxes.cls.cpu().numpy(),
                                        res.boxes.conf.cpu().numpy()):
                    x1, y1, x2, y2 = [float(v) for v in box]
                    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                    bw, bh = (x2 - x1) / w, (y2 - y1) / h
                    item = dict(conf=float(cf), box=[x1, y1, x2, y2],
                                yolo=[round(cx, 6), round(cy, 6), round(bw, 6), round(bh, 6)])
                    (bikes if int(cls) == COCO_BICYCLE else motos).append(item)

            stem = os.path.splitext(fname)[0]
            with open(os.path.join(lab_dir, stem + ".txt"), "w", encoding="utf-8") as f:
                for b in bikes:
                    f.write("0 " + " ".join("%.6f" % v for v in b["yolo"]) + "\n")

            if args.save_preview:
                draw_preview(path, bikes, motos, os.path.join(args.out, "preview", split),
                             stem, args.preview_scale)

            records.append(dict(split=split, image=fname, width=w, height=h,
                                n_bicycle=len(bikes), n_motorcycle=len(motos),
                                bicycles=bikes, motorcycles=motos))
            done += 1
            if done % 25 == 0 or done == total:
                el = time.time() - t0
                print("  %d/%d  已用 %.0fs  预计剩余 %.0fs" %
                      (done, total, el, el / done * (total - done)), flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "detections.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    n_box = sum(r["n_bicycle"] for r in records)
    n_moto = sum(r["n_motorcycle"] for r in records)
    print("\n完成: %d 张图, %d 个 bicycle 框, %d 个 motorcycle 框(仅用于排查误标)" %
          (len(records), n_box, n_moto))
    print("输出目录:", args.out)


if __name__ == "__main__":
    main()
