"""掩膜版违停判定：YOLO 检测 + SAM2 掩膜 + 接地点几何判定。

判定分级（对每辆车的掩膜）:
    合规   —— 接地点 90% 以上落在允许区内，且不占道
    压线   —— 接地点有 10%~90% 落在区外（车头/车轮跨线）
    出界   —— 接地点 90% 以上在区外
    占道   —— 车身与禁停区(通道)重叠超过阈值

同时输出"框规则"的结果做对照，量化掩膜比框精确多少。

用法:
    python violation_mask.py --source <图片目录> --scenes scenes.json --scene school_row
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, r"D:\CV-Datatext-Pipeline\sam2-repo")
sys.path.insert(0, r"D:\bick\yolo-lib")

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from scipy import ndimage  # noqa: E402

CKPT = r"D:\CV-Datatext-Pipeline\models\sam2\sam2.1_hiera_tiny.pt"
CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

GREEN = (0, 210, 90)
YELLOW = (255, 190, 0)
RED = (255, 45, 45)
GRAY = (150, 150, 150)


def rasterize(poly, W, H):
    img = Image.new("1", (W, H), 0)
    ImageDraw.Draw(img).polygon([(x * W, y * H) for x, y in poly], fill=1)
    return np.array(img, dtype=bool)


def contour(mask):
    return mask & ~ndimage.binary_erosion(mask, iterations=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--scene", default="default")
    ap.add_argument("--out", default=r"D:\bick\violation_mask_out")
    ap.add_argument("--weights", default=r"D:\bick\relabel\weights\yolo11x.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--max-side", type=int, default=1400)
    ap.add_argument("--deny-tol", type=float, default=0.05)
    ap.add_argument("--box-tol", type=float, default=0.02)
    args = ap.parse_args()

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from ultralytics import YOLO

    scenes = json.load(open(args.scenes, encoding="utf-8-sig"))
    sc = scenes.get(args.scene)
    if sc is None:
        raise SystemExit("scenes.json 里没有场景 %r，现有场景: %s" %
                         (args.scene, ", ".join(scenes) or "(空)"))

    yolo = YOLO(args.weights)
    sam = build_sam2(CFG, CKPT, device="cuda")
    pred = SAM2ImagePredictor(sam)

    os.makedirs(args.out, exist_ok=True)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 30)
    except Exception:
        font = ImageFont.load_default()

    files = ([os.path.join(args.source, f) for f in sorted(os.listdir(args.source))
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]
             if os.path.isdir(args.source) else [args.source])

    rows = []
    stat = {"合规": 0, "压线": 0, "出界": 0, "占道": 0}
    box_bad_total = 0
    mask_bad_total = 0

    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        # 优先用该图自己的区域，没有才回落到场景默认
        per_img = (sc.get("images") or {}).get(stem) or {}
        allow = per_img.get("allow") or sc.get("allow")
        deny = per_img.get("deny") or sc.get("deny") or []
        if not allow:
            # 没标定区域：只检测、不判定（原始新图走这条路径）
            has_zone = False
            allow, deny = [], []
        else:
            has_zone = True

        im = Image.open(path).convert("RGB")
        im.thumbnail((args.max_side, args.max_side * 2))
        W, H = im.size
        arr = np.array(im)

        allow_r = rasterize(allow, W, H) if has_zone else np.zeros((H, W), dtype=bool)
        deny_r = np.zeros_like(allow_r)
        for poly in deny:
            deny_r |= rasterize(poly, W, H)

        res = yolo.predict(arr, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                           classes=[1], device=0, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        pred.set_image(arr)

        canvas = im.copy()
        d = ImageDraw.Draw(canvas)
        if has_zone:
            d.line([(x * W, y * H) for x, y in allow] +
                   [(allow[0][0] * W, allow[0][1] * H)], fill=(80, 160, 255), width=6)
            for poly in deny:
                d.line([(x * W, y * H) for x, y in poly] +
                       [(poly[0][0] * W, poly[0][1] * H)], fill=(255, 140, 0), width=5)

        n_bad = 0
        for bi, b in enumerate(boxes, 1):
            masks, scores, _ = pred.predict(box=b, multimask_output=False)
            m = np.asarray(masks)[0]
            if m.ndim == 3:
                m = m[0]
            m = m > 0.0
            if m.sum() < 200:
                continue

            if not has_zone:
                verdict, color = "未标定", GRAY
                c_frac = d_frac = box_out = -1.0
                contact = np.zeros_like(m)
                ys, xs = np.where(m)
            else:
                # 接地点：掩膜最低 2% 高度的那一条
                ys, xs = np.where(m)
                y_max = ys.max()
                band = max(2, int(0.02 * H))
                contact = m & (np.arange(H)[:, None] >= y_max - band + 1)
                c_frac = (contact & allow_r).sum() / max(1, contact.sum())
                d_frac = (m & deny_r).sum() / max(1, m.sum())

                if d_frac > args.deny_tol:
                    verdict, color = "占道", RED
                elif c_frac >= 0.9:
                    verdict, color = "合规", GREEN
                elif c_frac > 0.1:
                    verdict, color = "压线", YELLOW
                else:
                    verdict, color = "出界", RED
            stat[verdict] = stat.get(verdict, 0) + 1

            if has_zone:
                # 框规则对照：车身在区外的比例
                bm = np.zeros_like(allow_r)
                x1, y1, x2, y2 = [int(v) for v in b]
                bm[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = True
                box_out = 1 - (bm & allow_r).sum() / max(1, bm.sum())
                box_bad = box_out > args.box_tol
                box_bad_total += int(box_bad)
                mask_bad_total += int(verdict != "合规")

            # 画轮廓
            c = contour(m)
            canvas_arr = np.array(canvas)
            canvas_arr[c] = (np.array(color, dtype=np.float32) * 0.85 +
                             canvas_arr[c].astype(np.float32) * 0.15)
            canvas = Image.fromarray(canvas_arr)

            box_rule = ("—" if not has_zone
                        else ("违停" if box_bad else "合规"))
            rows.append(dict(image=stem, bike=bi, verdict=verdict,
                             contact_in=round(float(c_frac), 3),
                             deny=round(float(d_frac), 3),
                             box_out=round(float(box_out), 3),
                             box_rule=box_rule,
                             score=round(float(scores[0]), 3)))
            if has_zone and verdict != "合规":
                n_bad += 1

        d = ImageDraw.Draw(canvas)
        d.rectangle([0, 0, 700, 48], fill=(0, 0, 0))
        if has_zone:
            d.text((10, 6), "违停 %d / 共 %d（掩膜判定）" % (n_bad, len(boxes)),
                   fill=(255, 60, 60) if n_bad else (0, 220, 90), font=font)
            print("%-28s 合规 %2d  违停 %2d" % (stem, len(boxes) - n_bad, n_bad), flush=True)
        else:
            d.text((10, 6), "未标定区域 · 仅检测 %d 辆（不判定）" % len(boxes),
                   fill=(180, 180, 180), font=font)
            print("%-28s 未标定，仅检测 %2d 辆" % (stem, len(boxes)), flush=True)
        canvas.save(os.path.join(args.out, os.path.basename(path)), quality=90)

    if rows:
        with open(os.path.join(args.out, "per_bike.csv"), "w",
                  encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("\n按掩膜判定:", stat)
    print("掩膜判违停 %d 辆 | 框规则判违停 %d 辆 | 框的误报 %d 辆" %
          (mask_bad_total, box_bad_total, max(0, box_bad_total - mask_bad_total)))
    print("输出:", args.out)


if __name__ == "__main__":
    main()
