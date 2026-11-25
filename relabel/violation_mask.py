"""掩膜版违停判定：YOLO 检测 + SAM2 掩膜 + 接地点几何判定。

判定分级（对每辆车的掩膜）:
    合规   —— 接地点 90% 以上落在允许区内，且不占道
    压线   —— 接地点有 10%~90% 落在区外（车头/车轮跨线）
    出界   —— 接地点 90% 以上在区外
    占道   —— 车身与禁停区(通道)重叠超过阈值

同时输出"框规则"的结果做对照，比较两套规则的差异（误报需人工真值评估）。

用法:
    python violation_mask.py --source <图片目录> --scenes scenes.json --scene school_row
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, r"D:\CV-Datatext-Pipeline\sam2-repo")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "yolo-lib"))

from violation_rules import (BAD_VERDICTS, bicycle_class_id, classify_mask,
                             count_verdicts, resolve_zones)

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
    ap.add_argument("--out", default=str(ROOT / "violation_mask_out"))
    ap.add_argument("--weights", default=str(ROOT / "relabel/weights/yolo11x.pt"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--max-side", type=int, default=1400)
    ap.add_argument("--deny-tol", type=float, default=0.05)
    ap.add_argument("--box-tol", type=float, default=0.02)
    ap.add_argument("--device", default="0", help="YOLO 设备：0 / cpu")
    ap.add_argument("--sam-device", default="cuda")
    ap.add_argument("--sam-checkpoint", default=CKPT)
    ap.add_argument("--sam-config", default=CFG)
    ap.add_argument("--contact-ratio", type=float, default=0.02,
                    help="接地点条带占目标掩膜高度的比例，仍为几何近似")
    ap.add_argument("--min-mask-area", type=int, default=200)
    ap.add_argument("--min-mask-score", type=float, default=0.0,
                    help="SAM 质量分数下限；不是违停概率，需用验证集选阈值")
    args = ap.parse_args()
    if not 0 < args.contact_ratio <= 1 or args.min_mask_area < 1:
        ap.error("contact-ratio 必须在 (0, 1] 内，min-mask-area 必须为正数")
    if not all(0 <= v <= 1 for v in (args.deny_tol, args.box_tol, args.min_mask_score)):
        ap.error("比例和质量分数阈值必须在 [0, 1] 内")
    if args.max_side < 1:
        ap.error("max-side 必须为正数")

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from ultralytics import YOLO

    with open(args.scenes, encoding="utf-8-sig") as f:
        scenes = json.load(f)
    sc = scenes.get(args.scene)
    if sc is None:
        raise SystemExit("scenes.json 里没有场景 %r，现有场景: %s" %
                         (args.scene, ", ".join(scenes) or "(空)"))

    yolo = YOLO(args.weights)
    bike_class = bicycle_class_id(yolo.names)
    print("自行车类别:", bike_class)
    sam = build_sam2(args.sam_config, args.sam_checkpoint, device=args.sam_device)
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
    stat = count_verdicts([])
    box_bad_total = 0
    mask_bad_total = 0

    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        allow, deny = resolve_zones(sc, stem)
        has_zone = bool(allow)

        with Image.open(path) as image:
            im = image.convert("RGB")
        im.thumbnail((args.max_side, args.max_side * 2))
        W, H = im.size
        arr = np.array(im)

        allow_r = rasterize(allow, W, H) if has_zone else np.zeros((H, W), dtype=bool)
        deny_r = np.zeros_like(allow_r)
        for poly in deny:
            deny_r |= rasterize(poly, W, H)

        # Ultralytics 的 ndarray 输入要求 BGR；SAM 的 set_image 要求 RGB。
        res = yolo.predict(arr[:, :, ::-1].copy(), imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                           classes=[bike_class], device=args.device, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        if has_zone and len(boxes):
            pred.set_image(arr)

        canvas = im.copy()
        d = ImageDraw.Draw(canvas)
        if has_zone:
            d.line([(x * W, y * H) for x, y in allow] +
                   [(allow[0][0] * W, allow[0][1] * H)], fill=(80, 160, 255), width=6)
            for poly in deny:
                d.line([(x * W, y * H) for x, y in poly] +
                       [(poly[0][0] * W, poly[0][1] * H)], fill=(255, 140, 0), width=5)

        image_rows = []
        for bi, b in enumerate(boxes, 1):
            # 未标定图只检测，避免无意义的 SAM 推理。
            score = -1.0
            m = np.zeros((H, W), dtype=bool)
            if has_zone:
                masks, scores, _ = pred.predict(box=b, multimask_output=False)
                m = np.asarray(masks).reshape(-1, H, W)[0] > 0.0
                score = float(np.asarray(scores).reshape(-1)[0])
            result = classify_mask(
                m, allow_r, deny_r, has_zone=has_zone, score=score,
                min_area=args.min_mask_area, min_score=args.min_mask_score,
                contact_ratio=args.contact_ratio, deny_tol=args.deny_tol)
            verdict = result["verdict"]
            color = {"合规": GREEN, "压线": YELLOW, "出界": RED,
                     "占道": RED}.get(verdict, GRAY)
            stat[verdict] += 1
            box_out = -1.0

            if has_zone:
                # 框规则对照：车身在区外的比例
                bm = np.zeros_like(allow_r)
                x1, y1, x2, y2 = [int(v) for v in b]
                bm[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = True
                box_out = 1 - (bm & allow_r).sum() / max(1, bm.sum())
                box_bad = box_out > args.box_tol
                box_bad_total += int(box_bad)
                mask_bad_total += int(verdict in BAD_VERDICTS)

            # 画轮廓
            c = contour(m)
            canvas_arr = np.array(canvas)
            canvas_arr[c] = (np.array(color, dtype=np.float32) * 0.85 +
                             canvas_arr[c].astype(np.float32) * 0.15)
            canvas = Image.fromarray(canvas_arr)

            box_rule = ("—" if not has_zone
                        else ("违停" if box_bad else "合规"))
            row = dict(image=stem, bike=bi, verdict=verdict,
                       contact_in=round(result["contact_in"], 3),
                       deny=round(result["deny"], 3),
                       box_out=round(float(box_out), 3), box_rule=box_rule,
                       score=round(score, 3), reason=result["reason"])
            rows.append(row)
            image_rows.append(row)
            if verdict in ("无法判定", "未标定"):
                d = ImageDraw.Draw(canvas)
                d.rectangle([float(v) for v in b], outline=GRAY, width=3)
                d.text((float(b[0]), float(b[1])), verdict, fill=GRAY, font=font)

        counts = count_verdicts(image_rows)
        n_bad = sum(counts[v] for v in BAD_VERDICTS)
        d = ImageDraw.Draw(canvas)
        d.rectangle([0, 0, W, 48], fill=(0, 0, 0))
        if has_zone:
            message = "合规 %d / 违停 %d / 无法判定 %d / 检测 %d" % (
                counts["合规"], n_bad, counts["无法判定"], len(boxes))
            color = RED if n_bad else GRAY if counts["无法判定"] else GREEN
        else:
            message = "未标定区域 · 仅检测 %d 辆（不判定）" % len(boxes)
            color = GRAY
        d.text((10, 6), message, fill=color, font=font)
        print("%s %s" % (stem, message), flush=True)
        canvas.save(os.path.join(args.out, os.path.basename(path)), quality=90)

    # 即使没有检测结果也写表头，避免读到上次运行残留的 CSV。
    with open(os.path.join(args.out, "per_bike.csv"), "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "bike", "verdict", "contact_in",
                                        "deny", "box_out", "box_rule", "score", "reason"])
        w.writeheader()
        w.writerows(rows)

    print("\n按掩膜判定:", stat)
    print("掩膜判违停 %d 辆 | 框规则判违停 %d 辆（均非误报统计，需人工真值）" %
          (mask_bad_total, box_bad_total))
    print("输出:", args.out)


if __name__ == "__main__":
    main()
