"""越界/违停判定 v2：支持「允许停放区 + 禁停区(通道)」，覆盖压线。

判定口径:
    * 出界   —— 车身落在允许区外的比例 > --tol（默认 2%，压线就会被抓）
    * 占道   —— 车身与任一禁停区(通道)重叠比例 > --deny-tol

场景配置 scenes.json:
{
  "campus_a": {
    "allow": [[0.30,0.10],[0.62,0.10],[0.62,0.95],[0.30,0.95]],
    "deny":  [[[0.00,0.00],[0.30,0.00],[0.30,1.00],[0.00,1.00]]]
  }
}

用法:
    python violation.py --source imgs --scenes scenes.json --scene campus_a
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, r"D:\bick\yolo-lib")


def point_in_poly(x, y, pts):
    inside = False
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            if x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
                inside = not inside
    return inside


def ratio_in(box, poly, W, H, n=14):
    x1, y1, x2, y2 = box
    hit = 0
    for i in range(n):
        for j in range(n):
            x = x1 + (x2 - x1) * (i + 0.5) / n
            y = y1 + (y2 - y1) * (j + 0.5) / n
            if point_in_poly(x / W, y / H, poly):
                hit += 1
    return hit / (n * n)


def load_scene(scenes, name, stem):
    sc = scenes.get(name, {})
    if "images" in sc and stem in sc["images"]:
        return sc["images"][stem]
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=r"D:\bick\yolo-lib")
    ap.add_argument("--weights", default=r"D:\bick\relabel\weights\yolo11x.pt")
    ap.add_argument("--source", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--scene", default="default")
    ap.add_argument("--out", default=r"D:\bick\violation_v2")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--device", default="0")
    ap.add_argument("--tol", type=float, default=0.02, help="容忍的出界比例")
    ap.add_argument("--deny-tol", type=float, default=0.05, help="容忍的占道比例")
    args = ap.parse_args()

    sys.path.insert(0, args.lib)
    from PIL import Image, ImageDraw, ImageFont
    from ultralytics import YOLO

    scenes = json.load(open(args.scenes, encoding="utf-8"))
    model = YOLO(args.weights)
    os.makedirs(args.out, exist_ok=True)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 32)
    except Exception:
        font = ImageFont.load_default()

    files = ([os.path.join(args.source, f) for f in sorted(os.listdir(args.source))
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]
             if os.path.isdir(args.source) else [args.source])

    rows = []
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        sc = load_scene(scenes, args.scene, stem)
        allow = sc.get("allow")
        deny = sc.get("deny") or []
        if not allow:
            print("跳过（该场景没有 allow 区域）:", stem)
            continue

        res = model.predict(path, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                            classes=[1], device=args.device, verbose=False)[0]
        im = Image.open(path).convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)

        ap_px = [(x * W, y * H) for x, y in allow]
        d.line(ap_px + [ap_px[0]], fill=(80, 160, 255), width=7)
        for poly in deny:
            dp = [(x * W, y * H) for x, y in poly]
            d.line(dp + [dp[0]], fill=(255, 140, 0), width=6)

        n_ok = n_bad = 0
        for box, cf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            b = [float(v) for v in box]
            out_ratio = 1.0 - ratio_in(b, allow, W, H)
            deny_ratio = max([ratio_in(b, p, W, H) for p in deny], default=0.0)
            reason = ""
            if out_ratio > args.tol:
                reason = "出界%.0f%%" % (out_ratio * 100)
            elif deny_ratio > args.deny_tol:
                reason = "占道%.0f%%" % (deny_ratio * 100)
            if reason:
                n_bad += 1
                color = (255, 40, 40)
                d.text((b[0] + 4, max(1, b[1] + 2)), reason, fill=(255, 80, 80), font=font)
            else:
                n_ok += 1
                color = (0, 220, 90)
            d.rectangle(b, outline=color, width=6)

        d.rectangle([0, 0, 620, 50], fill=(0, 0, 0))
        d.text((10, 6), "违停 %d / 共 %d" % (n_bad, n_bad + n_ok),
               fill=(255, 60, 60) if n_bad else (0, 220, 90), font=font)
        im.save(os.path.join(args.out, os.path.basename(path)), quality=90)
        rows.append(dict(file=os.path.basename(path), ok=n_ok, bad=n_bad))
        print("%-30s 合规 %2d  违停 %2d" % (stem, n_ok, n_bad))

    if rows:
        with open(os.path.join(args.out, "violations.csv"), "w",
                  encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print("\n合计违停 %d 辆 -> %s" % (sum(r["bad"] for r in rows), args.out))


if __name__ == "__main__":
    main()
