"""场景匹配：判断一张新图是否属于已标定的场地，并把区域多边形迁移过去。

原理：
    1) 用 SIFT 特征点匹配「新图」与「场地参考图」
    2) RANSAC 估单应矩阵，内点越多说明越可能是同一场景、同一地面
    3) 内点足够多时，用同一个 H 把标定好的多边形搬到新图上（不同角度也能用）

用法:
    python scene_match.py --ref <参考图> --queries <目录或图片> [--warp-scenes scenes.json --scene 名字]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

MAX_SIDE = 900
MIN_INLIERS = 45
MIN_RATIO = 0.008


def load_gray(path, max_side=MAX_SIDE):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, 1.0
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img, scale


def features(gray, detector):
    kp, des = detector.detectAndCompute(gray, None)
    return kp, des


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--scenes", default="", help="可选：用单应把该场景区域搬到每张图")
    ap.add_argument("--scene", default="")
    ap.add_argument("--out", default=r"D:\bick\scene_match_out")
    args = ap.parse_args()

    detector = cv2.SIFT_create(nfeatures=4000)
    ref_gray, ref_scale = load_gray(args.ref)
    ref_kp, ref_des = features(ref_gray, detector)
    ref_h, ref_w = ref_gray.shape[:2]
    print("参考图特征点: %d" % len(ref_kp))

    matcher = cv2.BFMatcher(cv2.NORM_L2)

    files = ([os.path.join(args.queries, f) for f in sorted(os.listdir(args.queries))
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]
             if os.path.isdir(args.queries) else [args.queries])

    scenes = {}
    if args.scenes and os.path.exists(args.scenes):
        scenes = json.load(open(args.scenes, encoding="utf-8-sig"))

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for path in files:
        gray, q_scale = load_gray(path)
        if gray is None:
            continue
        kp, des = features(gray, detector)
        if des is None or len(des) < 10:
            print("%-30s 特征点太少" % os.path.basename(path))
            continue
        raw = matcher.knnMatch(ref_des, des, k=2)
        good = [m for m, n in raw if m.distance < 0.75 * n.distance]
        inliers = 0
        H = None
        if len(good) >= 8:
            src = np.float32([ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if mask is not None:
                inliers = int(mask.sum())
        ratio = inliers / max(1, len(ref_kp))
        same = inliers >= MIN_INLIERS and ratio >= MIN_RATIO
        rows.append(dict(file=os.path.basename(path), kp=len(kp), good=len(good),
                         inliers=inliers, ratio=round(ratio, 3), same=same))
        print("%-34s 特征%4d 匹配%4d 内点%4d 占比%.2f  %s" %
              (os.path.basename(path), len(kp), len(good), inliers, ratio,
               "同场地 ✓" if same else "不同 ✗"))

        if same and scenes and args.scene and H is not None:
            sc = scenes.get(args.scene) or {}
            allow = (sc.get("images") or {}).get(
                os.path.splitext(os.path.basename(args.ref))[0], {}).get("allow") or sc.get("allow")
            if allow:
                pts = np.float32([[p[0] * ref_w, p[1] * ref_h] for p in allow]).reshape(-1, 1, 2)
                warped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
                qh, qw = gray.shape[:2]
                norm = [[round(float(x) / qw, 4), round(float(y) / qh, 4)] for x, y in warped]
                img = cv2.imread(path)
                # warped 在缩放后的坐标系里，画回原图要除以缩放比
                poly = np.array([[int(x / q_scale), int(y / q_scale)] for x, y in warped],
                                dtype=np.int32)
                cv2.polylines(img, [poly], True, (255, 160, 0), 10)
                cv2.putText(img, "warped zone", (12, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 160, 0), 3)
                cv2.imwrite(os.path.join(args.out, os.path.basename(path)), img)
                print("    区域已迁移:", norm)

    if rows:
        import csv
        with open(os.path.join(args.out, "match.csv"), "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print("\n输出:", args.out)


if __name__ == "__main__":
    main()
