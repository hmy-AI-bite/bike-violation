"""把粗画的多边形吸附到地面白线上。

思路：每条边沿法线方向搜索白色像素响应最强的位置，平移该边；
再用相邻边求交，重算四个角点。

用法:
    python snap_to_line.py --image <图> --scenes scenes.json --scene school_row [--apply]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np


def white_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = ((v > 120) & (s < 60)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask


def line_from_pts(p1, p2):
    """两点 -> 齐次直线 (a,b,c)，满足 ax+by+c=0"""
    a = p1[1] - p2[1]
    b = p2[0] - p1[0]
    c = p1[0] * p2[1] - p2[0] * p1[1]
    n = np.hypot(a, b)
    return np.array([a, b, c]) / (n if n else 1)


def intersect(l1, l2):
    x = l1[1] * l2[2] - l1[2] * l2[1]
    y = l1[2] * l2[0] - l1[0] * l2[2]
    d = l1[0] * l2[1] - l1[1] * l2[0]
    if abs(d) < 1e-9:
        return None
    return np.array([x / d, y / d])


def best_offset(mask, p1, p2, search=40, samples=120):
    """沿边采样，统计每个法向偏移上的白色响应，返回最佳偏移（像素）。"""
    direction = p2 - p1
    L = np.linalg.norm(direction)
    if L < 5:
        return 0.0
    direction = direction / L
    normal = np.array([-direction[1], direction[0]])
    h, w = mask.shape

    offsets = np.arange(-search, search + 1)
    score = np.zeros(len(offsets))
    for i, off in enumerate(offsets):
        cnt = 0
        for t in np.linspace(0.1, 0.9, samples):
            p = p1 + direction * (L * t) + normal * off
            x, y = int(round(p[0])), int(round(p[1]))
            if 0 <= x < w and 0 <= y < h and mask[y, x]:
                cnt += 1
        score[i] = cnt
    if score.max() <= 0:
        return 0.0
    # 取响应最强的若干偏移的中位数，避免单点噪声
    top = np.argsort(score)[-5:]
    return float(np.median(offsets[top]))


def snap(poly_px, mask, search=40):
    n = len(poly_px)
    lines = []
    shifts = []
    for i in range(n):
        p1, p2 = poly_px[i], poly_px[(i + 1) % n]
        off = best_offset(mask, p1, p2, search=search)
        direction = (p2 - p1) / (np.linalg.norm(p2 - p1) + 1e-9)
        normal = np.array([-direction[1], direction[0]])
        shift = normal * off
        shifts.append(off)
        lines.append(line_from_pts(p1 + shift, p2 + shift))

    new_pts = []
    for i in range(n):
        p = intersect(lines[i - 1], lines[i])
        new_pts.append(p if p is not None else poly_px[i])
    return np.array(new_pts), shifts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--scenes", default=r"D:\bick\dataset_school\scenes.json")
    ap.add_argument("--scene", default="school_row")
    ap.add_argument("--out", default=r"D:\bick\snap_out")
    ap.add_argument("--search", type=int, default=40)
    ap.add_argument("--apply", action="store_true", help="把结果写回 scenes.json 的场景默认区域")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    h, w = img.shape[:2]
    mask = white_mask(img)

    scenes = json.load(open(args.scenes, encoding="utf-8-sig"))
    poly = scenes[args.scene]["allow"]
    if not poly:
        raise SystemExit("场景没有 allow 多边形")
    poly_px = np.array([[p[0] * w, p[1] * h] for p in poly], np.float64)

    new_px, shifts = snap(poly_px, mask, search=args.search)
    disp = np.linalg.norm(new_px - poly_px, axis=1)
    print("每条边的吸附偏移(px):", [round(s, 1) for s in shifts])
    print("角点移动: 平均 %.1f px, 最大 %.1f px (图宽 %d)" % (disp.mean(), disp.max(), w))

    vis = img.copy()
    cv2.polylines(vis, [poly_px.astype(np.int32)], True, (0, 0, 255), 8)
    cv2.polylines(vis, [new_px.astype(np.int32)], True, (0, 255, 0), 8)
    for i, (a, b) in enumerate(zip(poly_px, new_px)):
        cv2.arrowedLine(vis, tuple(a.astype(int)), tuple(b.astype(int)), (255, 0, 0), 3)
    os.makedirs(args.out, exist_ok=True)
    out_img = os.path.join(args.out, "snap_" + os.path.basename(args.image))
    cv2.imwrite(out_img, vis)
    print("对比图:", out_img)
    print("红=原多边形  绿=吸附后")

    if args.apply:
        scenes[args.scene]["allow"] = [[round(float(x) / w, 4), round(float(y) / h, 4)]
                                       for x, y in new_px]
        json.dump(scenes, open(args.scenes, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("已写回 scenes.json")


if __name__ == "__main__":
    main()
