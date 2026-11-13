"""单应迁移 + 白线吸附：看能否把框压到白线上。

流程: 参考图区域 --单应--> 目标图粗略位置 --沿边搜索白线--> 精确位置
"""
import json
import os

import cv2
import numpy as np

MAX_SIDE = 900
REF = r"J:\project-1-at-2026-11-13-17-44-a1d4a70f\images\039f0796-A_013.jpg"
SCENES = r"D:\bick\dataset_school\scenes.json"
OUT = r"D:\bick\report_warp_snap"
QUERIES = [
    r"J:\project-1-at-2026-11-13-17-44-a1d4a70f\images\feb3e917-A_002.jpg",
    r"J:\project-1-at-2026-11-13-17-44-a1d4a70f\images\5bd0ea40-A_017.jpg",
    r"J:\project-1-at-2026-11-13-17-44-a1d4a70f\images\9acffb68-A_006.jpg",
]

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from snap_to_line import white_mask, snap  # noqa: E402


def gray_scaled(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    s = min(1.0, MAX_SIDE / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img, s


def main():
    det = cv2.SIFT_create(nfeatures=4000)
    g_ref, _ = gray_scaled(REF)
    kp_ref, des_ref = det.detectAndCompute(g_ref, None)
    h_ref, w_ref = g_ref.shape[:2]

    scene = json.load(open(SCENES, encoding="utf-8-sig"))["school_row"]
    poly = np.float32([[p[0] * w_ref, p[1] * h_ref] for p in scene["allow"]]).reshape(-1, 1, 2)

    os.makedirs(OUT, exist_ok=True)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    for q in QUERIES:
        g_q, s_q = gray_scaled(q)
        bgr = cv2.imread(q)
        kp_q, des_q = det.detectAndCompute(g_q, None)
        raw = bf.knnMatch(des_ref, des_q, k=2)
        good = [m for m, n in raw if m.distance < 0.75 * n.distance]
        src = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_q[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

        warped = cv2.perspectiveTransform(poly, H).reshape(-1, 2)
        bgr_small = cv2.resize(bgr, (g_q.shape[1], g_q.shape[0]), interpolation=cv2.INTER_AREA)
        m = white_mask(bgr_small)
        snapped, shifts = snap(warped, m, search=60)
        if snapped is None:
            snapped = warped

        vis = bgr.copy()
        # 坐标都在缩放坐标系里
        cv2.polylines(vis, [np.array([[int(x / s_q), int(y / s_q)] for x, y in warped], np.int32)],
                      True, (0, 165, 255), 10)
        cv2.polylines(vis, [np.array([[int(x / s_q), int(y / s_q)] for x, y in snapped], np.int32)],
                      True, (0, 255, 0), 10)
        cv2.putText(vis, "orange=warp only   green=warp+snap to white line", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 14)
        cv2.putText(vis, "orange=warp only   green=warp+snap to white line", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)
        out = os.path.join(OUT, os.path.basename(q))
        cv2.imwrite(out, vis)
        d = np.linalg.norm(snapped - warped, axis=1)
        print("%-30s 内点%4d  吸附位移 平均%.1fpx 最大%.1fpx (缩放后图宽 %d)" %
              (os.path.basename(q), int(mask.sum()), d.mean(), d.max(), g_q.shape[1]))
        print("   ->", out)


if __name__ == "__main__":
    main()
