"""对照实验：用全部特征 vs 只用地面特征 估单应，看区域框飘多少。"""
import json
import os
import sys

import cv2
import numpy as np

MAX_SIDE = 900
REF = r"J:\project-1-at-2026-11-13-17-44-a1d4a70f\images\039f0796-A_013.jpg"
SCENES = r"D:\bick\dataset_school\scenes.json"
OUT = r"D:\bick\report_homography"
QUERIES = [
    r"J:\project-1-at-2026-11-13-17-44-a1d4a70f\images\feb3e917-A_002.jpg",
    r"J:\project-1-at-2026-11-13-17-44-a1d4a70f\images\5bd0ea40-A_017.jpg",
]


def gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    s = min(1.0, MAX_SIDE / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img, s


def main():
    det = cv2.SIFT_create(nfeatures=4000)
    g_ref, _ = gray(REF)
    kp_ref, des_ref = det.detectAndCompute(g_ref, None)
    h_ref, w_ref = g_ref.shape[:2]
    print("参考图特征点:", len(kp_ref))

    scene = json.load(open(SCENES, encoding="utf-8-sig"))["school_row"]
    allow = scene["allow"]
    pts_ref = np.float32([[p[0] * w_ref, p[1] * h_ref] for p in allow]).reshape(-1, 1, 2)

    os.makedirs(OUT, exist_ok=True)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    for q in QUERIES:
        g_q, s_q = gray(q)
        kp_q, des_q = det.detectAndCompute(g_q, None)
        raw = bf.knnMatch(des_ref, des_q, k=2)
        good = [m for m, n in raw if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            print("匹配太少:", q)
            continue

        src = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_q[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H_all, mask_all = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        n_all = int(mask_all.sum())

        # 只用参考图下半部分（地面）的匹配点重估
        ys = src.reshape(-1, 2)[:, 1]
        ground = (ys > 0.45 * h_ref).reshape(-1, 1).astype(np.uint8)
        sel = (mask_all.reshape(-1, 1) & ground).astype(bool).reshape(-1)
        n_ground = int(sel.sum())
        H_g = None
        if n_ground >= 8:
            H_g, _ = cv2.findHomography(src[sel], dst[sel], cv2.RANSAC, 4.0)

        warped_all = cv2.perspectiveTransform(pts_ref, H_all).reshape(-1, 2)
        warped_g = cv2.perspectiveTransform(pts_ref, H_g).reshape(-1, 2) if H_g is not None else None

        print("\n%s" % os.path.basename(q))
        print("  全特征内点 %d，地面特征内点 %d" % (n_all, n_ground))
        if warped_g is not None:
            diff = np.linalg.norm(warped_all - warped_g, axis=1)
            print("  两种估计的角点偏移: 平均 %.1f px, 最大 %.1f px (图宽 %d)" %
                  (diff.mean(), diff.max(), g_q.shape[1]))
            print("  最大偏移占图宽: %.1f%%" % (100 * diff.max() / g_q.shape[1]))

        img = cv2.imread(q)
        pa = np.array([[int(x / s_q), int(y / s_q)] for x, y in warped_all], np.int32)
        cv2.polylines(img, [pa], True, (0, 165, 255), 10)
        if warped_g is not None:
            pg = np.array([[int(x / s_q), int(y / s_q)] for x, y in warped_g], np.int32)
            cv2.polylines(img, [pg], True, (0, 255, 0), 10)
        cv2.putText(img, "orange=all features  green=ground only", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 12)
        cv2.putText(img, "orange=all features  green=ground only", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4)
        cv2.imwrite(os.path.join(OUT, os.path.basename(q)), img)
        print("  输出:", os.path.join(OUT, os.path.basename(q)))


if __name__ == "__main__":
    main()
