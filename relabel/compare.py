"""对比「原标注」与「预训练模型自动标注」，分级判定并输出复核清单。

判定分级（对每个原框，找重叠度最高的自动框）:
  good      IoU >= 0.5                      -> 基本没问题
  loose     包含同一辆车但框太松:
            IoU >= 0.2 或 原框覆盖率 >= 0.6   -> 框偏大/偏了，需要修
  wrong     与任何自动框都几乎不重叠        -> 画错对象（人/车/幻觉框）
自动框在原标注里找不到对应 -> missing（原标注漏标）

用法:
    python compare.py
"""
import argparse
import csv
import hashlib
import json
import os
from collections import Counter


def load_txt(path):
    boxes = []
    if not os.path.exists(path):
        return boxes
    for line in open(path, encoding="utf-8-sig"):
        p = line.split()
        if len(p) < 5:
            continue
        try:
            boxes.append([int(float(p[0]))] + [float(v) for v in p[1:5]])
        except ValueError:
            pass
    return boxes  # [cls, xc, yc, w, h]


def xyxy(b):
    xc, yc, w, h = b[1], b[2], b[3], b[4]
    return [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]


def inter_area(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def classify(orig_boxes, auto_boxes):
    res = []
    used = set()
    for o in orig_boxes:
        oa = xyxy(o)
        o_area = area(oa) or 1e-9
        best_iou, best_cov, best_j = 0.0, 0.0, -1
        for j, a in enumerate(auto_boxes):
            aa = xyxy(a)
            inter = inter_area(oa, aa)
            if inter <= 0:
                continue
            union = o_area + area(aa) - inter
            iou = inter / union if union > 0 else 0.0
            cov = inter / o_area
            if iou > best_iou:
                best_iou = iou
            if cov > best_cov:
                best_cov, best_j = cov, j
        if best_iou >= 0.5:
            kind, used_j = "good", best_j
        elif best_iou >= 0.2 or best_cov >= 0.6:
            kind, used_j = "loose", best_j
        else:
            kind, used_j = "wrong", -1
        if used_j >= 0:
            used.add(used_j)
        res.append(dict(kind=kind, iou=round(best_iou, 3), cov=round(best_cov, 3)))
    missing = [j for j in range(len(auto_boxes)) if j not in used]
    return res, missing


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"D:\bick\bike_dataset")
    ap.add_argument("--auto", default=r"D:\bick\relabel\auto\labels")
    ap.add_argument("--out", default=r"D:\bick\relabel\report")
    ap.add_argument("--splits", default="train,val")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows, dup_map, h2files = [], Counter(), {}
    for split in args.splits.split(","):
        odir = os.path.join(args.src, "labels", split)
        adir = os.path.join(args.auto, split)
        idir = os.path.join(args.src, "images", split)
        stems = sorted({f[:-4] for f in os.listdir(odir) if f.endswith(".txt")} |
                       {f[:-4] for f in os.listdir(adir) if f.endswith(".txt")})
        for stem in stems:
            img = os.path.join(idir, stem + ".jpg")
            if os.path.exists(img):
                hv = md5(img)
                h2files.setdefault(hv, []).append("%s/%s" % (split, stem))
            o = load_txt(os.path.join(odir, stem + ".txt"))
            a = load_txt(os.path.join(adir, stem + ".txt"))
            res, missing = classify(o, a)
            cnt = Counter(r["kind"] for r in res)
            rows.append(dict(
                split=split, image=stem,
                n_orig=len(o), n_auto=len(a),
                good=cnt["good"], loose=cnt["loose"], wrong=cnt["wrong"],
                missing=len(missing),
                bad=cnt["loose"] + cnt["wrong"],
            ))

    dup_groups = {k: v for k, v in h2files.items() if len(v) > 1}
    dup_files = sum(len(v) for v in dup_groups.values())

    rows.sort(key=lambda r: (-r["wrong"], -r["loose"], -r["missing"], r["image"]))
    with open(os.path.join(args.out, "compare.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    S = lambda k: sum(r[k] for r in rows)
    n_img = len(rows)
    clean = [r for r in rows if r["bad"] == 0 and r["missing"] == 0]
    okish = [r for r in rows if r["wrong"] == 0 and r["missing"] <= 1 and r["loose"] <= max(1, r["n_orig"] // 3)]

    lines = [
        "图片总数: %d   原标注框: %d   自动标注框: %d" % (n_img, S("n_orig"), S("n_auto")),
        "",
        "原框判定:  好 %d (%.1f%%)   偏松 %d (%.1f%%)   画错对象 %d (%.1f%%)"
        % (S("good"), 100 * S("good") / max(1, S("n_orig")),
           S("loose"), 100 * S("loose") / max(1, S("n_orig")),
           S("wrong"), 100 * S("wrong") / max(1, S("n_orig"))),
        "自动框在原标注里找不到: %d" % S("missing"),
        "",
        "完全无需复核的图:        %d (%.0f%%)" % (len(clean), 100 * len(clean) / n_img),
        "小改即可(无画错、漏标<=1): %d (%.0f%%)" % (len(okish), 100 * len(okish) / n_img),
        "有画错对象的图:          %d (%.0f%%)" % (sum(1 for r in rows if r["wrong"]),
                                             100 * sum(1 for r in rows if r["wrong"]) / n_img),
        "",
        "重复图片: %d 组, 涉及 %d 张 (MD5 完全相同)" % (len(dup_groups), dup_files),
    ]
    for k, v in list(dup_groups.items())[:5]:
        lines.append("  " + " == ".join(v))

    lines += ["", "最需要复核的 25 张:"]
    for r in rows[:25]:
        lines.append("  %-18s 原%2d 自动%2d | 好%2d 松%2d 错%2d 漏%2d" %
                     (r["split"] + "/" + r["image"], r["n_orig"], r["n_auto"],
                      r["good"], r["loose"], r["wrong"], r["missing"]))

    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    with open(os.path.join(args.out, "review_queue.json"), "w", encoding="utf-8") as f:
        json.dump([r for r in rows if r["bad"] or r["missing"] > 1], f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "duplicates.json"), "w", encoding="utf-8") as f:
        json.dump(dup_groups, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
