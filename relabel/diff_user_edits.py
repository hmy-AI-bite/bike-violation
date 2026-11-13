"""对比「自动标注」和「用户精修后」的标注差异。"""
import json
import os
from collections import Counter

DS = r"D:\bick\dataset_v2"
AUTO = r"D:\bick\relabel\auto\labels"
OUT = r"D:\bick\relabel\report\user_edits.json"


def count(p):
    if not os.path.exists(p):
        return 0
    return sum(1 for line in open(p, encoding="utf-8-sig") if line.strip())


def auto_path(stem):
    for split in ("train", "val"):
        p = os.path.join(AUTO, split, stem + ".txt")
        if os.path.exists(p):
            return p
    return ""


rows = []
tot_before = tot_after = 0
for split in ("train", "val"):
    d = os.path.join(DS, "labels", split)
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".txt"):
            continue
        stem = fn[:-4]
        cur = count(os.path.join(d, fn))
        before = count(auto_path(stem))
        tot_before += before
        tot_after += cur
        rows.append(dict(split=split, stem=stem, before=before, after=cur,
                         delta=cur - before, pct=(cur / before) if before else 0))

changed = [r for r in rows if r["delta"] != 0]
empty_now = [r for r in rows if r["after"] == 0]
grew = [r for r in rows if r["delta"] > 0]

print("图片 %d 张" % len(rows))
print("自动标注 %d 框 -> 精修后 %d 框  (净删 %d, %.0f%%)" %
      (tot_before, tot_after, tot_before - tot_after,
       100 * (tot_before - tot_after) / tot_before))
print("被改动的图: %d 张 (%.0f%%)" % (len(changed), 100 * len(changed) / len(rows)))
print("删到 0 框（变负样本）: %d 张" % len(empty_now))
print("反而加了框的图: %d 张, 共加 %d 个" % (len(grew), sum(r["delta"] for r in grew)))
print()
print("改动最大的 15 张:")
for r in sorted(rows, key=lambda x: x["delta"])[:15]:
    print("  %-18s %2d -> %2d  (%+d)" % (r["split"] + "/" + r["stem"], r["before"],
                                         r["after"], r["delta"]))
print()
print("保留率分布:", dict(Counter(
    "0%" if r["pct"] == 0 else
    "<50%" if r["pct"] < 0.5 else
    "50-100%" if r["pct"] < 1 else
    "100%" if r["pct"] == 1 else ">100%" for r in rows)))

json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n明细:", OUT)
