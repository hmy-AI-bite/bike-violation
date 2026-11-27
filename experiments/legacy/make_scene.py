"""从一张图的标注里取停车区多边形，生成场景配置 scenes.json。"""

from pathlib import Path
import sys
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.py").is_file())))
import config
config.activate()

import json
import os

LAB = str(config.SCHOOL_SOURCE / "labels")
REF = "039f0796-A_013"
OUT = str(config.SCENES)


def polygon(stem):
    for line in open(os.path.join(LAB, stem + ".txt"), encoding="utf-8-sig"):
        q = line.split()
        if len(q) >= 9 and int(float(q[0])) == 1:
            v = [float(x) for x in q[1:]]
            return [[round(v[i], 4), round(v[i + 1], 4)] for i in range(0, len(v) - 1, 2)]
    return None


poly = polygon(REF)
print("参考区域:", poly)

scene = {
    "school_row": {
        "note": "以 %s 的停车区为准；deny 是左侧通道" % REF,
        "allow": poly,
        "deny": [[[0.0, 0.0], [0.30, 0.0], [0.30, 1.0], [0.0, 1.0]]],
    }
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(scene, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("写入", OUT)
