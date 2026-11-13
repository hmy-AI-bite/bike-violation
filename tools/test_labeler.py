"""无显示器环境下验证标注工具：加载、删除、保存、撤销、新建框。"""
import os
import shutil
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = r"D:\bick\tools\_test_data"
SRC = r"D:\bick\dataset_v2"


def build():
    if os.path.exists(ROOT):
        shutil.rmtree(ROOT)
    os.makedirs(os.path.join(ROOT, "images", "train"))
    os.makedirs(os.path.join(ROOT, "labels", "train"))
    names = sorted(os.listdir(os.path.join(SRC, "images", "train")))[:2]
    for fn in names:
        shutil.copy2(os.path.join(SRC, "images", "train", fn),
                     os.path.join(ROOT, "images", "train", fn))
        stem = os.path.splitext(fn)[0]
        shutil.copy2(os.path.join(SRC, "labels", "train", stem + ".txt"),
                     os.path.join(ROOT, "labels", "train", stem + ".txt"))
    return names


names = build()
from labeler import Doc, Main  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication(sys.argv)
w = Main(ROOT)

print("图片数:", len(w.items))
assert len(w.items) == 2, w.items

stem = os.path.splitext(names[0])[0]
lab = os.path.join(ROOT, "labels", "train", stem + ".txt")
n0 = len(w.doc.boxes)
print("首图框数:", n0)
assert n0 > 0

# 删除第一个框
w.canvas.sel = 0
w.delete_sel()
w.save()
n1 = sum(1 for l in open(lab, encoding="utf-8") if l.strip())
print("删除后框数:", n1)
assert n1 == n0 - 1, (n1, n0)
assert os.path.exists(lab + ".bak"), "未生成备份"

# 撤销
w.undo()
assert len(w.doc.boxes) == n0, "撤销失败"
print("撤销后框数:", len(w.doc.boxes))

# 新建一个框（模拟画框结果）
w.doc.push_undo()
w.doc.boxes.append([0, 0.1, 0.1, 0.3, 0.3])
w.doc.conf.append(-1.0)
w.save()
lines = [l for l in open(lab, encoding="utf-8") if l.strip()]
print("新建后框数:", len(lines))
assert len(lines) == n0 + 1, (len(lines), n0)

# 检查保存格式：5 列、归一化、类号 0
parts = lines[-1].split()
assert parts[0] == "0" and len(parts) == 5, parts
vals = [float(v) for v in parts[1:]]
assert all(0 <= v <= 1 for v in vals), vals
print("保存格式 OK:", lines[-1].strip())

# 翻页
w.next()
print("翻页后:", w.items[w.i][1])
assert w.i == 1

# 关闭时自动保存不报错
w.close()
print("\n全部检查通过")
shutil.rmtree(ROOT, ignore_errors=True)
