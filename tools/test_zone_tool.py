"""无显示器验证标定工具：按图存区域、闭合、保存、加载。"""
import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication  # noqa: E402

from zone_tool import Main  # noqa: E402

IMG = r"J:\project-1-at-2026-11-13-17-44-a1d4a70f\images"
OUT = r"D:\bick\tools\_test_scenes.json"

if os.path.exists(OUT):
    os.remove(OUT)

app = QApplication(sys.argv)
w = Main(IMG, OUT, "test_scene")
print("图片数:", len(w.files))

# 第 1 张：画允许区
for p in [(0.30, 0.10), (0.62, 0.10), (0.62, 0.95), (0.30, 0.95)]:
    w.canvas.pending.append(p)
w.close_poly()
assert len(w.doc()["allow"]) == 4, w.doc()

# 切模式画禁停区
w.toggle_mode()
for p in [(0.0, 0.0), (0.28, 0.0), (0.28, 1.0)]:
    w.canvas.pending.append(p)
w.close_poly()
assert len(w.doc()["deny"]) == 1

# 第 2 张应当是空的新条目（不会串用第 1 张的区域）
w.next()
assert w.doc()["allow"] == [], w.doc()
print("按图隔离 OK：第 2 张区域为空")

# 设为场景默认
w.prev()
w.promote()
assert len(w.scenes["test_scene"]["allow"]) == 4

w.save()
data = json.load(open(OUT, encoding="utf-8"))
print("保存的结构:", list(data["test_scene"].keys()))
assert "images" in data["test_scene"]
stem = os.path.splitext(w.files[0])[0]
assert stem in data["test_scene"]["images"]

# 重新加载
w2 = Main(IMG, OUT, "test_scene")
assert len(w2.doc()["allow"]) == 4
print("重新加载 OK")

w2.undo_point()
w2.close()
os.remove(OUT)
print("\n全部检查通过")
