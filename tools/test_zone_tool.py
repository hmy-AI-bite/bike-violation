"""无显示器验证标定工具：加顶点、闭合、保存、重新加载。"""
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

# 画允许区（4 点）
for p in [(0.30, 0.10), (0.62, 0.10), (0.62, 0.95), (0.30, 0.95)]:
    w.canvas.pending.append(p)
w.close_poly()
assert len(w.current["allow"]) == 4, w.current

# 切模式画禁停区（3 点）
w.toggle_mode()
for p in [(0.0, 0.0), (0.28, 0.0), (0.28, 1.0)]:
    w.canvas.pending.append(p)
w.close_poly()
assert len(w.current["deny"]) == 1 and len(w.current["deny"][0]) == 3

w.save()
data = json.load(open(OUT, encoding="utf-8"))
print("保存内容:", json.dumps(data, ensure_ascii=False))
assert data["test_scene"]["allow"][0] == [0.3, 0.1]

# 重新加载，确认能读回
w2 = Main(IMG, OUT, "test_scene")
assert len(w2.current["allow"]) == 4
print("重新加载 OK")

# 撤销点
w2.canvas.pending.append((0.5, 0.5))
w2.undo_point()
assert not w2.canvas.pending
print("撤销点 OK")

# 翻页
w2.next()
print("翻页后:", w2.files[w2.idx])
w2.close()

os.remove(OUT)
print("\n全部检查通过")
