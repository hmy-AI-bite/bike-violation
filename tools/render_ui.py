"""把工具界面离屏渲染成 PNG，用来检查外观是否正常。"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication  # noqa: E402

from labeler import Main  # noqa: E402

app = QApplication(sys.argv)
w = Main(r"D:\bick\dataset_v2")
w.resize(1400, 900)
w.show()
app.processEvents()
# 选中一个框，展示选中态
w.canvas.sel = 0
w.update_status()
w.canvas.update()
app.processEvents()

pix = w.grab()
out = r"D:\bick\tools\ui_preview.png"
pix.save(out)
print("saved", out, pix.width(), "x", pix.height())
