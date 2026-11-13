"""渲染标定工具界面，检查外观。"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "windows")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication  # noqa: E402

from zone_tool import Main  # noqa: E402

app = QApplication(sys.argv)
w = Main(r"J:\project-1-at-2026-11-13-17-44-a1d4a70f\images",
         r"D:\bick\dataset_school\scenes.json", "school_row")
w.resize(1300, 880)
w.show()
app.processEvents()
# 演示"正在画第 3 个点"的状态
w.canvas.pending = [(0.31, 0.09), (0.60, 0.09)]
w.canvas.cursor = w.canvas.to_screen(0.60 * w.canvas.pix.width(), 0.86 * w.canvas.pix.height())
w.update_lbl()
app.processEvents()

out = r"D:\bick\tools\zone_ui.png"
w.grab().save(out)
print("saved", out)
