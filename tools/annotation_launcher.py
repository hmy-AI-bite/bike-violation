"""统一打开车辆状态、车轮接地点与停车区域标注工具。"""
from pathlib import Path
import sys

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.py").is_file())))
import config
config.activate()

from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
from labeler import Main as BicycleWindow
from zone_tool import Main as ZoneWindow


class Launcher(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("自行车标注工具 整合版")
        self.resize(540, 340)
        self.bicycles = None
        self.zones = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)
        title = QLabel("选择标注任务")
        title.setStyleSheet("font-size:22px;font-weight:600")
        layout.addWidget(title)
        for name, callback in (
            ("自行车框与停车状态", lambda: self.open_bicycles(False)),
            ("车轮接地点 连续标注", lambda: self.open_bicycles(True)),
            ("允许停车区与禁停区", self.open_zones),
        ):
            button = QPushButton(name)
            button.setMinimumHeight(42)
            button.clicked.connect(callback)
            layout.addWidget(button)
        hint = QLabel("打开后选择自己的图片文件夹。\n滚轮缩放 · 中键拖动画面 · R 重置 · Ctrl+S 保存\n车辆与接地点共用同一个窗口，已有标注会保留。")
        hint.setWordWrap(True)
        layout.addWidget(hint)

    def open_bicycles(self, contacts=False):
        if self.bicycles is None:
            self.bicycles = BicycleWindow(contact_mode=contacts)
        else:
            self.bicycles.contact_mode_box.setChecked(contacts)
        self.bicycles.show()
        self.bicycles.raise_()
        self.bicycles.activateWindow()

    def open_zones(self):
        if self.zones is None:
            self.zones = ZoneWindow()
        self.zones.show()
        self.zones.raise_()
        self.zones.activateWindow()


def main():
    app = QApplication(sys.argv)
    launcher = Launcher()
    launcher.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
