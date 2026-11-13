"""停车区标定工具：点几下圈出「允许停放区」和「禁停区(通道)」，存成 scenes.json。

用法:
    python zone_tool.py <图片文件夹> [scenes.json 路径] [场景名]

操作:
    左键        加一个顶点
    Enter       闭合当前多边形（至少 3 个点）
    Backspace   撤销上一个顶点
    Tab         切换 允许区(蓝) / 禁停区(橙)
    S           保存到 scenes.json
    A / D       上一张 / 下一张
    C           清空当前模式的图形
    Esc         取消正在画的多边形

输出格式:
    {"场景名": {"allow": [[x,y],...], "deny": [[[...]],...]}}   # 归一化坐标
"""
import json
import os
import sys

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QFont, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMainWindow,
                               QMessageBox, QPushButton, QVBoxLayout, QWidget)

ALLOW_COLOR = QColor(80, 160, 255)
DENY_COLOR = QColor(255, 140, 0)


class Canvas(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.setMinimumSize(900, 600)
        self.setFocusPolicy(Qt.StrongFocus)
        self.pix = None
        self.pending = []          # 正在画的点（图片坐标）
        self.zoom = 1.0
        self.cursor = QPoint(0, 0)

    def set_image(self, path):
        self.pix = QPixmap(path)
        self.pending = []
        self.update()

    def _fit(self):
        if not self.pix:
            return 0.0, 0.0, 1.0
        iw, ih = self.pix.width(), self.pix.height()
        s = min(self.width() / iw, self.height() / ih) * self.zoom
        return (self.width() - iw * s) / 2, (self.height() - ih * s) / 2, s

    def to_img(self, pos):
        ox, oy, s = self._fit()
        return (pos.x() - ox) / s, (pos.y() - oy) / s

    def to_screen(self, x, y):
        ox, oy, s = self._fit()
        return QPoint(int(ox + x * s), int(oy + y * s))

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(26, 26, 28))
        if not self.pix:
            return
        ox, oy, s = self._fit()
        p.drawPixmap(int(ox), int(oy), int(self.pix.width() * s), int(self.pix.height() * s), self.pix)

        W, H = self.pix.width(), self.pix.height()
        doc = self.win.current

        def poly(pts, color, width=3):
            if len(pts) < 2:
                return
            p.setPen(QPen(color, width))
            p.setBrush(Qt.NoBrush)
            screen = [self.to_screen(x * W, y * H) for x, y in pts]
            p.drawPolyline(screen + ([screen[0]] if len(pts) >= 3 else []))

        if doc.get("allow"):
            poly(doc["allow"], ALLOW_COLOR, 4)
        for d in doc.get("deny", []):
            poly(d, DENY_COLOR, 4)

        # 正在画的多边形
        color = ALLOW_COLOR if self.win.mode == "allow" else DENY_COLOR
        if self.pending:
            screen = [self.to_screen(x * W, y * H) for x, y in self.pending] + [self.cursor]
            p.setPen(QPen(color, 2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            if len(screen) > 1:
                p.drawPolyline(screen)
            p.setBrush(color)
            for pt in screen[:-1]:
                p.drawEllipse(pt.x() - 4, pt.y() - 4, 8, 8)

        p.setFont(QFont("Microsoft YaHei", 10))
        p.setPen(QPen(QColor(255, 255, 0)))
        p.drawText(8, self.height() - 10,
                   "模式: %s | 已放 %d 点 | %s" %
                   ("允许区" if self.win.mode == "allow" else "禁停区",
                    len(self.pending),
                    "Enter 闭合" if self.pending else "左键点点"))

    def mouseMoveEvent(self, e):
        self.cursor = e.position().toPoint()
        if self.pending:
            self.update()

    def mousePressEvent(self, e):
        if not self.pix or e.button() != Qt.LeftButton:
            return
        x, y = self.to_img(e.position().toPoint())
        W, H = self.pix.width(), self.pix.height()
        self.pending.append((round(max(0.0, min(1.0, x / W)), 4),
                             round(max(0.0, min(1.0, y / H)), 4)))
        self.update()

    def wheelEvent(self, e):
        self.zoom = max(0.4, min(4.0, self.zoom * (1.1 if e.angleDelta().y() > 0 else 0.9)))
        self.update()


class Main(QMainWindow):
    def __init__(self, img_dir, scenes_path, scene):
        super().__init__()
        self.img_dir = img_dir
        self.scenes_path = scenes_path
        self.scene = scene
        self.mode = "allow"
        self.setWindowTitle("停车区标定 - %s" % scene)
        self.resize(1400, 900)

        self.files = sorted(f for f in os.listdir(img_dir)
                            if f.lower().endswith((".jpg", ".jpeg", ".png")))
        self.idx = 0
        self.scenes = {}
        if os.path.exists(scenes_path):
            try:
                self.scenes = json.load(open(scenes_path, encoding="utf-8"))
            except Exception:
                self.scenes = {}
        self.scenes.setdefault(scene, {"allow": [], "deny": []})
        self.current = self.scenes[scene]

        self.canvas = Canvas(self)
        bar = QHBoxLayout()
        self.lbl = QLabel("—")
        bar.addWidget(self.lbl, 1)
        for t, fn in (("闭合 (Enter)", self.close_poly), ("切模式 (Tab)", self.toggle_mode),
                      ("保存 (S)", self.save), ("上一张", self.prev), ("下一张", self.next),
                      ("清空 (C)", self.clear)):
            b = QPushButton(t)
            b.clicked.connect(fn)
            bar.addWidget(b)

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas, 1)
        lay.addLayout(bar)
        self.setCentralWidget(central)
        self.setStyleSheet("QWidget{background:#1c1c1e}QLabel{color:#eee}"
                           "QPushButton{background:#333;color:#eee;border:1px solid #555;"
                           "padding:5px 10px;border-radius:4px}QPushButton:hover{background:#444}")

        QShortcut(QKeySequence("Return"), self, self.close_poly)
        QShortcut(QKeySequence("Enter"), self, self.close_poly)
        QShortcut(QKeySequence("Tab"), self, self.toggle_mode)
        QShortcut(QKeySequence("S"), self, self.save)
        QShortcut(QKeySequence("A"), self, self.prev)
        QShortcut(QKeySequence("D"), self, self.next)
        QShortcut(QKeySequence("C"), self, self.clear)
        QShortcut(QKeySequence("Backspace"), self, self.undo_point)
        QShortcut(QKeySequence("Escape"), self, self.cancel)

        if self.files:
            self.load(0)
        self.canvas.setFocus()

    def load(self, i):
        self.idx = max(0, min(len(self.files) - 1, i))
        self.canvas.set_image(os.path.join(self.img_dir, self.files[self.idx]))
        self.update_lbl()

    def update_lbl(self):
        d = self.current
        self.lbl.setText("%s（%d/%d）  允许区 %d 点  禁停区 %d 块  模式:%s" %
                         (self.files[self.idx], self.idx + 1, len(self.files),
                          len(d.get("allow", [])), len(d.get("deny", [])), self.mode))

    def close_poly(self):
        if len(self.canvas.pending) < 3:
            self.statusBar().showMessage("至少 3 个点", 2000)
            return
        if self.mode == "allow":
            self.current["allow"] = self.canvas.pending[:]
        else:
            self.current.setdefault("deny", []).append(self.canvas.pending[:])
        self.canvas.pending = []
        self.update_lbl()
        self.canvas.update()

    def toggle_mode(self):
        self.mode = "deny" if self.mode == "allow" else "allow"
        self.canvas.pending = []
        self.update_lbl()
        self.canvas.update()

    def undo_point(self):
        if self.canvas.pending:
            self.canvas.pending.pop()
            self.canvas.update()
        self.update_lbl()

    def cancel(self):
        self.canvas.pending = []
        self.canvas.update()

    def clear(self):
        if self.mode == "allow":
            self.current["allow"] = []
        else:
            self.current["deny"] = []
        self.canvas.pending = []
        self.update_lbl()
        self.canvas.update()

    def save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.scenes_path)), exist_ok=True)
        json.dump(self.scenes, open(self.scenes_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        self.statusBar().showMessage("已保存 %s" % self.scenes_path, 3000)

    def prev(self):
        self.load(self.idx - 1)

    def next(self):
        self.load(self.idx + 1)

    def closeEvent(self, e):
        self.save()
        e.accept()


def main():
    img_dir = sys.argv[1] if len(sys.argv) > 1 else r"J:\project-1-at-2026-11-13-17-44-a1d4a70f\images"
    scenes = sys.argv[2] if len(sys.argv) > 2 else r"D:\bick\dataset_school\scenes.json"
    scene = sys.argv[3] if len(sys.argv) > 3 else "school_row"
    app = QApplication(sys.argv)
    w = Main(img_dir, scenes, scene)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
