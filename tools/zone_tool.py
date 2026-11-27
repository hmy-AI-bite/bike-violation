"""停车区标定工具：点几下圈出「允许停放区」和「禁停区(通道)」，存成 scenes.json。

用法:
    python zone_tool.py <图片文件夹> [scenes.json 路径] [场景名]

操作:
    左键        加一个顶点
    Enter       闭合当前多边形（至少 3 个点）
    Backspace   撤销上一个顶点
    Tab         切换 允许区(蓝) / 禁停区(橙)
    1 / 2       允许区 / 禁停区
    中键拖拽    移动画面；滚轮缩放；R 重置视角
    S / Ctrl+S  保存到 scenes.json
    A / D 或左右键 上一张 / 下一张
    Ctrl+O      选择图片文件夹
    Ctrl+Z      撤销上一个顶点
    C           清空当前模式的图形
    Esc         取消正在画的多边形

输出格式:
    {"场景名": {"allow": [[x,y],...], "deny": [[[...]],...]}}   # 归一化坐标
"""

from pathlib import Path
import sys
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.py").is_file())))
import config
config.activate()

import json
import os

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QFont, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
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
        self.pan_offset = QPoint(0, 0)
        self.pan_anchor = None
        self.cursor = QPoint(0, 0)

    def set_image(self, path):
        self.pix = QPixmap(path)
        self.pending = []
        self.pan_anchor = None
        self.unsetCursor()
        self.update()

    def _fit(self):
        if not self.pix:
            return 0.0, 0.0, 1.0
        iw, ih = self.pix.width(), self.pix.height()
        s = min(self.width() / iw, self.height() / ih) * self.zoom
        return ((self.width() - iw * s) / 2 + self.pan_offset.x(),
                (self.height() - ih * s) / 2 + self.pan_offset.y(), s)

    def reset_view(self):
        self.zoom = 1.0
        self.pan_offset = QPoint(0, 0)
        self.pan_anchor = None
        self.unsetCursor()
        self.update()

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
        doc = self.win.doc()

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
        if self.pan_anchor is not None:
            self.pan_offset += self.cursor - self.pan_anchor
            self.pan_anchor = self.cursor
            self.update()
            e.accept()
            return
        if self.pending:
            self.update()

    def mousePressEvent(self, e):
        if not self.pix:
            return
        if e.button() == Qt.MiddleButton:
            self.pan_anchor = e.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            e.accept()
            return
        if e.button() != Qt.LeftButton or self.pan_anchor is not None:
            return
        x, y = self.to_img(e.position().toPoint())
        W, H = self.pix.width(), self.pix.height()
        self.pending.append((round(max(0.0, min(1.0, x / W)), 4),
                             round(max(0.0, min(1.0, y / H)), 4)))
        self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MiddleButton and self.pan_anchor is not None:
            self.pan_offset += e.position().toPoint() - self.pan_anchor
            self.pan_anchor = None
            self.unsetCursor()
            self.update()
            e.accept()

    def wheelEvent(self, e):
        self.zoom = max(0.4, min(4.0, self.zoom * (1.1 if e.angleDelta().y() > 0 else 0.9)))
        self.update()


class Main(QMainWindow):
    def __init__(self, img_dir=None, scenes_path=None, scene=None):
        super().__init__()
        self.img_dir = str(img_dir or config.DATA_ROOT)
        self.scenes_path = None
        self.scene = scene or config.SCENE
        self.mode = "allow"
        self.setWindowTitle("停车区标定 - %s" % self.scene)
        self.resize(1400, 900)

        self.files = []
        self.idx = 0
        self.scenes = {}
        self.scenes.setdefault(self.scene, {"allow": [], "deny": []})
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
        top = QHBoxLayout()
        for title, fn in (("选择图片文件夹…", self.choose_dataset),
                          ("允许区 (1)", lambda: self.set_mode("allow")),
                          ("禁停区 (2)", lambda: self.set_mode("deny")),
                          ("重置视角 (R)", self.canvas.reset_view)):
            button = QPushButton(title)
            button.clicked.connect(fn)
            top.addWidget(button)
        self.dataset_lbl = QLabel("点击“选择图片文件夹…”开始；中键拖拽，滚轮缩放")
        self.dataset_lbl.setWordWrap(True)
        top.addWidget(self.dataset_lbl, 1)
        lay.addLayout(top)
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
        QShortcut(QKeySequence("Ctrl+S"), self, self.save)
        QShortcut(QKeySequence("Ctrl+O"), self, self.choose_dataset)
        QShortcut(QKeySequence("R"), self, self.canvas.reset_view)
        for key, mode in (("1", "allow"), ("2", "deny")):
            shortcut = QShortcut(QKeySequence(key), self, lambda mode=mode: self.set_mode(mode))
            shortcut.setAutoRepeat(False)
        QShortcut(QKeySequence("A"), self, self.prev)
        QShortcut(QKeySequence("D"), self, self.next)
        QShortcut(QKeySequence("Left"), self, self.prev)
        QShortcut(QKeySequence("Right"), self, self.next)
        QShortcut(QKeySequence("C"), self, self.clear)
        QShortcut(QKeySequence("Backspace"), self, self.undo_point)
        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo_point)
        QShortcut(QKeySequence("Escape"), self, self.cancel)
        QShortcut(QKeySequence("P"), self, self.promote)

        if img_dir:
            self.open_dataset(img_dir, scenes_path, self.scene)
        self.canvas.setFocus()

    def choose_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, "选择原始图片文件夹或数据集目录", self.img_dir)
        if folder:
            self.open_dataset(folder)

    def open_dataset(self, folder, scenes_path=None, scene=None):
        images, labels = config.annotation_paths(folder)
        files = sorted(str(p.relative_to(images)) for p in images.rglob("*")
                       if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
                       and not any(part.startswith(".") or part.lower() == "labels"
                                   for part in p.relative_to(images).parts[:-1]))
        if not files:
            QMessageBox.warning(self, "没有图片", "所选目录及其子目录中没有支持的图片。")
            return False
        stems = [Path(name).stem.casefold() for name in files]
        if len(set(stems)) != len(stems):
            QMessageBox.warning(self, "图片名冲突", "子目录中有同名图片，请分别选择对应子目录，避免区域覆盖。")
            return False
        target = Path(scenes_path) if scenes_path else labels.parent / "scenes.json"
        try:
            if self.scenes_path and target.resolve() == Path(self.scenes_path).resolve():
                scenes = self.scenes
            else:
                scenes = json.loads(target.read_text(encoding="utf-8-sig")) if target.exists() else {}
            if not isinstance(scenes, dict):
                raise ValueError("区域文件的顶层必须是对象")
            self.save()
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "打开失败", str(exc))
            return False
        self.scene = scene or self.scene
        self.scenes, self.files = scenes, files
        self.scenes.setdefault(self.scene, {"allow": [], "deny": []})
        self.img_dir, self.scenes_path = str(images), str(target)
        self.mode = "allow"
        self.canvas.reset_view()
        self.dataset_lbl.setText("图片：%s\n区域：%s" % (images, target))
        self.dataset_lbl.setToolTip(self.dataset_lbl.text())
        self.setWindowTitle("停车区标定 - %s" % self.scene)
        self.load(0)
        return True

    def doc(self):
        """当前图片自己的区域（按图存储，避免一张多边形套所有视角）。"""
        if not self.files:
            return self.scenes[self.scene]
        stem = Path(self.files[self.idx]).stem
        return self.scenes[self.scene].setdefault("images", {}).setdefault(
            stem, {"allow": [], "deny": []})

    def load(self, i):
        if not self.files:
            return
        self.idx = max(0, min(len(self.files) - 1, i))
        self.canvas.set_image(os.path.join(self.img_dir, self.files[self.idx]))
        self.update_lbl()

    def update_lbl(self):
        if not self.files:
            self.lbl.setText("请选择图片文件夹")
            return
        d = self.doc()
        self.lbl.setText("%s（%d/%d）  允许区 %d 点  禁停区 %d 块  模式:%s" %
                         (self.files[self.idx], self.idx + 1, len(self.files),
                          len(d.get("allow", [])), len(d.get("deny", [])),
                          "允许区 (1)" if self.mode == "allow" else "禁停区 (2)"))

    def close_poly(self):
        if not self.files:
            return
        if len(self.canvas.pending) < 3:
            self.statusBar().showMessage("至少 3 个点", 2000)
            return
        if self.mode == "allow":
            self.doc()["allow"] = self.canvas.pending[:]
        else:
            self.doc().setdefault("deny", []).append(self.canvas.pending[:])
        self.canvas.pending = []
        self.update_lbl()
        self.canvas.update()

    def toggle_mode(self):
        self.set_mode("deny" if self.mode == "allow" else "allow")

    def set_mode(self, mode):
        if self.mode == mode:
            return
        self.mode = mode
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
        d = self.doc()
        if self.mode == "allow":
            d["allow"] = []
        else:
            d["deny"] = []
        self.canvas.pending = []
        self.update_lbl()
        self.canvas.update()

    def promote(self):
        """把当前图的区域设为该场景的默认区域（固定机位用）。"""
        d = self.doc()
        self.scenes[self.scene]["allow"] = [list(p) for p in d.get("allow", [])]
        self.scenes[self.scene]["deny"] = [[list(q) for q in poly] for poly in d.get("deny", [])]
        self.statusBar().showMessage("已设为场景默认区域（%d 点）" % len(d.get("allow", [])), 3000)

    def save(self):
        if not self.scenes_path:
            return
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
    img_dir = sys.argv[1] if len(sys.argv) > 1 else str(config.SCHOOL_IMAGES)
    scenes = sys.argv[2] if len(sys.argv) > 2 else str(config.SCENES)
    scene = sys.argv[3] if len(sys.argv) > 3 else config.SCENE
    if len(sys.argv) <= 1 and not Path(img_dir).is_dir():
        img_dir = None
    app = QApplication(sys.argv)
    w = Main(img_dir, scenes, scene)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
