"""自行车数据集精修工具（PySide6）。

用法:
    python labeler.py [数据集目录]

默认数据集 D:\\bick\\dataset_v2（images/train、images/val、labels/...）

快捷键
    ← / A      上一张（自动保存）
    → / D      下一张（自动保存）
    Ctrl+S     保存
    空格       显示/隐藏所有框（对照原图）
    Delete     删除选中的框
    W          新建框模式；也可直接在空白处拖拽画框
    Ctrl+Z     撤销
    滚轮       缩放
    Esc        取消当前操作

鼠标
    点击框内    选中（可拖动整体移动）
    拖角点      调整大小
    空白处拖拽  画新框
"""
import json
import os
import sys

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import (QBrush, QColor, QFont, QImage, QKeySequence, QPainter,
                           QPen, QPixmap, QShortcut)
from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
                               QMainWindow, QMessageBox, QPushButton, QVBoxLayout,
                               QWidget)

CLASS_NAMES = ["bicycle", "parking_area"]
CONF_GREEN, CONF_YELLOW = 0.5, 0.35
HANDLE = 10  # 角点热区（屏幕像素）


class Doc:
    """一张图的标注文档（纯逻辑，不依赖 GUI）。"""

    def __init__(self, img_path, label_path):
        self.img_path = img_path
        self.label_path = label_path
        self.boxes = []       # [cls, x1, y1, x2, y2] 归一化
        self.conf = []        # 每个框的置信度，-1 表示未知
        self.undo = []
        self.dirty = False
        self.load()

    def load(self):
        self.boxes, self.conf = [], []
        if not os.path.exists(self.label_path):
            return
        for line in open(self.label_path, encoding="utf-8-sig"):
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            xc, yc, w, h = (float(v) for v in p[1:5])
            self.boxes.append([c, xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
            self.conf.append(-1.0)
        self.dirty = False

    def save(self):
        d = os.path.dirname(self.label_path)
        os.makedirs(d, exist_ok=True)
        lines = []
        for b in self.boxes:
            c = int(b[0])
            x1, y1 = max(0.0, min(1.0, b[1])), max(0.0, min(1.0, b[2]))
            x2, y2 = max(0.0, min(1.0, b[3])), max(0.0, min(1.0, b[4]))
            if x2 - x1 < 1e-4 or y2 - y1 < 1e-4:
                continue
            lines.append("%d %.6f %.6f %.6f %.6f" %
                         (c, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
        if lines:
            if not os.path.exists(self.label_path + ".bak"):
                bak = self.label_path + ".bak"
                try:
                    with open(bak, "w", encoding="utf-8") as f:
                        f.write(open(self.label_path, encoding="utf-8").read() if
                                os.path.exists(self.label_path) else "")
                except OSError:
                    pass
        with open(self.label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        self.dirty = False

    def push_undo(self):
        self.undo.append([b[:] for b in self.boxes])
        if len(self.undo) > 100:
            self.undo.pop(0)

    def do_undo(self):
        if self.undo:
            self.boxes = self.undo.pop()
            self.dirty = True
            return True
        return False


class Canvas(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.setMinimumSize(900, 600)
        self.setMouseTracking(True)
        self.pix = None
        self.mode = None
        self.sel = -1
        self.drag_from = None
        self.preview_box = None
        self.show_boxes = True
        self.zoom = 1.0
        self.setFocusPolicy(Qt.StrongFocus)

    # ---------- 坐标换算 ----------
    def _fit(self):
        if not self.pix:
            return 0.0, 0.0, 1.0
        iw, ih = self.pix.width(), self.pix.height()
        s = min(self.width() / iw, self.height() / ih) * self.zoom
        ox = (self.width() - iw * s) / 2
        oy = (self.height() - ih * s) / 2
        return ox, oy, s

    def to_img(self, pos):
        ox, oy, s = self._fit()
        return ((pos.x() - ox) / s, (pos.y() - oy) / s)

    def to_screen(self, x, y):
        ox, oy, s = self._fit()
        return QPoint(int(ox + x * s), int(oy + y * s))

    def box_rect(self, b):
        ox, oy, s = self._fit()
        return QRect(QPoint(int(ox + b[1] * self.pix.width() * s),
                            int(oy + b[2] * self.pix.height() * s)),
                     QPoint(int(ox + b[3] * self.pix.width() * s),
                            int(oy + b[4] * self.pix.height() * s))).normalized()

    def hit(self, pos):
        for i in range(len(self.win.doc.boxes) - 1, -1, -1):
            if self.box_rect(self.win.doc.boxes[i]).adjusted(-3, -3, 3, 3).contains(pos):
                return i
        return -1

    def corner(self, pos, idx):
        r = self.box_rect(self.win.doc.boxes[idx])
        pts = {"tl": r.topLeft(), "tr": r.topRight(),
               "bl": r.bottomLeft(), "br": r.bottomRight()}
        for k, p in pts.items():
            if (p - pos).manhattanLength() < HANDLE + 6:
                return k
        return None

    # ---------- 绘制 ----------
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(28, 28, 30))
        if not self.pix:
            return
        ox, oy, s = self._fit()
        p.drawPixmap(QRect(int(ox), int(oy), int(self.pix.width() * s),
                           int(self.pix.height() * s)), self.pix)
        if not self.show_boxes:
            return

        doc = self.win.doc
        f = QFont("Microsoft YaHei", 10)
        p.setFont(f)
        for i, b in enumerate(doc.boxes):
            r = self.box_rect(b)
            conf = doc.conf[i] if i < len(doc.conf) else -1
            if conf < 0:
                col = QColor(90, 170, 255)
            elif conf >= CONF_GREEN:
                col = QColor(0, 220, 90)
            elif conf >= CONF_YELLOW:
                col = QColor(255, 190, 0)
            else:
                col = QColor(255, 60, 60)
            if i == self.sel:
                col = QColor(255, 255, 0)
            p.setPen(QPen(col, 3 if i == self.sel else 2))
            p.setBrush(Qt.NoBrush)
            p.drawRect(r)
            tag = "%d" % (i + 1) if conf < 0 else "%d %.2f" % (i + 1, conf)
            p.fillRect(QRect(r.left(), max(0, r.top() - 18), 8 + 7 * len(tag), 18),
                       QBrush(col))
            p.setPen(QPen(QColor(0, 0, 0)))
            p.drawText(r.left() + 3, max(12, r.top() - 5), tag)

        if self.preview_box:
            p.setPen(QPen(QColor(0, 255, 255), 2, Qt.DashLine))
            p.drawRect(self.preview_box)

        if self.sel >= 0 and self.sel < len(doc.boxes):
            r = self.box_rect(doc.boxes[self.sel])
            p.setPen(QPen(QColor(255, 255, 0), 2))
            for pt in (r.topLeft(), r.topRight(), r.bottomLeft(), r.bottomRight()):
                p.fillRect(QRect(pt.x() - 5, pt.y() - 5, 10, 10), QColor(255, 255, 0))

    # ---------- 鼠标 ----------
    def mousePressEvent(self, e):
        if not self.pix or e.button() != Qt.LeftButton:
            return
        pos = e.position().toPoint()
        idx = self.hit(pos)
        if idx >= 0:
            self.sel = idx
            c = self.corner(pos, idx)
            self.mode = "resize" if c else "move"
            self.corner_key = c
            self.drag_from = self.to_img(pos)
            self.win.doc.push_undo()
        else:
            self.sel = -1
            self.mode = "create"
            self.drag_from = self.to_img(pos)
        self.win.update_status()
        self.update()

    def mouseMoveEvent(self, e):
        if not self.pix:
            return
        pos = e.position().toPoint()
        if self.mode is None:
            idx = self.hit(pos)
            if idx >= 0 and self.corner(pos, idx):
                self.setCursor(Qt.SizeFDiagCursor)
            elif idx >= 0:
                self.setCursor(Qt.SizeAllCursor)
            else:
                self.setCursor(Qt.CrossCursor)
            return

        cur = self.to_img(pos)
        doc = self.win.doc
        if self.mode == "create":
            a, b = self.drag_from, cur
            self.preview_box = QRect(self.to_screen(min(a[0], b[0]), min(a[1], b[1])),
                                     self.to_screen(max(a[0], b[0]), max(a[1], b[1])))
            self.update()
            return

        if self.sel < 0:
            return
        b = doc.boxes[self.sel]
        W, H = self.pix.width(), self.pix.height()
        dx = (cur[0] - self.drag_from[0]) / W
        dy = (cur[1] - self.drag_from[1]) / H
        if self.mode == "move":
            b[1] += dx
            b[3] += dx
            b[2] += dy
            b[4] += dy
        elif self.mode == "resize":
            k = self.corner_key
            if "l" in k:
                b[1] += dx
            else:
                b[3] += dx
            if "t" in k:
                b[2] += dy
            else:
                b[4] += dy
            if b[1] > b[3]:
                b[1], b[3] = b[3], b[1]
            if b[2] > b[4]:
                b[2], b[4] = b[4], b[2]
        self.drag_from = cur
        doc.dirty = True
        self.update()

    def mouseReleaseEvent(self, e):
        if self.mode == "create" and self.preview_box:
            a = self.drag_from
            c = self.to_img(e.position().toPoint())
            x1, y1 = min(a[0], c[0]) / self.pix.width(), min(a[1], c[1]) / self.pix.height()
            x2, y2 = max(a[0], c[0]) / self.pix.width(), max(a[1], c[1]) / self.pix.height()
            if (x2 - x1) * self.pix.width() > 6 and (y2 - y1) * self.pix.height() > 6:
                self.win.doc.push_undo()
                self.win.doc.boxes.append([0, x1, y1, x2, y2])
                self.win.doc.conf.append(-1.0)
                self.sel = len(self.win.doc.boxes) - 1
                self.win.doc.dirty = True
        self.mode = None
        self.preview_box = None
        self.win.update_status()
        self.update()

    def wheelEvent(self, e):
        self.zoom = max(0.3, min(4.0, self.zoom * (1.1 if e.angleDelta().y() > 0 else 0.9)))
        self.update()


class Main(QMainWindow):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.setWindowTitle("自行车标注精修")
        self.resize(1400, 900)

        self.items = []
        for split in ("train", "val"):
            d = os.path.join(root, "images", split)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.items.append((split, f))
        self.i = 0
        self.doc = None

        self.canvas = Canvas(self)
        bar = QHBoxLayout()
        self.lbl = QLabel("—")
        self.lbl.setStyleSheet("color:#eee;font:13px 'Microsoft YaHei'")
        bar.addWidget(self.lbl, 1)
        for text, fn in (("← 上一张", self.prev), ("下一张 →", self.next),
                         ("删除选中 (Del)", self.delete_sel), ("撤销 (Ctrl+Z)", self.undo),
                         ("保存 (Ctrl+S)", self.save)):
            btn = QPushButton(text)
            btn.clicked.connect(fn)
            bar.addWidget(btn)

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas, 1)
        lay.addLayout(bar)
        self.setCentralWidget(central)
        self.setStyleSheet("QPushButton{background:#333;color:#eee;border:1px solid #555;"
                           "padding:5px 10px;border-radius:4px}QPushButton:hover{background:#444}"
                           "QWidget{background:#1c1c1e}")

        QShortcut(QKeySequence("Left"), self, self.prev)
        QShortcut(QKeySequence("A"), self, self.prev)
        QShortcut(QKeySequence("Right"), self, self.next)
        QShortcut(QKeySequence("D"), self, self.next)
        QShortcut(QKeySequence("Ctrl+S"), self, self.save)
        QShortcut(QKeySequence("Delete"), self, self.delete_sel)
        QShortcut(QKeySequence("Backspace"), self, self.delete_sel)
        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo)
        QShortcut(QKeySequence("Space"), self, self.toggle_boxes)

        if self.items:
            self.load_index(0)
        else:
            QMessageBox.warning(self, "没有图片", "没找到 images/train 或 images/val")
        self.canvas.setFocus()

    # ---------- 数据 ----------
    def load_index(self, idx):
        self.save(silent=True)
        self.i = max(0, min(len(self.items) - 1, idx))
        split, fn = self.items[self.i]
        stem = os.path.splitext(fn)[0]
        img = os.path.join(self.root, "images", split, fn)
        lab = os.path.join(self.root, "labels", split, stem + ".txt")
        self.doc = Doc(img, lab)
        self.apply_conf(split, stem)
        self.canvas.pix = QPixmap(img)
        self.canvas.sel = -1
        self.update_status()
        self.canvas.update()

    def apply_conf(self, split, stem):
        """把 detections.json 里的置信度按 IoU 配回框上，用于上色。"""
        p = os.path.join(os.path.dirname(self.root), "relabel", "auto", "detections.json")
        if not hasattr(Main, "_dets"):
            Main._dets = {}
            if os.path.exists(p):
                try:
                    for rec in json.load(open(p, encoding="utf-8")):
                        Main._dets[(rec["split"], os.path.splitext(rec["image"])[0])] = rec
                except Exception:
                    Main._dets = {}
        rec = Main._dets.get((split, stem)) or Main._dets.get(("train", stem))
        if not rec:
            return
        dets = [b["box"] for b in rec.get("bicycles", [])]
        W = self.canvas.pix.width()
        H = self.canvas.pix.height()
        for i, b in enumerate(self.doc.boxes):
            if i >= len(self.doc.conf) or self.doc.conf[i] >= 0:
                continue
            best, bc = 0.0, -1
            for k, (dx1, dy1, dx2, dy2) in enumerate(dets):
                px1, py1, px2, py2 = b[1] * W, b[2] * H, b[3] * W, b[4] * H
                ix = max(0, min(px2, dx2) - max(px1, dx1))
                iy = max(0, min(py2, dy2) - max(py1, dy1))
                inter = ix * iy
                if inter <= 0:
                    continue
                ua = (px2 - px1) * (py2 - py1) + (dx2 - dx1) * (dy2 - dy1) - inter
                v = inter / ua if ua > 0 else 0
                if v > best:
                    best, bc = v, rec["bicycles"][k]["conf"]
            if best >= 0.5:
                self.doc.conf[i] = bc

    # ---------- 操作 ----------
    def prev(self):
        if self.i > 0:
            self.load_index(self.i - 1)

    def next(self):
        if self.i < len(self.items) - 1:
            self.load_index(self.i + 1)

    def delete_sel(self):
        if self.doc and 0 <= self.canvas.sel < len(self.doc.boxes):
            self.doc.push_undo()
            del self.doc.boxes[self.canvas.sel]
            if self.canvas.sel < len(self.doc.conf):
                del self.doc.conf[self.canvas.sel]
            self.canvas.sel = -1
            self.doc.dirty = True
            self.update_status()
            self.canvas.update()

    def undo(self):
        if self.doc and self.doc.do_undo():
            self.canvas.sel = -1
            self.update_status()
            self.canvas.update()

    def save(self, silent=False):
        if not self.doc:
            return
        self.doc.save()
        if not silent:
            self.statusBar().showMessage("已保存 %s" % os.path.basename(self.doc.label_path), 2500)

    def toggle_boxes(self):
        self.canvas.show_boxes = not self.canvas.show_boxes
        self.canvas.update()

    def update_status(self):
        if not self.doc:
            return
        sel = self.canvas.sel
        sel_txt = ""
        if 0 <= sel < len(self.doc.boxes):
            c = self.doc.conf[sel]
            sel_txt = " | 选中 #%d %s" % (sel + 1, "" if c < 0 else "conf %.2f" % c)
        self.lbl.setText("%s/%s   共 %d 框%s%s" %
                         (self.items[self.i][0], self.items[self.i][1],
                          len(self.doc.boxes), sel_txt,
                          "   ● 未保存" if self.doc.dirty else ""))

    def closeEvent(self, e):
        self.save(silent=True)
        e.accept()


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else r"D:\bick\dataset_v2"
    app = QApplication(sys.argv)
    w = Main(root)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
