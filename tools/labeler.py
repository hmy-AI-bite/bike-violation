"""自行车数据集精修工具（PySide6）。

用法:
    python labeler.py [数据集目录]

支持标准 YOLO 数据集，或直接选择原始图片文件夹（含子目录）。

快捷键
    ← / A      上一张（自动保存）
    → / D      下一张（自动保存）
    Ctrl+S     保存
    空格       显示/隐藏所有框（对照原图）
    Delete     删除选中的框
    W          新建框模式；也可直接在空白处拖拽画框
    Ctrl+Z     撤销
    滚轮       缩放
    中键拖拽   移动画面（不修改标注）
    R          重置缩放和画面位置
    1 / 2 / 3 / 4 / 5  合规 / 压线 / 出界 / 占道 / 无法判定（选中框后）
    Enter      保存并选择下一辆，连续点选两个接地点（任意顺序）
    Esc        取消接地点点选

鼠标
    点击框内    选中（可拖动整体移动）
    拖角点      调整大小
    空白处拖拽  画新框
"""

from pathlib import Path
import sys
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.py").is_file())))
import config
config.activate()

import json
import os
import copy
import math
import argparse

from PySide6.QtCore import QEasingCurve, QPoint, QRect, Qt, QVariantAnimation
from PySide6.QtGui import (QBrush, QColor, QFont, QFontDatabase, QImage, QKeySequence, QPainter,
                           QPen, QPixmap, QShortcut)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel,
                               QMainWindow, QMessageBox, QPushButton, QVBoxLayout,
                               QWidget)

CLASS_NAMES = ["bicycle", "parking_area"]
CONF_GREEN, CONF_YELLOW = 0.5, 0.35
HANDLE = 10  # 角点热区（屏幕像素）
PARKING_STATES = ["未标注", "合规", "压线", "出界", "占道", "无法判定"]
DEFAULT_PARKING_STATE = "合规"
PARKING_HOTKEYS = {"1": "合规", "2": "压线", "3": "出界", "4": "占道", "5": "无法判定"}
STATE_COLORS = {"未标注": "#5aaaff", "合规": "#00dc5a", "压线": "#ffbe00",
                "出界": "#ff5555", "占道": "#ff5555", "无法判定": "#bbbbbb"}
CONTACT_NAMES = {"point1": "接地点1", "point2": "接地点2"}


def empty_contacts():
    return {wheel: {"visibility": "unobservable", "point": None, "reviewed": False}
            for wheel in CONTACT_NAMES}


class Doc:
    """一张图的标注文档（纯逻辑，不依赖 GUI）。"""

    def __init__(self, img_path, label_path):
        self.img_path = img_path
        self.label_path = label_path
        self.boxes = []       # [cls, x1, y1, x2, y2] 归一化
        self.conf = []        # 每个框的置信度，-1 表示未知
        self.states = []      # 人工停车状态，与检测类别分开保存
        self.contacts = []    # 原图归一化坐标；看不清时不猜坐标
        self.undo = []
        self.dirty = False
        self.load()

    def load(self):
        self.boxes, self.conf, self.states, self.contacts = [], [], [], []
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
            self.states.append(DEFAULT_PARKING_STATE)
            self.contacts.append(empty_contacts())
        review_path = Path(self.label_path).with_suffix(".review.json")
        if review_path.exists():
            review = json.loads(review_path.read_text(encoding="utf-8"))
            # 已有人工记录时，匹配失败仍需复核，不能自动变成合规。
            self.states = ["未标注"] * len(self.boxes)
            for i, record in enumerate(review.get("objects", [])):
                if i >= len(self.boxes):
                    break
                # 外部修改或重排过检测框时，不把旧结论套在别的车上。
                if record.get("box") == self.yolo_box(self.boxes[i]):
                    state = record.get("parking_state", "未标注")
                    self.states[i] = state if state in PARKING_STATES else "未标注"
                    contacts = record.get("contact_points", {})
                    if not isinstance(contacts, dict):
                        contacts = {}
                    for wheel in CONTACT_NAMES:
                        # v2 used front/rear; preserve its coordinates, now as an
                        # unordered pair. Never require relabeling old points.
                        legacy = {"point1": "front", "point2": "rear"}[wheel]
                        item = contacts.get(wheel, contacts.get(legacy, {}))
                        if not isinstance(item, dict):
                            continue
                        visibility, point = item.get("visibility"), item.get("point")
                        if visibility in ("unobservable", "not_applicable"):
                            self.contacts[i][wheel] = {"visibility": visibility, "point": None,
                                                      "reviewed": item.get("reviewed", True) is True}
                        elif (visibility == "visible" and isinstance(point, list) and len(point) == 2
                              and all(isinstance(x, (int, float)) and math.isfinite(x)
                                      and 0 <= x <= 1 for x in point)):
                            self.contacts[i][wheel] = {"visibility": visibility, "point": point[:],
                                                      "reviewed": True}
        self.dirty = False

    @staticmethod
    def yolo_box(b):
        x1, y1, x2, y2 = [max(0.0, min(1.0, v)) for v in b[1:]]
        return [int(b[0]), *[round(v, 6) for v in
                ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)]]

    def state(self, index):
        return self.states[index] if index < len(self.states) else DEFAULT_PARKING_STATE

    def contact(self, index):
        return self.contacts[index] if index < len(self.contacts) else empty_contacts()

    def set_contact(self, index, wheel, visibility, point=None):
        if not 0 <= index < len(self.boxes) or wheel not in CONTACT_NAMES:
            raise ValueError("先选择车辆和车轮")
        if visibility not in ("visible", "unobservable", "not_applicable", "unmarked"):
            raise ValueError("无效的接地点可见性")
        if visibility == "visible":
            if point is None or len(point) != 2 or not all(math.isfinite(x) and 0 <= x <= 1 for x in point):
                raise ValueError("接地点必须位于原图范围内")
            point = [round(x, 8) for x in point]
        else:
            point = None
        self.push_undo()
        while len(self.contacts) < len(self.boxes):
            self.contacts.append(empty_contacts())
        self.contacts[index][wheel] = {"visibility": "unobservable" if visibility == "unmarked" else visibility,
                                       "point": point, "reviewed": visibility != "unmarked"}
        self.dirty = True

    def confirm_contacts(self, index):
        """Enter explicitly accepts any remaining default 'cannot see' values."""
        if not 0 <= index < len(self.boxes) or self.boxes[index][0] != 0:
            return
        if all(item["reviewed"] for item in self.contact(index).values()):
            return
        self.push_undo()
        while len(self.contacts) < len(self.boxes):
            self.contacts.append(empty_contacts())
        for item in self.contacts[index].values():
            item["reviewed"] = True
        self.dirty = True

    def save(self):
        d = os.path.dirname(self.label_path)
        os.makedirs(d, exist_ok=True)
        lines = []
        records = []
        for i, b in enumerate(self.boxes):
            c = int(b[0])
            x1, y1 = max(0.0, min(1.0, b[1])), max(0.0, min(1.0, b[2]))
            x2, y2 = max(0.0, min(1.0, b[3])), max(0.0, min(1.0, b[4]))
            if x2 - x1 < 1e-4 or y2 - y1 < 1e-4:
                continue
            lines.append("%d %.6f %.6f %.6f %.6f" %
                         (c, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
            records.append({"box": self.yolo_box(b), "parking_state": self.state(i),
                            "contact_points": self.contact(i)})
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
        review_path = Path(self.label_path).with_suffix(".review.json")
        backup = review_path.with_suffix(".json.bak")
        if review_path.exists() and not backup.exists():
            # Preserve the pre-contact human decisions before upgrading a sidecar.
            with backup.open("xb") as stream:
                stream.write(review_path.read_bytes())
        temporary = review_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"version": 3, "image": os.path.abspath(self.img_path),
                                        "contact_coordinate_system": "normalized_original_image",
                                        "contact_order": "unordered",
                                        "objects": records}, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(review_path)
        self.dirty = False

    def push_undo(self):
        self.undo.append(([b[:] for b in self.boxes], self.conf[:],
                          [self.state(i) for i in range(len(self.boxes))],
                          copy.deepcopy([self.contact(i) for i in range(len(self.boxes))])))
        if len(self.undo) > 100:
            self.undo.pop(0)

    def do_undo(self):
        if self.undo:
            self.boxes, self.conf, self.states, self.contacts = self.undo.pop()
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
        self.pan_offset = QPoint(0, 0)
        self.pan_anchor = None
        self.contact_target = None
        self.follow_animation = QVariantAnimation(self)
        self.follow_animation.setDuration(160)
        self.follow_animation.setEasingCurve(QEasingCurve.InOutCubic)
        self.follow_animation.valueChanged.connect(self._set_pan)
        self.setFocusPolicy(Qt.StrongFocus)

    # ---------- 坐标换算 ----------
    def _fit(self):
        if not self.pix:
            return 0.0, 0.0, 1.0
        iw, ih = self.pix.width(), self.pix.height()
        s = min(self.width() / iw, self.height() / ih) * self.zoom
        ox = (self.width() - iw * s) / 2 + self.pan_offset.x()
        oy = (self.height() - ih * s) / 2 + self.pan_offset.y()
        return ox, oy, s

    def reset_view(self):
        self.follow_animation.stop()
        self.zoom = 1.0
        self.pan_offset = QPoint(0, 0)
        self.pan_anchor = None
        self.unsetCursor()
        self.update()

    def _set_pan(self, offset):
        self.pan_offset = offset
        self.update()

    def follow_vehicle(self, index, anchor=None):
        """Follow the next vehicle with a short pan, preserving the user's zoom."""
        self.follow_animation.stop()
        center = self.box_rect(self.win.doc.boxes[index]).center()
        anchor = center if anchor is None else anchor
        target = QPoint(max(int(self.width() * .2), min(int(self.width() * .8), anchor.x())),
                        max(int(self.height() * .2), min(int(self.height() * .8), anchor.y())))
        offset = self.pan_offset + target - center
        if offset != self.pan_offset:
            self.follow_animation.setStartValue(self.pan_offset)
            self.follow_animation.setEndValue(offset)
            self.follow_animation.start()

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
            elif doc.state(i) != "未标注":
                col = QColor(STATE_COLORS[doc.state(i)])
            p.setPen(QPen(col, 3 if i == self.sel else 2))
            p.setBrush(Qt.NoBrush)
            p.drawRect(r)
            tag = "%d %s" % (i + 1, doc.state(i))
            p.fillRect(QRect(r.left(), max(0, r.top() - 18), p.fontMetrics().horizontalAdvance(tag) + 8, 18),
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
            for wheel, color in (("point1", "#ffb347"), ("point2", "#55e0dc")):
                item = doc.contact(self.sel)[wheel]
                if item["visibility"] == "visible":
                    x, y = item["point"]
                    pt = self.to_screen(x * self.pix.width(), y * self.pix.height())
                    p.setPen(QPen(QColor("black"), 3))
                    p.setBrush(QColor(color))
                    p.drawEllipse(pt, 6, 6)
                    p.setPen(QColor(color))
                    p.drawText(pt + QPoint(9, -9), CONTACT_NAMES[wheel])

    # ---------- 鼠标 ----------
    def mousePressEvent(self, e):
        if not self.pix:
            return
        # Freeze a follow transition before interpreting an annotation click.
        self.follow_animation.stop()
        if e.button() == Qt.MiddleButton:
            if self.mode is None:
                self.pan_anchor = e.position().toPoint()
                self.setCursor(Qt.ClosedHandCursor)
                e.accept()
            return
        if e.button() != Qt.LeftButton or self.pan_anchor is not None:
            return
        pos = e.position().toPoint()
        if self.contact_target is not None:
            index, wheel = self.contact_target
            x, y = self.to_img(pos)
            if not (0 <= x < self.pix.width() and 0 <= y < self.pix.height()):
                self.win.statusBar().showMessage("请点击图片内轮胎与地面相接的位置", 3000)
                return
            self.win.doc.set_contact(index, wheel, "visible", [x / self.pix.width(), y / self.pix.height()])
            self.win.contact_recorded()
            self.update()
            return
        idx = self.hit(pos)
        if self.win.contact_mode_box.isChecked():
            if idx >= 0 and self.win.doc.boxes[idx][0] == 0:
                self.sel = idx
                self.win.arm_next_contact()
                self.update()
            return
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
        if self.pan_anchor is not None:
            self.pan_offset += pos - self.pan_anchor
            self.pan_anchor = pos
            self.update()
            e.accept()
            return
        if self.mode is None:
            if self.contact_target is not None:
                self.setCursor(Qt.CrossCursor)
                return
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
        if e.button() == Qt.MiddleButton:
            if self.pan_anchor is not None:
                self.pan_offset += e.position().toPoint() - self.pan_anchor
                self.pan_anchor = None
                self.unsetCursor()
                self.update()
                e.accept()
            return
        if e.button() != Qt.LeftButton or self.pan_anchor is not None:
            return
        if self.mode == "create" and self.preview_box:
            a = self.drag_from
            c = self.to_img(e.position().toPoint())
            x1, y1 = min(a[0], c[0]) / self.pix.width(), min(a[1], c[1]) / self.pix.height()
            x2, y2 = max(a[0], c[0]) / self.pix.width(), max(a[1], c[1]) / self.pix.height()
            if (x2 - x1) * self.pix.width() > 6 and (y2 - y1) * self.pix.height() > 6:
                self.win.doc.push_undo()
                self.win.doc.boxes.append([0, x1, y1, x2, y2])
                self.win.doc.conf.append(-1.0)
                self.win.doc.states.append(DEFAULT_PARKING_STATE)
                self.sel = len(self.win.doc.boxes) - 1
                self.win.doc.dirty = True
        self.mode = None
        self.preview_box = None
        self.win.update_status()
        self.update()

    def wheelEvent(self, e):
        self.follow_animation.stop()
        self.zoom = max(0.3, min(4.0, self.zoom * (1.1 if e.angleDelta().y() > 0 else 0.9)))
        self.update()


class Main(QMainWindow):
    def __init__(self, root=None, contact_mode=False):
        super().__init__()
        # Some packaged/offscreen Qt environments do not enumerate Windows fonts.
        if not QFontDatabase.families():
            font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/msyh.ttc"
            font_id = QFontDatabase.addApplicationFont(str(font_path)) if font_path.is_file() else -1
            families = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
            if families:
                QApplication.instance().setFont(QFont(families[0], 9))
        self.root = str(root) if root else str(config.DATA_ROOT)
        self.setWindowTitle("自行车标注精修（含停车状态）")
        self.resize(1400, 900)

        self.items = []
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
        dataset_bar = QHBoxLayout()
        self.choose_btn = QPushButton("选择数据集…")
        self.choose_btn.clicked.connect(self.choose_dataset)
        dataset_bar.addWidget(self.choose_btn)
        self.reset_view_btn = QPushButton("重置视角 (R)")
        self.reset_view_btn.clicked.connect(self.canvas.reset_view)
        self.reset_view_btn.setToolTip("滚轮缩放；按住鼠标中键拖动画面")
        dataset_bar.addWidget(self.reset_view_btn)
        self.dataset_lbl = QLabel()
        self.dataset_lbl.setWordWrap(True)
        self.dataset_lbl.setStyleSheet("color:#eee;padding:6px")
        dataset_bar.addWidget(self.dataset_lbl, 1)
        lay.addLayout(dataset_bar)
        review_bar = QHBoxLayout()
        hint = QLabel("选中车框 → 1合规  2压线  3出界  4占道  5无法判定：")
        hint.setStyleSheet("color:#eee;padding:6px")
        review_bar.addWidget(hint)
        self.state_combo = QComboBox()
        self.state_combo.addItems(PARKING_STATES)
        self.state_combo.setEnabled(False)
        self.state_combo.activated.connect(self.set_parking_state)
        self.state_combo.setStyleSheet("color:#eee;padding:5px;min-width:120px")
        review_bar.addWidget(self.state_combo)
        self.review_progress = QLabel()
        self.review_progress.setStyleSheet("color:#eee;padding:6px")
        review_bar.addWidget(self.review_progress, 1)
        lay.addLayout(review_bar)
        contact_bar = QHBoxLayout()
        self.contact_mode_box = QCheckBox("连续点选：两点任意顺序 · Enter 下一辆")
        self.contact_mode_box.setChecked(contact_mode)
        self.contact_mode_box.setStyleSheet("color:#eee;padding:4px")
        self.contact_mode_box.setToolTip("连续点选时不拖动车框；取消勾选可编辑框。Esc 取消当前点选后，可点击另一个车框。")
        self.contact_mode_box.toggled.connect(self.contact_mode_changed)
        contact_bar.addWidget(self.contact_mode_box)
        self.contact_buttons = []
        for wheel, name in CONTACT_NAMES.items():
            button = QPushButton("标" + name)
            button.clicked.connect(lambda checked=False, wheel=wheel: self.pick_contact(wheel))
            contact_bar.addWidget(button)
            self.contact_buttons.append(button)
            button = QPushButton(name + "看不清")
            button.clicked.connect(lambda checked=False, wheel=wheel: self.mark_contact(wheel, "unobservable"))
            contact_bar.addWidget(button)
            self.contact_buttons.append(button)
        self.not_applicable_btn = QPushButton("倒车 / 轮胎离地")
        self.not_applicable_btn.clicked.connect(self.mark_contacts_not_applicable)
        self.contact_buttons.append(self.not_applicable_btn)
        contact_bar.addWidget(self.not_applicable_btn)
        self.contact_progress = QLabel("默认两个点都看不清；只点能看清的接地点，Enter 确认并下一辆")
        self.contact_progress.setWordWrap(True)
        self.contact_progress.setStyleSheet("color:#eee;padding:6px")
        contact_bar.addWidget(self.contact_progress, 1)
        lay.addLayout(contact_bar)
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
        QShortcut(QKeySequence("Ctrl+O"), self, self.choose_dataset)
        QShortcut(QKeySequence("R"), self, self.canvas.reset_view)
        QShortcut(QKeySequence("Escape"), self, self.cancel_contact)
        for key in ("Return", "Enter"):
            shortcut = QShortcut(QKeySequence(key), self, self.next_vehicle)
            shortcut.setAutoRepeat(False)
        for key, state in PARKING_HOTKEYS.items():
            shortcut = QShortcut(QKeySequence(key), self,
                                 lambda state=state: self.set_parking_state(PARKING_STATES.index(state)))
            shortcut.setAutoRepeat(False)

        if root:
            self.open_dataset(root)
        else:
            self.dataset_lbl.setText("点击“选择数据集…”打开原始图片文件夹")
            self.review_progress.setText("先选择图片文件夹，再逐车画框并设置停车状态")
        self.canvas.setFocus()

    # ---------- 数据 ----------
    def choose_dataset(self):
        folder = QFileDialog.getExistingDirectory(
            self, "选择原始图片文件夹或 YOLO 数据集目录", self.root)
        if folder:
            self.open_dataset(folder)

    def open_dataset(self, folder):
        root = Path(folder).resolve()
        images, labels = config.annotation_paths(root)
        items, destinations = [], set()
        for path in sorted(images.rglob("*")):
            if (not path.is_file() or path.suffix.lower() not in
                    (".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue
            relative = path.relative_to(images)
            if any(p.startswith(".") or p.lower() == "labels" for p in relative.parts[:-1]):
                continue
            destination = str(relative.with_suffix(".txt")).casefold()
            if destination in destinations:
                QMessageBox.warning(self, "文件名冲突", "同一目录存在同名、不同后缀的图片，"
                                    "请先重命名，避免标注覆盖：\n" + str(relative))
                return False
            destinations.add(destination)
            items.append((str(relative.parent), relative.name))
        if not items:
            QMessageBox.warning(self, "没有图片", "所选目录及其子目录中没有支持的图片。\n"
                                "支持 JPG、JPEG、PNG、BMP、WEBP。")
            return False
        try:
            self.save(silent=True)
        except OSError as exc:
            QMessageBox.critical(self, "保存失败", "当前标注未能保存，未切换数据集：\n" + str(exc))
            return False
        self.root = str(root)
        self.images_root, self.labels_root = str(images), str(labels)
        self.items, self.doc = items, None
        self._dets = None
        self.canvas.mode = self.canvas.preview_box = self.canvas.drag_from = None
        self.canvas.reset_view()
        self.canvas.show_boxes = True
        self.dataset_lbl.setText("图片：%s（%d 张）  |  标注：%s" % (images, len(items), labels))
        self.dataset_lbl.setToolTip(self.dataset_lbl.text())
        self.load_index(0)
        return True

    def load_index(self, idx):
        self.save(silent=True)
        self.canvas.follow_animation.stop()
        self.i = max(0, min(len(self.items) - 1, idx))
        split, fn = self.items[self.i]
        stem = os.path.splitext(fn)[0]
        img = os.path.join(self.images_root, split, fn)
        lab = os.path.join(self.labels_root, split, stem + ".txt")
        self.doc = Doc(img, lab)
        self.canvas.pix = QPixmap(img)
        self.apply_conf(split, stem)
        self.canvas.sel = -1
        self.canvas.contact_target = None
        if self.contact_mode_box.isChecked():
            self.contact_mode_changed(True)
        self.update_status()
        self.canvas.update()

    def apply_conf(self, split, stem):
        """把 detections.json 里的置信度按 IoU 配回框上，用于上色。"""
        p = config.output_path("relabel/auto/detections.json") if (
            Path(self.root).resolve() == config.NETWORK_DATASET.resolve()) else ""
        if self._dets is None:
            self._dets = {}
            if os.path.exists(p):
                try:
                    for rec in json.load(open(p, encoding="utf-8")):
                        self._dets[(rec["split"], os.path.splitext(rec["image"])[0])] = rec
                except Exception:
                    self._dets = {}
        rec = self._dets.get((split, stem)) or self._dets.get(("train", stem))
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
            if self.canvas.sel < len(self.doc.states):
                del self.doc.states[self.canvas.sel]
            if self.canvas.sel < len(self.doc.contacts):
                del self.doc.contacts[self.canvas.sel]
            self.canvas.sel = -1
            self.canvas.contact_target = None
            self.doc.dirty = True
            self.update_status()
            self.canvas.update()

    def undo(self):
        previous_sel = self.canvas.sel
        if self.doc and self.doc.do_undo():
            self.canvas.sel = -1
            self.canvas.contact_target = None
            if self.contact_mode_box.isChecked():
                if (0 <= previous_sel < len(self.doc.boxes)
                        and self.doc.boxes[previous_sel][0] == 0):
                    self.canvas.sel = previous_sel
                self.contact_mode_changed(True)
            self.update_status()
            self.canvas.update()

    def set_parking_state(self, index):
        sel = self.canvas.sel
        if not self.doc or not 0 <= sel < len(self.doc.boxes):
            return
        state = PARKING_STATES[index]
        if self.doc.state(sel) == state:
            return
        self.doc.push_undo()
        while len(self.doc.states) < len(self.doc.boxes):
            self.doc.states.append(DEFAULT_PARKING_STATE)
        self.doc.states[sel] = state
        self.doc.dirty = True
        self.update_status()
        self.canvas.update()

    def pick_contact(self, wheel):
        sel = self.canvas.sel
        if self.doc and 0 <= sel < len(self.doc.boxes):
            self.canvas.show_boxes = True
            self.canvas.mode = None
            self.canvas.contact_target = (sel, wheel)
            self.canvas.setCursor(Qt.CrossCursor)
            self.canvas.setFocus()
            self.update_status()

    def arm_next_contact(self):
        sel = self.canvas.sel
        self.canvas.contact_target = None
        if self.doc and 0 <= sel < len(self.doc.boxes) and self.doc.boxes[sel][0] == 0:
            for wheel in CONTACT_NAMES:
                if not self.doc.contact(sel)[wheel]["reviewed"]:
                    self.pick_contact(wheel)
                    return
        self.update_status()

    def contact_recorded(self):
        self.canvas.contact_target = None
        if self.contact_mode_box.isChecked():
            self.arm_next_contact()
        else:
            self.update_status()

    def contact_mode_changed(self, enabled):
        if not enabled:
            self.cancel_contact()
            return
        self.canvas.mode = self.canvas.preview_box = self.canvas.drag_from = None
        if not self.doc:
            return
        vehicles = [i for i, box in enumerate(self.doc.boxes) if box[0] == 0]
        if self.canvas.sel not in vehicles:
            unfinished = [i for i in vehicles if any(
                not item["reviewed"] for item in self.doc.contact(i).values())]
            self.canvas.sel = next(iter(unfinished or vehicles), -1)
        self.arm_next_contact()
        self.canvas.show_boxes = True
        if self.canvas.sel >= 0:
            self.canvas.follow_vehicle(self.canvas.sel)
        self.canvas.update()

    def next_vehicle(self):
        if not self.doc:
            return
        # A failed save must not advance past the current vehicle.
        try:
            self.doc.confirm_contacts(self.canvas.sel)
            if self.doc.dirty:
                self.save(silent=True)
        except OSError as exc:
            QMessageBox.critical(self, "保存失败", "未切换车辆，请先处理保存错误：\n" + str(exc))
            return
        vehicles = [i for i, box in enumerate(self.doc.boxes) if box[0] == 0]
        following = [i for i in vehicles if i > self.canvas.sel]
        if not following:
            self.canvas.contact_target = None
            self.update_status()
            self.canvas.update()
            self.statusBar().showMessage("本图已到最后一辆；按 D 或右箭头切换下一张图片", 5000)
            return
        anchor = (self.canvas.box_rect(self.doc.boxes[self.canvas.sel]).center()
                  if self.canvas.sel in vehicles else None)
        self.contact_mode_box.blockSignals(True)
        self.contact_mode_box.setChecked(True)
        self.contact_mode_box.blockSignals(False)
        self.canvas.mode = self.canvas.preview_box = self.canvas.drag_from = None
        self.canvas.sel = following[0]
        self.canvas.show_boxes = True
        self.arm_next_contact()
        self.canvas.follow_vehicle(self.canvas.sel, anchor)
        self.canvas.setFocus()
        self.canvas.update()

    def cancel_contact(self):
        self.canvas.contact_target = None
        self.canvas.unsetCursor()
        self.update_status()

    def mark_contact(self, wheel, visibility):
        sel = self.canvas.sel
        if self.doc and 0 <= sel < len(self.doc.boxes):
            self.doc.set_contact(sel, wheel, visibility)
            self.contact_recorded()
            self.canvas.update()

    def mark_contacts_not_applicable(self):
        sel = self.canvas.sel
        if self.doc and 0 <= sel < len(self.doc.boxes):
            # One undo restores both wheels together.
            self.doc.set_contact(sel, "point1", "not_applicable")
            self.doc.contacts[sel]["point2"] = {"visibility": "not_applicable", "point": None,
                                                "reviewed": True}
            self.cancel_contact()
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
        selected = 0 <= sel < len(self.doc.boxes)
        for button in self.contact_buttons:
            button.setEnabled(selected and self.doc.boxes[sel][0] == 0)
        visibility_text = {"unmarked": "未标", "visible": "已点", "unobservable": "看不清",
                           "not_applicable": "不适用"}
        done = sum(all(item["reviewed"] for item in self.doc.contact(i).values())
                   for i, box in enumerate(self.doc.boxes) if box[0] == 0)
        total = sum(box[0] == 0 for box in self.doc.boxes)
        text = "接地点已确认 %d/%d 辆" % (done, total)
        if selected:
            text += " | " + "；".join(CONTACT_NAMES[wheel] + "：" + visibility_text[item["visibility"]]
                                    + ("（默认）" if not item["reviewed"] else "")
                                    for wheel, item in self.doc.contact(sel).items())
        if self.canvas.contact_target is not None:
            text += " | 请在图上点" + CONTACT_NAMES[self.canvas.contact_target[1]] + "（任意车轮，Esc取消）"
        elif selected and self.contact_mode_box.isChecked():
            text += " | Enter 下一辆"
        self.contact_progress.setText(text)
        self.state_combo.setEnabled(selected)
        self.state_combo.setCurrentIndex(PARKING_STATES.index(self.doc.state(sel)) if selected else 0)
        reviewed = sum(self.doc.state(i) != "未标注" for i in range(len(self.doc.boxes)))
        self.review_progress.setText("本图已设置状态 %d/%d 辆（默认合规）；看不清选“无法判定”" %
                                     (reviewed, len(self.doc.boxes)))
        sel_txt = ""
        if 0 <= sel < len(self.doc.boxes):
            sel_txt = " | 选中 #%d：%s" % (sel + 1, self.doc.state(sel))
        self.lbl.setText("%s/%s   共 %d 框%s%s" %
                         (self.items[self.i][0], self.items[self.i][1],
                          len(self.doc.boxes), sel_txt,
                          "   ● 未保存" if self.doc.dirty else ""))

    def closeEvent(self, e):
        self.save(silent=True)
        e.accept()


def main():
    parser = argparse.ArgumentParser(description="自行车框、停车状态与车轮接地点标注")
    parser.add_argument("root", nargs="?", default=None)
    parser.add_argument("--contacts", action="store_true", help="打开即连续标两个无序接地点，Enter 下一辆")
    args = parser.parse_args()
    root = args.root or config.data_path('dataset_v2')
    if args.root is None and not Path(root).is_dir():
        root = None
    app = QApplication(sys.argv)
    w = Main(root, contact_mode=args.contacts)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
