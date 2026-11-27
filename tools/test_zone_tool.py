"""无显示器验证标定工具：按图存区域、闭合、保存、加载。"""

from pathlib import Path
import sys
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.py").is_file())))
import config
config.activate()

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from zone_tool import Main  # noqa: E402

import tempfile
from unittest.mock import patch
from PySide6.QtGui import QImage, QMouseEvent
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtTest import QTest

fixture = tempfile.TemporaryDirectory(prefix="bike_zones_", dir=config.ROOT / "tools")
IMG = str(Path(fixture.name) / "images")
Path(IMG).mkdir()
for name in ["sample_1.jpg", "sample_2.jpg"]:
    image = QImage(640, 480, QImage.Format_RGB32)
    image.fill(0xFFFFFFFF)
    assert image.save(str(Path(IMG) / name))
OUT = str(Path(fixture.name) / "scenes.json")

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
w2.show()
w2.activateWindow()
w2.canvas.setFocus()
app.processEvents()
before = json.loads(json.dumps(w2.doc()))
w2.canvas.zoom = 2.0
w2.canvas.pending = [(0.2, 0.2)]
pending = w2.canvas.pending[:]
initial = w2.canvas.to_screen(100, 100)
def mouse(kind, point, button, buttons):
    QApplication.sendEvent(w2.canvas, QMouseEvent(kind, QPointF(point), QPointF(point),
                                                button, buttons, Qt.NoModifier))
mouse(QEvent.MouseButtonPress, QPoint(300, 250), Qt.MiddleButton, Qt.MiddleButton)
mouse(QEvent.MouseMove, QPoint(410, 310), Qt.NoButton, Qt.MiddleButton)
mouse(QEvent.MouseButtonRelease, QPoint(420, 315), Qt.MiddleButton, Qt.NoButton)
assert w2.canvas.pan_offset == QPoint(120, 65)
assert (w2.canvas.to_screen(100, 100) - initial - QPoint(120, 65)).manhattanLength() <= 2
assert w2.canvas.pending == pending and w2.doc() == before
position = QPoint(600, 400)
x,y = w2.canvas.to_img(position)
mouse(QEvent.MouseButtonPress, position, Qt.LeftButton, Qt.LeftButton)
mouse(QEvent.MouseButtonRelease, position, Qt.LeftButton, Qt.NoButton)
assert w2.canvas.pending[-1] == (round(x/640,4), round(y/480,4))
QTest.keyClick(w2.canvas, Qt.Key_Z, Qt.ControlModifier)
assert w2.canvas.pending == pending
QTest.keyClick(w2.canvas, Qt.Key_1)
assert w2.mode == "allow" and w2.canvas.pending == pending
QTest.keyClick(w2.canvas, Qt.Key_2)
assert w2.mode == "deny" and not w2.canvas.pending
w2.canvas.pending = [(0.1,0.2),(0.4,0.2),(0.4,0.5)]
QTest.keyClick(w2.canvas, Qt.Key_Return)
assert len(w2.doc()["deny"]) == 2
QTest.keyClick(w2.canvas, Qt.Key_S, Qt.ControlModifier)
assert len(json.loads(Path(OUT).read_text(encoding='utf-8'))['test_scene']['images'][stem]['deny']) == 2
QTest.keyClick(w2.canvas, Qt.Key_R)
assert w2.canvas.zoom == 1.0 and w2.canvas.pan_offset == QPoint(0,0)
QTest.keyClick(w2.canvas, Qt.Key_Right)
assert w2.idx == 1 and not w2.doc()['deny']
QTest.keyClick(w2.canvas, Qt.Key_Left)
assert w2.idx == 0 and len(w2.doc()['deny']) == 2
app.processEvents()
assert not w2.grab().isNull()
print('中键平移、坐标、数字切模式、Ctrl+Z/Ctrl+S、R重置及左右翻页通过')

raw=Path(fixture.name)/'原始图片'
(raw/'角度一').mkdir(parents=True)
image=QImage(640,480,QImage.Format_RGB32);image.fill(0xFFFFFFFF)
assert image.save(str(raw/'角度一'/'new.jpg'))
with patch('zone_tool.QFileDialog.getExistingDirectory',return_value=str(raw)):
    w2.choose_dataset()
assert w2.files == [str(Path('角度一/new.jpg'))] and w2.mode=='allow'
assert Path(w2.scenes_path)==raw/'scenes.json'
w2.canvas.pending=[(0.1,0.1),(0.5,0.1),(0.5,0.5)]
w2.close_poly();w2.save()
assert len(json.loads((raw/'scenes.json').read_text(encoding='utf-8'))['test_scene']['images']['new']['allow'])==3
with patch('zone_tool.QFileDialog.getExistingDirectory',return_value=''):
    w2.choose_dataset()
assert Path(w2.img_dir)==raw
empty=Path(fixture.name)/'empty';empty.mkdir()
with patch('zone_tool.QMessageBox.warning'):
    assert not w2.open_dataset(empty)
assert Path(w2.img_dir)==raw
print('选择图片目录、子目录读取、区域输出位置与取消选择通过')
w2.close()
fixture.cleanup()
print("\n全部检查通过")
