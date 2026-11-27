"""无显示器环境下验证标注工具：加载、删除、保存、撤销、新建框。"""

from pathlib import Path
import sys
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.py").is_file())))
import config
config.activate()

import os
import shutil

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import tempfile
from unittest.mock import patch
from PySide6.QtGui import QImage, QMouseEvent
from PySide6.QtCore import QAbstractAnimation, QEvent, QPoint, QPointF, Qt

fixture = tempfile.TemporaryDirectory(prefix="bike_labeler_", dir=config.ROOT / "tools")
ROOT = fixture.name


def build():
    images = Path(ROOT) / "images/train"
    labels = Path(ROOT) / "labels/train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    names = ["sample_1.jpg", "sample_2.jpg"]
    for name in names:
        image = QImage(640, 480, QImage.Format_RGB32)
        image.fill(0xFFFFFFFF)
        assert image.save(str(images / name))
        (labels / (Path(name).stem + ".txt")).write_text(
            "0 0.3 0.4 0.2 0.3\n0 0.7 0.6 0.2 0.3\n", encoding="utf-8")
    return names


names = build()
from labeler import Doc, Main  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication(sys.argv)
w = Main(ROOT)

print("图片数:", len(w.items))
assert len(w.items) == 2, w.items

stem = os.path.splitext(names[0])[0]
lab = os.path.join(ROOT, "labels", "train", stem + ".txt")
n0 = len(w.doc.boxes)
print("首图框数:", n0)
assert n0 > 0

# 删除第一个框
w.canvas.sel = 0
w.delete_sel()
w.save()
n1 = sum(1 for l in open(lab, encoding="utf-8") if l.strip())
print("删除后框数:", n1)
assert n1 == n0 - 1, (n1, n0)
assert os.path.exists(lab + ".bak"), "未生成备份"

# 撤销
w.undo()
assert len(w.doc.boxes) == n0, "撤销失败"
print("撤销后框数:", len(w.doc.boxes))

# 新建一个框（模拟画框结果）
w.doc.push_undo()
w.doc.boxes.append([0, 0.1, 0.1, 0.3, 0.3])
w.doc.conf.append(-1.0)
w.save()
lines = [l for l in open(lab, encoding="utf-8") if l.strip()]
print("新建后框数:", len(lines))
assert len(lines) == n0 + 1, (len(lines), n0)

# 检查保存格式：5 列、归一化、类号 0
parts = lines[-1].split()
assert parts[0] == "0" and len(parts) == 5, parts
vals = [float(v) for v in parts[1:]]
assert all(0 <= v <= 1 for v in vals), vals
print("保存格式 OK:", lines[-1].strip())

# 翻页
w.next()
print("翻页后:", w.items[w.i][1])
assert w.i == 1

# 从按钮选择多层原始图片目录；同名图片保留各自的目录。
raw = Path(ROOT) / "原始图片"
for angle in ("角度一", "角度二"):
    directory = raw / angle
    directory.mkdir(parents=True)
    image = QImage(320, 240, QImage.Format_RGB32)
    image.fill(0xFFFFFFFF)
    assert image.save(str(directory / "同名.jpg"))
w.doc.boxes = [[0, 0.1, 0.2, 0.3, 0.4]]
w.doc.dirty = True
previous_label = Path(w.doc.label_path)
with patch("labeler.QFileDialog.getExistingDirectory", return_value=str(raw)):
    w.choose_btn.click()
assert w.root == str(raw.resolve()) and len(w.items) == 2
assert previous_label.read_text().startswith("0 0.200000 0.300000")
assert Path(w.doc.label_path) == raw / "labels/角度一/同名.txt"
w.doc.boxes = [[0, 0.2, 0.2, 0.5, 0.6]]
w.doc.dirty = True
w.next()
assert (raw / "labels/角度一/同名.txt").read_text().startswith("0 ")
assert Path(w.doc.label_path) == raw / "labels/角度二/同名.txt"
assert not w.doc.boxes
assert not (raw / "labels/角度二/同名.txt").exists()

# 取消选择、空目录和文件名冲突都保留当前数据集。
with patch("labeler.QFileDialog.getExistingDirectory", return_value=""):
    w.choose_btn.click()
assert w.i == 1 and w.root == str(raw.resolve())
empty = Path(ROOT) / "empty"
empty.mkdir()
with patch("labeler.QMessageBox.warning") as warning:
    assert not w.open_dataset(empty)
    assert warning.called
assert w.i == 1 and w.root == str(raw.resolve())
shutil.copyfile(raw / "角度一/同名.jpg", raw / "角度一/同名.png")
with patch("labeler.QMessageBox.warning") as warning:
    assert not w.open_dataset(raw)
    assert warning.called
assert w.i == 1
assert w.open_dataset(ROOT)
assert Path(w.doc.label_path).parent == Path(ROOT) / "labels/train"
print("选择数据集、自动保存、子目录隔离、取消和错误处理通过")

# 人工停车状态通过下拉框设置，保存重开、移动、删除与撤销都保持对应关系。
from labeler import PARKING_STATES
assert all(w.doc.state(i) == "合规" for i in range(len(w.doc.boxes)))
w.canvas.sel = 0
w.update_status()
assert w.state_combo.isEnabled()
w.state_combo.activated.emit(PARKING_STATES.index("出界"))
assert w.doc.state(0) == "出界" and w.doc.dirty
w.save()
reopened = Doc(w.doc.img_path, w.doc.label_path)
assert reopened.state(0) == "出界"
assert len(Path(w.doc.label_path).read_text().splitlines()[0].split()) == 5
assert Path(w.doc.label_path).read_text().splitlines()[0].startswith("0 ")
w.doc.push_undo()
w.doc.boxes[0][1] += 0.01
w.save()
assert Doc(w.doc.img_path, w.doc.label_path).state(0) == "出界"
w.delete_sel()
assert w.doc.state(0) == "合规"
w.undo()
assert w.doc.state(0) == "出界"
w.canvas.sel = 0
w.state_combo.activated.emit(PARKING_STATES.index("合规"))
assert w.doc.state(0) == "合规"
w.undo()
assert w.doc.state(0) == "出界"
w.save()
w.next()
w.prev()
assert w.doc.state(0) == "出界"
# 外部更改框文件后，不套用原来那辆车的人工结论。
label_path = Path(w.doc.label_path)
original = label_path.read_text()
lines = original.splitlines()
lines[0] = "0 0.800000 0.800000 0.100000 0.100000"
label_path.write_text("\n".join(lines) + "\n")
assert Doc(w.doc.img_path, str(label_path)).state(0) == "未标注"
label_path.write_text(original)
print("停车状态设置、保存重开、框移动、删除撤销与防错配检查通过")

# 整理后的 raw 子目录：标签必须写到独立 annotation_work，不污染原图。
raw_data_root = Path(ROOT) / "data"
campus_image = raw_data_root / "campus/unused/raw/new_batch/images/angle1/a.jpg"
campus_image.parent.mkdir(parents=True)
image = QImage(320, 240, QImage.Format_RGB32)
image.fill(0xFFFFFFFF)
assert image.save(str(campus_image))
original_image = campus_image.read_bytes()
with patch.object(config, "DATA_ROOT", raw_data_root):
    assert w.open_dataset(campus_image.parent)
    w.doc.boxes = [[0, 0.1, 0.2, 0.3, 0.4]]
    w.doc.dirty = True
    w.save()
    expected = raw_data_root / "campus/unused/annotation_work/new_batch/labels/angle1/a.txt"
    assert Path(w.doc.label_path) == expected and expected.is_file()
    assert expected.with_suffix(".review.json").is_file()
    assert not list((raw_data_root / "campus/unused/raw").rglob("*.txt"))
    assert campus_image.read_bytes() == original_image
print("raw 原图与 annotation_work 标签分离保存通过")

# 关闭时自动保存不报错
w.show()
app.processEvents()
assert not w.grab().isNull()

# 放大后中键平移只改视角；左键新建框仍使用正确的原图坐标。
canvas = w.canvas
canvas.zoom = 2.0
old_boxes = [b[:] for b in w.doc.boxes]
old_states = w.doc.states[:]
old_dirty, old_undo = w.doc.dirty, len(w.doc.undo)
before = canvas.to_screen(100, 100)
def mouse(kind, position, button, buttons):
    event = QMouseEvent(kind, QPointF(position), QPointF(position), button, buttons, Qt.NoModifier)
    QApplication.sendEvent(canvas, event)

mouse(QEvent.MouseButtonPress, QPoint(300, 250), Qt.MiddleButton, Qt.MiddleButton)
mouse(QEvent.MouseMove, QPoint(410, 310), Qt.NoButton, Qt.MiddleButton)
mouse(QEvent.MouseButtonRelease, QPoint(420, 315), Qt.MiddleButton, Qt.NoButton)
assert canvas.pan_offset == QPoint(120, 65)
assert canvas.to_screen(100, 100) == before + QPoint(120, 65)
assert canvas.pan_anchor is None and canvas.mode is None
assert w.doc.boxes == old_boxes and w.doc.states == old_states
assert w.doc.dirty == old_dirty and len(w.doc.undo) == old_undo

start, end = QPoint(600, 400), QPoint(675, 460)
a, b = canvas.to_img(start), canvas.to_img(end)
mouse(QEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton)
assert canvas.mode == "create", "测试应从已有框以外开始"
mouse(QEvent.MouseMove, end, Qt.NoButton, Qt.LeftButton)
mouse(QEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton)
expected = [0, a[0]/canvas.pix.width(), a[1]/canvas.pix.height(),
            b[0]/canvas.pix.width(), b[1]/canvas.pix.height()]
assert len(w.doc.boxes) == len(old_boxes) + 1
assert w.doc.state(len(w.doc.boxes) - 1) == "合规"
assert all(abs(x-y) < 1e-8 for x,y in zip(w.doc.boxes[-1], expected))
w.undo()
assert w.doc.boxes == old_boxes
assert [w.doc.state(i) for i in range(len(old_boxes))] == [
    old_states[i] if i < len(old_states) else "合规" for i in range(len(old_boxes))]
w.reset_view_btn.click()
assert canvas.zoom == 1.0 and canvas.pan_offset == QPoint(0, 0)
assert canvas.pan_anchor is None
print("中键平移不修改标注、平移后画框坐标和重置视角检查通过")

# 真实键盘事件检查 1～5 的映射，保存重开和撤销保留正确状态。
from PySide6.QtTest import QTest
from labeler import PARKING_HOTKEYS
w.activateWindow()
canvas.setFocus()
app.processEvents()
w.canvas.sel = 0
w.update_status()
for key, state in PARKING_HOTKEYS.items():
    QTest.keyClick(canvas, getattr(Qt, 'Key_' + key))
    app.processEvents()
    assert w.doc.state(0) == state, (key, state, w.doc.state(0))
    assert w.state_combo.currentText() == state
    w.save()
    assert Doc(w.doc.img_path, w.doc.label_path).state(0) == state
w.undo()
assert w.doc.state(0) == "占道"
w.canvas.sel = -1
snapshot = w.doc.states[:]
QTest.keyClick(canvas, Qt.Key_1)
assert w.doc.states == snapshot
# 旧版“未标注”显式记录不会被默认合规覆盖。
w.doc.states[0] = "未标注"
w.save()
assert Doc(w.doc.img_path, w.doc.label_path).state(0) == "未标注"
print("1～5快捷键、默认合规、状态持久化、撤销和旧状态保留检查通过")

# 接地点点击使用原图坐标，支持缩放/平移，且不移动框或改变停车状态。
w.canvas.sel = 0
w.update_status()
canvas.zoom = 1.7
canvas.pan_offset = QPoint(40, -20)
boxes_before = [b[:] for b in w.doc.boxes]
states_before = w.doc.states[:]
for wheel, original in (("point1", (90, 170)), ("point2", (220, 170))):
    position = canvas.to_screen(*original)
    expected_xy = canvas.to_img(position)
    expected_xy = [expected_xy[0] / canvas.pix.width(), expected_xy[1] / canvas.pix.height()]
    w.pick_contact(wheel)
    assert canvas.contact_target == (0, wheel)
    mouse(QEvent.MouseButtonPress, position, Qt.LeftButton, Qt.LeftButton)
    mouse(QEvent.MouseButtonRelease, position, Qt.LeftButton, Qt.NoButton)
    record = w.doc.contact(0)[wheel]
    assert record["visibility"] == "visible"
    assert all(abs(a-b) < 1e-7 for a,b in zip(record["point"], expected_xy))
    assert canvas.contact_target is None
assert w.doc.boxes == boxes_before and w.doc.states == states_before
w.save()
loaded = Doc(w.doc.img_path, w.doc.label_path)
assert loaded.contact(0) == w.doc.contact(0)
assert Path(w.doc.label_path).with_suffix(".review.json.bak").is_file()
assert all(len(line.split()) == 5 for line in Path(w.doc.label_path).read_text().splitlines())
before = loaded.contact(0)
w.mark_contact("point2", "unobservable")
assert w.doc.contact(0)["point2"] == {"visibility": "unobservable", "point": None, "reviewed": True}
w.undo()
assert w.doc.contact(0) == before
w.canvas.sel = 0
w.mark_contacts_not_applicable()
assert all(item["visibility"] == "not_applicable" for item in w.doc.contact(0).values())
w.undo()
assert w.doc.contact(0) == before
w.canvas.sel = 0
w.pick_contact("point1")
w.cancel_contact()
assert canvas.contact_target is None
# 图片外点击不记录，也不新建/移动框。
w.pick_contact("point1")
mouse(QEvent.MouseButtonPress, canvas.to_screen(-50, -50), Qt.LeftButton, Qt.LeftButton)
assert w.doc.contact(0) == before and canvas.contact_target == (0, "point1")
w.cancel_contact()
w.delete_sel()
assert not w.doc.boxes and not w.doc.contacts
w.undo()
assert w.doc.contact(0) == before
w.save()
w.next()
assert canvas.contact_target is None
print("接地点点击、缩放平移坐标、可见性、保存重开、备份、删除撤销与取消检查通过")

# 连续流程：直接点两个无序接地点，Enter 保存并跳车，跳过非自行车类别。
assert w.open_dataset(ROOT)
w.doc.boxes = [[0, .05, .10, .25, .40], [1, 0, 0, 1, 1],
               [0, .35, .15, .55, .45], [0, .65, .20, .85, .50]]
w.doc.conf = [-1.0] * 4
w.doc.states = ["合规"] * 4
w.doc.contacts = []
w.doc.undo = []
w.doc.dirty = True
w.canvas.sel = -1
original_boxes = [box[:] for box in w.doc.boxes]
original_states = w.doc.states[:]
w.contact_mode_box.setChecked(True)
QTest.qWait(180)
assert canvas.sel == 0 and canvas.contact_target == (0, "point1")
for wheel, xy in (("point1", (.10, .30)), ("point2", (.20, .30))):
    position = canvas.to_screen(xy[0]*canvas.pix.width(), xy[1]*canvas.pix.height())
    mouse(QEvent.MouseButtonPress, position, Qt.LeftButton, Qt.LeftButton)
    mouse(QEvent.MouseButtonRelease, position, Qt.LeftButton, Qt.NoButton)
    assert w.doc.contact(0)[wheel]["visibility"] == "visible"
    assert canvas.contact_target == ((0, "point2") if wheel == "point1" else None)
assert w.doc.boxes == original_boxes and w.doc.states == original_states
before_offset, before_zoom, image_index = QPoint(canvas.pan_offset), canvas.zoom, w.i
canvas.setFocus()
QTest.keyClick(canvas, Qt.Key_Return)
QTest.qWait(180)
assert canvas.sel == 2 and canvas.contact_target == (2, "point1")
assert not w.doc.dirty and Doc(w.doc.img_path, w.doc.label_path).contact(0) == w.doc.contact(0)
assert canvas.pan_offset != before_offset and canvas.zoom == before_zoom
assert w.doc.boxes == original_boxes and w.doc.states == original_states
w.mark_contact("point1", "unobservable")
assert canvas.contact_target == (2, "point2")
position = canvas.to_screen(.5*canvas.pix.width(), .4*canvas.pix.height())
mouse(QEvent.MouseButtonPress, position, Qt.LeftButton, Qt.LeftButton)
mouse(QEvent.MouseButtonRelease, position, Qt.LeftButton, Qt.NoButton)
assert canvas.contact_target is None
QTest.keyClick(canvas, Qt.Key_Enter)
QTest.qWait(180)
assert canvas.sel == 3 and canvas.contact_target == (3, "point1")
QTest.keyClick(canvas, Qt.Key_Return)
assert canvas.sel == 3 and w.i == image_index, "最后一辆不循环覆盖，也不自动翻图"
# 保存失败不推进；跳过的未标点仍保持未标，不自动猜测坐标。
w.doc.dirty = True
with patch.object(w.doc, "save", side_effect=OSError("test write failure")), patch("labeler.QMessageBox.critical") as error:
    w.next_vehicle()
    assert error.called and canvas.sel == 3
assert all(item["visibility"] == "unobservable" and item["reviewed"] for item in w.doc.contact(3).values())
# 重开本图自动选中首辆未完成的车，并从缺失的后轮继续。
w.doc.set_contact(3, "point1", "visible", [.7, .4])
w.doc.set_contact(3, "point2", "unmarked")
w.save()
w.load_index(w.i)
QTest.qWait(180)
assert canvas.sel == 3 and canvas.contact_target == (3, "point2")
# 点击可中止平移动画，随后坐标按停止后的视角换算。
canvas.follow_vehicle(0)
position = canvas.to_screen(.8*canvas.pix.width(), .4*canvas.pix.height())
mouse(QEvent.MouseButtonPress, position, Qt.LeftButton, Qt.LeftButton)
mouse(QEvent.MouseButtonRelease, position, Qt.LeftButton, Qt.NoButton)
assert canvas.follow_animation.state() == QAbstractAnimation.Stopped
assert w.doc.contact(3)["point2"]["visibility"] == "visible"
w.undo()
assert canvas.sel == 3 and canvas.contact_target == (3, "point2")
assert w.doc.contact(3)["point2"]["visibility"] == "unobservable"
assert not w.doc.contact(3)["point2"]["reviewed"]
w.contact_mode_box.setChecked(False)
assert canvas.contact_target is None
print("连续无序两点、Enter跳车保存、轻微平移、末车边界、断点续标和保存失败保护通过")

# 两点默认看不清，但尚未确认；Enter 接受默认值才计入完成。
from labeler import empty_contacts, CONTACT_NAMES
assert list(CONTACT_NAMES) == ["point1", "point2"]
assert all(item == {"visibility": "unobservable", "point": None, "reviewed": False}
           for item in empty_contacts().values())
w.doc.contacts = [empty_contacts() for _ in w.doc.boxes]
w.canvas.sel = 0
w.contact_mode_box.setChecked(True)
w.arm_next_contact()
w.doc.dirty = True
w.next_vehicle()
assert all(item["visibility"] == "unobservable" and item["point"] is None and item["reviewed"]
           for item in w.doc.contact(0).values())
assert not any(item["reviewed"] for item in w.doc.contact(2).values())
assert canvas.sel == 2 and canvas.contact_target == (2, "point1")
w.doc.set_contact(2, "point1", "visible", [.45, .4])
w.next_vehicle()
assert w.doc.contact(2)["point1"]["point"] == [.45, .4]
assert w.doc.contact(2)["point2"] == {"visibility": "unobservable", "point": None, "reviewed": True}
assert "前轮" not in w.contact_progress.text() and "后轮" not in w.contact_progress.text()

# 已完成的旧版 front/rear 精确转成无序两点，不丢坐标或人工状态。
import json
legacy_label = Path(ROOT) / "legacy_contacts.txt"
legacy_label.write_text("0 .5 .5 .4 .4\n", encoding="utf-8")
legacy_review = legacy_label.with_suffix(".review.json")
legacy_review.write_text(json.dumps({"version": 2, "objects": [{
    "box": [0, .5, .5, .4, .4], "parking_state": "出界",
    "contact_points": {"front": {"visibility": "visible", "point": [.12345678, .87654321]},
                       "rear": {"visibility": "unobservable", "point": None}}
}]}), encoding="utf-8")
legacy_doc = Doc(w.doc.img_path, str(legacy_label))
assert legacy_doc.state(0) == "出界"
assert legacy_doc.contact(0)["point1"]["point"] == [.12345678, .87654321]
assert all(item["reviewed"] for item in legacy_doc.contact(0).values())
legacy_doc.save()
neutral = json.loads(legacy_review.read_text(encoding="utf-8"))
assert neutral["version"] == 3 and neutral["contact_order"] == "unordered"
assert set(neutral["objects"][0]["contact_points"]) == {"point1", "point2"}
assert Doc(w.doc.img_path, str(legacy_label)).contact(0) == legacy_doc.contact(0)
assert legacy_review.with_suffix(".json.bak").is_file()
print("默认看不清、Enter确认、单点可见、无前后顺序与旧标注无损兼容检查通过")
w.close()
print("\n全部检查通过")
fixture.cleanup()
