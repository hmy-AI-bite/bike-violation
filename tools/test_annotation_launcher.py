"""验证整合入口复用窗口，并保留已有框、状态、接地点和区域。"""
from pathlib import Path
import os
import sys
import tempfile

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from annotation_launcher import Launcher, config
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage

app = QApplication(sys.argv)
with tempfile.TemporaryDirectory(dir=config.ROOT / "tools") as folder:
    root = Path(folder)
    image = QImage(640, 480, QImage.Format_RGB32)
    image.fill(0xFFFFFFFF)
    assert image.save(str(root / "sample.jpg"))
    launcher = Launcher()
    launcher.show()
    launcher.open_bicycles()
    bikes = launcher.bicycles
    assert not bikes.contact_mode_box.isChecked()
    assert bikes.open_dataset(root)
    bikes.doc.boxes = [[0, .2, .2, .6, .6]]
    bikes.doc.states = ["出界"]
    bikes.doc.conf = [-1]
    bikes.doc.set_contact(0, "point1", "visible", [.25, .55])
    bikes.save()
    launcher.open_bicycles(True)
    assert launcher.bicycles is bikes and bikes.contact_mode_box.isChecked()
    assert bikes.doc.state(0) == "出界"
    assert bikes.doc.contact(0)["point1"]["point"] == [.25, .55]
    bikes.close()
    launcher.open_bicycles(False)
    assert launcher.bicycles is bikes and not bikes.contact_mode_box.isChecked()
    launcher.open_zones()
    zones = launcher.zones
    assert zones.open_dataset(root)
    zones.canvas.pending = [(.1, .1), (.9, .1), (.9, .9)]
    zones.close_poly()
    zones.close()
    launcher.open_zones()
    assert launcher.zones is zones and len(zones.doc()["allow"]) == 3
    app.processEvents()
    assert not launcher.grab().isNull()
    zones.close()
    bikes.close()
    launcher.close()
print("整合入口、窗口复用、模式切换和标注保留检查通过")
