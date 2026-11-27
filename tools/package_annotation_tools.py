"""Build the standalone source distribution; no photos, models or local settings."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = "2026.11.27"


def build():
    output = ROOT / "runs/labeler_share"
    package = output / "自行车标注工具_整合版"
    (package / "tools").mkdir(parents=True, exist_ok=True)
    files = ["config.py", "tools/annotation_launcher.py", "tools/labeler.py", "tools/zone_tool.py",
             "tools/test_annotation_launcher.py", "tools/test_labeler.py", "tools/test_zone_tool.py"]
    for name in files:
        shutil.copy2(ROOT / name, package / name)
    contents = {
        "requirements.txt": "PySide6==6.12.0\n",
        "setup.bat": '''@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\\Scripts\\python.exe" goto install
where py >nul 2>nul
if errorlevel 1 (
  python -m venv .venv
) else (
  py -3 -m venv .venv
)
if errorlevel 1 goto failed
:install
".venv\\Scripts\\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Setup complete. Double-click start.bat.
pause
exit /b 0
:failed
echo Setup failed. Install 64-bit Python 3.11-3.13 with pip, then retry.
echo Internet access is needed for the first setup.
pause
exit /b 1
''',
        "使用说明.txt": '''自行车标注工具 整合版 2026.11.27

首次安装（Windows）
1. 安装 64 位 Python 3.11～3.13，安装时勾选 pip 和 Add Python to PATH。
2. 将整个压缩包解压，不要在压缩包内部运行。
3. 双击 setup.bat，首次需联网安装 PySide6。
4. 双击 start.bat，进入统一任务窗口。以后不需要重复安装。
此包是 Python 源码运行版，不是免安装 EXE。无需显卡、CUDA 或模型权重。

统一入口中的三个任务
自行车框与停车状态：选择自己的图片目录，一辆车一个框，逐车检查停车状态。
车轮接地点 连续标注：在同一车辆窗口开启连续点选，不区分前后轮。
允许停车区与禁停区：在区域工具中选择图片目录，画允许区和禁停通道。
车辆任务与接地点任务复用同一个窗口；直接关闭任务窗口会自动保存，之后可从统一入口重新打开。
目录、图片数量和输出文件位置请以各工具顶部显示为准。

通用视角操作
滚轮缩放；按住鼠标中键拖动画面；R 或“重置视角”恢复默认缩放和居中。
平移只改变显示位置，不修改框、接地点或区域坐标。
Ctrl+O 选择目录；Ctrl+S 保存；A/D 或左右键翻图。

自行车框与停车状态
空白处按左键拖动画框；点击框选中，拖动移动，拖角点调整大小。
选中框后：1 合规，2 压线，3 出界，4 占道，5 无法判定。
新建框、只有检测框而没有状态记录的标签默认合规，仍须人工检查。
已有人工状态保留；外部改过框导致状态记录无法匹配时，保留未标注待复核。
Delete 删除选中框，Ctrl+Z 撤销，空格隐藏或显示框。
连续点选模式下会保护车辆框；取消该勾选后再编辑框。

车轮接地点
两个点不分前后轮，能看清哪个就点哪个；默认看不清，不需要猜。
两个都清楚：点两下；只看清一个：点一下；都看不清：直接 Enter。
Enter 确认、保存并选择下一辆；最后一辆停留本图，A/D 翻图。
倒车或轮胎离地使用“不适用”按钮。Esc 取消当前点选，Ctrl+Z 撤销。
停车状态与接地点可见性分别记录，接地点看不清不会自动修改停车状态。

允许区与禁停区
默认允许区：1 允许区（蓝），2 禁停区（橙），Tab 也可切换。
左键逐个放顶点，Enter 闭合（至少 3 点）；Backspace/Ctrl+Z 撤销最后一个顶点。
Esc 取消未闭合图形。切换类型会取消未闭合图形，重复按当前类型数字键保留顶点。
允许区闭合后替换本图原允许区；禁停区可画多块。C 清空当前类型全部区域。
先 Enter 闭合，再 Ctrl+S/S 保存。未闭合顶点不会写入区域文件。
各图单独存区域；多个子目录有同名图片时，请分别选择子目录，避免覆盖。

保存与交付
.txt 保存自行车框，同名 .review.json 保存停车状态和接地点，必须成对交付。
scenes.json 保存按图标定的区域，需要和图片名、批次一起交付。
普通图片目录：通常写入 labels/ 和 scenes.json；标准 YOLO 数据集使用对应 labels/。
在项目的 data/ 来源目录中使用时，请检查顶部实际输出路径；项目配置可能分离保存到 annotation_work/。
原图片不修改。不要擅自重命名图片或改动子目录结构。
多人分配不同图片，完成后交回完整标签和区域文件，不要相互覆盖。
没有备注输入框的事项另记图片相对路径、最终框序号及原因。
关闭旧版窗口后再用新版打开同一批数据；更新前保留原有标签备份。

工具仅用于人工标注，不自动检测车辆，也不自动判断违停。
团队应先统一边界、压线与占道标准，再批量标注。
包内不含原始照片、开发者 .env、本机环境或模型文件。
''',
    }
    for name, script in (("start.bat", "annotation_launcher.py"),
                         ("start_bikes.bat", "labeler.py"),
                         ("start_zones.bat", "zone_tool.py")):
        contents[name] = '''@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\\Scripts\\python.exe" (
  echo Please run setup.bat first.
  pause
  exit /b 1
)
".venv\\Scripts\\python.exe" tools\\SCRIPT %*
if errorlevel 1 pause
'''.replace("SCRIPT", script)
    for name, content in contents.items():
        if name.endswith(".bat"):
            (package / name).write_bytes(content.replace("\n", "\r\n").encode("ascii"))
        else:
            (package / name).write_text(content, encoding="utf-8-sig" if name.endswith(".txt") and name != "requirements.txt" else "utf-8")
    files += list(contents)
    manifest = {"version": VERSION, "files": {
        name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in files}}
    (package / "VERSION.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    files.append("VERSION.json")
    archive = output / "自行车标注工具_整合版_20261127.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for name in files:
            z.write(package / name, Path(package.name) / name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(files)
        assert not any(".env" in name or "__pycache__" in name for name in z.namelist())
    print(archive)
    print("Bytes:", archive.stat().st_size)
    return archive


if __name__ == "__main__":
    build()
