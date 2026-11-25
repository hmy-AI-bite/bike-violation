@echo off
setlocal
cd /d "%~dp0"
set "PY=C:\Users\26803\miniconda3\envs\sam2\python.exe"
set "PYTHONPATH=%~dp0yolo-lib"
set "YOLO_CONFIG_DIR=%~dp0relabel\cfg"
set "MPLCONFIGDIR=%~dp0relabel\cfg\mpl"
if not exist "%PY%" goto NOPY
echo Running bicycle detection and parking rules on the existing school photos...
echo Please wait. Results will open in File Explorer when finished.
"%PY%" "%~dp0relabel\violation_mask.py" --source "%~dp0dataset_school\images\train" --scenes "%~dp0dataset_school\scenes.json" --scene school_row --weights "%~dp0runs\school_v3\weights\best.pt" --out "%~dp0runs\parking_detection"
if errorlevel 1 goto FAILED
start "" explorer.exe "%~dp0runs\parking_detection"
echo Done. Open the output JPG files to view results.
pause
exit /b 0
:NOPY
echo Python environment not found: %PY%
pause
exit /b 1
:FAILED
echo Detection failed. Keep this window open to inspect the error above.
pause
exit /b 1
