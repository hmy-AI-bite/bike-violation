@echo off
setlocal
cd /d "%~dp0"
set "PY=C:\Users\26803\AppData\Local\Programs\Python\Python313\pythonw.exe"
if not exist "%PY%" goto NOPY
set "SRC=J:\project-1-at-2026-11-13-17-44-a1d4a70f\images"
if not exist "%SRC%" set "SRC=D:\bick\dataset_school\images\train"
start "" "%PY%" "D:\bick\tools\zone_tool.py" "%SRC%" "D:\bick\dataset_school\scenes.json" "school_row"
exit /b 0
:NOPY
echo [ERROR] python not found: %PY%
pause