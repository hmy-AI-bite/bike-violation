@echo off
setlocal
cd /d "%~dp0"
set "PY=C:\Users\26803\AppData\Local\Programs\Python\Python313\pythonw.exe"
if not exist "%PY%" goto NOPY
start "" "%PY%" "D:\bick\tools\labeler.py" "D:\bick\dataset_v2"
exit /b 0
:NOPY
echo [ERROR] python not found: %PY%
pause