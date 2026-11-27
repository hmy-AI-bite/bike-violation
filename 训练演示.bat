@echo off
setlocal
cd /d "%~dp0"
if defined BICYCLE_PYTHON (
  "%BICYCLE_PYTHON%" "%~dp0config.py" run train-demo %*
) else (
  python "%~dp0config.py" run train-demo %*
)
if errorlevel 1 (
  echo Launch failed. Check Python and .env settings above.
  pause
  exit /b 1
)
pause
