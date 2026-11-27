@echo off
setlocal
cd /d "%~dp0"
if defined BICYCLE_PYTHON (
    "%BICYCLE_PYTHON%" "%~dp0config.py" run labeler "%~dp0data\campus\used\annotation_work\ground_truth_v1" --contacts
) else (
    python "%~dp0config.py" run labeler "%~dp0data\campus\used\annotation_work\ground_truth_v1" --contacts
)
if errorlevel 1 pause
