@echo off
cd /d "%~dp0"
set "PY=C:\Users\26803\miniconda3\envs\sam2\python.exe"
set "PYTHONPATH=D:\bick\yolo-lib"
set "YOLO_CONFIG_DIR=D:\bick\relabel\cfg"
set "MPLCONFIGDIR=D:\bick\relabel\cfg\mpl"
echo ==============================================================
echo  Real training on your GPU: 20 epochs, about 40 seconds.
echo  Watch: box_loss goes down, mAP50 goes up, GPU usage in Task Manager
echo ==============================================================
echo.
"%PY%" "D:\bick\relabel\train.py" --data "D:\bick\dataset_school\data.yaml" --weights "D:\bick\relabel\weights\yolo11s.pt" --imgsz 1280 --epochs 20 --batch 8 --lr0 0.002 --workers 0 --amp --name demo_visible
echo.
echo ==============================================================
echo  Done. Check runs\demo_visible\val_batch0_pred.jpg and results.png
echo ==============================================================
pause