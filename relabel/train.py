"""在复核后的数据集上微调 YOLO。

用法（先确认 dataset_v2/data.yaml 里的图片和标注已经审阅过）:
    python train.py --weights relabel/weights/yolo11s.pt --imgsz 960 --epochs 120 --batch 16
"""
import argparse
import os
import sys

sys.path.insert(0, r"D:\bick\yolo-lib")


def patch_no_multiprocessing():
    """受限沙箱下 multiprocessing 的 Pipe 会被拒绝（WinError 5），
    把 ultralytics 里所有线程池替换成顺序执行。"""
    class _SeqPool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def imap(self, func, iterable, chunksize=1):
            return map(func, iterable)

        def starmap(self, func, iterable, chunksize=1):
            return [func(*args) for args in iterable]

        def map(self, func, iterable, chunksize=1):
            return list(map(func, iterable))

        def close(self):
            pass

        def join(self):
            pass

    import sys as _sys
    for name, mod in list(_sys.modules.items()):
        if name.startswith("ultralytics") and hasattr(mod, "ThreadPool"):
            setattr(mod, "ThreadPool", _SeqPool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=r"D:\bick\yolo-lib")
    ap.add_argument("--data", default=r"D:\bick\dataset_v2\data.yaml")
    ap.add_argument("--weights", default=r"D:\bick\relabel\weights\yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache", default="disk", help="ram / disk / False")
    ap.add_argument("--patience", type=int, default=35)
    ap.add_argument("--project", default=r"D:\bick\runs")
    ap.add_argument("--name", default="bike_v2")
    ap.add_argument("--lr0", type=float, default=0.01, help="初始学习率；小数据集建议 0.002")
    ap.add_argument("--freeze", type=int, default=None, help="冻结前 N 层")
    ap.add_argument("--mosaic", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true", default=None, help="默认关闭（小数据集更稳）")
    args = ap.parse_args()

    sys.path.insert(0, args.lib)
    from ultralytics import YOLO

    patch_no_multiprocessing()
    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        cache=False if str(args.cache).lower() == "false" else args.cache,
        cos_lr=True,
        close_mosaic=15,
        lr0=args.lr0,
        lrf=0.01,
        warmup_epochs=3,
        freeze=args.freeze,
        mosaic=args.mosaic,
        amp=bool(args.amp),
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
