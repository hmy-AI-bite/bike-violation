"""在复核后的数据集上微调 YOLO。

用法（先确认 dataset_v2/data.yaml 里的图片和标注已经审阅过）:
    python train.py --weights relabel/weights/yolo11s.pt --imgsz 960 --epochs 120 --batch 16
"""
import argparse
import os
import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "yolo-lib"))


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


def check_finite_metrics(trainer):
    """明确报告 NaN/Inf；不能把验证失败当作正常训练完成。"""
    values = dict(getattr(trainer, "metrics", {}) or {})
    losses = getattr(trainer, "tloss", None)
    if losses is not None:
        values.update(trainer.label_loss_items(losses, prefix="train"))
    bad = [k for k, v in values.items() if not math.isfinite(float(v))]
    if bad:
        raise RuntimeError("第 %d 轮出现 NaN/Inf: %s；请检查数据和训练参数，勿使用异常轮权重" %
                           (trainer.epoch + 1, ", ".join(bad)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=str(ROOT / "yolo-lib"))
    ap.add_argument("--data", default=str(ROOT / "dataset_v2/data.yaml"))
    ap.add_argument("--weights", default=str(ROOT / "relabel/weights/yolo11s.pt"))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--cache", default="disk", help="ram / disk / False")
    ap.add_argument("--patience", type=int, default=35)
    ap.add_argument("--project", default=str(ROOT / "runs"))
    ap.add_argument("--name", default="bike_v2")
    ap.add_argument("--lr0", type=float, default=0.002, help="初始学习率；默认 0.002，需结合训练曲线调整")
    ap.add_argument("--freeze", type=int, default=None, help="冻结前 N 层")
    ap.add_argument("--mosaic", type=float, default=1.0)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                    help="默认启用混合精度；--no-amp 关闭")
    args = ap.parse_args()

    sys.path.insert(0, args.lib)
    from ultralytics import YOLO

    patch_no_multiprocessing()
    model = YOLO(args.weights)
    model.add_callback("on_train_epoch_end", check_finite_metrics)
    model.add_callback("on_fit_epoch_end", check_finite_metrics)
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
