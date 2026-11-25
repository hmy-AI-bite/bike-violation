"""Reproducible YOLO11x training; incoming photos are test-only."""
from pathlib import Path
import argparse
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'yolo-lib'), str(ROOT / 'relabel')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--name', default='bike11x_v1')
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args()
    import torch
    from ultralytics import YOLO
    from train import patch_no_multiprocessing, check_finite_metrics
    patch_no_multiprocessing()
    model = YOLO(str(ROOT / 'relabel/weights/yolo11x.pt'))

    def check_parameters(trainer):
        check_finite_metrics(trainer)
        finite = torch.stack([torch.isfinite(p).all() for p in trainer.model.parameters()]).all()
        if not bool(finite.item()):
            raise RuntimeError('Non-finite model weights; aborting before checkpoint save.')

    model.add_callback('on_train_epoch_end', check_parameters)
    model.add_callback('on_fit_epoch_end', check_finite_metrics)
    model.train(
        data=str(ROOT / 'dataset_v2/data.yaml'),
        project=str(ROOT / 'runs'), name=args.name, exist_ok=False,
        epochs=args.epochs, patience=12, imgsz=args.imgsz, batch=args.batch, device=0,
        workers=0, cache=False, amp=True, optimizer='AdamW', lr0=0.0001,
        lrf=0.1, warmup_epochs=3, cos_lr=True, freeze=10,
        mosaic=0.2, close_mosaic=10, scale=0.25,
        seed=0, deterministic=True, plots=True,
    )
    out = Path(model.trainer.save_dir)
    report = {
        'initial_weights': str(ROOT / 'relabel/weights/yolo11x.pt'),
        'training_dataset': str(ROOT / 'dataset_v2/data.yaml'),
        'incoming_test_data_used_for_training': False,
        'gpu': torch.cuda.get_device_name(0),
        'imgsz': args.imgsz,
        'batch': args.batch,
        'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
        'peak_gpu_reserved_mib': torch.cuda.max_memory_reserved()/1024**2,
        'best_weights': str(out / 'weights/best.pt'),
        'metrics': {k: float(v) for k,v in model.trainer.metrics.items()},
    }
    if not all(math.isfinite(v) for v in report['metrics'].values()):
        raise RuntimeError('Final metrics are not finite; run is not valid.')
    (out / 'run_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
