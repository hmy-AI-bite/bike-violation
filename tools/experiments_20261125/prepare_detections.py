"""Generate identical-parameter predictions for the three fixed test images."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'yolo-lib'), str(ROOT/'relabel')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT/'incoming/单车违停检测/原始图片')
    parser.add_argument('--finetuned', type=Path, default=ROOT/'runs/bike11x_v1/weights/best.pt')
    args = parser.parse_args()
    from ultralytics import YOLO
    from violation_rules import bicycle_class_id
    names = ['A_1_038.jpg', 'A_2_007.jpg', 'A_3_008.jpg']
    sources = {}
    for name in names:
        matches = list(args.source.rglob(name))
        if len(matches) != 1:
            parser.error(f'Expected exactly one {name} under {args.source}; found {len(matches)}')
        sources[name] = matches[0]
    weights = {'finetuned_x': args.finetuned, 'pretrained': ROOT/'relabel/weights/yolo11x.pt'}
    for path in weights.values():
        if not path.is_file():
            parser.error(f'Missing local weights: {path}')
    records = []
    for tag, path in weights.items():
        model = YOLO(str(path))
        for name, source in sources.items():
            result = model.predict(str(source), imgsz=1280, conf=.35, iou=.45,
                classes=[bicycle_class_id(model.names)], device=0, max_det=200, verbose=False)[0]
            boxes = [{'xyxy': box, 'confidence': confidence} for box, confidence in
                     zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist())]
            records.append(dict(model=tag, image=name, source=str(source), count=len(boxes), boxes=boxes))
            print(tag, name, len(boxes), flush=True)
        del model
    out = ROOT/'runs/incoming_20261125/comparison_x_v1'
    out.mkdir(parents=True, exist_ok=True)
    (out/'detections.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
