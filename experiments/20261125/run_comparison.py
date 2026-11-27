"""Compare cached identical-parameter YOLO predictions through SAM2 and rules.

Test images and weights are never modified. Manual zones are provisional.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.py").is_file())))
import config
config.activate()

import csv
import json
import time

ROOT = config.ROOT
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics.utils.patches import imread
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from violation_rules import classify_mask, count_verdicts, contact_band

OUT = Path(config.output_path('runs/incoming_20261125/comparison_x_v1'))
BASE = OUT.parent
CONFIGS = {
    'A_1_038.jpg': {
        'allow': [[382,188],[542,188],[685,1049],[282,1060]],
        'deny': [[[604,183],[858,173],[900,220],[900,517],[693,521]]],
    },
    'A_2_007.jpg': {
        'allow': [[28,942],[245,983],[869,463],[768,447]],
        'deny': [[[686,691],[900,492],[900,716]]],
    },
    'A_3_008.jpg': {
        'allow': [[39,317],[240,303],[852,979],[505,1068]],
        'deny': [[[230,295],[457,278],[843,511],[531,564]]],
        'roi': [[0,270],[300,270],[900,970],[900,1200],[0,1200]],
        'scope': 'Main foreground left parking row only; right row and distant background excluded.',
    },
}
COLORS = {'合规': (0,210,90), '压线': (255,190,0),
          '出界': (255,45,45), '占道': (255,45,45),
          '无法判定': (160,160,160), '范围外未评估': (160,160,160)}
LABELS = {'finetuned_x': '我们的微调大模型 YOLO11x',
          'pretrained': '官方预训练大模型 YOLO11x'}
FONT = config.load_font(23)
SMALL = config.load_font(18)


def main():
    detections = json.loads((OUT / 'detections.json').read_text(encoding='utf-8'))
    (OUT / 'test_zones.json').write_text(json.dumps({
        'reference_size': [900,1200], 'manual_provisional_zones': True,
        'purpose': 'Test inference only, not independent ground truth or training labels.',
        'images': CONFIGS}, ensure_ascii=False, indent=2), encoding='utf-8')
    sam = build_sam2(config.SAM2_CONFIG,
                    str(config.SAM2_CHECKPOINT), device='cuda')
    predictor = SAM2ImagePredictor(sam)
    summary, all_rows = [], []
    for name, cfg in CONFIGS.items():
        first = next(r for r in detections if r['image'] == name)
        bgr = imread(config.resolve_project_path(first['source']))
        h0, w0 = bgr.shape[:2]
        original = Image.fromarray(bgr[:, :, ::-1])
        original.thumbnail((1400,1900))
        rgb = np.array(original)
        h, w = rgb.shape[:2]
        predictor.set_image(rgb)

        def pts(poly):
            return [(x*w/900, y*h/1200) for x,y in poly]

        def raster(polys):
            im = Image.new('1', (w,h), 0)
            draw = ImageDraw.Draw(im)
            for poly in polys:
                draw.polygon(pts(poly), fill=1)
            return np.array(im, dtype=bool)

        allow, deny = raster([cfg['allow']]), raster(cfg['deny'])
        roi = raster([cfg['roi']]) if 'roi' in cfg else np.ones((h,w), dtype=bool)
        for tag in LABELS:
            rec = next(r for r in detections if r['image'] == name and r['model'] == tag)
            dest = OUT / tag
            dest.mkdir(exist_ok=True)
            canvas = rgb.astype(float)
            rows, masks, scaled = [], [], []
            started = time.time()
            for idx, box in enumerate(rec['boxes'], 1):
                b = np.array(box['xyxy']) * np.array([w/w0,h/h0,w/w0,h/h0])
                mm, ss, _ = predictor.predict(box=b, multimask_output=False)
                mask = np.asarray(mm).reshape(-1,h,w)[0] > 0
                score = float(np.asarray(ss).ravel()[0])
                result = classify_mask(mask, allow, deny, has_zone=True, score=score,
                                       min_area=200, min_score=0, contact_ratio=.02, deny_tol=.05)
                cx = min(w-1,max(0,int((b[0]+b[2])/2)))
                cy = min(h-1,max(0,int((b[1]+b[3])/2)))
                if not roi[cy,cx]:
                    result = dict(verdict='范围外未评估', contact_in=-1., deny=-1., reason='outside_main_row_roi')
                row = dict(image=name, model=tag, id=idx, detector_confidence=box['confidence'],
                           sam_quality=score, mask_area=int(mask.sum()), **result)
                rows.append(row)
                all_rows.append(row)
                masks.append(mask)
                scaled.append(b)
                color = np.array(COLORS[result['verdict']])
                canvas[mask] = canvas[mask]*.77 + color*.23
            evaluated = [r for r in rows if r['verdict'] != '范围外未评估']
            counts = count_verdicts(evaluated)
            excluded = len(rows)-len(evaluated)
            image = Image.fromarray(np.uint8(np.clip(canvas,0,255)))
            draw = ImageDraw.Draw(image)
            draw.line(pts(cfg['allow'])+[pts(cfg['allow'])[0]], fill=(60,150,255), width=5)
            for poly in cfg['deny']:
                draw.line(pts(poly)+[pts(poly)[0]], fill=(255,140,0), width=4)
            for b,row in zip(scaled,rows):
                color = COLORS[row['verdict']]
                draw.rectangle(tuple(b), outline=color, width=3)
                draw.text((b[0],max(80,b[1])), f"#{row['id']} {row['verdict']}",
                          font=SMALL, fill=color, stroke_width=1, stroke_fill='black')
            draw.rectangle((0,0,w,78), fill='black')
            draw.text((8,4), LABELS[tag], font=FONT, fill='white')
            text = f"候选{len(rows)} 合规{counts['合规']} 压线{counts['压线']} 出界{counts['出界']} 占道{counts['占道']} 未评估{excluded}"
            draw.text((8,41), text, font=FONT, fill='white')
            image.thumbnail((1200,1600))
            image.save(dest / name, quality=92)
            if masks:
                np.savez_compressed(dest/(Path(name).stem+'_masks.npz'), masks=np.stack(masks))
            alarms = [i for i,r in enumerate(rows) if r['verdict'] in ('压线','出界','占道')]
            for idx in alarms:
                mask=masks[idx]
                band=contact_band(mask,.02)
                arr=rgb.copy()
                arr[mask]=(arr[mask]*.55+np.array([0,200,100])*.45).astype('uint8')
                arr[band]=[255,0,0]
                b=scaled[idx]
                crop=(max(0,int(b[0])-60),max(0,int(b[1])-60),min(w,int(b[2])+60),min(h,int(b[3])+60))
                sheet=Image.new('RGB',(1200,650),'#eeeeee')
                sd=ImageDraw.Draw(sheet)
                for col,img in enumerate([Image.fromarray(rgb),Image.fromarray(arr)]):
                    panel=ImageOps.contain(img.crop(crop),(590,590))
                    sheet.paste(panel,(col*600+(600-panel.width)//2,50))
                sd.text((8,8),f"{name} #{idx+1} {rows[idx]['verdict']} | 原图 / 掩膜与最低条带",font=SMALL,fill='black')
                sheet.save(dest/(Path(name).stem+f'_alarm_{idx+1}.jpg'),quality=93)
            summary.append(dict(image=name,model=tag,candidates=len(rows),evaluated=len(evaluated),excluded=excluded,counts=counts))
            print(name,tag,text,'SAM seconds',round(time.time()-started,1),flush=True)
        comparison=Image.new('RGB',(1800,1240),'white')
        for col,tag in enumerate(LABELS):
            with Image.open(OUT/tag/name) as im:
                thumb=ImageOps.contain(im,(900,1200))
            comparison.paste(thumb,(col*900+(900-thumb.width)//2,0))
        cd=ImageDraw.Draw(comparison)
        cd.text((10,1207),'人工画测试区域；规则报警待核实，候选框数不等于真实车辆数。',font=FONT,fill='black')
        comparison.save(OUT/(Path(name).stem+'_comparison.jpg'),quality=93)
    (OUT/'results.json').write_text(json.dumps(dict(
        detection_input='Predictions prepared by prepare_detections.py with identical inference parameters.',
        detection_parameters=dict(imgsz=1280,conf=.35,iou=.45,max_det=200),
        sam_checkpoint='sam2.1_hiera_tiny',mask_input_limit=[1400,1900],
        manual_zones=True,no_training=True,independent_ground_truth=False,
        summary=summary,per_bike=all_rows),ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'per_bike.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    overview=Image.new('RGB',(1200,3*820),'white')
    for row,name in enumerate(CONFIGS):
        with Image.open(OUT/(Path(name).stem+'_comparison.jpg')) as im:
            thumb=ImageOps.contain(im,(1200,820))
        overview.paste(thumb,(0,row*820))
    overview.save(OUT/'overview.jpg',quality=92)


if __name__ == '__main__':
    main()
