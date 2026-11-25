"""Measure a resident YOLO11x + SAM2 pipeline and inspect boundary crops."""
from pathlib import Path
import gc
import importlib.util
import json
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'yolo-lib'),str(ROOT/'relabel'),r'D:\CV-Datatext-Pipeline\sam2-repo']
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO
from ultralytics.utils.patches import imread
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from violation_rules import bicycle_class_id, classify_mask, contact_band

HERE = ROOT / 'runs/incoming_20261125/comparison_x_v1'
OUT = HERE/'sam_memory_trial'
OUT.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('comparison_config',Path(__file__).resolve().with_name('run_comparison.py'))
config=importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
records=json.loads((HERE/'detections.json').read_text(encoding='utf-8'))
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)


def main():
    summaries=[]
    for tag,weights in [('finetuned_x',ROOT/'runs/bike11x_v1/weights/best.pt'),
                        ('pretrained',ROOT/'relabel/weights/yolo11x.pt')]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        samples=[]
        stop=threading.Event()

        def monitor():
            while not stop.is_set():
                try:
                    s=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],
                        text=True,timeout=5,creationflags=subprocess.CREATE_NO_WINDOW)
                    samples.append(float(s.strip().splitlines()[0]))
                except Exception:
                    pass
                stop.wait(.3)

        thread=threading.Thread(target=monitor,daemon=True)
        thread.start()
        yolo=YOLO(str(weights))
        sam=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',
            r'D:\CV-Datatext-Pipeline\models\sam2\sam2.1_hiera_tiny.pt',device='cuda')
        predictor=SAM2ImagePredictor(sam)
        rows=[]
        try:
            for name,cfg in config.CONFIGS.items():
                src=next(r['source'] for r in records if r['image']==name)
                rgb0=imread(src)[:,:,::-1].copy()
                h0,w0=rgb0.shape[:2]
                im=Image.fromarray(rgb0)
                im.thumbnail((1400,1900))
                rgb=np.array(im)
                h,w=rgb.shape[:2]
                torch.cuda.synchronize()
                start=time.perf_counter()
                result=yolo.predict(rgb0[:,:,::-1].copy(),imgsz=1280,conf=.35,iou=.45,
                    classes=[bicycle_class_id(yolo.names)],device=0,max_det=200,verbose=False)[0]
                boxes=result.boxes.xyxy.cpu().numpy()*np.array([w/w0,h/h0,w/w0,h/h0])
                predictor.set_image(rgb)

                def pts(poly):
                    return [(x*w/900,y*h/1200) for x,y in poly]

                def raster(polys):
                    image=Image.new('1',(w,h),0)
                    d=ImageDraw.Draw(image)
                    for poly in polys:d.polygon(pts(poly),fill=1)
                    return np.array(image,dtype=bool)

                allow=raster([cfg['allow']])
                deny=raster(cfg['deny'])
                roi=raster([cfg['roi']]) if 'roi' in cfg else np.ones((h,w),bool)
                canvas=rgb.astype(float)
                items=[]
                for i,b in enumerate(boxes,1):
                    masks,scores,_=predictor.predict(box=b,multimask_output=False)
                    mask=np.asarray(masks).reshape(-1,h,w)[0]>0
                    score=float(np.asarray(scores).ravel()[0])
                    verdict=classify_mask(mask,allow,deny,has_zone=True,score=score,
                        min_area=200,min_score=0,contact_ratio=.02,deny_tol=.05)
                    cx=min(w-1,max(0,int((b[0]+b[2])/2)))
                    cy=min(h-1,max(0,int((b[1]+b[3])/2)))
                    if not roi[cy,cx]:verdict=dict(verdict='范围外未评估',contact_in=-1,deny=-1,reason='outside_roi')
                    color=np.array(config.COLORS[verdict['verdict']])
                    canvas[mask]=canvas[mask]*.7+color*.3
                    canvas[contact_band(mask,.02)]=[255,0,255]
                    items.append(dict(id=i,box=b.tolist(),sam_quality=score,**verdict))
                torch.cuda.synchronize()
                elapsed=time.perf_counter()-start
                image=Image.fromarray(np.uint8(np.clip(canvas,0,255)))
                d=ImageDraw.Draw(image)
                d.line(pts(cfg['allow'])+[pts(cfg['allow'])[0]],fill=(30,130,255),width=5)
                for b,item in zip(boxes,items):
                    d.text((b[0],b[1]),f"#{item['id']} {item['verdict']}",font=font,
                        fill=config.COLORS[item['verdict']],stroke_width=1,stroke_fill='black')
                dest=OUT/tag
                dest.mkdir(exist_ok=True)
                image.save(dest/name,quality=94)
                rows.append(dict(image=name,candidates=len(boxes),seconds=elapsed,per_bike=items))
                print(tag,name,'boxes',len(boxes),'seconds',round(elapsed,2),flush=True)
                del result
        finally:
            stop.set()
            thread.join(timeout=6)
        summaries.append(dict(model=tag,weights=str(weights),
            torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
            torch_peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
            sampled_whole_gpu_peak_used_mib=max(samples) if samples else None,
            images=rows))
        del predictor,sam,yolo
        gc.collect()
        torch.cuda.empty_cache()
    crops=[('A_1_038.jpg','雨天：近端白线',(240,780,720,1090),'rain_near'),
           ('A_1_038.jpg','雨天：远端白线',(330,150,610,380),'rain_far'),
           ('A_2_007.jpg','斜侧：中间车辆',(180,570,710,850),'side_middle')]
    for name,label,crop,stem in crops:
        sheet=Image.new('RGB',(1400,650),'#eeeeee')
        draw=ImageDraw.Draw(sheet)
        for col,tag in enumerate(['finetuned_x','pretrained']):
            with Image.open(OUT/tag/name) as image:
                w,h=image.size
                panel=image.crop(tuple(int(v*(w/900 if i%2==0 else h/1200)) for i,v in enumerate(crop)))
            panel=ImageOps.contain(panel,(690,560))
            sheet.paste(panel,(col*700+(700-panel.width)//2,65))
            draw.text((col*700+10,5),'微调大模型＋SAM2' if col==0 else '官方大模型＋SAM2',font=font,fill='black')
            draw.text((col*700+10,32),label+'；紫色为最低条带近似',font=font,fill='black')
        sheet.save(OUT/(stem+'_comparison.jpg'),quality=94)
    report=dict(configuration='One resident YOLO11x plus SAM2.1 tiny; batch 1 inference; sequential box prompts; YOLO imgsz 1280; SAM source max 1400x1900.',
        gpu=torch.cuda.get_device_name(0),gpu_total_mib=torch.cuda.get_device_properties(0).total_memory/1024**2,
        notes='PyTorch peaks are process allocations/reservations; nvidia-smi sampled totals include desktop and other processes. No guarantee for larger SAM models or batched inference.',
        models=summaries)
    (OUT/'memory_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({m['model']:{k:v for k,v in m.items() if k!='images'} for m in summaries},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
