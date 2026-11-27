"""Audit frozen part outputs and generate a task-level comparison report.

Wheel silhouettes are candidate evidence, not ground-footprint truth. Association
thresholds are exploratory and fixed here before task-level scoring, not tuned.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import sys

import numpy as np
from PIL import Image,ImageDraw,ImageOps

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import config  # Font only; do not activate the lightweight yolo-lib shims.

BASE=ROOT/'runs/contact_eval_20261127_v1'
MODELS={'partglee':ROOT/'runs/parts_partglee_20261127_v1',
        'vlpart':ROOT/'runs/parts_vlpart_20261127_v2'}
BAD={'出界','压线','占道'}


def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def xyxy(row,width,height):
    _,cx,cy,bw,bh=row['box']
    return [(cx-bw/2)*width,(cy-bh/2)*height,(cx+bw/2)*width,(cy+bh/2)*height]


def box_iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/max(1e-12,aa+bb-inter)


def select_unique(scores,minimum,gap):
    """Return no identity for ties/weak overlap, never silently choose nearest."""
    if not scores:
        return None,0.,0.
    order=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)
    best=scores[order[0]];second=scores[order[1]] if len(order)>1 else 0.
    chosen=order[0] if best>=minimum and best-second>=gap else None
    return chosen,float(best),float(second)


def mask_owner(mask,boxes):
    height,width=mask.shape;area=int(mask.sum())
    scores=[]
    for x1,y1,x2,y2 in boxes:
        bounds=(min(width,max(0,math.floor(x1))),min(height,max(0,math.floor(y1))),
                min(width,max(0,math.ceil(x2))),min(height,max(0,math.ceil(y2))))
        scores.append(float(mask[bounds[1]:bounds[3],bounds[0]:bounds[2]].sum()/max(1,area)))
    return select_unique(scores,.8,.2)


def lowest_band(mask):
    ys=np.nonzero(mask)[0]
    if not len(ys):
        return np.zeros_like(mask)
    thickness=max(2,math.ceil(.02*(int(ys.max())-int(ys.min())+1)))
    return mask & (np.arange(mask.shape[0])[:,None]>=int(ys.max())-thickness+1)


def candidate_metrics(rows,flag):
    known=[r for r in rows if r['reference'] in BAD|{'合规'}]
    positives=[r for r in known if r['reference'] in BAD]
    negatives=[r for r in known if r['reference']=='合规']
    return dict(positive_denominator=len(positives),negative_denominator=len(negatives),
                hit=sum(r[flag] for r in positives),without_candidate=sum(not r[flag] for r in positives),
                false_candidate=sum(r[flag] for r in negatives),
                reference_unknown_candidates=sum(r[flag] for r in rows if r['reference']=='无法判定'),
                vehicles_with_assigned_wheel=sum(r['assigned_wheels']>0 for r in known))


def case_sheet(out,r,rgb,allow,baseline_mask,model_data,polygon):
    height,width=allow.shape;box=xyxy(r,width,height)
    crop=(max(0,int(box[0])-75),max(0,int(box[1])-85),min(width,int(box[2])+75),min(height,int(box[3])+110))
    sheet=Image.new('RGB',(1800,700),'#f3f2ed');d=ImageDraw.Draw(sheet);font=config.load_font(22)
    d.text((12,10),f"{r['image']} #{r['id']} | 人工状态：{r['reference']} | 绿色为模型车轮候选，红色仅表示线外投影",font=font,fill='black')
    panels=[('原图与人工目标框',None,None),('旧 SAM 掩膜 / 最低条带',baseline_mask,lowest_band(baseline_mask))]
    for name in ('partglee','vlpart'):
        all_wheels=np.zeros_like(allow)
        for part,mask in zip(model_data[name][0]['parts'],model_data[name][1]):
            if part['category']=='bicycle:wheel':
                all_wheels|=mask
        panels.append((name+' 全部车轮候选',all_wheels,all_wheels&~allow))
    for index,(title,mask,evidence) in enumerate(panels):
        arr=rgb.copy()
        if mask is not None:
            arr[mask]=(.6*arr[mask]+.4*np.array([20,210,140])).astype(np.uint8)
            arr[evidence]=[235,0,215] if index==1 else [250,45,45]
        image=Image.fromarray(arr);draw=ImageDraw.Draw(image)
        draw.line(polygon+polygon[:1],fill='#2299ff',width=3)
        draw.rectangle(box,outline='#ffce20',width=3)
        panel=ImageOps.contain(image.crop(crop),(430,600))
        d.text((index*450+10,55),title,font=font,fill='black')
        sheet.paste(panel,(index*450+(450-panel.width)//2,100))
    sheet.save(out/'cases'/f"{Path(r['image']).stem}_{r['id']:03d}.jpg",quality=94)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve()
    if out.exists():
        parser.error('Output exists; use a new directory')
    out.mkdir(parents=True);(out/'cases').mkdir()
    references=json.loads((BASE/'results.json').read_text(encoding='utf-8'))
    sources=json.loads((BASE/'inputs.json').read_text(encoding='utf-8'))
    source_images={Path(i['path']).name:Path(i['path']) for i in sources if Path(i['path']).suffix=='.jpg'}
    scenes=json.loads((BASE/'input_snapshot/image/scenes.json').read_text(encoding='utf-8'))['school_row']['images']
    models={n:{r['image']:r for r in json.loads((folder/'results.json').read_text(encoding='utf-8'))} for n,folder in MODELS.items()}
    protocol=dict(score_threshold=.5,training=False,labels_used_for_inference=False,
        silhouette_candidate='wheel mask outside fraction > .02',
        lower_candidate='each wheel lowest 2% strip outside fraction > .1',
        common_association='>=80% wheel pixels in one human box, with >=20 percentage point lead',
        hierarchical_association='PartGLEE parent class bicycle, parent score >=.5, parent/GT box IoU >=.5 and margin >=.1',
        caveat='Exploratory association/geometry controls, not paper rules. No candidate is not compliant. No pixel/part-parent ground truth.',
        independent_test=False)
    dump(out/'protocol.json',protocol)
    wheel_rows=[];vehicle_rows={n:[] for n in MODELS};hierarchical_rows=[];per_image=[]
    for image,path in sorted(source_images.items()):
        ref=[r for r in references if r['image']==image];width,height=ref[0]['processed_size']
        boxes=[xyxy(r,width,height) for r in ref]
        zone=Image.new('1',(width,height));xy=[(x*width,y*height) for x,y in scenes[Path(image).stem]['allow']]
        ImageDraw.Draw(zone).polygon(xy,fill=1);allow=np.asarray(zone,bool)
        data={};counts={'image':image,'vehicles':len(ref)}
        for name,folder in MODELS.items():
            record=models[name][image]
            with np.load(folder/'masks'/(Path(image).stem+'.npz')) as archive:
                masks=archive['masks']
            assert len(record['parts'])==len(masks)
            data[name]=(record,masks)
            state={r['id']:dict(image=image,id=r['id'],reference=r['reference'],assigned_wheels=0,whole_candidate=False,lower_candidate=False) for r in ref}
            hstate={r['id']:dict(image=image,id=r['id'],reference=r['reference'],assigned_wheels=0,whole_candidate=False,lower_candidate=False) for r in ref}
            count=0;assigned=0;hassigned=0
            for p,mask in zip(record['parts'],masks):
                if p['category']!='bicycle:wheel':
                    continue
                count+=1
                outside=float((mask&~allow).sum()/max(1,mask.sum()))
                lower=lowest_band(mask);lower_out=float((lower&~allow).sum()/max(1,lower.sum()))
                choice,best,second=mask_owner(mask,boxes)
                identifier=ref[choice]['id'] if choice is not None else None
                hid=None;hiou=0.;hsecond=0.
                if name=='partglee' and p['parent_class']=='bicycle' and p['parent_score']>=.5:
                    hc,hiou,hsecond=select_unique([box_iou(p['parent_box'],box) for box in boxes],.5,.1)
                    hid=ref[hc]['id'] if hc is not None else None
                for target,identifier_value in ((state,identifier),(hstate,hid)):
                    if identifier_value is not None:
                        target[identifier_value]['assigned_wheels']+=1
                        target[identifier_value]['whole_candidate']|=outside>.02
                        target[identifier_value]['lower_candidate']|=lower_out>.1
                assigned+=identifier is not None;hassigned+=hid is not None
                wheel_rows.append(dict(model=name,image=image,wheel_index=p['index'],score=p['score'],area=int(mask.sum()),
                    whole_outside=outside,lower_outside=lower_out,common_owner=identifier,overlap=best,runner_up=second,
                    parent_query=p.get('parent_query'),hierarchical_owner=hid,parent_iou=hiou,parent_runner_up_iou=hsecond))
            vehicle_rows[name]+=list(state.values())
            if name=='partglee':
                hierarchical_rows+=list(hstate.values())
            counts[name]=dict(wheels=count,common_assigned=assigned,hierarchical_assigned=hassigned)
        per_image.append(counts)
        with np.load(BASE/'masks'/(Path(image).stem+'.npz')) as archive:
            old_masks=archive['masks']
        with Image.open(path) as opened:
            im=opened.convert('RGB')
        im.thumbnail((1400,1900));rgb=np.array(im)
        for r,mask in zip(ref,old_masks):
            baseline_alarm_error=r['reference']!='无法判定' and ((r['reference'] in BAD)!=(r['band_verdict'] in BAD))
            if r['reference'] in BAD or baseline_alarm_error:
                case_sheet(out,r,rgb,allow,mask,data,xy)
    summary=dict(objects=len(references),per_image=per_image,
        common_association={n:{key:candidate_metrics(rows,key) for key in ['whole_candidate','lower_candidate']} for n,rows in vehicle_rows.items()},
        partglee_hierarchical_association={key:candidate_metrics(hierarchical_rows,key) for key in ['whole_candidate','lower_candidate']},
        wheel_totals={n:sum(r['model']==n for r in wheel_rows) for n in MODELS},
        associated_wheel_totals={n:sum(r['model']==n and r['common_owner'] is not None for r in wheel_rows) for n in MODELS},
        warning='Candidate flags are not validated final parking states; unmatched wheels remain unassigned; no-candidate not compliant.')
    dump(out/'summary.json',summary);dump(out/'wheel_candidates.json',wheel_rows)
    dump(out/'vehicle_candidates.json',vehicle_rows);dump(out/'partglee_parent_vehicle_candidates.json',hierarchical_rows)
    for filename,rows in [('wheel_candidates.csv',wheel_rows),('vehicle_candidates.csv',[dict(model=n,**r) for n,rs in vehicle_rows.items() for r in rs])]:
        with (out/filename).open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    shutil_source=Path(__file__).read_bytes();(out/'compare_part_models_source.py').write_bytes(shutil_source)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
