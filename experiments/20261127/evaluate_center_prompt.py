"""Change only SAM prompting: frozen box vs frozen box plus automatic center.

The center is weak instance guidance, never an annotated wheel/ground point.
No new annotation, training, threshold selection or default-pipeline mutation.
"""
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/"config.py").is_file())
sys.path.insert(0,str(ROOT))
import config
config.activate()

import argparse
import csv
import json
import shutil
import time

import numpy as np
from PIL import Image,ImageDraw,ImageOps

from evaluate_boundary_candidates import KNOWN,acceptance,digest,dump,metrics,raster
from violation_rules import BAD_VERDICTS,classify_mask,contact_band
from sam_prompts import center_point_prompt


def preview(rgb,old,new,poly,row,destination):
    h,w=old.shape
    _,cx,cy,bw,bh=row["box"]
    # Same prompt-centered crop for both; preserve enough border context.
    x1=max(0,int((cx-bw*.5)*w)-70);x2=min(w,int((cx+bw*.5)*w)+70)
    y1=max(0,int((cy-bh*.5)*h)-70);y2=min(h,int((cy+bh*.5)*h)+70)
    sheet=Image.new("RGB",(1200,750),"#f4f3ef")
    draw=ImageDraw.Draw(sheet);font=config.load_font(22)
    draw.text((12,10),f"{row['image']} #{row['id']} | 人工状态：{row['reference']}",font=font,fill="black")
    for i,(mask,name,verdict) in enumerate(((old,"原框提示",row["predictions"]["band"]),
                                           (new,"框+自动中心点",row["predictions"]["center_prompt"]))):
        array=rgb.copy();array[mask]=(.65*array[mask]+.35*np.array([25,200,115])).astype(np.uint8)
        array[contact_band(mask)]=[235,0,215]
        image=Image.fromarray(array);d=ImageDraw.Draw(image)
        if poly:
            xy=[(x*w,y*h) for x,y in poly];d.line(xy+xy[:1],fill="#2299ff",width=3)
        d.rectangle(((cx-bw*.5)*w,(cy-bh*.5)*h,(cx+bw*.5)*w,(cy+bh*.5)*h),outline="#ffcf30",width=2)
        if i:
            x,y=row["automatic_center"]
            d.ellipse((x-5,y-5,x+5,y+5),fill="#ff8020",outline="white",width=2)
        panel=ImageOps.contain(image.crop((x1,y1,x2,y2)),(580,640))
        draw.text((i*600+12,50),f"{name} → {verdict}",font=font,fill="black")
        sheet.paste(panel,(i*600+(600-panel.width)//2,95))
    sheet.save(destination,quality=93)


def write_report(out,rows,summary):
    base=summary["metrics"]["band"];new=summary["metrics"]["center_prompt"]
    lines=["# 自动中心点提示对照：2026-11-27", "",
           "这是固定掩膜规则对照之后的第二个小实验：只改变 SAM 提示。第一轮几何扩展没有通过开发门槛，因此检验自动增加实例提示能否改善逐车分割。", "",
           "## 改动与控制", "",
           "原方法：同一人工车辆框提示 SAM2.1 Tiny。新方法：同一框，加框中心的一个前景点；点由框自动计算，不读取人工接地点，不要求新标注。车辆框中心可能落在遮挡物上，因此它只是待验证的弱提示。", "",
           "图片、缩放上限 1400×1900、SAM 权重、人工区域、质量门槛和最低条带规则均沿用旧实验。没有训练或阈值搜索。每张图只编码一次。此实验不是论文网络复现，亦不预测真实接地点。", "",
           "## 全量报警结果", "",
           "| 方法 | 检出违停 / 8 | 漏报 | 合规误报 / 144 | 拒答 / 152 | 精确状态一致 / 152 |", "|---|---:|---:|---:|---:|---:|"]
    for name,m in (("原框提示",base),("框+自动中心点",new)):
        lines.append(f"| {name} | {m['tp']} | {m['fn']} | {m['fp']} | {m['n']-m['resolved']} | {m['exact_agreement']} |")
    lines += ["", "5 辆人工无法判定不计对错，保留全部输出；拒答没有算成正确合规。", "",
              "## 逐图结果", "",
              "| 图片 | 人工违停/合规 | 原检出/误报 | 新检出/误报 |", "|---|---:|---:|---:|"]
    for image,values in summary["by_image"].items():
        b=values["band"];n=values["center_prompt"]
        lines.append(f"| {image} | {b['positive']}/{b['negative']} | {b['tp']}/{b['fp']} | {n['tp']}/{n['fp']} |")
    lines += ["", "## 是否接入默认流程", ""]
    if summary["development_gate"]:
        lines.append("通过本轮开发门槛，但只有这 5 张开发图，不足以证明新场景泛化；保留实验入口，默认流程暂不替换。")
    else:
        lines.append("没有通过开发门槛，默认流程保持原方法。框中心不保证属于目标车辆，不能在密集遮挡下当作可信前景提示。")
    lines += ["", "门槛：152 辆均自动作答；总检出不降低、总误报不增加且至少一项改善；每张图都不增加误报、不减少检出。", "",
              "## 逐车变化", "",
              "| 图片/编号 | 人工状态 | 原结果 | 新结果 |", "|---|---|---|---|"]
    for row in rows:
        if row["predictions"]["band"]!=row["predictions"]["center_prompt"]:
            lines.append(f"| {row['image']} / {row['id']} | {row['reference']} | {row['predictions']['band']} | {row['predictions']['center_prompt']} |")
    lines += ["", "## 文件与复现", "",
              "- [逐车结果](per_bike.csv)", "- [统计与输入未变检查](summary.json)", "- [实验协议](protocol.json)",
              "- [局部可视化](cases/)", "- `masks/` 保存新提示产生的全部 157 辆掩膜；源码留有快照。", "",
              "运行：`python experiments/20261127/evaluate_center_prompt.py --out runs/center_prompt_20261127_v2`，必须使用新输出目录和现有 CUDA 环境。原始标注与旧缓存均未改写。"]
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        parser.error("Output exists; choose a new directory")
    base=ROOT/"runs/contact_eval_20261127_v1"
    original=json.loads((base/"results.json").read_text(encoding="utf-8"))
    manifest=json.loads((base/"inputs.json").read_text(encoding="utf-8"))
    images={Path(r["path"]).name:Path(r["path"]) for r in manifest if Path(r["path"]).suffix==".jpg"}
    scene_path=base/"input_snapshot/image/scenes.json"
    scenes=json.loads(scene_path.read_text(encoding="utf-8"))["school_row"]
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if not torch.cuda.is_available():
        raise RuntimeError("Use the existing CUDA environment")
    tracked=[base/"results.json",scene_path,Path(__file__),Path(__file__).with_name("evaluate_boundary_candidates.py"),
             ROOT/"relabel/violation_rules.py",ROOT/"relabel/sam_prompts.py",config.SAM2_CHECKPOINT,
             *sorted((base/"masks").glob("*.npz")),*images.values()]
    inputs=[dict(path=str(p.resolve()),sha256=digest(p),bytes=p.stat().st_size) for p in tracked]
    for item in manifest:
        if Path(item["path"]).suffix==".jpg" and digest(item["path"])!=item["sha256"]:
            raise ValueError("Image changed: "+item["path"])
    args.out.mkdir(parents=True)
    (args.out/"masks").mkdir();(args.out/"cases").mkdir()
    dump(args.out/"inputs.json",inputs)
    dump(args.out/"protocol.json",dict(date="2026-11-27",change="box + automatic box-center positive point",
        baseline="box only; original cached masks",human_contact_points_used=False,training=False,new_annotation=False,
        sam_config=config.SAM2_CONFIG,torch_version=torch.__version__,sam_checkpoint=str(config.SAM2_CHECKPOINT),
        resize=[1400,1900],multimask_output=False,point_labels=[1],rule="unchanged lowest 2% + .1/.9 thresholds",
        caveat="Box center can belong to an occluder; not a true contact point. Development comparison, not independent testing."))
    for source in (Path(__file__),Path(__file__).with_name("evaluate_boundary_candidates.py"),ROOT/"relabel/violation_rules.py",ROOT/"relabel/sam_prompts.py"):
        shutil.copy2(source,args.out/(source.stem+"_source.py"))
    predictor=SAM2ImagePredictor(build_sam2(config.SAM2_CONFIG,str(config.SAM2_CHECKPOINT),device="cuda"))
    rows=[];image_times={}
    with torch.inference_mode():
        for image,path in sorted(images.items()):
            started=time.perf_counter()
            with Image.open(path) as opened:
                im=opened.convert("RGB")
            im.thumbnail((1400,1900));rgb=np.array(im);w,h=im.size
            group=[r for r in original if r["image"]==image]
            assert all(r["processed_size"]==[w,h] for r in group)
            zone=scenes.get("images",{}).get(Path(image).stem,{})
            poly=zone.get("allow",scenes.get("allow",[]));allow=raster(poly,w,h)
            deny=np.zeros_like(allow)
            for polygon in zone.get("deny",scenes.get("deny",[])):
                deny|=raster(polygon,w,h)
            predictor.set_image(rgb)
            new_masks=[]
            with np.load(base/"masks"/(Path(image).stem+".npz")) as archive:
                old_masks=archive["masks"]
            assert len(old_masks)==len(group)
            for i,r in enumerate(group):
                _,cx,cy,bw,bh=r["box"]
                box=np.array([(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h])
                point_prompt=center_point_prompt(box,w,h)
                center=point_prompt["point_coords"][0].tolist()
                masks,scores,_=predictor.predict(box=box,**point_prompt,multimask_output=False)
                mask=np.asarray(masks).reshape(-1,h,w)[0]>0;new_masks.append(mask)
                score=float(np.asarray(scores).ravel()[0])
                result=classify_mask(mask,allow,deny,has_zone=bool(poly),score=score)
                row=dict(image=image,id=r["id"],box=r["box"],reference=r["reference"],
                         predictions=dict(band=r["band_verdict"],center_prompt=result["verdict"]),
                         automatic_center=center,sam_score=score,contact_in=result["contact_in"],
                         mask_area=int(mask.sum()),reason=result["reason"],case="")
                if row["predictions"]["band"]!=row["predictions"]["center_prompt"] or (r["reference"] in KNOWN and
                        ((r["reference"] in BAD_VERDICTS)!=(r["band_verdict"] in BAD_VERDICTS))):
                    row["case"]=f"cases/{Path(image).stem}_{r['id']:03d}.jpg"
                    preview(rgb,old_masks[i],mask,poly,row,args.out/row["case"])
                rows.append(row)
            np.savez_compressed(args.out/"masks"/(Path(image).stem+".npz"),masks=np.stack(new_masks))
            image_times[image]=round(time.perf_counter()-started,3)
            print(image,len(group),image_times[image],"seconds",flush=True)
    names=("band","center_prompt")
    totals={n:metrics(rows,n) for n in names}
    assert totals["band"]["tp"]==7 and totals["band"]["fp"]==6
    by_image={image:{n:metrics([r for r in rows if r["image"]==image],n) for n in names} for image in images}
    summary=dict(total_objects=len(rows),metrics=totals,by_image=by_image,
        development_gate=acceptance(totals["center_prompt"],totals["band"],by_image,"center_prompt"),
        inputs_unchanged=all(digest(i["path"])==i["sha256"] for i in inputs),
        image_times=image_times,default_pipeline_modified=False,independent_test=False)
    if not summary["inputs_unchanged"]:
        raise RuntimeError("Inputs changed during evaluation")
    dump(args.out/"results.json",rows);dump(args.out/"summary.json",summary)
    with (args.out/"per_bike.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        fields=["image","id","reference","band","center_prompt","sam_score","contact_in","mask_area","reason","case"]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({key:(row["predictions"][key] if key in names else row[key]) for key in fields})
    write_report(args.out,rows,summary)
    print(json.dumps(dict(metrics=totals,development_gate=summary["development_gate"],inputs_unchanged=summary["inputs_unchanged"]),ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
