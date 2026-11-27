"""Run public part-model checkpoints on the five frozen development images."""
from pathlib import Path
import argparse
import hashlib
import json
import logging
import math
import shutil
import time
import traceback
import types

import numpy as np
from PIL import Image,ImageDraw
import torch
import torch.nn.functional as F

from part_model_runtime import ROOT,PARTGLEE,CACHE,load_model


def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def sha(path):
    return hashlib.file_digest(Path(path).open("rb"),"sha256").hexdigest()


def attach_parent_export(model):
    """Export existing Q-Former lineage, without changing model predictions.

    Author's normal postprocess discards which object query generated a part.
    Repeat the same top-k selection only to retain those indices for inspection.
    This lineage is a model hypothesis, not a ground-truth identity guarantee.
    """
    state={}
    core=model.partglee
    original_top=core.hierarchical_topk_outputs
    def top(self,task,object_level_task,part_level_task,batch_name_list,object_outputs,
            part_outputs,test_topk,topk_object,topk_part,custom_object_categories_idx,custom_part_categories_idx):
        parent_top=torch.topk(object_outputs['pred_logits'].max(-1)[0],self.topk_object_queries_num,1)[1][0]
        object_top=torch.topk(object_outputs['pred_logits'].max(-1)[0],topk_object,1)[1][0]
        part_top=torch.topk(part_outputs['pred_logits'].max(-1)[0],topk_part,1)[1][0]
        parents=torch.cat([object_top,parent_top[part_top//self.num_part_queries]])
        state['parents']=parents
        state['parent_boxes']=object_outputs['pred_boxes'][0,parents]
        state['parent_classes']=object_outputs['pred_logits'][0,parents].argmax(-1)
        state['parent_scores']=object_outputs['pred_logits'][0,parents].sigmoid().max(-1)[0]
        return original_top(task,object_level_task,part_level_task,batch_name_list,object_outputs,
            part_outputs,test_topk,topk_object,topk_part,custom_object_categories_idx,custom_part_categories_idx)
    core.hierarchical_topk_outputs=types.MethodType(top,core)
    original_instance=model.instance_inference
    def instance(self,mask_cls,mask_pred,mask_box_result,task):
        indices=mask_cls.sigmoid().flatten().topk(self.test_topk_per_image,sorted=False).indices//mask_cls.shape[1]
        result=original_instance(mask_cls,mask_pred,mask_box_result,task)
        result.parent_query=state['parents'][indices]
        result.parent_class=state['parent_classes'][indices]
        result.parent_score=state['parent_scores'][indices]
        result.parent_boxes_normalized=state['parent_boxes'][indices]
        return result
    model.instance_inference=types.MethodType(instance,model)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['partglee','vlpart'],required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=5)
    args=parser.parse_args();args.out=args.out.resolve()
    if args.out.exists():
        parser.error('Output exists; select a new directory')
    args.out.mkdir(parents=True);(args.out/'masks').mkdir();(args.out/'previews').mkdir()
    base=ROOT/'runs/contact_eval_20261127_v1'
    sources=json.loads((base/'inputs.json').read_text(encoding='utf-8'))
    images=sorted([i for i in sources if Path(i['path']).suffix=='.jpg'],key=lambda i:Path(i['path']).name)[:args.limit]
    scenes=json.loads((base/'input_snapshot/image/scenes.json').read_text(encoding='utf-8'))['school_row']
    protocol=dict(model=args.model,score_threshold=.5,training=False,human_contacts_used=False,
                  input_short_side=800,input_long_side_max=1333,output_coordinates='same 1400x1900 thumbnail as baseline',
                  vocabulary='Pascal Part 93' if args.model=='vlpart' else 'Pascal joint 113',
                  partglee_attention='authors PyTorch grid_sample fallback; not compiled CUDA timing',
                  parent_export='Q-Former query lineage; not ground-truth identity' if args.model=='partglee' else 'No parent-instance output',
                  requested_images=len(images),checkpoint=str(CACHE/('PartGLEE_RN50.pth' if args.model=='partglee' else 'r50_pascalpart.pth')))
    dump(args.out/'protocol.json',protocol)
    for path in (Path(__file__),Path(__file__).with_name('part_model_runtime.py')):
        shutil.copy2(path,args.out/(path.stem+'_source.py'))
    tracked=[base/'results.json',base/'input_snapshot/image/scenes.json',Path(protocol['checkpoint']),
             Path(__file__),Path(__file__).with_name('part_model_runtime.py'),*[Path(i['path']) for i in images]]
    manifest=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in tracked]
    dump(args.out/'inputs.json',manifest)
    logging.getLogger('fvcore.common.checkpoint').setLevel(logging.ERROR)
    started=time.perf_counter()
    model,cfg,labels,load_info=load_model(args.model)
    if args.model=='partglee':
        labels=model.dataset_name_dicts['pascal_joint']
        attach_parent_export(model)
    dump(args.out/'checkpoint_load.json',load_info)
    print('MODEL_LOADED',args.model,len(labels),'classes',round(time.perf_counter()-started,2),'seconds',flush=True)
    from detectron2.data import transforms as T
    augment=T.ResizeShortestEdge([800,800],1333)
    results=[]
    torch.cuda.reset_peak_memory_stats()
    for item in images:
        path=Path(item['path']);stem=path.stem
        if sha(path)!=item['sha256']:
            raise RuntimeError('Image changed since baseline')
        with Image.open(path) as opened:
            image=opened.convert('RGB')
        image.thumbnail((1400,1900));width,height=image.size
        rgb=np.array(image)
        network=augment.get_transform(rgb).apply_image(rgb)
        if cfg.INPUT.FORMAT=='BGR':
            network=network[:,:,::-1].copy()
        tensor=torch.as_tensor(network.transpose(2,0,1).copy(),dtype=torch.float32)
        sample=dict(image=tensor,height=height,width=width)
        if args.model=='partglee':
            sample['dataset_name']='pascal_joint'
        torch.cuda.synchronize();started=time.perf_counter()
        with torch.inference_mode():
            instances=model([sample])[0]['instances'].to('cpu')
        torch.cuda.synchronize();elapsed=time.perf_counter()-started
        parts=[];masks=[]
        for i in range(len(instances)):
            score=float(instances.scores[i]);category=labels[int(instances.pred_classes[i])]
            if score<.5 or category not in ['bicycle','bicycle:wheel']:
                continue
            mask=instances.pred_masks[i].numpy().astype(bool)
            if not mask.any():
                continue
            row=dict(index=len(parts),category=category,score=score,box=instances.pred_boxes.tensor[i].tolist(),mask_area=int(mask.sum()))
            if args.model=='partglee':
                nh,nw=network.shape[:2];div=model.size_divisibility
                ph=math.ceil(nh/div)*div;pw=math.ceil(nw/div)*div
                parent=model.box_postprocess(instances.parent_boxes_normalized[i:i+1],ph/nh*height,pw/nw*width)[0]
                row.update(parent_query=int(instances.parent_query[i]),parent_class=labels[int(instances.parent_class[i])],
                           parent_score=float(instances.parent_score[i]),parent_box=parent.tolist())
            parts.append(row);masks.append(mask)
        np.savez_compressed(args.out/'masks'/(stem+'.npz'),masks=np.stack(masks) if masks else np.zeros((0,height,width),bool))
        record=dict(image=path.name,processed_size=[width,height],network_size=[network.shape[1],network.shape[0]],seconds=round(elapsed,4),
                    all_returned_instances=len(instances),parts=parts)
        results.append(record)
        dump(args.out/'results.json',results)
        canvas=rgb.copy();allow=scenes['images'][stem]['allow']
        region=Image.new('1',(width,height));ImageDraw.Draw(region).polygon([(x*width,y*height) for x,y in allow],fill=1)
        inside=np.asarray(region,bool)
        for part,mask in zip(parts,masks):
            if part['category']=='bicycle:wheel':
                canvas[mask]=(.5*canvas[mask]+.5*np.array([10,210,150])).astype(np.uint8)
                canvas[mask & ~inside]=[255,45,45]
        preview=Image.fromarray(canvas);draw=ImageDraw.Draw(preview)
        xy=[(x*width,y*height) for x,y in allow];draw.line(xy+xy[:1],fill='#2299ff',width=4)
        for part in parts:
            if part['category']=='bicycle:wheel':
                draw.rectangle(part['box'],outline='#ffcf30',width=2)
                draw.text((max(0,part['box'][0]),max(0,part['box'][1]-12)),f"W{part['index']} {part['score']:.2f}",fill='black',stroke_fill='white',stroke_width=1)
        preview.save(args.out/'previews'/(stem+'.jpg'),quality=93)
        print(path.name,'wheel',sum(p['category']=='bicycle:wheel' for p in parts),'bicycles',sum(p['category']=='bicycle' for p in parts),round(elapsed,2),'seconds',flush=True)
        del instances
    summary=dict(model=args.model,images=len(results),wheel_instances=sum(p['category']=='bicycle:wheel' for r in results for p in r['parts']),
                 object_instances=sum(p['category']=='bicycle' for r in results for p in r['parts']),
                 peak_allocated_gib=round(torch.cuda.max_memory_allocated()/2**30,3),
                 inputs_unchanged=all(sha(i['path'])==i['sha256'] for i in manifest),status='complete')
    dump(args.out/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
