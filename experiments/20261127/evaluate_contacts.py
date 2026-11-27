"""Development comparison with fixed human boxes/zones: SAM bottom band vs human points.

No detector is evaluated, no training is performed, and source annotations are
never edited. Human parking states are provisional reference labels, not an
independently adjudicated test set. Unknown contacts remain unknown.
Contact visibility and applicability do not invalidate human parking states.
Point-rule abstention is a method limitation; fallen-wheel/body crossing is not
implemented by this geometry-only diagnostic.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.py").is_file())))
import config
config.activate()

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import shutil
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from violation_rules import classify_mask, contact_band, resolve_zones

PAIR_NAMES = (("point1", "front"), ("point2", "rear"))
COLORS = {"合规": "#00b963", "出界": "#f04b48", "压线": "#f2b126",
          "无法判定": "#999999", "占道": "#f04b48", "未标定": "#999999"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalized_contacts(record):
    contacts = record.get("contact_points", {})
    result = []
    for new, old in PAIR_NAMES:
        item = contacts.get(new, contacts.get(old, {}))
        visibility = item.get("visibility", "unmarked")
        reviewed = item.get("reviewed", visibility in ("visible", "unobservable", "not_applicable")) is True
        point = item.get("point")
        if visibility == "visible":
            if not (isinstance(point, list) and len(point) == 2 and
                    all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in point)):
                raise ValueError("Invalid visible contact coordinates")
        else:
            point = None
        result.append(dict(visibility=visibility, reviewed=reviewed, point=point))
    return result


def classify_points(contacts, allow, width, height):
    """Point geometry only; abstention does not override the human parking state.

    Caller must separately establish whether points describe actual ground
    contacts. Fallen-wheel silhouette crossing/body overhang is not evaluated.
    """
    distances = [None, None]
    if not allow:
        return "未标定", distances
    poly = (np.asarray(allow, dtype=np.float32) * [width, height]).astype(np.float32)
    for i, item in enumerate(contacts):
        if item["visibility"] == "visible":
            x, y = item["point"]
            distances[i] = float(cv2.pointPolygonTest(poly, (x * width, y * height), True))
    if any(item["visibility"] == "not_applicable" for item in contacts):
        return "无法判定", distances
    known = [d for d in distances if d is not None]
    if any(d < 0 for d in known):
        return "出界", distances
    if len(known) != 2:
        return "无法判定", distances
    return ("压线" if any(d == 0 for d in known) else "合规"), distances


def pair_metrics(rows, column):
    counts = Counter((row["reference"], row[column]) for row in rows)
    tp = counts[("出界", "出界")]
    fp = counts[("合规", "出界")]
    positives = sum(r["reference"] == "出界" for r in rows)
    negatives = len(rows) - positives
    return dict(n=len(rows), reference_out=positives, reference_in=negatives,
                agreement=sum(r["reference"] == r[column] for r in rows),
                correct_out=tp, false_out=fp, missed_out=positives - tp,
                unresolved=sum(r[column] not in ("合规", "出界") for r in rows),
                out_precision=tp / (tp + fp) if tp + fp else None,
                out_recall=tp / positives if positives else None,
                in_false_alarm_fraction=fp / negatives if negatives else None,
                matrix={ref: dict(Counter(r[column] for r in rows if r["reference"] == ref))
                        for ref in ("合规", "出界")})


def read_batch(base):
    images = base / "image"
    labels = base / "labels/image"
    scenes_path = images / "scenes.json"
    scenes = json.loads(scenes_path.read_text(encoding="utf-8-sig"))
    scene = scenes["school_row"]
    batches = []
    for image in sorted(images.glob("*.jpg")):
        label = labels / (image.stem + ".txt")
        review = label.with_suffix(".review.json")
        lines = [line.split() for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        records = json.loads(review.read_text(encoding="utf-8-sig"))["objects"]
        if len(lines) != len(records):
            raise ValueError(f"Label/record count mismatch: {image.name}")
        allow, deny = resolve_zones(scene, image.stem)
        with Image.open(image) as opened:
            width, height = opened.size
        objects = []
        for index, (line, record) in enumerate(zip(lines, records), 1):
            box = [int(float(line[0])), *[round(float(v), 6) for v in line[1:]]]
            if len(box) != 5 or record.get("box") != box:
                raise ValueError(f"Label/record box mismatch: {image.name} #{index}")
            if box[0] != 0:
                continue
            contacts = normalized_contacts(record)
            point_verdict, distances = classify_points(contacts, allow, width, height)
            objects.append(dict(id=index, box=box, reference=record["parking_state"],
                                contacts=contacts, contacts_complete=all(c["reviewed"] for c in contacts),
                                point_verdict=point_verdict, distances_px=distances))
        batches.append(dict(image=image, label=label, review=review, allow=allow,
                            deny=deny, width=width, height=height, objects=objects))
    if not batches:
        raise ValueError("No reference images found")
    return scenes_path, batches


def raster(poly, size):
    image = Image.new("1", size, 0)
    if poly:
        ImageDraw.Draw(image).polygon([(x * size[0], y * size[1]) for x, y in poly], fill=1)
    return np.asarray(image, dtype=bool)


def category(row):
    reasons = []
    if not row["contacts_complete"]:
        reasons.append("接地点未确认，需补确认")
    if row["reference"] == "压线" and row["point_verdict"] == "出界":
        reasons.append("压线标签与接地点/边界不一致，待复核")
    if row["reference"] in ("合规", "出界") and row["point_verdict"] in ("合规", "出界"):
        if row["reference"] != row["point_verdict"]:
            reasons.append("人工状态与接地点/边界不一致，待复核")
        elif row["band_verdict"] != row["reference"]:
            reasons.append("条带表示或旧比例规则候选问题，需看掩膜复核")
    if row["band_reason"]:
        reasons.append("掩膜质量检查失败")
    if not reasons and row["point_verdict"] == "无法判定":
        reasons.append("接地点证据不足或不适用；不强行归因")
    return "；".join(reasons) or "本次两方案均与参考一致"


def case_image(rgb, mask, row, allow, out):
    """Show one identical crop twice, with human points and mask/band evidence."""
    h, w = rgb.shape[:2]
    _, cx, cy, bw, bh = row["box"]
    x1, y1, x2, y2 = (cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h
    points = [(c["point"][0]*w, c["point"][1]*h) for c in row["contacts"] if c["point"]]
    xs, ys = [x1, x2]+[p[0] for p in points], [y1, y2]+[p[1] for p in points]
    crop = (max(0, int(min(xs))-50), max(0, int(min(ys))-70),
            min(w, int(max(xs))+50), min(h, int(max(ys))+70))
    band = contact_band(mask, .02)
    modified = rgb.copy()
    modified[mask] = (modified[mask] * .6 + np.array([35, 200, 115]) * .4).astype(np.uint8)
    modified[band] = [240, 0, 220]
    panels = []
    for array in (rgb, modified):
        panel = Image.fromarray(array)
        draw = ImageDraw.Draw(panel)
        poly = [(x*w, y*h) for x,y in allow]
        draw.line(poly + poly[:1], fill="#25a5ff", width=3)
        draw.rectangle((x1,y1,x2,y2), outline="#ffe04b", width=2)
        for j,(x,y) in enumerate(points,1):
            draw.ellipse((x-5,y-5,x+5,y+5), fill="#ff952e", outline="black", width=1)
            draw.text((x+8,y-8),str(j),font=config.load_font(18),fill="#ff952e")
        panels.append(ImageOps.contain(panel.crop(crop),(620,700)))
    sheet = Image.new("RGB",(1280,840),"#f3f3ef")
    draw = ImageDraw.Draw(sheet)
    font = config.load_font(21)
    draw.text((12,8),f"{row['image']} #{row['id']} | 人工状态 {row['reference']} | 条带 {row['band_verdict']} | 人工点 {row['point_verdict']}",font=font,fill="black")
    draw.text((12,40),"左：人工框、接地点与边界 | 右：同图叠加 SAM 掩膜与紫色最低条带",font=font,fill="black")
    draw.text((12,73),"距离(px，正在线内)："+str([round(d,2) if d is not None else None for d in row['distances_px']]),font=font,fill="black")
    for i,panel in enumerate(panels):
        sheet.paste(panel,(i*640+(640-panel.width)//2,115))
    sheet.save(out,quality=93)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=config.DATA_ROOT/"campus/used/annotation_work/ground_truth_v1")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        parser.error("Output already exists; choose a new experiment directory")
    scenes_path,batches=read_batch(args.source)
    args.out.mkdir(parents=True)
    (args.out/"cases").mkdir()
    (args.out/"masks").mkdir()
    manifest=[]
    for path in [scenes_path,*[p for b in batches for p in (b["image"],b["label"],b["review"])]]:
        manifest.append(dict(path=str(path.resolve()),sha256=digest(path),bytes=path.stat().st_size))
        if path.suffix.lower() != ".jpg":
            target=args.out/"input_snapshot"/path.relative_to(args.source)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
    dump(args.out/"inputs.json",manifest)
    audit=dict(images=len(batches),objects=sum(len(b["objects"]) for b in batches),
               reference_states=dict(Counter(o["reference"] for b in batches for o in b["objects"])),
               contact_visibility=dict(Counter(c["visibility"] for b in batches for o in b["objects"] for c in o["contacts"])),
               unconfirmed_objects=[dict(image=b["image"].name,id=o["id"]) for b in batches for o in b["objects"] if not o["contacts_complete"]])
    dump(args.out/"audit.json",audit)
    print("AUDIT",json.dumps(audit,ensure_ascii=False),flush=True)
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment expects the existing CUDA environment")
    model=build_sam2(config.SAM2_CONFIG,str(config.SAM2_CHECKPOINT),device="cuda")
    predictor=SAM2ImagePredictor(model)
    all_rows=[]
    with torch.inference_mode():
        for batch in batches:
            started=time.perf_counter()
            with Image.open(batch["image"]) as opened:
                image=opened.convert("RGB")
            image.thumbnail((1400,1900))
            rgb=np.array(image)
            w,h=image.size
            predictor.set_image(rgb)
            allow=raster(batch["allow"],(w,h))
            deny=np.zeros((h,w),dtype=bool)
            for poly in batch["deny"]:
                deny |= raster(poly,(w,h))
            masks=[]
            for obj in batch["objects"]:
                _,cx,cy,bw,bh=obj["box"]
                prompt=np.asarray([(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h])
                predicted,scores,_=predictor.predict(box=prompt,multimask_output=False)
                mask=np.asarray(predicted).reshape(-1,h,w)[0]>0
                masks.append(mask)
                score=float(np.asarray(scores).ravel()[0])
                result=classify_mask(mask,allow,deny,has_zone=bool(batch["allow"]),score=score,
                                     min_area=200,min_score=0,contact_ratio=.02,deny_tol=.05)
                row={**obj,"image":batch["image"].name,"band_verdict":result["verdict"],
                     "band_in":result["contact_in"],"band_reason":result["reason"],
                     "sam_score":score,"mask_area":int(mask.sum()),"processed_size":[w,h]}
                row["paired_eligible"]=(row["contacts_complete"] and row["reference"] in ("合规","出界")
                                        and row["point_verdict"] in ("合规","出界"))
                row["attribution_candidate"]=category(row)
                needs_case=(row["band_verdict"]!=row["reference"] or not row["contacts_complete"]
                            or row["reference"]=="压线" or row["point_verdict"] in ("合规","出界")
                            and row["point_verdict"]!=row["reference"])
                if needs_case:
                    row["case_image"]=f"cases/{batch['image'].stem}_{obj['id']:03d}.jpg"
                    case_image(rgb,mask,row,batch["allow"],args.out/row["case_image"])
                else:
                    row["case_image"]=""
                all_rows.append(row)
            np.savez_compressed(args.out/"masks"/(batch["image"].stem+".npz"),masks=np.stack(masks))
            print(batch["image"].name,len(masks),"objects",round(time.perf_counter()-started,2),"seconds",flush=True)
    paired=[r for r in all_rows if r["paired_eligible"]]
    summary=dict(audit=audit,point_outcomes=dict(Counter(r["point_verdict"] for r in all_rows)),
                 band_outcomes=dict(Counter(r["band_verdict"] for r in all_rows)),
                 paired_comparison={"band":pair_metrics(paired,"band_verdict"),"human_points":pair_metrics(paired,"point_verdict")},
                 paired_ids=[dict(image=r["image"],id=r["id"]) for r in paired],
                 paired_corrections=[dict(image=r["image"],id=r["id"]) for r in paired
                                     if r["band_verdict"]!=r["reference"] and r["point_verdict"]==r["reference"]],
                 paired_regressions=[dict(image=r["image"],id=r["id"]) for r in paired
                                     if r["band_verdict"]==r["reference"] and r["point_verdict"]!=r["reference"]],
                 parameters=dict(manual_boxes=True,manual_zones=True,no_detector=True,no_training=True,
                                 independent_test=False,sam_config=config.SAM2_CONFIG,
                                 sam_checkpoint=str(config.SAM2_CHECKPOINT),sam_checkpoint_sha256=digest(config.SAM2_CHECKPOINT),
                                 torch_version=torch.__version__,source_limit=[1400,1900],contact_ratio=.02,
                                 min_mask_area=200,min_sam_score=0,deny_tol=.05,
                                 point_boundary="exact signed distance; no tuned pixel tolerance",
                                 scope="Wheel/parking-boundary subtask. Body overhang not evaluated; no deny zones provided.",
                                 caveat="Manual points plus point-rule vs legacy band-rule; not a learned keypoint model or end-to-end accuracy."))
    # Evidence was frozen before inference. Detect concurrent changes rather than
    # silently claiming that mutable inputs still equal this experiment's inputs.
    summary["source_inputs_unchanged"]=all(digest(item["path"])==item["sha256"] for item in manifest)
    dump(args.out/"results.json",all_rows)
    dump(args.out/"summary.json",summary)
    fields=["图片","车辆编号","人工状态","条带结果","人工点结果","接地点确认完成","纳入成对对照", "点1状态","点2状态",
            "点1边界距离_px","点2边界距离_px","条带在线内比例","归因候选","局部对照图"]
    with (args.out/"per_bike.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in all_rows:
            writer.writerow(dict(zip(fields,[row["image"],row["id"],row["reference"],row["band_verdict"],row["point_verdict"],
                row["contacts_complete"],row["paired_eligible"],*[c["visibility"] for c in row["contacts"]],
                *[round(d,3) if d is not None else "" for d in row["distances_px"]],round(row["band_in"],6),
                row["attribution_candidate"],row["case_image"]])))
    print("SUMMARY",json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
