"""Replay frozen SAM masks; compare candidate evidence without new annotation.

Run with the existing sam2 Python environment. No GPU, fitting or download is
needed. Input labels, masks, original images and the default detector stay intact.
"""
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config.py").is_file())
sys.path.insert(0, str(ROOT))
import config
config.activate()

import argparse
from collections import Counter
import csv
import hashlib
import json
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from boundary_candidates import candidate_predictions, lower_envelope, ownership_diagnostics
from violation_rules import BAD_VERDICTS, classify_mask, contact_band

NAMES = {"band": "原最低条带", "column_envelope": "逐列下边缘",
         "silhouette_2pct": "整车轮廓区外超过2%", "band_with_review": "原条带+掩膜归属检查"}
KNOWN = {"合规", "压线", "出界", "占道"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def raster(poly, width, height):
    image = Image.new("1", (width, height))
    if poly:
        ImageDraw.Draw(image).polygon([(x*width, y*height) for x, y in poly], fill=1)
    return np.asarray(image, dtype=bool)


def metrics(rows, method):
    known = [r for r in rows if r["reference"] in KNOWN]
    bad = [r for r in known if r["reference"] in BAD_VERDICTS]
    good = [r for r in known if r["reference"] == "合规"]
    return dict(n=len(known), positive=len(bad), negative=len(good),
                tp=sum(r["predictions"][method] in BAD_VERDICTS for r in bad),
                fn=sum(r["predictions"][method] == "合规" for r in bad),
                unresolved_positive=sum(r["predictions"][method] not in KNOWN for r in bad),
                fp=sum(r["predictions"][method] in BAD_VERDICTS for r in good),
                unresolved_negative=sum(r["predictions"][method] not in KNOWN for r in good),
                resolved=sum(r["predictions"][method] in KNOWN for r in known),
                exact_agreement=sum(r["predictions"][method] == r["reference"] for r in known),
                reference_unknown_outputs=dict(Counter(r["predictions"][method] for r in rows if r["reference"] not in KNOWN)))


def acceptance(candidate, baseline, by_image, method):
    # Abstention is not a corrected compliant prediction; every known sample
    # remains in the denominator. A strict development gate, not test accuracy.
    return bool(candidate["resolved"] == candidate["n"]
                and candidate["tp"] >= baseline["tp"] and candidate["fp"] <= baseline["fp"]
                and (candidate["tp"] > baseline["tp"] or candidate["fp"] < baseline["fp"])
                and all(s[method]["tp"] >= s["band"]["tp"]
                        and s[method]["fp"] <= s["band"]["fp"] for s in by_image.values()))


def comparison_image(rgb, mask, allow_poly, row, destination):
    h, w = mask.shape
    _, cx, cy, bw, bh = row["box"]
    ys, xs = np.nonzero(mask)
    crop = (max(0, int(min((cx-bw/2)*w, xs.min()))-45),
            max(0, int(min((cy-bh/2)*h, ys.min()))-45),
            min(w, int(max((cx+bw/2)*w, xs.max()))+46),
            min(h, int(max((cy+bh/2)*h, ys.max()))+46))
    sheet = Image.new("RGB", (1800, 710), "#f4f3ef")
    draw = ImageDraw.Draw(sheet)
    font = config.load_font(23)
    draw.text((18, 12), f"{row['image']} #{row['id']} | 人工：{row['reference']} | 掩膜共享 IoU：{row['max_mask_iou']:.2f}", font=font, fill="black")
    panels = [("原图与区域", None),
              ("原最低条带 → "+row["predictions"]["band"], contact_band(mask)),
              ("逐列下边缘 → "+row["predictions"]["column_envelope"], lower_envelope(mask))]
    for i, (title, evidence) in enumerate(panels):
        array = rgb.copy()
        if evidence is not None:
            array[mask] = (.65*array[mask]+.35*np.array([25,200,115])).astype(np.uint8)
            array[evidence] = [235, 0, 215]
        image = Image.fromarray(array)
        d = ImageDraw.Draw(image)
        poly = [(x*w,y*h) for x,y in allow_poly]
        if poly:
            d.line(poly+poly[:1],fill="#2299ff",width=3)
        d.rectangle(((cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h),outline="#ffcf30",width=2)
        panel=ImageOps.contain(image.crop(crop),(580,600))
        draw.text((i*600+12,55), title, font=font, fill="black")
        sheet.paste(panel,(i*600+(600-panel.width)//2,100))
    sheet.save(destination,quality=93)


def report(out, rows, summary):
    totals=summary["metrics"]
    lines=["# 固定掩膜的越界判定对照：2026-11-27", "",
           "## 本轮改了什么", "",
           "本轮新增可复现的离线候选规则实验，不改变默认检测程序。复用上一轮全部 5 图、157 辆的人工框、人工停车区域和 SAM 缓存；不训练模型、不新增标注，不把人工接地点作为算法输入。停车状态明确的 152 辆用于计分，人工无法判定的 5 辆只列输出。", "",
           "三个待验证方向：逐列下边缘能否弥补全局最低条带的缺失；整车轮廓越界能否提供倒车证据；掩膜归属检查能否识别不可靠输出。全部是本项目假设，不是复现论文网络。", "",
           "## 论文依据及适用限制", "",
           "- Kim 等（2025），第 3.1–3.2 节、图 3–4：自行车位置表示采用车轮与地面的接触点。它说明接地语义需要明确；没有证明掩膜最低点就是接地点，也没有证明本文候选方法适用于密集倒车。见 [本地论文](../docs/papers/01_2025_Road_User_Ground_Contact_Keypoints.pdf)。",
           "- HPS-Net（2023），第 III.B 节：IPM 基于地面平面假设，车位表示来自环视俯视图。借鉴区域表示，不能把校园斜拍图中整个车身的投影直接当作地面占用。见 [本地论文](../docs/papers/03_2023_HPS-Net_Polygon_Representations.pdf)。",
           "- 盛宣华等（2019），第 2 节、图 2：包含视频停留状态的时序统计。静态照片没有这些时序证据，本轮不引入停留时长或照搬其检测 mAP 作为违停指标。见 [本地论文](../docs/papers/06_2019_Shared_Bicycle_Illegal_Parking_YOLOv3.pdf)。", "",
           "上述链接在本报告位于 runs 子目录时需通过文末源目录访问；论文页码与具体证据另存于 protocol.json。", "",
           "## 预先固定的候选规则", "",
           "| 方法 | 计算方式 | 限制 |", "|---|---|---|",
           "| 原最低条带 | 掩膜整体最底部 2% 高度；线内比例 0.1/0.9 分档 | 可能只取到一个轮子或邻车 |",
           "| 逐列下边缘 | 每个非空列各取最低薄层，厚度仍按目标高度 2%；沿用 0.1/0.9 | 只是轮廓，不是车轮或接地点检测器 |",
           "| 整车轮廓区外超过2% | 整个掩膜在区外的像素比例 >0.02 则报警 | 沿用已有框对照的 2% 容差作控制实验；不声称可区分压线/出界，可能把投影当占地 |",
           "| 原条带+掩膜归属检查 | 最大逐车掩膜 IoU≥0.8 或掩膜落在提示框外比例>0.05，输出无法判定 | 这两阈值是探索性工程值；拒答不能算正确合规 |", "",
           "阈值在候选全量计分前写入 protocol.json，本轮没有参数网格搜索。但此前已检查这批图的基线错误与相关诊断值，因此整个实验属于开发分析，不是盲测或预注册研究。", "",
           "## 全量结果", "",
           "报警包含出界、压线、占道；漏报指人工违停却输出合规，拒答另列。", "",
           "| 方法 | 检出违停 / 8 | 漏报 | 合规误报 / 144 | 已知状态中拒答 | 自动作答 / 152 | 三类状态一致 / 152 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name,title in NAMES.items():
        m=totals[name]
        lines.append(f"| {title} | {m['tp']} | {m['fn']} | {m['fp']} | {152-m['resolved']} | {m['resolved']} | {m['exact_agreement']} |")
    lines += ["", "整车轮廓控制只输出二分类，不具备压线/出界细分类能力；其‘三类状态一致’仅供完整记账，不能据此宣称细分类改善。掩膜归属检查的拒答可能减少错误报警，但同时增加人工处理量，不能将拒答记成纠正。", "",
              "## 逐图报警结果", "",
              "表中为 检出违停/误报/拒答，分母与全量一致，拒答只计人工状态明确的车辆。", "",
              "| 图片 | 人工违停/合规 | 原条带 | 逐列下边缘 | 整车轮廓 | 归属检查 |", "|---|---:|---|---|---|---|"]
    for image,values in summary["by_image"].items():
        b=values["band"]
        cells=[f"{values[n]['tp']}/{values[n]['fp']}/{values[n]['n']-values[n]['resolved']}" for n in NAMES]
        lines.append(f"| {image} | {b['positive']}/{b['negative']} | "+" | ".join(cells)+" |")
    lines += ["", "## 接入默认程序的判据与结论", "",
              "本轮开发门槛：所有已知状态目标均自动作答；总检出不降低、总误报不增加且至少一项改善；每张图均不增加误报、不减少检出。通过也只表示值得继续验证，不能证明独立场景泛化。", ""]
    accepted=[NAMES[n] for n,v in summary["development_gate"].items() if v]
    if accepted:
        lines += ["通过本轮开发门槛："+"、".join(accepted)+"。尚未通过独立数据验证，保留为实验候选，不替换默认程序。"]
    else:
        lines += ["没有候选通过上述门槛，默认程序保持原规则。这是一次有结论的消融实验：不能仅凭局部改善就替换默认方法。"]
    lines += ["", "## 错误机制与下一步", "",
              "- 唯一原始报警漏报 A_1_005 #21 是倒车：原条带在线内比例为 1，但整个掩膜有约 6.4% 在线外。这直接说明全局最底部不能代表用户要求的倒车越界。",
              "- A_1_040 #24/#25 的掩膜 IoU 约 0.83，#29/#30 约 0.94；配合局部图，提示逐车掩膜归属不稳定。它不是独立分割真值，不能断言每个像素错误。",
              "- A_1_011 #1 的原条带在线内比例约 31%，但整体掩膜仅约 0.09% 在区外：极少量底部像素可以支配旧规则。",
              "- 不使用人工点补预测、不因看不清接地点修改停车状态，也不根据车框长宽比擅自判倒车。A_1_040 的用户补充仍只记录到图片级，不猜具体目标编号。", "",
              "后续应先验证实例分割归属或带边界上下文的停车状态表示；仅扩展几何采样不足以保证密集场景正确。优先复用已有车辆框与停车状态，并按整图/拍摄批次隔离评估。当前只有 8 个已知违停，不宜马上训练复杂网络或继续堆阈值。用户无需重做本批标注。", "",
              "## 可复现文件", "",
              "- [逐车预测与变化](per_bike.csv)", "- [机器可读统计](summary.json)",
              "- [固定协议与文献证据](protocol.json)", "- [输入 SHA-256](inputs.json)",
              "- [旧方法错误的三栏对照](cases/)", "",
              "运行：`python experiments/20261127/evaluate_boundary_candidates.py --out runs/boundary_ablation_20261127_v2`。必须使用未存在的输出目录。原始标注、人工点和缓存只读；源码与规则文件在本输出目录留有快照。"]
    text="\n".join(lines)+"\n"
    # Report lives two levels below the workspace, not one level below it.
    text=text.replace("(../docs/papers/", "(../../docs/papers/")
    text=text.replace("上述链接在本报告位于 runs 子目录时需通过文末源目录访问；论文页码与具体证据另存于 protocol.json。", "论文页码与具体证据另存于 protocol.json。本轮没有复现上述论文网络或沿用其效果指标。")
    (out/"report.md").write_text(text,encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,default=ROOT/"runs/contact_eval_20261127_v1")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        parser.error("Output exists; use a new directory")
    base=args.baseline
    source_rows=json.loads((base/"results.json").read_text(encoding="utf-8"))
    scenes_path=base/"input_snapshot/image/scenes.json"
    scenes=json.loads(scenes_path.read_text(encoding="utf-8"))["school_row"]
    original_inputs=json.loads((base/"inputs.json").read_text(encoding="utf-8"))
    image_sources={Path(i["path"]).name:Path(i["path"]) for i in original_inputs if Path(i["path"]).suffix==".jpg"}
    source_paths=[base/"results.json",base/"summary.json",scenes_path,
                  Path(__file__),Path(__file__).with_name("boundary_candidates.py"),
                  ROOT/"relabel/violation_rules.py",
                  *sorted((base/"masks").glob("*.npz")),
                  *sorted((base/"input_snapshot").rglob("*.txt")),
                  *sorted((base/"input_snapshot").rglob("*.review.json")),
                  *[ROOT/"docs/papers"/n for n in ("01_2025_Road_User_Ground_Contact_Keypoints.pdf",
                      "03_2023_HPS-Net_Polygon_Representations.pdf","06_2019_Shared_Bicycle_Illegal_Parking_YOLOv3.pdf")],
                  *image_sources.values()]
    manifest=[dict(path=str(p.resolve()),sha256=digest(p),bytes=p.stat().st_size) for p in source_paths]
    for item in original_inputs:
        if Path(item["path"]).suffix==".jpg" and digest(item["path"])!=item["sha256"]:
            raise ValueError("Original image changed: "+item["path"])
    args.out.mkdir(parents=True)
    (args.out/"cases").mkdir()
    dump(args.out/"inputs.json",manifest)
    dump(args.out/"protocol.json",dict(
        date="2026-11-27",stage="development ablation after baseline error inspection; not preregistered/independent test",
        human_contacts_used_for_prediction=False,new_annotation=False,training=False,
        frozen_manual_boxes=True,frozen_masks=True,frozen_zones=True,
        candidates=dict(band=dict(height_ratio=.02,inside_thresholds=[.1,.9]),
                        column_envelope=dict(height_ratio=.02,inside_thresholds=[.1,.9]),
                        silhouette_2pct=dict(outside_threshold=.02,source="existing box baseline tolerance; silhouette is not ground footprint"),
                        band_with_review=dict(max_mask_iou=.8,box_leak=.05,source="exploratory ownership thresholds, not literature values")),
        development_gate="No abstention on known labels, no image loses TP or adds FP; aggregate TP >= baseline and FP <= baseline, at least one strict improvement.",
        literature=[dict(file="01_2025_Road_User_Ground_Contact_Keypoints.pdf",pages=[5,6],sections="3.1-3.2",use="Wheel-ground semantics, not silhouette proxy"),
                    dict(file="03_2023_HPS-Net_Polygon_Representations.pdf",pages=[3],section="III.B",use="IPM flat-ground assumption and projection caveat"),
                    dict(file="06_2019_Shared_Bicycle_Illegal_Parking_YOLOv3.pdf",pages=[3,4],section="2 and Figure 2",use="Video temporal parking state is separate from static boundary task")]))
    for path in (Path(__file__),Path(__file__).with_name("boundary_candidates.py"),ROOT/"relabel/violation_rules.py"):
        shutil.copy2(path,args.out/(path.stem+"_source.py"))
    rows=[]
    for image in sorted(image_sources):
        group=[r for r in source_rows if r["image"]==image]
        with np.load(base/"masks"/(Path(image).stem+".npz")) as archive:
            masks=archive["masks"]
        assert masks.shape[0]==len(group)
        # The baseline stored masks in object order. Never assume id-1 if class
        # filtering or sparse IDs are introduced in later experiments.
        h,w=masks.shape[1:]
        assert all(r["processed_size"]==[w,h] for r in group)
        zone=scenes.get("images",{}).get(Path(image).stem,{})
        allow_poly=zone.get("allow",scenes.get("allow",[]))
        allow=raster(allow_poly,w,h)
        deny=np.zeros_like(allow)
        for poly in zone.get("deny",scenes.get("deny",[])):
            deny |= raster(poly,w,h)
        diagnostics=ownership_diagnostics(masks,[r["box"] for r in group])
        with Image.open(image_sources[image]) as opened:
            rendered=opened.convert("RGB")
        rendered.thumbnail((1400,1900))
        assert rendered.size==(w,h)
        rgb=np.asarray(rendered)
        for i,(r,mask,d) in enumerate(zip(group,masks,diagnostics)):
            baseline=classify_mask(mask,allow,deny,has_zone=bool(allow_poly),score=r["sam_score"])
            if baseline["verdict"]!=r["band_verdict"] or abs(baseline["contact_in"]-r["band_in"])>1e-10:
                raise ValueError(f"Baseline replay mismatch: {image} #{r['id']}")
            pred=candidate_predictions(mask,allow,baseline,d)
            neighbor=d.pop("neighbor_index")
            row=dict(image=image,id=r["id"],box=r["box"],reference=r["reference"],
                     predictions=pred,**d,neighbor_id=group[neighbor]["id"] if neighbor is not None else None,
                     silhouette_outside=float(np.count_nonzero(mask & ~allow)/max(1,mask.sum())))
            row["case"]=""
            is_baseline_alarm_error=r["reference"] in KNOWN and ((r["reference"] in BAD_VERDICTS)!=(pred["band"] in BAD_VERDICTS))
            if is_baseline_alarm_error:
                row["case"]=f"cases/{Path(image).stem}_{r['id']:03d}.jpg"
                comparison_image(rgb,mask,allow_poly,row,args.out/row["case"])
            rows.append(row)
        print(image,len(group),"baseline replayed",flush=True)
    by_image={image:{n:metrics([r for r in rows if r["image"]==image],n) for n in NAMES} for image in image_sources}
    totals={n:metrics(rows,n) for n in NAMES}
    assert totals["band"]["tp"]==7 and totals["band"]["fn"]==1 and totals["band"]["fp"]==6
    summary=dict(total_objects=len(rows),reference_unknown=sum(r["reference"] not in KNOWN for r in rows),
                 metrics=totals,by_image=by_image,
                 development_gate={n:acceptance(totals[n],totals["band"],by_image,n) for n in NAMES if n!="band"},
                 changes={n:[dict(image=r["image"],id=r["id"],reference=r["reference"],before=r["predictions"]["band"],after=r["predictions"][n])
                             for r in rows if r["predictions"][n]!=r["predictions"]["band"]] for n in NAMES if n!="band"},
                 inputs_unchanged=all(digest(i["path"])==i["sha256"] for i in manifest),
                 default_pipeline_modified=False,independent_test=False)
    if not summary["inputs_unchanged"]:
        raise RuntimeError("Inputs changed during replay")
    dump(args.out/"results.json",rows)
    dump(args.out/"summary.json",summary)
    fields=["image","id","reference",*NAMES,"box_leak","max_mask_iou","neighbor_id","silhouette_outside","case"]
    with (args.out/"per_bike.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in rows:
            writer.writerow({key:(r["predictions"][key] if key in NAMES else r[key]) for key in fields})
    report(args.out,rows,summary)
    print(json.dumps(dict(metrics=totals,development_gate=summary["development_gate"],inputs_unchanged=summary["inputs_unchanged"]),ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
