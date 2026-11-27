"""Build the dated development report and targeted review queue from frozen results."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/contact_eval_20261127_v1"
ROWS = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
SUMMARY = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))


def case_path(row):
    name = f"cases/{Path(row['image']).stem}_{row['id']:03d}.jpg"
    return name if (OUT / name).is_file() else ""


def main():
    queue = []
    for row in ROWS:
        reasons = []
        priority = 2
        if not row["contacts_complete"]:
            reasons.append("旧版接地点确认字段缺失；只影响点位实验的完成标记，不否定停车状态；看不清时不补猜")
        if row["reference"] == "压线" and row["point_verdict"] == "出界":
            reasons.append("人工压线与点位几何出界采用的证据可能不同；保留停车状态，检查线宽、区域边界和点位定义")
            priority = 1
        if (row["image"],row["id"]) == ("A_1_005.jpg",23):
            reasons.append("局部图中橙色点与车轮/邻车重叠；请确认属于该车的轮胎接地点，而非轮心或遮挡边缘")
            priority = 1
        if reasons:
            queue.append({"优先级":priority,"图片":row["image"],"车辆编号":row["id"],
                          "原人工状态":row["reference"],"需复核内容":"；".join(reasons),
                          "局部对照图":case_path(row)})
    queue.sort(key=lambda r:(r["优先级"],r["图片"],r["车辆编号"]))
    with (OUT / "needs_review.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(queue[0]));writer.writeheader();writer.writerows(queue)
    clarification = {
        "date": "2026-11-27",
        "source": "user clarification in current chat",
        "user_statement": "主要是在密集车辆里，只能看出它有没有违停，看不出接地点；40那张的出界是车倒了，然后后轮越界",
        "parking_state_and_contact_visibility_are_separate": True,
        "annotations_modified": False,
        "historical_metrics_recomputed": False,
        "image_level_event": {"image": "A_1_040.jpg", "event": "倒车后轮越界",
                              "object_id": None, "out_of_bounds_candidate_ids_in_snapshot": [10, 28],
                              "note": "用户说明到图片级，未擅自对应到某辆车；不将轮胎轮廓点直接解释为地面接触点"},
    }
    (OUT / "interpretation_update.json").write_text(json.dumps(clarification,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    band=SUMMARY["paired_comparison"]["band"]
    points=SUMMARY["paired_comparison"]["human_points"]
    assert band["n"] == points["n"] == 55
    incomplete=SUMMARY["audit"]["unconfirmed_objects"]
    assert len(incomplete) == 18
    lines=[
        "# 接地点对照实验：2026-11-27",
        "",
        "修订结论：人工停车状态与接地点可见性是两项独立标注。密集遮挡时可以判断是否违停，却看不清轮胎接地点；倒车后轮越界也可以构成用户定义的出界证据。不能因为接地点看不清或不适用，就要求把停车状态改为无法判定。",
        "",
        "用户补充：A_1_040 的出界涉及倒车后轮越界。快照中该图有两个出界目标（10、28），当前只记录图片级说明，不擅自认定具体编号。此前将‘接地点不适用但状态有结论’列为冲突的解释撤回；4 辆相应目标已从点位复核清单移除。本次未修改人工标注，也没有重新推理或训练模型。",
        "",
        "历史计算中，筛出的 55 辆最低条带与人工状态一致 51 辆，人工点规则一致 55 辆。这只是原筛选条件下的几何诊断结果；没有区分倒车、可见轮廓和真实落地点，不能据此认定接地点方案覆盖完整违停任务，更不能报告为自动模型准确率提升。",
        "",
        "## 数据核验",
        "",
        "| 项目 | 数量 |",
        "|---|---:|",
        "| 图片 | 5 |",
        "| 对应成功的车辆框和人工状态 | 157 |",
        "| 人工状态：合规 / 出界 / 压线 / 无法判定 | 144 / 7 / 1 / 5 |",
        "| 接地点：可见 / 看不清 / 旧版未标 / 不适用 | 124 / 156 / 26 / 8 |",
        "| 含旧版未确认接地点的车辆 | 18 |",
        "| 人工点几何判断：线内 / 线外 / 无法判定 | 50 / 6 / 101 |",
        "| 历史几何筛选纳入二分类对照 | 55（线内 50、出界 5；尚未按停车形态复核适用性） |",
        "",
        "26 个旧版未标点分布在 18 辆车中，不能按每车两个点简单折算成 13 辆。看不清的点没有补猜，未确认记录没有当成已经完成。所有框与同序号停车状态均能匹配；这只证明文件关联正确，不证明所有人工标签已独立复核。",
        "",
        "## 实验设置",
        "",
        "两方案使用同一批人工车辆框和人工停车区域。A 用人工框提示 SAM2.1 Tiny，取目标掩膜最低 2% 高度的条带，沿用 0.1/0.9 比例规则；B 使用人工标出的无序点，任意可见点明确在线外即输出出界，两点明确在线内才输出线内。倒车/离地或点位证据不足时，B 返回‘无法判定’表示这个方法无法作答，不代表人工停车状态无法判定。只有点确实对应适用的轮胎接地位置时，才能将 B 的结果解释为接地点几何结论。",
        "",
        "SAM 图像上限为 1400×1900，最小掩膜面积 200，质量分下限 0，禁停重叠阈值 0.05。本批 5 张图没有标定禁停区；此次实现只计算条带/点与区域的关系，没有实现倒车轮廓越界或车身占道判定。区域与人工点在原图坐标中计算带符号距离，正为线内、负为线外，没有为本批调像素容差。二维车身或车轮轮廓投影也不能一律等同于地面占用。",
        "",
        "本轮固定人工框，没有评估 YOLO 漏检、误检或重复框。没有掩膜真值，分割粘连只能列为候选原因，不能自动认定。",
        "",
        "## 全部车辆的最低条带基线",
        "",
        "先保留完整停车状态任务的结果，避免仅观察点位方法能够作答的目标：",
        "",
        "| 人工状态 | 车辆数 | 条带判合规 | 条带判压线 | 条带判出界 |",
        "|---|---:|---:|---:|---:|",
        *[f"| {state} | {sum(r['reference']==state for r in ROWS)} | " + " | ".join(
            str(sum(r['reference']==state and r['band_verdict']==pred for r in ROWS))
            for pred in ("合规","压线","出界")) + " |"
          for state in ("合规","压线","出界","无法判定")],
        "",
        "按旧逻辑将压线和出界都视为报警：8 辆人工出界/压线中有 7 辆报警、1 辆被判合规；144 辆人工合规中有 6 辆报警。人工无法判定的 5 辆不能自动算成误报。这是固定人工框和人工区域的开发基线，未包含检测错误，也未按遮挡/倒车分组。",
        "",
        "## 历史几何对照结果（适用范围待分组）",
        "",
        "| 人工参考 | 数量 | 最低条带 | 人工接地点 |",
        "|---|---:|---|---|",
        "| 合规 | 50 | 合规 49，压线 1 | 线内 50 |",
        "| 出界 | 5 | 出界 2，压线 3 | 出界 5 |",
        "| 状态一致数量 | 55 | 51 | 55 |",
        "",
        "这里只比较状态是否一致。现有旧报警逻辑也会对‘压线’报警，因此 3 辆被判为压线不等于 3 辆完全没有触发报警。接地点与标签来自同一批人工标注，结果存在同源性；本次还同时改变了接地点表示和规则表达，不能归结为某个关键点网络提升，也不能把 55/55 称为完整违停准确率。",
        "",
        "157 辆中接地点方法有 101 辆无法作答，其中人工状态为合规 94 辆、出界 2 辆、无法判定 5 辆。前 96 辆应保留人工状态，作为方法覆盖不足的证据；只有最后 5 辆人工本身标为无法判定。另有 1 辆人工压线而点位规则输出出界，需要检查证据定义，不能自动认定标签错误。历史成对对照只有 55 辆，其中出界 5 辆；真实接地点的适用子集须区分倒车等情形后再统计，不能直接沿用 55 作为分母。",
        "",
        "## 差异与归因候选",
        "",
        "| 图片 / 车辆编号 | 人工状态 | 条带 | 人工点 | 当前解释 |",
        "|---|---|---|---|---|",
        "| A_1_005 / 23 | 出界 | 压线 | 出界 | 条带在线内占比 11.1%；但人工点是否落在正确轮胎落地处还需复核，此例先保留为候选改善 |",
        "| A_1_011 / 1 | 合规 | 压线 | 线内 | 条带在线内占比 31.0%，最低掩膜像素与人工落地点位置不同；需检查掩膜归属 |",
        "| A_1_040 / 28 | 出界 | 压线 | 出界 | 标点在边界外约 16.5 原图像素；条带在线内占比 29.4%。用户补充该图有倒车后轮越界，尚未对应具体目标；此例不能直接作为真实接地点方案改善的证据 |",
        "| A_2_006 / 15 | 出界 | 压线 | 出界 | 可见点在所画边界外约 19.9 原图像素；条带在线内占比 71.1%，侧向出界没有被判成出界 |",
        "",
        "下图蓝线为已标区域边界，橙点为人工接地点，右图绿色为 SAM 掩膜、紫色为最低条带；黄色框是本次固定的人工提示框。",
        "",
    ]
    for stem,index in [("A_1_005",23),("A_1_011",1),("A_1_040",28),("A_2_006",15)]:
        lines += [f"![{stem} #{index} 局部对照](../runs/contact_eval_20261127_v1/cases/{stem}_{index:03d}.jpg)",""]
    lines += [
        "## 标注保留，点位实验的可选检查",
        "",
        "不需要整批重标，也不再要求因为倒车或看不清接地点而更改停车状态。下面两项只影响点位几何实验的解释，不是修改人工停车状态的指令：",
        "",
        "| 图片 / 编号 | 要核对的事情 |",
        "|---|---|",
        "| A_1_005 / 1 | 保留人工压线。一个点在所画多边形外约 25.4 原图像素，检查实际线宽、区域边界和点位定义是否一致 |",
        "| A_1_005 / 23 | 核对橙点是否属于该车的轮胎落地处，特别注意邻车遮挡；看不清就保留看不清 |",
        "",
        "快照还保留 18 辆旧版接地点确认字段缺失的记录。它们不是停车状态未标，也不阻碍分析遮挡场景；若以后整理点位训练集，再确认能看清的点，看不清的保留看不清。不要为了凑训练样本猜点。编号来自本次快照；删框或重排后应以框位置和局部图核对。",
        "",
        "| 图片 | 待确认车辆编号 |",
        "|---|---|",
    ]
    grouped={}
    for item in incomplete:
        grouped.setdefault(item["image"],[]).append(item["id"])
    lines += [f"| {name} | {', '.join(map(str,ids))} |" for name,ids in grouped.items()]
    lines += [
        "",
        "重点分歧示例：",
        "",
        "![A_1_005 #1：人工压线与所画边界存在差异](../runs/contact_eval_20261127_v1/cases/A_1_005_001.jpg)",
        "",
        "## 接下来的实验顺序",
        "",
        "1. 保留这 5 张开发评估图及全部人工停车状态，将‘接地点是否可见/适用’与‘停车状态是否明确’分开统计。已有记录可直接使用；无法从记录确定的停车形态先留待确认，不自动推断。",
        "2. 将评估证据区分为：直立且接地点可见；密集遮挡但停车状态明确；倒车/离地且存在车轮或车体越界证据。无法明确归类的目标保留未知分组，不混入真实接地点子集。A_1_040 的补充说明先按图片级保存。",
        "3. 保留最低条带基线，先检查它对全部 157 辆人工状态的输出；分别报告状态一致、出界/压线报警以及无法作答情况。点位方案只在适用子集比较几何判断，同时报告它覆盖了多少目标，不能把遮挡目标剔除后只看准确率。",
        "4. 模型方向采用多种证据：适用时用接地点辅助几何判断；遮挡和倒车场景另行验证‘目标车辆局部 + 周围停车边界’的状态判断。局部图须保留必要边界上下文，并检查邻车干扰。后者是待验证方案，尚无训练结果；不能仅把整车掩膜与区域的重叠当成地面越界真值。",
        "5. 分组和判据明确后再从其他图片准备训练数据，按场景/拍摄批次隔离训练与评估。无需继续把所有车都标成两个接地点，也无需区分前后轮；这 5 张继续作为开发集，不用来宣称独立测试成绩。",
        "",
        "## 文件与复现",
        "",
        "- [逐车结果与归因候选 CSV](../runs/contact_eval_20261127_v1/per_bike.csv)",
        "- [20 辆点位记录检查清单 CSV（18 辆旧版字段 + 2 项几何检查）](../runs/contact_eval_20261127_v1/needs_review.csv)",
        "- [用户补充与解释修订记录](../runs/contact_eval_20261127_v1/interpretation_update.json)",
        "- [机器可读统计](../runs/contact_eval_20261127_v1/summary.json)",
        "- [输入文件大小与 SHA-256](../runs/contact_eval_20261127_v1/inputs.json)",
        "- `input_snapshot/` 保存当时的框、状态、接地点和区域；`masks/` 保存逐图 SAM 掩膜。输入哈希在本次推理结束时全部未变。",
        "- `evaluate_contacts_source.py` 是本轮运行源码快照；仓库脚本后续仅将 NumPy 输入显式复制为可写数组，避免只读数组警告。",
        "",
        "```powershell",
        "# 新实验须换一个未存在的输出目录；读取人工框和接地点，不重训模型。",
        "& 'C:/Users/26803/miniconda3/envs/sam2/python.exe' experiments/20261127/evaluate_contacts.py --out runs/contact_eval_20261127_v2",
        "```",
        "",
        "CPU 检查已覆盖无序点交换、部分遮挡、单点出界、倒车不适用、边界、缺区域、非有限坐标与指标分母；GPU 完成 5 图 157 次人工框提示分割。所有判定及阈值与复核清单分别保存，没有自动改写人工答案。",
    ]
    assert len(queue) == 20
    assert Counter(r["reference"] for r in ROWS if r["point_verdict"]=="无法判定") == {"合规":94,"出界":2,"无法判定":5}
    assert sum(r["reference"] in ("压线","出界") and r["band_verdict"] in ("压线","出界") for r in ROWS) == 7
    assert sum(r["reference"]=="合规" and r["band_verdict"] in ("压线","出界") for r in ROWS) == 6
    (ROOT / "docs/接地点对照实验_2026-11-27.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    provenance={"run_source_sha256":hashlib.sha256((OUT/"evaluate_contacts_source.py").read_bytes()).hexdigest(),
                "paired_rows":55,"targeted_review_rows":len(queue),"unconfirmed_objects":len(incomplete),
                "visual_review":"Four paired disagreements, boundary geometry difference, and two not-applicable cases inspected; attributions remain candidates.",
                "interpretation_update":"Parking state is independent of contact visibility. Not-applicable contacts do not invalidate reference labels. A_1_040 fallen rear-wheel crossing recorded at image level; exact object unspecified."}
    (OUT/"review_provenance.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(provenance,ensure_ascii=False))


if __name__=="__main__":
    main()
