"""生成阶段实验报告（Markdown + Word 两份，内容同源）。

用法:
    python tools/build_report.py
"""
import os
import json

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_OUT = os.path.join(BASE, "docs", "实验报告_阶段一.md")
DOCX_OUT = os.path.join(BASE, "docs", "实验报告_阶段一.docx")

TITLE = "校园自行车违停检测阶段一实验报告"
SUBTITLE = "2026-11-22　｜　YOLO11 + SAM2 + 几何判定"

# 内容模型：h1/h2 标题、p 段落、ul 列表、table 表格、img 图片、note 提示
CONTENT = [
    {
        "h1": "一 项目目标与阶段结论"
    },
    {
        "p": "本阶段围绕校园自行车图片违停识别展开实验，完成检测、实例分割、区域几何判定、可视化和逐车 CSV 输出。当前已验证技术链路可以运行，并完成一批影响结果可信度的工程修复；跨场地图片识别、停车区域自动提取及违停召回率仍未完成验证。"
    },
    {
        "p": "近期工作继续聚焦图片：先建立可追溯的评估和失败分析，再验证停车区自动提议与接地点改进。视频接入、报警推送和边缘部署暂不纳入本轮计划。"
    },
    {
        "ul": [
            "输入：单张图片或图片目录，以及人工标定或后续自动提议的允许区与禁停区。",
            "输出：合规、压线、出界、占道、无法判定、未标定六种状态，以及标注图和逐车明细。",
            "判定思路：模型提供车辆和区域的位置证据，几何规则计算空间关系；其可靠性同时依赖检测、掩膜、接地点与区域边界。"
        ]
    },
    {
        "h1": "二 数据与数据治理"
    },
    {
        "table": {
            "header": [
                "数据集",
                "已核对规模",
                "用途与限制"
            ],
            "rows": [
                [
                    "原始 bike_dataset",
                    "553 个文件，153 张唯一图",
                    "按文件字节 MD5 统计；存在大量完全重复文件"
                ],
                [
                    "dataset_v2",
                    "112 张训练、41 张验证；335 个自行车框",
                    "人工精修后的通用自行车数据；相似图片和场景相关性仍需检查"
                ],
                [
                    "dataset_school",
                    "14 张训练、3 张验证；147 个车框、17 个区域标注",
                    "校园样本仅 17 张；区域多边形另存，检测训练使用其外接框"
                ]
            ]
        }
    },
    {
        "p": "原始去重记录包含 495 个属于重复组的文件，保留每组一份后实际多余副本为 400 个（553−153），并有 54 组完全重复图横跨训练与验证。495 不能解释为删除的副本数量。按哈希分组重划分可阻止完全相同文件跨集，但不能自动消除连拍、压缩变体或同场地相近视角带来的相关性。"
    },
    {
        "p": "历史标注对比记录显示，原 2156 个框中约 1.9% 与模型检出目标贴合，85.5% 不重叠。这是与模型候选框的对比结果，不能等同于经人工真值确认的标注准确率。项目已进行预训练模型重标和人工精修，精修后 335 框，保留 15 张空标注负样本。"
    },
    {
        "img": [
            "docs/images/labeler.jpg",
            "图 1 自行车标注精修工具",
            15
        ]
    },
    {
        "h1": "三 当前方法与判定口径"
    },
    {
        "h2": "3.1 自行车检测与分割"
    },
    {
        "p": "检测使用 YOLO11 预训练或微调权重。推理按模型 names 查找唯一 bicycle 类，兼容 COCO 的类别 1 和本项目微调数据的类别 0。SAM2.1 hiera-tiny 以检测框为提示生成车辆掩膜；本阶段未重新训练 SAM，也未证明其在重叠车辆上的分割已经可靠。"
    },
    {
        "p": "YOLO 的 NumPy 图像输入使用 BGR 通道，SAM 的图像输入使用 RGB 通道；两者使用同一缩放尺寸，保证检测框与掩膜坐标一致。"
    },
    {
        "h2": "3.2 接地点近似与几何判定"
    },
    {
        "p": "接地点暂用目标掩膜最低 2% 高度的条带近似，向上取整且至少两行。此次修复把比例基准从整图高度改为目标掩膜高度，减少远近尺度差异对条带厚度的影响。这仍不等同于真实轮胎接地点。"
    },
    {
        "table": {
            "header": [
                "优先级",
                "条件",
                "输出"
            ],
            "rows": [
                [
                    "1",
                    "有标定但掩膜面积不足、分数非有限或低于所设阈值",
                    "无法判定"
                ],
                [
                    "2",
                    "车辆掩膜与禁停区重叠比例 > 0.05",
                    "占道"
                ],
                [
                    "3",
                    "接地点条带落在允许区内的比例 ≥ 0.9",
                    "合规"
                ],
                [
                    "4",
                    "该比例 > 0.1 且 < 0.9",
                    "压线"
                ],
                [
                    "5",
                    "该比例 ≤ 0.1",
                    "出界"
                ],
                [
                    "独立分支",
                    "没有允许区域",
                    "未标定，仅检测"
                ]
            ]
        }
    },
    {
        "p": "阈值沿用现有规则，尚未通过独立验证集优化。最小掩膜面积默认 200 个缩放后像素；质量分数下限默认 0，仅排除非有限分数，可显式调高。SAM 质量分数不应解释为违停概率。"
    },
    {
        "h2": "3.3 停车区域与配置"
    },
    {
        "p": "当前主要通过工具沿地面划线内侧人工绘制多边形。图片配置中缺少 allow 或 deny 键时继承场景默认值；显式 allow=[] 表示未标定，deny=[] 表示清空禁停区。旧配置预先生成的空 allow 条目也遵循此语义，需要继承默认值时应删除该键。"
    },
    {
        "img": [
            "docs/images/zone_tool.jpg",
            "图 2 停车区域标定工具",
            15
        ]
    },
    {
        "p": "SIFT 与 RANSAC 单应迁移、白线吸附目前是独立实验方法，尚未在正式掩膜判定入口中自动串联。场景匹配成功也不保证区域迁移正确；跨视角变化、地面遮挡与参考标定误差仍会影响边界。"
    },
    {
        "h1": "四 已取得的成果与验证"
    },
    {
        "h2": "4.1 历史实验记录及解释边界"
    },
    {
        "table": {
            "header": [
                "实验记录",
                "观察结果",
                "可支持的结论"
            ],
            "rows": [
                [
                    "school_v3 训练 CSV",
                    "共 53 轮；整体 mAP50 最佳为第 5 轮 0.40917；第 37 轮起出现验证损失 NaN",
                    "存在数值异常；降低学习率不能据此认定已解决训练不稳定"
                ],
                [
                    "历史 bicycle mAP50",
                    "旧记录为 0.762，验证集仅 3 张",
                    "需补充对应权重、轮次与逐类验证产物；不能与整体 mAP 混用"
                ],
                [
                    "demo 训练",
                    "20 轮；整体 mAP50 最佳约 0.406",
                    "流程可运行；不足以证明跨场地泛化"
                ],
                [
                    "框规则与掩膜规则",
                    "旧记录分别报警 72 与 32 辆",
                    "报警数量减少 40 辆，约 55.6%；没有逐车真值时不能称为误报下降"
                ],
                [
                    "旧掩膜结果 CSV",
                    "初版 230 合规、21 压线、11 出界；重画区域后 236 合规、13 压线、13 出界",
                    "同为 262 条检测输出；新规则尚未对整批数据重测"
                ]
            ]
        }
    },
    {
        "p": "现有记录将 17 张学校图片描述为规范停放。若每个检出对象均经人工确认且不存在错检，旧输出中 26/262≈9.9% 可作为该批已检出对象的误报警比例。由于缺少逐车真值、检测关联与真实违停样本，这不是完整系统误报率，也不能据此断言区域是唯一瓶颈。"
    },
    {
        "p": "历史场景匹配记录为同场地 17/17、异场地 41/41；区域跨角度迁移曾观察到 5%～10% 图宽的偏移。这些结果来自有限样本，白线吸附后的误差没有独立真值评测，不能报告为普遍精度或保证值。"
    },
    {
        "img": [
            "runs/archive/report_warp_snap/feb3e917-A_002.jpg",
            "图 3 历史区域迁移与白线吸附示例 颜色表示两种区域候选",
            12
        ]
    },
    {
        "p": "历史耗时记录为 YOLO 约 48 ms、SAM 编码约 84 ms、分割约 9 ms/辆，21 辆场景全量约 325 ms。本次修复后未重做性能基准；这些数字仅保留为历史记录，不构成当前版本的性能承诺。"
    },
    {
        "h2": "4.2 本次工程修复成果"
    },
    {
        "table": {
            "header": [
                "修复项",
                "此前风险",
                "当前行为"
            ],
            "rows": [
                [
                    "类别映射",
                    "固定 classes=[1]，微调后可能过滤掉自行车或检测成区域",
                    "按模型名称识别 bicycle，不存在唯一类别则报错"
                ],
                [
                    "颜色通道",
                    "RGB 数组被 YOLO 当成 BGR",
                    "检测与 SAM 分别接收正确通道"
                ],
                [
                    "区域配置回落",
                    "显式空区域被 or 表达式覆盖",
                    "仅缺失键继承，清空配置生效"
                ],
                [
                    "接地点尺度",
                    "按整图高度取条带，目标远近影响不一致",
                    "按车辆掩膜高度计算"
                ],
                [
                    "失败统计",
                    "小掩膜跳过后仍可能计入合规",
                    "每个检测对象保留明细，失败单列为无法判定"
                ],
                [
                    "结果输出",
                    "零检测时可能保留上次 CSV；报警差被称为误报",
                    "成功运行即重写 CSV，新增 reason，输出口径更正"
                ],
                [
                    "训练入口",
                    "参数与项目建议不一致，异常训练缺少明确提示",
                    "默认 lr0=0.002、workers=0、AMP 开启，增加 NaN/Inf 检查"
                ]
            ]
        }
    },
    {
        "p": "训练与判定三个入口的项目内默认路径已改为按脚本位置定位，外部 SAM 权重和设备可通过参数覆盖。数据 YAML、其他脚本和启动器尚未做完整路径迁移。训练仍使用框架默认 auto 优化器，实际学习率需以日志为准；验证轮末检查发生在框架保存之后，异常轮权重可能已经生成，不应继续使用。"
    },
    {
        "h2": "4.3 回归与真实图片验证"
    },
    {
        "ul": [
            "13 项 CPU 回归测试全部通过，覆盖模型类别、RGB/BGR、区域继承与清空、合规/压线/出界阈值、禁停优先级、失败状态、计数、CSV 和训练参数及异常检查。",
            "单张真实图片 GPU 链路验证：使用 runs/school_v3/weights/best.pt 和 SAM2，对 9acffb68-A_006.jpg 检测到 11 个目标；规则输出合规 10、出界 1、无法判定 0，自行车类别正确识别为 0。",
            "上述单图验证证明当前入口与微调权重可以完成检测、分割和输出，不代表 11 个目标全部正确，也不证明这 1 次报警正确。此次没有重训模型、增加数据或修改标注。"
        ]
    },
    {
        "img": [
            "runs/rule_smoke_1795318890175/9acffb68-A_006.jpg",
            "图 4 修复后真实图片链路验证 结果尚未经过逐车真值评估",
            11
        ]
    },
    {
        "h1": "五 当前主要难点"
    },
    {
        "table": {
            "header": [
                "难点",
                "当前证据与影响",
                "尚需验证的方向"
            ],
            "rows": [
                [
                    "数据覆盖面不足",
                    "校园仅 17 张，缺少新场地与真实违停样本；无法估计违停召回率",
                    "按场地和拍摄批次留出测试集，补充边界案例与负样本"
                ],
                [
                    "停车框依赖人工",
                    "现有入口需要允许区配置，换场地后无法直接判定",
                    "自动提议区域多边形，人工修正；先验证可见划线场景"
                ],
                [
                    "密集车辆检测与分割",
                    "有漏检、重复框、粘车风险，SAM 轮廓不保证对应单辆车",
                    "分别记录检车和分割失败，避免将上游错误归因于规则"
                ],
                [
                    "接地点近似不充分",
                    "最低条带受车轮遮挡、碎片及粘车影响，按目标高度修复只解决尺度问题",
                    "对照条带法与轮胎接地点标注，验证关键点方案"
                ],
                [
                    "二维占道口径",
                    "车身像素与地面通道重叠不等于真实地面占用",
                    "区分车轮越界和车身伸入通道，明确人工判定标准"
                ],
                [
                    "跨场地泛化未知",
                    "同场地少量验证不能代表不同背景和拍摄角度",
                    "测试场地不参与训练或调参，分别统计各场地结果"
                ],
                [
                    "训练数值异常",
                    "school_v3 验证损失出现 NaN；新增检查只能暴露异常",
                    "检查数据、验证过程、优化器实际参数与数值精度"
                ],
                [
                    "评估与复现不完整",
                    "历史指标口径不一致，部分参数和外部路径依赖本机",
                    "冻结实验配置与版本，保留逐车匹配、失败原因和运行记录"
                ]
            ]
        }
    },
    {
        "p": "当前不能把问题全部归结为数据量，也不能只归结为区域标定。需要通过固定区域、人工接地点和人工车辆位置的对照实验，分离检测、分割、接地点与区域边界各自引入的误差。对于无划线、划线完全遮挡或管理规则不体现在图像中的场景，应允许人工提供区域并输出不确定状态。"
    },
    {
        "h1": "六 下一步规划"
    },
    {
        "p": "规划按验收条件推进，不以固定日期承诺完成。近期先开展不依赖新增标注的工程与复现实验；新增数据和标注由后续采集安排决定，未准备完成前不报告新精度或启动大规模训练。"
    },
    {
        "table": {
            "header": [
                "阶段",
                "具体工作",
                "交付与验收条件"
            ],
            "rows": [
                [
                    "P0 当前图片基线",
                    "冻结修复版本、权重、场景配置和参数；完整重跑现有图片并逐图登记失败；保留旧版结果作参照",
                    "形成新版 CSV、失败清单与可复现命令；各状态之和等于检测数；旧配置空值语义逐项确认"
                ],
                [
                    "P1 独立评估",
                    "明确合规、压线、出界、占道及无法确认的人工标准；按场地与拍摄批次划分；匹配预测与真实车辆",
                    "形成逐车真值与匹配表；分别报告检车 P/R、违停 P/R、规范车误报比例及无法判定率，并保留分母"
                ],
                [
                    "P2 误差分解与接地点",
                    "固定人工区域比较框与条带规则；用人工接地点替换近似条带做对照；再决定是否训练接地点模型",
                    "同一测试集、相同检测输入的对照表；证明主要误差来自哪一环节，模型方案有改善证据后再扩展"
                ],
                [
                    "P3 停车区域自动提议",
                    "整理现有多边形；在可见划线场景比较边界提取加几何组合与区域分割；支持人工拖点修正",
                    "输出候选多边形及不确定提示；在未见场地统计边界偏差、区域 IoU、成功提议率与人工修正成本"
                ],
                [
                    "P4 训练稳定性与复现",
                    "按控制变量方式排查 NaN；核对 auto 优化器实际参数、AMP、数据与验证损失；修正剩余路径",
                    "保留完整日志和环境说明；训练无 NaN/Inf，并在固定留出集验证；检查通过不等于精度达标"
                ]
            ]
        }
    },
    {
        "h2": "6.1 数据与标注准备要求"
    },
    {
        "ul": [
            "自行车：补充不同场地、距离、视角、密集程度和遮挡；保留海报、影子等容易误检的负样本。",
            "停车区域：标注地面划线内侧多边形，并记录边界是否可见；同时包含空停车区、少量车与满车遮挡。",
            "违停真值：补充明确出界、压线、占道与线内贴边样本，避免测试集全为规范停放。",
            "接地点试验：先对小批样本标注前后轮接地点及可见性；遮挡到无法确认时保留未知，不强行推测。",
            "场地和拍摄批次作为元数据保存；相近连拍及同源变体放在同一划分，避免测试结果因相关样本泄漏而虚高。"
        ]
    },
    {
        "h2": "6.2 阶段决策原则"
    },
    {
        "p": "先实现自动区域提议和少量人工修正，再根据跨场地测试决定是否支持高置信度自动判定。缺少独立验证之前，不把“无需标定”作为既有能力，也不预设必须采用某一个双类分割模型。视频、推送、蒸馏和部署继续后置。"
    },
    {
        "h1": "七 阶段总结"
    },
    {
        "p": "本阶段成果是形成了一条可运行、可检查的图片违停分析链路，并修复类别映射、颜色通道、失败计数、配置回落与条带尺度等明确缺陷。13 项规则回归及单图真实推理已通过；完整精度、真实违停召回率、跨场地泛化和停车区自动提取仍是待验证能力。下一阶段以独立评估和误差分解为依据改进图片识别，逐步减少人工绘制停车框的工作量。"
    },
    {
        "h1": "八 复现与证据索引"
    },
    {
        "ul": [
            "数据证据：dataset_v2 与 dataset_school 的 images/labels；runs/data_audit/network_legacy/dedupe.json。",
            "训练证据：runs/school_v3/results.csv 与 args.yaml；runs/demo_visible/results.csv。",
            "历史规则证据：runs/archive/violation_mask_out/per_bike.csv、runs/archive/violation_mask_out2/per_bike.csv。",
            "本次规则与测试：relabel/violation_rules.py、tools/test_violation_rules.py、docs/图片规则修复.md。",
            "本次实图结果：runs/rule_smoke_1795318890175/9acffb68-A_006.jpg 及 per_bike.csv。",
            "CPU 回归命令：在项目根目录使用 sam2 环境执行 python tools/test_violation_rules.py -v。",
            "单图复现：python relabel/violation_mask.py --source data/campus/used/annotated/school_v1/images/val/9acffb68-A_006.jpg --scenes data/campus/used/annotated/school_v1/scenes.json --scene school_row --weights runs/school_v3/weights/best.pt --out runs/rule_recheck。",
            "数据、权重及多数结果图不随代码入库；在其他机器复现需准备对应资产与 SAM 安装。Windows 沙箱若缓存路径受限，需将 YOLO_CONFIG_DIR、MPLCONFIGDIR、TORCHINDUCTOR_CACHE_DIR 指向项目可写目录。"
        ]
    },
    {
        "h1": "九 汇报提纲"
    },
    {
        "table": {
            "header": [
                "页",
                "主题",
                "建议表达"
            ],
            "rows": [
                [
                    "1",
                    "项目目标",
                    "图片识别及三种违停口径"
                ],
                [
                    "2",
                    "阶段成果",
                    "端到端链路、标注工具和工程修复"
                ],
                [
                    "3",
                    "数据治理",
                    "553 文件至 153 唯一图；多余副本 400"
                ],
                [
                    "4",
                    "方法",
                    "检测、掩膜、接地点近似与几何关系"
                ],
                [
                    "5",
                    "历史训练",
                    "整体与单类指标分开；明确 NaN 异常"
                ],
                [
                    "6",
                    "历史对比",
                    "72 与 32 是报警数量差，不直接称为精度提升"
                ],
                [
                    "7",
                    "工程修复",
                    "类别、通道、区域配置和失败状态"
                ],
                [
                    "8",
                    "验证证据",
                    "13 项回归与单图 GPU 运行"
                ],
                [
                    "9",
                    "当前难点",
                    "场景单一、区域靠人、遮挡和接地点近似"
                ],
                [
                    "10",
                    "评估方案",
                    "独立场地留出与逐车真值"
                ],
                [
                    "11",
                    "改进路线",
                    "误差分解、区域自动提议及人工修正"
                ],
                [
                    "12",
                    "下一步",
                    "先完成图片基线与评估，视频继续后置"
                ]
            ]
        }
    }
]


# Keep the dated model comparison in future Markdown/Word rebuilds without
# changing the existing report's original section content.
_parts_section = os.path.join(BASE, "docs", "part_experiment_section_20261127.json")
if os.path.isfile(_parts_section):
    with open(_parts_section, encoding="utf-8") as _parts_stream:
        CONTENT.extend(json.load(_parts_stream))


def render_md():
    out = ["# %s" % TITLE, "", "> %s" % SUBTITLE, ""]
    for item in CONTENT:
        if "h1" in item:
            out += ["## %s" % item["h1"], ""]
        elif "h2" in item:
            out += ["### %s" % item["h2"], ""]
        elif "p" in item:
            out += [item["p"], ""]
        elif "ul" in item:
            out += ["- %s" % s for s in item["ul"]] + [""]
        elif "table" in item:
            t = item["table"]
            out.append("| " + " | ".join(t["header"]) + " |")
            out.append("|" + "|".join(["---"] * len(t["header"])) + "|")
            for r in t["rows"]:
                out.append("| " + " | ".join(str(c) for c in r) + " |")
            out.append("")
        elif "img" in item:
            path, cap, _ = item["img"]
            out += ["![%s](%s)" % (cap, os.path.relpath(os.path.join(BASE, path),
                                                        os.path.dirname(MD_OUT)).replace("\\", "/")),
                    "", "*%s*" % cap, ""]
    return "\n".join(out)


def render_docx():
    doc = Document()
    for border in doc.styles.element.xpath(".//w:pBdr"):
        border.getparent().remove(border)
    style = doc.styles["Normal"]
    style.font.name = "微软雅黑"
    style.font.size = Pt(10.5)
    style._element.rPr.rFonts.set(
        __import__("docx").oxml.ns.qn("w:eastAsia"), "微软雅黑")

    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(2)
    section.left_margin = section.right_margin = Cm(2.2)
    style.paragraph_format.space_after = Pt(6)
    style.paragraph_format.line_spacing = 1.15
    for name in ("Title", "Heading 1", "Heading 2"):
        heading = doc.styles[name]
        heading.font.name = "微软雅黑"
        heading.font.color.rgb = RGBColor(0, 0, 0)
        heading._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "微软雅黑")
    t = doc.add_paragraph(style="Title")
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = t.add_run(TITLE)
    run.bold = True
    run.font.size = Pt(20)

    s = doc.add_paragraph()
    s.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s.add_run(SUBTITLE).font.size = Pt(11)

    for item in CONTENT:
        if "h1" in item:
            doc.add_heading(item["h1"], level=1)
        elif "h2" in item:
            doc.add_heading(item["h2"], level=2)
        elif "p" in item:
            doc.add_paragraph(item["p"])
        elif "ul" in item:
            for line in item["ul"]:
                doc.add_paragraph(line, style="List Bullet")
        elif "table" in item:
            tb = item["table"]
            table = doc.add_table(rows=1, cols=len(tb["header"]))
            table.style = "Light Grid Accent 1"
            repeat = OxmlElement("w:tblHeader")
            table.rows[0]._tr.get_or_add_trPr().append(repeat)
            for i, h in enumerate(tb["header"]):
                cell = table.rows[0].cells[i]
                cell.text = str(h)
                for p in cell.paragraphs:
                    for r in p.runs:
                        r.bold = True
                    p.paragraph_format.keep_with_next = True
            for row in tb["rows"]:
                cells = table.add_row().cells
                for i, c in enumerate(row):
                    cells[i].text = str(c)
            for row in table.rows:
                row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        paragraph.paragraph_format.space_after = Pt(4)
                        for run in paragraph.runs:
                            run.font.size = Pt(9)
            doc.add_paragraph()
        elif "img" in item:
            path, cap, width = item["img"]
            full = os.path.join(BASE, path)
            if os.path.exists(full):
                doc.add_picture(full, width=Cm(width))
                doc.paragraphs[-1].paragraph_format.keep_with_next = True
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                p = doc.add_paragraph(cap)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    r.italic = True
                    r.font.size = Pt(9)
            else:
                doc.add_paragraph("（图缺失：%s）" % path)

    doc.save(DOCX_OUT)


if __name__ == "__main__":
    os.makedirs(os.path.dirname(MD_OUT), exist_ok=True)
    with open(MD_OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(render_md())
    render_docx()
    print("Markdown:", MD_OUT, os.path.getsize(MD_OUT) // 1024, "KB")
    print("Word    :", DOCX_OUT, os.path.getsize(DOCX_OUT) // 1024, "KB")
