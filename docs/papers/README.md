# 图片违停识别相关论文

下载日期：2026-11-27。PDF 来自期刊、作者仓库或 arXiv；文件名编号即建议阅读顺序。实际下载状态、页数、来源和 SHA-256 见 `download_manifest.json`。

| 顺序 | 论文全文 | 重点阅读内容 | 原文与代码 |
|---|---|---|---|
| 1 | [Precise Position Estimation of Road Users by Extracting Object-Specific Key Points for Embedded Edge Cameras（2025）](01_2025_Road_User_Ground_Contact_Keypoints.pdf) | 第 3 节的单点与双点定义、图 4 的车轮接地点、第 4 节的位置误差与失败案例。思考接地点预测能否替代当前掩膜最低条带。 | [期刊原文](https://www.mdpi.com/2079-9292/14/7/1291) |
| 2 | [DMPR-PS A Novel Approach for Parking-Slot Detection Using Directional Marking-Point Regression（2019）](02_2019_DMPR-PS.pdf) | 标线关键点的位置、方向和形状如何表示，以及怎样通过几何规则组成停车位。 | [论文页面](https://ieeexplore.ieee.org/document/8784735/) · [作者代码](https://github.com/Teoge/DMPR-PS) |
| 3 | [Holistic Parking Slot Detection with Polygon-Shaped Representations（2023）](03_2023_HPS-Net_Polygon_Representations.pdf) | 多边形表示、顶点预测和角点损失。注意输出使用俯视坐标，不能直接等同于校园斜拍照片中的多边形。 | [arXiv 原文](https://arxiv.org/abs/2310.11629) |
| 4 | [Attentional Graph Neural Network for Parking-Slot Detection（2021）](04_2021_Attentional_GNN_Parking_Slot.pdf) | 如何学习标线关键点之间的连接关系；用于理解关键点配对错误，不必立即引入图网络。 | [arXiv 原文](https://arxiv.org/abs/2104.02576) · [作者代码](https://github.com/Jiaolong/gcn-parking-slot) |
| 5 | [Advancing Real-World Parking Slot Detection with Large-Scale Dataset and Semi-Supervised Baseline（2025）](05_2025_CRPS-D_SS-PSD.pdf) | CRPS-D 的场景覆盖、标注和半监督教师学生方法。思考如何利用尚未标注的图片。下载的是 arXiv 版本。 | [arXiv 原文](https://arxiv.org/abs/2509.13133) · [数据与代码](https://github.com/zzh362/CRPS-D) |
| 6 | [一种基于 YOLOv3 的共享单车违规停放检测方法（2019）](06_2019_Shared_Bicycle_Illegal_Parking_YOLOv3.pdf) | 中文直接相关工作，关注违停定义、区域设置、实验数据及评价指标。 | [期刊原文](https://www.ecice06.com/EN/10.19678/j.issn.1000-3428.0054915) · [期刊 PDF 入口](https://www.ecice06.com/EN/PDF/10.19678/j.issn.1000-3428.0054915) |
| 7 | [PartGLEE（2024）](07_2024_PartGLEE.pdf) | 物体与部件的层次关系；已运行 RN50 并导出查询关联。 | [论文](https://arxiv.org/abs/2407.16696) · [作者代码](https://github.com/ProvenceStar/PartGLEE) |
| 8 | [Going Denser with Open-Vocabulary Part Segmentation / VLPart（2023）](08_2023_VLPart.pdf) | 自行车车轮等部件；已运行 RN50 Pascal Part。 | [论文](https://arxiv.org/abs/2305.11173) · [作者代码](https://github.com/facebookresearch/VLPart) |

## 阅读时建议记录的问题

1. 输入是普通单目照片、环视图还是俯视图？是否需要相机标定？
2. 标注的是车辆框、车轮接地点、标线角点，还是整个停车区多边形？
3. 遮挡目标怎样处理？有没有不可见或无法判定状态？
4. 位置误差、检测准确率与违停准确率是否分别评估？指标分母是什么？
5. 测试是否包含训练中没有出现过的场地？代码、权重和数据能否获取？

多数停车位论文处理汽车环视图和单个车位，本项目处理校园斜拍图和成片自行车停车区。借鉴表示与评估方法，不能直接沿用其指标或假设预训练权重适用。第一篇研究道路使用者定位，也需要验证其方法对密集停放自行车的适用性。

下载原文不代表已复现实验；本次没有修改项目模型、数据和判定规则。

## 2026-11-27 补充检索：越界部位及对应车辆

用户已明确：不要求整车掩膜完全分开，任务是定位越界部位并确定所属车辆。补充检索后已保存两篇 PDF、源码及 RN50 检查点，并在现有 5 张图上完成 GPU 推理。实验报告见[部件定位与车辆关联对照](../实验报告_部件定位与车辆关联_2026-11-27.md)。两模型当前配置未能改善任务结果，下面的方法建议须结合实际失败阅读，不能再理解为已经有效的推荐。

| 优先级 | 论文与一手来源 | 可借鉴的内容 | 当前限制 |
|---|---|---|---|
| 已完成初测 | [PartGLEE，ECCV 2024](https://arxiv.org/html/2407.16696v1) · [作者代码](https://github.com/ProvenceStar/PartGLEE) | 联合定位物体与部件，建立层次关联，贴近“车轮属于哪辆车”的问题 | 已导出查询生成关系，但它不等于真实所属车辆；本轮高置信度车轮候选覆盖不足 |
| 部件识别对照 | [VLPart，ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Sun_Going_Denser_with_Open-Vocabulary_Part_Segmentation_ICCV_2023_paper.pdf) · [作者代码](https://github.com/facebookresearch/VLPart) | 物体及部件的开放词汇分割；[公开类别定义](https://raw.githubusercontent.com/facebookresearch/VLPart/main/vlpart/data/datasets/pascal_part.py)含 bicycle:wheel、bicycle:handlebar、bicycle:saddle | 类别标签不是同类车辆的实例归属；仓库已归档，安装文档以 Linux/macOS 与 Detectron2 为基础 |
| 关联机制参考 | [Objects as Context for Detecting Their Semantic Parts，CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Gonzalez-Garcia_Objects_as_Context_CVPR_2018_paper.pdf) | 使用整物体外观及相对位置辅助判断部件，支持保留整车上下文，避免只按最近车框分配车轮 | 不是校园停车模型，不直接照搬网络或效果数字 |
| 空间推理参考 | [遮挡车道重建与车辆掩膜空间推理，Applied Sciences 2026](https://www.mdpi.com/2076-3417/16/20/10080) | 将边界感知、车辆定位与违规规则分开，保存空间判定证据 | 面向大型运输车辆；缺完整分类 FP/TN，数据未公开，不作为校园模型性能或即插即用效果的依据 |

### 推荐的最小实验

以下是实验前的验证设计，现已完成首次整图推理及候选关联审计。实际观察为 PartGLEE 20 个、VLPart 133 个车轮候选；使用相同保守关联规则与车轮局部下缘候选规则时，分别只命中 0/8、1/8 个已知违停，未找回原倒车漏报。候选个数不是准确率，未关联不能记为合规。

**先联合识别车辆与部件，再筛选越界部位。** 前一版“先从全部线外像素找块、再找最近车辆”的路线缺少部件语义与实例归属依据，不能直接落成最终判定。

1. 保留 YOLO 与人工停车区域，首轮仍使用冻结的人工车辆框，避免混入漏检影响。
2. 优先验证 PartGLEE 预训练 RN50 的部件输出和层次关联；首个部件目标为车轮，不要求再标接地点。普通车与倒车都要保留必要的整车和标线上下文。
3. 将部件、停车边界和相邻车辆放到同一坐标中，产生局部越界候选。轮廓在线外只是候选证据，不自动等于地面越界；尤其不能把直立车轮的上部投影直接当作接地点越线。
4. 通过物体—部件关联和原图上下文确定所属车辆，再关联到现有车辆框；遇到同类车重叠时不以最近中心点作最终依据。保留关联不明的状态，单独计入覆盖率。
5. 输出原图上的局部高亮、所属车辆框和编号，并保存候选来源、部件类型、边界关系及关联依据。只高亮所依据的部位，避免把整个车群标为证据。

这是已做初测、尚未通过任务验证的新路线，不是已完成的默认算法改动。当前已记录关键轮胎遗漏、错误的大块车轮候选与归属覆盖不足；没有据此要求用户重复标注或启动训练。

### 实现位置与验证

- 新方法应先放在 `experiments/20261127/`，复用 `evaluate_boundary_candidates.py` 的冻结输入和记账方式，不继续向默认 `classify_mask` 叠加比例阈值。
- 验证通过后，再从 `relabel/violation_mask.py` 中的逐车处理位置接入部件证据分支；现有 YOLO 检测与区域解析保留。
- 使用已有 8 个明确违停、144 个合规目标计分，5 个无法判定保留。新旧比较检出、误报、拒答和重复报警；最终报警框与人工车辆框匹配，不能“找对一片车群”就算找对车辆。
- 现有标注没有逐像素越界真值，也没有每个部件的车辆归属真值。因此只能量化车辆级报警与定位；局部高亮及部件归属先做逐例可视核查，不宣称其像素 IoU 或精确归属率。相邻两辆都违停时，仅靠两个状态标签尤其无法验证部件究竟属于谁。
- 新候选先与原框提示基线、自动中心点候选分别对照。不能把 7/8 检出和 5 次误报的小幅变化当成已经解决局部定位。

### 复现条件核验

作者提供了 [PartGLEE RN50/SwinL 权重链接](https://github.com/ProvenceStar/PartGLEE/blob/master/assets/MODEL_ZOO.md)、[安装说明](https://github.com/ProvenceStar/PartGLEE/blob/master/assets/INSTALL.md)和[评估说明](https://github.com/ProvenceStar/PartGLEE/blob/master/assets/TEST.md)。本轮已下载并运行 RN50，未下载或测试 SwinL。独立环境为 experiments/.venv-parts，原 SAM 环境保持不变；因没有可用 CUDA 编译工具，使用作者已有的 PyTorch 注意力参考实现，并记录少量 API 兼容修改。没有与编译版逐像素对照，不称为标准环境原样复现，不报告部署 FPS。

LISA 的推理分割与 Grounded SAM 的文本提示可作为后续备选，但本轮没有证据证明普通文本提示就能稳定找到小车轮并正确归属，因此不优先将大模型问答直接当作违停裁决。
