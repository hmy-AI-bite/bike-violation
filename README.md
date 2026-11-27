# 校园自行车违停检测

图片输入后，由 YOLO 找自行车、SAM2 分割轮廓，再结合人工标定的停车区域，用几何规则输出合规、压线、出界、占道、无法判定或未标定状态。当前聚焦图片，视频跟踪和停车区域自动识别尚未完成。

数据目录已按来源、使用状态、原始/标注阶段整理。日常查找请先看 [数据目录说明](docs/数据目录说明.md)，逐图清单在 `data/manifests/images.csv`。

## 当前状态

- 2026-11-25：核验 66 张原图，去重后 60 张，与旧学校数据相比新增 49 张；这批只用于开发测试。
- YOLO11x 已在约 12 GB 显存的 RTX 5070 上完成微调，1280 分辨率、batch 4，13 轮后提前停止，最佳权重来自第 1 轮，没有 NaN/Inf。
- 新大模型在所选三张图上避免了旧微调 YOLO11s 的严重漏检；尚无独立标注证明其优于官方 YOLO11x。
- YOLO11x 与 SAM2.1 Tiny 单图、逐车分割时可同驻 GPU，整卡占用采样峰值约 2.4 GiB。
- 密集车群仍有漏检、重复框和分割粘连；最低掩膜条带并非真实接地点，轻微侧向越线仍可能漏报。当前不宣称违停准确率。
- 2026-11-27：完成固定人工框的 5 图 157 辆对照。自动中心点提示将合规误报从 6 降至 5，仍检出 7/8 个已知违停，倒车漏报尚未解决。它只作为默认关闭的实验开关；[实验过程、失败候选与验证限制](docs/规则与分割提示对照_2026-11-27.md)均已记录。
- 已保存 PartGLEE、VLPart 论文和 RN50 权重，并完成同批 5 图推理及局部候选关联对照。当前结果没有优于原方案，未替换默认程序；[新增实验报告](docs/实验报告_部件定位与车辆关联_2026-11-27.md)保留了漏检、错误部件及归属不明的结果。

数据、对比图和实验限制见 [2026-11-25 工作总结](docs/工作总结_2026-11-25.md)。旧状态文件已归档为 [2026-11-13 历史快照](docs/archive/STATUS_2026-11-13.md)，旧指标和结论不代表当前结果。

## 流程

```text
图片 → YOLO 自行车检测 → SAM2 逐车掩膜 → 接地点条带与区域几何判定
                            ↑                         ↑
                        检测框提示              scenes.json 人工标定
```

判定规则位于 `relabel/violation_rules.py`。未标定时仅检测，掩膜质量或面积不满足要求时输出“无法判定”。行为及阈值说明见 [图片规则修复](docs/图片规则修复.md)。

可选实验：`python config.py run detect --sam-center-point --out runs/parking_detection_center_trial`，自动给 SAM 增加车辆框中心前景点，无需补标。默认不开启；中心点不是接地点，密集遮挡时也可能落到邻车。CSV 的 `sam_prompt` 列记录本次使用 `box` 或 `box_center`。

## 安装依赖

使用 Python 3.10 或更新版本，在虚拟环境中安装依赖。需要 GPU 时先安装与驱动匹配的 CUDA 版 PyTorch 和 torchvision。SAM2 来自官方 Git 仓库，因此还需 Git。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install torch torchvision
python -m pip install --no-build-isolation -r requirements.txt
```

现有环境可直接复用。`yolo-lib` 和外部 SAM2 源码目录存在时会被加入模块搜索路径，否则使用已安装的包。仓库不包含模型权重，需准备 YOLO 和 SAM2 权重。GPU 训练、GUI 可以分别使用不同的 Python 环境。

## 路径配置

所有运行路径由根目录 `config.py` 提供，优先级为环境变量、`.env`、项目相对默认值。相对路径始终以仓库根目录为基准，与当前工作目录无关。

```powershell
Copy-Item .env.example .env
python config.py show
```

已有 `.env` 时编辑它，不要覆盖。`.env` 保存本机配置且不提交，`.env.example` 是可提交的示例。文件只接受字面量 `KEY=VALUE`，不执行命令，也不展开 shell 变量。

| 环境变量 | 用途与默认值 |
|---|---|
| `BICYCLE_DATA_ROOT` | 数据目录根，默认 `data` |
| `BICYCLE_OUTPUT_ROOT` | 输出目录根，默认仓库根目录 |
| `BICYCLE_WEIGHTS_DIR` | 官方权重目录，默认 `relabel/weights` |
| `BICYCLE_YOLO_LIB` | 可选本地 YOLO 库，默认 `yolo-lib` |
| `BICYCLE_SAM2_REPO` | 可选 SAM2 源码，默认 `vendor/sam2`；已安装 SAM2 时可不提供 |
| `BICYCLE_SAM2_CHECKPOINT` | SAM2 权重，默认权重目录下 `sam2.1_hiera_tiny.pt` |
| `BICYCLE_SAM2_CONFIG` | SAM2 配置资源名 |
| `BICYCLE_PYTHON` | 启动器使用的 Python，默认当前解释器 |
| `BICYCLE_GUI_PYTHON` | GUI Python，默认与上项相同 |
| `BICYCLE_SCHOOL_SOURCE` | 原始未划分学校数据，默认 `data/campus/used/raw/legacy_school_source`；该旧来源未在本机找到 |
| `BICYCLE_SCHOOL_IMAGES` | 默认输入图片，默认 `data/campus/used/annotated/school_v1/images/train` |
| `BICYCLE_SCENES`、`BICYCLE_SCENE` | 区域配置与场景名，默认 `data/campus/used/annotated/school_v1/scenes.json`、`school_row` |
| `BICYCLE_DETECT_WEIGHTS` | 一键检测权重，保留原默认 `runs/school_v3/weights/best.pt`，可改为新大模型 |
| `BICYCLE_FONT_PATH` | 可选中文字体；默认尝试系统字体，失败时回退 |

还可用 `BICYCLE_ENV_FILE` 指定其他配置文件。历史实验元数据中的原始路径保留作溯源，不作为运行配置。

数据集 `data.yaml` 使用相对的 `train`、`val`，省略机器绝对 `path`，从 YAML 所在目录解析。数据构建脚本会生成这种可迁移格式。

## 启动

Windows 可双击四个批处理入口，也可以调用配置启动器。批处理优先使用 `BICYCLE_PYTHON`，否则调用 PATH 中的 `python`，再由配置入口选择计算或 GUI 环境。

```powershell
python config.py run zones
python config.py run labeler
python config.py run detect
python config.py run train-demo
```

直接调用脚本时，原有命令行参数仍可覆盖默认值：

```powershell
python tools/zone_tool.py data/campus/used/annotated/school_v1/images/train data/campus/used/annotated/school_v1/scenes.json school_row
python relabel/violation_mask.py --source data/campus/used/annotated/school_v1/images/train --scenes data/campus/used/annotated/school_v1/scenes.json --scene school_row
python relabel/check_dataset.py --data data/network/used/annotated/bicycle_v2/data.yaml
python relabel/train.py --data data/network/used/annotated/bicycle_v2/data.yaml --imgsz 960 --epochs 120 --lr0 0.002
```

标注工具：点击顶部“选择数据集…”（Ctrl+O），可选择原始图片文件夹（递归读取子目录）或标准 YOLO 数据集。本项目 `data/<来源>/<使用状态>/raw/<批次>/images` 下的原图，标注分离保存到 `annotation_work/<批次>/labels`，并保留子目录结构；其他普通图片目录沿用所选目录下的 `labels/`；标准数据集继续使用对应的 `labels/train`、`labels/val`。顶部显示完整的图片与标注路径。切换数据集、翻页前自动保存；同目录同名但后缀不同的图片需先重命名，避免共用标注文件。

选中一个自行车框后，可在顶部下拉框设置人工停车状态：合规、压线、出界、占道、无法判定；新建框、没有停车状态文件的旧检测框默认“合规”；已有状态记录保留，已有记录与框不匹配时仍标为“未标注”待复核。选中框后可按 1 合规、2 压线、3 出界、4 占道、5 无法判定。框上显示对应结论，顶部显示本图已设置状态的数量。停车状态本身缺乏足够证据时选“无法判定”。接地点看不清或不适用不等于停车状态无法判定：密集遮挡仍可保留明确的状态；倒车后轮越界按用户确认的出界含义保留，另行分析其越界证据，不把轮廓点当成地面接地点。人工状态随保存、翻页一起写入标签旁的同名 `.review.json` 文件，保存的每条记录含 YOLO 框与停车状态。移动框保留状态，删除和撤销同步处理状态；外部修改框坐标后，旧结论不自动套用到新框。YOLO `.txt` 仍保留自行车检测类别，当前训练流程不会自动读取停车状态 JSON。

标注视角：滚轮缩放，按住鼠标中键拖动画面；点击“重置视角 (R)”或按 R 恢复居中与默认缩放。平移只改变显示位置，不修改框或停车状态。

标定工具：顶部选择图片文件夹（Ctrl+O，支持子目录）；默认允许区，1 允许区、2 禁停区，Tab 也可切换。左键加顶点，Enter 闭合，Backspace/Ctrl+Z 撤销上一个顶点，Esc 取消未闭合图形。滚轮缩放、中键拖动画面、R 重置；Ctrl+S/S 保存，A/D 或左右键翻页。切换模式会取消尚未闭合的图形，重复选择当前模式保留顶点；C 清空当前模式的区域。顶部显示区域文件保存路径，已闭合的区域按图存储。不同角度需分别检查区域。

## 实验与验证

逐车参考答案补充接地点：双击 `继续标注接地点.bat`，固定打开 `ground_truth_v1` 总目录，并默认启用连续点选。两个接地点不区分前后轮，按任意顺序点击即可。新记录默认“看不清”：能看清两个就点两下，只能看清一个就点一下，两个都看不清可直接 Enter；Enter 确认未点部分仍为看不清，自动保存并选下一辆，视角轻微平移、保留缩放。默认值与已人工确认分开记录，未操作过的车不计入完成。最后一辆停留本图，A/D 翻图。倒车或轮胎离地选“不适用”按钮。Esc 取消点选后可点击另一个车框，Ctrl+Z 撤销并继续补点，Ctrl+S 保存。取消“连续点选”勾选后可编辑车辆框。接地点存入 `.review.json` 的 `contact_points.point1/point2`，标明 `contact_order: unordered`，坐标相对原图归一化。旧版 front/rear 会按原坐标读取为接地点1/2，已标点与停车状态保留，无需重标。首次改写前备份为 `.review.json.bak`。

```powershell
python experiments/20261125/train_yolo11x.py --epochs 40 --name bike11x_v2 --imgsz 1280 --batch 4
python experiments/20261125/prepare_detections.py
python experiments/20261125/run_comparison.py
python experiments/20261125/measure_pipeline.py
```

后三项默认检查 `bike11x_v1`，不是刚创建的 `bike11x_v2`；可用 `prepare_detections.py --finetuned` 指向其他权重。固定测试图、阈值和人工测试区域属于实验设置，不等于独立真值。

```powershell
python tools/test_config.py -v
python tools/test_violation_rules.py -v
python tools/test_labeler.py
python tools/test_zone_tool.py
```

规则测试不加载模型；GUI 测试使用临时生成的样本，不读取或修改真实数据。GUI 测试需在安装 PySide6 的环境运行。

## 目录

```text
config.py                  统一路径、字体和启动器配置
.env.example               可提交的配置示例
requirements.txt           Python 依赖
relabel/                   数据治理、训练与判定命令
tools/                    标注 GUI、报告生成器与回归测试
experiments/20261125/      日期实验：大模型训练、对比、显存测量
experiments/legacy/        一次性旧实验与环境辅助脚本
experiments/ui/            GUI 界面渲染实验
docs/archive/             带日期的历史状态
docs/images/              文档中的对比图
data/network/             网络来源：used / unused / usage_unknown
data/campus/              校园实拍：used / unused / usage_unknown
data/manifests/           逐图索引、重复检查与迁移记录
runs/                     权重、日志、掩膜及其他输出，不提交
```

历史实验脚本仅作复现参考，未全部重测，其旧算法行为不能代替当前入口。更多说明见 [训练手册](docs/训练手册.md) 和 [数据治理](relabel/README.md)。
