# 校园自行车违停检测

用预训练模型 + 几何判定，识别「停在划线外 / 停在通道上 / 车头压线」的自行车。
先做图片，后续接视频监控。

## 结论先行

- **越界判定不需要模型学**：区域由人标定，模型只负责找车，是否越界是几何问题。
- **框不够精确**：框是轴对齐矩形，斜靠的车会把背景算进去。改用 SAM2 掩膜后，误报从 72 辆降到 32 辆（-56%）。
- **区域是唯一的短板**：多角度、多场地必须每处标定一次，或训练区域分割模型。

## 目录

```
bike_dataset/        原始 553 张 + YOLO 标注（标注质量差，仅作原始存档）
dataset_v2/          去重后的 153 张唯一图片 + 人工精修后的标注
dataset_school/      学校实拍 17 张（3072x4096）+ 停车区多边形
relabel/             重标、对比、训练、判定脚本
tools/               标注工具（精修 / 区域标定）
runs/                训练输出
violation_mask_out/  掩膜版判定的结果样例
```

## 脚本

| 脚本 | 作用 |
|---|---|
| `tools/labeler.py` | 精修自行车框：点选、拖动、画框、删除、自动保存 |
| `tools/zone_tool.py` | 标定停车区：点几下圈出允许区/禁停区，存 `scenes.json` |
| `relabel/autolabel.py` | 用 yolo11x 自动标注，可同时出带框图 |
| `relabel/compare.py` | 新旧标注对比，输出可疑清单 |
| `relabel/build_v2.py` | 按哈希去重 + 重新划分数据集 |
| `relabel/train.py` | 微调（小数据集必须降 lr0，否则会发散） |
| `relabel/preview.py` | 批量生成带框图 + 本地翻页查看器 |
| `relabel/violation.py` | 框规则越界判定（快速基线） |
| `relabel/violation_mask.py` | 掩膜规则判定：接地点 + 压线 + 占道 |

## 运行环境

复用本机 conda `sam2` 环境（torch 2.11 + CUDA 12.8），ultralytics 及其依赖装在
`yolo-lib/`（不进 git），通过 `PYTHONPATH` 叠加，不改动 conda 环境本身：

```bat
set PYTHONPATH=D:\bick\yolo-lib
set YOLO_CONFIG_DIR=D:\bick\relabel\cfg
```

## 典型流程

```bat
:: 1) 标定停车区（点几下，S 保存）
python tools\zone_tool.py "J:\project-1-at-2026-11-13-17-44-a1d4a70f\images" dataset_school\scenes.json school_row

:: 2) 掩膜版违停判定
python relabel\violation_mask.py --source "J:\...\images" --scenes dataset_school\scenes.json --scene school_row

:: 3) 精修标注（可选）
python tools\labeler.py dataset_v2

:: 4) 训练
python relabel\train.py --data dataset_v2\data.yaml --imgsz 960 --epochs 120 --lr0 0.002
```

## 已知问题

- 标注口径要统一：一辆完整车一个框，遮挡超过 70% 可不标，摩托车不算自行车。
- 负样本不足：海报/屏幕里的自行车、地面影子需要单独收集，标为空 txt。
- `violation_mask.py` 的 `scenes.json` 目前是拿一排车的外接四边形，不是地面划线；
  上线前要改成划线内侧，否则正常停放也会报「压线」。

## 依赖

- ultralytics 8.4（`yolo-lib/`）
- SAM2.1 hiera-tiny（`D:\CV-Datatext-Pipeline\models\sam2\sam2.1_hiera_tiny.pt`）
- 训练权重：`yolo11s.pt` / `yolo11x.pt`（`relabel/weights/`，不进 git）
