# 自行车检测数据集 v2：预训练模型辅助重标

## 这次做了什么

原始 `bike_dataset/` 有两个致命问题：

1. **标注不可用** —— 2156 个框里 85.5% 与任何真实自行车都不重叠，只有 1.9% 是贴合的好框（框普遍是图宽的 54%，大面积越界）。
2. **数据重复 + 泄漏** —— 553 个文件里只有 **153 张唯一图片**，495 个是 MD5 完全相同的副本，其中 54 组同时出现在 train 和 val。

处理方式：用 COCO 预训练的 `yolo11x` 重新推理，按内容哈希去重后重建数据集。

## 目录

```
relabel/
├── weights/yolo11x.pt        # COCO 预训练权重（bicycle 类本来就会）
├── autolabel.py              # 553 张全量推理 -> relabel/auto/labels/
├── compare.py                # 新旧标注对比 -> report/compare.csv
├── build_v2.py               # 去重 + 重划分 -> dataset_v2/
├── gallery_v2.py             # 生成复核联系表 -> report/review_sheets/
├── train.py                  # 微调脚本
└── report/
    ├── compare.csv           # 每张图：好/松/错/漏 统计
    ├── summary.txt           # 汇总
    ├── review_queue.json     # 需要复核的图（按严重程度排序）
    ├── dedupe.json           # 重复组明细
    └── review_sheets/        # 17 张联系表，9 图一版，可直接肉眼看
```

`dataset_v2/`：**112 train / 41 val**，153 张唯一图，标注来自 yolo11x。

## 复核怎么过

**快审（推荐先做）**：看 `report/review_sheets/sheet_01..17.jpg`，一张表 9 张图，
挑出"多框/漏框/框错"的图，只改这些。全部看完约 20 分钟。

**精审**：用 labelImg 打开 `dataset_v2`：

```bash
pip install labelImg
labelImg D:\bick\dataset_v2\images\train D:\bick\dataset_v2\labels\train D:\bick\dataset_v2\classes.txt
```

`classes.txt` 内容只有一行 `bicycle`。快捷键：`W` 画框、`A`/`D` 上一张/下一张、`Ctrl+S` 保存。

改完的标注直接覆盖 `dataset_v2/labels/{train,val}/*.txt` 即可，无需重新生成。

## 训练

```bat
set PYTHONPATH=D:\bick\yolo-lib
set YOLO_CONFIG_DIR=D:\bick\relabel\cfg
set MPLCONFIGDIR=D:\bick\relabel\cfg\mpl
C:\Users\26803\miniconda3\envs\sam2\python.exe D:\bick\relabel\train.py ^
   --weights D:\bick\relabel\weights\yolo11s.pt --imgsz 960 --epochs 120 --batch 16
```

先跑通再调参：`--epochs 30` 看 mAP 曲线是否正常上升。

## 复核后的迭代（可选）

如果还有漏检（夜景、远距离、遮挡），用第一版训练好的权重再自动标一次，
和人工改过的标签合并，只处理新增/冲突的图 —— 通常两轮就能收敛。
