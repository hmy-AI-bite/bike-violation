"""测速：YOLO 单独 vs YOLO+SAM2 每张图耗时。"""
import sys
import time

sys.path.insert(0, r"D:\CV-Datatext-Pipeline\sam2-repo")
sys.path.insert(0, r"D:\bick\yolo-lib")

import numpy as np
from PIL import Image

from ultralytics import YOLO

IMG = r"J:\project-1-at-2026-11-13-17-44-a1d4a70f\images\039f0796-A_013.jpg"
CKPT = r"D:\CV-Datatext-Pipeline\models\sam2\sam2.1_hiera_tiny.pt"
CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

yolo = YOLO(r"D:\bick\relabel\weights\yolo11x.pt")
im = Image.open(IMG).convert("RGB")
im.thumbnail((1400, 1900))
arr = np.array(im)

# 预热
yolo.predict(arr, imgsz=1280, conf=0.35, classes=[1], device=0, verbose=False)

t = time.time()
for _ in range(3):
    r = yolo.predict(arr, imgsz=1280, conf=0.35, classes=[1], device=0, verbose=False)[0]
t_yolo = (time.time() - t) / 3
print("YOLO 单图: %.0f ms, 检出 %d 辆" % (t_yolo * 1000, len(r.boxes)))

from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402

sam = build_sam2(CFG, CKPT, device="cuda")
pred = SAM2ImagePredictor(sam)
pred.set_image(arr)
boxes = r.boxes.xyxy.cpu().numpy()
pred.predict(box=boxes[0], multimask_output=False)  # 预热

t = time.time()
pred.set_image(arr)
t_embed = time.time() - t
t = time.time()
for b in boxes:
    pred.predict(box=b, multimask_output=False)
t_boxes = time.time() - t
print("SAM2 编码: %.0f ms" % (t_embed * 1000))
print("SAM2 逐框: %.0f ms/辆 x %d = %.0f ms" % (t_boxes / len(boxes) * 1000, len(boxes), t_boxes * 1000))
print("整图合计: %.0f ms (%.1f fps)" % ((t_yolo + t_embed + t_boxes) * 1000,
                                        1 / (t_yolo + t_embed + t_boxes)))
print("只对区外目标跑 SAM（假设 2 辆）: %.0f ms" %
      ((t_yolo + t_embed + 2 * t_boxes / len(boxes)) * 1000))
