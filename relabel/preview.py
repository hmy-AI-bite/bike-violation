"""生成带框的逐张预览图 + 本地 HTML 查看器。

框的颜色按置信度分档（置信度从 detections.json 里按 IoU 匹配回来）:
    绿   conf >= 0.50   大概率正确
    黄   0.35 ~ 0.50    要留意
    红   < 0.35         重点怀疑（误检高发）

输出:
    dataset_v2/preview/{train,val}/*.jpg   带框原尺寸图
    review.html                            翻页查看器（键盘 <- -> 切换）
"""
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

DS = r"D:\bick\dataset_v2"
DET = r"D:\bick\relabel\auto\detections.json"
SRC = r"D:\bick\bike_dataset"
VIEWER = r"D:\bick\review.html"

try:
    FONT = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)
    FONT_S = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 14)
except Exception:
    FONT = FONT_S = ImageFont.load_default()


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_detections():
    idx = {}
    for rec in json.load(open(DET, encoding="utf-8")):
        idx[(rec["split"], rec["image"])] = rec
    return idx


def find_source(stem):
    for split in ("train", "val"):
        for ext in (".jpg", ".jpeg", ".png"):
            if os.path.exists(os.path.join(SRC, "images", split, stem + ext)):
                return split
    return None


def main():
    dets = load_detections()
    manifest = []
    total = 0

    for split in ("train", "val"):
        idir = os.path.join(DS, "images", split)
        ldir = os.path.join(DS, "labels", split)
        odir = os.path.join(DS, "preview", split)
        os.makedirs(odir, exist_ok=True)

        for fn in sorted(os.listdir(idir)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = os.path.splitext(fn)[0]
            img = Image.open(os.path.join(idir, fn)).convert("RGB")
            W, H = img.size
            d = ImageDraw.Draw(img)

            src_split = find_source(stem) or split
            det = dets.get((src_split, fn)) or dets.get((src_split, stem + ".jpg")) or {}
            det_xyxy = [[b["box"], b["conf"]] for b in det.get("bicycles", [])]

            boxes = []
            for line in open(os.path.join(ldir, stem + ".txt"), encoding="utf-8"):
                p = line.split()
                if len(p) < 5:
                    continue
                xc, yc, w, h = map(float, p[1:5])
                x1, y1, x2, y2 = (xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H
                conf = 0.0
                best = 0.0
                for xyxy, c in det_xyxy:
                    v = iou([x1, y1, x2, y2], xyxy)
                    if v > best:
                        best, conf = v, c
                boxes.append(([x1, y1, x2, y2], conf if best > 0.5 else -1))

            for i, (box, conf) in enumerate(boxes, 1):
                color = (0, 200, 80) if conf >= 0.5 else ((255, 190, 0) if conf >= 0.35 else (255, 50, 50))
                d.rectangle(box, outline=color, width=3)
                label = "#%d" % i if conf < 0 else "#%d %.2f" % (i, conf)
                tx, ty = box[0] + 3, max(1, box[1] + 2)
                tw = d.textlength(label, font=FONT_S)
                d.rectangle([tx - 1, ty - 1, tx + tw + 2, ty + 16], fill=color)
                d.text((tx, ty), label, fill=(0, 0, 0) if conf >= 0.5 else (255, 255, 255), font=FONT_S)

            # 顶部信息条
            head = "%s/%s  %dx%d  %d boxes" % (split, stem, W, H, len(boxes))
            d.rectangle([0, 0, min(W, 420), 26], fill=(0, 0, 0))
            d.text((6, 3), head, fill=(255, 255, 0), font=FONT)

            out = os.path.join(odir, stem + ".jpg")
            img.save(out, quality=90)
            total += 1

            confs = [c for _, c in boxes if c >= 0]
            manifest.append(dict(
                split=split, name=stem, w=W, h=H, n=len(boxes),
                min_conf=round(min(confs), 3) if confs else None,
                src="dataset_v2/preview/%s/%s.jpg" % (split, stem),
            ))
            if total % 40 == 0:
                print("  %d ..." % total, flush=True)

    with open(os.path.join(DS, "preview", "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
    write_viewer(manifest)
    print("已生成 %d 张带框图 -> %s" % (total, os.path.join(DS, "preview")))
    print("查看器 ->", VIEWER)


def write_viewer(manifest):
    data = json.dumps(manifest, ensure_ascii=False)
    html = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>自行车框复核</title>
<style>
  html,body{margin:0;height:100%;background:#111;color:#eee;font:14px/1.5 "Microsoft YaHei",sans-serif;}
  #bar{padding:8px 12px;background:#1b1b1b;display:flex;gap:14px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #333}
  #bar b{color:#ffd54f}
  #wrap{position:absolute;top:52px;left:0;right:0;bottom:0;display:flex;align-items:center;justify-content:center;overflow:hidden}
  img{max-width:100%;max-height:100%;object-fit:contain;cursor:zoom-in}
  button{background:#2b2b2b;color:#eee;border:1px solid #444;border-radius:4px;padding:4px 10px;cursor:pointer}
  button:hover{background:#3a3a3a}
  input[type=range]{width:160px}
  .dim{color:#888}
</style></head><body>
<div id="bar">
  <button onclick="go(-1)">← 上一张</button>
  <button onclick="go(1)">下一张 →</button>
  <span><b id="name"></b></span>
  <span class="dim" id="meta"></span>
  <label><input type="checkbox" id="low" onchange="render()"> 只看低置信(&lt;0.5)</label>
  <span class="dim">进度 <b id="pos"></b></span>
  <span class="dim">绿=高置信 黄=可疑 红=重点怀疑 | 空格看原图</span>
</div>
<div id="wrap"><img id="pic" onclick="toggleOrig()"></div>
<script>
const ALL = __DATA__;
let list = ALL.slice(), i = 0, showOrig = false;
function render(){
  if(document.getElementById('low').checked){
    list = ALL.filter(x => x.min_conf === null || x.min_conf < 0.5);
    if(i >= list.length) i = 0;
  } else { list = ALL.slice(); }
  const it = list[i]; if(!it) return;
  document.getElementById('pic').src = it.src.replace('/preview/', showOrig ? '/images/' : '/preview/');
  document.getElementById('name').textContent = it.split + ' / ' + it.name;
  document.getElementById('meta').textContent = it.w + '×' + it.h + '  ' + it.n + ' 框'
      + (it.min_conf === null ? '' : '  最低置信 ' + it.min_conf);
  document.getElementById('pos').textContent = (i+1) + ' / ' + list.length;
}
function go(d){ i = (i + d + list.length) % list.length; render(); }
function toggleOrig(){ showOrig = !showOrig; render(); }
document.addEventListener('keydown', e => {
  if(e.key === 'ArrowLeft'){ go(-1); e.preventDefault(); }
  if(e.key === 'ArrowRight'){ go(1); e.preventDefault(); }
  if(e.key === ' '){ toggleOrig(); e.preventDefault(); }
});
render();
</script></body></html>"""
    html = html.replace("__DATA__", data)
    with open(VIEWER, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
