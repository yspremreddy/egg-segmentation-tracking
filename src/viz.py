"""Montage of worst/random predictions: image | SAM gt | prediction  (python src/viz.py ALGO [worst|best] [N])"""
import json, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parent.parent
algo = sys.argv[1]; mode = sys.argv[2] if len(sys.argv) > 2 else "worst"; N = int(sys.argv[3]) if len(sys.argv) > 3 else 12
split = sys.argv[4] if len(sys.argv) > 4 else "dev"
res = json.load(open(ROOT / "experiments" / f"{algo}_{split}.json"))["per_image"]
order = sorted(res, key=lambda n: res[n]["iou"], reverse=(mode == "best"))[:N]
def ov(img, m, col):
    o = img.copy(); o[m] = (o[m] * .5 + np.array(col) * .5).astype(np.uint8); return o
rows = []
for n in order:
    img = cv2.resize(cv2.imread(str(ROOT / "data/images" / f"{n}.jpg")), (200, 200))
    gt = cv2.resize(cv2.imread(str(ROOT / "outputs/sam_masks" / f"{n}.png"), 0), (200, 200)) > 0
    pr = cv2.resize(cv2.imread(str(ROOT / "outputs/classical_masks" / split / algo / f"{n}.png"), 0), (200, 200)) > 0
    t = np.hstack([ov(img, gt, (0, 255, 0)), ov(img, pr, (0, 0, 255))])
    cv2.putText(t, f"{n.replace('not_damaged_','')} iou={res[n]['iou']:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
    rows.append(t)
cols = 3
while len(rows) % cols: rows.append(np.zeros_like(rows[0]))
out = np.vstack([np.hstack(rows[i:i + cols]) for i in range(0, len(rows), cols)])
cv2.imwrite(str(ROOT / "outputs" / f"viz_{algo}_{split}_{mode}.jpg"), out)
