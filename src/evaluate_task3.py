"""Task 3: quantitative + qualitative evaluation of the classical method (v8) vs SAM ground truth.
Writes plots + tables to outputs/eval/ . Uses the already-saved eval predictions."""
import json
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parent.parent
OUT = R / "outputs" / "eval"; OUT.mkdir(parents=True, exist_ok=True)
ALGO = "v8_tuned"
ev = json.load(open(R / f"experiments/{ALGO}_eval.json"))
dv = json.load(open(R / f"experiments/{ALGO}_dev.json"))

def ncomp(m, min_area=300):
    n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    return int((st[1:, 4] >= min_area).sum())

rows = []
for n, r in ev["per_image"].items():
    gt = cv2.imread(str(R / f"outputs/sam_masks/{n}.png"), 0) > 0
    pr = cv2.imread(str(R / f"outputs/classical_masks/eval/{ALGO}/{n}.png"), 0) > 0
    mode = "good" if r["iou"] >= 0.7 else (
        "leak (over-seg)" if r["recall"] >= 0.85 and r["precision"] < r["recall"] - 0.15 else
        "missed egg(s)" if r["precision"] >= 0.85 and r["recall"] < r["precision"] - 0.15 else "boundary/mixed")
    rows.append(dict(name=n, **r, gt_eggs=ncomp(gt), pred_eggs=ncomp(pr), mode=mode, gt_frac=float(gt.mean())))
rows.sort(key=lambda x: x["iou"])
json.dump(rows, open(OUT / "eval_table.json", "w"), indent=1)

I = np.array([r["iou"] for r in rows]); D = np.array([r["dice"] for r in rows])
P = np.array([r["precision"] for r in rows]); Rc = np.array([r["recall"] for r in rows])
print("EVAL mean IoU %.3f  Dice %.3f  P %.3f  R %.3f  median IoU %.3f  IoU>0.5 %.0f%%  IoU>0.75 %.0f%%" %
      (I.mean(), D.mean(), P.mean(), Rc.mean(), np.median(I), 100 * (I > .5).mean(), 100 * (I > .75).mean()))
print("DEV  mean IoU %.3f  Dice %.3f" % (dv["summary"]["iou"], dv["summary"]["dice"]))
from collections import Counter
print(Counter(r["mode"] for r in rows))
cnt_ok = np.mean([r["gt_eggs"] == r["pred_eggs"] for r in rows]); print("egg-count exact match: %.0f%%" % (100 * cnt_ok))
multi = [r for r in rows if r["gt_eggs"] > 1]; print("multi-egg imgs:", len(multi), "mean IoU", np.mean([r["iou"] for r in multi]).round(3),
      "| single:", np.mean([r["iou"] for r in rows if r["gt_eggs"] == 1]).round(3))

# 1: histogram + per-image bars
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(I, bins=np.linspace(0, 1, 11), color="#4C78A8", edgecolor="w"); ax[0].axvline(I.mean(), color="k", ls="--", label=f"mean {I.mean():.2f}")
ax[0].set_xlabel("IoU"); ax[0].set_ylabel("# eval images"); ax[0].set_title("IoU distribution (eval, n=20)"); ax[0].legend()
cols = {"good": "#54A24B", "leak (over-seg)": "#E45756", "missed egg(s)": "#F58518", "boundary/mixed": "#B279A2"}
ax[1].barh([r["name"].replace("not_damaged_", "#") for r in rows], I, color=[cols[r["mode"]] for r in rows])
ax[1].set_xlabel("IoU"); ax[1].set_title("Per-image IoU, coloured by failure mode")
ax[1].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in cols.values()], labels=list(cols), fontsize=8, loc="lower right")
plt.tight_layout(); plt.savefig(OUT / "iou_hist_bars.png", dpi=130); plt.close()

# 2: precision-recall scatter
plt.figure(figsize=(5.2, 5))
for m, c in cols.items():
    s = [r for r in rows if r["mode"] == m]
    plt.scatter([r["recall"] for r in s], [r["precision"] for r in s], c=c, label=m, s=50)
for r in rows:
    if r["iou"] < 0.5: plt.annotate(r["name"].replace("not_damaged_", "#"), (r["recall"], r["precision"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
plt.xlabel("recall"); plt.ylabel("precision"); plt.xlim(0.2, 1.05); plt.ylim(0.2, 1.05); plt.plot([.2, 1], [.2, 1], "k:", lw=.5)
plt.title("Precision vs recall (eval)"); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(OUT / "precision_recall.png", dpi=130); plt.close()

# 3: dev vs eval metric bars
fig, ax = plt.subplots(figsize=(6, 3.6)); ks = ["iou", "dice", "precision", "recall"]; x = np.arange(4)
ax.bar(x - .2, [dv["summary"][k] for k in ks], .4, label="dev (n=90, tuned on)", color="#9ecae9")
ax.bar(x + .2, [ev["summary"][k] for k in ks], .4, label="eval (n=20, held out)", color="#4C78A8")
ax.set_xticks(x); ax.set_xticklabels(ks); ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.set_title("Dev vs held-out eval")
plt.tight_layout(); plt.savefig(OUT / "dev_vs_eval.png", dpi=130); plt.close()

# 4: qualitative failure gallery  (image | SAM gt | prediction) for worst 8
def ov(img, m, col): o = img.copy(); o[m] = (o[m] * .5 + np.array(col) * .5).astype(np.uint8); return o
tiles = []
for r in rows[:8]:
    n = r["name"]; img = cv2.resize(cv2.imread(str(R / f"data/images/{n}.jpg")), (260, 260))
    gt = cv2.resize(cv2.imread(str(R / f"outputs/sam_masks/{n}.png"), 0), (260, 260)) > 0
    pr = cv2.resize(cv2.imread(str(R / f"outputs/classical_masks/eval/{ALGO}/{n}.png"), 0), (260, 260)) > 0
    t = np.hstack([img, ov(img, gt, (0, 255, 0)), ov(img, pr, (0, 0, 255))])
    cv2.putText(t, f"{n.replace('not_damaged_', '#')} IoU {r['iou']:.2f} {r['mode']}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
    tiles.append(t)
cv2.imwrite(str(OUT / "failure_gallery.jpg"), np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, 8, 2)]))
