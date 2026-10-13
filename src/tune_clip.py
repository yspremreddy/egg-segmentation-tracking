"""Evaluate ellipse clipping on the cached DEV pool selection (never touches eval)."""
import json, sys, time
from pathlib import Path
import cv2, numpy as np
from tune import pools, gts
import seg_classical as S
R = Path(__file__).resolve().parent.parent
params = json.load(open(R / "experiments/tuned_params.json"))
imgs = {n: cv2.imread(str(R / f"data/images/{n}.jpg")) for n in pools}
sel = {n: S.select_from_pool(c, f, **params) for n, (c, f) in pools.items()}
edges = {n: S._edge_map(imgs[n]) for n in pools}
def score(preds):
    I, P, Rc = [], [], []
    for n, pr in preds.items():
        gt = gts[n]; i = (pr & gt).sum(); u = (pr | gt).sum()
        I.append(i / u if u else 1); P.append(i / max(pr.sum(), 1)); Rc.append(i / gt.sum())
    return np.mean(I), np.mean(P), np.mean(Rc), np.median(I)
if __name__ == "__main__":
    print("no clip        IoU %.4f P %.3f R %.3f med %.3f" % score(sel))
    for grow in (1.0, 1.06, 1.12):
        for ms in (0.3, 0.45, 0.6):
            t0 = time.time()
            pr = {n: S.clip_to_ellipse(imgs[n], sel[n], grow=grow, min_support=ms, edges=edges[n]) for n in pools}
            print("grow %.2f min_support %.2f  IoU %.4f P %.3f R %.3f med %.3f  (%.0fs)" % ((grow, ms) + score(pr) + (time.time() - t0,)), flush=True)
