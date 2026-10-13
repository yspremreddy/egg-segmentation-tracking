"""Grid-search the selection parameters on the cached DEV pool. python src/tune.py"""
import itertools, json, pickle
from pathlib import Path
import cv2, numpy as np
import seg_classical as S
R = Path(__file__).resolve().parent.parent
pool = pickle.load(open(R / "experiments/pool_dev.pkl", "rb"))
H = W = 640
gts, pools = {}, {}
for n, (packed, feats) in pool.items():
    gts[n] = cv2.imread(str(R / f"outputs/sam_masks/{n}.png"), 0) > 0
    pools[n] = ([np.unpackbits(p)[:H * W].reshape(H, W).astype(bool) for p in packed], feats)

def evaluate(**kw):
    ious, P, Rc = [], [], []
    for n, (comps, feats) in pools.items():
        pr = S.select_from_pool(comps, feats, **kw)
        gt = gts[n]
        i = (pr & gt).sum(); u = (pr | gt).sum()
        ious.append(i / u if u else 1.0)
        P.append(i / pr.sum() if pr.sum() else 0); Rc.append(i / gt.sum())
    return float(np.mean(ious)), float(np.mean(P)), float(np.mean(Rc)), float(np.median(ious))

if __name__ == "__main__":
    base = dict(min_fill=0.65, min_solid=0.88, max_frac=0.35, min_contrast=15.0, rel=0.5, tex_k=8.0, size_pow=0.5)
    print("base", [round(x, 4) for x in evaluate(**base)])
    grid = dict(min_fill=[0.55, 0.65, 0.75], min_solid=[0.8, 0.88, 0.93], max_frac=[0.25, 0.35, 0.45],
                min_contrast=[10, 15, 25], rel=[0.4, 0.6, 0.8, 1.01], tex_k=[4, 8, 16], size_pow=[0.0, 0.5, 1.0])
    res = []
    for k, vals in grid.items():            # one-at-a-time sensitivity first
        for v in vals:
            p = dict(base); p[k] = v
            res.append((k, v, evaluate(**p)))
            print(k, v, [round(x, 4) for x in res[-1][2]])
