"""Cache candidate pools for a split so selection parameters can be tuned quickly.
   python src/build_pool_cache.py dev   -> experiments/pool_dev.pkl"""
import json, pickle, sys
from pathlib import Path
import cv2, numpy as np
import seg_classical as S
ROOT = Path(__file__).resolve().parent.parent
split = sys.argv[1]
if split == "eval":
    raise SystemExit("eval pool is built only by the final run")
names = json.load(open(ROOT / "data/splits.json"))[split]
out = {}
for i, n in enumerate(names):
    img = cv2.imread(str(ROOT / f"data/images/{n}.jpg"))
    pool = S.build_pool(img)
    out[n] = ([np.packbits(p) for p in pool], S.pool_features(img, pool))
    print(i + 1, len(names), n, len(pool), flush=True)
pickle.dump(out, open(ROOT / "experiments" / f"pool_{split}.pkl", "wb"))
