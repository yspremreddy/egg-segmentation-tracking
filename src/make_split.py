"""
Build the dev/eval split required by Task 2 (4:1 ratio, leakage-aware).

The dataset contains photo series: the same scene (steel pot, ash, leaves,
marble counter, ...) is shot several times with the egg moved around. A plain
random split would put near-identical scenes in both dev and eval and inflate
the eval score. So:
  1. cluster images into scene groups (HSV colour histogram + 8x8 thumbnail,
     average-linkage agglomerative clustering, distance threshold 1.4 -> 48 groups)
  2. split whole GROUPS 80/20 with a fixed seed (GroupShuffleSplit)

Writes data/scene_groups.json and data/splits.json: {"dev": [...], "eval": [...]}
"""
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "data" / "images"
SEED = 42
THRESH = 1.4


def features(names):
    F = []
    for n in names:
        im = cv2.resize(cv2.imread(str(IMAGES_DIR / f"{n}.jpg")), (64, 64))
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, [12, 6, 6], [0, 180, 0, 256, 0, 256]).flatten()
        h /= h.sum()
        small = cv2.resize(im, (8, 8)).astype(np.float32).flatten() / 255
        F.append(np.concatenate([np.sqrt(h) * 2, small * 0.15]))
    return np.array(F)


def main():
    names = sorted(p.stem for p in IMAGES_DIR.glob("*.jpg"))
    groups = AgglomerativeClustering(n_clusters=None, distance_threshold=THRESH,
                                     linkage="average").fit_predict(features(names))
    json.dump(dict(zip(names, groups.tolist())), open(ROOT / "data" / "scene_groups.json", "w"))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    dev_idx, eval_idx = next(gss.split(names, groups=groups))
    split = {"dev": sorted(names[i] for i in dev_idx), "eval": sorted(names[i] for i in eval_idx)}
    assert not set(groups[dev_idx]) & set(groups[eval_idx])
    json.dump(split, open(ROOT / "data" / "splits.json", "w"), indent=1)
    print(f"{len(groups.tolist()) and len(set(groups))} scene groups; dev={len(split['dev'])} eval={len(split['eval'])}")


if __name__ == "__main__":
    main()
