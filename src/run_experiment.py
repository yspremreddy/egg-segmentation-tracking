"""
Run a classical segmentation algorithm (from src/seg_classical.py) over a split
and score it against the SAM masks (Task 1 ground truth).

    python src/run_experiment.py --algo v1_otsu --split dev

The eval split is locked: it can only be used with --final (done once at the end).
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

import seg_classical
from metrics import iou_dice

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "data" / "images"
GT = ROOT / "outputs" / "sam_masks"


def run(algo_name, split, save_dir=None):
    names = json.load(open(ROOT / "data" / "splits.json"))[split]
    fn = seg_classical.ALGOS[algo_name]
    rows = {}
    t0 = time.time()
    for n in names:
        img = cv2.imread(str(IMAGES / f"{n}.jpg"))
        gt = cv2.imread(str(GT / f"{n}.png"), 0) > 0
        pred = fn(img)
        rows[n] = iou_dice(pred, gt)
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(save_dir) / f"{n}.png"), pred.astype(np.uint8) * 255)
    summary = {k: float(np.mean([r[k] for r in rows.values()])) for k in ("iou", "dice", "precision", "recall")}
    summary["median_iou"] = float(np.median([r["iou"] for r in rows.values()]))
    summary["frac_iou_gt_0.5"] = float(np.mean([r["iou"] > 0.5 for r in rows.values()]))
    summary["sec_per_img"] = (time.time() - t0) / len(names)
    return summary, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "eval"])
    ap.add_argument("--final", action="store_true", help="required to touch the eval split")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    if a.split == "eval" and not a.final:
        raise SystemExit("eval split is locked until the algorithm is final; pass --final")
    out = ROOT / "outputs" / "classical_masks" / a.split / a.algo
    summary, rows = run(a.algo, a.split, out if a.save else None)
    print(a.algo, a.split, json.dumps({k: round(v, 4) for k, v in summary.items()}))
    log = ROOT / "experiments" / f"{a.algo}_{a.split}.json"
    json.dump({"summary": summary, "per_image": rows}, open(log, "w"), indent=1)
