"""
Task 1b - re-segment problem images with SAM, keeping ALL egg-like regions.

The PDF asks for "egg(s)", so images with several eggs need every egg masked.
For a subset of images (multi-egg or failed in v1) this:
  1. runs SAM's automatic mask generator and caches every candidate mask
     (outputs/sam_cache/<name>.npz) so selection can be re-tuned without re-running SAM
  2. selects egg-like candidates (compact, convex, not border-touching, not
     overlapping a larger kept region) and writes outputs/sam_masks/<name>.png
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "data" / "images"
OUT_DIR = ROOT / "outputs" / "sam_masks"
OVERLAY_DIR = OUT_DIR / "overlay"
CACHE_DIR = ROOT / "outputs" / "sam_cache"
CHECKPOINT = ROOT / "checkpoints" / "sam_vit_b_01ec64.pth"

# multi-egg images + v1 failures (identified from outputs/qa_sheet*.jpg)
SUBSET = [103, 105, 108, 113, 114, 115, 121, 128, 129, 132, 133, 134, 136, 140, 142,
          144, 148, 10, 11, 12, 13, 42, 44, 116, 117, 124, 135, 145, 28, 45]


def touches_border(m, margin=2):
    return m[:margin].any() or m[-margin:].any() or m[:, :margin].any() or m[:, -margin:].any()


def solidity(m):
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(c))
    return cv2.contourArea(c) / hull if hull > 0 else 0


def select_eggs(masks, shape, min_rel_area=0.3):
    h, w = shape[:2]
    img_area = h * w
    cands = []
    for m in masks:
        seg, area = m["segmentation"], m["area"]
        if area < 0.005 * img_area or area > 0.45 * img_area or touches_border(seg):
            continue
        ys, xs = np.where(seg)
        bbox = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
        if area / bbox < 0.6 or solidity(seg) < 0.9:
            continue
        cands.append((area, seg))
    cands.sort(key=lambda c: -c[0])
    kept = []
    for area, seg in cands:
        if any((seg & k).sum() / min(area, k.sum()) > 0.5 for _, k in kept):
            continue
        kept.append((area, seg))
    if kept:
        biggest = kept[0][0]
        kept = [k for k in kept if k[0] >= min_rel_area * biggest]
    return [k[1] for k in kept]


def write_outputs(name, img_bgr, egg_masks):
    union = np.zeros(img_bgr.shape[:2], bool)
    for m in egg_masks:
        union |= m
    mask_u8 = union.astype(np.uint8) * 255
    cv2.imwrite(str(OUT_DIR / f"{name}.png"), mask_u8)
    overlay = img_bgr.copy()
    overlay[union] = (overlay[union] * 0.55 + np.array([0, 0, 255]) * 0.45).astype(np.uint8)
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (0, 255, 0), 2)
    cv2.imwrite(str(OVERLAY_DIR / f"{name}.jpg"), overlay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cooldown", type=float, default=3.0)
    ap.add_argument("--reselect", action="store_true", help="re-run selection from cache only (no SAM)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    names = [f"not_damaged_{i}" for i in SUBSET]
    gen = None
    for name in tqdm(names, unit="img"):
        img_bgr = cv2.imread(str(IMAGES_DIR / f"{name}.jpg"))
        cache = CACHE_DIR / f"{name}.npz"
        if cache.exists():
            z = np.load(cache)
            segs = np.unpackbits(z["bits"], axis=-1)[..., : img_bgr.shape[1]].astype(bool)
            masks = [{"segmentation": s, "area": int(s.sum())} for s in segs]
        else:
            if args.reselect:
                continue
            if gen is None:
                sam = sam_model_registry["vit_b"](checkpoint=str(CHECKPOINT)).to("cpu")
                gen = SamAutomaticMaskGenerator(sam, points_per_side=24, pred_iou_thresh=0.86,
                                                stability_score_thresh=0.90, min_mask_region_area=200)
            masks = gen.generate(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            masks = [m for m in masks if m["area"] < 0.6 * img_bgr.shape[0] * img_bgr.shape[1]]
            np.savez_compressed(cache, bits=np.packbits(np.stack([m["segmentation"] for m in masks]), axis=-1))
            time.sleep(args.cooldown)
        eggs = select_eggs(masks, img_bgr.shape)
        if not eggs:
            print(f"{name}: no egg-like candidate, keeping v1 mask")
            continue
        write_outputs(name, img_bgr, eggs)


if __name__ == "__main__":
    main()
