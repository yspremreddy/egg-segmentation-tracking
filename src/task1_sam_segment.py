"""
Task 1 - Segmentation using Pretrained Model (SAM)

For every image in data/images/, run SAM's automatic mask generator and pick
the mask that best corresponds to "the egg" (the hand is usually the largest
region touching the image border, so the egg mask is picked as the largest
candidate region that does NOT touch the border and is roughly central).

Outputs, per image <name>.jpg:
  outputs/sam_masks/<name>.png        - binary mask (0/255), same size as image
  outputs/sam_masks/overlay/<name>.jpg - image with mask overlay for QA

These masks are treated as ground truth for Task 3.
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
CHECKPOINT = ROOT / "checkpoints" / "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    # SAM's automatic mask generator uses float64 internally for point
    # transforms, which MPS does not support -> fall back to CPU.
    return "cpu"


def touches_border(mask, margin=2):
    h, w = mask.shape
    return (
        mask[:margin, :].any()
        or mask[-margin:, :].any()
        or mask[:, :margin].any()
        or mask[:, -margin:].any()
    )


def pick_egg_mask(masks, img_shape):
    """
    Heuristic: among SAM's candidate masks, the egg is the region that
    - does not touch the image border (hand/arm usually does)
    - is reasonably large (not a tiny speck/finger highlight)
    - is the most "blob-like" (compact: high area / bounding-box-area ratio)
    among the remaining candidates, prefer the largest.
    """
    h, w = img_shape[:2]
    img_area = h * w
    candidates = []
    for m in masks:
        seg = m["segmentation"]
        area = m["area"]
        if area < 0.005 * img_area or area > 0.6 * img_area:
            continue
        if touches_border(seg):
            continue
        ys, xs = np.where(seg)
        if len(xs) == 0:
            continue
        bbox_area = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
        compactness = area / bbox_area
        if compactness < 0.55:  # eggs are elliptical/blob-like, filter ragged regions
            continue
        candidates.append((area, seg))

    if not candidates:
        # fallback: largest non-border-touching mask regardless of compactness
        for m in sorted(masks, key=lambda m: -m["area"]):
            if not touches_border(m["segmentation"]):
                return m["segmentation"]
        # last resort: largest mask overall
        return max(masks, key=lambda m: m["area"])["segmentation"]

    candidates.sort(key=lambda c: -c[0])
    return candidates[0][1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="process only first N images (debug)")
    parser.add_argument("--threads", type=int, default=3, help="CPU threads (fewer = cooler; M2 Air: 2-3)")
    parser.add_argument("--cooldown", type=float, default=0.0, help="seconds to sleep after each image")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Loading SAM ({MODEL_TYPE}) on {device} ...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=str(CHECKPOINT))
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=24,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.90,
        min_mask_region_area=200,
    )

    image_paths = sorted(IMAGES_DIR.glob("*.jpg"))
    if args.limit:
        image_paths = image_paths[: args.limit]
    print(f"Found {len(image_paths)} images.")

    failed = []
    todo = [p for p in image_paths if not (OUT_DIR / f"{p.stem}.png").exists()]
    print(f"{len(image_paths) - len(todo)} already done, {len(todo)} to go "
          f"(threads={args.threads}, cooldown={args.cooldown}s)")
    for img_path in tqdm(todo, desc="SAM segmenting", unit="img"):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            failed.append(img_path.name)
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        masks = mask_generator.generate(img_rgb)
        if not masks:
            failed.append(img_path.name)
            continue

        egg_mask = pick_egg_mask(masks, img_rgb.shape)
        mask_u8 = (egg_mask.astype(np.uint8)) * 255

        out_path = OUT_DIR / f"{img_path.stem}.png"
        cv2.imwrite(str(out_path), mask_u8)

        overlay = img_bgr.copy()
        color = np.zeros_like(overlay)
        color[:, :] = (0, 0, 255)
        alpha = 0.45
        mask_bool = egg_mask.astype(bool)
        overlay[mask_bool] = (
            overlay[mask_bool] * (1 - alpha) + color[mask_bool] * alpha
        ).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(str(OVERLAY_DIR / f"{img_path.stem}.jpg"), overlay)
        if args.cooldown:
            time.sleep(args.cooldown)

    print(f"Done. {len(image_paths) - len(failed)}/{len(image_paths)} succeeded.")
    if failed:
        print("Failed on:", failed)


if __name__ == "__main__":
    main()
