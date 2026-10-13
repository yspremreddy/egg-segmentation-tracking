"""
Task 1c - box-prompted SAM for images where automatic selection failed
(egg camouflaged by leaves / white tray / water, or close-up). Boxes are
hand-placed (x0, y0, x1, y1) in 640x640 pixel coords, one per egg.
For images in ADD, the egg is unioned into the existing mask instead of replacing it.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "data" / "images"
OUT_DIR = ROOT / "outputs" / "sam_masks"
OVERLAY_DIR = OUT_DIR / "overlay"
CHECKPOINT = ROOT / "checkpoints" / "sam_vit_b_01ec64.pth"

BOXES = {
    10: [[130, 270, 415, 475]],
    11: [[175, 140, 400, 348]],
    12: [[228, 225, 425, 452]],
    13: [[275, 155, 470, 288]],
    28: [[255, 60, 445, 228]],
    42: [[130, 5, 350, 155]],
    44: [[0, 60, 550, 390]],
    116: [[395, 355, 555, 495]],
    117: [[245, 105, 525, 260], [300, 295, 485, 490]],
    124: [[305, 220, 495, 365]],
    135: [[315, 245, 500, 395]],
    145: [[250, 285, 435, 435]],
    140: [[262, 0, 435, 180]],
}
ADD = {140}


def main():
    torch.set_num_threads(2)
    sam = sam_model_registry["vit_b"](checkpoint=str(CHECKPOINT)).to("cpu")
    pred = SamPredictor(sam)
    for n, boxes in BOXES.items():
        name = f"not_damaged_{n}"
        img_bgr = cv2.imread(str(IMAGES_DIR / f"{name}.jpg"))
        pred.set_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        union = np.zeros(img_bgr.shape[:2], bool)
        if n in ADD:
            union |= cv2.imread(str(OUT_DIR / f"{name}.png"), 0) > 0
        for b in boxes:
            m, _, _ = pred.predict(box=np.array(b), multimask_output=False)
            union |= m[0]
        mask_u8 = union.astype(np.uint8) * 255
        cv2.imwrite(str(OUT_DIR / f"{name}.png"), mask_u8)
        ov = img_bgr.copy()
        ov[union] = (ov[union] * 0.55 + np.array([0, 0, 255]) * 0.45).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, (0, 255, 0), 2)
        cv2.imwrite(str(OVERLAY_DIR / f"{name}.jpg"), ov)
        print("done", name)


if __name__ == "__main__":
    main()
