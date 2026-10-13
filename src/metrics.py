import numpy as np


def iou_dice(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    iou = inter / union if union else 1.0
    denom = pred.sum() + gt.sum()
    dice = 2 * inter / denom if denom else 1.0
    prec = inter / pred.sum() if pred.sum() else 0.0
    rec = inter / gt.sum() if gt.sum() else 0.0
    return dict(iou=float(iou), dice=float(dice), precision=float(prec), recall=float(rec))
