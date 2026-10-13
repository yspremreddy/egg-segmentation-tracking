"""Fast per-frame egg segmentation for video (Task 4).

Same ingredients as the Task 2 method (colour-threshold candidates + egg-shape test), but without the
~100 GrabCut runs per frame that made v8 ~4 s/image. Additions for video:
  * blobs that fail the single-egg test (two touching eggs) are split with a distance-transform watershed
  * regions touching the frame border are rejected (bright wall / shirt at the top corners)
Returns a list of boolean masks, one per egg-like region."""
import cv2
import numpy as np

K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
K9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
K13 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))  # removes thin caption/arrow strokes touching an egg


def _egg_like(comp, img_area, min_frac, max_frac, min_fill, min_solid, max_aspect):
    a = int(comp.sum())
    if not (min_frac * img_area <= a <= max_frac * img_area):
        return False
    if comp[0].any() or comp[-1].any() or comp[:, 0].any() or comp[:, -1].any():
        return False
    cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        return False
    hull = cv2.contourArea(cv2.convexHull(c))
    if hull == 0 or cv2.contourArea(c) / hull < min_solid:
        return False
    (_, _), (w, h), _ = cv2.fitEllipse(c)
    fill = a / (np.pi * w * h / 4 + 1e-6)
    return fill >= min_fill and max(w, h) / max(min(w, h), 1e-6) <= max_aspect


def _split(comp):
    """Split a merged blob into pieces: watershed on the distance transform."""
    u8 = comp.astype(np.uint8)
    dist = cv2.distanceTransform(u8, cv2.DIST_L2, 5)
    peaks = (dist > 0.6 * dist.max()).astype(np.uint8)
    n, markers = cv2.connectedComponents(peaks)
    if n <= 2:
        return []
    markers = markers.astype(np.int32) + 1
    markers[u8 == 0] = 0
    markers[(u8 > 0) & (peaks == 0)] = 0
    ws = cv2.watershed(cv2.cvtColor(u8 * 255, cv2.COLOR_GRAY2BGR), markers.copy())
    return [(ws == i) & comp for i in range(2, n + 1)]


def _brightness_contrast(v, comp, ring_w=12):
    """Mean V inside the region minus mean V in the ring just outside it (an egg is much brighter than what surrounds it)."""
    u8 = comp.astype(np.uint8)
    kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_w + 1, 2 * ring_w + 1))
    ring = cv2.dilate(u8, kk).astype(bool) & ~cv2.dilate(u8, np.ones((5, 5), np.uint8)).astype(bool)
    return float(v[comp].mean() - v[ring].mean()) if ring.any() else 0.0


def egg_candidates(bgr, min_frac=0.003, max_frac=0.12, min_fill=0.72, min_solid=0.88,
                   max_aspect=2.0, v_min=150, s_max=70, min_v_contrast=100.0, min_v_in=200.0):
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] > v_min) & (hsv[..., 1] < s_max)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, K13)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, K9)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    kw = dict(min_frac=min_frac, max_frac=max_frac, min_fill=min_fill, min_solid=min_solid, max_aspect=max_aspect)
    out = []
    for i in range(1, n):
        comp = lab == i
        if st[i, cv2.CC_STAT_AREA] < min_frac * m.size or st[i, cv2.CC_STAT_AREA] > 3 * max_frac * m.size:
            continue
        if _egg_like(comp, m.size, **kw):
            out.append(comp)
        elif st[i, cv2.CC_STAT_AREA] >= 2 * min_frac * m.size:
            found = [p for p in _split(comp) if _egg_like(p, m.size, **kw)]
            if not found:   # fallback: strip attachments (fingers, a second half-hidden egg) with growing openings
                for r in (15, 21, 27):
                    op = cv2.morphologyEx(comp.astype(np.uint8), cv2.MORPH_OPEN,
                                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r)))
                    k, lb, stt, _ = cv2.connectedComponentsWithStats(op, 8)
                    found = [lb == j for j in range(1, k) if _egg_like(lb == j, m.size, **kw)]
                    if found:
                        break
            out += found
    v = hsv[..., 2].astype(np.float32)
    return [c for c in out if v[c].mean() >= min_v_in and _brightness_contrast(v, c) >= min_v_contrast]
