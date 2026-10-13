"""Classical (no learning) egg segmentation algorithms. Registry: ALGOS[name](bgr) -> bool mask."""
import cv2
import numpy as np


def _largest_cc(mask, k=1, min_area=300):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
    out = np.zeros(mask.shape, bool)
    for i in order[:k]:
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out |= lab == i
    return out


def _fill_holes(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(out, cnts, -1, 1, -1)
    return out.astype(bool)


def v1_otsu(bgr):
    """Baseline: grayscale Otsu, largest bright component."""
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _largest_cc(m > 0)


def v2_hsv(bgr):
    """Eggs are bright + low saturation. Threshold V high / S low, open/close, fill holes."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    m = (hsv[..., 1] < 60) & (hsv[..., 2] > 150)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return _fill_holes(_largest_cc(m > 0))


def _egg_like(comp, img_area, min_frac=0.005, max_frac=0.35, min_fill=0.65, min_solid=0.88, allow_border=False):
    """Shape test on one binary component: size, not touching border, ellipse-like fill, convex."""
    a = comp.sum()
    if a < min_frac * img_area or a > max_frac * img_area:
        return False
    if not allow_border and (comp[0].any() or comp[-1].any() or comp[:, 0].any() or comp[:, -1].any()):
        return False
    cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(c))
    if hull == 0 or cv2.contourArea(c) / hull < min_solid:
        return False
    if len(c) < 5:
        return False
    (_, _), (w, h), _ = cv2.fitEllipse(c)
    return a / (np.pi * w * h / 4 + 1e-6) >= min_fill


def _select_eggs(m, **kw):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    out = np.zeros(m.shape, bool)
    for i in range(1, n):
        comp = lab == i
        if _egg_like(comp, m.size, **kw):
            out |= comp
    return out


def v3_hsv_shape(bgr):
    """v2 colour mask, but keep every shape-valid component (compact, convex, off-border)."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 1] < 60) & (hsv[..., 2] > 150)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    return _select_eggs(_fill_holes(m > 0))


def _candidates(bgr):
    """Many binary masks: multi-level thresholds of several colour channels, both polarities."""
    from skimage.filters import threshold_multiotsu
    b = cv2.GaussianBlur(bgr, (7, 7), 0)
    hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(b, cv2.COLOR_BGR2LAB)
    chans = [cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), hsv[..., 1], hsv[..., 2], lab[..., 1], lab[..., 2]]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for ch in chans:
        try:
            ths = list(threshold_multiotsu(ch, classes=3)) + list(threshold_multiotsu(ch, classes=4))
        except ValueError:
            continue
        for t in ths:
            for m in (ch > t, ch <= t):
                m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k)
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
                yield _fill_holes(m > 0)


def _contrast(lab, comp, ring_w=12):
    """LAB distance between the region and the ring of pixels just outside it."""
    kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_w + 1, 2 * ring_w + 1))
    ring = cv2.dilate(comp.astype(np.uint8), kk).astype(bool) & ~cv2.dilate(comp.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if ring.sum() < 20:
        return 0.0
    return float(np.linalg.norm(lab[comp].mean(0) - lab[ring].mean(0)))


def _nms(cands, overlap=0.5):
    kept = []
    for sc, comp in sorted(cands, key=lambda c: -c[0]):
        a = comp.sum()
        if any((comp & k).sum() / min(a, k.sum()) > overlap for _, k in kept):
            continue
        kept.append((sc, comp))
    return kept


def v4_multi_candidate(bgr, rel=0.6, min_contrast=15.0):
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (7, 7), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    seen, cands = set(), []
    for m in _candidates(bgr):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            key = (x // 4, y // 4, w // 4, h // 4)
            if key in seen:
                continue
            seen.add(key)
            comp = labels == i
            if not _egg_like(comp, m.size, min_fill=0.75, min_solid=0.92):
                continue
            c = _contrast(lab, comp)
            if c >= min_contrast:
                cands.append((c, comp))
    kept = _nms(cands)
    if not kept:
        return np.zeros(bgr.shape[:2], bool)
    top = kept[0][0]
    out = np.zeros(bgr.shape[:2], bool)
    for sc, comp in kept:
        if sc >= rel * top:
            out |= comp
    return out


def _grabcut_refine(bgr, comp, iters=3, scale=0.5):
    """Refine a rough region with GrabCut: eroded core = sure FG, region = probable FG,
    ring around it = probable BG, rest = sure BG. Runs at reduced resolution for speed."""
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    c = cv2.resize(comp.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    r = max(3, int(0.5 * np.sqrt(c.sum() / np.pi)))      # ~ egg radius / 2
    core = cv2.erode(c, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r)))
    ring = cv2.dilate(c, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r)))
    gc = np.full(c.shape, cv2.GC_BGD, np.uint8)
    gc[ring > 0] = cv2.GC_PR_BGD
    gc[c > 0] = cv2.GC_PR_FGD
    gc[core > 0] = cv2.GC_FGD
    if core.sum() < 10:
        return comp
    bg, fg = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.grabCut(small, gc, None, bg, fg, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return comp
    out = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    out = cv2.resize(out, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    out = _largest_cc(_fill_holes(out), min_area=100)
    return out if out.any() else comp


def v5_grabcut(bgr, rel=0.6, min_contrast=15.0, topn=6):
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (7, 7), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    seen, cands = set(), []
    for m in _candidates(bgr):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            key = (x // 4, y // 4, w // 4, h // 4)
            if key in seen:
                continue
            seen.add(key)
            comp = labels == i
            if not _egg_like(comp, m.size, min_fill=0.6, min_solid=0.85):
                continue
            c = _contrast(lab, comp)
            if c >= min_contrast:
                cands.append((c, comp))
    kept = _nms(cands)[:topn]
    refined = []
    for sc, comp in kept:
        r = _grabcut_refine(bgr, comp)
        if _egg_like(r, r.size, min_fill=0.55, min_solid=0.82):
            refined.append((_contrast(lab, r), r))
    refined = _nms(refined)
    out = np.zeros(bgr.shape[:2], bool)
    if refined:
        top = refined[0][0]
        for sc, comp in refined:
            if sc >= rel * top:
                out |= comp
    return out


def _score(lab, grad, comp):
    """Egg-likeness: contrast with surroundings x sqrt(size) / (1 + interior texture).
    Eggs are smooth, fairly large blobs; logos/text/shadow fragments are small or textured."""
    tex = float(grad[comp].mean())
    return _contrast(lab, comp) * np.sqrt(comp.sum()) / (1.0 + tex / 8.0)


def v6_scored(bgr, rel=0.5, min_contrast=15.0, topn=8):
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (7, 7), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)) / 8.0
    seen, cands = set(), []
    for m in _candidates(bgr):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            key = (x // 4, y // 4, w // 4, h // 4)
            if key in seen:
                continue
            seen.add(key)
            comp = labels == i
            if not _egg_like(comp, m.size, min_fill=0.6, min_solid=0.85):
                continue
            if _contrast(lab, comp) >= min_contrast:
                cands.append((_score(lab, grad, comp), comp))
    kept = _nms(cands)[:topn]
    refined = []
    for sc, comp in kept:
        r = _grabcut_refine(bgr, comp)
        if _egg_like(r, r.size, min_fill=0.55, min_solid=0.82):
            refined.append((_score(lab, grad, r), r))
    refined = _nms(refined)
    out = np.zeros(bgr.shape[:2], bool)
    if refined:
        top = refined[0][0]
        for sc, comp in refined:
            if sc >= rel * top:
                out |= comp
    return out


def _seed_candidates(bgr, centers=(0.2, 0.4, 0.6, 0.8), radii=(0.10, 0.17, 0.25)):
    """GrabCut candidates seeded from a grid of ellipses (finds low-contrast eggs that
    colour thresholds miss, e.g. white egg on white bag)."""
    h, w = bgr.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    for cy in centers:
        for cx in centers:
            for r in radii:
                a, b = r * w * 1.0, r * h * 0.8
                ell = (((xx - cx * w) / a) ** 2 + ((yy - cy * h) / b) ** 2) <= 1
                yield _grabcut_refine(bgr, ell, scale=0.4)


def v7_seeded(bgr, rel=0.5, min_contrast=15.0, topn=8):
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (7, 7), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)) / 8.0
    seen, cands = set(), []
    def add(comp, shape_kw):
        if not _egg_like(comp, comp.size, **shape_kw):
            return
        if _contrast(lab, comp) >= min_contrast:
            cands.append((_score(lab, grad, comp), comp))
    for m in _candidates(bgr):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        for i in range(1, n):
            key = tuple(stats[i, :4] // 4)
            if key in seen:
                continue
            seen.add(key)
            add(labels == i, dict(min_fill=0.6, min_solid=0.85))
    for comp in _seed_candidates(bgr):
        add(comp, dict(min_fill=0.6, min_solid=0.85))
    kept = _nms(cands)[:topn]
    refined = []
    for sc, comp in kept:
        r = _grabcut_refine(bgr, comp)
        if _egg_like(r, r.size, min_fill=0.55, min_solid=0.82):
            refined.append((_score(lab, grad, r), r))
    refined = _nms(refined)
    out = np.zeros(bgr.shape[:2], bool)
    if refined:
        top = refined[0][0]
        for sc, comp in refined:
            if sc >= rel * top:
                out |= comp
    return out


def build_pool(bgr, topn=14):
    """Expensive stage: all candidate regions (threshold + seeded GrabCut), refined with GrabCut.
    Returns list of bool masks. Selection (cheap, tunable) is done in select_from_pool."""
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (7, 7), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)) / 8.0
    raw, seen = [], set()
    def add(comp):
        if _egg_like(comp, comp.size, min_fill=0.5, min_solid=0.8, max_frac=0.45):
            c = _contrast(lab, comp)
            if c >= 10:
                raw.append((_score(lab, grad, comp), comp))
    for m in _candidates(bgr):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        for i in range(1, n):
            key = tuple(stats[i, :4] // 4)
            if key not in seen:
                seen.add(key)
                add(labels == i)
    for comp in _seed_candidates(bgr):
        add(comp)
    pool = []
    for sc, comp in _nms(raw)[:topn]:
        pool.append(comp)
        pool.append(_grabcut_refine(bgr, comp))
    return pool


def pool_features(bgr, pool):
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (7, 7), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)) / 8.0
    feats = []
    for comp in pool:
        a = comp.sum()
        if a == 0:
            feats.append(None); continue
        cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        hull = cv2.contourArea(cv2.convexHull(c))
        fill, ell_iou = 0.0, 0.0
        if len(c) >= 5:
            el = cv2.fitEllipse(c)
            (_, _), (w, h), _ = el
            fill = a / (np.pi * w * h / 4 + 1e-6)
            em = np.zeros(comp.shape, np.uint8)
            cv2.ellipse(em, el, 1, -1)
            em = em.astype(bool)
            ell_iou = (em & comp).sum() / max((em | comp).sum(), 1)
        feats.append(dict(ell_iou=float(ell_iou), area_frac=a / comp.size, solidity=cv2.contourArea(c) / hull if hull else 0, fill=fill,
                          border=bool(comp[0].any() or comp[-1].any() or comp[:, 0].any() or comp[:, -1].any()),
                          contrast=_contrast(lab, comp), tex=float(grad[comp].mean())))
    return feats


def select_from_pool(pool, feats, min_fill=0.65, min_solid=0.88, max_frac=0.35, min_contrast=15.0,
                     rel=0.5, tex_k=8.0, size_pow=0.5, min_ell_iou=0.0):
    cands = []
    for comp, f in zip(pool, feats):
        if f is None or f["border"] or not (0.005 <= f["area_frac"] <= max_frac):
            continue
        if f["fill"] < min_fill or f["solidity"] < min_solid or f["contrast"] < min_contrast:
            continue
        if f.get("ell_iou", 1.0) < min_ell_iou:
            continue
        cands.append((f["contrast"] * f["area_frac"] ** size_pow / (1 + f["tex"] / tex_k), comp))
    kept = _nms(cands)
    out = np.zeros(pool[0].shape if pool else (1, 1), bool)
    if kept:
        top = kept[0][0]
        for sc, comp in kept:
            if sc >= rel * top:
                out |= comp
    return out


ALGOS = {"v7_seeded": v7_seeded, "v6_scored": v6_scored, "v5_grabcut": v5_grabcut, "v4_multi_candidate": v4_multi_candidate, "v1_otsu": v1_otsu, "v2_hsv": v2_hsv, "v3_hsv_shape": v3_hsv_shape}


def v8_tuned(bgr):
    """v7 candidate pool + tuned selection (params from experiments/tuned_params.json, fit on dev)."""
    import json
    from pathlib import Path
    params = json.load(open(Path(__file__).resolve().parent.parent / "experiments" / "tuned_params.json"))
    pool = build_pool(bgr)
    if not pool:
        return np.zeros(bgr.shape[:2], bool)
    return select_from_pool(pool, pool_features(bgr, pool), **params)


ALGOS["v8_tuned"] = v8_tuned


# ---------------------------------------------------------------- ellipse clipping (post-process)
def _edge_map(bgr):
    """Canny edges on L and on the a/b chroma channels (white egg on white bag still has a luminance edge or a shadow edge)."""
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB)
    e = np.zeros(bgr.shape[:2], np.uint8)
    for ch in range(3):
        v = np.median(lab[..., ch])
        e |= cv2.Canny(lab[..., ch], max(10, 0.66 * v), max(30, 1.33 * v))
    return cv2.dilate(e, np.ones((3, 3), np.uint8)) > 0


def fit_egg_ellipse(comp, edges, iters=300, rng=None, min_inside=0.85, n_perim=90):
    """RANSAC ellipse fit to the contour of `comp`. The ellipse must lie mostly inside comp, have a plausible
    egg aspect, and its perimeter should sit on image edges. Returns (ellipse, edge_support) or (None, 0)."""
    rng = rng or np.random.default_rng(0)
    cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)[:, 0, :]
    if len(c) < 30:
        return None, 0.0
    best, best_s = None, 0.0
    H, W = comp.shape
    area = comp.sum()
    t = np.linspace(0, 2 * np.pi, n_perim, endpoint=False)
    for _ in range(iters):
        idx = rng.choice(len(c), 6, replace=False)
        try:
            (cx, cy), (w, h), ang = cv2.fitEllipse(c[idx].reshape(-1, 1, 2).astype(np.float32))
        except cv2.error:
            continue
        if min(w, h) < 8 or max(w, h) / min(w, h) > 2.3:
            continue
        ea = np.pi * w * h / 4
        if not (0.3 * area <= ea <= 1.1 * area):
            continue
        em = np.zeros((H, W), np.uint8)
        cv2.ellipse(em, ((cx, cy), (w, h), ang), 1, -1)
        em = em.astype(bool)
        if (em & comp).sum() / max(em.sum(), 1) < min_inside:
            continue
        a_, b_, th = w / 2, h / 2, np.deg2rad(ang)
        px = cx + a_ * np.cos(t) * np.cos(th) - b_ * np.sin(t) * np.sin(th)
        py = cy + a_ * np.cos(t) * np.sin(th) + b_ * np.sin(t) * np.cos(th)
        ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        if ok.mean() < 0.9:
            continue
        support = edges[py[ok].astype(int), px[ok].astype(int)].mean() * ok.mean()
        s = support * (ea / area) ** 0.3      # prefer the larger of two equally edge-supported ellipses
        if s > best_s:
            best_s, best = s, ((cx, cy), (w, h), ang)
    return best, best_s


def clip_to_ellipse(bgr, mask, grow=1.06, min_support=0.45, edges=None, **kw):
    """Eggs are convex: replace each region by region ∩ (slightly enlarged) edge-supported ellipse, when a good ellipse exists."""
    edges = _edge_map(bgr) if edges is None else edges
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros(mask.shape, bool)
    for i in range(1, n):
        comp = lab == i
        if st[i, cv2.CC_STAT_AREA] < 200:
            out |= comp
            continue
        ell, sup = fit_egg_ellipse(comp, edges, **kw)
        if ell is None or sup < min_support:
            out |= comp
            continue
        (cx, cy), (w, h), a = ell
        em = np.zeros(mask.shape, np.uint8)
        cv2.ellipse(em, ((cx, cy), (w * grow, h * grow), a), 1, -1)
        out |= comp & em.astype(bool)
    return out
