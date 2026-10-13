# Task 2 - Experiment log (classical egg segmentation)

All numbers: **dev split only** (90 images), scored against the SAM masks (Task 1) with
mean IoU / Dice / precision / recall over images. The eval split (20 images) has not been
touched; the runner refuses to evaluate it without `--final`.
Code: `src/seg_classical.py`, runner `src/run_experiment.py`, per-run JSON in `experiments/`.

## Split (leakage control)
The first split was a random per-image shuffle (`data/splits_v1_random.json`, kept for the record).
Looking at the data showed it is made of photo *series* (same steel pot / ash / leaves / marble
counter / hospital bag / foil, egg moved between shots), so a random split would put near-identical
scenes in both dev and eval. Replaced with a **group split**: images clustered into 48 scene groups
(HSV histogram + 8x8 thumbnail, average-linkage, threshold 1.4), whole groups assigned 80/20 with
seed 42 -> dev 90 / eval 20, zero shared groups. (`src/make_split.py`, `data/scene_groups.json`.)
Ratio is 4.5:1 rather than exactly 4:1 because groups cannot be cut.

## Runs

| # | Change | Why | Params | Dev IoU | Dice | Prec | Rec | Kept? |
|---|--------|-----|--------|---------|------|------|-----|-------|
| v1 | Gray Otsu, largest bright component | simplest baseline | blur 5 | 0.371 | 0.465 | 0.430 | 0.741 | baseline |
| v2 | HSV: S<60 & V>150, open/close 9px, fill holes, largest CC | eggs are bright + unsaturated | S<60,V>150 | 0.448 | 0.544 | 0.563 | 0.705 | kept as reference |
| v3 | v2 + keep every shape-valid component (off-border, ellipse fill>=.65, solidity>=.88) | "largest CC" picked bright walls/bags; want multi-egg + shape prior | as left | 0.329 | 0.378 | 0.447 | 0.339 | **discarded** - shape filter only helps if the right candidate exists; fixed colour threshold misses pale/cream/shaded eggs, many empty outputs (median IoU 0) |
| v4 | Candidate generator: multi-Otsu (3 and 4 class) on gray,S,V,a,b, both polarities; shape filter; rank by LAB contrast with a 12px ring; NMS | stop depending on one hand-set threshold | fill>=.75, solid>=.92, contrast>=15 | 0.442 | 0.508 | 0.601 | 0.456 | partially - precision up, but recall low |
| diag | Oracle analysis on v4 candidates: best-possible single-candidate IoU | find whether generation or selection is the bottleneck | - | any CC 0.65 -> +shape(.75/.92) 0.44 -> relaxed(.6/.85) 0.52 -> no shape filter 0.60 | | | | Generation is the bottleneck; shape thresholds too strict |
| v5 | + GrabCut refinement of each candidate (core=sure FG, mask=prob FG, ring=prob BG), relaxed shape (.6/.85) | pull boundaries onto real egg edge, recover partly-correct candidates | 3 iters, 0.5x scale, topn 6 | 0.524 | 0.589 | 0.670 | 0.549 | kept |
| v6 | New score = contrast x sqrt(area) / (1 + interior texture/8) | v5 picked small saturated distractors (blue text, red logos, shadow slivers) | tex_k 8 | 0.557 | 0.617 | 0.677 | 0.588 | kept |
| v7 | + extra candidates from GrabCut seeded with a 4x4 grid x 3 radii of ellipses | white/cream egg on white/silver bg never yields a threshold region | centres .2-.8, r .10/.17/.25 | 0.627 | 0.709 | 0.680 | 0.830 | kept (4.0 s/img - slow) |

## Observations after v7
* Recall jumped (0.59 -> 0.83) but precision did not: seeded GrabCut can grow into large
  background regions (steel pot interior, bag, tray) that are then ranked above the real egg because the score
  rewards area. Worst cases: 119, 40, 158, 123, 147, 125.
* Next: cache the candidate pool once, then tune only the *selection* step (shape strictness, max area,
  contrast threshold, area exponent, relative cut-off for extra eggs) so each trial is seconds, not minutes.

## Selection tuning (cached candidate pool)
`src/build_pool_cache.py` caches, per dev image, every candidate (threshold + seeded-GrabCut, each also
GrabCut-refined; ~14-22 regions) with shape/contrast/texture features. `src/tune.py` then scores
selection rules in ~1 s per setting. Sensitivity sweep on the base rule (fill .65, solidity .88, max area .35,
contrast>=15, rel .5, tex_k 8, size^0.5): solidity .93 hurts badly (0.53 IoU, recall .66), solidity .8 helps (0.61),
max area .45 helps (0.587), rel .8 raises precision (0.70) at the cost of recall (0.72).

| # | Change | Why | Dev IoU | Dice | Prec | Rec | Kept? |
|---|--------|-----|---------|------|------|-----|-------|
| base | v7 pool + strict selection | reference for tuning | 0.576 | - | 0.642 | 0.798 | - |
| v8 | coordinate descent over {fill, solidity, max area, min contrast, relative cut-off, texture weight, size exponent}, 3 rounds | strict thresholds rejected good (occluded / slightly non-convex) eggs | **0.666** | 0.748 | 0.750 | 0.831 | **kept - current final**. Params: fill .65, solidity .6 (effectively off), max area .45, contrast>=5, rel .7, tex_k 4, size^0.5 |
| v8b | add region-vs-fitted-ellipse IoU filter (0.5-0.9) and re-tune | worst cases were big irregular blobs | 0.666 | - | - | - | **discarded** - no threshold improved IoU; the blobs pass shape tests and win on score, not shape |
| diag | oracle: best single region in the pool per image | is ranking or generation the limit? | mean 0.723, median 0.762, 61% of images have a candidate with IoU>0.7 | | | | selection (0.666) is already close to the pool ceiling -> further gains need better candidate boundaries, not ranking |

Note: dev-set tuning of 7 parameters can overfit dev; the eval split gives the honest number (Task 3).

## Known failure modes of v8 (for Task 3 write-up)
* White/cream egg on a white or silver background (hospital bag, foil, tray, bright wall): low contrast, region grows into background.
* Egg inside a steel pot of water: reflections / boiling surface, bubbles break the boundary (124, 135, 145 are hardest).
* Seeded GrabCut growing into large textured background blobs that outrank a small egg (108, 127, 22).
* Eggs partially hidden by leaves (10-13) or by hand (20): ragged or fragmented masks.
* Speed: 6 s / image on CPU (pool of ~100 GrabCuts) - too slow for video, needs a fast variant for Task 4.

## Post-process attempt: clip to an edge-supported ellipse (dev only; eval untouched)
Motivation: 8 of 20 eval failures were the mask leaking past the egg, and eggs are convex. Ceiling test on dev: clipping each
predicted region to an ellipse fitted to the *SAM* mask would raise dev IoU 0.666 -> 0.766 (26/90 images improve by > 0.1).
Real method (`fit_egg_ellipse` / `clip_to_ellipse` in `src/seg_classical.py`, tuned with `src/tune_clip.py`): RANSAC ellipse fit to the region
contour; ellipse must lie >= 85 % inside the region, aspect <= 2.3, and its perimeter must sit on Canny edges (L, a, b channels);
result = region ∩ (ellipse x grow).

| variant | dev IoU | Prec | Rec |
|---------|---------|------|-----|
| v8 (no clip) | 0.666 | 0.750 | 0.831 |
| clip, min edge support 0.30 / 0.45 / 0.60, grow 1.0 / 1.06 / 1.12 (9 settings) | 0.658 - 0.667 | 0.750 - 0.769 | 0.805 - 0.831 |

**Discarded - no gain.** Visualising the leak cases showed why: the failures are white/cream eggs on white bags, trays or bright walls, where
the egg boundary has almost no edge (edge support 0.02-0.15 on 8 of 8 inspected cases), so the edge-supported fit cannot tell a correct
ellipse from a wrong one; the support threshold then switches the clip off exactly where it is needed. The 0.10 ceiling is real, but reaching
it needs a cue that is not edge- or colour-based. Because this was developed and rejected on dev, v8 and its single eval run are unchanged.
