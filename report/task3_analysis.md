# Task 3 - Evaluation and analysis

Ground truth = SAM masks (Task 1). Prediction = final classical method **v8** (Task 2), run **once** on the
held-out **eval split (20 images, 10 scene groups never seen during development/tuning)**.
Reproduce: `python src/run_experiment.py --algo v8_tuned --split eval --final --save && python src/evaluate_task3.py`.

## Quantitative results

| split | n | IoU | Dice | Precision | Recall |
|-------|---|-----|------|-----------|--------|
| dev (tuned on) | 90 | 0.666 | 0.748 | 0.750 | 0.831 |
| **eval (held out)** | 20 | **0.664** | **0.776** | **0.764** | **0.876** |

* Eval median IoU 0.65; 70 % of images have IoU > 0.5, 40 % have IoU > 0.75 (best: #105 0.98, #16 0.97, #102 0.92, #136 0.91).
* Dev and eval agree (0.666 vs 0.664), so tuning 7 selection parameters on dev did not visibly overfit it.
  (n=20 is small: one image moves mean IoU by ~0.03, so treat the number as +-0.05.)
* Recall (0.88) > precision (0.76): the method **finds** the egg but tends to **over-segment** it.
* Egg counting: predicted number of connected regions equals the SAM number in 80 % of images.
  Multi-egg images (5) score as well as single-egg (0.71 vs 0.65), so handling several eggs is not the main weakness.
* Plots: `outputs/eval/iou_hist_bars.png`, `precision_recall.png`, `dev_vs_eval.png`; per-image table `outputs/eval/eval_table.json`.

## Failure-mode breakdown (20 eval images)
| mode | count | rule | images |
|------|-------|------|--------|
| good (IoU >= 0.7) | 9 | | 105, 16, 102, 136, 2, 152, 151, 62, 142 |
| leak / over-segmentation | 8 | recall high, precision >= 0.15 lower | 104, 150, 52, 149, 111, 106, 156, 163 |
| missed egg(s) | 2 | precision high, recall >= 0.15 lower | 103, 121 |
| boundary / mixed | 1 | | 31 |

## Qualitative analysis (`outputs/eval/failure_gallery.jpg`: image | SAM | ours)
1. **White egg on white/bright background -> mask leaks into the background** (#104, #111, #106, #150, #149).
   The egg and bag/wall/counter have almost the same colour, so GrabCut's colour models for "egg" and "background"
   overlap and the foreground region spills into whatever is similarly bright (bag, shadow under the egg, wall).
   Recall stays ~1.0 while precision drops to 0.3-0.6. This is the single biggest error source (8 of 11 non-good images).
2. **Occluding object glued to the egg** (#52 earphone cable, #149/#150 pot rim): thin high-contrast structures touching the egg
   are absorbed into the mask (IoU 0.43-0.44 although the egg itself is found).
3. **Two-tone egg split by lighting** (#31): a shaded half of the egg looks like the red bucket, so only the lit half
   plus part of the bucket is segmented. A single colour/threshold model cannot treat light and shade sides as one object.
4. **Missed eggs in multi-egg scenes** (#103, #121): only 1 of 3 white eggs was kept in #103 (precision 0.93, recall 0.30).
   The other eggs are white-on-grey-marble like the one found, so the likely cause is my relative-score cut-off
   (a region is dropped if its score is < 0.7 x the best) rather than a missing candidate. I have **not** verified this
   on the eval pool because eval images were deliberately not used for tuning/diagnosis.
5. What works: clean single eggs with a clear colour/texture contrast to the background (hands, carpet, keyboard, towel) reach IoU 0.85-0.98.

## Caveats on the metric
* "Ground truth" is SAM, not human annotation. SAM itself is imperfect on a few images (egg in leaves, tray, boiling water; see Task 1),
  so IoU against SAM is a measure of *agreement with SAM*, not absolute accuracy. Where SAM is wrong, a better classical mask can score worse.
* Overlap metrics are boundary-sensitive: a few pixels of leakage costs more on small eggs.

## Potential improvements (not implemented)
* Replace per-image GrabCut colour models with a **shadow/edge-aware** cue (e.g. gradient-weighted GrabCut or a distance-from-ellipse
  penalty) so the mask cannot grow past a strong egg contour into similarly-coloured background (targets failure 1-2).
* After picking the best region, **fit an ellipse and clip the mask to a slightly dilated ellipse**; eggs are convex. *Tried on dev after
  this evaluation* (edge-supported RANSAC fit): no gain (0.658-0.667 vs 0.666), because white-on-white egg boundaries have no edges to support
  the fit; an oracle ellipse from the SAM mask would give 0.766, so the idea is sound but needs a non-edge cue. See `experiment_log.md`. The eval numbers above are from v8 and were not re-run.
* Replace the single relative cut-off with a per-region test (contrast, ellipse fit) so extra eggs are not discarded (failure 4).
* Light/shade-invariant colour space (chromaticity) for two-tone eggs (failure 3).
* Speed: ~4.3 s/image on CPU (about 100 GrabCut runs per image); needs a lighter variant for video (Task 4).
