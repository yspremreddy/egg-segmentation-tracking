# Task 4 - Video tracking

Video: https://www.youtube.com/watch?v=Y4JyV94y2_A ("Raw and Boiled Egg Spinning", Mocomi) - 60 s, 24 fps, 1440 frames, downloaded at 640x360.
Scenes: title card (f0-220), table scene with two white eggs and a hand (f221-~1005), fade to text cards (f~1005-1440).
Output: `outputs/video/egg_tracked.mp4` (+ per-frame `egg_tracked.csv`: frame, id, state, cx, cy, area).
Reproduce: `python src/video_track.py data/video_src/egg_video.mp4 outputs/video/egg_tracked.mp4`  (runs in seconds on CPU).

## Method
**1. Segmentation (`src/video_seg.py`).** The Task 2 method (v8: ~100 GrabCut runs, 4.3 s/image) is far too slow for 1440 frames,
so the video uses a light variant of the same ideas (~5 ms/frame):
* colour mask: bright and unsaturated (V>150, S<70) -> opening with a 13 px ellipse (this also removes thin caption/arrow strokes
  that touch an egg, e.g. "Raw Egg") -> closing 9 px
* each connected component must pass an egg-shape test: area 0.3-12 % of frame, not touching the border, solidity >= 0.88,
  ellipse fill >= 0.72, aspect <= 2
* a component that fails is rescued in two ways: watershed split on the distance transform (two touching eggs), then progressively
  larger openings (15/21/27 px) to strip attachments such as fingers or a second half-hidden egg
* brightness-contrast check as in v8's ring contrast: mean V inside >= 200 and >= 100 above the ring around it. This removed all false
  positives (bright wall patch, white shirt checks, finger highlights) that colour + shape alone let through.

**2. Tracking (`src/video_track.py`).**
* every track has a constant-velocity (alpha-beta) motion model; detections are matched to predicted positions with the Hungarian algorithm,
  gated by distance and by size
* a track is confirmed only after 3 consecutive detections (kills 1-frame false positives)
* **occlusion handling:** a confirmed track that loses its detection (hand in front, merged blob) *coasts* on its prediction for up to 24 frames
  (drawn as a dashed circle labelled "occluded"). Leftover detections may claim a coasting track through a much larger gate (400 px) - an egg
  flicked while hidden can reappear far away.
* **re-identification:** a track that stays lost becomes *dormant*; when a new egg is confirmed it takes the ID of the nearest dormant track
  instead of a new number. Eggs look identical, so last-seen position is the only cue.
* each track remembers its full-egg size (slowly decaying maximum) so a partly hidden egg (small sliver) can later match the full egg.
* scene-end reset: if no egg is visible and the frame is far from the last egg-bearing frame (mean abs diff > 30; hand occlusion stays < 10,
  the fade is > 45) the tracker is cleared, so no ghost markers sit on the text cards.

## Iterations (what failed and why)
| step | result | problem found | fix |
|------|--------|---------------|-----|
| fast segmentation, largest bright blobs | 3 false eggs per frame | bright wall/shirt | border rule + ring brightness contrast |
| touching eggs | n=0 detections on f300-420 | merged blob fails shape test | watershed split |
| first tracker | 6 IDs for 2 eggs | tracks died after 40 hidden frames; new ID every reappearance | dormant tracks + re-ID by last position |
| after re-ID | 3 IDs | egg re-seen as a 930 px sliver, then full 4500 px -> area-ratio gate rejected it | track full-egg size, gate ratio 0.12-1.8 |
| caption "Raw Egg" | egg lost f330-360 | white text touches the egg, merged blob | 13 px opening strips thin strokes |
| partially hidden egg | both eggs missed at f712-720 | second egg attached to first | opening fallback |
| end of scene | ghost "occluded" circles over text card | coasting tracks never dropped | scene-end reset |

## Result: coverage over the whole table scene
* exactly **2 persistent IDs** (Egg 1, Egg 2) over the whole table scene; no spurious IDs, no markers after the scene ends
* table-scene frames (f223-1009, 787 frames): both eggs segmented in **546 (69 %)**, one in 222 (28 %), none in 19 (2 %); the rest of the time
  the missing egg is carried by the occlusion model
* spot-checked ~30 frames by eye across the final iterations: IDs stay with the correct egg through the flick (Egg 1 sent right at ~f550), spinning and hand-holds.

## Limitations
* Two identical-looking eggs: if both are hidden at once or swapped under the hand, the ID assignment after reappearance is a position-based
  guess and could be wrong; I saw no swap in the spot checks, but I cannot prove there is none.
* When a hand covers part of an egg the mask is a fragment (e.g. f745); the mask is then too small although the track is correct.
* The thresholds (bright + unsaturated egg on a dark table) are tuned to this video; a coloured egg or bright background would need re-tuning
  or the heavier v8 pipeline.
* Video was downloaded at 360p (the only format YouTube served to yt-dlp's android client); the output has the same resolution.

## Quantitative check against hand-labelled frames
No ground-truth labels exist for this video, so I labelled egg centres on **39 raw frames (every 20th frame, f230-f990)** from the original video, not the
tracker overlay (`data/track_gt.json`). **Caveat: the labels were made by Claude (an AI) by viewing the frames, not by an independent human; positions
are eyeballed to about +-5 px, and identity (egg A = the left egg at the start, B = its neighbour) was followed by position continuity.**
Run: `python src/eval_tracking.py outputs/video/egg_tracked.csv`. A tracker output counts as a match if it is within 40 px (about one egg radius).

| | result |
|---|---|
| Fully visible egg instances (64) localised by a visible-state track | **100 % (64/64)** |
| ... with the correct ID | **100 % (64/64)** |
| False-positive visible detections (not near any labelled egg) | **0** |
| ID switches between consecutive matched labelled frames | **0** |
| Partly hidden / in-hand egg instances (13) localised | **15 % (2/13)** |
| Fully hidden egg instance (1) given an 'occluded' track of the right ID | 1/1 |

The centre error of the fully visible eggs is about 1 px, but that is below my label precision and most of those eggs are stationary, so
treat it as "within label precision", not as a measured sub-pixel accuracy. 0 ID switches only covers frames 20 apart: a swap during the 20
frames in between (e.g. around f730 where the two eggs overlap behind the fingers) would not be caught.

**Where it fails (the 14 non-fully-visible cases):** whenever a hand holds or covers an egg, the egg is not segmented, and a track is only carried
for 24 frames. Hand-held spells longer than about a second leave no track for that egg at all (A at f510-550 and f690-710). The coasting marker
is a constant-velocity guess, so when the hand carries the egg it can sit far from the truth (B at f770: marker at (259, 97), egg at (375, 218)).
Improving this would need a hand-held state or a longer, position-held coast; I did not implement it.
