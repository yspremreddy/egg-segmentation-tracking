# Egg segmentation and tracking (CV intern task)

Four tasks on the Roboflow "egg" dataset (110 photos) and one YouTube video:

| Task | What | Code | Main result |
|------|------|------|-------------|
| 1 | Egg segmentation with a pretrained model (SAM ViT-B) - used as ground truth | `src/task1_sam_segment.py`, `task1b_fix_sam.py`, `task1c_box_fix.py` | 110 masks in `outputs/sam_masks/` |
| 2 | Egg segmentation with classical image processing, 8 logged experiments | `src/seg_classical.py`, `run_experiment.py`, `tune.py` | dev IoU 0.666 |
| 3 | Evaluation against the SAM masks | `src/evaluate_task3.py` | held-out eval IoU 0.664, Dice 0.776 |
| 4 | Segment + track individual eggs in a video | `src/video_seg.py`, `src/video_track.py`, `eval_tracking.py` | 2 persistent IDs, 0 ID switches on 39 labelled frames |

**Report (PDF):** [`report/report.pdf`](report/report.pdf).  **Tracked video:** https://drive.google.com/file/d/1TFvqUhbgwCOfDZi-pZtQittgcJQT0uzP/view?usp=sharing

## Setup

Python 3.12+ (developed on 3.14, macOS, CPU only - SAM uses float64 internally, which Apple's MPS does not support).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Two things are downloaded, not stored in the repo:

```bash
# 1) SAM ViT-B weights (375 MB)
mkdir -p checkpoints
curl -L -o checkpoints/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# 2) Dataset: https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1  (CC BY 4.0)
#    Download as "Folder structure" and unzip into ./egg.v1i.folder/ (it contains train/ valid/ test/)
python src/prepare_data.py            # pools all 110 images into data/images/ (Roboflow's own split is ignored)
```

## Reproduce

**Split (Task 2).** The photos come in series of the same scene, so a random split leaks. Images are clustered into 48 scene groups and whole
groups go to dev (90 images) or eval (20 images). Already committed as `data/splits.json`; regenerate with `python src/make_split.py`.

**Task 1 - SAM masks** (CPU: ~50 min for step 1, ~16 min for step 2; resumable):
```bash
python src/task1_sam_segment.py --threads 2     # v1: one egg per image  -> outputs/sam_masks/
python src/task1b_fix_sam.py                    # v2: all eggs in multi-egg images + failures
python src/task1c_box_fix.py                    # v3: hand-placed box prompts for 13 hard images
```
The committed `outputs/sam_masks/` is the final result of all three steps, so you can skip this and go straight to Task 2.

**Task 2 - classical algorithms** (the eval split is locked: the runner refuses it without `--final`):
```bash
python src/run_experiment.py --algo v1_otsu --split dev      # ... v2_hsv, v3_hsv_shape, v4_multi_candidate, v5_grabcut, v6_scored, v7_seeded
python src/build_pool_cache.py dev                           # caches candidate regions for fast tuning (~10 min)
python src/tune.py                                           # selection-parameter search -> experiments/tuned_params.json (committed)
python src/run_experiment.py --algo v8_tuned --split dev --save
```
The experiment log (every run, parameters, dev results, why kept or discarded) is `report/experiment_log.md`.

**Task 3 - evaluation** (run once, on the held-out split):
```bash
python src/run_experiment.py --algo v8_tuned --split eval --final --save
python src/evaluate_task3.py        # metrics, plots, failure gallery -> outputs/eval/
```
Write-up: `report/task3_analysis.md`.

**Task 4 - video** (the source video is not redistributed; download it yourself):
```bash
mkdir -p data/video_src
yt-dlp --extractor-args "youtube:player_client=android" -f 18 \
       -o "data/video_src/egg_video.mp4" "https://www.youtube.com/watch?v=Y4JyV94y2_A"
python src/video_track.py data/video_src/egg_video.mp4 outputs/video/egg_tracked.mp4   # also writes egg_tracked.csv
python src/eval_tracking.py outputs/video/egg_tracked.csv                              # vs data/track_gt.json
```
If `yt-dlp` cannot fetch format 18, any 640x360 copy of the video works. Write-up: `report/task4_video.md`.

## Repository layout
```
src/            all code (see table above; seg_classical.py holds the Task 2 algorithms)
data/           splits.json, scene_groups.json, track_gt.json (hand-labelled tracking frames)
experiments/    per-run metric JSON for every algorithm version, tuned_params.json
outputs/        sam_masks/ (Task 1), classical_masks/eval/ (Task 2 predictions), eval/ (Task 3 plots), video/egg_tracked.csv
report/         experiment_log.md, task3_analysis.md, task4_video.md, report.pdf
```

## Notes and limitations
* "Ground truth" for Task 3 is SAM output, not human annotation, so IoU measures agreement with SAM.
* The eval split (20 images) was used once for the final number; an ellipse-clipping idea tried afterwards was developed and rejected on dev only.
* The tracking evaluation uses 39 frames labelled by Claude (an AI), not an independent human; see `report/task4_video.md`.

## AI tool disclosure
Claude Code (Anthropic) was used throughout: it wrote the functions and scripts in `src/`, ran the experiments, hand-labelled the 39 video frames,
and drafted the report text. See the report for details.
