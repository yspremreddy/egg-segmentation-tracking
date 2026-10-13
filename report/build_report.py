"""Build report/report.pdf.   python report/build_report.py [DRIVE_LINK]"""
import sys
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
DRIVE = sys.argv[1] if len(sys.argv) > 1 else "https://drive.google.com/file/d/1TFvqUhbgwCOfDZi-pZtQittgcJQT0uzP/view?usp=sharing"
REPO = sys.argv[2] if len(sys.argv) > 2 else "https://github.com/yspremreddy/egg-segmentation-tracking"

ss = getSampleStyleSheet()
B = ParagraphStyle("B", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13, alignment=TA_LEFT, spaceAfter=5)
SM = ParagraphStyle("SM", parent=B, fontSize=8, leading=10.5, spaceAfter=2)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=15, spaceBefore=10, spaceAfter=6, keepWithNext=1)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, spaceBefore=8, spaceAfter=4, keepWithNext=1)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8, leading=10, textColor=colors.HexColor("#444444"), spaceAfter=8)
BUL = ParagraphStyle("BUL", parent=B, leftIndent=12, bulletIndent=2, spaceAfter=2)
TITLE = ParagraphStyle("T", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=20, spaceAfter=4)


def P(t, st=B): return Paragraph(t, st)
def bullets(items): return [Paragraph(t, BUL, bulletText="•") for t in items]


def fig(name, width_cm=17.0, caption=None):
    p = FIG / name
    w, h = PILImage.open(p).size
    out = [Image(str(p), width=width_cm * cm, height=width_cm * cm * h / w)]
    if caption:
        out.append(P(caption, CAP))
    return KeepTogether(out)


def table(rows, widths, header=True, font=8, zebra=True):
    cells = [[Paragraph(str(c), ParagraphStyle("c", parent=SM, fontSize=font, leading=font + 2)) for c in r] for r in rows]
    t = Table(cells, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF5")))
    t.setStyle(TableStyle(st))
    return t


S = []
S += [P("Egg segmentation and tracking - CV intern task", TITLE),
      P(f"Code: {REPO}<br/>Tracked video (Google Drive, public): {DRIVE}", B),
      P("All numbers below are produced by the code in the repository (commands in README.md). Per-run metric files are in <font face='Courier'>experiments/</font>."),
      Spacer(1, 4)]

S += [P("Summary", H2), table([
    ["Task", "Method", "Result"],
    ["1 Pretrained segmentation", "SAM ViT-B automatic masks, egg-selection rules, then re-runs and box prompts for failures", "110/110 images masked; about 85-90% looked clean in a visual check (not quantified); residual errors listed in section 1"],
    ["2 Classical segmentation", "Multi-threshold candidates + seeded GrabCut, ranked by contrast / smoothness / size (v8); 8 logged runs", "dev (90 imgs) IoU 0.666, Dice 0.748"],
    ["3 Evaluation", "IoU, Dice, precision, recall vs SAM masks on a held-out split, run once", "eval (20 imgs) IoU 0.664, Dice 0.776, precision 0.764, recall 0.876"],
    ["4 Video tracking", "Fast classical segmentation + alpha-beta / Hungarian tracker with occlusion coasting and re-identification", "2 persistent IDs; on 39 labelled frames: 64/64 visible eggs localised with the right ID, 0 false positives, 0 ID switches; only 2/13 partly hidden eggs localised"],
], [3.2, 7.3, 6.5]), Spacer(1, 6)]

# ------------------------------------------------------------------ Task 1
S += [P("1. Segmentation with a pretrained model (SAM)", H1)]
S += [P("<b>Model.</b> Segment Anything, ViT-B checkpoint (<font face='Courier'>sam_vit_b_01ec64</font>), no training or fine-tuning. Run on CPU: SAM's automatic mask generator uses float64 internally, which Apple's MPS backend does not support. "
        "Generator settings: 24 points per side, predicted-IoU threshold 0.86, stability threshold 0.90, minimum region 200 px."),
      P("<b>Choosing the egg among SAM's masks.</b> The generator returns every region it finds (hand, bag, pot, ...), so a rule picks the egg. "
        "<i>v1:</i> the largest region that does not touch the image border, covers 0.5-60% of the image and is compact (area / bounding-box area >= 0.55). "
        "Checking all 110 overlays showed two problems, so two corrections were made:"),
      *bullets([
          "<b>Several eggs per image.</b> The brief says 'egg(s)', yet v1 keeps one. <i>v2</i> (30 images re-run, <font face='Courier'>task1b_fix_sam.py</font>) keeps every egg-like region: 0.5-45% of the image, compactness >= 0.6, solidity >= 0.9, off the border, de-duplicated (a region overlapping a larger kept one by more than 50% is dropped), and at least 0.3 x the area of the largest.",
          "<b>Failures</b> where the egg is camouflaged (white egg in a steel pot of water, in leaves, in a white tray, extreme close-up). <i>v3</i> (<font face='Courier'>task1c_box_fix.py</font>): one hand-placed box prompt per egg, SAM predictor, for 13 images (10-13, 28, 42, 44, 116, 117, 124, 135, 140, 145)."]),
      fig("qa_fix2.jpg", 13.5, "Figure 1: images after the box-prompt fix (overlay = SAM mask). Eggs in leaves (10-13) stay ragged because branches cross the egg; #117 (top egg) and #142 (extra blob) remain imperfect."),
      P("<b>Limits.</b> The masks are used as ground truth for Task 3 but are SAM output, not human annotation. Remaining imperfect masks: 10-13 (branches over the egg), 117, 142. "
        "I did not hand-correct them, because that would make the 'ground truth' partly manual rather than model-produced.")]

# ------------------------------------------------------------------ Task 2
S += [P("2. Segmentation with classical image processing", H1)]
S += [P("<b>Dev / eval split and leakage.</b> The dataset is made of photo series (the same steel pot, ash, leaves, marble counter, hospital bag, foil ... with the egg moved between shots). "
        "A first random per-image split would have put near-identical scenes in both parts. It was replaced by a <b>group split</b>: images are clustered into 48 scene groups (HSV colour histogram + 8x8 thumbnail, average-linkage clustering, threshold 1.4) "
        "and whole groups go to dev or eval with a fixed seed: <b>dev 90 / eval 20</b>, no shared group. The ratio is 4.5:1 rather than 4:1 because groups cannot be cut. "
        "The eval split is locked in the code: the runner refuses it without a <font face='Courier'>--final</font> flag, and it was evaluated once."),
      P("<b>Final algorithm (v8)</b>, in four steps:"),
      *bullets([
          "<b>Candidates.</b> Multi-Otsu thresholds (3 and 4 classes) on gray, saturation, value and the Lab a/b channels, both polarities, with morphological open/close; plus candidates from GrabCut seeded with a 4x4 grid of ellipses at 3 sizes (this is what finds a white egg on a white bag, where no threshold works).",
          "<b>Refinement.</b> Each candidate is refined with GrabCut: eroded core = sure foreground, region = probable foreground, ring = probable background, rest = background (3 iterations, half resolution).",
          "<b>Scoring.</b> contrast with a 12 px ring around the region (Lab distance) x area^0.5 / (1 + interior texture / 4). Eggs are smooth, fairly large blobs that stand out from their surroundings.",
          "<b>Selection.</b> Drop regions that touch the border, exceed 45% of the image, or have ellipse fill < 0.65; de-duplicate; keep the best region plus any other with at least 0.7 x its score (this is how several eggs are kept). Parameters tuned on dev only."])]

S += [P("Experiment log (dev split, 90 images, scored against SAM masks)", H2), table([
    ["Run", "What changed and why", "IoU", "Dice", "Prec", "Rec", "Outcome"],
    ["v1", "Gray Otsu, largest bright component. Simplest baseline.", ".371", ".465", ".430", ".741", "baseline"],
    ["v2", "HSV: S<60 and V>150, open/close, fill holes, largest component. Eggs are bright and unsaturated.", ".448", ".544", ".563", ".705", "kept as reference"],
    ["v3", "v2 + keep every shape-valid component (off-border, ellipse fill >= .65, solidity >= .88) to allow several eggs and reject bright walls.", ".329", ".378", ".447", ".339", "<b>discarded</b>: a fixed colour threshold misses pale/shaded eggs, so the shape filter had nothing to accept (median IoU 0)"],
    ["v4", "Candidates from multi-Otsu on 5 channels, both polarities; shape filter; rank by ring contrast; NMS. Removes dependence on one hand-set threshold.", ".442", ".508", ".601", ".456", "precision up, recall low"],
    ["diag", "Oracle: best possible single candidate per image. Best candidate: any component .65; after shape filter (.75/.92) .44; relaxed (.6/.85) .52; no shape filter .60.", "-", "-", "-", "-", "candidate <i>generation</i> is the bottleneck; shape thresholds too strict"],
    ["v5", "+ GrabCut refinement of each candidate; relaxed shape (.6/.85). Pulls boundaries onto the real egg edge.", ".524", ".589", ".670", ".549", "kept"],
    ["v6", "New score: contrast x sqrt(area) / (1 + texture/8). v5 chose small saturated distractors (blue text, red logos, shadow slivers).", ".557", ".617", ".677", ".588", "kept"],
    ["v7", "+ candidates from GrabCut seeded with a grid of ellipses. White egg on white bag never yields a threshold region.", ".627", ".709", ".680", ".830", "kept; 4 s/image"],
    ["v8", "Candidate pool cached, selection parameters tuned by coordinate descent (fill .65, solidity .6, max area .45, contrast >= 5, relative cut-off .7, texture weight 4, size exponent .5). Strict thresholds had rejected occluded or slightly non-convex eggs.", "<b>.666</b>", ".748", ".750", ".831", "<b>final</b>"],
    ["v8b", "+ filter on IoU between region and its fitted ellipse (0.5-0.9), re-tuned.", ".666", "-", "-", "-", "<b>discarded</b>: no threshold helped; the false blobs pass shape tests and win on score"],
    ["clip", "Post-process: clip each region to an edge-supported RANSAC ellipse (9 settings). An oracle ellipse from the SAM mask would give .766.", ".658-.667", "-", ".750-.769", ".805-.831", "<b>discarded</b>: white-on-white egg boundaries have no edges (support 0.02-0.15 on 8/8 inspected cases)"],
], [1.0, 7.0, 1.0, 1.0, 1.0, 1.0, 5.0], font=7.3),
      Spacer(1, 4),
      P("Tuning 7 selection parameters on dev can overfit dev; the held-out eval split in section 3 is the honest number. The oracle on the final pool (best single region per image) is IoU 0.72, so the selection step is already close to that pool's ceiling; further gains need better candidate boundaries rather than better ranking. "
        "Speed: about 4.3 s per image on CPU (about 100 GrabCut runs), which is why Task 4 uses a lighter variant."),
      fig("viz_v8_tuned_dev_worst.jpg", 15, "Figure 2: the 12 worst dev predictions of v8 (green = SAM, red = ours). Typical causes: white egg on white bag/tray, egg in boiling water, GrabCut growing into a large background blob.")]

# ------------------------------------------------------------------ Task 3
S += [P("3. Evaluation and analysis", H1)]
S += [P("Ground truth = SAM masks (section 1). Prediction = v8 run once on the held-out eval split (20 images, 10 scene groups never used for development or tuning). "
        "For several eggs the masks are compared as unions."),
      table([["split", "n", "IoU", "Dice", "Precision", "Recall"],
             ["dev (tuned on)", "90", "0.666", "0.748", "0.750", "0.831"],
             ["<b>eval (held out)</b>", "20", "<b>0.664</b>", "<b>0.776</b>", "<b>0.764</b>", "<b>0.876</b>"]], [4, 1.5, 2, 2, 2.5, 2.5], font=8.5),
      Spacer(1, 4),
      *bullets([
          "Eval median IoU 0.65; 70% of images have IoU > 0.5 and 40% > 0.75 (best #105 0.98, #16 0.97, #102 0.92, #136 0.91).",
          "Dev and eval agree (0.666 vs 0.664): tuning did not visibly overfit dev. With 20 images one image moves the mean by about 0.03, so read the eval number as +-0.05.",
          "Recall (0.88) exceeds precision (0.76): the method finds the egg but tends to over-segment it.",
          "The number of connected regions matched the SAM egg count in 80% of images; multi-egg images (5) scored 0.71, single-egg 0.65, so handling several eggs is not the main weakness."]),
      fig("iou_hist_bars.png", 15.5, "Figure 3: IoU distribution on the eval split (left) and per-image IoU coloured by failure mode (right)."),
      table([["Failure mode (20 eval images)", "Count", "Images"],
             ["good (IoU >= 0.7)", "9", "105, 16, 102, 136, 2, 152, 151, 62, 142"],
             ["leak / over-segmentation (high recall, precision >= 0.15 lower)", "8", "104, 150, 52, 149, 111, 106, 156, 163"],
             ["missed egg(s) (high precision, recall >= 0.15 lower)", "2", "103, 121"],
             ["boundary / mixed", "1", "31"]], [8.5, 1.5, 7], font=8),
      Spacer(1, 4),
      fig("failure_gallery.jpg", 14.5, "Figure 4: the 8 worst eval images: photo | SAM (green) | ours (red)."),
      P("Qualitative analysis", H2),
      *bullets([
          "<b>White egg on a white or bright background leaks into the background</b> (#104, #111, #106, #150, #149). Egg and bag/wall/counter are nearly the same colour, so GrabCut's colour models for egg and background overlap and the foreground spills into whatever is similarly bright. Recall stays near 1.0 while precision drops to 0.3-0.6. This is the biggest error source: 8 of the 11 images below IoU 0.7.",
          "<b>Touching objects get absorbed</b> (#52 earphone cable, #149/#150 pot rim): thin high-contrast structures attached to the egg join the mask although the egg itself is found.",
          "<b>Two-tone egg split by lighting</b> (#31): the shaded half of the egg resembles the red bucket behind it, so only the lit half plus part of the bucket is segmented.",
          "<b>Missed eggs in multi-egg scenes</b> (#103: 1 of 3 white eggs kept, precision 0.93, recall 0.30). Probable cause: my relative-score cut-off (a region is dropped below 0.7 x the best). I did <b>not</b> verify this, because eval images were deliberately not used for diagnosis.",
          "<b>What works:</b> a clean single egg with clear colour or texture contrast (hands, carpet, keyboard, towel) reaches IoU 0.85-0.98."]),
      fig("precision_recall.png", 8.5, "Figure 5: precision vs recall per eval image; leaks sit at high recall / lower precision, missed eggs at high precision / low recall."),
      P("Caveats and possible improvements", H2),
      *bullets([
          "SAM is the reference, so the score measures agreement with SAM; where SAM is wrong (leaves, tray, boiling water) a better classical mask can score worse.",
          "Overlap metrics are boundary-sensitive: a few pixels of leakage cost more on small eggs.",
          "<b>Tried after this evaluation, on dev only:</b> clipping the mask to a fitted ellipse (eggs are convex). The ceiling is real (an oracle ellipse gives dev IoU 0.766 vs 0.666) but an edge-supported fit gave no gain (0.658-0.667), because white-on-white egg boundaries contain no edges. The eval numbers above come from v8 and were not re-run.",
          "Not implemented: a cue that is not edge- or colour-based for the white-on-white case; a per-region test instead of the relative cut-off for extra eggs; a light/shade-invariant colour space for two-tone eggs; fewer seed ellipses for speed."])]

# ------------------------------------------------------------------ Task 4
S += [P("4. Video tracking", H1)]
S += [P(f"Video: youtube.com/watch?v=Y4JyV94y2_A ('Raw and Boiled Egg Spinning', Mocomi), 60 s, 24 fps, 1440 frames, used at 640x360. Title card, then a table scene with two white eggs and a hand spinning them (frames 221-1005), then text cards. "
        f"<b>Output video (Google Drive):</b> {DRIVE}"),
      P("<b>Segmentation (<font face='Courier'>video_seg.py</font>, about 5 ms/frame).</b> v8 takes 4 s per image, far too slow for 1440 frames, so the video uses a light variant of the same ideas:"),
      *bullets([
          "colour mask (V > 150, S < 70), opening with a 13 px ellipse (also strips thin caption and arrow strokes touching an egg), closing 9 px;",
          "each component must pass an egg-shape test: 0.3-12% of the frame, off the border, solidity >= 0.88, ellipse fill >= 0.72, aspect <= 2;",
          "a failing blob is rescued by a watershed split on the distance transform (two touching eggs), then by progressively larger openings (15/21/27 px) that strip attachments such as fingers;",
          "a region must be much brighter than its surroundings (mean V >= 200 and >= 100 above the ring around it, the same idea as the ring contrast in v8). This removed every false positive (bright wall, white shirt checks, finger highlights) that colour and shape let through."]),
      P("<b>Tracking (<font face='Courier'>video_track.py</font>).</b>"),
      *bullets([
          "Each track has a constant-velocity (alpha-beta) motion model; detections are matched to predicted positions with the Hungarian algorithm, gated by distance and size. A track is confirmed after 3 consecutive detections.",
          "<b>Occlusion:</b> a confirmed track that loses its detection coasts on its prediction for up to 24 frames (dashed circle labelled 'occluded'). Leftover detections may claim a coasting track through a larger gate (400 px), so an egg flicked while hidden can reappear far away.",
          "<b>Re-identification:</b> a track that stays lost becomes dormant; a newly confirmed egg takes the ID of the nearest dormant track instead of a new number. The two eggs look identical, so last-seen position is the only cue.",
          "Each track remembers the egg's full size (slowly decaying maximum) so a partly hidden egg can later match the full egg.",
          "Scene-end reset: with no egg visible, a frame far from the last egg-bearing frame (mean abs difference > 30; hand occlusion stays below 10, the fade is above 45) clears the tracker, so no markers sit on the text cards."]),
      fig("track_check3.jpg", 16.5, "Figure 6: tracked frames (f300, 460, 552, 600, 712, 720, 745, 800, 950, 1003, 1010, 1030). Coloured mask and label = visible egg; dashed circle = occluded; trail = recent path.")]

S += [P("Iterations", H2), table([
    ["Step", "Problem found", "Fix"],
    ["fast segmentation, largest bright blobs", "3 false eggs per frame (wall, shirt)", "border rule + ring brightness contrast"],
    ["touching eggs", "no detection on f300-420 (merged blob)", "watershed split"],
    ["first tracker", "6 IDs for 2 eggs (tracks died after 40 hidden frames, new ID each time)", "dormant tracks + re-identification by last position"],
    ["after re-ID", "3 IDs: an egg first seen as a 930 px sliver, then 4500 px, failed the size gate", "track full-egg size, size gate 0.12-1.8"],
    ["caption 'Raw Egg'", "egg lost f330-360 (white text touches the egg)", "13 px opening strips thin strokes"],
    ["partly hidden egg", "both eggs missed at f712-720 (second egg attached)", "progressive-opening fallback"],
    ["end of scene", "ghost 'occluded' circles over the text card", "scene-end reset"],
], [4, 8, 5], font=8), Spacer(1, 4)]

S += [P("Result", H2),
      P("Coverage over the whole table scene (787 frames): both eggs segmented in 546 frames (69%), one in 222 (28%), none in 19 (2%); exactly 2 persistent IDs (Egg 1, Egg 2), no spurious IDs, no markers after the scene ends. "
        "There are no ground-truth labels for this video, so I labelled egg centres on <b>39 raw frames (every 20th, f230-f990)</b> taken from the original video, not from the tracker output. "
        "<b>The labels were made by Claude (an AI) viewing the frames, not by an independent human; positions are eyeballed to about 5 px, and identity (egg A = left egg at the start, B = its neighbour) was followed by position continuity.</b> "
        "A tracker output matches a labelled egg if it is within 40 px (about one egg radius)."),
      table([["Measure", "Result"],
             ["Fully visible egg instances (64) localised by a visible-state track", "100% (64/64)"],
             ["... with the correct ID", "100% (64/64)"],
             ["False-positive visible detections", "0"],
             ["ID switches between consecutive matched labelled frames", "0"],
             ["Partly hidden / in-hand egg instances (13) localised", "15% (2/13)"],
             ["Fully hidden egg (1) given an 'occluded' track of the right ID", "1/1"]], [11, 6], font=8.5),
      Spacer(1, 4),
      P("Limitations", H2),
      *bullets([
          "<b>Hand occlusion is the weak point.</b> When a hand holds or covers an egg, the egg is not segmented and a track is carried for only 24 frames. Hand-held spells longer than about a second leave no track for that egg (A at f510-550 and f690-710). The coasting marker is a constant-velocity guess, so when the hand carries the egg it can sit far from the truth (B at f770: marker at (259, 97), egg at (375, 218)). A hand-held state or a longer position-held coast would be the next step; not implemented.",
          "The centre error of fully visible eggs is about 1 px, which is below my label precision and mostly concerns stationary eggs; read it as 'within label precision'. The 0 ID switches cover only labelled frames 20 apart, so a swap in between (for example around f730 where the eggs overlap behind the fingers) would not be caught.",
          "Two identical-looking eggs: after both are hidden or swapped under the hand, the ID is a position-based guess.",
          "Thresholds (bright unsaturated egg on a dark table) are tuned to this video; a coloured egg or bright background would need re-tuning or the heavier v8 pipeline. The video was fetched at 360p, the only format available, and the output has the same resolution."])]

# ------------------------------------------------------------------ AI
S += [P("AI tool disclosure", H1),
      P("Claude Code (Anthropic's coding agent, model Claude Sonnet 5.5) was used throughout this project. It wrote the functions and scripts in <font face='Courier'>src/</font>, ran the experiments and parameter sweeps, "
        "generated the contact sheets and plots, hand-labelled the 39 video frames used in the tracking evaluation, and drafted the text of this report from the logged results. "
        "The methods, parameters and numbers described here are those in the repository; they can be re-run with the commands in README.md.")]


def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 8); c.setFillColor(colors.grey)
    c.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"page {d.page}"); c.restoreState()


doc = SimpleDocTemplate(str(HERE / "report.pdf"), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                        title="Egg segmentation and tracking - CV intern task", author="")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("wrote", HERE / "report.pdf")
