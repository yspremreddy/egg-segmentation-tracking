"""Task 4 tracking evaluation against the hand-labelled frames (data/track_gt.json, 39 frames, stride 20).
   python src/eval_tracking.py outputs/video/egg_tracked.csv
Mapping used: GT egg A <-> tracker 'Egg 1', GT egg B <-> 'Egg 2' (fixed at f230 where the tracker starts).
A tracker output matches a GT egg if it is within 40 px of the labelled centre (~ egg radius)."""
import csv, json, sys
from collections import defaultdict
import numpy as np

GATE = 40.0
gt = json.load(open("data/track_gt.json"))["frames"]
rows = defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    rows[int(r["frame"])].append((int(r["id"]), r["state"], float(r["cx"]), float(r["cy"])))
MAP = {"A": 1, "B": 2}

stats = defaultdict(list); errs = []; fp = 0; fp_frames = []; idmatch_seq = defaultdict(list)
for f, eggs in sorted(gt.items(), key=lambda kv: int(kv[0])):
    f = int(f) if not isinstance(f, int) else f
    vis_out = [o for o in rows[f] if o[1] == "visible"]
    for name, (x, y, st) in eggs.items():
        if st == "hid":
            stats["hid_total"].append(1)
            occ = [o for o in rows[f] if o[0] == MAP[name] and o[1] == "occluded"]
            stats["hid_has_occluded_track"].append(1 if occ else 0)
            continue
        key = "vis" if st == "vis" else "part"
        d = [(np.hypot(o[2] - x, o[3] - y), o) for o in vis_out]
        d = [t for t in d if t[0] <= GATE]
        stats[key + "_total"].append(1)
        if d:
            dist, o = min(d, key=lambda t: t[0])
            stats[key + "_matched"].append(1)
            stats[key + "_idcorrect"].append(1 if o[0] == MAP[name] else 0)
            idmatch_seq[name].append((f, o[0]))
            if key == "vis": errs.append(dist)
        else:
            stats[key + "_matched"].append(0)
            # was the egg at least carried by an occluded (coasting) track of the right id?
            occ = [oo for oo in rows[f] if oo[0] == MAP[name] and oo[1] == "occluded" and np.hypot(oo[2] - x, oo[3] - y) <= 3 * GATE]
            stats[key + "_coasted_nearby"].append(1 if occ else 0)
    all_pts = [(e[0], e[1]) for e in eggs.values()]
    for o in vis_out:
        if all(np.hypot(o[2] - px, o[3] - py) > GATE for px, py in all_pts):
            fp += 1; fp_frames.append(f)

def pct(k, tot):
    a = stats[k]; t = stats[tot]
    return f"{100 * sum(a) / max(len(t), 1):.0f}% ({sum(a)}/{len(t)})"
n_frames = len(gt)
print(f"labelled frames: {n_frames}  | fully visible egg instances: {len(stats['vis_total'])}, partly hidden: {len(stats['part_total'])}, hidden: {len(stats['hid_total'])}")
print("Fully visible eggs: localised (visible track within 40 px):", pct("vis_matched", "vis_total"))
print("   ... correct ID among those localised:", pct("vis_idcorrect", "vis_matched"))
print("   mean centre error of matched fully-visible eggs: %.1f px (median %.1f)" % (np.mean(errs), np.median(errs)))
print("Partly hidden eggs: localised:", pct("part_matched", "part_total"), "| correct ID among localised:", pct("part_idcorrect", "part_matched"))
unm = [i for i, m in zip(range(len(stats['part_matched'])), stats['part_matched']) if not m]
print("   of the not-localised partly hidden eggs, carried by a coasting track of the right ID:", f"{sum(stats['part_coasted_nearby'])}/{len(stats['part_coasted_nearby'])}")
print("Hidden eggs with an 'occluded' track of the right ID:", f"{sum(stats['hid_has_occluded_track'])}/{len(stats['hid_total'])}")
print("False-positive visible detections (not near any labelled egg):", fp, "in frames", fp_frames)
sw = 0
for name, seq in idmatch_seq.items():
    for (f0, i0), (f1, i1) in zip(seq, seq[1:]):
        sw += i0 != i1
print("ID switches between consecutive matched labelled frames:", sw)
