"""Task 4 - segment + track individual eggs in a video.

    python src/video_track.py data/video_src/egg_video.mp4 outputs/video/egg_tracked.mp4

Pipeline per frame:
  1. video_seg.egg_candidates   -> one mask per egg-like region (fast classical segmentation)
  2. tracker                    -> persistent IDs
       * constant-velocity prediction (alpha-beta filter) for every track
       * Hungarian assignment detections <-> predicted positions, gated by distance and size ratio
       * a new track is only confirmed after `min_hits` consecutive detections (kills 1-frame false positives)
       * a confirmed track that loses its detection (hand occlusion, merged blob) COASTS on its predicted
         position for up to `max_missed` frames, then is dropped
Outputs the rendered video and a per-frame CSV (frame, id, state, cx, cy, area).
"""
import csv
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
import video_seg

COLORS = [(60, 76, 231), (219, 152, 52), (113, 204, 46), (15, 196, 241), (182, 89, 155), (156, 188, 26)]  # BGR


class Track:
    def __init__(self, tid, det):
        self.id, self.hits, self.missed, self.confirmed = tid, 1, 0, False
        self.pos = np.array(det["c"], float); self.vel = np.zeros(2)
        self.area = float(det["area"]); self.size = self.area; self.mask = det["mask"]
        self.trail = deque([tuple(self.pos)], maxlen=40)

    def predict(self):
        return self.pos + self.vel

    def update(self, det, alpha=0.7, beta=0.3):
        pred = self.predict(); meas = np.array(det["c"], float)
        resid = meas - pred
        self.pos = pred + alpha * resid
        self.vel = self.vel + beta * resid
        self.area = 0.8 * self.area + 0.2 * det["area"]
        self.size = max(0.995 * self.size, float(det["area"]))   # full-egg size: a hidden/partly visible egg is smaller, never much larger
        self.mask = det["mask"]; self.hits += 1; self.missed = 0
        self.trail.append(tuple(self.pos))

    def coast(self):
        self.pos = self.predict(); self.vel *= 0.8; self.missed += 1
        self.trail.append(tuple(self.pos))


class Tracker:
    def __init__(self, min_hits=3, max_missed=24, max_dist_factor=2.5, min_gate=40, dormant_frames=1200, far_gate=400):
        self.far_gate = far_gate   # stage-2 gate (px): an egg flicked while hidden can reappear far from where it vanished
        self.tracks, self.next_id = [], 1
        self.dormant = []          # (id, last_pos, last_area, frame_lost) of tracks that were lost
        self.frame, self.dormant_frames = 0, dormant_frames
        self.min_hits, self.max_missed = min_hits, max_missed
        self.k, self.min_gate = max_dist_factor, min_gate

    def step(self, dets):
        self.frame += 1
        T = self.tracks
        matched_t, matched_d = set(), set()
        if T and dets:
            cost = np.full((len(T), len(dets)), 1e6)
            for i, t in enumerate(T):
                p = t.predict(); r = np.sqrt(t.size / np.pi)
                gate = max(self.min_gate, self.k * r)
                for j, d in enumerate(dets):
                    dist = np.linalg.norm(np.array(d["c"]) - p)
                    ratio = d["area"] / max(t.size, 1)
                    if dist <= gate and 0.12 <= ratio <= 1.8:
                        cost[i, j] = dist / gate + 0.15 * abs(np.log(ratio))
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 1e5:
                    T[i].update(dets[j]); matched_t.add(i); matched_d.add(j)
        # stage 2: leftover detections may belong to a coasting (hidden) track -> large gate, nearest first
        for j, d in enumerate(dets):
            if j in matched_d:
                continue
            best = None
            for i, t in enumerate(T):
                if i in matched_t or not t.confirmed or t.missed == 0:
                    continue
                dist = np.linalg.norm(np.array(d["c"]) - t.pos)
                ratio = d["area"] / max(t.size, 1)
                if dist <= self.far_gate and 0.12 <= ratio <= 1.8 and (best is None or dist < best[0]):
                    best = (dist, i)
            if best:
                T[best[1]].update(d); T[best[1]].vel[:] = 0
                matched_t.add(best[1]); matched_d.add(j)
        for i, t in enumerate(T):
            if i not in matched_t:
                t.coast() if t.confirmed else setattr(t, "missed", t.missed + 1)
        for j, d in enumerate(dets):
            if j not in matched_d:
                T.append(Track(-1, d))
        for t in T:
            if not t.confirmed and t.hits >= self.min_hits:
                t.confirmed = True
                live_ids = {u.id for u in T if u.confirmed and u is not t}
                free = [d for d in self.dormant if d[0] not in live_ids]
                if free:   # re-identification: eggs look identical, so the best cue is where an egg was last seen
                    d = min(free, key=lambda d: np.linalg.norm(d[1] - t.pos))
                    t.id = d[0]; t.size = max(t.size, d[2])
                    self.dormant.remove(d)
                else:
                    t.id = self.next_id
                    self.next_id += 1
        for t in T:
            if t.confirmed and t.missed > self.max_missed:
                self.dormant.append((t.id, t.pos.copy(), t.size, self.frame))
        self.dormant = [d for d in self.dormant if self.frame - d[3] <= self.dormant_frames]
        self.tracks = [t for t in T if (t.confirmed and t.missed <= self.max_missed) or (not t.confirmed and t.missed <= 1)]
        return [t for t in self.tracks if t.confirmed]


def detections(frame):
    out = []
    for m in video_seg.egg_candidates(frame):
        ys, xs = np.where(m)
        out.append(dict(mask=m, c=(xs.mean(), ys.mean()), area=int(m.sum())))
    return out


def render(frame, tracks, fidx, fps):
    o = frame.copy()
    for t in tracks:
        col = COLORS[(t.id - 1) % len(COLORS)]
        live = t.missed == 0
        if live:
            o[t.mask] = (o[t.mask] * 0.45 + np.array(col) * 0.55).astype(np.uint8)
            cnts, _ = cv2.findContours(t.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(o, cnts, -1, col, 2)
        else:  # coasting: dashed circle at the predicted position
            r = int(np.sqrt(t.size / np.pi))
            for a in range(0, 360, 30):
                cv2.ellipse(o, (int(t.pos[0]), int(t.pos[1])), (r, r), 0, a, a + 15, col, 2)
        pts = np.array(t.trail, np.int32)
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(o, tuple(a), tuple(b), col, 1, cv2.LINE_AA)
        x, y = int(t.pos[0]), int(t.pos[1] - np.sqrt(t.size / np.pi) - 8)
        label = f"Egg {t.id}" + ("" if live else " (occluded)")
        cv2.putText(o, label, (x - 30, max(y, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(o, label, (x - 30, max(y, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
    cv2.putText(o, f"t={fidx / fps:5.1f}s  tracked eggs: {len(tracks)}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return o


def main(src, dst):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS); W, H = int(cap.get(3)), int(cap.get(4)); N = int(cap.get(7))
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    vw = None
    for fourcc in ("avc1", "mp4v"):
        vw = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*fourcc), fps, (W, H))
        if vw.isOpened():
            print("codec", fourcc); break
    tr = Tracker(); rows = []
    ref = None      # downscaled last frame in which an egg was actually visible
    for f in range(N):
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (64, 36)).astype(np.float32)
        # scene-end / cut detection: with no egg visible, a frame far from the last egg-bearing frame means the scene
        # changed (title card, fade). Hand occlusion stays < ~10 mean abs diff; scene change is > 45.
        if ref is not None and not any(t.missed == 0 for t in tr.tracks if t.confirmed) and np.abs(small - ref).mean() > 30:
            tr = Tracker(); ref = None
        confirmed = tr.step(detections(frame))
        if any(t.missed == 0 for t in confirmed):
            ref = small
        vw.write(render(frame, confirmed, f, fps))
        for t in confirmed:
            rows.append((f, t.id, "visible" if t.missed == 0 else "occluded", round(t.pos[0], 1), round(t.pos[1], 1), int(t.area)))
    vw.release()
    with open(Path(dst).with_suffix(".csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["frame", "id", "state", "cx", "cy", "area"]); w.writerows(rows)
    ids = sorted({r[1] for r in rows})
    print("frames", N, "| track ids:", ids)
    for i in ids:
        fr = [r[0] for r in rows if r[1] == i]; vis = sum(1 for r in rows if r[1] == i and r[2] == "visible")
        print(f"  Egg {i}: frames {min(fr)}-{max(fr)} ({len(fr)} frames, {vis} visible, {len(fr) - vis} coasted)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
