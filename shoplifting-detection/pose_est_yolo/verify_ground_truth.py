"""
Ground-truth verification of the shoplifting pipeline using coordinates.

Builds a deterministic person tracker (IoU + keypoint continuity) to obtain
stable identities across the whole video, then checks which person the
suspicious-motion alerts are assigned to vs. the ground truth:
  * the shoplifter is "the first person from bottom to middle",
  * i.e. the person who starts lower in the frame (bottom) and moves
    toward the middle of the frame.
"""
import csv
import os

import cv2
import numpy as np
from ultralytics import YOLO

from pose_est_yolo.shoplifting_pipeline import (
    analyze_person_pose, pick_device, MIN_CONF,
    WARNING_THRESHOLD, ALERT_THRESHOLD,
)

BASE = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.join(BASE, "sample_videos", "gettyimages-1995820194-640_adpp.mp4")
OUT_CSV = os.path.join(BASE, "sample_videos", "verify_records.csv")

MAX_GAP = 15          # frames a track may vanish and still be resumed
IOU_MATCH = 0.15      # IoU needed to consider same person
CENTER_MATCH = 130.0  # px, fallback nearest-center matching


class Track:
    def __init__(self, tid, frame, box, kpts, confs, score, reasons):
        self.id = tid
        self.last_frame = frame
        self.box = box                    # x1,y1,x2,y2
        self.kpts = kpts
        self.confs = confs
        self.score = score
        self.reasons = reasons
        self.prev_wrists = None
        self.first_frame = frame
        self.first_cy = (box[1] + box[3]) / 2.0
        self.cy_trace = [self.first_cy]
        self.score_trace = [score]
        self.alerts = 0
        self.warnings = 0


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def greedy_match(dets, tracks, frame):
    """Match detections to active tracks (IoU first, then nearest center)."""
    active = {tid: t for tid, t in tracks.items()
              if t.last_frame >= frame - MAX_GAP}
    used_dets = set()
    used_tracks = set()
    pairs = []

    # pass 1: IoU matching
    scored = []
    for di, d in enumerate(dets):
        for tid, t in active.items():
            iou = box_iou(d["box"], t.box)
            if iou > 0:
                scored.append((iou, di, tid))
    scored.sort(reverse=True)
    for _, di, tid in scored:
        if di in used_dets or tid in used_tracks:
            continue
        if box_iou(dets[di]["box"], active[tid].box) >= IOU_MATCH:
            pairs.append((di, tid))
            used_dets.add(di)
            used_tracks.add(tid)

    # pass 2: nearest-center matching for unmatched
    rest_dets = [di for di in range(len(dets)) if di not in used_dets]
    rest_trk = [tid for tid in active if tid not in used_tracks]
    if rest_dets and rest_trk:
        scored = []
        for di in rest_dets:
            dc = ((dets[di]["box"][0] + dets[di]["box"][2]) / 2,
                  (dets[di]["box"][1] + dets[di]["box"][3]) / 2)
            for tid in rest_trk:
                tc = ((active[tid].box[0] + active[tid].box[2]) / 2,
                      (active[tid].box[1] + active[tid].box[3]) / 2)
                dist = np.hypot(dc[0] - tc[0], dc[1] - tc[1])
                scored.append((dist, di, tid))
        scored.sort()
        for dist, di, tid in scored:
            if di in used_dets or tid in used_tracks or dist > CENTER_MATCH:
                continue
            pairs.append((di, tid))
            used_dets.add(di)
            used_tracks.add(tid)
    return pairs, used_dets, used_tracks


def main():
    device = pick_device()
    print(f"Device: {device.upper()}")
    model = YOLO(os.path.join(BASE, "yolov8n-pose.pt"))

    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracks = {}
    next_tid = 1
    frame_idx = 0
    rows = []
    peaks = {}

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "track", "x1", "y1", "x2", "y2", "cx", "cy",
                    "height", "score", "level"])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            r = model(frame, verbose=False, device=device)[0]

            dets = []
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                kpts_all = (r.keypoints.xy.cpu().numpy()
                            if r.keypoints is not None else None)
                conf_all = (r.keypoints.conf.cpu().numpy()
                            if r.keypoints is not None else None)
                bconfs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else None
                for i, b in enumerate(boxes):
                    dets.append({
                        "box": tuple(map(float, b)),
                        "kpts": kpts_all[i] if kpts_all is not None else None,
                        "confs": conf_all[i] if conf_all is not None else None,
                        "bconf": float(bconfs[i]) if bconfs is not None else 1.0,
                    })

            # suppress near-duplicate overlapping detections (same person):
            # two boxes are the same person if IoU > 0.35 or the overlap
            # covers >55% of the smaller box (double-detections of one body)
            kept = []
            for d in sorted(dets, key=lambda x: -x["bconf"]):
                dup = False
                for k in kept:
                    a, b = d["box"], k["box"]
                    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
                    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    if inter == 0:
                        continue
                    area_a = (a[2] - a[0]) * (a[3] - a[1])
                    area_b = (b[2] - b[0]) * (b[3] - b[1])
                    iou = inter / (area_a + area_b - inter)
                    overlap_small = inter / min(area_a, area_b)
                    if iou > 0.35 or overlap_small > 0.55:
                        dup = True
                        break
                if not dup:
                    kept.append(d)
            dets = kept

            pairs, used_dets, used_tracks = greedy_match(dets, tracks, frame_idx)

            # update matched tracks
            for di, tid in pairs:
                d = dets[di]
                t = tracks[tid]
                at_bottom_edge = (d["box"][3] + 2) >= height * 0.87
                if at_bottom_edge:
                    score, reasons, curr = 0.0, [], t.prev_wrists
                else:
                    score, reasons, curr = analyze_person_pose(
                        d["kpts"], d["confs"], t.prev_wrists, tid)
                t.prev_wrists = curr
                t.box = d["box"]
                t.kpts = d["kpts"]
                t.confs = d["confs"]
                t.score = score
                t.reasons = reasons
                t.last_frame = frame_idx
                cy = (d["box"][1] + d["box"][3]) / 2.0
                t.cy_trace.append(cy)
                t.score_trace.append(score)
                if score >= ALERT_THRESHOLD:
                    t.alerts += 1
                elif score >= WARNING_THRESHOLD:
                    t.warnings += 1
                level = "RED" if score >= ALERT_THRESHOLD else (
                    "YELLOW" if score >= WARNING_THRESHOLD else "GREEN")
                w.writerow([frame_idx, tid, *[f"{v:.1f}" for v in d["box"]],
                            f"{(d['box'][0]+d['box'][2])/2:.1f}", f"{cy:.1f}",
                            f"{d['box'][3]-d['box'][1]:.1f}", f"{score:.3f}", level])

            # create new tracks for unmatched detections
            for di in range(len(dets)):
                if di in used_dets:
                    continue
                d = dets[di]
                tid = next_tid
                next_tid += 1
                t = Track(tid, frame_idx, d["box"], d["kpts"], d["confs"], 0.0, [])
                tracks[tid] = t

            # count unmatched tracks (dropped)
            for tid, t in tracks.items():
                if tid not in used_tracks and t.last_frame < frame_idx - 1:
                    pass  # gap handled via MAX_GAP

    cap.release()

    # ---- report -----------------------------------------------------------
    print(f"\n=== Stable-track summary ({len(tracks)} tracks, {frame_idx} frames) ===")
    print(f"{'track':>6} {'frames':>8} {'first_cy':>8} {'last_cy':>8} {'cy_min':>8} "
          f"{'cy_max':>8} {'dir':>4} {'alerts':>7} {'warns':>7} {'peak':>6}")
    for tid in sorted(tracks):
        t = tracks[tid]
        frames = t.last_frame - t.first_frame + 1
        first_cy, last_cy = t.cy_trace[0], t.cy_trace[-1]
        cy_min, cy_max = min(t.cy_trace), max(t.cy_trace)
        direction = "bottom->mid" if last_cy < first_cy - 15 else (
            "mid->bottom" if last_cy > first_cy + 15 else "flat")
        peak = max(t.score_trace)
        print(f"{tid:>6} {frames:>8} {first_cy:>8.0f} {last_cy:>8.0f} {cy_min:>8.0f} "
              f"{cy_max:>8.0f} {direction:>4} {t.alerts:>7} {t.warnings:>7} {peak:>6.2f}")
        peaks[tid] = (first_cy, last_cy, direction, peak, t.alerts)
    print("\nrecords:", OUT_CSV)


if __name__ == "__main__":
    main()
