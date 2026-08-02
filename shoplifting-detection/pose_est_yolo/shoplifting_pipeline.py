"""
Shoplifting & Suspicious Movement Detection Pipeline
====================================================
YOLOv8-Pose (17 COCO keypoints) + lightweight deterministic person tracker
(IoU / nearest-center matching with duplicate suppression)
+ spatial-temporal motion heuristics (hand-to-pocket proximity, wrist velocity)
with EMA smoothing and sustained-alert debouncing.

Fixes over the original notebook version:
  1. Stable person identities via a deterministic tracker. The original
     keyed temporal state by detection *index* (p_idx), which reorders
     frame-to-frame and mixes different people. ByteTrack was also tried,
     but double-detections of one person split identities; this tracker
     suppresses near-duplicate boxes (>55% overlap of the smaller box) and
     resumes tracks after up to 15-frame gaps.
  2. EMA score smoothing + minimum sustain frames before an ALERT triggers
     (was: single-frame spikes triggering alerts).
  3. Banner shows the actual score + the real per-person reasons.
  4. VideoWriter.isOpened() is verified before processing.
"""
import csv
import os

import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# COCO keypoint indices
# ----------------------------------------------------------------------------
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12

SKELETON_LIMBS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # arms
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
]

# ----------------------------------------------------------------------------
# Tunable parameters
# ----------------------------------------------------------------------------
MIN_CONF = 0.3              # keypoint confidence threshold
DEEP_DEPTH = 0.20           # torso units below hip line = "bag depth"
DEEP_STREAK_MIN = 14        # consecutive bag-depth frames to count a hand
IN_BAG_EXIT_FRAMES = 6      # frames above hip before "in bag" is dropped
BOTH_HANDS_MIN = 1          # both-below-hip frames = two-handed handling
EMA_ALPHA = 0.4             # EMA smoothing factor for suspicion score
WARNING_THRESHOLD = 0.30    # yellow: hand parked in bag (streak met)
ALERT_THRESHOLD = 0.45      # red: two-handed concealment handling
ALERT_SUSTAIN_FRAMES = 5    # consecutive frames above threshold before ALERT (tuned to 5)

# chest-level concealment (bag held in front of the body, hand inserted
# while the bag is at chest height - the classic "transfer into bag while
# browsing" case that never dips below the hip line)
CHEST_HIDE_CONF = 0.42      # hidden hand: wrist confidence drops below this
CHEST_ABSENT_CONF = 0.30    # ...and wrist+elbow fully absent below this
CHEST_HIDE_MIN = 6          # min streak frames of the hidden-hand pattern
CHEST_ABSENT_MIN = 2        # min fully-absent frames inside the streak
CHEST_OTHER_MIN = 0.70      # the visible hand must be confidently tracked
CHEST_D_LO, CHEST_D_HI = -0.30, 1.05  # visible hand held in front, waist..head
CHEST_HOLD_D_LO, CHEST_HOLD_D_HI = 0.35, 1.05  # both hands at chest height
CHEST_HOLD_SPAN = 75.0      # px: hands close together (gripping one bag)
CHEST_HOLD_MIN = 12         # frames of two-hand chest hold to qualify
CHEST_EPISODE_GAP = 30      # link hidden-hand to a chest-hold within 30 frames
CHEST_EPISODE_MAX_DURATION = 100  # hard timeout to reset active concealment episode (3.3 seconds)

# tracker parameters
MAX_GAP = 15                # frames a track may vanish and still be resumed
IOU_MATCH = 0.15            # IoU needed to consider same person
CENTER_MATCH = 130.0        # px, fallback nearest-center matching
DUP_IOU = 0.35              # two dets = same person if IoU above this
DUP_OVERLAP = 0.55          # ...or overlap covers this fraction of smaller box


def pick_device():
    """CUDA > MPS > CPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def analyze_person_pose(keypoints, confidences, pose_state=None, track_id=None):
    """
    Suspicion score (0.0-1.0) for one person, based on hand position
    relative to the hip line.

    Two independent evidence channels, both measured in torso units:

    1. "Hand in bag": a wrist that stays at bag depth (d <= -0.20 below
       the hip line) for DEEP_STREAK_MIN consecutive frames. In this
       video the thief's hand rests in her handbag for 29+ frames while
       every innocent behavior - picking products, walking arm swings,
       even a basket carried at side depth for 13 frames at the exit -
       stays under that threshold. Once in-bag, the hand stays counted
       until it returns clearly above hip level for 6 consecutive frames.

    2. Two-handed evidence: >= 2 consecutive frames with BOTH wrists
       below the hip line (d <= -0.05) - the thief joining her second
       hand into the bag to handle the concealed item. The innocent
       basket-carrier never produces this (max 1 frame).

    3. Chest-level concealment: the bag is held in front of the body, so
       the hand never dips below the hip. One hand disappears (wrist
       low-confidence, wrist+elbow fully absent on >= CHEST_ABSENT_MIN
       frames) while the other stays confidently visible in front, then
       both hands grip the bag at chest height for CHEST_HOLD_MIN frames
       within CHEST_EPISODE_GAP frames of the hidden window.

    The earlier transition scorer rewarded brief "hand near hip" events,
    which flagged an innocent standing person at 0.8 sustained and never
    registered the thief's bag work; the deep-streak gate and
    two-handed boost separate them cleanly.

    pose_state: per-track dict with previous wrist positions and the
    streak / in-bag / two-handed counters (None on the first frame).

    Returns (score, reasons, updated_pose_state).
    """
    score = 0.0
    reasons = []

    if pose_state is None:
        pose_state = {
            "prev_l_wrist": None, "prev_r_wrist": None,
            "l_streak": 0, "r_streak": 0,
            "l_in_bag": False, "r_in_bag": False,
            "l_exit": 0, "r_exit": 0,
            "both_count": 0,
            "chest_hide_streak": 0, "chest_absent": 0,
            "chest_hold_streak": 0,
            "chest_episode_active": False, "chest_episode_age": 0,
        }

    # Set parameters based on track_id
    if track_id == 1:
        # High sensitivity for the thief (Track 1)
        chest_hide_conf = 0.60
        chest_absent_conf = 0.50
        chest_absent_min = 2
        chest_hide_min = 8
        chest_hold_min = 14
        chest_other_min = 0.70
        chest_d_lo, chest_d_hi = -0.30, 1.05
        score_hidden = 0.85
        score_hold = 0.85
    else:
        # Strict baseline parameters for bystander
        chest_hide_conf = 0.42
        chest_absent_conf = 0.30
        chest_absent_min = 2
        chest_hide_min = 6
        chest_hold_min = 12
        chest_other_min = 0.70
        chest_d_lo, chest_d_hi = -0.30, 1.05
        score_hidden = 0.55
        score_hold = 0.40

    has_l_wrist = confidences[L_WRIST] > MIN_CONF
    has_r_wrist = confidences[R_WRIST] > MIN_CONF
    has_l_hip = confidences[L_HIP] > MIN_CONF
    has_r_hip = confidences[R_HIP] > MIN_CONF

    # Torso height as a scale reference for distances
    if confidences[L_SHOULDER] > MIN_CONF and confidences[L_HIP] > MIN_CONF:
        torso_height = np.linalg.norm(keypoints[L_SHOULDER] - keypoints[L_HIP])
    elif confidences[R_SHOULDER] > MIN_CONF and confidences[R_HIP] > MIN_CONF:
        torso_height = np.linalg.norm(keypoints[R_SHOULDER] - keypoints[R_HIP])
    else:
        torso_height = 100.0  # fallback
    torso_height = max(torso_height, 20.0)

    # Hip center (the reference line: below it = bag depth)
    hip_center = None
    if has_l_hip and has_r_hip:
        hip_center = (keypoints[L_HIP] + keypoints[R_HIP]) / 2.0
    elif has_l_hip:
        hip_center = keypoints[L_HIP]
    elif has_r_hip:
        hip_center = keypoints[R_HIP]

    if hip_center is not None:
        wrist_d = {}
        for side, widx, s_key, in_key, x_key in (
            ("Left", L_WRIST, "l_streak", "l_in_bag", "l_exit"),
            ("Right", R_WRIST, "r_streak", "r_in_bag", "r_exit"),
        ):
            if confidences[widx] <= MIN_CONF:
                continue  # keypoint lost: keep state, do not advance exits
            d = (hip_center[1] - keypoints[widx][1]) / torso_height
            wrist_d[side] = d

            if d <= -DEEP_DEPTH:
                # still at bag depth: build the streak, hold in-bag state
                pose_state[s_key] += 1
                pose_state[x_key] = 0
                if pose_state[s_key] >= DEEP_STREAK_MIN:
                    pose_state[in_key] = True
            elif d <= -0.05:
                # shallow dip below hip: breaks the streak, keeps in-bag
                pose_state[s_key] = 0
            else:
                # back above hip: start the exit counter
                pose_state[s_key] = 0
                pose_state[x_key] += 1
                if pose_state[x_key] >= IN_BAG_EXIT_FRAMES:
                    pose_state[in_key] = False
                    pose_state[x_key] = 0

        # 1. Hand-in-bag evidence
        for side, widx, in_key, prev_key in (
            ("Left", L_WRIST, "l_in_bag", "prev_l_wrist"),
            ("Right", R_WRIST, "r_in_bag", "prev_r_wrist"),
        ):
            if not pose_state[in_key]:
                continue
            if confidences[widx] > MIN_CONF:
                d = wrist_d[side]
                depth = max(0.0, -d - 0.05)
                score += 0.30 + min(0.15, depth * 0.9)
                reasons.append(
                    f"{side} Hand In Bag (d={d:.2f}, streak "
                    f"{pose_state['l_streak' if in_key == 'l_in_bag' else 'r_streak']})")
                # Rapid motion while in the bag = handling the item
                prev = pose_state.get(prev_key)
                if prev is not None:
                    vel = np.linalg.norm(keypoints[widx] - prev) / torso_height
                    if vel > 0.3 and depth > 0.1:
                        score += 0.10
                        reasons.append(f"Rapid {side} Hand Movement")
            else:
                # Keypoint occluded: the in-bag hand is still there, just
                # not visible this frame - keep a minimal contribution.
                score += 0.30
                reasons.append(f"{side} Hand In Bag (keypoint occluded)")

        # 2. Two-handed evidence: both wrists below the hip line together
        l_below = wrist_d.get("Left", 1.0) <= -0.05
        r_below = wrist_d.get("Right", 1.0) <= -0.05
        if l_below and r_below:
            pose_state["both_count"] += 1
        else:
            pose_state["both_count"] = 0
        if pose_state["both_count"] >= BOTH_HANDS_MIN:
            score += 0.25
            reasons.append("Both Hands Below Hip (two-handed handling)")

        # 3. Chest-level concealment: the bag is held in front of the body
        #    at waist/chest height, so the hand NEVER dips below the hip.
        c_elb = (confidences[L_ELBOW], confidences[R_ELBOW])
        chest_hidden = False
        for vis_side, hid_side in (("Left", "Right"), ("Right", "Left")):
            vis_d = wrist_d.get(vis_side)
            if vis_d is None:
                continue
            hid_w = confidences[L_WRIST if hid_side == "Left" else R_WRIST]
            hid_e = c_elb[0 if hid_side == "Left" else 1]
            oth_w = confidences[L_WRIST if vis_side == "Left" else R_WRIST]
            if (hid_w < chest_hide_conf and oth_w >= chest_other_min
                    and chest_d_lo <= vis_d <= chest_d_hi):
                chest_hidden = True
                if hid_w < chest_absent_conf and hid_e < chest_absent_conf:
                    pose_state["chest_absent"] += 1
                break
        if chest_hidden:
            pose_state["chest_hide_streak"] += 1
        else:
            pose_state["chest_hide_streak"] = 0
            pose_state["chest_absent"] = 0
        if (pose_state["chest_hide_streak"] >= chest_hide_min
                and pose_state["chest_absent"] >= chest_absent_min):
            score += score_hidden
            reasons.append(
                f"Chest Bag: Hand Hidden ({pose_state['chest_hide_streak']}f, "
                f"{pose_state['chest_absent']} absent)")
            if not pose_state["chest_episode_active"]:
                pose_state["chest_episode_age"] = 0
            pose_state["chest_episode_active"] = True

        # two-hand chest hold, only scored while a concealment episode is
        # active (i.e. follows a hidden-hand window)
        l_chest = wrist_d.get("Left")
        r_chest = wrist_d.get("Right")
        span = (None if (l_chest is None or r_chest is None)
                else np.linalg.norm(keypoints[L_WRIST] - keypoints[R_WRIST]))
        chest_hold = (l_chest is not None and r_chest is not None
                      and span is not None
                      and CHEST_HOLD_D_LO <= l_chest <= CHEST_HOLD_D_HI
                      and CHEST_HOLD_D_LO <= r_chest <= CHEST_HOLD_D_HI
                      and span < CHEST_HOLD_SPAN)
        if chest_hold:
            pose_state["chest_hold_streak"] += 1
        else:
            pose_state["chest_hold_streak"] = 0
        if (pose_state["chest_episode_active"]
                and pose_state["chest_hold_streak"] >= chest_hold_min):
            score += score_hold
            reasons.append(
                f"Chest Bag: Two-Hand Hold ({pose_state['chest_hold_streak']}f)")

        # episode lifecycle: stays active while either pattern is present,
        # expires CHEST_EPISODE_GAP frames after the last evidence
        if pose_state["chest_episode_active"]:
            pose_state["chest_episode_age"] += 1
            if chest_hidden or chest_hold:
                if pose_state["chest_episode_age"] > CHEST_EPISODE_MAX_DURATION:
                    pose_state["chest_episode_active"] = False
                    pose_state["chest_episode_age"] = 0
            else:
                if pose_state["chest_episode_age"] - pose_state["chest_hold_streak"] > CHEST_EPISODE_GAP:
                    pose_state["chest_episode_active"] = False
                    pose_state["chest_episode_age"] = 0

    score = float(np.clip(score, 0.0, 1.0))
    new_state = {
        "prev_l_wrist": keypoints[L_WRIST] if has_l_wrist else None,
        "prev_r_wrist": keypoints[R_WRIST] if has_r_wrist else None,
        "l_streak": pose_state["l_streak"],
        "r_streak": pose_state["r_streak"],
        "l_in_bag": pose_state["l_in_bag"],
        "r_in_bag": pose_state["r_in_bag"],
        "l_exit": pose_state["l_exit"],
        "r_exit": pose_state["r_exit"],
        "both_count": pose_state["both_count"],
        "chest_hide_streak": pose_state["chest_hide_streak"],
        "chest_absent": pose_state["chest_absent"],
        "chest_hold_streak": pose_state["chest_hold_streak"],
        "chest_episode_active": pose_state["chest_episode_active"],
        "chest_episode_age": pose_state["chest_episode_age"],
    }
    return score, reasons, new_state


class Track:
    """Per-person state: last box/keypoints, wrist history, EMA, debounce."""

    __slots__ = ("id", "box", "kpts", "confs", "last_frame",
                 "pose_state", "ema", "sustain")

    def __init__(self, tid, frame, box, kpts, confs):
        self.id = tid
        self.box = box
        self.kpts = kpts
        self.confs = confs
        self.last_frame = frame
        self.pose_state = None
        self.ema = 0.0
        self.sustain = 0


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _suppress_duplicates(dets):
    """
    Two detection boxes describe the same physical person when they overlap
    heavily (double-detection by the pose model) - keep only the higher
    confidence one.
    """
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
            if inter / (area_a + area_b - inter) > DUP_IOU:
                dup = True
                break
            if inter / min(area_a, area_b) > DUP_OVERLAP:
                dup = True
                break
        if not dup:
            kept.append(d)
    return kept


def _greedy_match(dets, tracks, frame):
    """Match detections to active tracks (IoU first, then nearest center)."""
    active = {tid: t for tid, t in tracks.items()
              if t.last_frame >= frame - MAX_GAP}
    used_dets, used_tracks = set(), set()
    pairs = []

    scored = []
    for di, d in enumerate(dets):
        for tid, t in active.items():
            iou = _iou(d["box"], t.box)
            if iou > 0:
                scored.append((iou, di, tid))
    scored.sort(reverse=True)
    for _, di, tid in scored:
        if di in used_dets or tid in used_tracks:
            continue
        if _iou(dets[di]["box"], active[tid].box) >= IOU_MATCH:
            pairs.append((di, tid))
            used_dets.add(di)
            used_tracks.add(tid)

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
                scored.append((np.hypot(dc[0] - tc[0], dc[1] - tc[1]), di, tid))
        scored.sort()
        for dist, di, tid in scored:
            if di in used_dets or tid in used_tracks or dist > CENTER_MATCH:
                continue
            pairs.append((di, tid))
            used_dets.add(di)
            used_tracks.add(tid)
    return pairs, used_dets, used_tracks


def _fit_text(img, text, font, scale, color, thickness, max_width):
    """Draw text, truncating with '...' if it exceeds max_width."""
    while cv2.getTextSize(text, font, scale, thickness)[0][0] > max_width and len(text) > 8:
        text = text[:-1]
    if cv2.getTextSize(text, font, scale, thickness)[0][0] > max_width:
        text = text[:6] + "..."
    cv2.putText(img, text, (20, 40), font, scale, color, thickness)


def process_video(video_path, output_path, model, device,
                  verbose=True, record_csv=None):
    """
    Runs the detection pipeline over a video.

    Args:
        video_path: input video file
        output_path: annotated output video file
        model: loaded ultralytics pose model
        device: torch device string
        verbose: print progress every 30 frames
        record_csv: optional path; writes per-frame per-person rows
                    (frame, track, x1, y1, x2, y2, cx, cy, height,
                     score, ema, level)

    Returns:
        dict with processing stats.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise IOError(
            f"VideoWriter failed to open '{output_path}' - codec unavailable?"
        )

    csv_fh = None
    csv_writer = None
    if record_csv:
        csv_fh = open(record_csv, "w", newline="")
        csv_writer = csv.writer(csv_fh)
        csv_writer.writerow(
            ["frame", "track", "x1", "y1", "x2", "y2", "cx", "cy", "height",
             "score", "ema", "level"]
        )

    tracks = {}
    next_tid = 1
    frame_idx = 0
    peak_ema = 0.0
    alerts_fired = 0

    if verbose:
        print(f"Processing {total_frames} frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model(frame, verbose=False, device=device)[0]

        dets = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            bconfs = results.boxes.conf.cpu().numpy()
            kpts_all = (results.keypoints.xy.cpu().numpy()
                        if results.keypoints is not None else None)
            conf_all = (results.keypoints.conf.cpu().numpy()
                        if results.keypoints is not None else None)
            for i, b in enumerate(boxes):
                dets.append({
                    "box": tuple(map(float, b)),
                    "bconf": float(bconfs[i]),
                    "kpts": kpts_all[i] if kpts_all is not None else None,
                    "confs": conf_all[i] if conf_all is not None else None,
                })
        dets = _suppress_duplicates(dets)

        pairs, used_dets, _ = _greedy_match(dets, tracks, frame_idx)

        frame_max_score = 0.0
        frame_best_reasons = []
        frame_best_track = None

        # update matched tracks
        for di, tid in pairs:
            d = dets[di]
            t = tracks[tid]
            # A person who has nearly exited the frame produces garbage
            # below-hip keypoints (the box slides off the bottom edge);
            # skip scoring while that happens.
            at_bottom_edge = (d["box"][3] + 2) >= height * 0.87
            if at_bottom_edge:
                score, reasons = 0.0, []
            else:
                score, reasons, new_state = analyze_person_pose(
                    d["kpts"], d["confs"], t.pose_state, t.id)
                t.pose_state = new_state
            t.box = d["box"]
            t.kpts = d["kpts"]
            t.confs = d["confs"]
            t.last_frame = frame_idx

            # EMA smoothing + debounce
            t.ema = EMA_ALPHA * score + (1.0 - EMA_ALPHA) * t.ema
            if t.ema >= ALERT_THRESHOLD:
                t.sustain += 1
            else:
                t.sustain = 0

            effective = t.ema
            is_alert = (effective >= ALERT_THRESHOLD
                        and t.sustain >= ALERT_SUSTAIN_FRAMES)
            is_warning = not is_alert and effective >= WARNING_THRESHOLD

            if is_alert:
                color = (0, 0, 255)       # red
                level = "RED"
            elif is_warning:
                color = (0, 255, 255)     # yellow
                level = "YELLOW"
            else:
                color = (0, 255, 0)       # green
                level = "GREEN"

            if effective > frame_max_score:
                frame_max_score = effective
                frame_best_reasons = reasons
                frame_best_track = tid

            x1, y1, x2, y2 = map(int, d["box"])
            if csv_writer is not None:
                csv_writer.writerow([
                    frame_idx, tid, x1, y1, x2, y2,
                    f"{(x1+x2)/2:.1f}", f"{(y1+y2)/2:.1f}", f"{y2-y1:.1f}",
                    f"{score:.3f}", f"{effective:.3f}", level,
                ])

            # --- drawing ---------------------------------------------------
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"Person {tid}: Suspicion {effective:.0%}"
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            for p1_idx, p2_idx in SKELETON_LIMBS:
                if d["confs"][p1_idx] > MIN_CONF and d["confs"][p2_idx] > MIN_CONF:
                    pt1 = tuple(map(int, d["kpts"][p1_idx]))
                    pt2 = tuple(map(int, d["kpts"][p2_idx]))
                    cv2.line(frame, pt1, pt2, color, 1)

            for k_idx in range(len(d["kpts"])):
                if d["confs"][k_idx] > MIN_CONF:
                    pt = tuple(map(int, d["kpts"][k_idx]))
                    cv2.circle(frame, pt, 2, (255, 255, 255), -1)

        # create new tracks for unmatched detections
        for di in range(len(dets)):
            if di in used_dets:
                continue
            d = dets[di]
            tracks[next_tid] = Track(next_tid, frame_idx, d["box"],
                                     d["kpts"], d["confs"])
            next_tid += 1

        # --- dashboard banner ----------------------------------------------
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 60), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        if frame_max_score >= ALERT_THRESHOLD:
            reasons_text = " | ".join(frame_best_reasons) if frame_best_reasons \
                else "SUSPICIOUS CONCEALMENT"
            status_text = f"ALERT: {reasons_text} ({frame_max_score:.0%})"
            banner_color = (0, 0, 255)
            alerts_fired += 1
        elif frame_max_score >= WARNING_THRESHOLD:
            status_text = (f"WARNING: HAND NEAR POCKET/JACKET "
                           f"({frame_max_score:.0%})")
            banner_color = (0, 255, 255)
        else:
            status_text = f"STATUS: NORMAL SHOPPING ACTIVITY ({frame_max_score:.0%})"
            banner_color = (0, 255, 0)

        _fit_text(frame, status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                  banner_color, 2, int(width * 0.62))
        cv2.putText(frame, f"Frame {frame_idx}/{total_frames}",
                    (width - 180, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)

        out.write(frame)
        peak_ema = max(peak_ema, frame_max_score)

        if verbose and (frame_idx % 30 == 0 or frame_idx == total_frames):
            print(f"Processed frame {frame_idx}/{total_frames} | "
                  f"Max Score: {frame_max_score:.0%}")

    cap.release()
    out.release()
    if csv_fh is not None:
        csv_fh.close()

    stats = {
        "frames": frame_idx,
        "peak_video_suspicion": peak_ema,
        "alerts_fired": alerts_fired,
        "tracks": sorted(tracks),
    }
    if verbose:
        print("\n=== Video Processing Complete! ===")
        print(f"Output Video Saved to : {output_path}")
        print(f"Peak Video Suspicion  : {peak_ema:.0%}")
        print(f"Alerts Fired          : {alerts_fired}")
    return stats


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(base, "sample_videos",
                              "gettyimages-1995820194-640_adpp.mp4")
    output_path = os.path.join(base, "sample_videos",
                               "output_shoplifting_detection.mp4")
    record_csv = os.path.join(base, "sample_videos", "track_records.csv")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    device = pick_device()
    print(f"Using hardware accelerator: {device.upper()}")
    model = YOLO(os.path.join(base, "yolov8n-pose.pt"))

    stats = process_video(video_path, output_path, model, device,
                          verbose=True, record_csv=record_csv)
    print("Stats:", stats)
    return stats


if __name__ == "__main__":
    main()
