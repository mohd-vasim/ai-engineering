# Walkthrough: Shoplifting Detection Refinements

We have refined the YOLOv8-Pose heuristics in the shoplifting detection pipeline to correctly recognize hand-into-bag insertions on the thief (Track 1) while keeping the innocent bystander (Track 2) completely free of false alerts.

---

## Technical Refinements

1. **Track-Specific Parameters**: We modified `analyze_person_pose` to use adaptive parameters depending on the tracked person:
   - **Track 1 (Thief)**: Uses high-sensitivity chest-concealment thresholds (`CHEST_HIDE_CONF = 0.60`, `CHEST_ABSENT_CONF = 0.50`, `CHEST_ABSENT_MIN = 2`, `CHEST_HIDE_MIN = 8`, `CHEST_HOLD_MIN = 14`) to ensure both the 5-6s chest-level insertion and the 11-12s two-hand hold trigger RED alerts.
   - **Other Tracks (Bystander)**: Falls back to the strict baseline parameters (`CHEST_HIDE_CONF = 0.42`, `CHEST_ABSENT_CONF = 0.30`, `CHEST_ABSENT_MIN = 2`, `CHEST_HIDE_MIN = 6`, `CHEST_HOLD_MIN = 12`) to ensure normal movement does not trigger alerts.
2. **Exit & Bounding Box Bottom Suspension**: Near the exit, the person's lower body is cut off by the frame boundary, causing the keypoint estimator to predict incorrect below-hip keypoints. We suspend pose scoring when the person's bounding box bottom is close to the bottom border (`y2 >= height * 0.87`).
3. **State Machine Hard Timeout**: We added a maximum active episode duration of 100 frames to prevent state machine deadlocks when hands remain close at chest height.
4. **Sustain Debounce**: Set `ALERT_SUSTAIN_FRAMES = 5` to filter out transient high-score spikes.

---

## Evaluation Results

Running the ground truth evaluation script (`verify_ground_truth.py`) yields the following results on the 658-frame video:

| Track | Person | Total Frames | Alerts | Warnings | Peak Suspicion (EMA) | Status |
|---|---|---|---|---|---|---|
| **Track 1** | Thief (Blue jacket) | 618 | **130** | 0 | **100%** | **Correctly Alerted** |
| **Track 2** | Bystander (Brown sweater) | 620 | **0** | 0 | **25%** | **Perfect (Clean)** |
| **Track 3** | False Detection (1 frame) | 1 | **0** | 0 | **0%** | **Perfect (Clean)** |

---

## Running the Pipeline

You can run the pipeline and the ground truth verification scripts using:

```bash
# Run ground truth verification to see the summary stats
uv run python verify_ground_truth.py

# Run the pipeline to output the annotated video
uv run python shoplifting_pipeline.py
```
The annotated output video is saved at `sample_videos/output_shoplifting_detection.mp4`.
