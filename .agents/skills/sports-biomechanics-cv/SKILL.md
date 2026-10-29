---
name: sports-biomechanics-cv
description: Computer vision and 3D temporal kinematics framework for baseball pitching biomechanics and pitch tipping analysis. Covers MediaPipe Pose Landmarker (Heavy/Lite) in VIDEO running mode, 3D metric world landmarks, HSV broadcast camera-cut rejection, mound pitcher ROI tracking, Savitzky-Golay trajectory smoothing, 4-phase pitching delivery segmentation (Leg Lift, Foot Plant, Max External Rotation, Ball Release), intra-pitcher baseline Z-score normalization, and leakage-free GroupKFold validation by game_pk.
---

# Sports Biomechanics & Pitch Tipping Computer Vision (v2.0)

## When to Use This Skill
Activate this skill whenever developing, auditing, or extending:
- Video pose extraction from MLB / Baseball Savant 60fps broadcast clips (`src/pitch_predictor/vision/pose_extractor.py`).
- Biomechanical feature engineering, temporal phase segmentation, or trajectory filtering.
- Pitch tipping classification, probability calibration, intra-pitcher normalization, or SHAP attribution (`src/pitch_predictor/vision/tipping_model.py`).
- Hawk-Eye / Statcast skeletal overlays and kinematic timeline visualizations (`src/pitch_predictor/vision/overlay.py`, `pages/04_pitch_vision_lab.py`).

---

## 1. MediaPipe Pose Landmarker in Temporal Video Mode

1. **Prefer Heavy Landmarker with Graceful Fallback**:
   - Target `pose_landmarker_heavy.task` (`HEAVY_MODEL_URL`) for high-precision distal joint localization (throwing wrist/elbow during 100+ mph arm acceleration), falling back automatically to `pose_landmarker_lite.task` or deterministic `mock_mode` in offline CI environments.
2. **Temporal `RunningMode.VIDEO`**:
   - Use `vision.RunningMode.VIDEO` (`detect_for_video(mp_image, timestamp_ms)`) with strictly monotonically increasing `timestamp_ms` derived from frame index and clip FPS (`int(round(frame_idx * 1000.0 / fps))`). Never reuse a VIDEO landmarker across clips without resetting or instantiating per clip so timestamps never regress.
3. **3D Metric World Landmarks (`pose_world_landmarks`)**:
   - Extract both 2D normalized screen coordinates (`pose_landmarks`, `[0, 1]`, for pixel overlay rendering) and 3D hip-centered metric coordinates (`pose_world_landmarks`, meters, for perspective-invariant joint distances and 3D angles such as `glove_tuck_distance`, `hip_shoulder_separation`, and `forward_release_extension`).

---

## 2. Broadcast Scene Hygiene: Camera-Cut Rejection & Mound ROI Tracking

MLB broadcast videos include batter, catcher, home-plate umpire, and abrupt scene cuts to the outfield after release.
1. **Automatic Camera-Cut Detection (`detect_camera_cut_frame`)**:
   - Compute frame-to-frame HSV color histogram correlation (`cv2.compareHist(..., cv2.HISTCMP_CORREL)`).
   - Flag the first post-windup frame where correlation drops below threshold (e.g. `< 0.55`) as `camera_cut_frame`, truncating kinematic search strictly before the cut instead of relying on hardcoded frame numbers.
2. **Mound Pitcher ROI Isolation (`PitcherROITracker`)**:
   - Isolate the pitcher mound zone on standard center-field broadcast views and maintain an Exponential Moving Average (EMA) bounding box centered on the pitcher's mid-hip/mid-shoulder centroid so background actors never hijack single-pose detection.

---

## 3. Signal Processing & 4-Phase Pitching Delivery Segmentation

Raw frame-by-frame pose coordinates exhibit high-frequency jitter during rapid arm acceleration.
1. **Savitzky-Golay Filtering**:
   - Smooth joint coordinate and angular trajectories using `scipy.signal.savgol_filter(window_length=7, polyorder=2)` (adapting `window_length` safely to odd integers $\le N$ on short clips) to compute clean first-order angular derivatives ($\omega = d\theta / dt$).
2. **4 Canonical Delivery Phases (`detect_delivery_phases`)**:
   - **Phase 1 — Peak Leg Lift (`leg_lift_frame`)**: Minimum screen `y` (highest vertical elevation) of the lead knee (`LEFT_KNEE` for RHP, `RIGHT_KNEE` for LHP) prior to forward drive.
   - **Phase 2 — Foot Plant (`foot_plant_frame`)**: Frame after `leg_lift_frame` where horizontal stride extension reaches $\ge 90\%$ of peak stride and lead-ankle vertical velocity stabilizes.
   - **Phase 3 — Max External Rotation / Layback (`mer_frame`)**: Frame between `foot_plant_frame` and release with minimum elbow flexion / maximum forearm layback combined with peak torso rotation.
   - **Phase 4 — Ball Release (`release_frame`)**: Frame at or immediately after `mer_frame` (bounded before `camera_cut_frame`) maximizing throwing wrist extension + angular velocity peak, replacing naive `max(release_height)`.
   - Enforce causal ordering invariant: `leg_lift_frame < foot_plant_frame <= mer_frame <= release_frame`.

---

## 4. Extended 10-Feature MLB Pitch Tipping Vector

Combine the 5 structural release angles with 5 biomechanical tipping indicators documented by MLB scouting departments:
1. `arm_slot_angle` (°): Throwing shoulder-to-wrist elevation angle at release.
2. `release_height` (norm): Vertical release point height.
3. `elbow_flexion_angle` (°): 3D shoulder-elbow-wrist angle at release.
4. `trunk_tilt` (°): Spine inclination from vertical at release.
5. `stride_extension` (torsox): Lead-to-trail ankle distance normalized by torso length.
6. `glove_tuck_distance` (m): 3D distance between glove wrist and torso chest center at MER/release (flared glove vs tight tuck is a classic breaking-ball vs changeup tell).
7. `hip_shoulder_separation` (°): Torsional angle differential between pelvic hip axis and scapular shoulder axis at Foot Plant.
8. `arm_angular_velocity` (°/s): Peak throwing-arm angular velocity leading into Ball Release (detects involuntary offspeed arm deceleration).
9. `delivery_tempo_s` (s): Elapsed time from Peak Leg Lift to Ball Release (`(release_frame - leg_lift_frame) / fps`).
10. `forward_release_extension` (m): Forward 3D distance of the throwing wrist relative to mid-hip at Ball Release.

---

## 5. Intra-Pitcher Baseline Normalization & Honest GroupKFold Validation

1. **`PitcherBaselineNormalizer` (Zero-Leakage Sklearn Transformer)**:
   - Because a sidearm LHP's fastball (`~28°` arm slot) is lower than an overhand RHP's curveball (`~39°`), raw multi-pitcher angles conflate *pitcher identity* with *pitch category*.
   - Fit per-pitcher baseline $(\mu_{p,f}, \sigma_{p,f})$ strictly on training rows (`fit`), falling back to global training $(\mu_f, \sigma_f)$ for unseen pitchers or rows without `pitcher_id`.
   - Emit relative deviation features ($\Delta = x - \mu_{p,f}$ and $Z = (x - \mu_{p,f}) / \max(\sigma_{p,f}, \epsilon)$) alongside raw features inside the model pipeline.
2. **Honest Validation & Probability Calibration**:
   - Never inflate runtime training sets with synthetic Gaussian copies of test rows.
   - Evaluate with `StratifiedGroupKFold` / `GroupKFold(groups=game_pk)` so pitches from the same broadcast game/camera calibration never leak across train and validation folds.
   - Wrap tree classifiers with `CalibratedClassifierCV` (sigmoid/isotonic when sample counts permit) and report out-of-fold `cv_accuracy`, `cv_f1_macro`, `cv_brier_score`, and `cv_log_loss`.
   - Compute SHAP attributions via `shap.TreeExplainer` on the underlying fitted tree estimator(s) over the transformed feature matrix, mapping delta/Z-score attributions cleanly back to human-readable anatomical explanations.
