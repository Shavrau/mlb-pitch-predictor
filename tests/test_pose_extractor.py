"""
Unit and integration tests for Biomechanical Pose Extraction (v2 — 3D Temporal Kinematics).
Tests cover geometric angle calculations, 3D tipping features, HSV camera-cut rejection,
Savitzky-Golay smoothing, 4-phase delivery detection, mound ROI tracking, and MediaPipe Pose.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pitch_predictor.vision.pose_extractor import (
    EXTENDED_BIOMECHANICAL_FEATURE_COLS,
    PitcherROITracker,
    PoseExtractor,
    batch_extract_poses,
    calculate_arm_slot_angle,
    calculate_elbow_flexion,
    calculate_glove_tuck_distance,
    calculate_hip_shoulder_separation,
    calculate_stride_extension,
    calculate_trunk_tilt,
    detect_camera_cut_frame,
    detect_delivery_phases,
    detect_release_frame,
    smooth_kinematic_series,
)

# ---------------------------------------------------------------------------
# Geometric Unit Tests (Exact Analytic Vectors)
# ---------------------------------------------------------------------------


def test_calculate_arm_slot_angle_right_handed():
    """Verify arm slot angle for a right-handed pitcher with known geometric positions."""
    shoulder = {"x": 0.5, "y": 0.5, "z": 0.0}
    wrist_overhand = {"x": 0.5, "y": 0.2, "z": 0.0}
    assert math.isclose(calculate_arm_slot_angle(shoulder, wrist_overhand, p_throws="R"), 90.0, abs_tol=1e-3)

    wrist_sidearm = {"x": 0.8, "y": 0.5, "z": 0.0}
    assert math.isclose(calculate_arm_slot_angle(shoulder, wrist_sidearm, p_throws="R"), 0.0, abs_tol=1e-3)

    wrist_three_quarters = {"x": 0.7, "y": 0.3, "z": 0.0}
    angle_3q = calculate_arm_slot_angle(shoulder, wrist_three_quarters, p_throws="R")
    assert 40.0 <= angle_3q <= 50.0


def test_calculate_arm_slot_angle_left_handed():
    """Verify arm slot angle for a left-handed pitcher."""
    shoulder = {"x": 0.5, "y": 0.5, "z": 0.0}
    wrist_overhand = {"x": 0.5, "y": 0.2, "z": 0.0}
    assert math.isclose(calculate_arm_slot_angle(shoulder, wrist_overhand, p_throws="L"), 90.0, abs_tol=1e-3)

    wrist_sidearm = {"x": 0.2, "y": 0.5, "z": 0.0}
    assert math.isclose(calculate_arm_slot_angle(shoulder, wrist_sidearm, p_throws="L"), 0.0, abs_tol=1e-3)


def test_calculate_elbow_flexion_right_angle():
    """Verify elbow flexion calculation for orthogonal vectors (90 degrees)."""
    shoulder = {"x": 0.0, "y": 1.0, "z": 0.0}
    elbow = {"x": 0.0, "y": 0.0, "z": 0.0}
    wrist = {"x": 1.0, "y": 0.0, "z": 0.0}

    angle = calculate_elbow_flexion(shoulder, elbow, wrist)
    assert math.isclose(angle, 90.0, abs_tol=1e-3)


def test_calculate_elbow_flexion_straight_arm():
    """Verify elbow flexion for fully extended straight arm (180 degrees)."""
    shoulder = {"x": 0.0, "y": 1.0, "z": 0.0}
    elbow = {"x": 0.0, "y": 0.5, "z": 0.0}
    wrist = {"x": 0.0, "y": 0.0, "z": 0.0}

    angle = calculate_elbow_flexion(shoulder, elbow, wrist)
    assert math.isclose(angle, 180.0, abs_tol=1e-3)


def test_calculate_trunk_tilt_vertical_and_inclined():
    """Verify torso tilt relative to vertical."""
    ls = {"x": 0.4, "y": 0.3}
    rs = {"x": 0.6, "y": 0.3}
    lh = {"x": 0.4, "y": 0.7}
    rh = {"x": 0.6, "y": 0.7}
    assert math.isclose(calculate_trunk_tilt(ls, rs, lh, rh), 0.0, abs_tol=1e-3)

    ls_tilt = {"x": 0.6, "y": 0.5}
    rs_tilt = {"x": 0.8, "y": 0.5}
    tilt = calculate_trunk_tilt(ls_tilt, rs_tilt, lh, rh)
    assert tilt > 0.0


def test_calculate_stride_extension():
    """Verify stride extension relative to torso height."""
    lead_ankle = {"x": 0.8, "y": 0.9}
    trail_ankle = {"x": 0.2, "y": 0.9}
    torso_len = 0.4

    stride = calculate_stride_extension(lead_ankle, trail_ankle, torso_len)
    assert math.isclose(stride, 1.5, abs_tol=1e-3)


# ---------------------------------------------------------------------------
# New v2 3D & Temporal Biomechanics Unit Tests
# ---------------------------------------------------------------------------


def test_extended_biomechanical_feature_cols_count():
    """Verify all 10 extended MLB tipping features are declared."""
    assert len(EXTENDED_BIOMECHANICAL_FEATURE_COLS) == 10
    expected_new = {
        "glove_tuck_distance",
        "hip_shoulder_separation",
        "arm_angular_velocity",
        "delivery_tempo_s",
        "forward_release_extension",
    }
    assert expected_new.issubset(set(EXTENDED_BIOMECHANICAL_FEATURE_COLS))


def test_calculate_glove_tuck_and_hip_shoulder_separation():
    """Verify 3D glove tuck distance and hip-shoulder separation angle calculations."""
    ls = {"x": -0.20, "y": -0.40, "z": 0.0}
    rs = {"x": 0.20, "y": -0.40, "z": 0.0}
    lh = {"x": -0.15, "y": 0.0, "z": -0.15}
    rh = {"x": 0.15, "y": 0.0, "z": 0.15}

    tucked_glove = {"x": -0.05, "y": -0.25, "z": -0.08}
    flared_glove = {"x": -0.45, "y": -0.30, "z": -0.35}

    d_tucked = calculate_glove_tuck_distance(tucked_glove, ls, rs, lh, rh)
    d_flared = calculate_glove_tuck_distance(flared_glove, ls, rs, lh, rh)
    assert d_flared > d_tucked > 0.0

    # Hips rotated 45 deg in X-Z plane relative to shoulders along X axis -> ~45 deg separation
    sep_deg = calculate_hip_shoulder_separation(ls, rs, lh, rh)
    assert 40.0 <= sep_deg <= 50.0


def test_detect_camera_cut_frame_hsv_histogram():
    """Verify automatic broadcast camera-cut detection when scene switches abruptly."""
    frames = []
    # First 25 frames: pitcher mound palette (earthy brown/green BGR)
    for i in range(25):
        img = np.zeros((120, 160, 3), dtype=np.uint8)
        img[:60, :] = (45, 95, 60)
        img[60:, :] = (55, 80, 125)
        img[:, (i % 10) : (i % 10) + 20] = (70, 110, 90)
        frames.append(img)

    # Frame 25 onward: abrupt cut to bright blue sky / outfield wall
    for _ in range(10):
        cut_img = np.full((120, 160, 3), (240, 180, 20), dtype=np.uint8)
        frames.append(cut_img)

    cut_idx = detect_camera_cut_frame(frames, min_frame=8, corr_threshold=0.65)
    assert cut_idx == 25


def test_smooth_kinematic_series_reduces_jitter():
    """Verify Savitzky-Golay smoothing reduces high-frequency pose jitter while preserving trend."""
    t = np.linspace(0, np.pi, 31)
    clean_signal = 50.0 + 20.0 * np.sin(t)
    rng = np.random.default_rng(7)
    noisy_signal = clean_signal + rng.normal(0.0, 3.0, size=len(t))

    smoothed = smooth_kinematic_series(noisy_signal, window_length=7, polyorder=2)
    assert len(smoothed) == len(noisy_signal)
    err_noisy = np.mean(np.abs(noisy_signal - clean_signal))
    err_smooth = np.mean(np.abs(smoothed - clean_signal))
    assert err_smooth < err_noisy


def test_detect_delivery_phases_ordering_rhp_and_lhp():
    """Verify 4-phase delivery segmentation enforces Leg Lift < Foot Plant <= MER <= Release."""
    frames_data = []
    for f in range(60):
        # Simulate canonical 60-frame delivery:
        # Leg lift peak around frame 15 (minimum lead_knee_y)
        lead_knee_y = 0.72 - 0.25 * math.exp(-((f - 15) ** 2) / 36.0)
        # Stride ramps up and plateaus around frame 34 (foot plant)
        stride = 0.65 + 0.80 / (1.0 + math.exp(-(f - 28) / 3.0))
        # Forearm layback / MER minimum elbow flexion around frame 42, extending to 150 at release frame 47
        if f < 42:
            flexion = 115.0 - 28.0 * math.exp(-((f - 41) ** 2) / 18.0)
        else:
            flexion = 87.0 + 62.0 * min(1.0, (f - 41) / 6.0)
        # Forward extension & release height peak around frame 47
        forward_ext = 0.20 + 0.55 * math.exp(-((f - 47) ** 2) / 25.0)
        rel_h = 0.48 + 0.24 * math.exp(-((f - 47) ** 2) / 30.0)
        arm_angle = 32.0 + 22.0 * math.exp(-((f - 47) ** 2) / 40.0)

        frames_data.append({
            "frame": f,
            "lead_knee_y": lead_knee_y,
            "stride": stride,
            "flexion": flexion,
            "forward_ext": forward_ext,
            "rel_h": rel_h,
            "arm_angle": arm_angle,
            "trunk_tilt": 10.0 + 0.12 * f,
            "glove_tuck": 0.28 - 0.08 * min(1.0, f / 47.0),
            "hip_shoulder_sep": 18.0 + 26.0 * math.exp(-((f - 34) ** 2) / 30.0),
            "confidence": 0.95,
        })

    phases = detect_delivery_phases(frames_data, fps=60.0, max_delivery_frame=55)
    assert phases["leg_lift_frame"] < phases["foot_plant_frame"]
    assert phases["foot_plant_frame"] <= phases["mer_frame"]
    assert phases["mer_frame"] <= phases["release_frame"]
    assert 12 <= phases["leg_lift_frame"] <= 18
    assert 43 <= phases["release_frame"] <= 50
    assert phases["delivery_tempo_s"] > 0.3
    assert phases["arm_angular_velocity"] > 0.0
    assert phases["hip_shoulder_separation"] > 30.0


def test_pitcher_roi_tracker_isolates_mound_and_remaps():
    """Verify PitcherROITracker crops mound region and remaps coordinates back to full frame."""
    tracker = PitcherROITracker(ema_alpha=0.5)
    full_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    crop, bbox = tracker.crop_frame(full_frame)
    assert crop.shape[0] < 720 and crop.shape[1] < 1280

    local_lm = {
        "MID_TEST": {"x": 0.5, "y": 0.5, "z": -0.1, "visibility": 0.95},
        "LEFT_HIP": {"x": 0.48, "y": 0.60, "z": 0.0, "visibility": 0.95},
        "RIGHT_HIP": {"x": 0.52, "y": 0.60, "z": 0.0, "visibility": 0.95},
    }
    remapped = tracker.remap_landmarks(local_lm, bbox, full_width=1280, full_height=720)
    assert 0.35 <= remapped["MID_TEST"]["x"] <= 0.68
    assert 0.35 <= remapped["MID_TEST"]["y"] <= 0.95


def test_detect_release_frame():
    """Verify detection of the pitch release frame based on maximum release height."""
    frames_data = [
        {"frame": 10, "rel_h": 0.40, "arm_angle": 30.0, "flexion": 110.0, "trunk_tilt": 8.0, "stride": 1.1},
        {"frame": 15, "rel_h": 0.55, "arm_angle": 42.0, "flexion": 125.0, "trunk_tilt": 10.0, "stride": 1.3},
        {"frame": 20, "rel_h": 0.72, "arm_angle": 51.0, "flexion": 148.0, "trunk_tilt": 12.0, "stride": 1.5},
        {"frame": 25, "rel_h": 0.60, "arm_angle": 45.0, "flexion": 130.0, "trunk_tilt": 15.0, "stride": 1.4},
        {"frame": 30, "rel_h": 0.35, "arm_angle": 20.0, "flexion": 95.0, "trunk_tilt": 18.0, "stride": 1.2},
    ]

    release = detect_release_frame(frames_data)
    assert release["release_frame"] == 20
    assert release["release_height"] == 0.72
    assert release["arm_slot_angle"] == 51.0
    assert release["elbow_flexion_angle"] == 148.0
    assert release["trunk_tilt"] == 12.0
    assert release["stride_extension"] == 1.5


def test_detect_release_frame_empty():
    """Verify graceful handling when no frames are provided."""
    release = detect_release_frame([])
    assert release["release_frame"] is None
    assert release["release_height"] is None
    assert release["arm_slot_angle"] is None


# ---------------------------------------------------------------------------
# Mock & Headless Video Pose Extractor Tests
# ---------------------------------------------------------------------------


def test_pose_extractor_initialization():
    """Verify PoseExtractor initializes with default or custom model path."""
    extractor = PoseExtractor(model_path="models/pose_landmarker_lite.task")
    assert extractor.model_path.exists() or not extractor.has_detector


def test_pose_extractor_extract_frame_mock():
    """Verify frame extraction handles synthetic/empty frame gracefully."""
    extractor = PoseExtractor(mock_mode=True)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    landmarks = extractor.extract_frame_landmarks(dummy_frame)
    assert landmarks is not None
    assert "RIGHT_SHOULDER" in landmarks
    assert "RIGHT_WRIST" in landmarks


def test_pose_extractor_extract_video_mock_includes_extended_features_and_phases():
    """Verify extract_video_poses outputs all 10 extended features and 4 phases in mock mode."""
    extractor = PoseExtractor(mock_mode=True)
    result = extractor.extract_video_poses("non_existent_video.mp4", p_throws="R")

    assert "summary" in result
    assert "frames" in result
    assert "phases" in result
    for col in EXTENDED_BIOMECHANICAL_FEATURE_COLS:
        assert result["summary"].get(col) is not None, f"Missing {col} in summary"
    phases = result["phases"]
    assert phases["leg_lift_frame"] < phases["foot_plant_frame"] <= phases["mer_frame"] <= phases["release_frame"]
    assert len(result["frames"]) > 0


# ---------------------------------------------------------------------------
# Integration Tests on Real MLB Clips
# ---------------------------------------------------------------------------


def test_pose_extractor_real_clip_paul_skenes():
    """Verify pose extraction on real Paul Skenes Fastball clip if available."""
    video_path = Path("data/videos/694973/745170_31_4_Fastball.mp4")
    if not video_path.exists():
        pytest.skip("Real video clip not present locally")

    extractor = PoseExtractor()
    result = extractor.extract_video_poses(video_path, p_throws="R", sample_step=2)

    assert result["summary"]["release_frame"] is not None
    assert 20.0 <= result["summary"]["arm_slot_angle"] <= 88.0
    assert result["summary"]["release_height"] > 0.45
    assert result["summary"]["elbow_flexion_angle"] > 80.0
    assert result["summary"]["glove_tuck_distance"] is not None
    assert len(result["frames"]) >= 10


def test_batch_extract_poses():
    """Verify batch extraction merges manifest with all 10 biomechanical features."""
    manifest_df = pd.DataFrame([
        {
            "game_pk": 745170,
            "at_bat_number": 31,
            "pitch_number": 4,
            "pitcher_id": 694973,
            "pitch_type": "FF",
            "pitch_category": "Fastball",
            "video_path": "data/videos/694973/745170_31_4_Fastball.mp4",
            "status": "COMPLETED",
        }
    ])

    extractor = PoseExtractor(mock_mode=True)
    features_df = batch_extract_poses(manifest_df, extractor=extractor)

    assert len(features_df) == 1
    for col in EXTENDED_BIOMECHANICAL_FEATURE_COLS:
        assert col in features_df.columns
    assert "release_frame" in features_df.columns
    assert "leg_lift_frame" in features_df.columns
    assert features_df["pitch_category"].iloc[0] == "Fastball"
