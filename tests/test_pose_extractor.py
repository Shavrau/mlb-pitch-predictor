"""
Unit and integration tests for Biomechanical Pose Extraction (Passo 2).
Tests cover geometric angle calculations, release frame detection,
and video pose extraction with MediaPipe Pose.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pitch_predictor.vision.pose_extractor import (
    PoseExtractor,
    batch_extract_poses,
    calculate_arm_slot_angle,
    calculate_elbow_flexion,
    calculate_stride_extension,
    calculate_trunk_tilt,
    detect_release_frame,
)

# ---------------------------------------------------------------------------
# Geometric Unit Tests (Exact Analytic Vectors)
# ---------------------------------------------------------------------------


def test_calculate_arm_slot_angle_right_handed():
    """Verify arm slot angle for a right-handed pitcher with known geometric positions."""
    # Shoulder at (0.5, 0.5), Wrist directly above at (0.5, 0.2) -> 90 degrees (Overhand)
    shoulder = {"x": 0.5, "y": 0.5, "z": 0.0}
    wrist_overhand = {"x": 0.5, "y": 0.2, "z": 0.0}
    assert math.isclose(calculate_arm_slot_angle(shoulder, wrist_overhand, p_throws="R"), 90.0, abs_tol=1e-3)

    # Wrist directly to the side at (0.8, 0.5) -> 0 degrees (Sidearm / submarine horizontal)
    wrist_sidearm = {"x": 0.8, "y": 0.5, "z": 0.0}
    assert math.isclose(calculate_arm_slot_angle(shoulder, wrist_sidearm, p_throws="R"), 0.0, abs_tol=1e-3)

    # Wrist at 45 degrees up and to the right
    wrist_three_quarters = {"x": 0.7, "y": 0.3, "z": 0.0}
    angle_3q = calculate_arm_slot_angle(shoulder, wrist_three_quarters, p_throws="R")
    assert 40.0 <= angle_3q <= 50.0


def test_calculate_arm_slot_angle_left_handed():
    """Verify arm slot angle for a left-handed pitcher."""
    shoulder = {"x": 0.5, "y": 0.5, "z": 0.0}
    # Left hand throwing arm extends to the left (x < shoulder.x)
    wrist_overhand = {"x": 0.5, "y": 0.2, "z": 0.0}
    assert math.isclose(calculate_arm_slot_angle(shoulder, wrist_overhand, p_throws="L"), 90.0, abs_tol=1e-3)

    wrist_sidearm = {"x": 0.2, "y": 0.5, "z": 0.0}
    assert math.isclose(calculate_arm_slot_angle(shoulder, wrist_sidearm, p_throws="L"), 0.0, abs_tol=1e-3)


def test_calculate_elbow_flexion_right_angle():
    """Verify elbow flexion calculation for orthogonal vectors (90 degrees)."""
    # Shoulder at (0, 1, 0), Elbow at (0, 0, 0), Wrist at (1, 0, 0)
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
    # Perfectly upright torso (shoulders aligned above hips)
    ls = {"x": 0.4, "y": 0.3}
    rs = {"x": 0.6, "y": 0.3}
    lh = {"x": 0.4, "y": 0.7}
    rh = {"x": 0.6, "y": 0.7}
    assert math.isclose(calculate_trunk_tilt(ls, rs, lh, rh), 0.0, abs_tol=1e-3)

    # 45-degree lateral tilt
    ls_tilt = {"x": 0.6, "y": 0.5}
    rs_tilt = {"x": 0.8, "y": 0.5}
    tilt = calculate_trunk_tilt(ls_tilt, rs_tilt, lh, rh)
    assert tilt > 0.0


def test_calculate_stride_extension():
    """Verify stride extension relative to torso height."""
    lead_ankle = {"x": 0.8, "y": 0.9}
    trail_ankle = {"x": 0.2, "y": 0.9}
    torso_len = 0.4  # Mid-hip to mid-shoulder distance

    stride = calculate_stride_extension(lead_ankle, trail_ankle, torso_len)
    assert math.isclose(stride, 1.5, abs_tol=1e-3)


def test_detect_release_frame():
    """Verify detection of the pitch release frame based on maximum release height."""
    frames_data = [
        {"frame": 10, "rel_h": 0.40, "arm_angle": 30.0, "flexion": 110.0, "trunk_tilt": 8.0, "stride": 1.1},
        {"frame": 15, "rel_h": 0.55, "arm_angle": 42.0, "flexion": 125.0, "trunk_tilt": 10.0, "stride": 1.3},
        {"frame": 20, "rel_h": 0.72, "arm_angle": 51.0, "flexion": 148.0, "trunk_tilt": 12.0, "stride": 1.5},  # Peak
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


def test_pose_extractor_extract_video_mock():
    """Verify extract_video_poses works in mock mode for non-existent video in CI."""
    extractor = PoseExtractor(mock_mode=True)
    result = extractor.extract_video_poses("non_existent_video.mp4", p_throws="R")

    assert "summary" in result
    assert "frames" in result
    assert result["summary"]["arm_slot_angle"] is not None
    assert result["summary"]["release_height"] is not None
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
    # Skenes throws with a 3/4 to high slot (~40-85 deg) and high release height
    assert 30.0 <= result["summary"]["arm_slot_angle"] <= 85.0
    assert result["summary"]["release_height"] > 0.5
    assert result["summary"]["elbow_flexion_angle"] > 100.0
    assert len(result["frames"]) >= 10


def test_batch_extract_poses():
    """Verify batch extraction merges manifest with biomechanical features."""
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
    assert "arm_slot_angle" in features_df.columns
    assert "release_height" in features_df.columns
    assert "elbow_flexion_angle" in features_df.columns
    assert "trunk_tilt" in features_df.columns
    assert "stride_extension" in features_df.columns
    assert "release_frame" in features_df.columns
    assert features_df["pitch_category"].iloc[0] == "Fastball"
