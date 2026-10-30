"""
Unit tests for Statcast Biomechanical Skeleton Overlay and Base Print Extraction.
"""

from __future__ import annotations

import numpy as np

from pitch_predictor.vision.overlay import (
    draw_pitcher_skeleton,
    extract_and_annotate_release_frame,
    extract_base_release_frame,
    find_preloaded_base_frame,
)
from pitch_predictor.vision.pose_extractor import PoseExtractor


def test_draw_pitcher_skeleton_on_blank_frame():
    """Verify drawing skeleton on a synthetic image without throwing errors."""
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    extractor = PoseExtractor(mock_mode=True)
    landmarks = extractor.extract_frame_landmarks(blank_frame)

    metrics = {
        "arm_slot_angle": 49.8,
        "elbow_flexion_angle": 146.1,
        "release_height": 0.71,
        "trunk_tilt": 11.7,
    }

    annotated = draw_pitcher_skeleton(blank_frame, landmarks, p_throws="R", metrics=metrics)

    assert annotated is not None
    assert annotated.shape == (480, 640, 3)
    # Check that pixels were drawn (not all zero)
    assert annotated.sum() > 0


def test_extract_and_annotate_release_frame_mock():
    """Verify release frame annotation works in mock mode or when video is absent."""
    annotated = extract_and_annotate_release_frame(
        video_path="non_existent.mp4",
        release_frame_idx=20,
        p_throws="R",
        mock_mode=True,
    )
    assert annotated is not None
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape[2] == 3  # RGB channels


def test_extract_base_release_frame_preloaded_and_mock():
    """Verify base print frame extraction returns valid RGB numpy arrays."""
    preloaded = find_preloaded_base_frame(
        pitch_category="Fastball",
        pitcher_id=694973,
        exact_match_only=False,
    )
    assert preloaded is not None
    assert isinstance(preloaded, np.ndarray)
    assert preloaded.shape[2] == 3

    mock_base = extract_base_release_frame(
        video_path="non_existent.mp4",
        release_frame_idx=20,
        mock_mode=True,
    )
    assert mock_base is not None
    assert isinstance(mock_base, np.ndarray)
    assert mock_base.shape[2] == 3


def test_get_joint_color_by_z_thresholds():
    """Verify adaptive joint heatmap colors resolve correctly according to Z-score thresholds."""
    from pitch_predictor.vision.overlay import (
        COLOR_HEATMAP_ANOMALY_BGR,
        COLOR_HEATMAP_ANOMALY_RGB,
        COLOR_HEATMAP_NORMAL_BGR,
        COLOR_HEATMAP_NORMAL_RGB,
        COLOR_HEATMAP_WARNING_BGR,
        COLOR_HEATMAP_WARNING_RGB,
        get_joint_color_by_z,
    )

    # Normal (< 1.0)
    assert get_joint_color_by_z(0.0, bgr=True) == COLOR_HEATMAP_NORMAL_BGR
    assert get_joint_color_by_z(0.85, bgr=False) == COLOR_HEATMAP_NORMAL_RGB
    assert get_joint_color_by_z(-0.42, bgr=True) == COLOR_HEATMAP_NORMAL_BGR
    assert get_joint_color_by_z(None, bgr=True) == COLOR_HEATMAP_NORMAL_BGR

    # Warning (1.0 <= |Z| < 1.75)
    assert get_joint_color_by_z(1.0, bgr=True) == COLOR_HEATMAP_WARNING_BGR
    assert get_joint_color_by_z(-1.45, bgr=False) == COLOR_HEATMAP_WARNING_RGB
    assert get_joint_color_by_z(1.74, bgr=True) == COLOR_HEATMAP_WARNING_BGR

    # Pitch Tipping Anomaly (|Z| >= 1.75)
    assert get_joint_color_by_z(1.75, bgr=True) == COLOR_HEATMAP_ANOMALY_BGR
    assert get_joint_color_by_z(-2.30, bgr=False) == COLOR_HEATMAP_ANOMALY_RGB
    assert get_joint_color_by_z(3.50, bgr=True) == COLOR_HEATMAP_ANOMALY_BGR


def test_draw_pitcher_skeleton_with_anatomical_heatmap_z_scores():
    """Verify drawing skeleton with explicit Z-scores applies heatmap without error."""
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    extractor = PoseExtractor(mock_mode=True)
    landmarks = extractor.extract_frame_landmarks(blank_frame)

    metrics = {
        "arm_slot_angle": 55.0,
        "elbow_flexion_angle": 130.0,
        "release_height": 0.65,
        "trunk_tilt": 18.0,
        "glove_tuck_distance": 0.38,
        "hip_shoulder_separation": 28.0,
        "z_scores": {
            "arm_slot_angle": 0.4,       # Normal (< 1.0)
            "elbow_flexion_angle": -1.3,  # Warning (1.0 <= |Z| < 1.75)
            "release_height": -2.1,      # Tipping anomaly (|Z| >= 1.75)
            "trunk_tilt": 1.9,           # Tipping anomaly (|Z| >= 1.75)
            "glove_tuck_distance": 1.1,  # Warning
            "hip_shoulder_separation": -0.5, # Normal
        },
    }

    annotated = draw_pitcher_skeleton(blank_frame, landmarks, p_throws="R", metrics=metrics)
    assert annotated is not None
    assert annotated.shape == (480, 640, 3)
    assert annotated.sum() > 0

