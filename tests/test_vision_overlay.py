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
