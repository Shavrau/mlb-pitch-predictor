"""
Unit and integration tests for multi-pitcher ingestion, registry, and UI features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pitch_predictor.scenarios import MLB_PITCHERS
from pitch_predictor.vision.overlay import (
    extract_and_annotate_release_frame,
    find_preloaded_release_frame,
)


def test_mlb_pitchers_registry_structure_and_priors() -> None:
    """Verify that all MLB pitchers in registry have valid properties and priors sum to ~1.0."""
    assert len(MLB_PITCHERS) >= 5
    required_pitchers = ["Paul Skenes", "Corbin Burnes", "Logan Webb", "Chris Sale", "Yoshinobu Yamamoto"]
    for name in required_pitchers:
        assert name in MLB_PITCHERS
        p = MLB_PITCHERS[name]
        assert "pitcher_id" in p
        assert isinstance(p["pitcher_id"], int)
        assert p["p_throws"] in ("R", "L")
        assert "signature_arsenal" in p
        assert "bio_profile" in p
        bio = p["bio_profile"]
        assert 15.0 <= bio["arm_slot_angle"] <= 85.0
        assert 70.0 <= bio["elbow_flexion_angle"] <= 165.0
        assert 0.50 <= bio["release_height"] <= 0.85

        # Priors sum to ~1.0
        prior_sum = p["pitcher_fb_prior"] + p["pitcher_bb_prior"] + p["pitcher_os_prior"]
        assert abs(prior_sum - 1.0) < 0.001


def test_yoshinobu_yamamoto_profile() -> None:
    """Verify Yoshinobu Yamamoto is registered with Dodgers, RHP, and signature splitter."""
    yy = MLB_PITCHERS["Yoshinobu Yamamoto"]
    assert yy["pitcher_id"] == 808967
    assert yy["team"] == "Los Angeles Dodgers"
    assert yy["p_throws"] == "R"
    assert "Splitter" in yy["signature_arsenal"]
    assert 40.0 <= yy["bio_profile"]["arm_slot_angle"] <= 55.0


def test_chris_sale_lefty_profile() -> None:
    """Verify Chris Sale is correctly registered as a left-handed pitcher with low release slot."""
    sale = MLB_PITCHERS["Chris Sale"]
    assert sale["p_throws"] == "L"
    assert sale["pitcher_id"] == 519242
    assert sale["bio_profile"]["arm_slot_angle"] < 35.0  # Low 3/4 / sidearm slot


def test_find_preloaded_release_frame_for_each_pitcher() -> None:
    """Verify preloaded release frames are found for all four MLB star pitchers."""
    for name, p_data in MLB_PITCHERS.items():
        pid = p_data["pitcher_id"]
        # Fastball
        fb_frame = find_preloaded_release_frame(
            pitch_category="Fastball",
            pitcher_id=pid,
            exact_match_only=False,
        )
        assert fb_frame is not None, f"Could not find Fastball release frame for {name} ({pid})"
        assert isinstance(fb_frame, np.ndarray)
        assert fb_frame.shape[2] == 3

        # Breaking Ball
        bb_frame = find_preloaded_release_frame(
            pitch_category="Breaking Ball",
            pitcher_id=pid,
            exact_match_only=False,
        )
        assert bb_frame is not None, f"Could not find Breaking Ball release frame for {name} ({pid})"


def test_extract_and_annotate_release_frame_multi_handedness() -> None:
    """Verify extract_and_annotate_release_frame works for both right-handed and left-handed pitchers."""
    # Right-handed (Skenes)
    rhp_frame = extract_and_annotate_release_frame(
        video_path="",
        release_frame_idx=200,
        p_throws="R",
        metrics={"arm_slot_angle": 50.0, "elbow_flexion_angle": 145.0, "release_height": 0.71, "trunk_tilt": 12.0},
        pitch_category="Fastball",
        pitcher_id=694973,
    )
    assert rhp_frame is not None
    assert rhp_frame.shape == (720, 1280, 3) or rhp_frame.shape == (540, 720, 3)

    # Left-handed (Sale)
    lhp_frame = extract_and_annotate_release_frame(
        video_path="",
        release_frame_idx=200,
        p_throws="L",
        metrics={"arm_slot_angle": 25.0, "elbow_flexion_angle": 105.0, "release_height": 0.62, "trunk_tilt": 24.0},
        pitch_category="Fastball",
        pitcher_id=519242,
    )
    assert lhp_frame is not None
    assert lhp_frame.shape == (720, 1280, 3) or lhp_frame.shape == (540, 720, 3)


def test_manifest_and_bio_csv_exist_for_all_pitchers() -> None:
    """Verify manifest and biomechanical features exist on disk for all 4 pitchers."""
    for name, p_data in MLB_PITCHERS.items():
        pid = p_data["pitcher_id"]
        m_path = Path(f"data/videos/{pid}/manifest.csv")
        b_path = Path(f"data/videos/{pid}/biomechanical_features.csv")
        assert m_path.exists(), f"Manifest missing for {name} ({pid})"
        assert b_path.exists(), f"Biomechanical features missing for {name} ({pid})"
