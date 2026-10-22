"""
Unit tests for the Unified Fallback & Offline Inference Engine.
"""

from __future__ import annotations

from pitch_predictor.fallback_engine import (
    predict_multimodal_fallback,
    predict_tactical_fallback,
    simulate_12_counts_fallback,
)
from pitch_predictor.scenarios import MLB_PITCHERS
from pitch_predictor.scout_analytics import generate_12_counts_batch


def test_predict_tactical_fallback_behind_in_count_favors_fastball() -> None:
    """Verify that on 3-0 (pitcher behind), the fallback engine projects Fastball to force a strike."""
    raw = {
        "balls": 3,
        "strikes": 0,
        "prev_pitch_category": "Offspeed",
        "prev_pitch_type": "CH",
        "prev_pitch_zone": 13,
        "prev_pitch_result": "B",
        "pitcher_fb_prior": 0.55,
        "pitcher_bb_prior": 0.30,
        "pitcher_os_prior": 0.15,
        "p_throws": "R",
        "stand": "R",
    }
    result = predict_tactical_fallback(raw)

    assert result["status"] == "fallback"
    assert result["is_fallback"] is True
    assert result["prediction"] == "Fastball"
    assert result["probabilities"]["Fastball"] > 0.50
    # Probabilities must sum to 1.0
    total_p = sum(result["probabilities"].values())
    assert abs(total_p - 1.0) < 1e-3


def test_predict_tactical_fallback_ahead_in_count_favors_breaking_ball() -> None:
    """Verify that on 0-2 (pitcher ahead), the fallback engine projects Breaking Ball for putaway."""
    raw = {
        "balls": 0,
        "strikes": 2,
        "prev_pitch_category": "Fastball",
        "prev_pitch_type": "FF",
        "prev_pitch_zone": 2,
        "prev_pitch_result": "S",
        "pitcher_fb_prior": 0.55,
        "pitcher_bb_prior": 0.30,
        "pitcher_os_prior": 0.15,
        "p_throws": "R",
        "stand": "R",
    }
    result = predict_tactical_fallback(raw)

    assert result["prediction"] == "Breaking Ball"
    assert result["probabilities"]["Breaking Ball"] > result["probabilities"]["Fastball"]


def test_predict_multimodal_fallback_agreement() -> None:
    """Verify alignment AGREEMENT when both situational and biomechanical cues point to Fastball."""
    raw = {
        "balls": 3,
        "strikes": 1,
        "prev_pitch_category": "Fastball",
        "pitcher_fb_prior": 0.65,
        "pitcher_bb_prior": 0.25,
        "pitcher_os_prior": 0.10,
        "arm_slot_angle": 50.0,
        "elbow_flexion_angle": 145.0,  # fully extended -> Fastball
        "release_height": 0.71,
    }
    result = predict_multimodal_fallback(raw)

    assert result["statcast_prediction"] == "Fastball"
    assert result["vision_prediction"] == "Fastball"
    assert result["prediction"] == "Fastball"
    assert result["alignment_status"] == "AGREEMENT"
    assert "Alinhamento total" in result["summary_text"]


def test_predict_multimodal_fallback_tipping_override() -> None:
    """Verify TIPPING_OVERRIDE when situational context is Fastball but physical tipping signals Breaking Ball."""
    raw = {
        "balls": 3,
        "strikes": 0,
        "prev_pitch_category": "Fastball",
        "pitcher_fb_prior": 0.50,
        "pitcher_bb_prior": 0.30,
        "pitcher_os_prior": 0.20,
        "arm_slot_angle": 35.0,  # low arm slot
        "elbow_flexion_angle": 105.0,  # bent elbow -> Breaking Ball tipping!
        "release_height": 0.65,
    }
    result = predict_multimodal_fallback(raw)

    assert result["vision_prediction"] == "Breaking Ball"
    # Even if statcast would pick Fastball on 3-0, vision tipping can override
    assert result["alignment_status"] in ("TIPPING_OVERRIDE", "DIVERGENT")


def test_simulate_12_counts_fallback_covers_all_counts() -> None:
    """Verify that simulate_12_counts_fallback generates predictions for all 12 MLB counts."""
    profile = MLB_PITCHERS["Corbin Burnes"]
    batch = generate_12_counts_batch(profile, stand="R")

    matrix = simulate_12_counts_fallback(batch)
    assert len(matrix) == 12
    for b in range(4):
        for s in range(3):
            assert f"{b}-{s}" in matrix
            assert matrix[f"{b}-{s}"] in ("Fastball", "Breaking Ball", "Offspeed")
