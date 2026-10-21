"""
Unit tests for the Databricks Serverless Model Serving client and feature contract (Red Phase TDD).

Covers:
1. Strict 24-column feature contract (`EXPECTED_FEATURE_COLUMNS`).
2. Tactical MLB feature derivation (`derive_tactical_features`) including edge cases:
   - First Pitch boundary enforcement (`pitch_number == 1` / `prev_pitch_category == "First Pitch"`)
   - Platoon handedness (`same_hand`)
   - All 12 MLB counts (`count_state` and `count_leverage`), including 0-2 and 3-2
   - Baserunner configurations (`runners_on_base` and `scoring_position`), including bases loaded
   - Negative `score_diff` handling
3. Input schema and domain validation (`validate_inference_record`).
4. Pitcher repertoire prior normalization (`normalize_priors`).
5. Databricks Serving REST payload formatting (`build_rest_payload`).
6. `PitchServingClient.predict` execution, latency reporting, and cold-start retry resilience.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from pitch_predictor.serving_client import (
    BIOMECHANICAL_FEATURE_COLUMNS,
    EXPECTED_FEATURE_COLUMNS,
    PitchServingClient,
    build_multimodal_payload,
    build_rest_payload,
    derive_multimodal_features,
    derive_tactical_features,
    normalize_priors,
    validate_inference_record,
)


def _sample_raw_inputs(**overrides: Any) -> dict[str, Any]:
    """Create a valid baseline dictionary of raw tactical inputs."""
    base: dict[str, Any] = {
        "prev_pitch_category": "Fastball",
        "prev_pitch_type": "FF",
        "prev_pitch_zone": 5,
        "prev_pitch_result": "S",
        "p_throws": "R",
        "stand": "L",
        "balls": 1,
        "strikes": 1,
        "outs_when_up": 1,
        "inning": 5,
        "inning_topbot": "Top",
        "score_diff": 0,
        "on_1b_flag": 0,
        "on_2b_flag": 0,
        "on_3b_flag": 0,
        "n_thruorder_pitcher": 2,
        "pitcher_fb_prior": 0.5543,
        "pitcher_bb_prior": 0.3053,
        "pitcher_os_prior": 0.1404,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Contract of 24 Feature Columns
# ---------------------------------------------------------------------------


def test_expected_feature_columns_exact_order_and_length() -> None:
    """Verify EXPECTED_FEATURE_COLUMNS has the exact 24 columns in the required order."""
    expected = [
        "prev_pitch_category",
        "prev_pitch_type",
        "prev_pitch_zone",
        "prev_pitch_result",
        "p_throws",
        "stand",
        "same_hand",
        "balls",
        "strikes",
        "count_state",
        "count_leverage",
        "outs_when_up",
        "inning",
        "inning_topbot",
        "score_diff",
        "on_1b_flag",
        "on_2b_flag",
        "on_3b_flag",
        "runners_on_base",
        "scoring_position",
        "n_thruorder_pitcher",
        "pitcher_fb_prior",
        "pitcher_bb_prior",
        "pitcher_os_prior",
    ]
    assert len(EXPECTED_FEATURE_COLUMNS) == 24
    assert EXPECTED_FEATURE_COLUMNS == expected


# ---------------------------------------------------------------------------
# 2. Tactical Feature Derivation
# ---------------------------------------------------------------------------


def test_derive_tactical_features_first_pitch_forces_defaults() -> None:
    """When prev_pitch_category is 'First Pitch', force type='None', zone=0, result='First Pitch'."""
    raw = _sample_raw_inputs(
        prev_pitch_category="First Pitch",
        prev_pitch_type="SL",
        prev_pitch_zone=14,
        prev_pitch_result="B",
    )
    derived = derive_tactical_features(raw)

    assert derived["prev_pitch_category"] == "First Pitch"
    assert derived["prev_pitch_type"] == "None"
    assert derived["prev_pitch_zone"] == 0
    assert derived["prev_pitch_result"] == "First Pitch"
    assert list(derived.keys()) == EXPECTED_FEATURE_COLUMNS


@pytest.mark.parametrize(
    ("p_throws", "stand", "expected_same_hand"),
    [
        ("R", "R", 1),
        ("L", "L", 1),
        ("R", "L", 0),
        ("L", "R", 0),
    ],
)
def test_derive_tactical_features_same_hand(
    p_throws: str, stand: str, expected_same_hand: int
) -> None:
    """Verify same_hand is 1 when pitcher and batter have identical handedness, else 0."""
    raw = _sample_raw_inputs(p_throws=p_throws, stand=stand)
    derived = derive_tactical_features(raw)
    assert derived["same_hand"] == expected_same_hand


@pytest.mark.parametrize(
    ("balls", "strikes", "expected_state", "expected_leverage"),
    [
        # Pitcher Ahead counts
        (0, 1, "0-1", "Pitcher Ahead"),
        (0, 2, "0-2", "Pitcher Ahead"),
        (1, 2, "1-2", "Pitcher Ahead"),
        # Pitcher Behind counts
        (1, 0, "1-0", "Pitcher Behind"),
        (2, 0, "2-0", "Pitcher Behind"),
        (3, 0, "3-0", "Pitcher Behind"),
        (3, 1, "3-1", "Pitcher Behind"),
        # Neutral counts (including 0-0 and full count 3-2)
        (0, 0, "0-0", "Neutral"),
        (1, 1, "1-1", "Neutral"),
        (2, 1, "2-1", "Neutral"),
        (2, 2, "2-2", "Neutral"),
        (3, 2, "3-2", "Neutral"),
    ],
)
def test_derive_tactical_features_count_state_and_leverage(
    balls: int, strikes: int, expected_state: str, expected_leverage: str
) -> None:
    """Verify count_state string formatting and count_leverage mapping across all 12 MLB counts."""
    raw = _sample_raw_inputs(balls=balls, strikes=strikes)
    derived = derive_tactical_features(raw)
    assert derived["count_state"] == expected_state
    assert derived["count_leverage"] == expected_leverage


@pytest.mark.parametrize(
    ("on_1b", "on_2b", "on_3b", "expected_runners", "expected_risp"),
    [
        (0, 0, 0, 0, 0),
        (1, 0, 0, 1, 0),
        (0, 1, 0, 1, 1),
        (0, 0, 1, 1, 1),
        (1, 1, 0, 2, 1),
        (1, 0, 1, 2, 1),
        (0, 1, 1, 2, 1),
        (1, 1, 1, 3, 1),  # Bases loaded
    ],
)
def test_derive_tactical_features_runners_and_scoring_position(
    on_1b: int,
    on_2b: int,
    on_3b: int,
    expected_runners: int,
    expected_risp: int,
) -> None:
    """Verify runners_on_base sum and scoring_position (RISP: 2B or 3B occupied)."""
    raw = _sample_raw_inputs(on_1b_flag=on_1b, on_2b_flag=on_2b, on_3b_flag=on_3b)
    derived = derive_tactical_features(raw)
    assert derived["runners_on_base"] == expected_runners
    assert derived["scoring_position"] == expected_risp


def test_derive_tactical_features_negative_score_diff_and_bases_loaded_full_count() -> None:
    """Verify high-leverage edge case: 3-2 count, bases loaded, negative score_diff."""
    raw = _sample_raw_inputs(
        balls=3,
        strikes=2,
        on_1b_flag=1,
        on_2b_flag=1,
        on_3b_flag=1,
        score_diff=-3,
        outs_when_up=2,
        inning=9,
        inning_topbot="Bot",
    )
    derived = derive_tactical_features(raw)
    assert derived["count_state"] == "3-2"
    assert derived["count_leverage"] == "Neutral"
    assert derived["runners_on_base"] == 3
    assert derived["scoring_position"] == 1
    assert derived["score_diff"] == -3


# ---------------------------------------------------------------------------
# 3. Record Validation
# ---------------------------------------------------------------------------


def test_validate_inference_record_accepts_valid_record() -> None:
    """Verify validate_inference_record accepts a complete, valid 24-column feature dictionary."""
    valid_record = derive_tactical_features(_sample_raw_inputs())
    assert validate_inference_record(valid_record) is True


def test_validate_inference_record_raises_on_missing_column() -> None:
    """Verify ValueError is raised when any of the 24 required columns is missing."""
    incomplete_record = derive_tactical_features(_sample_raw_inputs())
    incomplete_record.pop("pitcher_os_prior")
    with pytest.raises(ValueError, match="pitcher_os_prior|missing|column"):
        validate_inference_record(incomplete_record)


@pytest.mark.parametrize(
    ("invalid_field", "invalid_value"),
    [
        ("balls", 4),
        ("strikes", 3),
        ("prev_pitch_zone", 10),
        ("prev_pitch_zone", 15),
    ],
)
def test_validate_inference_record_raises_on_out_of_domain_values(
    invalid_field: str, invalid_value: int
) -> None:
    """Verify ValueError when balls > 3, strikes > 2, prev_pitch_zone == 10, or prev_pitch_zone > 14."""
    record = derive_tactical_features(_sample_raw_inputs())
    record[invalid_field] = invalid_value
    with pytest.raises(ValueError):
        validate_inference_record(record)


# ---------------------------------------------------------------------------
# 4. Pitcher Priors Normalization & REST Payload Builder
# ---------------------------------------------------------------------------


def test_normalize_priors_standard_and_unnormalized() -> None:
    """Verify normalize_priors scales positive numbers to sum to 1.0."""
    fb, bb, os = normalize_priors(0.5, 0.3, 0.2)
    assert fb == pytest.approx(0.5)
    assert bb == pytest.approx(0.3)
    assert os == pytest.approx(0.2)
    assert (fb + bb + os) == pytest.approx(1.0)

    fb2, bb2, os2 = normalize_priors(50.0, 30.0, 20.0)
    assert fb2 == pytest.approx(0.5)
    assert bb2 == pytest.approx(0.3)
    assert os2 == pytest.approx(0.2)


def test_normalize_priors_zero_sum_returns_league_baseline() -> None:
    """Verify normalize_priors falls back to MLB league averages when sum is 0."""
    fb, bb, os = normalize_priors(0.0, 0.0, 0.0)
    assert fb == pytest.approx(0.5543, abs=1e-4)
    assert bb == pytest.approx(0.3053, abs=1e-4)
    assert os == pytest.approx(0.1404, abs=1e-4)
    assert (fb + bb + os) == pytest.approx(1.0, abs=1e-4)


def test_build_rest_payload_structure_and_keys() -> None:
    """Verify build_rest_payload returns {'dataframe_records': [...]} with exact 24 keys per record."""
    record = derive_tactical_features(_sample_raw_inputs())
    payload = build_rest_payload([record])

    assert isinstance(payload, dict)
    assert "dataframe_records" in payload
    assert isinstance(payload["dataframe_records"], list)
    assert len(payload["dataframe_records"]) == 1
    assert list(payload["dataframe_records"][0].keys()) == EXPECTED_FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# 5. PitchServingClient Inference & Retry Resilience
# ---------------------------------------------------------------------------


def test_pitch_serving_client_predict_success() -> None:
    """Verify PitchServingClient.predict returns prediction, latency_ms, features, and status='success'."""
    client = PitchServingClient(
        endpoint_name="pitch-predictor-prod",
        host="https://adb-test.azuredatabricks.net",
        token="dapi_test_token",
    )
    raw = _sample_raw_inputs(balls=0, strikes=2)

    with patch.object(
        client, "_send_request", return_value={"predictions": ["Fastball"]}
    ) as mock_send:
        result = client.predict(raw)

    assert mock_send.call_count == 1
    assert result["status"] == "success"
    assert result["prediction"] == "Fastball"
    assert "latency_ms" in result
    assert isinstance(result["latency_ms"], (int, float))
    assert result["latency_ms"] >= 0
    assert "features" in result
    assert list(result["features"].keys()) == EXPECTED_FEATURE_COLUMNS
    assert result["features"]["count_state"] == "0-2"
    assert result["features"]["count_leverage"] == "Pitcher Ahead"


def test_pitch_serving_client_retries_on_cold_start_connection_error() -> None:
    """Verify PitchServingClient retries when the 1st attempt raises ConnectionError and succeeds on 2nd."""
    client = PitchServingClient(
        endpoint_name="pitch-predictor-prod",
        host="https://adb-test.azuredatabricks.net",
        token="dapi_test_token",
        max_retries=3,
        backoff_seconds=0.01,
    )
    raw = _sample_raw_inputs(balls=3, strikes=2)

    with patch.object(
        client,
        "_send_request",
        side_effect=[
            ConnectionError("Endpoint scale-to-zero cold-start in progress"),
            {"predictions": ["Breaking Ball"]},
        ],
    ) as mock_send:
        result = client.predict(raw)

    assert mock_send.call_count == 2
    assert result["status"] == "success"
    assert result["prediction"] == "Breaking Ball"
    assert result["features"]["count_state"] == "3-2"


def test_derive_and_build_multimodal_payload() -> None:
    """Verify derive_multimodal_features and build_multimodal_payload structure and fallbacks."""
    raw = _sample_raw_inputs(arm_slot_angle=52.5)
    multi_feats = derive_multimodal_features(raw)

    assert "arm_slot_angle" in multi_feats
    assert multi_feats["arm_slot_angle"] == 52.5
    assert "elbow_flexion_angle" in multi_feats
    assert multi_feats["elbow_flexion_angle"] == 135.0  # default fallback

    payload = build_multimodal_payload([multi_feats])
    assert "dataframe_records" in payload
    records = payload["dataframe_records"]
    assert len(records) == 1
    assert set(BIOMECHANICAL_FEATURE_COLUMNS).issubset(records[0].keys())
    assert set(EXPECTED_FEATURE_COLUMNS).issubset(records[0].keys())


def test_pitch_serving_client_predict_multimodal_structured_response() -> None:
    """Verify predict_multimodal returns probabilities, alignment status, and explanation."""
    client = PitchServingClient(
        endpoint_name="pitch-predictor-prod",
        host="https://adb-test.azuredatabricks.net",
        token="dapi_test_token",
    )
    raw = _sample_raw_inputs(balls=0, strikes=2, arm_slot_angle=20.0)

    mock_multimodal_item = {
        "prediction": "Breaking Ball",
        "prob_fastball": 0.15,
        "prob_breaking_ball": 0.75,
        "prob_offspeed": 0.10,
        "statcast_prediction": "Fastball",
        "vision_prediction": "Breaking Ball",
        "alignment_status": "TIPPING_OVERRIDE",
        "summary_text": "Pitch tipping detectado no release point.",
    }

    with patch.object(
        client, "_send_request", return_value={"predictions": [mock_multimodal_item]}
    ) as mock_send:
        result = client.predict_multimodal(raw)

    assert mock_send.call_count == 1
    assert result["status"] == "success"
    assert result["prediction"] == "Breaking Ball"
    assert result["probabilities"]["Breaking Ball"] == 0.75
    assert result["statcast_prediction"] == "Fastball"
    assert result["vision_prediction"] == "Breaking Ball"
    assert result["alignment_status"] == "TIPPING_OVERRIDE"
    assert "tipping" in result["summary_text"].lower()


def test_pitch_serving_client_predict_multimodal_string_response_compatibility() -> None:
    """Verify predict_multimodal backwards-compatibility if endpoint returns simple strings."""
    client = PitchServingClient(
        endpoint_name="pitch-predictor-prod",
        host="https://adb-test.azuredatabricks.net",
        token="dapi_test_token",
    )
    raw = _sample_raw_inputs(balls=1, strikes=0)

    with patch.object(
        client, "_send_request", return_value={"predictions": ["Fastball"]}
    ) as mock_send:
        result = client.predict_multimodal(raw)

    assert mock_send.call_count == 1
    assert result["status"] == "success"
    assert result["prediction"] == "Fastball"
    assert result["probabilities"]["Fastball"] > 0.5
    assert result["alignment_status"] == "AGREEMENT"


def test_pitch_serving_client_predict_multimodal_batch() -> None:
    """Verify predict_multimodal_batch across multiple game states."""
    client = PitchServingClient(
        endpoint_name="pitch-predictor-prod",
        host="https://adb-test.azuredatabricks.net",
        token="dapi_test_token",
    )
    inputs = [
        _sample_raw_inputs(balls=0, strikes=0),
        _sample_raw_inputs(balls=0, strikes=2),
    ]

    mock_resp = {
        "predictions": [
            {
                "prediction": "Fastball",
                "prob_fastball": 0.6,
                "prob_breaking_ball": 0.3,
                "prob_offspeed": 0.1,
                "statcast_prediction": "Fastball",
                "vision_prediction": "Fastball",
                "alignment_status": "AGREEMENT",
                "summary_text": "Alinhado.",
            },
            {
                "prediction": "Breaking Ball",
                "prob_fastball": 0.2,
                "prob_breaking_ball": 0.7,
                "prob_offspeed": 0.1,
                "statcast_prediction": "Breaking Ball",
                "vision_prediction": "Breaking Ball",
                "alignment_status": "AGREEMENT",
                "summary_text": "Alinhado.",
            },
        ]
    }

    with patch.object(client, "_send_request", return_value=mock_resp) as mock_send:
        result = client.predict_multimodal_batch(inputs)

    assert mock_send.call_count == 1
    assert result["status"] == "success"
    assert len(result["predictions"]) == 2
    assert result["predictions"][0]["prediction"] == "Fastball"
    assert result["predictions"][1]["prediction"] == "Breaking Ball"

