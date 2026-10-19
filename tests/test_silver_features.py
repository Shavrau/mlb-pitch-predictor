"""
Unit and integration tests for Silver Feature Engineering (TDD).

Tests:
1. Sequence features generation (LAG on pitch_category, pitch_type, zone, type/result).
2. Cross at-bat boundary isolation (pitch_number == 1 never leaks from previous at-bat).
3. Platoon and count state interactions (same_hand, count_state, count_leverage).
4. Base runner situational aggregations (runners_on_base, scoring_position).
5. Pitcher historical priors calculation with strict zero future leakage.
6. Target filtering (drops unclassified / non-tactical pitches).
"""

import pandas as pd
import pytest


def create_sample_bronze_df() -> pd.DataFrame:
    """Create a minimal synthetic Bronze DataFrame with 2 at-bats and 2 games."""
    data = [
        # Game 1, At-Bat 1 (Pitcher 100 vs Batter 200, R vs L)
        {
            "game_pk": 1001,
            "game_date": "2024-04-01",
            "game_type": "R",
            "at_bat_number": 1,
            "pitch_number": 1,
            "pitcher": 100,
            "batter": 200,
            "pitch_type": "FF",
            "pitch_name": "4-Seam Fastball",
            "pitch_category": "Fastball",
            "balls": 0,
            "strikes": 0,
            "outs_when_up": 0,
            "inning": 1,
            "inning_topbot": "Top",
            "on_1b_flag": 0,
            "on_2b_flag": 0,
            "on_3b_flag": 0,
            "bat_score": 0,
            "fld_score": 0,
            "score_diff": 0,
            "p_throws": "R",
            "stand": "L",
            "n_thruorder_pitcher": 1,
            "type": "B",
            "zone": 14,
        },
        {
            "game_pk": 1001,
            "game_date": "2024-04-01",
            "game_type": "R",
            "at_bat_number": 1,
            "pitch_number": 2,
            "pitcher": 100,
            "batter": 200,
            "pitch_type": "SL",
            "pitch_name": "Slider",
            "pitch_category": "Breaking Ball",
            "balls": 1,
            "strikes": 0,
            "outs_when_up": 0,
            "inning": 1,
            "inning_topbot": "Top",
            "on_1b_flag": 0,
            "on_2b_flag": 0,
            "on_3b_flag": 0,
            "bat_score": 0,
            "fld_score": 0,
            "score_diff": 0,
            "p_throws": "R",
            "stand": "L",
            "n_thruorder_pitcher": 1,
            "type": "S",
            "zone": 5,
        },
        {
            "game_pk": 1001,
            "game_date": "2024-04-01",
            "game_type": "R",
            "at_bat_number": 1,
            "pitch_number": 3,
            "pitcher": 100,
            "batter": 200,
            "pitch_type": "CH",
            "pitch_name": "Changeup",
            "pitch_category": "Offspeed",
            "balls": 1,
            "strikes": 1,
            "outs_when_up": 0,
            "inning": 1,
            "inning_topbot": "Top",
            "on_1b_flag": 0,
            "on_2b_flag": 0,
            "on_3b_flag": 0,
            "bat_score": 0,
            "fld_score": 0,
            "score_diff": 0,
            "p_throws": "R",
            "stand": "L",
            "n_thruorder_pitcher": 1,
            "type": "S",
            "zone": 13,
        },
        # Game 1, At-Bat 2 (Pitcher 100 vs Batter 300, R vs R, runners on 2nd and 3rd)
        {
            "game_pk": 1001,
            "game_date": "2024-04-01",
            "game_type": "R",
            "at_bat_number": 2,
            "pitch_number": 1,
            "pitcher": 100,
            "batter": 300,
            "pitch_type": "FF",
            "pitch_name": "4-Seam Fastball",
            "pitch_category": "Fastball",
            "balls": 0,
            "strikes": 0,
            "outs_when_up": 1,
            "inning": 1,
            "inning_topbot": "Top",
            "on_1b_flag": 0,
            "on_2b_flag": 1,
            "on_3b_flag": 1,
            "bat_score": 0,
            "fld_score": 0,
            "score_diff": 0,
            "p_throws": "R",
            "stand": "R",
            "n_thruorder_pitcher": 1,
            "type": "S",
            "zone": 2,
        },
        # Game 2, At-Bat 1 (Pitcher 100 vs Batter 400 on next day: 2024-04-02)
        {
            "game_pk": 1002,
            "game_date": "2024-04-02",
            "game_type": "R",
            "at_bat_number": 1,
            "pitch_number": 1,
            "pitcher": 100,
            "batter": 400,
            "pitch_type": "SL",
            "pitch_name": "Slider",
            "pitch_category": "Breaking Ball",
            "balls": 0,
            "strikes": 0,
            "outs_when_up": 0,
            "inning": 1,
            "inning_topbot": "Top",
            "on_1b_flag": 1,
            "on_2b_flag": 0,
            "on_3b_flag": 0,
            "bat_score": 2,
            "fld_score": 1,
            "score_diff": -1,
            "p_throws": "R",
            "stand": "R",
            "n_thruorder_pitcher": 1,
            "type": "B",
            "zone": 14,
        },
        # Game 2, Row with Unclassified / Non-tactical Pitch (e.g. Knuckleball KN)
        {
            "game_pk": 1002,
            "game_date": "2024-04-02",
            "game_type": "R",
            "at_bat_number": 1,
            "pitch_number": 2,
            "pitcher": 100,
            "batter": 400,
            "pitch_type": "KN",
            "pitch_name": "Knuckleball",
            "pitch_category": None,
            "balls": 1,
            "strikes": 0,
            "outs_when_up": 0,
            "inning": 1,
            "inning_topbot": "Top",
            "on_1b_flag": 1,
            "on_2b_flag": 0,
            "on_3b_flag": 0,
            "bat_score": 2,
            "fld_score": 1,
            "score_diff": -1,
            "p_throws": "R",
            "stand": "R",
            "n_thruorder_pitcher": 1,
            "type": "B",
            "zone": 14,
        },
    ]
    return pd.DataFrame(data)


def test_pitch_sequence_features_and_no_cross_at_bat_leakage():
    """Verify intra-at-bat LAG features and boundary reset across at-bats."""
    from pitch_predictor.features import derive_sequence_features

    df = create_sample_bronze_df()
    res = derive_sequence_features(df)

    # Pitch 1 of At-Bat 1 must be 'First Pitch' / 0
    ab1_p1 = res[(res["at_bat_number"] == 1) & (res["pitch_number"] == 1) & (res["game_pk"] == 1001)].iloc[0]
    assert ab1_p1["prev_pitch_category"] == "First Pitch"
    assert ab1_p1["prev_pitch_type"] == "None"
    assert ab1_p1["prev_pitch_zone"] == 0
    assert ab1_p1["prev_pitch_result"] == "First Pitch"

    # Pitch 2 of At-Bat 1 must inherit from Pitch 1 (FF, Fastball, zone 14, B)
    ab1_p2 = res[(res["at_bat_number"] == 1) & (res["pitch_number"] == 2) & (res["game_pk"] == 1001)].iloc[0]
    assert ab1_p2["prev_pitch_category"] == "Fastball"
    assert ab1_p2["prev_pitch_type"] == "FF"
    assert ab1_p2["prev_pitch_zone"] == 14
    assert ab1_p2["prev_pitch_result"] == "B"

    # Pitch 3 of At-Bat 1 must inherit from Pitch 2 (SL, Breaking Ball, zone 5, S)
    ab1_p3 = res[(res["at_bat_number"] == 1) & (res["pitch_number"] == 3) & (res["game_pk"] == 1001)].iloc[0]
    assert ab1_p3["prev_pitch_category"] == "Breaking Ball"
    assert ab1_p3["prev_pitch_type"] == "SL"
    assert ab1_p3["prev_pitch_zone"] == 5
    assert ab1_p3["prev_pitch_result"] == "S"

    # CRITICAL: Pitch 1 of At-Bat 2 must NOT inherit Pitch 3 of At-Bat 1!
    ab2_p1 = res[(res["at_bat_number"] == 2) & (res["pitch_number"] == 1) & (res["game_pk"] == 1001)].iloc[0]
    assert ab2_p1["prev_pitch_category"] == "First Pitch"
    assert ab2_p1["prev_pitch_type"] == "None"
    assert ab2_p1["prev_pitch_zone"] == 0
    assert ab2_p1["prev_pitch_result"] == "First Pitch"


def test_situational_platoon_and_count_features():
    """Verify derivation of same_hand, count_state, count_leverage, and base runner flags."""
    from pitch_predictor.features import derive_situational_features

    df = create_sample_bronze_df()
    res = derive_situational_features(df)

    # R vs L -> same_hand == 0
    row_rvl = res[(res["p_throws"] == "R") & (res["stand"] == "L")].iloc[0]
    assert row_rvl["same_hand"] == 0

    # R vs R -> same_hand == 1
    row_rvr = res[(res["p_throws"] == "R") & (res["stand"] == "R")].iloc[0]
    assert row_rvr["same_hand"] == 1

    # Count state & leverage checks:
    # 0 balls, 0 strikes -> '0-0', 'Neutral'
    row_00 = res[(res["balls"] == 0) & (res["strikes"] == 0)].iloc[0]
    assert row_00["count_state"] == "0-0"
    assert row_00["count_leverage"] == "Neutral"

    # 1 ball, 0 strikes -> '1-0', 'Pitcher Behind'
    row_10 = res[(res["balls"] == 1) & (res["strikes"] == 0)].iloc[0]
    assert row_10["count_state"] == "1-0"
    assert row_10["count_leverage"] == "Pitcher Behind"

    # Base runners & scoring position
    # At-bat 1 has empty bases -> runners_on_base == 0, scoring_position == 0
    row_empty = res[res["at_bat_number"] == 1].iloc[0]
    assert row_empty["runners_on_base"] == 0
    assert row_empty["scoring_position"] == 0

    # At-bat 2 has runners on 2nd and 3rd -> runners_on_base == 2, scoring_position == 1
    row_risp = res[res["at_bat_number"] == 2].iloc[0]
    assert row_risp["runners_on_base"] == 2
    assert row_risp["scoring_position"] == 1


def test_pitcher_historical_priors_zero_leakage():
    """Verify that pitcher historical priors only use strictly preceding pitches."""
    from pitch_predictor.features import derive_pitcher_historical_priors

    df = create_sample_bronze_df()
    # Sort chronologically by game_date, game_pk, at_bat_number, pitch_number
    df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)

    res = derive_pitcher_historical_priors(df)

    # First pitch ever observed for pitcher 100 has default league priors
    p1 = res[(res["pitcher"] == 100) & (res["game_pk"] == 1001) & (res["at_bat_number"] == 1) & (res["pitch_number"] == 1)].iloc[0]
    assert "pitcher_fb_prior" in p1
    assert "pitcher_bb_prior" in p1
    assert "pitcher_os_prior" in p1
    # Check probabilities sum to ~1.0
    total_prior = p1["pitcher_fb_prior"] + p1["pitcher_bb_prior"] + p1["pitcher_os_prior"]
    assert pytest.approx(total_prior, 0.01) == 1.0

    # Check Game 2 pitch 1: Pitcher 100 has already thrown pitches in Game 1 (FF, SL, CH, FF)
    # The prior in Game 2 should incorporate the Game 1 pitch counts without leaking Game 2 itself!
    g2_p1 = res[(res["game_pk"] == 1002) & (res["at_bat_number"] == 1) & (res["pitch_number"] == 1)].iloc[0]
    assert g2_p1["pitcher_fb_prior"] > 0.3  # Pitched Fastballs in game 1


def test_build_silver_features_end_to_end_and_target_filtering():
    """Verify end-to-end Silver pipeline and dropping of null/unclassified pitch categories."""
    from pitch_predictor.features import SILVER_FEATURE_COLUMNS, build_silver_features

    df = create_sample_bronze_df()
    silver_df = build_silver_features(df)

    # The raw sample had 6 rows, including 1 with pitch_type='KN' and pitch_category=None
    # The Silver dataset must drop rows where pitch_category is null
    assert len(silver_df) == 5
    assert silver_df["pitch_category"].isnull().sum() == 0

    # Verify all expected columns exist
    for col in SILVER_FEATURE_COLUMNS:
        assert col in silver_df.columns, f"Missing required Silver column: {col}"

    # Verify no nulls in critical feature columns
    assert silver_df["prev_pitch_category"].isnull().sum() == 0
    assert silver_df["same_hand"].isnull().sum() == 0
    assert silver_df["count_state"].isnull().sum() == 0
    assert silver_df["scoring_position"].isnull().sum() == 0


def test_declarative_silver_pipeline_registration():
    """Verify Lakeflow Spark Declarative Pipeline registration for the Silver table."""
    from unittest.mock import MagicMock

    from pitch_predictor.pipelines.bronze_pitches import register_silver_pipeline

    mock_dp = MagicMock()
    mock_dp.table.side_effect = lambda **kwargs: (lambda fn: fn)

    register_silver_pipeline(mock_dp)

    mock_dp.table.assert_called_once()
    _, tbl_kwargs = mock_dp.table.call_args
    assert tbl_kwargs["name"] == "silver_pitch_features"
    assert tbl_kwargs["cluster_by"] == ["game_date", "pitcher"]

