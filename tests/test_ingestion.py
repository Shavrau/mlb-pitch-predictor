"""TDD test suite for Statcast Bronze data ingestion, quality rules, EDA profiling, and @dp pipeline definition."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from pitch_predictor.ingestion import (
    BRONZE_COLUMNS,
    BRONZE_EXPECTATIONS,
    POST_RELEASE_LEAKAGE_COLS,
    PRE_PITCH_FEATURE_COLS,
    PRIMARY_KEY_COLS,
    clean_and_project_statcast,
    generate_bronze_eda_report,
    map_pitch_category,
)
from pitch_predictor.pipelines.bronze_pitches import register_bronze_pipeline


@pytest.fixture
def sample_raw_statcast_df() -> pd.DataFrame:
    """Synthetic raw Statcast DataFrame mimicking pybaseball.statcast() output with edge cases."""
    return pd.DataFrame(
        {
            "game_pk": [745001, 745001, 745001, 745001, 745002, 745003],
            "game_date": [
                "2024-04-01",
                "2024-04-01",
                "2024-04-01",
                "2024-04-01",  # Duplicate key row below
                "2024-04-01",
                "2024-03-15",  # Spring training game
            ],
            "game_type": ["R", "R", "R", "R", "R", "S"],
            "at_bat_number": [1, 1, 1, 1, 2, 1],
            "pitch_number": [1, 2, 3, 3, 1, 1],
            "pitcher": [605483, 605483, 605483, 605483, 592789, 605483],
            "batter": [660271, 660271, 660271, 660271, 665742, 660271],
            "pitch_type": ["FF", "SL", "CH", "CH", "PO", "FF"],
            "pitch_name": [
                "4-Seam Fastball",
                "Slider",
                "Changeup",
                "Changeup",
                "Pitch Out",
                "4-Seam Fastball",
            ],
            "balls": [0, 0, 1, 1, 0, 0],
            "strikes": [0, 1, 1, 1, 0, 0],
            "outs_when_up": [0, 0, 0, 0, 1, 0],
            "inning": [1, 1, 1, 1, 1, 1],
            "inning_topbot": ["Top", "Top", "Top", "Top", "Bot", "Top"],
            "on_1b": [None, None, 665742.0, 665742.0, None, None],
            "on_2b": [None, None, None, None, 660271.0, None],
            "on_3b": [None, None, None, None, None, None],
            "bat_score": [0, 0, 0, 0, 1, 0],
            "fld_score": [0, 0, 0, 0, 0, 0],
            "p_throws": ["R", "R", "R", "R", "L", "R"],
            "stand": ["L", "L", "L", "L", "R", "L"],
            "n_thruorder_pitcher": [1, 1, 1, 1, 1, 1],
            "type": ["S", "B", "X", "X", "B", "S"],
            "zone": [5.0, 14.0, 8.0, 8.0, None, 4.0],
            # Post-release columns that exist in raw Statcast
            "release_speed": [96.5, 85.2, 88.1, 88.1, 90.0, 95.0],
            "release_spin_rate": [2400, 2600, 1750, 1750, 2100, 2350],
            "pfx_x": [-0.5, 0.8, -1.1, -1.1, -0.2, -0.4],
            "pfx_z": [1.4, 0.1, 0.4, 0.4, 1.1, 1.3],
            "plate_x": [0.1, 1.2, -0.2, -0.2, 2.5, 0.0],
            "plate_z": [2.5, 1.1, 2.0, 2.0, 3.5, 2.6],
        }
    )


def test_constants_and_anti_leakage_contract():
    """Verify primary keys, pre-pitch features, and strict separation from post-release columns."""
    assert PRIMARY_KEY_COLS == ["game_pk", "at_bat_number", "pitch_number"]
    for pk in PRIMARY_KEY_COLS:
        assert pk in BRONZE_COLUMNS

    # No post-release physics column may ever appear in PRE_PITCH_FEATURE_COLS or BRONZE_COLUMNS
    overlap_features = set(PRE_PITCH_FEATURE_COLS).intersection(POST_RELEASE_LEAKAGE_COLS)
    assert not overlap_features, f"Leakage detected in PRE_PITCH_FEATURE_COLS: {overlap_features}"

    overlap_bronze = set(BRONZE_COLUMNS).intersection(POST_RELEASE_LEAKAGE_COLS)
    assert not overlap_bronze, f"Post-release cols should be dropped at Bronze projection: {overlap_bronze}"


@pytest.mark.parametrize(
    "pitch_type,expected_category",
    [
        ("FF", "Fastball"),
        ("SI", "Fastball"),
        ("FC", "Fastball"),
        ("SL", "Breaking Ball"),
        ("ST", "Breaking Ball"),
        ("CU", "Breaking Ball"),
        ("KC", "Breaking Ball"),
        ("SV", "Breaking Ball"),
        ("CS", "Breaking Ball"),
        ("CH", "Offspeed"),
        ("FS", "Offspeed"),
        ("FO", "Offspeed"),
        ("SC", "Offspeed"),
        ("PO", None),
        ("IN", None),
        ("KN", None),
        (None, None),
    ],
)
def test_map_pitch_category(pitch_type, expected_category):
    """Verify Statcast pitch_type codes map accurately to the 3 tactical categories."""
    assert map_pitch_category(pitch_type) == expected_category


def test_clean_and_project_statcast(sample_raw_statcast_df):
    """Verify regular season filtering, deduplication, base flags, score_diff, and column projection."""
    cleaned = clean_and_project_statcast(sample_raw_statcast_df, regular_season_only=True)

    # Spring training ('S') row and duplicate (745001, 1, 3) must be removed -> 4 unique regular-season rows
    assert len(cleaned) == 4
    assert (cleaned["game_type"] == "R").all()

    # Check primary key uniqueness
    assert not cleaned.duplicated(subset=PRIMARY_KEY_COLS).any()

    # Check base occupancy flags (structural NaN converted to 0/1)
    assert set(cleaned["on_1b_flag"].unique()).issubset({0, 1})
    assert set(cleaned["on_2b_flag"].unique()).issubset({0, 1})
    assert set(cleaned["on_3b_flag"].unique()).issubset({0, 1})
    assert cleaned.loc[cleaned["pitch_number"] == 3, "on_1b_flag"].iloc[0] == 1

    # Check score differential
    assert "score_diff" in cleaned.columns

    # Check post-release columns were stripped
    for col in POST_RELEASE_LEAKAGE_COLS:
        assert col not in cleaned.columns

    # Check pitch_category mapping
    categories = cleaned.set_index(["game_pk", "at_bat_number", "pitch_number"])["pitch_category"]
    assert categories.loc[(745001, 1, 1)] == "Fastball"
    assert categories.loc[(745001, 1, 2)] == "Breaking Ball"
    assert categories.loc[(745001, 1, 3)] == "Offspeed"
    assert pd.isna(categories.loc[(745002, 2, 1)])  # 'PO' Pitch Out -> None/NaN


def test_generate_bronze_eda_report(sample_raw_statcast_df):
    """Verify EDA report structure, counts, missingness, and class distributions."""
    cleaned = clean_and_project_statcast(sample_raw_statcast_df, regular_season_only=True)
    report = generate_bronze_eda_report(cleaned)

    assert report["total_rows"] == 4
    assert report["unique_games"] == 2
    assert report["unique_pitchers"] == 2
    assert report["duplicate_primary_keys"] == 0
    assert report["domain_violations"]["invalid_balls"] == 0
    assert report["domain_violations"]["invalid_strikes"] == 0
    assert report["domain_violations"]["invalid_outs"] == 0

    # 3 classified pitches (1 Fastball, 1 Breaking Ball, 1 Offspeed) + 1 unclassified ('PO')
    assert report["pitch_category_distribution"]["Fastball"]["count"] == 1
    assert report["pitch_category_distribution"]["Breaking Ball"]["count"] == 1
    assert report["pitch_category_distribution"]["Offspeed"]["count"] == 1
    assert report["unclassified_pitches_count"] == 1


def test_declarative_pipeline_dp_registration():
    """Verify Lakeflow Spark Declarative Pipeline (@dp) registration uses create_streaming_table and create_auto_cdc_flow."""
    mock_dp = MagicMock()
    mock_spark = MagicMock()

    # Make decorators pass functions through
    mock_dp.temporary_view.side_effect = lambda **kwargs: (lambda fn: fn)

    register_bronze_pipeline(dp_module=mock_dp, spark_session=mock_spark)

    # 1. Must register temporary view for staged Statcast stream
    mock_dp.temporary_view.assert_called_once()
    _, tv_kwargs = mock_dp.temporary_view.call_args
    assert tv_kwargs["name"] == "statcast_pitches_staged"

    # 2. Must create streaming table 'raw_pitches' with expectations and liquid clustering
    mock_dp.create_streaming_table.assert_called_once()
    _, st_kwargs = mock_dp.create_streaming_table.call_args
    assert st_kwargs["name"] == "raw_pitches"
    assert st_kwargs["cluster_by"] == ["game_date", "pitcher"]
    assert st_kwargs["expect_all_or_drop"] == BRONZE_EXPECTATIONS

    # 3. Must wire Auto CDC flow keyed on (game_pk, at_bat_number, pitch_number)
    mock_dp.create_auto_cdc_flow.assert_called_once()
    _, cdc_kwargs = mock_dp.create_auto_cdc_flow.call_args
    assert cdc_kwargs["target"] == "raw_pitches"
    assert cdc_kwargs["source"] == "statcast_pitches_staged"
    assert cdc_kwargs["keys"] == ["game_pk", "at_bat_number", "pitch_number"]
    assert cdc_kwargs["stored_as_scd_type"] == 1


def test_real_statcast_extraction_and_validation():
    """
    Integration test executing real MLB Statcast extraction via pybaseball
    and validating schema, primary key uniqueness, domain bounds, and EDA report.
    """
    from pitch_predictor.ingestion import fetch_statcast_pitches

    # Fetch 1 week of regular season MLB games (2024-04-01 to 2024-04-07)
    df = fetch_statcast_pitches(
        start_dt="2024-04-01",
        end_dt="2024-04-07",
        regular_season_only=True,
    )

    assert not df.empty, "Real Statcast extraction must return non-empty DataFrame"
    assert list(df.columns) == BRONZE_COLUMNS

    # Verify primary key integrity on real MLB data
    assert not df[PRIMARY_KEY_COLS].isna().any().any()
    assert not df.duplicated(subset=PRIMARY_KEY_COLS).any()

    # Verify domain boundaries on real MLB data
    assert df["balls"].between(0, 4).all()
    assert df["strikes"].between(0, 3).all()
    assert df["outs_when_up"].between(0, 2).all()
    assert set(df["p_throws"].dropna().unique()).issubset({"L", "R"})
    assert set(df["stand"].dropna().unique()).issubset({"L", "R"})

    # Verify EDA report on real data
    report = generate_bronze_eda_report(df)
    assert report["total_rows"] > 20000
    assert report["unique_games"] > 70
    assert set(report["pitch_category_distribution"].keys()) == {
        "Fastball",
        "Breaking Ball",
        "Offspeed",
    }


def test_resolve_ingestion_window_explicit_dates():
    """Verify explicit dates are preserved and validated."""
    from pitch_predictor.ingestion import resolve_ingestion_window

    s, e = resolve_ingestion_window("2024-05-01", "2024-05-07")
    assert s == "2024-05-01"
    assert e == "2024-05-07"

    # Single date
    s, e = resolve_ingestion_window("2024-05-01", "2024-05-01")
    assert s == "2024-05-01"
    assert e == "2024-05-01"

    # Inverted dates raise ValueError
    with pytest.raises(ValueError, match="cannot be after end_dt"):
        resolve_ingestion_window("2024-05-10", "2024-05-01")


def test_resolve_ingestion_window_scheduled_auto_continuation():
    """Verify scheduled auto mode picks up next day after latest ingested date up to max batch."""
    from pitch_predictor.ingestion import resolve_ingestion_window

    # Auto continuation after 2024-04-28 with 7-day batch
    s, e = resolve_ingestion_window(
        start_dt="auto",
        end_dt="auto",
        latest_ingested_date="2024-04-28",
        max_days_per_batch=7,
    )
    assert s == "2024-04-29"
    assert e == "2024-05-05"

    # Auto continuation bounded by reference date (e.g. yesterday)
    s, e = resolve_ingestion_window(
        start_dt="auto",
        end_dt="auto",
        latest_ingested_date="2024-04-28",
        max_days_per_batch=7,
        reference_date="2024-04-30",
    )
    assert s == "2024-04-29"
    assert e == "2024-04-30"


def test_resolve_ingestion_window_scheduled_auto_initial():
    """Verify scheduled auto mode with no prior data uses default opening day."""
    from pitch_predictor.ingestion import resolve_ingestion_window

    s, e = resolve_ingestion_window(
        start_dt="auto",
        end_dt="auto",
        latest_ingested_date=None,
        default_start="2024-03-28",
        max_days_per_batch=7,
    )
    assert s == "2024-03-28"
    assert e == "2024-04-03"


def test_generate_season_month_chunks():
    """Verify generation of chronological monthly intervals for MLB regular seasons."""
    from pitch_predictor.ingestion import generate_season_month_chunks

    chunks_2023 = generate_season_month_chunks(2023)
    assert len(chunks_2023) == 6
    assert chunks_2023[0] == ("2023-03-30", "2023-04-30")
    assert chunks_2023[-1] == ("2023-09-01", "2023-10-01")

    chunks_2022 = generate_season_month_chunks(2022)
    assert len(chunks_2022) == 6
    assert chunks_2022[0] == ("2022-04-07", "2022-04-30")
    assert chunks_2022[-1] == ("2022-09-01", "2022-10-05")

    chunks_2021 = generate_season_month_chunks(2021)
    assert len(chunks_2021) == 6
    assert chunks_2021[0] == ("2021-04-01", "2021-04-30")
    assert chunks_2021[-1] == ("2021-09-01", "2021-10-03")


def test_resolve_dual_track_ingestion():
    """Verify dual-track planning for current season forward and historical backward ingestion."""
    from pitch_predictor.ingestion import resolve_dual_track_ingestion

    # Case 1: Ingestion starting at 2024-04-01 -> gaps detected and 2023 planned
    plan1 = resolve_dual_track_ingestion(
        current_min_date="2024-04-01",
        current_max_date="2024-05-05",
        earliest_season_limit=2021,
    )
    assert plan1["gap_2024_window"] == ("2024-03-28", "2024-03-31")
    assert plan1["backward_season"] == 2023
    assert len(plan1["backward_chunks"]) == 6

    # Case 2: 2023 already ingested -> planning 2022 next
    plan2 = resolve_dual_track_ingestion(
        current_min_date="2023-03-30",
        current_max_date="2024-05-05",
        earliest_season_limit=2021,
    )
    assert plan1["gap_2024_window"] is not None
    assert plan2["gap_2024_window"] is None
    assert plan2["backward_season"] == 2022
    assert len(plan2["backward_chunks"]) == 6

    # Case 3: Target earliest season limit reached (e.g. 2021) -> stops backward ingestion
    plan3 = resolve_dual_track_ingestion(
        current_min_date="2021-04-01",
        current_max_date="2024-05-05",
        earliest_season_limit=2021,
    )
    assert plan3["backward_season"] is None
    assert plan3["backward_chunks"] == []



