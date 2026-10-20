"""Statcast pitch data acquisition, cleaning, projection, and Bronze EDA profiling."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pandas as pd

from pitch_predictor.utils import get_logger

logger = get_logger(__name__)

# Composite primary key for pitch-level deduplication and Phase 2 Computer Vision join
PRIMARY_KEY_COLS: list[str] = [
    "game_pk",
    "at_bat_number",
    "pitch_number",
]

# Pre-pitch situational and matchup columns (Zero post-release leakage)
PRE_PITCH_FEATURE_COLS: list[str] = [
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "inning_topbot",
    "on_1b_flag",
    "on_2b_flag",
    "on_3b_flag",
    "bat_score",
    "fld_score",
    "score_diff",
    "p_throws",
    "stand",
    "n_thruorder_pitcher",
]

# Post-release tracking columns explicitly excluded from Bronze/Silver pre-pitch features
POST_RELEASE_LEAKAGE_COLS: list[str] = [
    "release_speed",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "release_spin_rate",
    "release_extension",
    "spin_axis",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "effective_speed",
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
]

# Raw columns extracted from Statcast before derived Bronze flags
RAW_STATCAST_SELECTED_COLS: list[str] = [
    "game_pk",
    "game_date",
    "game_type",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "batter",
    "pitch_type",
    "pitch_name",
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "inning_topbot",
    "on_1b",
    "on_2b",
    "on_3b",
    "bat_score",
    "fld_score",
    "p_throws",
    "stand",
    "n_thruorder_pitcher",
    "type",
    "zone",
]

# Final projected columns stored in Bronze table (`raw_pitches`)
BRONZE_COLUMNS: list[str] = [
    "game_pk",
    "game_date",
    "game_type",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "batter",
    "pitch_type",
    "pitch_name",
    "pitch_category",
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "inning_topbot",
    "on_1b_flag",
    "on_2b_flag",
    "on_3b_flag",
    "bat_score",
    "fld_score",
    "score_diff",
    "p_throws",
    "stand",
    "n_thruorder_pitcher",
    "type",
    "zone",
]

# Mapping from Statcast pitch_type to 3 tactical categories
PITCH_CATEGORY_MAP: dict[str, str] = {
    # Fastballs
    "FF": "Fastball",
    "SI": "Fastball",
    "FC": "Fastball",
    "FA": "Fastball",
    # Breaking Balls
    "SL": "Breaking Ball",
    "ST": "Breaking Ball",
    "CU": "Breaking Ball",
    "KC": "Breaking Ball",
    "SV": "Breaking Ball",
    "CS": "Breaking Ball",
    # Offspeed
    "CH": "Offspeed",
    "FS": "Offspeed",
    "FO": "Offspeed",
    "SC": "Offspeed",
}

# Lakeflow Spark Declarative Pipeline (@dp) Data Quality Expectations
BRONZE_EXPECTATIONS: dict[str, str] = {
    "valid_primary_key": (
        "game_pk IS NOT NULL AND at_bat_number IS NOT NULL AND pitch_number IS NOT NULL"
    ),
    "valid_balls": "balls BETWEEN 0 AND 3",
    "valid_strikes": "strikes BETWEEN 0 AND 2",
    "valid_outs": "outs_when_up BETWEEN 0 AND 2",
    "valid_handedness": "p_throws IN ('L', 'R') AND stand IN ('L', 'R')",
}


def map_pitch_category(pitch_type: str | None) -> str | None:
    """Map raw Statcast pitch_type code to Fastball, Breaking Ball, Offspeed, or None."""
    if pitch_type is None or pd.isna(pitch_type):
        return None
    return PITCH_CATEGORY_MAP.get(str(pitch_type).strip().upper())


def clean_and_project_statcast(
    raw_df: pd.DataFrame,
    regular_season_only: bool = True,
) -> pd.DataFrame:
    """
    Filter, deduplicate, and project raw Statcast DataFrame into the Bronze schema.

    - Filters to regular season (`game_type == 'R'`) when enabled.
    - Converts structural base runner IDs (`on_1b`, `on_2b`, `on_3b`) into 0/1 flags.
    - Computes `score_diff = fld_score - bat_score`.
    - Maps `pitch_type` -> `pitch_category`.
    - Enforces composite key uniqueness on `(game_pk, at_bat_number, pitch_number)`.
    - Drops all post-release physical tracking columns to prevent leakage and reduce volume.
    """
    if raw_df.empty:
        return pd.DataFrame(columns=BRONZE_COLUMNS)

    df = raw_df.copy()

    # 1. Filter regular season games if requested
    if regular_season_only and "game_type" in df.columns:
        df = df[df["game_type"] == "R"].copy()

    # 2. Ensure all expected raw columns exist
    for col in RAW_STATCAST_SELECTED_COLS:
        if col not in df.columns:
            df[col] = None

    # 3. Structural missingness -> binary base occupancy flags (0 = empty, 1 = runner on base)
    df["on_1b_flag"] = df["on_1b"].notna().astype("int64")
    df["on_2b_flag"] = df["on_2b"].notna().astype("int64")
    df["on_3b_flag"] = df["on_3b"].notna().astype("int64")

    # 4. Score differential from pitcher's perspective
    df["bat_score"] = pd.to_numeric(df["bat_score"], errors="coerce").fillna(0).astype("int64")
    df["fld_score"] = pd.to_numeric(df["fld_score"], errors="coerce").fillna(0).astype("int64")
    df["score_diff"] = (df["fld_score"] - df["bat_score"]).astype("int64")

    # 5. Map tactical pitch_category
    df["pitch_category"] = df["pitch_type"].apply(map_pitch_category)

    # 6. Normalize date and core integer types
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    for int_col in [
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "pitcher",
        "batter",
        "balls",
        "strikes",
        "outs_when_up",
        "inning",
    ]:
        df[int_col] = pd.to_numeric(df[int_col], errors="coerce")

    df = df.dropna(subset=PRIMARY_KEY_COLS).copy()
    for pk_col in PRIMARY_KEY_COLS:
        df[pk_col] = df[pk_col].astype("int64")

    # 7. Deduplicate on composite primary key and sort chronologically
    df = df.drop_duplicates(subset=PRIMARY_KEY_COLS, keep="last")
    df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(
        drop=True
    )

    return df[BRONZE_COLUMNS]


def resolve_ingestion_window(
    start_dt: str | None = None,
    end_dt: str | None = None,
    latest_ingested_date: str | None = None,
    max_days_per_batch: int = 7,
    reference_date: str | None = None,
    default_start: str = "2024-03-28",
) -> tuple[str, str]:
    """
    Resolve the ingestion date window for scheduled or ad-hoc runs.

    Rules:
    1. If explicit valid dates (YYYY-MM-DD) are provided and neither is 'auto'/'latest', use them.
    2. If start_dt is 'auto' (or None) and latest_ingested_date is provided:
       - Next start date is latest_ingested_date + 1 day.
       - End date is min(reference_date, next_start + max_days_per_batch - 1) if reference_date is provided.
    3. If latest_ingested_date is None (initial ingestion):
       - Default start date is Opening Day of 2024 regular season ("2024-03-28").
       - End date is start date + max_days_per_batch - 1.
    4. Guarantees end_dt >= start_dt and formats as 'YYYY-MM-DD'.
    """
    is_auto_start = start_dt is None or str(start_dt).strip().lower() in {"auto", "latest", ""}
    is_auto_end = end_dt is None or str(end_dt).strip().lower() in {"auto", "latest", ""}

    if not is_auto_start and not is_auto_end:
        s_date = pd.to_datetime(start_dt).strftime("%Y-%m-%d")
        e_date = pd.to_datetime(end_dt).strftime("%Y-%m-%d")
        if s_date > e_date:
            raise ValueError(f"start_dt ({s_date}) cannot be after end_dt ({e_date})")
        return s_date, e_date

    # Determine start date
    if latest_ingested_date is not None and str(latest_ingested_date).strip():
        last_dt = pd.to_datetime(latest_ingested_date).date()
        s_dt = last_dt + timedelta(days=1)
    else:
        s_dt = pd.to_datetime(default_start).date()

    # Determine end date
    max_batch_end = s_dt + timedelta(days=max(0, max_days_per_batch - 1))
    if reference_date is not None and str(reference_date).strip():
        ref_dt = pd.to_datetime(reference_date).date()
        e_dt = min(max_batch_end, max(s_dt, ref_dt))
    else:
        e_dt = max_batch_end

    return s_dt.strftime("%Y-%m-%d"), e_dt.strftime("%Y-%m-%d")


def generate_season_month_chunks(season: int) -> list[tuple[str, str]]:
    """
    Generate chronological monthly intervals for an MLB regular season.
    Specific official Opening Day and final regular season dates are mapped for recent seasons.
    """
    if season == 2023:
        return [
            ("2023-03-30", "2023-04-30"),
            ("2023-05-01", "2023-05-31"),
            ("2023-06-01", "2023-06-30"),
            ("2023-07-01", "2023-07-31"),
            ("2023-08-01", "2023-08-31"),
            ("2023-09-01", "2023-10-01"),
        ]
    if season == 2022:
        return [
            ("2022-04-07", "2022-04-30"),
            ("2022-05-01", "2022-05-31"),
            ("2022-06-01", "2022-06-30"),
            ("2022-07-01", "2022-07-31"),
            ("2022-08-01", "2022-08-31"),
            ("2022-09-01", "2022-10-05"),
        ]
    if season == 2021:
        return [
            ("2021-04-01", "2021-04-30"),
            ("2021-05-01", "2021-05-31"),
            ("2021-06-01", "2021-06-30"),
            ("2021-07-01", "2021-07-31"),
            ("2021-08-01", "2021-08-31"),
            ("2021-09-01", "2021-10-03"),
        ]
    return [
        (f"{season}-03-28", f"{season}-04-30"),
        (f"{season}-05-01", f"{season}-05-31"),
        (f"{season}-06-01", f"{season}-06-30"),
        (f"{season}-07-01", f"{season}-07-31"),
        (f"{season}-08-01", f"{season}-08-31"),
        (f"{season}-09-01", f"{season}-10-02"),
    ]


def resolve_dual_track_ingestion(
    current_min_date: str | None = None,
    current_max_date: str | None = None,
    earliest_season_limit: int = 2021,
    reference_date: str | None = None,
    max_days_forward_batch: int = 7,
) -> dict[str, Any]:
    """
    Plan dual-track ingestion for both:
    1. Forward track: Continuing current season from max_game_date up to reference_date (yesterday).
    2. Backward track: Backfilling missing historical data (e.g. 2024 opening gap, and 1 full historical
       season per execution until earliest_season_limit is satisfied).
    """
    plan: dict[str, Any] = {
        "forward_window": None,
        "gap_2024_window": None,
        "backward_season": None,
        "backward_chunks": [],
    }

    # Forward window
    if current_max_date is not None and str(current_max_date).strip():
        s_fwd, e_fwd = resolve_ingestion_window(
            start_dt="auto",
            end_dt="auto",
            latest_ingested_date=current_max_date,
            max_days_per_batch=max_days_forward_batch,
            reference_date=reference_date,
        )
        plan["forward_window"] = (s_fwd, e_fwd)
    else:
        # Initial run: no records in raw_pitches yet -> ingest opening week batch
        s_fwd, e_fwd = resolve_ingestion_window(
            start_dt="auto",
            end_dt="auto",
            latest_ingested_date=None,
            max_days_per_batch=max_days_forward_batch,
            reference_date=reference_date,
        )
        plan["forward_window"] = (s_fwd, e_fwd)

    # 2024 Opening Day gap check
    if (
        current_min_date is not None
        and str(current_min_date) > "2024-03-28"
        and str(current_min_date).startswith("2024")
    ):
        plan["gap_2024_window"] = ("2024-03-28", "2024-03-31")

    # Backward season check
    if current_min_date is not None and str(current_min_date).strip():
        min_season = int(pd.to_datetime(current_min_date).strftime("%Y"))
        if min_season > earliest_season_limit:
            target_season = min_season - 1
            plan["backward_season"] = target_season
            plan["backward_chunks"] = generate_season_month_chunks(target_season)

    return plan


def fetch_statcast_pitches(
    start_dt: str,
    end_dt: str | None = None,
    regular_season_only: bool = True,
) -> pd.DataFrame:
    """
    Fetch real pitch-by-pitch data from MLB Statcast via `pybaseball` and project to Bronze schema.
    """
    try:
        from pybaseball import statcast
    except ImportError as exc:
        raise ImportError(
            "pybaseball is required for Statcast extraction. Install with `pip install pybaseball`."
        ) from exc

    end_dt = end_dt or start_dt
    logger.info("Fetching MLB Statcast pitches from %s to %s...", start_dt, end_dt)
    raw_df = statcast(start_dt=start_dt, end_dt=end_dt, verbose=False)
    if raw_df is None or raw_df.empty:
        logger.warning("No Statcast records returned for window %s to %s.", start_dt, end_dt)
        return pd.DataFrame(columns=BRONZE_COLUMNS)

    cleaned_df = clean_and_project_statcast(raw_df, regular_season_only=regular_season_only)
    logger.info(
        "Fetched %d raw rows -> %d cleaned Bronze rows across %d games.",
        len(raw_df),
        len(cleaned_df),
        cleaned_df["game_pk"].nunique(),
    )
    return cleaned_df


def generate_bronze_eda_report(df: pd.DataFrame) -> dict[str, Any]:
    """
    Generate a deterministic exploratory data analysis (EDA) summary for Bronze Statcast data.
    """
    total_rows = len(df)
    if total_rows == 0:
        return {
            "total_rows": 0,
            "unique_games": 0,
            "unique_pitchers": 0,
            "unique_batters": 0,
            "date_min": None,
            "date_max": None,
            "duplicate_primary_keys": 0,
            "missingness": {},
            "pitch_type_distribution": {},
            "pitch_category_distribution": {},
            "unclassified_pitches_count": 0,
            "domain_violations": {
                "invalid_balls": 0,
                "invalid_strikes": 0,
                "invalid_outs": 0,
            },
        }

    missingness = {
        col: {
            "null_count": int(df[col].isna().sum()),
            "null_pct": round(float(df[col].isna().mean() * 100.0), 2),
        }
        for col in df.columns
    }

    pitch_type_counts = df["pitch_type"].value_counts(dropna=False)
    pitch_type_dist = {
        str(k): {
            "count": int(v),
            "pct": round(float(v / total_rows * 100.0), 2),
        }
        for k, v in pitch_type_counts.items()
    }

    classified_df = df[df["pitch_category"].notna()]
    classified_total = max(len(classified_df), 1)
    cat_counts = classified_df["pitch_category"].value_counts()
    pitch_cat_dist = {
        str(k): {
            "count": int(v),
            "pct_of_total": round(float(v / total_rows * 100.0), 2),
            "pct_of_classified": round(float(v / classified_total * 100.0), 2),
        }
        for k, v in cat_counts.items()
    }

    invalid_balls = int((~df["balls"].between(0, 3)).sum())
    invalid_strikes = int((~df["strikes"].between(0, 2)).sum())
    invalid_outs = int((~df["outs_when_up"].between(0, 2)).sum())

    return {
        "total_rows": total_rows,
        "unique_games": int(df["game_pk"].nunique()),
        "unique_pitchers": int(df["pitcher"].nunique()),
        "unique_batters": int(df["batter"].nunique()),
        "date_min": str(df["game_date"].min()),
        "date_max": str(df["game_date"].max()),
        "duplicate_primary_keys": int(df.duplicated(subset=PRIMARY_KEY_COLS).sum()),
        "missingness": missingness,
        "pitch_type_distribution": pitch_type_dist,
        "pitch_category_distribution": pitch_cat_dist,
        "unclassified_pitches_count": int(df["pitch_category"].isna().sum()),
        "domain_violations": {
            "invalid_balls": invalid_balls,
            "invalid_strikes": invalid_strikes,
            "invalid_outs": invalid_outs,
        },
    }
