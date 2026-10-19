"""
Silver Feature Engineering module for pre-pitch prediction.

Implements feature transformations for the Silver layer (silver_pitch_features):
1. Intra-at-bat pitch sequence dynamics (LAG features on category, type, zone, result).
2. Platoon and matchup interactions (same_hand, count_state, count_leverage).
3. Baserunner situational leverage (runners_on_base, scoring_position).
4. Strict zero-leakage historical pitcher priors (smoothed cumulative proportions).
5. Clean projection and target filtering for ML/AutoML.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# Global League Priors from EDA Baseline
LEAGUE_PRIOR_FB = 0.5543
LEAGUE_PRIOR_BB = 0.3053
LEAGUE_PRIOR_OS = 0.1404

# Mapping for Count Leverage
COUNT_LEVERAGE_MAP = {
    "0-1": "Pitcher Ahead",
    "0-2": "Pitcher Ahead",
    "1-2": "Pitcher Ahead",
    "1-0": "Pitcher Behind",
    "2-0": "Pitcher Behind",
    "3-0": "Pitcher Behind",
    "3-1": "Pitcher Behind",
    "0-0": "Neutral",
    "1-1": "Neutral",
    "2-1": "Neutral",
    "2-2": "Neutral",
    "3-2": "Neutral",
}

SILVER_FEATURE_COLUMNS: list[str] = [
    # Primary Key & CV Join Keys
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "batter",
    # Temporal Anchor
    "game_date",
    # Tactical ML Target
    "pitch_category",
    # Intra-At-Bat Sequence Features
    "prev_pitch_category",
    "prev_pitch_type",
    "prev_pitch_zone",
    "prev_pitch_result",
    # Platoon Handedness
    "p_throws",
    "stand",
    "same_hand",
    # Count State & Leverage
    "balls",
    "strikes",
    "count_state",
    "count_leverage",
    # Situational Context
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
    # Pitcher Historical Priors (Zero Leakage)
    "pitcher_fb_prior",
    "pitcher_bb_prior",
    "pitcher_os_prior",
]


def derive_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive intra-at-bat pitch sequence features using LAG.
    Guarantees strict boundary isolation: pitch_number == 1 receives default values
    and never inherits from a previous at-bat.
    """
    data = df.copy()
    # Ensure correct sequence ordering
    sort_cols = ["game_pk", "at_bat_number", "pitch_number"]
    data = data.sort_values(sort_cols).reset_index(drop=True)

    group = data.groupby(["game_pk", "at_bat_number"])

    data["prev_pitch_category"] = group["pitch_category"].shift(1).fillna("First Pitch")
    data["prev_pitch_type"] = group["pitch_type"].shift(1).fillna("None")

    # Zone is integer: fill NA with 0
    prev_zone = group["zone"].shift(1).fillna(0)
    data["prev_pitch_zone"] = prev_zone.astype(int)

    # Result / type: fill NA with 'First Pitch'
    data["prev_pitch_result"] = group["type"].shift(1).fillna("First Pitch")

    return data


def derive_situational_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive platoon, count state, leverage, and baserunner features.
    """
    data = df.copy()

    # Handedness interaction
    data["same_hand"] = (data["p_throws"] == data["stand"]).astype(int)

    # Count state (e.g. '0-0', '3-2')
    data["count_state"] = data["balls"].astype(str) + "-" + data["strikes"].astype(str)

    # Count leverage
    data["count_leverage"] = data["count_state"].map(COUNT_LEVERAGE_MAP).fillna("Neutral")

    # Baserunners summary
    data["runners_on_base"] = data["on_1b_flag"] + data["on_2b_flag"] + data["on_3b_flag"]
    data["scoring_position"] = ((data["on_2b_flag"] == 1) | (data["on_3b_flag"] == 1)).astype(int)

    return data


def derive_pitcher_historical_priors(df: pd.DataFrame, k_smooth: float = 15.0) -> pd.DataFrame:
    """
    Calculate strictly historical, zero-leakage pitch category priors for each pitcher.
    Uses smoothed cumulative counts strictly prior to the current pitch.
    """
    data = df.copy()
    sort_cols = ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    data = data.sort_values(sort_cols).reset_index(drop=True)

    # Indicator columns for valid pitch categories
    is_fb = (data["pitch_category"] == "Fastball").astype(float)
    is_bb = (data["pitch_category"] == "Breaking Ball").astype(float)
    is_os = (data["pitch_category"] == "Offspeed").astype(float)

    temp_indicators = pd.DataFrame(
        {"pitcher": data["pitcher"], "is_fb": is_fb, "is_bb": is_bb, "is_os": is_os}
    )

    # Groupby pitcher and calculate cumulative sums of strictly PRECEDING pitches
    shifted = temp_indicators.groupby("pitcher")[["is_fb", "is_bb", "is_os"]].shift(1).fillna(0.0)
    cum_counts = shifted.groupby(temp_indicators["pitcher"]).cumsum()

    cum_total = cum_counts["is_fb"] + cum_counts["is_bb"] + cum_counts["is_os"]

    # Empirical Bayes / Laplace smoothing with global league priors
    data["pitcher_fb_prior"] = (cum_counts["is_fb"] + k_smooth * LEAGUE_PRIOR_FB) / (cum_total + k_smooth)
    data["pitcher_bb_prior"] = (cum_counts["is_bb"] + k_smooth * LEAGUE_PRIOR_BB) / (cum_total + k_smooth)
    data["pitcher_os_prior"] = (cum_counts["is_os"] + k_smooth * LEAGUE_PRIOR_OS) / (cum_total + k_smooth)

    return data


def build_silver_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end transformation from Bronze DataFrame to Silver features.
    Filters out non-tactical / unclassified rows and projects SILVER_FEATURE_COLUMNS.
    """
    # 1. Filter out unclassified / non-tactical pitches
    valid_df = df[df["pitch_category"].notnull()].copy()

    # 2. Derive sequence features
    seq_df = derive_sequence_features(valid_df)

    # 3. Derive situational features
    sit_df = derive_situational_features(seq_df)

    # 4. Derive pitcher historical priors
    silver_df = derive_pitcher_historical_priors(sit_df)

    # 5. Return projected columns
    return silver_df[SILVER_FEATURE_COLUMNS].copy()


def build_silver_features_spark(bronze_df: Any) -> Any:
    """
    Spark / PySpark implementation of the Silver feature transformations for @dp pipeline.
    Uses window functions with zero-leakage specifications.
    """
    import pyspark.sql.functions as F
    from pyspark.sql import Window

    # 1. Filter out unclassified pitches
    filtered = bronze_df.filter(F.col("pitch_category").isNotNull())

    # 2. Sequence Window partitioned by (game_pk, at_bat_number) ordered by pitch_number
    ab_window = Window.partitionBy("game_pk", "at_bat_number").orderBy("pitch_number")

    with_sequence = (
        filtered.withColumn("prev_pitch_category", F.coalesce(F.lag("pitch_category", 1).over(ab_window), F.lit("First Pitch")))
        .withColumn("prev_pitch_type", F.coalesce(F.lag("pitch_type", 1).over(ab_window), F.lit("None")))
        .withColumn("prev_pitch_zone", F.coalesce(F.lag("zone", 1).over(ab_window), F.lit(0)).cast("int"))
        .withColumn("prev_pitch_result", F.coalesce(F.lag("type", 1).over(ab_window), F.lit("First Pitch")))
    )

    # 3. Situational and Platoon features
    with_situational = (
        with_sequence.withColumn("same_hand", F.when(F.col("p_throws") == F.col("stand"), 1).otherwise(0))
        .withColumn("count_state", F.concat_ws("-", F.col("balls"), F.col("strikes")))
        .withColumn(
            "count_leverage",
            F.when(F.col("count_state").isin("0-1", "0-2", "1-2"), "Pitcher Ahead")
            .when(F.col("count_state").isin("1-0", "2-0", "3-0", "3-1"), "Pitcher Behind")
            .otherwise("Neutral"),
        )
        .withColumn("runners_on_base", F.col("on_1b_flag") + F.col("on_2b_flag") + F.col("on_3b_flag"))
        .withColumn("scoring_position", F.when((F.col("on_2b_flag") == 1) | (F.col("on_3b_flag") == 1), 1).otherwise(0))
    )

    # 4. Pitcher historical priors Window (strictly preceding pitches)
    p_window = (
        Window.partitionBy("pitcher")
        .orderBy("game_date", "game_pk", "at_bat_number", "pitch_number")
        .rowsBetween(Window.unboundedPreceding, -1)
    )

    k_smooth = F.lit(15.0)
    fb_indicator = F.when(F.col("pitch_category") == "Fastball", 1.0).otherwise(0.0)
    bb_indicator = F.when(F.col("pitch_category") == "Breaking Ball", 1.0).otherwise(0.0)
    os_indicator = F.when(F.col("pitch_category") == "Offspeed", 1.0).otherwise(0.0)

    cum_fb = F.coalesce(F.sum(fb_indicator).over(p_window), F.lit(0.0))
    cum_bb = F.coalesce(F.sum(bb_indicator).over(p_window), F.lit(0.0))
    cum_os = F.coalesce(F.sum(os_indicator).over(p_window), F.lit(0.0))
    cum_total = cum_fb + cum_bb + cum_os

    with_priors = (
        with_situational.withColumn("pitcher_fb_prior", (cum_fb + k_smooth * F.lit(LEAGUE_PRIOR_FB)) / (cum_total + k_smooth))
        .withColumn("pitcher_bb_prior", (cum_bb + k_smooth * F.lit(LEAGUE_PRIOR_BB)) / (cum_total + k_smooth))
        .withColumn("pitcher_os_prior", (cum_os + k_smooth * F.lit(LEAGUE_PRIOR_OS)) / (cum_total + k_smooth))
    )

    return with_priors.select(SILVER_FEATURE_COLUMNS)
