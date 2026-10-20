# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # 01 - Statcast Data Extraction & Landing for `@dp` Declarative Pipeline
# MAGIC
# MAGIC This notebook extracts incremental pitch-by-pitch MLB Statcast data via `pybaseball`,
# MAGIC projects the leakage-free Bronze columns, generates an Exploratory Data Analysis (EDA)
# MAGIC profile report, and stages the Parquet batch into a Unity Catalog Volume for incremental
# MAGIC ingestion by the Lakeflow Spark Declarative Pipeline (`@dp` Auto CDC flow).

# COMMAND ----------
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

# Ensure local package is importable when running in Databricks workspace
try:
    from pitch_predictor.ingestion import (
        fetch_statcast_pitches,
        generate_bronze_eda_report,
        resolve_dual_track_ingestion,
        resolve_ingestion_window,
    )
except ImportError:
    candidate_dirs = [
        Path.cwd().parent / "src",
        Path.cwd() / "src",
        Path("/Workspace/Users/rhian.souza@programmers.com.br/.bundle/pitch_predictor/dev/files/src"),
        Path("/Workspace") / "Shared" / "pitch_predictor" / "src",
    ]
    for c_dir in candidate_dirs:
        if c_dir.exists() and str(c_dir) not in sys.path:
            sys.path.insert(0, str(c_dir))
    from pitch_predictor.ingestion import (
        fetch_statcast_pitches,
        generate_bronze_eda_report,
        resolve_dual_track_ingestion,
        resolve_ingestion_window,
    )

# COMMAND ----------
# Read configuration / parameters
# Default to "auto" for automated scheduled runs
default_start = "auto"
default_end = "auto"

dbutils.widgets.text("catalog", "pitch_predictor", "Catalog")
dbutils.widgets.text("schema", "baseball_ml", "Schema")
dbutils.widgets.text("volume", "pitch_data", "Landing Volume")
dbutils.widgets.text("start_dt", default_start, "Start Date (YYYY-MM-DD or auto)")
dbutils.widgets.text("end_dt", default_end, "End Date (YYYY-MM-DD or auto)")
dbutils.widgets.text("earliest_season_limit", "2021", "Earliest Season Limit (YYYY)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")
start_dt = dbutils.widgets.get("start_dt")
end_dt = dbutils.widgets.get("end_dt")
earliest_season_limit = int(dbutils.widgets.get("earliest_season_limit"))

landing_dir = f"/Volumes/{catalog}/{schema}/{volume}/statcast_landing"
print(f"Requested extraction parameters: start_dt='{start_dt}', end_dt='{end_dt}', earliest_season_limit={earliest_season_limit}")
print(f"Target UC Landing Volume: {landing_dir}")

# COMMAND ----------
# Ensure schema and managed volume exist in Unity Catalog
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume}")
os.makedirs(landing_dir, exist_ok=True)

# COMMAND ----------
# Query current chronological bounds from Bronze table
current_min_date = None
current_max_date = None
try:
    bounds = spark.table(f"{catalog}.{schema}.raw_pitches").selectExpr("min(game_date)", "max(game_date)").collect()
    if bounds and bounds[0][0] and bounds[0][1]:
        current_min_date = str(bounds[0][0])
        current_max_date = str(bounds[0][1])
        print(f"Current bounds in raw_pitches: min='{current_min_date}', max='{current_max_date}'")
except Exception as e:
    print(f"Notice: raw_pitches table bounds query ({e}).")

# COMMAND ----------
# Execute Ingestion: Dual-Track (Scheduled/Auto) vs Explicit Window (Ad-hoc)
is_auto = str(start_dt).strip().lower() in {"auto", ""} and str(end_dt).strip().lower() in {"auto", ""}
total_staged_pitches = 0

if is_auto:
    yesterday_str = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    plan = resolve_dual_track_ingestion(
        current_min_date=current_min_date,
        current_max_date=current_max_date,
        earliest_season_limit=earliest_season_limit,
        reference_date=yesterday_str,
    )
    print("Dual-Track Execution Plan:", json.dumps({k: str(v) for k, v in plan.items()}, indent=2))

    # Track 1: Forward Catch-Up (Current season up to yesterday)
    if plan["forward_window"]:
        s_fwd, e_fwd = plan["forward_window"]
        print(f"\n--- [Track 1: Forward Ingestion] {s_fwd} -> {e_fwd} ---")
        fwd_df = fetch_statcast_pitches(start_dt=s_fwd, end_dt=e_fwd, regular_season_only=True)
        if not fwd_df.empty:
            batch_file = f"{landing_dir}/statcast_{s_fwd}_{e_fwd}.parquet"
            fwd_df.to_parquet(batch_file, index=False)
            print(f"Staged {len(fwd_df)} rows to {batch_file}")
            total_staged_pitches += len(fwd_df)
        else:
            print("No regular-season games in forward window.")

    # Track 2A: 2024 Opening Day gap backfill
    if plan["gap_2024_window"]:
        s_gap, e_gap = plan["gap_2024_window"]
        print(f"\n--- [Track 2A: 2024 Opening Day Gap Backfill] {s_gap} -> {e_gap} ---")
        gap_df = fetch_statcast_pitches(start_dt=s_gap, end_dt=e_gap, regular_season_only=True)
        if not gap_df.empty:
            batch_file = f"{landing_dir}/statcast_{s_gap}_{e_gap}.parquet"
            gap_df.to_parquet(batch_file, index=False)
            print(f"Staged {len(gap_df)} rows to {batch_file}")
            total_staged_pitches += len(gap_df)

    # Track 2B: Historical Season Backward Backfill (1 full season per run in 6 monthly chunks)
    if plan["backward_season"]:
        target_season = plan["backward_season"]
        print(f"\n--- [Track 2B: Historical Season {target_season} Backfill] ---")
        for s_m, e_m in plan["backward_chunks"]:
            print(f"Ingesting monthly chunk {s_m} -> {e_m} for season {target_season}...")
            chunk_df = fetch_statcast_pitches(start_dt=s_m, end_dt=e_m, regular_season_only=True)
            if not chunk_df.empty:
                batch_file = f"{landing_dir}/statcast_{s_m}_{e_m}.parquet"
                chunk_df.to_parquet(batch_file, index=False)
                print(f" -> Staged {len(chunk_df)} rows to {batch_file}")
                total_staged_pitches += len(chunk_df)
            else:
                print(f" -> No regular-season games in chunk {s_m} -> {e_m}")
    else:
        print(f"\nHistorical season limit ({earliest_season_limit}) already reached or not required.")

else:
    # Explicit single window requested
    print(f"\n--- [Explicit Ingestion Window] {start_dt} -> {end_dt} ---")
    bronze_df = fetch_statcast_pitches(start_dt=start_dt, end_dt=end_dt, regular_season_only=True)
    if not bronze_df.empty:
        batch_file = f"{landing_dir}/statcast_{start_dt}_{end_dt}.parquet"
        bronze_df.to_parquet(batch_file, index=False)
        print(f"Staged {len(bronze_df)} rows to {batch_file}")
        total_staged_pitches += len(bronze_df)
    else:
        print(f"No regular-season games in window {start_dt} -> {end_dt}")

print(f"\nIngestion step complete. Total pitches staged in this run: {total_staged_pitches}")

