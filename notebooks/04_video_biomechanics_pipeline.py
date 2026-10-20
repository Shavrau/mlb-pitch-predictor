# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # 04 - Incremental Video Ingestion & Biomechanical Pose Estimation
# MAGIC
# MAGIC This notebook executes an automated, incremental workflow on Databricks Serverless:
# MAGIC 1. Queries newly processed pitches from `silver_pitch_features` for the target pitcher.
# MAGIC 2. Compares keys `(game_pk, at_bat_number, pitch_number)` with the Volume manifest (`manifest.csv`).
# MAGIC 3. Downloads missing video clips incrementally directly into the Unity Catalog Managed Volume.
# MAGIC 4. Runs 3D skeletal pose estimation (MediaPipe Pose in headless mode) to extract release kinematics.
# MAGIC 5. Appends new records to `biomechanical_features.parquet` and registers/updates the Delta table `{catalog}.{schema}.biomechanical_features`.
# MAGIC 6. Retrains and synchronizes the Multimodal Pitch Predictor model.

# COMMAND ----------
import os
import sys
from pathlib import Path

# Add src to sys.path first to ensure package resolution on Serverless
candidate_dirs = [
    Path.cwd().parent / "src",
    Path.cwd() / "src",
    Path("/Workspace/Users/rhian.souza@programmers.com.br/.bundle/pitch_predictor/dev/files/src"),
    Path("/Workspace") / "Shared" / "pitch_predictor" / "src",
]
for c_dir in candidate_dirs:
    if c_dir.exists() and str(c_dir) not in sys.path:
        sys.path.insert(0, str(c_dir))

from pitch_predictor.vision.incremental_pipeline import process_incremental_video_batch

import pandas as pd

# COMMAND ----------
# Widgets / Parameters
dbutils.widgets.text("catalog", "pitch_predictor", "Catalog")
dbutils.widgets.text("schema", "baseball_ml_dev", "Schema")
dbutils.widgets.text("volume", "pitch_data", "Volume")
dbutils.widgets.text("pitcher_id", "694973", "Pitcher ID (Paul Skenes)")
dbutils.widgets.text("max_pitches", "15", "Max Pitches Per Execution")
dbutils.widgets.text("sync_models", "true", "Sync Multimodal Models (true/false)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")
pitcher_id = int(dbutils.widgets.get("pitcher_id"))
max_pitches = int(dbutils.widgets.get("max_pitches"))
sync_models = dbutils.widgets.get("sync_models").lower() in ("true", "1", "yes")

print("=" * 65)
print("[JOB TASK] INCREMENTAL VIDEO & BIOMECHANICS PIPELINE")
print(f"Target Catalog: {catalog}")
print(f"Target Schema:  {schema}")
print(f"Target Volume:  {volume}")
print(f"Pitcher ID:     {pitcher_id}")
print(f"Batch Limit:    {max_pitches} pitches")
print(f"Sync Models:    {sync_models}")
print("=" * 65)

# COMMAND ----------
# Step 1: Query candidate pitches from Silver Layer
silver_table = f"{catalog}.{schema}.silver_pitch_features"
candidates_df = None

try:
    print(f"Attempting to query candidate pitches from Silver table: {silver_table}...")
    spark_df = spark.table(silver_table).filter(f"pitcher = {pitcher_id}")
    if spark_df.count() > 0:
        candidates_df = spark_df.toPandas()
        print(f"Found {len(candidates_df)} pitches in {silver_table} for pitcher {pitcher_id}.")
    else:
        print(f"No pitches found in {silver_table} for pitcher {pitcher_id}.")
except Exception as e:
    print(f"Silver table query failed or table not found ({e}). Falling back to Statcast live query...")

if candidates_df is None or candidates_df.empty:
    import pybaseball
    print(f"Fetching recent pitches for pitcher {pitcher_id} via pybaseball...")
    candidates_df = pybaseball.statcast_pitcher("2024-05-15", "2024-06-15", player_id=pitcher_id)
    if "pitch_category" not in candidates_df.columns:
        from pitch_predictor.ingestion import map_pitch_category
        candidates_df["pitch_category"] = candidates_df["pitch_type"].apply(map_pitch_category)

print(f"Total candidate pitches available for evaluation: {len(candidates_df)}")

# COMMAND ----------
# Step 2: Resolve paths on Databricks Volume
volume_base = Path(f"/Volumes/{catalog}/{schema}/{volume}/videos/{pitcher_id}")
volume_base.mkdir(parents=True, exist_ok=True)
print(f"Target Volume Directory: {volume_base}")

# COMMAND ----------
# Step 3: Run Incremental Batch Processing
summary = process_incremental_video_batch(
    candidates_df=candidates_df,
    output_dir=volume_base,
    pitcher_id=pitcher_id,
    max_pitches=max_pitches,
    sync_models=sync_models,
    sync_catalog=False,  # Direct POSIX access in Volume inside Databricks
    catalog=catalog,
    schema=schema,
    volume=volume,
)

print("\n" + "=" * 65)
print("EXECUTION SUMMARY:")
print(f"Status:                 {summary.get('status')}")
print(f"Pitches Processed:      {summary.get('processed')}")
print(f"Total Manifest Records: {summary.get('total_manifest_records')}")
print("=" * 65)

# COMMAND ----------
# Step 4: Register / Update Delta Table in Unity Catalog
parquet_file = volume_base / "biomechanical_features.parquet"
if parquet_file.exists():
    try:
        print("\nUpdating Delta Table in Unity Catalog...")
        delta_table_name = f"{catalog}.{schema}.biomechanical_features"
        bio_spark_df = spark.read.parquet(str(parquet_file))
        bio_spark_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(delta_table_name)
        print(f"[OK] Materialized Unity Catalog Delta Table: {delta_table_name} ({bio_spark_df.count()} rows).")
    except Exception as exc:
        print(f"[WARN] Could not update Delta Table {delta_table_name}: {exc}")
else:
    print("No biomechanical features file found to register.")

# COMMAND ----------
print("[COMPLETE] Incremental video & biomechanics pipeline completed successfully.")
