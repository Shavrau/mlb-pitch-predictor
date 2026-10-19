# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # 00 - Environment Setup (Unity Catalog, Schema, and Volume)
# MAGIC
# MAGIC This notebook ensures that the required Unity Catalog, Schema, and Managed Volume
# MAGIC exist before any ingestion or pipeline task runs, guaranteeing 100% reproducibility across environments.

# COMMAND ----------
# Widgets / parameters
dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "baseball_ml", "Schema")
dbutils.widgets.text("volume", "pitch_data", "Volume")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")

print(f"Target Catalog: {catalog}")
print(f"Target Schema: {catalog}.{schema}")
print(f"Target Volume: {catalog}.{schema}.{volume}")

# COMMAND ----------
# Step 1: Attempt Catalog creation (if allowed; otherwise fall back to existing catalog)
try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
    print(f"Catalog '{catalog}' verified/created.")
except Exception as e:
    print(f"Notice: Could not create catalog '{catalog}' (may require metastore admin privileges). Assuming catalog already exists: {e}")

# COMMAND ----------
# Step 2: Ensure Schema exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema} COMMENT 'MLB Pitch Predictor dataset and feature tables'")
print(f"Schema '{catalog}.{schema}' verified/created.")

# COMMAND ----------
# Step 3: Ensure Managed Volume exists for staging raw Statcast Parquet batches
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume} COMMENT 'Managed volume for staging raw Statcast Parquet data'")
print(f"Volume '{catalog}.{schema}.{volume}' verified/created.")

# COMMAND ----------
# Step 4: Verify volume landing path
volume_path = f"/Volumes/{catalog}/{schema}/{volume}"
landing_path = f"{volume_path}/statcast_landing"

import os
os.makedirs(landing_path, exist_ok=True)
print(f"Unity Catalog landing directory ready at: {landing_path}")
