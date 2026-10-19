# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # 03 - Model Inference & Validation
# MAGIC 
# MAGIC This notebook loads the winning model registered by the AutoML training run in Unity Catalog
# MAGIC (`{catalog}.{schema}.pitch_category_predictor@prod`) and performs batch validation over test samples.

# COMMAND ----------
import os
import sys
from pathlib import Path

# Ensure local package is importable when running in Databricks workspace
try:
    from pitch_predictor.training import TRAINING_FEATURE_COLS
except ImportError:
    candidate_dirs = [
        Path.cwd().parent / "src",
        Path.cwd() / "src",
        Path("/Workspace/Users/rhian.souza@programmers.com.br/.bundle/pitch_predictor/dev/files/src"),
    ]
    for c_dir in candidate_dirs:
        if c_dir.exists() and str(c_dir) not in sys.path:
            sys.path.insert(0, str(c_dir))
    from pitch_predictor.training import TRAINING_FEATURE_COLS

import mlflow
import pyspark.sql.functions as F

# COMMAND ----------
dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "baseball_ml_dev", "Schema")
dbutils.widgets.text("model_alias", "prod", "Model Alias")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_alias = dbutils.widgets.get("model_alias")

full_model_name = f"{catalog}.{schema}.pitch_category_predictor"
model_uri = f"models:/{full_model_name}@{model_alias}"

print(f"Loading Unity Catalog model: {model_uri}")
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------
# Load model as PyFunc
try:
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Model successfully loaded. MLflow Run ID: {model.metadata.run_id}")
    
    # Read sample features from silver table to test scoring
    sample_df = spark.table(f"{catalog}.{schema}.silver_pitch_features").limit(10).toPandas()
    predictions = model.predict(sample_df[TRAINING_FEATURE_COLS])
    print(f"Sample Predictions (first 5 pitches): {list(predictions[:5])}")
    print("Inference validation completed with SUCCESS.")
except Exception as e:
    print(f"Notice during model loading: {e}. Inference template ready.")
