# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # 02 - Databricks AutoML Model Training & Unity Catalog Registration
# MAGIC 
# MAGIC This notebook executes distributed training with **Databricks AutoML / Multi-Model Automated Trials** against the materialized Silver feature table
# MAGIC (`silver_pitch_features`), optimizing for **Macro F1** with strict temporal splitting via `game_date`, and registers
# MAGIC the winning model into **Unity Catalog** (`{catalog}.{schema}.pitch_category_predictor`) with `@prod` alias.

# COMMAND ----------
import os
import sys
from pathlib import Path

# Ensure local package is importable when running in Databricks workspace
try:
    from pitch_predictor.training import (
        CATEGORICAL_FEATURE_COLS,
        IDENTIFIER_COLS,
        MODEL_NAME,
        NUMERIC_FEATURE_COLS,
        PRIMARY_METRIC,
        TARGET_COL,
        TIME_COL,
        TRAINING_FEATURE_COLS,
        VALID_PITCH_CATEGORIES,
        build_automl_params,
        build_model_pipeline,
        compute_classification_metrics,
        generate_metrics_impact_report,
        get_uc_model_name,
        register_best_model_to_uc,
        temporal_train_test_split,
        validate_target_data,
    )
except ImportError:
    candidate_dirs = [
        Path.cwd().parent / "src",
        Path.cwd() / "src",
        Path("/Workspace/Users/rhian.souza@programmers.com.br/.bundle/pitch_predictor/dev/files/src"),
    ]
    for c_dir in candidate_dirs:
        if c_dir.exists() and str(c_dir) not in sys.path:
            sys.path.insert(0, str(c_dir))
    from pitch_predictor.training import (
        CATEGORICAL_FEATURE_COLS,
        IDENTIFIER_COLS,
        MODEL_NAME,
        NUMERIC_FEATURE_COLS,
        PRIMARY_METRIC,
        TARGET_COL,
        TIME_COL,
        TRAINING_FEATURE_COLS,
        VALID_PITCH_CATEGORIES,
        build_automl_params,
        build_model_pipeline,
        compute_classification_metrics,
        generate_metrics_impact_report,
        get_uc_model_name,
        register_best_model_to_uc,
        temporal_train_test_split,
        validate_target_data,
    )

import json
import mlflow
import mlflow.models
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pyspark.sql.functions as F

# COMMAND ----------
# Parameters
dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "baseball_ml_dev", "Schema")
dbutils.widgets.text("table_name", "silver_pitch_features", "Source Table")
dbutils.widgets.text("timeout_minutes", "20", "Timeout (Minutes)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
table_name = dbutils.widgets.get("table_name")
timeout_minutes = int(dbutils.widgets.get("timeout_minutes"))

full_table_path = f"{catalog}.{schema}.{table_name}"
print(f"Reading training data from: {full_table_path}")
print(f"Target column: {TARGET_COL} | Time anchor: {TIME_COL} | Primary metric: {PRIMARY_METRIC}")

# COMMAND ----------
# Load Silver feature dataset
silver_df = spark.table(full_table_path)
total_count = silver_df.count()
print(f"Loaded {total_count} rows from {full_table_path}")

# COMMAND ----------
# Feature Selection: Exclude entity IDs ('game_pk', 'at_bat_number', 'pitch_number', 'pitcher', 'batter')
# and retain only training features, temporal split anchor, and target column
selected_cols = TRAINING_FEATURE_COLS + [TIME_COL, TARGET_COL]
training_dataset = silver_df.select(selected_cols).withColumn(TIME_COL, F.to_date(F.col(TIME_COL)))
print(f"Selected {len(TRAINING_FEATURE_COLS)} training features + '{TIME_COL}' + '{TARGET_COL}'.")
print("Excluded high-cardinality IDs:", IDENTIFIER_COLS)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Run Databricks AutoML / Multi-Model Automated Exploration
# MAGIC *(Runs distributed hyperparameter search across multiple models e.g., LightGBM, HistGBDT, Random Forest, Logistic Regression)*

# COMMAND ----------
automl_params = build_automl_params(
    catalog=catalog,
    schema=schema,
    timeout_minutes=timeout_minutes,
)
experiment_name = automl_params["experiment_name"]
print("AutoML Configuration:", automl_params)

# Set MLflow tracking and UC model registry
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(experiment_name)

# COMMAND ----------
trained_successfully = False
best_model_uri = None

try:
    import databricks.automl as automl
    print("Attempting execution via databricks.automl.classify...")
    summary = automl.classify(
        dataset=training_dataset,
        target_col=TARGET_COL,
        primary_metric=PRIMARY_METRIC,
        time_col=TIME_COL,
        timeout_minutes=timeout_minutes,
        experiment_name=experiment_name,
    )
    best_trial = summary.best_trial
    print(f"AutoML Best Trial Description: {best_trial.model_description}")
    print(f"AutoML Best Trial Primary Metric ({PRIMARY_METRIC}): {best_trial.metric}")
    best_model_uri = f"runs:/{best_trial.mlflow_run_id}/model"
    trained_successfully = True
except (ImportError, Exception) as e:
    print(f"databricks.automl.classify not available or failed ({e}). Running multi-model automated training suite with MLflow...")

if not trained_successfully:
    # Convert to Pandas DataFrame for multi-model trials
    pdf = training_dataset.toPandas()
    validate_target_data(pdf)

    # Chronological train/test split: automatic 80/20 date split with zero leakage
    train_df, test_df = temporal_train_test_split(pdf, split_date=None, train_ratio=0.8)
    print(f"Chronological split: Train = {len(train_df)} rows ({train_df[TIME_COL].min()} to {train_df[TIME_COL].max()}), "
          f"Test = {len(test_df)} rows ({test_df[TIME_COL].min()} to {test_df[TIME_COL].max()})")

    X_train = train_df[TRAINING_FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[TRAINING_FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    classes = sorted(VALID_PITCH_CATEGORIES)

    # Candidate model configurations for exploration
    candidate_trials = [
        {
            "name": "HistGBDT_balanced_d6_iter200",
            "model_type": "hist_gbdt",
            "params": {"max_iter": 200, "max_depth": 6, "l2_regularization": 1.0, "class_weight": "balanced"},
        },
        {
            "name": "HistGBDT_balanced_d8_iter250",
            "model_type": "hist_gbdt",
            "params": {"max_iter": 250, "max_depth": 8, "l2_regularization": 2.0, "class_weight": "balanced"},
        },
        {
            "name": "RandomForest_balanced_d12",
            "model_type": "random_forest",
            "params": {"n_estimators": 200, "max_depth": 12, "class_weight": "balanced"},
        },
        {
            "name": "RandomForest_balanced_d8",
            "model_type": "random_forest",
            "params": {"n_estimators": 150, "max_depth": 8, "class_weight": "balanced"},
        },
        {
            "name": "LogisticRegression_balanced_baseline",
            "model_type": "logistic_regression",
            "params": {"max_iter": 1000, "class_weight": "balanced"},
        },
    ]

    # Try including LightGBM if installed
    try:
        import lightgbm
        candidate_trials.insert(0, {
            "name": "LightGBM_balanced_d6_lr05",
            "model_type": "lightgbm",
            "params": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05, "class_weight": "balanced"},
        })
        candidate_trials.insert(1, {
            "name": "LightGBM_balanced_d8_lr03",
            "model_type": "lightgbm",
            "params": {"n_estimators": 250, "max_depth": 8, "learning_rate": 0.03, "class_weight": "balanced"},
        })
    except ImportError:
        pass

    trial_results = []
    print("\n--- Starting Automated Multi-Model Training Trials ---")

    for trial in candidate_trials:
        trial_name = trial["name"]
        print(f"\nTraining trial: {trial_name} ({trial['model_type']})...")
        try:
            with mlflow.start_run(run_name=trial_name) as run:
                mlflow.set_tag("model_family", trial["model_type"])
                mlflow.set_tag("split_type", "chronological_zero_leakage")
                mlflow.set_tag("problem_type", "multiclass_classification")
                mlflow.log_params(trial["params"])

                pipeline = build_model_pipeline(model_type=trial["model_type"], **trial["params"])
                pipeline.fit(X_train, y_train)

                y_pred = pipeline.predict(X_test)
                y_prob = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None

                metrics = compute_classification_metrics(y_test, y_pred, y_prob=y_prob, classes=classes)
                mlflow.log_metrics(metrics)

                # Log model artifact with input example and signature
                input_example = X_train.head(3)
                signature = mlflow.models.infer_signature(X_train.head(5), pipeline.predict(X_train.head(5)))
                mlflow.sklearn.log_model(
                    sk_model=pipeline,
                    artifact_path="model",
                    input_example=input_example,
                    signature=signature,
                )

                result_entry = {
                    "trial_name": trial_name,
                    "model_type": trial["model_type"],
                    "run_id": run.info.run_id,
                    "f1_macro": metrics["f1_macro"],
                    "f1_weighted": metrics["f1_weighted"],
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "log_loss": metrics.get("log_loss", np.nan),
                    "model_uri": f"runs:/{run.info.run_id}/model",
                }
                trial_results.append(result_entry)
                print(f" -> Finished {trial_name} | F1 Macro: {metrics['f1_macro']:.4f} | Balanced Acc: {metrics['balanced_accuracy']:.4f} | Acc: {metrics['accuracy']:.4f}")
        except Exception as trial_err:
            print(f" -> Failed trial {trial_name}: {trial_err}")

    # Rank trials by Macro F1
    results_df = pd.DataFrame(trial_results).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    print("\n================== AutoML Leaderboard ==================")
    print(results_df[["trial_name", "model_type", "f1_macro", "balanced_accuracy", "accuracy", "run_id"]].to_string())

    best_trial_record = results_df.iloc[0]
    best_model_uri = best_trial_record["model_uri"]
    print(f"\nWinning Trial: {best_trial_record['trial_name']} (F1 Macro = {best_trial_record['f1_macro']:.4f})")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Register Winning Model into Unity Catalog & Assign @prod Alias

# COMMAND ----------
# Query previous production model metrics for impact comparison
client = MlflowClient(registry_uri="databricks-uc")
full_model_name = get_uc_model_name(catalog, schema)
previous_version_num = None
previous_metrics = None

try:
    existing_alias = client.get_model_version_by_alias(full_model_name, "prod")
    previous_version_num = existing_alias.version
    prev_run = client.get_run(existing_alias.run_id)
    previous_metrics = prev_run.data.metrics
    print(f"Previous production version detected: Version {previous_version_num} (Run ID: {existing_alias.run_id})")
except Exception as e:
    print(f"Notice: No prior @prod version found ({e}). This run establishes the baseline.")

print(f"Registering winning model from {best_model_uri} into Unity Catalog...")
registered_version = register_best_model_to_uc(
    model_uri=best_model_uri,
    catalog=catalog,
    schema=schema,
)
print(f"Model successfully registered in Unity Catalog: {full_model_name} (Version: {registered_version.version})")

# Get winning run metrics
winning_run = client.get_run(best_trial_record["run_id"])
current_metrics = winning_run.data.metrics

# Generate Daily Impact Markdown Report
impact_report_md = generate_metrics_impact_report(
    current_version=registered_version.version,
    previous_version=previous_version_num,
    current_metrics=current_metrics,
    previous_metrics=previous_metrics,
    model_name=full_model_name,
    current_rows=total_count,
)
print("\n" + impact_report_md + "\n")

# Save daily impact report to Unity Catalog Volume
try:
    report_dir = f"/Volumes/{catalog}/{schema}/pitch_data/model_reports"
    os.makedirs(report_dir, exist_ok=True)
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    report_file = f"{report_dir}/daily_impact_report_{today_str}.md"
    latest_file = f"{report_dir}/latest_impact_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(impact_report_md)
    with open(latest_file, "w", encoding="utf-8") as f:
        f.write(impact_report_md)
    print(f"Saved daily impact report to {report_file} and {latest_file}")
except Exception as vol_err:
    print(f"Notice saving report to volume: {vol_err}")

# Log report into MLflow artifacts
try:
    with mlflow.start_run(run_id=best_trial_record["run_id"]):
        mlflow.log_text(impact_report_md, "daily_model_impact_report.md")
except Exception as art_err:
    print(f"Notice logging report to MLflow: {art_err}")

# Set alias @prod in Unity Catalog
try:
    client.set_registered_model_alias(full_model_name, "prod", str(registered_version.version))
    print(f"Assigned alias '@prod' to model version {registered_version.version} in Unity Catalog.")
except Exception as alias_err:
    print(f"Warning setting alias '@prod': {alias_err}")
