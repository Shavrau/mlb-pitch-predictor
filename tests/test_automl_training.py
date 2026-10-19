"""
TDD test suite for Databricks AutoML Model Training, Feature Selection, and Unity Catalog Model Registration.

Tests:
1. Feature selection validation from SILVER_FEATURE_COLUMNS:
   - Primary keys and direct identifiers ('game_pk', 'at_bat_number', 'pitch_number', 'pitcher', 'batter')
     MUST NOT be used as direct training features.
   - 'game_date' MUST NOT be a training feature (reserved strictly as time_col anchor for temporal split).
   - 'pitch_category' MUST be the prediction target.
   - Exactly all remaining situational, platoon, sequence, and prior columns must be training features.
2. Databricks AutoML configuration parameters:
   - target_col == 'pitch_category'
   - primary_metric in ('f1_macro', 'f1_score', 'f1')
   - time_col == 'game_date'
   - timeout_minutes configured properly
   - experiment_name points to workspace MLflow path.
3. Unity Catalog Model Registration:
   - 3-level namespace '{catalog}.{schema}.pitch_category_predictor'
   - MLflow registry_uri set to 'databricks-uc'
4. Temporal split zero-leakage isolation:
   - Strict chronological ordering with no future leakage or overlapping games.
5. Target category consistency:
   - Strictly 3 classes: ['Fastball', 'Breaking Ball', 'Offspeed'].
   - Detection and rejection of nulls, atypical classes, or degenerate class distributions.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pitch_predictor.features import SILVER_FEATURE_COLUMNS


def create_sample_silver_df() -> pd.DataFrame:
    """Create a synthetic DataFrame conforming to silver_pitch_features schema across multiple dates."""
    data = [
        # Game 101, 2024-04-01
        {
            "game_pk": 101,
            "at_bat_number": 1,
            "pitch_number": 1,
            "pitcher": 5001,
            "batter": 6001,
            "game_date": "2024-04-01",
            "pitch_category": "Fastball",
            "prev_pitch_category": "First Pitch",
            "prev_pitch_type": "None",
            "prev_pitch_zone": 0,
            "prev_pitch_result": "First Pitch",
            "p_throws": "R",
            "stand": "L",
            "same_hand": 0,
            "balls": 0,
            "strikes": 0,
            "count_state": "0-0",
            "count_leverage": "Neutral",
            "outs_when_up": 0,
            "inning": 1,
            "inning_topbot": "Top",
            "score_diff": 0,
            "on_1b_flag": 0,
            "on_2b_flag": 0,
            "on_3b_flag": 0,
            "runners_on_base": 0,
            "scoring_position": 0,
            "n_thruorder_pitcher": 1,
            "pitcher_fb_prior": 0.55,
            "pitcher_bb_prior": 0.30,
            "pitcher_os_prior": 0.15,
        },
        {
            "game_pk": 101,
            "at_bat_number": 1,
            "pitch_number": 2,
            "pitcher": 5001,
            "batter": 6001,
            "game_date": "2024-04-01",
            "pitch_category": "Breaking Ball",
            "prev_pitch_category": "Fastball",
            "prev_pitch_type": "FF",
            "prev_pitch_zone": 5,
            "prev_pitch_result": "S",
            "p_throws": "R",
            "stand": "L",
            "same_hand": 0,
            "balls": 0,
            "strikes": 1,
            "count_state": "0-1",
            "count_leverage": "Pitcher Ahead",
            "outs_when_up": 0,
            "inning": 1,
            "inning_topbot": "Top",
            "score_diff": 0,
            "on_1b_flag": 0,
            "on_2b_flag": 0,
            "on_3b_flag": 0,
            "runners_on_base": 0,
            "scoring_position": 0,
            "n_thruorder_pitcher": 1,
            "pitcher_fb_prior": 0.56,
            "pitcher_bb_prior": 0.29,
            "pitcher_os_prior": 0.15,
        },
        {
            "game_pk": 101,
            "at_bat_number": 1,
            "pitch_number": 3,
            "pitcher": 5001,
            "batter": 6001,
            "game_date": "2024-04-01",
            "pitch_category": "Offspeed",
            "prev_pitch_category": "Breaking Ball",
            "prev_pitch_type": "SL",
            "prev_pitch_zone": 14,
            "prev_pitch_result": "B",
            "p_throws": "R",
            "stand": "L",
            "same_hand": 0,
            "balls": 1,
            "strikes": 1,
            "count_state": "1-1",
            "count_leverage": "Neutral",
            "outs_when_up": 0,
            "inning": 1,
            "inning_topbot": "Top",
            "score_diff": 0,
            "on_1b_flag": 0,
            "on_2b_flag": 0,
            "on_3b_flag": 0,
            "runners_on_base": 0,
            "scoring_position": 0,
            "n_thruorder_pitcher": 1,
            "pitcher_fb_prior": 0.54,
            "pitcher_bb_prior": 0.31,
            "pitcher_os_prior": 0.15,
        },
        # Game 102, 2024-04-02
        {
            "game_pk": 102,
            "at_bat_number": 1,
            "pitch_number": 1,
            "pitcher": 5002,
            "batter": 6002,
            "game_date": "2024-04-02",
            "pitch_category": "Fastball",
            "prev_pitch_category": "First Pitch",
            "prev_pitch_type": "None",
            "prev_pitch_zone": 0,
            "prev_pitch_result": "First Pitch",
            "p_throws": "L",
            "stand": "R",
            "same_hand": 0,
            "balls": 0,
            "strikes": 0,
            "count_state": "0-0",
            "count_leverage": "Neutral",
            "outs_when_up": 0,
            "inning": 1,
            "inning_topbot": "Top",
            "score_diff": 0,
            "on_1b_flag": 1,
            "on_2b_flag": 0,
            "on_3b_flag": 0,
            "runners_on_base": 1,
            "scoring_position": 0,
            "n_thruorder_pitcher": 1,
            "pitcher_fb_prior": 0.52,
            "pitcher_bb_prior": 0.32,
            "pitcher_os_prior": 0.16,
        },
        # Game 103, 2024-04-03
        {
            "game_pk": 103,
            "at_bat_number": 1,
            "pitch_number": 1,
            "pitcher": 5001,
            "batter": 6003,
            "game_date": "2024-04-03",
            "pitch_category": "Breaking Ball",
            "prev_pitch_category": "First Pitch",
            "prev_pitch_type": "None",
            "prev_pitch_zone": 0,
            "prev_pitch_result": "First Pitch",
            "p_throws": "R",
            "stand": "R",
            "same_hand": 1,
            "balls": 0,
            "strikes": 0,
            "count_state": "0-0",
            "count_leverage": "Neutral",
            "outs_when_up": 1,
            "inning": 2,
            "inning_topbot": "Bot",
            "score_diff": 1,
            "on_1b_flag": 0,
            "on_2b_flag": 1,
            "on_3b_flag": 0,
            "runners_on_base": 1,
            "scoring_position": 1,
            "n_thruorder_pitcher": 1,
            "pitcher_fb_prior": 0.53,
            "pitcher_bb_prior": 0.31,
            "pitcher_os_prior": 0.16,
        },
        # Game 104, 2024-04-04
        {
            "game_pk": 104,
            "at_bat_number": 2,
            "pitch_number": 1,
            "pitcher": 5003,
            "batter": 6001,
            "game_date": "2024-04-04",
            "pitch_category": "Offspeed",
            "prev_pitch_category": "First Pitch",
            "prev_pitch_type": "None",
            "prev_pitch_zone": 0,
            "prev_pitch_result": "First Pitch",
            "p_throws": "R",
            "stand": "L",
            "same_hand": 0,
            "balls": 0,
            "strikes": 0,
            "count_state": "0-0",
            "count_leverage": "Neutral",
            "outs_when_up": 2,
            "inning": 3,
            "inning_topbot": "Top",
            "score_diff": -2,
            "on_1b_flag": 0,
            "on_2b_flag": 0,
            "on_3b_flag": 1,
            "runners_on_base": 1,
            "scoring_position": 1,
            "n_thruorder_pitcher": 2,
            "pitcher_fb_prior": 0.50,
            "pitcher_bb_prior": 0.30,
            "pitcher_os_prior": 0.20,
        },
    ]
    df = pd.DataFrame(data)
    # Ensure all columns from SILVER_FEATURE_COLUMNS exist
    for col in SILVER_FEATURE_COLUMNS:
        assert col in df.columns, f"Fixture missing column: {col}"
    return df


def test_feature_selection_contract():
    """
    Verify feature selection contract:
    - Primary keys and IDs ('game_pk', 'at_bat_number', 'pitch_number', 'pitcher', 'batter')
      MUST NOT be included in training features.
    - 'game_date' MUST NOT be in training features (temporal anchor only).
    - 'pitch_category' MUST NOT be in training features (target only).
    - All training features belong to SILVER_FEATURE_COLUMNS.
    - The union of training features, identifiers, time anchor, and target covers all SILVER_FEATURE_COLUMNS.
    """
    from pitch_predictor.training import (
        IDENTIFIER_COLS,
        TARGET_COL,
        TIME_COL,
        TRAINING_FEATURE_COLS,
    )

    # Required constants
    assert IDENTIFIER_COLS == ["game_pk", "at_bat_number", "pitch_number", "pitcher", "batter"]
    assert TIME_COL == "game_date"
    assert TARGET_COL == "pitch_category"

    # Leakage & ID isolation checks
    for id_col in IDENTIFIER_COLS:
        assert id_col not in TRAINING_FEATURE_COLS, (
            f"Identifier '{id_col}' found in TRAINING_FEATURE_COLS! Direct IDs cause extreme overfitting."
        )

    assert TIME_COL not in TRAINING_FEATURE_COLS, "TIME_COL 'game_date' must not be in TRAINING_FEATURE_COLS"
    assert TARGET_COL not in TRAINING_FEATURE_COLS, "TARGET_COL 'pitch_category' must not be in TRAINING_FEATURE_COLS"

    # Silver feature coverage
    all_silver = set(SILVER_FEATURE_COLUMNS)
    training_set = set(TRAINING_FEATURE_COLS)
    excluded_set = set(IDENTIFIER_COLS) | {TIME_COL, TARGET_COL}

    assert training_set.issubset(all_silver), "TRAINING_FEATURE_COLS contains columns not in SILVER_FEATURE_COLUMNS"
    assert training_set.intersection(excluded_set) == set(), "Overlapping columns between features and excluded keys"
    assert training_set | excluded_set == all_silver, "Some Silver features are unassigned"


def test_select_automl_features_dataframe():
    """
    Verify select_automl_features drops high-cardinality IDs and preserves only
    training features, time_col (for chronological split), and target_col.
    """
    from pitch_predictor.training import (
        IDENTIFIER_COLS,
        TARGET_COL,
        TIME_COL,
        TRAINING_FEATURE_COLS,
        select_automl_features,
    )

    df = create_sample_silver_df()
    prepared_df = select_automl_features(df)

    # Identifiers must be completely absent from prepared dataset
    for id_col in IDENTIFIER_COLS:
        assert id_col not in prepared_df.columns, f"ID column '{id_col}' was not dropped in select_automl_features"

    # Time anchor and Target must be present
    assert TIME_COL in prepared_df.columns
    assert TARGET_COL in prepared_df.columns

    # All training features must be present
    for feature in TRAINING_FEATURE_COLS:
        assert feature in prepared_df.columns, f"Feature '{feature}' missing from prepared DataFrame"

    expected_cols = set(TRAINING_FEATURE_COLS) | {TIME_COL, TARGET_COL}
    assert set(prepared_df.columns) == expected_cols
    assert len(prepared_df) == len(df)


def test_automl_parameters_configuration():
    """
    Verify configuration parameters for Databricks AutoML:
    - target_col: 'pitch_category'
    - primary_metric: 'f1_macro' (or 'f1_score'/'f1')
    - time_col: 'game_date'
    - timeout_minutes: positive integer
    - experiment_name: structured workspace path
    """
    from pitch_predictor.training import (
        PRIMARY_METRIC,
        TARGET_COL,
        TIME_COL,
        build_automl_params,
    )

    assert PRIMARY_METRIC in ["f1_macro", "f1_score", "f1"]
    assert TARGET_COL == "pitch_category"
    assert TIME_COL == "game_date"

    params = build_automl_params(catalog="main", schema="baseball_ml_dev", timeout_minutes=40)

    assert params["target_col"] == "pitch_category"
    assert params["primary_metric"] in ["f1_macro", "f1_score", "f1"]
    assert params["time_col"] == "game_date"
    assert params["timeout_minutes"] == 40
    assert "experiment_name" in params
    assert "baseball_ml_dev" in params["experiment_name"]


def test_unity_catalog_model_naming_and_registration():
    """
    Verify Unity Catalog model registration:
    - Registered model name must follow 3-level namespace '{catalog}.{schema}.pitch_category_predictor'.
    - Registry URI must be strictly set to 'databricks-uc'.
    - MLflow register_model is invoked with the expected parameters.
    """
    from pitch_predictor.training import (
        MODEL_NAME,
        get_uc_model_name,
        register_best_model_to_uc,
    )

    assert MODEL_NAME == "pitch_category_predictor"

    model_name_dev = get_uc_model_name("main", "baseball_ml_dev")
    assert model_name_dev == "main.baseball_ml_dev.pitch_category_predictor"
    parts = model_name_dev.split(".")
    assert len(parts) == 3, f"Expected 3-level UC namespace 'catalog.schema.model', got '{model_name_dev}'"

    model_name_prod = get_uc_model_name("prod_catalog", "ml_baseball")
    assert model_name_prod == "prod_catalog.ml_baseball.pitch_category_predictor"

    # Test MLflow registration logic with mocked mlflow
    mock_mlflow = MagicMock()
    with patch("pitch_predictor.training.mlflow", mock_mlflow):
        register_best_model_to_uc(
            model_uri="runs:/fake_run_123/model",
            catalog="main",
            schema="baseball_ml_dev",
        )

        # Invariant: registry_uri must be 'databricks-uc' for Unity Catalog model governance
        mock_mlflow.set_registry_uri.assert_called_once_with("databricks-uc")
        mock_mlflow.register_model.assert_called_once_with(
            model_uri="runs:/fake_run_123/model",
            name="main.baseball_ml_dev.pitch_category_predictor",
        )


def test_temporal_train_test_split_zero_leakage():
    """
    Verify chronological train/test split:
    - Train data must strictly precede or match the cutoff date.
    - Test data must strictly be after the cutoff date.
    - Zero overlapping games (game_pk) between train and test.
    """
    from pitch_predictor.training import temporal_train_test_split

    df = create_sample_silver_df()
    split_date = "2024-04-02"

    train_df, test_df = temporal_train_test_split(df, split_date=split_date)

    # Chronological invariant
    assert train_df["game_date"].max() <= split_date
    assert test_df["game_date"].min() > split_date

    # No game overlap
    train_games = set(train_df["game_pk"])
    test_games = set(test_df["game_pk"])
    overlap = train_games.intersection(test_games)
    assert not overlap, f"Leakage detected: game_pks found in both train and test splits: {overlap}"

    # Partition completeness
    assert len(train_df) + len(test_df) == len(df)

    # Test with datetime.date objects (PySpark to_date().toPandas() compatibility)
    df_date_objs = df.copy()
    df_date_objs["game_date"] = pd.to_datetime(df_date_objs["game_date"]).dt.date
    train_df_dt, test_df_dt = temporal_train_test_split(df_date_objs, split_date="2024-04-02")
    assert len(train_df_dt) == len(train_df)
    assert len(test_df_dt) == len(test_df)



def test_target_classes_consistency_and_validation():
    """
    Verify validation of pitch_category classes:
    - Valid classes: 'Fastball', 'Breaking Ball', 'Offspeed'.
    - If unclassified rows or invalid labels appear, validate_target_data must raise ValueError.
    - If any of the 3 target classes is completely missing from training dataset, it must warn or raise.
    """
    from pitch_predictor.training import (
        VALID_PITCH_CATEGORIES,
        validate_target_data,
    )

    assert set(VALID_PITCH_CATEGORIES) == {"Fastball", "Breaking Ball", "Offspeed"}

    # Valid DataFrame passes validation
    valid_df = create_sample_silver_df()
    assert validate_target_data(valid_df) is True

    # DataFrame with invalid/non-tactical category raises ValueError
    invalid_category_df = valid_df.copy()
    invalid_category_df.loc[0, "pitch_category"] = "Knuckleball"
    with pytest.raises(ValueError, match="Invalid pitch category"):
        validate_target_data(invalid_category_df)

    # DataFrame with null target raises ValueError
    null_target_df = valid_df.copy()
    null_target_df.loc[0, "pitch_category"] = None
    with pytest.raises(ValueError, match="Null values found"):
        validate_target_data(null_target_df)

    # DataFrame missing a class raises ValueError (multiclass training requires all target categories)
    missing_class_df = valid_df[valid_df["pitch_category"] != "Offspeed"].copy()
    with pytest.raises(ValueError, match="Missing required pitch categories"):
        validate_target_data(missing_class_df)


def test_feature_type_partitioning():
    """
    Verify feature type classification:
    - CATEGORICAL_FEATURE_COLS and NUMERIC_FEATURE_COLS are disjoint.
    - Union of CATEGORICAL_FEATURE_COLS and NUMERIC_FEATURE_COLS exactly equals TRAINING_FEATURE_COLS.
    """
    from pitch_predictor.training import (
        CATEGORICAL_FEATURE_COLS,
        NUMERIC_FEATURE_COLS,
        TRAINING_FEATURE_COLS,
    )

    cat_set = set(CATEGORICAL_FEATURE_COLS)
    num_set = set(NUMERIC_FEATURE_COLS)
    train_set = set(TRAINING_FEATURE_COLS)

    assert cat_set.intersection(num_set) == set(), "Overlap between categorical and numeric features"
    assert cat_set.union(num_set) == train_set, "Features partition does not match TRAINING_FEATURE_COLS"


def test_compute_classification_metrics():
    """
    Verify compute_classification_metrics accurately computes:
    - f1_macro, f1_weighted, accuracy, balanced_accuracy, log_loss
    - Per-class F1 metrics
    """
    import numpy as np

    from pitch_predictor.training import (
        VALID_PITCH_CATEGORIES,
        compute_classification_metrics,
    )

    y_true = np.array(["Fastball", "Breaking Ball", "Offspeed", "Fastball", "Breaking Ball"])
    y_pred = np.array(["Fastball", "Breaking Ball", "Offspeed", "Breaking Ball", "Breaking Ball"])
    
    # Probabilities for 3 classes: [Breaking Ball, Fastball, Offspeed] (alphabetical)
    classes = sorted(VALID_PITCH_CATEGORIES)
    y_prob = np.array([
        [0.1, 0.8, 0.1],  # True: Fastball -> Correct
        [0.8, 0.1, 0.1],  # True: Breaking Ball -> Correct
        [0.1, 0.1, 0.8],  # True: Offspeed -> Correct
        [0.6, 0.3, 0.1],  # True: Fastball -> Predicted Breaking Ball
        [0.7, 0.2, 0.1],  # True: Breaking Ball -> Correct
    ])

    metrics = compute_classification_metrics(y_true, y_pred, y_prob, classes=classes)

    assert "f1_macro" in metrics
    assert "f1_weighted" in metrics
    assert "accuracy" in metrics
    assert "balanced_accuracy" in metrics
    assert "log_loss" in metrics
    assert metrics["accuracy"] == 4 / 5
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert "f1_Fastball" in metrics
    assert "f1_Breaking_Ball" in metrics
    assert "f1_Offspeed" in metrics


def test_build_and_train_trial_pipeline():
    """
    Verify that build_model_pipeline creates an sklearn Pipeline that successfully trains
    and predicts on Silver feature data without throwing errors or leaking data.
    """
    from pitch_predictor.training import (
        TARGET_COL,
        TRAINING_FEATURE_COLS,
        build_model_pipeline,
    )

    df = create_sample_silver_df()
    X = df[TRAINING_FEATURE_COLS]
    y = df[TARGET_COL]

    pipeline = build_model_pipeline(model_type="hist_gbdt")
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)

    assert len(preds) == len(df)
    assert probs.shape == (len(df), 3)
    assert set(preds).issubset(set(df[TARGET_COL].unique()))


def test_compute_metrics_diff_and_impact_report():
    """
    Verify computing metric deltas between successive training runs and formatting
    a markdown impact comparison report.
    """
    from pitch_predictor.training import (
        compute_metrics_diff,
        generate_metrics_impact_report,
    )

    prev_metrics = {
        "f1_macro": 0.5047,
        "balanced_accuracy": 0.5473,
        "accuracy": 0.5247,
        "f1_Fastball": 0.5778,
        "f1_Breaking_Ball": 0.5095,
        "f1_Offspeed": 0.4267,
        "log_loss": 0.9105,
    }

    curr_metrics = {
        "f1_macro": 0.5013,
        "balanced_accuracy": 0.5745,
        "accuracy": 0.5100,
        "f1_Fastball": 0.5351,
        "f1_Breaking_Ball": 0.5188,
        "f1_Offspeed": 0.4500,
        "log_loss": 0.9163,
    }

    diffs = compute_metrics_diff(prev_metrics, curr_metrics)

    assert "f1_macro" in diffs
    assert pytest.approx(diffs["balanced_accuracy"]["delta"], 0.0001) == 0.0272
    assert pytest.approx(diffs["f1_Offspeed"]["delta"], 0.0001) == 0.0233
    assert diffs["f1_Offspeed"]["pct_change"] > 5.0

    report = generate_metrics_impact_report(
        current_version=2,
        previous_version=1,
        current_metrics=curr_metrics,
        previous_metrics=prev_metrics,
        current_rows=108424,
        previous_rows=25907,
    )

    assert "Relatório Diário de Impacto no Modelo" in report
    assert "Version 2" in report
    assert "Version 1" in report
    assert "108,424" in report
    assert "25,907" in report
    assert "Balanced Accuracy" in report
    assert "0.5745" in report


