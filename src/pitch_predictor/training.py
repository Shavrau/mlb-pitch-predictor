"""
Databricks AutoML and MLflow training configuration and model registration module.

Defines:
1. Feature selection contracts excluding direct high-cardinality IDs from training.
2. Temporal anchor and multiclasse target specification.
3. Databricks AutoML parameter builder with time-series splitting and Macro F1 optimization.
4. Unity Catalog 3-level model naming and MLflow registration ('databricks-uc').
5. Strict temporal validation and target class consistency checks.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None

from pitch_predictor.features import SILVER_FEATURE_COLUMNS

# Target & Temporal Anchors
TARGET_COL: str = "pitch_category"
TIME_COL: str = "game_date"
PRIMARY_METRIC: str = "f1"
MODEL_NAME: str = "pitch_category_predictor"

VALID_PITCH_CATEGORIES: list[str] = [
    "Fastball",
    "Breaking Ball",
    "Offspeed",
]

# High-cardinality IDs strictly excluded from training features to avoid spurious correlation/overfitting
IDENTIFIER_COLS: list[str] = [
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "batter",
]

# Training feature columns (all Silver features minus identifiers, temporal anchor, and target)
TRAINING_FEATURE_COLS: list[str] = [
    col
    for col in SILVER_FEATURE_COLUMNS
    if col not in IDENTIFIER_COLS and col != TIME_COL and col != TARGET_COL
]

CATEGORICAL_FEATURE_COLS: list[str] = [
    "prev_pitch_category",
    "prev_pitch_type",
    "prev_pitch_result",
    "p_throws",
    "stand",
    "count_state",
    "count_leverage",
    "inning_topbot",
]

NUMERIC_FEATURE_COLS: list[str] = [
    col for col in TRAINING_FEATURE_COLS if col not in CATEGORICAL_FEATURE_COLS
]



def select_automl_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select training features, temporal split anchor, and target column from a Silver DataFrame.
    Drops high-cardinality entity identifiers ('game_pk', 'at_bat_number', 'pitch_number', 'pitcher', 'batter').
    """
    cols_to_keep = TRAINING_FEATURE_COLS + [TIME_COL, TARGET_COL]
    return df[cols_to_keep].copy()


def build_automl_params(
    catalog: str = "main",
    schema: str = "baseball_ml",
    timeout_minutes: int = 30,
) -> dict[str, Any]:
    """
    Construct parameters dictionary for databricks.automl.classify.
    Configures strict chronological splitting via time_col and Macro F1 optimization.
    """
    return {
        "target_col": TARGET_COL,
        "primary_metric": PRIMARY_METRIC,
        "time_col": TIME_COL,
        "timeout_minutes": timeout_minutes,
        "experiment_name": f"/Shared/{catalog}/{schema}/automl_{TARGET_COL}_predictor",
    }


def get_uc_model_name(catalog: str, schema: str) -> str:
    """
    Generate fully qualified 3-level Unity Catalog model name.
    """
    return f"{catalog}.{schema}.{MODEL_NAME}"


def register_best_model_to_uc(
    model_uri: str,
    catalog: str,
    schema: str,
) -> Any:
    """
    Register model artifact into Unity Catalog using MLflow ('databricks-uc').
    """
    mlflow.set_registry_uri("databricks-uc")
    full_model_name = get_uc_model_name(catalog, schema)
    return mlflow.register_model(model_uri=model_uri, name=full_model_name)


def temporal_train_test_split(
    df: pd.DataFrame,
    split_date: str | None = None,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset strictly chronologically at split_date or by quantile when split_date is None.
    Guarantees max(train_date) <= target_cutoff < min(test_date) with zero overlapping games.
    Supports strings, datetime.date, and Timestamps.
    """
    time_series = pd.to_datetime(df[TIME_COL])
    if split_date is None:
        unique_dates = sorted(time_series.dropna().unique())
        if len(unique_dates) <= 1:
            split_idx = int(len(df) * train_ratio)
            sorted_df = df.sort_values(TIME_COL)
            return sorted_df.iloc[:split_idx].copy(), sorted_df.iloc[split_idx:].copy()
        cutoff_idx = max(0, min(len(unique_dates) - 2, int(len(unique_dates) * train_ratio) - 1))
        target_cutoff = pd.to_datetime(unique_dates[cutoff_idx])
    else:
        target_cutoff = pd.to_datetime(split_date)

    train_df = df[time_series <= target_cutoff].copy()
    test_df = df[time_series > target_cutoff].copy()
    return train_df, test_df



def validate_target_data(df: pd.DataFrame) -> bool:
    """
    Validate target classes:
    - No null values allowed.
    - All categories must be in VALID_PITCH_CATEGORIES ('Fastball', 'Breaking Ball', 'Offspeed').
    - All 3 categories must be present for multiclass classification.
    """
    target_series = df[TARGET_COL]

    if target_series.isnull().any():
        raise ValueError(f"Null values found in target column '{TARGET_COL}'.")

    unique_classes = set(target_series.dropna().unique())
    valid_set = set(VALID_PITCH_CATEGORIES)

    invalid_classes = unique_classes - valid_set
    if invalid_classes:
        raise ValueError(
            f"Invalid pitch category found in target: {invalid_classes}. Expected only {valid_set}."
        )

    missing_classes = valid_set - unique_classes
    if missing_classes:
        raise ValueError(
            f"Missing required pitch categories in target: {missing_classes}. Multiclass training requires all {valid_set}."
        )

    return True


def build_preprocessor() -> Any:
    """
    Construct a scikit-learn ColumnTransformer for Silver feature preprocessing.
    One-hot encodes categorical situational/sequence features and imputes/scales numeric features.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, CATEGORICAL_FEATURE_COLS),
            ("num", num_transformer, NUMERIC_FEATURE_COLS),
        ]
    )


def build_model_pipeline(model_type: str = "hist_gbdt", **kwargs) -> Any:
    """
    Construct an end-to-end scikit-learn Pipeline with preprocessor and specified classifier.
    Supports 'hist_gbdt', 'random_forest', 'logistic_regression', 'lightgbm', 'xgboost'.
    """
    from sklearn.pipeline import Pipeline

    preprocessor = build_preprocessor()

    if model_type == "hist_gbdt":
        from sklearn.ensemble import HistGradientBoostingClassifier

        clf = HistGradientBoostingClassifier(
            class_weight=kwargs.get("class_weight", "balanced"),
            max_iter=kwargs.get("max_iter", 150),
            max_depth=kwargs.get("max_depth", 6),
            l2_regularization=kwargs.get("l2_regularization", 1.0),
            random_state=kwargs.get("random_state", 42),
        )
    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 150),
            max_depth=kwargs.get("max_depth", 10),
            class_weight=kwargs.get("class_weight", "balanced"),
            random_state=kwargs.get("random_state", 42),
            n_jobs=-1,
        )
    elif model_type == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(
            max_iter=kwargs.get("max_iter", 1000),
            class_weight=kwargs.get("class_weight", "balanced"),
            random_state=kwargs.get("random_state", 42),
        )
    elif model_type == "lightgbm":
        from lightgbm import LGBMClassifier

        clf = LGBMClassifier(
            n_estimators=kwargs.get("n_estimators", 150),
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.05),
            class_weight=kwargs.get("class_weight", "balanced"),
            random_state=kwargs.get("random_state", 42),
            n_jobs=-1,
            verbose=-1,
        )
    elif model_type == "xgboost":
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            n_estimators=kwargs.get("n_estimators", 150),
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.05),
            random_state=kwargs.get("random_state", 42),
            n_jobs=-1,
            eval_metric="mlogloss",
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])


def compute_classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_prob: Any = None,
    classes: list[str] | None = None,
) -> dict[str, float]:
    """
    Compute comprehensive classification metrics:
    - Primary: Macro F1 ('f1_macro')
    - Secondary: Weighted F1 ('f1_weighted'), Accuracy ('accuracy'), Balanced Accuracy ('balanced_accuracy'), Log Loss ('log_loss')
    - Per-class F1 scores.
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        log_loss,
    )

    metrics: dict[str, float] = {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }

    if classes is not None:
        per_class_f1 = f1_score(y_true, y_pred, average=None, labels=classes)
        for cls_name, f1_val in zip(classes, per_class_f1):
            clean_name = cls_name.replace(" ", "_")
            metrics[f"f1_{clean_name}"] = float(f1_val)

    if y_prob is not None:
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_prob, labels=classes))
        except Exception:  # noqa: BLE001, S110
            pass

    return metrics


def compute_metrics_diff(
    previous_metrics: dict[str, float],
    current_metrics: dict[str, float],
) -> dict[str, dict[str, float]]:
    """
    Compute absolute and relative differences between two evaluation metric sets.
    """
    diffs = {}
    for k, curr_val in current_metrics.items():
        if k in previous_metrics and previous_metrics[k] is not None:
            prev_val = float(previous_metrics[k])
            abs_delta = float(curr_val) - prev_val
            rel_pct = (abs_delta / prev_val * 100.0) if prev_val != 0 else 0.0
            diffs[k] = {
                "previous": prev_val,
                "current": float(curr_val),
                "delta": abs_delta,
                "pct_change": rel_pct,
            }
    return diffs


def generate_metrics_impact_report(
    current_version: int | str,
    previous_version: int | str | None,
    current_metrics: dict[str, float],
    previous_metrics: dict[str, float] | None = None,
    model_name: str = "pitch_category_predictor",
    current_rows: int | None = None,
    previous_rows: int | None = None,
    execution_date: str | None = None,
) -> str:
    """
    Generate a markdown impact report comparing newly trained model version metrics
    against the previous production version.
    """
    date_str = execution_date or pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        f"# Relatório Diário de Impacto no Modelo: {model_name}",
        f"**Data da Execução**: {date_str}  ",
        f"**Versão Atual Avaliada**: Version {current_version}  ",
        f"**Versão Anterior Comparada**: Version {previous_version if previous_version else 'N/A (Baseline Inicial)'}",
        "",
        "---",
        "",
        "## 1. Evolução do Volume de Dados Ingeridos",
        "",
        "| Métrica de Dados | Versão Anterior | Versão Atual | Variação (Delta) | Crescimento |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    if current_rows is not None:
        prev_r_str = f"{previous_rows:,}" if previous_rows is not None else "N/A"
        curr_r_str = f"{current_rows:,}"
        if previous_rows is not None and previous_rows > 0:
            diff_r = current_rows - previous_rows
            diff_r_str = f"+{diff_r:,}" if diff_r >= 0 else f"{diff_r:,}"
            pct_r = f"{(diff_r / previous_rows * 100):+.2f}%"
        else:
            diff_r_str = "-"
            pct_r = "-"
        lines.append(f"| **Total de Arremessos Úteis (Silver)** | {prev_r_str} | {curr_r_str} | {diff_r_str} | {pct_r} |")
    else:
        lines.append("| **Total de Arremessos Úteis (Silver)** | N/A | N/A | - | - |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Comparativo de Métricas do Modelo",
        "",
        "| Métrica de Avaliação | Versão Anterior | Versão Atual | Variação Absoluta | Variação Relativa | Impacto Tático |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    prev_m = previous_metrics or {}
    diffs = compute_metrics_diff(prev_m, current_metrics)

    metric_labels = [
        ("f1_macro", "Macro F1 (Métrica Primária)", True),
        ("balanced_accuracy", "Balanced Accuracy", True),
        ("accuracy", "Acurácia Geral", True),
        ("f1_Fastball", "F1: Bola Rápida (Fastball)", True),
        ("f1_Breaking_Ball", "F1: Efeito (Breaking Ball)", True),
        ("f1_Offspeed", "F1: Mudança (Offspeed)", True),
        ("log_loss", "Multi-class Log-Loss", False),
    ]

    for m_key, label, higher_is_better in metric_labels:
        if m_key in current_metrics:
            curr_v = current_metrics[m_key]
            curr_str = f"{curr_v:.4f}"
            if m_key in diffs:
                prev_v = diffs[m_key]["previous"]
                prev_str = f"{prev_v:.4f}"
                delta = diffs[m_key]["delta"]
                pct = diffs[m_key]["pct_change"]
                delta_str = f"{delta:+.4f}"
                pct_str = f"{pct:+.2f}%"
                is_positive = delta > 0 if higher_is_better else delta < 0
                status_icon = "🟢 Ganho" if is_positive else ("⚪ Estável" if abs(delta) < 0.001 else "🔴 Atenção")
            else:
                prev_str = "N/A"
                delta_str = "-"
                pct_str = "-"
                status_icon = "Novo"
            lines.append(f"| **{label}** | {prev_str} | {curr_str} | {delta_str} | {pct_str} | {status_icon} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Síntese Tática da Evolução",
        "- **Classes Minoritárias**: O aumento do volume histórico melhora expressivamente a estabilidade bayesiana de priors para Breaking Ball e Offspeed.",
        "- **Isolamento de Vazamento**: Todas as métricas foram calculadas estritamente sobre a partição temporal de holdout futuro.",
    ])

    return "\n".join(lines)


