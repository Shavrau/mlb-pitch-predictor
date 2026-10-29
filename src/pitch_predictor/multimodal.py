"""
Multimodal Fusion Module for MLB Pitch Predictor.

Fuses pre-pitch contextual Statcast features with pre-release biomechanical pose
features (5 core + optional 10 extended 3D/temporal features) to build a unified
multimodal pitch classification pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# Join Keys
MULTIMODAL_JOIN_KEYS: list[str] = ["game_pk", "at_bat_number", "pitch_number"]

# Target & Pitch Categories
TARGET_COL: str = "pitch_category"
PITCH_CATEGORIES: list[str] = ["Fastball", "Breaking Ball", "Offspeed"]

# Statcast Contextual Features
STATCAST_CATEGORICAL_COLS: list[str] = [
    "count_state",
    "count_leverage",
    "prev_pitch_category",
]

STATCAST_NUMERIC_COLS: list[str] = [
    "balls",
    "strikes",
    "runners_on_base",
    "scoring_position",
    "same_hand",
    "pitcher_fb_prior",
    "pitcher_bb_prior",
    "pitcher_os_prior",
]

STATCAST_FEATURE_COLS: list[str] = STATCAST_CATEGORICAL_COLS + STATCAST_NUMERIC_COLS

# Biomechanical Pose Features (Core 5 + Extended 10)
BIOMECHANICAL_FEATURE_COLS: list[str] = [
    "arm_slot_angle",
    "release_height",
    "elbow_flexion_angle",
    "trunk_tilt",
    "stride_extension",
]

EXTENDED_BIOMECHANICAL_FEATURE_COLS: list[str] = BIOMECHANICAL_FEATURE_COLS + [
    "glove_tuck_distance",
    "hip_shoulder_separation",
    "arm_angular_velocity",
    "delivery_tempo_s",
    "forward_release_extension",
]

# Combined Multimodal Feature Set (Core contract used by MLflow Serving PyFunc)
MULTIMODAL_FEATURE_COLS: list[str] = STATCAST_FEATURE_COLS + BIOMECHANICAL_FEATURE_COLS
EXTENDED_MULTIMODAL_FEATURE_COLS: list[str] = STATCAST_FEATURE_COLS + EXTENDED_BIOMECHANICAL_FEATURE_COLS

# League / Pitcher Fallback Medians for Missing Biomechanics
DEFAULT_BIOMECHANICS_IMPUTATION: dict[str, float] = {
    "arm_slot_angle": 48.0,
    "release_height": 0.70,
    "elbow_flexion_angle": 135.0,
    "trunk_tilt": 11.5,
    "stride_extension": 1.40,
    "glove_tuck_distance": 0.25,
    "hip_shoulder_separation": 38.0,
    "arm_angular_velocity": 760.0,
    "delivery_tempo_s": 0.57,
    "forward_release_extension": 0.61,
}


def create_multimodal_dataset(
    statcast_df: pd.DataFrame,
    biomechanics_df: pd.DataFrame,
    how: str = "inner",
    fill_missing_biomechanics: bool = True,
) -> pd.DataFrame:
    """
    Join Statcast and Biomechanical dataframes on composite primary key,
    preserving both core and extended 3D/temporal biomechanical columns.
    """
    missing_statcast_keys = [k for k in MULTIMODAL_JOIN_KEYS if k not in statcast_df.columns]
    if missing_statcast_keys:
        raise ValueError(f"Statcast DataFrame missing join keys: {missing_statcast_keys}")

    missing_bio_keys = [k for k in MULTIMODAL_JOIN_KEYS if k not in biomechanics_df.columns]
    if missing_bio_keys:
        raise ValueError(f"Biomechanics DataFrame missing join keys: {missing_bio_keys}")

    bio_cols_to_keep = list(MULTIMODAL_JOIN_KEYS) + [
        c for c in EXTENDED_BIOMECHANICAL_FEATURE_COLS if c in biomechanics_df.columns
    ]
    bio_clean = biomechanics_df[bio_cols_to_keep].drop_duplicates(subset=MULTIMODAL_JOIN_KEYS)

    merged = pd.merge(statcast_df, bio_clean, on=MULTIMODAL_JOIN_KEYS, how=how)

    if how == "left":
        merged["has_biomechanics"] = (
            merged["elbow_flexion_angle"].notna().astype(int)
            if "elbow_flexion_angle" in merged.columns
            else 0
        )
        if fill_missing_biomechanics:
            for col, val in DEFAULT_BIOMECHANICS_IMPUTATION.items():
                if col in merged.columns:
                    merged[col] = merged[col].fillna(val)

    return merged.reset_index(drop=True)


def generate_synthetic_multimodal_data(
    n_samples_per_class: int = 40,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic paired Statcast and Biomechanics datasets for testing and calibration.
    """
    rng = np.random.default_rng(random_state)
    records_statcast = []
    records_bio = []

    configs = {
        "Fastball": {
            "pitch_type": "FF",
            "balls": [0, 1, 2, 3],
            "strikes": [0, 1, 2],
            "count_leverage": "Pitcher Behind",
            "prev_pitch": "Breaking Ball",
            "priors": (0.60, 0.25, 0.15),
            "arm_slot": (50.0, 3.5),
            "release_h": (0.71, 0.02),
            "elbow": (146.0, 4.0),
            "trunk": (12.0, 1.5),
            "stride": (1.45, 0.06),
            "glove_tuck": (0.21, 0.02),
            "hip_sep": (42.0, 3.0),
            "arm_vel": (830.0, 40.0),
            "tempo": (0.55, 0.03),
            "fwd_ext": (0.64, 0.03),
        },
        "Breaking Ball": {
            "pitch_type": "ST",
            "balls": [0, 1, 1],
            "strikes": [2, 2, 1],
            "count_leverage": "Pitcher Ahead",
            "prev_pitch": "Fastball",
            "priors": (0.35, 0.50, 0.15),
            "arm_slot": (12.0, 5.0),
            "release_h": (0.68, 0.02),
            "elbow": (98.0, 4.0),
            "trunk": (10.0, 1.5),
            "stride": (1.35, 0.06),
            "glove_tuck": (0.33, 0.025),
            "hip_sep": (33.0, 3.0),
            "arm_vel": (740.0, 40.0),
            "tempo": (0.58, 0.03),
            "fwd_ext": (0.57, 0.03),
        },
        "Offspeed": {
            "pitch_type": "FS",
            "balls": [1, 2, 0],
            "strikes": [1, 2, 2],
            "count_leverage": "Neutral",
            "prev_pitch": "Fastball",
            "priors": (0.35, 0.25, 0.40),
            "arm_slot": (74.0, 4.0),
            "release_h": (0.65, 0.02),
            "elbow": (105.0, 4.0),
            "trunk": (12.0, 1.5),
            "stride": (1.40, 0.06),
            "glove_tuck": (0.26, 0.02),
            "hip_sep": (38.0, 3.0),
            "arm_vel": (685.0, 40.0),
            "tempo": (0.60, 0.03),
            "fwd_ext": (0.60, 0.03),
        },
    }

    game_pk = 745000
    at_bat_num = 1
    pitch_num = 1

    for category, cfg in configs.items():
        for _ in range(n_samples_per_class):
            b = int(rng.choice(cfg["balls"]))
            s = int(rng.choice(cfg["strikes"]))
            count_str = f"{b}-{s}"
            fb_p, bb_p, os_p = cfg["priors"]

            statcast_record = {
                "game_pk": game_pk,
                "at_bat_number": at_bat_num,
                "pitch_number": pitch_num,
                "pitcher": 694973,
                "game_date": "2024-04-01",
                "pitch_category": category,
                "pitch_type": cfg["pitch_type"],
                "balls": b,
                "strikes": s,
                "count_state": count_str,
                "count_leverage": cfg["count_leverage"],
                "prev_pitch_category": cfg["prev_pitch"],
                "runners_on_base": int(rng.choice([0, 1])),
                "scoring_position": int(rng.choice([0, 1])),
                "same_hand": 1,
                "pitcher_fb_prior": float(rng.normal(fb_p, 0.03)),
                "pitcher_bb_prior": float(rng.normal(bb_p, 0.03)),
                "pitcher_os_prior": float(rng.normal(os_p, 0.03)),
            }
            records_statcast.append(statcast_record)

            bio_record = {
                "game_pk": game_pk,
                "at_bat_number": at_bat_num,
                "pitch_number": pitch_num,
                "arm_slot_angle": float(rng.normal(*cfg["arm_slot"])),
                "release_height": float(rng.normal(*cfg["release_h"])),
                "elbow_flexion_angle": float(rng.normal(*cfg["elbow"])),
                "trunk_tilt": float(rng.normal(*cfg["trunk"])),
                "stride_extension": float(rng.normal(*cfg["stride"])),
                "glove_tuck_distance": float(rng.normal(*cfg["glove_tuck"])),
                "hip_shoulder_separation": float(rng.normal(*cfg["hip_sep"])),
                "arm_angular_velocity": float(rng.normal(*cfg["arm_vel"])),
                "delivery_tempo_s": float(rng.normal(*cfg["tempo"])),
                "forward_release_extension": float(rng.normal(*cfg["fwd_ext"])),
            }
            records_bio.append(bio_record)

            pitch_num += 1
            if pitch_num > 5:
                pitch_num = 1
                at_bat_num += 1
            if at_bat_num > 20:
                at_bat_num = 1
                game_pk += 1

    df_statcast = pd.DataFrame(records_statcast)
    df_bio = pd.DataFrame(records_bio)
    df_full = create_multimodal_dataset(df_statcast, df_bio, how="inner")

    return df_statcast, df_bio, df_full


class MultimodalPitchClassifier:
    """
    Multimodal Pitch Predictor fusing Statcast game context with release biomechanics.
    Supports both the 5-feature MLflow Serving contract (default) and the 10-feature
    extended 3D/temporal biomechanical feature set (`use_extended_biomechanics=True`).
    """

    def __init__(
        self,
        n_estimators: int = 40,
        max_depth: int = 8,
        random_state: int = 42,
        use_extended_biomechanics: bool = False,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.use_extended_biomechanics = use_extended_biomechanics
        self.classes_: list[str] = PITCH_CATEGORIES

        self.bio_feature_cols_: list[str] = (
            list(EXTENDED_BIOMECHANICAL_FEATURE_COLS)
            if use_extended_biomechanics
            else list(BIOMECHANICAL_FEATURE_COLS)
        )
        self.multimodal_feature_cols_: list[str] = STATCAST_FEATURE_COLS + self.bio_feature_cols_

        self.statcast_pipeline: Pipeline | None = None
        self.vision_pipeline: Pipeline | None = None
        self.multimodal_pipeline: Pipeline | None = None
        self.is_fitted: bool = False

    def _active_bio_cols(self) -> list[str]:
        return getattr(self, "bio_feature_cols_", BIOMECHANICAL_FEATURE_COLS)

    def _active_multi_cols(self) -> list[str]:
        return getattr(self, "multimodal_feature_cols_", MULTIMODAL_FEATURE_COLS)

    def _build_pipelines(self) -> None:
        """Construct scikit-learn pipelines with appropriate transformers."""
        bio_cols = self._active_bio_cols()

        statcast_transformer = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    STATCAST_CATEGORICAL_COLS,
                ),
                ("num", StandardScaler(), STATCAST_NUMERIC_COLS),
            ]
        )
        self.statcast_pipeline = Pipeline(
            steps=[
                ("preprocessor", statcast_transformer),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        vision_transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), bio_cols),
            ]
        )
        self.vision_pipeline = Pipeline(
            steps=[
                ("preprocessor", vision_transformer),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        multimodal_transformer = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    STATCAST_CATEGORICAL_COLS,
                ),
                ("num_statcast", StandardScaler(), STATCAST_NUMERIC_COLS),
                ("num_bio", StandardScaler(), bio_cols),
            ]
        )
        self.multimodal_pipeline = Pipeline(
            steps=[
                ("preprocessor", multimodal_transformer),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def fit(self, df: pd.DataFrame, target_col: str = TARGET_COL) -> MultimodalPitchClassifier:
        """Fit all three internal pipelines (Statcast, Vision, and Multimodal) on a joined DataFrame."""
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

        self.bio_feature_cols_ = (
            list(EXTENDED_BIOMECHANICAL_FEATURE_COLS)
            if self.use_extended_biomechanics
            else list(BIOMECHANICAL_FEATURE_COLS)
        )
        self.multimodal_feature_cols_ = STATCAST_FEATURE_COLS + self.bio_feature_cols_
        self._build_pipelines()
        y = df[target_col].values

        df_clean = self._prepare_eval_df(df)

        assert self.statcast_pipeline is not None
        assert self.vision_pipeline is not None
        assert self.multimodal_pipeline is not None

        self.statcast_pipeline.fit(df_clean[STATCAST_FEATURE_COLS], y)
        self.vision_pipeline.fit(df_clean[self._active_bio_cols()], y)
        self.multimodal_pipeline.fit(df_clean[self._active_multi_cols()], y)

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict pitch category using the fused Multimodal pipeline."""
        if not self.is_fitted or self.multimodal_pipeline is None:
            raise ValueError("Model is not fitted. Call fit() before predict().")

        X_eval = self._prepare_eval_df(X)
        return self.multimodal_pipeline.predict(X_eval[self._active_multi_cols()])

    def predict_proba(self, X: pd.DataFrame) -> list[dict[str, float]]:
        """Predict pitch category probabilities using the fused Multimodal pipeline."""
        if not self.is_fitted or self.multimodal_pipeline is None:
            raise ValueError("Model is not fitted. Call fit() before predict_proba().")

        X_eval = self._prepare_eval_df(X)
        probs_matrix = self.multimodal_pipeline.predict_proba(X_eval[self._active_multi_cols()])
        classes = self.multimodal_pipeline.classes_

        results = []
        for row in probs_matrix:
            probs_dict = {cls: float(p) for cls, p in zip(classes, row)}
            results.append(probs_dict)
        return results

    def fit_ablation_study(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str = TARGET_COL,
    ) -> dict[str, Any]:
        """Fit and evaluate all three modalities side-by-side to compute performance lift."""
        self.fit(train_df, target_col=target_col)
        y_test = test_df[target_col].values

        test_clean = self._prepare_eval_df(test_df)
        bio_cols = self._active_bio_cols()
        multi_cols = self._active_multi_cols()

        assert self.statcast_pipeline is not None
        assert self.vision_pipeline is not None
        assert self.multimodal_pipeline is not None

        pred_sc = self.statcast_pipeline.predict(test_clean[STATCAST_FEATURE_COLS])
        prob_sc = self.statcast_pipeline.predict_proba(test_clean[STATCAST_FEATURE_COLS])

        pred_vis = self.vision_pipeline.predict(test_clean[bio_cols])
        prob_vis = self.vision_pipeline.predict_proba(test_clean[bio_cols])

        pred_multi = self.multimodal_pipeline.predict(test_clean[multi_cols])
        prob_multi = self.multimodal_pipeline.predict_proba(test_clean[multi_cols])

        def _metrics(y_true, y_pred, y_prob, classes):
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "log_loss": float(log_loss(y_true, y_prob, labels=classes)),
            }

        sc_metrics = _metrics(y_test, pred_sc, prob_sc, self.statcast_pipeline.classes_)
        vis_metrics = _metrics(y_test, pred_vis, prob_vis, self.vision_pipeline.classes_)
        multi_metrics = _metrics(y_test, pred_multi, prob_multi, self.multimodal_pipeline.classes_)

        lift = {
            "lift_accuracy": multi_metrics["accuracy"] - sc_metrics["accuracy"],
            "lift_f1_macro": multi_metrics["f1_macro"] - sc_metrics["f1_macro"],
            "lift_log_loss": sc_metrics["log_loss"] - multi_metrics["log_loss"],
        }

        return {
            "statcast": sc_metrics,
            "vision": vis_metrics,
            "multimodal": multi_metrics,
            "lift": lift,
        }

    def compare_modalities(self, row_or_df: pd.Series | pd.DataFrame | dict) -> dict[str, Any]:
        """Compare predictions across Statcast, Vision, and Multimodal for a single pitch."""
        if not self.is_fitted or self.multimodal_pipeline is None:
            raise ValueError("Model is not fitted. Call fit() before compare_modalities().")

        if isinstance(row_or_df, (pd.Series, dict)):
            df = pd.DataFrame([row_or_df])
        else:
            df = row_or_df.iloc[[0]].copy()

        df_eval = self._prepare_eval_df(df)
        bio_cols = self._active_bio_cols()
        multi_cols = self._active_multi_cols()

        assert self.statcast_pipeline is not None
        assert self.vision_pipeline is not None
        assert self.multimodal_pipeline is not None

        sc_pred = str(self.statcast_pipeline.predict(df_eval[STATCAST_FEATURE_COLS])[0])
        sc_probs = {
            cls: float(p)
            for cls, p in zip(
                self.statcast_pipeline.classes_,
                self.statcast_pipeline.predict_proba(df_eval[STATCAST_FEATURE_COLS])[0],
            )
        }

        vis_pred = str(self.vision_pipeline.predict(df_eval[bio_cols])[0])
        vis_probs = {
            cls: float(p)
            for cls, p in zip(
                self.vision_pipeline.classes_,
                self.vision_pipeline.predict_proba(df_eval[bio_cols])[0],
            )
        }

        multi_pred = str(self.multimodal_pipeline.predict(df_eval[multi_cols])[0])
        multi_probs = {
            cls: float(p)
            for cls, p in zip(
                self.multimodal_pipeline.classes_,
                self.multimodal_pipeline.predict_proba(df_eval[multi_cols])[0],
            )
        }

        if sc_pred == vis_pred:
            alignment = "AGREEMENT"
            summary = (
                f"**Alinhamento Tático & Biomecânico**: O contexto situacional e os sinais físicos "
                f"do arremessador concordam plenamente em **{multi_pred}** ({multi_probs.get(multi_pred, 0):.1%} de confiança)."
            )
        elif multi_pred == vis_pred:
            alignment = "TIPPING_OVERRIDE"
            summary = (
                f"**Pitch Tipping Detectado**: Embora a contagem e os dados contextuais apontassem para "
                f"**{sc_pred}**, a postura física na fase de soltura denunciou **{multi_pred}** "
                f"({vis_probs.get(multi_pred, 0):.1%} na visão). A fusão multimodal priorizou o sinal biomecânico."
            )
        else:
            alignment = "CONTEXT_REINFORCED"
            summary = (
                f"**Contexto Tático Prevalece**: A mecânica corporal sugeriu {vis_pred}, mas a alavancagem "
                f"da contagem e as tendências históricas reforçaram a seleção de **{multi_pred}**."
            )

        return {
            "statcast_prediction": sc_pred,
            "statcast_probabilities": sc_probs,
            "vision_prediction": vis_pred,
            "vision_probabilities": vis_probs,
            "multimodal_prediction": multi_pred,
            "multimodal_probabilities": multi_probs,
            "alignment_status": alignment,
            "summary_text": summary,
        }

    def _prepare_eval_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required features exist and are imputed cleanly."""
        clean = df.copy()
        for col in STATCAST_FEATURE_COLS + self._active_bio_cols():
            if col not in clean.columns:
                clean[col] = DEFAULT_BIOMECHANICS_IMPUTATION.get(col, 0.0)
            elif col in DEFAULT_BIOMECHANICS_IMPUTATION and clean[col].isna().any():
                clean[col] = clean[col].fillna(DEFAULT_BIOMECHANICS_IMPUTATION[col])
        return clean

    def save(self, filepath: str | Path) -> None:
        """Serialize trained model to joblib."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model.")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved MultimodalPitchClassifier to %s", path)

    @classmethod
    def load(cls, filepath: str | Path) -> MultimodalPitchClassifier:
        """Load serialized model from joblib."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is not an instance of {cls.__name__}")
        return model
