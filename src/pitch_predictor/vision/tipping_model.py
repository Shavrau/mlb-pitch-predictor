"""
Pitch Tipping Classifier v2 — Intra-Pitcher Baseline Normalization,
Honest GroupKFold Validation, Probability Calibration & SHAP Explainability.

Trains an interpretable biomechanical classifier on 3D & temporal skeletal features
normalized relative to each pitcher's baseline delivery (Δ and Z-Score), preventing
pitcher-identity confounding across overhand RHPs and sidearm LHPs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, train_test_split

logger = logging.getLogger(__name__)

# Core 5 biomechanical features (preserved for 100% backwards compatibility)
BIOMECHANICAL_FEATURE_COLS: list[str] = [
    "arm_slot_angle",
    "release_height",
    "elbow_flexion_angle",
    "trunk_tilt",
    "stride_extension",
]

# Full 10-feature MLB Pitch Tipping vector (v2)
EXTENDED_BIOMECHANICAL_FEATURE_COLS: list[str] = BIOMECHANICAL_FEATURE_COLS + [
    "glove_tuck_distance",
    "hip_shoulder_separation",
    "arm_angular_velocity",
    "delivery_tempo_s",
    "forward_release_extension",
]

# Friendly labels and units for SHAP explanations
FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "arm_slot_angle": "Ângulo do Braço (Arm Slot)",
    "release_height": "Altura de Soltura (Release Point)",
    "elbow_flexion_angle": "Flexão do Cotovelo (Elbow Flexion)",
    "trunk_tilt": "Inclinação do Tronco (Trunk Tilt)",
    "stride_extension": "Extensão da Passada (Stride)",
    "glove_tuck_distance": "Recolhimento da Luva (Glove Tuck)",
    "hip_shoulder_separation": "Separação Quadril-Ombro (Torção)",
    "arm_angular_velocity": "Velocidade Angular do Braço",
    "delivery_tempo_s": "Cadência do Movimento (Tempo)",
    "forward_release_extension": "Extensão Frontal na Soltura",
}

FEATURE_UNITS: dict[str, str] = {
    "arm_slot_angle": "°",
    "release_height": " norm",
    "elbow_flexion_angle": "°",
    "trunk_tilt": "°",
    "stride_extension": "x",
    "glove_tuck_distance": " m",
    "hip_shoulder_separation": "°",
    "arm_angular_velocity": "°/s",
    "delivery_tempo_s": " s",
    "forward_release_extension": " m",
}


class PitcherBaselineNormalizer(BaseEstimator, TransformerMixin):
    """
    Leakage-free scikit-learn transformer computing intra-pitcher biomechanical
    deviations (Δ and Z-Score) strictly from training-fold statistics.

    Falls back cleanly to global training population (μ, σ) when `pitcher_id`
    is absent or belongs to an unseen pitcher at inference time.
    """

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        pitcher_col: str = "pitcher_id",
        include_raw: bool = True,
        eps: float = 1e-3,
    ) -> None:
        self.feature_cols = feature_cols
        self.pitcher_col = pitcher_col
        self.include_raw = include_raw
        self.eps = eps

    def fit(self, X: pd.DataFrame, y: Any = None) -> PitcherBaselineNormalizer:
        if self.feature_cols is not None:
            cols = [c for c in self.feature_cols if c in X.columns]
        else:
            cols = [c for c in EXTENDED_BIOMECHANICAL_FEATURE_COLS if c in X.columns]
        if not cols:
            cols = list(BIOMECHANICAL_FEATURE_COLS)

        self.feature_cols_: list[str] = cols
        self.global_means_: dict[str, float] = {}
        self.global_stds_: dict[str, float] = {}

        for col in self.feature_cols_:
            series = pd.to_numeric(X[col], errors="coerce").dropna()
            mean_val = float(series.mean()) if not series.empty else 0.0
            std_val = float(series.std(ddof=0)) if len(series) > 1 else 1.0
            self.global_means_[col] = mean_val
            self.global_stds_[col] = max(std_val, self.eps)

        self.pitcher_stats_: dict[Any, dict[str, tuple[float, float]]] = {}
        if self.pitcher_col in X.columns:
            for pid, grp in X.groupby(self.pitcher_col):
                if pd.isna(pid) or len(grp) < 2:
                    continue
                p_dict: dict[str, tuple[float, float]] = {}
                for col in self.feature_cols_:
                    s = pd.to_numeric(grp[col], errors="coerce").dropna()
                    if len(s) >= 2:
                        p_mean = float(s.mean())
                        p_std = max(float(s.std(ddof=0)), self.eps)
                    elif len(s) == 1:
                        p_mean = float(s.iloc[0])
                        p_std = self.global_stds_[col]
                    else:
                        p_mean = self.global_means_[col]
                        p_std = self.global_stds_[col]
                    p_dict[col] = (p_mean, p_std)
                self.pitcher_stats_[pid] = p_dict

        return self

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        names: list[str] = []
        if self.include_raw:
            names.extend(self.feature_cols_)
        names.extend([f"{c}_delta" for c in self.feature_cols_])
        names.extend([f"{c}_zscore" for c in self.feature_cols_])
        return names

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        has_pid = self.pitcher_col in X.columns
        pids = X[self.pitcher_col].tolist() if has_pid else [None] * len(X)

        raw_data: dict[str, list[float]] = {c: [] for c in self.feature_cols_}
        delta_data: dict[str, list[float]] = {f"{c}_delta": [] for c in self.feature_cols_}
        z_data: dict[str, list[float]] = {f"{c}_zscore": [] for c in self.feature_cols_}

        for row_idx in range(len(X)):
            pid = pids[row_idx]
            p_stats = self.pitcher_stats_.get(pid) if pid is not None else None
            for col in self.feature_cols_:
                mu, sigma = (
                    p_stats[col]
                    if (p_stats is not None and col in p_stats)
                    else (self.global_means_[col], self.global_stds_[col])
                )
                if col in X.columns:
                    raw_val = X[col].iloc[row_idx]
                    val = float(raw_val) if pd.notna(raw_val) else mu
                else:
                    val = mu

                delta = val - mu
                zscore = delta / max(sigma, self.eps)
                raw_data[col].append(val)
                delta_data[f"{col}_delta"].append(delta)
                z_data[f"{col}_zscore"].append(zscore)

        out_cols: dict[str, list[float]] = {}
        if self.include_raw:
            out_cols.update(raw_data)
        out_cols.update(delta_data)
        out_cols.update(z_data)
        return pd.DataFrame(out_cols, index=X.index)


def generate_synthetic_biomechanics(
    n_samples_per_class: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic 10-feature biomechanical pitching dataset
    across multiple pitchers and games (for testing and offline calibration).
    """
    rng = np.random.default_rng(random_state)
    records = []

    configs = {
        "Fastball": {
            "pitch_type": "FF",
            "arm_slot_angle": (50.0, 4.0),
            "release_height": (0.71, 0.02),
            "elbow_flexion_angle": (145.0, 5.0),
            "trunk_tilt": (12.0, 1.8),
            "stride_extension": (1.45, 0.08),
            "glove_tuck_distance": (0.21, 0.025),
            "hip_shoulder_separation": (42.0, 3.5),
            "arm_angular_velocity": (830.0, 45.0),
            "delivery_tempo_s": (0.55, 0.03),
            "forward_release_extension": (0.64, 0.03),
        },
        "Breaking Ball": {
            "pitch_type": "ST",
            "arm_slot_angle": (16.0, 6.5),
            "release_height": (0.67, 0.025),
            "elbow_flexion_angle": (98.0, 5.5),
            "trunk_tilt": (10.2, 1.5),
            "stride_extension": (1.35, 0.08),
            "glove_tuck_distance": (0.33, 0.03),
            "hip_shoulder_separation": (33.0, 3.5),
            "arm_angular_velocity": (745.0, 45.0),
            "delivery_tempo_s": (0.58, 0.035),
            "forward_release_extension": (0.57, 0.03),
        },
        "Offspeed": {
            "pitch_type": "FS",
            "arm_slot_angle": (68.0, 5.5),
            "release_height": (0.65, 0.025),
            "elbow_flexion_angle": (108.0, 6.0),
            "trunk_tilt": (12.8, 1.8),
            "stride_extension": (1.39, 0.08),
            "glove_tuck_distance": (0.26, 0.025),
            "hip_shoulder_separation": (38.0, 3.5),
            "arm_angular_velocity": (690.0, 45.0),
            "delivery_tempo_s": (0.60, 0.035),
            "forward_release_extension": (0.60, 0.03),
        },
    }

    for category, cfg in configs.items():
        pt = cfg["pitch_type"]
        for i in range(n_samples_per_class):
            records.append({
                "game_pk": 745000 + (i % 6),
                "pitcher_id": 694973,
                "pitch_category": category,
                "pitch_type": pt,
                "arm_slot_angle": float(rng.normal(*cfg["arm_slot_angle"])),
                "release_height": float(rng.normal(*cfg["release_height"])),
                "elbow_flexion_angle": float(rng.normal(*cfg["elbow_flexion_angle"])),
                "trunk_tilt": float(rng.normal(*cfg["trunk_tilt"])),
                "stride_extension": float(rng.normal(*cfg["stride_extension"])),
                "glove_tuck_distance": float(rng.normal(*cfg["glove_tuck_distance"])),
                "hip_shoulder_separation": float(rng.normal(*cfg["hip_shoulder_separation"])),
                "arm_angular_velocity": float(rng.normal(*cfg["arm_angular_velocity"])),
                "delivery_tempo_s": float(rng.normal(*cfg["delivery_tempo_s"])),
                "forward_release_extension": float(rng.normal(*cfg["forward_release_extension"])),
            })

    return pd.DataFrame(records)


def _multiclass_brier_score(y_true: np.ndarray, y_proba: np.ndarray, classes: list[str]) -> float:
    """Compute standard multiclass Brier score: mean_i sum_k (p_{ik} - y_{ik})^2."""
    n = len(y_true)
    if n == 0:
        return 0.0
    y_onehot = np.zeros((n, len(classes)), dtype=float)
    cls_to_idx = {c: idx for idx, c in enumerate(classes)}
    for i, lbl in enumerate(y_true):
        if lbl in cls_to_idx:
            y_onehot[i, cls_to_idx[lbl]] = 1.0
    return float(np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1)))


class PitchTippingClassifier:
    """
    Calibrated Biomechanical Classifier with Intra-Pitcher Normalization,
    Honest GroupKFold(game_pk) Validation, and Additive SHAP Explainability.
    """

    def __init__(
        self,
        n_estimators: int = 120,
        max_depth: int = 6,
        random_state: int = 42,
        calibrate_probabilities: bool = True,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.calibrate_probabilities = calibrate_probabilities

        self.normalizer_: PitcherBaselineNormalizer | None = None
        self.base_rf_: RandomForestClassifier | None = None
        self.model: Any = None
        self.active_feature_cols_: list[str] = list(BIOMECHANICAL_FEATURE_COLS)
        self.transformed_feature_names_: list[str] = []
        self.classes_: list[str] = []
        self.explainer: Any = None
        self.validation_metrics_: dict[str, Any] = {}
        self.is_fitted: bool = False

    def _make_rf(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=self.random_state,
        )

    def _fit_estimator_with_optional_calibration(self, X_trans: pd.DataFrame, y: pd.Series) -> Any:
        min_cls = int(y.value_counts().min()) if not y.empty else 0
        if self.calibrate_probabilities and min_cls >= 3 and y.nunique() >= 2:
            cv_k = min(3, min_cls)
            cal = CalibratedClassifierCV(
                estimator=self._make_rf(),
                method="sigmoid",
                cv=cv_k,
            )
            cal.fit(X_trans, y)
            return cal
        rf = self._make_rf()
        rf.fit(X_trans, y)
        return rf

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "pitch_category",
        test_size: float = 0.2,
        group_col: str = "game_pk",
    ) -> dict[str, Any]:
        """
        Fit intra-pitcher baseline normalizer + calibrated Random Forest,
        evaluating honestly via StratifiedGroupKFold(groups=game_pk) when game_pk is available.
        """
        for col in BIOMECHANICAL_FEATURE_COLS:
            if col not in df.columns:
                raise ValueError(f"Required biomechanical feature '{col}' missing from training DataFrame.")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' missing from training DataFrame.")

        # Detect available extended biomechanical columns with >= 50% coverage
        active_cols = list(BIOMECHANICAL_FEATURE_COLS)
        for ext_col in EXTENDED_BIOMECHANICAL_FEATURE_COLS:
            if ext_col not in active_cols and ext_col in df.columns:
                non_na_ratio = float(df[ext_col].notna().mean()) if len(df) > 0 else 0.0
                if non_na_ratio >= 0.50:
                    active_cols.append(ext_col)

        self.active_feature_cols_ = active_cols
        clean_df = df.dropna(subset=BIOMECHANICAL_FEATURE_COLS + [target_col]).reset_index(drop=True).copy()

        # Impute sporadic NaNs in optional extended columns with column median
        for col in self.active_feature_cols_:
            if clean_df[col].isna().any():
                med = float(pd.to_numeric(clean_df[col], errors="coerce").median())
                clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce").fillna(0.0 if np.isnan(med) else med)

        y = clean_df[target_col].astype(str)
        all_classes = sorted(y.unique().tolist())

        has_groups = (
            group_col in clean_df.columns
            and clean_df[group_col].notna().all()
            and int(clean_df[group_col].nunique()) >= 3
            and len(clean_df) >= 12
        )

        metrics: dict[str, Any] = {}
        if has_groups:
            groups = clean_df[group_col].values
            n_groups = int(clean_df[group_col].nunique())
            n_splits = min(5, n_groups)

            # Build folds via StratifiedGroupKFold (fallback to GroupKFold if needed)
            splits: list[tuple[np.ndarray, np.ndarray]] = []
            scheme_name = f"StratifiedGroupKFold({group_col}, k={n_splits})"
            try:
                sgkf = StratifiedGroupKFold(n_splits=n_splits)
                candidate_splits = list(sgkf.split(clean_df, y, groups))
                if all(len(np.unique(y.iloc[tr])) == len(all_classes) for tr, _ in candidate_splits):
                    splits = candidate_splits
            except Exception:  # noqa: BLE001
                splits = []

            if not splits:
                gkf = GroupKFold(n_splits=n_splits)
                splits = list(gkf.split(clean_df, y, groups))
                scheme_name = f"GroupKFold({group_col}, k={n_splits})"

            oof_proba = np.zeros((len(clean_df), len(all_classes)), dtype=float)
            oof_pred = np.empty(len(clean_df), dtype=object)

            for tr_idx, va_idx in splits:
                tr_df = clean_df.iloc[tr_idx]
                va_df = clean_df.iloc[va_idx]
                y_tr = y.iloc[tr_idx]

                fold_norm = PitcherBaselineNormalizer(feature_cols=self.active_feature_cols_)
                fold_norm.fit(tr_df)
                X_tr = fold_norm.transform(tr_df)
                X_va = fold_norm.transform(va_df)

                fold_est = self._fit_estimator_with_optional_calibration(X_tr, y_tr)
                fold_probs = fold_est.predict_proba(X_va)
                fold_classes = list(fold_est.classes_)

                for c_idx, cls_name in enumerate(fold_classes):
                    if cls_name in all_classes:
                        target_idx = all_classes.index(cls_name)
                        oof_proba[va_idx, target_idx] = fold_probs[:, c_idx]

                # Normalize rows in case a fold lacked a rare class
                row_sums = oof_proba[va_idx].sum(axis=1, keepdims=True)
                zero_rows = (row_sums.squeeze(-1) <= 1e-8)
                if np.any(zero_rows):
                    oof_proba[va_idx[zero_rows]] = 1.0 / len(all_classes)
                oof_proba[va_idx] = oof_proba[va_idx] / np.maximum(oof_proba[va_idx].sum(axis=1, keepdims=True), 1e-8)

                pred_indices = np.argmax(oof_proba[va_idx], axis=1)
                oof_pred[va_idx] = [all_classes[idx] for idx in pred_indices]

            cv_acc = float(accuracy_score(y.values, oof_pred))
            cv_f1 = float(f1_score(y.values, oof_pred, average="macro", zero_division=0))
            cv_brier = _multiclass_brier_score(y.values, oof_proba, all_classes)
            cv_ll = float(log_loss(y.values, oof_proba, labels=all_classes))

            metrics = {
                "accuracy": cv_acc,
                "f1_macro": cv_f1,
                "cv_accuracy": cv_acc,
                "cv_f1_macro": cv_f1,
                "cv_brier_score": cv_brier,
                "cv_log_loss": cv_ll,
                "cv_splits": n_splits,
                "unique_games": n_groups,
                "total_samples": len(clean_df),
                "train_samples": round(len(clean_df) * (n_splits - 1) / n_splits),
                "test_samples": round(len(clean_df) / n_splits),
                "validation_scheme": scheme_name,
            }
        elif len(clean_df) >= 12 and y.value_counts().min() >= 2:
            tr_df, te_df, y_tr, y_te = train_test_split(
                clean_df,
                y,
                test_size=test_size,
                random_state=self.random_state,
                stratify=y,
            )
            eval_norm = PitcherBaselineNormalizer(feature_cols=self.active_feature_cols_)
            eval_norm.fit(tr_df)
            X_tr = eval_norm.transform(tr_df)
            X_te = eval_norm.transform(te_df)

            eval_est = self._fit_estimator_with_optional_calibration(X_tr, y_tr)
            te_probs_raw = eval_est.predict_proba(X_te)
            te_probs = np.zeros((len(te_df), len(all_classes)), dtype=float)
            for c_idx, cls_name in enumerate(eval_est.classes_):
                te_probs[:, all_classes.index(cls_name)] = te_probs_raw[:, c_idx]
            y_pred = [all_classes[idx] for idx in np.argmax(te_probs, axis=1)]

            acc = float(accuracy_score(y_te.values, y_pred))
            f1 = float(f1_score(y_te.values, y_pred, average="macro", zero_division=0))
            brier = _multiclass_brier_score(y_te.values, te_probs, all_classes)
            ll = float(log_loss(y_te.values, te_probs, labels=all_classes))
            metrics = {
                "accuracy": acc,
                "f1_macro": f1,
                "cv_accuracy": acc,
                "cv_f1_macro": f1,
                "cv_brier_score": brier,
                "cv_log_loss": ll,
                "cv_splits": 1,
                "unique_games": int(clean_df[group_col].nunique()) if group_col in clean_df.columns else 1,
                "total_samples": len(clean_df),
                "train_samples": len(tr_df),
                "test_samples": len(te_df),
                "validation_scheme": "StratifiedHoldout",
            }
        else:
            temp_norm = PitcherBaselineNormalizer(feature_cols=self.active_feature_cols_).fit(clean_df)
            X_all_t = temp_norm.transform(clean_df)
            rf_small = self._make_rf().fit(X_all_t, y)
            y_pred = rf_small.predict(X_all_t)
            acc = float(accuracy_score(y, y_pred))
            f1 = float(f1_score(y, y_pred, average="macro", zero_division=0))
            metrics = {
                "accuracy": acc,
                "f1_macro": f1,
                "cv_accuracy": acc,
                "cv_f1_macro": f1,
                "cv_brier_score": 0.0,
                "cv_log_loss": 0.0,
                "cv_splits": 1,
                "unique_games": 1,
                "total_samples": len(clean_df),
                "train_samples": len(clean_df),
                "test_samples": len(clean_df),
                "validation_scheme": "LowSampleInSample",
            }

        # Fit final production normalizer + base RF (for SHAP) + calibrated estimator (for probabilities)
        self.normalizer_ = PitcherBaselineNormalizer(feature_cols=self.active_feature_cols_).fit(clean_df)
        X_full_trans = self.normalizer_.transform(clean_df)
        self.transformed_feature_names_ = list(X_full_trans.columns)

        self.base_rf_ = self._make_rf()
        self.base_rf_.fit(X_full_trans, y)
        self.model = self._fit_estimator_with_optional_calibration(X_full_trans, y)
        self.classes_ = list(self.model.classes_)

        try:
            import shap

            self.explainer = shap.TreeExplainer(self.base_rf_)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not initialize shap.TreeExplainer: %s", exc)
            self.explainer = None

        self.validation_metrics_ = metrics
        self.is_fitted = True
        return metrics

    def _transform_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted or self.normalizer_ is None:
            raise RuntimeError("Model is not fitted. Call fit() before inference.")
        return self.normalizer_.transform(X)

    def predict(self, X: pd.DataFrame) -> list[str]:
        """Predict pitch categories for input biomechanical features."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")
        X_trans = self._transform_input(X)
        return [str(p) for p in self.model.predict(X_trans)]

    def predict_proba(self, X: pd.DataFrame) -> dict[str, float]:
        """Return calibrated class probability mapping for the first record in X."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict_proba().")
        X_trans = self._transform_input(X.iloc[:1])
        probs = self.model.predict_proba(X_trans)[0]
        return {str(cls): float(p) for cls, p in zip(self.classes_, probs)}

    def explain(self, X_sample: pd.DataFrame) -> dict[str, Any]:
        """
        Explain pitch tipping decision for a single pitch instance by aggregating
        additive SHAP attributions across raw, intra-pitcher Δ, and Z-score representations.
        """
        if not self.is_fitted or self.model is None or self.normalizer_ is None:
            raise RuntimeError("Model is not fitted. Call fit() before explain().")

        row_df = X_sample.iloc[:1]
        X_trans = self._transform_input(row_df)
        probabilities = self.predict_proba(row_df)
        predicted_category = max(probabilities, key=probabilities.get)
        confidence = float(probabilities.get(predicted_category, 0.0))

        # Compute per-transformed-column attributions via SHAP TreeExplainer
        trans_shap_map: dict[str, float] = {}
        if self.explainer is not None and self.base_rf_ is not None:
            try:
                shap_values = self.explainer.shap_values(X_trans)
                rf_classes = list(self.base_rf_.classes_)
                class_idx = rf_classes.index(predicted_category)

                if isinstance(shap_values, list):
                    class_shap = shap_values[class_idx][0]
                elif hasattr(shap_values, "shape") and len(shap_values.shape) == 3:
                    class_shap = shap_values[0, :, class_idx]
                else:
                    class_shap = shap_values[0]

                for t_col, sv in zip(X_trans.columns, class_shap):
                    trans_shap_map[t_col] = float(sv)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error computing SHAP values: %s. Falling back to RF importances.", exc)

        if not trans_shap_map and self.base_rf_ is not None:
            for t_col, imp in zip(X_trans.columns, self.base_rf_.feature_importances_):
                trans_shap_map[t_col] = float(imp)

        # Aggregate additive SHAP attributions back to each physical biomechanical feature
        feature_attributions: list[dict[str, Any]] = []
        for col in self.active_feature_cols_:
            total_sv = (
                trans_shap_map.get(col, 0.0)
                + trans_shap_map.get(f"{col}_delta", 0.0)
                + trans_shap_map.get(f"{col}_zscore", 0.0)
            )
            raw_val = float(X_trans[col].iloc[0]) if col in X_trans.columns else float(row_df[col].iloc[0])
            delta_val = float(X_trans[f"{col}_delta"].iloc[0]) if f"{col}_delta" in X_trans.columns else 0.0
            z_val = float(X_trans[f"{col}_zscore"].iloc[0]) if f"{col}_zscore" in X_trans.columns else 0.0

            feature_attributions.append({
                "feature": col,
                "display_name": FEATURE_DISPLAY_NAMES.get(col, col),
                "value": raw_val,
                "delta_vs_baseline": delta_val,
                "zscore_vs_baseline": z_val,
                "unit": FEATURE_UNITS.get(col, ""),
                "shap_value": float(total_sv),
                "impact_abs": abs(float(total_sv)),
            })

        feature_attributions.sort(key=lambda x: x["impact_abs"], reverse=True)

        top_2 = feature_attributions[:2]
        bullets = []
        for f in top_2:
            sign_str = "indicando fortemente" if f["shap_value"] >= 0 else "afastando a hipótese de"
            delta_tag = f" [Δ {f['delta_vs_baseline']:+.2f}{f['unit']}]" if abs(f["delta_vs_baseline"]) > 1e-3 else ""
            bullets.append(
                f"- **{f['display_name']}** ({f['value']:.2f}{f['unit']}{delta_tag}): {sign_str} {predicted_category}"
            )

        summary_text = (
            f"Previsão Biomecânica Calibrada: **{predicted_category}** ({confidence * 100:.1f}% de probabilidade).\n"
            f"Sinais Físicos Principais (*Pitch Tipping*):\n" + "\n".join(bullets)
        )

        return {
            "predicted_category": predicted_category,
            "confidence": confidence,
            "probabilities": probabilities,
            "top_tipping_features": feature_attributions,
            "validation_metrics": self.validation_metrics_,
            "summary_text": summary_text,
        }

    def save(self, model_path: str | Path) -> Path:
        """Serialize fitted model to disk using joblib."""
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Pitch Tipping v2 model saved to %s", path)
        return path

    @classmethod
    def load(cls, model_path: str | Path) -> PitchTippingClassifier:
        """Deserialize trained model from disk."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")
        instance = joblib.load(path)
        if not isinstance(instance, cls):
            raise TypeError(f"Loaded object is not an instance of {cls.__name__}")
        return instance
