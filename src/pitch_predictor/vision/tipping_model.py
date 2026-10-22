"""
Pitch Tipping Classifier and SHAP Explainability Engine (Passo 3).

Trains an interpretable classifier on biomechanical skeletal features
(arm slot angle, release height, elbow flexion, trunk tilt, stride extension)
to detect whether a pitcher's delivery mechanics "tip" the upcoming pitch category
(Fastball, Breaking Ball, Offspeed).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Core biomechanical features extracted from pose estimation
BIOMECHANICAL_FEATURE_COLS: list[str] = [
    "arm_slot_angle",
    "release_height",
    "elbow_flexion_angle",
    "trunk_tilt",
    "stride_extension",
]

# Friendly labels and units for SHAP explanations
FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "arm_slot_angle": "Ângulo do Braço (Arm Slot)",
    "release_height": "Altura de Soltura (Release Point)",
    "elbow_flexion_angle": "Flexão do Cotovelo (Elbow Flexion)",
    "trunk_tilt": "Inclinação do Tronco (Trunk Tilt)",
    "stride_extension": "Extensão da Passada (Stride)",
}

FEATURE_UNITS: dict[str, str] = {
    "arm_slot_angle": "°",
    "release_height": " norm",
    "elbow_flexion_angle": "°",
    "trunk_tilt": "°",
    "stride_extension": "x",
}


def generate_synthetic_biomechanics(
    n_samples_per_class: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generates a realistic synthetic biomechanical pitching dataset based on
    MLB pitcher release metrics (useful for testing, calibration, and augmentation).

    Distributions:
    - Fastball (FF): Arm slot ~48-52°, release height ~0.70-0.72, elbow flexion ~140-150° (extended)
    - Breaking Ball (ST/SL): Arm slot ~5-20° (lower/sidearm release), elbow flexion ~95-105° (bent)
    - Offspeed (FS/CH): Arm slot ~65-75° (higher arm slot), release height ~0.64-0.67, elbow ~105-115°
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
        },
        "Breaking Ball": {
            "pitch_type": "ST",
            "arm_slot_angle": (12.0, 7.0),
            "release_height": (0.67, 0.025),
            "elbow_flexion_angle": (98.0, 5.5),
            "trunk_tilt": (10.0, 1.5),
            "stride_extension": (1.35, 0.08),
        },
        "Offspeed": {
            "pitch_type": "FS",
            "arm_slot_angle": (70.0, 5.5),
            "release_height": (0.65, 0.025),
            "elbow_flexion_angle": (108.0, 6.0),
            "trunk_tilt": (12.5, 1.8),
            "stride_extension": (1.38, 0.08),
        },
    }

    for category, cfg in configs.items():
        pt = cfg["pitch_type"]
        for _ in range(n_samples_per_class):
            records.append({
                "pitch_category": category,
                "pitch_type": pt,
                "arm_slot_angle": float(rng.normal(*cfg["arm_slot_angle"])),
                "release_height": float(rng.normal(*cfg["release_height"])),
                "elbow_flexion_angle": float(rng.normal(*cfg["elbow_flexion_angle"])),
                "trunk_tilt": float(rng.normal(*cfg["trunk_tilt"])),
                "stride_extension": float(rng.normal(*cfg["stride_extension"])),
            })

    return pd.DataFrame(records)


class PitchTippingClassifier:
    """
    Random Forest Classifier with SHAP explainability for pitch tipping detection.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        self.classes_: list[str] = []
        self.explainer: Any = None
        self.is_fitted: bool = False

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "pitch_category",
        test_size: float = 0.2,
    ) -> dict[str, float]:
        """
        Fits the pitch tipping classifier on biomechanical feature columns.
        """
        for col in BIOMECHANICAL_FEATURE_COLS:
            if col not in df.columns:
                raise ValueError(f"Required biomechanical feature '{col}' missing from training DataFrame.")

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' missing from training DataFrame.")

        clean_df = df.dropna(subset=BIOMECHANICAL_FEATURE_COLS + [target_col]).copy()
        X = clean_df[BIOMECHANICAL_FEATURE_COLS]
        y = clean_df[target_col]

        if len(clean_df) >= 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            }
        else:
            # Low sample fallback
            self.model.fit(X, y)
            y_pred = self.model.predict(X)
            metrics = {
                "accuracy": float(accuracy_score(y, y_pred)),
                "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
                "train_samples": len(X),
                "test_samples": len(X),
            }

        self.classes_ = list(self.model.classes_)

        # Initialize SHAP explainer
        try:
            import shap

            self.explainer = shap.TreeExplainer(self.model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not initialize shap.TreeExplainer: %s", exc)
            self.explainer = None

        self.is_fitted = True
        return metrics

    def predict(self, X: pd.DataFrame) -> list[str]:
        """Predict pitch categories for input biomechanical features."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")
        features = X[BIOMECHANICAL_FEATURE_COLS]
        return list(self.model.predict(features))

    def predict_proba(self, X: pd.DataFrame) -> dict[str, float]:
        """Return class probability mapping for the first record in X."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() before predict_proba().")
        features = X[BIOMECHANICAL_FEATURE_COLS]
        probs = self.model.predict_proba(features)[0]
        return {cls: float(p) for cls, p in zip(self.classes_, probs)}

    def explain(self, X_sample: pd.DataFrame) -> dict[str, Any]:
        """
        Explain pitch tipping decision for a single pitch instance using SHAP attributions.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() before explain().")

        features = X_sample[BIOMECHANICAL_FEATURE_COLS].iloc[:1]
        predicted_category = self.model.predict(features)[0]
        probabilities = self.predict_proba(features)
        confidence = probabilities.get(predicted_category, 0.0)

        feature_attributions: list[dict[str, Any]] = []

        if self.explainer is not None:
            try:
                shap_values = self.explainer.shap_values(features)
                class_idx = self.classes_.index(predicted_category)

                # Format SHAP values based on library version (list or 3D array)
                if isinstance(shap_values, list):
                    class_shap = shap_values[class_idx][0]
                elif hasattr(shap_values, "shape") and len(shap_values.shape) == 3:
                    class_shap = shap_values[0, :, class_idx]
                else:
                    class_shap = shap_values[0]

                for col, sv in zip(BIOMECHANICAL_FEATURE_COLS, class_shap):
                    val = float(features[col].iloc[0])
                    feature_attributions.append({
                        "feature": col,
                        "display_name": FEATURE_DISPLAY_NAMES.get(col, col),
                        "value": val,
                        "unit": FEATURE_UNITS.get(col, ""),
                        "shap_value": float(sv),
                        "impact_abs": abs(float(sv)),
                    })
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error computing SHAP values: %s. Using feature importances.", exc)

        # Fallback to model feature importances if SHAP failed
        if not feature_attributions:
            importances = self.model.feature_importances_
            for col, imp in zip(BIOMECHANICAL_FEATURE_COLS, importances):
                val = float(features[col].iloc[0])
                feature_attributions.append({
                    "feature": col,
                    "display_name": FEATURE_DISPLAY_NAMES.get(col, col),
                    "value": val,
                    "unit": FEATURE_UNITS.get(col, ""),
                    "shap_value": float(imp),
                    "impact_abs": abs(float(imp)),
                })

        # Sort features by absolute impact descending
        feature_attributions.sort(key=lambda x: x["impact_abs"], reverse=True)

        # Generate readable natural language summary
        top_2 = feature_attributions[:2]
        bullets = []
        for f in top_2:
            sign_str = "indicando fortemente" if f["shap_value"] > 0 else "afastando a hipótese de"
            bullets.append(f"- **{f['display_name']}** ({f['value']:.1f}{f['unit']}): {sign_str} {predicted_category}")

        summary_text = (
            f"Previsão Biomecânica: **{predicted_category}** ({confidence * 100:.1f}% de certeza).\n"
            f"Sinais Físicos Principais (*Pitch Tipping*):\n" + "\n".join(bullets)
        )

        return {
            "predicted_category": predicted_category,
            "confidence": float(confidence),
            "probabilities": probabilities,
            "top_tipping_features": feature_attributions,
            "summary_text": summary_text,
        }

    def save(self, model_path: str | Path) -> Path:
        """Serialize fitted model to disk using joblib."""
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Pitch Tipping model saved to %s", path)
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
