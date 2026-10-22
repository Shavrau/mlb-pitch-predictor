"""
MLflow Custom PyFunc Model for Multimodal Pitch Serving.

Wraps MultimodalPitchClassifier (Statcast situational context + release biomechanics)
into an MLflow pyfunc model deployable to Databricks Serverless Model Serving.
"""

from __future__ import annotations

import logging
from typing import Any

import joblib
import mlflow.pyfunc
import pandas as pd
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, DataType, Schema

from pitch_predictor.multimodal import (
    BIOMECHANICAL_FEATURE_COLS,
    DEFAULT_BIOMECHANICS_IMPUTATION,
    MULTIMODAL_FEATURE_COLS,
    STATCAST_FEATURE_COLS,
    MultimodalPitchClassifier,
)

logger = logging.getLogger(__name__)


class MultimodalServingPyfuncModel(mlflow.pyfunc.PythonModel):
    """
    MLflow Custom PyFunc serving wrapper for the Multimodal Pitch Predictor.

    Capabilities:
    - Zero-downtime serving on Databricks Serverless Model Serving.
    - Automatic imputation fallback when biomechanical features are omitted.
    - Decoupled multimodal evaluation: computes unimodal baselines (Statcast & Vision)
      alongside fused multimodal probabilities.
    - Diagnostics generation: produces alignment status ('AGREEMENT', 'TIPPING_OVERRIDE',
      'CONTEXT_REINFORCED') and readable tactical summary.
    """

    def __init__(self, classifier: MultimodalPitchClassifier | None = None) -> None:
        self.classifier = classifier

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load serialized MultimodalPitchClassifier from MLflow artifact repository."""
        if self.classifier is None:
            model_path = context.artifacts.get("model_path")
            if model_path is None:
                raise ValueError("Artifact 'model_path' not found in MLflow context.")
            self.classifier = joblib.load(model_path)
            logger.info("Successfully loaded MultimodalPitchClassifier from %s", model_path)

    def _prepare_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate, impute, and format input dataframe to ensure all features exist.
        """
        clean_df = df.copy()

        # Handle missing Statcast features
        if "count_state" not in clean_df.columns:
            if "balls" in clean_df.columns and "strikes" in clean_df.columns:
                clean_df["count_state"] = (
                    clean_df["balls"].astype(int).astype(str)
                    + "-"
                    + clean_df["strikes"].astype(int).astype(str)
                )
            else:
                clean_df["count_state"] = "0-0"

        if "count_leverage" not in clean_df.columns:
            clean_df["count_leverage"] = "Neutral"

        if "prev_pitch_category" not in clean_df.columns:
            clean_df["prev_pitch_category"] = "First Pitch"

        # Defaults for numeric Statcast features
        numeric_defaults = {
            "balls": 0,
            "strikes": 0,
            "runners_on_base": 0,
            "scoring_position": 0,
            "same_hand": 1,
            "pitcher_fb_prior": 0.5543,
            "pitcher_bb_prior": 0.3053,
            "pitcher_os_prior": 0.1404,
        }
        for num_col, def_val in numeric_defaults.items():
            if num_col not in clean_df.columns:
                clean_df[num_col] = def_val
            else:
                clean_df[num_col] = clean_df[num_col].fillna(def_val)

        # Impute missing or NaN biomechanical features
        for bio_col in BIOMECHANICAL_FEATURE_COLS:
            def_bio = DEFAULT_BIOMECHANICS_IMPUTATION[bio_col]
            if bio_col not in clean_df.columns:
                clean_df[bio_col] = def_bio
            else:
                clean_df[bio_col] = clean_df[bio_col].fillna(def_bio)

        return clean_df

    def predict(
        self,
        context,
        model_input,
        params=None,
    ):
        """
        Run vectorized multimodal inference across batch input records.

        Returns:
        --------
        pd.DataFrame containing:
            - prediction: Top fused pitch category string
            - prob_fastball: Probability for Fastball
            - prob_breaking_ball: Probability for Breaking Ball
            - prob_offspeed: Probability for Offspeed
            - statcast_prediction: Statcast-only baseline prediction
            - vision_prediction: Biomechanics-only baseline prediction
            - alignment_status: AGREEMENT, TIPPING_OVERRIDE, or CONTEXT_REINFORCED
            - summary_text: Human-readable tactical and physical explanation
        """
        if self.classifier is None:
            raise ValueError("Classifier is not loaded or fitted.")

        if isinstance(model_input, (list, dict)):
            df = pd.DataFrame(model_input if isinstance(model_input, list) else [model_input])
        else:
            df = model_input.copy()

        clean_df = self._prepare_inputs(df)

        sc_pipeline = self.classifier.statcast_pipeline
        vis_pipeline = self.classifier.vision_pipeline
        multi_pipeline = self.classifier.multimodal_pipeline

        if sc_pipeline is None or vis_pipeline is None or multi_pipeline is None:
            raise ValueError("Internal pipelines are not initialized on classifier.")

        # Vectorized batch predictions
        sc_preds = sc_pipeline.predict(clean_df[STATCAST_FEATURE_COLS])

        vis_preds = vis_pipeline.predict(clean_df[BIOMECHANICAL_FEATURE_COLS])
        vis_probs = vis_pipeline.predict_proba(clean_df[BIOMECHANICAL_FEATURE_COLS])
        vis_classes = list(vis_pipeline.classes_)

        multi_preds = multi_pipeline.predict(clean_df[MULTIMODAL_FEATURE_COLS])
        multi_probs = multi_pipeline.predict_proba(clean_df[MULTIMODAL_FEATURE_COLS])
        multi_classes = list(multi_pipeline.classes_)

        fb_idx = multi_classes.index("Fastball") if "Fastball" in multi_classes else 0
        bb_idx = multi_classes.index("Breaking Ball") if "Breaking Ball" in multi_classes else 1
        os_idx = multi_classes.index("Offspeed") if "Offspeed" in multi_classes else 2

        results: list[dict[str, Any]] = []
        for i in range(len(clean_df)):
            p_pred = str(multi_preds[i])
            sc_p = str(sc_preds[i])
            vis_p = str(vis_preds[i])

            p_fb = float(multi_probs[i][fb_idx])
            p_bb = float(multi_probs[i][bb_idx])
            p_os = float(multi_probs[i][os_idx])

            # Class probability for top predicted class
            top_prob = p_fb if p_pred == "Fastball" else (p_bb if p_pred == "Breaking Ball" else p_os)

            # Alignment and Diagnosis
            if sc_p == vis_p:
                alignment = "AGREEMENT"
                summary = (
                    f"Alinhamento Tático & Biomecânico: Contexto situacional e mecânica corporal "
                    f"concordam em {p_pred} ({top_prob:.1%} de confiança)."
                )
            elif p_pred == vis_p:
                alignment = "TIPPING_OVERRIDE"
                vis_prob_val = float(vis_probs[i][vis_classes.index(vis_p)])
                summary = (
                    f"Pitch Tipping Detectado: Contexto situacional sugeria {sc_p}, mas postura física "
                    f"na soltura denunciou {p_pred} ({vis_prob_val:.1%} na visão). Prioridade biomecânica ativa."
                )
            else:
                alignment = "CONTEXT_REINFORCED"
                summary = (
                    f"Contexto Tático Prevalece: Gesto físico sugeriu {vis_p}, mas leverage e tendências "
                    f"históricas reforçaram {p_pred} ({top_prob:.1%})."
                )

            results.append(
                {
                    "prediction": p_pred,
                    "prob_fastball": round(p_fb, 4),
                    "prob_breaking_ball": round(p_bb, 4),
                    "prob_offspeed": round(p_os, 4),
                    "statcast_prediction": sc_p,
                    "vision_prediction": vis_p,
                    "alignment_status": alignment,
                    "summary_text": summary,
                }
            )

        return pd.DataFrame(results)


def get_pyfunc_model_signature() -> ModelSignature:
    """
    Construct MLflow ModelSignature declaring the contract for both tactical and physical features.
    """
    inputs = Schema(
        [
            # Statcast Contextual Features
            ColSpec(DataType.string, "count_state"),
            ColSpec(DataType.string, "count_leverage"),
            ColSpec(DataType.string, "prev_pitch_category"),
            ColSpec(DataType.long, "balls"),
            ColSpec(DataType.long, "strikes"),
            ColSpec(DataType.long, "runners_on_base"),
            ColSpec(DataType.long, "scoring_position"),
            ColSpec(DataType.long, "same_hand"),
            ColSpec(DataType.double, "pitcher_fb_prior"),
            ColSpec(DataType.double, "pitcher_bb_prior"),
            ColSpec(DataType.double, "pitcher_os_prior"),
            # Biomechanical Features (optional with fallback imputation)
            ColSpec(DataType.double, "arm_slot_angle", required=False),
            ColSpec(DataType.double, "release_height", required=False),
            ColSpec(DataType.double, "elbow_flexion_angle", required=False),
            ColSpec(DataType.double, "trunk_tilt", required=False),
            ColSpec(DataType.double, "stride_extension", required=False),
        ]
    )

    outputs = Schema(
        [
            ColSpec(DataType.string, "prediction"),
            ColSpec(DataType.double, "prob_fastball"),
            ColSpec(DataType.double, "prob_breaking_ball"),
            ColSpec(DataType.double, "prob_offspeed"),
            ColSpec(DataType.string, "statcast_prediction"),
            ColSpec(DataType.string, "vision_prediction"),
            ColSpec(DataType.string, "alignment_status"),
            ColSpec(DataType.string, "summary_text"),
        ]
    )

    return ModelSignature(inputs=inputs, outputs=outputs)


def create_pyfunc_input_example() -> pd.DataFrame:
    """Generate a valid input example dataframe matching the declared signature."""
    return pd.DataFrame(
        [
            {
                "count_state": "1-2",
                "count_leverage": "Pitcher Ahead",
                "prev_pitch_category": "Fastball",
                "balls": 1,
                "strikes": 2,
                "runners_on_base": 0,
                "scoring_position": 0,
                "same_hand": 1,
                "pitcher_fb_prior": 0.5543,
                "pitcher_bb_prior": 0.3053,
                "pitcher_os_prior": 0.1404,
                "arm_slot_angle": 48.0,
                "release_height": 0.70,
                "elbow_flexion_angle": 135.0,
                "trunk_tilt": 11.5,
                "stride_extension": 1.40,
            }
        ]
    )
