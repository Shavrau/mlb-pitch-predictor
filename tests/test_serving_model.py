"""
Unit and integration tests for MultimodalServingPyfuncModel.

Verifies:
- Input handling with complete multimodal data (Statcast + Biomechanics).
- Automatic imputation fallback when biomechanical features are omitted.
- Compatibility with 24-column Silver Statcast feature records.
- Vectorized batch predictions returning structured output (prediction, probabilities, alignment_status, summary_text).
- MLflow PyFunc saving, loading, and signature compliance.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pitch_predictor.multimodal import (
    BIOMECHANICAL_FEATURE_COLS,
    PITCH_CATEGORIES,
    STATCAST_FEATURE_COLS,
    MultimodalPitchClassifier,
    generate_synthetic_multimodal_data,
)
from pitch_predictor.serving_client import derive_tactical_features
from pitch_predictor.serving_model import (
    MultimodalServingPyfuncModel,
    create_pyfunc_input_example,
    get_pyfunc_model_signature,
)


@pytest.fixture
def trained_multimodal_classifier() -> MultimodalPitchClassifier:
    """Fixture providing a fitted MultimodalPitchClassifier on synthetic data."""
    _, _, df_full = generate_synthetic_multimodal_data(n_samples_per_class=30, random_state=42)
    clf = MultimodalPitchClassifier(n_estimators=10, max_depth=4, random_state=42)
    clf.fit(df_full)
    return clf


@pytest.fixture
def pyfunc_serving_model(
    trained_multimodal_classifier: MultimodalPitchClassifier,
) -> MultimodalServingPyfuncModel:
    """Fixture providing an initialized MultimodalServingPyfuncModel."""
    model = MultimodalServingPyfuncModel(classifier=trained_multimodal_classifier)
    return model


def test_pyfunc_predict_complete_multimodal(
    pyfunc_serving_model: MultimodalServingPyfuncModel,
) -> None:
    """Verify inference when both Statcast and biomechanics features are provided."""
    sample_data = pd.DataFrame(
        [
            {
                "balls": 0,
                "strikes": 2,
                "count_state": "0-2",
                "count_leverage": "Pitcher Ahead",
                "prev_pitch_category": "Fastball",
                "runners_on_base": 0,
                "scoring_position": 0,
                "same_hand": 1,
                "pitcher_fb_prior": 0.55,
                "pitcher_bb_prior": 0.35,
                "pitcher_os_prior": 0.10,
                "arm_slot_angle": 48.0,
                "release_height": 0.70,
                "elbow_flexion_angle": 135.0,
                "trunk_tilt": 11.5,
                "stride_extension": 1.40,
            },
            {
                "balls": 3,
                "strikes": 0,
                "count_state": "3-0",
                "count_leverage": "Pitcher Behind",
                "prev_pitch_category": "Offspeed",
                "runners_on_base": 2,
                "scoring_position": 1,
                "same_hand": 0,
                "pitcher_fb_prior": 0.40,
                "pitcher_bb_prior": 0.40,
                "pitcher_os_prior": 0.20,
                "arm_slot_angle": 19.5,
                "release_height": 0.68,
                "elbow_flexion_angle": 118.0,
                "trunk_tilt": 10.0,
                "stride_extension": 1.35,
            },
        ]
    )

    preds_df = pyfunc_serving_model.predict(context=None, model_input=sample_data)

    assert isinstance(preds_df, pd.DataFrame)
    assert len(preds_df) == 2
    expected_cols = [
        "prediction",
        "prob_fastball",
        "prob_breaking_ball",
        "prob_offspeed",
        "statcast_prediction",
        "vision_prediction",
        "alignment_status",
        "summary_text",
    ]
    for col in expected_cols:
        assert col in preds_df.columns, f"Missing column {col} in output"

    for _, row in preds_df.iterrows():
        assert row["prediction"] in PITCH_CATEGORIES
        assert row["statcast_prediction"] in PITCH_CATEGORIES
        assert row["vision_prediction"] in PITCH_CATEGORIES
        assert row["alignment_status"] in {
            "AGREEMENT",
            "TIPPING_OVERRIDE",
            "CONTEXT_REINFORCED",
        }
        assert len(row["summary_text"]) > 10
        total_prob = (
            row["prob_fastball"] + row["prob_breaking_ball"] + row["prob_offspeed"]
        )
        assert np.isclose(total_prob, 1.0, atol=1e-3)


def test_pyfunc_predict_missing_biomechanics_fallback(
    pyfunc_serving_model: MultimodalServingPyfuncModel,
) -> None:
    """Verify that omitting biomechanical columns automatically uses default imputation."""
    statcast_only_data = pd.DataFrame(
        [
            {
                "balls": 1,
                "strikes": 1,
                "count_state": "1-1",
                "count_leverage": "Neutral",
                "prev_pitch_category": "Breaking Ball",
                "runners_on_base": 1,
                "scoring_position": 0,
                "same_hand": 1,
                "pitcher_fb_prior": 0.50,
                "pitcher_bb_prior": 0.30,
                "pitcher_os_prior": 0.20,
            }
        ]
    )

    preds_df = pyfunc_serving_model.predict(context=None, model_input=statcast_only_data)

    assert len(preds_df) == 1
    assert preds_df["prediction"].iloc[0] in PITCH_CATEGORIES
    assert preds_df["alignment_status"].iloc[0] in {
        "AGREEMENT",
        "TIPPING_OVERRIDE",
        "CONTEXT_REINFORCED",
    }


def test_pyfunc_predict_accepts_24_tactical_columns(
    pyfunc_serving_model: MultimodalServingPyfuncModel,
) -> None:
    """Verify compatibility with the full 24-column Silver feature payload."""
    raw_inputs = {
        "balls": 1,
        "strikes": 2,
        "prev_pitch_category": "Fastball",
        "prev_pitch_type": "FF",
        "prev_pitch_zone": 5,
        "prev_pitch_result": "called_strike",
        "p_throws": "R",
        "stand": "R",
        "outs_when_up": 2,
        "inning": 7,
        "inning_topbot": "Top",
        "score_diff": -1,
        "on_1b": False,
        "on_2b": True,
        "on_3b": False,
        "n_thruorder_pitcher": 2,
        "pitcher_fb_prior": 0.60,
        "pitcher_bb_prior": 0.25,
        "pitcher_os_prior": 0.15,
    }
    derived = derive_tactical_features(raw_inputs)
    derived_df = pd.DataFrame([derived])

    preds_df = pyfunc_serving_model.predict(context=None, model_input=derived_df)
    assert len(preds_df) == 1
    assert preds_df["prediction"].iloc[0] in PITCH_CATEGORIES


def test_pyfunc_signature_and_input_example() -> None:
    """Verify MLflow ModelSignature generation and input example format."""
    signature = get_pyfunc_model_signature()
    assert signature is not None
    assert signature.inputs is not None
    assert signature.outputs is not None

    input_names = [col.name for col in signature.inputs.inputs]
    for col in STATCAST_FEATURE_COLS:
        assert col in input_names
    for col in BIOMECHANICAL_FEATURE_COLS:
        assert col in input_names

    output_names = [col.name for col in signature.outputs.inputs]
    assert "prediction" in output_names
    assert "prob_fastball" in output_names
    assert "alignment_status" in output_names

    example = create_pyfunc_input_example()
    assert isinstance(example, pd.DataFrame)
    assert len(example) >= 1
    for col in STATCAST_FEATURE_COLS + BIOMECHANICAL_FEATURE_COLS:
        assert col in example.columns


def test_pyfunc_save_and_load_with_mlflow(
    trained_multimodal_classifier: MultimodalPitchClassifier,
) -> None:
    """Verify MLflow pyfunc packaging, serialization, and deserialization."""
    import mlflow.pyfunc

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_save_dir = Path(tmp_dir) / "mlflow_multimodal_model"
        signature = get_pyfunc_model_signature()
        input_example = create_pyfunc_input_example()

        pyfunc_model = MultimodalServingPyfuncModel(
            classifier=trained_multimodal_classifier
        )

        mlflow.pyfunc.save_model(
            path=str(model_save_dir),
            python_model=pyfunc_model,
            signature=signature,
            input_example=input_example,
        )

        assert (model_save_dir / "MLmodel").exists()

        # Load back via pyfunc
        loaded_pyfunc = mlflow.pyfunc.load_model(str(model_save_dir))
        eval_data = input_example.copy()
        loaded_preds = loaded_pyfunc.predict(eval_data)

        assert isinstance(loaded_preds, pd.DataFrame)
        assert len(loaded_preds) == len(eval_data)
        assert "prediction" in loaded_preds.columns
        assert loaded_preds["prediction"].iloc[0] in PITCH_CATEGORIES
