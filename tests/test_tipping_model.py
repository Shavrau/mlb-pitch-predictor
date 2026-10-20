"""
Unit and integration tests for Pitch Tipping Biomechanical Classifier and SHAP Explainability (Passo 3).
"""

from __future__ import annotations

import pandas as pd
import pytest

from pitch_predictor.vision.tipping_model import (
    BIOMECHANICAL_FEATURE_COLS,
    PitchTippingClassifier,
    generate_synthetic_biomechanics,
)


@pytest.fixture
def sample_biomechanics_df() -> pd.DataFrame:
    """Fixture providing a balanced biomechanical dataset for 3 pitch categories."""
    return generate_synthetic_biomechanics(
        n_samples_per_class=30,
        random_state=42,
    )


def test_generate_synthetic_biomechanics_distribution():
    """Verify that synthetic biomechanics matches realistic MLB domain distributions."""
    df = generate_synthetic_biomechanics(n_samples_per_class=20, random_state=123)

    assert len(df) == 60
    assert set(df["pitch_category"].unique()) == {"Fastball", "Breaking Ball", "Offspeed"}
    for col in BIOMECHANICAL_FEATURE_COLS:
        assert col in df.columns
        assert not df[col].isna().any()

    # Fastball: Higher arm slot and extended elbow flexion
    fb = df[df["pitch_category"] == "Fastball"]
    bb = df[df["pitch_category"] == "Breaking Ball"]
    assert fb["elbow_flexion_angle"].mean() > bb["elbow_flexion_angle"].mean()


def test_pitch_tipping_classifier_fit_and_predict(sample_biomechanics_df: pd.DataFrame):
    """Verify classifier training, predict and predict_proba contracts."""
    clf = PitchTippingClassifier(random_state=42)
    metrics = clf.fit(sample_biomechanics_df, target_col="pitch_category")

    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert metrics["accuracy"] >= 0.70

    # Test single-row inference
    test_sample = pd.DataFrame([{
        "arm_slot_angle": 50.0,
        "release_height": 0.71,
        "elbow_flexion_angle": 145.0,
        "trunk_tilt": 11.5,
        "stride_extension": 1.4,
    }])
    pred_category = clf.predict(test_sample)[0]
    probabilities = clf.predict_proba(test_sample)

    assert pred_category in {"Fastball", "Breaking Ball", "Offspeed"}
    assert isinstance(probabilities, dict)
    assert set(probabilities.keys()) == {"Fastball", "Breaking Ball", "Offspeed"}
    assert pytest.approx(sum(probabilities.values()), abs=1e-3) == 1.0


def test_pitch_tipping_classifier_explain_shap(sample_biomechanics_df: pd.DataFrame):
    """Verify SHAP attributions calculation and natural language explainability."""
    clf = PitchTippingClassifier(random_state=42)
    clf.fit(sample_biomechanics_df, target_col="pitch_category")

    # Sweeper-like sample (bent elbow, lower side arm slot)
    sweeper_sample = pd.DataFrame([{
        "arm_slot_angle": 15.0,
        "release_height": 0.65,
        "elbow_flexion_angle": 95.0,
        "trunk_tilt": 9.5,
        "stride_extension": 1.3,
    }])

    explanation = clf.explain(sweeper_sample)
    assert "predicted_category" in explanation
    assert "confidence" in explanation
    assert "top_tipping_features" in explanation
    assert "summary_text" in explanation

    top_features = explanation["top_tipping_features"]
    assert len(top_features) > 0
    # Elbow flexion and arm slot are top physical tells
    top_feature_names = [f["feature"] for f in top_features]
    assert any(feat in top_feature_names for feat in ["elbow_flexion_angle", "arm_slot_angle"])


def test_pitch_tipping_classifier_save_and_load(tmp_path, sample_biomechanics_df: pd.DataFrame):
    """Verify joblib model serialization and deserialization."""
    clf = PitchTippingClassifier(random_state=42)
    clf.fit(sample_biomechanics_df)

    model_file = tmp_path / "pitch_tipping_model.joblib"
    saved_path = clf.save(model_file)
    assert saved_path.exists()

    loaded_clf = PitchTippingClassifier.load(model_file)
    assert loaded_clf.is_fitted

    test_sample = sample_biomechanics_df[BIOMECHANICAL_FEATURE_COLS].iloc[:1]
    assert clf.predict(test_sample)[0] == loaded_clf.predict(test_sample)[0]


def test_pitch_tipping_classifier_unfitted_raises():
    """Verify error when attempting prediction or explanation before fit."""
    clf = PitchTippingClassifier()
    dummy = pd.DataFrame([{col: 0.0 for col in BIOMECHANICAL_FEATURE_COLS}])

    with pytest.raises(RuntimeError, match="Model is not fitted"):
        clf.predict(dummy)

    with pytest.raises(RuntimeError, match="Model is not fitted"):
        clf.explain(dummy)
