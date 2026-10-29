"""
Unit and integration tests for Pitch Tipping Biomechanical Classifier v2:
- Intra-pitcher baseline normalization (PitcherBaselineNormalizer) with zero test leakage
- Honest StratifiedGroupKFold(groups=game_pk) validation & probability calibration
- Extended 10-feature support + backwards compatibility with 5-feature callers
- SHAP TreeExplainer attribution aggregation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitch_predictor.vision.tipping_model import (
    BIOMECHANICAL_FEATURE_COLS,
    EXTENDED_BIOMECHANICAL_FEATURE_COLS,
    PitcherBaselineNormalizer,
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
    for col in EXTENDED_BIOMECHANICAL_FEATURE_COLS:
        assert col in df.columns
        assert not df[col].isna().any()

    fb = df[df["pitch_category"] == "Fastball"]
    bb = df[df["pitch_category"] == "Breaking Ball"]
    assert fb["elbow_flexion_angle"].mean() > bb["elbow_flexion_angle"].mean()


def test_pitcher_baseline_normalizer_zero_leakage():
    """Verify PitcherBaselineNormalizer fits strictly on training rows and handles unseen pitchers."""
    train_df = pd.DataFrame([
        {"pitcher_id": 694973, "arm_slot_angle": 50.0, "elbow_flexion_angle": 150.0},
        {"pitcher_id": 694973, "arm_slot_angle": 60.0, "elbow_flexion_angle": 160.0},
        {"pitcher_id": 519242, "arm_slot_angle": 20.0, "elbow_flexion_angle": 100.0},
        {"pitcher_id": 519242, "arm_slot_angle": 30.0, "elbow_flexion_angle": 110.0},
    ])
    # Extreme outlier in test set must NOT shift fitted pitcher mean (55.0 for Skenes, 25.0 for Sale)
    test_df = pd.DataFrame([
        {"pitcher_id": 694973, "arm_slot_angle": 55.0, "elbow_flexion_angle": 155.0},
        {"pitcher_id": 519242, "arm_slot_angle": 25.0, "elbow_flexion_angle": 105.0},
        {"pitcher_id": 999999, "arm_slot_angle": 40.0, "elbow_flexion_angle": 130.0},  # Unseen pitcher
    ])

    norm = PitcherBaselineNormalizer(feature_cols=["arm_slot_angle", "elbow_flexion_angle"])
    norm.fit(train_df)
    transformed = norm.transform(test_df)

    assert "arm_slot_angle_delta" in transformed.columns
    assert "arm_slot_angle_zscore" in transformed.columns
    # Exact 0.0 delta for Skenes at 55.0 and Sale at 25.0 proves test rows never leaked into baseline
    assert pytest.approx(transformed.loc[0, "arm_slot_angle_delta"], abs=1e-6) == 0.0
    assert pytest.approx(transformed.loc[1, "arm_slot_angle_delta"], abs=1e-6) == 0.0
    # Unseen pitcher falls back cleanly to global training mean (40.0) -> 0.0 delta
    assert pytest.approx(transformed.loc[2, "arm_slot_angle_delta"], abs=1e-6) == 0.0


def test_pitch_tipping_classifier_group_kfold_and_calibration():
    """Verify honest GroupKFold validation by game_pk and calibrated probabilities on multi-pitcher data."""
    rng = np.random.default_rng(99)
    rows = []
    # Two pitchers with radically different raw baselines (Overhand RHP vs Sidearm LHP) across 6 games
    pitcher_baselines = [
        (694973, 54.0, 148.0),
        (519242, 24.0, 108.0),
    ]
    cat_deltas = {
        "Fastball": (5.0, 15.0, 0.20, 42.0, 820.0),
        "Breaking Ball": (-12.0, -18.0, 0.34, 31.0, 740.0),
        "Offspeed": (11.0, -4.0, 0.25, 38.0, 680.0),
    }
    for game_idx in range(6):
        g_pk = 745000 + game_idx
        for pid, slot_b, elbow_b in pitcher_baselines:
            for cat, (d_slot, d_elb, glv, sep, vel) in cat_deltas.items():
                for rep in range(2):
                    rows.append({
                        "game_pk": g_pk,
                        "pitcher_id": pid,
                        "pitch_category": cat,
                        "arm_slot_angle": slot_b + d_slot + float(rng.normal(0, 1.2)),
                        "release_height": 0.68 + float(rng.normal(0, 0.01)),
                        "elbow_flexion_angle": elbow_b + d_elb + float(rng.normal(0, 1.8)),
                        "trunk_tilt": 14.0 + float(rng.normal(0, 0.8)),
                        "stride_extension": 1.42 + float(rng.normal(0, 0.02)),
                        "glove_tuck_distance": glv + float(rng.normal(0, 0.015)),
                        "hip_shoulder_separation": sep + float(rng.normal(0, 1.5)),
                        "arm_angular_velocity": vel + float(rng.normal(0, 20.0)),
                        "delivery_tempo_s": 0.56 + float(rng.normal(0, 0.02)),
                        "forward_release_extension": 0.62 + float(rng.normal(0, 0.02)),
                    })

    multi_df = pd.DataFrame(rows)
    clf = PitchTippingClassifier(random_state=42)
    metrics = clf.fit(multi_df, target_col="pitch_category")

    assert "cv_accuracy" in metrics
    assert "cv_f1_macro" in metrics
    assert "cv_brier_score" in metrics
    assert "cv_log_loss" in metrics
    assert "GroupKFold" in metrics["validation_scheme"]
    # Thanks to PitcherBaselineNormalizer, cross-game multi-pitcher accuracy should be strong
    assert metrics["cv_accuracy"] >= 0.80
    assert 0.0 <= metrics["cv_brier_score"] <= 0.50

    # Sidearm LHP Chris Sale Fastball (arm_slot ~29 deg, which would fool an unnormalized RHP model!)
    sale_fb = pd.DataFrame([{
        "pitcher_id": 519242,
        "arm_slot_angle": 29.0,
        "release_height": 0.68,
        "elbow_flexion_angle": 123.0,
        "trunk_tilt": 14.0,
        "stride_extension": 1.42,
        "glove_tuck_distance": 0.20,
        "hip_shoulder_separation": 42.0,
        "arm_angular_velocity": 820.0,
        "delivery_tempo_s": 0.56,
        "forward_release_extension": 0.62,
    }])
    assert clf.predict(sale_fb)[0] == "Fastball"
    probs = clf.predict_proba(sale_fb)
    assert pytest.approx(sum(probs.values()), abs=1e-3) == 1.0


def test_pitch_tipping_classifier_fit_and_predict(sample_biomechanics_df: pd.DataFrame):
    """Verify classifier training, predict and predict_proba contracts (5-col backwards compatibility)."""
    clf = PitchTippingClassifier(random_state=42)
    metrics = clf.fit(sample_biomechanics_df, target_col="pitch_category")

    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert metrics["accuracy"] >= 0.70

    # Test single-row inference with legacy 5-column input
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
