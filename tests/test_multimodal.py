"""
Unit and integration tests for Multimodal Fusion (Passo 5).

Tests:
1. Multimodal constants and feature definitions.
2. Dataset join on (game_pk, at_bat_number, pitch_number) with zero leakage.
3. Synthetic paired dataset generation.
4. MultimodalPitchClassifier training, predict, predict_proba.
5. Ablation study: Statcast-only vs Vision-only vs Multimodal lift.
6. Situational modality comparison (Agreement vs Tipping Override).
7. Model serialization and reloading fidelity.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from pitch_predictor.multimodal import (
    BIOMECHANICAL_FEATURE_COLS,
    MULTIMODAL_FEATURE_COLS,
    MULTIMODAL_JOIN_KEYS,
    STATCAST_FEATURE_COLS,
    MultimodalPitchClassifier,
    create_multimodal_dataset,
    generate_synthetic_multimodal_data,
)


def test_multimodal_feature_constants():
    """Verify join keys and feature lists integrity."""
    assert MULTIMODAL_JOIN_KEYS == ["game_pk", "at_bat_number", "pitch_number"]
    assert "arm_slot_angle" in BIOMECHANICAL_FEATURE_COLS
    assert "elbow_flexion_angle" in BIOMECHANICAL_FEATURE_COLS
    assert "count_state" in STATCAST_FEATURE_COLS
    assert "pitcher_fb_prior" in STATCAST_FEATURE_COLS

    for col in BIOMECHANICAL_FEATURE_COLS:
        assert col in MULTIMODAL_FEATURE_COLS
    for col in STATCAST_FEATURE_COLS:
        assert col in MULTIMODAL_FEATURE_COLS


def test_create_multimodal_dataset_inner_join():
    """Verify inner join matches exactly on composite primary key."""
    statcast_df = pd.DataFrame([
        {
            "game_pk": 745170,
            "at_bat_number": 31,
            "pitch_number": 4,
            "pitcher": 694973,
            "game_date": "2024-04-01",
            "pitch_category": "Fastball",
            "balls": 1,
            "strikes": 2,
            "count_state": "1-2",
            "count_leverage": "Pitcher Ahead",
            "prev_pitch_category": "Fastball",
            "runners_on_base": 0,
            "scoring_position": 0,
            "same_hand": 1,
            "pitcher_fb_prior": 0.55,
            "pitcher_bb_prior": 0.30,
            "pitcher_os_prior": 0.15,
        },
        {
            "game_pk": 745170,
            "at_bat_number": 31,
            "pitch_number": 5,
            "pitcher": 694973,
            "game_date": "2024-04-01",
            "pitch_category": "Breaking Ball",
            "balls": 2,
            "strikes": 2,
            "count_state": "2-2",
            "count_leverage": "Neutral",
            "prev_pitch_category": "Fastball",
            "runners_on_base": 0,
            "scoring_position": 0,
            "same_hand": 1,
            "pitcher_fb_prior": 0.55,
            "pitcher_bb_prior": 0.30,
            "pitcher_os_prior": 0.15,
        },
    ])

    bio_df = pd.DataFrame([
        {
            "game_pk": 745170,
            "at_bat_number": 31,
            "pitch_number": 4,
            "arm_slot_angle": 49.8,
            "release_height": 0.71,
            "elbow_flexion_angle": 146.1,
            "trunk_tilt": 11.7,
            "stride_extension": 0.20,
        }
    ])

    merged = create_multimodal_dataset(statcast_df, bio_df, how="inner")
    assert len(merged) == 1
    assert merged.iloc[0]["game_pk"] == 745170
    assert merged.iloc[0]["pitch_number"] == 4
    assert merged.iloc[0]["arm_slot_angle"] == pytest.approx(49.8)
    assert merged.iloc[0]["pitch_category"] == "Fastball"


def test_create_multimodal_dataset_left_join_with_missing_imputation():
    """Verify left join with imputation for missing biomechanics."""
    statcast_df = pd.DataFrame([
        {
            "game_pk": 100,
            "at_bat_number": 1,
            "pitch_number": 1,
            "pitcher": 694973,
            "game_date": "2024-04-01",
            "pitch_category": "Fastball",
            "balls": 0,
            "strikes": 0,
            "count_state": "0-0",
            "count_leverage": "Neutral",
            "prev_pitch_category": "First Pitch",
            "runners_on_base": 0,
            "scoring_position": 0,
            "same_hand": 1,
            "pitcher_fb_prior": 0.55,
            "pitcher_bb_prior": 0.30,
            "pitcher_os_prior": 0.15,
        },
        {
            "game_pk": 100,
            "at_bat_number": 1,
            "pitch_number": 2,
            "pitcher": 694973,
            "game_date": "2024-04-01",
            "pitch_category": "Breaking Ball",
            "balls": 0,
            "strikes": 1,
            "count_state": "0-1",
            "count_leverage": "Pitcher Ahead",
            "prev_pitch_category": "Fastball",
            "runners_on_base": 0,
            "scoring_position": 0,
            "same_hand": 1,
            "pitcher_fb_prior": 0.55,
            "pitcher_bb_prior": 0.30,
            "pitcher_os_prior": 0.15,
        },
    ])

    bio_df = pd.DataFrame([
        {
            "game_pk": 100,
            "at_bat_number": 1,
            "pitch_number": 1,
            "arm_slot_angle": 50.0,
            "release_height": 0.70,
            "elbow_flexion_angle": 145.0,
            "trunk_tilt": 12.0,
            "stride_extension": 1.45,
        }
    ])

    merged = create_multimodal_dataset(statcast_df, bio_df, how="left", fill_missing_biomechanics=True)
    assert len(merged) == 2
    assert "has_biomechanics" in merged.columns
    assert merged.loc[merged["pitch_number"] == 1, "has_biomechanics"].iloc[0] == 1
    assert merged.loc[merged["pitch_number"] == 2, "has_biomechanics"].iloc[0] == 0
    # Imputed values should not be NaN
    assert not merged["elbow_flexion_angle"].isna().any()


def test_generate_synthetic_multimodal_data():
    """Verify generator yields aligned Statcast and Biomechanical data."""
    statcast_df, bio_df, full_df = generate_synthetic_multimodal_data(n_samples_per_class=20, random_state=42)

    assert len(statcast_df) == 60
    assert len(bio_df) == 60
    assert len(full_df) == 60

    for col in MULTIMODAL_JOIN_KEYS:
        assert col in statcast_df.columns
        assert col in bio_df.columns

    for col in STATCAST_FEATURE_COLS:
        assert col in full_df.columns
    for col in BIOMECHANICAL_FEATURE_COLS:
        assert col in full_df.columns

    assert set(full_df["pitch_category"].unique()) == {"Fastball", "Breaking Ball", "Offspeed"}


def test_multimodal_classifier_fit_and_predict():
    """Verify fitting MultimodalPitchClassifier and making predictions."""
    _, _, full_df = generate_synthetic_multimodal_data(n_samples_per_class=30, random_state=42)

    model = MultimodalPitchClassifier(n_estimators=25, random_state=42)
    model.fit(full_df)

    sample = full_df.iloc[[0]]
    pred = model.predict(sample)[0]
    assert pred in ["Fastball", "Breaking Ball", "Offspeed"]

    probs = model.predict_proba(sample)[0]
    assert len(probs) == 3
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-3)


def test_multimodal_classifier_ablation_study():
    """Verify ablation study compares Statcast, Vision, and Multimodal and computes Lift."""
    _, _, full_df = generate_synthetic_multimodal_data(n_samples_per_class=40, random_state=42)

    train_df = full_df.iloc[:90].copy()
    test_df = full_df.iloc[90:].copy()

    model = MultimodalPitchClassifier(n_estimators=30, random_state=42)
    ablation_results = model.fit_ablation_study(train_df, test_df)

    assert "statcast" in ablation_results
    assert "vision" in ablation_results
    assert "multimodal" in ablation_results
    assert "lift" in ablation_results

    # Multimodal accuracy should be high on synthetic multimodal signals
    assert ablation_results["multimodal"]["accuracy"] >= 0.70
    assert "lift_f1_macro" in ablation_results["lift"]


def test_multimodal_classifier_compare_modalities():
    """Verify comparing modalities identifies agreement vs tipping override."""
    _, _, full_df = generate_synthetic_multimodal_data(n_samples_per_class=30, random_state=42)
    model = MultimodalPitchClassifier(n_estimators=25, random_state=42)
    model.fit(full_df)

    # Test with a single row
    comparison = model.compare_modalities(full_df.iloc[0])
    assert "statcast_prediction" in comparison
    assert "vision_prediction" in comparison
    assert "multimodal_prediction" in comparison
    assert "alignment_status" in comparison
    assert comparison["alignment_status"] in ["AGREEMENT", "TIPPING_OVERRIDE", "CONTEXT_REINFORCED"]


def test_multimodal_classifier_save_and_load():
    """Verify serialization to joblib and reloading."""
    _, _, full_df = generate_synthetic_multimodal_data(n_samples_per_class=20, random_state=42)
    model = MultimodalPitchClassifier(n_estimators=15, random_state=42)
    model.fit(full_df)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "multimodal_test_model.joblib"
        model.save(save_path)
        assert save_path.exists()

        loaded_model = MultimodalPitchClassifier.load(save_path)
        pred_original = model.predict(full_df.iloc[[0]])[0]
        pred_loaded = loaded_model.predict(full_df.iloc[[0]])[0]
        assert pred_original == pred_loaded


def test_multimodal_classifier_unfitted_raises():
    """Verify calling predict on unfitted model raises ValueError."""
    model = MultimodalPitchClassifier()
    df = pd.DataFrame([{"arm_slot_angle": 50.0}])
    with pytest.raises(ValueError, match="Model is not fitted"):
        model.predict(df)
