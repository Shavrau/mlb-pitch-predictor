#!/usr/bin/env python3
"""
CLI Tool for Training the Pitch Tipping Biomechanical Model with SHAP Explainability.

Trains a Random Forest classifier on biomechanical skeletal features
(arm slot, release height, elbow flexion, trunk tilt, stride extension),
evaluates classification metrics, computes SHAP attributions to explain
pitch tipping tells, and synchronizes the trained model to the Unity Catalog Volume.

Usage:
    python scripts/train_tipping_model.py --data-path data/videos/694973/biomechanical_features.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from pitch_predictor.vision.tipping_model import (
    BIOMECHANICAL_FEATURE_COLS,
    PitchTippingClassifier,
    generate_synthetic_biomechanics,
)
from pitch_predictor.vision.video_ingestion import upload_to_unity_catalog_volume


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Pitch Tipping Classifier on biomechanical features with SHAP explanations.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/videos/694973/biomechanical_features.parquet"),
        help="Path to biomechanical features Parquet or CSV file",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("models/pitch_tipping_model.joblib"),
        help="Target local output path for serialized model (default: models/pitch_tipping_model.joblib)",
    )
    parser.add_argument(
        "--n-augment",
        type=int,
        default=40,
        help="Number of augmented/synthetic samples per category to ensure robust calibration (default: 40)",
    )
    parser.add_argument(
        "--catalog",
        type=str,
        default="pitch_predictor",
        help="Databricks Unity Catalog name (default: pitch_predictor)",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default="baseball_ml_dev",
        help="Databricks Unity Catalog schema name (default: baseball_ml_dev)",
    )
    parser.add_argument(
        "--volume",
        type=str,
        default="pitch_data",
        help="Databricks Unity Catalog volume name (default: pitch_data)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="dev",
        help="Databricks CLI profile for authentication (default: dev)",
    )
    parser.add_argument(
        "--sync-catalog",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synchronize trained model to Unity Catalog Volume (default: True)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("[BASEBALL] PITCH TIPPING MODEL TRAINING & SHAP EXPLAINABILITY")
    print(f"Data Source:   {args.data_path}")
    print(f"Output Model:  {args.output_model}")
    print(f"Catalog Sync:  {args.sync_catalog} ({args.catalog}.{args.schema}.{args.volume})")
    print("=" * 60)

    # 1. Load data
    train_dfs = []
    if args.data_path.exists():
        print(f"Loading real biomechanical features from {args.data_path}...")
        if args.data_path.suffix == ".parquet":
            real_df = pd.read_parquet(args.data_path)
        else:
            real_df = pd.read_csv(args.data_path)

        valid_real = real_df.dropna(subset=BIOMECHANICAL_FEATURE_COLS + ["pitch_category"])
        if not valid_real.empty:
            print(f"Loaded {len(valid_real)} real pitch biomechanics records.")
            train_dfs.append(valid_real)
    else:
        print(f"Notice: Data file '{args.data_path}' not found. Training on calibrated synthetic data.")

    # Augment with domain-calibrated distributions
    if args.n_augment > 0:
        synth_df = generate_synthetic_biomechanics(n_samples_per_class=args.n_augment, random_state=42)
        train_dfs.append(synth_df)

    combined_df = pd.concat(train_dfs, ignore_index=True)
    print(f"Total dataset size for training: {len(combined_df)} samples across classes:")
    print(combined_df["pitch_category"].value_counts().to_string())

    # 2. Train model
    print("\nTraining PitchTippingClassifier (Random Forest)...")
    clf = PitchTippingClassifier(random_state=42)
    metrics = clf.fit(combined_df, target_col="pitch_category")

    print("\n--- Model Evaluation Metrics ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1-Macro:  {metrics['f1_macro']:.4f}")
    print(f"Train/Test split: {metrics['train_samples']} / {metrics['test_samples']} samples")

    # 3. SHAP Feature Attribution Demonstration
    print("\n--- SHAP Pitch Tipping Analysis (Example Scenarios) ---")
    test_cases = [
        ("Fastball Delivery (Extended arm, 3/4 slot)", pd.DataFrame([{
            "arm_slot_angle": 50.0,
            "release_height": 0.71,
            "elbow_flexion_angle": 146.0,
            "trunk_tilt": 11.7,
            "stride_extension": 1.45,
        }])),
        ("Sweeper Delivery (Bent elbow, lower arm slot)", pd.DataFrame([{
            "arm_slot_angle": 12.0,
            "release_height": 0.67,
            "elbow_flexion_angle": 98.0,
            "trunk_tilt": 10.0,
            "stride_extension": 1.35,
        }])),
        ("Splitter Delivery (High arm slot, moderate flexion)", pd.DataFrame([{
            "arm_slot_angle": 72.0,
            "release_height": 0.65,
            "elbow_flexion_angle": 108.0,
            "trunk_tilt": 12.0,
            "stride_extension": 1.38,
        }])),
    ]

    for label, sample in test_cases:
        explanation = clf.explain(sample)
        print(f"\nScenario: {label}")
        print(f" -> {explanation['summary_text']}")

    # 4. Save model locally
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    saved_path = clf.save(args.output_model)
    print(f"\n[SUCCESS] Model serialized locally to: {saved_path}")

    # 5. Synchronize to Unity Catalog Volume
    if args.sync_catalog:
        vol_model_path = f"/Volumes/{args.catalog}/{args.schema}/{args.volume}/models/pitch_tipping_model.joblib"
        print(f"\n[CATALOG SYNC] Uploading model to Unity Catalog Volume: {vol_model_path}...")
        success = upload_to_unity_catalog_volume(saved_path, vol_model_path, profile=args.profile)
        if success:
            print(f"[OK] Synced model to Unity Catalog: {vol_model_path}")
        else:
            print(f"[WARN] Notice: Failed to upload model to UC Volume: {vol_model_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
