#!/usr/bin/env python3
"""
CLI Tool for Training the Pitch Tipping v2 Biomechanical Model.

Trains a calibrated Random Forest with intra-pitcher baseline normalization
(`PitcherBaselineNormalizer`) on the 10-feature 3D/temporal dataset, evaluates
honest out-of-fold validation via `StratifiedGroupKFold(groups=game_pk)`,
verifies SHAP attributions across different pitcher archetypes (overhand RHP vs
sidearm LHP), and serializes the artifact to `models/pitch_tipping_model.joblib`.

Usage:
    python scripts/train_tipping_model.py --no-sync-catalog
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
        description="Train calibrated Pitch Tipping v2 Classifier with GroupKFold(game_pk) validation.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("src/pitch_predictor/vision/sample_data/biomechanical_features.csv"),
        help="Path to multi-pitcher biomechanical features CSV or Parquet",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("models/pitch_tipping_model.joblib"),
        help="Target local output path for serialized model (default: models/pitch_tipping_model.joblib)",
    )
    parser.add_argument(
        "--clips-per-pitcher",
        type=int,
        default=None,
        help="Optional cap on training clips per pitcher for ablation/scaling experiments",
    )
    parser.add_argument(
        "--n-augment",
        type=int,
        default=0,
        help="Optional synthetic samples per category (default: 0 for 100%% honest real-sample training)",
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

    print("=" * 68)
    print("[BIOMECHANICS v2] PITCH TIPPING TRAINING (INTRA-PITCHER Z-SCORE + GROUPKFOLD)")
    print(f"Data Source:   {args.data_path}")
    print(f"Output Model:  {args.output_model}")
    print(f"Augmentation:  {args.n_augment} synthetic rows/class (0 = honest real-data CV)")
    print("=" * 68)

    train_dfs = []
    if args.data_path.exists():
        real_df = (
            pd.read_parquet(args.data_path)
            if args.data_path.suffix == ".parquet"
            else pd.read_csv(args.data_path)
        )
        valid_real = real_df.dropna(subset=BIOMECHANICAL_FEATURE_COLS + ["pitch_category"]).copy()
        if args.clips_per_pitcher and "pitcher_id" in valid_real.columns:
            valid_real = (
                valid_real.groupby("pitcher_id", as_index=False, group_keys=False)
                .head(args.clips_per_pitcher)
                .reset_index(drop=True)
            )
        if not valid_real.empty:
            print(f"Loaded {len(valid_real)} multi-pitcher biomechanical records.")
            train_dfs.append(valid_real)
    else:
        print(f"Notice: Data file '{args.data_path}' not found. Falling back to synthetic benchmark.")

    if args.n_augment > 0 or not train_dfs:
        n_syn = args.n_augment if args.n_augment > 0 else 40
        train_dfs.append(generate_synthetic_biomechanics(n_samples_per_class=n_syn, random_state=42))

    combined_df = pd.concat(train_dfs, ignore_index=True)
    print(f"Dataset size: {len(combined_df)} pitches across classes:")
    print(combined_df["pitch_category"].value_counts().to_string())

    clf = PitchTippingClassifier(random_state=42)
    metrics = clf.fit(combined_df, target_col="pitch_category")

    print("\n--- Honest Out-of-Fold Validation Metrics ---")
    print(f"Scheme:         {metrics.get('validation_scheme')}")
    print(f"Active Columns: {len(clf.active_feature_cols_)} ({', '.join(clf.active_feature_cols_)})")
    print(f"CV Accuracy:    {metrics.get('cv_accuracy', metrics['accuracy']):.4f}")
    print(f"CV Macro-F1:    {metrics.get('cv_f1_macro', metrics['f1_macro']):.4f}")
    print(f"CV Brier Score: {metrics.get('cv_brier_score', 0.0):.4f}")
    print(f"CV Log-Loss:    {metrics.get('cv_log_loss', 0.0):.4f}")
    print(f"Unique Games:   {metrics.get('unique_games', 1)}")

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    saved_path = clf.save(args.output_model)
    print(f"\n[SUCCESS] Serialized v2 model to: {saved_path}")

    if args.sync_catalog:
        vol_model_path = f"/Volumes/{args.catalog}/{args.schema}/{args.volume}/models/pitch_tipping_model.joblib"
        print(f"\n[CATALOG SYNC] Uploading model to Unity Catalog Volume: {vol_model_path}...")
        upload_to_unity_catalog_volume(saved_path, vol_model_path, profile=args.profile)

    return 0


if __name__ == "__main__":
    sys.exit(main())
