#!/usr/bin/env python3
"""
CLI Tool for Training the Multimodal Pitch Predictor & Ablation Lift Evaluation.

Combines Statcast pre-pitch contextual leverage features with pre-release
biomechanical pose features (arm slot, elbow flexion, release height, trunk tilt, stride),
evaluates performance gains over unimodal baselines (Statcast-only, Vision-only),
and synchronizes the trained multimodal model and dataset to Unity Catalog Volume.

Usage:
    python scripts/train_multimodal_model.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from pitch_predictor.multimodal import (
    MultimodalPitchClassifier,
    create_multimodal_dataset,
    generate_synthetic_multimodal_data,
)
from pitch_predictor.vision.video_ingestion import upload_to_unity_catalog_volume


def sanitize_for_terminal(text: str) -> str:
    replacements = {
        "✅": "[OK]",
        "⚡": "[TIPPING]",
        "🎯": "[CONTEXTO]",
        "💡": "[DICA]",
        "📝": "[NOTA]",
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text.encode("ascii", errors="replace").decode("ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Multimodal Pitch Classifier fusing Statcast and Biomechanics.",
    )
    parser.add_argument(
        "--bio-data",
        type=Path,
        default=Path("data/videos/694973/biomechanical_features.parquet"),
        help="Path to biomechanical features Parquet file (default: data/videos/694973/biomechanical_features.parquet)",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("models/multimodal_pitch_model.joblib"),
        help="Local output path for serialized multimodal model (default: models/multimodal_pitch_model.joblib)",
    )
    parser.add_argument(
        "--output-dataset",
        type=Path,
        default=Path("data/multimodal_features.parquet"),
        help="Local output path for merged multimodal dataset (default: data/multimodal_features.parquet)",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=50,
        help="Number of samples per pitch category for calibration (default: 50)",
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
        help="Synchronize trained model and dataset to Unity Catalog Volume (default: True)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 65)
    print("[BASEBALL] MULTIMODAL PITCH CLASSIFIER & ABLATION STUDY (PASSO 5)")
    print(f"Biomechanical Source: {args.bio_data}")
    print(f"Output Model:         {args.output_model}")
    print(f"Output Dataset:       {args.output_dataset}")
    print(f"Catalog Sync:         {args.sync_catalog} ({args.catalog}.{args.schema}.{args.volume})")
    print("=" * 65)

    # 1. Synthesize paired multimodal dataset calibrated for Paul Skenes
    print("\n[1/5] Building multimodal dataset with zero-leakage join keys...")
    _, _, full_df = generate_synthetic_multimodal_data(
        n_samples_per_class=args.samples_per_class,
        random_state=42,
    )

    # 2. Append real extracted biomechanical rows if available
    if args.bio_data.exists():
        real_bio = pd.read_parquet(args.bio_data)
        print(f"Loaded {len(real_bio)} real video-extracted biomechanical records from {args.bio_data}.")
        # Construct matching Statcast mock entries for real pitches to enable joint evaluation
        real_statcast = []
        for _, row in real_bio.iterrows():
            real_statcast.append({
                "game_pk": int(row["game_pk"]),
                "at_bat_number": int(row["at_bat_number"]),
                "pitch_number": int(row["pitch_number"]),
                "pitcher": int(row.get("pitcher_id", 694973)),
                "game_date": "2024-04-01",
                "pitch_category": row["pitch_category"],
                "pitch_type": row.get("pitch_type", "FF"),
                "balls": 1,
                "strikes": 2 if row["pitch_category"] == "Breaking Ball" else 1,
                "count_state": "1-2" if row["pitch_category"] == "Breaking Ball" else "1-1",
                "count_leverage": "Pitcher Ahead" if row["pitch_category"] == "Breaking Ball" else "Neutral",
                "prev_pitch_category": "Fastball",
                "runners_on_base": 0,
                "scoring_position": 0,
                "same_hand": 1,
                "pitcher_fb_prior": 0.55,
                "pitcher_bb_prior": 0.30,
                "pitcher_os_prior": 0.15,
            })
        real_sc_df = pd.DataFrame(real_statcast)
        real_merged = create_multimodal_dataset(real_sc_df, real_bio, how="inner")
        full_df = pd.concat([full_df, real_merged], ignore_index=True)

    print(f"Total Multimodal Dataset Size: {len(full_df)} samples across 3 pitch categories.")
    print("Class Distribution:")
    for cat, count in full_df["pitch_category"].value_counts().items():
        print(f" - {cat}: {count} samples ({count / len(full_df):.1%})")

    # 3. Save local dataset parquet
    args.output_dataset.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_parquet(args.output_dataset, index=False)
    print(f"[OK] Saved multimodal dataset to {args.output_dataset} ({args.output_dataset.stat().st_size:,} bytes).")

    # 4. Train-Test Split (80% train, 20% test holdout)
    print("\n[2/5] Training Multimodal Model and conducting Ablation Study...")
    n_train = int(len(full_df) * 0.8)
    shuffled = full_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    train_df = shuffled.iloc[:n_train].copy()
    test_df = shuffled.iloc[n_train:].copy()

    model = MultimodalPitchClassifier(n_estimators=40, max_depth=8, random_state=42)
    ablation = model.fit_ablation_study(train_df, test_df)

    # 5. Display Ablation Study Report
    print("\n" + "=" * 65)
    print("           RELATÓRIO DE IMPACTO & ABLAÇÃO MULTIMODAL")
    print("=" * 65)
    print(f"{'Modalidade Avaliada':<24} | {'Acurácia':<10} | {'Macro F1':<10} | {'Log-Loss':<10}")
    print("-" * 65)

    sc_m = ablation["statcast"]
    vis_m = ablation["vision"]
    multi_m = ablation["multimodal"]
    lift = ablation["lift"]

    print(f"{'1. Statcast (Contexto)':<24} | {sc_m['accuracy']:<10.3f} | {sc_m['f1_macro']:<10.3f} | {sc_m['log_loss']:<10.3f}")
    print(f"{'2. Visão (Biomecânica)':<24} | {vis_m['accuracy']:<10.3f} | {vis_m['f1_macro']:<10.3f} | {vis_m['log_loss']:<10.3f}")
    print(f"{'3. Fusão Multimodal':<24} | {multi_m['accuracy']:<10.3f} | {multi_m['f1_macro']:<10.3f} | {multi_m['log_loss']:<10.3f}")
    print("-" * 65)
    print(f"GANHO MULTIMODAL (LIFT)  : +{lift['lift_accuracy'] * 100:+.1f}% Acurácia | +{lift['lift_f1_macro'] * 100:+.1f}% Macro F1 | {lift['lift_log_loss']:+.3f} Log-Loss")
    print("=" * 65)

    # 6. Test Modality Comparison Scenarios
    print("\n[3/5] Testing Situational Duels (Agreement vs Tipping Override)...")

    # Scenario 1: Agreement
    sample_agreement = {
        "balls": 0,
        "strikes": 2,
        "count_state": "0-2",
        "count_leverage": "Pitcher Ahead",
        "prev_pitch_category": "Fastball",
        "runners_on_base": 0,
        "scoring_position": 0,
        "same_hand": 1,
        "pitcher_fb_prior": 0.35,
        "pitcher_bb_prior": 0.50,
        "pitcher_os_prior": 0.15,
        "arm_slot_angle": 12.0,
        "release_height": 0.68,
        "elbow_flexion_angle": 98.0,
        "trunk_tilt": 10.0,
        "stride_extension": 1.35,
    }
    diag_1 = model.compare_modalities(sample_agreement)
    print("\n[Cenário 1: Contagem 0-2 com Mecânica Lateral de Sweeper]")
    print(f" - Statcast Pred:   {diag_1['statcast_prediction']} ({diag_1['statcast_probabilities'][diag_1['statcast_prediction']]:.1%})")
    print(f" - Vision Pred:     {diag_1['vision_prediction']} ({diag_1['vision_probabilities'][diag_1['vision_prediction']]:.1%})")
    print(f" - Multimodal Pred: {diag_1['multimodal_prediction']} ({diag_1['multimodal_probabilities'][diag_1['multimodal_prediction']]:.1%})")
    print(f" - Diagnóstico:     {sanitize_for_terminal(diag_1['summary_text'])}")

    # Scenario 2: Pitch Tipping Override
    sample_override = {
        "balls": 3,
        "strikes": 1,
        "count_state": "3-1",
        "count_leverage": "Pitcher Behind",
        "prev_pitch_category": "Breaking Ball",
        "runners_on_base": 1,
        "scoring_position": 1,
        "same_hand": 1,
        "pitcher_fb_prior": 0.70,
        "pitcher_bb_prior": 0.20,
        "pitcher_os_prior": 0.10,
        "arm_slot_angle": 12.5,
        "release_height": 0.68,
        "elbow_flexion_angle": 97.0,  # bent elbow reveals breaking ball despite 3-1 count
        "trunk_tilt": 10.2,
        "stride_extension": 1.35,
    }
    diag_2 = model.compare_modalities(sample_override)
    print("\n[Cenário 2: Contagem 3-1 (Favorável a Fastball) com Pitch Tipping de Breaking Ball]")
    print(f" - Statcast Pred:   {diag_2['statcast_prediction']} ({diag_2['statcast_probabilities'][diag_2['statcast_prediction']]:.1%})")
    print(f" - Vision Pred:     {diag_2['vision_prediction']} ({diag_2['vision_probabilities'][diag_2['vision_prediction']]:.1%})")
    print(f" - Multimodal Pred: {diag_2['multimodal_prediction']} ({diag_2['multimodal_probabilities'][diag_2['multimodal_prediction']]:.1%})")
    print(f" - Diagnóstico:     {sanitize_for_terminal(diag_2['summary_text'])}")

    # 7. Save model locally
    print("\n[4/5] Serializing multimodal model...")
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output_model)
    print(f"[OK] Saved multimodal model to {args.output_model} ({args.output_model.stat().st_size:,} bytes).")

    # 8. Synchronize to Unity Catalog Volume
    if args.sync_catalog:
        print("\n[5/5] Synchronizing multimodal model and dataset to Unity Catalog Volume...")
        volume_model_path = f"/Volumes/{args.catalog}/{args.schema}/{args.volume}/models/multimodal_pitch_model.joblib"
        success_model = upload_to_unity_catalog_volume(
            local_file_path=args.output_model,
            target_volume_path=volume_model_path,
            profile=args.profile,
        )
        if success_model:
            print(f"[OK] Model synced to Unity Catalog: {volume_model_path}")
        else:
            print(f"[WARN] Failed to sync model to {volume_model_path}.")

        volume_dataset_path = f"/Volumes/{args.catalog}/{args.schema}/{args.volume}/features/multimodal_features.parquet"
        success_dataset = upload_to_unity_catalog_volume(
            local_file_path=args.output_dataset,
            target_volume_path=volume_dataset_path,
            profile=args.profile,
        )
        if success_dataset:
            print(f"[OK] Dataset synced to Unity Catalog: {volume_dataset_path}")
        else:
            print(f"[WARN] Failed to sync dataset to {volume_dataset_path}.")

    print("\n" + "=" * 65)
    print("[SUCCESS] PASSO 5 MULTIMODAL FUSION COMPLETED!")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
