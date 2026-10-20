#!/usr/bin/env python3
"""
CLI Tool for Batch Biomechanical Pose Extraction from Pitching Video Clips.

Reads the video manifest produced in Passo 1, extracts 3D skeletal landmarks
via MediaPipe Pose, detects release frames, calculates pitching biomechanics
(arm slot, release height, elbow flexion, trunk tilt, stride extension),
and saves the consolidated dataset as Parquet and CSV.

Usage:
    python scripts/extract_pitch_poses.py --manifest data/videos/694973/manifest.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from pitch_predictor.vision.pose_extractor import PoseExtractor, batch_extract_poses
from pitch_predictor.vision.video_ingestion import (
    resolve_volume_video_path,
    upload_to_unity_catalog_volume,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract biomechanical pose features from pitch video clips.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/videos/694973/manifest.csv"),
        help="Path to video manifest CSV or Parquet (default: data/videos/694973/manifest.csv)",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("data/videos/694973"),
        help="Directory containing the video files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Target output Parquet path (defaults to manifest_dir/biomechanical_features.parquet)",
    )
    parser.add_argument(
        "--p-throws",
        type=str,
        default="R",
        choices=["R", "L"],
        help="Pitcher throwing hand ('R' or 'L')",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in synthetic mock mode without processing video files",
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
        help="Databricks CLI authentication profile (default: dev)",
    )
    parser.add_argument(
        "--sync-catalog",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synchronize extracted features to Unity Catalog Volume (default: True)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.manifest.exists():
        print(f"Error: Manifest file '{args.manifest}' does not exist.")
        return 1

    print(f"Loading manifest from {args.manifest}...")
    if args.manifest.suffix == ".parquet":
        manifest_df = pd.read_parquet(args.manifest)
    else:
        manifest_df = pd.read_csv(args.manifest)

    completed_videos = manifest_df[manifest_df["status"] == "COMPLETED"]
    print(f"Found {len(completed_videos)} completed videos to process out of {len(manifest_df)} total records.")

    if completed_videos.empty:
        print("No completed videos found in manifest to process.")
        return 0

    output_path = args.output
    if output_path is None:
        output_path = args.manifest.parent / "biomechanical_features.parquet"

    print(f"Initializing PoseExtractor (mock_mode={args.mock})...")
    extractor = PoseExtractor(mock_mode=args.mock)

    print("Extracting poses and calculating biomechanical features...")
    features_df = batch_extract_poses(
        manifest_df=completed_videos,
        video_dir=args.video_dir,
        p_throws=args.p_throws,
        extractor=extractor,
    )

    # Save to Parquet and CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_path, index=False)
    csv_path = output_path.with_suffix(".csv")
    features_df.to_csv(csv_path, index=False)

    print(f"\n[SUCCESS] Extracted features for {len(features_df)} pitches.")
    print(f"Saved Parquet to: {output_path}")
    print(f"Saved CSV to:     {csv_path}")

    # Synchronize to Unity Catalog Volume
    if args.sync_catalog:
        pitcher_id = None
        if "pitcher_id" in features_df.columns and not features_df["pitcher_id"].empty:
            pitcher_id = int(features_df["pitcher_id"].iloc[0])

        vol_parquet_path = resolve_volume_video_path(
            catalog=args.catalog,
            schema=args.schema,
            volume=args.volume,
            pitcher_id=pitcher_id,
            filename="biomechanical_features.parquet",
        )
        vol_csv_path = resolve_volume_video_path(
            catalog=args.catalog,
            schema=args.schema,
            volume=args.volume,
            pitcher_id=pitcher_id,
            filename="biomechanical_features.csv",
        )

        print("\n[CATALOG SYNC] Uploading features to Unity Catalog Volume...")
        upload_to_unity_catalog_volume(output_path, vol_parquet_path, profile=args.profile)
        upload_to_unity_catalog_volume(csv_path, vol_csv_path, profile=args.profile)
        print(f"Synced Parquet to: {vol_parquet_path}")
        print(f"Synced CSV to:     {vol_csv_path}")

    # Display preview summary table
    cols_to_show = [
        "pitch_category",
        "pitch_type",
        "arm_slot_angle",
        "release_height",
        "elbow_flexion_angle",
        "trunk_tilt",
        "release_frame",
    ]
    available_cols = [c for c in cols_to_show if c in features_df.columns]
    print("\nBiomechanical Summary:")
    print(features_df[available_cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
