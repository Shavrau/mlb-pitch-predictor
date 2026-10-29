#!/usr/bin/env python3
"""
CLI Tool for Batch 3D Temporal Biomechanical Pose Extraction from Pitching Video Clips.

Reads a video manifest, extracts 2D screen + 3D metric world landmarks via
MediaPipe Pose (RunningMode.VIDEO), rejects broadcast camera cuts, isolates mound ROI,
segments 4 canonical delivery phases (Leg Lift, Foot Plant, MER, Ball Release),
computes all 10 MLB Pitch Tipping features, and optionally exports per-frame
kinematic timeseries for synchronized UI charts.

Usage:
    python scripts/extract_pitch_poses.py --manifest data/videos/694973/manifest.csv --clips-per-pitcher 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from pitch_predictor.vision.pose_extractor import (
    EXTENDED_BIOMECHANICAL_FEATURE_COLS,
    PoseExtractor,
)
from pitch_predictor.vision.video_ingestion import (
    resolve_volume_video_path,
    upload_to_unity_catalog_volume,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 3D temporal biomechanical pose features from pitch video clips.",
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
        "--clips-per-pitcher",
        type=int,
        default=None,
        help="Maximum number of clips to process per pitcher (optional parameterized scale flag)",
    )
    parser.add_argument(
        "--clips-per-category",
        type=int,
        default=None,
        help="Maximum number of clips to process per pitch category per pitcher",
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        default=2,
        help="Frame stride during temporal video decoding (default: 2)",
    )
    parser.add_argument(
        "--p-throws",
        type=str,
        default="R",
        choices=["R", "L"],
        help="Default pitcher throwing hand ('R' or 'L') when omitted from manifest",
    )
    parser.add_argument(
        "--heavy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer pose_landmarker_heavy.task when available (default: True)",
    )
    parser.add_argument(
        "--download-heavy",
        action="store_true",
        help="Download pose_landmarker_heavy.task automatically if not cached locally",
    )
    parser.add_argument(
        "--save-timeseries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export per-frame smoothed kinematic_timeseries.csv alongside summary features",
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


def _apply_clip_limits(
    df: pd.DataFrame,
    clips_per_pitcher: int | None,
    clips_per_category: int | None,
) -> pd.DataFrame:
    filtered = df.copy()
    group_cols = ["pitcher_id"] if "pitcher_id" in filtered.columns else []

    if clips_per_category is not None and clips_per_category > 0 and "pitch_category" in filtered.columns:
        cat_groups = group_cols + ["pitch_category"]
        filtered = filtered.groupby(cat_groups, as_index=False, group_keys=False).head(clips_per_category)

    if clips_per_pitcher is not None and clips_per_pitcher > 0:
        if group_cols:
            filtered = filtered.groupby(group_cols, as_index=False, group_keys=False).head(clips_per_pitcher)
        else:
            filtered = filtered.head(clips_per_pitcher)

    return filtered.reset_index(drop=True)


def main() -> int:
    args = parse_args()

    if not args.manifest.exists():
        print(f"Error: Manifest file '{args.manifest}' does not exist.")
        return 1

    print(f"Loading manifest from {args.manifest}...")
    manifest_df = (
        pd.read_parquet(args.manifest)
        if args.manifest.suffix == ".parquet"
        else pd.read_csv(args.manifest)
    )

    completed_videos = (
        manifest_df[manifest_df["status"] == "COMPLETED"].copy()
        if "status" in manifest_df.columns
        else manifest_df.copy()
    )
    completed_videos = _apply_clip_limits(
        completed_videos,
        clips_per_pitcher=args.clips_per_pitcher,
        clips_per_category=args.clips_per_category,
    )
    print(f"Processing {len(completed_videos)} videos (out of {len(manifest_df)} manifest rows).")

    if completed_videos.empty:
        print("No completed videos found in manifest to process.")
        return 0

    output_path = args.output or (args.manifest.parent / "biomechanical_features.parquet")

    print(
        f"Initializing PoseExtractor (mock_mode={args.mock}, prefer_heavy={args.heavy}, "
        f"download_heavy={args.download_heavy})..."
    )
    extractor = PoseExtractor(
        mock_mode=args.mock,
        prefer_heavy=args.heavy,
        auto_download_heavy=args.download_heavy,
    )

    summary_records = []
    timeseries_rows = []

    for _, row in completed_videos.iterrows():
        rec = row.to_dict()
        vpath = rec.get("video_path")
        if not vpath and args.video_dir:
            clean_cat = str(rec.get("pitch_category", "")).replace(" ", "_")
            vpath = str(args.video_dir / f"{rec.get('game_pk')}_{rec.get('at_bat_number')}_{rec.get('pitch_number')}_{clean_cat}.mp4")

        row_throws = str(rec.get("p_throws", args.p_throws))
        res = extractor.extract_video_poses(
            video_path=vpath or "",
            p_throws=row_throws,
            sample_step=args.sample_step,
        )
        summary = res.get("summary", {})
        for col in EXTENDED_BIOMECHANICAL_FEATURE_COLS:
            rec[col] = summary.get(col)

        rec["leg_lift_frame"] = summary.get("leg_lift_frame")
        rec["foot_plant_frame"] = summary.get("foot_plant_frame")
        rec["mer_frame"] = summary.get("mer_frame")
        rec["release_frame"] = summary.get("release_frame")
        rec["camera_cut_frame"] = res.get("camera_cut_frame")
        rec["pose_confidence"] = summary.get("confidence")
        summary_records.append(rec)

        if args.save_timeseries:
            for fr in res.get("frames", []):
                timeseries_rows.append({
                    "pitcher_id": rec.get("pitcher_id", 694973),
                    "game_pk": rec.get("game_pk"),
                    "at_bat_number": rec.get("at_bat_number"),
                    "pitch_number": rec.get("pitch_number"),
                    "pitch_category": rec.get("pitch_category"),
                    "frame": fr.get("frame"),
                    "arm_slot_angle": fr.get("arm_angle_smooth", fr.get("arm_angle")),
                    "elbow_flexion_angle": fr.get("flexion"),
                    "glove_tuck_distance": fr.get("glove_tuck_smooth", fr.get("glove_tuck")),
                    "hip_shoulder_separation": fr.get("hip_shoulder_sep"),
                    "arm_angular_velocity": fr.get("arm_vel_deg_s", 0.0),
                    "release_height": fr.get("rel_h"),
                })

    features_df = pd.DataFrame(summary_records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_path, index=False)
    csv_path = output_path.with_suffix(".csv")
    features_df.to_csv(csv_path, index=False)

    print(f"\n[SUCCESS] Extracted 10-feature 3D biomechanics for {len(features_df)} pitches.")
    print(f"Saved Parquet to: {output_path}")
    print(f"Saved CSV to:     {csv_path}")

    if args.save_timeseries and timeseries_rows:
        ts_path = output_path.parent / "kinematic_timeseries.csv"
        pd.DataFrame(timeseries_rows).to_csv(ts_path, index=False)
        print(f"Saved Timeseries: {ts_path} ({len(timeseries_rows)} frame records)")

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

    cols_to_show = [
        "pitch_category",
        "pitch_type",
        "arm_slot_angle",
        "elbow_flexion_angle",
        "glove_tuck_distance",
        "hip_shoulder_separation",
        "arm_angular_velocity",
        "release_frame",
    ]
    available_cols = [c for c in cols_to_show if c in features_df.columns]
    print("\nBiomechanical v2 Summary:")
    print(features_df[available_cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
