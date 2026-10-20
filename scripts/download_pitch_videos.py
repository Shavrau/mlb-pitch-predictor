"""
CLI script to incrementally download MLB pitch video clips from Baseball Savant.

Usage:
    python scripts/download_pitch_videos.py --pitcher-id 694973 --samples-per-class 12
    python scripts/download_pitch_videos.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pybaseball
import requests

from pitch_predictor.vision.video_ingestion import (
    download_pitch_clip,
    filter_balanced_pitch_sample,
    resolve_play_id,
    resolve_savant_video_url,
    resolve_volume_video_path,
    update_video_manifest,
    upload_to_unity_catalog_volume,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("download_pitch_videos")

# Default pilot pitcher: Paul Skenes (Pittsburgh Pirates, #694973)
DEFAULT_PITCHER_ID = 694973
DEFAULT_START_DATE = "2024-05-15"
DEFAULT_END_DATE = "2024-06-15"
DEFAULT_SAMPLES_PER_CLASS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MLB pitch video clips incrementally from Baseball Savant."
    )
    parser.add_argument(
        "--pitcher-id",
        type=int,
        default=DEFAULT_PITCHER_ID,
        help=f"MLB player ID for target pitcher (default: {DEFAULT_PITCHER_ID} - Paul Skenes)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=DEFAULT_START_DATE,
        help=f"Statcast query start date YYYY-MM-DD (default: {DEFAULT_START_DATE})",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=DEFAULT_END_DATE,
        help=f"Statcast query end date YYYY-MM-DD (default: {DEFAULT_END_DATE})",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=DEFAULT_SAMPLES_PER_CLASS,
        help=f"Max samples per pitch category (default: {DEFAULT_SAMPLES_PER_CLASS})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Destination directory for MP4 clips (default: data/videos/{pitcher_id})",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default="",
        help="Path to manifest CSV (default: {output_dir}/manifest.csv)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Query and display selected pitches without downloading files",
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
        help="Synchronize downloaded videos and manifest to Unity Catalog Volume (default: True)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pitcher_id = args.pitcher_id
    output_dir = Path(args.output_dir or f"data/videos/{pitcher_id}")
    manifest_path = Path(args.manifest_path or (output_dir / "manifest.csv"))

    logger.info("=" * 60)
    logger.info("⚾ MLB PITCH VIDEO INGESTION (INCREMENTAL)")
    logger.info("Pitcher ID: %d", pitcher_id)
    logger.info("Query Period: %s to %s", args.start_date, args.end_date)
    logger.info("Samples Per Category: %d", args.samples_per_class)
    logger.info("Output Directory: %s", output_dir)
    logger.info("Manifest Path: %s", manifest_path)
    logger.info("Dry-Run: %s", args.dry_run)
    logger.info("=" * 60)

    # 1. Query Statcast data for target pitcher
    logger.info("Fetching Statcast pitches via PyBaseball...")
    try:
        raw_df = pybaseball.statcast_pitcher(
            args.start_date,
            args.end_date,
            player_id=pitcher_id,
        )
    except (ValueError, KeyError, requests.RequestException, RuntimeError, OSError) as exc:
        logger.error("Failed to query Statcast data: %s", exc)
        return 1

    if raw_df.empty:
        logger.warning("No pitches found for pitcher %d in period %s to %s", pitcher_id, args.start_date, args.end_date)
        return 0

    logger.info("Total pitches fetched from Statcast: %d", len(raw_df))

    # 2. Filter balanced pitch sample
    balanced_df = filter_balanced_pitch_sample(
        raw_df,
        pitcher_id=pitcher_id,
        samples_per_class=args.samples_per_class,
    )

    if balanced_df.empty:
        logger.warning("No valid categorized pitches available after filtering.")
        return 0

    category_counts = balanced_df["pitch_category"].value_counts().to_dict()
    logger.info("Balanced pitch sample selected (%d pitches): %s", len(balanced_df), category_counts)

    if args.dry_run:
        logger.info("[DRY-RUN] Selected Pitches Preview:")
        cols = ["game_pk", "at_bat_number", "pitch_number", "pitch_type", "pitch_category", "game_date"]
        preview_cols = [c for c in cols if c in balanced_df.columns]
        for _, row in balanced_df[preview_cols].iterrows():
            logger.info(" - Game %s | AB #%s | Pitch #%s | %s (%s) | Date: %s",
                        row["game_pk"], row.get("at_bat_number", "N/A"), row["pitch_number"], row["pitch_type"], row["pitch_category"], row.get("game_date", "N/A"))
        logger.info("[DRY-RUN] Completed without downloading any video files.")
        return 0

    # 3. Process downloads incrementally
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_keys = set()
    if manifest_path.exists():
        manifest_df = update_video_manifest(manifest_path, [])
        for _, row in manifest_df.iterrows():
            if str(row.get("status")) == "COMPLETED":
                existing_keys.add((int(row["game_pk"]), int(row.get("at_bat_number", 1)), int(row["pitch_number"])))

    logger.info("Existing downloaded clips in manifest: %d", len(existing_keys))

    new_records: list[dict] = []
    downloaded = 0
    skipped = 0
    failed = 0

    for idx, row in balanced_df.iterrows():
        game_pk = int(row["game_pk"])
        at_bat_number = int(row.get("at_bat_number", 1))
        pitch_number = int(row["pitch_number"])
        pitch_type = str(row.get("pitch_type", "UNK"))
        pitch_category = str(row.get("pitch_category", "Unknown"))

        if (game_pk, at_bat_number, pitch_number) in existing_keys:
            logger.info("[%d/%d] Skipping already downloaded pitch: Game %d, AB #%d, Pitch #%d",
                        idx + 1, len(balanced_df), game_pk, at_bat_number, pitch_number)
            skipped += 1
            continue

        logger.info("[%d/%d] Resolving video for Game %d, AB #%d, Pitch #%d (%s - %s)...",
                    idx + 1, len(balanced_df), game_pk, at_bat_number, pitch_number, pitch_type, pitch_category)

        # Resolve playId
        play_id = resolve_play_id(game_pk, pitch_number, at_bat_number=at_bat_number, pitcher_id=pitcher_id)
        if not play_id:
            logger.warning("Could not resolve playId for Game %d, AB #%d, Pitch #%d", game_pk, at_bat_number, pitch_number)
            new_records.append({
                "game_pk": game_pk,
                "at_bat_number": at_bat_number,
                "pitch_number": pitch_number,
                "pitcher_id": pitcher_id,
                "pitch_type": pitch_type,
                "pitch_category": pitch_category,
                "play_id": None,
                "video_url": None,
                "video_path": None,
                "status": "PLAY_ID_NOT_FOUND",
            })
            failed += 1
            continue

        # Resolve video URL
        video_url = resolve_savant_video_url(play_id)
        if not video_url:
            logger.warning("Could not resolve video URL for playId %s", play_id)
            new_records.append({
                "game_pk": game_pk,
                "at_bat_number": at_bat_number,
                "pitch_number": pitch_number,
                "pitcher_id": pitcher_id,
                "pitch_type": pitch_type,
                "pitch_category": pitch_category,
                "play_id": play_id,
                "video_url": None,
                "video_path": None,
                "status": "VIDEO_URL_NOT_FOUND",
            })
            failed += 1
            continue

        clean_cat = pitch_category.replace(" ", "_")
        filename = f"{game_pk}_{at_bat_number}_{pitch_number}_{clean_cat}.mp4"
        target_path = output_dir / filename

        # Download clip
        logger.info("Downloading clip to %s...", target_path.name)
        success = download_pitch_clip(video_url, target_path)

        vol_path = None
        if success:
            logger.info("✅ Successfully saved locally: %s", target_path.name)
            if args.sync_catalog:
                vol_path = resolve_volume_video_path(
                    catalog=args.catalog,
                    schema=args.schema,
                    volume=args.volume,
                    pitcher_id=pitcher_id,
                    filename=filename,
                )
                logger.info("☁️ Syncing to Unity Catalog Volume: %s...", vol_path)
                upload_to_unity_catalog_volume(target_path, vol_path, profile=args.profile)

            new_records.append({
                "game_pk": game_pk,
                "at_bat_number": at_bat_number,
                "pitch_number": pitch_number,
                "pitcher_id": pitcher_id,
                "pitch_type": pitch_type,
                "pitch_category": pitch_category,
                "play_id": play_id,
                "video_url": video_url,
                "video_path": str(target_path).replace("\\", "/"),
                "volume_path": vol_path,
                "status": "COMPLETED",
            })
            downloaded += 1
        else:
            logger.error("❌ Download failed for: %s", filename)
            new_records.append({
                "game_pk": game_pk,
                "at_bat_number": at_bat_number,
                "pitch_number": pitch_number,
                "pitcher_id": pitcher_id,
                "pitch_type": pitch_type,
                "pitch_category": pitch_category,
                "play_id": play_id,
                "video_url": video_url,
                "video_path": None,
                "volume_path": None,
                "status": "DOWNLOAD_FAILED",
            })
            failed += 1

    # Update manifest
    if new_records:
        update_video_manifest(manifest_path, new_records)
        logger.info("Manifest updated at %s (total records written: %d)", manifest_path, len(new_records))
        if args.sync_catalog:
            manifest_vol_path = resolve_volume_video_path(
                catalog=args.catalog,
                schema=args.schema,
                volume=args.volume,
                pitcher_id=pitcher_id,
                filename="manifest.csv",
            )
            logger.info("☁️ Syncing manifest to Unity Catalog Volume: %s...", manifest_vol_path)
            upload_to_unity_catalog_volume(manifest_path, manifest_vol_path, profile=args.profile)

    logger.info("=" * 60)
    logger.info("🏁 INGESTION RUN FINISHED")
    logger.info("Downloaded: %d | Skipped: %d | Failed: %d | Total: %d",
                downloaded, skipped, failed, len(balanced_df))
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
