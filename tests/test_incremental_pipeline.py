"""
Unit tests for Incremental Video & Biomechanics Lakeflow Pipeline.

Tests:
1. filter_pending_pitches excludes existing completed pitches from manifest.
2. filter_pending_pitches balances categories among pending candidates.
3. process_incremental_video_batch executes idempotently in mock mode.
4. process_incremental_video_batch detects when dataset is already up-to-date.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from pitch_predictor.vision.incremental_pipeline import (
    filter_pending_pitches,
    process_incremental_video_batch,
)


def test_filter_pending_pitches_excludes_existing_completed():
    """Verify that pitches already completed in the manifest are excluded."""
    candidates_df = pd.DataFrame([
        {"game_pk": 100, "at_bat_number": 1, "pitch_number": 1, "pitch_category": "Fastball", "pitch_type": "FF"},
        {"game_pk": 100, "at_bat_number": 1, "pitch_number": 2, "pitch_category": "Fastball", "pitch_type": "FF"},
        {"game_pk": 100, "at_bat_number": 2, "pitch_number": 1, "pitch_category": "Breaking Ball", "pitch_type": "SL"},
    ])

    manifest_df = pd.DataFrame([
        {"game_pk": 100, "at_bat_number": 1, "pitch_number": 1, "status": "COMPLETED"},
        {"game_pk": 100, "at_bat_number": 1, "pitch_number": 2, "status": "DOWNLOAD_FAILED"},
    ])

    pending = filter_pending_pitches(candidates_df, manifest_df, max_samples=10, balanced=False)

    # Pitch 1 was COMPLETED -> must be excluded
    # Pitch 2 was DOWNLOAD_FAILED -> eligible for retry
    # Pitch 3 was never attempted -> eligible
    assert len(pending) == 2
    assert (100, 1, 1) not in [
        (r["game_pk"], r["at_bat_number"], r["pitch_number"]) for _, r in pending.iterrows()
    ]
    assert (100, 1, 2) in [
        (r["game_pk"], r["at_bat_number"], r["pitch_number"]) for _, r in pending.iterrows()
    ]


def test_filter_pending_pitches_balanced_selection():
    """Verify that pending candidate selection balances across pitch categories."""
    records = []
    # 10 Fastballs
    for i in range(1, 11):
        records.append({"game_pk": 200, "at_bat_number": i, "pitch_number": 1, "pitch_category": "Fastball", "pitch_type": "FF"})
    # 2 Breaking Balls
    for i in range(1, 3):
        records.append({"game_pk": 201, "at_bat_number": i, "pitch_number": 1, "pitch_category": "Breaking Ball", "pitch_type": "ST"})
    # 5 Offspeed
    for i in range(1, 6):
        records.append({"game_pk": 202, "at_bat_number": i, "pitch_number": 1, "pitch_category": "Offspeed", "pitch_type": "FS"})

    df = pd.DataFrame(records)
    pending = filter_pending_pitches(df, manifest_df=None, max_samples=6, balanced=True)

    # Max samples 6 balanced over 3 categories -> ~2 per category
    assert len(pending) == 6
    counts = pending["pitch_category"].value_counts().to_dict()
    assert counts.get("Fastball", 0) == 2
    assert counts.get("Breaking Ball", 0) == 2
    assert counts.get("Offspeed", 0) == 2


def test_process_incremental_video_batch_mock_mode():
    """Verify end-to-end execution of incremental batch processing in mock mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        candidates_df = pd.DataFrame([
            {"game_pk": 300, "at_bat_number": 1, "pitch_number": 1, "pitcher": 694973, "pitch_category": "Fastball", "pitch_type": "FF"},
            {"game_pk": 300, "at_bat_number": 1, "pitch_number": 2, "pitcher": 694973, "pitch_category": "Breaking Ball", "pitch_type": "ST"},
            {"game_pk": 300, "at_bat_number": 2, "pitch_number": 1, "pitcher": 694973, "pitch_category": "Offspeed", "pitch_type": "FS"},
        ])

        summary = process_incremental_video_batch(
            candidates_df=candidates_df,
            output_dir=temp_dir,
            pitcher_id=694973,
            max_pitches=3,
            mock_mode=True,
            sync_models=False,
            sync_catalog=False,
        )

        assert summary["processed"] == 3
        assert summary["status"] == "SUCCESS"

        # Check manifest created
        manifest_file = temp_dir / "manifest.csv"
        assert manifest_file.exists()
        manifest_df = pd.read_csv(manifest_file)
        assert len(manifest_df) == 3
        assert (manifest_df["status"] == "COMPLETED").all()

        # Check biomechanical features created
        parquet_file = temp_dir / "biomechanical_features.parquet"
        assert parquet_file.exists()
        bio_df = pd.read_parquet(parquet_file)
        assert len(bio_df) == 3
        assert "arm_slot_angle" in bio_df.columns
        assert "elbow_flexion_angle" in bio_df.columns


def test_process_incremental_video_batch_up_to_date():
    """Verify that when all candidates are already completed, the runner exits gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        manifest_file = temp_dir / "manifest.csv"
        pd.DataFrame([
            {"game_pk": 400, "at_bat_number": 1, "pitch_number": 1, "status": "COMPLETED"},
        ]).to_csv(manifest_file, index=False)

        candidates_df = pd.DataFrame([
            {"game_pk": 400, "at_bat_number": 1, "pitch_number": 1, "pitcher": 694973, "pitch_category": "Fastball", "pitch_type": "FF"},
        ])

        summary = process_incremental_video_batch(
            candidates_df=candidates_df,
            output_dir=temp_dir,
            pitcher_id=694973,
            max_pitches=5,
            mock_mode=True,
            sync_models=False,
            sync_catalog=False,
        )

        assert summary["processed"] == 0
        assert summary["status"] == "UP_TO_DATE"
