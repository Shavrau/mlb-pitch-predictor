"""
Unit tests for Baseball Savant / MLB video ingestion and manifest management.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from pitch_predictor.vision.video_ingestion import (
    download_pitch_clip,
    filter_balanced_pitch_sample,
    resolve_play_id,
    resolve_savant_video_url,
    update_video_manifest,
)


@pytest.fixture
def sample_statcast_df() -> pd.DataFrame:
    """Creates a sample Statcast DataFrame with diverse pitch categories."""
    return pd.DataFrame(
        [
            # Paul Skenes (694973) pitches
            {"game_pk": 745494, "at_bat_number": 1, "pitch_number": 1, "pitcher": 694973, "pitch_type": "FF", "game_date": "2024-06-05"},
            {"game_pk": 745494, "at_bat_number": 1, "pitch_number": 2, "pitcher": 694973, "pitch_type": "FF", "game_date": "2024-06-05"},
            {"game_pk": 745494, "at_bat_number": 2, "pitch_number": 1, "pitcher": 694973, "pitch_type": "SL", "game_date": "2024-06-05"},
            {"game_pk": 745494, "at_bat_number": 2, "pitch_number": 2, "pitcher": 694973, "pitch_type": "ST", "game_date": "2024-06-05"},
            {"game_pk": 745494, "at_bat_number": 3, "pitch_number": 1, "pitcher": 694973, "pitch_type": "CH", "game_date": "2024-06-05"},
            {"game_pk": 745494, "at_bat_number": 3, "pitch_number": 2, "pitcher": 694973, "pitch_type": "FS", "game_date": "2024-06-05"},
            # Different pitcher
            {"game_pk": 745494, "at_bat_number": 4, "pitch_number": 1, "pitcher": 999999, "pitch_type": "FF", "game_date": "2024-06-05"},
        ]
    )


def test_filter_balanced_pitch_sample(sample_statcast_df: pd.DataFrame) -> None:
    """Validates that filter_balanced_pitch_sample samples evenly across categories."""
    filtered = filter_balanced_pitch_sample(
        sample_statcast_df,
        pitcher_id=694973,
        samples_per_class=1,
    )

    assert len(filtered) == 3
    assert set(filtered["pitch_category"].unique()) == {"Fastball", "Breaking Ball", "Offspeed"}
    assert (filtered["pitcher"] == 694973).all()


def test_resolve_play_id_found_with_at_bat() -> None:
    """Validates direct at_bat_number lookup in MLB Stats API live feed."""
    mock_payload = {
        "liveData": {
            "plays": {
                "allPlays": [
                    {
                        "about": {"atBatIndex": 0},
                        "matchup": {"pitcher": {"id": 694973}},
                        "playEvents": [
                            {"isPitch": False, "pitchNumber": None},
                            {
                                "isPitch": True,
                                "pitchNumber": 1,
                                "playId": "mock-play-uuid-ab1",
                            },
                        ],
                    }
                ]
            }
        }
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_payload

    with patch("requests.get", return_value=mock_resp):
        play_id = resolve_play_id(game_pk=745494, pitch_number=1, at_bat_number=1, pitcher_id=694973)

    assert play_id == "mock-play-uuid-ab1"


def test_resolve_play_id_found_fallback() -> None:
    """Validates fallback scan in MLB Stats API when at_bat_number is omitted."""
    mock_payload = {
        "liveData": {
            "plays": {
                "allPlays": [
                    {
                        "matchup": {"pitcher": {"id": 694973}},
                        "playEvents": [
                            {
                                "isPitch": True,
                                "pitchNumber": 2,
                                "playId": "mock-play-uuid-scan",
                            },
                        ],
                    }
                ]
            }
        }
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_payload

    with patch("requests.get", return_value=mock_resp):
        play_id = resolve_play_id(game_pk=745494, pitch_number=2, pitcher_id=694973)

    assert play_id == "mock-play-uuid-scan"


def test_resolve_play_id_not_found() -> None:
    """Validates playId resolution returns None when pitch does not exist."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"liveData": {"plays": {"allPlays": []}}}

    with patch("requests.get", return_value=mock_resp):
        play_id = resolve_play_id(game_pk=745494, pitch_number=99, pitcher_id=694973)

    assert play_id is None


def test_resolve_savant_video_url_success() -> None:
    """Validates extraction of sporty-clips mp4 URL from sporty-videos HTML."""
    html_content = """
    <html>
      <body>
        <video id="sporty">
          <source src="https://sporty-clips.mlb.com/sample_pitch_clip.mp4" type="video/mp4">
        </video>
      </body>
    </html>
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = html_content

    with patch("requests.get", return_value=mock_resp):
        url = resolve_savant_video_url(play_id="mock-play-uuid-1234")

    assert url == "https://sporty-clips.mlb.com/sample_pitch_clip.mp4"


def test_resolve_savant_video_url_missing() -> None:
    """Validates None is returned when no MP4 link is present in sporty-videos."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "<html><body>No video available</body></html>"

    with patch("requests.get", return_value=mock_resp):
        url = resolve_savant_video_url(play_id="missing-play-uuid")

    assert url is None


def test_download_pitch_clip_success(tmp_path: Path) -> None:
    """Validates successful binary stream write of MP4 file."""
    target_file = tmp_path / "pitch_clip.mp4"
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_content.return_value = [b"\x00\x00\x00\x18ftypmp42", b"binarydata"]

    with patch("requests.get", return_value=mock_resp):
        success = download_pitch_clip(
            video_url="https://sporty-clips.mlb.com/clip.mp4",
            target_path=target_file,
        )

    assert success is True
    assert target_file.exists()
    assert target_file.stat().st_size > 0


def test_download_pitch_clip_failure(tmp_path: Path) -> None:
    """Validates download failure handling without leaving corrupted file."""
    target_file = tmp_path / "failed_clip.mp4"

    with patch("requests.get", side_effect=requests.RequestException("Network error")):
        success = download_pitch_clip(
            video_url="https://sporty-clips.mlb.com/fail.mp4",
            target_path=target_file,
        )

    assert success is False
    assert not target_file.exists()


def test_update_video_manifest_idempotency(tmp_path: Path) -> None:
    """Validates manifest file creation, appending, and deduplication by (game_pk, at_bat_number, pitch_number)."""
    manifest_file = tmp_path / "video_manifest.csv"

    record_1 = {
        "game_pk": 745494,
        "at_bat_number": 1,
        "pitch_number": 1,
        "pitcher_id": 694973,
        "pitch_type": "FF",
        "pitch_category": "Fastball",
        "video_path": "data/videos/694973/745494_1_1_Fastball.mp4",
        "status": "COMPLETED",
    }
    record_2 = {
        "game_pk": 745494,
        "at_bat_number": 1,
        "pitch_number": 2,
        "pitcher_id": 694973,
        "pitch_type": "SL",
        "pitch_category": "Breaking Ball",
        "video_path": "data/videos/694973/745494_1_2_Breaking_Ball.mp4",
        "status": "COMPLETED",
    }

    # Initial write
    df_first = update_video_manifest(manifest_file, [record_1, record_2])
    assert len(df_first) == 2
    assert manifest_file.exists()

    # Attempt to write duplicate record_1 along with a new record_3
    record_3 = {
        "game_pk": 745494,
        "at_bat_number": 2,
        "pitch_number": 1,
        "pitcher_id": 694973,
        "pitch_type": "CH",
        "pitch_category": "Offspeed",
        "video_path": "data/videos/694973/745494_2_1_Offspeed.mp4",
        "status": "COMPLETED",
    }
    df_second = update_video_manifest(manifest_file, [record_1, record_3])

    assert len(df_second) == 3
    # Check that record_1 is not duplicated
    assert ((df_second["at_bat_number"] == 1) & (df_second["pitch_number"] == 1)).sum() == 1
    assert ((df_second["at_bat_number"] == 2) & (df_second["pitch_number"] == 1)).sum() == 1
