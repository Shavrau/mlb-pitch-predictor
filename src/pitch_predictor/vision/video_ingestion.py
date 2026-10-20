"""
Video Ingestion module for MLB Pitch Predictor.

Handles retrieving official MLB/Baseball Savant video clips,
balanced sampling across pitch categories, and maintaining an
incremental video manifest for Databricks Volume or local storage.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from pitch_predictor.ingestion import map_pitch_category

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"


def filter_balanced_pitch_sample(
    df: pd.DataFrame,
    pitcher_id: int,
    samples_per_class: int = 12,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Filters Statcast pitches for a target pitcher and samples a balanced set per category.

    Args:
        df: Input Statcast DataFrame.
        pitcher_id: MLB Player ID of the target pitcher.
        samples_per_class: Maximum number of pitches per category (Fastball, Breaking Ball, Offspeed).
        random_state: Random seed for sampling reproducibility.

    Returns:
        Balanced DataFrame containing sampled pitches.
    """
    if df.empty:
        return pd.DataFrame()

    pitcher_df = df[df["pitcher"] == pitcher_id].copy()
    if pitcher_df.empty:
        logger.warning("No pitches found for pitcher %d", pitcher_id)
        return pd.DataFrame()

    if "pitch_category" not in pitcher_df.columns:
        pitcher_df["pitch_category"] = pitcher_df["pitch_type"].apply(map_pitch_category)

    valid_df = pitcher_df[pitcher_df["pitch_category"].notna()].copy()
    categories = ["Fastball", "Breaking Ball", "Offspeed"]
    sampled_dfs = []

    for category in categories:
        cat_df = valid_df[valid_df["pitch_category"] == category]
        n_available = len(cat_df)
        if n_available == 0:
            logger.warning("No %s pitches available for pitcher %d", category, pitcher_id)
            continue

        n_samples = min(samples_per_class, n_available)
        sampled = cat_df.sample(n=n_samples, random_state=random_state)
        sampled_dfs.append(sampled)

    if not sampled_dfs:
        return pd.DataFrame()

    return pd.concat(sampled_dfs, ignore_index=True)


def resolve_play_id(
    game_pk: int,
    pitch_number: int,
    at_bat_number: int | None = None,
    pitcher_id: int | None = None,
    timeout: int = 10,
) -> str | None:
    """
    Resolves the unique playId (UUID) for a pitch event from the MLB Stats API live feed.

    Args:
        game_pk: MLB Game ID.
        pitch_number: Pitch number within the at-bat.
        at_bat_number: Optional 1-indexed at-bat number (corresponds to atBatIndex + 1).
        pitcher_id: Optional MLB pitcher ID to verify the pitcher on the mound.
        timeout: HTTP request timeout in seconds.

    Returns:
        String playId if found, otherwise None.
    """
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    headers = {"User-Agent": DEFAULT_USER_AGENT}

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code != 200:
            logger.warning("MLB Stats API returned status %d for game_pk %d", response.status_code, game_pk)
            return None

        data = response.json()
        all_plays = data.get("liveData", {}).get("plays", {}).get("allPlays", [])

        # Direct index lookup if at_bat_number is provided
        if at_bat_number is not None:
            target_idx = at_bat_number - 1
            if 0 <= target_idx < len(all_plays):
                candidate_play = all_plays[target_idx]
                for ev in candidate_play.get("playEvents", []):
                    if ev.get("isPitch") and ev.get("pitchNumber") == pitch_number:
                        play_id = ev.get("playId")
                        if play_id:
                            return str(play_id)

        # Fallback: scan allPlays
        for play in all_plays:
            if pitcher_id is not None:
                match_pitcher = play.get("matchup", {}).get("pitcher", {}).get("id")
                if match_pitcher != pitcher_id:
                    continue

            for ev in play.get("playEvents", []):
                if ev.get("isPitch") and ev.get("pitchNumber") == pitch_number:
                    play_id = ev.get("playId")
                    if play_id:
                        return str(play_id)

    except requests.RequestException as exc:
        logger.error("Network error while resolving playId for game_pk %d: %s", game_pk, exc)
    except (ValueError, KeyError) as exc:
        logger.error("JSON parsing error for game_pk %d: %s", game_pk, exc)

    return None


def resolve_savant_video_url(play_id: str, timeout: int = 10) -> str | None:
    """
    Retrieves the direct MP4 video link from Baseball Savant's sporty-videos endpoint.

    Args:
        play_id: UUID of the pitch event.
        timeout: HTTP request timeout in seconds.

    Returns:
        Direct MP4 URL if found, otherwise None.
    """
    url = f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"
    headers = {"User-Agent": DEFAULT_USER_AGENT}

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code != 200:
            logger.warning("Baseball Savant returned status %d for playId %s", response.status_code, play_id)
            return None

        # Regex search for sporty-clips MP4 URLs
        matches = re.findall(r'https?://[^\s"\'<>]+\.mp4', response.text)
        if matches:
            return matches[0]

    except requests.RequestException as exc:
        logger.error("Network error retrieving sporty-video for playId %s: %s", play_id, exc)

    return None


def download_pitch_clip(
    video_url: str,
    target_path: Path | str,
    timeout: int = 20,
    chunk_size: int = 65536,
) -> bool:
    """
    Streams and downloads an MP4 video clip to target_path atomically.

    Args:
        video_url: Direct MP4 URL.
        target_path: Local or volume destination path.
        timeout: HTTP request timeout in seconds.
        chunk_size: Byte chunk size for streaming.

    Returns:
        True if download succeeded, False otherwise.
    """
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_suffix(".tmp")
    headers = {"User-Agent": DEFAULT_USER_AGENT}

    try:
        response = requests.get(video_url, headers=headers, stream=True, timeout=timeout)
        if response.status_code != 200:
            logger.warning("Download failed with status %d for %s", response.status_code, video_url)
            return False

        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

        # Atomic rename after full download
        temp_path.replace(target)
        return True

    except (requests.RequestException, OSError) as exc:
        logger.error("Failed to download clip from %s to %s: %s", video_url, target, exc)
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        return False


def update_video_manifest(
    manifest_path: Path | str,
    new_records: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Appends new video records to the manifest file idempotently.

    Deduplicates records by (game_pk, at_bat_number, pitch_number).

    Args:
        manifest_path: Path to the CSV or Parquet manifest file.
        new_records: List of dictionaries with video metadata.

    Returns:
        Consolidated pandas DataFrame.
    """
    manifest_p = Path(manifest_path)
    manifest_p.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "pitcher_id",
        "pitch_type",
        "pitch_category",
        "play_id",
        "video_url",
        "video_path",
        "status",
    ]

    new_df = pd.DataFrame(new_records)
    if new_df.empty:
        if manifest_p.exists():
            return pd.read_csv(manifest_p) if manifest_p.suffix == ".csv" else pd.read_parquet(manifest_p)
        return pd.DataFrame(columns=columns)

    if manifest_p.exists():
        existing_df = pd.read_csv(manifest_p) if manifest_p.suffix == ".csv" else pd.read_parquet(manifest_p)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    # Ensure at_bat_number column exists
    if "at_bat_number" not in combined_df.columns:
        combined_df["at_bat_number"] = 1

    # Deduplicate keeping latest record
    combined_df = combined_df.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"], keep="last")

    # Save to disk
    if manifest_p.suffix == ".parquet":
        combined_df.to_parquet(manifest_p, index=False)
    else:
        combined_df.to_csv(manifest_p, index=False)

    return combined_df
