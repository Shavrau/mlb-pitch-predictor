"""
Vision module for MLB Pitch Predictor.

Handles video clip ingestion, biomechanical pose estimation,
and pitch tipping classification from video streams.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Check for optional computer vision dependencies
try:
    import cv2  # noqa: F401
    import mediapipe  # noqa: F401

    HAS_VISION_DEPS = True
except ImportError:
    HAS_VISION_DEPS = False

from pitch_predictor.vision.pose_extractor import (
    PoseExtractor,
    batch_extract_poses,
    calculate_arm_slot_angle,
    calculate_elbow_flexion,
    calculate_stride_extension,
    calculate_trunk_tilt,
    detect_release_frame,
)
from pitch_predictor.vision.tipping_model import (
    BIOMECHANICAL_FEATURE_COLS,
    PitchTippingClassifier,
    generate_synthetic_biomechanics,
)
from pitch_predictor.vision.video_ingestion import (
    download_pitch_clip,
    filter_balanced_pitch_sample,
    resolve_play_id,
    resolve_savant_video_url,
    resolve_volume_video_path,
    update_video_manifest,
    upload_to_unity_catalog_volume,
)

__all__ = [
    "BIOMECHANICAL_FEATURE_COLS",
    "HAS_VISION_DEPS",
    "PitchTippingClassifier",
    "PoseExtractor",
    "batch_extract_poses",
    "calculate_arm_slot_angle",
    "calculate_elbow_flexion",
    "calculate_stride_extension",
    "calculate_trunk_tilt",
    "detect_release_frame",
    "download_pitch_clip",
    "filter_balanced_pitch_sample",
    "generate_synthetic_biomechanics",
    "resolve_play_id",
    "resolve_savant_video_url",
    "resolve_volume_video_path",
    "update_video_manifest",
    "upload_to_unity_catalog_volume",
]
