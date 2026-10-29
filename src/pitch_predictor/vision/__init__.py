"""
Vision module for MLB Pitch Predictor.

Handles video clip ingestion, 3D temporal biomechanical pose estimation,
intra-pitcher baseline normalization, and calibrated pitch tipping classification.
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

from pitch_predictor.multimodal import (
    MULTIMODAL_FEATURE_COLS,
    MULTIMODAL_JOIN_KEYS,
    MultimodalPitchClassifier,
    create_multimodal_dataset,
    generate_synthetic_multimodal_data,
)
from pitch_predictor.vision.incremental_pipeline import (
    filter_pending_pitches,
    process_incremental_video_batch,
)
from pitch_predictor.vision.overlay import (
    draw_pitcher_skeleton,
    extract_and_annotate_release_frame,
)
from pitch_predictor.vision.pose_extractor import (
    EXTENDED_BIOMECHANICAL_FEATURE_COLS,
    PitcherROITracker,
    PoseExtractor,
    batch_extract_poses,
    calculate_arm_slot_angle,
    calculate_elbow_flexion,
    calculate_glove_tuck_distance,
    calculate_hip_shoulder_separation,
    calculate_stride_extension,
    calculate_trunk_tilt,
    detect_camera_cut_frame,
    detect_delivery_phases,
    detect_release_frame,
    smooth_kinematic_series,
)
from pitch_predictor.vision.tipping_model import (
    BIOMECHANICAL_FEATURE_COLS,
    PitcherBaselineNormalizer,
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
    "EXTENDED_BIOMECHANICAL_FEATURE_COLS",
    "HAS_VISION_DEPS",
    "MULTIMODAL_FEATURE_COLS",
    "MULTIMODAL_JOIN_KEYS",
    "MultimodalPitchClassifier",
    "PitchTippingClassifier",
    "PitcherBaselineNormalizer",
    "PitcherROITracker",
    "PoseExtractor",
    "batch_extract_poses",
    "calculate_arm_slot_angle",
    "calculate_elbow_flexion",
    "calculate_glove_tuck_distance",
    "calculate_hip_shoulder_separation",
    "calculate_stride_extension",
    "calculate_trunk_tilt",
    "create_multimodal_dataset",
    "detect_camera_cut_frame",
    "detect_delivery_phases",
    "detect_release_frame",
    "download_pitch_clip",
    "draw_pitcher_skeleton",
    "extract_and_annotate_release_frame",
    "filter_balanced_pitch_sample",
    "filter_pending_pitches",
    "generate_synthetic_biomechanics",
    "generate_synthetic_multimodal_data",
    "process_incremental_video_batch",
    "resolve_play_id",
    "resolve_savant_video_url",
    "resolve_volume_video_path",
    "smooth_kinematic_series",
    "update_video_manifest",
    "upload_to_unity_catalog_volume",
]
