"""
Statcast Biomechanical Overlay Engine (Passo 4).

Draws a high-visibility, broadcast-grade skeletal overlay directly onto
pitching video frames at release point, showing:
- Throwing arm kinematic chain (Shoulder -> Elbow -> Wrist) in Statcast Neon Cyan
- Trunk / Spine inclination vector in Neon Pink
- Lower-body kinetic chain (Hips -> Knees -> Ankles) in Gold
- Release point targeting reticle
- Broadcast HUD summary badge with exact biomechanical metrics
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Check OpenCV availability
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

# Broadcast Color Palette (BGR format for OpenCV)
COLOR_NEON_CYAN = (204, 255, 0)      # #00FFCC - Throwing arm & release point
COLOR_NEON_PINK = (127, 0, 255)      # #FF007F - Spine & trunk tilt
COLOR_GOLD = (0, 215, 255)           # #FFD700 - Lower body & stride
COLOR_GLOVE_ARM = (235, 206, 135)    # #87CEEB - Glove side
COLOR_WHITE = (255, 255, 255)
COLOR_DARK_HUD = (20, 24, 33)


def _to_pixel_coords(point: dict[str, float] | None, width: int, height: int) -> tuple[int, int] | None:
    """Converts normalized [0, 1] landmark coordinates to pixel space."""
    if not point:
        return None
    x = int(np.clip(point.get("x", 0.0) * width, 0, width - 1))
    y = int(np.clip(point.get("y", 0.0) * height, 0, height - 1))
    return x, y


def draw_pitcher_skeleton(
    frame: np.ndarray,
    landmarks: dict[str | int, Any] | None,
    p_throws: str = "R",
    metrics: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Renders Statcast-styled 3D skeletal overlay and tactical HUD onto an image frame.
    Returns RGB image array ready for Streamlit display.
    """
    if frame is None or not HAS_CV2 or landmarks is None:
        return frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)

    img = frame.copy()
    h, w = img.shape[:2]

    # Convert coordinates
    r_sh = _to_pixel_coords(landmarks.get("RIGHT_SHOULDER"), w, h)
    l_sh = _to_pixel_coords(landmarks.get("LEFT_SHOULDER"), w, h)
    r_el = _to_pixel_coords(landmarks.get("RIGHT_ELBOW"), w, h)
    l_el = _to_pixel_coords(landmarks.get("LEFT_ELBOW"), w, h)
    r_wr = _to_pixel_coords(landmarks.get("RIGHT_WRIST"), w, h)
    l_wr = _to_pixel_coords(landmarks.get("LEFT_WRIST"), w, h)
    r_hip = _to_pixel_coords(landmarks.get("RIGHT_HIP"), w, h)
    l_hip = _to_pixel_coords(landmarks.get("LEFT_HIP"), w, h)
    r_ank = _to_pixel_coords(landmarks.get("RIGHT_ANKLE"), w, h)
    l_ank = _to_pixel_coords(landmarks.get("LEFT_ANKLE"), w, h)

    # 1. Draw Lower Body (Legs & Hips)
    if r_hip and l_hip:
        cv2.line(img, r_hip, l_hip, COLOR_GOLD, 3, cv2.LINE_AA)
        mid_hip = ((r_hip[0] + l_hip[0]) // 2, (r_hip[1] + l_hip[1]) // 2)

        # Trunk vector (Mid-hip to mid-shoulder)
        if r_sh and l_sh:
            mid_sh = ((r_sh[0] + l_sh[0]) // 2, (r_sh[1] + l_sh[1]) // 2)
            cv2.line(img, mid_hip, mid_sh, COLOR_NEON_PINK, 4, cv2.LINE_AA)
            cv2.circle(img, mid_sh, 6, COLOR_NEON_PINK, -1)

    if r_hip and r_ank:
        cv2.line(img, r_hip, r_ank, COLOR_GOLD, 2, cv2.LINE_AA)
    if l_hip and l_ank:
        cv2.line(img, l_hip, l_ank, COLOR_GOLD, 2, cv2.LINE_AA)

    # 2. Draw Shoulders line
    if r_sh and l_sh:
        cv2.line(img, r_sh, l_sh, COLOR_WHITE, 2, cv2.LINE_AA)

    # 3. Draw Arms based on Handedness
    is_righty = p_throws.upper() == "R"
    throw_sh, throw_el, throw_wr = (r_sh, r_el, r_wr) if is_righty else (l_sh, l_el, l_wr)
    glove_sh, glove_el, glove_wr = (l_sh, l_el, l_wr) if is_righty else (r_sh, r_el, r_wr)

    # Glove arm (Subdued)
    if glove_sh and glove_el:
        cv2.line(img, glove_sh, glove_el, COLOR_GLOVE_ARM, 2, cv2.LINE_AA)
        cv2.circle(img, glove_el, 4, COLOR_GLOVE_ARM, -1)
    if glove_el and glove_wr:
        cv2.line(img, glove_el, glove_wr, COLOR_GLOVE_ARM, 2, cv2.LINE_AA)
        cv2.circle(img, glove_wr, 5, COLOR_GLOVE_ARM, -1)

    # Throwing arm (High visibility Neon Cyan)
    if throw_sh and throw_el:
        cv2.line(img, throw_sh, throw_el, COLOR_NEON_CYAN, 4, cv2.LINE_AA)
        cv2.circle(img, throw_sh, 7, COLOR_WHITE, -1)
        cv2.circle(img, throw_el, 8, COLOR_NEON_CYAN, -1)
        cv2.circle(img, throw_el, 12, COLOR_NEON_CYAN, 1, cv2.LINE_AA)

    if throw_el and throw_wr:
        cv2.line(img, throw_el, throw_wr, COLOR_NEON_CYAN, 4, cv2.LINE_AA)
        # Release Point Reticle on throwing wrist
        cv2.circle(img, throw_wr, 10, COLOR_NEON_CYAN, -1)
        cv2.circle(img, throw_wr, 16, COLOR_WHITE, 2, cv2.LINE_AA)
        cv2.drawMarker(img, throw_wr, COLOR_NEON_PINK, cv2.MARKER_CROSS, 20, 2)

    # 4. Statcast HUD Info Card Overlay (Top-Left Corner)
    if metrics:
        card_w, card_h = 260, 115
        hud_box = img[15:15 + card_h, 15:15 + card_w]
        if hud_box.shape[0] == card_h and hud_box.shape[1] == card_w:
            overlay = hud_box.copy()
            cv2.rectangle(overlay, (0, 0), (card_w, card_h), COLOR_DARK_HUD, -1)
            cv2.addWeighted(overlay, 0.75, hud_box, 0.25, 0, hud_box)
            cv2.rectangle(img, (15, 15), (15 + card_w, 15 + card_h), COLOR_NEON_CYAN, 1)

            # Header
            cv2.putText(img, "STATCAST BIOMECHANICS", (25, 33), cv2.FONT_HERSHEY_DUPLEX, 0.45, COLOR_NEON_CYAN, 1, cv2.LINE_AA)
            cv2.line(img, (25, 38), (15 + card_w - 10, 38), COLOR_WHITE, 1)

            # Metric Rows
            arm_slot = metrics.get("arm_slot_angle")
            elbow_fl = metrics.get("elbow_flexion_angle")
            rel_ht = metrics.get("release_height")
            trunk = metrics.get("trunk_tilt")

            y_pos = 55
            if arm_slot is not None:
                cv2.putText(img, f"ARM SLOT: {arm_slot:.1f} deg", (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_WHITE, 1, cv2.LINE_AA)
                y_pos += 18
            if elbow_fl is not None:
                cv2.putText(img, f"ELBOW FLEX: {elbow_fl:.1f} deg", (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_WHITE, 1, cv2.LINE_AA)
                y_pos += 18
            if rel_ht is not None:
                cv2.putText(img, f"RELEASE HT: {rel_ht:.3f} norm", (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_WHITE, 1, cv2.LINE_AA)
                y_pos += 18
            if trunk is not None:
                cv2.putText(img, f"TRUNK TILT: {trunk:.1f} deg", (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_WHITE, 1, cv2.LINE_AA)

    # Return as RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def extract_and_annotate_release_frame(
    video_path: str | Path,
    release_frame_idx: int = 0,
    p_throws: str = "R",
    metrics: dict[str, Any] | None = None,
    extractor: Any | None = None,
    mock_mode: bool = False,
) -> np.ndarray:
    """
    Extracts the release frame from a video clip and overlays the skeletal pose.
    Falls back to a styled synthetic visualization if the video cannot be opened.
    """
    path = Path(video_path)

    if not mock_mode and path.exists() and HAS_CV2:
        try:
            cap = cv2.VideoCapture(str(path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, release_frame_idx)
            ret, frame = cap.read()
            cap.release()

            if ret:
                if extractor is None:
                    from pitch_predictor.vision.pose_extractor import PoseExtractor

                    extractor = PoseExtractor()

                landmarks = extractor.extract_frame_landmarks(frame)
                return draw_pitcher_skeleton(frame, landmarks, p_throws=p_throws, metrics=metrics)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not extract frame from %s: %s. Using synthetic fallback.", path, exc)

    # Synthetic Broadcast Frame Fallback
    synth_canvas = np.full((540, 720, 3), 28, dtype=np.uint8)
    # Add subtle baseball field grass green hue
    synth_canvas[320:, :] = (35, 55, 38)

    from pitch_predictor.vision.pose_extractor import PoseExtractor

    mock_extractor = PoseExtractor(mock_mode=True)
    mock_landmarks = mock_extractor.extract_frame_landmarks(synth_canvas)

    return draw_pitcher_skeleton(synth_canvas, mock_landmarks, p_throws=p_throws, metrics=metrics)
