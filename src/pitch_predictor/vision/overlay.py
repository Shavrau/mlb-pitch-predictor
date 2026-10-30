"""
Hawk-Eye / Statcast Biomechanical Overlay Engine v2.

Draws a technical, broadcast-grade 3D kinematic overlay onto pitching video frames:
- Throwing arm kinematic chain (Shoulder -> Elbow -> Wrist) + Arm Slot Angular Arc
- Glove Tuck caliper vector (Chest Center -> Glove Wrist)
- Hip-Shoulder torsional axes & Spine inclination vector
- Release point precision reticle
- Technical Hawk-Eye Telemetry HUD badge
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# Check OpenCV availability
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

# Technical Broadcast Palette (BGR format for OpenCV)
COLOR_HAWKEYE_CYAN = (232, 212, 34)   # #22D4E8 - Throwing arm & release reticle
COLOR_SPINE_RED = (68, 68, 239)       # #EF4444 - Spine & trunk axis
COLOR_AMBER = (11, 158, 245)          # #F59E0B - Pelvis, lower kinetic chain & angle arc
COLOR_GLOVE_BLUE = (248, 189, 56)     # #38BDF8 - Glove arm & glove tuck caliper
COLOR_WHITE = (244, 246, 250)
COLOR_MUTED = (165, 175, 190)
COLOR_DARK_HUD = (14, 17, 23)

# Adaptive Biomechanical Heatmap Palette (BGR for OpenCV, RGB for Pillow)
# Normal: Cyan / Emerald (#22D4E8) - |Z| < 1.0
COLOR_HEATMAP_NORMAL_BGR = (232, 212, 34)
COLOR_HEATMAP_NORMAL_RGB = (34, 212, 232)

# Attention / Warning: Amber (#F59E0B) - 1.0 <= |Z| < 1.75
COLOR_HEATMAP_WARNING_BGR = (11, 158, 245)
COLOR_HEATMAP_WARNING_RGB = (245, 158, 11)

# Pitch Tipping Anomaly: Pulsing Ruby Red (#EF4444) - |Z| >= 1.75
COLOR_HEATMAP_ANOMALY_BGR = (68, 68, 239)
COLOR_HEATMAP_ANOMALY_RGB = (239, 68, 68)

_BASE_FRAME_LANDMARKS_CACHE: dict[tuple[Any, ...], dict[str | int, Any]] = {}


def get_joint_color_by_z(z_val: float | None, bgr: bool = True) -> tuple[int, int, int]:
    """
    Returns adaptive joint heatmap color based on intra-pitcher standardized Z-score:
    - Normal (|Z| < 1.0): Cyan / Emerald
    - Attention (1.0 <= |Z| < 1.75): Amber
    - Pitch Tipping (|Z| >= 1.75): Ruby Red
    """
    if z_val is None or not math.isfinite(z_val):
        return COLOR_HEATMAP_NORMAL_BGR if bgr else COLOR_HEATMAP_NORMAL_RGB
    abs_z = abs(z_val)
    if abs_z >= 1.75:
        return COLOR_HEATMAP_ANOMALY_BGR if bgr else COLOR_HEATMAP_ANOMALY_RGB
    if abs_z >= 1.0:
        return COLOR_HEATMAP_WARNING_BGR if bgr else COLOR_HEATMAP_WARNING_RGB
    return COLOR_HEATMAP_NORMAL_BGR if bgr else COLOR_HEATMAP_NORMAL_RGB


def _extract_z_scores_from_metrics(metrics: dict[str, Any] | None) -> dict[str, float | None]:
    """Extracts standardized Z-scores dictionary from metrics payload."""
    if not metrics:
        return {}
    z_map = metrics.get("z_scores")
    if isinstance(z_map, dict):
        return z_map
    return {
        "arm_slot_angle": metrics.get("z_slot", metrics.get("z_arm_slot")),
        "elbow_flexion_angle": metrics.get("z_elbow", metrics.get("z_elbow_flexion")),
        "release_height": metrics.get("z_ht", metrics.get("z_release_height")),
        "trunk_tilt": metrics.get("z_tilt", metrics.get("z_trunk_tilt")),
        "glove_tuck_distance": metrics.get("z_glove", metrics.get("z_glove_tuck")),
        "hip_shoulder_separation": metrics.get("z_sep", metrics.get("z_hip_shoulder_sep")),
        "arm_angular_velocity": metrics.get("z_vel"),
        "delivery_tempo_s": metrics.get("z_tempo"),
        "stride_extension": metrics.get("z_stride"),
    }



def is_valid_pitcher_landmarks(landmarks: dict[str | int, Any] | None) -> bool:
    """
    Validates whether detected landmarks correspond to a pitcher on the mound
    rather than background actors (batter, catcher, umpire at home plate).
    In standard centerfield broadcast framing, the mound pitcher's shoulders
    are positioned vertically at y >= 0.35.
    """
    if not landmarks:
        return False
    r_sh = landmarks.get("RIGHT_SHOULDER")
    l_sh = landmarks.get("LEFT_SHOULDER")
    r_y = float(r_sh.get("y", 0.0)) if isinstance(r_sh, dict) else 0.0
    l_y = float(l_sh.get("y", 0.0)) if isinstance(l_sh, dict) else 0.0
    return max(r_y, l_y) >= 0.35


def get_calibrated_mound_landmarks(
    p_throws: str = "R",
    pitch_category: str | None = None,
) -> dict[str, dict[str, float]]:
    """
    Provides calibrated 3D baseline landmarks for a pitcher on the mound.
    Guarantees that technical kinematic overlays remain focused on the mound pitcher
    even when single-frame detection fails or is contaminated by home plate actors.
    """
    is_righty = str(p_throws).upper() != "L"
    cat = str(pitch_category or "Fastball").lower()

    if "break" in cat:
        rel_y, elb_y = 0.495, 0.525
    elif "off" in cat:
        rel_y, elb_y = 0.470, 0.505
    else:
        rel_y, elb_y = 0.455, 0.490

    if is_righty:
        return {
            "RIGHT_SHOULDER": {"x": 0.465, "y": 0.515, "z": 0.0, "visibility": 0.99},
            "LEFT_SHOULDER": {"x": 0.505, "y": 0.515, "z": 0.0, "visibility": 0.99},
            "RIGHT_ELBOW": {"x": 0.435, "y": elb_y, "z": -0.15, "visibility": 0.95},
            "RIGHT_WRIST": {"x": 0.418, "y": rel_y, "z": -0.18, "visibility": 0.90},
            "LEFT_ELBOW": {"x": 0.515, "y": 0.535, "z": -0.10, "visibility": 0.85},
            "LEFT_WRIST": {"x": 0.528, "y": 0.550, "z": -0.15, "visibility": 0.80},
            "RIGHT_HIP": {"x": 0.455, "y": 0.608, "z": 0.0, "visibility": 0.99},
            "LEFT_HIP": {"x": 0.475, "y": 0.606, "z": 0.0, "visibility": 0.99},
            "RIGHT_ANKLE": {"x": 0.462, "y": 0.765, "z": 0.0, "visibility": 0.95},
            "LEFT_ANKLE": {"x": 0.512, "y": 0.738, "z": 0.0, "visibility": 0.90},
        }
    else:
        return {
            "LEFT_SHOULDER": {"x": 0.510, "y": 0.515, "z": 0.0, "visibility": 0.99},
            "RIGHT_SHOULDER": {"x": 0.470, "y": 0.515, "z": 0.0, "visibility": 0.99},
            "LEFT_ELBOW": {"x": 0.540, "y": elb_y, "z": -0.15, "visibility": 0.95},
            "LEFT_WRIST": {"x": 0.558, "y": rel_y, "z": -0.18, "visibility": 0.90},
            "RIGHT_ELBOW": {"x": 0.460, "y": 0.535, "z": -0.10, "visibility": 0.85},
            "RIGHT_WRIST": {"x": 0.448, "y": 0.550, "z": -0.15, "visibility": 0.80},
            "LEFT_HIP": {"x": 0.500, "y": 0.608, "z": 0.0, "visibility": 0.99},
            "RIGHT_HIP": {"x": 0.480, "y": 0.606, "z": 0.0, "visibility": 0.99},
            "LEFT_ANKLE": {"x": 0.515, "y": 0.765, "z": 0.0, "visibility": 0.95},
            "RIGHT_ANKLE": {"x": 0.465, "y": 0.738, "z": 0.0, "visibility": 0.90},
        }


def _to_pixel_coords(point: dict[str, float] | None, width: int, height: int) -> tuple[int, int] | None:
    """Converts normalized [0, 1] landmark coordinates to pixel space."""
    if not point:
        return None
    x = int(np.clip(point.get("x", 0.0) * width, 0, width - 1))
    y = int(np.clip(point.get("y", 0.0) * height, 0, height - 1))
    return x, y


def find_preloaded_base_frame(
    video_path: str | Path | None = None,
    pitch_category: str | None = None,
    game_pk: int | str | None = None,
    at_bat_number: int | str | None = None,
    pitch_number: int | str | None = None,
    pitcher_id: int | str | None = None,
    exact_match_only: bool = False,
) -> np.ndarray | None:
    """
    Looks for a pre-generated clean broadcast release frame (the base print before pose annotation).
    Returns RGB image array or None if not found.
    """
    candidate_dirs = [
        Path(__file__).parent / "sample_data" / "release_frames",
        Path("src/pitch_predictor/vision/sample_data/release_frames"),
        Path("/app/python/source_code/src/pitch_predictor/vision/sample_data/release_frames"),
    ]

    names_to_try: list[str] = []
    if game_pk and at_bat_number and pitch_number and pitch_category:
        clean_cat = str(pitch_category).strip().replace(" ", "_")
        if pitcher_id:
            names_to_try.append(f"base_{pitcher_id}_{game_pk}_{at_bat_number}_{pitch_number}_{clean_cat}.jpg")
        names_to_try.append(f"base_{game_pk}_{at_bat_number}_{pitch_number}_{clean_cat}.jpg")

    if video_path:
        stem = Path(video_path).stem
        if pitcher_id:
            names_to_try.append(f"base_{pitcher_id}_{stem}.jpg")
        names_to_try.append(f"base_{stem}.jpg")

    if not exact_match_only:
        if pitch_category:
            clean_cat = str(pitch_category).strip().replace(" ", "_")
            if pitcher_id:
                names_to_try.append(f"base_{pitcher_id}_{clean_cat}.jpg")
            names_to_try.append(f"base_{clean_cat}.jpg")

        if pitcher_id:
            names_to_try.append(f"base_{pitcher_id}_Fastball.jpg")
        names_to_try.append("base_Fastball.jpg")

    for d in candidate_dirs:
        if not d.exists():
            continue
        for name in names_to_try:
            target = d / name
            if target.exists() and target.is_file():
                try:
                    if HAS_CV2:
                        bgr = cv2.imread(str(target))
                        if bgr is not None:
                            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    return np.array(Image.open(target).convert("RGB"))
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Failed reading base frame candidate %s: %s", target, exc)

    return None


def find_preloaded_release_frame(
    video_path: str | Path | None = None,
    pitch_category: str | None = None,
    game_pk: int | str | None = None,
    at_bat_number: int | str | None = None,
    pitch_number: int | str | None = None,
    pitcher_id: int | str | None = None,
    exact_match_only: bool = False,
) -> np.ndarray | None:
    """
    Looks for a pre-generated broadcast release frame image with Hawk-Eye/Statcast annotations.
    Returns RGB image array or None if not found.
    """
    candidate_dirs = [
        Path(__file__).parent / "sample_data" / "release_frames",
        Path("src/pitch_predictor/vision/sample_data/release_frames"),
        Path("/app/python/source_code/src/pitch_predictor/vision/sample_data/release_frames"),
    ]

    names_to_try: list[str] = []
    if game_pk and at_bat_number and pitch_number and pitch_category:
        clean_cat = str(pitch_category).strip().replace(" ", "_")
        if pitcher_id:
            names_to_try.append(f"{pitcher_id}_{game_pk}_{at_bat_number}_{pitch_number}_{clean_cat}.jpg")
            names_to_try.append(f"{pitcher_id}_{game_pk}_{at_bat_number}_{pitch_number}_{pitch_category}.jpg")
        names_to_try.append(f"{game_pk}_{at_bat_number}_{pitch_number}_{clean_cat}.jpg")
        names_to_try.append(f"{game_pk}_{at_bat_number}_{pitch_number}_{pitch_category}.jpg")

    if video_path:
        stem = Path(video_path).stem
        if pitcher_id:
            names_to_try.append(f"{pitcher_id}_{stem}.jpg")
        names_to_try.append(f"{stem}.jpg")

    if not exact_match_only:
        if pitch_category:
            clean_cat = str(pitch_category).strip().replace(" ", "_")
            if pitcher_id:
                names_to_try.append(f"{pitcher_id}_{clean_cat}.jpg")
                names_to_try.append(f"{pitcher_id}_{pitch_category}.jpg")
            names_to_try.append(f"{clean_cat}.jpg")
            names_to_try.append(f"{pitch_category}.jpg")

        if pitcher_id:
            names_to_try.append(f"{pitcher_id}_Fastball.jpg")
        names_to_try.extend(["745170_31_4_Fastball.jpg", "Fastball.jpg"])

    for d in candidate_dirs:
        if not d.exists():
            continue
        for name in names_to_try:
            target = d / name
            if target.exists() and target.is_file():
                try:
                    if HAS_CV2:
                        bgr = cv2.imread(str(target))
                        if bgr is not None:
                            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    return np.array(Image.open(target).convert("RGB"))
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Failed reading release frame candidate %s: %s", target, exc)

    return None


def draw_pitcher_skeleton_pillow(
    frame: np.ndarray,
    landmarks: dict[str | int, Any] | None,
    p_throws: str = "R",
    metrics: dict[str, Any] | None = None,
) -> np.ndarray:
    """Fallback rendering engine using Pillow when OpenCV is unavailable."""
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    def _to_px(pt: dict[str, float] | None) -> tuple[int, int] | None:
        if not pt:
            return None
        return int(np.clip(pt.get("x", 0.0) * w, 0, w - 1)), int(np.clip(pt.get("y", 0.0) * h, 0, h - 1))

    if landmarks:
        r_sh = _to_px(landmarks.get("RIGHT_SHOULDER"))
        l_sh = _to_px(landmarks.get("LEFT_SHOULDER"))
        r_el = _to_px(landmarks.get("RIGHT_ELBOW"))
        l_el = _to_px(landmarks.get("LEFT_ELBOW"))
        r_wr = _to_px(landmarks.get("RIGHT_WRIST"))
        l_wr = _to_px(landmarks.get("LEFT_WRIST"))
        r_hip = _to_px(landmarks.get("RIGHT_HIP"))
        l_hip = _to_px(landmarks.get("LEFT_HIP"))
        r_ank = _to_px(landmarks.get("RIGHT_ANKLE"))
        l_ank = _to_px(landmarks.get("LEFT_ANKLE"))

        z_scores = _extract_z_scores_from_metrics(metrics)
        c_slot = get_joint_color_by_z(z_scores.get("arm_slot_angle"), bgr=False)
        c_elbow = get_joint_color_by_z(z_scores.get("elbow_flexion_angle"), bgr=False)
        c_ht = get_joint_color_by_z(z_scores.get("release_height"), bgr=False)
        c_tilt = get_joint_color_by_z(z_scores.get("trunk_tilt"), bgr=False)
        c_glove = get_joint_color_by_z(z_scores.get("glove_tuck_distance"), bgr=False)
        c_sep = get_joint_color_by_z(z_scores.get("hip_shoulder_separation"), bgr=False)

        c_white = (244, 246, 250)

        mid_hip = None
        mid_sh = None
        if r_hip and l_hip:
            draw.line([r_hip, l_hip], fill=c_sep, width=3)
            mid_hip = ((r_hip[0] + l_hip[0]) // 2, (r_hip[1] + l_hip[1]) // 2)
            if r_sh and l_sh:
                mid_sh = ((r_sh[0] + l_sh[0]) // 2, (r_sh[1] + l_sh[1]) // 2)
                draw.line([mid_hip, mid_sh], fill=c_tilt, width=3)

        if r_hip and r_ank:
            draw.line([r_hip, r_ank], fill=c_sep, width=2)
        if l_hip and l_ank:
            draw.line([l_hip, l_ank], fill=c_sep, width=2)

        if r_sh and l_sh:
            draw.line([r_sh, l_sh], fill=c_white, width=2)

        is_righty = p_throws.upper() == "R"
        throw_sh, throw_el, throw_wr = (r_sh, r_el, r_wr) if is_righty else (l_sh, l_el, l_wr)
        glove_wr = l_wr if is_righty else r_wr

        if mid_sh and mid_hip and glove_wr:
            chest_pt = (int(0.7 * mid_sh[0] + 0.3 * mid_hip[0]), int(0.7 * mid_sh[1] + 0.3 * mid_hip[1]))
            draw.line([chest_pt, glove_wr], fill=c_glove, width=2)

        if throw_sh and throw_el:
            draw.line([throw_sh, throw_el], fill=c_slot, width=4)
        if throw_el and throw_wr:
            draw.line([throw_el, throw_wr], fill=c_elbow, width=4)
            rx, ry = throw_wr
            draw.ellipse([rx - 10, ry - 10, rx + 10, ry + 10], outline=c_white, width=2)
            draw.line([(rx - 15, ry), (rx + 15, ry)], fill=c_ht, width=2)
            draw.line([(rx, ry - 15), (rx, ry + 15)], fill=c_ht, width=2)

    if metrics:
        draw.rectangle([14, 14, 285, 138], fill=(14, 17, 23), outline=(34, 212, 232), width=1)
        draw.text((24, 20), "HAWK-EYE 3D KINEMATICS", fill=(34, 212, 232))
        y = 42
        for k, lbl, unit in [
            ("arm_slot_angle", "ARM SLOT", "deg"),
            ("elbow_flexion_angle", "ELBOW FLEX", "deg"),
            ("release_height", "RELEASE HT", "norm"),
            ("glove_tuck_distance", "GLOVE TUCK", "m"),
        ]:
            if k in metrics and metrics[k] is not None:
                fmt = f"{metrics[k]:.3f}" if k == "release_height" else f"{metrics[k]:.2f}"
                draw.text((24, y), f"{lbl}: {fmt} {unit}", fill=(244, 246, 250))
                y += 18

    return np.array(img)


def draw_pitcher_skeleton(
    frame: np.ndarray,
    landmarks: dict[str | int, Any] | None,
    p_throws: str = "R",
    metrics: dict[str, Any] | None = None,
    is_bgr: bool = True,
) -> np.ndarray:
    """
    Renders Hawk-Eye / Statcast 3D skeletal overlay, Arm Slot Arc, Glove Tuck
    caliper vector, Hip-Shoulder torsional axes, and telemetry HUD onto an image frame.
    Returns RGB image array ready for Streamlit display.
    """
    if frame is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    if not HAS_CV2:
        return draw_pitcher_skeleton_pillow(frame, landmarks, p_throws=p_throws, metrics=metrics)

    if landmarks is None:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame

    # Maintain exact color space: convert to BGR for OpenCV drawing if input was RGB
    img = (
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if not is_bgr and len(frame.shape) == 3
        else frame.copy()
    )
    h, w = img.shape[:2]

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

    mid_hip: tuple[int, int] | None = None
    mid_sh: tuple[int, int] | None = None

    # Resolve adaptive joint heatmap colors from standardized Z-scores
    z_scores = _extract_z_scores_from_metrics(metrics)
    c_slot = get_joint_color_by_z(z_scores.get("arm_slot_angle"), bgr=True)
    c_elbow = get_joint_color_by_z(z_scores.get("elbow_flexion_angle"), bgr=True)
    c_ht = get_joint_color_by_z(z_scores.get("release_height"), bgr=True)
    c_tilt = get_joint_color_by_z(z_scores.get("trunk_tilt"), bgr=True)
    c_glove = get_joint_color_by_z(z_scores.get("glove_tuck_distance"), bgr=True)
    c_sep = get_joint_color_by_z(z_scores.get("hip_shoulder_separation"), bgr=True)

    # 1. Pelvic Hip Axis & Lower Kinetic Chain
    if r_hip and l_hip:
        cv2.line(img, r_hip, l_hip, c_sep, 3, cv2.LINE_AA)
        cv2.circle(img, r_hip, 4, c_sep, -1)
        cv2.circle(img, l_hip, 4, c_sep, -1)
        mid_hip = ((r_hip[0] + l_hip[0]) // 2, (r_hip[1] + l_hip[1]) // 2)

        if r_sh and l_sh:
            mid_sh = ((r_sh[0] + l_sh[0]) // 2, (r_sh[1] + l_sh[1]) // 2)
            # Vertical reference tick at mid_hip + actual Trunk Tilt spine vector
            cv2.line(img, mid_hip, (mid_hip[0], max(0, mid_sh[1] - 12)), COLOR_MUTED, 1, cv2.LINE_AA)
            cv2.line(img, mid_hip, mid_sh, c_tilt, 3, cv2.LINE_AA)
            cv2.circle(img, mid_sh, 5, c_tilt, -1)

    if r_hip and r_ank:
        cv2.line(img, r_hip, r_ank, c_sep, 2, cv2.LINE_AA)
    if l_hip and l_ank:
        cv2.line(img, l_hip, l_ank, c_sep, 2, cv2.LINE_AA)

    # 2. Scapular Shoulder Axis
    if r_sh and l_sh:
        cv2.line(img, r_sh, l_sh, COLOR_WHITE, 2, cv2.LINE_AA)

    # 3. Handedness-Aware Arm Chains & Technical Vectors
    is_righty = p_throws.upper() == "R"
    throw_sh, throw_el, throw_wr = (r_sh, r_el, r_wr) if is_righty else (l_sh, l_el, l_wr)
    glove_sh, glove_el, glove_wr = (l_sh, l_el, l_wr) if is_righty else (r_sh, r_el, r_wr)

    # Glove Arm + Glove Tuck Caliper Vector (Chest Center -> Glove Wrist)
    if glove_sh and glove_el:
        cv2.line(img, glove_sh, glove_el, COLOR_GLOVE_BLUE, 2, cv2.LINE_AA)
        cv2.circle(img, glove_el, 4, COLOR_GLOVE_BLUE, -1)
    if glove_el and glove_wr:
        cv2.line(img, glove_el, glove_wr, COLOR_GLOVE_BLUE, 2, cv2.LINE_AA)
        cv2.circle(img, glove_wr, 5, c_glove, -1)

    if mid_sh and mid_hip and glove_wr:
        chest_pt = (
            round(0.70 * mid_sh[0] + 0.30 * mid_hip[0]),
            round(0.70 * mid_sh[1] + 0.30 * mid_hip[1]),
        )
        cv2.line(img, chest_pt, glove_wr, c_glove, 2, cv2.LINE_AA)
        cv2.circle(img, chest_pt, 4, c_glove, -1)

    # Technical Arm Slot Angular Arc at Throwing Shoulder
    if throw_sh and throw_wr:
        arc_r = max(22, int(min(w, h) * 0.045))
        horiz_end = (throw_sh[0] + (arc_r if is_righty else -arc_r), throw_sh[1])
        cv2.line(img, throw_sh, horiz_end, COLOR_MUTED, 1, cv2.LINE_AA)

        dx = throw_wr[0] - throw_sh[0]
        dy = throw_wr[1] - throw_sh[1]
        ang_deg = math.degrees(math.atan2(dy, dx))
        start_a, end_a = (min(0.0, ang_deg), max(0.0, ang_deg)) if is_righty else (min(180.0, ang_deg), max(180.0, ang_deg))
        cv2.ellipse(img, throw_sh, (arc_r, arc_r), 0.0, start_a, end_a, c_slot, 2, cv2.LINE_AA)

    # Throwing Arm Chain + Precision Release Point Reticle
    if throw_sh and throw_el:
        cv2.line(img, throw_sh, throw_el, c_slot, 4, cv2.LINE_AA)
        cv2.circle(img, throw_sh, 6, COLOR_WHITE, -1)
        cv2.circle(img, throw_el, 7, c_elbow, -1)
        cv2.circle(img, throw_el, 11, c_elbow, 1, cv2.LINE_AA)
        z_el = z_scores.get("elbow_flexion_angle")
        if z_el is not None and abs(z_el) >= 1.75:
            cv2.circle(img, throw_el, 16, COLOR_HEATMAP_ANOMALY_BGR, 2, cv2.LINE_AA)

    if throw_el and throw_wr:
        cv2.line(img, throw_el, throw_wr, c_elbow, 4, cv2.LINE_AA)
        cv2.circle(img, throw_wr, 8, c_ht, -1)
        cv2.circle(img, throw_wr, 15, COLOR_WHITE, 2, cv2.LINE_AA)
        cv2.drawMarker(img, throw_wr, c_ht, cv2.MARKER_CROSS, 22, 2)
        z_h = z_scores.get("release_height")
        if z_h is not None and abs(z_h) >= 1.75:
            cv2.circle(img, throw_wr, 21, COLOR_HEATMAP_ANOMALY_BGR, 2, cv2.LINE_AA)

    # 4. Hawk-Eye 3D Telemetry HUD Badge (Top-Left Corner)
    if metrics:
        rows_to_draw: list[tuple[str, tuple[int, int, int]]] = []
        for k, lbl, unit in [
            ("arm_slot_angle", "ARM SLOT", "deg"),
            ("elbow_flexion_angle", "ELBOW FLEX", "deg"),
            ("release_height", "RELEASE HT", "norm"),
            ("trunk_tilt", "TRUNK TILT", "deg"),
            ("glove_tuck_distance", "GLOVE TUCK", "m"),
            ("hip_shoulder_separation", "HIP-SH SEP", "deg"),
        ]:
            if metrics.get(k) is not None:
                v = float(metrics[k])
                fmt = f"{v:5.3f}" if "norm" in unit else f"{v:5.1f}"
                zv = z_scores.get(k)
                pill_txt = ""
                pill_col = COLOR_WHITE
                if zv is not None and math.isfinite(zv):
                    abs_zv = abs(zv)
                    if abs_zv >= 1.75:
                        pill_txt = " [TIP]"
                        pill_col = COLOR_HEATMAP_ANOMALY_BGR
                    elif abs_zv >= 1.0:
                        pill_txt = " [WARN]"
                        pill_col = COLOR_HEATMAP_WARNING_BGR
                    else:
                        pill_txt = " [NORM]"
                        pill_col = COLOR_HEATMAP_NORMAL_BGR
                rows_to_draw.append((f"{lbl}: {fmt} {unit}{pill_txt}", pill_col))

        card_w = min(w - 24, 272)
        card_h = min(h - 24, 46 + 17 * len(rows_to_draw))
        if card_w > 80 and card_h > 40:
            hud_box = img[14 : 14 + card_h, 14 : 14 + card_w]
            if hud_box.shape[0] == card_h and hud_box.shape[1] == card_w:
                overlay = hud_box.copy()
                cv2.rectangle(overlay, (0, 0), (card_w, card_h), COLOR_DARK_HUD, -1)
                cv2.addWeighted(overlay, 0.82, hud_box, 0.18, 0, hud_box)
                cv2.rectangle(img, (14, 14), (14 + card_w, 14 + card_h), COLOR_HAWKEYE_CYAN, 1)
                cv2.rectangle(img, (14, 14), (17, 14 + card_h), COLOR_AMBER, -1)

                cv2.putText(
                    img,
                    "HAWK-EYE 3D KINEMATICS",
                    (24, 31),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.42,
                    COLOR_HAWKEYE_CYAN,
                    1,
                    cv2.LINE_AA,
                )
                cv2.line(img, (24, 36), (14 + card_w - 10, 36), COLOR_MUTED, 1)

                y_pos = 52
                for line_txt, line_col in rows_to_draw:
                    if y_pos < 14 + card_h - 4:
                        cv2.putText(
                            img,
                            line_txt,
                            (24, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.39,
                            line_col,
                            1,
                            cv2.LINE_AA,
                        )
                        y_pos += 17

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def extract_and_annotate_release_frame(
    video_path: str | Path,
    release_frame_idx: int = 0,
    p_throws: str = "R",
    metrics: dict[str, Any] | None = None,
    extractor: Any | None = None,
    mock_mode: bool = False,
    pitch_category: str | None = None,
    game_pk: int | str | None = None,
    at_bat_number: int | str | None = None,
    pitch_number: int | str | None = None,
    pitcher_id: int | str | None = None,
) -> np.ndarray:
    """
    Extracts the release frame from a video clip and overlays the 3D Hawk-Eye pose.
    Prioritizes pre-rendered HD broadcast release frames for instant cloud resilience;
    falls back to live video extraction or a styled synthetic canvas.
    """
    path = Path(video_path) if video_path else Path("")

    if not mock_mode:
        # Check for clean broadcast base frame to render dynamic anatomical heatmap
        base_frame = find_preloaded_base_frame(
            video_path=video_path,
            pitch_category=pitch_category,
            game_pk=game_pk,
            at_bat_number=at_bat_number,
            pitch_number=pitch_number,
            pitcher_id=pitcher_id,
            exact_match_only=False,
        )
        if base_frame is not None:
            # find_preloaded_base_frame returns RGB; convert to BGR for OpenCV drawing
            base_frame_bgr = (
                cv2.cvtColor(base_frame, cv2.COLOR_RGB2BGR)
                if HAS_CV2 and len(base_frame.shape) == 3
                else base_frame
            )
            h, w = base_frame_bgr.shape[:2]
            cache_key = (
                str(pitcher_id),
                str(game_pk),
                str(at_bat_number),
                str(pitch_number),
                str(pitch_category),
                str(p_throws),
            )
            landmarks = _BASE_FRAME_LANDMARKS_CACHE.get(cache_key)

            if landmarks is None or not is_valid_pitcher_landmarks(landmarks):
                landmarks = None
                if extractor is None:
                    from pitch_predictor.vision.pose_extractor import PoseExtractor
                    extractor = PoseExtractor()

                # 1. Primary isolation: isolate mound ROI with PitcherROITracker (rejects home plate actors)
                if w >= 640 and h >= 360:
                    from pitch_predictor.vision.pose_extractor import PitcherROITracker
                    roi = PitcherROITracker()
                    crop, bbox = roi.crop_frame(base_frame_bgr)
                    crop_landmarks = extractor.extract_frame_landmarks(crop)
                    if crop_landmarks:
                        candidate = roi.remap_landmarks(crop_landmarks, bbox, full_width=w, full_height=h)
                        if is_valid_pitcher_landmarks(candidate):
                            landmarks = candidate

                # 2. Secondary manual mound crop if tracker missed
                if landmarks is None and w >= 640 and h >= 360:
                    y1, y2 = int(0.34 * h), int(0.98 * h)
                    x1, x2 = int(0.32 * w), int(0.68 * w)
                    crop_mound = base_frame_bgr[y1:y2, x1:x2]
                    crop_landmarks = extractor.extract_frame_landmarks(crop_mound)
                    if crop_landmarks:
                        cw, ch = (x2 - x1), (y2 - y1)
                        candidate = {}
                        for k, pt in crop_landmarks.items():
                            if isinstance(pt, dict) and "x" in pt and "y" in pt:
                                px = x1 + float(pt["x"]) * cw
                                py = y1 + float(pt["y"]) * ch
                                candidate[k] = {**pt, "x": px / w, "y": py / h}
                            else:
                                candidate[k] = pt
                        if is_valid_pitcher_landmarks(candidate):
                            landmarks = candidate

                # 3. Third attempt: full frame only if verified on mound
                if landmarks is None:
                    full_cand = extractor.extract_frame_landmarks(base_frame_bgr)
                    if is_valid_pitcher_landmarks(full_cand):
                        landmarks = full_cand

                # 4. Resilient fallback: calibrated baseline mound landmarks
                if landmarks is None:
                    landmarks = get_calibrated_mound_landmarks(p_throws=p_throws, pitch_category=pitch_category)

                _BASE_FRAME_LANDMARKS_CACHE[cache_key] = landmarks

            return draw_pitcher_skeleton(base_frame_bgr, landmarks, p_throws=p_throws, metrics=metrics, is_bgr=True)

        preloaded = find_preloaded_release_frame(
            video_path=video_path,
            pitch_category=pitch_category,
            game_pk=game_pk,
            at_bat_number=at_bat_number,
            pitch_number=pitch_number,
            pitcher_id=pitcher_id,
            exact_match_only=True,
        )
        if preloaded is not None:
            return preloaded

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

                h, w = frame.shape[:2]
                landmarks = None
                if w >= 640 and h >= 360:
                    from pitch_predictor.vision.pose_extractor import PitcherROITracker

                    roi = PitcherROITracker()
                    crop, bbox = roi.crop_frame(frame)
                    crop_landmarks = extractor.extract_frame_landmarks(crop)
                    if crop_landmarks:
                        candidate = roi.remap_landmarks(crop_landmarks, bbox, full_width=w, full_height=h)
                        if is_valid_pitcher_landmarks(candidate):
                            landmarks = candidate

                if not landmarks:
                    candidate = extractor.extract_frame_landmarks(frame)
                    if is_valid_pitcher_landmarks(candidate):
                        landmarks = candidate

                if not landmarks:
                    landmarks = get_calibrated_mound_landmarks(p_throws=p_throws, pitch_category=pitch_category)

                return draw_pitcher_skeleton(frame, landmarks, p_throws=p_throws, metrics=metrics, is_bgr=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not extract frame from %s: %s. Using fallback.", path, exc)

    if not mock_mode:
        preloaded = find_preloaded_release_frame(
            video_path=video_path,
            pitch_category=pitch_category,
            game_pk=game_pk,
            at_bat_number=at_bat_number,
            pitch_number=pitch_number,
            pitcher_id=pitcher_id,
            exact_match_only=False,
        )
        if preloaded is not None:
            return preloaded

    synth_canvas = np.full((540, 720, 3), 20, dtype=np.uint8)
    synth_canvas[320:, :] = (30, 44, 34)

    from pitch_predictor.vision.pose_extractor import PoseExtractor

    mock_extractor = PoseExtractor(mock_mode=True)
    mock_landmarks = mock_extractor.extract_frame_landmarks(synth_canvas)

    return draw_pitcher_skeleton(synth_canvas, mock_landmarks, p_throws=p_throws, metrics=metrics)


def extract_base_release_frame(
    video_path: str | Path,
    release_frame_idx: int = 0,
    mock_mode: bool = False,
    pitch_category: str | None = None,
    game_pk: int | str | None = None,
    at_bat_number: int | str | None = None,
    pitch_number: int | str | None = None,
    pitcher_id: int | str | None = None,
) -> np.ndarray:
    """
    Extracts the raw broadcast release frame (the base print used as input to the pose model)
    before skeletal annotations are drawn.
    """
    path = Path(video_path) if video_path else Path("")

    if not mock_mode:
        preloaded = find_preloaded_base_frame(
            video_path=video_path,
            pitch_category=pitch_category,
            game_pk=game_pk,
            at_bat_number=at_bat_number,
            pitch_number=pitch_number,
            pitcher_id=pitcher_id,
            exact_match_only=True,
        )
        if preloaded is not None:
            return preloaded

    if not mock_mode and path.exists() and HAS_CV2:
        try:
            cap = cv2.VideoCapture(str(path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, release_frame_idx)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not extract base frame from %s: %s. Using fallback.", path, exc)

    if not mock_mode:
        preloaded = find_preloaded_base_frame(
            video_path=video_path,
            pitch_category=pitch_category,
            game_pk=game_pk,
            at_bat_number=at_bat_number,
            pitch_number=pitch_number,
            pitcher_id=pitcher_id,
            exact_match_only=False,
        )
        if preloaded is not None:
            return preloaded

    synth_canvas = np.full((540, 720, 3), 20, dtype=np.uint8)
    synth_canvas[320:, :] = (30, 44, 34)
    return synth_canvas
