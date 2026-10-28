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
from PIL import Image, ImageDraw

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
    Looks for a pre-generated broadcast release frame image with Statcast annotations.
    Supports pitcher-specific release frame resolution and generic fallbacks.
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

        # Generic fallbacks
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

        c_cyan = (0, 255, 204)
        c_pink = (255, 0, 127)
        c_gold = (255, 215, 0)
        c_white = (255, 255, 255)

        # 1. Lower body
        if r_hip and l_hip:
            draw.line([r_hip, l_hip], fill=c_gold, width=3)
            mid_hip = ((r_hip[0] + l_hip[0]) // 2, (r_hip[1] + l_hip[1]) // 2)
            if r_sh and l_sh:
                mid_sh = ((r_sh[0] + l_sh[0]) // 2, (r_sh[1] + l_sh[1]) // 2)
                draw.line([mid_hip, mid_sh], fill=c_pink, width=4)
                draw.ellipse([mid_sh[0] - 4, mid_sh[1] - 4, mid_sh[0] + 4, mid_sh[1] + 4], fill=c_pink)

        if r_hip and r_ank:
            draw.line([r_hip, r_ank], fill=c_gold, width=2)
        if l_hip and l_ank:
            draw.line([l_hip, l_ank], fill=c_gold, width=2)

        # 2. Shoulders
        if r_sh and l_sh:
            draw.line([r_sh, l_sh], fill=c_white, width=2)

        # 3. Throwing arm
        is_righty = p_throws.upper() == "R"
        throw_sh, throw_el, throw_wr = (r_sh, r_el, r_wr) if is_righty else (l_sh, l_el, l_wr)
        if throw_sh and throw_el:
            draw.line([throw_sh, throw_el], fill=c_cyan, width=4)
            draw.ellipse([throw_sh[0] - 5, throw_sh[1] - 5, throw_sh[0] + 5, throw_sh[1] + 5], fill=c_white)
            draw.ellipse([throw_el[0] - 6, throw_el[1] - 6, throw_el[0] + 6, throw_el[1] + 6], fill=c_cyan)

        if throw_el and throw_wr:
            draw.line([throw_el, throw_wr], fill=c_cyan, width=4)
            rx, ry = throw_wr
            draw.ellipse([rx - 8, ry - 8, rx + 8, ry + 8], fill=c_cyan)
            draw.ellipse([rx - 14, ry - 14, rx + 14, ry + 14], outline=c_white, width=2)
            draw.line([(rx - 18, ry), (rx + 18, ry)], fill=c_pink, width=2)
            draw.line([(rx, ry - 18), (rx, ry + 18)], fill=c_pink, width=2)

    # 4. Statcast HUD Info Card
    if metrics:
        draw.rectangle([15, 15, 275, 125], fill=(20, 24, 33), outline=(0, 255, 204), width=1)
        draw.text((25, 22), "STATCAST BIOMECHANICS", fill=(0, 255, 204))
        draw.line([(25, 36), (265, 36)], fill=(255, 255, 255), width=1)
        y = 48
        if "arm_slot_angle" in metrics:
            draw.text((25, y), f"ARM SLOT: {metrics['arm_slot_angle']:.1f} deg", fill=(255, 255, 255))
            y += 18
        if "elbow_flexion_angle" in metrics:
            draw.text((25, y), f"ELBOW FLEX: {metrics['elbow_flexion_angle']:.1f} deg", fill=(255, 255, 255))
            y += 18
        if "release_height" in metrics:
            draw.text((25, y), f"RELEASE HT: {metrics['release_height']:.3f} norm", fill=(255, 255, 255))

    return np.array(img)


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
    if frame is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    if not HAS_CV2:
        return draw_pitcher_skeleton_pillow(frame, landmarks, p_throws=p_throws, metrics=metrics)

    if landmarks is None:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame

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
    pitch_category: str | None = None,
    game_pk: int | str | None = None,
    at_bat_number: int | str | None = None,
    pitch_number: int | str | None = None,
    pitcher_id: int | str | None = None,
) -> np.ndarray:
    """
    Extracts the release frame from a video clip and overlays the skeletal pose.
    Prioritizes pre-rendered HD broadcast release frames for instant cloud resilience;
    falls back to video frame extraction or a styled synthetic canvas.
    """
    path = Path(video_path) if video_path else Path("")

    # 1. Exact pre-rendered release frame file match
    if not mock_mode:
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

    # 2. Extract from video clip if available
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
                # If large broadcast frame, isolate pitcher on mound to avoid batter detection
                if w >= 1000 and h >= 600:
                    y1, y2, x1, x2 = int(0.38 * h), int(0.98 * h), int(0.38 * w), int(0.65 * w)
                    crop = frame[y1:y2, x1:x2]
                    crop_landmarks = extractor.extract_frame_landmarks(crop)
                    if crop_landmarks:
                        cw, ch = (x2 - x1), (y2 - y1)
                        landmarks = {}
                        for k, pt in crop_landmarks.items():
                            if isinstance(pt, dict) and "x" in pt:
                                px = x1 + pt["x"] * cw
                                py = y1 + pt["y"] * ch
                                landmarks[k] = {"x": px / w, "y": py / h, "z": pt.get("z", 0.0)}

                if not landmarks:
                    landmarks = extractor.extract_frame_landmarks(frame)

                return draw_pitcher_skeleton(frame, landmarks, p_throws=p_throws, metrics=metrics)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not extract frame from %s: %s. Using fallback.", path, exc)

    # 3. Fallback to generic preloaded frame by category
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

    # 4. Synthetic Broadcast Frame Fallback
    synth_canvas = np.full((540, 720, 3), 28, dtype=np.uint8)
    synth_canvas[320:, :] = (35, 55, 38)

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

    # 1. Exact pre-rendered base frame file match
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

    # 2. Extract raw frame from video clip if available
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

    # 3. Fallback to generic preloaded base frame by category
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

    # 4. Synthetic Broadcast Base Frame Fallback
    synth_canvas = np.full((540, 720, 3), 28, dtype=np.uint8)
    synth_canvas[320:, :] = (38, 55, 35)
    return synth_canvas
