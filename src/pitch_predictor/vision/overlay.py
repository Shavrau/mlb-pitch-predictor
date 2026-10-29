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

        c_cyan = (34, 212, 232)
        c_red = (239, 68, 68)
        c_amber = (245, 158, 11)
        c_blue = (56, 189, 248)
        c_white = (244, 246, 250)

        mid_hip = None
        mid_sh = None
        if r_hip and l_hip:
            draw.line([r_hip, l_hip], fill=c_amber, width=3)
            mid_hip = ((r_hip[0] + l_hip[0]) // 2, (r_hip[1] + l_hip[1]) // 2)
            if r_sh and l_sh:
                mid_sh = ((r_sh[0] + l_sh[0]) // 2, (r_sh[1] + l_sh[1]) // 2)
                draw.line([mid_hip, mid_sh], fill=c_red, width=3)

        if r_hip and r_ank:
            draw.line([r_hip, r_ank], fill=c_amber, width=2)
        if l_hip and l_ank:
            draw.line([l_hip, l_ank], fill=c_amber, width=2)

        if r_sh and l_sh:
            draw.line([r_sh, l_sh], fill=c_white, width=2)

        is_righty = p_throws.upper() == "R"
        throw_sh, throw_el, throw_wr = (r_sh, r_el, r_wr) if is_righty else (l_sh, l_el, l_wr)
        glove_wr = l_wr if is_righty else r_wr

        if mid_sh and mid_hip and glove_wr:
            chest_pt = (int(0.7 * mid_sh[0] + 0.3 * mid_hip[0]), int(0.7 * mid_sh[1] + 0.3 * mid_hip[1]))
            draw.line([chest_pt, glove_wr], fill=c_blue, width=2)

        if throw_sh and throw_el:
            draw.line([throw_sh, throw_el], fill=c_cyan, width=4)
        if throw_el and throw_wr:
            draw.line([throw_el, throw_wr], fill=c_cyan, width=4)
            rx, ry = throw_wr
            draw.ellipse([rx - 10, ry - 10, rx + 10, ry + 10], outline=c_white, width=2)
            draw.line([(rx - 15, ry), (rx + 15, ry)], fill=c_amber, width=2)
            draw.line([(rx, ry - 15), (rx, ry + 15)], fill=c_amber, width=2)

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

    img = frame.copy()
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

    # 1. Pelvic Hip Axis & Lower Kinetic Chain
    if r_hip and l_hip:
        cv2.line(img, r_hip, l_hip, COLOR_AMBER, 3, cv2.LINE_AA)
        cv2.circle(img, r_hip, 4, COLOR_AMBER, -1)
        cv2.circle(img, l_hip, 4, COLOR_AMBER, -1)
        mid_hip = ((r_hip[0] + l_hip[0]) // 2, (r_hip[1] + l_hip[1]) // 2)

        if r_sh and l_sh:
            mid_sh = ((r_sh[0] + l_sh[0]) // 2, (r_sh[1] + l_sh[1]) // 2)
            # Vertical reference tick at mid_hip + actual Trunk Tilt spine vector
            cv2.line(img, mid_hip, (mid_hip[0], max(0, mid_sh[1] - 12)), COLOR_MUTED, 1, cv2.LINE_AA)
            cv2.line(img, mid_hip, mid_sh, COLOR_SPINE_RED, 3, cv2.LINE_AA)
            cv2.circle(img, mid_sh, 5, COLOR_SPINE_RED, -1)

    if r_hip and r_ank:
        cv2.line(img, r_hip, r_ank, COLOR_AMBER, 2, cv2.LINE_AA)
    if l_hip and l_ank:
        cv2.line(img, l_hip, l_ank, COLOR_AMBER, 2, cv2.LINE_AA)

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
        cv2.circle(img, glove_wr, 5, COLOR_GLOVE_BLUE, -1)

    if mid_sh and mid_hip and glove_wr:
        chest_pt = (
            round(0.70 * mid_sh[0] + 0.30 * mid_hip[0]),
            round(0.70 * mid_sh[1] + 0.30 * mid_hip[1]),
        )
        cv2.line(img, chest_pt, glove_wr, COLOR_GLOVE_BLUE, 1, cv2.LINE_AA)
        cv2.circle(img, chest_pt, 3, COLOR_GLOVE_BLUE, -1)

    # Technical Arm Slot Angular Arc at Throwing Shoulder
    if throw_sh and throw_wr:
        arc_r = max(22, int(min(w, h) * 0.045))
        horiz_end = (throw_sh[0] + (arc_r if is_righty else -arc_r), throw_sh[1])
        cv2.line(img, throw_sh, horiz_end, COLOR_MUTED, 1, cv2.LINE_AA)

        dx = throw_wr[0] - throw_sh[0]
        dy = throw_wr[1] - throw_sh[1]
        ang_deg = math.degrees(math.atan2(dy, dx))
        start_a, end_a = (min(0.0, ang_deg), max(0.0, ang_deg)) if is_righty else (min(180.0, ang_deg), max(180.0, ang_deg))
        cv2.ellipse(img, throw_sh, (arc_r, arc_r), 0.0, start_a, end_a, COLOR_AMBER, 2, cv2.LINE_AA)

    # Throwing Arm Chain + Precision Release Point Reticle
    if throw_sh and throw_el:
        cv2.line(img, throw_sh, throw_el, COLOR_HAWKEYE_CYAN, 4, cv2.LINE_AA)
        cv2.circle(img, throw_sh, 6, COLOR_WHITE, -1)
        cv2.circle(img, throw_el, 7, COLOR_HAWKEYE_CYAN, -1)
        cv2.circle(img, throw_el, 11, COLOR_HAWKEYE_CYAN, 1, cv2.LINE_AA)

    if throw_el and throw_wr:
        cv2.line(img, throw_el, throw_wr, COLOR_HAWKEYE_CYAN, 4, cv2.LINE_AA)
        cv2.circle(img, throw_wr, 8, COLOR_HAWKEYE_CYAN, -1)
        cv2.circle(img, throw_wr, 15, COLOR_WHITE, 2, cv2.LINE_AA)
        cv2.drawMarker(img, throw_wr, COLOR_AMBER, cv2.MARKER_CROSS, 22, 2)

    # 4. Hawk-Eye 3D Telemetry HUD Badge (Top-Left Corner)
    if metrics:
        rows_to_draw: list[str] = []
        if metrics.get("arm_slot_angle") is not None:
            rows_to_draw.append(f"ARM SLOT:   {float(metrics['arm_slot_angle']):5.1f} deg")
        if metrics.get("elbow_flexion_angle") is not None:
            rows_to_draw.append(f"ELBOW FLEX: {float(metrics['elbow_flexion_angle']):5.1f} deg")
        if metrics.get("release_height") is not None:
            rows_to_draw.append(f"RELEASE HT: {float(metrics['release_height']):5.3f} norm")
        if metrics.get("trunk_tilt") is not None:
            rows_to_draw.append(f"TRUNK TILT: {float(metrics['trunk_tilt']):5.1f} deg")
        if metrics.get("glove_tuck_distance") is not None:
            rows_to_draw.append(f"GLOVE TUCK: {float(metrics['glove_tuck_distance']):5.2f} m")
        if metrics.get("hip_shoulder_separation") is not None:
            rows_to_draw.append(f"HIP-SH SEP: {float(metrics['hip_shoulder_separation']):5.1f} deg")

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
                for line_txt in rows_to_draw:
                    if y_pos < 14 + card_h - 4:
                        cv2.putText(
                            img,
                            line_txt,
                            (24, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.39,
                            COLOR_WHITE,
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
                if w >= 960 and h >= 540:
                    from pitch_predictor.vision.pose_extractor import PitcherROITracker

                    roi = PitcherROITracker()
                    crop, bbox = roi.crop_frame(frame)
                    crop_landmarks = extractor.extract_frame_landmarks(crop)
                    if crop_landmarks:
                        landmarks = roi.remap_landmarks(crop_landmarks, bbox, full_width=w, full_height=h)

                if not landmarks:
                    landmarks = extractor.extract_frame_landmarks(frame)

                return draw_pitcher_skeleton(frame, landmarks, p_throws=p_throws, metrics=metrics)
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
