"""
Biomechanical Pose Extraction v2 — 3D Temporal Kinematics with MediaPipe Pose.

Extracts 2D screen landmarks and 3D metric world landmarks (meters) from MLB
pitching video clips in temporal VIDEO mode, featuring:
- Automatic broadcast HSV camera-cut rejection (`detect_camera_cut_frame`)
- Mound pitcher ROI isolation & EMA tracking (`PitcherROITracker`)
- Savitzky-Golay trajectory smoothing (`smooth_kinematic_series`)
- 4-Phase delivery segmentation (`detect_delivery_phases`: Leg Lift, Foot Plant, MER, Ball Release)
- 10 MLB Pitch Tipping features (`EXTENDED_BIOMECHANICAL_FEATURE_COLS`)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

# Check for vision dependencies
try:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.vision import PoseLandmark

    HAS_VISION_DEPS = True
except ImportError:
    cv2 = None
    mp = None
    python = None
    vision = None
    PoseLandmark = None
    HAS_VISION_DEPS = False

# MediaPipe Pose Landmarker model URLs and local cache paths
HEAVY_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
HEAVY_MODEL_PATH = Path("models/pose_landmarker_heavy.task")
DEFAULT_MODEL_PATH = Path("models/pose_landmarker_lite.task")

# Core 5 + Extended 10 biomechanical feature sets
CORE_BIOMECHANICAL_FEATURE_COLS: list[str] = [
    "arm_slot_angle",
    "release_height",
    "elbow_flexion_angle",
    "trunk_tilt",
    "stride_extension",
]

EXTENDED_BIOMECHANICAL_FEATURE_COLS: list[str] = CORE_BIOMECHANICAL_FEATURE_COLS + [
    "glove_tuck_distance",
    "hip_shoulder_separation",
    "arm_angular_velocity",
    "delivery_tempo_s",
    "forward_release_extension",
]

# Landmark index constants matching MediaPipe Pose
LANDMARK_INDEX_MAP = {
    "NOSE": 0,
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,
    "RIGHT_ANKLE": 28,
}


# ---------------------------------------------------------------------------
# Signal Smoothing & Broadcast Scene Hygiene
# ---------------------------------------------------------------------------


def smooth_kinematic_series(
    values: list[float] | np.ndarray,
    window_length: int = 7,
    polyorder: int = 2,
) -> np.ndarray:
    """
    Smooth a 1D kinematic signal using a Savitzky-Golay polynomial filter,
    adapting window length safely for short sequences.
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n < 5:
        return arr.copy()

    # Handle NaNs via linear interpolation if present
    if np.isnan(arr).any():
        valid = ~np.isnan(arr)
        if valid.sum() >= 2:
            arr = np.interp(np.arange(n), np.flatnonzero(valid), arr[valid])
        else:
            arr = np.nan_to_num(arr, nan=0.0)

    win = min(window_length, n)
    if win % 2 == 0:
        win -= 1
    if win <= polyorder:
        return arr.copy()

    return np.asarray(savgol_filter(arr, window_length=win, polyorder=polyorder), dtype=float)


def _compute_hsv_hist(frame_bgr: np.ndarray) -> np.ndarray:
    """Compute normalized 2D Hue-Saturation histogram (or RGB fallback without cv2)."""
    if cv2 is not None:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist
    # NumPy fallback
    small = frame_bgr[::4, ::4, :].astype(np.float32)
    hist_parts = [np.histogram(small[:, :, c], bins=16, range=(0, 256))[0] for c in range(3)]
    vec = np.concatenate(hist_parts).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-6)


def _compare_hists(h1: np.ndarray, h2: np.ndarray) -> float:
    """Return correlation in [-1, 1] between two frame histograms."""
    if cv2 is not None:
        return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
    denom = float(np.linalg.norm(h1) * np.linalg.norm(h2))
    if denom < 1e-6:
        return 1.0
    return float(np.dot(h1.ravel(), h2.ravel()) / denom)


def detect_camera_cut_frame(
    frames_bgr: list[np.ndarray],
    min_frame: int = 10,
    corr_threshold: float = 0.55,
) -> int | None:
    """
    Detect abrupt broadcast camera scene cuts (e.g., center-field mound camera
    cutting to outfield/infield tracking camera after pitch release) via HSV
    histogram correlation drop.
    """
    if len(frames_bgr) <= max(1, min_frame):
        return None

    prev_hist = _compute_hsv_hist(frames_bgr[0])
    for idx in range(1, len(frames_bgr)):
        curr_hist = _compute_hsv_hist(frames_bgr[idx])
        if idx >= min_frame:
            corr = _compare_hists(prev_hist, curr_hist)
            if corr < corr_threshold:
                return idx
        prev_hist = curr_hist
    return None


class PitcherROITracker:
    """
    Isolates the pitcher mound region in wide MLB broadcast shots and tracks
    the pitcher's torso centroid via Exponential Moving Average (EMA) so
    background actors (batter, catcher, umpire) never hijack pose estimation.
    """

    def __init__(
        self,
        init_cx: float = 0.51,
        init_cy: float = 0.66,
        box_w: float = 0.36,
        box_h: float = 0.64,
        ema_alpha: float = 0.35,
    ) -> None:
        self.cx = init_cx
        self.cy = init_cy
        self.box_w = box_w
        self.box_h = box_h
        self.ema_alpha = ema_alpha

    def crop_frame(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Crop the active mound ROI from a full broadcast frame."""
        h, w = frame.shape[:2]
        if w < 640 or h < 360:
            return frame, (0, 0, w, h)

        half_w = self.box_w / 2.0
        half_h = self.box_h / 2.0
        nx1 = max(0.0, min(1.0 - self.box_w, self.cx - half_w))
        ny1 = max(0.0, min(1.0 - self.box_h, self.cy - half_h))
        nx2 = min(1.0, nx1 + self.box_w)
        ny2 = min(1.0, ny1 + self.box_h)

        x1, y1 = round(nx1 * w), round(ny1 * h)
        x2, y2 = max(x1 + 32, round(nx2 * w)), max(y1 + 32, round(ny2 * h))
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    def remap_landmarks(
        self,
        landmarks: dict[str | int, Any],
        bbox: tuple[int, int, int, int],
        full_width: int,
        full_height: int,
    ) -> dict[str | int, Any]:
        """Remap 2D normalized crop coordinates back to full-frame [0, 1] and update EMA."""
        x1, y1, x2, y2 = bbox
        cw = max(1, x2 - x1)
        ch = max(1, y2 - y1)

        remapped: dict[str | int, Any] = {}
        for key, pt in landmarks.items():
            if isinstance(pt, dict) and "x" in pt and "y" in pt:
                fx = (x1 + float(pt["x"]) * cw) / max(1, full_width)
                fy = (y1 + float(pt["y"]) * ch) / max(1, full_height)
                new_pt = dict(pt)
                new_pt["x"] = float(np.clip(fx, 0.0, 1.0))
                new_pt["y"] = float(np.clip(fy, 0.0, 1.0))
                remapped[key] = new_pt
            else:
                remapped[key] = pt

        # Update EMA centroid from mid-hip if visible
        lh = remapped.get("LEFT_HIP")
        rh = remapped.get("RIGHT_HIP")
        if isinstance(lh, dict) and isinstance(rh, dict):
            vis = min(float(lh.get("visibility", 1.0)), float(rh.get("visibility", 1.0)))
            if vis >= 0.45:
                mid_x = (lh["x"] + rh["x"]) / 2.0
                mid_y = (lh["y"] + rh["y"]) / 2.0
                # Clamp drift to plausible mound zone
                if 0.28 <= mid_x <= 0.72 and 0.32 <= mid_y <= 0.90:
                    self.cx = (1.0 - self.ema_alpha) * self.cx + self.ema_alpha * mid_x
                    self.cy = (1.0 - self.ema_alpha) * self.cy + self.ema_alpha * mid_y

        return remapped


# ---------------------------------------------------------------------------
# Biomechanical Geometric & 3D Metric Calculations
# ---------------------------------------------------------------------------


def calculate_arm_slot_angle(
    shoulder: dict[str, float],
    wrist: dict[str, float],
    p_throws: str = "R",
) -> float:
    """
    Calculate arm slot angle (elevation angle in degrees relative to horizontal).
    - Overhand: ~70° to 90°
    - Three-quarters: ~40° to 65°
    - Sidearm: ~15° to 35°
    - Submarine: < 15°
    """
    _ = p_throws
    dx = abs(wrist["x"] - shoulder["x"])
    dy = -(wrist["y"] - shoulder["y"])
    angle_rad = math.atan2(dy, dx)
    return float(math.degrees(angle_rad))


def calculate_elbow_flexion(
    shoulder: dict[str, float],
    elbow: dict[str, float],
    wrist: dict[str, float],
) -> float:
    """Calculate 3D elbow flexion angle in degrees (180° = straight, 90° = right angle)."""
    v1 = np.array([
        shoulder["x"] - elbow["x"],
        shoulder["y"] - elbow["y"],
        shoulder.get("z", 0.0) - elbow.get("z", 0.0),
    ])
    v2 = np.array([
        wrist["x"] - elbow["x"],
        wrist["y"] - elbow["y"],
        wrist.get("z", 0.0) - elbow.get("z", 0.0),
    ])

    norm1 = float(np.linalg.norm(v1))
    norm2 = float(np.linalg.norm(v2))
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 90.0

    cos_ang = float(np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0))
    return float(math.degrees(math.acos(cos_ang)))


def calculate_trunk_tilt(
    left_shoulder: dict[str, float],
    right_shoulder: dict[str, float],
    left_hip: dict[str, float],
    right_hip: dict[str, float],
) -> float:
    """Calculate torso inclination relative to vertical (0° = upright)."""
    mid_shoulder_x = (left_shoulder["x"] + right_shoulder["x"]) / 2.0
    mid_shoulder_y = (left_shoulder["y"] + right_shoulder["y"]) / 2.0
    mid_hip_x = (left_hip["x"] + right_hip["x"]) / 2.0
    mid_hip_y = (left_hip["y"] + right_hip["y"]) / 2.0

    trunk_dx = abs(mid_shoulder_x - mid_hip_x)
    trunk_dy = -(mid_shoulder_y - mid_hip_y)
    if trunk_dy <= 1e-6:
        return 0.0
    return float(math.degrees(math.atan2(trunk_dx, trunk_dy)))


def calculate_stride_extension(
    lead_ankle: dict[str, float],
    trail_ankle: dict[str, float],
    torso_length: float,
) -> float:
    """Calculate stride distance normalized by torso height."""
    dx = lead_ankle["x"] - trail_ankle["x"]
    dz = lead_ankle.get("z", 0.0) - trail_ankle.get("z", 0.0)
    horizontal_stride = math.sqrt(dx * dx + dz * dz) if abs(dz) > 1e-4 else abs(dx)
    if torso_length < 1e-4:
        return float(horizontal_stride)
    return float(horizontal_stride / torso_length)


def calculate_glove_tuck_distance(
    glove_wrist: dict[str, float],
    left_shoulder: dict[str, float],
    right_shoulder: dict[str, float],
    left_hip: dict[str, float] | None = None,
    right_hip: dict[str, float] | None = None,
) -> float:
    """
    Calculate 3D distance between the glove-hand wrist and the pitcher's upper-chest center.
    A flared glove (larger distance) vs tight chest tuck is a premier MLB breaking-ball tell.
    """
    mid_sh = np.array([
        (left_shoulder["x"] + right_shoulder["x"]) / 2.0,
        (left_shoulder["y"] + right_shoulder["y"]) / 2.0,
        (left_shoulder.get("z", 0.0) + right_shoulder.get("z", 0.0)) / 2.0,
    ])
    if left_hip is not None and right_hip is not None:
        mid_hip = np.array([
            (left_hip["x"] + right_hip["x"]) / 2.0,
            (left_hip["y"] + right_hip["y"]) / 2.0,
            (left_hip.get("z", 0.0) + right_hip.get("z", 0.0)) / 2.0,
        ])
        chest_center = 0.70 * mid_sh + 0.30 * mid_hip
    else:
        chest_center = mid_sh

    gw = np.array([
        glove_wrist["x"],
        glove_wrist["y"],
        glove_wrist.get("z", 0.0),
    ])
    return float(np.linalg.norm(gw - chest_center))


def calculate_hip_shoulder_separation(
    left_shoulder: dict[str, float],
    right_shoulder: dict[str, float],
    left_hip: dict[str, float],
    right_hip: dict[str, float],
) -> float:
    """
    Calculate torsional hip-shoulder separation angle in degrees [0°, 90°]
    projected onto the transverse (X-Z) plane (with 2D X-Y fallback when Z is absent).
    """
    s_dx = right_shoulder["x"] - left_shoulder["x"]
    s_dz = right_shoulder.get("z", 0.0) - left_shoulder.get("z", 0.0)
    h_dx = right_hip["x"] - left_hip["x"]
    h_dz = right_hip.get("z", 0.0) - left_hip.get("z", 0.0)

    if abs(s_dz) < 1e-5 and abs(h_dz) < 1e-5:
        s_vec = np.array([s_dx, right_shoulder["y"] - left_shoulder["y"]], dtype=float)
        h_vec = np.array([h_dx, right_hip["y"] - left_hip["y"]], dtype=float)
    else:
        s_vec = np.array([s_dx, s_dz], dtype=float)
        h_vec = np.array([h_dx, h_dz], dtype=float)

    ns = float(np.linalg.norm(s_vec))
    nh = float(np.linalg.norm(h_vec))
    if ns < 1e-6 or nh < 1e-6:
        return 0.0

    cos_theta = float(np.clip(np.dot(s_vec, h_vec) / (ns * nh), -1.0, 1.0))
    angle = math.degrees(math.acos(cos_theta))
    return float(min(angle, 180.0 - angle))


def calculate_forward_release_extension(
    throwing_wrist: dict[str, float],
    left_hip: dict[str, float],
    right_hip: dict[str, float],
) -> float:
    """Calculate forward 3D distance of the throwing wrist relative to mid-hip center."""
    mid_hx = (left_hip["x"] + right_hip["x"]) / 2.0
    mid_hz = (left_hip.get("z", 0.0) + right_hip.get("z", 0.0)) / 2.0
    dx = throwing_wrist["x"] - mid_hx
    dz = throwing_wrist.get("z", 0.0) - mid_hz
    return float(math.sqrt(dx * dx + dz * dz))


# ---------------------------------------------------------------------------
# 4-Phase Delivery Segmentation & Release Detection
# ---------------------------------------------------------------------------


def detect_delivery_phases(
    frames_data: list[dict[str, Any]],
    fps: float = 60.0,
    max_delivery_frame: int | None = 240,
) -> dict[str, Any]:
    """
    Segment a pitching clip into the 4 canonical biomechanical phases:
    1. Peak Leg Lift (`leg_lift_frame`)
    2. Foot Plant (`foot_plant_frame`)
    3. Max External Rotation / Layback (`mer_frame`)
    4. Ball Release (`release_frame`)

    Computes all 10 extended pitch-tipping features from smoothed trajectories.
    """
    if not frames_data:
        empty: dict[str, Any] = {col: None for col in EXTENDED_BIOMECHANICAL_FEATURE_COLS}
        empty.update({
            "leg_lift_frame": None,
            "foot_plant_frame": None,
            "mer_frame": None,
            "release_frame": None,
            "confidence": 0.0,
        })
        return empty

    candidates = frames_data
    if max_delivery_frame is not None:
        win = [f for f in frames_data if int(f.get("frame", 0)) <= max_delivery_frame]
        if win:
            candidates = win

    n = len(candidates)
    frame_nums = np.array([int(f.get("frame", i)) for i, f in enumerate(candidates)], dtype=int)

    rel_h_s = smooth_kinematic_series([float(f.get("rel_h", 0.65)) for f in candidates])
    arm_s = smooth_kinematic_series([float(f.get("arm_angle", 45.0)) for f in candidates])
    flex_s = smooth_kinematic_series([float(f.get("flexion", 130.0)) for f in candidates])
    tilt_s = smooth_kinematic_series([float(f.get("trunk_tilt", 12.0)) for f in candidates])
    stride_s = smooth_kinematic_series([float(f.get("stride", 1.35)) for f in candidates])
    knee_y_s = smooth_kinematic_series([float(f.get("lead_knee_y", 0.70 - 0.1 * (i == n // 4))) for i, f in enumerate(candidates)])
    glove_s = smooth_kinematic_series([float(f.get("glove_tuck", 0.24)) for f in candidates])
    sep_s = smooth_kinematic_series([float(f.get("hip_shoulder_sep", 32.0)) for f in candidates])
    fwd_s = smooth_kinematic_series([float(f.get("forward_ext", 0.55)) for f in candidates])

    # Angular velocity in deg/s using frame timestamps
    fps_safe = max(15.0, float(fps))
    if n >= 3:
        t_sec = frame_nums.astype(float) / fps_safe
        # Ensure strictly positive dt for gradient
        if np.any(np.diff(t_sec) <= 0):
            t_sec = np.arange(n, dtype=float) / fps_safe
        d_arm = np.gradient(arm_s, t_sec)
        d_flex = np.gradient(flex_s, t_sec)
        arm_vel_series = np.sqrt(d_arm * d_arm + d_flex * d_flex)
    else:
        arm_vel_series = np.full(n, 450.0, dtype=float)

    # Store smoothed angular velocity back onto candidate dicts for timeline export
    for i, rec in enumerate(candidates):
        rec["arm_vel_deg_s"] = float(arm_vel_series[i])
        rec["arm_angle_smooth"] = float(arm_s[i])
        rec["glove_tuck_smooth"] = float(glove_s[i])

    if n < 4:
        best_idx = int(np.argmax(rel_h_s))
        rf = int(frame_nums[best_idx])
        llf = max(0, int(frame_nums[0]) - (1 if int(frame_nums[0]) == rf and rf > 0 else 0))
        if llf >= rf:
            rf = llf + 1
        fpf = min(rf, llf + 1)
        return {
            "leg_lift_frame": llf,
            "foot_plant_frame": fpf,
            "mer_frame": rf,
            "release_frame": rf,
            "arm_slot_angle": float(candidates[best_idx].get("arm_angle", arm_s[best_idx])),
            "release_height": float(candidates[best_idx].get("rel_h", rel_h_s[best_idx])),
            "elbow_flexion_angle": float(candidates[best_idx].get("flexion", flex_s[best_idx])),
            "trunk_tilt": float(candidates[best_idx].get("trunk_tilt", tilt_s[best_idx])),
            "stride_extension": float(candidates[best_idx].get("stride", stride_s[best_idx])),
            "glove_tuck_distance": float(glove_s[best_idx]),
            "hip_shoulder_separation": float(sep_s[best_idx]),
            "arm_angular_velocity": float(np.max(arm_vel_series)),
            "delivery_tempo_s": float(max(1.0 / fps_safe, (rf - llf) / fps_safe)),
            "forward_release_extension": float(fwd_s[best_idx]),
            "confidence": float(candidates[best_idx].get("confidence", 0.95)),
        }

    # 1. Peak Leg Lift: minimum screen y of lead knee in the early/mid delivery window
    lift_search_end = max(1, min(n - 3, round(0.65 * n)))
    idx_lift = int(np.argmin(knee_y_s[: lift_search_end + 1]))
    idx_lift = min(idx_lift, n - 3)

    # 2. Ball Release: peak combined release height + forward wrist extension after leg lift
    rel_start = min(n - 1, idx_lift + 2)
    rel_norm = (rel_h_s - np.min(rel_h_s)) / max(1e-5, float(np.ptp(rel_h_s)))
    fwd_norm = (fwd_s - np.min(fwd_s)) / max(1e-5, float(np.ptp(fwd_s)))
    has_fwd_signal = float(np.ptp(fwd_s)) > 1e-3
    score_rel = 0.65 * rel_norm + (0.35 * fwd_norm if has_fwd_signal else 0.35 * rel_norm)
    idx_rel = rel_start + int(np.argmax(score_rel[rel_start:]))

    # 3. Foot Plant: stride reaches >= 88% of peak stride between Leg Lift and Release
    fp_start = idx_lift + 1
    fp_end = max(fp_start, idx_rel)
    seg_stride = stride_s[fp_start : fp_end + 1]
    min_st = float(np.min(seg_stride))
    max_st = float(np.max(seg_stride))
    if max_st - min_st > 0.03:
        thresh_st = min_st + 0.88 * (max_st - min_st)
        hits = np.flatnonzero(seg_stride >= thresh_st)
        idx_fp = fp_start + int(hits[0]) if len(hits) > 0 else (fp_start + fp_end) // 2
    else:
        idx_fp = (fp_start + fp_end) // 2
    idx_fp = max(idx_lift + 1, min(idx_fp, idx_rel))

    # 4. Max External Rotation (MER): minimum elbow flexion (forearm layback) in [idx_fp, idx_rel]
    mer_seg = flex_s[idx_fp : idx_rel + 1]
    idx_mer = idx_fp + int(np.argmin(mer_seg)) if len(mer_seg) > 0 else idx_fp
    idx_mer = max(idx_fp, min(idx_mer, idx_rel))

    ll_frame = int(frame_nums[idx_lift])
    fp_frame = int(frame_nums[idx_fp])
    mer_frame = int(frame_nums[idx_mer])
    rel_frame = int(frame_nums[idx_rel])

    # Guarantee strict leg_lift_frame < foot_plant_frame <= mer_frame <= release_frame
    if fp_frame <= ll_frame:
        fp_frame = ll_frame + 1
    mer_frame = max(mer_frame, fp_frame)
    rel_frame = max(rel_frame, mer_frame)

    peak_vel = float(np.max(arm_vel_series[idx_fp : idx_rel + 1])) if idx_rel >= idx_fp else float(np.max(arm_vel_series))
    glove_tuck_val = float(np.mean(glove_s[idx_mer : idx_rel + 1]))
    hip_sh_sep_val = float(np.max(sep_s[max(0, idx_fp - 2) : min(n, idx_fp + 3)]))
    tempo_s = float(max(1.0 / fps_safe, (rel_frame - ll_frame) / fps_safe))

    return {
        "leg_lift_frame": ll_frame,
        "foot_plant_frame": fp_frame,
        "mer_frame": mer_frame,
        "release_frame": rel_frame,
        "arm_slot_angle": float(arm_s[idx_rel]),
        "release_height": float(rel_h_s[idx_rel]),
        "elbow_flexion_angle": float(flex_s[idx_rel]),
        "trunk_tilt": float(tilt_s[idx_rel]),
        "stride_extension": float(stride_s[idx_rel]),
        "glove_tuck_distance": glove_tuck_val,
        "hip_shoulder_separation": hip_sh_sep_val,
        "arm_angular_velocity": peak_vel,
        "delivery_tempo_s": tempo_s,
        "forward_release_extension": float(fwd_s[idx_rel]),
        "confidence": float(candidates[idx_rel].get("confidence", 0.95)),
    }


def detect_release_frame(
    frames_data: list[dict[str, Any]],
    max_delivery_frame: int | None = 240,
) -> dict[str, Any]:
    """
    Detect the pitch release frame from a sequence of frame biomechanics.
    Preserves exact single-frame peak semantics for backwards-compatible callers,
    while attaching extended biomechanical fields when present.
    """
    if not frames_data:
        return {
            "release_frame": None,
            "release_height": None,
            "arm_slot_angle": None,
            "elbow_flexion_angle": None,
            "trunk_tilt": None,
            "stride_extension": None,
            "confidence": 0.0,
        }

    candidates = frames_data
    if max_delivery_frame is not None:
        window_candidates = [f for f in frames_data if f.get("frame", 0) <= max_delivery_frame]
        if window_candidates:
            candidates = window_candidates

    best_candidate = max(candidates, key=lambda f: f.get("rel_h", 0.0))

    out: dict[str, Any] = {
        "release_frame": int(best_candidate.get("frame", 0)),
        "release_height": float(best_candidate.get("rel_h", 0.0)),
        "arm_slot_angle": float(best_candidate.get("arm_angle", 0.0)),
        "elbow_flexion_angle": float(best_candidate.get("flexion", 0.0)),
        "trunk_tilt": float(best_candidate.get("trunk_tilt", 0.0)),
        "stride_extension": float(best_candidate.get("stride", 0.0)),
        "confidence": float(best_candidate.get("confidence", 0.95)),
    }
    return out


# ---------------------------------------------------------------------------
# PoseExtractor Class (MediaPipe Pose Heavy/Lite + RunningMode.VIDEO + 3D World)
# ---------------------------------------------------------------------------


class PoseExtractor:
    """
    Extracts 2D normalized and 3D metric world pose landmarks plus 4-phase
    temporal pitching biomechanics from video frames and clips.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        mock_mode: bool = False,
        min_detection_confidence: float = 0.5,
        prefer_heavy: bool = True,
        auto_download_heavy: bool = False,
    ) -> None:
        self.mock_mode = mock_mode or not HAS_VISION_DEPS
        self.min_confidence = min_detection_confidence
        self.prefer_heavy = prefer_heavy
        self.auto_download_heavy = auto_download_heavy
        self.detector = None
        self.has_detector = False

        if model_path is not None:
            self.model_path = Path(model_path)
        elif prefer_heavy and HEAVY_MODEL_PATH.exists() or auto_download_heavy:
            self.model_path = HEAVY_MODEL_PATH
        elif DEFAULT_MODEL_PATH.exists():
            self.model_path = DEFAULT_MODEL_PATH
        else:
            self.model_path = HEAVY_MODEL_PATH if prefer_heavy else DEFAULT_MODEL_PATH

        if not self.mock_mode:
            self._initialize_detector()

    def _initialize_detector(self) -> None:
        """Initialize the MediaPipe PoseLandmarker in IMAGE mode for single-frame calls."""
        if not HAS_VISION_DEPS or vision is None:
            self.mock_mode = True
            return

        if not self.model_path.exists():
            url = HEAVY_MODEL_URL if "heavy" in self.model_path.name else DEFAULT_MODEL_URL
            self._download_model(url=url, dest=self.model_path)
            if not self.model_path.exists() and self.model_path != DEFAULT_MODEL_PATH and DEFAULT_MODEL_PATH.exists():
                self.model_path = DEFAULT_MODEL_PATH

        if self.model_path.exists():
            try:
                base_options = python.BaseOptions(model_asset_path=str(self.model_path))
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    min_pose_detection_confidence=self.min_confidence,
                    min_pose_presence_confidence=self.min_confidence,
                    num_poses=1,
                )
                self.detector = vision.PoseLandmarker.create_from_options(options)
                self.has_detector = True
                logger.info("Initialized MediaPipe PoseLandmarker from %s", self.model_path)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to initialize MediaPipe PoseLandmarker: %s. Using mock fallback.", e)
                self.mock_mode = True
        else:
            self.mock_mode = True

    def _create_video_detector(self) -> Any | None:
        """Create a dedicated RunningMode.VIDEO PoseLandmarker instance for sequential clip tracking."""
        if self.mock_mode or not HAS_VISION_DEPS or vision is None or not self.model_path.exists():
            return None
        try:
            base_options = python.BaseOptions(model_asset_path=str(self.model_path))
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=self.min_confidence,
                min_pose_presence_confidence=self.min_confidence,
                min_tracking_confidence=self.min_confidence,
                num_poses=1,
            )
            return vision.PoseLandmarker.create_from_options(options)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Falling back to IMAGE PoseLandmarker for video clip: %s", exc)
            return None

    def _download_model(self, url: str = DEFAULT_MODEL_URL, dest: Path | None = None) -> None:
        """Download the MediaPipe pose landmarker task model."""
        import requests

        target = dest or self.model_path
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading MediaPipe pose landmarker model from %s to %s...", url, target)
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with open(target, "wb") as f:
                    f.write(r.content)
                logger.info("Successfully downloaded pose model (%d bytes)", len(r.content))
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not download pose model: %s", e)

    @staticmethod
    def _parse_mediapipe_result(result: Any) -> tuple[dict[str | int, Any] | None, dict[str | int, Any] | None]:
        """Parse both 2D screen landmarks and 3D metric world landmarks (meters) from a MediaPipe result."""
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None, None

        landmarks_raw = result.pose_landmarks[0]
        landmarks_2d: dict[str | int, Any] = {}
        for name, idx in LANDMARK_INDEX_MAP.items():
            if idx < len(landmarks_raw):
                lm = landmarks_raw[idx]
                pt = {
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z),
                    "visibility": float(getattr(lm, "visibility", 1.0)),
                }
                landmarks_2d[name] = pt
                landmarks_2d[idx] = pt
        for idx, lm in enumerate(landmarks_raw):
            landmarks_2d[idx] = {
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z),
                "visibility": float(getattr(lm, "visibility", 1.0)),
            }

        landmarks_world: dict[str | int, Any] | None = None
        world_list = getattr(result, "pose_world_landmarks", None)
        if world_list and len(world_list) > 0:
            w_raw = world_list[0]
            landmarks_world = {}
            for name, idx in LANDMARK_INDEX_MAP.items():
                if idx < len(w_raw):
                    wlm = w_raw[idx]
                    wpt = {
                        "x": float(wlm.x),
                        "y": float(wlm.y),
                        "z": float(wlm.z),
                        "visibility": float(getattr(wlm, "visibility", 1.0)),
                    }
                    landmarks_world[name] = wpt
                    landmarks_world[idx] = wpt

        return landmarks_2d, landmarks_world

    def extract_frame_landmarks(self, frame: np.ndarray) -> dict[str | int, Any] | None:
        """
        Extract 33 pose landmarks from a single BGR image frame.
        Returns dictionary accessible by both joint name (e.g. 'RIGHT_SHOULDER') and index (12).
        """
        if self.mock_mode or not self.has_detector:
            return self._generate_mock_landmarks()

        if cv2 is None or mp is None:
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.detector.detect(mp_image)
        landmarks_2d, landmarks_world = self._parse_mediapipe_result(result)
        if landmarks_2d is not None and landmarks_world is not None:
            landmarks_2d["_world"] = landmarks_world
        return landmarks_2d

    def _generate_mock_landmarks(self, rel_progress: float = 0.5) -> dict[str | int, Any]:
        """Generate realistic synthetic biomechanical landmarks for testing without video."""
        wrist_y = 0.35 - (0.05 * rel_progress)
        wrist_x = 0.65 + (0.05 * rel_progress)

        points = {
            "NOSE": {"x": 0.50, "y": 0.20, "z": 0.0, "visibility": 0.99},
            "LEFT_SHOULDER": {"x": 0.45, "y": 0.35, "z": 0.05, "visibility": 0.95},
            "RIGHT_SHOULDER": {"x": 0.55, "y": 0.35, "z": -0.05, "visibility": 0.95},
            "LEFT_ELBOW": {"x": 0.40, "y": 0.45, "z": 0.10, "visibility": 0.90},
            "RIGHT_ELBOW": {"x": 0.62, "y": 0.32, "z": -0.08, "visibility": 0.90},
            "LEFT_WRIST": {"x": 0.42, "y": 0.52, "z": 0.12, "visibility": 0.88},
            "RIGHT_WRIST": {"x": wrist_x, "y": wrist_y, "z": -0.10, "visibility": 0.92},
            "LEFT_HIP": {"x": 0.48, "y": 0.60, "z": 0.02, "visibility": 0.95},
            "RIGHT_HIP": {"x": 0.52, "y": 0.60, "z": -0.02, "visibility": 0.95},
            "LEFT_KNEE": {"x": 0.58, "y": 0.72, "z": 0.04, "visibility": 0.92},
            "RIGHT_KNEE": {"x": 0.42, "y": 0.75, "z": -0.04, "visibility": 0.92},
            "LEFT_ANKLE": {"x": 0.70, "y": 0.90, "z": 0.05, "visibility": 0.90},
            "RIGHT_ANKLE": {"x": 0.35, "y": 0.92, "z": -0.05, "visibility": 0.90},
        }

        dict_out: dict[str | int, Any] = {}
        for name, pt in points.items():
            dict_out[name] = pt
            idx = LANDMARK_INDEX_MAP.get(name)
            if idx is not None:
                dict_out[idx] = pt
        return dict_out

    def extract_video_poses(
        self,
        video_path: str | Path,
        p_throws: str = "R",
        sample_step: int = 1,
    ) -> dict[str, Any]:
        """
        Extract frame-by-frame 2D + 3D world poses in temporal VIDEO mode,
        reject post-release camera cuts, track mound ROI, smooth trajectories,
        and segment the 4 canonical pitching delivery phases.
        """
        path = Path(video_path)

        if self.mock_mode or not path.exists() or cv2 is None:
            mock_frames = []
            for i in range(45):
                progress = i / 44.0
                lead_knee_y = 0.74 - 0.22 * math.exp(-((i - 11) ** 2) / 24.0)
                stride = 0.75 + 0.72 / (1.0 + math.exp(-(i - 22) / 3.0))
                if i < 31:
                    flexion = 118.0 - 24.0 * math.exp(-((i - 30) ** 2) / 14.0)
                else:
                    flexion = 94.0 + 54.0 * min(1.0, (i - 30) / 6.0)
                rel_h = 0.50 + 0.22 * math.exp(-((i - 36) ** 2) / 36.0)
                arm_angle = 36.0 + 18.0 * math.exp(-((i - 36) ** 2) / 45.0)
                trunk_tilt = 8.5 + 5.5 * progress
                glove_tuck = 0.29 - 0.08 * min(1.0, i / 36.0)
                hip_sep = 18.0 + 24.0 * math.exp(-((i - 24) ** 2) / 25.0)
                fwd_ext = 0.28 + 0.36 * math.exp(-((i - 36) ** 2) / 28.0)
                mock_frames.append({
                    "frame": i,
                    "rel_h": rel_h,
                    "arm_angle": arm_angle,
                    "flexion": flexion,
                    "trunk_tilt": trunk_tilt,
                    "stride": stride,
                    "lead_knee_y": lead_knee_y,
                    "glove_tuck": glove_tuck,
                    "hip_shoulder_sep": hip_sep,
                    "forward_ext": fwd_ext,
                    "confidence": 0.95,
                })
            phases = detect_delivery_phases(mock_frames, fps=60.0)
            return {
                "summary": phases,
                "phases": phases,
                "camera_cut_frame": None,
                "frames": mock_frames,
                "video_path": str(path),
                "fps": 60.0,
            }

        cap = cv2.VideoCapture(str(path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
        if fps <= 1.0 or fps > 240.0:
            fps = 60.0

        frames_data: list[dict[str, Any]] = []
        frame_idx = 0
        last_ts_ms = -1
        camera_cut_frame: int | None = None
        prev_hist: np.ndarray | None = None
        roi_tracker = PitcherROITracker()
        video_detector = self._create_video_detector()

        is_lefty = p_throws.upper() == "L"
        sh_key = "LEFT_SHOULDER" if is_lefty else "RIGHT_SHOULDER"
        el_key = "LEFT_ELBOW" if is_lefty else "RIGHT_ELBOW"
        wr_key = "LEFT_WRIST" if is_lefty else "RIGHT_WRIST"
        glove_wr_key = "RIGHT_WRIST" if is_lefty else "LEFT_WRIST"
        lead_knee_key = "RIGHT_KNEE" if is_lefty else "LEFT_KNEE"
        lead_ankle_key = "RIGHT_ANKLE" if is_lefty else "LEFT_ANKLE"
        trail_ankle_key = "LEFT_ANKLE" if is_lefty else "RIGHT_ANKLE"

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Automatic HSV scene-cut rejection
                curr_hist = _compute_hsv_hist(frame)
                if prev_hist is not None and frame_idx >= 30:
                    corr = _compare_hists(prev_hist, curr_hist)
                    if corr < 0.52:
                        camera_cut_frame = frame_idx
                        break
                prev_hist = curr_hist

                if frame_idx % max(1, sample_step) == 0:
                    h, w = frame.shape[:2]
                    crop_bgr, bbox = roi_tracker.crop_frame(frame)
                    rgb_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)

                    if video_detector is not None:
                        ts_ms = round(frame_idx * 1000.0 / fps)
                        if ts_ms <= last_ts_ms:
                            ts_ms = last_ts_ms + 1
                        last_ts_ms = ts_ms
                        mp_res = video_detector.detect_for_video(mp_img, ts_ms)
                    else:
                        mp_res = self.detector.detect(mp_img)

                    lm_2d_crop, lm_world = self._parse_mediapipe_result(mp_res)
                    if lm_2d_crop is not None:
                        lm_2d = roi_tracker.remap_landmarks(lm_2d_crop, bbox, full_width=w, full_height=h)
                    else:
                        lm_2d = None

                    if lm_2d and sh_key in lm_2d and wr_key in lm_2d:
                        sh_2d = lm_2d[sh_key]
                        lm_2d.get(el_key, sh_2d)
                        wr_2d = lm_2d[wr_key]
                        ls_2d = lm_2d.get("LEFT_SHOULDER", sh_2d)
                        rs_2d = lm_2d.get("RIGHT_SHOULDER", sh_2d)
                        lh_2d = lm_2d.get("LEFT_HIP", {"x": 0.48, "y": 0.60, "z": 0.0})
                        rh_2d = lm_2d.get("RIGHT_HIP", {"x": 0.52, "y": 0.60, "z": 0.0})
                        lead_knee_2d = lm_2d.get(lead_knee_key, {"x": 0.55, "y": 0.72, "z": 0.0})
                        lead_ank_2d = lm_2d.get(lead_ankle_key, {"x": 0.70, "y": 0.90, "z": 0.0})
                        trail_ank_2d = lm_2d.get(trail_ankle_key, {"x": 0.35, "y": 0.90, "z": 0.0})

                        # Prefer 3D metric world landmarks (meters) for 3D angles & distances
                        src_3d = lm_world if (lm_world and sh_key in lm_world and wr_key in lm_world) else lm_2d
                        sh_3d = src_3d[sh_key]
                        el_3d = src_3d.get(el_key, sh_3d)
                        wr_3d = src_3d[wr_key]
                        glove_3d = src_3d.get(glove_wr_key, sh_3d)
                        ls_3d = src_3d.get("LEFT_SHOULDER", sh_3d)
                        rs_3d = src_3d.get("RIGHT_SHOULDER", sh_3d)
                        lh_3d = src_3d.get("LEFT_HIP", lh_2d)
                        rh_3d = src_3d.get("RIGHT_HIP", rh_2d)

                        arm_angle = calculate_arm_slot_angle(sh_2d, wr_2d, p_throws=p_throws)
                        flexion = calculate_elbow_flexion(sh_3d, el_3d, wr_3d)
                        trunk_tilt = calculate_trunk_tilt(ls_2d, rs_2d, lh_2d, rh_2d)

                        torso_len = abs(((rs_2d["y"] + ls_2d["y"]) / 2.0) - ((lh_2d["y"] + rh_2d["y"]) / 2.0))
                        stride = calculate_stride_extension(lead_ank_2d, trail_ank_2d, torso_len)
                        rel_h = 1.0 - wr_2d["y"]

                        glove_tuck = calculate_glove_tuck_distance(glove_3d, ls_3d, rs_3d, lh_3d, rh_3d)
                        hip_sep = calculate_hip_shoulder_separation(ls_3d, rs_3d, lh_3d, rh_3d)
                        fwd_ext = calculate_forward_release_extension(wr_3d, lh_3d, rh_3d)

                        frames_data.append({
                            "frame": frame_idx,
                            "rel_h": float(rel_h),
                            "arm_angle": float(arm_angle),
                            "flexion": float(flexion),
                            "trunk_tilt": float(trunk_tilt),
                            "stride": float(stride),
                            "lead_knee_y": float(lead_knee_2d["y"]),
                            "glove_tuck": float(glove_tuck),
                            "hip_shoulder_sep": float(hip_sep),
                            "forward_ext": float(fwd_ext),
                            "confidence": float(wr_2d.get("visibility", 0.9)),
                        })

                frame_idx += 1
        finally:
            cap.release()
            if video_detector is not None:
                try:
                    video_detector.close()
                except Exception:  # noqa: BLE001, S110
                    pass

        max_win = (camera_cut_frame - 1) if camera_cut_frame is not None else 240
        phases = detect_delivery_phases(frames_data, fps=fps, max_delivery_frame=max_win)
        return {
            "summary": phases,
            "phases": phases,
            "camera_cut_frame": camera_cut_frame,
            "frames": frames_data,
            "video_path": str(path),
            "fps": fps,
            "total_frames_analyzed": len(frames_data),
        }

    def extract_video_timeseries(
        self,
        video_path: str | Path,
        p_throws: str = "R",
        sample_step: int = 1,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Return a clean per-frame kinematic DataFrame and 4-phase summary dict."""
        res = self.extract_video_poses(video_path=video_path, p_throws=p_throws, sample_step=sample_step)
        df = pd.DataFrame(res.get("frames", []))
        return df, res.get("phases", res.get("summary", {}))

    def extract_from_video(
        self,
        video_path: str | Path,
        p_throws: str = "R",
        sample_step: int = 1,
    ) -> dict[str, Any]:
        """Convenience alias for extract_video_poses."""
        return self.extract_video_poses(video_path, p_throws=p_throws, sample_step=sample_step)


# ---------------------------------------------------------------------------
# Batch Processing Functions
# ---------------------------------------------------------------------------


def batch_extract_poses(
    manifest_df: pd.DataFrame,
    video_dir: Path | None = None,
    p_throws: str = "R",
    extractor: PoseExtractor | None = None,
) -> pd.DataFrame:
    """
    Process all videos in the manifest and return a consolidated DataFrame
    with metadata, all 10 extended biomechanical features, and 4-phase frame indices.
    """
    if extractor is None:
        extractor = PoseExtractor()

    records = []
    for _, row in manifest_df.iterrows():
        rec = row.to_dict()
        video_path = rec.get("video_path")
        if not video_path and video_dir:
            clean_cat = str(rec.get("pitch_category", "")).replace(" ", "_")
            fname = f"{rec.get('game_pk')}_{rec.get('at_bat_number')}_{rec.get('pitch_number')}_{clean_cat}.mp4"
            video_path = str(video_dir / fname)

        row_throws = str(rec.get("p_throws", p_throws))
        res = extractor.extract_video_poses(video_path or "", p_throws=row_throws)
        summary = res.get("summary", {})

        for col in EXTENDED_BIOMECHANICAL_FEATURE_COLS:
            rec[col] = summary.get(col)

        rec["leg_lift_frame"] = summary.get("leg_lift_frame")
        rec["foot_plant_frame"] = summary.get("foot_plant_frame")
        rec["mer_frame"] = summary.get("mer_frame")
        rec["release_frame"] = summary.get("release_frame")
        rec["camera_cut_frame"] = res.get("camera_cut_frame")
        rec["pose_confidence"] = summary.get("confidence")

        records.append(rec)

    return pd.DataFrame(records)
