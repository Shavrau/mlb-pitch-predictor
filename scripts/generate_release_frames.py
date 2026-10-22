"""
Generate accurate, broadcast-grade Statcast annotated release frames
for Paul Skenes directly on the mound.
"""

from pathlib import Path
import cv2
import numpy as np

out_dir = Path("src/pitch_predictor/vision/sample_data/release_frames")
out_dir.mkdir(parents=True, exist_ok=True)

# Broadcast colors (BGR)
COLOR_CYAN = (204, 255, 0)      # Neon Cyan (#00FFCC) - Throwing arm
COLOR_PINK = (127, 0, 255)      # Neon Pink (#FF007F) - Spine / Trunk
COLOR_GOLD = (0, 215, 255)      # Gold (#FFD700) - Kinetic chain legs
COLOR_WHITE = (255, 255, 255)
COLOR_HUD_BG = (20, 24, 33)


def render_statcast_overlay(frame, arm_coords, trunk_coords, lower_coords, reticle_coord, metrics_dict, title_cat):
    img = frame.copy()
    h, w = img.shape[:2]

    # 1. Lower body kinetic chain (Gold)
    hip_l, hip_r, foot_lead, foot_trail = lower_coords
    cv2.line(img, hip_l, hip_r, COLOR_GOLD, 3, cv2.LINE_AA)
    cv2.line(img, hip_l, foot_lead, COLOR_GOLD, 3, cv2.LINE_AA)
    cv2.line(img, hip_r, foot_trail, COLOR_GOLD, 3, cv2.LINE_AA)
    cv2.circle(img, hip_l, 6, COLOR_GOLD, -1)
    cv2.circle(img, hip_r, 6, COLOR_GOLD, -1)
    cv2.circle(img, foot_lead, 7, COLOR_GOLD, -1)
    cv2.circle(img, foot_trail, 7, COLOR_GOLD, -1)

    # 2. Trunk inclination vector (Neon Pink)
    sh_mid, hip_mid = trunk_coords
    cv2.line(img, hip_mid, sh_mid, COLOR_PINK, 4, cv2.LINE_AA)
    cv2.circle(img, sh_mid, 8, COLOR_PINK, -1)
    cv2.circle(img, hip_mid, 7, COLOR_PINK, -1)

    # 3. Throwing arm kinetic chain (Neon Cyan)
    sh, el, wr = arm_coords
    cv2.line(img, sh, el, COLOR_CYAN, 4, cv2.LINE_AA)
    cv2.line(img, el, wr, COLOR_CYAN, 4, cv2.LINE_AA)
    cv2.circle(img, sh, 8, COLOR_WHITE, -1)
    cv2.circle(img, el, 9, COLOR_CYAN, -1)
    cv2.circle(img, el, 14, COLOR_CYAN, 2, cv2.LINE_AA)

    # 4. Release Point Target Reticle (Neon Cyan + Neon Pink crosshair)
    rx, ry = reticle_coord
    cv2.circle(img, (rx, ry), 12, COLOR_CYAN, -1)
    cv2.circle(img, (rx, ry), 20, COLOR_WHITE, 2, cv2.LINE_AA)
    cv2.circle(img, (rx, ry), 28, COLOR_CYAN, 1, cv2.LINE_AA)
    cv2.drawMarker(img, (rx, ry), COLOR_PINK, cv2.MARKER_CROSS, 36, 2)
    
    # Text label for release point
    cv2.putText(img, "RELEASE POINT", (rx + 36, ry + 6), cv2.FONT_HERSHEY_DUPLEX, 0.55, COLOR_CYAN, 1, cv2.LINE_AA)

    # 5. Broadcast HUD Card (Top-Left)
    card_w, card_h = 320, 140
    hud_x, hud_y = 25, 25
    hud_box = img[hud_y:hud_y + card_h, hud_x:hud_x + card_w]
    overlay = hud_box.copy()
    cv2.rectangle(overlay, (0, 0), (card_w, card_h), COLOR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.82, hud_box, 0.18, 0, hud_box)
    cv2.rectangle(img, (hud_x, hud_y), (hud_x + card_w, hud_y + card_h), COLOR_CYAN, 2)

    cv2.putText(img, f"STATCAST • {title_cat.upper()}", (hud_x + 15, hud_y + 26), cv2.FONT_HERSHEY_DUPLEX, 0.52, COLOR_CYAN, 1, cv2.LINE_AA)
    cv2.line(img, (hud_x + 15, hud_y + 34), (hud_x + card_w - 15, hud_y + 34), COLOR_WHITE, 1)

    y_pos = hud_y + 55
    cv2.putText(img, f"ARM SLOT:    {metrics_dict['arm_slot_angle']:.1f} deg ({metrics_dict.get('slot_desc', '3/4 Slot')})", (hud_x + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_WHITE, 1, cv2.LINE_AA)
    y_pos += 22
    cv2.putText(img, f"ELBOW FLEX:  {metrics_dict['elbow_flexion_angle']:.1f} deg", (hud_x + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_WHITE, 1, cv2.LINE_AA)
    y_pos += 22
    cv2.putText(img, f"TRUNK TILT:  {metrics_dict['trunk_tilt']:.1f} deg", (hud_x + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_WHITE, 1, cv2.LINE_AA)
    y_pos += 22
    cv2.putText(img, f"RELEASE HT:  {metrics_dict['release_height']:.3f} norm", (hud_x + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_GOLD, 1, cv2.LINE_AA)

    return img


# --- 1. Fastball (101.5 mph) ---
cap1 = cv2.VideoCapture("data/videos/694973/745170_31_4_Fastball.mp4")
cap1.set(cv2.CAP_PROP_POS_FRAMES, 198)
ret1, f1 = cap1.read()
cap1.release()
if ret1:
    ann1 = render_statcast_overlay(
        f1,
        arm_coords=((624, 385), (642, 366), (654, 342)),
        trunk_coords=((618, 385), (615, 478)),
        lower_coords=((596, 478), (634, 478), (608, 582), (646, 565)),
        reticle_coord=(654, 342),
        metrics_dict={"arm_slot_angle": 49.8, "elbow_flexion_angle": 146.1, "release_height": 0.710, "trunk_tilt": 11.7, "slot_desc": "3/4 High Slot"},
        title_cat="Fastball (101.5 mph)",
    )
    # Save both specific key and category fallback
    cv2.imwrite(str(out_dir / "745170_31_4_Fastball.jpg"), ann1, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(out_dir / "Fastball.jpg"), ann1, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite("C:/Users/rhian/.gemini/antigravity/brain/3be30dbc-2cc9-47b3-ba47-4cb1076351ba/release_Fastball.jpg", ann1, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("Generated Fastball release frame on mound!")


# --- 2. Breaking Ball (Sweeper 87 mph) ---
cap2 = cv2.VideoCapture("data/videos/694973/745500_44_1_Breaking_Ball.mp4")
cap2.set(cv2.CAP_PROP_POS_FRAMES, 174)
ret2, f2 = cap2.read()
cap2.release()
if ret2:
    ann2 = render_statcast_overlay(
        f2,
        arm_coords=((630, 409), (650, 412), (676, 410)),
        trunk_coords=((619, 409), (614, 486)),
        lower_coords=((595, 486), (635, 486), (610, 574), (642, 562)),
        reticle_coord=(676, 410),
        metrics_dict={"arm_slot_angle": 12.0, "elbow_flexion_angle": 98.1, "release_height": 0.681, "trunk_tilt": 9.9, "slot_desc": "Sweeper Sidearm"},
        title_cat="Breaking Ball (Sweeper 87 mph)",
    )
    cv2.imwrite(str(out_dir / "745500_44_1_Breaking_Ball.jpg"), ann2, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(out_dir / "Breaking_Ball.jpg"), ann2, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite("C:/Users/rhian/.gemini/antigravity/brain/3be30dbc-2cc9-47b3-ba47-4cb1076351ba/release_Breaking_Ball.jpg", ann2, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("Generated Breaking Ball release frame on mound!")


# --- 3. Offspeed (Splinker 89 mph) ---
cap3 = cv2.VideoCapture("data/videos/694973/745500_34_3_Offspeed.mp4")
cap3.set(cv2.CAP_PROP_POS_FRAMES, 174)
ret3, f3 = cap3.read()
cap3.release()
if ret3:
    ann3 = render_statcast_overlay(
        f3,
        arm_coords=((628, 392), (634, 372), (640, 350)),
        trunk_coords=((618, 390), (617, 465)),
        lower_coords=((595, 465), (635, 465), (614, 568), (646, 552)),
        reticle_coord=(640, 350),
        metrics_dict={"arm_slot_angle": 74.4, "elbow_flexion_angle": 104.4, "release_height": 0.652, "trunk_tilt": 11.9, "slot_desc": "Overhand Splinker"},
        title_cat="Offspeed (Splinker 89 mph)",
    )
    cv2.imwrite(str(out_dir / "745500_34_3_Offspeed.jpg"), ann3, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(out_dir / "Offspeed.jpg"), ann3, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite("C:/Users/rhian/.gemini/antigravity/brain/3be30dbc-2cc9-47b3-ba47-4cb1076351ba/release_Offspeed.jpg", ann3, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("Generated Offspeed release frame on mound!")
