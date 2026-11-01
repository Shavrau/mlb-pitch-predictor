"""
Pitch Tunneling & Trajectory Disguise Analytics Module (Statcast 3D Aerodynamics).

Calculates and visualizes:
1. 3D flight trajectories from Release Point (54.5 ft) to Home Plate (1.4 ft).
2. The Commitment Point / Decision Point at 23.8 ft (~150-180 ms before contact).
3. The virtual "Tunnel Cylinder" (initial ~30 ft where trajectories overlap within ~4 inches).
4. Late aerodynamic separation (IVB and HB break divergence at the plate).
5. Precision vector SVGs for perspective flight arena and 2D Commitment cross-section.
"""

from __future__ import annotations

import math
from typing import Any

from pitch_predictor.scenarios import MLB_PITCHERS
from pitch_predictor.ui_components import PITCH_CATEGORY_STYLES, STATCAST_ZONES, _compact_html, resolve_ui_theme_mode

PITCH_DISTANCE_FT: float = 60.5
RELEASE_Y_FT: float = 54.5
COMMITMENT_Y_FT: float = 23.8
HOME_PLATE_Y_FT: float = 1.4
BALL_DIAMETER_INCHES: float = 2.94

# ---------------------------------------------------------------------------
# Calibrated Pitch Arsenals for Official MLB Pitchers
# ---------------------------------------------------------------------------

PITCHER_ARSENALS: dict[str, dict[str, dict[str, Any]]] = {
    "Paul Skenes": {
        "4-Seam Fastball [FF]": {
            "name": "4-Seam Fastball",
            "code": "FF",
            "category": "Fastball",
            "speed_mph": 101.5,
            "spin_rpm": 2480,
            "spin_axis": "1:15",
            "spin_axis_deg": 225,
            "ivb_inches": 18.2,
            "hb_inches": -7.5,
            "rel_x_ft": 2.1,
            "rel_z_ft": 5.9,
            "color": "#EF4444",
            "default_zone": 2,
            "description": "Potência extrema no terço superior da zona com alto efeito Magnus (High Ride).",
        },
        "Splinker [FS]": {
            "name": "Splinker (Split-Sinker)",
            "code": "FS",
            "category": "Offspeed",
            "speed_mph": 94.8,
            "spin_rpm": 1750,
            "spin_axis": "2:15",
            "spin_axis_deg": 255,
            "ivb_inches": 2.4,
            "hb_inches": -14.2,
            "rel_x_ft": 2.15,
            "rel_z_ft": 5.85,
            "color": "#10B981",
            "default_zone": 8,
            "description": "Arremesso híbrido único que parece 4-Seam no túnel e despenca 16 polegadas abaixo da linha esperada.",
        },
        "Sweeper [ST]": {
            "name": "Sweeper",
            "code": "ST",
            "category": "Breaking Ball",
            "speed_mph": 85.5,
            "spin_rpm": 2650,
            "spin_axis": "8:30",
            "spin_axis_deg": 75,
            "ivb_inches": -1.2,
            "hb_inches": 15.8,
            "rel_x_ft": 2.2,
            "rel_z_ft": 5.8,
            "color": "#3B82F6",
            "default_zone": 14,
            "description": "Varredura horizontal violenta para o lado da luva induzindo swings no vazio fora da placa.",
        },
    },
    "Corbin Burnes": {
        "Cutter [FC]": {
            "name": "Cutter",
            "code": "FC",
            "category": "Fastball",
            "speed_mph": 95.4,
            "spin_rpm": 2780,
            "spin_axis": "11:45",
            "spin_axis_deg": 170,
            "ivb_inches": 8.5,
            "hb_inches": 4.5,
            "rel_x_ft": 2.3,
            "rel_z_ft": 5.7,
            "color": "#EF4444",
            "default_zone": 3,
            "description": "Arremesso assinatura com mais de 2.700 RPM que corta late no canto externo.",
        },
        "Curveball [CU]": {
            "name": "Curveball",
            "code": "CU",
            "category": "Breaking Ball",
            "speed_mph": 82.2,
            "spin_rpm": 2890,
            "spin_axis": "6:15",
            "spin_axis_deg": 5,
            "ivb_inches": -14.5,
            "hb_inches": -6.2,
            "rel_x_ft": 2.32,
            "rel_z_ft": 5.75,
            "color": "#3B82F6",
            "default_zone": 8,
            "description": "Quebra vertical profunda (12-6) compartilhando o mesmo túnel alto do Cutter.",
        },
        "Changeup [CH]": {
            "name": "Changeup",
            "code": "CH",
            "category": "Offspeed",
            "speed_mph": 88.6,
            "spin_rpm": 1850,
            "spin_axis": "2:30",
            "spin_axis_deg": 250,
            "ivb_inches": 5.2,
            "hb_inches": -13.8,
            "rel_x_ft": 2.35,
            "rel_z_ft": 5.65,
            "color": "#10B981",
            "default_zone": 7,
            "description": "Queda para o lado do braço contra rebatedores canhotos (platoon buster).",
        },
    },
    "Logan Webb": {
        "Sinker [SI]": {
            "name": "Sinker",
            "code": "SI",
            "category": "Fastball",
            "speed_mph": 92.8,
            "spin_rpm": 2180,
            "spin_axis": "2:00",
            "spin_axis_deg": 240,
            "ivb_inches": 5.5,
            "hb_inches": -16.5,
            "rel_x_ft": 2.6,
            "rel_z_ft": 5.4,
            "color": "#EF4444",
            "default_zone": 7,
            "description": "Máquina de induzir groundballs na borda interna/baixa da placa.",
        },
        "Changeup [CH]": {
            "name": "Changeup",
            "code": "CH",
            "category": "Offspeed",
            "speed_mph": 87.2,
            "spin_rpm": 1820,
            "spin_axis": "2:15",
            "spin_axis_deg": 245,
            "ivb_inches": -3.8,
            "hb_inches": -15.8,
            "rel_x_ft": 2.62,
            "rel_z_ft": 5.38,
            "color": "#10B981",
            "default_zone": 13,
            "description": "Túnel gêmeo do Sinker: slot e fade idênticos, mas com -5.6 mph e queda vertical abrupta.",
        },
        "Slider [SL]": {
            "name": "Slider",
            "code": "SL",
            "category": "Breaking Ball",
            "speed_mph": 83.5,
            "spin_rpm": 2450,
            "spin_axis": "9:00",
            "spin_axis_deg": 90,
            "ivb_inches": 1.2,
            "hb_inches": 12.5,
            "rel_x_ft": 2.58,
            "rel_z_ft": 5.42,
            "color": "#3B82F6",
            "default_zone": 9,
            "description": "Quebra horizontal para fora da placa, criando bifurcação tática com o Changeup.",
        },
    },
    "Chris Sale": {
        "4-Seam Fastball [FF]": {
            "name": "4-Seam Fastball",
            "code": "FF",
            "category": "Fastball",
            "speed_mph": 94.8,
            "spin_rpm": 2350,
            "spin_axis": "10:45",
            "spin_axis_deg": 135,
            "ivb_inches": 14.2,
            "hb_inches": 12.8,
            "rel_x_ft": -3.1,
            "rel_z_ft": 5.2,
            "color": "#EF4444",
            "default_zone": 1,
            "description": "Velocidade vinda de um ângulo de soltura canhoto extremamente lateral e baixo.",
        },
        "Sweeping Slider [SL]": {
            "name": "Sweeping Slider",
            "code": "SL",
            "category": "Breaking Ball",
            "speed_mph": 80.5,
            "spin_rpm": 2620,
            "spin_axis": "3:30",
            "spin_axis_deg": 285,
            "ivb_inches": -2.5,
            "hb_inches": -16.8,
            "rel_x_ft": -3.15,
            "rel_z_ft": 5.15,
            "color": "#3B82F6",
            "default_zone": 13,
            "description": "O slider mais intimidador para canhotos: parte do túnel da bola rápida e quebra 29 polegadas longe do bastão.",
        },
        "Changeup [CH]": {
            "name": "Changeup",
            "code": "CH",
            "category": "Offspeed",
            "speed_mph": 86.4,
            "spin_rpm": 1720,
            "spin_axis": "10:15",
            "spin_axis_deg": 125,
            "ivb_inches": 4.8,
            "hb_inches": 14.5,
            "rel_x_ft": -3.05,
            "rel_z_ft": 5.18,
            "color": "#10B981",
            "default_zone": 9,
            "description": "Arma primária contra destros mantendo o disfarce de braço cruzado.",
        },
    },
    "Yoshinobu Yamamoto": {
        "4-Seam Fastball [FF]": {
            "name": "4-Seam Fastball",
            "code": "FF",
            "category": "Fastball",
            "speed_mph": 95.8,
            "spin_rpm": 2420,
            "spin_axis": "1:15",
            "spin_axis_deg": 225,
            "ivb_inches": 16.4,
            "hb_inches": -6.2,
            "rel_x_ft": 1.8,
            "rel_z_ft": 5.5,
            "color": "#EF4444",
            "default_zone": 2,
            "description": "Linha limpa no topo da zona ancorada na mecânica flexível de dardo.",
        },
        "Splitter [FS]": {
            "name": "Splitter",
            "code": "FS",
            "category": "Offspeed",
            "speed_mph": 90.4,
            "spin_rpm": 1380,
            "spin_axis": "1:45",
            "spin_axis_deg": 235,
            "ivb_inches": 2.1,
            "hb_inches": -10.5,
            "rel_x_ft": 1.82,
            "rel_z_ft": 5.48,
            "color": "#10B981",
            "default_zone": 8,
            "description": "Queda 'fantasma' com rotação mínima, desaparecendo no instante do swing do rebatedor.",
        },
        "Rainbow Curve [CU]": {
            "name": "Rainbow Curve",
            "code": "CU",
            "category": "Breaking Ball",
            "speed_mph": 77.5,
            "spin_rpm": 2820,
            "spin_axis": "6:30",
            "spin_axis_deg": 15,
            "ivb_inches": -18.2,
            "hb_inches": 7.4,
            "rel_x_ft": 1.78,
            "rel_z_ft": 5.55,
            "color": "#3B82F6",
            "default_zone": 8,
            "description": "Arco clássico e pronunciado com diferença abissal de tempo (+180 ms em relação à 4-Seam).",
        },
    },
}


# ---------------------------------------------------------------------------
# Trajectory Computation & Tunneling Physics
# ---------------------------------------------------------------------------


def get_pitcher_arsenal(pitcher_name: str) -> dict[str, dict[str, Any]]:
    """Return the pitch arsenal dictionary for a pitcher, falling back to Paul Skenes."""
    return PITCHER_ARSENALS.get(pitcher_name, PITCHER_ARSENALS["Paul Skenes"])


def compute_3d_trajectory_points(
    pitch_spec: dict[str, Any],
    p_throws: str = "R",
    steps: int = 50,
) -> list[dict[str, float]]:
    """
    Compute 3D aerodynamic trajectory coordinates (x_ft, y_ft, z_ft, t_s)
    from release point (y = 54.5 ft) down to home plate (y = 1.4 ft).
    """
    speed_mph = float(pitch_spec["speed_mph"])
    v0_fps = speed_mph * 1.467
    flight_dist_ft = RELEASE_Y_FT - HOME_PLATE_Y_FT
    flight_time_s = flight_dist_ft / v0_fps

    rel_x = float(pitch_spec.get("rel_x_ft", 2.0 if p_throws == "R" else -2.0))
    rel_z = float(pitch_spec.get("rel_z_ft", 5.8))

    ivb_ft = float(pitch_spec.get("ivb_inches", 10.0)) / 12.0
    hb_ft = float(pitch_spec.get("hb_inches", 0.0)) / 12.0

    points = []
    # Pitchers calibrate initial launch angles so their pitch tunnel shares the same sightline
    aim_z_adj = (ivb_ft * 0.35)
    aim_x_adj = (hb_ft * 0.35)

    for i in range(steps + 1):
        progress = i / float(steps)
        curr_y = RELEASE_Y_FT - progress * flight_dist_ft
        curr_t = progress * flight_time_s

        # Aerodynamic break accelerates late in flight (after commitment point)
        accel_factor = progress ** 2.8

        curr_x = rel_x + (hb_ft * accel_factor) - (aim_x_adj * progress)
        curr_z = rel_z - (32.174 * 0.5 * (curr_t ** 2)) + (ivb_ft * accel_factor) - (aim_z_adj * progress)

        points.append({
            "progress": round(progress, 3),
            "y_ft": round(curr_y, 2),
            "x_ft": round(curr_x, 3),
            "z_ft": round(curr_z, 3),
            "t_s": round(curr_t, 3),
        })

    return points


def compute_tunneling_metrics(
    pitch_a: dict[str, Any],
    pitch_b: dict[str, Any],
    p_throws: str = "R",
) -> dict[str, Any]:
    """
    Compute precise Statcast tunneling separation metrics between two pitches:
    - Release Distance (inches) at y = 54.5 ft
    - Tunnel Separation Distance (inches) at the Commitment Point (y = 23.8 ft)
    - Plate Separation Distance (inches) at y = 1.4 ft
    - Velocity differential (mph)
    - Flight time delta (ms)
    - Tunnel Differential Ratio (Plate Sep / Tunnel Sep)
    - Composite Tunneling Score (0-100) and Deception Grade
    """
    pts_a = compute_3d_trajectory_points(pitch_a, p_throws=p_throws, steps=100)
    pts_b = compute_3d_trajectory_points(pitch_b, p_throws=p_throws, steps=100)

    # Release Point (step 0)
    d_rel_ft = math.hypot(pts_a[0]["x_ft"] - pts_b[0]["x_ft"], pts_a[0]["z_ft"] - pts_b[0]["z_ft"])
    d_rel_inches = round(d_rel_ft * 12.0, 2)

    # Commitment Point (find point closest to y = 23.8 ft)
    comm_idx = min(range(len(pts_a)), key=lambda idx: abs(pts_a[idx]["y_ft"] - COMMITMENT_Y_FT))
    d_tunnel_ft = math.hypot(
        pts_a[comm_idx]["x_ft"] - pts_b[comm_idx]["x_ft"],
        pts_a[comm_idx]["z_ft"] - pts_b[comm_idx]["z_ft"],
    )
    d_tunnel_inches = round(d_tunnel_ft * 12.0, 2)

    # Plate Point (step -1)
    d_plate_ft = math.hypot(pts_a[-1]["x_ft"] - pts_b[-1]["x_ft"], pts_a[-1]["z_ft"] - pts_b[-1]["z_ft"])
    d_plate_inches = round(d_plate_ft * 12.0, 2)

    speed_diff = round(abs(float(pitch_a["speed_mph"]) - float(pitch_b["speed_mph"])), 1)
    time_delta_ms = round(abs(pts_a[-1]["t_s"] - pts_b[-1]["t_s"]) * 1000.0, 1)

    tunnel_ratio = round(d_plate_inches / max(0.1, d_tunnel_inches), 2)

    # Composite Tunneling Score (0..100):
    # Rewards: d_tunnel < 3.5 in, d_plate > 14 in, speed_diff > 8 mph, time_delta > 40 ms
    score = 50.0
    # Tunnel tightness: up to +25 pts
    if d_tunnel_inches <= 2.5:
        score += 25.0
    elif d_tunnel_inches <= 3.8:
        score += 20.0 - (d_tunnel_inches - 2.5) * 6.0
    else:
        score -= min(15.0, (d_tunnel_inches - 3.8) * 8.0)

    # Late divergence: up to +20 pts
    if d_plate_inches >= 18.0:
        score += 20.0
    elif d_plate_inches >= 12.0:
        score += 12.0 + (d_plate_inches - 12.0) * 1.3
    else:
        score -= 5.0

    # Timing disruption: up to +15 pts
    score += min(15.0, (time_delta_ms / 60.0) * 15.0)

    final_score = int(min(100, max(15, round(score))))

    if final_score >= 88:
        grade = "Tier S (Disfarce de Elite)"
        badge_color = "#10B981"
    elif final_score >= 75:
        grade = "Tier A (Forte Disfarce)"
        badge_color = "#3B82F6"
    else:
        grade = "Tier B (Disfarce Moderado)"
        badge_color = "#F59E0B"

    # Tactical analysis bullet
    if d_tunnel_inches <= BALL_DIAMETER_INCHES:
        tactical_desc = (
            f"No Ponto de Decisão (23.8 ft), as trajetórias de **{pitch_a['name']}** e **{pitch_b['name']}** "
            f"estão separadas por apenas **{d_tunnel_inches:.1f}\"** (menor que o diâmetro de uma bola de beisebol, que é 2.94\"). "
            f"O rebatedor é obrigado a iniciar o swing sem conseguir distinguir a quebra, sofrendo uma divergência de **{d_plate_inches:.1f}\"** na placa com **{time_delta_ms:.0f} ms** de atraso temporal."
        )
    else:
        tactical_desc = (
            f"Os arremessos mantêm boa sobreposição inicial com **{d_tunnel_inches:.1f}\"** de distância no túnel. "
            f"A divergência final atinge **{d_plate_inches:.1f}\"** na luva do receptor com variação de velocidade de **{speed_diff} mph**, forçando o rebatedor a adivinhar a zona."
        )

    return {
        "d_rel_inches": d_rel_inches,
        "d_tunnel_inches": d_tunnel_inches,
        "d_plate_inches": d_plate_inches,
        "speed_diff_mph": speed_diff,
        "time_delta_ms": time_delta_ms,
        "tunnel_ratio": tunnel_ratio,
        "tunnel_score": final_score,
        "grade": grade,
        "badge_color": badge_color,
        "tactical_desc": tactical_desc,
        "points_a": pts_a,
        "points_b": pts_b,
    }


# ---------------------------------------------------------------------------
# Precision SVG Rendering: 3D Tunneling Arena
# ---------------------------------------------------------------------------


def render_tunneling_arena_svg(
    pitch_a: dict[str, Any],
    pitch_b: dict[str, Any],
    metrics: dict[str, Any],
    p_throws: str = "R",
    theme_mode: str = "dark",
) -> str:
    """
    Render a precision vector SVG illustrating 3D Pitch Tunneling in dual-mode:
    - Viewpoint from behind the catcher towards the pitcher's mound.
    - Translucent virtual "Tunnel Cylinder" extending to Commitment Point (23.8 ft).
    - Glowing commitment decision ring with distance label.
    - Two contrasting animated trajectory arcs with ball particles moving at relative speeds.
    - Side-by-side radar cards summarizing aerodynamics.
    """
    is_light = resolve_ui_theme_mode(theme_mode) == "light"

    bg_color = "#F8FAFC" if is_light else "#0A0B0E"
    border_color = "#C2CFDD" if is_light else "#222631"
    grid_line = "#CBD5E1" if is_light else "#1E222D"
    text_color = "#0F172A" if is_light else "#F1F5F9"
    muted_color = "#475569" if is_light else "#94A3B8"

    color_a = pitch_a.get("color", "#EF4444")
    color_b = pitch_b.get("color", "#3B82F6")

    speed_a = float(pitch_a["speed_mph"])
    speed_b = float(pitch_b["speed_mph"])

    # Duration proportional to flight speed
    dur_a = f"{max(0.38, round(40.0 / speed_a, 2))}s"
    dur_b = f"{max(0.40, round(40.0 / speed_b, 2))}s"

    rx = 295 if p_throws == "R" else 345
    ry = 82

    # Perspective projections
    # Commitment Point (23.8 ft)
    cx_tunnel = 320 + (12 if p_throws == "R" else -12)
    cy_tunnel = 152

    # Plate endpoints (Zone 2 for fastball vs zone 8/14 for breaking/offspeed)
    za = int(pitch_a.get("default_zone", 2))
    zb = int(pitch_b.get("default_zone", 8))

    zone_map = {
        1: (282, 185), 2: (320, 185), 3: (358, 185),
        4: (282, 215), 5: (320, 215), 6: (358, 215),
        7: (282, 245), 8: (320, 245), 9: (358, 245),
        11: (245, 165), 12: (395, 165), 13: (245, 265), 14: (395, 265),
    }
    tx_a, ty_a = zone_map.get(za, (320, 185))
    tx_b, ty_b = zone_map.get(zb, (320, 245))

    # Control points for Bézier curves
    # Early path stays inside the tunnel cylinder!
    c1x_a = int(rx + (cx_tunnel - rx) * 0.7)
    c1y_a = int(ry + (cy_tunnel - ry) * 0.7)
    c2x_a = int(cx_tunnel + (tx_a - cx_tunnel) * 0.6)
    c2y_a = int(cy_tunnel + (ty_a - cy_tunnel) * 0.6 - 8)

    c1x_b = int(rx + (cx_tunnel - rx) * 0.7)
    c1y_b = int(ry + (cy_tunnel - ry) * 0.7)
    c2x_b = int(cx_tunnel + (tx_b - cx_tunnel) * 0.6)
    c2y_b = int(cy_tunnel + (ty_b - cy_tunnel) * 0.6 + 6)

    path_a = f"M {rx} {ry} C {c1x_a} {c1y_a}, {c2x_a} {c2y_a}, {tx_a} {ty_a}"
    path_b = f"M {rx} {ry} C {c1x_b} {c1y_b}, {c2x_b} {c2y_b}, {tx_b} {ty_b}"

    d_tunnel = metrics.get("d_tunnel_inches", 2.8)
    d_plate = metrics.get("d_plate_inches", 16.4)
    t_score = metrics.get("tunnel_score", 95)
    t_grade = metrics.get("grade", "Tier S (Disfarce de Elite)")
    badge_color = metrics.get("badge_color", "#10B981")

    svg_content = f"""
    <div style="width:100%;display:flex;justify-content:center;margin:8px 0;">
      <svg viewBox="0 0 640 310" width="100%" style="max-width:640px;background:{bg_color};border:1px solid {border_color};border-radius:10px;display:block;">
        <defs>
          <linearGradient id="tunnelCylinder" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#38BDF8" stop-opacity="0.25" />
            <stop offset="100%" stop-color="#F59E0B" stop-opacity="0.10" />
          </linearGradient>
          <filter id="neonGlowA" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="0" stdDeviation="3.5" flood-color="{color_a}" flood-opacity="0.8" />
          </filter>
          <filter id="neonGlowB" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="0" stdDeviation="3.5" flood-color="{color_b}" flood-opacity="0.8" />
          </filter>
        </defs>

        <!-- BACKGROUND DIRT & MOUND -->
        <polygon points="280,82 360,82 450,290 190,290" fill="#1E222D" fill-opacity="0.4" stroke="{grid_line}" stroke-width="1" />
        <ellipse cx="320" cy="85" rx="38" ry="8" fill="#7C4A2D" stroke="#A0522D" stroke-width="1.2" />
        <rect x="314" y="83" width="12" height="3" fill="#FFFFFF" />

        <!-- 1. VIRTUAL TUNNEL CYLINDER (54.5 ft to 23.8 ft) -->
        <polygon points="{rx-8},{ry} {rx+8},{ry} {cx_tunnel+16},{cy_tunnel} {cx_tunnel-16},{cy_tunnel}" fill="url(#tunnelCylinder)" stroke="#38BDF8" stroke-width="1.2" stroke-dasharray="3,3" />

        <!-- 2. COMMITMENT POINT DECISION RING (23.8 ft) -->
        <ellipse cx="{cx_tunnel}" cy="{cy_tunnel}" rx="22" ry="14" fill="none" stroke="#F59E0B" stroke-width="2" stroke-dasharray="4,3" />
        <circle cx="{cx_tunnel}" cy="{cy_tunnel}" r="3" fill="#F59E0B" />
        <line x1="{cx_tunnel+22}" y1="{cy_tunnel}" x2="495" y2="{cy_tunnel}" stroke="#F59E0B" stroke-width="1" stroke-dasharray="2,2" />
        <rect x="495" y="{cy_tunnel-13}" width="135" height="26" rx="4" fill="{bg_color}" stroke="#F59E0B" stroke-width="1.2" />
        <text x="502" y="{cy_tunnel+4}" fill="#F59E0B" font-family="'JetBrains Mono', monospace" font-size="9" font-weight="800">DECISÃO (23.8 ft) • Δ {d_tunnel:.1f}"</text>

        <!-- STRIKE ZONE AT HOME PLATE -->
        <rect x="260" y="170" width="120" height="90" fill="none" stroke="{muted_color}" stroke-width="1.5" stroke-dasharray="2,2" />
        <polygon points="295,282 345,282 336,291 320,298 304,291" fill="#FFFFFF" stroke="#64748B" stroke-width="1.2" />

        <!-- 3. TRAJECTORY PATH B (Offspeed / Breaking) -->
        <path d="{path_b}" fill="none" stroke="{color_b}" stroke-width="3" stroke-linecap="round" filter="url(#neonGlowB)" stroke-opacity="0.9" />
        <path d="{path_b}" fill="none" stroke="#FFFFFF" stroke-width="1" stroke-dasharray="3,3" />
        <circle r="7" fill="#FFFFFF" stroke="{color_b}" stroke-width="2.5" filter="url(#neonGlowB)">
          <animateMotion dur="{dur_b}" repeatCount="indefinite" path="{path_b}" />
          <animate attributeName="r" values="3.5;6.5;9.5" dur="{dur_b}" repeatCount="indefinite" />
        </circle>
        <circle cx="{tx_b}" cy="{ty_b}" r="8" fill="{color_b}" fill-opacity="0.3" stroke="{color_b}" stroke-width="2" />

        <!-- 4. TRAJECTORY PATH A (Fastball / Primary) -->
        <path d="{path_a}" fill="none" stroke="{color_a}" stroke-width="3" stroke-linecap="round" filter="url(#neonGlowA)" stroke-opacity="0.9" />
        <path d="{path_a}" fill="none" stroke="#FFFFFF" stroke-width="1" stroke-dasharray="3,3" />
        <circle r="7" fill="#FFFFFF" stroke="{color_a}" stroke-width="2.5" filter="url(#neonGlowA)">
          <animateMotion dur="{dur_a}" repeatCount="indefinite" path="{path_a}" />
          <animate attributeName="r" values="3.5;6.5;9.5" dur="{dur_a}" repeatCount="indefinite" />
        </circle>
        <circle cx="{tx_a}" cy="{ty_a}" r="8" fill="{color_a}" fill-opacity="0.3" stroke="{color_a}" stroke-width="2" />

        <!-- SEPARATION BRACKET AT PLATE -->
        <line x1="{tx_a}" y1="{ty_a}" x2="{tx_b}" y2="{ty_b}" stroke="#FFFFFF" stroke-width="1.5" stroke-dasharray="2,2" />
        <circle cx="{(tx_a+tx_b)//2}" cy="{(ty_a+ty_b)//2}" r="11" fill="{bg_color}" stroke="{badge_color}" stroke-width="1.5" />
        <text x="{(tx_a+tx_b)//2}" y="{(ty_a+ty_b)//2 + 4}" text-anchor="middle" fill="{text_color}" font-family="'JetBrains Mono', monospace" font-size="8.5" font-weight="800">{d_plate:.0f}"</text>

        <!-- TOP SCOREBOARD BANNER -->
        <rect x="12" y="10" width="220" height="52" rx="6" fill="{bg_color}" stroke="{border_color}" stroke-width="1.2" />
        <line x1="12" y1="12" x2="232" y2="12" stroke="{color_a}" stroke-width="2.5" />
        <text x="22" y="27" fill="{muted_color}" font-family="'JetBrains Mono', monospace" font-size="9" font-weight="700">ARREMESSO A ({pitch_a['code']})</text>
        <text x="22" y="44" fill="{text_color}" font-family="'JetBrains Mono', monospace" font-size="13" font-weight="800">{pitch_a['speed_mph']} mph • {pitch_a['name'][:12]}</text>
        <text x="22" y="55" fill="{color_a}" font-family="'JetBrains Mono', monospace" font-size="8.5" font-weight="700">{pitch_a['spin_rpm']} RPM • IVB: {pitch_a['ivb_inches']:+.1f}" • HB: {pitch_a['hb_inches']:+.1f}"</text>

        <rect x="408" y="10" width="220" height="52" rx="6" fill="{bg_color}" stroke="{border_color}" stroke-width="1.2" />
        <line x1="408" y1="12" x2="628" y2="12" stroke="{color_b}" stroke-width="2.5" />
        <text x="418" y="27" fill="{muted_color}" font-family="'JetBrains Mono', monospace" font-size="9" font-weight="700">ARREMESSO B ({pitch_b['code']})</text>
        <text x="418" y="44" fill="{text_color}" font-family="'JetBrains Mono', monospace" font-size="13" font-weight="800">{pitch_b['speed_mph']} mph • {pitch_b['name'][:12]}</text>
        <text x="418" y="55" fill="{color_b}" font-family="'JetBrains Mono', monospace" font-size="8.5" font-weight="700">{pitch_b['spin_rpm']} RPM • IVB: {pitch_b['ivb_inches']:+.1f}" • HB: {pitch_b['hb_inches']:+.1f}"</text>

        <!-- CENTER BADGE: TUNNELING SCORE -->
        <rect x="250" y="10" width="140" height="32" rx="5" fill="{bg_color}" stroke="{badge_color}" stroke-width="1.4" />
        <text x="320" y="23" text-anchor="middle" fill="{badge_color}" font-family="'JetBrains Mono', monospace" font-size="8.5" font-weight="800">SCORE: {t_score}/100</text>
        <text x="320" y="36" text-anchor="middle" fill="{text_color}" font-family="'JetBrains Mono', monospace" font-size="9" font-weight="800">{t_grade}</text>
      </svg>
    </div>
    """
    return _compact_html(svg_content)


# ---------------------------------------------------------------------------
# Precision SVG Rendering: 2D Commitment Point Cross-Section Target
# ---------------------------------------------------------------------------


def render_commitment_cross_section_svg(
    pitch_a: dict[str, Any],
    pitch_b: dict[str, Any],
    metrics: dict[str, Any],
    theme_mode: str = "dark",
) -> str:
    """
    Render a 2D cross-section target showing both ball silhouettes inside the 4-inch
    virtual tunnel at the Commitment Point (y = 23.8 ft).
    Proves whether the two pitches visually overlap to the batter's eye.
    """
    is_light = resolve_ui_theme_mode(theme_mode) == "light"

    bg_color = "#F8FAFC" if is_light else "#0A0B0E"
    border_color = "#C2CFDD" if is_light else "#222631"
    text_color = "#0F172A" if is_light else "#F1F5F9"
    muted_color = "#475569" if is_light else "#94A3B8"

    color_a = pitch_a.get("color", "#EF4444")
    color_b = pitch_b.get("color", "#3B82F6")

    d_tunnel = metrics.get("d_tunnel_inches", 2.8)

    # Scale: 1 inch = 18 pixels (4 inch circle = 72px radius)
    scale = 16.0
    cx = 120
    cy = 110

    # Center Ball A at (cx - d_tunnel/2 * scale, cy)
    # Center Ball B at (cx + d_tunnel/2 * scale, cy)
    offset_px = (d_tunnel * scale) / 2.0
    bx_a = cx - offset_px * 0.7
    by_a = cy - offset_px * 0.3
    bx_b = cx + offset_px * 0.7
    by_b = cy + offset_px * 0.3

    # Ball radius = 2.94 / 2 * scale ~= 23.5 px
    ball_r = (BALL_DIAMETER_INCHES / 2.0) * scale

    overlap = d_tunnel <= BALL_DIAMETER_INCHES
    overlap_label = "SOBREPOSIÇÃO VISUAL TOTAL" if overlap else "DISFARCE PARCIAL"
    overlap_color = "#10B981" if overlap else "#F59E0B"

    svg = f"""
    <div style="width:100%;display:flex;justify-content:center;margin:4px 0;">
      <svg viewBox="0 0 240 220" width="100%" style="max-width:240px;background:{bg_color};border:1px solid {border_color};border-radius:8px;display:block;">
        <!-- TITLE -->
        <text x="120" y="20" text-anchor="middle" fill="{muted_color}" font-family="'JetBrains Mono', monospace" font-size="9" font-weight="700">CORTE A 23.8 PÉS (COMMITMENT)</text>

        <!-- 4-INCH VIRTUAL TUNNEL CYLINDER TARGET -->
        <circle cx="{cx}" cy="{cy}" r="64" fill="none" stroke="#38BDF8" stroke-width="1.5" stroke-dasharray="4,4" />
        <circle cx="{cx}" cy="{cy}" r="32" fill="none" stroke="{border_color}" stroke-width="1" />
        <line x1="{cx-70}" y1="{cy}" x2="{cx+70}" y2="{cy}" stroke="{border_color}" stroke-width="1" />
        <line x1="{cx}" y1="{cy-70}" x2="{cx}" y2="{cy+70}" stroke="{border_color}" stroke-width="1" />
        <text x="{cx+40}" y="{cy-44}" fill="#38BDF8" font-family="'JetBrains Mono', monospace" font-size="8">TÚNEL 4"</text>

        <!-- BALL A SILHOUETTE -->
        <circle cx="{bx_a}" cy="{by_a}" r="{ball_r}" fill="{color_a}" fill-opacity="0.35" stroke="{color_a}" stroke-width="2" />
        <circle cx="{bx_a}" cy="{by_a}" r="3" fill="{color_a}" />
        <text x="{bx_a}" y="{by_a-12}" text-anchor="middle" fill="{color_a}" font-family="'JetBrains Mono', monospace" font-size="9" font-weight="800">{pitch_a['code']}</text>

        <!-- BALL B SILHOUETTE -->
        <circle cx="{bx_b}" cy="{by_b}" r="{ball_r}" fill="{color_b}" fill-opacity="0.35" stroke="{color_b}" stroke-width="2" />
        <circle cx="{bx_b}" cy="{by_b}" r="3" fill="{color_b}" />
        <text x="{bx_b}" y="{by_b+18}" text-anchor="middle" fill="{color_b}" font-family="'JetBrains Mono', monospace" font-size="9" font-weight="800">{pitch_b['code']}</text>

        <!-- SEPARATION BRACKET -->
        <line x1="{bx_a}" y1="{by_a}" x2="{bx_b}" y2="{by_b}" stroke="#FFFFFF" stroke-width="1.5" />
        <text x="{cx}" y="{cy-8}" text-anchor="middle" fill="#FFFFFF" font-family="'JetBrains Mono', monospace" font-size="9" font-weight="800">Δ {d_tunnel:.1f}"</text>

        <!-- FOOTER STATUS -->
        <rect x="20" y="190" width="200" height="22" rx="4" fill="{overlap_color}" fill-opacity="0.15" stroke="{overlap_color}" stroke-width="1" />
        <text x="120" y="204" text-anchor="middle" fill="{overlap_color}" font-family="'JetBrains Mono', monospace" font-size="8.5" font-weight="800">{overlap_label}</text>
      </svg>
    </div>
    """
    return _compact_html(svg)
