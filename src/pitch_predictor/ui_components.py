"""
SVG and HTML UI components for the MLB Statcast Broadcast Tactical Interface.

Provides:
- `STATCAST_ZONES`: Metadata for Statcast zones 0, 1-9 (Strike), and 11-14 (Chase).
- `PITCH_CATEGORY_STYLES`: Semantic colors, badges, and descriptions for Fastball,
  Breaking Ball, and Offspeed categories.
- `render_baseball_diamond_svg`: Interactive vector infield diamond with amber glow (`#F59E0B`)
  on occupied bases and outs indicator.
- `render_strike_zone_svg`: Catcher's perspective Statcast Strike Zone (3x3 inner grid + 4 outer
  shadow/chase corners) with dynamic highlight on the selected zone.
- `render_prediction_card_html`: Broadcast-grade HTML card displaying the predicted pitch category,
  tactical context badges, and inference latency.
- `render_scoreboard_banner_html`: MLB TV-style electronic scoreboard header.
- `get_custom_css`: Dark-mode stadium control room CSS stylesheet.
"""

from __future__ import annotations

import base64
import functools
from pathlib import Path
from typing import Any

STATCAST_ZONES: dict[int, dict[str, str]] = {
    0: {
        "label": "Zona 0 — First Pitch (Início de Duelo)",
        "region": "First Pitch",
        "short": "FP",
    },
    1: {"label": "Zona 1 — Strike Alta-Esquerda", "region": "Strike", "short": "Z1"},
    2: {"label": "Zona 2 — Strike Alta-Centro", "region": "Strike", "short": "Z2"},
    3: {"label": "Zona 3 — Strike Alta-Direita", "region": "Strike", "short": "Z3"},
    4: {"label": "Zona 4 — Strike Média-Esquerda", "region": "Strike", "short": "Z4"},
    5: {"label": "Zona 5 — Coração da Zona (Heart)", "region": "Strike", "short": "Z5"},
    6: {"label": "Zona 6 — Strike Média-Direita", "region": "Strike", "short": "Z6"},
    7: {"label": "Zona 7 — Strike Baixa-Esquerda", "region": "Strike", "short": "Z7"},
    8: {"label": "Zona 8 — Strike Baixa-Centro", "region": "Strike", "short": "Z8"},
    9: {"label": "Zona 9 — Strike Baixa-Direita", "region": "Strike", "short": "Z9"},
    11: {
        "label": "Zona 11 — Chase/Waste Superior-Esquerda",
        "region": "Chase",
        "short": "Z11",
    },
    12: {
        "label": "Zona 12 — Chase/Waste Superior-Direita",
        "region": "Chase",
        "short": "Z12",
    },
    13: {
        "label": "Zona 13 — Chase/Waste Inferior-Esquerda",
        "region": "Chase",
        "short": "Z13",
    },
    14: {
        "label": "Zona 14 — Chase/Waste Inferior-Direita",
        "region": "Chase",
        "short": "Z14",
    },
}

PITCH_CATEGORY_STYLES: dict[str, dict[str, str]] = {
    "Fastball": {
        "primary": "#EF4444",
        "secondary": "#DC2626",
        "glow": "rgba(239, 68, 68, 0.35)",
        "bg_tint": "rgba(239, 68, 68, 0.12)",
        "icon": "⚡",
        "label_pt": "Bola Rápida (Fastball)",
        "arsenal": "4-Seam (FF) • Sinker (SI) • Cutter (FC)",
        "tactical_insight": "Ataque de velocidade e potência para desafiar o rebatedor na zona ou elevar o nível dos olhos.",
    },
    "Breaking Ball": {
        "primary": "#3B82F6",
        "secondary": "#2563EB",
        "glow": "rgba(59, 130, 246, 0.35)",
        "bg_tint": "rgba(59, 130, 246, 0.12)",
        "icon": "🌀",
        "label_pt": "Bola de Efeito (Breaking Ball)",
        "arsenal": "Slider (SL) • Sweeper (ST) • Curveball (CU) • Knuckle-Curve (KC)",
        "tactical_insight": "Quebra acentuada horizontal/vertical buscando swing no vazio (whiff) ou contato fraco.",
    },
    "Offspeed": {
        "primary": "#10B981",
        "secondary": "#059669",
        "glow": "rgba(16, 185, 129, 0.35)",
        "bg_tint": "rgba(16, 185, 129, 0.12)",
        "icon": "🛑",
        "label_pt": "Mudança de Velocidade (Offspeed)",
        "arsenal": "Changeup (CH) • Splitter (FS)",
        "tactical_insight": "Variação de velocidade tunelada com a bola rápida para desestabilizar o timing do swing e induzir bolas rasteiras (groundballs).",
    },
}


def _compact_html(raw: str) -> str:
    """
    Collapse multi-line HTML/SVG markup into a single line without leading spaces.

    Streamlit's `st.markdown(..., unsafe_allow_html=True)` runs CommonMark before
    injecting HTML. Since `<svg>` is not a CommonMark Type-6 block tag, multi-line
    SVG markup or indented lines can be split and rendered as `<pre><code>` blocks.
    Collapsing into a single line guarantees unbroken HTML/SVG rendering.
    """
    return " ".join(line.strip() for line in raw.splitlines() if line.strip())


def render_baseball_diamond_svg(
    on_1b: int, on_2b: int, on_3b: int, outs: int = 0
) -> str:
    """
    Render an interactive-style SVG diagram of the baseball diamond.

    Occupied bases are illuminated in vibrant amber (`#F59E0B`) with a neon drop-shadow.
    Empty bases are rendered in passive slate (`#334155`).
    """
    color_1b = "#F59E0B" if int(on_1b) > 0 else "#334155"
    color_2b = "#F59E0B" if int(on_2b) > 0 else "#334155"
    color_3b = "#F59E0B" if int(on_3b) > 0 else "#334155"

    stroke_1b = "#FEF3C7" if int(on_1b) > 0 else "#64748B"
    stroke_2b = "#FEF3C7" if int(on_2b) > 0 else "#64748B"
    stroke_3b = "#FEF3C7" if int(on_3b) > 0 else "#64748B"

    filter_1b = 'filter="url(#amberGlow)"' if int(on_1b) > 0 else ""
    filter_2b = 'filter="url(#amberGlow)"' if int(on_2b) > 0 else ""
    filter_3b = 'filter="url(#amberGlow)"' if int(on_3b) > 0 else ""

    out1_fill = "#EF4444" if outs >= 1 else "#1E293B"
    out2_fill = "#EF4444" if outs >= 2 else "#1E293B"
    runners_count = (
        (1 if int(on_1b) > 0 else 0)
        + (1 if int(on_2b) > 0 else 0)
        + (1 if int(on_3b) > 0 else 0)
    )
    risp_active = int(on_2b) > 0 or int(on_3b) > 0
    status_label = (
        "RISP ATIVO (POSIÇÃO DE ANOTAÇÃO)"
        if risp_active
        else ("CORREDOR NA 1B" if runners_count == 1 else "BASES LIMPAS")
    )
    status_color = "#F59E0B" if risp_active else ("#38BDF8" if runners_count == 1 else "#94A3B8")

    raw_svg = f"""
    <div style="display:flex;justify-content:center;align-items:center;margin:6px 0;">
      <svg viewBox="0 0 240 235" width="100%" style="max-width:250px;background:#0B0F19;border-radius:12px;border:1px solid #26354A;padding:6px;display:block;">
        <defs>
          <filter id="amberGlow" x="-40%" y="-40%" width="180%" height="180%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        <path d="M 22 105 A 110 110 0 0 1 218 105" fill="none" stroke="#1E293B" stroke-width="2" stroke-dasharray="4,4" />
        <polygon points="120,38 192,110 120,182 48,110" fill="#0F172A" stroke="#334155" stroke-width="2.2" />
        <circle cx="120" cy="110" r="13" fill="#1E293B" stroke="#475569" stroke-width="1.5" />
        <rect x="115" y="108" width="10" height="4" rx="1" fill="#E2E8F0" />
        <g transform="translate(120, 38) rotate(45)" {filter_2b}>
          <rect x="-13" y="-13" width="26" height="26" rx="3" fill="{color_2b}" stroke="{stroke_2b}" stroke-width="2.2" />
        </g>
        <text x="120" y="16" text-anchor="middle" fill="#E2E8F0" font-size="11" font-weight="700" font-family="sans-serif">2B</text>
        <g transform="translate(48, 110) rotate(45)" {filter_3b}>
          <rect x="-13" y="-13" width="26" height="26" rx="3" fill="{color_3b}" stroke="{stroke_3b}" stroke-width="2.2" />
        </g>
        <text x="18" y="114" text-anchor="middle" fill="#E2E8F0" font-size="11" font-weight="700" font-family="sans-serif">3B</text>
        <g transform="translate(192, 110) rotate(45)" {filter_1b}>
          <rect x="-13" y="-13" width="26" height="26" rx="3" fill="{color_1b}" stroke="{stroke_1b}" stroke-width="2.2" />
        </g>
        <text x="222" y="114" text-anchor="middle" fill="#E2E8F0" font-size="11" font-weight="700" font-family="sans-serif">1B</text>
        <polygon points="120,192 110,182 110,173 130,173 130,182" fill="#F8FAFC" stroke="#94A3B8" stroke-width="1.5" />
        <text x="72" y="218" fill="#94A3B8" font-size="10" font-weight="700" font-family="sans-serif">OUTS:</text>
        <circle cx="116" cy="214" r="6" fill="{out1_fill}" stroke="#64748B" stroke-width="1.5" />
        <circle cx="134" cy="214" r="6" fill="{out2_fill}" stroke="#64748B" stroke-width="1.5" />
        <text x="120" y="231" text-anchor="middle" fill="{status_color}" font-size="8.5" font-weight="700" font-family="sans-serif">{status_label}</text>
      </svg>
    </div>
    """
    return _compact_html(raw_svg)


def render_strike_zone_svg(
    selected_zone: int, prev_category: str = "First Pitch"
) -> str:
    """
    Render the official MLB Statcast Strike Zone SVG from the catcher's perspective.

    Includes:
    - Outer Chase/Shadow L-shaped quadrants: Zones 11 (Top-Left), 12 (Top-Right),
      13 (Bottom-Left), and 14 (Bottom-Right).
    - Inner 3x3 Strike Zone grid: Zones 1..9 (with Zone 5 as the Heart of the Plate).
    - Visual pitch marker highlighting `selected_zone`.
    """
    cat_style = PITCH_CATEGORY_STYLES.get(prev_category, {})
    active_color = cat_style.get("primary", "#38BDF8")

    def _chase_style(zone_id: int) -> tuple[str, str, str]:
        if selected_zone == zone_id:
            return (active_color, "#F8FAFC", "0.78")
        return ("#162032", "#334155", "0.85")

    def _strike_style(zone_id: int) -> tuple[str, str, str]:
        if selected_zone == zone_id:
            return (active_color, "#FFFFFF", "0.88")
        if zone_id == 5:
            return ("#1E293B", "#38BDF8", "0.95")
        return ("#0F172A", "#38BDF8", "0.90")

    zone_centers = {
        1: (75, 65),
        2: (120, 65),
        3: (165, 65),
        4: (75, 110),
        5: (120, 110),
        6: (165, 110),
        7: (75, 155),
        8: (120, 155),
        9: (165, 155),
        11: (40, 32),
        12: (200, 32),
        13: (40, 188),
        14: (200, 188),
    }

    f11, s11, o11 = _chase_style(11)
    f12, s12, o12 = _chase_style(12)
    f13, s13, o13 = _chase_style(13)
    f14, s14, o14 = _chase_style(14)

    strike_cells_svg = []
    for idx, z_id in enumerate(range(1, 10)):
        row = idx // 3
        col = idx % 3
        x = 52 + col * 45
        y = 42 + row * 45
        fill_c, stroke_c, opac = _strike_style(z_id)
        text_c = "#FFFFFF" if selected_zone == z_id else "#94A3B8"
        font_w = "800" if selected_zone == z_id else "600"
        strike_cells_svg.append(
            f'<rect x="{x}" y="{y}" width="45" height="45" fill="{fill_c}" '
            f'fill-opacity="{opac}" stroke="{stroke_c}" stroke-width="1.5" />\n'
            f'<text x="{x + 22.5}" y="{y + 27}" text-anchor="middle" fill="{text_c}" '
            f'font-size="13" font-weight="{font_w}" font-family="sans-serif">{z_id}</text>'
        )

    marker_svg = ""
    if selected_zone in zone_centers:
        cx, cy = zone_centers[selected_zone]
        marker_svg = (
            f'<circle cx="{cx}" cy="{cy}" r="14" fill="none" stroke="#FEF08A" '
            f'stroke-width="2.2" stroke-dasharray="3,2" />\n'
            f'<circle cx="{cx}" cy="{cy}" r="5" fill="#FEF08A" />'
        )

    zone_meta = STATCAST_ZONES.get(selected_zone, STATCAST_ZONES[0])
    caption_text = zone_meta["label"]

    raw_svg = f"""
    <div style="display:flex;justify-content:center;align-items:center;margin:6px 0;">
      <svg viewBox="0 0 240 245" width="100%" style="max-width:255px;background:#0B0F19;border-radius:12px;border:1px solid #26354A;padding:6px;display:block;">
        <rect x="20" y="14" width="100" height="96" rx="6" fill="{f11}" fill-opacity="{o11}" stroke="{s11}" stroke-width="1.5" stroke-dasharray="4,3" />
        <text x="36" y="32" text-anchor="middle" fill="#CBD5E1" font-size="11" font-weight="700" font-family="sans-serif">11</text>
        <rect x="120" y="14" width="100" height="96" rx="6" fill="{f12}" fill-opacity="{o12}" stroke="{s12}" stroke-width="1.5" stroke-dasharray="4,3" />
        <text x="204" y="32" text-anchor="middle" fill="#CBD5E1" font-size="11" font-weight="700" font-family="sans-serif">12</text>
        <rect x="20" y="110" width="100" height="96" rx="6" fill="{f13}" fill-opacity="{o13}" stroke="{s13}" stroke-width="1.5" stroke-dasharray="4,3" />
        <text x="36" y="196" text-anchor="middle" fill="#CBD5E1" font-size="11" font-weight="700" font-family="sans-serif">13</text>
        <rect x="120" y="110" width="100" height="96" rx="6" fill="{f14}" fill-opacity="{o14}" stroke="{s14}" stroke-width="1.5" stroke-dasharray="4,3" />
        <text x="204" y="196" text-anchor="middle" fill="#CBD5E1" font-size="11" font-weight="700" font-family="sans-serif">14</text>
        {chr(10).join(strike_cells_svg)}
        <rect x="52" y="42" width="135" height="135" fill="none" stroke="#38BDF8" stroke-width="2.6" />
        {marker_svg}
        <polygon points="85,216 155,216 142,226 120,233 98,226" fill="#E2E8F0" stroke="#64748B" stroke-width="1.4" />
        <text x="120" y="242" text-anchor="middle" fill="#38BDF8" font-size="8.5" font-weight="700" font-family="sans-serif">{caption_text}</text>
      </svg>
    </div>
    """
    return _compact_html(raw_svg)


def render_prediction_card_html(
    prediction: str, features: dict[str, Any], latency_ms: float = 0.0
) -> str:
    """
    Render the broadcast tactical prediction card with semantic category colors.

    - Fastball: `#EF4444` / `#DC2626`
    - Breaking Ball: `#3B82F6` / `#2563EB`
    - Offspeed: `#10B981` / `#059669`
    """
    style = PITCH_CATEGORY_STYLES.get(
        prediction,
        {
            "primary": "#38BDF8",
            "secondary": "#0284C7",
            "glow": "rgba(56, 189, 248, 0.35)",
            "bg_tint": "rgba(56, 189, 248, 0.12)",
            "icon": "⚾",
            "label_pt": prediction,
            "arsenal": "Arremesso Tático Previsto",
            "tactical_insight": "Predição gerada pelo modelo em produção no Unity Catalog.",
        },
    )

    primary = style["primary"]
    secondary = style["secondary"]
    glow = style["glow"]
    bg_tint = style["bg_tint"]
    icon = style["icon"]
    label_pt = style["label_pt"]
    arsenal = style["arsenal"]
    insight = style["tactical_insight"]

    count_state = features.get("count_state", "0-0")
    leverage = features.get("count_leverage", "Neutral")
    p_throws = features.get("p_throws", "R")
    stand = features.get("stand", "R")
    platoon_str = (
        "Mesma Mão (Favorável ao Arremessador)"
        if features.get("same_hand", 0) == 1
        else "Mão Oposta (Platoon Invertido)"
    )
    risp_str = (
        "RISP (Posição de Anotar)"
        if features.get("scoring_position", 0) == 1
        else (f"{features.get('runners_on_base', 0)} em Base" if features.get('runners_on_base', 0) > 0 else "Bases Limpas")
    )

    raw_html = f"""
    <div style="background:linear-gradient(145deg,#151E2E 0%,#0B0F19 100%);border:2px solid {primary};border-radius:14px;padding:20px 22px;box-shadow:0 8px 26px {glow};margin-bottom:14px;position:relative;overflow:hidden;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
        <span style="background:{secondary};color:#FFFFFF;font-size:11px;font-weight:800;letter-spacing:0.08em;padding:4px 10px;border-radius:6px;text-transform:uppercase;">PREDIÇÃO TÁTICA • STATCAST AI</span>
        <span style="background:#1E293B;color:#94A3B8;font-size:11px;font-weight:600;padding:4px 9px;border-radius:6px;border:1px solid #334155;">⚡ {latency_ms:.1f} ms</span>
      </div>
      <div style="background:{bg_tint};border-left:5px solid {primary};border-radius:8px;padding:14px 16px;margin:10px 0 14px 0;">
        <div style="font-size:28px;font-weight:900;color:#F8FAFC;line-height:1.15;">{icon} {prediction}</div>
        <div style="font-size:13px;font-weight:700;color:{primary};margin-top:4px;">{label_pt} — {arsenal}</div>
      </div>
      <div style="font-size:12.5px;color:#CBD5E1;line-height:1.45;margin-bottom:14px;">{insight}</div>
      <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:8px;border-top:1px solid #26354A;padding-top:12px;font-size:11.5px;">
        <div style="background:#0F172A;padding:7px 10px;border-radius:6px;border:1px solid #1E293B;">
          <span style="color:#64748B;display:block;font-size:10px;font-weight:700;">CONTAGEM &amp; LEVERAGE</span>
          <strong style="color:#F8FAFC;">{count_state} • {leverage}</strong>
        </div>
        <div style="background:#0F172A;padding:7px 10px;border-radius:6px;border:1px solid #1E293B;">
          <span style="color:#64748B;display:block;font-size:10px;font-weight:700;">PLATOON &amp; BASES</span>
          <strong style="color:#F8FAFC;">{p_throws}HP vs {stand}HB • {risp_str}</strong>
        </div>
      </div>
      <div style="margin-top:6px;font-size:10.5px;color:#64748B;text-align:right;">Confronto: {platoon_str}</div>
    </div>
    """
    return _compact_html(raw_html)


def render_scoreboard_banner_html(features: dict[str, Any], endpoint_name: str) -> str:
    """Render a top MLB Broadcast Scorebug / Scoreboard bar summarizing the live game state."""
    inning_arrow = "▲" if features.get("inning_topbot") == "Top" else "▼"
    inning_label = f"{inning_arrow} {features.get('inning', 1)}ª ENTRADA"
    outs = features.get("outs_when_up", 0)
    count_state = features.get("count_state", "0-0")
    leverage = features.get("count_leverage", "Neutral")
    score_diff = int(features.get("score_diff", 0))
    score_str = f"+{score_diff}" if score_diff > 0 else str(score_diff)

    leverage_color = (
        "#10B981"
        if leverage == "Pitcher Ahead"
        else ("#EF4444" if leverage == "Pitcher Behind" else "#38BDF8")
    )
    leverage_map = {
        "Pitcher Ahead": "Vantagem Arremessador",
        "Pitcher Behind": "Pressão no Arremessador",
        "Neutral": "Contagem Neutra",
    }
    leverage_pt = leverage_map.get(leverage, leverage)

    raw_banner = f"""
    <div style="background:linear-gradient(90deg,#0F172A 0%,#151E2E 50%,#0F172A 100%);border:1px solid #26354A;border-left:5px solid #38BDF8;border-radius:12px;padding:12px 20px;margin-bottom:18px;display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:12px;">
      <div style="display:flex;align-items:center;gap:14px;">
        <div style="font-size:26px;">⚾</div>
        <div>
          <div style="font-size:17px;font-weight:900;color:#F8FAFC;letter-spacing:0.04em;">MLB STATCAST • TACTICAL PITCH PREDICTOR</div>
          <div style="font-size:11.5px;color:#94A3B8;">Databricks Serverless Real-Time Inference • Endpoint: <code style="color:#38BDF8;">{endpoint_name}</code></div>
        </div>
      </div>
      <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
        <div style="background:#0B0F19;border:1px solid #334155;border-radius:8px;padding:6px 12px;text-align:center;">
          <div style="font-size:9.5px;color:#64748B;font-weight:700;">ENTRADA</div>
          <div style="font-size:13.5px;color:#F8FAFC;font-weight:800;">{inning_label}</div>
        </div>
        <div style="background:#0B0F19;border:1px solid #334155;border-radius:8px;padding:6px 12px;text-align:center;">
          <div style="font-size:9.5px;color:#64748B;font-weight:700;">CONTAGEM</div>
          <div style="font-size:13.5px;color:#F59E0B;font-weight:800;">{count_state} ({outs} Out)</div>
        </div>
        <div style="background:#0B0F19;border:1px solid #334155;border-radius:8px;padding:6px 12px;text-align:center;">
          <div style="font-size:9.5px;color:#64748B;font-weight:700;">SITUAÇÃO</div>
          <div style="font-size:13.5px;color:{leverage_color};font-weight:800;">{leverage_pt}</div>
        </div>
        <div style="background:#0B0F19;border:1px solid #334155;border-radius:8px;padding:6px 12px;text-align:center;">
          <div style="font-size:9.5px;color:#64748B;font-weight:700;">DIF. PLACAR</div>
          <div style="font-size:13.5px;color:#E2E8F0;font-weight:800;">{score_str}</div>
        </div>
      </div>
    </div>
    """
    return _compact_html(raw_banner)


def get_custom_css() -> str:
    """Return custom CSS implementing the Statcast dark broadcast theme."""
    raw_css = """
    <style>
      :root {
        --bg-main: #0B0F19;
        --surface-card: #151E2E;
        --surface-border: #26354A;
        --pitch-fastball: #EF4444;
        --pitch-breaking: #3B82F6;
        --pitch-offspeed: #10B981;
        --base-occupied: #F59E0B;
        --strike-zone: #38BDF8;
      }
      .stApp {
        background-color: #0B0F19;
        color: #F8FAFC;
      }
      .block-container,
      div[data-testid="stMainBlockContainer"] {
        max-width: 96% !important;
        padding-top: 2rem !important;
        padding-left: 2.2rem !important;
        padding-right: 2.2rem !important;
      }
      div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #151E2E;
        border: 1px solid #26354A !important;
        border-radius: 12px !important;
        padding: 14px !important;
      }
      .section-header {
        font-size: 13px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #38BDF8;
        margin-bottom: 8px;
        border-bottom: 1px solid #26354A;
        padding-bottom: 5px;
      }
      div.stButton > button {
        border-radius: 8px;
        font-weight: 700;
        transition: all 0.15s ease-in-out;
      }
    </style>
    """
    return _compact_html(raw_css)


# ---------------------------------------------------------------------------
# 5. Retratos Retro Pixel Art dos Astros da MLB (24x24)
# ---------------------------------------------------------------------------

PITCHER_PIXEL_MAPS: dict[str, dict[str, Any]] = {
    "Paul Skenes": {
        "palette": {
            ".": None,
            "K": "#111111",  # Black Pirates cap / jersey
            "G": "#FDB827",  # Gold Pirates logo / trim
            "S": "#FCD5B5",  # Skin
            "D": "#E5B995",  # Skin shadow
            "H": "#4E342E",  # Brown hair & signature mustache
            "E": "#231510",  # Dark eyes
            "W": "#FFFFFF",  # Jersey number
        },
        "rows": [
            "........................",
            ".........GGGG...........",
            ".......KKKKKKKK.........",
            "......KKKKKKKKKK........",
            ".....KKKKGGKKKKKK.......",
            ".....KKKKGGKKKKKK.......",
            "....KKKKKKKKKKKKKK......",
            "...KKKKKKKKKKKKKKKKK....",
            "....HHHSSSSSSSSHHH......",
            "....HHHSSSSSSSSHHH......",
            "....HHSEESSSSEESHH......",
            "....HHSSSSDDSSSSHH......",
            ".....HSSSSDEDSSSH.......",
            ".....HSSHEEEEHSSH.......",
            ".....HSSEEEEEESSH.......",
            "......SSSDDDDSS.........",
            ".......SSSSSSSS.........",
            "........SSSSSS..........",
            ".....KKKKGGGGKKKK.......",
            "...KKKKKKGGGGKKKKKK.....",
            "..KKKKKKKGGGGKKKKKKK....",
            ".KKKKWKKKKKKKKKKKKKKK...",
            ".KKKWWKKKKKKKKKKKKKKK...",
            ".KKKWWKKKKKKKKKKKKKKK...",
        ],
    },
    "Corbin Burnes": {
        "palette": {
            ".": None,
            "K": "#111111",  # Black cap crown
            "O": "#DF4601",  # Orioles Orange visor & jersey
            "Y": "#E5C158",  # Blonde curly hair
            "A": "#C69B34",  # Full blonde beard
            "S": "#FDE0D0",  # Fair skin
            "D": "#E8BEAA",  # Skin shadow
            "U": "#38BDF8",  # Blue eyes
            "W": "#FFFFFF",  # Logo detail
        },
        "rows": [
            "........................",
            ".........OOOO...........",
            ".......KKKKKKKK.........",
            "......KKKKKKKKKK........",
            ".....KKKKWKKKKKKK.......",
            ".....KKKKWWKKKKKK.......",
            "....KKKKKKKKKKKKKK......",
            "...OOOOOOOOOOOOOOOOO....",
            "....YYYSSSSSSSSYYY......",
            "...YYYYYSSSSSSSSYYYY....",
            "...YYYSUUSSSSUUSYYYY....",
            "...YYYYSSSDDDSSSYYYY....",
            "....YYSSSSDEDSSSSYY.....",
            ".....YSSAAAAAASSY.......",
            ".....YSAAAAAAAASY.......",
            "......AAAAAAAAAA........",
            ".......AAAAAAAA.........",
            "........SSSSSS..........",
            ".....OOOOOOOOOOOO.......",
            "...OOOOOOOOOOOOOOOO.....",
            "..OOOOOKKOOOOKKOOOOO....",
            ".OOOOOOKKOOOOKKOOOOOO...",
            ".OOOOOOOOOOOOOOOOOOOO...",
            ".OOOOOOOOOOOOOOOOOOOO...",
        ],
    },
    "Logan Webb": {
        "palette": {
            ".": None,
            "K": "#18181B",  # Black Giants cap / collar
            "O": "#FD5A1E",  # Giants Orange SF logo & visor
            "H": "#382415",  # Brown hair & sideburns
            "S": "#F5CEAB",  # Skin tone
            "D": "#DCAB83",  # Skin shadow
            "E": "#26180F",  # Dark eyes
            "W": "#FDFBF7",  # Giants off-white cream jersey
        },
        "rows": [
            "........................",
            ".........OOOO...........",
            ".......KKKKKKKK.........",
            "......KKKKKKKKKK........",
            ".....KKKKOKKKKKKK.......",
            ".....KKKKOOKKKKKK.......",
            "....KKKKKKKKKKKKKK......",
            "...OOOOOOOOOOOOOOOOO....",
            "....HHHSSSSSSSSHHH......",
            "....HHHSSSSSSSSHHH......",
            "....HHSEESSSSEESHH......",
            "....HHSSSSDDSSSSHH......",
            "....HHSSSSDDSSSSHH......",
            ".....HSSSSDDSSSSHH......",
            ".....HSSSSKKSSSSHH......",
            "......SSSDDDDSS.........",
            ".......SSSSSSSS.........",
            "........SSSSSS..........",
            ".....WWWWOKOWWWW........",
            "...WWWWWWOKOWWWWWW......",
            "..WWWWWWWOKOWWWWWWW.....",
            ".WWWWWWWWOKOWWWWWWWW....",
            ".WWWWWWWWKKKWWWWWWWW....",
            ".WWWWWWWWWWWWWWWWWWW....",
        ],
    },
    "Chris Sale": {
        "palette": {
            ".": None,
            "N": "#13274F",  # Braves Navy cap & jersey
            "R": "#CE1141",  # Scarlet Red visor & collar
            "K": "#1F1F1F",  # Dark hair
            "S": "#EED8C6",  # Pale lean skin
            "D": "#D1B49D",  # Sunken cheek shadow
            "T": "#4B5563",  # Southpaw dark stubble / scruff
            "E": "#1C1917",  # Intense dark eyes
            "W": "#FFFFFF",  # Logo
        },
        "rows": [
            "........................",
            ".........RRRR...........",
            ".......NNNNNNNN.........",
            "......NNNNNNNNNN........",
            ".....NNNNWWNNNNNN.......",
            ".....NNNWWWWNNNNN.......",
            "....NNNNNNNNNNNNNN......",
            "...RRRRRRRRRRRRRRRRR....",
            "....KKKSSSSSSSSKKK......",
            "....KKKSSSSSSSSKKK......",
            "....KKSEESSSSEESKK......",
            "....KKDDSSDDSSDDDK......",
            ".....KDSSSDEDSSDK.......",
            ".....KDSSSTTSSSDK.......",
            ".....KDSSTTTTSSDK.......",
            "......DDTTTTTTDD........",
            ".......DTTTTTTD.........",
            "........SSSSSS..........",
            ".....NNNNRRRRNNNN.......",
            "...NNNNNNRRRRNNNNNN.....",
            "..NNNNNNNRRRRNNNNNNN....",
            ".NNNNNNNNNNNNNNNNNNNN...",
            ".NNNNNNNNNNNNNNNNNNNN...",
            ".NNNNNNNNNNNNNNNNNNNN...",
        ],
    },
    "Yoshinobu Yamamoto": {
        "palette": {
            ".": None,
            "B": "#005A9C",  # Dodgers Royal Blue cap & jersey
            "W": "#FFFFFF",  # White LA logo & collar
            "K": "#0F172A",  # Sleek black hair
            "S": "#F8D7B0",  # Warm skin tone
            "D": "#DEB386",  # Skin shadow
            "E": "#0F172A",  # Almond dark eyes
            "R": "#EF4444",  # Dodgers red jersey accent (#18)
        },
        "rows": [
            "........................",
            ".........WWWW...........",
            ".......BBBBBBBB.........",
            "......BBBBBBBBBB........",
            ".....BBBBWWBBBBBB.......",
            ".....BBBBWWBBBBBB.......",
            "....BBBBBBBBBBBBBB......",
            "...BBBBBBBBBBBBBBBBB....",
            "....KKKSSSSSSSSKKK......",
            "....KKKSSSSSSSSKKK......",
            "....KKWEESSSSEEWKK......",
            "....KKSSSSDDSSSSKK......",
            ".....KSSSSDEDSSSK.......",
            ".....KSSSSDDSSSSK.......",
            ".....KSSSDDDDDSKK.......",
            "......SSSSDDSSSS........",
            ".......SSSSSSSS.........",
            "........SSSSSS..........",
            ".....BBBBWWWWBBBB.......",
            "...BBBBBBWWWWBBBBBB.....",
            "..BBBBBBBWKKWBBBBBBB....",
            ".BBBRBBBBWKKWBBBBBBBB...",
            ".BBRRBBBBBBBBBBBBBBBB...",
            ".BBRRBBBBBBBBBBBBBBBB...",
        ],
    },
}


def render_pixel_pitcher_svg(pitcher_name: str) -> str:
    """
    Renderiza um retrato retro 8-bit / 16-bit pixel-art do arremessador da MLB em SVG.

    Desenha boné oficial da franquia, traços característicos (mustache do Skenes,
    barba loira do Burnes, sideburns do Webb, etc.) e uniforme de jogo.
    """
    data = PITCHER_PIXEL_MAPS.get(pitcher_name, PITCHER_PIXEL_MAPS["Paul Skenes"])
    palette = data["palette"]
    rows = data["rows"]

    rects_svg: list[str] = []
    for y, row_str in enumerate(rows):
        cur_char = row_str[0]
        cur_len = 1
        start_x = 0
        for x in range(1, len(row_str)):
            if row_str[x] == cur_char:
                cur_len += 1
            else:
                color = palette.get(cur_char)
                if color:
                    rects_svg.append(
                        f'<rect x="{start_x}" y="{y}" width="{cur_len}" height="1" fill="{color}"/>'
                    )
                cur_char = row_str[x]
                cur_len = 1
                start_x = x
        color = palette.get(cur_char)
        if color:
            rects_svg.append(
                f'<rect x="{start_x}" y="{y}" width="{cur_len}" height="1" fill="{color}"/>'
            )

    rects_body = "".join(rects_svg)
    p_id = pitcher_name.replace(" ", "_")

    svg = f"""
    <svg viewBox="0 0 24 24" width="100%" height="135" shape-rendering="crispEdges" style="background: radial-gradient(circle at 50% 30%, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.98)); border-radius: 6px; border: 1px solid rgba(0, 255, 204, 0.25); display: block; margin: 0 auto 6px auto;">
        <defs>
            <pattern id="pixelGrid_{p_id}" width="2" height="2" patternUnits="userSpaceOnUse">
                <rect width="2" height="1" fill="rgba(255,255,255,0.02)"/>
            </pattern>
        </defs>
        <rect width="24" height="24" fill="url(#pixelGrid_{p_id})"/>
        {rects_body}
    </svg>
    """
    return _compact_html(svg)


# ---------------------------------------------------------------------------
# 6. Retratos Oficiais em Alta Definição dos Astros da MLB
# ---------------------------------------------------------------------------

PITCHER_HEADSHOT_FILES: dict[str, str] = {
    "Paul Skenes": "paul_skenes.png",
    "Corbin Burnes": "corbin_burnes.png",
    "Logan Webb": "logan_webb.png",
    "Chris Sale": "chris_sale.png",
    "Yoshinobu Yamamoto": "yoshinobu_yamamoto.png",
}

PITCHER_HEADSHOT_URLS: dict[str, str] = {
    "Paul Skenes": "https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_426,q_auto:best/v1/people/694973/headshot/67/current",
    "Corbin Burnes": "https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_426,q_auto:best/v1/people/669203/headshot/67/current",
    "Logan Webb": "https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_426,q_auto:best/v1/people/657277/headshot/67/current",
    "Chris Sale": "https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_426,q_auto:best/v1/people/519242/headshot/67/current",
    "Yoshinobu Yamamoto": "https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_426,q_auto:best/v1/people/808967/headshot/67/current",
}


@functools.lru_cache(maxsize=16)
def get_pitcher_headshot_src(pitcher_name: str) -> str:
    """Retorna o data URI base64 local da foto do arremessador, ou a URL oficial da MLB."""
    filename = PITCHER_HEADSHOT_FILES.get(pitcher_name)
    if filename:
        local_path = Path(__file__).resolve().parent / "assets" / "headshots" / filename
        if local_path.is_file():
            try:
                encoded = base64.b64encode(local_path.read_bytes()).decode("utf-8")
                return f"data:image/png;base64,{encoded}"
            except OSError:
                pass
    return PITCHER_HEADSHOT_URLS.get(pitcher_name, PITCHER_HEADSHOT_URLS["Paul Skenes"])


def render_pitcher_photo_html(pitcher_name: str) -> str:
    """Renderiza a foto real do arremessador da MLB em moldura retrô com iluminação neon."""
    src = get_pitcher_headshot_src(pitcher_name)
    raw_html = f"""
    <div style="background: radial-gradient(circle at 50% 30%, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.98)); border-radius: 6px; border: 1px solid rgba(0, 255, 204, 0.35); padding: 4px; margin: 2px auto 6px auto; height: 135px; display: flex; align-items: center; justify-content: center; box-shadow: inset 0 0 15px rgba(0, 255, 204, 0.1);">
        <img src="{src}" alt="{pitcher_name}" style="max-height: 125px; max-width: 100%; object-fit: contain; filter: drop-shadow(0 4px 6px rgba(0,0,0,0.7)); display: block; margin: 0 auto;" />
    </div>
    """
    return _compact_html(raw_html)



