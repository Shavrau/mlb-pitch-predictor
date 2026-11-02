"""

SVG and HTML UI components for the MLB Broadcast Editorial Dual-Mode Interface (Design System v2.0).

Provides:

- `STATCAST_ZONES`: Metadata for Statcast zones 0, 1-9 (Strike), and 11-14 (Chase).

- `PITCH_CATEGORY_STYLES`: Semantic colors, Statcast code badges, geometric markers, and descriptions

  for Fastball, Breaking Ball, and Offspeed categories (zero decorative emojis).

- `resolve_ui_theme_mode`: Resolves active Dual Mode (`"dark"` default Dark Broadcast or `"light"` Light Editorial).

- `get_custom_css`: Adaptive Dual-Mode CSS stylesheet with `--pp-*` custom properties, `color-scheme` on `:root`,

  tabular monospace numerals, and dynamic MLB franchise accents.

- `render_baseball_diamond_svg`: Crisp vector infield diamond with amber highlight (`#F59E0B`) on occupied bases.

- `render_strike_zone_svg`: Catcher's perspective Statcast Strike Zone (3x3 inner grid + 4 outer chase corners).

- `render_prediction_card_html`: Apple TV+ / Savant editorial scorebug card for tactical pitch predictions.

- `render_scoreboard_banner_html`: MLB Broadcast Scorebug header summarizing live game state.

- `render_pitcher_photo_html`: Official MLB headshot framed with franchise primary color header strip.

- `render_pitch_flight_arena_svg`: 3D perspective animated trajectory arena with Statcast Radar HUD.

"""

from __future__ import annotations

import base64
import functools
import math
from pathlib import Path
from typing import Any

from pitch_predictor.scenarios import MLB_PITCHERS

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
        "glow": "rgba(239, 68, 68, 0.16)",
        "bg_tint": "rgba(239, 68, 68, 0.10)",
        "icon": "▲",
        "code_badge": "4-SEAM [FF]",
        "label_pt": "Bola Rápida (Fastball)",
        "arsenal": "4-Seam (FF) • Sinker (SI) • Cutter (FC)",
        "tactical_insight": "Ataque de velocidade e potência para desafiar o rebatedor na zona ou elevar o nível dos olhos.",
    },
    "Breaking Ball": {
        "primary": "#3B82F6",
        "secondary": "#2563EB",
        "glow": "rgba(59, 130, 246, 0.16)",
        "bg_tint": "rgba(59, 130, 246, 0.10)",
        "icon": "◆",
        "code_badge": "BREAKING [SL/ST]",
        "label_pt": "Bola de Efeito (Breaking Ball)",
        "arsenal": "Slider (SL) • Sweeper (ST) • Curveball (CU) • Knuckle-Curve (KC)",
        "tactical_insight": "Quebra acentuada horizontal/vertical buscando swing no vazio (whiff) ou contato fraco.",
    },
    "Offspeed": {
        "primary": "#10B981",
        "secondary": "#059669",
        "glow": "rgba(16, 185, 129, 0.16)",
        "bg_tint": "rgba(16, 185, 129, 0.10)",
        "icon": "●",
        "code_badge": "OFFSPEED [CH/FS]",
        "label_pt": "Mudança de Velocidade (Offspeed)",
        "arsenal": "Changeup (CH) • Splitter (FS)",
        "tactical_insight": "Variação de velocidade com soltura idêntica à bola rápida (disfarce de trajetória) para desestabilizar o timing do swing e induzir groundballs.",
    },
}


def resolve_ui_theme_mode(mode: str | None = None) -> str:
    """

    Resolve the active UI theme mode (`"diamond"` or `"light"`).

    Defaults to `"diamond"` (Classic Diamond Onyx & Clay).

    Dark Broadcast has been consolidated into Classic Diamond.

    Zero green colors: Corinthians & MLB authentic tribute (Onyx, Chalk White, Infield Clay & Rawlings Leather).

    """

    candidate = mode

    if candidate is None:
        try:
            import streamlit as st

            candidate = st.session_state.get("ui_theme_mode")

        except Exception:  # noqa: BLE001
            candidate = None

    if candidate:
        cand_str = str(candidate).strip().lower()

        if "light" in cand_str:
            return "light"

    return "diamond"


def _compact_html(raw: str) -> str:
    """

    Collapse multi-line HTML/SVG markup into a single line without leading spaces.

    Streamlit's `st.markdown(..., unsafe_allow_html=True)` runs CommonMark before

    injecting HTML. Collapsing into a single line guarantees unbroken HTML/SVG rendering.

    """

    return " ".join(line.strip() for line in raw.splitlines() if line.strip())


def render_baseball_diamond_svg(
    on_1b: int, on_2b: int, on_3b: int, outs: int = 0
) -> str:
    """

    Render a precision editorial vector diagram of the baseball diamond.

    Occupied bases are illuminated in stadium amber (`#F59E0B`).

    Empty bases are rendered in crisp passive slate (`#334155`).

    """

    color_1b = "#F59E0B" if int(on_1b) > 0 else "#334155"

    color_2b = "#F59E0B" if int(on_2b) > 0 else "#334155"

    color_3b = "#F59E0B" if int(on_3b) > 0 else "#334155"

    stroke_1b = "#FEF3C7" if int(on_1b) > 0 else "#64748B"

    stroke_2b = "#FEF3C7" if int(on_2b) > 0 else "#64748B"

    stroke_3b = "#FEF3C7" if int(on_3b) > 0 else "#64748B"

    out1_fill = "#EF4444" if outs >= 1 else "none"

    out2_fill = "#EF4444" if outs >= 2 else "none"

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

    status_color = (
        "#F59E0B" if risp_active else ("#0284C7" if runners_count == 1 else "#94A3B8")
    )

    raw_svg = f"""

    <div style="display:flex;justify-content:center;align-items:center;margin:4px 0;">

      <svg viewBox="0 0 240 235" width="100%" style="max-width:248px;background:var(--pp-surface,#12141A);border-radius:8px;border:1px solid var(--pp-border,#222631);padding:6px;display:block;">

        <path d="M 22 105 A 110 110 0 0 1 218 105" fill="none" stroke="#475569" stroke-opacity="0.45" stroke-width="1.5" stroke-dasharray="4,4" />

        <polygon points="120,38 192,110 120,182 48,110" fill="rgba(15,23,42,0.35)" stroke="#64748B" stroke-width="1.8" />

        <circle cx="120" cy="110" r="12" fill="rgba(30,41,59,0.55)" stroke="#64748B" stroke-width="1.3" />

        <rect x="115" y="108" width="10" height="4" rx="1" fill="#E2E8F0" />

        <g transform="translate(120, 38) rotate(45)">

          <rect x="-12" y="-12" width="24" height="24" rx="2" fill="{color_2b}" stroke="{stroke_2b}" stroke-width="2" />

        </g>

        <text x="120" y="16" text-anchor="middle" fill="#94A3B8" font-size="10.5" font-weight="700" font-family="JetBrains Mono, monospace">2B</text>

        <g transform="translate(48, 110) rotate(45)">

          <rect x="-12" y="-12" width="24" height="24" rx="2" fill="{color_3b}" stroke="{stroke_3b}" stroke-width="2" />

        </g>

        <text x="18" y="114" text-anchor="middle" fill="#94A3B8" font-size="10.5" font-weight="700" font-family="JetBrains Mono, monospace">3B</text>

        <g transform="translate(192, 110) rotate(45)">

          <rect x="-12" y="-12" width="24" height="24" rx="2" fill="{color_1b}" stroke="{stroke_1b}" stroke-width="2" />

        </g>

        <text x="222" y="114" text-anchor="middle" fill="#94A3B8" font-size="10.5" font-weight="700" font-family="JetBrains Mono, monospace">1B</text>

        <polygon points="120,192 110,182 110,173 130,173 130,182" fill="#F8FAFC" stroke="#64748B" stroke-width="1.4" />

        <text x="72" y="218" fill="#94A3B8" font-size="9.5" font-weight="700" font-family="JetBrains Mono, monospace">OUTS:</text>

        <circle cx="116" cy="214" r="5.5" fill="{out1_fill}" stroke="#64748B" stroke-width="1.5" />

        <circle cx="134" cy="214" r="5.5" fill="{out2_fill}" stroke="#64748B" stroke-width="1.5" />

        <text x="120" y="231" text-anchor="middle" fill="{status_color}" font-size="8.2" font-weight="700" font-family="JetBrains Mono, monospace">{status_label}</text>

      </svg>

    </div>

    """

    return _compact_html(raw_svg)


def render_strike_zone_svg(
    selected_zone: int,
    prev_category: str = "First Pitch",
    theme_mode: str | None = None,
) -> str:
    """

    Render the official MLB Statcast Strike Zone SVG from the catcher's perspective.

    Dual-Mode adaptive: in Light Mode or consuming theme variables, the 3x3 inner grid lines

    use `#94A3B8` (1.5px stroke width) and the target impact reticle is rendered in

    stadium amber `#D97706` for maximum contrast across dark and light surfaces.

    """

    is_light = resolve_ui_theme_mode(theme_mode) == "light"

    cat_style = PITCH_CATEGORY_STYLES.get(prev_category, {})

    active_color = cat_style.get("primary", "#38BDF8")

    def _chase_style(zone_id: int) -> tuple[str, str, str]:

        if selected_zone == zone_id:
            return (active_color, "#F8FAFC", "0.68")

        if is_light:
            return ("#E2E8F0", "#94A3B8", "0.40")

        return ("#334155", "#64748B", "0.18")

    def _strike_style(zone_id: int) -> tuple[str, str, str]:

        if selected_zone == zone_id:
            return (active_color, "#FFFFFF", "0.82")

        if is_light:
            if zone_id == 5:
                return ("#E1E8F0", "#94A3B8", "0.55")

            return ("#F1F5F9", "#94A3B8", "0.45")

        if zone_id == 5:
            return ("#1E293B", "#94A3B8", "0.42")

        return ("#0F172A", "#94A3B8", "0.30")

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

        stroke_color = "#94A3B8" if selected_zone != z_id else stroke_c

        text_c = (
            "#FFFFFF"
            if selected_zone == z_id
            else ("#334155" if is_light else "#94A3B8")
        )

        font_w = "800" if selected_zone == z_id else "600"

        strike_cells_svg.append(
            f'<rect x="{x}" y="{y}" width="45" height="45" fill="{fill_c}" '
            f'fill-opacity="{opac}" stroke="{stroke_color}" stroke-width="1.5" />\n'
            f'<text x="{x + 22.5}" y="{y + 27}" text-anchor="middle" fill="{text_c}" '
            f'font-size="12" font-weight="{font_w}" font-family="JetBrains Mono, monospace">{z_id}</text>'
        )

    marker_svg = ""

    if selected_zone in zone_centers:
        cx, cy = zone_centers[selected_zone]

        marker_svg = (
            f'<circle cx="{cx}" cy="{cy}" r="13" fill="none" stroke="#D97706" '
            f'stroke-width="2.2" stroke-dasharray="3,2" />\n'
            f'<circle cx="{cx}" cy="{cy}" r="4.5" fill="#D97706" stroke="#FFFFFF" stroke-width="1.2" />'
        )

    zone_meta = STATCAST_ZONES.get(selected_zone, STATCAST_ZONES[0])

    caption_text = zone_meta["label"]

    outer_zone_stroke = "#0284C7" if is_light else "#38BDF8"

    caption_color = "#0369A1" if is_light else "#0284C7"

    raw_svg = f"""

    <div style="display:flex;justify-content:center;align-items:center;margin:4px 0;">

      <svg viewBox="0 0 240 245" width="100%" style="max-width:250px;background:var(--pp-surface,#12141A);border-radius:8px;border:1px solid var(--pp-border,#222631);padding:6px;display:block;">

        <rect x="20" y="14" width="100" height="96" rx="5" fill="{f11}" fill-opacity="{o11}" stroke="{s11}" stroke-width="1.2" stroke-dasharray="4,3" />

        <text x="36" y="32" text-anchor="middle" fill="#94A3B8" font-size="10.5" font-weight="700" font-family="JetBrains Mono, monospace">11</text>

        <rect x="120" y="14" width="100" height="96" rx="5" fill="{f12}" fill-opacity="{o12}" stroke="{s12}" stroke-width="1.2" stroke-dasharray="4,3" />

        <text x="204" y="32" text-anchor="middle" fill="#94A3B8" font-size="10.5" font-weight="700" font-family="JetBrains Mono, monospace">12</text>

        <rect x="20" y="110" width="100" height="96" rx="5" fill="{f13}" fill-opacity="{o13}" stroke="{s13}" stroke-width="1.2" stroke-dasharray="4,3" />

        <text x="36" y="196" text-anchor="middle" fill="#94A3B8" font-size="10.5" font-weight="700" font-family="JetBrains Mono, monospace">13</text>

        <rect x="120" y="110" width="100" height="96" rx="5" fill="{f14}" fill-opacity="{o14}" stroke="{s14}" stroke-width="1.2" stroke-dasharray="4,3" />

        <text x="204" y="196" text-anchor="middle" fill="#94A3B8" font-size="10.5" font-weight="700" font-family="JetBrains Mono, monospace">14</text>

        {chr(10).join(strike_cells_svg)}

        <rect x="52" y="42" width="135" height="135" fill="none" stroke="{outer_zone_stroke}" stroke-width="2.2" />

        {marker_svg}

        <ellipse cx="120" cy="225" rx="52" ry="12" fill="#C85A17" fill-opacity="0.22" stroke="#C85A17" stroke-width="0.8" stroke-dasharray="3,2" />

        <polygon points="84,217 156,217 143,227 120,234 97,227" fill="rgba(0,0,0,0.45)" />

        <polygon points="83,214 157,214 144,226 120,234 96,226" fill="#0A0B0E" stroke="#334155" stroke-width="1.2" />

        <polygon points="85,215 155,215 142,225 120,232 98,225" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.0" />

        <line x1="42" y1="210" x2="42" y2="236" stroke="#94A3B8" stroke-opacity="0.35" stroke-width="1.2" stroke-dasharray="3,2" />

        <line x1="198" y1="210" x2="198" y2="236" stroke="#94A3B8" stroke-opacity="0.35" stroke-width="1.2" stroke-dasharray="3,2" />

        <text x="120" y="243" text-anchor="middle" fill="{caption_color}" font-size="8.2" font-weight="700" font-family="JetBrains Mono, monospace">{caption_text}</text>

      </svg>

    </div>

    """

    return _compact_html(raw_svg)


def render_kinematic_card_html(
    label: str,
    value: str,
    delta: str,
    is_warning: bool = False,
    theme_mode: str | None = None,
) -> str:
    """

    Renderiza um card de telemetria cinemática 3D com contraste rigoroso em Light e Dark mode.

    Garante legibilidade absoluta da label técnica, valor mono e badge de delta.

    """
    resolved = resolve_ui_theme_mode(theme_mode)

    delta_clean = delta.strip()

    is_neg = "-" in delta_clean or "↓" in delta_clean

    is_pos = "+" in delta_clean or "↑" in delta_clean

    if is_warning:
        badge_bg = "rgba(239, 68, 68, 0.14)"

        badge_border = "rgba(239, 68, 68, 0.35)"

        badge_color = "#DC2626"

    elif is_pos:
        badge_bg = (
            "rgba(2, 132, 199, 0.14)"
            if resolved == "light"
            else "rgba(56, 189, 248, 0.14)"
        )

        badge_border = (
            "rgba(2, 132, 199, 0.35)"
            if resolved == "light"
            else "rgba(56, 189, 248, 0.35)"
        )

        badge_color = "#0284C7" if resolved == "light" else "#38BDF8"

    elif is_neg:
        badge_bg = "rgba(239, 68, 68, 0.14)"

        badge_border = "rgba(239, 68, 68, 0.35)"

        badge_color = "#DC2626"

    else:
        badge_bg = "var(--pp-surface-alt, #E1E8F0)"

        badge_border = "var(--pp-border, #C2CFDD)"

        badge_color = "var(--pp-muted, #475569)"

    raw_html = f"""

    <div style="background:var(--pp-surface,#F1F5F9);border:1px solid var(--pp-border,#C2CFDD);border-radius:8px;padding:12px 14px;box-sizing:border-box;margin-bottom:8px;">

      <div style="font-size:11px;font-weight:700;color:var(--pp-muted,#475569);text-transform:uppercase;letter-spacing:0.04em;line-height:1.2;margin-bottom:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{label}</div>

      <div style="font-size:24px;font-weight:800;font-family:'JetBrains Mono',monospace;font-variant-numeric:tabular-nums;color:var(--pp-text,#0F172A);line-height:1.15;margin-bottom:6px;">{value}</div>

      <div style="display:inline-flex;align-items:center;background:{badge_bg};border:1px solid {badge_border};border-radius:4px;padding:2px 7px;font-size:11px;font-weight:700;color:{badge_color};font-family:'JetBrains Mono',monospace;line-height:1.2;">

        {delta_clean}

      </div>

    </div>

    """

    return " ".join(line.strip() for line in raw_html.splitlines() if line.strip())


def render_prediction_card_html(
    prediction: str, features: dict[str, Any], latency_ms: float = 0.0
) -> str:
    """

    Render the broadcast editorial tactical prediction card with semantic category colors:

    - Fastball: `#EF4444` / `#DC2626`

    - Breaking Ball: `#3B82F6` / `#2563EB`

    - Offspeed: `#10B981` / `#059669`

    """

    style = PITCH_CATEGORY_STYLES.get(
        prediction,
        {
            "primary": "#38BDF8",
            "secondary": "#0284C7",
            "glow": "rgba(56, 189, 248, 0.16)",
            "bg_tint": "rgba(56, 189, 248, 0.10)",
            "icon": "▲",
            "code_badge": "STATCAST [AI]",
            "label_pt": prediction,
            "arsenal": "Arremesso Tático Previsto",
            "tactical_insight": "Predição gerada pelo modelo em produção no Unity Catalog.",
        },
    )

    primary = style["primary"]

    secondary = style["secondary"]

    bg_tint = style["bg_tint"]

    icon = style["icon"]

    code_badge = style.get("code_badge", "STATCAST [AI]")

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
        else (
            f"{features.get('runners_on_base', 0)} em Base"
            if features.get("runners_on_base", 0) > 0
            else "Bases Limpas"
        )
    )

    raw_html = f"""

    <div style="background:var(--pp-surface,#12141A);border:1px solid var(--pp-border,#222631);border-top:3px solid {primary};border-radius:8px;padding:18px 20px;margin-bottom:12px;">

      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;gap:8px;flex-wrap:wrap;">

        <div style="display:flex;align-items:center;gap:8px;">

          <span style="background:{secondary};color:#FFFFFF;font-family:'JetBrains Mono',monospace;font-size:10.5px;font-weight:800;letter-spacing:0.06em;padding:3px 9px;border-radius:4px;text-transform:uppercase;">PREDIÇÃO TÁTICA • STATCAST AI</span>

          <span style="font-family:'JetBrains Mono',monospace;font-size:10.5px;font-weight:700;color:{primary};letter-spacing:0.04em;">{code_badge}</span>

        </div>

        <span style="background:var(--pp-surface-alt,#181B23);color:var(--pp-muted,#94A3B8);font-family:'JetBrains Mono',monospace;font-variant-numeric:tabular-nums;font-size:11px;font-weight:700;padding:3px 9px;border-radius:4px;border:1px solid var(--pp-border,#222631);">{latency_ms:.1f} ms</span>

      </div>

      <div style="background:{bg_tint};border-left:3px solid {primary};border-radius:4px;padding:12px 15px;margin:8px 0 12px 0;">

        <div style="font-size:25px;font-weight:800;color:var(--pp-text,#F1F5F9);line-height:1.15;letter-spacing:0.01em;">{icon} {prediction}</div>

        <div style="font-size:12.5px;font-weight:700;color:{primary};margin-top:4px;">{label_pt} — {arsenal}</div>

      </div>

      <div style="font-size:12.5px;color:var(--pp-text,#F1F5F9);opacity:0.88;line-height:1.45;margin-bottom:12px;">{insight}</div>

      <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:8px;border-top:1px solid var(--pp-border,#222631);padding-top:10px;font-size:11.5px;">

        <div style="background:var(--pp-surface-alt,#181B23);padding:7px 10px;border-radius:4px;border:1px solid var(--pp-border,#222631);">

          <span style="color:var(--pp-muted,#94A3B8);display:block;font-family:'JetBrains Mono',monospace;font-size:9.5px;font-weight:700;letter-spacing:0.04em;">CONTAGEM &amp; LEVERAGE</span>

          <strong style="color:var(--pp-text,#F1F5F9);font-family:'JetBrains Mono',monospace;font-variant-numeric:tabular-nums;">{count_state} • {leverage}</strong>

        </div>

        <div style="background:var(--pp-surface-alt,#181B23);padding:7px 10px;border-radius:4px;border:1px solid var(--pp-border,#222631);">

          <span style="color:var(--pp-muted,#94A3B8);display:block;font-family:'JetBrains Mono',monospace;font-size:9.5px;font-weight:700;letter-spacing:0.04em;">PLATOON &amp; BASES</span>

          <strong style="color:var(--pp-text,#F1F5F9);font-family:'JetBrains Mono',monospace;">{p_throws}HP vs {stand}HB • {risp_str}</strong>

        </div>

      </div>

      <div style="margin-top:6px;font-size:10.5px;color:var(--pp-muted,#94A3B8);text-align:right;">Confronto: {platoon_str}</div>

    </div>

    """

    return _compact_html(raw_html)


def render_scoreboard_banner_html(
    features: dict[str, Any], endpoint_name: str, theme_mode: str | None = None
) -> str:
    """Render an Apple TV+ / MLB Broadcast Scorebug bar summarizing the live game state."""
    resolved = resolve_ui_theme_mode(theme_mode)

    inning_arrow = "▲" if features.get("inning_topbot") == "Top" else "▼"

    inning_label = f"{inning_arrow} {features.get('inning', 1)}ª ENTRADA"

    outs = features.get("outs_when_up", 0)

    count_state = features.get("count_state", "0-0")

    leverage = features.get("count_leverage", "Neutral")

    score_diff = int(features.get("score_diff", 0))

    score_str = f"+{score_diff}" if score_diff > 0 else str(score_diff)

    leverage_color = (
        ("#0284C7" if resolved == "light" else "#38BDF8")
        if leverage == "Pitcher Ahead"
        else ("#EF4444" if leverage == "Pitcher Behind" else "#94A3B8")
    )

    leverage_map = {
        "Pitcher Ahead": "Vantagem Arremessador",
        "Pitcher Behind": "Pressão no Arremessador",
        "Neutral": "Contagem Neutra",
    }

    leverage_pt = leverage_map.get(leverage, leverage)

    raw_banner = f"""

    <div style="background:var(--pp-surface,#12141A);border:1px solid var(--pp-border,#222631);border-left:3px solid #38BDF8;border-radius:8px;padding:12px 18px;margin-bottom:16px;display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:12px;">

      <div style="display:flex;align-items:center;gap:12px;">

        <div style="background:#38BDF8;color:#0A0B0E;font-family:'JetBrains Mono',monospace;font-size:10.5px;font-weight:800;letter-spacing:0.06em;padding:4px 8px;border-radius:4px;">MLB • LIVE</div>

        <div>

          <div style="font-size:16px;font-weight:800;color:var(--pp-text,#F1F5F9);letter-spacing:0.03em;">MLB STATCAST • TACTICAL PITCH PREDICTOR</div>

          <div style="font-size:11px;color:var(--pp-muted,#94A3B8);">Databricks Serverless Real-Time Inference • Endpoint: <code style="color:#38BDF8;font-family:'JetBrains Mono',monospace;">{endpoint_name}</code></div>

        </div>

      </div>

      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-family:'JetBrains Mono',monospace;font-variant-numeric:tabular-nums;">

        <div style="background:var(--pp-surface-alt,#181B23);border:1px solid var(--pp-border,#222631);border-radius:6px;padding:5px 11px;text-align:center;">

          <div style="font-size:9px;color:var(--pp-muted,#94A3B8);font-weight:700;">ENTRADA</div>

          <div style="font-size:12.5px;color:var(--pp-text,#F1F5F9);font-weight:800;">{inning_label}</div>

        </div>

        <div style="background:var(--pp-surface-alt,#181B23);border:1px solid var(--pp-border,#222631);border-radius:6px;padding:5px 11px;text-align:center;">

          <div style="font-size:9px;color:var(--pp-muted,#94A3B8);font-weight:700;">CONTAGEM</div>

          <div style="font-size:12.5px;color:#F59E0B;font-weight:800;">{count_state} ({outs} Out)</div>

        </div>

        <div style="background:var(--pp-surface-alt,#181B23);border:1px solid var(--pp-border,#222631);border-radius:6px;padding:5px 11px;text-align:center;">

          <div style="font-size:9px;color:var(--pp-muted,#94A3B8);font-weight:700;">SITUAÇÃO</div>

          <div style="font-size:12.5px;color:{leverage_color};font-weight:800;">{leverage_pt}</div>

        </div>

        <div style="background:var(--pp-surface-alt,#181B23);border:1px solid var(--pp-border,#222631);border-radius:6px;padding:5px 11px;text-align:center;">

          <div style="font-size:9px;color:var(--pp-muted,#94A3B8);font-weight:700;">DIF. PLACAR</div>

          <div style="font-size:12.5px;color:var(--pp-text,#F1F5F9);font-weight:800;">{score_str}</div>

        </div>

      </div>

    </div>

    """

    return _compact_html(raw_banner)


def get_custom_css(theme_mode: str | None = None, accent_hex: str = "#38BDF8") -> str:
    """

    Return authentic MLB Ballpark custom CSS implementing Classic Diamond (Dark) & Light Editorial.

    Dark Broadcast has been replaced by the bespoke Classic Diamond theme.

    Zero green colors: Corinthians & MLB authentic tribute (Onyx, Infield Clay, Chalk White & Rawlings Leather).

    """

    resolved = resolve_ui_theme_mode(theme_mode)

    if resolved == "light":
        # Light Editorial Theme (Zero Green - Official Press Box & Scorer Parchment)

        pp_bg = "#F4F6F9"

        pp_surface = "#FFFFFF"

        pp_surface_alt = "#EDF2F7"

        pp_border = "#CBD5E1"

        pp_text = "#0F172A"

        pp_muted = "#475569"

        pp_sidebar_bg = "#E8EDF3"

        color_scheme = "light"

        scroll_thumb = "#94A3B8"

        scroll_track = "#E2E8F0"

        metric_label_color = "#1E293B"

        # Zero green (Corinthians standard - Floodlight Blue & Crimson):

        metric_delta_pos_bg = "rgba(2, 132, 199, 0.12)"

        metric_delta_pos_border = "rgba(2, 132, 199, 0.35)"

        metric_delta_pos_color = "#0369A1"

        metric_delta_pos_fill = "#0284C7"

        metric_delta_neg_bg = "rgba(220, 38, 38, 0.12)"

        metric_delta_neg_border = "rgba(220, 38, 38, 0.35)"

        metric_delta_neg_color = "#991B1B"

        metric_delta_neg_fill = "#B91C1C"

        slider_tick_color = "#475569"

        alert_success_bg = "#F0F9FF"

        alert_success_border = "#BAE6FD"

        alert_success_text = "#0369A1"

        alert_info_bg = "#EFF6FF"

        alert_info_border = "#BFDBFE"

        alert_info_text = "#1E3A8A"

        pitch_offspeed_var = "#0284C7"

        panel_bg = "#FFFFFF"

        panel_border = "#CBD5E1"

        panel_shadow = (
            "0 2px 8px rgba(15, 23, 42, 0.05), inset 0 1px 0 rgba(255, 255, 255, 0.8)"
        )

        metric_bg = "linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%)"

        metric_border = "#CBD5E1"

        metric_shadow = (
            "0 2px 6px rgba(15, 23, 42, 0.04), inset 0 1px 0 rgba(255, 255, 255, 0.8)"
        )

        tab_bg_active = "rgba(2, 132, 199, 0.08)"

        pinstripe_css = """

        background-color: var(--pp-bg) !important;

        background-image: 

          linear-gradient(to right, rgba(15, 23, 42, 0.035) 1px, transparent 1px),

          linear-gradient(to bottom, rgba(15, 23, 42, 0.035) 1px, transparent 1px) !important;

        background-size: 28px 28px !important;

        background-attachment: fixed !important;

        """

    else:
        # Classic Diamond Theme (Zero Green - Onyx Ballpark, Floodlight Vignette & Infield Clay)

        pp_bg = "#08090C"

        pp_surface = "#11131B"

        pp_surface_alt = "#171A24"

        pp_border = "#262C3D"

        pp_text = "#F8FAFC"

        pp_muted = "#94A3B8"

        pp_sidebar_bg = "#07080B"

        color_scheme = "dark"

        scroll_thumb = "#3A4459"

        scroll_track = "#08090C"

        metric_label_color = "#E2E8F0"

        metric_delta_pos_bg = "rgba(2, 132, 199, 0.18)"

        metric_delta_pos_border = "rgba(2, 132, 199, 0.40)"

        metric_delta_pos_color = "#38BDF8"

        metric_delta_pos_fill = "#0284C7"

        metric_delta_neg_bg = "rgba(220, 38, 38, 0.18)"

        metric_delta_neg_border = "rgba(220, 38, 38, 0.40)"

        metric_delta_neg_color = "#FCA5A5"

        metric_delta_neg_fill = "#F87171"

        slider_tick_color = "#94A3B8"

        alert_success_bg = "rgba(2, 132, 199, 0.20)"

        alert_success_border = "#0284C7"

        alert_success_text = "#BAE6FD"

        alert_info_bg = "rgba(200, 90, 23, 0.22)"

        alert_info_border = "#C85A17"

        alert_info_text = "#FED7AA"

        pitch_offspeed_var = "#38BDF8"

        panel_bg = "linear-gradient(180deg, #131622 0%, #0C0E15 100%)"

        panel_border = "var(--pp-border)"

        panel_shadow = (
            "0 4px 16px rgba(0, 0, 0, 0.42), inset 0 1px 0 rgba(255, 255, 255, 0.04)"
        )

        metric_bg = "linear-gradient(180deg, #121520 0%, #0A0C13 100%)"

        metric_border = "#23293A"

        metric_shadow = (
            "0 3px 12px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.03)"
        )

        tab_bg_active = "rgba(255, 255, 255, 0.05)"

        pinstripe_css = """

        background-color: var(--pp-bg) !important;

        background-image: 

          radial-gradient(ellipse 90% 45% at 50% -10%, rgba(56, 189, 248, 0.08) 0%, rgba(200, 90, 23, 0.035) 45%, transparent 75%),

          repeating-linear-gradient(to right, transparent, transparent 47px, rgba(248, 250, 252, 0.022) 48px, transparent 49px) !important;

        background-attachment: fixed !important;

        """

    raw_css = f"""

    <style>

      @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=JetBrains+Mono:wght@500;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

      :root {{

        color-scheme: {color_scheme};

        --pp-bg: {pp_bg};

        --pp-surface: {pp_surface};

        --pp-surface-alt: {pp_surface_alt};

        --pp-border: {pp_border};

        --pp-text: {pp_text};

        --pp-muted: {pp_muted};

        --pp-accent: {accent_hex};

        --bg-main: {pp_bg};

        --surface-card: {pp_surface};

        --surface-border: {pp_border};

        --pitch-fastball: #EF4444;

        --pitch-breaking: #3B82F6;

        --pitch-offspeed: {pitch_offspeed_var};

        --base-occupied: #F59E0B;

        --strike-zone: #38BDF8;

        --pp-clay: #C85A17;

        --pp-dirt: #B45309;

        --pp-seam: #DC2626;

        --pp-chalk: #F8FAFC;

        --pp-leather: #78350F;

        --pp-onyx: #090A0E;

        --pp-scoreboard: #0F141C;

        scrollbar-width: thin;

        scrollbar-color: {scroll_thumb} {scroll_track};

      }}

      .stApp {{

        background-color: var(--pp-bg) !important;

        {pinstripe_css}

        color: var(--pp-text) !important;

        font-family: 'Space Grotesk', system-ui, sans-serif;

      }}

      /* Remove default Streamlit top navbar and toolbar */
      header[data-testid="stHeader"],
      header.stAppHeader,
      .stAppHeader,
      div[data-testid="stHeader"] {{
        background: transparent !important;
        background-color: transparent !important;
        color: transparent !important;
        height: 0px !important;
        min-height: 0px !important;
        max-height: 0px !important;
        padding: 0px !important;
        margin: 0px !important;
        border: none !important;
        box-shadow: none !important;
        pointer-events: none !important;
        visibility: hidden !important;
        z-index: 100 !important;
      }}

      /* Hide default Streamlit toolbar and decoration */
      [data-testid="stToolbar"],
      .stToolbar,
      [data-testid="stDecoration"],
      .stDecoration,
      #MainMenu,
      [data-testid="stStatusWidget"],
      header[data-testid="stHeader"] [data-testid="stToolbar"] {{
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
        width: 0px !important;
        pointer-events: none !important;
      }}

      /* Preserve sidebar expand toggle if collapsed */
      [data-testid="stSidebarCollapsedControl"],
      header[data-testid="stHeader"] [data-testid="stSidebarCollapsedControl"] {{
        pointer-events: auto !important;
        visibility: visible !important;
        display: flex !important;
        position: fixed !important;
        top: 10px !important;
        left: 10px !important;
        z-index: 999999 !important;
        background: var(--pp-surface) !important;
        border: 1px solid var(--pp-border) !important;
        border-radius: 6px !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15) !important;
      }}

      [data-testid="stSidebarCollapsedControl"] button {{
        color: var(--pp-text) !important;
      }}

      section[data-testid="stSidebar"] {{

        background-color: {pp_sidebar_bg} !important;

        border-right: 1px solid var(--pp-border) !important;

      }}

      section[data-testid="stSidebar"] * {{

        color: var(--pp-text);

      }}

      .block-container,

      div[data-testid="stMainBlockContainer"] {{

        max-width: 96% !important;

        padding-top: 1.5rem !important;

        padding-left: 2rem !important;

        padding-right: 2rem !important;

      }}

      /* Dugout Tactical Panels (st.container border=True) */
      div[data-testid="stVerticalBlockBorderWrapper"] {{

        background: {panel_bg} !important;

        border: 1px solid {panel_border} !important;

        border-radius: 8px !important;

        padding: 16px !important;

        box-shadow: {panel_shadow} !important;

        transition: border-color 0.15s ease !important;

      }}

      div[data-testid="stVerticalBlockBorderWrapper"]:hover {{

        border-color: rgba(255, 255, 255, 0.14) !important;

      }}

      /* Electronic Ballpark Scoreboard Tabs (st.tabs) */
      div[data-baseweb="tab-list"] {{

        background-color: transparent !important;

        border-bottom: 1px solid var(--pp-border) !important;

        gap: 6px !important;

        padding-bottom: 0px !important;

      }}

      button[data-baseweb="tab"] {{

        font-family: 'Barlow Condensed', 'Space Grotesk', sans-serif !important;

        font-size: 14.5px !important;

        font-weight: 600 !important;

        letter-spacing: 0.01em !important;

        text-transform: none !important;

        color: var(--pp-muted) !important;

        background-color: transparent !important;

        border-top-left-radius: 6px !important;

        border-top-right-radius: 6px !important;

        border: none !important;

        border-bottom: 2.5px solid transparent !important;

        padding: 8px 16px !important;

        transition: all 0.15s ease !important;

      }}

      button[data-baseweb="tab"]:hover {{

        color: var(--pp-text) !important;

        background-color: rgba(255, 255, 255, 0.03) !important;

      }}

      button[data-baseweb="tab"][aria-selected="true"] {{

        color: var(--pp-text) !important;

        border-bottom: 2.5px solid var(--pp-accent) !important;

        background-color: {tab_bg_active} !important;

      }}

      div[data-baseweb="tab-highlight"] {{

        background-color: var(--pp-accent) !important;

        height: 2.5px !important;

      }}

      div[data-testid="stMetricValue"],

      .pp-mono {{

        font-family: 'JetBrains Mono', monospace !important;

        font-variant-numeric: tabular-nums !important;

      }}

      .pp-display {{

        font-family: 'Barlow Condensed', 'Space Grotesk', sans-serif !important;

        letter-spacing: 0.03em;

      }}

      .pp-seam-divider {{

        display: flex;

        align-items: center;

        justify-content: center;

        margin: 12px 0;

      }}

      .pp-dugout-clipboard {{

        position: relative;

        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);

        margin: 18px 0 12px 0;

      }}

      .section-header {{

        font-family: 'Barlow Condensed', 'Space Grotesk', sans-serif;

        font-size: 14.5px;

        font-weight: 600;

        text-transform: none;

        letter-spacing: 0.01em;

        color: var(--pp-text);

        margin-bottom: 8px;

        border-bottom: 1px solid var(--pp-border);

        border-left: 3px solid var(--pp-accent);

        padding: 2px 0 5px 9px;

      }}

      h1, h2, h3, h4, h5, h6,

      .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {{

        color: var(--pp-text) !important;

        font-family: 'Barlow Condensed', 'Space Grotesk', sans-serif !important;

        letter-spacing: 0.02em;

      }}

      /* Dugout Clipboards (st.expander) */
      div[data-testid="stExpander"] {{

        background-color: var(--pp-surface) !important;

        border: 1px solid var(--pp-border) !important;

        border-left: 3.5px solid var(--pp-accent) !important;

        border-radius: 8px !important;

        margin-bottom: 12px !important;

        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12) !important;

      }}

      div[data-testid="stExpander"] details {{

        border-color: transparent !important;

        background-color: var(--pp-surface) !important;

        border-radius: 8px !important;

      }}

      summary[data-testid="stExpanderSummary"],

      summary[data-testid="stExpanderSummary"] *,

      div[data-testid="stExpander"] summary,

      div[data-testid="stExpander"] summary * {{

        color: var(--pp-text) !important;

        font-family: 'Barlow Condensed', 'Space Grotesk', sans-serif !important;

        font-weight: 600 !important;

        font-size: 14.5px !important;

        letter-spacing: 0.01em !important;

        text-transform: none !important;

      }}

      summary[data-testid="stExpanderSummary"]:hover,

      summary[data-testid="stExpanderSummary"]:hover * {{

        color: var(--pp-accent) !important;

      }}

      div[data-testid="stExpanderDetails"] {{

        background-color: var(--pp-surface) !important;

        color: var(--pp-text) !important;

        padding-top: 6px !important;

      }}

      /* Stalker Radar Gun Telemetry Boxes (st.metric) */
      div[data-testid="stMetric"] {{

        background: {metric_bg} !important;

        border: 1px solid {metric_border} !important;

        border-left: 3.5px solid var(--pp-accent) !important;

        border-radius: 8px !important;

        padding: 12px 16px !important;

        box-shadow: {metric_shadow} !important;

        position: relative !important;

      }}

      div[data-testid="stMetric"] label,

      div[data-testid="stMetricLabel"],

      div[data-testid="stMetricLabel"] p,

      div[data-testid="stMetricLabel"] span,

      div[data-testid="stMetricLabel"] * {{

        color: {metric_label_color} !important;

        font-family: 'JetBrains Mono', monospace !important;

        font-size: 11px !important;

        font-weight: 600 !important;

        text-transform: uppercase !important;

        letter-spacing: 0.04em !important;

        opacity: 1 !important;

        visibility: visible !important;

      }}

      div[data-testid="stMetricValue"],

      div[data-testid="stMetricValue"] * {{

        color: var(--pp-text) !important;

        font-family: 'JetBrains Mono', monospace !important;

        font-variant-numeric: tabular-nums !important;

        font-size: 26px !important;

        font-weight: 700 !important;

        letter-spacing: -0.01em !important;

      }}

      div[data-testid="stMetricDelta"] {{

        display: inline-flex !important;

        align-items: center !important;

        background-color: {metric_delta_pos_bg} !important;

        border: 1px solid {metric_delta_pos_border} !important;

        border-radius: 4px !important;

        padding: 2px 7px !important;

        margin-top: 4px !important;

      }}

      div[data-testid="stMetricDelta"] svg {{

        fill: {metric_delta_pos_fill} !important;

      }}

      div[data-testid="stMetricDelta"] div,

      div[data-testid="stMetricDelta"] span {{

        color: {metric_delta_pos_color} !important;

        font-weight: 800 !important;

        font-family: 'JetBrains Mono', monospace !important;

        font-size: 11.5px !important;

      }}

      div[data-testid="stMetricDelta"]:has(svg[data-testid*="Down" i]),

      div[data-testid="stMetricDelta"]:has(svg[data-testid*="down" i]),

      div[data-testid="stMetricDelta"]:has(svg[data-testid*="Decrease" i]) {{

        background-color: {metric_delta_neg_bg} !important;

        border: 1px solid {metric_delta_neg_border} !important;

      }}

      div[data-testid="stMetricDelta"]:has(svg[data-testid*="Down" i]) svg,

      div[data-testid="stMetricDelta"]:has(svg[data-testid*="down" i]) svg,

      div[data-testid="stMetricDelta"]:has(svg[data-testid*="Decrease" i]) svg {{

        fill: {metric_delta_neg_fill} !important;

      }}

      div[data-testid="stMetricDelta"]:has(svg[data-testid*="Down" i]) div,

      div[data-testid="stMetricDelta"]:has(svg[data-testid*="down" i]) div,

      div[data-testid="stMetricDelta"]:has(svg[data-testid*="Decrease" i]) div,

      div[data-testid="stMetricDelta"]:has(svg[data-testid*="Down" i]) span,

      div[data-testid="stMetricDelta"]:has(svg[data-testid*="down" i]) span,

      div[data-testid="stMetricDelta"]:has(svg[data-testid*="Decrease" i]) span {{

        color: {metric_delta_neg_color} !important;

      }}

      /* Streamlit Notifications / Alerts (st.success, st.info, st.warning) */

      div[data-testid="stAlert"] {{

        border-radius: 8px !important;

      }}

      div[data-testid="stAlert"] [data-testid="stNotificationContentSuccess"],

      div[data-testid="stAlert"]:has(svg[data-testid*="check" i]),

      div[data-testid="stAlert"]:has([aria-label*="success" i]) {{

        background-color: {alert_success_bg} !important;

        border: 1px solid {alert_success_border} !important;

        color: {alert_success_text} !important;

      }}

      div[data-testid="stAlert"] [data-testid="stNotificationContentSuccess"] *,

      div[data-testid="stAlert"]:has(svg[data-testid*="check" i]) *,

      div[data-testid="stAlert"]:has([aria-label*="success" i]) * {{

        color: {alert_success_text} !important;

      }}

      div[data-testid="stAlert"] [data-testid="stNotificationContentInfo"],

      div[data-testid="stAlert"]:has(svg[data-testid*="info" i]),

      div[data-testid="stAlert"]:has([aria-label*="info" i]) {{

        background-color: {alert_info_bg} !important;

        border: 1px solid {alert_info_border} !important;

        color: {alert_info_text} !important;

      }}

      div[data-testid="stAlert"] [data-testid="stNotificationContentInfo"] *,

      div[data-testid="stAlert"]:has(svg[data-testid*="info" i]) *,

      div[data-testid="stAlert"]:has([aria-label*="info" i]) * {{

        color: {alert_info_text} !important;

      }}

      /* Slider Subtext & Thumb Values */

      div[data-testid="stSlider"] [data-testid="stThumbValue"] {{

        color: var(--pp-text) !important;

        font-family: 'JetBrains Mono', monospace !important;

        font-weight: 700 !important;

      }}

      div[data-testid="stSlider"] [data-testid="stTickBarMin"],

      div[data-testid="stSlider"] [data-testid="stTickBarMax"] {{

        color: {slider_tick_color} !important;

        font-family: 'JetBrains Mono', monospace !important;

        font-weight: 700 !important;

      }}

      div[data-testid="stSlider"] label,

      div[data-testid="stCheckbox"] label,

      div[data-testid="stSelectbox"] label,

      div[data-testid="stRadio"] label,

      div[data-testid="stTextInput"] label,

      div[data-testid="stFileUploader"] label,

      div[data-testid="stWidgetLabel"] label,

      div[data-testid="stWidgetLabel"] p {{

        color: var(--pp-text) !important;

        font-weight: 700 !important;

        font-size: 13.5px !important;

        letter-spacing: 0.02em !important;

      }}

      div[data-testid="stRadio"] div[role="radiogroup"] label p {{

        color: var(--pp-text) !important;

        font-weight: 600 !important;

      }}

      /* Selectbox & React Aria ComboBox (Streamlit 1.66+) */

      div[data-testid="stSelectbox"] div[data-rac][role="group"],

      div[data-testid="stSelectbox"] div[role="combobox"],

      div[data-testid="stSelectbox"] input,

      div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,

      div[data-baseweb="select"] > div {{

        background-color: var(--pp-surface) !important;

        border-color: var(--pp-border) !important;

        color: var(--pp-text) !important;

      }}

      div[data-testid="stSelectbox"] svg {{

        fill: var(--pp-muted) !important;

        color: var(--pp-muted) !important;

      }}

      div[data-testid="stSelectbox"] input::placeholder {{

        color: var(--pp-muted) !important;

      }}

      div[data-testid="stSelectbox"] div[role="listbox"],

      div[data-baseweb="popover"] ul {{

        background-color: var(--pp-surface) !important;

        border: 1px solid var(--pp-border) !important;

        color: var(--pp-text) !important;

      }}

      div[data-testid="stSelectbox"] div[role="option"],

      div[data-baseweb="popover"] li {{

        background-color: var(--pp-surface) !important;

        color: var(--pp-text) !important;

      }}

      div[data-testid="stSelectbox"] div[role="option"]:hover,

      div[data-testid="stSelectbox"] div[role="option"][aria-selected="true"],

      div[data-baseweb="popover"] li:hover {{

        background-color: var(--pp-surface-alt) !important;

        color: var(--pp-text) !important;

      }}

      div[data-baseweb="select"] * {{

        color: var(--pp-text) !important;

      }}

      /* Segmented Control & stButtonGroup (Streamlit 1.66+) */

      div[data-testid="stButtonGroup"] div[role="radiogroup"],

      div[data-testid="stSegmentedControl"] div[role="radiogroup"] {{

        background-color: var(--pp-surface-alt) !important;

        border: 1px solid var(--pp-border) !important;

        border-radius: 8px !important;

        padding: 3px !important;

        gap: 4px !important;

      }}

      div[data-testid="stButtonGroup"] button[data-variant="segmented_control"],

      div[data-testid="stSegmentedControl"] button {{

        font-weight: 700 !important;

        font-size: 12.5px !important;

        border-radius: 6px !important;

        border: none !important;

        transition: all 0.15s ease !important;

      }}

      div[data-testid="stButtonGroup"] button[data-variant="segmented_control"]:not([aria-checked="true"]),

      div[data-testid="stSegmentedControl"] button:not([aria-checked="true"]) {{

        background-color: transparent !important;

        color: var(--pp-muted) !important;

      }}

      div[data-testid="stButtonGroup"] button[data-variant="segmented_control"]:not([aria-checked="true"]):hover,

      div[data-testid="stSegmentedControl"] button:not([aria-checked="true"]):hover {{

        background-color: rgba(128, 128, 128, 0.14) !important;

        color: var(--pp-text) !important;

      }}

      div[data-testid="stButtonGroup"] button[data-variant="segmented_control"][aria-checked="true"],

      div[data-testid="stSegmentedControl"] button[aria-checked="true"] {{

        background-color: var(--pp-surface) !important;

        color: var(--pp-text) !important;

        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15) !important;

      }}

      button[kind="primary"],

      div[data-testid="stBaseButton-primary"] button {{

        background: linear-gradient(135deg, #C85A17 0%, #A2420F 100%) !important;

        color: #F8FAFC !important;

        border: 1px solid #78350F !important;

        font-family: 'Space Grotesk', sans-serif !important;

        font-size: 14.5px !important;

        font-weight: 600 !important;

        letter-spacing: 0.01em !important;

        text-transform: none !important;

        box-shadow: 0 2px 6px rgba(200, 90, 23, 0.25) !important;

        border-radius: 6px !important;

        transition: background 0.15s ease, border-color 0.15s ease !important;

      }}

      button[kind="primary"]:hover,

      div[data-testid="stBaseButton-primary"] button:hover {{

        background: linear-gradient(135deg, #DE661E 0%, #B44A12 100%) !important;

        box-shadow: 0 3px 10px rgba(200, 90, 23, 0.35) !important;

      }}

      button[kind="secondary"],

      div[data-testid="stBaseButton-secondary"] button {{

        background-color: var(--pp-surface) !important;

        border: 1px solid var(--pp-border) !important;

        color: var(--pp-text) !important;

        border-radius: 6px !important;

        font-weight: 600 !important;

      }}

      .pp-editorial-card {{

        background: var(--pp-surface);

        border: 1px solid var(--pp-border);

        border-radius: 8px;

        padding: 14px 18px;

      }}

      .pp-kicker {{

        font-family: 'JetBrains Mono', monospace;

        font-size: 10px;

        font-weight: 600;

        color: var(--pp-muted);

        text-transform: uppercase;

        letter-spacing: 0.04em;

        margin-bottom: 4px;

      }}

      .pp-text-primary {{

        color: var(--pp-text) !important;

      }}

      div.stButton > button {{

        border-radius: 6px;

        font-weight: 700;

        letter-spacing: 0.02em;

        transition: border-color 0.14s ease, transform 0.12s ease;

      }}

      @media (forced-colors: active) {{

        div[data-testid="stVerticalBlockBorderWrapper"] {{

          border: 1px solid CanvasText !important;

        }}

      }}

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
            "K": "#111111",
            "G": "#FDB827",
            "S": "#FCD5B5",
            "D": "#E5B995",
            "H": "#4E342E",
            "E": "#231510",
            "W": "#FFFFFF",
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
            "K": "#111111",
            "O": "#DF4601",
            "Y": "#E5C158",
            "A": "#C69B34",
            "S": "#FDE0D0",
            "D": "#E8BEAA",
            "U": "#38BDF8",
            "W": "#FFFFFF",
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
            "K": "#18181B",
            "O": "#FD5A1E",
            "H": "#382415",
            "S": "#F5CEAB",
            "D": "#DCAB83",
            "E": "#26180F",
            "W": "#FDFBF7",
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
            "N": "#13274F",
            "R": "#CE1141",
            "K": "#1F1F1F",
            "S": "#EED8C6",
            "D": "#D1B49D",
            "T": "#4B5563",
            "E": "#1C1917",
            "W": "#FFFFFF",
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
            "B": "#005A9C",
            "W": "#FFFFFF",
            "K": "#0F172A",
            "S": "#F8D7B0",
            "D": "#DEB386",
            "E": "#0F172A",
            "R": "#EF4444",
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

    meta = MLB_PITCHERS.get(pitcher_name, MLB_PITCHERS["Paul Skenes"])

    team_hex = meta.get("team_primary_hex", "#F59E0B")

    svg = f"""

    <svg viewBox="0 0 24 24" width="100%" height="135" shape-rendering="crispEdges" style="background:#12141A;border-radius:6px;border:1px solid {team_hex};display:block;margin:0 auto 6px auto;">

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


def render_pitcher_photo_html(
    pitcher_name: str, height_px: int = 138, theme_mode: str | None = None
) -> str:
    """

    Render official MLB headshot framed in Topps Chrome / Statcast foil card style.

    Features tactical laser-cut chamfered corners, stencil jersey watermark, and Stalker radar top velo tag.

    Adapted for both Classic Diamond (Dark Onyx) and Light Editorial (Platinum Silver).

    """

    src = get_pitcher_headshot_src(pitcher_name)

    meta = MLB_PITCHERS.get(pitcher_name, MLB_PITCHERS["Paul Skenes"])

    team_hex = meta.get("team_primary_hex", "#F59E0B")

    team_abbr = meta.get("team_abbr", "MLB")

    jersey = meta.get("jersey_number", "#00")

    top_spd = meta.get("top_speed", "100+ MPH")

    img_max_h = max(64, int(height_px) - 12)

    resolved = resolve_ui_theme_mode(theme_mode)

    is_light = resolved == "light"

    if is_light:
        card_bg = "linear-gradient(145deg, #FFFFFF 0%, #F1F5F9 100%)"

        border_c = "#CBD5E1"

        badge_bg = "#0F172A"

        badge_color = "#38BDF8"

        wm_color = "#0F172A"

        wm_opac = "0.08"

        shadow_style = "box-shadow: 0 4px 14px rgba(15, 23, 42, 0.12);"

    else:
        card_bg = "linear-gradient(145deg, #161922 0%, #0A0B0E 100%)"

        border_c = "#2C3446"

        badge_bg = "#05070B"

        badge_color = "#F8FAFC"

        wm_color = team_hex

        wm_opac = "0.12"

        shadow_style = "box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);"

    clip_style = "clip-path: polygon(0 8px, 8px 0, calc(100% - 8px) 0, 100% 8px, 100% calc(100% - 8px), calc(100% - 8px) 100%, 8px 100%, 0 calc(100% - 8px)); border-radius: 4px;"

    decorations = f"""

    <span style="position:absolute;bottom:2px;left:8px;font-family:'Barlow Condensed',sans-serif;font-size:42px;font-weight:900;color:{wm_color};opacity:{wm_opac};line-height:1;pointer-events:none;user-select:none;">{jersey}</span>

    <span style="position:absolute;top:5px;left:7px;background:{badge_bg};border:1px solid {border_c};border-left:2px solid {team_hex};border-radius:3px;padding:1px 5px;font-family:'JetBrains Mono',monospace;font-size:8.5px;font-weight:800;color:{badge_color};">{top_spd}</span>

    """

    raw_html = f"""

    <div style="background:{card_bg};{clip_style}border:1px solid {border_c};border-top:3px solid {team_hex};padding:4px;margin:2px auto 6px auto;height:{int(height_px)}px;box-sizing:border-box;position:relative;display:flex;align-items:flex-end;justify-content:center;overflow:hidden;{shadow_style}">

      <span style="position:absolute;top:5px;right:7px;font-family:'JetBrains Mono',monospace;font-size:9.5px;font-weight:800;color:{team_hex};letter-spacing:0.04em;">{team_abbr} {jersey}</span>

      {decorations}

      <img src="{src}" alt="{pitcher_name}" style="max-height:{img_max_h}px;max-width:100%;object-fit:contain;filter:drop-shadow(0 4px 8px rgba(0,0,0,0.30));display:block;margin:0 auto;position:relative;z-index:2;" />

    </div>

    """

    return _compact_html(raw_html)


def render_pitch_flight_arena_svg(
    pitch_category: str = "Fastball",
    pitch_name: str = "4-Seam Fastball",
    speed_mph: str = "100 mph",
    target_zone: int = 5,
    p_throws: str = "R",
    ai_category: str | None = None,
    result_badge: str = "PRONTO PARA O ARREMESSO",
    is_success: bool = True,
) -> str:
    """

    Renderiza a Arena Broadcast 3D em perspectiva (visão do rebatedor/catcher para o montinho)

    com animação vetorial contínua da trajetória da bola rumo à Zona de Strike.

    """

    cat_style = PITCH_CATEGORY_STYLES.get(
        pitch_category, PITCH_CATEGORY_STYLES["Fastball"]
    )

    primary_color = cat_style["primary"]

    zone_coords: dict[int, tuple[int, int]] = {
        1: (278, 162),
        2: (320, 162),
        3: (362, 162),
        4: (278, 198),
        5: (320, 198),
        6: (362, 198),
        7: (278, 234),
        8: (320, 234),
        9: (362, 234),
        11: (238, 138),
        12: (402, 138),
        13: (238, 260),
        14: (402, 260),
    }

    tx, ty = zone_coords.get(target_zone, (320, 198))

    rx = 298 if p_throws == "R" else 342

    ry = 86

    hand_sign = 1 if p_throws == "R" else -1

    tunnel_x = int(rx + (tx - rx) * 0.42 - hand_sign * 8)

    tunnel_y = int(ry + (ty - ry) * 0.35 - 14)

    if pitch_category == "Breaking Ball":
        c2x = int(tx - hand_sign * 54)

        c2y = int(ty - 46)

        flight_dur = "0.62s"

        rpm_label = "2,880 RPM • SWEEP"

        traj_label = "CURVA ACENTUADA [BRK]"

    elif pitch_category == "Offspeed":
        c2x = int(tx + hand_sign * 26)

        c2y = int(ty - 38)

        flight_dur = "0.56s"

        rpm_label = "1,650 RPM • DROP TARDIO"

        traj_label = "QUEDA TARDIA [OFF]"

    else:
        c2x = int(rx + (tx - rx) * 0.72)

        c2y = int(ry + (ty - ry) * 0.68 - 10)

        flight_dur = "0.42s"

        rpm_label = "2,450 RPM • HIGH RIDE"

        traj_label = "LINHA EXPLOSIVA [FF]"

    main_path = f"M {rx} {ry} C {tunnel_x} {tunnel_y}, {c2x} {c2y}, {tx} {ty}"

    ghost_svg = ""

    if ai_category and ai_category != pitch_category:
        ai_style = PITCH_CATEGORY_STYLES.get(
            ai_category, PITCH_CATEGORY_STYLES["Fastball"]
        )

        ai_color = ai_style["primary"]

        if ai_category == "Breaking Ball":
            gx, gy = tx + hand_sign * 34, ty + 24

            gc2x, gc2y = tx - hand_sign * 40, ty - 35

        elif ai_category == "Offspeed":
            gx, gy = tx - hand_sign * 20, ty + 28

            gc2x, gc2y = tx + hand_sign * 18, ty - 30

        else:
            gx, gy = 320, 172

            gc2x, gc2y = int(rx + (gx - rx) * 0.72), int(ry + (gy - ry) * 0.68 - 10)

        ghost_path = f"M {rx} {ry} C {tunnel_x} {tunnel_y}, {gc2x} {gc2y}, {gx} {gy}"

        ghost_svg = (
            f'<path d="{ghost_path}" fill="none" stroke="{ai_color}" stroke-width="2.2" '
            f'stroke-dasharray="5,5" stroke-opacity="0.65" />'
            f'<circle cx="{gx}" cy="{gy}" r="7" fill="none" stroke="{ai_color}" '
            f'stroke-width="1.8" stroke-dasharray="3,2" stroke-opacity="0.75" />'
            f'<text x="{gx}" y="{gy - 10}" text-anchor="middle" fill="{ai_color}" '
            f'font-size="8.5" font-weight="700" font-family="JetBrains Mono, monospace">IA ESPERAVA: {ai_category.upper()}</text>'
        )

    grid_cells = []

    for idx, z_id in enumerate(range(1, 10)):
        r_idx = idx // 3

        c_idx = idx % 3

        gx_cell = 257 + c_idx * 42

        gy_cell = 144 + r_idx * 36

        is_sel = z_id == target_zone

        cell_fill = primary_color if is_sel else "#12141A"

        cell_opac = "0.45" if is_sel else "0.30"

        cell_stroke = "#FFFFFF" if is_sel else "#38BDF8"

        grid_cells.append(
            f'<rect x="{gx_cell}" y="{gy_cell}" width="42" height="36" '
            f'fill="{cell_fill}" fill-opacity="{cell_opac}" stroke="{cell_stroke}" stroke-width="1.1" />'
        )

    verdict_color = "#38BDF8" if is_success else "#EF4444"

    raw_svg = f"""

    <div style="margin:6px 0 10px 0;border-radius:8px;overflow:hidden;border:1px solid #222631;border-top:3px solid {primary_color};">

      <svg viewBox="0 0 640 290" width="100%" style="display:block;background:#0A0B0E;">

        <defs>

          <linearGradient id="tunnelBg" x1="0" y1="0" x2="0" y2="1">

            <stop offset="0%" stop-color="#0A0B0E" />

            <stop offset="55%" stop-color="#12141A" />

            <stop offset="100%" stop-color="#0D1117" />

          </linearGradient>

          <radialGradient id="moundDirt" cx="50%" cy="50%" r="50%">

            <stop offset="0%" stop-color="#7C4A2D" />

            <stop offset="100%" stop-color="#4A2B18" />

          </radialGradient>

          <filter id="pitchGlow" x="-30%" y="-30%" width="160%" height="160%">

            <feGaussianBlur stdDeviation="3" result="blur" />

            <feMerge>

              <feMergeNode in="blur" />

              <feMergeNode in="SourceGraphic" />

            </feMerge>

          </filter>

        </defs>

        <rect x="0" y="0" width="640" height="290" fill="url(#tunnelBg)" />

        <polygon points="275,84 365,84 445,270 195,270" fill="#12141A" fill-opacity="0.65" stroke="#222631" stroke-width="1.2" />

        <line x1="275" y1="84" x2="257" y2="144" stroke="#334155" stroke-width="1" stroke-dasharray="3,3" />

        <line x1="365" y1="84" x2="383" y2="144" stroke="#334155" stroke-width="1" stroke-dasharray="3,3" />

        <line x1="275" y1="84" x2="257" y2="252" stroke="#334155" stroke-width="1" stroke-dasharray="3,3" />

        <line x1="365" y1="84" x2="383" y2="252" stroke="#334155" stroke-width="1" stroke-dasharray="3,3" />

        <ellipse cx="320" cy="92" rx="42" ry="10" fill="url(#moundDirt)" stroke="#9A5B37" stroke-width="1.2" />

        <rect x="313" y="89" width="14" height="3" fill="#F8FAFC" />

        <circle cx="{rx}" cy="{ry}" r="6" fill="{primary_color}" fill-opacity="0.3" stroke="#F59E0B" stroke-width="1.8" />

        <circle cx="{rx}" cy="{ry}" r="2.5" fill="#FFFFFF" />

        <text x="{rx}" y="{ry - 10}" text-anchor="middle" fill="#F59E0B" font-size="8.5" font-weight="700" font-family="JetBrains Mono, monospace">PONTO DE SOLTURA ({p_throws}HP)</text>

        <ellipse cx="320" cy="276" rx="115" ry="14" fill="#5C3620" fill-opacity="0.75" />

        <polygon points="295,266 345,266 336,275 320,282 304,275" fill="#F8FAFC" stroke="#94A3B8" stroke-width="1.5" />

        {"".join(grid_cells)}

        <rect x="257" y="144" width="126" height="108" fill="none" stroke="#38BDF8" stroke-width="2" />

        {ghost_svg}

        <circle cx="{tunnel_x}" cy="{tunnel_y}" r="3.5" fill="#F59E0B" fill-opacity="0.95" />

        <text x="{tunnel_x + (14 if p_throws == "L" else -14)}" y="{tunnel_y - 6}" text-anchor="{"start" if p_throws == "L" else "end"}" fill="#F59E0B" font-size="8" font-family="JetBrains Mono, monospace">PONTO DE QUEBRA (DISFARCE)</text>

        <path id="mainPitchArc" d="{main_path}" fill="none" stroke="{primary_color}" stroke-width="4" stroke-linecap="round" filter="url(#pitchGlow)" stroke-dasharray="320" stroke-dashoffset="320">

          <animate attributeName="stroke-dashoffset" from="320" to="0" dur="{flight_dur}" fill="freeze" repeatCount="indefinite" />

        </path>

        <path d="{main_path}" fill="none" stroke="#FFFFFF" stroke-width="1.3" stroke-opacity="0.5" stroke-dasharray="4,4" />

        <circle r="7.5" fill="#FFFFFF" stroke="{primary_color}" stroke-width="2.4" filter="url(#pitchGlow)">

          <animateMotion dur="{flight_dur}" repeatCount="indefinite" path="{main_path}" />

          <animate attributeName="r" values="3.5;6.5;10.0" dur="{flight_dur}" repeatCount="indefinite" />

        </circle>

        <circle cx="{tx}" cy="{ty}" r="12" fill="none" stroke="#F59E0B" stroke-width="2" stroke-dasharray="3,2" />

        <circle cx="{tx}" cy="{ty}" r="4.5" fill="{primary_color}" />

        <rect x="12" y="10" width="202" height="62" rx="6" fill="#12141A" fill-opacity="0.95" stroke="#222631" stroke-width="1.2" />

        <line x1="12" y1="12" x2="214" y2="12" stroke="{primary_color}" stroke-width="2.5" />

        <text x="22" y="27" fill="#94A3B8" font-size="9" font-weight="700" font-family="JetBrains Mono, monospace">STATCAST RADAR • {p_throws}HP</text>

        <text x="22" y="47" fill="#F1F5F9" font-size="15" font-weight="800" font-family="JetBrains Mono, monospace">{speed_mph.upper()} • {pitch_name[:15]}</text>

        <text x="22" y="63" fill="{primary_color}" font-size="9.5" font-weight="700" font-family="JetBrains Mono, monospace">{rpm_label}</text>

        <rect x="405" y="10" width="223" height="62" rx="6" fill="#12141A" fill-opacity="0.95" stroke="#222631" stroke-width="1.2" />

        <line x1="405" y1="12" x2="628" y2="12" stroke="{verdict_color}" stroke-width="2.5" />

        <text x="415" y="27" fill="#94A3B8" font-size="9" font-weight="700" font-family="JetBrains Mono, monospace">{traj_label}</text>

        <text x="415" y="47" fill="{verdict_color}" font-size="12" font-weight="800" font-family="JetBrains Mono, monospace">{result_badge[:24]}</text>

        <text x="415" y="63" fill="#CBD5E1" font-size="9" font-weight="700" font-family="JetBrains Mono, monospace">ZONA ALVO: #{target_zone} ({STATCAST_ZONES.get(target_zone, STATCAST_ZONES[5])["region"].upper()})</text>

      </svg>

    </div>

    """

    return _compact_html(raw_svg)


def render_baseball_seams_divider_svg(
    width_px: int = 600,
    theme_mode: str | None = None,
) -> str:
    """

    Render a precision horizontal baseball seam divider SVG.

    Features the iconic Rawlings MLB baseball dual-stitch pattern with red V-stitches

    (#DC2626) and a subtle guideline. Guaranteed CommonMark-safe single-line markup.

    """

    stitches: list[str] = []

    for x in range(30, 580, 24):
        if 280 <= x <= 320:
            continue

        stitches.append(f"M {x - 5} 7 L {x} 12 L {x - 5} 17")

    path_d = " ".join(stitches)

    resolved = resolve_ui_theme_mode(theme_mode)

    line_stroke = "#334155" if resolved != "light" else "#CBD5E1"

    raw_svg = f"""

    <div class="pp-seam-divider" style="display:flex;justify-content:center;align-items:center;margin:12px 0;width:100%;">

      <svg viewBox="0 0 600 24" width="100%" style="max-width:{int(width_px)}px;display:block;">

        <line x1="10" y1="12" x2="590" y2="12" stroke="{line_stroke}" stroke-width="1.2" stroke-dasharray="2,4" />

        <path d="{path_d}" fill="none" stroke="#DC2626" stroke-width="1.8" stroke-linecap="round" />

        <circle cx="300" cy="12" r="5" fill="#DC2626" stroke="#FEF2F2" stroke-width="1" />

      </svg>

    </div>

    """

    return _compact_html(raw_svg)


def render_spin_axis_clock_svg(
    spin_axis: str = "1:15",
    spin_rpm: int = 2480,
    pitch_name: str = "4-Seam Fastball",
    tilt_deg: float | None = None,
    category: str = "Fastball",
    theme_mode: str | None = None,
) -> str:
    """

    Render an authentic MLB Statcast 12-Hour Spin Axis Clock Gauge SVG.

    Visualizes pitch tilt orientation, active Magnus effect vector, and RPM telemetry.

    Guaranteed CommonMark-safe single-line markup.

    """

    clock_angle_deg = 37.5

    if tilt_deg is not None:
        clock_angle_deg = float(tilt_deg) % 360.0

    else:
        try:
            parts = spin_axis.split(":")

            h = int(parts[0]) % 12

            m = int(parts[1]) if len(parts) > 1 else 0

            clock_angle_deg = (h * 30.0 + m * 0.5) % 360.0

        except Exception:  # noqa: BLE001
            clock_angle_deg = 37.5

    cx, cy = 82, 88

    r_dial = 46

    math_rad = math.radians(clock_angle_deg - 90.0)

    x2 = round(cx + (r_dial - 8) * math.cos(math_rad), 1)

    y2 = round(cy + (r_dial - 8) * math.sin(math_rad), 1)

    cat_style = PITCH_CATEGORY_STYLES.get(category, PITCH_CATEGORY_STYLES["Fastball"])

    pointer_color = cat_style["primary"]

    if "Fastball" in category or "4-Seam" in pitch_name:
        magnus_pct = "98% ATIVO"

        magnus_color = "#38BDF8"

    elif "Breaking" in category or "Slider" in pitch_name or "Curve" in pitch_name:
        magnus_pct = "88% ATIVO"

        magnus_color = "#3B82F6"

    else:
        magnus_pct = "76% ATIVO"

        magnus_color = "#F59E0B"

    resolved = resolve_ui_theme_mode(theme_mode)

    if resolved == "light":
        card_bg = "#FFFFFF"

        border_c = "#C2CFDD"

        dial_fill = "#E6ECF2"

        dial_stroke = "#94A3B8"

        text_primary = "#0F172A"

        text_muted = "#475569"

        panel_bg = "#F1F5F9"

    elif resolved in ("diamond", "ballpark"):
        card_bg = "linear-gradient(135deg, #12141C 0%, #090A0E 100%)"

        border_c = "#262C3D"

        dial_fill = "#07080B"

        dial_stroke = "#3A4459"

        text_primary = "#F8FAFC"

        text_muted = "#94A3B8"

        panel_bg = "#171B26"

    else:
        card_bg = "linear-gradient(135deg, #12151C 0%, #0A0B0E 100%)"

        border_c = "#222631"

        dial_fill = "#07080B"

        dial_stroke = "#334155"

        text_primary = "#F1F5F9"

        text_muted = "#94A3B8"

        panel_bg = "#181B23"

    raw_svg = f"""

    <div style="display:flex;justify-content:center;align-items:center;margin:4px 0;width:100%;">

      <svg viewBox="0 0 290 176" width="100%" style="max-width:290px;background:{card_bg};border:1px solid {border_c};border-radius:8px;padding:8px;display:block;box-sizing:border-box;">

        <text x="145" y="19" text-anchor="middle" fill="{text_muted}" font-family="JetBrains Mono, monospace" font-size="8.8" font-weight="700" letter-spacing="0.04em">EIXO DE ROTAÇÃO (SPIN AXIS CLOCK)</text>

        <circle cx="{cx}" cy="{cy}" r="{r_dial}" fill="{dial_fill}" stroke="{dial_stroke}" stroke-width="1.4" />

        <circle cx="{cx}" cy="{cy}" r="{r_dial - 2}" fill="none" stroke="{dial_stroke}" stroke-width="0.8" stroke-dasharray="2,5" />

        <text x="{cx}" y="{cy - 34}" text-anchor="middle" fill="{text_muted}" font-size="7.5" font-family="JetBrains Mono, monospace" font-weight="700">12:00</text>

        <text x="{cx + 34}" y="{cy + 3}" text-anchor="middle" fill="{text_muted}" font-size="7.5" font-family="JetBrains Mono, monospace" font-weight="700">3:00</text>

        <text x="{cx}" y="{cy + 40}" text-anchor="middle" fill="{text_muted}" font-size="7.5" font-family="JetBrains Mono, monospace" font-weight="700">6:00</text>

        <text x="{cx - 34}" y="{cy + 3}" text-anchor="middle" fill="{text_muted}" font-size="7.5" font-family="JetBrains Mono, monospace" font-weight="700">9:00</text>

        <line x1="{cx}" y1="{cy}" x2="{x2}" y2="{y2}" stroke="{pointer_color}" stroke-width="2.8" stroke-linecap="round" />

        <circle cx="{x2}" cy="{y2}" r="3.8" fill="{pointer_color}" stroke="#FFFFFF" stroke-width="1" />

        <circle cx="{cx}" cy="{cy}" r="3.5" fill="#FFFFFF" />

        <rect x="156" y="32" width="122" height="38" rx="4" fill="{panel_bg}" stroke="{border_c}" stroke-width="1" />

        <text x="164" y="46" fill="{text_muted}" font-size="7.8" font-family="JetBrains Mono, monospace" font-weight="700">EIXO REAL</text>

        <text x="164" y="62" fill="{pointer_color}" font-size="12.5" font-weight="800" font-family="JetBrains Mono, monospace">{spin_axis} TILT</text>

        <rect x="156" y="74" width="122" height="38" rx="4" fill="{panel_bg}" stroke="{border_c}" stroke-width="1" />

        <text x="164" y="88" fill="{text_muted}" font-size="7.8" font-family="JetBrains Mono, monospace" font-weight="700">ROTAÇÃO</text>

        <text x="164" y="104" fill="{text_primary}" font-size="12.5" font-weight="800" font-family="JetBrains Mono, monospace">{spin_rpm:,} RPM</text>

        <rect x="156" y="116" width="122" height="38" rx="4" fill="{panel_bg}" stroke="{border_c}" stroke-width="1" />

        <text x="164" y="130" fill="{text_muted}" font-size="7.8" font-family="JetBrains Mono, monospace" font-weight="700">EFEITO MAGNUS</text>

        <text x="164" y="146" fill="{magnus_color}" font-size="11" font-weight="800" font-family="JetBrains Mono, monospace">{magnus_pct}</text>

      </svg>

    </div>

    """

    return _compact_html(raw_svg)


def render_dugout_clipboard_frame_html(
    title: str,
    content_html: str,
    stadium: str = "PNC PARK • PITTSBURGH",
    theme_mode: str | None = None,
) -> str:
    """

    Wrap content in an authentic MLB Dugout Tactical Clipboard frame.

    Features a top brushed-steel clip with brass rivets, stadium header strip,

    and acrylic dugout clipboard texture. Guaranteed CommonMark-safe single-line markup.

    """

    resolved = resolve_ui_theme_mode(theme_mode)

    if resolved == "light":
        frame_bg = "#FFFFFF"

        border_c = "#CBD5E1"

        header_border = "#E2E8F0"

        title_color = "#C85A17"

        stadium_color = "#64748B"

        clip_bg = "linear-gradient(180deg, #94A3B8 0%, #64748B 100%)"

        clip_border = "#475569"

        text_c = "#0F172A"

    elif resolved in ("diamond", "ballpark"):
        frame_bg = "linear-gradient(180deg, #13161F 0%, #090A0E 100%)"

        border_c = "#2E364A"

        header_border = "#1E2433"

        title_color = "#D97706"

        stadium_color = "#94A3B8"

        clip_bg = "linear-gradient(180deg, #64748B 0%, #334155 100%)"

        clip_border = "#94A3B8"

        text_c = "#F8FAFC"

    else:
        frame_bg = "#141720"

        border_c = "#2C3242"

        header_border = "#222631"

        title_color = "#C85A17"

        stadium_color = "#94A3B8"

        clip_bg = "linear-gradient(180deg, #64748B 0%, #334155 100%)"

        clip_border = "#94A3B8"

        text_c = "#E2E8F0"

    raw_html = f"""

    <div class="pp-dugout-clipboard" style="background:{frame_bg};border:1.5px solid {border_c};border-radius:10px;padding:26px 18px 16px 18px;position:relative;box-shadow:0 8px 24px rgba(0,0,0,0.35);margin:18px 0 12px 0;">

      <div style="position:absolute;top:-13px;left:50%;transform:translateX(-50%);background:{clip_bg};border:1px solid {clip_border};border-radius:4px;padding:3px 28px;display:flex;align-items:center;gap:12px;box-shadow:0 3px 8px rgba(0,0,0,0.45);z-index:2;">

        <div style="width:6px;height:6px;border-radius:50%;background:#D97706;border:1px solid #78350F;"></div>

        <span style="font-family:'JetBrains Mono',monospace;font-size:9.5px;font-weight:800;color:#F8FAFC;letter-spacing:0.08em;white-space:nowrap;">MLB DUGOUT TACTICAL CARD</span>

        <div style="width:6px;height:6px;border-radius:50%;background:#D97706;border:1px solid #78350F;"></div>

      </div>

      <div style="display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid {header_border};padding-bottom:8px;margin-bottom:14px;flex-wrap:wrap;gap:6px;">

        <span style="color:{title_color};font-family:'JetBrains Mono',monospace;font-size:11.5px;font-weight:800;letter-spacing:0.04em;">{title}</span>

        <span style="color:{stadium_color};font-size:10px;font-family:'JetBrains Mono',monospace;font-weight:600;">{stadium}</span>

      </div>

      <div style="color:{text_c};font-size:13px;line-height:1.55;">

        {content_html}

      </div>

    </div>

    """

    return _compact_html(raw_html)


def render_broadcast_scorebug_header_html(
    pitcher_name: str = "Paul Skenes",
    balls: int = 0,
    strikes: int = 0,
    outs: int = 0,
    speed_mph: str = "101.8 MPH",
    on_1b: int = 0,
    on_2b: int = 0,
    on_3b: int = 0,
    inning: str = "▲ 1st",
    matchup_title: str = "MLB STATCAST BROADCAST",
    theme_mode: str | None = None,
) -> str:
    """

    Render official MLB broadcast scorebug header (Apple TV+ / MLB Network style).

    Displays Balls/Strikes/Outs electronic LED dots, Stalker radar gun speed box,

    mini-diamond base runners, and pitcher identification.

    Guaranteed CommonMark-safe single-line markup.

    """

    meta = MLB_PITCHERS.get(pitcher_name, MLB_PITCHERS["Paul Skenes"])

    team_hex = meta.get("team_primary_hex", "#F59E0B")

    jersey = meta.get("jersey_number", "#00")

    p_throws = meta.get("p_throws", "R")

    ball_dots = []

    for b_idx in range(1, 4):
        f = "#F59E0B" if balls >= b_idx else "rgba(148,163,184,0.25)"

        ball_dots.append(
            f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:{f};margin:0 1px;"></span>'
        )

    b_html = "".join(ball_dots)

    strike_dots = []

    for s_idx in range(1, 3):
        f = "#EF4444" if strikes >= s_idx else "rgba(148,163,184,0.25)"

        strike_dots.append(
            f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:{f};margin:0 1px;"></span>'
        )

    s_html = "".join(strike_dots)

    out_dots = []

    for o_idx in range(1, 3):
        f = "#DC2626" if outs >= o_idx else "rgba(148,163,184,0.25)"

        out_dots.append(
            f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:{f};margin:0 1px;"></span>'
        )

    o_html = "".join(out_dots)

    c_1b = "#F59E0B" if int(on_1b) > 0 else "#334155"

    c_2b = "#F59E0B" if int(on_2b) > 0 else "#334155"

    c_3b = "#F59E0B" if int(on_3b) > 0 else "#334155"

    resolved = resolve_ui_theme_mode(theme_mode)

    if resolved == "light":
        wrap_bg = "linear-gradient(90deg, #FFFFFF 0%, #F8FAFC 100%)"

        border_c = "#CBD5E1"

        inner_bg = "#F1F5F9"

        text_primary = "#0F172A"

        text_muted = "#475569"

        radar_bg = "#0F172A"

        radar_color = "#38BDF8"

        runner_stroke = "#94A3B8"

        box_shadow = "box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);"

    else:
        wrap_bg = "linear-gradient(90deg, #12141C 0%, #0A0C12 100%)"

        border_c = "#262C3D"

        inner_bg = "#07080B"

        text_primary = "#F8FAFC"

        text_muted = "#94A3B8"

        radar_bg = "#05070B"

        radar_color = "#F59E0B"

        runner_stroke = "#475569"

        box_shadow = "box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);"

    raw_html = f"""

    <div class="pp-scorebug-banner" style="background:{wrap_bg};border:1px solid {border_c};border-radius:8px;padding:8px 14px;margin-bottom:14px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;{box_shadow}">

      <div style="display:flex;align-items:center;gap:10px;">

        <div style="background:{inner_bg};border:1px solid {border_c};border-left:3px solid {team_hex};border-radius:4px;padding:3px 9px;">

          <div style="font-family:'JetBrains Mono',monospace;font-size:8px;font-weight:800;color:{text_muted};letter-spacing:0.06em;">{matchup_title}</div>

          <div style="font-family:'Barlow Condensed',sans-serif;font-size:15px;font-weight:800;color:{text_primary};letter-spacing:0.02em;">{pitcher_name.upper()} <span style="color:{team_hex};font-size:12px;font-weight:700;">{jersey} • {p_throws}HP</span></div>

        </div>

      </div>

      <div style="background:{radar_bg};border:1px solid {border_c};border-radius:4px;padding:3px 12px;text-align:center;">

        <div style="font-family:'JetBrains Mono',monospace;font-size:7.5px;font-weight:800;color:{text_muted};letter-spacing:0.08em;">RADAR GUN</div>

        <div style="font-family:'JetBrains Mono',monospace;font-size:15px;font-weight:900;color:{radar_color};letter-spacing:0.04em;">{speed_mph}</div>

      </div>

      <div style="display:flex;align-items:center;gap:12px;">

        <div style="background:{inner_bg};border:1px solid {border_c};border-radius:4px;padding:4px 9px;display:flex;gap:8px;align-items:center;">

          <div style="font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:800;display:flex;align-items:center;gap:3px;"><span style="color:{text_muted};">B</span> {b_html}</div>

          <div style="font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:800;display:flex;align-items:center;gap:3px;"><span style="color:{text_muted};">S</span> {s_html}</div>

          <div style="font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:800;display:flex;align-items:center;gap:3px;"><span style="color:{text_muted};">O</span> {o_html}</div>

        </div>

        <svg viewBox="0 0 28 28" width="26" height="26" style="display:block;">

          <polygon points="14,3 25,14 14,25 3,14" fill="none" stroke="{runner_stroke}" stroke-width="1.2" />

          <g transform="translate(14, 3.5) rotate(45)"><rect x="-2.5" y="-2.5" width="5" height="5" fill="{c_2b}" /></g>

          <g transform="translate(3, 14) rotate(45)"><rect x="-2.5" y="-2.5" width="5" height="5" fill="{c_3b}" /></g>

          <g transform="translate(25, 14) rotate(45)"><rect x="-2.5" y="-2.5" width="5" height="5" fill="{c_1b}" /></g>

        </svg>

        <span style="font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:800;color:{text_primary};letter-spacing:0.04em;">{inning}</span>

      </div>

    </div>

    """

    return _compact_html(raw_html)


def render_topps_chrome_card_html(
    pitcher_name: str,
    theme_mode: str | None = None,
) -> str:
    """

    Render collectible Topps Chrome athlete card with radar velo tag,

    franchise styling, stencil watermark, and pitch repertoire pills.

    Guaranteed CommonMark-safe single-line markup.

    """

    meta = MLB_PITCHERS.get(pitcher_name, MLB_PITCHERS["Paul Skenes"])

    team_hex = meta.get("team_primary_hex", "#F59E0B")

    team_full = meta.get("team_full", "Major League Baseball")

    jersey = meta.get("jersey_number", "#00")

    p_throws = meta.get("p_throws", "R")

    top_spd = meta.get("top_speed", "100+ MPH")

    role = meta.get("role", "Ace Pitcher")

    arsenal = meta.get("arsenal", "4-Seam (FF) • Slider (SL)")

    src = get_pitcher_headshot_src(pitcher_name)

    resolved = resolve_ui_theme_mode(theme_mode)

    if resolved == "light":
        card_bg = "linear-gradient(145deg, #FFFFFF 0%, #F8FAFC 45%, #EDF2F7 100%)"

        border_c = "#CBD5E1"

        inner_box = "#F1F5F9"

        text_primary = "#0F172A"

        text_muted = "#475569"

        radar_bg = "#0F172A"

        radar_color = "#38BDF8"

        watermark_color = "#0F172A"

        watermark_opac = "0.06"

        shadow_style = "box-shadow: 0 6px 18px rgba(15, 23, 42, 0.12);"

    else:
        card_bg = "linear-gradient(145deg, #161922 0%, #0A0B0E 100%)"

        border_c = "#2C3446"

        inner_box = "#07080B"

        text_primary = "#F8FAFC"

        text_muted = "#94A3B8"

        radar_bg = "#05070B"

        radar_color = "#F59E0B"

        watermark_color = team_hex

        watermark_opac = "0.10"

        shadow_style = "box-shadow: 0 6px 18px rgba(0, 0, 0, 0.45);"

    clip_poly = "clip-path: polygon(0 8px, 8px 0, calc(100% - 8px) 0, 100% 8px, 100% calc(100% - 8px), calc(100% - 8px) 100%, 8px 100%, 0 calc(100% - 8px)); border-radius: 4px;"

    raw_html = f"""

    <div class="pp-topps-card" style="background:{card_bg};{clip_poly}border:1.5px solid {border_c};border-top:3px solid {team_hex};padding:12px;position:relative;{shadow_style}overflow:hidden;box-sizing:border-box;">

      <span style="position:absolute;bottom:8px;right:10px;font-family:'Barlow Condensed',sans-serif;font-size:62px;font-weight:900;color:{watermark_color};opacity:{watermark_opac};line-height:1;pointer-events:none;user-select:none;">{jersey}</span>

      <div style="background:{inner_box};border:1px solid {border_c};border-radius:4px;padding:3px 8px;display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">

        <span style="font-family:'JetBrains Mono',monospace;font-size:8px;font-weight:800;color:{text_muted};letter-spacing:0.06em;">TOPPS CHROME • STATCAST</span>

        <span style="background:{radar_bg};color:{radar_color};padding:1px 6px;border-radius:3px;font-family:'JetBrains Mono',monospace;font-size:10.5px;font-weight:900;letter-spacing:0.04em;">{top_spd}</span>

      </div>

      <div style="display:flex;gap:10px;align-items:center;margin-bottom:8px;">

        <img src="{src}" alt="{pitcher_name}" style="width:58px;height:58px;object-fit:cover;border-radius:50%;border:2px solid {team_hex};background:{inner_box};flex-shrink:0;" />

        <div>

          <div style="font-family:'JetBrains Mono',monospace;font-size:8.5px;font-weight:800;color:{team_hex};letter-spacing:0.04em;">{team_full.upper()}</div>

          <div style="font-family:'Barlow Condensed',sans-serif;font-size:18px;font-weight:900;color:{text_primary};line-height:1.1;">{pitcher_name.upper()}</div>

          <div style="font-family:'JetBrains Mono',monospace;font-size:9.5px;color:{text_muted};font-weight:700;">{jersey} • {p_throws}HP • {role}</div>

        </div>

      </div>

      <div style="background:{inner_box};border:1px solid {border_c};border-left:2px solid {team_hex};border-radius:4px;padding:4px 8px;font-size:10px;color:{text_primary};line-height:1.4;">

        <strong style="color:{text_muted};font-size:8.5px;font-family:'JetBrains Mono',monospace;display:block;">ARSENAL:</strong>

        {arsenal}

      </div>

    </div>

    """

    return _compact_html(raw_html)


def render_broadcast_station_bar_html(theme_mode: str | None = None) -> str:
    """

    Render top station ticker / broadcast network header ribbon.

    Displays live transmission indicator, network branding, and active MLB franchise tags.

    Guaranteed CommonMark-safe single-line markup.

    """

    resolved = resolve_ui_theme_mode(theme_mode)

    is_light = resolved == "light"

    if is_light:
        bar_bg = "linear-gradient(90deg, #FFFFFF 0%, #F1F5F9 100%)"

        border_c = "#CBD5E1"

        text_primary = "#0F172A"

        text_muted = "#475569"

        chip_bg = "#FFFFFF"

        shadow_style = "box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);"

    else:
        bar_bg = "linear-gradient(90deg, #12141C 0%, #0A0B0E 100%)"

        border_c = "#262C3D"

        text_primary = "#F8FAFC"

        text_muted = "#94A3B8"

        chip_bg = "#07080B"

        shadow_style = "box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);"

    raw_html = f"""

    <div class="pp-station-bar" style="background:{bar_bg};border:1px solid {border_c};border-top:2.5px solid var(--pp-accent, #38BDF8);border-radius:6px;padding:6px 12px;margin-bottom:12px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;{shadow_style}">

      <div style="display:flex;align-items:center;gap:8px;">

        <span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#DC2626;box-shadow:0 0 6px rgba(220,38,38,0.7);"></span>

        <span style="font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:800;color:{text_primary};letter-spacing:0.06em;">MLB STATCAST AI NETWORK</span>

        <span style="font-family:'JetBrains Mono',monospace;font-size:9.5px;color:{text_muted};">• TACTICAL RADAR & BIOMECHANICS</span>

      </div>

      <div style="display:flex;align-items:center;gap:6px;font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:800;">

        <span style="background:{chip_bg};border:1px solid {border_c};border-left:2px solid #F59E0B;padding:2px 6px;border-radius:3px;color:{text_primary};">PIT #30</span>

        <span style="background:{chip_bg};border:1px solid {border_c};border-left:2px solid #38BDF8;padding:2px 6px;border-radius:3px;color:{text_primary};">BAL #39</span>

        <span style="background:{chip_bg};border:1px solid {border_c};border-left:2px solid #F97316;padding:2px 6px;border-radius:3px;color:{text_primary};">SF #62</span>

        <span style="background:{chip_bg};border:1px solid {border_c};border-left:2px solid #DC2626;padding:2px 6px;border-radius:3px;color:{text_primary};">ATL #51</span>

        <span style="background:{chip_bg};border:1px solid {border_c};border-left:2px solid #0284C7;padding:2px 6px;border-radius:3px;color:{text_primary};">LAD #18</span>

      </div>

    </div>

    """

    return _compact_html(raw_html)


def render_ballpark_audio_listener_html(enabled: bool = True) -> str:
    """
    Render a zero-asset Web Audio API listener synthesizing realistic ballpark audio effects.

    When enabled, primary button clicks produce an authentic, satisfying 102 mph leather catcher's
    mitt pocket "POP" (fast pitch impact and leather vibration).

    100% reversible via sidebar toggle. Pure client-side synthesis with 0 network latency.
    """
    if not enabled:
        return ""

    raw_html = """
    <div id="pp-ballpark-audio-host" style="display:none;" aria-hidden="true">
      <script>
      (function() {
        if (window.__ppAudioInitialized) return;
        window.__ppAudioInitialized = true;

        var audioCtx = null;
        function getAudioContext() {
          if (!audioCtx) {
            var AudioContextClass = window.AudioContext || window.webkitAudioContext;
            if (AudioContextClass) {
              audioCtx = new AudioContextClass();
            }
          }
          if (audioCtx && audioCtx.state === 'suspended') {
            audioCtx.resume();
          }
          return audioCtx;
        }

        function playMittPop() {
          try {
            var ctx = getAudioContext();
            if (!ctx) return;
            var now = ctx.currentTime;

            // 1. Kinetic thud of 102 mph baseball compressing the leather pocket
            var osc = ctx.createOscillator();
            var oscGain = ctx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(155, now);
            osc.frequency.exponentialRampToValueAtTime(38, now + 0.055);

            oscGain.gain.setValueAtTime(0.35, now);
            oscGain.gain.exponentialRampToValueAtTime(0.001, now + 0.065);

            osc.connect(oscGain);
            oscGain.connect(ctx.destination);
            osc.start(now);
            osc.stop(now + 0.07);

            // 2. High-frequency leather slap burst
            var bufferSize = Math.floor(ctx.sampleRate * 0.04);
            var noiseBuffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
            var output = noiseBuffer.getChannelData(0);
            for (var i = 0; i < bufferSize; i++) {
              output[i] = (Math.random() * 2 - 1) * Math.exp(-i / (bufferSize * 0.25));
            }

            var whiteNoise = ctx.createBufferSource();
            whiteNoise.buffer = noiseBuffer;

            var filter = ctx.createBiquadFilter();
            filter.type = 'bandpass';
            filter.frequency.setValueAtTime(1900, now);
            filter.Q.setValueAtTime(1.8, now);

            var noiseGain = ctx.createGain();
            noiseGain.gain.setValueAtTime(0.28, now);
            noiseGain.gain.exponentialRampToValueAtTime(0.001, now + 0.045);

            whiteNoise.connect(filter);
            filter.connect(noiseGain);
            noiseGain.connect(ctx.destination);

            whiteNoise.start(now);
            whiteNoise.stop(now + 0.05);
          } catch (e) {
            // Audio context failure gracefully swallowed
          }
        }

        function handleInteraction(e) {
          var target = e.target;
          if (!target) return;
          var btn = target.closest('button');
          if (btn) {
            playMittPop();
          }
        }

        document.addEventListener('click', handleInteraction, true);
        try {
          if (window.parent && window.parent.document && window.parent.document !== document) {
            window.parent.document.addEventListener('click', handleInteraction, true);
          }
        } catch (err) {}
      })();
      </script>
    </div>
    """
    return _compact_html(raw_html)


def render_scouting_grades_card_html(
    pitcher_name: str, theme_mode: str | None = None
) -> str:
    """
    Render bespoke MLB 20-80 Scouting Scale & Stuff+ Index Card for Scouting Report.

    Authentic scout grading (20: Descarte • 50: Média MLB • 80: Elite) combined with
    advanced Statcast pitch-level Stuff+ metrics.
    Zero green (Corinthians standard): uses Gold/Amber (#F59E0B), Floodlight Blue (#38BDF8/#0284C7),
    Infield Clay (#C85A17), and Slate (#94A3B8).
    CommonMark-safe single-line markup.
    """
    from pitch_predictor.scenarios import get_pitcher_profile

    resolved = resolve_ui_theme_mode(theme_mode)
    is_light = resolved == "light"

    profile = get_pitcher_profile(pitcher_name)
    grades = profile.get("scouting_grades", {})
    team_hex = profile.get("team_primary_hex", "#38BDF8")
    p_name = profile.get("name", pitcher_name)
    overall_stuff = grades.get("overall_stuff", 125)

    if is_light:
        card_bg = "#FFFFFF"
        border_c = "#CBD5E1"
        text_primary = "#0F172A"
        text_muted = "#475569"
        row_bg = "#F8FAFC"
        bar_track = "#E2E8F0"
        shadow = "box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);"
    else:
        card_bg = "linear-gradient(180deg, #131622 0%, #0C0E15 100%)"
        border_c = "#262C3D"
        text_primary = "#F8FAFC"
        text_muted = "#94A3B8"
        row_bg = "rgba(255, 255, 255, 0.025)"
        bar_track = "#1E2330"
        shadow = "box-shadow: 0 4px 16px rgba(0, 0, 0, 0.45);"

    def _grade_badge(grade_str: str, label_str: str) -> str:
        g = int(grade_str) if str(grade_str).isdigit() else 50
        if g >= 80:
            bg = "rgba(245, 158, 11, 0.16)"
            bd = "#F59E0B"
            tc = "#F59E0B"
        elif g >= 70:
            bg = "rgba(2, 132, 199, 0.16)" if is_light else "rgba(56, 189, 248, 0.16)"
            bd = "#0284C7" if is_light else "#38BDF8"
            tc = "#0284C7" if is_light else "#38BDF8"
        elif g >= 60:
            bg = "rgba(200, 90, 23, 0.16)"
            bd = "#C85A17"
            tc = "#C85A17"
        else:
            bg = "rgba(148, 163, 184, 0.15)"
            bd = "#94A3B8"
            tc = "#94A3B8"
        return f"<span style=\"background:{bg};border:1px solid {bd};color:{tc};font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:800;padding:2px 7px;border-radius:4px;letter-spacing:0.04em;\">{grade_str} • {label_str.upper()}</span>"

    def _stuff_bar(stuff_val: int) -> str:
        pct = max(10, min(100, int((stuff_val - 70) / (150 - 70) * 100)))
        fill_color = (
            "#F59E0B"
            if stuff_val >= 135
            else ("#0284C7" if is_light else "#38BDF8")
            if stuff_val >= 115
            else "#C85A17"
        )
        return f'<div style="background:{bar_track};width:100%;height:6px;border-radius:3px;overflow:hidden;position:relative;"><div style="background:{fill_color};width:{pct}%;height:100%;border-radius:3px;"></div></div>'

    rows_html = []
    for key in ("fastball", "secondary", "breaking", "command"):
        item = grades.get(key)
        if not item:
            continue
        badge = _grade_badge(item["grade"], item["label"])
        bar = _stuff_bar(item["stuff_plus"])
        whiff_txt = item.get("whiff_rate", "—")
        rows_html.append(
            f'<div style="background:{row_bg};border:1px solid {border_c};border-radius:6px;padding:9px 12px;margin-bottom:6px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">'
            f'<div style="min-width:180px;"><div style="font-family:\'Barlow Condensed\',sans-serif;font-size:14.5px;font-weight:800;color:{text_primary};letter-spacing:0.03em;">{item["name"]}</div>'
            f'<div style="margin-top:2px;">{badge}</div></div>'
            f'<div style="min-width:120px;flex:1;max-width:200px;margin:0 12px;"><div style="display:flex;justify-content:space-between;font-family:\'JetBrains Mono\',monospace;font-size:9.5px;font-weight:800;color:{text_muted};margin-bottom:3px;"><span>STUFF+</span><span style="color:{text_primary};">{item["stuff_plus"]}</span></div>{bar}</div>'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;font-weight:800;color:{text_primary};text-align:right;"><span style="font-size:9px;color:{text_muted};display:block;">WHIFF%</span>{whiff_txt}</div>'
            f"</div>"
        )

    all_rows = "".join(rows_html)

    raw_html = f"""
    <div style="background:{card_bg};border:1px solid {border_c};border-top:3px solid {team_hex};border-radius:8px;padding:16px;margin:12px 0;{shadow}">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:12px;border-bottom:1px solid {border_c};padding-bottom:10px;">
        <div>
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:18px;font-weight:800;color:{text_primary};letter-spacing:0.04em;text-transform:uppercase;">GRADUAÇÃO DE OLHEIRO DA MLB • ESCALA 20-80 &amp; ÍNDICE STUFF+</div>
          <div style="font-size:11.5px;color:{text_muted};margin-top:2px;">Dossiê Físico de Repertório • {p_name} ({profile.get("team", "")}) • 50 = Média MLB • 80 = Elite Hall of Fame</div>
        </div>
        <div style="background:{row_bg};border:1px solid {border_c};border-left:3px solid {team_hex};border-radius:6px;padding:6px 12px;text-align:right;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:800;color:{text_muted};letter-spacing:0.06em;">OVERALL STUFF+</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:800;color:{team_hex};font-variant-numeric:tabular-nums;">{overall_stuff}</div>
        </div>
      </div>
      <div>{all_rows}</div>
    </div>
    """
    return _compact_html(raw_html)
