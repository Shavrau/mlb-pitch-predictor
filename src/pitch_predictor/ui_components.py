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
    Resolve the active UI theme mode (`"dark"` or `"light"`).
    Defaults strictly to `"dark"` (Dark Broadcast `#0A0B0E`).
    """
    candidate = mode
    if candidate is None:
        try:
            import streamlit as st

            candidate = st.session_state.get("ui_theme_mode")
        except Exception:  # noqa: BLE001
            candidate = None

    if candidate and "light" in str(candidate).strip().lower():
        return "light"
    return "dark"


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
    status_color = "#F59E0B" if risp_active else ("#0284C7" if runners_count == 1 else "#94A3B8")

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
    selected_zone: int, prev_category: str = "First Pitch"
) -> str:
    """
    Render the official MLB Statcast Strike Zone SVG from the catcher's perspective.
    """
    cat_style = PITCH_CATEGORY_STYLES.get(prev_category, {})
    active_color = cat_style.get("primary", "#38BDF8")

    def _chase_style(zone_id: int) -> tuple[str, str, str]:
        if selected_zone == zone_id:
            return (active_color, "#F8FAFC", "0.68")
        return ("#334155", "#64748B", "0.18")

    def _strike_style(zone_id: int) -> tuple[str, str, str]:
        if selected_zone == zone_id:
            return (active_color, "#FFFFFF", "0.82")
        if zone_id == 5:
            return ("#1E293B", "#38BDF8", "0.42")
        return ("#0F172A", "#38BDF8", "0.30")

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
            f'fill-opacity="{opac}" stroke="{stroke_c}" stroke-width="1.2" />\n'
            f'<text x="{x + 22.5}" y="{y + 27}" text-anchor="middle" fill="{text_c}" '
            f'font-size="12" font-weight="{font_w}" font-family="JetBrains Mono, monospace">{z_id}</text>'
        )

    marker_svg = ""
    if selected_zone in zone_centers:
        cx, cy = zone_centers[selected_zone]
        marker_svg = (
            f'<circle cx="{cx}" cy="{cy}" r="13" fill="none" stroke="#F59E0B" '
            f'stroke-width="2" stroke-dasharray="3,2" />\n'
            f'<circle cx="{cx}" cy="{cy}" r="4.5" fill="#F59E0B" />'
        )

    zone_meta = STATCAST_ZONES.get(selected_zone, STATCAST_ZONES[0])
    caption_text = zone_meta["label"]

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
        <rect x="52" y="42" width="135" height="135" fill="none" stroke="#38BDF8" stroke-width="2.2" />
        {marker_svg}
        <polygon points="85,216 155,216 142,226 120,233 98,226" fill="#E2E8F0" stroke="#64748B" stroke-width="1.4" />
        <text x="120" y="242" text-anchor="middle" fill="#0284C7" font-size="8.2" font-weight="700" font-family="JetBrains Mono, monospace">{caption_text}</text>
      </svg>
    </div>
    """
    return _compact_html(raw_svg)


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


def render_scoreboard_banner_html(features: dict[str, Any], endpoint_name: str) -> str:
    """Render an Apple TV+ / MLB Broadcast Scorebug bar summarizing the live game state."""
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
    Return Dual-Mode custom CSS implementing Broadcast Editorial Design System v2.0.
    Defaults strictly to Dark Broadcast (`#0A0B0E`) and adapts cleanly when `theme_mode="light"`.
    """
    resolved = resolve_ui_theme_mode(theme_mode)
    if resolved == "light":
        pp_bg = "#F8FAFC"
        pp_surface = "#FFFFFF"
        pp_surface_alt = "#F1F5F9"
        pp_border = "#CBD5E1"
        pp_text = "#0F172A"
        pp_muted = "#475569"
        pp_sidebar_bg = "#F1F5F9"
        color_scheme = "light"
        scroll_thumb = "#94A3B8"
        scroll_track = "#F1F5F9"
    else:
        pp_bg = "#0A0B0E"
        pp_surface = "#12141A"
        pp_surface_alt = "#181B23"
        pp_border = "#222631"
        pp_text = "#F1F5F9"
        pp_muted = "#94A3B8"
        pp_sidebar_bg = "#0E1015"
        color_scheme = "dark"
        scroll_thumb = "#334155"
        scroll_track = "#0E1015"

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
        --pitch-offspeed: #10B981;
        --base-occupied: #F59E0B;
        --strike-zone: #38BDF8;
        scrollbar-width: thin;
        scrollbar-color: {scroll_thumb} {scroll_track};
      }}
      .stApp {{
        background-color: var(--pp-bg) !important;
        color: var(--pp-text) !important;
        font-family: 'Space Grotesk', system-ui, sans-serif;
      }}
      header[data-testid="stHeader"] {{
        background-color: var(--pp-bg) !important;
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
      div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: var(--pp-surface);
        border: 1px solid var(--pp-border) !important;
        border-radius: 8px !important;
        padding: 14px !important;
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
      .section-header {{
        font-family: 'Barlow Condensed', 'Space Grotesk', sans-serif;
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--pp-text);
        margin-bottom: 8px;
        border-bottom: 1px solid var(--pp-border);
        border-left: 3px solid var(--pp-accent);
        padding: 2px 0 5px 9px;
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


def render_pitcher_photo_html(pitcher_name: str, height_px: int = 138) -> str:
    """Renderiza a foto oficial do arremessador da MLB com faixa cromática da franquia."""
    src = get_pitcher_headshot_src(pitcher_name)
    meta = MLB_PITCHERS.get(pitcher_name, MLB_PITCHERS["Paul Skenes"])
    team_hex = meta.get("team_primary_hex", "#F59E0B")
    team_abbr = meta.get("team_abbr", "MLB")
    jersey = meta.get("jersey_number", "#00")
    img_max_h = max(64, int(height_px) - 12)

    raw_html = f"""
    <div style="background:linear-gradient(180deg,rgba(24,27,35,0.96) 0%,rgba(10,11,14,0.98) 100%);border-radius:6px;border:1px solid #222631;border-top:3px solid {team_hex};padding:4px;margin:2px auto 6px auto;height:{int(height_px)}px;position:relative;display:flex;align-items:flex-end;justify-content:center;overflow:hidden;">
      <span style="position:absolute;top:5px;right:7px;font-family:'JetBrains Mono',monospace;font-size:9.5px;font-weight:800;color:{team_hex};letter-spacing:0.04em;">{team_abbr} {jersey}</span>
      <img src="{src}" alt="{pitcher_name}" style="max-height:{img_max_h}px;max-width:100%;object-fit:contain;filter:drop-shadow(0 4px 8px rgba(0,0,0,0.65));display:block;margin:0 auto;" />
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
    cat_style = PITCH_CATEGORY_STYLES.get(pitch_category, PITCH_CATEGORY_STYLES["Fastball"])
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
        ai_style = PITCH_CATEGORY_STYLES.get(ai_category, PITCH_CATEGORY_STYLES["Fastball"])
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

    verdict_color = "#10B981" if is_success else "#EF4444"

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
        <text x="{tunnel_x + (14 if p_throws == 'L' else -14)}" y="{tunnel_y - 6}" text-anchor="{'start' if p_throws == 'L' else 'end'}" fill="#F59E0B" font-size="8" font-family="JetBrains Mono, monospace">PONTO DE QUEBRA (DISFARCE)</text>
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
        <text x="415" y="63" fill="#CBD5E1" font-size="9" font-weight="700" font-family="JetBrains Mono, monospace">ZONA ALVO: #{target_zone} ({STATCAST_ZONES.get(target_zone, STATCAST_ZONES[5])['region'].upper()})</text>
      </svg>
    </div>
    """
    return _compact_html(raw_svg)
