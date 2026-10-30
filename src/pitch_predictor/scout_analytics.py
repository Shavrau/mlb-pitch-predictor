"""
MLB Scouting Report, 12-Count Tactical Decision Matrix, Radar SVG, and Batter Game Plan Generator.

Provides:
- `MLB_12_COUNTS`: Canonical list of all 12 `(balls, strikes)` tuples in an MLB at-bat.
- `generate_12_counts_batch`: Builds the 12-record 24-feature tactical batch for a given pitcher
  and game situation, enforcing `First Pitch` on `0-0`.
- `simulate_tactical_12_counts_fallback`: Deterministic baseball-aware fallback matrix generator
  for instant offline/initial visualization.
- `compute_matrix_distribution`: Aggregates predicted categories across the 12 counts into
  normalized proportions (`Fastball`, `Breaking Ball`, `Offspeed`).
- `generate_batter_game_plan`: Synthesizes a 3-phase tactical approach for the batter
  (`early_counts`, `hitter_counts`, `two_strike_protection`).
- `render_count_matrix_html`: CommonMark-safe 4x3 HTML tactical count heatmap.
- `render_radar_chart_svg`: CommonMark-safe triangular SVG radar chart comparing MLB League Average,
  Pitcher Historical Priors, and AI Situational Matrix distribution.
- `render_game_plan_cards_html`: CommonMark-safe 3-column HTML tactical cards for the batter.
- `export_scouting_report_markdown`: Formatted Markdown report ready for coaching staff download.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from pitch_predictor.serving_client import (
    MLB_LEAGUE_PRIORS,
    PITCHER_AHEAD_COUNTS,
    PITCHER_BEHIND_COUNTS,
    derive_tactical_features,
    normalize_priors,
)
from pitch_predictor.ui_components import PITCH_CATEGORY_STYLES, _compact_html

MLB_12_COUNTS: list[tuple[int, int]] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
    (3, 0),
    (3, 1),
    (3, 2),
]


def generate_12_counts_batch(
    pitcher_profile: dict[str, Any],
    stand: str = "L",
    on_1b: int = 0,
    on_2b: int = 0,
    on_3b: int = 0,
    outs_when_up: int = 1,
    inning: int = 5,
    inning_topbot: str = "Top",
    score_diff: int = 0,
    n_thruorder_pitcher: int = 2,
    prev_pitch_category: str = "Fastball",
    prev_pitch_type: str = "FF",
    prev_pitch_zone: int = 5,
    prev_pitch_result: str = "S",
) -> list[dict[str, Any]]:
    """
    Generate a batch of 12 validated 24-column feature dictionaries covering all MLB counts.

    Enforces `First Pitch` semantics on `(0, 0)` and applies the specified previous-pitch
    context on the remaining 11 counts.
    """
    p_throws = str(pitcher_profile.get("p_throws", "R")).upper()
    fb_prior = float(pitcher_profile.get("pitcher_fb_prior", MLB_LEAGUE_PRIORS[0]))
    bb_prior = float(pitcher_profile.get("pitcher_bb_prior", MLB_LEAGUE_PRIORS[1]))
    os_prior = float(pitcher_profile.get("pitcher_os_prior", MLB_LEAGUE_PRIORS[2]))
    fb_norm, bb_norm, os_norm = normalize_priors(fb_prior, bb_prior, os_prior)

    non_zero_cat = (
        "Fastball" if prev_pitch_category == "First Pitch" else prev_pitch_category
    )
    non_zero_type = "FF" if prev_pitch_type == "None" else prev_pitch_type
    non_zero_zone = 5 if int(prev_pitch_zone) == 0 else int(prev_pitch_zone)
    non_zero_res = "S" if prev_pitch_result == "First Pitch" else prev_pitch_result

    batch_records: list[dict[str, Any]] = []
    for balls, strikes in MLB_12_COUNTS:
        if balls == 0 and strikes == 0:
            cat, p_type, zone, res = ("First Pitch", "None", 0, "First Pitch")
        else:
            cat, p_type, zone, res = (
                non_zero_cat,
                non_zero_type,
                non_zero_zone,
                non_zero_res,
            )

        raw_item: dict[str, Any] = {
            "prev_pitch_category": cat,
            "prev_pitch_type": p_type,
            "prev_pitch_zone": zone,
            "prev_pitch_result": res,
            "p_throws": p_throws,
            "stand": stand,
            "balls": balls,
            "strikes": strikes,
            "outs_when_up": outs_when_up,
            "inning": inning,
            "inning_topbot": inning_topbot,
            "score_diff": score_diff,
            "on_1b_flag": int(on_1b),
            "on_2b_flag": int(on_2b),
            "on_3b_flag": int(on_3b),
            "n_thruorder_pitcher": n_thruorder_pitcher,
            "pitcher_fb_prior": fb_norm,
            "pitcher_bb_prior": bb_norm,
            "pitcher_os_prior": os_norm,
        }
        batch_records.append(derive_tactical_features(raw_item))

    return batch_records


def simulate_tactical_12_counts_fallback(
    batch_records: list[dict[str, Any]],
) -> dict[str, str]:
    """
    Compute a deterministic, baseball-authentic 12-count prediction map based on
    pitcher priors, count leverage, platoon handedness, and RISP when offline or pre-loading.
    Delegates to the unified fallback engine.
    """
    from pitch_predictor.fallback_engine import simulate_12_counts_fallback

    return simulate_12_counts_fallback(batch_records)


def compute_matrix_distribution(
    predictions_by_count: dict[str, str] | list[str],
) -> dict[str, float]:
    """
    Calculate the normalized distribution of `Fastball`, `Breaking Ball`, and `Offspeed`
    across the 12-count prediction matrix.
    """
    values = (
        list(predictions_by_count.values())
        if isinstance(predictions_by_count, dict)
        else list(predictions_by_count)
    )
    if not values:
        return {
            "Fastball": MLB_LEAGUE_PRIORS[0],
            "Breaking Ball": MLB_LEAGUE_PRIORS[1],
            "Offspeed": MLB_LEAGUE_PRIORS[2],
        }

    counts = Counter(values)
    total = float(len(values))
    fb = counts.get("Fastball", 0) / total
    bb = counts.get("Breaking Ball", 0) / total
    os_val = counts.get("Offspeed", 0) / total

    norm_fb, norm_bb, norm_os = normalize_priors(fb, bb, os_val)
    return {
        "Fastball": norm_fb,
        "Breaking Ball": norm_bb,
        "Offspeed": norm_os,
    }


def _dominant_category_for_counts(
    predictions_by_count: dict[str, str], target_counts: list[str]
) -> str:
    """Return the most frequent pitch category across a subset of counts."""
    subset = [predictions_by_count.get(c, "Fastball") for c in target_counts]
    if not subset:
        return "Fastball"
    counts = Counter(subset)
    return counts.most_common(1)[0][0]


def generate_batter_game_plan(
    pitcher_name: str,
    predictions_by_count: dict[str, str],
    priors: dict[str, float],
    stand: str = "L",
    risp_active: bool = False,
) -> dict[str, dict[str, str]]:
    """
    Synthesize a 3-stage tactical Batter Game Plan from the 12-count AI matrix.

    Returns a dictionary with keys:
    - `"early_counts"` (`0-0`, `1-0`, `0-1`, `1-1`)
    - `"hitter_counts"` (`2-0`, `2-1`, `3-0`, `3-1`)
    - `"two_strike_protection"` (`0-2`, `1-2`, `2-2`, `3-2`)
    """
    early_list = ["0-0", "1-0", "0-1", "1-1"]
    hitter_list = ["2-0", "2-1", "3-0", "3-1"]
    two_strike_list = ["0-2", "1-2", "2-2", "3-2"]

    dom_early = _dominant_category_for_counts(predictions_by_count, early_list)
    dom_hitter = _dominant_category_for_counts(predictions_by_count, hitter_list)
    dom_two_strike = _dominant_category_for_counts(predictions_by_count, two_strike_list)

    risp_context = (
        "Com corredores em posição de anotar corrida (RISP), o arremessador evita o centro da zona (Zona 5)."
        if risp_active
        else "Sem pressão imediata de RISP, o arremessador ataca a zona para adiantar a contagem."
    )
    platoon_context = (
        "Rebatedor Canhoto (LHB)" if stand.upper() == "L" else "Rebatedor Destro (RHB)"
    )

    early_covered = " • ".join(
        f"{c}: {predictions_by_count.get(c, 'Fastball')}" for c in early_list
    )
    hitter_covered = " • ".join(
        f"{c}: {predictions_by_count.get(c, 'Fastball')}" for c in hitter_list
    )
    two_strike_covered = " • ".join(
        f"{c}: {predictions_by_count.get(c, 'Breaking Ball')}" for c in two_strike_list
    )

    fb_pct = float(priors.get("pitcher_fb_prior", MLB_LEAGUE_PRIORS[0])) * 100.0
    bb_pct = float(priors.get("pitcher_bb_prior", MLB_LEAGUE_PRIORS[1])) * 100.0
    os_pct = float(priors.get("pitcher_os_prior", MLB_LEAGUE_PRIORS[2])) * 100.0

    return {
        "early_counts": {
            "title": "1. Abertura & Contagens Neutras (0-0, 1-0, 0-1, 1-1)",
            "dominant_pitch": dom_early,
            "counts_covered": early_covered,
            "strategy": (
                f"Contra {pitcher_name} ({platoon_context}), ancore o plano de swing em {dom_early} "
                f"no terço médio/alto da zona (Zonas 2, 5 e 6). {risp_context}"
            ),
            "chase_warning": (
                f"Evite swings em arremessos secundários baixos nas Zonas 13 e 14 antes de 2 strikes "
                f"(Repertório base: FB {fb_pct:.0f}% / BB {bb_pct:.0f}% / OS {os_pct:.0f}%)."
            ),
        },
        "hitter_counts": {
            "title": "2. Vantagem do Rebatedor / Green Light (2-0, 2-1, 3-0, 3-1)",
            "dominant_pitch": dom_hitter,
            "counts_covered": hitter_covered,
            "strategy": (
                f"Zona de dano máximo: o modelo projeta {dom_hitter} como chamada dominante para "
                f"forçar o strike e evitar o walk. Busque contato forte puxando a bola (pull-side)."
            ),
            "chase_warning": (
                "Se o arremesso sair do centro da zona (Zonas 4-5-6), aceite a base por bolas (Walk / BB) "
                "e não expanda a zona em 3-1."
            ),
        },
        "two_strike_protection": {
            "title": "3. Proteção com 2 Strikes & Putaway (0-2, 1-2, 2-2, 3-2)",
            "dominant_pitch": dom_two_strike,
            "counts_covered": two_strike_covered,
            "strategy": (
                f"Modo de batalha (B-Swing): o modelo aponta {dom_two_strike} como principal arma de "
                f"finalização. Encurte o swing, ajuste o tempo para a bola rápida e rebata para o campo oposto."
            ),
            "chase_warning": (
                "Atenção redobrada nas zonas periféricas (Chase/Shadow): proteja as bordas (Zonas 7-9) com foul balls e "
                "cuidado com Sliders e Changeups caindo nas Zonas 13 e 14."
            ),
        },
    }


def render_count_matrix_html(
    predictions_by_count: dict[str, str],
    theme_mode: str = "dark",
    active_count: str | None = None,
) -> str:
    """
    Render the 4x3 MLB Tactical Count Matrix HTML (Balls 0..3 rows x Strikes 0..2 columns).

    Strictly CommonMark-safe (single line, no 4-space indentation, no HTML comments)
    and Dual-Mode adaptive via `--pp-*` CSS custom properties.
    """
    is_light = str(theme_mode).strip().lower().startswith("light")
    muted_color = "var(--pp-muted, #475569)" if is_light else "var(--pp-muted, #94A3B8)"
    border_color = "var(--pp-border, #C2CFDD)" if is_light else "var(--pp-border, #222631)"

    putaway_color = "#D97706" if is_light else "#F59E0B"
    header_cells = [
        f'<div style="background:var(--pp-surface-alt,#181B23);border:1px solid {border_color};border-radius:6px;padding:8px;text-align:center;font-family:\'JetBrains Mono\',monospace;font-size:10.5px;font-weight:700;letter-spacing:0.04em;color:{muted_color};">BALLS \\ STRIKES</div>',
        f'<div style="background:var(--pp-surface-alt,#181B23);border:1px solid {border_color};border-radius:6px;padding:8px;text-align:center;font-family:\'JetBrains Mono\',monospace;font-size:11px;font-weight:700;letter-spacing:0.04em;color:var(--pp-text,#F1F5F9);">0 STRIKES</div>',
        f'<div style="background:var(--pp-surface-alt,#181B23);border:1px solid {border_color};border-radius:6px;padding:8px;text-align:center;font-family:\'JetBrains Mono\',monospace;font-size:11px;font-weight:700;letter-spacing:0.04em;color:var(--pp-text,#F1F5F9);">1 STRIKE</div>',
        f'<div style="background:var(--pp-surface-alt,#181B23);border:1px solid {border_color};border-radius:6px;padding:8px;text-align:center;font-family:\'JetBrains Mono\',monospace;font-size:11px;font-weight:700;letter-spacing:0.04em;color:{putaway_color};">2 STRIKES [PUTAWAY]</div>',
    ]

    grid_items: list[str] = list(header_cells)

    for balls in range(4):
        grid_items.append(
            f'<div style="background:var(--pp-surface-alt,#181B23);border:1px solid {border_color};border-radius:6px;padding:10px;display:flex;flex-direction:column;justify-content:center;align-items:center;font-family:\'JetBrains Mono\',monospace;font-variant-numeric:tabular-nums;">'
            f'<span style="font-size:12.5px;font-weight:800;color:var(--pp-text,#F1F5F9);">{balls} BALL{"S" if balls != 1 else ""}</span>'
            f'<span style="font-size:9.5px;color:{muted_color};font-weight:700;">LINHA {balls}-X</span>'
            f"</div>"
        )
        for strikes in range(3):
            count_key = f"{balls}-{strikes}"
            pred_cat = predictions_by_count.get(count_key, "Fastball")
            style = PITCH_CATEGORY_STYLES.get(
                pred_cat, PITCH_CATEGORY_STYLES["Fastball"]
            )
            primary = style["primary"]
            secondary = style.get("secondary", primary)
            badge_color = secondary if is_light else primary
            bg_tint = style["bg_tint"]
            icon = style["icon"]
            code_badge = style.get("code_badge", style["arsenal"].split("•")[0].strip())

            if count_key in PITCHER_AHEAD_COUNTS:
                lev_label = "AHEAD"
                lev_color = "#059669" if is_light else "#10B981"
            elif count_key in PITCHER_BEHIND_COUNTS:
                lev_label = "BEHIND"
                lev_color = "#D97706" if is_light else "#F59E0B"
            else:
                lev_label = "EVEN"
                lev_color = "var(--pp-muted, #475569)" if is_light else "var(--pp-muted, #94A3B8)"

            is_active = active_count == count_key
            if is_active:
                cell_border = f"2px solid {primary}"
                cell_shadow = (
                    "box-shadow: 0 2px 8px rgba(15, 23, 42, 0.18), 0 0 0 1.5px var(--pp-accent, #0284C7);"
                    if is_light
                    else "box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35), 0 0 0 1px var(--pp-accent, #38BDF8);"
                )
            else:
                cell_border = f"1px solid {border_color}"
                cell_shadow = (
                    "box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);"
                    if is_light
                    else "box-shadow: 0 1px 3px rgba(0, 0, 0, 0.20);"
                )

            pill_bg = "var(--pp-surface-alt, #E1E8F0)" if is_light else "var(--pp-bg, #0A0B0E)"
            pill_text = "var(--pp-text, #0F172A)" if is_light else "var(--pp-text, #F1F5F9)"

            cell_html = (
                f'<div style="background:{bg_tint};border:{cell_border};border-left:4px solid {primary};border-radius:6px;padding:10px 12px;display:flex;flex-direction:column;justify-content:space-between;{cell_shadow}">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
                f'<span style="background:{pill_bg};color:{pill_text};font-family:\'JetBrains Mono\',monospace;font-variant-numeric:tabular-nums;font-size:12.5px;font-weight:800;padding:2px 7px;border-radius:4px;border:1px solid {border_color};">{count_key}</span>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:9.5px;font-weight:700;color:{lev_color};text-transform:uppercase;letter-spacing:0.04em;">{lev_label}</span>'
                f"</div>"
                f'<div style="font-size:14px;font-weight:800;color:var(--pp-text,#F1F5F9);margin:2px 0;">{icon} {pred_cat}</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:{badge_color};font-weight:700;letter-spacing:0.04em;">{code_badge}</div>'
                f"</div>"
            )
            grid_items.append(cell_html)

    raw_html = f"""
    <div style="background:var(--pp-surface,#12141A);border:1px solid {border_color};border-radius:8px;padding:12px;margin:6px 0;">
      <div style="display:grid;grid-template-columns:108px repeat(3,1fr);gap:8px;align-items:stretch;">
        {"".join(grid_items)}
      </div>
    </div>
    """
    return _compact_html(raw_html)


def _extract_dist_triplet(
    dist: dict[str, float] | tuple[float, float, float] | list[float],
) -> tuple[float, float, float]:
    """Normalize a dict or tuple distribution into `(fb, bb, os)` floats."""
    if isinstance(dist, dict):
        fb = float(dist.get("Fastball", dist.get("pitcher_fb_prior", MLB_LEAGUE_PRIORS[0])))
        bb = float(dist.get("Breaking Ball", dist.get("pitcher_bb_prior", MLB_LEAGUE_PRIORS[1])))
        os_val = float(dist.get("Offspeed", dist.get("pitcher_os_prior", MLB_LEAGUE_PRIORS[2])))
        return normalize_priors(fb, bb, os_val)
    items = list(dist)
    if len(items) >= 3:
        return normalize_priors(float(items[0]), float(items[1]), float(items[2]))
    return MLB_LEAGUE_PRIORS


def render_radar_chart_svg(
    league_dist: dict[str, float] | tuple[float, float, float],
    pitcher_dist: dict[str, float] | tuple[float, float, float],
    model_dist: dict[str, float] | tuple[float, float, float],
) -> str:
    """
    Render a 3-axis SVG Radar Chart comparing:
    1. MLB League Average (`#94A3B8` dashed)
    2. Pitcher Historical Prior (`#F59E0B` amber)
    3. AI Situational Matrix Call (`#38BDF8` sky blue)

    Strictly CommonMark-safe via `_compact_html` and XML-valid.
    """
    l_fb, l_bb, l_os = _extract_dist_triplet(league_dist)
    p_fb, p_bb, p_os = _extract_dist_triplet(pitcher_dist)
    m_fb, m_bb, m_os = _extract_dist_triplet(model_dist)

    cx, cy = 190.0, 142.0
    max_r = 92.0
    angles = (-math.pi / 2.0, math.pi / 6.0, 5.0 * math.pi / 6.0)

    def _pt(val: float, angle: float, scale: float = 1.0) -> str:
        norm_r = min(1.0, max(0.08, val / 0.80)) * max_r * scale
        x = cx + norm_r * math.cos(angle)
        y = cy + norm_r * math.sin(angle)
        return f"{x:.1f},{y:.1f}"

    def _ring_points(frac: float) -> str:
        pts = []
        for ang in angles:
            x = cx + max_r * frac * math.cos(ang)
            y = cy + max_r * frac * math.sin(ang)
            pts.append(f"{x:.1f},{y:.1f}")
        return " ".join(pts)

    league_pts = " ".join(
        [_pt(l_fb, angles[0]), _pt(l_bb, angles[1]), _pt(l_os, angles[2])]
    )
    pitcher_pts = " ".join(
        [_pt(p_fb, angles[0]), _pt(p_bb, angles[1]), _pt(p_os, angles[2])]
    )
    model_pts = " ".join(
        [_pt(m_fb, angles[0]), _pt(m_bb, angles[1]), _pt(m_os, angles[2])]
    )

    top_x, top_y = cx, cy - max_r
    br_x, br_y = cx + max_r * math.cos(angles[1]), cy + max_r * math.sin(angles[1])
    bl_x, bl_y = cx + max_r * math.cos(angles[2]), cy + max_r * math.sin(angles[2])

    raw_svg = f"""
    <div style="display:flex;justify-content:center;align-items:center;margin:6px 0;">
      <svg viewBox="0 0 380 285" width="100%" style="max-width:390px;background:var(--pp-surface,#12141A);border-radius:8px;border:1px solid var(--pp-border,#222631);padding:8px;display:block;">
        <polygon points="{_ring_points(1.0)}" fill="none" stroke="#475569" stroke-opacity="0.45" stroke-width="1.4" />
        <polygon points="{_ring_points(0.75)}" fill="none" stroke="#475569" stroke-opacity="0.32" stroke-width="1" stroke-dasharray="3,3" />
        <polygon points="{_ring_points(0.50)}" fill="none" stroke="#475569" stroke-opacity="0.32" stroke-width="1" stroke-dasharray="3,3" />
        <polygon points="{_ring_points(0.25)}" fill="none" stroke="#475569" stroke-opacity="0.32" stroke-width="1" stroke-dasharray="3,3" />
        <line x1="{cx}" y1="{cy}" x2="{top_x:.1f}" y2="{top_y:.1f}" stroke="#64748B" stroke-opacity="0.45" stroke-width="1.1" />
        <line x1="{cx}" y1="{cy}" x2="{br_x:.1f}" y2="{br_y:.1f}" stroke="#64748B" stroke-opacity="0.45" stroke-width="1.1" />
        <line x1="{cx}" y1="{cy}" x2="{bl_x:.1f}" y2="{bl_y:.1f}" stroke="#64748B" stroke-opacity="0.45" stroke-width="1.1" />
        <polygon points="{league_pts}" fill="rgba(148,163,184,0.12)" stroke="#94A3B8" stroke-width="1.8" stroke-dasharray="4,3" />
        <polygon points="{pitcher_pts}" fill="rgba(245,158,11,0.16)" stroke="#F59E0B" stroke-width="2.2" />
        <polygon points="{model_pts}" fill="rgba(56,189,248,0.22)" stroke="#38BDF8" stroke-width="2.6" />
        <text x="{cx}" y="24" text-anchor="middle" fill="#EF4444" font-size="11" font-weight="800" font-family="JetBrains Mono, monospace">▲ Fastball ({m_fb * 100:.0f}%)</text>
        <text x="312" y="212" text-anchor="middle" fill="#3B82F6" font-size="11" font-weight="800" font-family="JetBrains Mono, monospace">◆ Breaking Ball ({m_bb * 100:.0f}%)</text>
        <text x="68" y="212" text-anchor="middle" fill="#10B981" font-size="11" font-weight="800" font-family="JetBrains Mono, monospace">● Offspeed ({m_os * 100:.0f}%)</text>
        <rect x="24" y="246" width="12" height="4" fill="#94A3B8" />
        <text x="42" y="251" fill="#94A3B8" font-size="10" font-weight="600" font-family="sans-serif">Média MLB</text>
        <rect x="122" y="246" width="12" height="4" fill="#F59E0B" />
        <text x="140" y="251" fill="#F59E0B" font-size="10" font-weight="700" font-family="sans-serif">Prior Pitcher</text>
        <rect x="234" y="246" width="12" height="4" fill="#38BDF8" />
        <text x="252" y="251" fill="#0284C7" font-size="10" font-weight="800" font-family="sans-serif">Matriz IA (12 Contagens)</text>
      </svg>
    </div>
    """
    return _compact_html(raw_svg)


def render_game_plan_cards_html(game_plan: dict[str, dict[str, str]]) -> str:
    """
    Render the 3 Batter Tactical Approach Cards (`early_counts`, `hitter_counts`,
    `two_strike_protection`) as a CommonMark-safe Dual-Mode responsive HTML grid.
    """
    cards_markup: list[str] = []
    section_order = ("early_counts", "hitter_counts", "two_strike_protection")

    for key in section_order:
        sec = game_plan.get(key, {})
        title = sec.get("title", key)
        dom_pitch = sec.get("dominant_pitch", "Fastball")
        covered = sec.get("counts_covered", "")
        strategy = sec.get("strategy", "")
        warning = sec.get("chase_warning", "")

        style = PITCH_CATEGORY_STYLES.get(dom_pitch, PITCH_CATEGORY_STYLES["Fastball"])
        primary = style["primary"]
        bg_tint = style["bg_tint"]
        icon = style["icon"]
        code_badge = style.get("code_badge", "")

        card_html = (
            f'<div style="background:var(--pp-surface,#12141A);border:1px solid var(--pp-border,#222631);border-top:3px solid {primary};border-radius:8px;padding:15px;display:flex;flex-direction:column;justify-content:space-between;">'
            f"<div>"
            f'<div style="font-size:13px;font-weight:800;color:var(--pp-text,#F1F5F9);margin-bottom:8px;">{title}</div>'
            f'<div style="background:{bg_tint};border-left:3px solid {primary};padding:8px 10px;border-radius:4px;margin-bottom:10px;display:flex;justify-content:space-between;align-items:center;">'
            f"<div>"
            f'<span style="font-size:9.5px;color:var(--pp-muted,#94A3B8);display:block;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;">FOCO PRINCIPAL DO PLANO</span>'
            f'<strong style="font-size:15px;color:var(--pp-text,#F1F5F9);">{icon} {dom_pitch}</strong>'
            f"</div>"
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:10px;font-weight:700;color:{primary};">{code_badge}</span>'
            f"</div>"
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10.5px;color:var(--pp-muted,#94A3B8);font-weight:700;margin-bottom:8px;">{covered}</div>'
            f'<p style="font-size:12.5px;color:var(--pp-text,#F1F5F9);opacity:0.9;line-height:1.45;margin:0 0 10px 0;">{strategy}</p>'
            f"</div>"
            f'<div style="background:var(--pp-surface-alt,#181B23);border:1px solid var(--pp-border,#222631);border-left:3px solid #F59E0B;border-radius:4px;padding:8px 10px;font-size:11px;color:var(--pp-text,#F1F5F9);">'
            f'<strong style="font-family:\'JetBrains Mono\',monospace;color:#F59E0B;letter-spacing:0.03em;">[DISCIPLINA DE PLACA]</strong> {warning}'
            f"</div>"
            f"</div>"
        )
        cards_markup.append(card_html)

    raw_html = f"""
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin:6px 0;">
      {"".join(cards_markup)}
    </div>
    """
    return _compact_html(raw_html)


def export_scouting_report_markdown(
    pitcher_name: str,
    predictions_by_count: dict[str, str],
    game_plan: dict[str, dict[str, str]],
    priors: dict[str, float],
    stand: str = "L",
    risp_active: bool = False,
    **kwargs: Any,
) -> str:
    """
    Generate a complete, formatted Markdown Scouting Report for the coaching staff and batter.
    """
    fb_p, bb_p, os_p = _extract_dist_triplet(priors)
    model_dist = compute_matrix_distribution(predictions_by_count)
    stand_label = "Canhoto (LHB)" if stand.upper() == "L" else "Destro (RHB)"
    risp_label = "Sim (RISP Ativo — 2B/3B)" if risp_active else "Não (Bases Limpas / 1B)"

    lines: list[str] = [
        f"# ⚾ MLB Statcast Scouting Report — {pitcher_name}",
        "",
        "## 1. Contexto Tático do Confronto",
        f"- **Arremessador Analisado:** {pitcher_name}",
        f"- **Lado do Rebatedor (Platoon):** {stand_label}",
        f"- **Corredores em Posição de Anotação (RISP):** {risp_label}",
        f"- **Repertório Base (Priors):** Fastball `{fb_p * 100:.1f}%` | Breaking Ball `{bb_p * 100:.1f}%` | Offspeed `{os_p * 100:.1f}%`",
        f"- **Tendência Situacional da IA (12 Contagens):** Fastball `{model_dist['Fastball'] * 100:.1f}%` | Breaking Ball `{model_dist['Breaking Ball'] * 100:.1f}%` | Offspeed `{model_dist['Offspeed'] * 100:.1f}%`",
        "",
        "## 2. Matriz de Decisão Tática (12 Contagens da MLB)",
        "",
        "| Balls \\ Strikes | 0 Strikes | 1 Strike | 2 Strikes (Putaway) |",
        "| :--- | :---: | :---: | :---: |",
    ]

    for b in range(4):
        c0 = f"{b}-0: **{predictions_by_count.get(f'{b}-0', 'Fastball')}**"
        c1 = f"{b}-1: **{predictions_by_count.get(f'{b}-1', 'Fastball')}**"
        c2 = f"{b}-2: **{predictions_by_count.get(f'{b}-2', 'Breaking Ball')}**"
        lines.append(f"| **{b} Balls** | {c0} | {c1} | {c2} |")

    lines.extend(
        [
            "",
            "## 3. Game Plan do Rebatedor (Abordagem por Fase da Contagem)",
            "",
        ]
    )

    for sec_key in ("early_counts", "hitter_counts", "two_strike_protection"):
        sec = game_plan.get(sec_key, {})
        lines.extend(
            [
                f"### {sec.get('title', sec_key)}",
                f"- **Arremesso Dominante Previsto:** `{sec.get('dominant_pitch', 'Fastball')}`",
                f"- **Contagens Mapeadas:** {sec.get('counts_covered', '')}",
                f"- **Estratégia de Swing:** {sec.get('strategy', '')}",
                f"- **Alerta de Chase/Disciplina:** {sec.get('chase_warning', '')}",
                "",
            ]
        )

    return "\n".join(lines)



# ---------------------------------------------------------------------------
# Sprint 2 Deliverables: Head-to-Head Scout Comparison & Printable Dugout Card
# ---------------------------------------------------------------------------


def render_head_to_head_radar_svg(
    league_dist: dict[str, float] | tuple[float, float, float],
    dist_a: dict[str, float] | tuple[float, float, float],
    dist_b: dict[str, float] | tuple[float, float, float],
    name_a: str = "Arremessador A",
    name_b: str = "Arremessador B",
    color_a: str = "#F59E0B",
    color_b: str = "#38BDF8",
) -> str:
    """
    Render a 3-axis triangular SVG Radar Chart comparing:
    1. MLB League Average (`#94A3B8` dashed)
    2. Pitcher A (`color_a`)
    3. Pitcher B (`color_b`)

    Strictly CommonMark-safe via `_compact_html` and XML-valid.
    """
    l_fb, l_bb, l_os = _extract_dist_triplet(league_dist)
    a_fb, a_bb, a_os = _extract_dist_triplet(dist_a)
    b_fb, b_bb, b_os = _extract_dist_triplet(dist_b)

    cx, cy = 205.0, 142.0
    max_r = 92.0
    angles = (-math.pi / 2.0, math.pi / 6.0, 5.0 * math.pi / 6.0)

    def _pt(val: float, angle: float, scale: float = 1.0) -> str:
        norm_r = min(1.0, max(0.08, val / 0.80)) * max_r * scale
        x = cx + norm_r * math.cos(angle)
        y = cy + norm_r * math.sin(angle)
        return f"{x:.1f},{y:.1f}"

    def _ring_points(frac: float) -> str:
        pts = []
        for ang in angles:
            x = cx + max_r * frac * math.cos(ang)
            y = cy + max_r * frac * math.sin(ang)
            pts.append(f"{x:.1f},{y:.1f}")
        return " ".join(pts)

    league_pts = " ".join(
        [_pt(l_fb, angles[0]), _pt(l_bb, angles[1]), _pt(l_os, angles[2])]
    )
    a_pts = " ".join(
        [_pt(a_fb, angles[0]), _pt(a_bb, angles[1]), _pt(a_os, angles[2])]
    )
    b_pts = " ".join(
        [_pt(b_fb, angles[0]), _pt(b_bb, angles[1]), _pt(b_os, angles[2])]
    )

    top_x, top_y = cx, cy - max_r
    br_x, br_y = cx + max_r * math.cos(angles[1]), cy + max_r * math.sin(angles[1])
    bl_x, bl_y = cx + max_r * math.cos(angles[2]), cy + max_r * math.sin(angles[2])

    clean_name_a = name_a.split("(")[0].strip()[:14]
    clean_name_b = name_b.split("(")[0].strip()[:14]

    raw_svg = f"""
    <div style="display:flex;justify-content:center;align-items:center;margin:6px 0;">
      <svg viewBox="0 0 410 295" width="100%" style="max-width:420px;background:var(--pp-surface,#12141A);border-radius:8px;border:1px solid var(--pp-border,#222631);padding:8px;display:block;">
        <polygon points="{_ring_points(1.0)}" fill="none" stroke="#475569" stroke-opacity="0.45" stroke-width="1.4" />
        <polygon points="{_ring_points(0.75)}" fill="none" stroke="#475569" stroke-opacity="0.32" stroke-width="1" stroke-dasharray="3,3" />
        <polygon points="{_ring_points(0.50)}" fill="none" stroke="#475569" stroke-opacity="0.32" stroke-width="1" stroke-dasharray="3,3" />
        <polygon points="{_ring_points(0.25)}" fill="none" stroke="#475569" stroke-opacity="0.32" stroke-width="1" stroke-dasharray="3,3" />
        <line x1="{cx}" y1="{cy}" x2="{top_x:.1f}" y2="{top_y:.1f}" stroke="#64748B" stroke-opacity="0.45" stroke-width="1.1" />
        <line x1="{cx}" y1="{cy}" x2="{br_x:.1f}" y2="{br_y:.1f}" stroke="#64748B" stroke-opacity="0.45" stroke-width="1.1" />
        <line x1="{cx}" y1="{cy}" x2="{bl_x:.1f}" y2="{bl_y:.1f}" stroke="#64748B" stroke-opacity="0.45" stroke-width="1.1" />
        <polygon points="{league_pts}" fill="rgba(148,163,184,0.10)" stroke="#94A3B8" stroke-width="1.8" stroke-dasharray="4,3" />
        <polygon points="{a_pts}" fill="rgba(245,158,11,0.18)" stroke="{color_a}" stroke-width="2.5" />
        <polygon points="{b_pts}" fill="rgba(56,189,248,0.18)" stroke="{color_b}" stroke-width="2.5" stroke-dasharray="6,2" />
        <text x="{cx}" y="20" text-anchor="middle" fill="#EF4444" font-size="11" font-weight="800" font-family="JetBrains Mono, monospace">▲ Fastball</text>
        <text x="{cx}" y="34" text-anchor="middle" font-size="10.5" font-weight="700" font-family="JetBrains Mono, monospace"><tspan fill="{color_a}">{clean_name_a} {a_fb * 100:.0f}%</tspan> <tspan fill="#64748B">|</tspan> <tspan fill="{color_b}">{clean_name_b} {b_fb * 100:.0f}%</tspan></text>
        <text x="340" y="210" text-anchor="middle" fill="#3B82F6" font-size="11" font-weight="800" font-family="JetBrains Mono, monospace">◆ Breaking</text>
        <text x="340" y="224" text-anchor="middle" font-size="10.5" font-weight="700" font-family="JetBrains Mono, monospace"><tspan fill="{color_a}">{a_bb * 100:.0f}%</tspan> <tspan fill="#64748B">|</tspan> <tspan fill="{color_b}">{b_bb * 100:.0f}%</tspan></text>
        <text x="70" y="210" text-anchor="middle" fill="#10B981" font-size="11" font-weight="800" font-family="JetBrains Mono, monospace">● Offspeed</text>
        <text x="70" y="224" text-anchor="middle" font-size="10.5" font-weight="700" font-family="JetBrains Mono, monospace"><tspan fill="{color_a}">{a_os * 100:.0f}%</tspan> <tspan fill="#64748B">|</tspan> <tspan fill="{color_b}">{b_os * 100:.0f}%</tspan></text>
        <rect x="20" y="258" width="12" height="4" fill="#94A3B8" />
        <text x="38" y="263" fill="#94A3B8" font-size="10" font-weight="600" font-family="sans-serif">Média MLB</text>
        <rect x="135" y="258" width="12" height="4" fill="{color_a}" />
        <text x="153" y="263" fill="{color_a}" font-size="10" font-weight="700" font-family="sans-serif">{name_a[:16]}</text>
        <rect x="270" y="258" width="12" height="4" fill="{color_b}" />
        <text x="288" y="263" fill="{color_b}" font-size="10" font-weight="700" font-family="sans-serif">{name_b[:16]}</text>
      </svg>
    </div>
    """
    return _compact_html(raw_svg)


def compute_head_to_head_divergence(
    preds_a: dict[str, str],
    preds_b: dict[str, str],
) -> dict[str, Any]:
    """
    Compare predictions of Pitcher A vs. Pitcher B across all 12 MLB counts.

    Returns:
    - `total_counts`: 12
    - `agreement_counts`: number of counts where both pitchers throw the same category
    - `divergence_counts`: number of counts with tactical disagreement
    - `agreement_pct`: agreement ratio (0.0 to 1.0)
    - `divergence_pct`: divergence ratio (0.0 to 1.0)
    - `divergent_counts`: list of count strings where calls differ
    - `consensus_counts`: list of count strings where calls match
    - `contrasts`: list of 12 dicts with detailed comparisons per count
    - `summary_text`: tactical narrative explaining the divergence
    """
    contrasts: list[dict[str, Any]] = []
    divergent: list[str] = []
    consensus: list[str] = []

    for b, s in MLB_12_COUNTS:
        ck = f"{b}-{s}"
        p_a = preds_a.get(ck, "Fastball")
        p_b = preds_b.get(ck, "Fastball")
        is_div = p_a != p_b

        if ck in PITCHER_AHEAD_COUNTS:
            lev = "AHEAD"
        elif ck in PITCHER_BEHIND_COUNTS:
            lev = "BEHIND"
        else:
            lev = "EVEN"

        record = {
            "count": ck,
            "balls": b,
            "strikes": s,
            "leverage": lev,
            "pitch_a": p_a,
            "pitch_b": p_b,
            "is_divergent": is_div,
        }
        contrasts.append(record)

        if is_div:
            divergent.append(ck)
        else:
            consensus.append(ck)

    total = len(MLB_12_COUNTS)
    n_div = len(divergent)
    n_agree = len(consensus)
    div_pct = float(n_div) / float(total)
    agree_pct = float(n_agree) / float(total)

    if n_div >= 7:
        summary_text = (
            f"Alta divergência tática ({div_pct * 100:.1f}% — {n_div} de 12 contagens com chamadas opostas). "
            f"Os dois arremessadores operam com filosofias de duelo antagônicas: enquanto um ataca agressivamente com velocidade, "
            f"o outro prioriza quebras e indução de contato fraco fora do centro da zona."
        )
    elif n_div >= 4:
        summary_text = (
            f"Divergência tática moderada ({div_pct * 100:.1f}% — {n_div} de 12 contagens). "
            f"Ambos compartilham tendências similares em contagens neutras, mas divergem significativamente "
            f"no momento decisivo com 2 strikes (putaway) e em contagens desfavoráveis."
        )
    else:
        summary_text = (
            f"Forte consenso tático ({agree_pct * 100:.1f}% de alinhamento — {n_agree} de 12 contagens idênticas). "
            f"Ambos os perfis respeitam a cartilha canônica da liga em quase todas as situações de pressão."
        )

    return {
        "total_counts": total,
        "agreement_counts": n_agree,
        "divergence_counts": n_div,
        "agreement_pct": agree_pct,
        "divergence_pct": div_pct,
        "divergent_counts": divergent,
        "consensus_counts": consensus,
        "contrasts": contrasts,
        "summary_text": summary_text,
    }


def generate_comparative_game_plan(
    name_a: str,
    name_b: str,
    preds_a: dict[str, str],
    preds_b: dict[str, str],
    priors_a: dict[str, float],
    priors_b: dict[str, float],
    stand: str = "L",
    risp_active: bool = False,
) -> dict[str, dict[str, str]]:
    """
    Synthesize a comparative 3-stage Game Plan contrasting the batter's tactical adjustments
    when facing Pitcher A vs. Pitcher B.
    """
    plan_a = generate_batter_game_plan(name_a, preds_a, priors_a, stand, risp_active)
    plan_b = generate_batter_game_plan(name_b, preds_b, priors_b, stand, risp_active)

    clean_a = name_a.split("(")[0].strip()
    clean_b = name_b.split("(")[0].strip()

    early_a = plan_a["early_counts"]["dominant_pitch"]
    early_b = plan_b["early_counts"]["dominant_pitch"]

    hitter_a = plan_a["hitter_counts"]["dominant_pitch"]
    hitter_b = plan_b["hitter_counts"]["dominant_pitch"]

    two_a = plan_a["two_strike_protection"]["dominant_pitch"]
    two_b = plan_b["two_strike_protection"]["dominant_pitch"]

    return {
        "early_counts": {
            "title": "1. Ajuste em Contagens Neutras (0-0, 1-0, 0-1, 1-1)",
            "dominant_a": early_a,
            "dominant_b": early_b,
            "strategy": (
                f"Ao enfrentar {clean_a}: ancore o swing em {early_a} na metade superior da zona. "
                f"Contra {clean_b}: reajuste a antecipação para {early_b}, preparando o tempo de reação "
                f"para a quebra ou atraso no ponto de soltura."
            ),
            "chase_warning": (
                f"Transição crítica: {clean_a} desafia na zona com maior frequência, enquanto "
                f"{clean_b} tenta expandir as bordas logo nos primeiros arremessos."
            ),
        },
        "hitter_counts": {
            "title": "2. Ajuste em Vantagem do Rebatedor (2-0, 2-1, 3-0, 3-1)",
            "dominant_a": hitter_a,
            "dominant_b": hitter_b,
            "strategy": (
                f"Em contagens favoráveis: {clean_a} projeta {hitter_a} para forçar o strike; "
                f"já {clean_b} aposta em {hitter_b}. Ataque a bola com agressividade puxando para o pull-side "
                f"apenas na zona de conforto."
            ),
            "chase_warning": (
                "Não persiga bolas fora do quadrante central (Zonas 4, 5 e 6). Aceite o walk se o arremesso escapar."
            ),
        },
        "two_strike_protection": {
            "title": "3. Ajuste de Sobrevivência com 2 Strikes (0-2, 1-2, 2-2, 3-2)",
            "dominant_a": two_a,
            "dominant_b": two_b,
            "strategy": (
                f"Modo B-Swing com 2 strikes: contra {clean_a}, a principal arma de eliminação é {two_a}; "
                f"contra {clean_b}, o foco defensivo é {two_b}. Encurte a pegada no bastão e proteja as laterais."
            ),
            "chase_warning": (
                f"Atenção às zonas de chase (11 a 14): {clean_a} busca whiffs em elevação/wipeout, enquanto "
                f"{clean_b} induz groundouts na terra com quedas tardias."
            ),
        },
    }


def export_scouting_card_html(
    pitcher_name: str,
    predictions_by_count: dict[str, str],
    game_plan: dict[str, dict[str, str]],
    priors: dict[str, float],
    stand: str = "L",
    risp_active: bool = False,
    theme_mode: str = "light",
) -> str:
    """
    Generate a high-fidelity, standalone printable HTML Dugout Card.
    Optimized for printing (Ctrl+P) and PDF export via `@media print` CSS rules.
    """
    fb_p, bb_p, os_p = _extract_dist_triplet(priors)
    model_dist = compute_matrix_distribution(predictions_by_count)
    stand_label = "Rebatedor Canhoto (LHB)" if stand.upper() == "L" else "Rebatedor Destro (RHB)"
    risp_label = "RISP Ativo (Corredores em 2B/3B)" if risp_active else "Bases Limpas / Sem RISP"

    grid_cells: list[str] = []
    for b in range(4):
        for s in range(3):
            ck = f"{b}-{s}"
            cat = predictions_by_count.get(ck, "Fastball")
            style = PITCH_CATEGORY_STYLES.get(cat, PITCH_CATEGORY_STYLES["Fastball"])
            c_badge = style.get("code_badge", "PITCH")
            prim = style.get("secondary", style["primary"])

            if ck in PITCHER_AHEAD_COUNTS:
                badge_txt = "AHEAD"
                badge_bg = "#ECFDF5"
                badge_c = "#065F46"
            elif ck in PITCHER_BEHIND_COUNTS:
                badge_txt = "BEHIND"
                badge_bg = "#FEF3C7"
                badge_c = "#92400E"
            else:
                badge_txt = "EVEN"
                badge_bg = "#F1F5F9"
                badge_c = "#334155"

            cell = f"""
            <div class="card-cell" style="border-left:4px solid {prim};">
              <div class="cell-head">
                <span class="count-pill">{ck}</span>
                <span class="lev-pill" style="background:{badge_bg};color:{badge_c};">{badge_txt}</span>
              </div>
              <div class="pitch-name">{style['icon']} {cat}</div>
              <div class="pitch-code" style="color:{prim};">{c_badge}</div>
            </div>
            """
            grid_cells.append(cell)

    early = game_plan.get("early_counts", {})
    hitter = game_plan.get("hitter_counts", {})
    two_s = game_plan.get("two_strike_protection", {})

    html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <title>MLB Statcast Dugout Card — {pitcher_name}</title>
  <style>
    @page {{
      size: A4 portrait;
      margin: 10mm;
    }}
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #FFFFFF;
      color: #0F172A;
      padding: 16px;
      line-height: 1.4;
      font-size: 13px;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }}
    .no-print {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: #0F172A;
      color: #FFFFFF;
      padding: 12px 20px;
      border-radius: 8px;
      margin-bottom: 20px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    .print-btn {{
      background: #2563EB;
      color: #FFFFFF;
      border: none;
      padding: 8px 18px;
      border-radius: 6px;
      font-weight: 700;
      font-size: 13px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .print-btn:hover {{
      background: #1D4ED8;
    }}
    .dugout-card {{
      border: 2px solid #CBD5E1;
      border-radius: 10px;
      padding: 18px;
      background: #FFFFFF;
    }}
    .header-bar {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      border-bottom: 2px solid #E2E8F0;
      padding-bottom: 12px;
      margin-bottom: 14px;
    }}
    .title-area h1 {{
      font-size: 20px;
      font-weight: 800;
      color: #0F172A;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }}
    .title-area .sub {{
      font-size: 12px;
      color: #64748B;
      font-weight: 600;
      margin-top: 2px;
    }}
    .meta-box {{
      text-align: right;
      font-family: monospace;
      font-size: 11px;
      color: #475569;
    }}
    .priors-bar {{
      display: flex;
      gap: 12px;
      background: #F8FAFC;
      border: 1px solid #E2E8F0;
      border-radius: 6px;
      padding: 8px 14px;
      margin-bottom: 14px;
      font-size: 11.5px;
      font-weight: 600;
    }}
    .priors-bar span strong {{
      color: #0F172A;
    }}
    .section-title {{
      font-size: 13px;
      font-weight: 800;
      text-transform: uppercase;
      color: #1E293B;
      letter-spacing: 0.05em;
      margin: 12px 0 8px 0;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .matrix-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-bottom: 16px;
    }}
    .card-cell {{
      background: #F8FAFC;
      border: 1px solid #E2E8F0;
      border-radius: 6px;
      padding: 8px 10px;
    }}
    .cell-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 4px;
    }}
    .count-pill {{
      font-family: monospace;
      font-weight: 800;
      font-size: 12px;
      color: #0F172A;
    }}
    .lev-pill {{
      font-size: 9px;
      font-weight: 800;
      padding: 1px 5px;
      border-radius: 3px;
      font-family: monospace;
    }}
    .pitch-name {{
      font-size: 13px;
      font-weight: 800;
      color: #0F172A;
    }}
    .pitch-code {{
      font-family: monospace;
      font-size: 9.5px;
      font-weight: 700;
    }}
    .plan-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin-bottom: 12px;
    }}
    .plan-card {{
      background: #F8FAFC;
      border: 1px solid #E2E8F0;
      border-radius: 6px;
      padding: 10px;
    }}
    .plan-card h4 {{
      font-size: 11.5px;
      font-weight: 800;
      color: #0F172A;
      margin-bottom: 4px;
    }}
    .plan-card .focus {{
      font-size: 11px;
      font-weight: 700;
      color: #2563EB;
      margin-bottom: 6px;
    }}
    .plan-card p {{
      font-size: 11px;
      color: #334155;
      line-height: 1.35;
      margin-bottom: 6px;
    }}
    .plan-card .alert {{
      background: #FEF3C7;
      border: 1px solid #FDE68A;
      border-radius: 4px;
      padding: 4px 6px;
      font-size: 10px;
      color: #92400E;
      font-weight: 600;
    }}
    .footer-note {{
      border-top: 1px solid #E2E8F0;
      padding-top: 8px;
      display: flex;
      justify-content: space-between;
      font-size: 10px;
      color: #64748B;
      font-family: monospace;
    }}
    @media print {{
      .no-print {{
        display: none !important;
      }}
      body {{
        padding: 0;
      }}
      .dugout-card {{
        border: 1px solid #94A3B8;
        padding: 12px;
      }}
    }}
  </style>
</head>
<body>
  <div class="no-print">
    <div>
      <strong>⚾ MLB STATCAST DUGOUT CARD — VISUALIZADOR DE IMPRESSÃO</strong>
      <span style="font-size:11px;opacity:0.8;display:block;">Pronto para impressão em A4 ou salvamento em PDF para uso no dugout</span>
    </div>
    <button class="print-btn" onclick="window.print()">
      🖨️ Imprimir / Salvar em PDF
    </button>
  </div>

  <div class="dugout-card">
    <div class="header-bar">
      <div class="title-area">
        <h1>{pitcher_name}</h1>
        <div class="sub">MLB STATCAST TACTICAL DOSSIER • DUGOUT GAME PLAN</div>
      </div>
      <div class="meta-box">
        <div><strong>PLATOON:</strong> {stand_label}</div>
        <div><strong>SITUAÇÃO:</strong> {risp_label}</div>
      </div>
    </div>

    <div class="priors-bar">
      <span><strong>Priors Base:</strong> FB {fb_p * 100:.0f}% • BB {bb_p * 100:.0f}% • OS {os_p * 100:.0f}%</span>
      <span>|</span>
      <span><strong>Matriz IA (12 Contagens):</strong> Fastball {model_dist['Fastball'] * 100:.0f}% • Breaking {model_dist['Breaking Ball'] * 100:.0f}% • Offspeed {model_dist['Offspeed'] * 100:.0f}%</span>
    </div>

    <div class="section-title">Matriz Tática de Decisão (12 Contagens da MLB)</div>
    <div class="matrix-grid">
      {"".join(grid_cells)}
    </div>

    <div class="section-title">Game Plan do Rebatedor por Fase da Contagem</div>
    <div class="plan-grid">
      <div class="plan-card">
        <h4>1. Neutras (0-0, 1-0, 0-1, 1-1)</h4>
        <div class="focus">Foco: {early.get('dominant_pitch', 'Fastball')}</div>
        <p>{early.get('strategy', '')}</p>
        <div class="alert">[CHASE] {early.get('chase_warning', '')}</div>
      </div>
      <div class="plan-card">
        <h4>2. Vantagem (2-0, 2-1, 3-0, 3-1)</h4>
        <div class="focus">Foco: {hitter.get('dominant_pitch', 'Fastball')}</div>
        <p>{hitter.get('strategy', '')}</p>
        <div class="alert">[ZONA] {hitter.get('chase_warning', '')}</div>
      </div>
      <div class="plan-card">
        <h4>3. Putaway (0-2, 1-2, 2-2, 3-2)</h4>
        <div class="focus">Foco: {two_s.get('dominant_pitch', 'Breaking Ball')}</div>
        <p>{two_s.get('strategy', '')}</p>
        <div class="alert">[2-STRIKES] {two_s.get('chase_warning', '')}</div>
      </div>
    </div>

    <div class="footer-note">
      <span>MLB Statcast Multimodal AI • Databricks Serverless Model Serving</span>
      <span>Uso Exclusivo no Dugout & Comissão Técnica</span>
    </div>
  </div>
</body>
</html>"""
    return html_content
