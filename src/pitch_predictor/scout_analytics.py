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


def render_count_matrix_html(predictions_by_count: dict[str, str]) -> str:
    """
    Render the 4x3 MLB Tactical Count Matrix HTML (Balls 0..3 rows x Strikes 0..2 columns).

    Strictly CommonMark-safe (single line, no 4-space indentation, no HTML comments).
    """
    header_cells = [
        '<div style="background:#0F172A;border:1px solid #26354A;border-radius:8px;padding:8px;text-align:center;font-size:11px;font-weight:800;color:#94A3B8;">BALLS \\ STRIKES</div>',
        '<div style="background:#0F172A;border:1px solid #26354A;border-radius:8px;padding:8px;text-align:center;font-size:12px;font-weight:800;color:#38BDF8;">0 STRIKES</div>',
        '<div style="background:#0F172A;border:1px solid #26354A;border-radius:8px;padding:8px;text-align:center;font-size:12px;font-weight:800;color:#38BDF8;">1 STRIKE</div>',
        '<div style="background:#0F172A;border:1px solid #26354A;border-radius:8px;padding:8px;text-align:center;font-size:12px;font-weight:800;color:#F59E0B;">2 STRIKES (FINALIZAÇÃO)</div>',
    ]

    grid_items: list[str] = list(header_cells)

    for balls in range(4):
        grid_items.append(
            f'<div style="background:#0F172A;border:1px solid #26354A;border-radius:8px;padding:10px;display:flex;flex-direction:column;justify-content:center;align-items:center;">'
            f'<span style="font-size:13px;font-weight:900;color:#F8FAFC;">{balls} BALL{"S" if balls != 1 else ""}</span>'
            f'<span style="font-size:10px;color:#64748B;font-weight:700;">LINHA {balls}-X</span>'
            f"</div>"
        )
        for strikes in range(3):
            count_key = f"{balls}-{strikes}"
            pred_cat = predictions_by_count.get(count_key, "Fastball")
            style = PITCH_CATEGORY_STYLES.get(
                pred_cat, PITCH_CATEGORY_STYLES["Fastball"]
            )
            primary = style["primary"]
            secondary = style["secondary"]
            bg_tint = style["bg_tint"]
            icon = style["icon"]

            if count_key in PITCHER_AHEAD_COUNTS:
                lev_label = "Pitcher Ahead"
                lev_color = "#10B981"
            elif count_key in PITCHER_BEHIND_COUNTS:
                lev_label = "Pitcher Behind"
                lev_color = "#F59E0B"
            else:
                lev_label = "Neutral"
                lev_color = "#94A3B8"

            cell_html = (
                f'<div style="background:{bg_tint};border:2px solid {primary};border-radius:10px;padding:10px 12px;display:flex;flex-direction:column;justify-content:space-between;box-shadow:0 4px 12px rgba(0,0,0,0.25);">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
                f'<span style="background:#0B0F19;color:#F8FAFC;font-size:13px;font-weight:900;padding:2px 8px;border-radius:5px;border:1px solid {secondary};">{count_key}</span>'
                f'<span style="font-size:10px;font-weight:700;color:{lev_color};text-transform:uppercase;">{lev_label}</span>'
                f"</div>"
                f'<div style="font-size:15px;font-weight:900;color:#F8FAFC;margin:2px 0;">{icon} {pred_cat}</div>'
                f'<div style="font-size:10.5px;color:{primary};font-weight:700;">{style["arsenal"].split("•")[0].strip()}</div>'
                f"</div>"
            )
            grid_items.append(cell_html)

    raw_html = f"""
    <div style="background:#151E2E;border:1px solid #26354A;border-radius:12px;padding:14px;margin:8px 0;">
      <div style="display:grid;grid-template-columns:110px repeat(3,1fr);gap:10px;align-items:stretch;">
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
    1. MLB League Average (`#64748B` dashed)
    2. Pitcher Historical Prior (`#F59E0B` amber)
    3. AI Situational Matrix Call (`#38BDF8` neon cyan)

    Strictly CommonMark-safe via `_compact_html`.
    """
    l_fb, l_bb, l_os = _extract_dist_triplet(league_dist)
    p_fb, p_bb, p_os = _extract_dist_triplet(pitcher_dist)
    m_fb, m_bb, m_os = _extract_dist_triplet(model_dist)

    cx, cy = 190.0, 142.0
    max_r = 92.0
    # Angles in radians: Top (-90 deg), Bottom-Right (+30 deg), Bottom-Left (+150 deg)
    angles = (-math.pi / 2.0, math.pi / 6.0, 5.0 * math.pi / 6.0)

    def _pt(val: float, angle: float, scale: float = 1.0) -> str:
        # Scale 0..1 value onto radar radius (using 0.85 as full outer ring so 65%-80% stands out clearly)
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
      <svg viewBox="0 0 380 285" width="100%" style="max-width:390px;background:#0B0F19;border-radius:12px;border:1px solid #26354A;padding:8px;display:block;">
        <polygon points="{_ring_points(1.0)}" fill="#0F172A" stroke="#26354A" stroke-width="1.5" />
        <polygon points="{_ring_points(0.75)}" fill="none" stroke="#1E293B" stroke-width="1" stroke-dasharray="3,3" />
        <polygon points="{_ring_points(0.50)}" fill="none" stroke="#1E293B" stroke-width="1" stroke-dasharray="3,3" />
        <polygon points="{_ring_points(0.25)}" fill="none" stroke="#1E293B" stroke-width="1" stroke-dasharray="3,3" />
        <line x1="{cx}" y1="{cy}" x2="{top_x:.1f}" y2="{top_y:.1f}" stroke="#334155" stroke-width="1.2" />
        <line x1="{cx}" y1="{cy}" x2="{br_x:.1f}" y2="{br_y:.1f}" stroke="#334155" stroke-width="1.2" />
        <line x1="{cx}" y1="{cy}" x2="{bl_x:.1f}" y2="{bl_y:.1f}" stroke="#334155" stroke-width="1.2" />
        <polygon points="{league_pts}" fill="rgba(148,163,184,0.12)" stroke="#94A3B8" stroke-width="2" stroke-dasharray="4,3" />
        <polygon points="{pitcher_pts}" fill="rgba(245,158,11,0.18)" stroke="#F59E0B" stroke-width="2.4" />
        <polygon points="{model_pts}" fill="rgba(56,189,248,0.26)" stroke="#38BDF8" stroke-width="2.8" />
        <text x="{cx}" y="24" text-anchor="middle" fill="#EF4444" font-size="11.5" font-weight="800" font-family="sans-serif">⚡ Fastball ({m_fb * 100:.0f}%)</text>
        <text x="312" y="212" text-anchor="middle" fill="#3B82F6" font-size="11.5" font-weight="800" font-family="sans-serif">🌀 Breaking Ball ({m_bb * 100:.0f}%)</text>
        <text x="68" y="212" text-anchor="middle" fill="#10B981" font-size="11.5" font-weight="800" font-family="sans-serif">🛑 Offspeed ({m_os * 100:.0f}%)</text>
        <rect x="24" y="246" width="12" height="4" fill="#94A3B8" />
        <text x="42" y="251" fill="#CBD5E1" font-size="10" font-weight="600" font-family="sans-serif">Média MLB</text>
        <rect x="122" y="246" width="12" height="4" fill="#F59E0B" />
        <text x="140" y="251" fill="#F59E0B" font-size="10" font-weight="700" font-family="sans-serif">Prior Pitcher</text>
        <rect x="234" y="246" width="12" height="4" fill="#38BDF8" />
        <text x="252" y="251" fill="#38BDF8" font-size="10" font-weight="800" font-family="sans-serif">Matriz IA (12 Contagens)</text>
      </svg>
    </div>
    """
    return _compact_html(raw_svg)


def render_game_plan_cards_html(game_plan: dict[str, dict[str, str]]) -> str:
    """
    Render the 3 Batter Tactical Approach Cards (`early_counts`, `hitter_counts`,
    `two_strike_protection`) as a CommonMark-safe responsive HTML grid.
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

        card_html = (
            f'<div style="background:#151E2E;border:1px solid #26354A;border-top:4px solid {primary};border-radius:12px;padding:16px;display:flex;flex-direction:column;justify-content:space-between;">'
            f"<div>"
            f'<div style="font-size:13px;font-weight:800;color:#F8FAFC;margin-bottom:8px;">{title}</div>'
            f'<div style="background:{bg_tint};border-left:4px solid {primary};padding:8px 10px;border-radius:6px;margin-bottom:10px;">'
            f'<span style="font-size:10px;color:#94A3B8;display:block;font-weight:700;text-transform:uppercase;">FOCO PRINCIPAL DO PLANO</span>'
            f'<strong style="font-size:16px;color:#F8FAFC;">{icon} {dom_pitch}</strong>'
            f"</div>"
            f'<div style="font-size:11px;color:#38BDF8;font-weight:700;margin-bottom:8px;">{covered}</div>'
            f'<p style="font-size:12px;color:#CBD5E1;line-height:1.45;margin:0 0 10px 0;">{strategy}</p>'
            f"</div>"
            f'<div style="background:#0B0F19;border:1px solid #334155;border-radius:8px;padding:8px 10px;font-size:11px;color:#F59E0B;">'
            f"<strong>⚠️ Alerta de Disciplina:</strong> {warning}"
            f"</div>"
            f"</div>"
        )
        cards_markup.append(card_html)

    raw_html = f"""
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin:8px 0;">
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
