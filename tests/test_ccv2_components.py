"""
Unit and headless AppTest integration tests for Broadcast Editorial Dual-Mode Design System v2.0
and Streamlit Custom Components v2 (`src/pitch_predictor/ccv2_components.py`).
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from pitch_predictor.scenarios import MLB_PITCHERS
from pitch_predictor.scout_analytics import (
    render_count_matrix_html,
    render_game_plan_cards_html,
    render_radar_chart_svg,
)
from pitch_predictor.serving_client import MLB_LEAGUE_PRIORS, derive_tactical_features
from pitch_predictor.ui_components import (
    PITCH_CATEGORY_STYLES,
    get_custom_css,
    render_prediction_card_html,
    render_scoreboard_banner_html,
)

DECORATIVE_EMOJIS = ("⚡", "🌀", "🛑", "⚾", "⚠️", "💡", "🔮", "🚀", "🌪️")


def test_mlb_pitchers_have_official_franchise_broadcast_metadata() -> None:
    """Verify all 5 canonical MLB pitchers include team_abbr, jersey_number, and official franchise hex colors."""
    expected_abbrs = {
        "Paul Skenes": "PIT",
        "Corbin Burnes": "BAL",
        "Logan Webb": "SF",
        "Chris Sale": "ATL",
        "Yoshinobu Yamamoto": "LAD",
    }
    for name, expected_abbr in expected_abbrs.items():
        profile = MLB_PITCHERS[name]
        assert profile.get("team_abbr") == expected_abbr
        assert str(profile.get("jersey_number", "")).startswith("#")
        primary_hex = str(profile.get("team_primary_hex", ""))
        secondary_hex = str(profile.get("team_secondary_hex", ""))
        assert primary_hex.startswith("#") and len(primary_hex) == 7
        assert secondary_hex.startswith("#") and len(secondary_hex) == 7


def test_dual_mode_css_defaults_to_dark_broadcast_and_supports_light_editorial() -> None:
    """Verify get_custom_css supports Dual Mode with Dark Broadcast (#0A0B0E) default and Light Editorial (#F8FAFC)."""
    from pitch_predictor.ui_components import resolve_ui_theme_mode

    assert resolve_ui_theme_mode(None) == "dark"
    assert resolve_ui_theme_mode("light") == "light"
    assert resolve_ui_theme_mode("Light Editorial") == "light"
    assert resolve_ui_theme_mode("Dark Broadcast") == "dark"

    dark_css = get_custom_css("dark")
    light_css = get_custom_css("light")

    assert "#0A0B0E" in dark_css.upper()
    assert "--PP-BG" in dark_css.upper()
    assert "TABULAR-NUMS" in dark_css.upper()

    assert "#E6ECF2" in light_css.upper()
    assert "#F1F5F9" in light_css.upper()
    assert "#0F172A" in light_css.upper()
    assert "STEXPANDER" in light_css.upper()
    assert "PP-EDITORIAL-CARD" in light_css.upper()


def test_analytical_components_are_free_of_decorative_ai_emojis() -> None:
    """Verify analytical cards, matrix, radar, and styles contain zero decorative emojis."""
    for cat, style in PITCH_CATEGORY_STYLES.items():
        assert "code_badge" in style, f"Missing Statcast code_badge on {cat}"
        for emoji in DECORATIVE_EMOJIS:
            assert emoji not in style.get("icon", "")
            assert emoji not in style.get("label_pt", "")

    features = derive_tactical_features(
        {
            "prev_pitch_category": "Fastball",
            "prev_pitch_type": "FF",
            "prev_pitch_zone": 2,
            "prev_pitch_result": "S",
            "balls": 0,
            "strikes": 2,
            "p_throws": "R",
            "stand": "L",
        }
    )
    preds_12 = {f"{b}-{s}": "Fastball" for b in range(4) for s in range(3)}
    preds_12["0-2"] = "Breaking Ball"
    preds_12["2-1"] = "Offspeed"

    game_plan = {
        "early_counts": {
            "title": "1. Abertura & Contagens Neutras",
            "dominant_pitch": "Fastball",
            "counts_covered": "0-0: Fastball",
            "strategy": "Ataque direto.",
            "chase_warning": "Disciplina baixa.",
        },
        "hitter_counts": {
            "title": "2. Vantagem do Rebatedor",
            "dominant_pitch": "Fastball",
            "counts_covered": "2-0: Fastball",
            "strategy": "Zona central.",
            "chase_warning": "Aceite BB.",
        },
        "two_strike_protection": {
            "title": "3. Proteção com 2 Strikes",
            "dominant_pitch": "Breaking Ball",
            "counts_covered": "0-2: Breaking Ball",
            "strategy": "Encurte swing.",
            "chase_warning": "Proteja bordas.",
        },
    }

    rendered_markups = [
        render_prediction_card_html("Breaking Ball", features, latency_ms=34.2),
        render_scoreboard_banner_html(features, "dev_endpoint"),
        render_count_matrix_html(preds_12),
        render_radar_chart_svg(MLB_LEAGUE_PRIORS, MLB_LEAGUE_PRIORS, MLB_LEAGUE_PRIORS),
        render_game_plan_cards_html(game_plan),
    ]

    for markup in rendered_markups:
        for emoji in DECORATIVE_EMOJIS:
            assert emoji not in markup, f"Found banned decorative emoji '{emoji}' in analytical UI markup"


def test_ccv2_count_inspection_detail_and_component_mount() -> None:
    """Verify CCv2 interactive count matrix and strike zone mount cleanly in AppTest and return valid state."""
    from pitch_predictor.ccv2_components import build_count_inspection_detail

    preds_12 = {f"{b}-{s}": ("Breaking Ball" if s == 2 else "Fastball") for b in range(4) for s in range(3)}
    detail = build_count_inspection_detail(
        count_key="0-2",
        predictions_by_count=preds_12,
        priors={"pitcher_fb_prior": 0.55, "pitcher_bb_prior": 0.31, "pitcher_os_prior": 0.14},
        stand="L",
        risp_active=True,
    )
    assert detail["count_key"] == "0-2"
    assert detail["predicted_category"] == "Breaking Ball"
    assert detail["leverage"] == "Pitcher Ahead"
    assert len(detail["recommended_zones"]) >= 1
    assert len(detail["tactical_note"]) > 15

    def _app_script() -> None:
        import streamlit as st

        from pitch_predictor.ccv2_components import (
            interactive_count_matrix_ccv2,
            interactive_strike_zone_ccv2,
        )

        p_map = {f"{b}-{s}": "Fastball" for b in range(4) for s in range(3)}
        sel_count = interactive_count_matrix_ccv2(
            predictions_by_count=p_map,
            initial_count="1-2",
            accent_hex="#FDB827",
            key="ccv2_matrix_test",
        )
        sel_zone = interactive_strike_zone_ccv2(
            selected_zone=14,
            pitch_category="Breaking Ball",
            accent_hex="#FDB827",
            key="ccv2_zone_test",
        )
        st.write(f"SELECTED_COUNT={sel_count}")
        st.write(f"SELECTED_ZONE={sel_zone}")

    at = AppTest.from_function(_app_script).run()
    assert not at.exception, f"CCv2 AppTest raised exception: {at.exception}"
    combined = "\n".join(m.value for m in at.markdown)
    assert "SELECTED_COUNT=1-2" in combined
    assert "SELECTED_ZONE=14" in combined


def test_light_mode_count_matrix_and_strike_zone_contrast() -> None:
    """Verify Light Mode contrast improvements: AHEAD/BEHIND/EVEN tags, #94A3B8 1.5px grid, #D97706 reticle."""
    from pitch_predictor.ui_components import render_strike_zone_svg
    from pitch_predictor.scout_analytics import render_count_matrix_html

    # 1. Strike Zone SVG
    sz_light = render_strike_zone_svg(selected_zone=5, prev_category="Fastball", theme_mode="light")
    assert "#94A3B8" in sz_light
    assert 'stroke-width="1.5"' in sz_light
    assert "#D97706" in sz_light  # Stadium amber target reticle

    sz_dark = render_strike_zone_svg(selected_zone=5, prev_category="Fastball", theme_mode="dark")
    assert "#D97706" in sz_dark

    # 2. 4x3 Tactical Count Matrix HTML
    p_map = {f"{b}-{s}": ("Breaking Ball" if s == 2 else "Fastball") for b in range(4) for s in range(3)}
    p_map["2-0"] = "Offspeed"
    matrix_light = render_count_matrix_html(p_map, theme_mode="light", active_count="0-2")
    assert "AHEAD" in matrix_light
    assert "BEHIND" in matrix_light
    assert "EVEN" in matrix_light
    assert "#475569" in matrix_light  # Crisp --pp-muted in Light Mode
    assert "box-shadow:" in matrix_light
