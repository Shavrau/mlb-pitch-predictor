"""
Unit tests for tactical MLB scenarios, At-Bat state progression, and SVG/HTML UI components (Red Phase TDD).

Covers:
1. `PITCHER_ARCHETYPES`: Minimum 4 archetypes (League Average/Média da Liga, Power Pitcher,
   Breaking Specialist, Offspeed Master) with priors summing to ~1.0.
2. `TACTICAL_SCENARIOS`: Every preset scenario produces a valid 24-column feature dictionary
   when passed through `derive_tactical_features` and `validate_inference_record`.
3. `advance_at_bat_state`: Official MLB count progression rules:
   - Strike ('S') on 0-0 -> 0-1 and prev_pitch_result='S'
   - Two-strike Foul ('Foul' on strikes=2) -> stays at strikes=2
   - Strike ('S') on strikes=2 -> signals 'Strikeout (K)' and resets count to 0-0
   - Ball ('B') on balls=3 -> signals 'Walk (BB)' and resets count to 0-0
4. `STATCAST_ZONES`: Zones 1-9 marked as 'Strike', zones 11-14 marked as 'Chase' (zone 10 excluded).
5. `render_baseball_diamond_svg`: Valid `<svg` markup and `#F59E0B` amber highlight when base is occupied.
6. `render_strike_zone_svg`: Valid `<svg` markup highlighting the selected Statcast zone.
7. `render_prediction_card_html`: Semantic colors for Fastball (`#EF4444`/`#DC2626`),
   Breaking Ball (`#3B82F6`/`#2563EB`), and Offspeed (`#10B981`/`#059669`).
"""

from __future__ import annotations

from typing import Any

import pytest

from pitch_predictor.scenarios import (
    PITCHER_ARCHETYPES,
    TACTICAL_SCENARIOS,
    advance_at_bat_state,
)
from pitch_predictor.serving_client import (
    derive_tactical_features,
    validate_inference_record,
)
from pitch_predictor.ui_components import (
    STATCAST_ZONES,
    render_baseball_diamond_svg,
    render_prediction_card_html,
    render_strike_zone_svg,
)

# ---------------------------------------------------------------------------
# 1. Pitcher Archetypes & Tactical Scenarios
# ---------------------------------------------------------------------------


def test_pitcher_archetypes_contains_required_profiles_and_valid_priors() -> None:
    """Verify PITCHER_ARCHETYPES has >= 4 profiles and each profile's priors sum to ~1.0."""
    assert isinstance(PITCHER_ARCHETYPES, dict)
    assert len(PITCHER_ARCHETYPES) >= 4

    joined_keys = " | ".join(PITCHER_ARCHETYPES.keys()).lower()
    assert "power pitcher" in joined_keys
    assert "breaking specialist" in joined_keys
    assert "offspeed master" in joined_keys
    assert ("league" in joined_keys) or ("média" in joined_keys) or ("media" in joined_keys)

    for name, profile in PITCHER_ARCHETYPES.items():
        fb = profile.get("pitcher_fb_prior", profile.get("fb", 0.0))
        bb = profile.get("pitcher_bb_prior", profile.get("bb", 0.0))
        os = profile.get("pitcher_os_prior", profile.get("os", 0.0))
        assert (fb + bb + os) == pytest.approx(
            1.0, abs=1e-3
        ), f"Archetype '{name}' priors do not sum to 1.0"


def test_tactical_scenarios_produce_valid_24_feature_records() -> None:
    """Verify all TACTICAL_SCENARIOS generate valid 24-column records via derive_tactical_features."""
    assert len(TACTICAL_SCENARIOS) > 0

    scenarios_iter = (
        TACTICAL_SCENARIOS.values()
        if isinstance(TACTICAL_SCENARIOS, dict)
        else TACTICAL_SCENARIOS
    )

    for scenario in scenarios_iter:
        raw_inputs = scenario.get("inputs", scenario) if isinstance(scenario, dict) else scenario
        features = derive_tactical_features(raw_inputs)
        assert validate_inference_record(features) is True


# ---------------------------------------------------------------------------
# 2. Official MLB At-Bat State Progression
# ---------------------------------------------------------------------------


def _initial_at_bat_state(**overrides: Any) -> dict[str, Any]:
    """Create a default 0-0 First Pitch state dictionary for testing at-bat transitions."""
    state: dict[str, Any] = {
        "prev_pitch_category": "First Pitch",
        "prev_pitch_type": "None",
        "prev_pitch_zone": 0,
        "prev_pitch_result": "First Pitch",
        "p_throws": "R",
        "stand": "L",
        "balls": 0,
        "strikes": 0,
        "outs_when_up": 1,
        "inning": 7,
        "inning_topbot": "Top",
        "score_diff": 1,
        "on_1b_flag": 0,
        "on_2b_flag": 0,
        "on_3b_flag": 0,
        "n_thruorder_pitcher": 2,
        "pitcher_fb_prior": 0.5543,
        "pitcher_bb_prior": 0.3053,
        "pitcher_os_prior": 0.1404,
    }
    state.update(overrides)
    return state


def test_advance_at_bat_state_strike_on_0_0_advances_to_0_1() -> None:
    """On outcome='S' from 0-0, count advances to balls=0, strikes=1, prev_pitch_result='S'."""
    state = _initial_at_bat_state(balls=0, strikes=0)
    next_state = advance_at_bat_state(
        current_state=state,
        pitch_category="Fastball",
        pitch_type="FF",
        pitch_zone=5,
        outcome="S",
    )

    assert next_state["balls"] == 0
    assert next_state["strikes"] == 1
    assert next_state["prev_pitch_category"] == "Fastball"
    assert next_state["prev_pitch_type"] == "FF"
    assert next_state["prev_pitch_zone"] == 5
    assert next_state["prev_pitch_result"] == "S"


def test_advance_at_bat_state_foul_with_two_strikes_keeps_two_strikes() -> None:
    """Official baseball rule: outcome='Foul' with strikes=2 keeps strikes=2 without strikeout."""
    state = _initial_at_bat_state(balls=1, strikes=2)
    next_state = advance_at_bat_state(
        current_state=state,
        pitch_category="Fastball",
        pitch_type="FF",
        pitch_zone=2,
        outcome="Foul",
    )

    assert next_state["balls"] == 1
    assert next_state["strikes"] == 2
    assert next_state["prev_pitch_result"] == "S"


def test_advance_at_bat_state_strike_with_two_strikes_signals_strikeout_and_resets() -> None:
    """On outcome='S' with strikes=2, signals 'Strikeout (K)' and resets count to 0-0."""
    state = _initial_at_bat_state(balls=1, strikes=2, outs_when_up=1)
    next_state = advance_at_bat_state(
        current_state=state,
        pitch_category="Breaking Ball",
        pitch_type="SL",
        pitch_zone=14,
        outcome="S",
    )

    assert next_state["balls"] == 0
    assert next_state["strikes"] == 0
    assert next_state["prev_pitch_category"] == "First Pitch"
    event_signal = next_state.get("at_bat_event") or next_state.get("event") or ""
    assert "Strikeout (K)" in event_signal


def test_advance_at_bat_state_ball_with_three_balls_signals_walk_and_resets() -> None:
    """On outcome='B' with balls=3, signals 'Walk (BB)' and resets count to 0-0."""
    state = _initial_at_bat_state(balls=3, strikes=1)
    next_state = advance_at_bat_state(
        current_state=state,
        pitch_category="Offspeed",
        pitch_type="CH",
        pitch_zone=13,
        outcome="B",
    )

    assert next_state["balls"] == 0
    assert next_state["strikes"] == 0
    assert next_state["prev_pitch_category"] == "First Pitch"
    event_signal = next_state.get("at_bat_event") or next_state.get("event") or ""
    assert "Walk (BB)" in event_signal


# ---------------------------------------------------------------------------
# 3. Statcast Zones & SVG / HTML Tactical Visual Components
# ---------------------------------------------------------------------------


def test_statcast_zones_strike_and_chase_classification() -> None:
    """Verify zones 1-9 are marked as 'Strike' and zones 11-14 are marked as 'Chase'."""
    for zone_id in range(1, 10):
        assert zone_id in STATCAST_ZONES
        assert "Strike" in str(STATCAST_ZONES[zone_id])

    for zone_id in range(11, 15):
        assert zone_id in STATCAST_ZONES
        assert "Chase" in str(STATCAST_ZONES[zone_id])

    assert 10 not in STATCAST_ZONES


def test_render_baseball_diamond_svg_highlights_occupied_bases() -> None:
    """Verify render_baseball_diamond_svg returns valid SVG and highlights occupied base with #F59E0B."""
    svg_empty = render_baseball_diamond_svg(on_1b=0, on_2b=0, on_3b=0, outs=0)
    assert "<svg" in svg_empty

    svg_runner_on_2b = render_baseball_diamond_svg(on_1b=0, on_2b=1, on_3b=0, outs=2)
    assert "<svg" in svg_runner_on_2b
    assert "#F59E0B" in svg_runner_on_2b.upper()


def test_render_strike_zone_svg_renders_svg_and_highlights_selected_zone() -> None:
    """Verify render_strike_zone_svg returns valid SVG markup and highlights the selected zone."""
    svg_zone_5 = render_strike_zone_svg(selected_zone=5, prev_category="Fastball")
    assert "<svg" in svg_zone_5
    assert "5" in svg_zone_5


@pytest.mark.parametrize(
    ("prediction", "expected_colors"),
    [
        ("Fastball", ("#EF4444", "#DC2626")),
        ("Breaking Ball", ("#3B82F6", "#2563EB")),
        ("Offspeed", ("#10B981", "#059669")),
    ],
)
def test_render_prediction_card_html_semantic_colors(
    prediction: str, expected_colors: tuple[str, str]
) -> None:
    """Verify render_prediction_card_html includes category name and semantic hex color."""
    features = derive_tactical_features(_initial_at_bat_state(balls=0, strikes=2))
    html = render_prediction_card_html(
        prediction=prediction,
        features=features,
        latency_ms=42.5,
    )

    assert prediction in html
    html_upper = html.upper()
    assert any(color in html_upper for color in expected_colors)


def test_all_html_and_svg_renderers_are_commonmark_safe_single_line() -> None:
    """
    Regression test: ensure no HTML/SVG renderer contains newlines or 4-space indentation
    that would cause Streamlit's CommonMark parser to emit <pre><code> blocks.
    """
    from pitch_predictor.ui_components import (
        get_custom_css,
        render_pitcher_photo_html,
        render_pixel_pitcher_svg,
        render_scoreboard_banner_html,
    )

    features = derive_tactical_features(_initial_at_bat_state(balls=1, strikes=2))
    from pitch_predictor.ui_components import (
        render_baseball_seams_divider_svg,
        render_dugout_clipboard_frame_html,
        render_spin_axis_clock_svg,
    )

    fragments = [
        render_baseball_diamond_svg(1, 1, 0, outs=1),
        render_strike_zone_svg(selected_zone=14, prev_category="Breaking Ball"),
        render_prediction_card_html("Fastball", features, latency_ms=105.0),
        render_scoreboard_banner_html(features, "dev_rhian_souza_pitch_predictor_endpoint_dev"),
        render_pixel_pitcher_svg("Paul Skenes"),
        render_pixel_pitcher_svg("Corbin Burnes"),
        render_pixel_pitcher_svg("Logan Webb"),
        render_pixel_pitcher_svg("Chris Sale"),
        render_pixel_pitcher_svg("Yoshinobu Yamamoto"),
        render_pitcher_photo_html("Paul Skenes"),
        render_pitcher_photo_html("Corbin Burnes"),
        render_pitcher_photo_html("Logan Webb"),
        render_pitcher_photo_html("Chris Sale"),
        render_pitcher_photo_html("Yoshinobu Yamamoto"),
        render_baseball_seams_divider_svg(width_px=600),
        render_spin_axis_clock_svg(spin_axis="1:15", spin_rpm=2480, pitch_name="4-Seam Fastball"),
        render_dugout_clipboard_frame_html(title="Plano de Jogo", content_html="<p>Teste</p>"),
        get_custom_css(),
        get_custom_css(theme_mode="ballpark"),
    ]
    for frag in fragments:
        assert "\n" not in frag, "HTML/SVG fragment must not contain newlines (CommonMark <pre><code> hazard)"
        assert "    " not in frag, "HTML/SVG fragment must not contain 4-space indentation"
        assert "<!--" not in frag, "HTML/SVG fragment must not contain HTML comments"


def test_render_pitcher_photo_html_loads_official_headshots() -> None:
    """Verify render_pitcher_photo_html renders real photos for all 5 MLB stars."""
    from pitch_predictor.ui_components import render_pitcher_photo_html

    pitchers = ["Paul Skenes", "Corbin Burnes", "Logan Webb", "Chris Sale", "Yoshinobu Yamamoto"]
    for p_name in pitchers:
        html = render_pitcher_photo_html(p_name)
        assert "<img" in html
        assert f'alt="{p_name}"' in html
        assert ("data:image/png;base64," in html) or ("https://img.mlbstatic.com" in html)
        assert "\n" not in html



def test_render_pitch_flight_arena_svg_renders_animated_trajectory_and_ghost_tunnel() -> None:
    """Verify render_pitch_flight_arena_svg renders 3D animated trajectory and ghost AI comparison."""
    from pitch_predictor.ui_components import render_pitch_flight_arena_svg

    for cat in ("Fastball", "Breaking Ball", "Offspeed"):
        svg = render_pitch_flight_arena_svg(
            pitch_category=cat,
            pitch_name=f"Test {cat}",
            speed_mph="98 mph",
            target_zone=5,
            p_throws="R",
            ai_category="Fastball" if cat != "Fastball" else "Breaking Ball",
            result_badge="WHIFF!",
            is_success=True,
        )
        assert "<svg" in svg
        assert "<animateMotion" in svg
        assert "STATCAST RADAR" in svg
        assert "IA ESPERAVA:" in svg
        assert "\n" not in svg
        assert "<!--" not in svg



# ---------------------------------------------------------------------------
# Ballpark MLB Authentic Theme Components
# ---------------------------------------------------------------------------


def test_resolve_ui_theme_mode_supports_ballpark_dark_light() -> None:
    """Verify resolve_ui_theme_mode resolves ballpark, light, and dark appropriately."""
    from pitch_predictor.ui_components import resolve_ui_theme_mode

    assert resolve_ui_theme_mode("ballpark") == "ballpark"
    assert resolve_ui_theme_mode("⚾ Ballpark MLB") == "ballpark"
    assert resolve_ui_theme_mode("baseball") == "ballpark"
    assert resolve_ui_theme_mode("light") == "light"
    assert resolve_ui_theme_mode("Light Editorial") == "light"
    assert resolve_ui_theme_mode("dark") == "dark"
    assert resolve_ui_theme_mode("Dark Broadcast") == "dark"
    assert resolve_ui_theme_mode(None) in ("ballpark", "dark", "light")


def test_get_custom_css_ballpark_palette() -> None:
    """Verify get_custom_css in ballpark mode injects authentic MLB colors and styles."""
    from pitch_predictor.ui_components import get_custom_css

    css = get_custom_css(theme_mode="ballpark")
    assert "--pp-grass: #1B4332;" in css
    assert "--pp-clay: #C85A17;" in css
    assert "--pp-seam: #DC2626;" in css
    assert "--pp-chalk: #F8FAFC;" in css
    assert "--pp-leather: #78350F;" in css
    assert "--pp-bg: #08120C;" in css
    assert ".pp-seam-divider" in css
    assert ".pp-dugout-clipboard" in css


def test_render_baseball_seams_divider_svg() -> None:
    """Verify render_baseball_seams_divider_svg returns valid SVG with Rawlings red V-stitches."""
    from pitch_predictor.ui_components import render_baseball_seams_divider_svg

    for mode in ("ballpark", "dark", "light"):
        svg = render_baseball_seams_divider_svg(width_px=650, theme_mode=mode)
        assert "<svg" in svg
        assert "#DC2626" in svg
        assert chr(10) not in svg
        assert "<!--" not in svg


def test_render_spin_axis_clock_svg() -> None:
    """Verify render_spin_axis_clock_svg renders 12-hour clock face and telemetry cards."""
    from pitch_predictor.ui_components import render_spin_axis_clock_svg

    for axis in ("1:15", "10:30", "6:00", "12:00", "7:45"):
        svg = render_spin_axis_clock_svg(
            spin_axis=axis,
            spin_rpm=2450,
            pitch_name="4-Seam (FF)",
            category="Fastball",
            theme_mode="ballpark",
        )
        assert "<svg" in svg
        assert "12:00" in svg
        assert "3:00" in svg
        assert "6:00" in svg
        assert "9:00" in svg
        assert f"{axis} TILT" in svg
        assert "2,450 RPM" in svg
        assert "EFEITO MAGNUS" in svg
        assert chr(10) not in svg
        assert "<!--" not in svg


def test_render_dugout_clipboard_frame_html() -> None:
    """Verify render_dugout_clipboard_frame_html wraps content with clip, rivets, and stadium header."""
    from pitch_predictor.ui_components import render_dugout_clipboard_frame_html

    for mode in ("ballpark", "dark", "light"):
        html = render_dugout_clipboard_frame_html(
            title="GAME PLAN • DOIS STRIKES",
            content_html="<div class='test-plan'>Plano de ataque</div>",
            stadium="ESTÁDIO: PNC PARK",
            theme_mode=mode,
        )
        assert "MLB DUGOUT TACTICAL CARD" in html
        assert "GAME PLAN • DOIS STRIKES" in html
        assert "ESTÁDIO: PNC PARK" in html
        assert "test-plan" in html
        assert chr(10) not in html
        assert "<!--" not in html