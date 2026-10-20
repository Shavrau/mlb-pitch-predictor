"""
Headless UI automation tests for `app.py` using `streamlit.testing.v1.AppTest`.

Covers:
1. Default Landing Page (`pages/00_home_apresentacao.py` — Visão Geral & Apresentação Descomplicada):
   - Initial render of `app.py` loads the Home Presentation page without exceptions (`not at.exception`).
   - Presence of the 400ms challenge hero & impact metrics (`0.40`, `160 km/h`), the 3 pitch family cards
     (`Bola Rápida` / `Fastball`, `Bola com Efeito` / `Breaking Ball`, `Mudança de Velocidade` / `Offspeed`),
     and the Interactive Quick Demo controls (`home_pitcher_style`, `home_game_moment`, action button).
   - Quick Demo prediction flow with mocked `PitchServingClient.predict` returning `"Breaking Ball"`,
     rendering latency (`42.1 ms`) and a natural-language tactical explanation in Portuguese, plus
     graceful `ConnectionError` handling (`st.error`).
2. Page 1 (`pages/01_live_predictor.py` — Simulador Tático Ao Vivo):
   - Rendering of core SVG/HTML components (Baseball Diamond SVG, Statcast Strike Zone SVG, Scoreboard).
   - Widget interactions: scenario preset switching, pitcher archetype switching, base checkboxes & quick buttons,
     count radio buttons, Statcast zone buttons, prediction button, and Pitch-by-Pitch Tracker state machine.
3. Page 2 (`pages/02_scout_report.py` — MLB Scouting Report & Matriz):
   - Multipage navigation, 4x3 Count Matrix (12 counts), inline SVG Radar Chart, Batter Game Plan cards,
     and 12-count batch inference (`predict_batch`).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from pitch_predictor.serving_client import PitchServingClient, derive_tactical_features

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
MINIGAME_PAGE_PATH = str(Path(__file__).resolve().parent.parent / "pages" / "03_minigame.py")


# ---------------------------------------------------------------------------
# 1. Default Home Presentation Page (pages/00_home_apresentacao.py)
# ---------------------------------------------------------------------------


def test_app_initial_render_loads_home_presentation_page() -> None:
    """Verify `app.py` loads the Home Presentation page by default with hero, pitch cards, and quick demo."""
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not at.exception, f"App raised unexpected exception: {at.exception}"

    all_markdown = "\n".join(m.value for m in at.markdown)

    # 1. Hero title / subtitle of the presentation page
    assert "MLB Pitch Predictor" in all_markdown or "MLB PITCH PREDICTOR" in all_markdown.upper()
    assert (
        "Decodificando o Arremesso com Inteligência Artificial" in all_markdown
        or "Desafio dos 400 Milissegundos" in all_markdown
        or "400" in all_markdown
    )

    # 2. Impact metrics (0.40s / 0.4 segundos and 160 km/h)
    assert "0.40" in all_markdown or "0.4" in all_markdown
    assert "160 km/h" in all_markdown

    # 3. Three pitch family explanation cards
    assert "Fastball" in all_markdown and "Bola Rápida" in all_markdown
    assert "Breaking Ball" in all_markdown and (
        "Bola com Efeito" in all_markdown or "Bola de Efeito" in all_markdown
    )
    assert "Offspeed" in all_markdown and "Mudança de Velocidade" in all_markdown

    # 4. Quick Demo interactive selectors and action button
    style_select = at.selectbox(key="home_pitcher_style")
    moment_select = at.selectbox(key="home_game_moment")
    assert len(style_select.options) >= 4
    assert len(moment_select.options) >= 3

    demo_buttons = [
        b
        for b in at.button
        if b.key == "btn_home_quick_predict"
        or "Inteligência Artificial" in b.label
        or "Consultar" in b.label
    ]
    assert len(demo_buttons) >= 1


def test_home_quick_demo_prediction_flow_and_explanation() -> None:
    """Verify the Quick Demo on the Home page calls PitchServingClient.predict, explains in Portuguese, and handles errors."""
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not at.exception

    # Select "Rebatedor Contra a Parede" (or 0-2 / 2 strikes scenario) in the quick demo
    moment_select = at.selectbox(key="home_game_moment")
    target_moment = next(
        (
            opt
            for opt in moment_select.options
            if "Rebatedor" in opt or "Parede" in opt or "0-2" in opt or "2 Strikes" in opt
        ),
        moment_select.options[0],
    )
    moment_select.select(target_moment).run()
    assert not at.exception

    mock_features = derive_tactical_features(
        {
            "prev_pitch_category": "Fastball",
            "prev_pitch_type": "FF",
            "prev_pitch_zone": 2,
            "prev_pitch_result": "S",
            "balls": 0,
            "strikes": 2,
            "p_throws": "R",
            "stand": "R",
        }
    )
    mock_return = {
        "status": "success",
        "prediction": "Breaking Ball",
        "latency_ms": 42.1,
        "features": mock_features,
        "endpoint_name": "dev_rhian_souza_pitch_predictor_endpoint_dev",
        "attempts": 1,
    }

    with patch.object(PitchServingClient, "predict", return_value=mock_return) as mock_pred:
        demo_btn = next(
            b
            for b in at.button
            if b.key == "btn_home_quick_predict"
            or "Inteligência Artificial" in b.label
            or "Consultar" in b.label
        )
        demo_btn.click().run()

    assert not at.exception
    assert mock_pred.call_count == 1

    all_md = "\n".join(m.value for m in at.markdown)
    assert "Breaking Ball" in all_md
    assert "42.1 ms" in all_md
    # Verify natural-language tactical explanation in Portuguese is rendered
    assert any(
        term in all_md.lower()
        for term in ("arremessador", "rebatedor", "efeito", "vantagem", "zona")
    )

    # Verify graceful error handling when PitchServingClient.predict raises ConnectionError
    with patch.object(
        PitchServingClient,
        "predict",
        side_effect=ConnectionError("Home demo endpoint connection error"),
    ):
        demo_btn = next(
            b
            for b in at.button
            if b.key == "btn_home_quick_predict"
            or "Inteligência Artificial" in b.label
            or "Consultar" in b.label
        )
        demo_btn.click().run()

    assert not at.exception
    assert len(at.error) >= 1
    assert "Home demo endpoint connection error" in at.error[0].value


# ---------------------------------------------------------------------------
# 2. Page 1 — Live Tactical Simulator (pages/01_live_predictor.py)
# ---------------------------------------------------------------------------


def test_app_initial_render_no_exceptions() -> None:
    """Verify `pages/01_live_predictor.py` renders without any unhandled exception and shows SVG components."""
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.switch_page("pages/01_live_predictor.py").run()
    assert not at.exception, f"Live predictor raised unexpected exception: {at.exception}"

    # Verify default session state initialized properly
    assert at.session_state["balls"] == 0
    assert at.session_state["strikes"] == 0
    assert at.session_state["prev_pitch_category"] == "First Pitch"
    assert at.session_state["prev_pitch_zone"] == 0

    # Verify SVGs and Scoreboard HTML are present in rendered markdown blocks
    all_markdown = "\n".join(m.value for m in at.markdown)
    assert "<svg" in all_markdown
    assert "MLB STATCAST • TACTICAL PITCH PREDICTOR" in all_markdown
    assert "BASES LIMPAS" in all_markdown


def test_app_scenario_and_archetype_switching() -> None:
    """Verify selecting preset scenarios and pitcher archetypes updates session state and widgets."""
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.switch_page("pages/01_live_predictor.py").run()
    assert not at.exception

    # Switch scenario to Full Count & Bases Loaded (3-2 High Leverage)
    scenario_select = at.selectbox(key="selected_scenario")
    scenario_select.select("🔥 Contagem Cheia & Bases Lotadas (3-2 High Leverage)").run()
    assert not at.exception

    assert at.session_state["balls"] == 3
    assert at.session_state["strikes"] == 2
    assert at.session_state["outs_when_up"] == 2
    assert at.session_state["on_1b_flag"] is True
    assert at.session_state["on_2b_flag"] is True
    assert at.session_state["on_3b_flag"] is True
    assert at.session_state["prev_pitch_category"] == "Breaking Ball"
    assert at.session_state["prev_pitch_zone"] == 14

    # Verify SVG diamond reflects RISP ATIVO and #F59E0B
    all_markdown = "\n".join(m.value for m in at.markdown)
    assert "RISP ATIVO" in all_markdown
    assert "#F59E0B" in all_markdown.upper()

    # Switch pitcher archetype to Left-Handed Crafty Ace
    arch_select = at.selectbox(key="selected_archetype")
    arch_select.select("Lefty Crafty Ace (Chris Sale / Blake Snell)").run()
    assert not at.exception
    assert at.session_state["p_throws"] == "L"
    assert at.session_state["fb_pct_slider"] == 48.0


def test_app_bases_checkboxes_count_and_statcast_zone_buttons() -> None:
    """Verify interactive base checkboxes, count radios, and clickable Statcast zone matrix."""
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.switch_page("pages/01_live_predictor.py").run()
    assert not at.exception

    # Check 2nd base checkbox -> should activate RISP
    at.checkbox(key="on_2b_flag").check().run()
    assert not at.exception
    assert at.session_state["on_2b_flag"] is True
    all_md = "\n".join(m.value for m in at.markdown)
    assert "RISP ATIVO" in all_md

    # Click '🔥 Lotar Bases' button
    load_bases_btn = next(b for b in at.button if "Lotar Bases" in b.label)
    load_bases_btn.click().run()
    assert not at.exception
    assert at.session_state["on_1b_flag"] is True
    assert at.session_state["on_2b_flag"] is True
    assert at.session_state["on_3b_flag"] is True

    # Click '🧹 Limpar Bases' button
    clear_bases_btn = next(b for b in at.button if "Limpar Bases" in b.label)
    clear_bases_btn.click().run()
    assert not at.exception
    assert at.session_state["on_1b_flag"] is False
    assert at.session_state["on_2b_flag"] is False
    assert at.session_state["on_3b_flag"] is False

    # Change balls and strikes via radio buttons
    at.radio(key="balls").set_value(1).run()
    at.radio(key="strikes").set_value(2).run()
    assert not at.exception
    assert at.session_state["balls"] == 1
    assert at.session_state["strikes"] == 2

    # Click Zone 5 button in Statcast matrix (from First Pitch -> transitions to Fastball in Zone 5)
    at.button(key="btn_z5").click().run()
    assert not at.exception
    assert at.session_state["prev_pitch_zone"] == 5
    assert at.session_state["prev_pitch_category"] == "Fastball"

    # Click Zone 14 (Chase corner)
    at.button(key="btn_z14").click().run()
    assert not at.exception
    assert at.session_state["prev_pitch_zone"] == 14

    # Click Zone 0 (First Pitch reset)
    at.button(key="btn_z0").click().run()
    assert not at.exception
    assert at.session_state["prev_pitch_zone"] == 0
    assert at.session_state["prev_pitch_category"] == "First Pitch"


def test_app_predict_button_renders_prediction_card_and_handles_errors() -> None:
    """Verify clicking the prediction button calls PitchServingClient.predict and renders the card."""
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.switch_page("pages/01_live_predictor.py").run()
    assert not at.exception

    mock_features = derive_tactical_features(
        {
            "prev_pitch_category": "First Pitch",
            "balls": 0,
            "strikes": 0,
            "p_throws": "R",
            "stand": "L",
        }
    )
    mock_return = {
        "status": "success",
        "prediction": "Fastball",
        "latency_ms": 38.4,
        "features": mock_features,
        "endpoint_name": "dev_rhian_souza_pitch_predictor_endpoint_dev",
        "attempts": 1,
    }

    with patch.object(PitchServingClient, "predict", return_value=mock_return) as mock_pred:
        predict_btn = next(b for b in at.button if "Prever Próximo Arremesso" in b.label)
        predict_btn.click().run()

    assert not at.exception
    assert mock_pred.call_count == 1
    assert at.session_state["last_prediction_result"]["prediction"] == "Fastball"

    all_md = "\n".join(m.value for m in at.markdown)
    assert "PREDIÇÃO TÁTICA • STATCAST AI" in all_md
    assert "Bola Rápida (Fastball)" in all_md
    assert "#EF4444" in all_md.upper()

    # Now verify graceful error handling when endpoint raises ConnectionError
    with patch.object(
        PitchServingClient,
        "predict",
        side_effect=ConnectionError("Simulated endpoint timeout"),
    ):
        predict_btn = next(b for b in at.button if "Prever Próximo Arremesso" in b.label)
        predict_btn.click().run()

    assert not at.exception
    assert len(at.error) >= 1
    assert "Simulated endpoint timeout" in at.error[0].value


def test_app_sequential_pitch_by_pitch_tracker_mode() -> None:
    """Verify the Pitch-by-Pitch Tracker state machine inside the Streamlit UI."""
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.switch_page("pages/01_live_predictor.py").run()
    assert not at.exception

    # Initial count is 0-0
    assert at.session_state["balls"] == 0
    assert at.session_state["strikes"] == 0

    # 1. Click '🔴 Strike' -> count becomes 0-1
    strike_btn = next(b for b in at.button if "🔴 Strike" in b.label)
    strike_btn.click().run()
    assert not at.exception
    assert at.session_state["balls"] == 0
    assert at.session_state["strikes"] == 1
    assert len(at.session_state["at_bat_history"]) == 1

    # 2. Click '🔴 Strike' again -> count becomes 0-2
    strike_btn = next(b for b in at.button if "🔴 Strike" in b.label)
    strike_btn.click().run()
    assert not at.exception
    assert at.session_state["balls"] == 0
    assert at.session_state["strikes"] == 2
    assert len(at.session_state["at_bat_history"]) == 2

    # 3. Click '🟡 Foul' on 0-2 -> count MUST stay 0-2 (2-strike foul rule)
    foul_btn = next(b for b in at.button if "🟡 Foul" in b.label)
    foul_btn.click().run()
    assert not at.exception
    assert at.session_state["balls"] == 0
    assert at.session_state["strikes"] == 2
    assert len(at.session_state["at_bat_history"]) == 3

    # 4. Click '🟢 Ball' -> count becomes 1-2
    ball_btn = next(b for b in at.button if "🟢 Ball" in b.label)
    ball_btn.click().run()
    assert not at.exception
    assert at.session_state["balls"] == 1
    assert at.session_state["strikes"] == 2

    # 5. Click '🔴 Strike' on 1-2 -> Strikeout (K)! Resets count to 0-0 and increments outs
    strike_btn = next(b for b in at.button if "🔴 Strike" in b.label)
    strike_btn.click().run()
    assert not at.exception
    assert at.session_state["balls"] == 0
    assert at.session_state["strikes"] == 0
    assert at.session_state["outs_when_up"] == 1
    assert "Strikeout (K)" in at.session_state["last_at_bat_event"]
    assert len(at.dataframe) >= 1

    # 6. Clear at-bat history
    clear_hist_btn = next(b for b in at.button if "Limpar Histórico do Duelo" in b.label)
    clear_hist_btn.click().run()
    assert not at.exception
    assert len(at.session_state["at_bat_history"]) == 0


# ---------------------------------------------------------------------------
# 3. Page 2 — MLB Scouting Report & Decision Matrix (pages/02_scout_report.py)
# ---------------------------------------------------------------------------


def test_app_multipage_switch_to_scout_report_renders_matrix_radar_and_game_plan() -> None:
    """Verify switching from `app.py` to `pages/02_scout_report.py` renders Matrix, Radar SVG, and Game Plan."""
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not at.exception

    at.switch_page("pages/02_scout_report.py").run()
    assert not at.exception, f"Page 2 raised unexpected exception: {at.exception}"

    all_md = "\n".join(m.value for m in at.markdown)
    # 1. Verify Header & 4x3 Matrix of all 12 counts (0-0 through 3-2)
    assert "MLB SCOUTING REPORT" in all_md
    for balls in range(4):
        for strikes in range(3):
            assert f"{balls}-{strikes}" in all_md

    # 2. Verify SVG Radar Chart is rendered inline without CommonMark code-block leaks
    assert "<svg" in all_md
    assert "<polygon" in all_md
    assert "Matriz IA (12 Contagens)" in all_md
    assert "&lt;polygon" not in all_md
    assert "&lt;rect" not in all_md

    # 3. Verify all 3 Batter Game Plan cards are rendered
    assert "1. Abertura & Contagens Neutras" in all_md
    assert "2. Vantagem do Rebatedor" in all_md
    assert "3. Proteção com 2 Strikes" in all_md


def test_scout_report_batch_inference_button_updates_matrix_and_handles_errors() -> None:
    """Verify clicking the 12-count batch inference button on Page 2 calls predict_batch and updates UI."""
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.switch_page("pages/02_scout_report.py").run()
    assert not at.exception

    # Switch pitcher archetype and batter handedness
    at.selectbox(key="scout_archetype").select(
        "Offspeed Master (Logan Webb / Kevin Gausman)"
    ).run()
    at.radio(key="scout_stand").set_value("R").run()
    assert not at.exception

    mock_12_preds = [
        "Fastball",
        "Offspeed",
        "Offspeed",
        "Fastball",
        "Offspeed",
        "Breaking Ball",
        "Fastball",
        "Offspeed",
        "Offspeed",
        "Fastball",
        "Fastball",
        "Offspeed",
    ]
    mock_batch_return = {
        "status": "success",
        "predictions": mock_12_preds,
        "latency_ms": 92.4,
        "features_list": [],
        "endpoint_name": "dev_rhian_souza_pitch_predictor_endpoint_dev",
        "attempts": 1,
    }

    with patch.object(
        PitchServingClient, "predict_batch", return_value=mock_batch_return
    ) as mock_batch:
        batch_btn = next(
            b for b in at.button if "Batch 12 Contagens" in b.label or "Gerar Scouting Report" in b.label
        )
        batch_btn.click().run()

    assert not at.exception
    assert mock_batch.call_count == 1
    assert at.session_state["scout_batch_latency_ms"] == 92.4
    assert at.session_state["scout_predictions_map"]["0-1"] == "Offspeed"
    assert at.session_state["scout_predictions_map"]["3-2"] == "Offspeed"

    all_md = "\n".join(m.value for m in at.markdown)
    assert "Offspeed" in all_md
    assert "#10B981" in all_md.upper()

    # Verify graceful error handling when predict_batch raises ConnectionError
    with patch.object(
        PitchServingClient,
        "predict_batch",
        side_effect=ConnectionError("Batch endpoint timeout"),
    ):
        batch_btn = next(
            b for b in at.button if "Batch 12 Contagens" in b.label or "Gerar Scouting Report" in b.label
        )
        batch_btn.click().run()

    assert not at.exception
    assert len(at.error) >= 1
    assert "Batch endpoint timeout" in at.error[0].value


# ---------------------------------------------------------------------------
# 4. Retro Arcade Survival Minigame (pages/03_minigame.py)
# ---------------------------------------------------------------------------


def test_minigame_initial_render_and_hud() -> None:
    """Verify pages/03_minigame.py renders retro arcade HUD, RPG sliders, and cabinet buttons."""
    at = AppTest.from_file(MINIGAME_PAGE_PATH, default_timeout=30).run()
    assert not at.exception, f"Minigame raised exception: {at.exception}"

    all_md = "\n".join(m.value for m in at.markdown)
    assert "PONTUAÇÃO" in all_md
    assert "RECORDE" in all_md
    assert "VIDAS" in all_md
    assert "COMBO" in all_md
    assert "Atributos do Arremessador" in all_md
    assert "Gabinete de Arremesso" in all_md

    # Check sliders exist
    assert any("Velocidade" in s.label for s in at.slider)
    assert any("Efeito" in s.label for s in at.slider)
    assert any("Variação" in s.label for s in at.slider)

    # Check pitch button exists
    assert any("ARREMESSAR CONTRA A IA" in b.label for b in at.button)


def test_minigame_pitch_execution_with_mocked_prediction() -> None:
    """Verify executing a pitch in the minigame queries serving client and updates state."""
    at = AppTest.from_file(MINIGAME_PAGE_PATH, default_timeout=30).run()
    assert not at.exception

    mock_return = {
        "status": "success",
        "prediction": "Fastball",
        "latency_ms": 38.5,
        "features": {},
        "endpoint_name": "dev_rhian_souza_pitch_predictor_endpoint_dev",
    }

    with patch.object(
        PitchServingClient, "predict", return_value=mock_return
    ) as mock_pred:
        pitch_btn = next(b for b in at.button if "ARREMESSAR CONTRA A IA" in b.label)
        pitch_btn.click().run()

    assert not at.exception
    assert mock_pred.call_count == 1
    assert len(at.session_state["arcade_state"].history) == 1
    assert at.session_state["last_outcome"] is not None


def test_app_multipage_switch_to_vision_lab() -> None:
    """Verify switching from app entrypoint to Vision Lab page renders without exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not at.exception

    at.switch_page("pages/04_pitch_vision_lab.py").run()
    assert not at.exception
    assert len(at.title) >= 1
    assert "Laborat" in at.title[0].value or "Vis" in at.title[0].value


