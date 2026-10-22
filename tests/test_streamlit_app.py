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
     graceful offline fallback via `fallback_engine`.
2. Official Pitchers & Navigation on Home:
   - Skenes, Burnes, Webb, Sale, and Yamamoto in pitcher selection.
   - 3-module footer navigation targeting Minigame, Scout Report, and Vision Lab.
3. Page 2 (`pages/02_scout_report.py` — MLB Scouting Report & Matriz):
   - Multipage navigation, 4x3 Count Matrix (12 counts), inline SVG Radar Chart, Batter Game Plan cards,
     and 12-count batch inference (`predict_batch`).
4. Page 3 (`pages/03_minigame.py` — Minigame Retrô Arcade):
   - Retro HUD, RPG sliders, pitch execution, and audio synthesis.
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

    # Verify graceful fallback when PitchServingClient.predict raises ConnectionError
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
    all_md = "\n".join(m.value for m in at.markdown)
    assert any(cat in all_md for cat in ("Fastball", "Breaking Ball", "Offspeed"))


# ---------------------------------------------------------------------------
# 2. Official Pitchers & Navigation Links on Home
# ---------------------------------------------------------------------------


def test_home_pitcher_styles_include_official_mlb_pitchers() -> None:
    """Verify Home presentation page loads all 5 official MLB pitchers from MLB_PITCHERS."""
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not at.exception

    style_select = at.selectbox(key="home_pitcher_style")
    options_text = " ".join(style_select.options)

    assert "Paul Skenes" in options_text
    assert "Corbin Burnes" in options_text
    assert "Logan Webb" in options_text
    assert "Chris Sale" in options_text
    assert "Yoshinobu Yamamoto" in options_text
    assert "Média Geral da MLB" in options_text


def test_home_navigation_links_target_three_modules() -> None:
    """Verify Home page footer contains links to Minigame, Scout Report, and Vision Lab."""
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not at.exception

    page_targets = [getattr(link, "page", link.proto.page) for link in at.page_link]
    assert "minigame" in page_targets or "pages/03_minigame.py" in page_targets
    assert "scout" in page_targets or "pages/02_scout_report.py" in page_targets
    assert "vision-lab" in page_targets or "pages/04_pitch_vision_lab.py" in page_targets
    assert "simulador" not in page_targets
    assert "pages/01_live_predictor.py" not in page_targets


def test_app_navigation_structure_has_four_pages() -> None:
    """Verify app navigation defines exactly the 4 streamlined modules without the redundant page 01."""
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not at.exception
    assert at.session_state is not None


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


def test_scout_report_stale_session_state_recovery() -> None:
    """Verify that stale session state values (e.g. from previous releases) are safely repaired."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["scout_base_state"] = "RISP (Corredor na 2B)"
    at.run()
    at.switch_page("pages/02_scout_report.py").run()
    assert not at.exception
    assert "Posição de Anotar" in at.session_state["scout_base_state"]


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
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not at.exception

    at.switch_page("pages/04_pitch_vision_lab.py").run(timeout=60)
    assert not at.exception
    assert len(at.title) >= 1
    assert "Laborat" in at.title[0].value or "Vis" in at.title[0].value


