"""
Headless UI automation tests for `pages/02_scout_report.py` using `streamlit.testing.v1.AppTest`.
Validates Sprint 2 deliverables: Head-to-Head comparison mode, printable Dugout card, and zero regressions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

PAGE_PATH = str(Path(__file__).resolve().parent.parent / "pages" / "02_scout_report.py")


def test_scout_page_renders_single_pitcher_mode_without_exceptions() -> None:
    """Verify that the Scout Report page renders cleanly in default Single Pitcher mode."""
    at = AppTest.from_file(PAGE_PATH, default_timeout=60).run()
    assert not at.exception, f"Scout Report page raised an exception: {at.exception}"

    # Radio for view mode
    assert len(at.radio) >= 2
    view_mode_radio = at.radio(key="scout_view_mode")
    assert view_mode_radio is not None
    assert view_mode_radio.value == "🎯 Dossiê Individual (Single Pitcher)"

    # Markdown content validation
    all_markdown = "\n".join(m.value for m in at.markdown)
    assert "MLB SCOUTING REPORT" in all_markdown
    assert "Parâmetros do Confronto" in all_markdown
    assert "01. Matriz Tática Interativa" in all_markdown
    assert "02. Comparativo de Valor Situacional" in all_markdown
    assert "03. Plano de Jogo do Rebatedor" in all_markdown
    assert "04. Exportação do Dossiê Tático" in all_markdown


def test_scout_page_renders_head_to_head_mode_without_exceptions() -> None:
    """Verify that switching to Head-to-Head mode renders both pitchers, radar, and comparison table."""
    at = AppTest.from_file(PAGE_PATH, default_timeout=60).run()
    assert not at.exception

    # Switch to Head-to-Head mode
    view_mode_radio = at.radio(key="scout_view_mode")
    view_mode_radio.set_value("⚔️ Comparação Head-to-Head (2 Arremessadores)").run()
    assert not at.exception, f"Head-to-Head mode raised an exception: {at.exception}"

    all_markdown = "\n".join(m.value for m in at.markdown)
    assert "Seleção de Arremessadores & Contexto do Confronto Head-to-Head" in all_markdown
    assert "Perfis dos Atletas no Duelo Head-to-Head" in all_markdown
    assert "01. Telemetria de Divergência Tática" in all_markdown
    assert "02. Matriz Comparativa Contagem a Contagem" in all_markdown
    assert "03. Radar de Confronto Multi-Polígono" in all_markdown
    assert "04. Game Plan Comparativo" in all_markdown
    assert "05. Exportação dos Cartões Imprimíveis de Dugout" in all_markdown
    assert "DIVERGÊNCIA TÁTICA" in all_markdown
    assert "CONSENSO DE DUELO" in all_markdown


def test_scout_page_head_to_head_switch_pitchers() -> None:
    """Verify selecting different pitchers in Head-to-Head updates without exception."""
    at = AppTest.from_file(PAGE_PATH, default_timeout=60).run()
    assert not at.exception

    view_mode_radio = at.radio(key="scout_view_mode")
    view_mode_radio.set_value("⚔️ Comparação Head-to-Head (2 Arremessadores)").run()
    assert not at.exception

    pitcher_a_sel = at.selectbox(key="scout_h2h_pitcher_a")
    pitcher_b_sel = at.selectbox(key="scout_h2h_pitcher_b")

    assert pitcher_a_sel is not None
    assert pitcher_b_sel is not None

    # Change Pitcher A to Chris Sale
    pitcher_a_sel.set_value("Lefty Crafty Ace (Chris Sale / Blake Snell)").run()
    assert not at.exception

    all_markdown = "\n".join(m.value for m in at.markdown)
    assert "Chris Sale" in all_markdown
