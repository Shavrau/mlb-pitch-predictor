"""
Unit and AppTest tests for 3D Pitch Tunneling and Trajectory Disguise.

Validates:
1. `src/pitch_predictor/tunneling_analytics.py` physics and metrics computation.
2. `pages/06_pitch_tunneling.py` headless rendering via `streamlit.testing.v1.AppTest`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pitch_predictor.tunneling_analytics import (
    PITCHER_ARSENALS,
    compute_3d_trajectory_points,
    compute_tunneling_metrics,
    get_pitcher_arsenal,
    render_commitment_cross_section_svg,
    render_tunneling_arena_svg,
)

PAGE_PATH = str(Path(__file__).resolve().parent.parent / "pages" / "06_pitch_tunneling.py")


def test_pitcher_arsenal_integrity() -> None:
    """Verify that all 5 pitchers have configured arsenals with required fields."""
    for p_name in ["Paul Skenes", "Corbin Burnes", "Logan Webb", "Chris Sale", "Yoshinobu Yamamoto"]:
        arsenal = get_pitcher_arsenal(p_name)
        assert len(arsenal) >= 2, f"{p_name} should have at least 2 pitches"
        for p_key, spec in arsenal.items():
            assert "speed_mph" in spec
            assert spec["speed_mph"] > 70.0
            assert "spin_rpm" in spec
            assert spec["spin_rpm"] > 1000
            assert "ivb_inches" in spec
            assert "hb_inches" in spec
            assert "spin_axis" in spec


def test_compute_3d_trajectory_points() -> None:
    """Verify that trajectory points span from 54.5 ft to 1.4 ft with positive time."""
    arsenal = get_pitcher_arsenal("Paul Skenes")
    pitch = arsenal["4-Seam Fastball [FF]"]

    pts = compute_3d_trajectory_points(pitch, p_throws="R", steps=50)
    assert len(pts) == 51
    assert pts[0]["y_ft"] == 54.5
    assert pts[-1]["y_ft"] == 1.4
    assert pts[-1]["t_s"] > 0.30
    assert pts[-1]["t_s"] < 0.60


def test_compute_tunneling_metrics_skenes() -> None:
    """Verify that Skenes 4-Seam vs Splinker produces Tier S elite tunneling."""
    arsenal = get_pitcher_arsenal("Paul Skenes")
    p_a = arsenal["4-Seam Fastball [FF]"]
    p_b = arsenal["Splinker [FS]"]

    m = compute_tunneling_metrics(p_a, p_b, p_throws="R")
    assert isinstance(m, dict)
    assert "d_tunnel_inches" in m
    assert m["d_tunnel_inches"] > 0
    assert m["d_plate_inches"] > m["d_tunnel_inches"]
    assert m["tunnel_score"] >= 70
    assert "tactical_desc" in m
    assert len(m["points_a"]) == 101
    assert len(m["points_b"]) == 101


def test_render_svgs_both_modes() -> None:
    """Verify that both perspective arena and cross-section SVGs render cleanly."""
    arsenal = get_pitcher_arsenal("Corbin Burnes")
    p_a = arsenal["Cutter [FC]"]
    p_b = arsenal["Curveball [CU]"]
    m = compute_tunneling_metrics(p_a, p_b, p_throws="R")

    dark_arena = render_tunneling_arena_svg(p_a, p_b, m, p_throws="R", theme_mode="dark")
    assert "<svg" in dark_arena
    assert "COMMITMENT" in dark_arena or "DECISÃO" in dark_arena

    light_arena = render_tunneling_arena_svg(p_a, p_b, m, p_throws="R", theme_mode="light")
    assert "<svg" in light_arena
    assert "#F8FAFC" in light_arena or "F8FAFC" in light_arena

    cross_svg = render_commitment_cross_section_svg(p_a, p_b, m, theme_mode="dark")
    assert "<svg" in cross_svg
    assert "TÚNEL" in cross_svg


def test_pitch_tunneling_page_apptest() -> None:
    """Verify that pages/06_pitch_tunneling.py renders cleanly without exceptions."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(PAGE_PATH, default_timeout=60).run()
    assert not at.exception, f"Pitch Tunneling page raised an exception: {at.exception}"

    all_markdown = "\n".join(m.value for m in at.markdown)
    assert "3D BALL AERODYNAMICS & TUNNELING" in all_markdown
    assert "Disfarce de Trajetória 3D" in all_markdown
    assert "TUNNELING SCORE" in all_markdown
    assert "DISTÂNCIA NO TÚNEL (23.8 FT)" in all_markdown
    assert "QUEBRA NA PLACA (1.4 FT)" in all_markdown
    assert "ARENA DE VOO EM PERSPECTIVA 3D" in all_markdown
    assert "CORTE TRANSVERSAL NO PONTO DE DECISÃO" in all_markdown
    assert "TABELA DE PARÂMETROS AERODINÂMICOS" in all_markdown
    assert "DOSSIÊ TÁTICO DE ENFRENTAMENTO" in all_markdown
