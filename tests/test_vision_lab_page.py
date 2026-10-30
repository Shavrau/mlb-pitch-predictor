"""
Headless UI automation tests for `pages/04_pitch_vision_lab.py` using `streamlit.testing.v1.AppTest`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

PAGE_PATH = str(Path(__file__).resolve().parent.parent / "pages" / "04_pitch_vision_lab.py")


def test_vision_lab_page_renders_without_exceptions() -> None:
    """Verify that the Vision Lab page renders cleanly without any unhandled exceptions."""
    at = AppTest.from_file(PAGE_PATH, default_timeout=60).run()
    assert not at.exception, f"Vision Lab page raised an exception: {at.exception}"

    assert len(at.title) >= 1
    assert "Laborat" in at.title[0].value or "Vis" in at.title[0].value

    all_markdown = "\n".join(m.value for m in at.markdown)
    assert "Pitch Tipping" in all_markdown
    assert "Paul Skenes" in "\n".join(i.value for i in at.info) or "Paul Skenes" in all_markdown


def test_vision_lab_page_elements_and_metrics() -> None:
    """Verify presence of pitch selector, kinematic metrics, base print, and vision-only predictor."""
    at = AppTest.from_file(PAGE_PATH, default_timeout=60).run()
    assert not at.exception

    # Selectbox for pitch selection
    assert len(at.selectbox) >= 1
    pitch_selector = at.selectbox(key="vision_pitch_selector")
    assert pitch_selector is not None
    assert len(pitch_selector.options) >= 3

    # Metrics (Arm slot, elbow flexion, release height, trunk tilt, plus vision prediction metric)
    metric_labels = [m.label for m in at.metric]
    assert any("Arm Slot" in lbl for lbl in metric_labels)
    assert any("Cotovelo" in lbl for lbl in metric_labels)
    assert any("Soltura" in lbl for lbl in metric_labels)
    assert any("Tronco" in lbl for lbl in metric_labels)
    assert any("Pose" in lbl or "Previsto" in lbl for lbl in metric_labels)


def test_vision_lab_page_base_print_and_vision_only_focus() -> None:
    """Verify base print image is shown for the pose model and page focuses solely on computer vision."""
    at = AppTest.from_file(PAGE_PATH, default_timeout=60).run()
    assert not at.exception

    all_markdown = "\n".join(m.value for m in at.markdown)
    # Must show the Base Print and 3D Pose Model sections
    assert "Print Base" in all_markdown
    assert "Modelo da Pose" in all_markdown

    # Must be exclusively focused on Computer Vision (no Statcast contextual / Tri-Modal comparator)
    assert "Comparador Tri-Modal" not in all_markdown
    assert "Modelo Statcast (Contextual)" not in all_markdown
    assert "Fusão Multimodal" not in all_markdown


def test_vision_lab_page_switching_pitch_selection() -> None:
    """Verify selecting a different pitch category re-runs and updates analysis without error."""
    at = AppTest.from_file(PAGE_PATH, default_timeout=60).run()
    assert not at.exception

    pitch_selector = at.selectbox(key="vision_pitch_selector")
    # Switch to fifth option (Breaking Ball)
    pitch_selector.select_index(4).run()
    assert not at.exception

    all_info = "\n".join(i.value for i in at.info)
    all_markdown = "\n".join(m.value for m in at.markdown)
    assert "Breaking Ball" in all_info or "Breaking Ball" in all_markdown


def test_vision_lab_page_switching_pitcher_selection() -> None:
    """Verify selecting different MLB pitchers updates profile, handedness, and arsenals without error."""
    at = AppTest.from_file(PAGE_PATH, default_timeout=60).run()
    assert not at.exception

    pitcher_selector = at.selectbox(key="vision_pitcher_selector")
    assert pitcher_selector is not None
    assert len(pitcher_selector.options) == 5

    # Switch to Chris Sale (index 3 - Lefty / LHP)
    pitcher_selector.select_index(3).run()
    assert not at.exception

    all_content = "\n".join(i.value for i in at.info) + "\n" + "\n".join(m.value for m in at.markdown)
    assert "Chris Sale" in all_content
    assert "LHP" in all_content or "Canhoto" in all_content

    # Switch to Yoshinobu Yamamoto (index 4 - Dodgers)
    pitcher_selector = at.selectbox(key="vision_pitcher_selector")
    pitcher_selector.select_index(4).run()
    assert not at.exception

    all_content = "\n".join(i.value for i in at.info) + "\n" + "\n".join(m.value for m in at.markdown)
    assert "Yoshinobu Yamamoto" in all_content
    assert "Los Angeles Dodgers" in all_content or "LAD" in all_content


def test_vision_lab_page_adaptive_z_score_tipping_threshold() -> None:
    """Verify adaptive Z-score calculation on biomechanical features and telemetry metrics."""
    at = AppTest.from_file(PAGE_PATH, default_timeout=60).run()
    assert not at.exception

    # Metrics should reflect Z-score in their deltas
    deltas = [m.delta for m in at.metric if m.delta]
    assert any("Z=" in d for d in deltas)

    # Enable custom pose simulator to verify adaptive tipping behavior
    custom_cb = at.checkbox(key="vision_use_custom_pose")
    assert custom_cb is not None
    custom_cb.check().run()
    assert not at.exception
