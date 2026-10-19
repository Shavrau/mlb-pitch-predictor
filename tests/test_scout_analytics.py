"""
Unit tests for MLB Scouting Report, 12-Count Batch Inference, and Tactical Decision Matrix (Red Phase TDD).

Covers:
1. `PitchServingClient.predict_batch(raw_inputs_list)` in `pitch_predictor.serving_client`:
   - Batch inference of 12 MLB counts in a single `_send_request` call (`dataframe_records` len == 12).
   - Structured return dictionary (`status`, `predictions`, `latency_ms`, `features_list`, `endpoint_name`, `attempts`).
   - Cold-start retry resilience when the 1st attempt raises `ConnectionError` and the 2nd succeeds.
2. `pitch_predictor.scout_analytics`:
   - `MLB_12_COUNTS`: Exact list of 12 `(balls, strikes)` tuples from `(0, 0)` to `(3, 2)`.
   - `generate_12_counts_batch`: Generates 12 valid 24-column feature dicts; enforces First Pitch defaults on `0-0`
     and propagates `scoring_position == 1` when `on_2b=1`.
   - `compute_matrix_distribution`: Computes (`Fastball`, `Breaking Ball`, `Offspeed`) proportions summing to `1.0`.
   - `generate_batter_game_plan`: Returns structured game plan with `"early_counts"`, `"hitter_counts"`, and
     `"two_strike_protection"`, each containing `"title"`, `"dominant_pitch"`, `"counts_covered"`,
     `"strategy"`, and `"chase_warning"`.
   - `render_count_matrix_html`: Renders all 12 counts (`0-0` to `3-2`), includes semantic category hex colors,
     and is strictly CommonMark-safe (`"\\n" not in html`, `"    " not in html`, `"<!--" not in html`).
   - `render_radar_chart_svg`: Renders valid `<svg` and `<polygon` with the 3 axes (`Fastball`, `Breaking`, `Offspeed`)
     and is strictly CommonMark-safe.
   - `render_game_plan_cards_html`: Renders HTML for the 3 batter approach cards and is strictly CommonMark-safe.
   - `export_scouting_report_markdown`: Generates complete Markdown report with pitcher name, all 12 counts,
     and the 3 Game Plan sections.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from pitch_predictor.scenarios import PITCHER_ARCHETYPES
from pitch_predictor.scout_analytics import (
    MLB_12_COUNTS,
    compute_matrix_distribution,
    export_scouting_report_markdown,
    generate_12_counts_batch,
    generate_batter_game_plan,
    render_count_matrix_html,
    render_game_plan_cards_html,
    render_radar_chart_svg,
)
from pitch_predictor.serving_client import (
    EXPECTED_FEATURE_COLUMNS,
    PitchServingClient,
    validate_inference_record,
)


def _sample_12_predictions_map() -> dict[str, str]:
    """Return a realistic 12-count prediction mapping across all MLB counts."""
    return {
        "0-0": "Fastball",
        "0-1": "Breaking Ball",
        "0-2": "Breaking Ball",
        "1-0": "Fastball",
        "1-1": "Offspeed",
        "1-2": "Breaking Ball",
        "2-0": "Fastball",
        "2-1": "Fastball",
        "2-2": "Breaking Ball",
        "3-0": "Fastball",
        "3-1": "Fastball",
        "3-2": "Offspeed",
    }


# ---------------------------------------------------------------------------
# 1. Batch Inference in PitchServingClient
# ---------------------------------------------------------------------------


def test_pitch_serving_client_predict_batch_single_request_for_12_counts() -> None:
    """Verify predict_batch sends all 12 count records in a single _send_request call."""
    client = PitchServingClient(
        endpoint_name="pitch-predictor-prod",
        host="https://adb-test.azuredatabricks.net",
        token="dapi_test_token",
    )
    profile = PITCHER_ARCHETYPES["Power Pitcher (Paul Skenes / Spencer Strider)"]
    raw_batch = generate_12_counts_batch(pitcher_profile=profile, stand="L")

    expected_preds = [
        "Fastball",
        "Breaking Ball",
        "Breaking Ball",
        "Fastball",
        "Fastball",
        "Breaking Ball",
        "Fastball",
        "Fastball",
        "Breaking Ball",
        "Fastball",
        "Fastball",
        "Offspeed",
    ]

    with patch.object(
        client, "_send_request", return_value={"predictions": expected_preds}
    ) as mock_send:
        result = client.predict_batch(raw_batch)

    assert mock_send.call_count == 1
    sent_payload = mock_send.call_args[0][0]
    assert "dataframe_records" in sent_payload
    assert len(sent_payload["dataframe_records"]) == 12

    assert result["status"] == "success"
    assert result["predictions"] == expected_preds
    assert len(result["predictions"]) == 12
    assert isinstance(result["latency_ms"], (int, float))
    assert result["latency_ms"] >= 0
    assert len(result["features_list"]) == 12
    for feat_dict in result["features_list"]:
        assert list(feat_dict.keys()) == EXPECTED_FEATURE_COLUMNS
    assert result["endpoint_name"] == "pitch-predictor-prod"
    assert result["attempts"] == 1


def test_pitch_serving_client_predict_batch_retries_on_cold_start() -> None:
    """Verify predict_batch retries on ConnectionError in attempt 1 and succeeds on attempt 2."""
    client = PitchServingClient(
        endpoint_name="pitch-predictor-prod",
        host="https://adb-test.azuredatabricks.net",
        token="dapi_test_token",
        max_retries=2,
        backoff_seconds=0.01,
    )
    profile = PITCHER_ARCHETYPES["Média da Liga MLB (League Average)"]
    raw_batch = generate_12_counts_batch(pitcher_profile=profile, stand="R")
    expected_preds = ["Fastball"] * 12

    with patch.object(
        client,
        "_send_request",
        side_effect=[
            ConnectionError("Endpoint warming up from scale-to-zero"),
            {"predictions": expected_preds},
        ],
    ) as mock_send:
        result = client.predict_batch(raw_batch)

    assert mock_send.call_count == 2
    assert result["status"] == "success"
    assert result["predictions"] == expected_preds
    assert result["attempts"] == 2


# ---------------------------------------------------------------------------
# 2. MLB_12_COUNTS & 12-Count Tactical Batch Generator
# ---------------------------------------------------------------------------


def test_mlb_12_counts_exact_order() -> None:
    """Verify MLB_12_COUNTS contains the exact 12 (balls, strikes) pairs from (0,0) to (3,2)."""
    expected_counts = [
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
    assert len(MLB_12_COUNTS) == 12
    assert MLB_12_COUNTS == expected_counts


def test_generate_12_counts_batch_validates_all_records_and_0_0_first_pitch() -> None:
    """Verify generate_12_counts_batch returns 12 valid 24-col dicts with First Pitch on 0-0 and RISP on 2B."""
    profile = PITCHER_ARCHETYPES["Breaking Specialist (Corbin Burnes / Sonny Gray)"]
    batch = generate_12_counts_batch(
        pitcher_profile=profile,
        stand="L",
        on_1b=0,
        on_2b=1,
        on_3b=0,
        outs_when_up=1,
        inning=6,
        inning_topbot="Top",
        score_diff=-1,
        n_thruorder_pitcher=2,
    )

    assert isinstance(batch, list)
    assert len(batch) == 12

    observed_counts = []
    for rec in batch:
        assert validate_inference_record(rec) is True
        assert list(rec.keys()) == EXPECTED_FEATURE_COLUMNS
        assert rec["scoring_position"] == 1
        assert rec["on_2b_flag"] == 1
        observed_counts.append((rec["balls"], rec["strikes"]))

    assert observed_counts == MLB_12_COUNTS

    # 0-0 record must strictly enforce First Pitch boundary semantics
    rec_0_0 = batch[0]
    assert rec_0_0["count_state"] == "0-0"
    assert rec_0_0["prev_pitch_category"] == "First Pitch"
    assert rec_0_0["prev_pitch_type"] == "None"
    assert rec_0_0["prev_pitch_zone"] == 0
    assert rec_0_0["prev_pitch_result"] == "First Pitch"


# ---------------------------------------------------------------------------
# 3. Matrix Distribution & Batter Game Plan Generation
# ---------------------------------------------------------------------------


def test_compute_matrix_distribution_sums_to_one() -> None:
    """Verify compute_matrix_distribution returns Fastball, Breaking Ball, Offspeed proportions summing to 1.0."""
    preds_map = _sample_12_predictions_map()
    dist = compute_matrix_distribution(preds_map)

    assert set(dist.keys()) == {"Fastball", "Breaking Ball", "Offspeed"}
    assert dist["Fastball"] == pytest.approx(6 / 12)
    assert dist["Breaking Ball"] == pytest.approx(4 / 12)
    assert dist["Offspeed"] == pytest.approx(2 / 12)
    assert sum(dist.values()) == pytest.approx(1.0)


def test_generate_batter_game_plan_structure_and_keys() -> None:
    """Verify generate_batter_game_plan returns early_counts, hitter_counts, and two_strike_protection."""
    preds_map = _sample_12_predictions_map()
    priors = {
        "pitcher_fb_prior": 0.64,
        "pitcher_bb_prior": 0.28,
        "pitcher_os_prior": 0.08,
    }
    plan = generate_batter_game_plan(
        pitcher_name="Power Pitcher (Paul Skenes / Spencer Strider)",
        predictions_by_count=preds_map,
        priors=priors,
        stand="L",
        risp_active=True,
    )

    assert isinstance(plan, dict)
    expected_sections = ("early_counts", "hitter_counts", "two_strike_protection")
    required_fields = (
        "title",
        "dominant_pitch",
        "counts_covered",
        "strategy",
        "chase_warning",
    )

    for section_key in expected_sections:
        assert section_key in plan
        section = plan[section_key]
        assert isinstance(section, dict)
        for field in required_fields:
            assert field in section
            assert section[field]  # non-empty value


# ---------------------------------------------------------------------------
# 4. CommonMark-Safe HTML/SVG Rendering & Markdown Report Export
# ---------------------------------------------------------------------------


def test_render_count_matrix_html_renders_12_counts_colors_and_is_commonmark_safe() -> None:
    """Verify render_count_matrix_html includes all 12 counts, semantic colors, and is CommonMark-safe."""
    preds_map = _sample_12_predictions_map()
    html = render_count_matrix_html(preds_map)

    for b, s in MLB_12_COUNTS:
        assert f"{b}-{s}" in html

    html_upper = html.upper()
    assert "#EF4444" in html_upper or "#DC2626" in html_upper  # Fastball
    assert "#3B82F6" in html_upper or "#2563EB" in html_upper  # Breaking Ball
    assert "#10B981" in html_upper or "#059669" in html_upper  # Offspeed

    assert "\n" not in html
    assert "    " not in html
    assert "<!--" not in html


def test_render_radar_chart_svg_renders_polygon_axes_and_is_commonmark_safe() -> None:
    """Verify render_radar_chart_svg returns valid SVG with <polygon, 3 axes, and is CommonMark-safe."""
    league_dist = {"Fastball": 0.5543, "Breaking Ball": 0.3053, "Offspeed": 0.1404}
    pitcher_dist = {"Fastball": 0.6400, "Breaking Ball": 0.2800, "Offspeed": 0.0800}
    model_dist = compute_matrix_distribution(_sample_12_predictions_map())

    svg = render_radar_chart_svg(
        league_dist=league_dist,
        pitcher_dist=pitcher_dist,
        model_dist=model_dist,
    )

    assert "<svg" in svg
    assert "<polygon" in svg
    assert "Fastball" in svg
    assert "Breaking" in svg
    assert "Offspeed" in svg

    assert "\n" not in svg
    assert "    " not in svg
    assert "<!--" not in svg


def test_render_game_plan_cards_html_renders_three_cards_and_is_commonmark_safe() -> None:
    """Verify render_game_plan_cards_html renders all 3 approach cards and is CommonMark-safe."""
    preds_map = _sample_12_predictions_map()
    priors = {
        "pitcher_fb_prior": 0.46,
        "pitcher_bb_prior": 0.22,
        "pitcher_os_prior": 0.32,
    }
    game_plan = generate_batter_game_plan(
        pitcher_name="Offspeed Master (Logan Webb / Kevin Gausman)",
        predictions_by_count=preds_map,
        priors=priors,
        stand="R",
        risp_active=False,
    )

    html = render_game_plan_cards_html(game_plan)

    assert game_plan["early_counts"]["title"] in html
    assert game_plan["hitter_counts"]["title"] in html
    assert game_plan["two_strike_protection"]["title"] in html

    assert "\n" not in html
    assert "    " not in html
    assert "<!--" not in html


def test_export_scouting_report_markdown_contains_pitcher_counts_and_game_plan() -> None:
    """Verify export_scouting_report_markdown generates complete Markdown with pitcher, 12 counts, and 3 sections."""
    pitcher_name = "Power Pitcher (Paul Skenes / Spencer Strider)"
    preds_map = _sample_12_predictions_map()
    priors = {
        "pitcher_fb_prior": 0.64,
        "pitcher_bb_prior": 0.28,
        "pitcher_os_prior": 0.08,
    }
    game_plan = generate_batter_game_plan(
        pitcher_name=pitcher_name,
        predictions_by_count=preds_map,
        priors=priors,
        stand="L",
        risp_active=True,
    )

    md_report = export_scouting_report_markdown(
        pitcher_name=pitcher_name,
        predictions_by_count=preds_map,
        game_plan=game_plan,
        priors=priors,
        stand="L",
        risp_active=True,
    )

    assert isinstance(md_report, str)
    assert pitcher_name in md_report
    for b, s in MLB_12_COUNTS:
        assert f"{b}-{s}" in md_report

    assert game_plan["early_counts"]["title"] in md_report
    assert game_plan["hitter_counts"]["title"] in md_report
    assert game_plan["two_strike_protection"]["title"] in md_report
