"""
Unit and AppTest headless UI tests for Databricks MLOps & Governance Hub.

Validates:
1. `src/pitch_predictor/mlops_analytics.py` functions and resilience fallbacks.
2. `pages/05_mlops_hub.py` rendering, metric cards, tabs, and interactive live probe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pitch_predictor.mlops_analytics import (
    ENDPOINT_NAME,
    JOB_ID,
    MODEL_NAME,
    get_lakeflow_job_telemetry,
    get_mlflow_model_audit,
    get_serving_endpoint_details,
    get_unity_catalog_lineage,
    probe_endpoint_latency,
    render_medallion_pipeline_svg,
)

PAGE_PATH = str(Path(__file__).resolve().parent.parent / "pages" / "05_mlops_hub.py")


# ---------------------------------------------------------------------------
# Unit Tests for mlops_analytics.py
# ---------------------------------------------------------------------------


def test_get_serving_endpoint_details() -> None:
    """Verify endpoint details return valid schema whether live or fallback."""
    details = get_serving_endpoint_details()
    assert isinstance(details, dict)
    assert details["name"] == ENDPOINT_NAME
    assert "status" in details
    assert "online" in details
    assert "workload_size" in details
    assert "scale_to_zero" in details
    assert details["scale_to_zero"] is True


def test_probe_endpoint_latency() -> None:
    """Verify latency probe returns structured response with RTT and probabilities."""
    res = probe_endpoint_latency()
    assert isinstance(res, dict)
    assert res.get("success") is True
    assert res.get("status_code") == 200
    assert "latency_ms" in res
    assert res["latency_ms"] > 0
    assert "prediction" in res
    assert res["prediction"] in {"Fastball", "Breaking Ball", "Offspeed"}
    probs = res.get("probabilities", {})
    assert "Fastball" in probs
    assert "Breaking Ball" in probs
    assert "Offspeed" in probs


def test_get_unity_catalog_lineage() -> None:
    """Verify Unity Catalog lineage returns catalog, schema, 3 medallion tables, and volume."""
    lineage = get_unity_catalog_lineage()
    assert lineage["catalog"] == "pitch_predictor"
    assert lineage["schema"] == "baseball_ml_dev"

    tables = lineage.get("tables", [])
    assert len(tables) >= 3
    table_names = {t["name"] for t in tables}
    assert "raw_pitches" in table_names
    assert "silver_pitch_features" in table_names
    assert "biomechanical_features" in table_names

    volumes = lineage.get("volumes", [])
    assert len(volumes) >= 1
    assert volumes[0]["name"] == "savant_videos"


def test_get_mlflow_model_audit() -> None:
    """Verify MLflow model registry audit contains version data, benchmarks, and SHAP."""
    audit = get_mlflow_model_audit()
    assert audit["full_name"] == MODEL_NAME
    versions = audit.get("versions", [])
    assert len(versions) >= 1

    benchmarks = audit.get("benchmarks", [])
    assert len(benchmarks) >= 4
    models_evaluated = {b["model"] for b in benchmarks}
    assert "Multimodal PyFunc Fusion (Champion)" in models_evaluated

    shap_items = audit.get("shap_features", [])
    assert len(shap_items) >= 5
    feature_names = {s["feature"] for s in shap_items}
    assert "elbow_flexion_angle" in feature_names
    assert "pitcher_fb_prior" in feature_names


def test_get_lakeflow_job_telemetry() -> None:
    """Verify Lakeflow Job telemetry contains ID 819238473316716 and 6 tasks."""
    job = get_lakeflow_job_telemetry()
    assert job["job_id"] == JOB_ID
    assert "Serverless" in job.get("compute_type", "")

    tasks = job.get("tasks", [])
    assert len(tasks) == 6
    task_keys = [t["task_key"] for t in tasks]
    assert "setup_environment" in task_keys
    assert "bronze_declarative_pipeline" in task_keys
    assert "video_biomechanics_pipeline" in task_keys
    assert "automl_training" in task_keys
    assert "model_inference" in task_keys


def test_render_medallion_pipeline_svg_both_modes() -> None:
    """Verify architecture SVG renders cleanly in both Dark Broadcast and Light Editorial."""
    dark_svg = render_medallion_pipeline_svg(theme_mode="dark")
    assert "<svg" in dark_svg
    assert "raw_pitches" in dark_svg
    assert "silver_pitch_features" in dark_svg
    assert "biomechanical_features" in dark_svg
    assert "#0A0B0E" in dark_svg or "0A0B0E" in dark_svg

    light_svg = render_medallion_pipeline_svg(theme_mode="light")
    assert "<svg" in light_svg
    assert "#F8FAFC" in light_svg or "F8FAFC" in light_svg


# ---------------------------------------------------------------------------
# AppTest Headless UI Automation Tests
# ---------------------------------------------------------------------------


def test_mlops_page_renders_cleanly_without_exceptions() -> None:
    """Verify that the MLOps hub page renders without any unhandled exceptions."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(PAGE_PATH, default_timeout=60).run()
    assert not at.exception, f"MLOps page raised an exception: {at.exception}"

    all_markdown = "\n".join(m.value for m in at.markdown)
    assert "PRODUCTION MLOPS & GOVERNANCE" in all_markdown
    assert "Databricks MLOps & Governança" in all_markdown
    assert "SERVING ENDPOINT" in all_markdown
    assert "LATÊNCIA P90 (RTT)" in all_markdown
    assert "MODELO NO UNITY CATALOG" in all_markdown
    assert "LAKEFLOW JOB (6 TASKS)" in all_markdown
    assert "METADADOS DO ENDPOINT" in all_markdown
    assert "TESTE DE LATÊNCIA EM TEMPO REAL" in all_markdown
    assert "VISÃO DE ARQUITETURA MEDALHÃO END-TO-END" in all_markdown
    assert "AUDITORIA DE CONFORMIDADE: INVARIANTE 3 (ZERO-LEAKAGE)" in all_markdown
    assert "BENCHMARK DE MODELOS AVALIADOS" in all_markdown
    assert "EXPLICABILIDADE GLOBAL SHAP" in all_markdown
    assert "METADADOS DO LAKEFLOW JOB" in all_markdown
    assert "DAG DE EXECUÇÃO DAS 6 TAREFAS" in all_markdown


def test_mlops_page_probe_interaction() -> None:
    """Verify firing the latency probe button runs cleanly and presents results."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(PAGE_PATH, default_timeout=60).run()
    assert not at.exception

    # Locate the fire probe button
    probe_btn = at.button(key="probe_fire_btn")
    assert probe_btn is not None

    # Click probe button
    probe_btn.click().run()
    assert not at.exception, f"Probe click raised an exception: {at.exception}"

    all_markdown = "\n".join(m.value for m in at.markdown)
    assert "RTT:" in all_markdown or "ms" in all_markdown
    assert "Predição:" in all_markdown
