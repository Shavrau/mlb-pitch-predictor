"""
Layer 2 Verification Script: Streamlit HTTP Health Probes & Real Databricks Serverless Endpoint E2E Test.

1. Starts `streamlit run app.py` headless on port 8501 as a background subprocess.
2. Polls `http://localhost:8501/_stcore/health` until HTTP 200 ("ok").
3. Verifies `GET http://localhost:8501/` returns HTTP 200 and valid Streamlit HTML shell.
4. Queries the real Databricks Serverless Model Serving endpoint (`dev_rhian_souza_pitch_predictor_endpoint_dev`)
   via `PitchServingClient(profile="dev")` across 5 distinct MLB tactical scenarios (including
   First Pitch 0-0, Pitcher Ahead 0-2, and Full Count 3-2 Bases Loaded).
5. Terminates the Streamlit subprocess cleanly in the `finally` block.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

# Ensure local package is importable when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pitch_predictor.scenarios import TACTICAL_SCENARIOS
from pitch_predictor.serving_client import (
    DEFAULT_ENDPOINT_NAME,
    EXPECTED_FEATURE_COLUMNS,
    PitchServingClient,
)

STREAMLIT_PORT = 8501
BASE_URL = f"http://localhost:{STREAMLIT_PORT}"
HEALTH_URL = f"{BASE_URL}/_stcore/health"


def wait_for_streamlit_health(timeout_sec: float = 35.0) -> dict[str, Any]:
    """Poll Streamlit `/_stcore/health` until it responds with HTTP 200 and 'ok'."""
    t0 = time.perf_counter()
    last_err = None
    while (time.perf_counter() - t0) < timeout_sec:
        try:
            resp = requests.get(HEALTH_URL, timeout=3)
            if resp.status_code == 200 and resp.text.strip().lower() == "ok":
                return {
                    "url": HEALTH_URL,
                    "status_code": resp.status_code,
                    "body": resp.text.strip(),
                    "startup_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                }
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
        time.sleep(0.5)
    raise RuntimeError(f"Streamlit health check timed out after {timeout_sec}s: {last_err}")


def verify_streamlit_root_html() -> dict[str, Any]:
    """Verify `GET http://localhost:8501/` returns HTTP 200 and Streamlit HTML structure."""
    resp = requests.get(BASE_URL, timeout=5)
    resp.raise_for_status()
    html = resp.text
    has_doctype = "<!doctype html>" in html.lower()
    has_root_div = 'id="root"' in html or "streamlit" in html.lower()
    if not (has_doctype and has_root_div):
        raise AssertionError("Root HTML response did not match expected Streamlit shell.")
    return {
        "url": BASE_URL,
        "status_code": resp.status_code,
        "content_length": len(html),
        "has_doctype": has_doctype,
        "has_streamlit_shell": has_root_div,
    }


def verify_real_databricks_endpoint() -> dict[str, Any]:
    """Invoke the live Databricks Serverless Endpoint across all tactical scenarios."""
    client = PitchServingClient(
        endpoint_name=DEFAULT_ENDPOINT_NAME,
        profile="dev",
        timeout_seconds=90,
        max_retries=3,
        backoff_seconds=3.0,
    )

    endpoint_status = client.check_endpoint_status()
    scenario_results: list[dict[str, Any]] = []

    for scenario_name, meta in TACTICAL_SCENARIOS.items():
        raw_inputs = meta["inputs"]
        result = client.predict(raw_inputs)
        features = result["features"]
        assert list(features.keys()) == EXPECTED_FEATURE_COLUMNS
        assert result["prediction"] in {"Fastball", "Breaking Ball", "Offspeed"}

        scenario_results.append(
            {
                "scenario": scenario_name,
                "badge": meta["badge"],
                "count_state": features["count_state"],
                "count_leverage": features["count_leverage"],
                "runners_on_base": features["runners_on_base"],
                "scoring_position": features["scoring_position"],
                "prev_pitch_category": features["prev_pitch_category"],
                "prev_pitch_zone": features["prev_pitch_zone"],
                "prediction": result["prediction"],
                "latency_ms": result["latency_ms"],
                "attempts": result["attempts"],
            }
        )

    return {
        "endpoint_name": client.endpoint_name,
        "endpoint_status": endpoint_status,
        "scenarios_tested": len(scenario_results),
        "results": scenario_results,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_ROOT / "app.py"),
        "--server.headless",
        "true",
        "--server.port",
        str(STREAMLIT_PORT),
        "--browser.gatherUsageStats",
        "false",
    ]

    print(f"[1/4] Starting Streamlit server in background on port {STREAMLIT_PORT}...")
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        print(f"[2/4] Probing Streamlit health endpoint: {HEALTH_URL} ...")
        health_info = wait_for_streamlit_health()
        print(f"      -> Health Probe OK: {health_info}")

        print(f"[3/4] Probing Streamlit root HTML page: {BASE_URL} ...")
        root_info = verify_streamlit_root_html()
        print(f"      -> Root HTML OK: {root_info}")

        print(
            f"[4/4] Executing REAL E2E inference against Databricks endpoint "
            f"'{DEFAULT_ENDPOINT_NAME}'..."
        )
        endpoint_report = verify_real_databricks_endpoint()
        for item in endpoint_report["results"]:
            print(
                f"      * {item['scenario']} [{item['count_state']} | {item['count_leverage']}] "
                f"=> Predicted: {item['prediction']} ({item['latency_ms']} ms)"
            )

        summary = {
            "layer": "Layer 2 — HTTP Health Probes & Real Databricks Endpoint Verification",
            "status": "PASSED",
            "health_probe": health_info,
            "root_html_probe": root_info,
            "databricks_e2e": endpoint_report,
        }
        print("\n=== LAYER 2 VERIFICATION SUMMARY ===")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    finally:
        print("\n[Cleanup] Terminating background Streamlit server...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("[Cleanup] Streamlit server stopped cleanly.")


if __name__ == "__main__":
    raise SystemExit(main())
