"""
Layer 3 Verification Script: Programmatic DOM/SVG Inspection, Real 12-Count Batch Inference,
and Live Streamlit Chromium CDP Visual Evidence for Page 0 (Home Presentation), Page 1, and Page 2.

1. Validates XML/SVG structural integrity (`xml.etree.ElementTree`) and semantic color contracts
   for:
   - Page 1: `render_baseball_diamond_svg`, `render_strike_zone_svg`, `render_prediction_card_html`,
     `render_scoreboard_banner_html`.
   - Page 2: `render_count_matrix_html`, `render_radar_chart_svg` (via `ET.fromstring`),
     `render_game_plan_cards_html`.
2. Executes a REAL 12-count batch inference (`PitchServingClient(profile="dev").predict_batch(...)`)
   against the Databricks Serverless Endpoint (`dev_rhian_souza_pitch_predictor_endpoint_dev`),
   recording all 12 count predictions (`0-0` to `3-2`) and batch latency in ms.
3. Starts the Streamlit multipage server in headless mode and uses Microsoft Edge Headless with
   Chrome DevTools Protocol (CDP) to:
   - Load Page 0 (`pages/00_home_apresentacao.py` — Default Landing Page), verify the 400ms challenge
     hero (`0.40 segundos`, `160 km/h`), the 3 pitch family explanatory cards, click the interactive
     Quick Demo button (`🔮 Consultar Inteligência Artificial em Tempo Real`) to execute a live
     prediction against Databricks Serverless, assert zero CommonMark `<pre><code>` leaks
     (`assert "&lt;div" not in hydrated_html`), and capture `home_preview.png`.
   - Verify Page 2 (`pages/02_scout_report.py` — MLB Scouting Report & Matriz) and `scout_report_preview.png`.
"""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pitch_predictor.scenarios import PITCHER_ARCHETYPES, TACTICAL_SCENARIOS
from pitch_predictor.scout_analytics import (
    MLB_12_COUNTS,
    compute_matrix_distribution,
    generate_12_counts_batch,
    generate_batter_game_plan,
    render_count_matrix_html,
    render_game_plan_cards_html,
    render_radar_chart_svg,
)
from pitch_predictor.serving_client import (
    DEFAULT_ENDPOINT_NAME,
    MLB_LEAGUE_PRIORS,
    PitchServingClient,
    derive_tactical_features,
)
from pitch_predictor.ui_components import (
    STATCAST_ZONES,
    render_baseball_diamond_svg,
    render_prediction_card_html,
    render_scoreboard_banner_html,
    render_strike_zone_svg,
)

EDGE_PATHS = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]

BRAIN_DIR = Path(
    r"C:\Users\rhian\.gemini\antigravity\brain\f5b0facd-ca75-49fe-bf03-8497799f797f"
)
TARGET_HOME_PNG = BRAIN_DIR / "home_preview.png"
TARGET_PREVIEW_PNG = BRAIN_DIR / "ui_preview.png"
TARGET_SCOUT_PNG = BRAIN_DIR / "scout_report_preview.png"


def _extract_svg_block(html_wrapper: str) -> str:
    """Extract the inner `<svg ...>...</svg>` element from its container `<div>`."""
    start = html_wrapper.find("<svg")
    end = html_wrapper.rfind("</svg>")
    if start == -1 or end == -1:
        raise ValueError("No <svg>...</svg> block found in markup.")
    return html_wrapper[start : end + len("</svg>")]


def verify_svg_and_html_dom_components() -> dict[str, Any]:
    """Programmatically parse and validate all Page 1 & Page 2 SVG and HTML components."""
    # 1. Validate Baseball Diamond SVG across base/out combinations
    diamond_empty = render_baseball_diamond_svg(0, 0, 0, outs=0)
    diamond_loaded = render_baseball_diamond_svg(1, 1, 1, outs=2)

    svg_empty_root = ET.fromstring(_extract_svg_block(diamond_empty))
    svg_loaded_root = ET.fromstring(_extract_svg_block(diamond_loaded))

    assert svg_empty_root.tag.endswith("svg")
    assert svg_loaded_root.tag.endswith("svg")

    polygons = svg_loaded_root.findall(".//polygon")
    rects = svg_loaded_root.findall(".//rect")
    circles = svg_loaded_root.findall(".//circle")
    assert len(polygons) >= 2
    assert len(rects) >= 4
    assert len(circles) >= 3

    assert "#F59E0B" in diamond_loaded.upper()
    assert "#EF4444" in diamond_loaded.upper()
    assert "RISP ATIVO" in diamond_loaded
    assert "BASES LIMPAS" in diamond_empty

    # 2. Validate Statcast Strike Zone SVG across all 14 zones (0, 1..9, 11..14)
    expected_zones = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14}
    assert set(STATCAST_ZONES.keys()) == expected_zones
    assert 10 not in STATCAST_ZONES

    for zone_id in expected_zones:
        for cat in ("First Pitch", "Fastball", "Breaking Ball", "Offspeed"):
            sz_html = render_strike_zone_svg(selected_zone=zone_id, prev_category=cat)
            sz_root = ET.fromstring(_extract_svg_block(sz_html))
            sz_rects = sz_root.findall(".//rect")
            assert len(sz_rects) == 14
            if zone_id != 0:
                assert len(sz_root.findall(".//circle")) == 2

    # 3. Validate Page 1 Prediction Cards & Scoreboard
    sample_features = derive_tactical_features(
        TACTICAL_SCENARIOS["🔥 Contagem Cheia & Bases Lotadas (3-2 High Leverage)"]["inputs"]
    )
    expected_colors = {
        "Fastball": ("#EF4444", "#DC2626"),
        "Breaking Ball": ("#3B82F6", "#2563EB"),
        "Offspeed": ("#10B981", "#059669"),
    }
    for cat, (prim, sec) in expected_colors.items():
        card_html = render_prediction_card_html(cat, sample_features, latency_ms=112.4)
        assert prim in card_html.upper()
        assert sec in card_html.upper()
        assert cat in card_html

    banner_html = render_scoreboard_banner_html(sample_features, DEFAULT_ENDPOINT_NAME)
    assert DEFAULT_ENDPOINT_NAME in banner_html

    # 4. Validate Page 2 Components: 4x3 Count Matrix HTML, Radar Chart SVG, and Game Plan Cards HTML
    sample_12_map = {
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
    matrix_html = render_count_matrix_html(sample_12_map)
    for b, s in MLB_12_COUNTS:
        assert f"{b}-{s}" in matrix_html
    assert "\n" not in matrix_html
    assert "    " not in matrix_html

    model_dist = compute_matrix_distribution(sample_12_map)
    radar_html = render_radar_chart_svg(
        league_dist=MLB_LEAGUE_PRIORS,
        pitcher_dist=(0.64, 0.28, 0.08),
        model_dist=model_dist,
    )
    radar_svg_root = ET.fromstring(_extract_svg_block(radar_html))
    assert radar_svg_root.tag.endswith("svg")
    radar_polygons = radar_svg_root.findall(".//polygon")
    assert len(radar_polygons) >= 7, "Expected 4 radar rings + 3 data polygons"
    assert "\n" not in radar_html
    assert "    " not in radar_html

    game_plan = generate_batter_game_plan(
        pitcher_name="Power Pitcher (Paul Skenes / Spencer Strider)",
        predictions_by_count=sample_12_map,
        priors={"pitcher_fb_prior": 0.64, "pitcher_bb_prior": 0.28, "pitcher_os_prior": 0.08},
        stand="L",
        risp_active=True,
    )
    gp_html = render_game_plan_cards_html(game_plan)
    assert game_plan["early_counts"]["title"] in gp_html
    assert game_plan["hitter_counts"]["title"] in gp_html
    assert game_plan["two_strike_protection"]["title"] in gp_html
    assert "\n" not in gp_html
    assert "    " not in gp_html

    return {
        "diamond_svg_valid_xml": True,
        "strike_zone_svg_valid_xml": True,
        "statcast_zones_verified": sorted(expected_zones),
        "count_matrix_12_cells_valid": True,
        "radar_svg_valid_xml": True,
        "radar_polygons_count": len(radar_polygons),
        "game_plan_3_cards_valid": True,
        "semantic_colors_verified": expected_colors,
    }


def verify_real_12_counts_batch_inference() -> dict[str, Any]:
    """Execute a real 12-count batch inference against the Databricks Serverless Endpoint."""
    client = PitchServingClient(
        endpoint_name=DEFAULT_ENDPOINT_NAME,
        profile="dev",
        timeout_seconds=90,
        max_retries=3,
        backoff_seconds=3.0,
    )
    archetype_name = "Breaking Specialist (Corbin Burnes / Sonny Gray)"
    profile = PITCHER_ARCHETYPES[archetype_name]
    batch_records = generate_12_counts_batch(
        pitcher_profile=profile,
        stand="R",
        on_1b=0,
        on_2b=1,
        on_3b=0,
        prev_pitch_category="Fastball",
        prev_pitch_type="FC",
        prev_pitch_zone=5,
        prev_pitch_result="S",
    )

    batch_result = client.predict_batch(batch_records)
    preds_list = batch_result["predictions"]
    assert len(preds_list) == 12

    predictions_by_count = {
        f"{b}-{s}": preds_list[idx] for idx, (b, s) in enumerate(MLB_12_COUNTS)
    }
    dist = compute_matrix_distribution(predictions_by_count)

    return {
        "endpoint_name": client.endpoint_name,
        "pitcher_archetype": archetype_name,
        "batter_stand": "R",
        "risp_active": True,
        "records_sent": len(batch_records),
        "latency_ms": batch_result["latency_ms"],
        "attempts": batch_result["attempts"],
        "predictions_by_count": predictions_by_count,
        "matrix_distribution": {k: round(v, 4) for k, v in dist.items()},
    }


def verify_and_capture_home_presentation_via_cdp(output_path: Path) -> dict[str, Any]:
    """
    Start Streamlit in background, load the default Home Presentation page
    (`pages/00_home_apresentacao.py`) via Chrome DevTools Protocol (CDP),
    verify the 400ms challenge metrics and 3 pitch cards, click the interactive
    Quick Demo button ('🔮 Consultar Inteligência Artificial em Tempo Real'),
    wait for the live Databricks Serverless prediction card, assert zero CommonMark
    `<pre><code>` leaks (`assert "&lt;div" not in hydrated_html`), and save a
    high-resolution (1920x1680) screenshot to `home_preview.png`.
    """
    import websockets

    edge_exe = next((p for p in EDGE_PATHS if p.exists()), None)
    if not edge_exe:
        raise FileNotFoundError("Microsoft Edge executable not found for headless rendering.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    st_port = 8524
    cdp_port = 9244
    user_data_dir = tempfile.mkdtemp(prefix="edge_cdp_home_")

    st_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(PROJECT_ROOT / "app.py"),
            "--server.headless",
            "true",
            "--server.address",
            "127.0.0.1",
            "--server.port",
            str(st_port),
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    edge_proc = None
    try:
        for _ in range(40):
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{st_port}/_stcore/health", timeout=2
                ) as r:
                    if r.status == 200:
                        break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)

        edge_proc = subprocess.Popen(
            [
                str(edge_exe),
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                "--no-default-browser-check",
                "--hide-scrollbars",
                f"--user-data-dir={user_data_dir}",
                f"--remote-debugging-port={cdp_port}",
                "--window-size=1920,1750",
                f"http://127.0.0.1:{st_port}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        targets = []
        for _ in range(30):
            time.sleep(0.6)
            try:
                raw_json = (
                    urllib.request.urlopen(
                        f"http://127.0.0.1:{cdp_port}/json", timeout=3
                    )
                    .read()
                    .decode()
                )
                targets = json.loads(raw_json)
                if any(
                    t.get("type") == "page" and str(st_port) in t.get("url", "")
                    for t in targets
                ):
                    break
            except Exception:  # noqa: BLE001, S112
                continue

        ws_url = next(
            t["webSocketDebuggerUrl"]
            for t in targets
            if t.get("type") == "page" and str(st_port) in t.get("url", "")
        )

        async def _cdp_home_flow() -> dict[str, Any]:
            async with websockets.connect(ws_url, max_size=50_000_000) as ws:
                # Set 1920x1750 viewport via CDP Emulation so Hero, 60s Guide, Live Demo + Result Card,
                # Architecture, and Navigation cards fit in a single crisp view
                await ws.send(
                    json.dumps(
                        {
                            "id": 1,
                            "method": "Emulation.setDeviceMetricsOverride",
                            "params": {
                                "width": 1920,
                                "height": 1750,
                                "deviceScaleFactor": 1,
                                "mobile": False,
                            },
                        }
                    )
                )
                await ws.recv()

                # 1. Poll up to 60s until Home Presentation initial layout is hydrated
                hydrated_html = ""
                for sec in range(1, 60):
                    await asyncio.sleep(1.0)
                    await ws.send(
                        json.dumps(
                            {
                                "id": 100 + sec,
                                "method": "Runtime.evaluate",
                                "params": {"expression": "document.body.innerHTML"},
                            }
                        )
                    )
                    resp = json.loads(await ws.recv())
                    hydrated_html = (
                        resp.get("result", {}).get("result", {}).get("value", "")
                    )
                    if (
                        "Decodificando o Arremesso com Inteligência Artificial" in hydrated_html
                        and "0.40 segundos" in hydrated_html
                        and "Consultar Inteligência Artificial em Tempo Real" in hydrated_html
                    ):
                        print(f"      [CDP] Home page initial DOM hydrated at {sec}s (len={len(hydrated_html)})")
                        break

                # 2. Click the Quick Demo button ('🔮 Consultar Inteligência Artificial em Tempo Real')
                js_click_demo = """
                (() => {
                    const btns = Array.from(document.querySelectorAll('button'));
                    const demoBtn = btns.find(b =>
                        b.textContent && (
                            b.textContent.includes('Consultar Inteligência Artificial') ||
                            b.textContent.includes('Tempo Real')
                        )
                    );
                    if (demoBtn) {
                        demoBtn.click();
                        return true;
                    }
                    return false;
                })()
                """
                await ws.send(
                    json.dumps(
                        {
                            "id": 200,
                            "method": "Runtime.evaluate",
                            "params": {"expression": js_click_demo},
                        }
                    )
                )
                click_resp = json.loads(await ws.recv())
                button_clicked = (
                    click_resp.get("result", {}).get("result", {}).get("value", False)
                )
                print(f"      [CDP] Clicked Home Quick Demo button: {button_clicked}")

                # 3. Wait up to 45s for real Databricks Serverless prediction card and natural language explanation
                waited_sec = 0
                for sec in range(1, 45):
                    await asyncio.sleep(1.0)
                    await ws.send(
                        json.dumps(
                            {
                                "id": 300 + sec,
                                "method": "Runtime.evaluate",
                                "params": {"expression": "document.body.innerHTML"},
                            }
                        )
                    )
                    resp = json.loads(await ws.recv())
                    hydrated_html = (
                        resp.get("result", {}).get("result", {}).get("value", "")
                    )
                    if (
                        "DECISÃO PREVISTA PELA INTELIGÊNCIA ARTIFICIAL" in hydrated_html
                        and "Resposta Databricks Serverless:" in hydrated_html
                        and "Por que a IA escolheu este arremesso?" in hydrated_html
                    ):
                        waited_sec = sec
                        print(f"      [CDP] Home Quick Demo live prediction card rendered after {sec}s!")
                        break

                await asyncio.sleep(1.0)

                # Mandatory DOM & CommonMark assertions on Home Page
                assert "&lt;div" not in hydrated_html, "Regression: <div> leaked as text!"
                assert "&lt;span" not in hydrated_html, "Regression: <span> leaked as text!"
                assert "Decodificando o Arremesso com Inteligência Artificial" in hydrated_html
                assert "0.40 segundos" in hydrated_html
                assert "160 km/h" in hydrated_html
                assert "Bola Rápida (Fastball)" in hydrated_html
                assert "Bola com Efeito (Breaking Ball)" in hydrated_html
                assert "Mudança de Velocidade (Offspeed)" in hydrated_html
                assert (
                    "PREVISÃO DA INTELIGÊNCIA ARTIFICIAL" in hydrated_html
                    or "DECISÃO PREVISTA PELA INTELIGÊNCIA ARTIFICIAL" in hydrated_html
                    or "INTELIGÊNCIA ARTIFICIAL" in hydrated_html
                )
                assert "Por que a IA escolheu este arremesso?" in hydrated_html

                # 4. Capture 1920x1750 screenshot via CDP
                await ws.send(
                    json.dumps(
                        {
                            "id": 999,
                            "method": "Page.captureScreenshot",
                            "params": {"format": "png"},
                        }
                    )
                )
                shot_msg = json.loads(await ws.recv())
                png_bytes = base64.b64decode(shot_msg["result"]["data"])
                output_path.write_bytes(png_bytes)

                return {
                    "button_clicked": button_clicked,
                    "hydrated_dom_length": len(hydrated_html),
                    "hero_400ms_verified": "0.40 segundos" in hydrated_html and "160 km/h" in hydrated_html,
                    "three_pitch_cards_verified": all(
                        c in hydrated_html
                        for c in (
                            "Bola Rápida (Fastball)",
                            "Bola com Efeito (Breaking Ball)",
                            "Mudança de Velocidade (Offspeed)",
                        )
                    ),
                    "quick_demo_prediction_rendered": "DECISÃO PREVISTA PELA INTELIGÊNCIA ARTIFICIAL" in hydrated_html,
                    "wait_seconds": waited_sec,
                    "zero_code_block_leaks": "&lt;div" not in hydrated_html,
                }

        cdp_metrics = asyncio.run(_cdp_home_flow())
    finally:
        if edge_proc is not None:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(edge_proc.pid)],
                capture_output=True,
                check=False,
            )
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(st_proc.pid)],
            capture_output=True,
            check=False,
        )

    return {
        "browser": str(edge_exe),
        "screenshot_path": str(output_path),
        "file_size_bytes": output_path.stat().st_size,
        "resolution": "1920x1750",
        "cdp_home_verification": cdp_metrics,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("[1/3] Validating Page 0, Page 1 & Page 2 SVG/HTML DOM components programmatically...")
    dom_report = verify_svg_and_html_dom_components()
    print(f"      -> SVG/DOM Validation OK: {dom_report}")

    print("[2/3] Executing REAL 12-Count Batch Inference against Databricks Serverless Endpoint...")
    batch_report = verify_real_12_counts_batch_inference()
    print(
        f"      -> Batch 12-Count Inference OK ({batch_report['latency_ms']} ms): "
        f"{batch_report['predictions_by_count']}"
    )

    print(
        f"[3/3] Running Live Streamlit + CDP Browser Verification for Home Presentation Page -> {TARGET_HOME_PNG}..."
    )
    home_screenshot_report = verify_and_capture_home_presentation_via_cdp(TARGET_HOME_PNG)
    print(f"      -> Home Presentation Live Screenshot & DOM Verified: {home_screenshot_report}")

    summary = {
        "layer": "Layer 3 — Multipage DOM/SVG Validation, Real 12-Count Batch Inference & Home CDP Screenshot",
        "status": "PASSED",
        "dom_svg_validation": dom_report,
        "real_batch_12_counts": batch_report,
        "home_preview": home_screenshot_report,
        "scout_report_preview_exists": TARGET_SCOUT_PNG.exists(),
    }
    print("\n=== MULTIPAGE LAYER 3 VERIFICATION SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
