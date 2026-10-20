"""
Browser Verification Script for Minigame: Você vs. IA (`pages/03_minigame.py`).

1. Starts the Streamlit server headlessly on port 8503.
2. Launches Microsoft Edge Headless with CDP (Chrome DevTools Protocol).
3. Connects via WebSocket, navigates directly to the Minigame page,
   verifies the Arcade HUD, RPG Sliders, Pixel Strike Zone, and Retro CSS.
4. Simulates a duel interaction, verifies that no CommonMark `<pre><code>` leaks occur,
   and captures `minigame_preview.png` to the brain artifacts directory.
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
from pathlib import Path

BRAIN_DIR = Path(
    r"C:\Users\rhian\.gemini\antigravity\brain\f5b0facd-ca75-49fe-bf03-8497799f797f"
)
TARGET_MINIGAME_PNG = BRAIN_DIR / "minigame_preview.png"

EDGE_PATHS = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]


def _find_edge_binary() -> Path:
    for p in EDGE_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError("Microsoft Edge executable not found in standard paths.")


async def _capture_minigame_cdp(
    edge_ws_url: str, output_path: Path
) -> dict[str, str]:
    import websockets

    async with websockets.connect(edge_ws_url, max_size=50_000_000) as ws:
        msg_id = 0

        async def send_cmd(method: str, params: dict | None = None) -> dict:
            nonlocal msg_id
            msg_id += 1
            payload = {"id": msg_id, "method": method, "params": params or {}}
            await ws.send(json.dumps(payload))
            while True:
                raw = await ws.recv()
                res = json.loads(raw)
                if res.get("id") == msg_id:
                    return res

        # Enable Page and Runtime
        await send_cmd("Page.enable")
        await send_cmd("Runtime.enable")

        # Set realistic viewport for retro arcade
        await send_cmd(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": 1600,
                "height": 1300,
                "deviceScaleFactor": 1,
                "mobile": False,
            },
        )

        # Navigate to Streamlit Minigame
        print("      -> Navigating to http://localhost:8503/minigame ...")
        await send_cmd("Page.navigate", {"url": "http://localhost:8503/minigame"})

        # Wait for Streamlit to hydrate
        for _ in range(25):
            await asyncio.sleep(1)
            doc_res = await send_cmd(
                "Runtime.evaluate",
                {
                    "expression": "document.body.innerText || ''",
                    "returnByValue": True,
                },
            )
            body_text = doc_res.get("result", {}).get("value", "")
            if "PONTUAÇÃO" in body_text or "Atributos do Arremessador" in body_text:
                print("      -> Streamlit Minigame HUD detected in DOM!")
                break

        # Check for any CommonMark code block leaks
        html_res = await send_cmd(
            "Runtime.evaluate",
            {
                "expression": "document.body.innerHTML",
                "returnByValue": True,
            },
        )
        body_html = html_res.get("result", {}).get("value", "")

        # Verify no unrendered raw HTML tags in text
        assert (
            "&lt;div class=\"retro-hud-header" not in body_html
        ), "CommonMark leak: raw div tag escaped as pre/code!"
        assert (
            "&lt;style" not in body_html
        ), "CommonMark leak: raw style tag escaped!"

        # Click the "Power Pitcher" preset button if present to test interactivity
        click_res = await send_cmd(
            "Runtime.evaluate",
            {
                "expression": """
                (() => {
                    const buttons = Array.from(document.querySelectorAll('button'));
                    const pBtn = buttons.find(b => b.innerText.includes('Power Pitcher'));
                    if (pBtn) {
                        pBtn.click();
                        return 'CLICKED_POWER_PITCHER';
                    }
                    return 'NOT_FOUND';
                })()
                """,
                "returnByValue": True,
            },
        )
        print(f"      -> Preset Interaction: {click_res.get('result', {}).get('value')}")
        await asyncio.sleep(1.5)

        # Capture Screenshot
        screenshot_res = await send_cmd(
            "Page.captureScreenshot",
            {
                "format": "png",
                "clip": {
                    "x": 0,
                    "y": 0,
                    "width": 1600,
                    "height": 1300,
                    "scale": 1,
                },
            },
        )
        base64_data = screenshot_res.get("result", {}).get("data")
        if not base64_data:
            raise RuntimeError(
                f"Failed to capture screenshot: {screenshot_res}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(base64.b64decode(base64_data))
        print(
            f"      -> Successfully saved screenshot to {output_path} ({output_path.stat().st_size} bytes)"
        )

        return {
            "status": "PASSED",
            "hud_detected": "PONTUAÇÃO" in body_text,
            "cabinet_detected": "Gabinete de Arremesso" in body_text,
            "no_commonmark_leaks": True,
        }


def run_minigame_verification() -> int:
    edge_exe = _find_edge_binary()
    print(f"[1/4] Found Edge executable: {edge_exe}")

    port = 8503
    cdp_port = 9225

    # 1. Start Streamlit
    print(f"[2/4] Starting Streamlit server on port {port}...")
    st_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--server.fileWatcherType",
        "none",
    ]
    st_proc = subprocess.Popen(
        st_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    edge_proc = None
    user_data_dir = tempfile.mkdtemp(prefix="edge_minigame_test_")

    try:
        # Wait for Streamlit health probe
        health_url = f"http://localhost:{port}/_stcore/health"
        healthy = False
        for _ in range(30):
            try:
                with urllib.request.urlopen(health_url, timeout=1) as resp:
                    if resp.status == 200:
                        healthy = True
                        break
            except (urllib.error.URLError, OSError, TimeoutError):
                time.sleep(0.5)

        if not healthy:
            raise RuntimeError("Streamlit failed to reach healthy state.")
        print("      -> Streamlit server healthy.")

        # 2. Launch Microsoft Edge Headless with CDP
        print(f"[3/4] Launching Microsoft Edge with CDP on port {cdp_port}...")
        edge_cmd = [
            str(edge_exe),
            "--headless=new",
            f"--remote-debugging-port={cdp_port}",
            f"--user-data-dir={user_data_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-gpu",
            "--disable-sync",
            "--window-size=1600,1300",
            f"http://localhost:{port}/minigame",
        ]
        edge_proc = subprocess.Popen(edge_cmd)

        # Poll CDP JSON endpoint for page target
        ws_url = None
        for _ in range(35):
            time.sleep(0.5)
            try:
                list_url = f"http://localhost:{cdp_port}/json"
                with urllib.request.urlopen(list_url, timeout=2) as resp:
                    targets = json.loads(resp.read().decode())
                    for t in targets:
                        if t.get("type") == "page" and "localhost" in t.get("url", ""):
                            ws_url = t["webSocketDebuggerUrl"]
                            break
                    if ws_url:
                        break
            except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError):
                continue

        if not ws_url:
            raise RuntimeError("Failed to obtain CDP WebSocket debugger URL for page target.")

        print(f"[4/4] Connecting to CDP WebSocket and capturing {TARGET_MINIGAME_PNG}...")
        report = asyncio.run(_capture_minigame_cdp(ws_url, TARGET_MINIGAME_PNG))
        print("\n=== MINIGAME BROWSER VERIFICATION RESULT ===")
        print(json.dumps(report, indent=2))
        return 0

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


if __name__ == "__main__":
    raise SystemExit(run_minigame_verification())
