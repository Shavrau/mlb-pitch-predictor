"""Baseball ML Pitch Predictor package."""

from __future__ import annotations

import pandas as _pd  # noqa: F401 — Pre-warm pandas in sys.modules for fast Streamlit AppTest startup

from pitch_predictor import (
    scenarios,
    scout_analytics,
    serving_client,
    ui_components,
)

__version__ = "0.1.0"
__all__ = [
    "scenarios",
    "scout_analytics",
    "serving_client",
    "ui_components",
]
