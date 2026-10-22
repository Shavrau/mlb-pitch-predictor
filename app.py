"""
MLB Pitch Predictor — Statcast Tactical Broadcast Interface (Multipage Entrypoint).

Configures the global Streamlit page layout and routes between:
0. `pages/00_home_apresentacao.py` — Visão Geral & Apresentação Descomplicada (Default)
1. `pages/03_minigame.py` — Minigame Retrô Arcade: Você vs. IA no Montinho
2. `pages/02_scout_report.py` — MLB Scouting Report, Matriz 4x3 de 12 Contagens & Game Plan
3. `pages/04_pitch_vision_lab.py` — Laboratório de Visão Computacional, Pose 3D & Pitch Tipping
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on sys.path for Databricks Apps and container execution
SRC_DIR = str(Path(__file__).resolve().parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import streamlit as st

st.set_page_config(
    page_title="MLB Pitch Predictor | Statcast Tactical AI",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

home_page = st.Page(
    "pages/00_home_apresentacao.py",
    title="Visão Geral & Apresentação",
    icon="🏠",
    url_path="home",
    default=True,
)

minigame_page = st.Page(
    "pages/03_minigame.py",
    title="Minigame: Você vs. IA",
    icon="🎮",
    url_path="minigame",
)

scout_report_page = st.Page(
    "pages/02_scout_report.py",
    title="MLB Scouting Report & Matriz",
    icon="📋",
    url_path="scout",
)

vision_lab_page = st.Page(
    "pages/04_pitch_vision_lab.py",
    title="Laboratório de Visão (Pitch Tipping)",
    icon="🔬",
    url_path="vision-lab",
)

pg = st.navigation(
    {
        "⚾ Módulos da Aplicação": [
            home_page,
            minigame_page,
            scout_report_page,
            vision_lab_page,
        ]
    }
)
pg.run()
