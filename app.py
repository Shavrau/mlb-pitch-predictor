"""
MLB Pitch Predictor — Broadcast Editorial Dual-Mode Interface (Multipage Entrypoint).

Configures the global Streamlit layout, Dual Mode selector (Dark Broadcast `#0A0B0E` default
+ Light Editorial `#F8FAFC`), and routes between:
0. `pages/00_home_apresentacao.py` — Visão Geral & Apresentação Executiva (Default)
1. `pages/03_minigame.py` — Minigame Console HUD: Você vs. IA no Montinho
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
    page_icon=":material/sports_baseball:",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "ui_theme_mode" not in st.session_state or st.session_state.ui_theme_mode not in ("dark", "light"):
    st.session_state.ui_theme_mode = "dark"

with st.sidebar:
    st.caption("MLB STATCAST • BROADCAST UI")
    appearance_choice = st.segmented_control(
        "Aparência",
        options=["Dark Broadcast", "Light Editorial"],
        default="Light Editorial" if st.session_state.ui_theme_mode == "light" else "Dark Broadcast",
        selection_mode="single",
        key="ui_appearance_toggle",
    )
    if appearance_choice == "Light Editorial":
        st.session_state.ui_theme_mode = "light"
    else:
        st.session_state.ui_theme_mode = "dark"

home_page = st.Page(
    "pages/00_home_apresentacao.py",
    title="Visão Geral & Apresentação",
    icon=":material/dashboard:",
    url_path="home",
    default=True,
)

minigame_page = st.Page(
    "pages/03_minigame.py",
    title="Minigame: Você vs. IA",
    icon=":material/sports_esports:",
    url_path="minigame",
)

scout_report_page = st.Page(
    "pages/02_scout_report.py",
    title="MLB Scouting Report & Matriz",
    icon=":material/analytics:",
    url_path="scout",
)

vision_lab_page = st.Page(
    "pages/04_pitch_vision_lab.py",
    title="Laboratório de Visão (Pitch Tipping)",
    icon=":material/biotech:",
    url_path="vision-lab",
)

pg = st.navigation(
    {
        "MLB Statcast • Módulos": [
            home_page,
            minigame_page,
            scout_report_page,
            vision_lab_page,
        ]
    }
)
pg.run()
