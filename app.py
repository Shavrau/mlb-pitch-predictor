"""
MLB Pitch Predictor — Statcast Tactical Broadcast Interface (Multipage Entrypoint).

Configures the global Streamlit page layout and routes between:
0. `pages/00_home_apresentacao.py` — Visão Geral & Apresentação Descomplicada (Default)
1. `pages/01_live_predictor.py` — Simulador Tático em Tempo Real & Pitch-by-Pitch Tracker
2. `pages/02_scout_report.py` — MLB Scouting Report, Matriz 4x3 de 12 Contagens & Game Plan
"""

from __future__ import annotations

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

live_predictor_page = st.Page(
    "pages/01_live_predictor.py",
    title="Simulador Tático (Ao Vivo)",
    icon="⚾",
    url_path="simulador",
)

scout_report_page = st.Page(
    "pages/02_scout_report.py",
    title="MLB Scouting Report & Matriz",
    icon="📋",
    url_path="scout",
)

pg = st.navigation(
    {
        "⚾ Módulos da Aplicação": [
            home_page,
            minigame_page,
            live_predictor_page,
            scout_report_page,
        ]
    }
)
pg.run()
