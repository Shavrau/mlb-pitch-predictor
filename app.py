"""

MLB Pitch Predictor — Broadcast Editorial Dual-Mode Interface (Multipage Entrypoint).

Configures the global Streamlit layout, Dual Mode selector (Dark Broadcast `#0A0B0E` default

+ Light Editorial `#F8FAFC`), and routes between:

0. `pages/00_home_apresentacao.py` — Visão Geral & Apresentação Executiva (Default)

1. `pages/03_minigame.py` — Minigame Console HUD: Você vs. IA no Montinho

2. `pages/02_scout_report.py` — MLB Scouting Report, Matriz 4x3 de 12 Contagens & Game Plan

3. `pages/04_pitch_vision_lab.py` — Laboratório de Visão Computacional, Pose 3D & Pitch Tipping

4. `pages/06_pitch_tunneling.py` — Disfarce de Trajetória 3D (Pitch Tunneling & Commitment Point)

5. `pages/05_mlops_hub.py` — Databricks MLOps, Linhagem Unity Catalog & Governança Serverless

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

if "ui_theme_mode" not in st.session_state or st.session_state.ui_theme_mode not in (
    "diamond",
    "ballpark",
    "light",
):
    st.session_state.ui_theme_mode = "diamond"

with st.sidebar:
    st.caption("MLB STATCAST • BROADCAST UI")

    theme_default = (
        "Light Editorial"
        if st.session_state.ui_theme_mode == "light"
        else "Classic Diamond"
    )

    appearance_choice = st.segmented_control(
        "Aparência",
        options=["Classic Diamond", "Light Editorial"],
        default=theme_default,
        selection_mode="single",
        key="ui_appearance_toggle",
    )

    if appearance_choice == "Light Editorial":
        st.session_state.ui_theme_mode = "light"

    else:
        st.session_state.ui_theme_mode = "diamond"

    if "ballpark_audio_enabled" not in st.session_state:
        st.session_state.ballpark_audio_enabled = True

    ballpark_audio = st.toggle(
        "Som do Ballpark (Efeitos de Estádio)",
        value=st.session_state.ballpark_audio_enabled,
        key="ballpark_audio_toggle",
        help="Sintetiza via Web Audio o estalo da luva de couro recebendo uma bola rápida de 102 mph ao acionar botões de ação.",
    )
    st.session_state.ballpark_audio_enabled = ballpark_audio

    from pitch_predictor.ui_components import (
        get_custom_css,
        render_ballpark_audio_listener_html,
    )

    st.markdown(
        get_custom_css(theme_mode=st.session_state.ui_theme_mode),
        unsafe_allow_html=True,
    )
    if ballpark_audio:
        st.markdown(
            render_ballpark_audio_listener_html(enabled=True),
            unsafe_allow_html=True,
        )

    st.markdown(
        """

        <div style="background:var(--pp-surface);border:1px solid var(--pp-border);border-radius:6px;padding:8px 10px;margin-top:10px;font-family:'JetBrains Mono',monospace;font-size:9.5px;color:var(--pp-muted);">

          <div style="color:var(--pp-text);font-weight:600;letter-spacing:0.02em;margin-bottom:2px;">MLB STATCAST NETWORK</div>

          <div>5 Astros • 33 Marcos 3D</div>

          <div style="color:var(--pp-clay,#C85A17);font-weight:700;margin-top:3px;">● TELEMETRIA ATIVA</div>

        </div>

        """,
        unsafe_allow_html=True,
    )

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

tunneling_page = st.Page(
    "pages/06_pitch_tunneling.py",
    title="Disfarce de Trajetória 3D",
    icon=":material/alt_route:",
    url_path="tunneling",
)

mlops_page = st.Page(
    "pages/05_mlops_hub.py",
    title="Databricks MLOps & Governança",
    icon=":material/monitoring:",
    url_path="mlops",
)

pg = st.navigation(
    {
        "MLB Statcast • Módulos": [
            home_page,
            minigame_page,
            scout_report_page,
            vision_lab_page,
            tunneling_page,
            mlops_page,
        ]
    }
)

pg.run()
