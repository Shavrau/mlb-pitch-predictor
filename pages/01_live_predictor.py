"""
Page 01: MLB Pitch Predictor — Live Tactical Broadcast Simulator (Single-Pitch & At-Bat Tracker).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from pitch_predictor.scenarios import (
    PITCH_TYPES_BY_CATEGORY,
    PITCHER_ARCHETYPES,
    TACTICAL_SCENARIOS,
    advance_at_bat_state,
)
from pitch_predictor.serving_client import (
    DEFAULT_ENDPOINT_NAME,
    PitchServingClient,
    build_rest_payload,
    derive_tactical_features,
    normalize_priors,
)
from pitch_predictor.ui_components import (
    STATCAST_ZONES,
    get_custom_css,
    render_baseball_diamond_svg,
    render_prediction_card_html,
    render_scoreboard_banner_html,
    render_strike_zone_svg,
)


@st.cache_resource(show_spinner=False)
def get_serving_client(endpoint_name: str = DEFAULT_ENDPOINT_NAME) -> PitchServingClient:
    """Return a cached instance of the Databricks Serverless Model Serving client."""
    return PitchServingClient(endpoint_name=endpoint_name)


def init_session_state() -> None:
    """Initialize all reactive tactical state variables in `st.session_state`."""
    default_scenario_key = next(iter(TACTICAL_SCENARIOS.keys()))
    default_inputs = TACTICAL_SCENARIOS[default_scenario_key]["inputs"]

    defaults: dict[str, Any] = {
        "selected_scenario": default_scenario_key,
        "selected_archetype": "Média da Liga MLB (League Average)",
        "prev_pitch_category": default_inputs["prev_pitch_category"],
        "prev_pitch_type": default_inputs["prev_pitch_type"],
        "prev_pitch_zone": default_inputs["prev_pitch_zone"],
        "prev_pitch_result": default_inputs["prev_pitch_result"],
        "p_throws": default_inputs["p_throws"],
        "stand": default_inputs["stand"],
        "balls": default_inputs["balls"],
        "strikes": default_inputs["strikes"],
        "outs_when_up": default_inputs["outs_when_up"],
        "inning": default_inputs["inning"],
        "inning_topbot": default_inputs["inning_topbot"],
        "score_diff": default_inputs["score_diff"],
        "on_1b_flag": bool(default_inputs["on_1b_flag"]),
        "on_2b_flag": bool(default_inputs["on_2b_flag"]),
        "on_3b_flag": bool(default_inputs["on_3b_flag"]),
        "n_thruorder_pitcher": default_inputs["n_thruorder_pitcher"],
        "fb_pct_slider": round(default_inputs["pitcher_fb_prior"] * 100.0, 1),
        "bb_pct_slider": round(default_inputs["pitcher_bb_prior"] * 100.0, 1),
        "os_pct_slider": round(default_inputs["pitcher_os_prior"] * 100.0, 1),
        "at_bat_history": [],
        "last_prediction_result": None,
        "last_at_bat_event": None,
        "endpoint_status_cache": None,
    }

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def apply_scenario_to_state(scenario_name: str) -> None:
    """Load a preset MLB tactical scenario into `st.session_state`."""
    scenario = TACTICAL_SCENARIOS.get(scenario_name)
    if not scenario:
        return
    inputs = scenario["inputs"]
    st.session_state.prev_pitch_category = inputs["prev_pitch_category"]
    st.session_state.prev_pitch_type = inputs["prev_pitch_type"]
    st.session_state.prev_pitch_zone = int(inputs["prev_pitch_zone"])
    st.session_state.prev_pitch_result = inputs["prev_pitch_result"]
    st.session_state.p_throws = inputs["p_throws"]
    st.session_state.stand = inputs["stand"]
    st.session_state.balls = int(inputs["balls"])
    st.session_state.strikes = int(inputs["strikes"])
    st.session_state.outs_when_up = int(inputs["outs_when_up"])
    st.session_state.inning = int(inputs["inning"])
    st.session_state.inning_topbot = inputs["inning_topbot"]
    st.session_state.score_diff = int(inputs["score_diff"])
    st.session_state.on_1b_flag = bool(inputs["on_1b_flag"])
    st.session_state.on_2b_flag = bool(inputs["on_2b_flag"])
    st.session_state.on_3b_flag = bool(inputs["on_3b_flag"])
    st.session_state.n_thruorder_pitcher = int(inputs["n_thruorder_pitcher"])
    st.session_state.fb_pct_slider = round(float(inputs["pitcher_fb_prior"]) * 100.0, 1)
    st.session_state.bb_pct_slider = round(float(inputs["pitcher_bb_prior"]) * 100.0, 1)
    st.session_state.os_pct_slider = round(float(inputs["pitcher_os_prior"]) * 100.0, 1)
    st.session_state.at_bat_history = []
    st.session_state.last_at_bat_event = f"Cenário carregado: {scenario_name}"


def apply_archetype_to_state(archetype_name: str) -> None:
    """Apply a pitcher archetype's repertoire priors and handedness to `st.session_state`."""
    profile = PITCHER_ARCHETYPES.get(archetype_name)
    if not profile:
        return
    st.session_state.fb_pct_slider = round(float(profile["pitcher_fb_prior"]) * 100.0, 1)
    st.session_state.bb_pct_slider = round(float(profile["pitcher_bb_prior"]) * 100.0, 1)
    st.session_state.os_pct_slider = round(float(profile["pitcher_os_prior"]) * 100.0, 1)
    if "p_throws" in profile:
        st.session_state.p_throws = profile["p_throws"]


def select_zone_callback(zone_id: int) -> None:
    """Update `prev_pitch_zone` from the interactive Statcast Strike Zone button matrix."""
    st.session_state.prev_pitch_zone = zone_id
    if zone_id == 0:
        st.session_state.prev_pitch_category = "First Pitch"
        st.session_state.prev_pitch_type = "None"
        st.session_state.prev_pitch_result = "First Pitch"
    else:
        if st.session_state.prev_pitch_category == "First Pitch":
            st.session_state.prev_pitch_category = "Fastball"
            st.session_state.prev_pitch_type = "FF"
            st.session_state.prev_pitch_result = "S" if zone_id in range(1, 10) else "B"


def set_bases_callback(b1: bool, b2: bool, b3: bool) -> None:
    """Quick helper to clear or load all bases."""
    st.session_state.on_1b_flag = b1
    st.session_state.on_2b_flag = b2
    st.session_state.on_3b_flag = b3


def build_raw_inputs_from_state() -> dict[str, Any]:
    """Collect current UI widgets from `st.session_state` into a raw_inputs dictionary."""
    fb_prior, bb_prior, os_prior = normalize_priors(
        st.session_state.fb_pct_slider,
        st.session_state.bb_pct_slider,
        st.session_state.os_pct_slider,
    )
    return {
        "prev_pitch_category": st.session_state.prev_pitch_category,
        "prev_pitch_type": st.session_state.prev_pitch_type,
        "prev_pitch_zone": int(st.session_state.prev_pitch_zone),
        "prev_pitch_result": st.session_state.prev_pitch_result,
        "p_throws": st.session_state.p_throws,
        "stand": st.session_state.stand,
        "balls": int(st.session_state.balls),
        "strikes": int(st.session_state.strikes),
        "outs_when_up": int(st.session_state.outs_when_up),
        "inning": int(st.session_state.inning),
        "inning_topbot": st.session_state.inning_topbot,
        "score_diff": int(st.session_state.score_diff),
        "on_1b_flag": 1 if st.session_state.on_1b_flag else 0,
        "on_2b_flag": 1 if st.session_state.on_2b_flag else 0,
        "on_3b_flag": 1 if st.session_state.on_3b_flag else 0,
        "n_thruorder_pitcher": int(st.session_state.n_thruorder_pitcher),
        "pitcher_fb_prior": fb_prior,
        "pitcher_bb_prior": bb_prior,
        "pitcher_os_prior": os_prior,
    }


def simulate_pitch_outcome_callback(
    outcome: str, sim_category: str, sim_type: str, sim_zone: int
) -> None:
    """Advance the at-bat state machine after a simulated pitch outcome."""
    current_raw = build_raw_inputs_from_state()
    count_before = f"{current_raw['balls']}-{current_raw['strikes']}"

    next_state = advance_at_bat_state(
        current_state=current_raw,
        pitch_category=sim_category,
        pitch_type=sim_type,
        pitch_zone=sim_zone,
        outcome=outcome,
    )

    event_label = next_state.get("at_bat_event", outcome)
    pitch_num = len(st.session_state.at_bat_history) + 1
    st.session_state.at_bat_history.append(
        {
            "#": pitch_num,
            "Contagem": count_before,
            "Categoria": sim_category,
            "Tipo": sim_type,
            "Zona": f"Z{sim_zone}",
            "Resultado": event_label,
            "Nova Contagem": f"{next_state['balls']}-{next_state['strikes']}",
        }
    )

    st.session_state.balls = int(next_state["balls"])
    st.session_state.strikes = int(next_state["strikes"])
    st.session_state.outs_when_up = int(next_state["outs_when_up"])
    st.session_state.prev_pitch_category = next_state["prev_pitch_category"]
    st.session_state.prev_pitch_type = next_state["prev_pitch_type"]
    st.session_state.prev_pitch_zone = int(next_state["prev_pitch_zone"])
    st.session_state.prev_pitch_result = next_state["prev_pitch_result"]
    st.session_state.last_at_bat_event = event_label


# Initialize session state and inject custom broadcast CSS
init_session_state()
st.markdown(get_custom_css(), unsafe_allow_html=True)

client = get_serving_client()
raw_inputs = build_raw_inputs_from_state()
current_features = derive_tactical_features(raw_inputs)

# ---------------------------------------------------------------------------
# Sidebar: Presets Táticos, Arquétipos da MLB & Status do Endpoint
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚾ Central Tática MLB")
    st.caption("Configure rapidamente situações reais de jogo e perfis de arremessadores.")

    st.markdown("### 🎬 Cenários de Jogo (Presets)")
    chosen_scenario = st.selectbox(
        "Escolha uma situação tática:",
        options=list(TACTICAL_SCENARIOS.keys()),
        key="selected_scenario",
        on_change=lambda: apply_scenario_to_state(st.session_state.selected_scenario),
    )
    st.info(TACTICAL_SCENARIOS[chosen_scenario]["description"])
    if st.button("🔄 Recarregar Cenário Selecionado", use_container_width=True):
        apply_scenario_to_state(chosen_scenario)
        st.rerun()

    st.divider()

    st.markdown("### 🏆 Arquétipos de Arremessador")
    chosen_archetype = st.selectbox(
        "Perfil de Repertório (Priors):",
        options=list(PITCHER_ARCHETYPES.keys()),
        key="selected_archetype",
        on_change=lambda: apply_archetype_to_state(st.session_state.selected_archetype),
    )
    arch_info = PITCHER_ARCHETYPES[chosen_archetype]
    st.caption(f"**Arsenal:** {arch_info['signature_arsenal']}")
    st.caption(arch_info["description"])

    st.divider()

    st.markdown("### 🔌 Databricks Model Serving")
    st.code(client.endpoint_name, language="text")
    if st.button("📡 Verificar Status do Endpoint", use_container_width=True):
        with st.spinner("Consultando estado do endpoint no Databricks..."):
            st.session_state.endpoint_status_cache = client.check_endpoint_status()

    if st.session_state.endpoint_status_cache:
        ep_stat = st.session_state.endpoint_status_cache
        if ep_stat.get("online"):
            st.success(f"Online • Estado: {ep_stat.get('ready')}")
        else:
            st.warning(f"Status: {ep_stat.get('ready')}")

# ---------------------------------------------------------------------------
# Top Electronic Scoreboard Header
# ---------------------------------------------------------------------------
st.markdown(
    render_scoreboard_banner_html(current_features, client.endpoint_name),
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Main 3-Column Tactical Console
# ---------------------------------------------------------------------------
col_game, col_zone, col_pred = st.columns([1.05, 1.15, 1.2], gap="medium")

# ===========================================================================
# COLUNA 1: Situação de Jogo, Diamante de Bases & Platoon
# ===========================================================================
with col_game:
    with st.container(border=True):
        st.markdown(
            '<div class="section-header">🏟️ 1. Situação de Jogo & Placar</div>',
            unsafe_allow_html=True,
        )
        c_inn, c_half = st.columns(2)
        with c_inn:
            st.number_input("Entrada (Inning)", min_value=1, max_value=15, step=1, key="inning")
        with c_half:
            st.selectbox(
                "Parte da Entrada",
                options=["Top", "Bot"],
                format_func=lambda x: "▲ Alta (Visitante)" if x == "Top" else "▼ Baixa (Mandante)",
                key="inning_topbot",
            )

        c_outs, c_score = st.columns(2)
        with c_outs:
            st.selectbox(
                "Eliminações (Outs)",
                options=[0, 1, 2],
                format_func=lambda x: f"{x} Out (Eliminação)" if x == 1 else f"{x} Outs (Eliminações)",
                key="outs_when_up",
            )
        with c_score:
            st.slider(
                "Diferença de Placar (Arremessador)",
                min_value=-10,
                max_value=10,
                step=1,
                key="score_diff",
                help="Positivo = time do arremessador vencendo; Negativo = perdendo.",
            )

    with st.container(border=True):
        st.markdown(
            '<div class="section-header">💎 2. Diamante de Bases Interativo</div>',
            unsafe_allow_html=True,
        )
        b_col1, b_col2, b_col3 = st.columns(3)
        with b_col1:
            st.checkbox("1ª Base (1B)", key="on_1b_flag")
        with b_col2:
            st.checkbox("2ª Base (2B)", key="on_2b_flag")
        with b_col3:
            st.checkbox("3ª Base (3B)", key="on_3b_flag")

        q_col1, q_col2 = st.columns(2)
        with q_col1:
            st.button(
                "🧹 Limpar Bases",
                on_click=set_bases_callback,
                args=(False, False, False),
                use_container_width=True,
            )
        with q_col2:
            st.button(
                "🔥 Lotar Bases",
                on_click=set_bases_callback,
                args=(True, True, True),
                use_container_width=True,
            )

        st.markdown(
            render_baseball_diamond_svg(
                on_1b=1 if st.session_state.on_1b_flag else 0,
                on_2b=1 if st.session_state.on_2b_flag else 0,
                on_3b=1 if st.session_state.on_3b_flag else 0,
                outs=int(st.session_state.outs_when_up),
            ),
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        st.markdown(
            '<div class="section-header">🧤 3. Confronto de Mãos (Platoon) & Passagem na Ordem</div>',
            unsafe_allow_html=True,
        )
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.radio(
                "Mão do Arremessador",
                options=["R", "L"],
                format_func=lambda x: "Destro (RHP)" if x == "R" else "Canhoto (LHP)",
                horizontal=True,
                key="p_throws",
            )
        with p_col2:
            st.radio(
                "Lado do Rebatedor",
                options=["R", "L"],
                format_func=lambda x: "Destro (RHB)" if x == "R" else "Canhoto (LHB)",
                horizontal=True,
                key="stand",
            )

        st.select_slider(
            "Passagem pela Ordem de Rebatedores (Times Through the Order)",
            options=[1, 2, 3, 4],
            format_func=lambda x: f"{x}ª Passagem na Ordem",
            key="n_thruorder_pitcher",
        )

# ===========================================================================
# COLUNA 2: Contagem, Zona Statcast Interativa & Arsenal (Priors)
# ===========================================================================
with col_zone:
    with st.container(border=True):
        st.markdown(
            '<div class="section-header">🔢 4. Contagem do Duelo (Balls & Strikes)</div>',
            unsafe_allow_html=True,
        )
        cnt_col1, cnt_col2 = st.columns(2)
        with cnt_col1:
            st.radio(
                "Balls (Bolas)",
                options=[0, 1, 2, 3],
                horizontal=True,
                key="balls",
            )
        with cnt_col2:
            st.radio(
                "Strikes",
                options=[0, 1, 2],
                horizontal=True,
                key="strikes",
            )

    with st.container(border=True):
        st.markdown(
            '<div class="section-header">🎯 5. Zona Statcast & Arremesso Anterior</div>',
            unsafe_allow_html=True,
        )

        prev_cat = st.selectbox(
            "Categoria do Arremesso Anterior",
            options=["First Pitch", "Fastball", "Breaking Ball", "Offspeed"],
            key="prev_pitch_category",
        )

        if prev_cat == "First Pitch":
            st.session_state.prev_pitch_type = "None"
            st.session_state.prev_pitch_zone = 0
            st.session_state.prev_pitch_result = "First Pitch"
            st.caption("ℹ️ Em **First Pitch**, tipo = `None`, zona = `0` e resultado = `First Pitch`.")
        else:
            available_types = PITCH_TYPES_BY_CATEGORY.get(prev_cat, [("FF", "FF")])
            type_codes = [code for code, _ in available_types]
            type_labels = {code: label for code, label in available_types}
            if st.session_state.prev_pitch_type not in type_codes:
                st.session_state.prev_pitch_type = type_codes[0]
            if st.session_state.prev_pitch_zone == 0:
                st.session_state.prev_pitch_zone = 5

            t_col1, t_col2 = st.columns([1.4, 1.0])
            with t_col1:
                st.selectbox(
                    "Tipo Específico (Statcast)",
                    options=type_codes,
                    format_func=lambda c: type_labels.get(c, c),
                    key="prev_pitch_type",
                )
            with t_col2:
                if st.session_state.prev_pitch_result == "First Pitch":
                    st.session_state.prev_pitch_result = "S"
                st.selectbox(
                    "Resultado Anterior",
                    options=["S", "B", "X"],
                    format_func=lambda r: {
                        "S": "S — Strike / Foul (Falta)",
                        "B": "B — Ball (Bola Fora)",
                        "X": "X — Bola em Jogo (In Play)",
                    }.get(r, r),
                    key="prev_pitch_result",
                )

        # Visual SVG Strike Zone
        st.markdown(
            render_strike_zone_svg(
                selected_zone=int(st.session_state.prev_pitch_zone),
                prev_category=st.session_state.prev_pitch_category,
            ),
            unsafe_allow_html=True,
        )

        st.caption("**Clique para selecionar a Zona Statcast:**")
        st.button(
            "0 • First Pitch",
            key="btn_z0",
            type="primary" if st.session_state.prev_pitch_zone == 0 else "secondary",
            on_click=select_zone_callback,
            args=(0,),
            use_container_width=True,
        )
        # Top Chase Row (11, 12)
        ch_top1, ch_top2 = st.columns(2)
        with ch_top1:
            st.button(
                "↖ Z11",
                key="btn_z11",
                type="primary" if st.session_state.prev_pitch_zone == 11 else "secondary",
                on_click=select_zone_callback,
                args=(11,),
                use_container_width=True,
            )
        with ch_top2:
            st.button(
                "↗ Z12",
                key="btn_z12",
                type="primary" if st.session_state.prev_pitch_zone == 12 else "secondary",
                on_click=select_zone_callback,
                args=(12,),
                use_container_width=True,
            )
        # 3x3 Strike Zone Rows (1..9)
        for row_zones in ((1, 2, 3), (4, 5, 6), (7, 8, 9)):
            z_cols = st.columns(3)
            for idx, z_num in enumerate(row_zones):
                with z_cols[idx]:
                    st.button(
                        f"{z_num}",
                        key=f"btn_z{z_num}",
                        type="primary" if st.session_state.prev_pitch_zone == z_num else "secondary",
                        on_click=select_zone_callback,
                        args=(z_num,),
                        use_container_width=True,
                    )
        # Bottom Chase Row (13, 14)
        ch_bot1, ch_bot2 = st.columns(2)
        with ch_bot1:
            st.button(
                "↙ Z13",
                key="btn_z13",
                type="primary" if st.session_state.prev_pitch_zone == 13 else "secondary",
                on_click=select_zone_callback,
                args=(13,),
                use_container_width=True,
            )
        with ch_bot2:
            st.button(
                "↘ Z14",
                key="btn_z14",
                type="primary" if st.session_state.prev_pitch_zone == 14 else "secondary",
                on_click=select_zone_callback,
                args=(14,),
                use_container_width=True,
            )

    with st.container(border=True):
        st.markdown(
            '<div class="section-header">📊 6. Arsenal & Frequências Históricas (Priors)</div>',
            unsafe_allow_html=True,
        )
        st.slider("⚡ Fastball Prior (%)", 0.0, 100.0, step=1.0, key="fb_pct_slider")
        st.slider("🌀 Breaking Ball Prior (%)", 0.0, 100.0, step=1.0, key="bb_pct_slider")
        st.slider("🛑 Offspeed Prior (%)", 0.0, 100.0, step=1.0, key="os_pct_slider")

        fb_n, bb_n, os_n = normalize_priors(
            st.session_state.fb_pct_slider,
            st.session_state.bb_pct_slider,
            st.session_state.os_pct_slider,
        )
        st.caption(
            f"**Proporção Normalizada do Arsenal (Soma = 100%):** "
            f"⚡ FB `{fb_n * 100:.1f}%` • 🌀 BB `{bb_n * 100:.1f}%` • 🛑 OS `{os_n * 100:.1f}%`"
        )

# ===========================================================================
# COLUNA 3: Predição Tática em Tempo Real & Simulador Sequencial Pitch-by-Pitch
# ===========================================================================
with col_pred:
    # Refresh derived features after all widget bindings
    raw_inputs = build_raw_inputs_from_state()
    current_features = derive_tactical_features(raw_inputs)

    with st.container(border=True):
        st.markdown(
            '<div class="section-header">🚀 7. Inferência Tática Serverless</div>',
            unsafe_allow_html=True,
        )

        predict_clicked = st.button(
            "🎯 Prever Próximo Arremesso",
            type="primary",
            use_container_width=True,
        )

        if predict_clicked:
            with st.spinner(
                "⏳ Conectando ao Databricks Serverless Model Serving... "
                "(Acordando contêiner caso em scale-to-zero)"
            ):
                try:
                    result = client.predict(raw_inputs)
                    st.session_state.last_prediction_result = result
                except Exception as exc:  # noqa: BLE001
                    st.error(
                        f"⚠️ Não foi possível obter resposta do endpoint `{client.endpoint_name}`: {exc}"
                    )

        if st.session_state.last_prediction_result:
            pred_res = st.session_state.last_prediction_result
            st.markdown(
                render_prediction_card_html(
                    prediction=pred_res["prediction"],
                    features=pred_res["features"],
                    latency_ms=pred_res["latency_ms"],
                ),
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "👆 Configure a situação de jogo ou escolha um **Cenário Tático** na barra lateral "
                "e clique em **🎯 Prever Próximo Arremesso** para consultar o modelo em tempo real."
            )

    with st.container(border=True):
        st.markdown(
            '<div class="section-header">⚾ 8. Simulador Sequencial (Pitch-by-Pitch Tracker)</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Evolua o duelo arremesso a arremesso aplicando as regras oficiais da MLB "
            "(2-strike foul, Strikeout, Walk e bola em jogo)."
        )

        active_pred_cat = (
            st.session_state.last_prediction_result["prediction"]
            if st.session_state.last_prediction_result
            else "Fastball"
        )
        default_sim_zone = 5 if active_pred_cat == "Fastball" else (14 if active_pred_cat == "Breaking Ball" else 8)

        sim_c1, sim_c2 = st.columns(2)
        with sim_c1:
            sim_cat = st.selectbox(
                "Arremesso Executado",
                options=["Fastball", "Breaking Ball", "Offspeed"],
                index=["Fastball", "Breaking Ball", "Offspeed"].index(active_pred_cat)
                if active_pred_cat in ["Fastball", "Breaking Ball", "Offspeed"]
                else 0,
                key="sim_pitch_cat_select",
            )
        with sim_c2:
            sim_zone = st.selectbox(
                "Zona Alvo",
                options=[z for z in STATCAST_ZONES if z != 0],
                index=[z for z in STATCAST_ZONES if z != 0].index(default_sim_zone),
                format_func=lambda z: f"Zona {z} ({STATCAST_ZONES[z]['region']})",
                key="sim_pitch_zone_select",
            )

        sim_type_code = PITCH_TYPES_BY_CATEGORY[sim_cat][0][0]

        out_b1, out_b2, out_b3, out_b4 = st.columns(4)
        with out_b1:
            st.button(
                "🔴 Strike",
                on_click=simulate_pitch_outcome_callback,
                args=("S", sim_cat, sim_type_code, sim_zone),
                use_container_width=True,
                help="Registra Strike (ou Strikeout se já houver 2 strikes).",
            )
        with out_b2:
            st.button(
                "🟡 Foul",
                on_click=simulate_pitch_outcome_callback,
                args=("Foul", sim_cat, sim_type_code, sim_zone),
                use_container_width=True,
                help="Registra Foul Ball (mantém 2 strikes se a contagem já tiver 2 strikes).",
            )
        with out_b3:
            st.button(
                "🟢 Ball",
                on_click=simulate_pitch_outcome_callback,
                args=("B", sim_cat, sim_type_code, sim_zone),
                use_container_width=True,
                help="Registra Bola (ou Walk se já houver 3 bolas).",
            )
        with out_b4:
            st.button(
                "🔵 In Play",
                on_click=simulate_pitch_outcome_callback,
                args=("X", sim_cat, sim_type_code, sim_zone),
                use_container_width=True,
                help="Bola colocada em jogo; encerra o duelo atual e reseta para 0-0.",
            )

        if st.session_state.last_at_bat_event:
            ev = st.session_state.last_at_bat_event
            if "Strikeout" in ev or "Walk" in ev or "In Play" in ev:
                st.success(f"🔔 **Desfecho do Duelo:** {ev} — Contagem reiniciada para o próximo rebatedor!")
            else:
                st.caption(f"Último evento registrado: **{ev}**")

        if st.session_state.at_bat_history:
            st.dataframe(
                pd.DataFrame(st.session_state.at_bat_history),
                use_container_width=True,
                hide_index=True,
            )
            if st.button("🗑️ Limpar Histórico do Duelo", use_container_width=True):
                st.session_state.at_bat_history = []
                st.session_state.last_at_bat_event = None
                st.rerun()

    with st.expander("🔍 Inspecionar Payload JSON (Contrato de 24 Features)"):
        payload_preview = build_rest_payload([current_features])
        st.json(payload_preview)
