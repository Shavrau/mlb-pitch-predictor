"""
Page 02: MLB Scouting Report, 12-Count Tactical Decision Matrix & Batter Game Plan.

Executes batch inference across all 12 MLB counts (`0-0` to `3-2`) in a single request
to the Databricks Serverless Model Serving endpoint and synthesizes a complete pre-game
and in-game scouting dossier for the batter and coaching staff.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pitch_predictor.scenarios import PITCHER_ARCHETYPES
from pitch_predictor.scout_analytics import (
    MLB_12_COUNTS,
    compute_matrix_distribution,
    export_scouting_report_markdown,
    generate_12_counts_batch,
    generate_batter_game_plan,
    render_count_matrix_html,
    render_game_plan_cards_html,
    render_radar_chart_svg,
    simulate_tactical_12_counts_fallback,
)
from pitch_predictor.serving_client import (
    DEFAULT_ENDPOINT_NAME,
    MLB_LEAGUE_PRIORS,
    PitchServingClient,
)
from pitch_predictor.ui_components import get_custom_css

SCOUT_BASE_OPTIONS = [
    "RISP (Corredor na 2B — Posição de Anotar)",
    "Bases Limpas",
    "Corredor na 1ª Base (Double Play / Eliminação Dupla)",
    "Bases Lotadas (Pressão Máxima)",
]


@st.cache_resource(show_spinner=False)
def get_serving_client(endpoint_name: str = DEFAULT_ENDPOINT_NAME) -> PitchServingClient:
    """Return a cached instance of the Databricks Serverless Model Serving client."""
    return PitchServingClient(endpoint_name=endpoint_name)


SCOUT_ARCHETYPE_OPTIONS = list(PITCHER_ARCHETYPES.keys())


def init_scout_state() -> None:
    """Initialize session state keys for the Scouting Report page."""
    default_arch = "Breaking Specialist (Corbin Burnes / Sonny Gray)"
    if "scout_archetype" not in st.session_state or st.session_state.scout_archetype not in SCOUT_ARCHETYPE_OPTIONS:
        st.session_state.scout_archetype = default_arch
    if "scout_stand" not in st.session_state or st.session_state.scout_stand not in ("L", "R"):
        st.session_state.scout_stand = "L"
    if "scout_base_state" not in st.session_state or st.session_state.scout_base_state not in SCOUT_BASE_OPTIONS:
        st.session_state.scout_base_state = SCOUT_BASE_OPTIONS[0]
    if "scout_prev_cat" not in st.session_state or st.session_state.scout_prev_cat not in ("Fastball", "Breaking Ball", "Offspeed"):
        st.session_state.scout_prev_cat = "Fastball"
    if "scout_predictions_map" not in st.session_state:
        st.session_state.scout_predictions_map = None
    if "scout_batch_latency_ms" not in st.session_state:
        st.session_state.scout_batch_latency_ms = None
    if "scout_batch_source" not in st.session_state:
        st.session_state.scout_batch_source = "Simulação Tática Instantânea (Pré-Carregada)"


def _resolve_bases(base_label: str) -> tuple[int, int, int]:
    """Convert base situation selector label into `(on_1b, on_2b, on_3b)` flags."""
    if "Lotadas" in base_label:
        return (1, 1, 1)
    if "RISP" in base_label:
        return (0, 1, 0)
    if "1ª Base" in base_label:
        return (1, 0, 0)
    return (0, 0, 0)


init_scout_state()
st.markdown(get_custom_css(), unsafe_allow_html=True)
client = get_serving_client()

# ---------------------------------------------------------------------------
# Header Banner
# ---------------------------------------------------------------------------
st.markdown(
    '<div style="background:linear-gradient(90deg,#0F172A 0%,#151E2E 50%,#0F172A 100%);border:1px solid #26354A;border-left:5px solid #F59E0B;border-radius:12px;padding:14px 20px;margin-bottom:16px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">'
    '<div><div style="font-size:18px;font-weight:900;color:#F8FAFC;letter-spacing:0.04em;">📋 MLB SCOUTING REPORT &amp; MATRIZ TÁTICA DE DECISÃO</div>'
    '<div style="font-size:12px;color:#94A3B8;">Inferência em Lote das 12 Contagens (0-0 a 3-2) • Comparativo de Valor da IA • Game Plan do Rebatedor</div></div>'
    f'<div style="background:#0B0F19;border:1px solid #334155;border-radius:8px;padding:6px 12px;font-size:11.5px;color:#38BDF8;">Endpoint: <code>{client.endpoint_name}</code></div>'
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Top Control Panel: Pitcher Profile, Platoon, Base State & Batch Trigger
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown(
        '<div class="section-header">🎛️ Parâmetros do Confronto para Varredura das 12 Contagens</div>',
        unsafe_allow_html=True,
    )
    ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([1.4, 0.85, 1.0, 0.95])
    with ctrl_c1:
        selected_arch = st.selectbox(
            "Arquétipo de Arremessador (Pitcher Profile)",
            options=list(PITCHER_ARCHETYPES.keys()),
            key="scout_archetype",
        )
    with ctrl_c2:
        selected_stand = st.radio(
            "Lado do Rebatedor",
            options=["L", "R"],
            format_func=lambda s: "Canhoto (LHB)" if s == "L" else "Destro (RHB)",
            horizontal=True,
            key="scout_stand",
        )
    with ctrl_c3:
        selected_bases = st.selectbox(
            "Situação de Corredores",
            options=SCOUT_BASE_OPTIONS,
            key="scout_base_state",
        )
    with ctrl_c4:
        selected_prev_cat = st.selectbox(
            "Arremesso Anterior (Contagens > 0-0)",
            options=["Fastball", "Breaking Ball", "Offspeed"],
            key="scout_prev_cat",
        )

    profile = PITCHER_ARCHETYPES[selected_arch]
    on_1b, on_2b, on_3b = _resolve_bases(selected_bases)
    risp_active = bool(on_2b or on_3b)

    prev_type_map = {"Fastball": "FF", "Breaking Ball": "SL", "Offspeed": "CH"}
    prev_zone_map = {"Fastball": 5, "Breaking Ball": 14, "Offspeed": 8}

    batch_records = generate_12_counts_batch(
        pitcher_profile=profile,
        stand=selected_stand,
        on_1b=on_1b,
        on_2b=on_2b,
        on_3b=on_3b,
        prev_pitch_category=selected_prev_cat,
        prev_pitch_type=prev_type_map.get(selected_prev_cat, "FF"),
        prev_pitch_zone=prev_zone_map.get(selected_prev_cat, 5),
        prev_pitch_result="S",
    )

    btn_col1, btn_col2 = st.columns([1.2, 1.8])
    with btn_col1:
        run_batch_clicked = st.button(
            "📡 Gerar Scouting Report (Batch 12 Contagens via Endpoint)",
            type="primary",
            use_container_width=True,
        )
    with btn_col2:
        st.caption(
            f"**Arsenal {selected_arch}:** {profile['signature_arsenal']} — {profile['description']}"
        )

    if run_batch_clicked:
        with st.spinner(
            "⏳ Executando inferência em lote (12 contagens simultâneas) no Databricks Serverless Model Serving..."
        ):
            try:
                batch_res = client.predict_batch(batch_records)
                preds_list = batch_res["predictions"]
                st.session_state.scout_predictions_map = {
                    f"{b}-{s}": preds_list[idx]
                    for idx, (b, s) in enumerate(MLB_12_COUNTS)
                }
                st.session_state.scout_batch_latency_ms = batch_res["latency_ms"]
                st.session_state.scout_batch_source = (
                    f"Databricks Serverless Endpoint ({batch_res['latency_ms']:.1f} ms para 12 contagens)"
                )
            except Exception as exc:  # noqa: BLE001
                st.error(
                    f"⚠️ Erro ao consultar endpoint `{client.endpoint_name}` em lote: {exc}"
                )

    # Always keep a current 12-count prediction map ready for immediate visual inspection
    if (
        not st.session_state.scout_predictions_map or not run_batch_clicked
    ) and st.session_state.scout_batch_latency_ms is None:
        st.session_state.scout_predictions_map = simulate_tactical_12_counts_fallback(
            batch_records
        )

predictions_map: dict[str, str] = st.session_state.scout_predictions_map
model_dist = compute_matrix_distribution(predictions_map)
priors_dict = {
    "pitcher_fb_prior": float(profile["pitcher_fb_prior"]),
    "pitcher_bb_prior": float(profile["pitcher_bb_prior"]),
    "pitcher_os_prior": float(profile["pitcher_os_prior"]),
}
game_plan = generate_batter_game_plan(
    pitcher_name=selected_arch,
    predictions_by_count=predictions_map,
    priors=priors_dict,
    stand=selected_stand,
    risp_active=risp_active,
)

# ---------------------------------------------------------------------------
# Section 1: 4x3 Tactical Count Matrix
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown(
        '<div class="section-header">🎯 1. Matriz Tática de Decisão nas 12 Contagens (Balls × Strikes)</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Fonte ativa: **{st.session_state.scout_batch_source}** • "
        f"Platoon: **{profile['p_throws']}HP vs {selected_stand}HB** • "
        f"Situação: **{selected_bases}**"
    )
    st.markdown(render_count_matrix_html(predictions_map), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 2: AI Value Comparison — Radar Chart SVG + Quantitative Table
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown(
        '<div class="section-header">📐 2. Comparativo de Valor da IA vs. Repertório Histórico e Média da Liga</div>',
        unsafe_allow_html=True,
    )
    rad_col, tbl_col = st.columns([1.05, 1.15], gap="medium")
    with rad_col:
        st.markdown(
            render_radar_chart_svg(
                league_dist=MLB_LEAGUE_PRIORS,
                pitcher_dist=(
                    priors_dict["pitcher_fb_prior"],
                    priors_dict["pitcher_bb_prior"],
                    priors_dict["pitcher_os_prior"],
                ),
                model_dist=model_dist,
            ),
            unsafe_allow_html=True,
        )
    with tbl_col:
        st.markdown("##### 📊 Desvio Tático Situacional (IA vs. Prior Estático)")
        st.caption(
            "Demonstra como o modelo vai além da média histórica do arremessador, "
            "ajustando a chamada conforme contagem, vantagem tática, platoon e corredores em base."
        )
        comp_df = pd.DataFrame(
            [
                {
                    "Categoria": "⚡ Fastball (Bola Rápida)",
                    "Média MLB": f"{MLB_LEAGUE_PRIORS[0] * 100:.1f}%",
                    "Prior Pitcher": f"{priors_dict['pitcher_fb_prior'] * 100:.1f}%",
                    "Matriz IA (12 Contagens)": f"{model_dist['Fastball'] * 100:.1f}%",
                    "Ajuste Situacional (Δ)": f"{(model_dist['Fastball'] - priors_dict['pitcher_fb_prior']) * 100:+.1f}%",
                },
                {
                    "Categoria": "🌀 Breaking Ball (Efeito)",
                    "Média MLB": f"{MLB_LEAGUE_PRIORS[1] * 100:.1f}%",
                    "Prior Pitcher": f"{priors_dict['pitcher_bb_prior'] * 100:.1f}%",
                    "Matriz IA (12 Contagens)": f"{model_dist['Breaking Ball'] * 100:.1f}%",
                    "Ajuste Situacional (Δ)": f"{(model_dist['Breaking Ball'] - priors_dict['pitcher_bb_prior']) * 100:+.1f}%",
                },
                {
                    "Categoria": "🛑 Offspeed (Mudança)",
                    "Média MLB": f"{MLB_LEAGUE_PRIORS[2] * 100:.1f}%",
                    "Prior Pitcher": f"{priors_dict['pitcher_os_prior'] * 100:.1f}%",
                    "Matriz IA (12 Contagens)": f"{model_dist['Offspeed'] * 100:.1f}%",
                    "Ajuste Situacional (Δ)": f"{(model_dist['Offspeed'] - priors_dict['pitcher_os_prior']) * 100:+.1f}%",
                },
            ]
        )
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Section 3: Batter Game Plan (3 Tactical Approach Cards)
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown(
        '<div class="section-header">🏏 3. Plano de Jogo do Rebatedor (Game Plan) — Estratégia por Fase da Contagem</div>',
        unsafe_allow_html=True,
    )
    st.markdown(render_game_plan_cards_html(game_plan), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 4: Coaching Staff Export (Download .md + One-Click Copy Block)
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown(
        '<div class="section-header">📥 4. Exportar Dossiê Tático para a Comissão Técnica</div>',
        unsafe_allow_html=True,
    )
    md_report = export_scouting_report_markdown(
        pitcher_name=selected_arch,
        predictions_by_count=predictions_map,
        game_plan=game_plan,
        priors=priors_dict,
        stand=selected_stand,
        risp_active=risp_active,
    )

    dl_col, info_col = st.columns([1.0, 1.5])
    with dl_col:
        safe_slug = selected_arch.lower().split("(")[0].strip().replace(" ", "_")
        st.download_button(
            label="📄 Baixar Relatório Completo (.md)",
            data=md_report,
            file_name=f"scouting_report_{safe_slug}.md",
            mime="text/markdown",
            type="primary",
            use_container_width=True,
        )
    with info_col:
        st.caption(
            "O relatório inclui o perfil do arremessador, a tabela completa das 12 contagens "
            "e o Game Plan de 3 estágios pronto para impressão no dugout."
        )

    with st.expander("📋 Visualizar / Copiar Relatório em Markdown (1-Click Copy)"):
        st.code(md_report, language="markdown")
