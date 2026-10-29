"""
Page 02: MLB Scouting Report, 12-Count Tactical Decision Matrix & Batter Game Plan.

Executes batch inference across all 12 MLB counts (`0-0` to `3-2`) in a single request
to the Databricks Serverless Model Serving endpoint and synthesizes a pre-game and in-game
Dugout Dossier featuring an interactive Custom Component v2 (`st.components.v2`) 4x3 Count Matrix.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pitch_predictor.ccv2_components import (
    build_count_inspection_detail,
    interactive_count_matrix_ccv2,
)
from pitch_predictor.scenarios import MLB_PITCHERS, PITCHER_ARCHETYPES
from pitch_predictor.scout_analytics import (
    MLB_12_COUNTS,
    compute_matrix_distribution,
    export_scouting_report_markdown,
    generate_12_counts_batch,
    generate_batter_game_plan,
    render_game_plan_cards_html,
    render_radar_chart_svg,
    simulate_tactical_12_counts_fallback,
)
from pitch_predictor.serving_client import (
    DEFAULT_ENDPOINT_NAME,
    MLB_LEAGUE_PRIORS,
    PitchServingClient,
)
from pitch_predictor.ui_components import (
    PITCH_CATEGORY_STYLES,
    _compact_html,
    get_custom_css,
    render_strike_zone_svg,
    resolve_ui_theme_mode,
)

SCOUT_BASE_OPTIONS = [
    "RISP (Corredor na 2B — Posição de Anotar)",
    "Bases Limpas",
    "Corredor na 1ª Base (Double Play / Eliminação Dupla)",
    "Bases Lotadas (Pressão Máxima)",
]

ARCHETYPE_FRANCHISE_HEX: dict[str, str] = {
    "Média da Liga MLB (League Average)": "#38BDF8",
    "Power Pitcher (Paul Skenes / Spencer Strider)": MLB_PITCHERS["Paul Skenes"]["team_primary_hex"],
    "Breaking Specialist (Corbin Burnes / Sonny Gray)": MLB_PITCHERS["Corbin Burnes"]["team_primary_hex"],
    "Offspeed Master (Logan Webb / Kevin Gausman)": MLB_PITCHERS["Logan Webb"]["team_primary_hex"],
    "Lefty Crafty Ace (Chris Sale / Blake Snell)": MLB_PITCHERS["Chris Sale"]["team_primary_hex"],
}


@st.cache_resource(show_spinner=False)
def get_serving_client(endpoint_name: str = DEFAULT_ENDPOINT_NAME) -> PitchServingClient:
    """Return a cached instance of the Databricks Serverless Model Serving client."""
    return PitchServingClient(endpoint_name=endpoint_name)


SCOUT_ARCHETYPE_OPTIONS = list(PITCHER_ARCHETYPES.keys())


def init_scout_state() -> None:
    """Initialize and validate session state keys for the Scouting Report page."""
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
    if "scout_inspected_count" not in st.session_state:
        st.session_state.scout_inspected_count = "0-2"


def _resolve_bases(base_label: str) -> tuple[int, int, int]:
    """Convert base situation selector label into `(on_1b, on_2b, on_3b)` flags."""
    if "Lotadas" in base_label:
        return (1, 1, 1)
    if "RISP" in base_label:
        return (0, 1, 0)
    if "1ª Base" in base_label:
        return (1, 0, 0)
    return (0, 0, 0)


def _render_compact_12_count_ribbon_html(predictions_by_count: dict[str, str]) -> str:
    """Render a compact horizontal telemetry strip summarizing all 12 count calls."""
    pills: list[str] = []
    for b, s in MLB_12_COUNTS:
        c_key = f"{b}-{s}"
        cat = predictions_by_count.get(c_key, "Fastball")
        style = PITCH_CATEGORY_STYLES.get(cat, PITCH_CATEGORY_STYLES["Fastball"])
        prim = style["primary"]
        code = style.get("code_badge", "FF").split("[")[-1].rstrip("]")
        pills.append(
            f'<span style="display:inline-flex;align-items:center;gap:4px;background:var(--pp-surface-alt,#181B23);border:1px solid var(--pp-border,#222631);border-left:3px solid {prim};border-radius:4px;padding:2px 7px;font-family:\'JetBrains Mono\',monospace;font-variant-numeric:tabular-nums;font-size:10.5px;">'
            f'<strong style="color:var(--pp-text,#F1F5F9);">{c_key}</strong>'
            f'<span style="color:{prim};font-weight:700;">{code}</span>'
            f"</span>"
        )
    raw = (
        '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;padding-top:8px;border-top:1px solid var(--pp-border,#222631);">'
        + "".join(pills)
        + "</div>"
    )
    return _compact_html(raw)


init_scout_state()
active_theme = resolve_ui_theme_mode()
accent_hex = ARCHETYPE_FRANCHISE_HEX.get(st.session_state.scout_archetype, "#F59E0B")
st.markdown(get_custom_css(theme_mode=active_theme, accent_hex=accent_hex), unsafe_allow_html=True)
client = get_serving_client()

# ---------------------------------------------------------------------------
# Dugout Dossier Header Banner
# ---------------------------------------------------------------------------
st.markdown(
    _compact_html(
        f"""
        <div style="background:var(--pp-surface,#12141A);border:1px solid var(--pp-border,#222631);border-top:3px solid {accent_hex};border-radius:8px;padding:14px 20px;margin-bottom:14px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
          <div>
            <div class="pp-display" style="font-size:20px;font-weight:800;color:var(--pp-text,#F1F5F9);letter-spacing:0.04em;text-transform:uppercase;">MLB SCOUTING REPORT &amp; MATRIZ TÁTICA DE DECISÃO</div>
            <div style="font-size:12px;color:var(--pp-muted,#94A3B8);">Inferência em Lote das 12 Contagens (0-0 a 3-2) • Inspeção Interativa CCv2 • Game Plan do Rebatedor</div>
          </div>
          <div style="background:var(--pp-surface-alt,#181B23);border:1px solid var(--pp-border,#222631);border-radius:4px;padding:5px 11px;font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--pp-muted,#94A3B8);">Endpoint: <code style="color:var(--pp-text,#F1F5F9);">{client.endpoint_name}</code></div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Top Control Panel: Pitcher Profile, Platoon, Base State & Batch Trigger
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown(
        '<div class="section-header">Parâmetros do Confronto • Varredura das 12 Contagens</div>',
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
    accent_hex = ARCHETYPE_FRANCHISE_HEX.get(selected_arch, "#F59E0B")
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

    btn_col1, btn_col2 = st.columns([1.2, 1.8], vertical_alignment="center")
    with btn_col1:
        run_batch_clicked = st.button(
            "Gerar Scouting Report (Batch 12 Contagens via Endpoint)",
            icon=":material/sensors:",
            type="primary",
            width="stretch",
        )
    with btn_col2:
        st.caption(
            f"**Arsenal {selected_arch}:** {profile['signature_arsenal']} — {profile['description']}"
        )

    if run_batch_clicked:
        with st.spinner(
            "Executando inferência em lote (12 contagens simultâneas) no Databricks Serverless Model Serving..."
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
                    f"Erro ao consultar endpoint `{client.endpoint_name}` em lote: {exc}",
                    icon=":material/error:",
                )

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
# Section 1: Interactive 4x3 Tactical Count Matrix (CCv2) + Dugout Inspector
# ---------------------------------------------------------------------------
@st.fragment
def _render_interactive_matrix_section() -> None:
    with st.container(border=True):
        st.markdown(
            '<div class="section-header">01. Matriz Tática Interativa nas 12 Contagens (Clique em qualquer contagem para inspecionar)</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Fonte ativa: **{st.session_state.scout_batch_source}** • "
            f"Platoon: **{profile['p_throws']}HP vs {selected_stand}HB** • "
            f"Situação: **{selected_bases}**"
        )

        mat_col, insp_col = st.columns([1.55, 1.0], gap="medium")
        with mat_col:
            selected_count = interactive_count_matrix_ccv2(
                predictions_by_count=predictions_map,
                initial_count=st.session_state.get("scout_inspected_count", "0-2"),
                accent_hex=accent_hex,
                theme_mode=active_theme,
                key="scout_ccv2_matrix",
            )
            st.session_state.scout_inspected_count = selected_count
            st.markdown(
                _render_compact_12_count_ribbon_html(predictions_map),
                unsafe_allow_html=True,
            )

        with insp_col:
            detail = build_count_inspection_detail(
                count_key=selected_count,
                predictions_by_count=predictions_map,
                priors=priors_dict,
                stand=selected_stand,
                risp_active=risp_active,
            )
            d_cat = detail["predicted_category"]
            d_style = PITCH_CATEGORY_STYLES.get(d_cat, PITCH_CATEGORY_STYLES["Fastball"])
            d_prim = d_style["primary"]
            zones_str = " • ".join(detail["recommended_zones"])
            first_zone_str = detail["recommended_zones"][0].split()[0].lstrip("Z")
            target_z = int(first_zone_str) if first_zone_str.isdigit() else 5
            probs = detail["probabilities"]

            insp_html = _compact_html(
                f"""
                <div style="background:var(--pp-surface-alt,#181B23);border:1px solid var(--pp-border,#222631);border-left:3px solid {d_prim};border-radius:6px;padding:12px 14px;margin-bottom:8px;">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <span style="font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:800;color:var(--pp-muted,#94A3B8);letter-spacing:0.05em;">INSPEÇÃO DE CONTAGEM • {detail['count_key']}</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:800;color:{d_prim};">{detail['code_badge']}</span>
                  </div>
                  <div style="font-size:18px;font-weight:800;color:var(--pp-text,#F1F5F9);margin-bottom:4px;">{d_style['icon']} {d_cat} <span style="font-size:12px;color:var(--pp-muted,#94A3B8);font-weight:600;">({detail['leverage_pt']})</span></div>
                  <div style="font-family:'JetBrains Mono',monospace;font-variant-numeric:tabular-nums;font-size:10.5px;color:var(--pp-muted,#94A3B8);margin-bottom:8px;">
                    FB {probs['Fastball'] * 100:.0f}% • BRK {probs['Breaking Ball'] * 100:.0f}% • OFF {probs['Offspeed'] * 100:.0f}%
                  </div>
                  <div style="font-size:12px;color:var(--pp-text,#F1F5F9);line-height:1.42;margin-bottom:8px;">{detail['tactical_note']}</div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:{d_prim};font-weight:700;">ALVOS: {zones_str}</div>
                </div>
                """
            )
            st.markdown(insp_html, unsafe_allow_html=True)
            st.markdown(
                render_strike_zone_svg(selected_zone=target_z, prev_category=d_cat),
                unsafe_allow_html=True,
            )


_render_interactive_matrix_section()

# ---------------------------------------------------------------------------
# Section 2: AI Value Comparison — Radar Chart SVG + Quantitative Table
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown(
        '<div class="section-header">02. Comparativo de Valor Situacional da IA vs. Repertório Base e Média da MLB</div>',
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
        st.markdown("##### Desvio Tático Situacional (IA vs. Prior Estático)")
        st.caption(
            "Quantifica como o modelo ajusta a distribuição histórica do arremessador conforme "
            "contagem, vantagem tática, platoon e corredores em base."
        )
        comp_df = pd.DataFrame(
            [
                {
                    "Categoria": "FF • Fastball (Bola Rápida)",
                    "Média MLB": f"{MLB_LEAGUE_PRIORS[0] * 100:.1f}%",
                    "Prior Pitcher": f"{priors_dict['pitcher_fb_prior'] * 100:.1f}%",
                    "Matriz IA (12 Contagens)": f"{model_dist['Fastball'] * 100:.1f}%",
                    "Ajuste Situacional (Δ)": f"{(model_dist['Fastball'] - priors_dict['pitcher_fb_prior']) * 100:+.1f}%",
                },
                {
                    "Categoria": "BRK • Breaking Ball (Efeito)",
                    "Média MLB": f"{MLB_LEAGUE_PRIORS[1] * 100:.1f}%",
                    "Prior Pitcher": f"{priors_dict['pitcher_bb_prior'] * 100:.1f}%",
                    "Matriz IA (12 Contagens)": f"{model_dist['Breaking Ball'] * 100:.1f}%",
                    "Ajuste Situacional (Δ)": f"{(model_dist['Breaking Ball'] - priors_dict['pitcher_bb_prior']) * 100:+.1f}%",
                },
                {
                    "Categoria": "OFF • Offspeed (Mudança)",
                    "Média MLB": f"{MLB_LEAGUE_PRIORS[2] * 100:.1f}%",
                    "Prior Pitcher": f"{priors_dict['pitcher_os_prior'] * 100:.1f}%",
                    "Matriz IA (12 Contagens)": f"{model_dist['Offspeed'] * 100:.1f}%",
                    "Ajuste Situacional (Δ)": f"{(model_dist['Offspeed'] - priors_dict['pitcher_os_prior']) * 100:+.1f}%",
                },
            ]
        )
        st.dataframe(comp_df, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Section 3: Batter Game Plan (3 Tactical Approach Cards)
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown(
        '<div class="section-header">03. Plano de Jogo do Rebatedor (Game Plan) — Abordagem por Fase da Contagem</div>',
        unsafe_allow_html=True,
    )
    st.markdown(render_game_plan_cards_html(game_plan), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 4: Coaching Staff Export (Download .md + One-Click Copy Block)
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown(
        '<div class="section-header">04. Exportação do Dossiê Tático para a Comissão Técnica</div>',
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

    dl_col, info_col = st.columns([1.0, 1.5], vertical_alignment="center")
    with dl_col:
        safe_slug = selected_arch.lower().split("(")[0].strip().replace(" ", "_")
        st.download_button(
            label="Baixar Relatório Completo (.md)",
            icon=":material/download:",
            data=md_report,
            file_name=f"scouting_report_{safe_slug}.md",
            mime="text/markdown",
            type="primary",
            width="stretch",
        )
    with info_col:
        st.caption(
            "Dossiê formatado contendo priors do arremessador, matriz completa das 12 contagens "
            "e Game Plan em 3 estágios pronto para impressão ou tablet de dugout."
        )

    with st.expander("Visualizar / Copiar Relatório em Markdown"):
        st.code(md_report, language="markdown")
