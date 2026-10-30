"""
Page 02: MLB Scouting Report, 12-Count Tactical Decision Matrix, Head-to-Head Comparison & Batter Game Plan.

Executes batch inference across all 12 MLB counts (`0-0` to `3-2`) in a single request
to the Databricks Serverless Model Serving endpoint and synthesizes:
1. Single Pitcher Tactical Dossier with interactive Custom Component v2 (`st.components.v2`) Count Matrix.
2. Head-to-Head Pitcher Comparison with side-by-side athlete profile cards, multi-polygon SVG radar chart,
   12-count divergence telemetry, and comparative batter game plan.
3. Printable Dugout Scouting Card (HTML / PDF) and coaching staff Markdown export.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components_v1

from pitch_predictor.ccv2_components import (
    build_count_inspection_detail,
    interactive_count_matrix_ccv2,
)
from pitch_predictor.scenarios import MLB_PITCHERS, PITCHER_ARCHETYPES
from pitch_predictor.scout_analytics import (
    MLB_12_COUNTS,
    compute_head_to_head_divergence,
    compute_matrix_distribution,
    export_scouting_card_html,
    export_scouting_report_markdown,
    generate_12_counts_batch,
    generate_batter_game_plan,
    generate_comparative_game_plan,
    render_game_plan_cards_html,
    render_head_to_head_radar_svg,
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
    render_pitcher_photo_html,
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


def _archetype_to_pitcher_name(archetype: str) -> str:
    """Map archetype string to canonical MLB pitcher name for headshot rendering."""
    if "Paul Skenes" in archetype:
        return "Paul Skenes"
    if "Corbin Burnes" in archetype:
        return "Corbin Burnes"
    if "Logan Webb" in archetype:
        return "Logan Webb"
    if "Chris Sale" in archetype:
        return "Chris Sale"
    if "Yamamoto" in archetype:
        return "Yoshinobu Yamamoto"
    return "Paul Skenes"


@st.cache_resource(show_spinner=False)
def get_serving_client(endpoint_name: str = DEFAULT_ENDPOINT_NAME) -> PitchServingClient:
    """Return a cached instance of the Databricks Serverless Model Serving client."""
    return PitchServingClient(endpoint_name=endpoint_name)


SCOUT_ARCHETYPE_OPTIONS = list(PITCHER_ARCHETYPES.keys())


def init_scout_state() -> None:
    """Initialize and validate session state keys for the Scouting Report page."""
    default_arch = "Breaking Specialist (Corbin Burnes / Sonny Gray)"
    if "scout_view_mode" not in st.session_state or st.session_state.scout_view_mode not in (
        "🎯 Dossiê Individual (Single Pitcher)",
        "⚔️ Comparação Head-to-Head (2 Arremessadores)",
    ):
        st.session_state.scout_view_mode = "🎯 Dossiê Individual (Single Pitcher)"

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

    # Head-to-Head state keys
    if "scout_h2h_pitcher_a" not in st.session_state or st.session_state.scout_h2h_pitcher_a not in SCOUT_ARCHETYPE_OPTIONS:
        st.session_state.scout_h2h_pitcher_a = "Power Pitcher (Paul Skenes / Spencer Strider)"
    if "scout_h2h_pitcher_b" not in st.session_state or st.session_state.scout_h2h_pitcher_b not in SCOUT_ARCHETYPE_OPTIONS:
        st.session_state.scout_h2h_pitcher_b = "Offspeed Master (Logan Webb / Kevin Gausman)"
    if "scout_h2h_stand" not in st.session_state or st.session_state.scout_h2h_stand not in ("L", "R"):
        st.session_state.scout_h2h_stand = "L"
    if "scout_h2h_base_state" not in st.session_state or st.session_state.scout_h2h_base_state not in SCOUT_BASE_OPTIONS:
        st.session_state.scout_h2h_base_state = SCOUT_BASE_OPTIONS[0]
    if "scout_h2h_prev_cat" not in st.session_state or st.session_state.scout_h2h_prev_cat not in ("Fastball", "Breaking Ball", "Offspeed"):
        st.session_state.scout_h2h_prev_cat = "Fastball"
    if "scout_h2h_preds_a" not in st.session_state:
        st.session_state.scout_h2h_preds_a = None
    if "scout_h2h_preds_b" not in st.session_state:
        st.session_state.scout_h2h_preds_b = None
    if "scout_h2h_latency_ms" not in st.session_state:
        st.session_state.scout_h2h_latency_ms = None


def _resolve_bases(base_label: str) -> tuple[int, int, int]:
    """Convert base situation selector label into `(on_1b, on_2b, on_3b)` flags."""
    if "Lotadas" in base_label:
        return (1, 1, 1)
    if "RISP" in base_label:
        return (0, 1, 0)
    if "1ª Base" in base_label:
        return (1, 0, 0)
    return (0, 0, 0)


def _render_compact_12_count_ribbon_html(
    predictions_by_count: dict[str, str], theme_mode: str = "dark"
) -> str:
    """Render a compact horizontal telemetry strip summarizing all 12 count calls."""
    is_light = str(theme_mode).strip().lower().startswith("light")
    pills: list[str] = []
    for b, s in MLB_12_COUNTS:
        c_key = f"{b}-{s}"
        cat = predictions_by_count.get(c_key, "Fastball")
        style = PITCH_CATEGORY_STYLES.get(cat, PITCH_CATEGORY_STYLES["Fastball"])
        prim = style.get("secondary", style["primary"]) if is_light else style["primary"]
        code = style.get("code_badge", "FF").split("[")[-1].rstrip("]")
        pill_border = "var(--pp-border,#C2CFDD)" if is_light else "var(--pp-border,#222631)"
        pills.append(
            f'<span style="display:inline-flex;align-items:center;gap:4px;background:var(--pp-surface-alt,#181B23);border:1px solid {pill_border};border-left:3px solid {prim};border-radius:4px;padding:2px 7px;font-family:\'JetBrains Mono\',monospace;font-variant-numeric:tabular-nums;font-size:10.5px;">'
            f'<strong style="color:var(--pp-text,#F1F5F9);">{c_key}</strong>'
            f'<span style="color:{prim};font-weight:700;">{code}</span>'
            f"</span>"
        )
    div_border = "var(--pp-border,#C2CFDD)" if is_light else "var(--pp-border,#222631)"
    raw = (
        f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;padding-top:8px;border-top:1px solid {div_border};">'
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
            <div style="font-size:12px;color:var(--pp-muted,#94A3B8);">Inferência em Lote das 12 Contagens (0-0 a 3-2) • Comparação Head-to-Head • Cartão Imprimível de Dugout</div>
          </div>
          <div style="background:var(--pp-surface-alt,#181B23);border:1px solid var(--pp-border,#222631);border-radius:4px;padding:5px 11px;font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--pp-muted,#94A3B8);">Endpoint: <code style="color:var(--pp-text,#F1F5F9);">{client.endpoint_name}</code></div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# View Mode Selector: Single Pitcher Dossier vs. Head-to-Head Comparison
# ---------------------------------------------------------------------------
view_mode = st.radio(
    "Modo de Análise do Scouting Report",
    options=[
        "🎯 Dossiê Individual (Single Pitcher)",
        "⚔️ Comparação Head-to-Head (2 Arremessadores)",
    ],
    horizontal=True,
    key="scout_view_mode",
    help="Alterne entre o dossiê detalhado de um arremessador ou a comparação tática direta entre dois astros da MLB.",
)

prev_type_map = {"Fastball": "FF", "Breaking Ball": "SL", "Offspeed": "CH"}
prev_zone_map = {"Fastball": 5, "Breaking Ball": 14, "Offspeed": 8}

# ===========================================================================
# MODE 1: SINGLE PITCHER DOSSIER
# ===========================================================================
if view_mode == "🎯 Dossiê Individual (Single Pitcher)":
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
    # Section 4: Coaching Staff Export (Download .md + Printable Dugout Card)
    # ---------------------------------------------------------------------------
    with st.container(border=True):
        st.markdown(
            '<div class="section-header">04. Exportação do Dossiê Tático & Cartão Imprimível de Dugout</div>',
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
        dugout_html = export_scouting_card_html(
            pitcher_name=selected_arch,
            predictions_by_count=predictions_map,
            game_plan=game_plan,
            priors=priors_dict,
            stand=selected_stand,
            risp_active=risp_active,
            theme_mode=active_theme,
        )

        safe_slug = selected_arch.lower().split("(")[0].strip().replace(" ", "_")
        dl_c1, dl_c2 = st.columns([1.0, 1.0], gap="medium")
        with dl_c1:
            st.download_button(
                label="Baixar Relatório Completo (.md)",
                icon=":material/download:",
                data=md_report,
                file_name=f"scouting_report_{safe_slug}.md",
                mime="text/markdown",
                type="primary",
                width="stretch",
            )
        with dl_c2:
            st.download_button(
                label="Baixar Cartão Imprimível Dugout (HTML / PDF)",
                icon=":material/print:",
                data=dugout_html,
                file_name=f"cartao_tatico_dugout_{safe_slug}.html",
                mime="text/html",
                type="secondary",
                width="stretch",
            )

        st.caption(
            "O Cartão Imprimível Dugout é formatado em folha A4 com `@media print` sem poluição de tela, "
            "contendo a matriz 4x3 de 12 contagens e estratégias de disciplina de placa prontas para tablet ou prancheta."
        )

        with st.expander("Visualizar Cartão Tático Imprimível de Dugout (Preview)"):
            components_v1.html(dugout_html, height=540, scrolling=True)

        with st.expander("Visualizar / Copiar Relatório em Markdown"):
            st.code(md_report, language="markdown")


# ===========================================================================
# MODE 2: HEAD-TO-HEAD PITCHER COMPARISON (SPRINT 2)
# ===========================================================================
else:
    with st.container(border=True):
        st.markdown(
            '<div class="section-header">Seleção de Arremessadores & Contexto do Confronto Head-to-Head</div>',
            unsafe_allow_html=True,
        )
        h2h_c1, h2h_c2, h2h_c3, h2h_c4 = st.columns([1.15, 1.15, 0.85, 0.85])
        with h2h_c1:
            pitcher_a_sel = st.selectbox(
                "Arremessador A (Referência)",
                options=SCOUT_ARCHETYPE_OPTIONS,
                key="scout_h2h_pitcher_a",
            )
        with h2h_c2:
            # Default to second archetype or Offspeed Master
            b_default_idx = 3 if len(SCOUT_ARCHETYPE_OPTIONS) > 3 else 1
            pitcher_b_sel = st.selectbox(
                "Arremessador B (Comparado)",
                options=SCOUT_ARCHETYPE_OPTIONS,
                index=b_default_idx if st.session_state.scout_h2h_pitcher_b == SCOUT_ARCHETYPE_OPTIONS[b_default_idx] else 0,
                key="scout_h2h_pitcher_b",
            )
        with h2h_c3:
            h2h_stand = st.radio(
                "Lado do Rebatedor",
                options=["L", "R"],
                format_func=lambda s: "Canhoto (LHB)" if s == "L" else "Destro (RHB)",
                horizontal=True,
                key="scout_h2h_stand",
            )
        with h2h_c4:
            h2h_bases = st.selectbox(
                "Situação de Corredores",
                options=SCOUT_BASE_OPTIONS,
                key="scout_h2h_base_state",
            )

        profile_a = PITCHER_ARCHETYPES[pitcher_a_sel]
        profile_b = PITCHER_ARCHETYPES[pitcher_b_sel]
        accent_a = ARCHETYPE_FRANCHISE_HEX.get(pitcher_a_sel, "#FDB827")
        accent_b = ARCHETYPE_FRANCHISE_HEX.get(pitcher_b_sel, "#38BDF8")
        if accent_a == accent_b:
            accent_b = "#06B6D4"  # ensure contrasting colors if same franchise

        on_1b, on_2b, on_3b = _resolve_bases(h2h_bases)
        risp_active = bool(on_2b or on_3b)

        batch_a = generate_12_counts_batch(
            pitcher_profile=profile_a,
            stand=h2h_stand,
            on_1b=on_1b,
            on_2b=on_2b,
            on_3b=on_3b,
            prev_pitch_category="Fastball",
            prev_pitch_type="FF",
            prev_pitch_zone=5,
            prev_pitch_result="S",
        )
        batch_b = generate_12_counts_batch(
            pitcher_profile=profile_b,
            stand=h2h_stand,
            on_1b=on_1b,
            on_2b=on_2b,
            on_3b=on_3b,
            prev_pitch_category="Fastball",
            prev_pitch_type="FF",
            prev_pitch_zone=5,
            prev_pitch_result="S",
        )

        run_h2h_btn = st.button(
            "Executar Comparação Head-to-Head (24 Lances via Endpoint)",
            icon=":material/compare_arrows:",
            type="primary",
            width="stretch",
        )

        if run_h2h_btn:
            with st.spinner("Executando inferência simultânea para os dois arremessadores no Databricks Serverless..."):
                try:
                    res_a = client.predict_batch(batch_a)
                    res_b = client.predict_batch(batch_b)
                    st.session_state.scout_h2h_preds_a = {
                        f"{b}-{s}": res_a["predictions"][idx]
                        for idx, (b, s) in enumerate(MLB_12_COUNTS)
                    }
                    st.session_state.scout_h2h_preds_b = {
                        f"{b}-{s}": res_b["predictions"][idx]
                        for idx, (b, s) in enumerate(MLB_12_COUNTS)
                    }
                    st.session_state.scout_h2h_latency_ms = res_a["latency_ms"] + res_b["latency_ms"]
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Erro na inferência em lote Head-to-Head: {exc}", icon=":material/error:")

        if not st.session_state.scout_h2h_preds_a or not run_h2h_btn:
            st.session_state.scout_h2h_preds_a = simulate_tactical_12_counts_fallback(batch_a)
            st.session_state.scout_h2h_preds_b = simulate_tactical_12_counts_fallback(batch_b)

    preds_a: dict[str, str] = st.session_state.scout_h2h_preds_a
    preds_b: dict[str, str] = st.session_state.scout_h2h_preds_b

    priors_a = {
        "pitcher_fb_prior": float(profile_a["pitcher_fb_prior"]),
        "pitcher_bb_prior": float(profile_a["pitcher_bb_prior"]),
        "pitcher_os_prior": float(profile_a["pitcher_os_prior"]),
    }
    priors_b = {
        "pitcher_fb_prior": float(profile_b["pitcher_fb_prior"]),
        "pitcher_bb_prior": float(profile_b["pitcher_bb_prior"]),
        "pitcher_os_prior": float(profile_b["pitcher_os_prior"]),
    }

    dist_a = compute_matrix_distribution(preds_a)
    dist_b = compute_matrix_distribution(preds_b)
    h2h_div = compute_head_to_head_divergence(preds_a, preds_b)
    comp_plan = generate_comparative_game_plan(
        name_a=pitcher_a_sel,
        name_b=pitcher_b_sel,
        preds_a=preds_a,
        preds_b=preds_b,
        priors_a=priors_a,
        priors_b=priors_b,
        stand=h2h_stand,
        risp_active=risp_active,
    )

    # ---------------------------------------------------------------------------
    # Athlete Profile Cards Side-by-Side
    # ---------------------------------------------------------------------------
    with st.container(border=True):
        st.markdown(
            '<div class="section-header">Perfis dos Atletas no Duelo Head-to-Head</div>',
            unsafe_allow_html=True,
        )
        card_a_col, card_b_col = st.columns(2, gap="medium")
        with card_a_col:
            st.markdown(
                render_pitcher_photo_html(
                    _archetype_to_pitcher_name(pitcher_a_sel),
                    height_px=148,
                    theme_mode=active_theme,
                ),
                unsafe_allow_html=True,
            )
            name_a_clean = pitcher_a_sel.split("(")[0].strip()
            st.markdown(
                _compact_html(
                    f"""
                    <div style="background:var(--pp-surface-alt,#181B23);border:1px solid var(--pp-border,#222631);border-left:4px solid {accent_a};border-radius:6px;padding:12px;margin-top:6px;">
                      <div style="font-size:15px;font-weight:800;color:var(--pp-text,#F1F5F9);margin-bottom:3px;">{name_a_clean}</div>
                      <div style="font-family:'JetBrains Mono',monospace;font-size:10.5px;color:{accent_a};font-weight:700;margin-bottom:6px;">{profile_a['p_throws']}HP • {profile_a['signature_arsenal']}</div>
                      <div style="font-size:11.5px;color:var(--pp-muted,#94A3B8);line-height:1.4;margin-bottom:8px;">{profile_a['description']}</div>
                      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--pp-text,#F1F5F9);background:var(--pp-surface,#12141A);padding:4px 8px;border-radius:4px;border:1px solid var(--pp-border,#222631);">
                        PRIORS BASE: FB {priors_a['pitcher_fb_prior'] * 100:.0f}% • BRK {priors_a['pitcher_bb_prior'] * 100:.0f}% • OFF {priors_a['pitcher_os_prior'] * 100:.0f}%
                      </div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

        with card_b_col:
            st.markdown(
                render_pitcher_photo_html(
                    _archetype_to_pitcher_name(pitcher_b_sel),
                    height_px=148,
                    theme_mode=active_theme,
                ),
                unsafe_allow_html=True,
            )
            name_b_clean = pitcher_b_sel.split("(")[0].strip()
            st.markdown(
                _compact_html(
                    f"""
                    <div style="background:var(--pp-surface-alt,#181B23);border:1px solid var(--pp-border,#222631);border-left:4px solid {accent_b};border-radius:6px;padding:12px;margin-top:6px;">
                      <div style="font-size:15px;font-weight:800;color:var(--pp-text,#F1F5F9);margin-bottom:3px;">{name_b_clean}</div>
                      <div style="font-family:'JetBrains Mono',monospace;font-size:10.5px;color:{accent_b};font-weight:700;margin-bottom:6px;">{profile_b['p_throws']}HP • {profile_b['signature_arsenal']}</div>
                      <div style="font-size:11.5px;color:var(--pp-muted,#94A3B8);line-height:1.4;margin-bottom:8px;">{profile_b['description']}</div>
                      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--pp-text,#F1F5F9);background:var(--pp-surface,#12141A);padding:4px 8px;border-radius:4px;border:1px solid var(--pp-border,#222631);">
                        PRIORS BASE: FB {priors_b['pitcher_fb_prior'] * 100:.0f}% • BRK {priors_b['pitcher_bb_prior'] * 100:.0f}% • OFF {priors_b['pitcher_os_prior'] * 100:.0f}%
                      </div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

    # ---------------------------------------------------------------------------
    # Section 1: Divergence Telemetry & Consensus Metrics
    # ---------------------------------------------------------------------------
    with st.container(border=True):
        st.markdown(
            '<div class="section-header">01. Telemetria de Divergência Tática (12 Contagens)</div>',
            unsafe_allow_html=True,
        )
        is_light = active_theme == "light"
        badge_div_bg = "rgba(239, 68, 68, 0.15)" if not is_light else "#FEE2E2"
        badge_div_border = "#EF4444" if not is_light else "#DC2626"
        badge_div_color = "#FCA5A5" if not is_light else "#991B1B"

        badge_agr_bg = "rgba(16, 185, 129, 0.15)" if not is_light else "#ECFDF5"
        badge_agr_border = "#10B981" if not is_light else "#059669"
        badge_agr_color = "#6EE7B7" if not is_light else "#064E3B"

        st.markdown(
            _compact_html(
                f"""
                <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:12px;">
                  <div style="flex:1;min-width:180px;background:{badge_div_bg};border:1px solid {badge_div_border};border-radius:8px;padding:12px 16px;">
                    <span style="font-size:10px;font-weight:800;font-family:'JetBrains Mono',monospace;text-transform:uppercase;color:{badge_div_color};letter-spacing:0.04em;">DIVERGÊNCIA TÁTICA</span>
                    <div style="font-size:24px;font-weight:800;color:{badge_div_color};font-family:'JetBrains Mono',monospace;">{h2h_div['divergence_pct'] * 100:.1f}%</div>
                    <span style="font-size:11px;color:{badge_div_color};opacity:0.9;">{h2h_div['divergence_counts']} de 12 contagens com chamadas opostas</span>
                  </div>
                  <div style="flex:1;min-width:180px;background:{badge_agr_bg};border:1px solid {badge_agr_border};border-radius:8px;padding:12px 16px;">
                    <span style="font-size:10px;font-weight:800;font-family:'JetBrains Mono',monospace;text-transform:uppercase;color:{badge_agr_color};letter-spacing:0.04em;">CONSENSO DE DUELO</span>
                    <div style="font-size:24px;font-weight:800;color:{badge_agr_color};font-family:'JetBrains Mono',monospace;">{h2h_div['agreement_pct'] * 100:.1f}%</div>
                    <span style="font-size:11px;color:{badge_agr_color};opacity:0.9;">{h2h_div['agreement_counts']} de 12 contagens idênticas</span>
                  </div>
                </div>
                <div style="background:var(--pp-surface-alt,#181B23);border:1px solid var(--pp-border,#222631);border-left:3px solid var(--pp-accent,#38BDF8);border-radius:6px;padding:10px 14px;font-size:12px;color:var(--pp-text,#F1F5F9);line-height:1.45;">
                  <strong>Análise do Modelo:</strong> {h2h_div['summary_text']}
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    # ---------------------------------------------------------------------------
    # Section 2: Side-by-Side 12-Count Tactical Decision Comparison Table
    # ---------------------------------------------------------------------------
    with st.container(border=True):
        st.markdown(
            '<div class="section-header">02. Matriz Comparativa Contagem a Contagem (Balls / Strikes)</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Visualização paralela dos arremessos previstos pelo modelo para ambos os arremessadores "
            "em cada uma das 12 contagens regulamentares da MLB."
        )

        rows_markup: list[str] = []
        is_light = active_theme == "light"
        border_c = "var(--pp-border, #C2CFDD)" if is_light else "var(--pp-border, #222631)"

        for item in h2h_div["contrasts"]:
            ck = item["count"]
            lev = item["leverage"]
            pa = item["pitch_a"]
            pb = item["pitch_b"]
            is_div = item["is_divergent"]

            style_a = PITCH_CATEGORY_STYLES.get(pa, PITCH_CATEGORY_STYLES["Fastball"])
            style_b = PITCH_CATEGORY_STYLES.get(pb, PITCH_CATEGORY_STYLES["Fastball"])
            prim_a = style_a.get("secondary", style_a["primary"]) if is_light else style_a["primary"]
            prim_b = style_b.get("secondary", style_b["primary"]) if is_light else style_b["primary"]

            if is_div:
                status_pill = (
                    '<span style="background:rgba(245,158,11,0.14);color:#B45309;border:1px solid #F59E0B;padding:2px 6px;border-radius:4px;font-size:10px;font-weight:800;font-family:\'JetBrains Mono\',monospace;">⚔️ DIVERGÊNCIA</span>'
                    if is_light
                    else '<span style="background:rgba(245,158,11,0.18);color:#F59E0B;border:1px solid #F59E0B;padding:2px 6px;border-radius:4px;font-size:10px;font-weight:800;font-family:\'JetBrains Mono\',monospace;">⚔️ DIVERGÊNCIA</span>'
                )
            else:
                status_pill = (
                    '<span style="background:rgba(16,185,129,0.14);color:#065F46;border:1px solid #10B981;padding:2px 6px;border-radius:4px;font-size:10px;font-weight:800;font-family:\'JetBrains Mono\',monospace;">🤝 CONSENSO</span>'
                    if is_light
                    else '<span style="background:rgba(16,185,129,0.18);color:#10B981;border:1px solid #10B981;padding:2px 6px;border-radius:4px;font-size:10px;font-weight:800;font-family:\'JetBrains Mono\',monospace;">🤝 CONSENSO</span>'
                )

            row_html = f"""
            <tr style="border-bottom:1px solid {border_c};">
              <td style="padding:8px 10px;font-family:'JetBrains Mono',monospace;font-weight:800;font-size:12px;color:var(--pp-text,#F1F5F9);">{ck}</td>
              <td style="padding:8px 10px;font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700;color:var(--pp-muted,#94A3B8);">{lev}</td>
              <td style="padding:8px 10px;border-left:3px solid {prim_a};background:var(--pp-surface-alt,#181B23);">
                <strong style="color:var(--pp-text,#F1F5F9);font-size:12.5px;">{style_a['icon']} {pa}</strong>
                <span style="font-family:'JetBrains Mono',monospace;font-size:9.5px;color:{prim_a};display:block;font-weight:700;">{style_a.get('code_badge','')}</span>
              </td>
              <td style="padding:8px 10px;border-left:3px solid {prim_b};background:var(--pp-surface-alt,#181B23);">
                <strong style="color:var(--pp-text,#F1F5F9);font-size:12.5px;">{style_b['icon']} {pb}</strong>
                <span style="font-family:'JetBrains Mono',monospace;font-size:9.5px;color:{prim_b};display:block;font-weight:700;">{style_b.get('code_badge','')}</span>
              </td>
              <td style="padding:8px 10px;text-align:center;">{status_pill}</td>
            </tr>
            """
            rows_markup.append(row_html)

        name_a_clean = pitcher_a_sel.split("(")[0].strip()
        name_b_clean = pitcher_b_sel.split("(")[0].strip()

        table_html = f"""
        <div style="background:var(--pp-surface,#12141A);border:1px solid {border_c};border-radius:8px;overflow:hidden;margin:6px 0;">
          <table style="width:100%;border-collapse:collapse;font-size:12px;text-align:left;">
            <thead>
              <tr style="background:var(--pp-surface-alt,#181B23);border-bottom:1px solid {border_c};">
                <th style="padding:10px;font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--pp-muted,#94A3B8);">CONTAGEM</th>
                <th style="padding:10px;font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--pp-muted,#94A3B8);">PRESSÃO</th>
                <th style="padding:10px;font-family:'JetBrains Mono',monospace;font-size:11.5px;color:{accent_a};font-weight:800;">{name_a_clean}</th>
                <th style="padding:10px;font-family:'JetBrains Mono',monospace;font-size:11.5px;color:{accent_b};font-weight:800;">{name_b_clean}</th>
                <th style="padding:10px;text-align:center;font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--pp-muted,#94A3B8);">STATUS</th>
              </tr>
            </thead>
            <tbody>
              {"".join(rows_markup)}
            </tbody>
          </table>
        </div>
        """
        st.markdown(_compact_html(table_html), unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # Section 3: Comparative Multi-Polygon Radar SVG & Quantitative Table
    # ---------------------------------------------------------------------------
    with st.container(border=True):
        st.markdown(
            '<div class="section-header">03. Radar de Confronto Multi-Polígono & Análise Quantitativa</div>',
            unsafe_allow_html=True,
        )
        h2h_rad_col, h2h_tbl_col = st.columns([1.1, 1.1], gap="medium")
        with h2h_rad_col:
            st.markdown(
                render_head_to_head_radar_svg(
                    league_dist=MLB_LEAGUE_PRIORS,
                    dist_a=dist_a,
                    dist_b=dist_b,
                    name_a=pitcher_a_sel,
                    name_b=pitcher_b_sel,
                    color_a=accent_a,
                    color_b=accent_b,
                ),
                unsafe_allow_html=True,
            )

        with h2h_tbl_col:
            st.markdown("##### Comparativo Quantitativo de Repertório")
            st.caption("Distribuição das 12 contagens projetadas para ambos os arremessadores vs. Média da MLB.")
            h2h_df = pd.DataFrame(
                [
                    {
                        "Categoria": "FF • Fastball",
                        "Média MLB": f"{MLB_LEAGUE_PRIORS[0] * 100:.0f}%",
                        f"IA {name_a_clean}": f"{dist_a['Fastball'] * 100:.1f}%",
                        f"IA {name_b_clean}": f"{dist_b['Fastball'] * 100:.1f}%",
                        "Diferença (A - B)": f"{(dist_a['Fastball'] - dist_b['Fastball']) * 100:+.1f}%",
                    },
                    {
                        "Categoria": "BRK • Breaking Ball",
                        "Média MLB": f"{MLB_LEAGUE_PRIORS[1] * 100:.0f}%",
                        f"IA {name_a_clean}": f"{dist_a['Breaking Ball'] * 100:.1f}%",
                        f"IA {name_b_clean}": f"{dist_b['Breaking Ball'] * 100:.1f}%",
                        "Diferença (A - B)": f"{(dist_a['Breaking Ball'] - dist_b['Breaking Ball']) * 100:+.1f}%",
                    },
                    {
                        "Categoria": "OFF • Offspeed",
                        "Média MLB": f"{MLB_LEAGUE_PRIORS[2] * 100:.0f}%",
                        f"IA {name_a_clean}": f"{dist_a['Offspeed'] * 100:.1f}%",
                        f"IA {name_b_clean}": f"{dist_b['Offspeed'] * 100:.1f}%",
                        "Diferença (A - B)": f"{(dist_a['Offspeed'] - dist_b['Offspeed']) * 100:+.1f}%",
                    },
                ]
            )
            st.dataframe(h2h_df, width="stretch", hide_index=True)

    # ---------------------------------------------------------------------------
    # Section 4: Comparative Batter Game Plan (A vs B)
    # ---------------------------------------------------------------------------
    with st.container(border=True):
        st.markdown(
            '<div class="section-header">04. Game Plan Comparativo: Como Ajustar o Swing ao Trocar de Arremessador</div>',
            unsafe_allow_html=True,
        )

        comp_cards: list[str] = []
        is_light = active_theme == "light"
        border_c = "var(--pp-border, #C2CFDD)" if is_light else "var(--pp-border, #222631)"

        for key in ("early_counts", "hitter_counts", "two_strike_protection"):
            phase = comp_plan[key]
            card_html = f"""
            <div style="background:var(--pp-surface,#12141A);border:1px solid {border_c};border-top:3px solid var(--pp-accent,#38BDF8);border-radius:8px;padding:14px;display:flex;flex-direction:column;justify-content:space-between;">
              <div>
                <div style="font-size:13.5px;font-weight:800;color:var(--pp-text,#F1F5F9);margin-bottom:8px;">{phase['title']}</div>
                <div style="display:flex;gap:6px;margin-bottom:10px;">
                  <span style="flex:1;background:var(--pp-surface-alt,#181B23);border-left:3px solid {accent_a};padding:6px 8px;border-radius:4px;font-size:11px;color:var(--pp-text,#F1F5F9);">
                    <strong style="color:{accent_a};display:block;font-size:9.5px;font-family:'JetBrains Mono',monospace;">{name_a_clean[:12]}</strong>
                    {phase['dominant_a']}
                  </span>
                  <span style="flex:1;background:var(--pp-surface-alt,#181B23);border-left:3px solid {accent_b};padding:6px 8px;border-radius:4px;font-size:11px;color:var(--pp-text,#F1F5F9);">
                    <strong style="color:{accent_b};display:block;font-size:9.5px;font-family:'JetBrains Mono',monospace;">{name_b_clean[:12]}</strong>
                    {phase['dominant_b']}
                  </span>
                </div>
                <p style="font-size:12px;color:var(--pp-text,#F1F5F9);opacity:0.9;line-height:1.45;margin-bottom:10px;">{phase['strategy']}</p>
              </div>
              <div style="background:var(--pp-surface-alt,#181B23);border:1px solid {border_c};border-left:3px solid #F59E0B;border-radius:4px;padding:6px 10px;font-size:10.5px;color:var(--pp-text,#F1F5F9);">
                <strong style="font-family:'JetBrains Mono',monospace;color:#F59E0B;">[DISCIPLINA]</strong> {phase['chase_warning']}
              </div>
            </div>
            """
            comp_cards.append(card_html)

        grid_html = f"""
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:12px;margin:6px 0;">
          {"".join(comp_cards)}
        </div>
        """
        st.markdown(_compact_html(grid_html), unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # Section 5: Dual Dugout Cards & Comparative Export
    # ---------------------------------------------------------------------------
    with st.container(border=True):
        st.markdown(
            '<div class="section-header">05. Exportação dos Cartões Imprimíveis de Dugout (A & B)</div>',
            unsafe_allow_html=True,
        )
        plan_a_single = generate_batter_game_plan(pitcher_a_sel, preds_a, priors_a, h2h_stand, risp_active)
        plan_b_single = generate_batter_game_plan(pitcher_b_sel, preds_b, priors_b, h2h_stand, risp_active)

        card_a_html = export_scouting_card_html(
            pitcher_name=pitcher_a_sel,
            predictions_by_count=preds_a,
            game_plan=plan_a_single,
            priors=priors_a,
            stand=h2h_stand,
            risp_active=risp_active,
            theme_mode=active_theme,
        )
        card_b_html = export_scouting_card_html(
            pitcher_name=pitcher_b_sel,
            predictions_by_count=preds_b,
            game_plan=plan_b_single,
            priors=priors_b,
            stand=h2h_stand,
            risp_active=risp_active,
            theme_mode=active_theme,
        )

        slug_a = pitcher_a_sel.lower().split("(")[0].strip().replace(" ", "_")
        slug_b = pitcher_b_sel.lower().split("(")[0].strip().replace(" ", "_")

        h2h_dl_c1, h2h_dl_c2 = st.columns(2, gap="medium")
        with h2h_dl_c1:
            st.download_button(
                label=f"Baixar Cartão Imprimível: {name_a_clean} (HTML / PDF)",
                icon=":material/print:",
                data=card_a_html,
                file_name=f"cartao_tatico_dugout_{slug_a}.html",
                mime="text/html",
                type="primary",
                width="stretch",
            )
        with h2h_dl_c2:
            st.download_button(
                label=f"Baixar Cartão Imprimível: {name_b_clean} (HTML / PDF)",
                icon=":material/print:",
                data=card_b_html,
                file_name=f"cartao_tatico_dugout_{slug_b}.html",
                mime="text/html",
                type="secondary",
                width="stretch",
            )

        with st.expander(f"Visualizar Cartão Tático de {name_a_clean}"):
            components_v1.html(card_a_html, height=520, scrolling=True)

        with st.expander(f"Visualizar Cartão Tático de {name_b_clean}"):
            components_v1.html(card_b_html, height=520, scrolling=True)
