"""
Página 06 — Disfarce de Trajetória 3D • Pitch Tunneling & Break Point (Statcast 3D Aerodynamics).

Análise visual e balística da sobreposição de voo entre dois arremessos do mesmo atleta:
1. Ponto de Decisão / Commitment Point (23.8 pés do home plate, ~150-180 ms antes do contato).
2. Cilindro do Túnel Virtual (primeiros 30 pés onde as trajetórias compartilham a mesma linha de visão).
3. Divergência e Quebra Tardia (IVB vertical e HB horizontal separando os arremessos na placa).
4. Seção transversal 2D no Ponto de Compromisso com validação de sobreposição visual (< 2.94 pol).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pitch_predictor.scenarios import (
    MLB_PITCHERS,
    get_pitcher_names,
    get_pitcher_profile,
)
from pitch_predictor.tunneling_analytics import (
    compute_tunneling_metrics,
    get_pitcher_arsenal,
    render_commitment_cross_section_svg,
    render_tunneling_arena_svg,
)
from pitch_predictor.ui_components import (
    PITCH_CATEGORY_STYLES,
    _compact_html,
    get_custom_css,
    render_baseball_seams_divider_svg,
    render_pitcher_photo_html,
    render_spin_axis_clock_svg,
    resolve_ui_theme_mode,
)

# ---------------------------------------------------------------------------
# Configuração de Tema & Estilos
# ---------------------------------------------------------------------------

active_theme = resolve_ui_theme_mode()

# ---------------------------------------------------------------------------
# Barra de Seleção de Arremessador e Par de Arremessos
# ---------------------------------------------------------------------------

pitcher_names = get_pitcher_names()
if "tunnel_pitcher_name" not in st.session_state or st.session_state.tunnel_pitcher_name not in pitcher_names:
    st.session_state.tunnel_pitcher_name = "Paul Skenes"

selected_pitcher = st.session_state.tunnel_pitcher_name
pitcher_meta = get_pitcher_profile(selected_pitcher)
team_hex = pitcher_meta.get("team_primary_hex", "#F59E0B")
p_throws = pitcher_meta.get("p_throws", "R")

st.markdown(get_custom_css(theme_mode=active_theme, accent_hex=team_hex), unsafe_allow_html=True)

# Header
st.markdown("<div class='pp-kicker'>MLB STATCAST • 3D BALL AERODYNAMICS & TUNNELING</div>", unsafe_allow_html=True)
st.markdown(
    "<h1 style='margin:0;font-size:26px;font-weight:800;letter-spacing:-0.02em;color:var(--pp-text);'>"
    "Disfarce de Trajetória 3D • Pitch Tunneling & Break Point"
    "</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='font-size:13px;color:var(--pp-muted);margin-top:2px;margin-bottom:14px;'>"
    "Simulação balística da sobreposição de voo entre dois arremessos do mesmo arremessador até o Ponto de Decisão (Commitment Point a 23.8 ft)."
    "</div>",
    unsafe_allow_html=True,
)

# Selection Toolbar
with st.container(border=True):
    t_c1, t_c2, t_c3 = st.columns([0.34, 0.33, 0.33], gap="medium")

    with t_c1:
        chosen_pitcher = st.selectbox(
            "Arremessador Oficial MLB",
            options=pitcher_names,
            index=pitcher_names.index(selected_pitcher),
            key="tunnel_pitcher_selector",
        )
        if chosen_pitcher != st.session_state.tunnel_pitcher_name:
            st.session_state.tunnel_pitcher_name = chosen_pitcher
            st.rerun()

    arsenal = get_pitcher_arsenal(chosen_pitcher)
    pitch_keys = list(arsenal.keys())

    with t_c2:
        pitch_a_key = st.selectbox(
            "Arremesso A (Referência / Rápido)",
            options=pitch_keys,
            index=0,
            key="tunnel_pitch_a_selector",
        )

    with t_c3:
        # Default to second pitch if available
        def_b_idx = 1 if len(pitch_keys) > 1 else 0
        pitch_b_key = st.selectbox(
            "Arremesso B (Variação / Quebra)",
            options=pitch_keys,
            index=def_b_idx,
            key="tunnel_pitch_b_selector",
        )

pitch_a = arsenal[pitch_a_key]
pitch_b = arsenal[pitch_b_key]

# Compute precise metrics
metrics = compute_tunneling_metrics(pitch_a, pitch_b, p_throws=p_throws)

st.markdown(render_baseball_seams_divider_svg(width_px=750, theme_mode=active_theme), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Faixa Superior de KPIs de Tunneling
# ---------------------------------------------------------------------------

kpi1, kpi2, kpi3, kpi4 = st.columns(4, gap="medium")

with kpi1, st.container(border=True):
    st.markdown("<div class='pp-kicker'>TUNNELING SCORE</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:22px;font-weight:800;color:{metrics['badge_color']};font-family:\"JetBrains Mono\",monospace;margin:2px 0;'>"
        f"{metrics['tunnel_score']}/100"
        f"</div>"
        f"<div style='font-size:11px;color:var(--pp-text);font-weight:700;'>"
        f"{metrics['grade']}"
        f"</div>",
        unsafe_allow_html=True,
    )

with kpi2, st.container(border=True):
    st.markdown("<div class='pp-kicker'>DISTÂNCIA NO TÚNEL (23.8 FT)</div>", unsafe_allow_html=True)
    sub_ball = metrics["d_tunnel_inches"] <= 2.94
    tag_color = "#10B981" if sub_ball else "#F59E0B"
    tag_label = "Menor que a bola (2.94\")" if sub_ball else "Separação sutil"
    st.markdown(
        f"<div style='font-size:22px;font-weight:800;color:var(--pp-text);font-family:\"JetBrains Mono\",monospace;margin:2px 0;'>"
        f"{metrics['d_tunnel_inches']:.1f} pol"
        f"</div>"
        f"<div style='font-size:11px;color:{tag_color};font-weight:700;'>"
        f"● {tag_label}"
        f"</div>",
        unsafe_allow_html=True,
    )

with kpi3, st.container(border=True):
    st.markdown("<div class='pp-kicker'>QUEBRA NA PLACA (1.4 FT)</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:22px;font-weight:800;color:#0284C7;font-family:\"JetBrains Mono\",monospace;margin:2px 0;'>"
        f"{metrics['d_plate_inches']:.1f} pol"
        f"</div>"
        f"<div style='font-size:11px;color:var(--pp-muted);'>"
        f"Divergência: <strong style='color:var(--pp-text);'>{metrics['tunnel_ratio']:.1f}x</strong> o túnel"
        f"</div>",
        unsafe_allow_html=True,
    )

with kpi4, st.container(border=True):
    st.markdown("<div class='pp-kicker'>DESCOMPASSO DE TEMPO</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:22px;font-weight:800;color:#8B5CF6;font-family:\"JetBrains Mono\",monospace;margin:2px 0;'>"
        f"+{metrics['time_delta_ms']:.0f} ms"
        f"</div>"
        f"<div style='font-size:11px;color:var(--pp-muted);'>"
        f"Diferença de velocidade: <strong style='color:var(--pp-text);'>{metrics['speed_diff_mph']:.1f} mph</strong>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Arena 3D em Perspectiva (SVG Animado com Túnel e Ponto de Decisão)
# ---------------------------------------------------------------------------

with st.container(border=True):
    st.markdown(
        "<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;'>"
        "<div>"
        "<div class='pp-kicker'>ARENA DE VOO EM PERSPECTIVA 3D (VISÃO DO REBATEDOR)</div>"
        "<div style='font-size:15px;font-weight:800;color:var(--pp-text);'>Trajetórias Animadas • Ponto de Decisão a 23.8 ft</div>"
        "</div>"
        f"<span style='background:{team_hex};color:#0A0B0E;font-family:\"JetBrains Mono\",monospace;font-size:11px;font-weight:800;padding:3px 8px;border-radius:4px;'>{pitcher_meta['team_abbr']} • {p_throws}HP</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    arena_svg = render_tunneling_arena_svg(
        pitch_a=pitch_a,
        pitch_b=pitch_b,
        metrics=metrics,
        p_throws=p_throws,
        theme_mode=active_theme,
    )
    st.markdown(arena_svg, unsafe_allow_html=True)

    st.caption(
        "**Física do Túnel:** Até o anel âmbar (*Ponto de Decisão a 23.8 pés*), ambos os arremessos viajam no interior do cilindro virtual de 4 polegadas. "
        "O olho humano leva cerca de 150 ms para disparar o comando de swing do cérebro para os braços — após o anel, o rebatedor já está comprometido com o golpe, "
        "enquanto a bola de quebra curva bruscamente em direção à luva."
    )

# ---------------------------------------------------------------------------
# Análise Detalhada: Seção Transversal e Dossiê Tático
# ---------------------------------------------------------------------------

col_left, col_right = st.columns([0.45, 0.55], gap="large")

with col_left:
    with st.container(border=True):
        st.markdown("<div class='pp-kicker'>CORTE TRANSVERSAL NO PONTO DE DECISÃO</div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:14px;font-weight:800;color:var(--pp-text);margin-bottom:6px;'>"
            "Visão Frontal do Túnel de 4 Polegadas a 23.8 ft"
            "</div>",
            unsafe_allow_html=True,
        )

        cross_svg = render_commitment_cross_section_svg(
            pitch_a=pitch_a,
            pitch_b=pitch_b,
            metrics=metrics,
            theme_mode=active_theme,
        )
        st.markdown(cross_svg, unsafe_allow_html=True)

        st.caption(
            "Círculo tracejado azul = diâmetro de 4 polegadas do túnel virtual. "
            "As duas esferas ilustram a silhueta real de cada bola vista da perspectiva dos olhos do rebatedor."
        )

    with st.container(border=True):
        st.markdown("<div class='pp-kicker'>TABELA DE PARÂMETROS AERODINÂMICOS</div>", unsafe_allow_html=True)
        aero_df = pd.DataFrame([
            {
                "Métrica": "Velocidade de Soltura",
                pitch_a["name"]: f"{pitch_a['speed_mph']} mph",
                pitch_b["name"]: f"{pitch_b['speed_mph']} mph",
                "Divergência": f"Δ {metrics['speed_diff_mph']:.1f} mph",
            },
            {
                "Métrica": "Taxa de Rotação (Spin)",
                pitch_a["name"]: f"{pitch_a['spin_rpm']} RPM",
                pitch_b["name"]: f"{pitch_b['spin_rpm']} RPM",
                "Divergência": f"{abs(pitch_a['spin_rpm'] - pitch_b['spin_rpm']):,} RPM",
            },
            {
                "Métrica": "Eixo de Giro (Spin Axis)",
                pitch_a["name"]: f"{pitch_a['spin_axis']} ({pitch_a['spin_axis_deg']}°)",
                pitch_b["name"]: f"{pitch_b['spin_axis']} ({pitch_b['spin_axis_deg']}°)",
                "Divergência": f"{abs(pitch_a['spin_axis_deg'] - pitch_b['spin_axis_deg'])}° tilt",
            },
            {
                "Métrica": "Quebra Vertical Induzida (IVB)",
                pitch_a["name"]: f"{pitch_a['ivb_inches']:+.1f}\"",
                pitch_b["name"]: f"{pitch_b['ivb_inches']:+.1f}\"",
                "Divergência": f"{abs(pitch_a['ivb_inches'] - pitch_b['ivb_inches']):.1f}\" vertical",
            },
            {
                "Métrica": "Quebra Horizontal (HB)",
                pitch_a["name"]: f"{pitch_a['hb_inches']:+.1f}\"",
                pitch_b["name"]: f"{pitch_b['hb_inches']:+.1f}\"",
                "Divergência": f"{abs(pitch_a['hb_inches'] - pitch_b['hb_inches']):.1f}\" lateral",
            },
        ])
        st.dataframe(aero_df, hide_index=True, width="stretch")

    with st.container(border=True):
        st.markdown("<div class='pp-kicker'>EIXOS DE ROTAÇÃO COMPARATIVOS (SPIN AXIS CLOCK)</div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:13px;font-weight:700;color:var(--pp-text);margin-bottom:6px;'>"
            "Orientação de Tilt e Eficiência Magnus no Momento de Soltura"
            "</div>",
            unsafe_allow_html=True,
        )
        clock_col1, clock_col2 = st.columns(2, gap="small")
        with clock_col1:
            clock_a = render_spin_axis_clock_svg(
                spin_axis=pitch_a["spin_axis"],
                spin_rpm=pitch_a["spin_rpm"],
                pitch_name=pitch_a["name"],
                tilt_deg=pitch_a.get("spin_axis_deg"),
                category=pitch_a["category"],
                theme_mode=active_theme,
            )
            st.markdown(clock_a, unsafe_allow_html=True)
        with clock_col2:
            clock_b = render_spin_axis_clock_svg(
                spin_axis=pitch_b["spin_axis"],
                spin_rpm=pitch_b["spin_rpm"],
                pitch_name=pitch_b["name"],
                tilt_deg=pitch_b.get("spin_axis_deg"),
                category=pitch_b["category"],
                theme_mode=active_theme,
            )
            st.markdown(clock_b, unsafe_allow_html=True)

with col_right:
    with st.container(border=True):
        st.markdown("<div class='pp-kicker'>DOSSIÊ TÁTICO DE ENFRENTAMENTO (SCOUTING)</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:17px;font-weight:800;color:var(--pp-text);margin-bottom:8px;'>"
            f"Como o Rebatedor Processa {pitch_a['code']} vs. {pitch_b['code']}"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="background:var(--pp-surface-alt,#181B23);border:1px solid var(--pp-border,#222631);border-left:4px solid {team_hex};border-radius:6px;padding:12px 14px;margin-bottom:12px;">
              <div style="font-family:'JetBrains Mono',monospace;font-size:10.5px;font-weight:800;color:{team_hex};text-transform:uppercase;margin-bottom:4px;">DIAGNÓSTICO DO DUELO</div>
              <div style="font-size:13px;line-height:1.5;color:var(--pp-text);">{metrics['tactical_desc']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="font-size:12.5px;color:var(--pp-muted);line-height:1.5;">
              <strong style="color:var(--pp-text);">Plano de Ação do Rebatedor no Bastão:</strong>
              <ul style="margin-top:6px;padding-left:18px;">
                <li><strong>Zero Identificação pela Soltura:</strong> O arremessador utiliza o mesmo slot de braço e ponto de liberação, anulando qualquer tentativa de leitura prévia (Pitch Tipping).</li>
                <li><strong>A Janela de 150 Milissegundos:</strong> O rebatedor tem apenas 0.15s para reconhecer o giro das costuras antes do Commitment Point. Atrasar o início do swing em contagens favoráveis (ex: 2-0 ou 3-1) é a única defesa matemática.</li>
                <li><strong>Ajuste de Altura de Swing:</strong> A bola rápida desafia no topo da zona com resistência à gravidade (IVB positivo), enquanto a bola de efeito quebra para fora do plano do taco. Em contagens de 2 strikes, focar exclusivamente na metade inferior reduz a taxa de swing no vazio (Whiff).</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Pitcher Photo Card
    st.markdown(
        render_pitcher_photo_html(selected_pitcher, height_px=140, theme_mode=active_theme),
        unsafe_allow_html=True,
    )
