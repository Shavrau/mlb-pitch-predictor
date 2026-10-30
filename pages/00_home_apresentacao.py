"""
Page 00: Visão Geral & Apresentação Executiva (MLB Pitch Predictor — Broadcast Editorial v2.0).

Interface editorial inspirada em Apple TV+ Friday Night Baseball & Baseball Savant:
- Hero do Desafio dos 400 Milissegundos com barras de tempo proporcionais e tipografia tabular.
- Guia das 3 famílias de arremessos com códigos oficiais do Statcast (`[FF]`, `[SL/ST]`, `[CH/FS]`).
- Demonstrador tático em tempo real isolado em `@st.fragment` com identidade cromática dinâmica
  da franquia MLB do arremessador selecionado e suporte completo a Dual Mode.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from pitch_predictor.fallback_engine import predict_tactical_fallback
from pitch_predictor.scenarios import MLB_PITCHERS
from pitch_predictor.serving_client import (
    DEFAULT_ENDPOINT_NAME,
    PitchServingClient,
)
from pitch_predictor.ui_components import (
    PITCH_CATEGORY_STYLES,
    _compact_html,
    get_custom_css,
    resolve_ui_theme_mode,
)


@st.cache_resource(show_spinner=False)
def get_serving_client(endpoint_name: str = DEFAULT_ENDPOINT_NAME) -> PitchServingClient:
    """Return a cached instance of the Databricks Serverless Model Serving client."""
    return PitchServingClient(endpoint_name=endpoint_name)


def _build_friendly_pitcher_styles() -> dict[str, dict[str, Any]]:
    styles: dict[str, dict[str, Any]] = {}
    for p_name, meta in MLB_PITCHERS.items():
        abbr = meta.get("team_abbr", "MLB")
        jersey = meta.get("jersey_number", "#00")
        hand = f"{meta['p_throws']}HP"
        label = f"{p_name} — {abbr} {jersey} • {meta['team']} ({hand})"
        styles[label] = {
            "pitcher_name": p_name,
            "team_abbr": abbr,
            "jersey_number": jersey,
            "team_primary_hex": meta.get("team_primary_hex", "#38BDF8"),
            "p_throws": meta["p_throws"],
            "pitcher_fb_prior": meta["pitcher_fb_prior"],
            "pitcher_bb_prior": meta["pitcher_bb_prior"],
            "pitcher_os_prior": meta["pitcher_os_prior"],
            "short_desc": f"{meta['signature_arsenal']} • Slot: {meta['arm_slot_type']}",
        }
    styles["Média Geral da MLB — Referência da Liga (RHP)"] = {
        "pitcher_name": "Média Geral da MLB",
        "team_abbr": "MLB",
        "jersey_number": "AVG",
        "team_primary_hex": "#38BDF8",
        "p_throws": "R",
        "pitcher_fb_prior": 0.5543,
        "pitcher_bb_prior": 0.3053,
        "pitcher_os_prior": 0.1404,
        "short_desc": "Distribuição basal da Major League (FB 55.4% • BRK 30.5% • OFF 14.0%)",
    }
    return styles


FRIENDLY_PITCHER_STYLES: dict[str, dict[str, Any]] = _build_friendly_pitcher_styles()

FRIENDLY_GAME_MOMENTS: dict[str, dict[str, Any]] = {
    "[0-0] Início do Duelo — Primeiro Arremesso (0 Bolas, 0 Strikes)": {
        "prev_pitch_category": "First Pitch",
        "prev_pitch_type": "None",
        "prev_pitch_zone": 0,
        "prev_pitch_result": "First Pitch",
        "stand": "L",
        "balls": 0,
        "strikes": 0,
        "outs_when_up": 0,
        "inning": 1,
        "inning_topbot": "Top",
        "score_diff": 0,
        "on_1b_flag": 0,
        "on_2b_flag": 0,
        "on_3b_flag": 0,
        "n_thruorder_pitcher": 1,
        "context_badge": "PRIMEIRO ARREMESSO • 0-0",
        "short_situation": "Duelo equilibrado. O arremessador busca abrir com strike sem expor o centro da placa.",
    },
    "[3-0] Arremessador Encurralado — Risco de Ceder Base (3 Bolas, 0 Strikes)": {
        "prev_pitch_category": "Offspeed",
        "prev_pitch_type": "CH",
        "prev_pitch_zone": 13,
        "prev_pitch_result": "B",
        "stand": "L",
        "balls": 3,
        "strikes": 0,
        "outs_when_up": 1,
        "inning": 6,
        "inning_topbot": "Bot",
        "score_diff": -1,
        "on_1b_flag": 0,
        "on_2b_flag": 1,
        "on_3b_flag": 0,
        "n_thruorder_pitcher": 2,
        "context_badge": "PRESSÃO NO MONTINHO • 3-0 • RISP",
        "short_situation": "Com corredor na 2ª base e 3 bolas fora, errar a próxima bola concede a 1ª base automática (Walk).",
    },
    "[0-2] Rebatedor Contra a Parede — 2 Strikes (Hora de Eliminar)": {
        "prev_pitch_category": "Fastball",
        "prev_pitch_type": "FF",
        "prev_pitch_zone": 2,
        "prev_pitch_result": "S",
        "stand": "R",
        "balls": 0,
        "strikes": 2,
        "outs_when_up": 1,
        "inning": 5,
        "inning_topbot": "Bot",
        "score_diff": 2,
        "on_1b_flag": 0,
        "on_2b_flag": 0,
        "on_3b_flag": 0,
        "n_thruorder_pitcher": 2,
        "context_badge": "VANTAGEM DO ARREMESSADOR • 0-2",
        "short_situation": "Rebatedor a um strike de ser eliminado. Margem tática ideal para expandir zonas de Chase (13 e 14).",
    },
    "[3-2] Pressão Máxima — Bases Cheias & 2 Eliminados (Full Count)": {
        "prev_pitch_category": "Breaking Ball",
        "prev_pitch_type": "SL",
        "prev_pitch_zone": 14,
        "prev_pitch_result": "B",
        "stand": "L",
        "balls": 3,
        "strikes": 2,
        "outs_when_up": 2,
        "inning": 8,
        "inning_topbot": "Top",
        "score_diff": 1,
        "on_1b_flag": 1,
        "on_2b_flag": 1,
        "on_3b_flag": 1,
        "n_thruorder_pitcher": 3,
        "context_badge": "LANCE DECISIVO • 3-2 BASES CHEIAS",
        "short_situation": "Todas as bases ocupadas na 8ª entrada com 2 eliminados e contagem cheia. Qualquer erro altera o placar.",
    },
}


def build_concise_tactical_bullet(
    prediction: str,
    features: dict[str, Any],
) -> str:
    """Generate a concise 1-2 sentence tactical explanation in Portuguese."""
    count_state = str(features.get("count_state", "0-0"))
    leverage = str(features.get("count_leverage", "Neutral"))

    if count_state in {"0-2", "1-2"} or leverage == "Pitcher Ahead":
        return (
            f"Na contagem **{count_state}**, o arremessador tem **vantagem total** e pode expandir a zona. "
            f"A IA projeta um arremesso de efeito ({prediction}) para induzir swing no vazio (Strikeout)."
        )
    if count_state in {"3-0", "3-1"} or leverage == "Pitcher Behind":
        return (
            f"Na contagem **{count_state}**, o arremessador está **encurralado** e evita nova bola fora. "
            f"A IA prevê ataque direto à zona de strike ({prediction}) para impedir a base por bolas (Walk)."
        )
    if int(features.get("scoring_position", 0)) == 1:
        return (
            f"Com corredores prontos para anotar corrida na contagem **{count_state}**, a IA pondera "
            f"o repertório histórico do arremessador ({prediction}) buscando contato fraco ou groundball."
        )
    return (
        f"Na abertura do duelo (**{count_state}**), a IA aponta {prediction} como a chamada de maior eficiência "
        f"para conquistar o primeiro strike mantendo o disfarce de trajetória."
    )


def render_intuitive_result_card_html(
    prediction: str,
    latency_ms: float,
    context_badge: str,
    concise_bullet: str,
    franchise_hex: str = "#38BDF8",
    pitcher_tag: str = "MLB • STATCAST",
) -> str:
    """Render an Apple TV+ / Baseball Savant Broadcast Scorebug result card."""
    style = PITCH_CATEGORY_STYLES.get(prediction, PITCH_CATEGORY_STYLES["Fastball"])
    primary = style["primary"]
    secondary = style["secondary"]
    bg_tint = style["bg_tint"]
    icon = style["icon"]
    code_badge = style.get("code_badge", "STATCAST [AI]")
    label_pt = style["label_pt"]

    parts = concise_bullet.split("**")
    formatted: list[str] = []
    for idx, part in enumerate(parts):
        if idx % 2 == 1:
            formatted.append(f'<strong style="color:var(--pp-text,#F1F5F9);">{part}</strong>')
        else:
            formatted.append(part)
    clean_bullet = "".join(formatted)

    raw_html = f"""
    <div style="background:var(--pp-surface,#12141A);border:1px solid var(--pp-border,#222631);border-top:3px solid {franchise_hex};border-radius:8px;padding:18px 20px;margin-top:12px;">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:12px;">
        <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
          <span style="background:{franchise_hex};color:#0A0B0E;font-family:'JetBrains Mono',monospace;font-size:10.5px;font-weight:800;letter-spacing:0.05em;padding:3px 8px;border-radius:4px;">{pitcher_tag}</span>
          <span style="background:{secondary};color:#FFFFFF;font-family:'JetBrains Mono',monospace;font-size:10.5px;font-weight:800;letter-spacing:0.05em;padding:3px 9px;border-radius:4px;text-transform:uppercase;">{context_badge}</span>
        </div>
        <span style="background:var(--pp-surface-alt,#181B23);color:var(--pp-text,#F1F5F9);font-family:'JetBrains Mono',monospace;font-variant-numeric:tabular-nums;font-size:11px;font-weight:700;padding:3px 10px;border-radius:4px;border:1px solid var(--pp-border,#222631);">Resposta Databricks Serverless: {latency_ms:.1f} ms</span>
      </div>
      <div style="background:{bg_tint};border-left:3px solid {primary};border-radius:4px;padding:12px 16px;margin-bottom:12px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
        <div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:800;color:var(--pp-muted,#94A3B8);text-transform:uppercase;letter-spacing:0.06em;">DECISÃO PREVISTA PELA INTELIGÊNCIA ARTIFICIAL</div>
          <div style="font-size:24px;font-weight:800;color:var(--pp-text,#F1F5F9);margin-top:2px;">{icon} {label_pt} <span style="font-size:15px;color:{primary};font-weight:700;">({prediction})</span></div>
        </div>
        <span style="font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:800;color:{primary};border:1px solid {primary};padding:3px 9px;border-radius:4px;">{code_badge}</span>
      </div>
      <div style="background:var(--pp-surface-alt,#181B23);border:1px solid var(--pp-border,#222631);border-radius:6px;padding:10px 14px;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:10.5px;font-weight:800;color:#F59E0B;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:4px;">Por que a IA escolheu este arremesso?</div>
        <div style="font-size:13px;color:var(--pp-text,#F1F5F9);opacity:0.92;line-height:1.45;">{clean_bullet}</div>
      </div>
    </div>
    """
    return _compact_html(raw_html)


if "home_pitcher_style" not in st.session_state or st.session_state.home_pitcher_style not in FRIENDLY_PITCHER_STYLES:
    st.session_state.home_pitcher_style = next(iter(FRIENDLY_PITCHER_STYLES.keys()))
if "home_game_moment" not in st.session_state or st.session_state.home_game_moment not in FRIENDLY_GAME_MOMENTS:
    st.session_state.home_game_moment = next(iter(FRIENDLY_GAME_MOMENTS.keys()))
if "home_demo_result" not in st.session_state:
    st.session_state.home_demo_result = None

active_theme = resolve_ui_theme_mode()
active_style_cfg = FRIENDLY_PITCHER_STYLES[st.session_state.home_pitcher_style]
st.markdown(
    get_custom_css(theme_mode=active_theme, accent_hex=active_style_cfg["team_primary_hex"]),
    unsafe_allow_html=True,
)
client = get_serving_client()

# ===========================================================================
# 1. EDITORIAL BROADCAST HERO — O DESAFIO DOS 400 MILISSEGUNDOS
# ===========================================================================
hero_html = _compact_html(
    """
    <div style="background:var(--pp-surface,#12141A);border:1px solid var(--pp-border,#222631);border-top:3px solid var(--pp-accent,#38BDF8);border-radius:8px;padding:20px 24px;margin-bottom:16px;">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">
        <div>
          <span style="background:var(--pp-surface-alt,#181B23);color:var(--pp-muted,#94A3B8);border:1px solid var(--pp-border,#222631);font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:800;padding:3px 9px;border-radius:4px;letter-spacing:0.06em;text-transform:uppercase;">DATABRICKS SERVERLESS AI • MLB STATCAST TELEMETRY</span>
          <h1 class="pp-display" style="font-size:26px;font-weight:800;color:var(--pp-text,#F1F5F9);margin:8px 0 4px 0;text-transform:uppercase;">MLB Pitch Predictor — Decodificando o Arremesso com Inteligência Artificial</h1>
          <p style="font-size:14px;color:var(--pp-muted,#94A3B8);margin:0;">A bola cruza 18,44 metros a <strong style="color:var(--pp-text,#F1F5F9);font-family:'JetBrains Mono',monospace;">160 km/h</strong> em <strong style="color:var(--pp-text,#F1F5F9);font-family:'JetBrains Mono',monospace;">0.40 segundos</strong>. O modelo antecipa a chamada tática antes de ele acontecer.</p>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-top:16px;">
        <div style="background:var(--pp-surface-alt,#181B23);border:1px solid var(--pp-border,#222631);border-left:3px solid #EF4444;border-radius:6px;padding:11px 14px;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:9.5px;color:var(--pp-muted,#94A3B8);font-weight:800;letter-spacing:0.05em;">DESAFIO DOS 400 MILISSEGUNDOS</div>
          <div style="font-family:'JetBrains Mono',monospace;font-variant-numeric:tabular-nums;font-size:21px;font-weight:800;color:#EF4444;margin:3px 0;">0.40 segundos</div>
          <div style="width:100%;height:4px;background:var(--pp-border,#222631);border-radius:2px;margin:5px 0;"><div style="width:100%;height:100%;background:#EF4444;border-radius:2px;"></div></div>
          <div style="font-size:11.5px;color:var(--pp-muted,#94A3B8);">Tempo total de voo da bola a 160 km/h</div>
        </div>
        <div style="background:var(--pp-surface-alt,#181B23);border:1px solid var(--pp-border,#222631);border-left:3px solid #F59E0B;border-radius:6px;padding:11px 14px;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:9.5px;color:var(--pp-muted,#94A3B8);font-weight:800;letter-spacing:0.05em;">JANELA NEURAL HUMANA</div>
          <div style="font-family:'JetBrains Mono',monospace;font-variant-numeric:tabular-nums;font-size:21px;font-weight:800;color:#F59E0B;margin:3px 0;">0.15 segundos</div>
          <div style="width:100%;height:4px;background:var(--pp-border,#222631);border-radius:2px;margin:5px 0;"><div style="width:37.5%;height:100%;background:#F59E0B;border-radius:2px;"></div></div>
          <div style="font-size:11.5px;color:var(--pp-muted,#94A3B8);">Tempo de o olho processar o movimento</div>
        </div>
        <div style="background:var(--pp-surface-alt,#181B23);border:1px solid var(--pp-border,#222631);border-left:3px solid #10B981;border-radius:6px;padding:11px 14px;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:9.5px;color:var(--pp-muted,#94A3B8);font-weight:800;letter-spacing:0.05em;">INFERÊNCIA PRÉ-ARREMESSO</div>
          <div style="font-family:'JetBrains Mono',monospace;font-variant-numeric:tabular-nums;font-size:21px;font-weight:800;color:#10B981;margin:3px 0;">24 Variáveis</div>
          <div style="width:100%;height:4px;background:var(--pp-border,#222631);border-radius:2px;margin:5px 0;"><div style="width:12.5%;height:100%;background:#10B981;border-radius:2px;"></div></div>
          <div style="font-size:11.5px;color:var(--pp-muted,#94A3B8);">Contexto Statcast + cinemática 3D (&lt;50 ms)</div>
        </div>
      </div>
    </div>
    """
)
st.markdown(hero_html, unsafe_allow_html=True)

# ===========================================================================
# 2. TAXONOMIA STATCAST: AS 3 FAMÍLIAS DE ARREMESSO
# ===========================================================================
cards_pitches_html = _compact_html(
    """
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin-bottom:16px;">
      <div style="background:var(--pp-surface,#12141A);border:1px solid var(--pp-border,#222631);border-top:3px solid #EF4444;border-radius:8px;padding:12px 15px;">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;">
          <span style="font-size:14.5px;font-weight:800;color:var(--pp-text,#F1F5F9);">▲ Bola Rápida (Fastball)</span>
          <span style="font-family:'JetBrains Mono',monospace;background:rgba(239,68,68,0.14);color:#EF4444;border:1px solid #EF4444;font-size:10px;font-weight:800;padding:2px 6px;border-radius:4px;">FF/SI • ~160 km/h</span>
        </div>
        <div style="font-size:12px;color:var(--pp-muted,#94A3B8);margin-top:6px;line-height:1.4;">Trajetória explosiva de alta sustentação vertical. Desafia o tempo de reação na zona.</div>
      </div>
      <div style="background:var(--pp-surface,#12141A);border:1px solid var(--pp-border,#222631);border-top:3px solid #3B82F6;border-radius:8px;padding:12px 15px;">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;">
          <span style="font-size:14.5px;font-weight:800;color:var(--pp-text,#F1F5F9);">◆ Bola com Efeito (Breaking Ball)</span>
          <span style="font-family:'JetBrains Mono',monospace;background:rgba(59,130,246,0.14);color:#3B82F6;border:1px solid #3B82F6;font-size:10px;font-weight:800;padding:2px 6px;border-radius:4px;">SL/ST • ~135 km/h</span>
        </div>
        <div style="font-size:12px;color:var(--pp-muted,#94A3B8);margin-top:6px;line-height:1.4;">Sliders, Sweepers e Curvas de alto giro (2800+ RPM) projetados para whiff nas bordas.</div>
      </div>
      <div style="background:var(--pp-surface,#12141A);border:1px solid var(--pp-border,#222631);border-top:3px solid #10B981;border-radius:8px;padding:12px 15px;">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;">
          <span style="font-size:14.5px;font-weight:800;color:var(--pp-text,#F1F5F9);">● Mudança de Velocidade (Offspeed)</span>
          <span style="font-family:'JetBrains Mono',monospace;background:rgba(16,185,129,0.14);color:#10B981;border:1px solid #10B981;font-size:10px;font-weight:800;padding:2px 6px;border-radius:4px;">CH/FS • ~125 km/h</span>
        </div>
        <div style="font-size:12px;color:var(--pp-muted,#94A3B8);margin-top:6px;line-height:1.4;">Changeup e Splitter com disfarce de trajetória idêntico à bola rápida e queda tardia.</div>
      </div>
    </div>
    """
)
st.markdown(cards_pitches_html, unsafe_allow_html=True)


# ===========================================================================
# 3. DEMONSTRADOR TÁTICO INTERATIVO EM TEMPO REAL (@st.fragment)
# ===========================================================================
@st.fragment
def _render_live_tactical_demo() -> None:
    with st.container(border=True):
        st.markdown(
            '<div class="section-header">Simulador Tático Instantâneo • Databricks Model Serving</div>',
            unsafe_allow_html=True,
        )

        col_style, col_moment = st.columns(2)
        with col_style:
            chosen_style = st.selectbox(
                "Perfil Oficial do Arremessador (MLB Roster)",
                options=list(FRIENDLY_PITCHER_STYLES.keys()),
                key="home_pitcher_style",
                help="Escolha o astro da MLB no montinho para aplicar seus priors e lateralidade reais",
            )
            st.caption(FRIENDLY_PITCHER_STYLES[chosen_style]["short_desc"])

        with col_moment:
            chosen_moment = st.selectbox(
                "Situação Tática da Partida",
                options=list(FRIENDLY_GAME_MOMENTS.keys()),
                key="home_game_moment",
                help="Escolha o cenário de contagem e pressão nas bases",
            )
            st.caption(FRIENDLY_GAME_MOMENTS[chosen_moment]["short_situation"])

        style_cfg = FRIENDLY_PITCHER_STYLES[chosen_style]
        moment_cfg = FRIENDLY_GAME_MOMENTS[chosen_moment]

        demo_raw_inputs: dict[str, Any] = {
            "prev_pitch_category": moment_cfg["prev_pitch_category"],
            "prev_pitch_type": moment_cfg["prev_pitch_type"],
            "prev_pitch_zone": moment_cfg["prev_pitch_zone"],
            "prev_pitch_result": moment_cfg["prev_pitch_result"],
            "p_throws": style_cfg["p_throws"],
            "stand": moment_cfg["stand"],
            "balls": moment_cfg["balls"],
            "strikes": moment_cfg["strikes"],
            "outs_when_up": moment_cfg["outs_when_up"],
            "inning": moment_cfg["inning"],
            "inning_topbot": moment_cfg["inning_topbot"],
            "score_diff": moment_cfg["score_diff"],
            "on_1b_flag": moment_cfg["on_1b_flag"],
            "on_2b_flag": moment_cfg["on_2b_flag"],
            "on_3b_flag": moment_cfg["on_3b_flag"],
            "n_thruorder_pitcher": moment_cfg["n_thruorder_pitcher"],
            "pitcher_fb_prior": style_cfg["pitcher_fb_prior"],
            "pitcher_bb_prior": style_cfg["pitcher_bb_prior"],
            "pitcher_os_prior": style_cfg["pitcher_os_prior"],
        }

        quick_predict_clicked = st.button(
            "Consultar Inteligência Artificial em Tempo Real",
            key="btn_home_quick_predict",
            icon=":material/online_prediction:",
            type="primary",
            width="stretch",
        )

        if quick_predict_clicked:
            with st.spinner("Consultando endpoint Databricks Serverless..."):
                try:
                    pred_res = client.predict(demo_raw_inputs)
                except Exception:  # noqa: BLE001
                    pred_res = predict_tactical_fallback(demo_raw_inputs)

                concise_bullet = build_concise_tactical_bullet(
                    prediction=pred_res["prediction"],
                    features=pred_res["features"],
                )
                context_badge = moment_cfg["context_badge"]
                if pred_res.get("is_fallback"):
                    context_badge += " • FALLBACK LOCAL"

                st.session_state.home_demo_result = {
                    "prediction": pred_res["prediction"],
                    "latency_ms": pred_res["latency_ms"],
                    "bullet": concise_bullet,
                    "context_badge": context_badge,
                    "features": pred_res["features"],
                    "franchise_hex": style_cfg["team_primary_hex"],
                    "pitcher_tag": f"{style_cfg['team_abbr']} {style_cfg['jersey_number']} • {style_cfg['pitcher_name'].upper()}",
                }

        if st.session_state.home_demo_result:
            res = st.session_state.home_demo_result
            st.markdown(
                render_intuitive_result_card_html(
                    prediction=res["prediction"],
                    latency_ms=res["latency_ms"],
                    context_badge=res["context_badge"],
                    concise_bullet=res["bullet"],
                    franchise_hex=res.get("franchise_hex", style_cfg["team_primary_hex"]),
                    pitcher_tag=res.get("pitcher_tag", "MLB • STATCAST"),
                ),
                unsafe_allow_html=True,
            )


_render_live_tactical_demo()

# ===========================================================================
# 4. PIPELINE DE ARQUITETURA LAKEHOUSE SERVERLESS
# ===========================================================================
arch_summary_html = _compact_html(
    """
    <div style="background:var(--pp-surface,#12141A);border:1px solid var(--pp-border,#222631);border-radius:8px;padding:12px 16px;margin:14px 0;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:800;color:var(--pp-muted,#94A3B8);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;">PIPELINE END-TO-END • DATABRICKS UNITY CATALOG &amp; MODEL SERVING</div>
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;font-family:'JetBrains Mono',monospace;font-size:11.5px;color:var(--pp-text,#F1F5F9);">
        <span><strong>01.</strong> Ingestão Statcast &amp; Savant MP4</span>
        <span style="color:var(--pp-muted,#94A3B8);">/</span>
        <span><strong>02.</strong> Lakeflow Spark CDC (Zero-Leakage)</span>
        <span style="color:var(--pp-muted,#94A3B8);">/</span>
        <span><strong>03.</strong> MediaPipe 3D Pose + MLflow @prod</span>
        <span style="color:var(--pp-muted,#94A3B8);">/</span>
        <span><strong>04.</strong> Serverless REST (&lt;50 ms)</span>
      </div>
    </div>
    """
)
st.markdown(arch_summary_html, unsafe_allow_html=True)

# ===========================================================================
# 5. NAVEGAÇÃO RÁPIDA ENTRE MÓDULOS
# ===========================================================================
nav_c1, nav_c2, nav_c3 = st.columns(3, gap="medium")
with nav_c1, st.container(border=True):
    st.markdown("<div class='pp-kicker'>MÓDULO 01 • ARCADE RETRO</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:16px;font-weight:800;color:var(--pp-text);margin-bottom:6px;'>"
        "🎮 Minigame: Você vs. IA"
        "</div>"
        "<div style='font-size:12.5px;color:var(--pp-muted);line-height:1.45;min-height:54px;margin-bottom:10px;'>"
        "Duelo lance a lance no montinho. Escolha seu arremesso, mire na zona de strike e tente enganar a IA."
        "</div>",
        unsafe_allow_html=True,
    )
    st.page_link(
        "pages/03_minigame.py",
        label="Jogar Minigame →",
        icon=":material/sports_esports:",
        width="stretch",
    )

with nav_c2, st.container(border=True):
    st.markdown("<div class='pp-kicker'>MÓDULO 02 • STATCAST INTEL</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:16px;font-weight:800;color:var(--pp-text);margin-bottom:6px;'>"
        "📊 MLB Scouting Report"
        "</div>"
        "<div style='font-size:12.5px;color:var(--pp-muted);line-height:1.45;min-height:54px;margin-bottom:10px;'>"
        "Matriz 4x3 de 12 contagens, plano tático de jogo e tendências situacionais em contagens favoráveis e desfavoráveis."
        "</div>",
        unsafe_allow_html=True,
    )
    st.page_link(
        "pages/02_scout_report.py",
        label="Abrir Scouting Report →",
        icon=":material/analytics:",
        width="stretch",
    )

with nav_c3, st.container(border=True):
    st.markdown("<div class='pp-kicker'>MÓDULO 03 • VISÃO COMPUTACIONAL 3D</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:16px;font-weight:800;color:var(--pp-text);margin-bottom:6px;'>"
        "🔬 Laboratório de Biomecânica"
        "</div>"
        "<div style='font-size:12.5px;color:var(--pp-muted);line-height:1.45;min-height:54px;margin-bottom:10px;'>"
        "Cinemática 3D com 33 marcos da entrega, detecção de pitch tipping no release frame e estudo de ablação SHAP."
        "</div>",
        unsafe_allow_html=True,
    )
    st.page_link(
        "pages/04_pitch_vision_lab.py",
        label="Abrir Laboratório de Visão →",
        icon=":material/biotech:",
        width="stretch",
    )
