"""
Page 00: Visão Geral & Apresentação Descomplicada (MLB Pitch Predictor).

Interface visual e intuitiva projetada para apresentações rápidas, permitindo que
qualquer pessoa compreenda o problema e experimente a IA em poucos segundos,
sem necessidade de conhecimento prévio de beisebol ou jargões técnicos.
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
)


@st.cache_resource(show_spinner=False)
def get_serving_client(endpoint_name: str = DEFAULT_ENDPOINT_NAME) -> PitchServingClient:
    """Return a cached instance of the Databricks Serverless Model Serving client."""
    return PitchServingClient(endpoint_name=endpoint_name)


FRIENDLY_PITCHER_STYLES: dict[str, dict[str, Any]] = {
    f"🚀 Paul Skenes ({MLB_PITCHERS['Paul Skenes']['team']})": {
        "p_throws": MLB_PITCHERS["Paul Skenes"]["p_throws"],
        "pitcher_fb_prior": MLB_PITCHERS["Paul Skenes"]["pitcher_fb_prior"],
        "pitcher_bb_prior": MLB_PITCHERS["Paul Skenes"]["pitcher_bb_prior"],
        "pitcher_os_prior": MLB_PITCHERS["Paul Skenes"]["pitcher_os_prior"],
        "short_desc": f"102 mph Fastball / Splinker • {MLB_PITCHERS['Paul Skenes']['arm_slot_type']}",
    },
    f"🌀 Corbin Burnes ({MLB_PITCHERS['Corbin Burnes']['team']})": {
        "p_throws": MLB_PITCHERS["Corbin Burnes"]["p_throws"],
        "pitcher_fb_prior": MLB_PITCHERS["Corbin Burnes"]["pitcher_fb_prior"],
        "pitcher_bb_prior": MLB_PITCHERS["Corbin Burnes"]["pitcher_bb_prior"],
        "pitcher_os_prior": MLB_PITCHERS["Corbin Burnes"]["pitcher_os_prior"],
        "short_desc": f"Cutter / Sweeper Specialist (2800+ RPM) • {MLB_PITCHERS['Corbin Burnes']['arm_slot_type']}",
    },
    f"🛑 Logan Webb ({MLB_PITCHERS['Logan Webb']['team']})": {
        "p_throws": MLB_PITCHERS["Logan Webb"]["p_throws"],
        "pitcher_fb_prior": MLB_PITCHERS["Logan Webb"]["pitcher_fb_prior"],
        "pitcher_bb_prior": MLB_PITCHERS["Logan Webb"]["pitcher_bb_prior"],
        "pitcher_os_prior": MLB_PITCHERS["Logan Webb"]["pitcher_os_prior"],
        "short_desc": f"Sinker / Changeup Master (Groundball King) • {MLB_PITCHERS['Logan Webb']['arm_slot_type']}",
    },
    f"🏹 Chris Sale ({MLB_PITCHERS['Chris Sale']['team']})": {
        "p_throws": MLB_PITCHERS["Chris Sale"]["p_throws"],
        "pitcher_fb_prior": MLB_PITCHERS["Chris Sale"]["pitcher_fb_prior"],
        "pitcher_bb_prior": MLB_PITCHERS["Chris Sale"]["pitcher_bb_prior"],
        "pitcher_os_prior": MLB_PITCHERS["Chris Sale"]["pitcher_os_prior"],
        "short_desc": f"Lefty Crafty Ace (NL Cy Young '24) • {MLB_PITCHERS['Chris Sale']['arm_slot_type']}",
    },
    f"🎌 Yoshinobu Yamamoto ({MLB_PITCHERS['Yoshinobu Yamamoto']['team']})": {
        "p_throws": MLB_PITCHERS["Yoshinobu Yamamoto"]["p_throws"],
        "pitcher_fb_prior": MLB_PITCHERS["Yoshinobu Yamamoto"]["pitcher_fb_prior"],
        "pitcher_bb_prior": MLB_PITCHERS["Yoshinobu Yamamoto"]["pitcher_bb_prior"],
        "pitcher_os_prior": MLB_PITCHERS["Yoshinobu Yamamoto"]["pitcher_os_prior"],
        "short_desc": f"Devastating Splitter / Rainbow Curve • {MLB_PITCHERS['Yoshinobu Yamamoto']['arm_slot_type']}",
    },
    "⚖️ Média Geral da MLB (Padrão da Liga)": {
        "p_throws": "R",
        "pitcher_fb_prior": 0.5543,
        "pitcher_bb_prior": 0.3053,
        "pitcher_os_prior": 0.1404,
        "short_desc": "Repertório equilibrado padrão da Major League",
    },
}

FRIENDLY_GAME_MOMENTS: dict[str, dict[str, Any]] = {
    "🟢 Início do Duelo (0 Bolas, 0 Strikes — Primeiro Arremesso 0-0)": {
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
        "short_situation": "Duelo equilibrado. O arremessador busca abrir com Strike sem correr riscos.",
    },
    "🟡 Arremessador Encurralado (3 Bolas, 0 Strikes — Risco de ceder base!)": {
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
        "context_badge": "PRESSÃO TOTAL • 3-0",
        "short_situation": "O arremessador já errou 3 bolas. Se errar a próxima, cede a 1ª base por bolas (Walk).",
    },
    "🔴 Rebatedor Contra a Parede (0 Bolas, 2 Strikes — Hora de Eliminar!)": {
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
        "short_situation": "Rebatedor a um strike de ser eliminado. O arremessador tem margem para arriscar efeitos fora da zona de strike.",
    },
    "💥 Pressão Máxima (Bases Cheias com Corredores em Posição de Pontuar)": {
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
        "short_situation": "Todas as bases ocupadas, 2 eliminados e contagem cheia. Qualquer erro decide o jogo.",
    },
}


def build_concise_tactical_bullet(
    prediction: str,
    features: dict[str, Any],
) -> str:
    """Generate punchy, 1-2 sentence tactical bullet explaining the AI decision."""
    count_state = str(features.get("count_state", "0-0"))
    leverage = str(features.get("count_leverage", "Neutral"))

    if count_state in {"0-2", "1-2"} or leverage == "Pitcher Ahead":
        return (
            f"Na contagem **{count_state}**, o arremessador tem **vantagem total** e pode expandir a zona. "
            f"A IA projeta um arremesso de efeito para fazer o rebatedor abanar no vazio (Strikeout)."
        )
    if count_state in {"3-0", "3-1"} or leverage == "Pitcher Behind":
        return (
            f"Na contagem **{count_state}**, o arremessador está **encurralado** e não pode arriscar nova bola fora. "
            f"A IA prevê ataque direto à zona de strike para evitar a base por bolas (Walk)."
        )
    if int(features.get("scoring_position", 0)) == 1:
        return (
            f"Com corredores prontos para anotar corrida na contagem **{count_state}**, a IA equilibra "
            f"o repertório histórico do arremessador para induzir contato fraco ou groundball."
        )
    return (
        f"No início do duelo (**{count_state}**), a IA indica a escolha mais eficiente para abrir a contagem "
        f"conquistando o primeiro strike sem correr riscos desnecessários."
    )


def render_intuitive_result_card_html(
    prediction: str,
    latency_ms: float,
    context_badge: str,
    concise_bullet: str,
) -> str:
    """Render an ultra-clean, punchy result card without text overload."""
    style = PITCH_CATEGORY_STYLES.get(prediction, PITCH_CATEGORY_STYLES["Fastball"])
    primary = style["primary"]
    secondary = style["secondary"]
    glow = style["glow"]
    bg_tint = style["bg_tint"]
    icon = style["icon"]
    label_pt = style["label_pt"]

    # Format bold tags cleanly
    parts = concise_bullet.split("**")
    formatted: list[str] = []
    for idx, part in enumerate(parts):
        if idx % 2 == 1:
            formatted.append(f'<strong style="color:#F8FAFC;">{part}</strong>')
        else:
            formatted.append(part)
    clean_bullet = "".join(formatted)

    raw_html = f"""
    <div style="background:linear-gradient(135deg,#131D2E 0%,#0A0E17 100%);border:2px solid {primary};border-radius:14px;padding:18px 22px;box-shadow:0 8px 24px {glow};margin-top:14px;">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:12px;">
        <span style="background:{secondary};color:#FFFFFF;font-size:11px;font-weight:800;letter-spacing:0.06em;padding:4px 10px;border-radius:6px;text-transform:uppercase;">{context_badge}</span>
        <span style="background:#0F172A;color:#38BDF8;font-size:11.5px;font-weight:700;padding:4px 10px;border-radius:6px;border:1px solid #334155;">⚡ Resposta Databricks: {latency_ms:.1f} ms</span>
      </div>
      <div style="background:{bg_tint};border-left:5px solid {primary};border-radius:10px;padding:12px 16px;margin-bottom:12px;">
        <div style="font-size:10.5px;font-weight:800;color:#94A3B8;text-transform:uppercase;letter-spacing:0.06em;">PREVISÃO DA INTELIGÊNCIA ARTIFICIAL</div>
        <div style="font-size:26px;font-weight:900;color:#F8FAFC;margin-top:2px;">{icon} {label_pt} <span style="font-size:17px;color:{primary};font-weight:700;">({prediction})</span></div>
      </div>
      <div style="background:#0B0F19;border:1px solid #26354A;border-radius:8px;padding:10px 14px;">
        <div style="font-size:11px;font-weight:800;color:#F59E0B;text-transform:uppercase;margin-bottom:4px;">💡 Por que a IA escolheu este arremesso?</div>
        <div style="font-size:13px;color:#CBD5E1;line-height:1.45;">{clean_bullet}</div>
      </div>
    </div>
    """
    return _compact_html(raw_html)


# Initialize session state for Quick Demo
if "home_pitcher_style" not in st.session_state or st.session_state.home_pitcher_style not in FRIENDLY_PITCHER_STYLES:
    st.session_state.home_pitcher_style = next(iter(FRIENDLY_PITCHER_STYLES.keys()))
if "home_game_moment" not in st.session_state or st.session_state.home_game_moment not in FRIENDLY_GAME_MOMENTS:
    st.session_state.home_game_moment = next(iter(FRIENDLY_GAME_MOMENTS.keys()))
if "home_demo_result" not in st.session_state:
    st.session_state.home_demo_result = None

st.markdown(get_custom_css(), unsafe_allow_html=True)
client = get_serving_client()

# ===========================================================================
# 1. HERO VISUAL — DIRETO AO PONTO (O DESAFIO DOS 400 MILISSEGUNDOS)
# ===========================================================================
hero_html = _compact_html(
    """
    <div style="background:linear-gradient(135deg,#0E1626 0%,#141E30 50%,#090D15 100%);border:1px solid #26354A;border-left:5px solid #38BDF8;border-radius:14px;padding:20px 24px;margin-bottom:18px;">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
        <div>
          <span style="background:rgba(56,189,248,0.15);color:#38BDF8;font-size:10.5px;font-weight:800;padding:3px 10px;border-radius:999px;letter-spacing:0.06em;text-transform:uppercase;">DATABRICKS SERVERLESS AI • MLB STATCAST</span>
          <h1 style="font-size:24px;font-weight:900;color:#F8FAFC;margin:6px 0 4px 0;">⚾ MLB Pitch Predictor — Decodificando o Arremesso com Inteligência Artificial</h1>
          <p style="font-size:14px;color:#94A3B8;margin:0;">A bola chega a <strong>160 km/h</strong> em <strong>0.40 segundos</strong>. A Inteligência Artificial antecipa o arremesso antes dele acontecer.</p>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-top:14px;">
        <div style="background:#080C14;border:1px solid #26354A;border-top:3px solid #EF4444;border-radius:10px;padding:10px 14px;">
          <div style="font-size:10px;color:#94A3B8;font-weight:800;text-transform:uppercase;">DESAFIO DOS 400 MILISSEGUNDOS</div>
          <div style="font-size:20px;font-weight:900;color:#EF4444;margin:2px 0;">⚡ 0.40 segundos</div>
          <div style="font-size:11.5px;color:#94A3B8;">Tempo de voo da bola a 160 km/h</div>
        </div>
        <div style="background:#080C14;border:1px solid #26354A;border-top:3px solid #F59E0B;border-radius:10px;padding:10px 14px;">
          <div style="font-size:10px;color:#94A3B8;font-weight:800;text-transform:uppercase;">LIMITE HUMANO</div>
          <div style="font-size:20px;font-weight:900;color:#F59E0B;margin:2px 0;">🧠 0.15 segundos</div>
          <div style="font-size:11.5px;color:#94A3B8;">Tempo do olho processar o movimento</div>
        </div>
        <div style="background:#080C14;border:1px solid #26354A;border-top:3px solid #10B981;border-radius:10px;padding:10px 14px;">
          <div style="font-size:10px;color:#94A3B8;font-weight:800;text-transform:uppercase;">NOSSO MODELO</div>
          <div style="font-size:20px;font-weight:900;color:#10B981;margin:2px 0;">🎯 24 Variáveis</div>
          <div style="font-size:11.5px;color:#94A3B8;">Previsão instantânea pré-arremesso</div>
        </div>
      </div>
    </div>
    """
)
st.markdown(hero_html, unsafe_allow_html=True)

# ===========================================================================
# 2. GUIA VISUAL: AS 3 FAMÍLIAS DE ARREMESSO (EM 3 CARDS RÁPIDOS)
# ===========================================================================
cards_pitches_html = _compact_html(
    """
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin-bottom:18px;">
      <div style="background:rgba(239,68,68,0.08);border:1px solid #EF4444;border-radius:10px;padding:12px 14px;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <span style="font-size:15px;font-weight:900;color:#F8FAFC;">⚡ Bola Rápida (Fastball)</span>
          <span style="background:#EF4444;color:#FFF;font-size:10px;font-weight:800;padding:2px 6px;border-radius:4px;">~160 km/h</span>
        </div>
        <div style="font-size:12px;color:#CBD5E1;margin-top:6px;line-height:1.35;">Reta e potente. Vence o rebatedor na pura velocidade de reação.</div>
      </div>
      <div style="background:rgba(59,130,246,0.08);border:1px solid #3B82F6;border-radius:10px;padding:12px 14px;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <span style="font-size:15px;font-weight:900;color:#F8FAFC;">🌀 Bola com Efeito (Breaking Ball)</span>
          <span style="background:#3B82F6;color:#FFF;font-size:10px;font-weight:800;padding:2px 6px;border-radius:4px;">~135 km/h</span>
        </div>
        <div style="font-size:12px;color:#CBD5E1;margin-top:6px;line-height:1.35;">Curvas e Sliders. Desvia repentinamente no ar para induzir o erro.</div>
      </div>
      <div style="background:rgba(16,185,129,0.08);border:1px solid #10B981;border-radius:10px;padding:12px 14px;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <span style="font-size:15px;font-weight:900;color:#F8FAFC;">🛑 Mudança de Velocidade (Offspeed)</span>
          <span style="background:#10B981;color:#FFF;font-size:10px;font-weight:800;padding:2px 6px;border-radius:4px;">~125 km/h</span>
        </div>
        <div style="font-size:12px;color:#CBD5E1;margin-top:6px;line-height:1.35;">Changeup. Parece rápida, mas chega lenta quebrando o tempo do swing.</div>
      </div>
    </div>
    """
)
st.markdown(cards_pitches_html, unsafe_allow_html=True)

# ===========================================================================
# 3. DEMONSTRADOR RÁPIDO INTERATIVO (O CORAÇÃO INTUITIVO DA APRESENTAÇÃO)
# ===========================================================================
with st.container(border=True):
    st.markdown(
        '<div style="font-size:16px;font-weight:800;color:#F8FAFC;margin-bottom:8px;">🎮 Experimente a Inteligência Artificial ao Vivo</div>',
        unsafe_allow_html=True,
    )

    col_style, col_moment = st.columns(2)
    with col_style:
        chosen_style = st.selectbox(
            "👤 Estilo do Arremessador:",
            options=list(FRIENDLY_PITCHER_STYLES.keys()),
            key="home_pitcher_style",
            help="Escolha o perfil do jogador no montinho",
        )
        st.caption(f"💡 *{FRIENDLY_PITCHER_STYLES[chosen_style]['short_desc']}*")

    with col_moment:
        chosen_moment = st.selectbox(
            "⏱️ Situação do Jogo:",
            options=list(FRIENDLY_GAME_MOMENTS.keys()),
            key="home_game_moment",
            help="Escolha o momento de pressão da partida",
        )
        st.caption(f"💡 *{FRIENDLY_GAME_MOMENTS[chosen_moment]['short_situation']}*")

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
        "🔮 Consultar Inteligência Artificial em Tempo Real",
        key="btn_home_quick_predict",
        type="primary",
        use_container_width=True,
    )

    if quick_predict_clicked:
        with st.spinner("⏳ Consultando modelo no Databricks Serverless..."):
            try:
                pred_res = client.predict(demo_raw_inputs)
            except Exception:  # noqa: BLE001
                # Unified baseball fallback engine
                pred_res = predict_tactical_fallback(demo_raw_inputs)

            concise_bullet = build_concise_tactical_bullet(
                prediction=pred_res["prediction"],
                features=pred_res["features"],
            )
            context_badge = moment_cfg["context_badge"]
            if pred_res.get("is_fallback"):
                context_badge += " • MODO LOCAL"

            st.session_state.home_demo_result = {
                "prediction": pred_res["prediction"],
                "latency_ms": pred_res["latency_ms"],
                "bullet": concise_bullet,
                "context_badge": context_badge,
                "features": pred_res["features"],
            }

    if st.session_state.home_demo_result:
        res = st.session_state.home_demo_result
        st.markdown(
            render_intuitive_result_card_html(
                prediction=res["prediction"],
                latency_ms=res["latency_ms"],
                context_badge=res["context_badge"],
                concise_bullet=res["bullet"],
            ),
            unsafe_allow_html=True,
        )

# ===========================================================================
# 4. ARQUITETURA EM 4 PASSOS VISUAIS (SEM TEXTO LONGO)
# ===========================================================================
arch_summary_html = _compact_html(
    """
    <div style="background:#090D15;border:1px solid #1E293B;border-radius:10px;padding:12px 16px;margin:16px 0;">
      <div style="font-size:11px;font-weight:800;color:#94A3B8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;">ARQUITETURA DA SOLUÇÃO NO DATABRICKS</div>
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;font-size:12px;color:#CBD5E1;">
        <span>📡 <strong>1. Sensores Statcast</strong></span>
        <span style="color:#38BDF8;">➔</span>
        <span>⚙️ <strong>2. Lakeflow Spark CDC</strong></span>
        <span style="color:#38BDF8;">➔</span>
        <span>🧠 <strong>3. Modelo MLflow (24 Vars)</strong></span>
        <span style="color:#38BDF8;">➔</span>
        <span>⚡ <strong>4. Serverless REST (&lt;50ms)</strong></span>
      </div>
    </div>
    """
)
st.markdown(arch_summary_html, unsafe_allow_html=True)

# ===========================================================================
# 5. ATALHOS DIRETOS PARA AS OUTRAS PÁGINAS
# ===========================================================================
nav_c1, nav_c2, nav_c3 = st.columns(3)
with nav_c1:
    st.page_link(
        "pages/03_minigame.py",
        label="🎮 Jogar Minigame: Você vs. IA (Fliperama) →",
        icon="🎮",
    )
with nav_c2:
    st.page_link(
        "pages/02_scout_report.py",
        label="📋 Abrir MLB Scouting Report & Matriz →",
        icon="📋",
    )
with nav_c3:
    st.page_link(
        "pages/04_pitch_vision_lab.py",
        label="🔬 Abrir Laboratório de Visão & Tipping →",
        icon="🔬",
    )
