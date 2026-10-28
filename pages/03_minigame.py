"""
Página de Minigame Retro Arcade & Arena Broadcast 3D:
Modo Híbrido Visual (MLB Pitch Predictor).

Oferece:
- Passo 1: Seleção de Arremessador Estrela da MLB em 5 cards verticais com fotos oficiais e Tiers (S, A, B).
- Passo 2 (Modo Híbrido Visual):
  1. 🏟️ Modo Arena Broadcast 3D (Você no Montinho vs. IA): Simulador com animação vetorial 3D da
     trajetória/curva da bola (Fastball, Breaking Ball, Offspeed), Radar Statcast (MPH/RPM),
     comparação de túnel contra a previsão da IA e snapshot real da pose 3D no ponto de soltura.
  2. 👁️ Modo Desafio Olho Biônico (Leitura de Pose 3D vs. IA): Veja o Print Real no Montinho,
     o Esqueleto 3D MediaPipe e os ângulos biomecânicos antes de a bola chegar na placa e tente
     adivinhar qual arremesso está vindo, competindo contra o modelo de Visão Computacional!
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from pitch_predictor.fallback_engine import predict_tactical_fallback
from pitch_predictor.minigame_engine import (
    MLB_BATTERS,
    QUICK_AIM_MAP,
    ArcadeGameState,
    get_pitcher_tier_info,
    register_high_score,
    resolve_arcade_pitch,
)
from pitch_predictor.scenarios import get_pitcher_profile
from pitch_predictor.serving_client import PitchServingClient
from pitch_predictor.ui_components import (
    _compact_html,
    render_baseball_diamond_svg,
    render_pitch_flight_arena_svg,
    render_pitcher_photo_html,
    render_strike_zone_svg,
)
from pitch_predictor.vision.overlay import (
    extract_and_annotate_release_frame,
    extract_base_release_frame,
)

# ---------------------------------------------------------------------------
# 1. Custom CSS Retrô Arcade & Broadcast Arena 3D
# ---------------------------------------------------------------------------


def _get_retro_arcade_css() -> str:
    raw_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Rajdhani:wght@600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    .arcade-font {
        font-family: 'Press Start 2P', monospace, monospace !important;
    }

    .retro-cabinet-card {
        background: radial-gradient(circle at 50% 20%, rgba(13, 27, 42, 0.96), rgba(7, 14, 24, 0.99));
        border: 2px solid #00FFCC;
        box-shadow: 0 0 15px rgba(0, 255, 204, 0.28), inset 0 0 15px rgba(0, 255, 204, 0.08);
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
    }

    .retro-hud-header {
        background: linear-gradient(180deg, #16102C 0%, #090614 100%);
        border: 2px solid #FF007F;
        box-shadow: 0 0 22px rgba(255, 0, 127, 0.38);
        border-radius: 10px;
        padding: 10px 16px;
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }

    .retro-hud-item {
        font-family: 'Press Start 2P', monospace;
        font-size: 10.5px;
        letter-spacing: 0.8px;
        margin: 4px 6px;
    }

    .neon-cyan {
        color: #00FFCC;
        text-shadow: 0 0 8px #00FFCC;
    }

    .neon-pink {
        color: #FF007F;
        text-shadow: 0 0 8px #FF007F;
    }

    .neon-gold {
        color: #FFCC00;
        text-shadow: 0 0 8px #FFCC00;
    }

    .neon-green {
        color: #10B981;
        text-shadow: 0 0 8px #10B981;
    }

    .neon-purple {
        color: #A855F7;
        text-shadow: 0 0 8px #A855F7;
    }

    .quality-tag {
        display: inline-block;
        background: rgba(0, 255, 204, 0.12);
        color: #00FFCC;
        border: 1px solid rgba(0, 255, 204, 0.4);
        border-radius: 10px;
        padding: 2px 6px;
        font-size: 9.5px;
        font-weight: 700;
        margin: 2px 2px;
        line-height: 1.2;
    }

    .retro-pitcher-card {
        background: radial-gradient(circle at 50% 15%, rgba(17, 24, 39, 0.95), rgba(7, 14, 24, 0.98));
        border: 2px solid #00FFCC;
        box-shadow: 0 0 14px rgba(0, 255, 204, 0.25);
        border-radius: 8px;
        padding: 10px 8px;
        margin-bottom: 8px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 490px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .retro-pitcher-card:hover {
        box-shadow: 0 0 22px rgba(0, 255, 204, 0.5);
        transform: translateY(-2px);
    }

    .tier-badge-s {
        display: inline-block;
        background: rgba(255, 204, 0, 0.2);
        color: #FFCC00;
        border: 1px solid #FFCC00;
        border-radius: 4px;
        padding: 2px 6px;
        font-family: 'Press Start 2P', monospace;
        font-size: 9px;
        font-weight: 700;
    }

    .tier-badge-a {
        display: inline-block;
        background: rgba(0, 255, 204, 0.2);
        color: #00FFCC;
        border: 1px solid #00FFCC;
        border-radius: 4px;
        padding: 2px 6px;
        font-family: 'Press Start 2P', monospace;
        font-size: 9px;
        font-weight: 700;
    }

    .tier-badge-b {
        display: inline-block;
        background: rgba(168, 85, 247, 0.2);
        color: #C084FC;
        border: 1px solid #A855F7;
        border-radius: 4px;
        padding: 2px 6px;
        font-family: 'Press Start 2P', monospace;
        font-size: 9px;
        font-weight: 700;
    }

    .timing-meter-bar {
        border-radius: 6px;
        padding: 6px 12px;
        font-family: 'Press Start 2P', monospace;
        font-size: 10px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .timing-cold {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10B981;
        color: #10B981;
    }

    .timing-alert {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid #F59E0B;
        color: #F59E0B;
    }

    .timing-hot {
        background: rgba(239, 68, 68, 0.2);
        border: 1px solid #EF4444;
        color: #EF4444;
    }

    .retro-gameover-banner {
        background: linear-gradient(180deg, #3B0712 0%, #1A0408 100%);
        border: 3px solid #EF4444;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.6);
        border-radius: 8px;
        padding: 24px;
        text-align: center;
        margin-bottom: 24px;
    }

    .bio-gauge-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 11.5px;
        color: #E2E8F0;
        padding: 4px 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
    }
    </style>
    """
    return _compact_html(raw_css)


# ---------------------------------------------------------------------------
# 2. Sintetizador de Áudio 8-bit Chiptune (Web Audio API)
# ---------------------------------------------------------------------------


def _render_chiptune_audio(sound_type: str | None) -> str:
    """Gera script JS embutido que sintetiza efeitos sonoros 8-bit via Web Audio API."""
    if not sound_type:
        return ""

    raw_js = f"""
    <script>
    (function() {{
        try {{
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            if (!AudioContext) return;
            const ctx = new AudioContext();
            const now = ctx.currentTime;

            function playTone(freq, type, duration, startTime, gainVal) {{
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.type = type;
                osc.frequency.setValueAtTime(freq, startTime);
                gain.gain.setValueAtTime(gainVal || 0.1, startTime);
                gain.gain.exponentialRampToValueAtTime(0.0001, startTime + duration);
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.start(startTime);
                osc.stop(startTime + duration);
            }}

            const snd = "{sound_type}";
            if (snd === "strike" || snd === "whiff") {{
                playTone(587.33, "square", 0.12, now, 0.12);
                playTone(880.00, "square", 0.15, now + 0.06, 0.15);
            }} else if (snd === "hit" || snd === "home_run") {{
                playTone(261.63, "sawtooth", 0.25, now, 0.2);
                playTone(174.61, "sawtooth", 0.35, now + 0.08, 0.25);
            }} else if (snd === "strikeout" || snd === "inning_cleared") {{
                playTone(523.25, "square", 0.10, now, 0.12);
                playTone(659.25, "square", 0.10, now + 0.08, 0.12);
                playTone(783.99, "square", 0.12, now + 0.16, 0.15);
                playTone(1046.50, "square", 0.25, now + 0.24, 0.18);
            }} else if (snd === "game_over") {{
                playTone(349.23, "sawtooth", 0.25, now, 0.2);
                playTone(293.66, "sawtooth", 0.25, now + 0.22, 0.2);
                playTone(246.94, "sawtooth", 0.30, now + 0.44, 0.2);
                playTone(196.00, "sawtooth", 0.60, now + 0.70, 0.25);
            }} else if (snd === "ball" || snd === "walk") {{
                playTone(220.00, "sine", 0.15, now, 0.1);
            }}
        }} catch (e) {{
            // Audio autoplay policy fallback
        }}
    }})();
    </script>
    """
    return _compact_html(raw_js)


@st.cache_data(ttl=300)
def _load_all_vision_samples() -> pd.DataFrame:
    """
    Carrega os 60 lances reais calibrados dos 5 arremessadores da MLB de forma intercalada,
    para que cada rodada do Desafio Olho Biônico alterne automaticamente o arremessador e o lance!
    """
    bio_path = Path("src/pitch_predictor/vision/sample_data/biomechanical_features.csv")
    man_path = Path("src/pitch_predictor/vision/sample_data/manifest.csv")
    if bio_path.exists() and man_path.exists():
        bio_df = pd.read_csv(bio_path)
        man_df = pd.read_csv(man_path)
        m_sub = man_df[
            [
                "pitcher_id",
                "pitcher_name",
                "game_pk",
                "at_bat_number",
                "pitch_number",
                "pitch_type",
                "video_path",
            ]
        ].copy()
        merged = pd.merge(
            bio_df,
            m_sub,
            on=["pitcher_id", "game_pk", "at_bat_number", "pitch_number"],
            how="left",
        )
        if not merged.empty:
            # Intercala os 5 arremessadores (Skenes -> Burnes -> Webb -> Sale -> Yamamoto)
            merged["_within_idx"] = merged.groupby("pitcher_id").cumcount()
            interleaved = merged.sort_values(["_within_idx", "pitcher_id"]).drop(columns=["_within_idx"])
            return interleaved.reset_index(drop=True)
    return pd.DataFrame()


def _predict_cv_from_angles(
    arm_slot: float,
    elbow_flex: float,
    rel_ht: float,
    trunk_tilt: float,
    base_profile: dict[str, float],
) -> str:
    """Classifica o arremesso pela cinemática corporal em relação à linha de base do arremessador."""
    slot_diff = arm_slot - float(base_profile.get("arm_slot_angle", 50.0))
    elbow_diff = elbow_flex - float(base_profile.get("elbow_flexion_angle", 140.0))
    if slot_diff < -3.0 or elbow_diff < -4.0:
        return "Breaking Ball"
    if slot_diff > 2.2:
        return "Offspeed"
    return "Fastball"


def _render_image_compat(image_data, caption: str) -> None:
    try:
        st.image(image_data, caption=caption, use_container_width=True)
    except TypeError:
        st.image(image_data, caption=caption)


# ---------------------------------------------------------------------------
# 3. Inicialização de Session State
# ---------------------------------------------------------------------------

if "minigame_step" not in st.session_state:
    st.session_state.minigame_step = "selection"

if "minigame_pitcher" not in st.session_state:
    st.session_state.minigame_pitcher = "Paul Skenes"

if "arcade_state" not in st.session_state:
    st.session_state.arcade_state = ArcadeGameState(pitcher_name=st.session_state.minigame_pitcher)

if "arcade_leaderboard" not in st.session_state:
    st.session_state.arcade_leaderboard = [
        {"initials": "ACE", "score": 3500},
        {"initials": "KNG", "score": 2500},
        {"initials": "MLB", "score": 1500},
        {"initials": "DEV", "score": 1000},
        {"initials": "BOT", "score": 500},
    ]

if "minigame_audio" not in st.session_state:
    st.session_state.minigame_audio = True

if "last_outcome" not in st.session_state:
    st.session_state.last_outcome = None

if "last_sound_trigger" not in st.session_state:
    st.session_state.last_sound_trigger = None

if "minigame_mode" not in st.session_state:
    st.session_state.minigame_mode = "🏟️ Modo 1: Arena Broadcast 3D (Você no Montinho)"

if "bionic_clip_idx" not in st.session_state:
    st.session_state.bionic_clip_idx = 0

if "bionic_last_result" not in st.session_state:
    st.session_state.bionic_last_result = None

if "bionic_human_hits" not in st.session_state:
    st.session_state.bionic_human_hits = 0

if "bionic_cv_hits" not in st.session_state:
    st.session_state.bionic_cv_hits = 0

if "bionic_rounds" not in st.session_state:
    st.session_state.bionic_rounds = 0

# Renderiza CSS
st.markdown(_get_retro_arcade_css(), unsafe_allow_html=True)

# Dispara áudio se houver evento pendente
if st.session_state.minigame_audio and st.session_state.last_sound_trigger:
    st.markdown(
        _render_chiptune_audio(st.session_state.last_sound_trigger),
        unsafe_allow_html=True,
    )
    st.session_state.last_sound_trigger = None

game: ArcadeGameState = st.session_state.arcade_state

# ---------------------------------------------------------------------------
# 4. Painel Superior: Arcade HUD (Sempre Visível)
# ---------------------------------------------------------------------------

hearts_display = " ".join(["❤️" for _ in range(game.lives)])
if game.lives < 3:
    hearts_display += " " + " ".join(["🖤" for _ in range(3 - game.lives)])
if game.lives == 0:
    hearts_display = "💀 GAME OVER"

outs_display = ""
for i in range(2):
    outs_display += "● " if game.outs > i else "○ "

combo_badge = f"x{game.combo_multiplier}"
if game.combo_multiplier >= 3:
    combo_badge += " 🔥 COMBO!"
elif game.combo_multiplier == 2:
    combo_badge += " ⚡"

# Visual do Medidor de Timing
if game.timing_meter == 0:
    timing_html = """
    <div class="timing-meter-bar timing-cold">
        <span>TIMING DO REBATEDOR:</span>
        <span>🟩 FRIO (0/3) — Desprevenido</span>
    </div>
    """
elif game.timing_meter == 1:
    timing_html = """
    <div class="timing-meter-bar timing-alert">
        <span>TIMING DO REBATEDOR:</span>
        <span>🟨 ALERTA (1/3) — Antecipando repetição</span>
    </div>
    """
else:
    timing_html = f"""
    <div class="timing-meter-bar timing-hot">
        <span>TIMING DO REBATEDOR:</span>
        <span>🟥 NA ESPERA! ({game.timing_meter}/3) — Perigo de Rebatida!</span>
    </div>
    """

hud_html = f"""
<div class="retro-hud-header">
    <div class="retro-hud-item neon-gold">
        PONTUAÇÃO: <span style="font-size: 13px;">{game.score:06d}</span>
    </div>
    <div class="retro-hud-item neon-cyan">
        RECORDE: <span style="font-size: 13px;">{game.high_score:06d}</span>
    </div>
    <div class="retro-hud-item">
        VIDAS: <span>{hearts_display}</span>
    </div>
    <div class="retro-hud-item" style="color: #60A5FA;">
        CONTAGEM: <span style="font-weight: 700; color: #FFFFFF;">{game.balls}B - {game.strikes}S</span>
    </div>
    <div class="retro-hud-item neon-pink">
        OUTS: <span>{outs_display}</span>
    </div>
    <div class="retro-hud-item neon-green">
        COMBO: <span>{combo_badge}</span>
    </div>
</div>
{timing_html}
"""
st.markdown(_compact_html(hud_html), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 5. Tela de GAME OVER (Caso Vidas == 0)
# ---------------------------------------------------------------------------

if game.game_over:
    game_over_html = f"""
    <div class="retro-gameover-banner">
        <h1 class="arcade-font neon-pink" style="font-size: 24px; margin-bottom: 12px;">💀 FIM DE JOGO 💀</h1>
        <p class="arcade-font neon-gold" style="font-size: 13px; margin-bottom: 8px;">
            SUA PONTUAÇÃO FINAL: {game.score:06d} PONTOS
        </p>
        <p style="color: #94A3B8; font-size: 14px; margin-bottom: 16px;">
            A inteligência artificial encontrou brechas no seu repertório! Registre suas iniciais no Hall da Fama.
        </p>
    </div>
    """
    st.markdown(_compact_html(game_over_html), unsafe_allow_html=True)

    col_in1, col_in2, col_in3 = st.columns([1, 1, 1])
    with col_in2:
        initials_input = st.text_input(
            "Digite 3 Iniciais (ex: AAA, RHI):",
            value="AAA",
            max_chars=3,
            key="retro_initials",
        )
        if st.button("💾 GRAVAR NO HALL DA FAMA", use_container_width=True):
            st.session_state.arcade_leaderboard = register_high_score(
                initials=initials_input,
                score=game.score,
                leaderboard=st.session_state.arcade_leaderboard,
            )
            st.success("✅ Iniciais registradas com sucesso!")

        if st.button("🔄 JOGAR NOVAMENTE", use_container_width=True):
            old_high = max(game.high_score, game.score)
            st.session_state.arcade_state = ArcadeGameState(
                high_score=old_high,
                pitcher_name=st.session_state.minigame_pitcher,
            )
            st.session_state.last_outcome = None
            st.session_state.bionic_last_result = None
            st.session_state.last_sound_trigger = None
            st.session_state.minigame_step = "selection"
            st.rerun()

    st.markdown("---")
    st.markdown("### 🏆 Hall da Fama Retrô (Top 10)")
    for rank, entry in enumerate(st.session_state.arcade_leaderboard, 1):
        st.markdown(f"**#{rank:02d}** `{entry['initials']}` — **{entry['score']:06d} pts**")
    st.stop()


# ---------------------------------------------------------------------------
# 6. PASSO 1: Seleção de Arremessador Estrela da MLB (Cards Verticais)
# ---------------------------------------------------------------------------

if st.session_state.minigame_step == "selection":
    st.markdown("## ⚾ Passo 1: Escolha seu Arremessador Estrela")
    st.caption(
        "Cada astro da MLB possui arsenal calibrado em Tiers (S, A, B), animação 3D de trajetória e poses reais capturadas no montinho:"
    )

    pitcher_keys = [
        "Paul Skenes",
        "Corbin Burnes",
        "Logan Webb",
        "Chris Sale",
        "Yoshinobu Yamamoto",
    ]

    cols = st.columns(5, gap="small")

    for idx, p_name in enumerate(pitcher_keys):
        p_info = get_pitcher_tier_info(p_name)
        tags_html = " ".join([f'<span class="quality-tag">{tag}</span>' for tag in p_info["tags"]])

        p_pitches = p_info["pitches"]
        pitch_s = p_pitches[0]
        pitch_a = p_pitches[1]
        pitch_b = p_pitches[2]

        photo_html = render_pitcher_photo_html(p_name)

        with cols[idx]:
            card_html = f"""
            <div class="retro-pitcher-card">
                <div style="margin-bottom: 4px; min-height: 44px;">
                    {tags_html}
                </div>
                <div class="arcade-font neon-gold" style="font-size: 11px; margin-bottom: 2px; line-height: 1.3;">
                    {p_info['name']}
                </div>
                <div style="font-size: 10px; color: #94A3B8; margin-bottom: 4px;">
                    {p_info['team']} • <strong style="color: #60A5FA;">{p_info['p_throws']}HP</strong>
                </div>

                {photo_html}

                <div style="background: rgba(0,0,0,0.4); border-radius: 6px; padding: 6px; font-size: 10px; text-align: left; margin-bottom: 6px;">
                    <div style="margin-bottom: 4px;">
                        <span class="tier-badge-s">TIER S</span>
                        <strong style="color: #FFFFFF; font-size: 10px; display: block; margin-top: 1px;">{pitch_s['name']} ({pitch_s['speed']})</strong>
                        <span style="color: #94A3B8; font-size: 8.5px;">{pitch_s['desc']}</span>
                    </div>
                    <div style="margin-bottom: 4px;">
                        <span class="tier-badge-a">TIER A</span>
                        <strong style="color: #FFFFFF; font-size: 10px; display: block; margin-top: 1px;">{pitch_a['name']} ({pitch_a['speed']})</strong>
                        <span style="color: #94A3B8; font-size: 8.5px;">{pitch_a['desc']}</span>
                    </div>
                    <div>
                        <span class="tier-badge-b">TIER B</span>
                        <strong style="color: #FFFFFF; font-size: 10px; display: block; margin-top: 1px;">{pitch_b['name']} ({pitch_b['speed']})</strong>
                        <span style="color: #94A3B8; font-size: 8.5px;">{pitch_b['desc']}</span>
                    </div>
                </div>
                <div style="font-size: 9px; color: #CBD5E1; margin-bottom: 4px; min-height: 28px; line-height: 1.2;">
                    {p_info['description']}
                </div>
            </div>
            """
            st.markdown(_compact_html(card_html), unsafe_allow_html=True)

            if st.button(
                f"🎯 Escolher {p_name}",
                key=f"btn_pick_{p_name.replace(' ', '_')}",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.minigame_pitcher = p_name
                st.session_state.arcade_state.pitcher_name = p_name
                st.session_state.bionic_clip_idx = 0
                st.session_state.bionic_last_result = None
                st.session_state.minigame_step = "duel"
                st.rerun()

    st.stop()


# ---------------------------------------------------------------------------
# 7. PASSO 2: Arena Híbrida Visual (Duelo Broadcast 3D + Desafio Olho Biônico)
# ---------------------------------------------------------------------------

pitcher_name = st.session_state.minigame_pitcher
p_info = get_pitcher_tier_info(pitcher_name)
pitches = p_info["pitches"]
full_profile = get_pitcher_profile(pitcher_name)
pitcher_id = int(full_profile.get("pitcher_id", 694973))
p_throws = str(p_info.get("p_throws", "R"))

# Barra Superior do Duelo: Jogador Ativo + Seletor de Modo Híbrido + Botão Bullpen
MODE_OPTIONS = [
    "🏟️ Modo 1: Arena Broadcast 3D (Você no Montinho)",
    "👁️ Modo 2: Desafio Olho Biônico (Adivinhe pela Pose 3D)",
]
if st.session_state.minigame_mode not in MODE_OPTIONS:
    st.session_state.minigame_mode = MODE_OPTIONS[0]

col_top_l, col_top_m, col_top_r = st.columns([1.3, 2.2, 0.8])
with col_top_l:
    st.markdown(
        f"**No Montinho:** <span class='neon-cyan'>{pitcher_name}</span> ({p_throws}HP) | "
        f"*{p_info['team']}*",
        unsafe_allow_html=True,
    )
with col_top_m:
    active_mode = st.radio(
        "Modo de Jogo:",
        MODE_OPTIONS,
        index=MODE_OPTIONS.index(st.session_state.minigame_mode),
        horizontal=True,
        label_visibility="collapsed",
        key="minigame_mode_radio",
    )
    st.session_state.minigame_mode = active_mode
with col_top_r:
    if st.button("🔄 Trocar (Bullpen)", use_container_width=True):
        st.session_state.minigame_step = "selection"
        st.rerun()


# ===========================================================================
# MODO 2: DESAFIO OLHO BIÔNICO (ADIVINHE O ARREMESSO PELA POSE 3D VS. IA)
# ===========================================================================
if "Modo 2" in st.session_state.minigame_mode:
    samples_df = _load_all_vision_samples()
    total_samples = len(samples_df)
    if total_samples == 0:
        st.warning("Dataset biomecânico não encontrado.")
        st.stop()

    clip_idx = int(st.session_state.bionic_clip_idx) % total_samples
    sample_row = samples_df.iloc[clip_idx]

    cur_pid = int(sample_row["pitcher_id"])
    cur_pname = str(sample_row.get("pitcher_name", pitcher_name))
    cur_p_info = get_pitcher_tier_info(cur_pname)
    cur_pitches = cur_p_info["pitches"]
    cur_full_profile = get_pitcher_profile(cur_pname)
    cur_p_throws = str(cur_p_info.get("p_throws", "R"))

    s_gpk = int(sample_row["game_pk"])
    s_ab = int(sample_row["at_bat_number"])
    s_pn = int(sample_row["pitch_number"])
    real_cat = str(sample_row["pitch_category"])
    real_type = str(sample_row.get("pitch_type", "FF"))
    r_frame = int(sample_row.get("release_frame", 180))
    v_path = str(
        sample_row.get(
            "video_path",
            f"data/videos/{cur_pid}/{s_gpk}_{s_ab}_{s_pn}_{real_cat}.mp4",
        )
    )

    arm_slot = float(sample_row.get("arm_slot_angle", 50.0))
    elbow_flex = float(sample_row.get("elbow_flexion_angle", 140.0))
    rel_ht = float(sample_row.get("release_height", 0.70))
    trunk_tilt = float(sample_row.get("trunk_tilt", 12.0))

    base_bio = cur_full_profile.get("bio_profile", {})
    slot_diff = arm_slot - float(base_bio.get("arm_slot_angle", 50.0))
    elbow_diff = elbow_flex - float(base_bio.get("elbow_flexion_angle", 140.0))
    ht_diff = rel_ht - float(base_bio.get("release_height", 0.70))
    tilt_diff = trunk_tilt - float(base_bio.get("trunk_tilt", 12.0))

    cv_ai_guess = _predict_cv_from_angles(arm_slot, elbow_flex, rel_ht, trunk_tilt, base_bio)

    # Se houver resultado da rodada imediatamente anterior, exibe o veredito no topo
    b_res = st.session_state.bionic_last_result
    if b_res:
        h_ok = b_res["human_ok"]
        rounds = max(1, st.session_state.bionic_rounds)
        h_acc = (st.session_state.bionic_human_hits / rounds) * 100.0
        cv_acc = (st.session_state.bionic_cv_hits / rounds) * 100.0
        verdict_icon = "🎯 ACERTOU NA MOSCA!" if h_ok else "❌ ENGANADO PELO DISFARCE!"
        if h_ok:
            st.success(
                f"**Rodada Anterior ({b_res['pitcher_name']}) — {verdict_icon}** "
                f"Você apostou `{b_res['user_guess']}` e o arremesso real foi **{b_res['real_cat']} ({b_res['pitch_name']} • {b_res['speed']})**! "
                f"| Placar Acumulado: 👤 Você `{st.session_state.bionic_human_hits}/{rounds}` (`{h_acc:.0f}%`) vs. 🤖 IA `{st.session_state.bionic_cv_hits}/{rounds}` (`{cv_acc:.0f}%`)"
            )
        else:
            st.error(
                f"**Rodada Anterior ({b_res['pitcher_name']}) — {verdict_icon}** "
                f"Você apostou `{b_res['user_guess']}`, mas {b_res['pitcher_name']} lançou **{b_res['real_cat']} ({b_res['pitch_name']} • {b_res['speed']})** (IA previu `{b_res['cv_guess']}`)! "
                f"| Placar Acumulado: 👤 Você `{st.session_state.bionic_human_hits}/{rounds}` (`{h_acc:.0f}%`) vs. 🤖 IA `{st.session_state.bionic_cv_hits}/{rounds}` (`{cv_acc:.0f}%`)"
            )

    st.markdown(
        f"### 👁️ Desafio Olho Biônico — Rodada #{clip_idx + 1}/{total_samples} • Arremessador: **{cur_pname}** ({cur_p_throws}HP)"
    )
    st.caption(
        f"Analise o **Print Real no Montinho**, o **Esqueleto 3D MediaPipe** e os **Desvios Angulares** de **{cur_pname}** no Frame #{r_frame}. "
        "Ao chutar um arremesso, o jogo corrige sua resposta e já avança automaticamente para o próximo arremessador!"
    )

    bio_col1, bio_col2, bio_col3 = st.columns([1.15, 1.15, 1.1], gap="medium")

    with bio_col1:
        st.markdown(f"**📸 1. Print Base de {cur_pname} (Frame #{r_frame})**")
        base_img = extract_base_release_frame(
            video_path=v_path,
            release_frame_idx=r_frame,
            pitch_category=real_cat,
            game_pk=s_gpk,
            at_bat_number=s_ab,
            pitch_number=s_pn,
            pitcher_id=cur_pid,
        )
        _render_image_compat(
            base_img,
            caption=f"{cur_pname} no ponto de soltura (Frame #{r_frame} • Jogo #{s_gpk})",
        )

    with bio_col2:
        st.markdown("**🦴 2. Modelo da Pose 3D (Esqueleto Extraído)**")
        pose_img = extract_and_annotate_release_frame(
            video_path=v_path,
            release_frame_idx=r_frame,
            p_throws=cur_p_throws,
            metrics={
                "arm_slot_angle": arm_slot,
                "elbow_flexion_angle": elbow_flex,
                "release_height": rel_ht,
                "trunk_tilt": trunk_tilt,
            },
            pitch_category=real_cat,
            game_pk=s_gpk,
            at_bat_number=s_ab,
            pitch_number=s_pn,
            pitcher_id=cur_pid,
        )
        _render_image_compat(
            pose_img,
            caption=f"Esqueleto 3D MediaPipe de {cur_pname} ({cur_p_throws}HP) no Frame #{r_frame}",
        )

    with bio_col3:
        st.markdown("**📐 3. Leitura Biomecânica & Seu Chute**")
        gauges_html = f"""
        <div class="retro-cabinet-card" style="padding: 12px;">
            <div class="arcade-font neon-cyan" style="font-size: 9.5px; margin-bottom: 8px;">
                TELEMETRIA DE {cur_pname.upper()} (FRAME #{r_frame}):
            </div>
            <div class="bio-gauge-row">
                <span>📐 Arm Slot (Ângulo Braço):</span>
                <strong style="color: #00FFCC;">{arm_slot:.1f}° ({slot_diff:+.1f}° vs base)</strong>
            </div>
            <div class="bio-gauge-row">
                <span>💪 Flexão do Cotovelo:</span>
                <strong style="color: #60A5FA;">{elbow_flex:.1f}° ({elbow_diff:+.1f}° vs base)</strong>
            </div>
            <div class="bio-gauge-row">
                <span>📏 Altura de Soltura:</span>
                <strong style="color: #FACC15;">{rel_ht:.2f} ({ht_diff:+.2f} vs base)</strong>
            </div>
            <div class="bio-gauge-row" style="border-bottom: none;">
                <span>🧍 Inclinação do Tronco:</span>
                <strong style="color: #F472B6;">{trunk_tilt:.1f}° ({tilt_diff:+.1f}° vs base)</strong>
            </div>
            <div style="margin-top: 8px; padding: 6px; background: rgba(0, 255, 204, 0.08); border-radius: 6px; font-size: 10.5px; color: #94A3B8;">
                💡 <strong>Dica:</strong> Arm Slot negativo/cotovelo mais dobrado indica <em>Breaking Ball</em>; Arm Slot mais alto indica <em>Offspeed</em>; alinhado à base indica <em>Fastball</em>.
            </div>
        </div>
        """
        st.markdown(_compact_html(gauges_html), unsafe_allow_html=True)

        guess_fb = st.button("🔥 CHUTAR: FASTBALL (Bola Rápida)", use_container_width=True, type="primary")
        guess_bb = st.button("🌀 CHUTAR: BREAKING BALL (Curva/Slider)", use_container_width=True)
        guess_os = st.button("🛑 CHUTAR: OFFSPEED (Changeup/Splitter)", use_container_width=True)
        next_clip = st.button("⏭️ Pular para Próximo Arremessador", use_container_width=True)

        if next_clip:
            next_idx = (clip_idx + 1) % total_samples
            st.session_state.bionic_clip_idx = next_idx
            next_pname = str(samples_df.iloc[next_idx].get("pitcher_name", cur_pname))
            st.session_state.minigame_pitcher = next_pname
            st.session_state.bionic_last_result = None
            st.rerun()

        user_guess = None
        if guess_fb:
            user_guess = "Fastball"
        elif guess_bb:
            user_guess = "Breaking Ball"
        elif guess_os:
            user_guess = "Offspeed"

        if user_guess:
            human_ok = user_guess == real_cat
            cv_ok = cv_ai_guess == real_cat
            st.session_state.bionic_rounds += 1
            if human_ok:
                st.session_state.bionic_human_hits += 1
                pts = 250 * game.combo_multiplier
                game.score += pts
                game.high_score = max(game.high_score, game.score)
                game.combo_multiplier = min(5, game.combo_multiplier + 1)
                st.session_state.last_sound_trigger = "strikeout"
            else:
                game.combo_multiplier = 1
                st.session_state.last_sound_trigger = "hit"
            if cv_ok:
                st.session_state.bionic_cv_hits += 1

            matched_p = next((p for p in cur_pitches if p["category"] == real_cat), cur_pitches[0])
            st.session_state.bionic_last_result = {
                "pitcher_name": cur_pname,
                "p_throws": cur_p_throws,
                "user_guess": user_guess,
                "cv_guess": cv_ai_guess,
                "real_cat": real_cat,
                "real_type": real_type,
                "pitch_name": matched_p["name"],
                "speed": matched_p["speed"],
                "human_ok": human_ok,
                "cv_ok": cv_ok,
                "clip_idx": clip_idx,
            }
            # Avança IMEDIATAMENTE para o próximo arremessador e próximo lance!
            next_idx = (clip_idx + 1) % total_samples
            st.session_state.bionic_clip_idx = next_idx
            next_pname = str(samples_df.iloc[next_idx].get("pitcher_name", cur_pname))
            st.session_state.minigame_pitcher = next_pname
            st.rerun()


# ===========================================================================
# MODO 1 (PADRÃO): ARENA BROADCAST 3D — VOCÊ NO MONTINHO VS. IA
# ===========================================================================
else:
    col_batter, col_mound, col_feedback = st.columns([0.95, 1.55, 1.05], gap="medium")

    # --- COLUNA 1: REBATEDOR, BASES & SNAPSHOT 3D DO ARREMESSADOR ---
    with col_batter:
        st.markdown("### 👤 No Bastão")

        batter_profile = next(
            (b for b in MLB_BATTERS if b["name"] == game.current_batter_name),
            MLB_BATTERS[0],
        )

        batter_card_html = f"""
        <div class="retro-cabinet-card" style="padding: 10px;">
            <div class="arcade-font neon-pink" style="font-size: 10px; margin-bottom: 4px;">
                ASTRO DA MLB:
            </div>
            <div style="font-size: 16px; font-weight: 700; color: #FFFFFF;">
                {batter_profile['name']}
            </div>
            <div style="font-size: 11px; color: #60A5FA; margin-bottom: 6px;">
                {batter_profile['team']} • <strong>{batter_profile['stand']}HB</strong>
            </div>
            <div style="font-size: 11px; color: #CBD5E1; margin-bottom: 6px;">
                {batter_profile['description']}
            </div>
            <div style="background: rgba(255, 0, 127, 0.15); border: 1px solid #FF007F; border-radius: 4px; padding: 4px 6px; font-size: 10px; color: #FF99CC;">
                ⚡ <strong>Traço Tático:</strong> {batter_profile['trait']}
            </div>
        </div>
        """
        st.markdown(_compact_html(batter_card_html), unsafe_allow_html=True)

        st.markdown(
            render_baseball_diamond_svg(game.on_1b, game.on_2b, game.on_3b, game.outs),
            unsafe_allow_html=True,
        )

    # --- COLUNA 2: GABINETE DE ARREMESSO + ARENA BROADCAST 3D ANIMADA ---
    with col_mound:
        st.markdown("### 🎮 Gabinete de Arremesso")
        st.caption("Escolha o arremesso por Tier, aponte na zona e veja a trajetória 3D em tempo real:")

        tier_options = [
            f"⭐ Tier S: {pitches[0]['name']} ({pitches[0]['speed']})",
            f"⚡ Tier A: {pitches[1]['name']} ({pitches[1]['speed']})",
            f"🎯 Tier B: {pitches[2]['name']} ({pitches[2]['speed']})",
        ]

        selected_tier_str = st.radio(
            "1. Selecione o Arremesso:",
            tier_options,
            index=0,
            label_visibility="collapsed",
        )

        if "Tier S" in selected_tier_str:
            chosen_pitch = pitches[0]
        elif "Tier A" in selected_tier_str:
            chosen_pitch = pitches[1]
        else:
            chosen_pitch = pitches[2]

        aim_choice = st.radio(
            "2. Ponto de Destino:",
            list(QUICK_AIM_MAP.keys()),
            format_func=lambda k: QUICK_AIM_MAP[k]["label"],
            horizontal=True,
        )

        target_zone = QUICK_AIM_MAP[aim_choice]["zone"]
        target_category = chosen_pitch["category"]
        target_type = chosen_pitch["code"]
        target_tier = chosen_pitch["tier"]

        # Determina estado visual da Arena 3D (mostra o último lance resolvido ou o preview da seleção)
        last_out = st.session_state.last_outcome
        if last_out:
            res_code = last_out.get("result", "")
            arena_ok = res_code in ("STRIKE", "WHIFF", "STRIKEOUT", "INNING_CLEARED", "OUT", "FOUL")
            arena_badge = f"{res_code} • IA: {last_out.get('ai_prediction', '')}"
            arena_ai_cat = last_out.get("ai_prediction")
        else:
            arena_ok = True
            arena_badge = f"PREVIEW: TIER {target_tier} ({target_type})"
            arena_ai_cat = None

        # Arena Broadcast 3D Animada (Trajetória Reta vs Curva vs Queda Tardia)
        st.markdown(
            render_pitch_flight_arena_svg(
                pitch_category=target_category,
                pitch_name=chosen_pitch["name"],
                speed_mph=chosen_pitch["speed"],
                target_zone=target_zone,
                p_throws=p_throws,
                ai_category=arena_ai_cat,
                result_badge=arena_badge,
                is_success=arena_ok,
            ),
            unsafe_allow_html=True,
        )

        pitch_btn = st.button(
            "🕹️ ARREMESSAR CONTRA A IA!",
            type="primary",
            use_container_width=True,
        )

        if pitch_btn:
            with st.spinner("🧠 A IA está prevendo seu arremesso no Databricks..."):
                client = PitchServingClient(profile="dev")

                prev_cat = "First Pitch"
                prev_tp = "None"
                prev_zn = 0
                prev_res = "First Pitch"

                if game.history:
                    last_h = game.history[-1]
                    prev_cat = last_h.get("pitch_category", "First Pitch")
                    prev_tp = last_h.get("pitch_type", "None")
                    prev_zn = last_h.get("pitch_zone", 0)
                    res_raw = last_h.get("result", "First Pitch")
                    if "STRIKE" in res_raw or "WHIFF" in res_raw:
                        prev_res = "S"
                    elif "BALL" in res_raw:
                        prev_res = "B"
                    else:
                        prev_res = "X"

                fb_p, bb_p, os_p = p_info["priors"]

                raw_payload = {
                    "balls": game.balls,
                    "strikes": game.strikes,
                    "outs_when_up": game.outs,
                    "inning": 1,
                    "inning_topbot": "Top",
                    "score_diff": 0,
                    "on_1b_flag": game.on_1b,
                    "on_2b_flag": game.on_2b,
                    "on_3b_flag": game.on_3b,
                    "p_throws": p_throws,
                    "stand": batter_profile["stand"],
                    "n_thruorder_pitcher": 1,
                    "prev_pitch_category": prev_cat,
                    "prev_pitch_type": prev_tp,
                    "prev_pitch_zone": prev_zn,
                    "prev_pitch_result": prev_res,
                    "pitcher_fb_prior": fb_p,
                    "pitcher_bb_prior": bb_p,
                    "pitcher_os_prior": os_p,
                }

                try:
                    prediction_res = client.predict(raw_payload)
                    ai_pred = prediction_res.get("prediction", "Fastball")
                except (ConnectionError, TimeoutError, RuntimeError, ValueError):
                    fallback_res = predict_tactical_fallback(raw_payload)
                    ai_pred = fallback_res["prediction"]

                next_state, outcome = resolve_arcade_pitch(
                    state=game,
                    user_category=target_category,
                    user_type=target_type,
                    user_zone=target_zone,
                    ai_prediction=ai_pred,
                    pitch_tier=target_tier,
                    probabilistic=True,
                )

                res_type = outcome.get("result", "")
                if res_type == "HOME_RUN":
                    st.session_state.last_sound_trigger = "home_run"
                elif res_type == "HIT":
                    st.session_state.last_sound_trigger = "hit"
                elif res_type == "STRIKEOUT" or outcome.get("inning_cleared"):
                    st.session_state.last_sound_trigger = "strikeout"
                elif res_type == "WHIFF":
                    st.session_state.last_sound_trigger = "whiff"
                elif res_type in ("STRIKE", "FOUL"):
                    st.session_state.last_sound_trigger = "strike"
                elif res_type in ("WALK", "BALL"):
                    st.session_state.last_sound_trigger = "ball"

                if next_state.game_over:
                    st.session_state.last_sound_trigger = "game_over"

                st.session_state.arcade_state = next_state
                st.session_state.last_outcome = outcome
                st.rerun()

    # --- COLUNA 3: FEEDBACK DO DUELO + POSE 3D REAL NO PONTO DE SOLTURA ---
    with col_feedback:
        st.markdown("### 📡 Resolução do Lance")

        outcome = st.session_state.last_outcome
        if outcome:
            res = outcome.get("result", "")
            ai_guess = outcome.get("ai_prediction", "")
            user_p = outcome.get("user_category", "")
            tier = outcome.get("pitch_tier", "A")

            is_success = res in ("STRIKE", "WHIFF", "STRIKEOUT", "INNING_CLEARED", "OUT", "FOUL")
            box_border = "#00FFCC" if is_success else "#FF007F"
            badge_bg = (
                "rgba(0, 255, 204, 0.2)" if is_success else "rgba(255, 0, 127, 0.2)"
            )

            feedback_html = f"""
            <div class="retro-cabinet-card" style="border-color: {box_border}; margin-bottom: 10px;">
                <div class="arcade-font neon-gold" style="font-size: 10px; margin-bottom: 6px;">
                    RESULTADO DO DUELO:
                </div>
                <div style="font-size: 13px; color: #FFFFFF; margin-bottom: 4px;">
                    Seu Arremesso: <strong style="color: #00FFCC;">Tier {tier} • {user_p} ({outcome.get('user_type')})</strong>
                </div>
                <div style="font-size: 13px; color: #FFFFFF; margin-bottom: 6px;">
                    Previsão da IA: <strong style="color: #FF007F;">{ai_guess}</strong>
                </div>
                <div style="padding: 6px 10px; background: {badge_bg}; border-radius: 4px; font-weight: 700; font-size: 12px; color: #FFFFFF;">
                    {outcome.get('message', '')}
                </div>
            </div>
            """
            st.markdown(_compact_html(feedback_html), unsafe_allow_html=True)
        else:
            st.info(
                "👉 Escolha seu Tier de arremesso e clique em **ARREMESSAR CONTRA A IA**!",
                icon=":material/sports_baseball:",
            )

        st.markdown(
            render_strike_zone_svg(target_zone, target_category),
            unsafe_allow_html=True,
        )

        # Histórico dos últimos 3 lances
        st.markdown("#### 📜 Últimos Lances")
        if game.history:
            for idx, h in enumerate(reversed(game.history[-3:]), 1):
                st.markdown(
                    f"<div style='font-size: 12px; color: #CBD5E1; margin-bottom: 4px;'>"
                    f"<strong>#{len(game.history) - idx + 1}</strong> Tier {h.get('pitch_tier', 'A')} {h.get('pitch_type')} "
                    f"➔ IA: <code>{h.get('ai_prediction')}</code> ➔ <strong>{h.get('result')}</strong>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Nenhum arremesso realizado nesta partida.")

        st.markdown("---")
        st.session_state.minigame_audio = st.toggle(
            "🔊 Áudio 8-Bit",
            value=st.session_state.minigame_audio,
        )
        if st.button("🔄 Novo Jogo", use_container_width=True):
            st.session_state.arcade_state = ArcadeGameState(
                high_score=game.high_score,
                pitcher_name=pitcher_name,
            )
            st.session_state.last_outcome = None
            st.session_state.bionic_last_result = None
            st.session_state.last_sound_trigger = None
            st.session_state.minigame_step = "selection"
            st.rerun()
