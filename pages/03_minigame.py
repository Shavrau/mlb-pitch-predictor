"""
Página de Minigame Retro Arcade: Você vs. IA no Montinho (MLB Pitch Predictor).

Estética fliperama dos anos 90 (8-bit / 16-bit), sistema RPG de distribuição
de 100 pontos de atributos, modo sobrevivência infinita com High Score e efeitos
sonoros sintetizados chiptune via Web Audio API.
"""

from __future__ import annotations

import streamlit as st

from pitch_predictor.minigame_engine import (
    ArcadeGameState,
    calculate_priors_from_stats,
    register_high_score,
    resolve_arcade_pitch,
)
from pitch_predictor.scenarios import PITCH_TYPES_BY_CATEGORY
from pitch_predictor.serving_client import PitchServingClient
from pitch_predictor.ui_components import (
    _compact_html,
    render_baseball_diamond_svg,
    render_strike_zone_svg,
)

# ---------------------------------------------------------------------------
# 1. Custom CSS Retrô Arcade (Fliperama Anos 90)
# ---------------------------------------------------------------------------


def _get_retro_arcade_css() -> str:
    raw_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Rajdhani:wght@600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    .arcade-font {
        font-family: 'Press Start 2P', monospace, monospace !important;
    }

    .retro-cabinet-card {
        background: radial-gradient(circle at 50% 20%, rgba(13, 27, 42, 0.95), rgba(7, 14, 24, 0.98));
        border: 2px solid #00FFCC;
        box-shadow: 0 0 15px rgba(0, 255, 204, 0.3), inset 0 0 15px rgba(0, 255, 204, 0.1);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }

    .retro-hud-header {
        background: linear-gradient(180deg, #1A102F 0%, #0D081B 100%);
        border: 2px solid #FF007F;
        box-shadow: 0 0 20px rgba(255, 0, 127, 0.4);
        border-radius: 8px;
        padding: 12px 18px;
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }

    .retro-hud-item {
        font-family: 'Press Start 2P', monospace;
        font-size: 11px;
        letter-spacing: 1px;
        margin: 4px 8px;
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

    .retro-badge {
        display: inline-block;
        padding: 4px 8px;
        font-family: 'Press Start 2P', monospace;
        font-size: 9px;
        border-radius: 4px;
        margin-right: 6px;
    }

    .scanline-overlay {
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%),
                    linear-gradient(90deg, rgba(255, 0, 0, 0.03), rgba(0, 255, 0, 0.01), rgba(0, 0, 255, 0.03));
        background-size: 100% 4px, 6px 100%;
        border-radius: 8px;
        padding: 4px;
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
            // Ignore audio autoplay restrictions gracefully
        }}
    }})();
    </script>
    """
    return _compact_html(raw_js)


# ---------------------------------------------------------------------------
# 3. Inicialização de Session State
# ---------------------------------------------------------------------------

if "arcade_state" not in st.session_state:
    st.session_state.arcade_state = ArcadeGameState()

if "arcade_leaderboard" not in st.session_state:
    st.session_state.arcade_leaderboard = [
        {"initials": "ACE", "score": 3500},
        {"initials": "KNG", "score": 2500},
        {"initials": "MLB", "score": 1500},
        {"initials": "DEV", "score": 1000},
        {"initials": "BOT", "score": 500},
    ]

if "rpg_vel" not in st.session_state:
    st.session_state.rpg_vel = 50
if "rpg_efe" not in st.session_state:
    st.session_state.rpg_efe = 30
if "rpg_tim" not in st.session_state:
    st.session_state.rpg_tim = 20

if "minigame_audio" not in st.session_state:
    st.session_state.minigame_audio = True

if "last_outcome" not in st.session_state:
    st.session_state.last_outcome = None

if "last_sound_trigger" not in st.session_state:
    st.session_state.last_sound_trigger = None

# Renderiza CSS
st.markdown(_get_retro_arcade_css(), unsafe_allow_html=True)

# Dispara áudio sintetizado se houver som pendente
if st.session_state.minigame_audio and st.session_state.last_sound_trigger:
    st.markdown(
        _render_chiptune_audio(st.session_state.last_sound_trigger),
        unsafe_allow_html=True,
    )
    st.session_state.last_sound_trigger = None

game: ArcadeGameState = st.session_state.arcade_state

# ---------------------------------------------------------------------------
# 4. Painel Superior: Arcade HUD
# ---------------------------------------------------------------------------

# Vidas em corações
hearts_display = " ".join(["❤️" for _ in range(game.lives)])
if game.lives < 3:
    hearts_display += " " + " ".join(["🖤" for _ in range(3 - game.lives)])
if game.lives == 0:
    hearts_display = "💀 GAME OVER"

# Outs display (● ● ○)
outs_display = ""
for i in range(2):
    outs_display += "● " if game.outs > i else "○ "

# Combo display
combo_badge = f"x{game.combo_multiplier}"
if game.combo_multiplier >= 3:
    combo_badge += " 🔥 COMBO!"
elif game.combo_multiplier == 2:
    combo_badge += " ⚡"

hud_html = f"""
<div class="retro-hud-header">
    <div class="retro-hud-item neon-gold">
        PONTUAÇÃO: <span style="font-size: 14px;">{game.score:06d}</span>
    </div>
    <div class="retro-hud-item neon-cyan">
        RECORDE: <span style="font-size: 14px;">{game.high_score:06d}</span>
    </div>
    <div class="retro-hud-item">
        VIDAS: <span>{hearts_display}</span>
    </div>
    <div class="retro-hud-item neon-pink">
        OUTS: <span>{outs_display}</span>
    </div>
    <div class="retro-hud-item neon-green">
        COMBO: <span>{combo_badge}</span>
    </div>
</div>
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
            A IA do Databricks encontrou brechas no seu repertório! Registre suas iniciais no Hall da Fama.
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
        if st.button("💾 GRAVAR INICIAIS NO HALL DA FAMA", use_container_width=True):
            st.session_state.arcade_leaderboard = register_high_score(
                initials=initials_input,
                score=game.score,
                leaderboard=st.session_state.arcade_leaderboard,
            )
            st.success("✅ Iniciais registradas com sucesso!")

        if st.button("🔄 JOGAR NOVAMENTE (NOVO DESAFIO)", use_container_width=True):
            old_high = max(game.high_score, game.score)
            st.session_state.arcade_state = ArcadeGameState(high_score=old_high)
            st.session_state.last_outcome = None
            st.session_state.last_sound_trigger = None
            st.rerun()

    # Mostra ranking e encerra página
    st.markdown("---")
    st.markdown("### 🏆 Hall da Fama Retrô (Top 10 Recordes)")
    for rank, entry in enumerate(st.session_state.arcade_leaderboard, 1):
        st.markdown(
            f"**#{rank:02d}** `{entry['initials']}` — **{entry['score']:06d} pts**"
        )
    st.stop()


# ---------------------------------------------------------------------------
# 6. Layout Principal: 3 Colunas Arcade
# ---------------------------------------------------------------------------

col_rpg, col_console, col_feedback = st.columns([1.1, 1.3, 1.1])

# --- COLUNA 1: RPG CHARACTER BUILDER (100 PONTOS DE ATRIBUTOS) ---
with col_rpg:
    st.markdown("### 🧙‍♂️ Atributos do Arremessador")
    st.caption("Distribua 100 pontos entre os três pilares do seu repertório:")

    # Presets rápidos
    st.markdown("**Perfis Pré-Configurados:**")
    cp1, cp2 = st.columns(2)
    with cp1:
        if st.button("⚡ Power Pitcher", use_container_width=True):
            st.session_state.rpg_vel = 70
            st.session_state.rpg_efe = 20
            st.session_state.rpg_tim = 10
            st.rerun()
        if st.button("🛑 Changeup Ace", use_container_width=True):
            st.session_state.rpg_vel = 30
            st.session_state.rpg_efe = 20
            st.session_state.rpg_tim = 50
            st.rerun()
    with cp2:
        if st.button("🌀 Spin Specialist", use_container_width=True):
            st.session_state.rpg_vel = 20
            st.session_state.rpg_efe = 65
            st.session_state.rpg_tim = 15
            st.rerun()
        if st.button("⚖️ Equilibrado", use_container_width=True):
            st.session_state.rpg_vel = 34
            st.session_state.rpg_efe = 33
            st.session_state.rpg_tim = 33
            st.rerun()

    vel = st.slider(
        "⚡ Velocidade (Fastball):",
        0,
        100,
        int(st.session_state.rpg_vel),
        key="slider_vel",
    )
    efe = st.slider(
        "🌀 Efeito / Giro (Breaking):",
        0,
        100,
        int(st.session_state.rpg_efe),
        key="slider_efe",
    )
    tim = st.slider(
        "🛑 Variação / Timing (Offspeed):",
        0,
        100,
        int(st.session_state.rpg_tim),
        key="slider_tim",
    )

    st.session_state.rpg_vel = vel
    st.session_state.rpg_efe = efe
    st.session_state.rpg_tim = tim

    total_pts = vel + efe + tim
    fb_p, bb_p, os_p = calculate_priors_from_stats(vel, efe, tim)

    if total_pts == 100:
        st.success(f"✅ Total: {total_pts}/100 pontos distribuídos!")
    else:
        st.warning(
            f"⚖️ Total: {total_pts}/100 pts. Normalizado proporcionalmente: "
            f"FB: {fb_p*100:.0f}% | BB: {bb_p*100:.0f}% | OS: {os_p*100:.0f}%"
        )

    # Card do Rebatedor Atual
    batter_card_html = f"""
    <div class="retro-cabinet-card" style="margin-top: 14px;">
        <div class="arcade-font neon-pink" style="font-size: 10px; margin-bottom: 6px;">
            REBATEDOR NO DUELO:
        </div>
        <div style="font-size: 16px; font-weight: 700; color: #FFFFFF;">
            👤 {game.current_batter_name}
        </div>
        <div style="font-size: 12px; color: #94A3B8; margin-top: 4px;">
            Contagem: <strong style="color: #60A5FA;">{game.balls} Bolas</strong> — <strong style="color: #F87171;">{game.strikes} Strikes</strong>
        </div>
    </div>
    """
    st.markdown(_compact_html(batter_card_html), unsafe_allow_html=True)

    # Mini Diamante de Corredores
    st.markdown(
        render_baseball_diamond_svg(game.on_1b, game.on_2b, game.on_3b, game.outs),
        unsafe_allow_html=True,
    )


# --- COLUNA 2: CONSOLE DE ARREMESSO (ESCOLHA DO JOGADOR) ---
with col_console:
    st.markdown("### 🎮 Gabinete de Arremesso")
    st.caption("Escolha o arremesso para tentar enganar a inteligência artificial:")

    cat_col1, cat_col2, cat_col3 = st.columns(3)
    user_category = st.radio(
        "1. Categoria do Arremesso:",
        ["Fastball", "Breaking Ball", "Offspeed"],
        format_func=lambda c: (
            "⚡ Bola Rápida (Fastball)"
            if c == "Fastball"
            else (
                "🌀 Efeito (Breaking Ball)"
                if c == "Breaking Ball"
                else "🛑 Mudança (Offspeed)"
            )
        ),
        horizontal=True,
    )

    # Tipos compatíveis
    type_candidates = PITCH_TYPES_BY_CATEGORY.get(user_category, [])
    type_labels = [label for code, label in type_candidates if code != "None"]
    selected_type_label = st.selectbox("2. Tipo Específico:", type_labels)
    user_type = selected_type_label.split(" — ")[0]

    st.markdown("**3. Localização na Zona de Strike Statcast:**")
    st.caption(
        "Zonas 1-9: Na Zona (Risco de Rebatida se a IA adivinhar!) • Zonas 11-14: Fora (Induz Whiff se a IA errar!)"
    )

    # Seletor interativo de zona Statcast
    cz1, cz2, cz3 = st.columns(3)
    with cz1:
        z11 = st.button("Z11 ↗️ Chase Alto-Esq", use_container_width=True)
    with cz2:
        st.markdown(
            "<div style='text-align: center; color: #64748B; font-size: 11px;'>CHASE SUPERIOR</div>",
            unsafe_allow_html=True,
        )
    with cz3:
        z12 = st.button("Z12 ↖️ Chase Alto-Dir", use_container_width=True)

    # Grid 3x3
    g1, g2, g3 = st.columns(3)
    with g1:
        z1 = st.button("Z1 (Alto-Esq)", use_container_width=True)
        z4 = st.button("Z4 (Meio-Esq)", use_container_width=True)
        z7 = st.button("Z7 (Baixo-Esq)", use_container_width=True)
    with g2:
        z2 = st.button("Z2 (Alto-Centro)", use_container_width=True)
        z5 = st.button("🔥 Z5 (MEATBALL)", use_container_width=True)
        z8 = st.button("Z8 (Baixo-Centro)", use_container_width=True)
    with g3:
        z3 = st.button("Z3 (Alto-Dir)", use_container_width=True)
        z6 = st.button("Z6 (Meio-Dir)", use_container_width=True)
        z9 = st.button("Z9 (Baixo-Dir)", use_container_width=True)

    cz4, cz5, cz6 = st.columns(3)
    with cz4:
        z13 = st.button("Z13 ↘️ Chase Baixo-Esq", use_container_width=True)
    with cz5:
        st.markdown(
            "<div style='text-align: center; color: #64748B; font-size: 11px;'>CHASE INFERIOR</div>",
            unsafe_allow_html=True,
        )
    with cz6:
        z14 = st.button("Z14 ↙️ Chase Baixo-Dir", use_container_width=True)

    # Determina zona selecionada
    if "selected_minigame_zone" not in st.session_state:
        st.session_state.selected_minigame_zone = 5

    if z1:
        st.session_state.selected_minigame_zone = 1
    elif z2:
        st.session_state.selected_minigame_zone = 2
    elif z3:
        st.session_state.selected_minigame_zone = 3
    elif z4:
        st.session_state.selected_minigame_zone = 4
    elif z5:
        st.session_state.selected_minigame_zone = 5
    elif z6:
        st.session_state.selected_minigame_zone = 6
    elif z7:
        st.session_state.selected_minigame_zone = 7
    elif z8:
        st.session_state.selected_minigame_zone = 8
    elif z9:
        st.session_state.selected_minigame_zone = 9
    elif z11:
        st.session_state.selected_minigame_zone = 11
    elif z12:
        st.session_state.selected_minigame_zone = 12
    elif z13:
        st.session_state.selected_minigame_zone = 13
    elif z14:
        st.session_state.selected_minigame_zone = 14

    user_zone = st.session_state.selected_minigame_zone
    st.info(
        f"🎯 Zona Selecionada: **Zona {user_zone}** "
        + ("(Na Zona de Strike)" if 1 <= user_zone <= 9 else "(Bola Fora / Chase)")
    )

    # SVG da zona selecionada
    st.markdown(
        render_strike_zone_svg(user_zone, user_category), unsafe_allow_html=True
    )

    # Botão de Execução Arcade
    st.markdown("<br>", unsafe_allow_html=True)
    pitch_btn = st.button(
        "🕹️ ARREMESSAR CONTRA A IA!",
        type="primary",
        use_container_width=True,
    )

    if pitch_btn:
        with st.spinner("🧠 A IA está lendo a situação tática no Databricks..."):
            client = PitchServingClient(profile="dev")

            # Determina histórico anterior do duelo
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
                "p_throws": "R",
                "stand": "R",
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
                # Fallback resiliente baseado nos priors se endpoint indisponível
                if fb_p >= bb_p and fb_p >= os_p:
                    ai_pred = "Fastball"
                elif bb_p >= os_p:
                    ai_pred = "Breaking Ball"
                else:
                    ai_pred = "Offspeed"

            # Resolução do Duelo pela Game Engine
            next_state, outcome = resolve_arcade_pitch(
                state=game,
                user_category=user_category,
                user_type=user_type,
                user_zone=user_zone,
                ai_prediction=ai_pred,
            )

            # Efeito sonoro correspondente
            res_type = outcome.get("result", "")
            if res_type == "HOME_RUN":
                st.session_state.last_sound_trigger = "home_run"
            elif res_type == "HIT":
                st.session_state.last_sound_trigger = "hit"
            elif res_type == "STRIKEOUT" or outcome.get("inning_cleared"):
                st.session_state.last_sound_trigger = "strikeout"
            elif res_type == "WHIFF":
                st.session_state.last_sound_trigger = "whiff"
            elif res_type == "STRIKE":
                st.session_state.last_sound_trigger = "strike"
            elif res_type == "WALK" or res_type == "BALL":
                st.session_state.last_sound_trigger = "ball"

            if next_state.game_over:
                st.session_state.last_sound_trigger = "game_over"

            st.session_state.arcade_state = next_state
            st.session_state.last_outcome = outcome
            st.rerun()


# --- COLUNA 3: FEEDBACK DO DUELO EM TEMPO REAL ---
with col_feedback:
    st.markdown("### 📡 Resolução do Duelo")

    outcome = st.session_state.last_outcome
    if outcome:
        res = outcome.get("result", "")
        ai_guess = outcome.get("ai_prediction", "")
        user_p = outcome.get("user_category", "")

        is_success = res in ("STRIKE", "WHIFF", "STRIKEOUT", "INNING_CLEARED")
        box_border = "#00FFCC" if is_success else "#FF007F"
        badge_bg = (
            "rgba(0, 255, 204, 0.2)" if is_success else "rgba(255, 0, 127, 0.2)"
        )

        feedback_html = f"""
        <div class="retro-cabinet-card" style="border-color: {box_border};">
            <div class="arcade-font neon-gold" style="font-size: 11px; margin-bottom: 8px;">
                RESULTADO DO ÚLTIMO ARREMESSO:
            </div>
            <div style="font-size: 14px; color: #FFFFFF; margin-bottom: 8px;">
                Seu Arremesso: <strong style="color: #00FFCC;">{user_p} ({outcome.get('user_type')})</strong>
            </div>
            <div style="font-size: 14px; color: #FFFFFF; margin-bottom: 8px;">
                Previsão da IA: <strong style="color: #FF007F;">{ai_guess}</strong>
            </div>
            <div style="padding: 8px; background: {badge_bg}; border-radius: 4px; font-weight: 700; font-size: 13px; color: #FFFFFF;">
                {outcome.get('message', '')}
            </div>
        </div>
        """
        st.markdown(_compact_html(feedback_html), unsafe_allow_html=True)
    else:
        st.info(
            "👉 Escolha seu arremesso e zona no console ao lado e clique em **ARREMESSAR CONTRA A IA** para começar!"
        )

    # Histórico de Arremessos
    st.markdown("#### 📜 Histórico de Arremessos")
    if game.history:
        for idx, h in enumerate(reversed(game.history[-5:]), 1):
            st.markdown(
                f"**#{len(game.history) - idx + 1}** {h.get('pitch_category')} ({h.get('pitch_type')}) "
                f"na Z{h.get('pitch_zone')} ➔ IA previu: `{h.get('ai_prediction')}` ➔ **{h.get('result')}**"
            )
    else:
        st.caption("Nenhum arremesso realizado nesta partida.")

    # Opções de Áudio e Reiniciar
    st.markdown("---")
    ca1, ca2 = st.columns(2)
    with ca1:
        st.session_state.minigame_audio = st.toggle(
            "🔊 Efeitos Sonoros 8-Bit",
            value=st.session_state.minigame_audio,
        )
    with ca2:
        if st.button("🔄 Reiniciar Jogo", use_container_width=True):
            st.session_state.arcade_state = ArcadeGameState(
                high_score=game.high_score
            )
            st.session_state.last_outcome = None
            st.session_state.last_sound_trigger = None
            st.rerun()
