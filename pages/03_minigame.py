"""
Página de Minigame Console Sports Broadcast HUD & Arena 3D (MLB Pitch Predictor).

Oferece:
- Passo 1: Seleção de Arremessador Estrela da MLB em 5 cards verticais com identidade cromática
  oficial de cada franquia (PIT, BAL, SF, ATL, LAD), fotos oficiais e arsenal por Tiers (S, A, B).
- Passo 2 (Modo Híbrido Visual):
  1. Modo 1 — Arena Broadcast 3D (Você no Montinho vs. IA): Simulador com animação vetorial 3D da
     trajetória da bola, Radar Statcast (MPH/RPM), Zona de Strike Interativa CCv2 (`st.components.v2`)
     de 13 zonas e comparação de disfarce de trajetória contra a previsão da IA.
  2. Modo 2 — Desafio Olho Biônico (Leitura de Pose 3D vs. IA): Veja o Print Real no Montinho,
     o Esqueleto 3D MediaPipe e os ângulos biomecânicos antes de a bola chegar à placa e tente
     antecipar o arremesso competindo contra o modelo de Visão Computacional.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from pitch_predictor.ccv2_components import interactive_strike_zone_ccv2
from pitch_predictor.fallback_engine import predict_tactical_fallback
from pitch_predictor.minigame_engine import (
    MLB_BATTERS,
    QUICK_AIM_MAP,
    ArcadeGameState,
    get_pitcher_tier_info,
    register_high_score,
    resolve_arcade_pitch,
)
from pitch_predictor.scenarios import MLB_PITCHERS, get_pitcher_profile
from pitch_predictor.serving_client import PitchServingClient
from pitch_predictor.ui_components import (
    _compact_html,
    get_custom_css,
    render_baseball_diamond_svg,
    render_pitch_flight_arena_svg,
    render_pitcher_photo_html,
    resolve_ui_theme_mode,
)
from pitch_predictor.vision.overlay import (
    extract_and_annotate_release_frame,
    extract_base_release_frame,
)

# ---------------------------------------------------------------------------
# 1. Console Sports Broadcast HUD Stylesheet (Cohesive Graphite + Stadium Amber)
# ---------------------------------------------------------------------------


def _get_console_broadcast_css(accent_hex: str = "#F59E0B") -> str:
    raw_css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=JetBrains+Mono:wght@500;700;800&family=Press+Start+2P&display=swap');

    .arcade-font {{
        font-family: 'JetBrains Mono', monospace !important;
        font-variant-numeric: tabular-nums !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }}

    .console-cabinet-card {{
        background: var(--pp-surface, #12141A);
        border: 1px solid var(--pp-border, #222631);
        border-top: 3px solid {accent_hex};
        border-radius: 8px;
        padding: 13px 15px;
        margin-bottom: 10px;
    }}

    .console-hud-header {{
        background: var(--pp-surface, #12141A);
        border: 1px solid var(--pp-border, #222631);
        border-left: 4px solid {accent_hex};
        border-radius: 8px;
        padding: 10px 16px;
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }}

    .console-hud-item {{
        font-family: 'JetBrains Mono', monospace;
        font-variant-numeric: tabular-nums;
        font-size: 11.5px;
        font-weight: 700;
        letter-spacing: 0.04em;
        color: var(--pp-muted, #94A3B8);
    }}

    .hud-val-amber {{
        color: #F59E0B;
        font-weight: 800;
        font-size: 14px;
    }}

    .hud-val-sky {{
        color: #38BDF8;
        font-weight: 800;
        font-size: 14px;
    }}

    .hud-val-emerald {{
        color: #10B981;
        font-weight: 800;
        font-size: 13px;
    }}

    .hud-val-crimson {{
        color: #EF4444;
        font-weight: 800;
        font-size: 13px;
    }}

    .quality-tag {{
        display: inline-block;
        background: var(--pp-surface-alt, #181B23);
        color: var(--pp-text, #F1F5F9);
        border: 1px solid var(--pp-border, #222631);
        border-radius: 4px;
        padding: 2px 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 9px;
        font-weight: 700;
        margin: 2px 2px;
        line-height: 1.25;
    }}

    .console-pitcher-card {{
        background: var(--pp-surface, #12141A);
        border: 1px solid var(--pp-border, #222631);
        border-radius: 8px;
        padding: 10px 9px;
        margin-bottom: 8px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 486px;
        text-align: center;
        transition: transform 0.16s ease, border-color 0.16s ease;
    }}

    .console-pitcher-card:hover {{
        transform: translateY(-2px);
    }}

    .tier-badge-s {{
        display: inline-block;
        background: rgba(245, 158, 11, 0.16);
        color: #F59E0B;
        border: 1px solid #F59E0B;
        border-radius: 4px;
        padding: 1px 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 9.5px;
        font-weight: 800;
    }}

    .tier-badge-a {{
        display: inline-block;
        background: rgba(56, 189, 248, 0.16);
        color: #38BDF8;
        border: 1px solid #38BDF8;
        border-radius: 4px;
        padding: 1px 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 9.5px;
        font-weight: 800;
    }}

    .tier-badge-b {{
        display: inline-block;
        background: rgba(16, 185, 129, 0.16);
        color: #10B981;
        border: 1px solid #10B981;
        border-radius: 4px;
        padding: 1px 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 9.5px;
        font-weight: 800;
    }}

    .timing-meter-bar {{
        border-radius: 6px;
        padding: 6px 14px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 700;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}

    .timing-cold {{
        background: rgba(16, 185, 129, 0.10);
        border: 1px solid rgba(16, 185, 129, 0.45);
        color: #10B981;
    }}

    .timing-alert {{
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.50);
        color: #F59E0B;
    }}

    .timing-hot {{
        background: rgba(239, 68, 68, 0.14);
        border: 1px solid rgba(239, 68, 68, 0.60);
        color: #EF4444;
    }}

    .console-gameover-banner {{
        background: var(--pp-surface, #12141A);
        border: 1px solid var(--pp-border, #222631);
        border-top: 4px solid #EF4444;
        border-radius: 8px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
    }}

    .arcade-hotkey-strip {{
        display: flex;
        align-items: center;
        justify-content: center;
        flex-wrap: wrap;
        gap: 12px;
        padding: 7px 14px;
        background: var(--pp-surface, #12141A);
        border: 1px solid var(--pp-border, #222631);
        border-radius: 6px;
        margin: 8px 0 12px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
    }}

    .hotkey-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }}

    .hotkey-badge kbd {{
        display: inline-block;
        padding: 2px 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 800;
        line-height: 1.3;
        color: var(--pp-text, #F1F5F9);
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid var(--pp-border, #334155);
        border-radius: 4px;
        box-shadow: 0 1px 0 rgba(0, 0, 0, 0.25);
    }}

    .pp-light .hotkey-badge kbd {{
        background: #FFFFFF;
        border-color: #CBD5E1;
        color: #0F172A;
        box-shadow: 0 1px 0 rgba(0, 0, 0, 0.08);
    }}

    .hotkey-label {{
        font-size: 10px;
        font-weight: 700;
        color: var(--pp-muted, #94A3B8);
        letter-spacing: 0.04em;
    }}

    .hotkey-sep {{
        color: var(--pp-border, #334155);
        font-size: 12px;
    }}

    .bio-gauge-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 11.5px;
        color: var(--pp-text, #F1F5F9);
        padding: 5px 0;
        border-bottom: 1px solid var(--pp-border, #222631);
        font-family: 'JetBrains Mono', monospace;
        font-variant-numeric: tabular-nums;
    }}
    </style>
    """
    return _compact_html(raw_css)


# ---------------------------------------------------------------------------
# 2. Sintetizador de Áudio Chiptune (Web Audio API)
# ---------------------------------------------------------------------------


def _render_chiptune_audio(sound_type: str | None) -> str:
    """Gera script JS embutido que sintetiza efeitos sonoros via Web Audio API."""
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


PID_TO_NAME: dict[int, str] = {
    694973: "Paul Skenes",
    669203: "Corbin Burnes",
    657277: "Logan Webb",
    519242: "Chris Sale",
    808967: "Yoshinobu Yamamoto",
}

PITCH_CODE_LABEL: dict[str, str] = {
    "FF": "4-Seam Fastball",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "ST": "Sweeper",
    "CU": "Curveball",
    "KC": "Knuckle Curve",
    "CH": "Changeup",
    "FS": "Splitter",
}


@st.cache_data(ttl=300)
def _load_all_vision_samples() -> pd.DataFrame:
    """Carrega os lances reais calibrados dos 5 arremessadores da MLB."""
    bio_path = Path("src/pitch_predictor/vision/sample_data/biomechanical_features.csv")
    man_path = Path("src/pitch_predictor/vision/sample_data/manifest.csv")
    if bio_path.exists() and man_path.exists():
        bio_df = pd.read_csv(bio_path)
        man_df = pd.read_csv(man_path)
        m_sub = man_df[
            [
                "pitcher_id",
                "game_pk",
                "at_bat_number",
                "pitch_number",
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
            merged["pitcher_name"] = (
                merged["pitcher_id"]
                .map(PID_TO_NAME)
                .fillna(merged.get("pitcher_name", "Paul Skenes"))
            )
            return merged.reset_index(drop=True)
    return pd.DataFrame()


def _build_shuffled_bionic_deck(samples_df: pd.DataFrame) -> list[int]:
    """Gera um baralho embaralhado alternando arremessadores e categorias."""
    if samples_df.empty:
        return []
    rng = random.SystemRandom()
    by_pitcher: dict[int, list[int]] = {}
    for idx, row in samples_df.iterrows():
        pid = int(row["pitcher_id"])
        by_pitcher.setdefault(pid, []).append(int(idx))
    for pool_list in by_pitcher.values():
        rng.shuffle(pool_list)

    pids = list(by_pitcher.keys())
    max_len = max(len(v) for v in by_pitcher.values())
    deck: list[int] = []
    last_pid: int | None = None
    last_cat: str | None = None

    for round_i in range(max_len):
        round_pids = pids[:]
        rng.shuffle(round_pids)
        if last_pid is not None and len(round_pids) > 1 and round_pids[0] == last_pid:
            swap_idx = rng.randint(1, len(round_pids) - 1)
            round_pids[0], round_pids[swap_idx] = round_pids[swap_idx], round_pids[0]
        for pid in round_pids:
            pool = by_pitcher[pid]
            if round_i < len(pool):
                if last_cat is not None and round_i + 1 < len(pool):
                    cur_cat = str(samples_df.iloc[pool[round_i]]["pitch_category"])
                    if cur_cat == last_cat:
                        for alt_pos in range(round_i + 1, len(pool)):
                            if str(samples_df.iloc[pool[alt_pos]]["pitch_category"]) != last_cat:
                                pool[round_i], pool[alt_pos] = pool[alt_pos], pool[round_i]
                                break
                chosen_idx = pool[round_i]
                deck.append(chosen_idx)
                last_pid = pid
                last_cat = str(samples_df.iloc[chosen_idx]["pitch_category"])
    return deck


def _resolve_pitch_label_and_speed(
    cur_pitches: list[dict[str, str]],
    real_cat: str,
    real_type: str,
) -> tuple[str, str]:
    by_code = next((p for p in cur_pitches if p.get("code") == real_type), None)
    if by_code:
        return by_code["name"], by_code["speed"]
    by_cat = next((p for p in cur_pitches if p.get("category") == real_cat), cur_pitches[0])
    specific_label = PITCH_CODE_LABEL.get(real_type, by_cat["name"])
    return specific_label, by_cat["speed"]


def _render_bionic_verdict_html(
    b_res: dict,
    human_hits: int,
    cv_hits: int,
    rounds: int,
) -> str:
    """Renderiza painel Broadcast Console com o resultado da rodada anterior."""
    h_ok = bool(b_res.get("human_ok", False))
    cv_ok = bool(b_res.get("cv_ok", False))
    safe_rounds = max(1, rounds)
    h_acc = (human_hits / safe_rounds) * 100.0
    cv_acc = (cv_hits / safe_rounds) * 100.0

    border_col = "#10B981" if h_ok else "#EF4444"
    pts = int(b_res.get("points_earned", 0))
    combo = int(b_res.get("combo", 1))
    round_num = int(b_res.get("round_num", safe_rounds))

    verdict_title = (
        f"LEITURA EXATA (+{pts} PTS • COMBO x{combo})"
        if h_ok
        else "ENGANADO PELO DISFARCE DE TRAJETÓRIA (COMBO RESETADO)"
    )

    user_badge_col = "#10B981" if h_ok else "#EF4444"
    user_badge_txt = "ACERTOU" if h_ok else "ERROU"

    cv_badge_col = "#38BDF8" if cv_ok else "#F59E0B"
    cv_badge_txt = "IA ACERTOU" if cv_ok else "IA ERROU"

    if human_hits > cv_hits:
        leader_txt = "HUMANO NA LIDERANÇA"
        leader_col = "#F59E0B"
    elif human_hits < cv_hits:
        leader_txt = "IA BIOMECÂNICA NA LIDERANÇA"
        leader_col = "#38BDF8"
    else:
        leader_txt = "EMPATE TÉCNICO HUMANO VS. IA"
        leader_col = "#10B981"

    raw_html = f"""
    <div style="background:var(--pp-surface,#12141A);border:1px solid var(--pp-border,#222631);border-left:4px solid {border_col};border-radius:8px;padding:14px 18px;margin-bottom:14px;">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;border-bottom:1px solid var(--pp-border,#222631);padding-bottom:8px;margin-bottom:10px;gap:8px;">
        <div class="arcade-font" style="font-size:12px;font-weight:800;color:{border_col};">{verdict_title}</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--pp-muted,#94A3B8);font-weight:700;">
          RODADA #{round_num} — <span style="color:#F59E0B;">{b_res['pitcher_name'].upper()}</span> ({b_res['p_throws']}HP • {b_res.get('pitcher_team', 'MLB')})
        </div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;align-items:stretch;">
        <div style="background:var(--pp-surface-alt,#181B23);border:1px solid var(--pp-border,#222631);border-radius:6px;padding:10px 12px;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:9.5px;color:var(--pp-muted,#94A3B8);text-transform:uppercase;margin-bottom:4px;">ARREMESSO EXECUTADO</div>
          <div style="font-size:15.5px;font-weight:800;color:#F59E0B;margin-bottom:2px;">{b_res['real_cat']} — {b_res['pitch_name']} ({b_res['speed']})</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:10.5px;color:var(--pp-muted,#94A3B8);">
            Frame #{b_res.get('frame_idx', 180)} • Slot: <strong>{b_res.get('arm_slot', 50.0):.1f}°</strong> ({b_res.get('slot_diff', 0.0):+.1f}°) • Cotovelo: <strong>{b_res.get('elbow_flex', 140.0):.1f}°</strong>
          </div>
        </div>
        <div style="background:var(--pp-surface-alt,#181B23);border:1px solid var(--pp-border,#222631);border-radius:6px;padding:10px 12px;display:flex;flex-direction:column;justify-content:center;gap:6px;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-size:12px;color:var(--pp-text,#F1F5F9);">Seu Palpite: <code>{b_res['user_guess']}</code></span>
            <span style="color:{user_badge_col};border:1px solid {user_badge_col};padding:1px 8px;border-radius:4px;font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:800;">{user_badge_txt}</span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-size:12px;color:var(--pp-text,#F1F5F9);">Leitura IA 3D: <code>{b_res['cv_guess']}</code></span>
            <span style="color:{cv_badge_col};border:1px solid {cv_badge_col};padding:1px 8px;border-radius:4px;font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:800;">{cv_badge_txt}</span>
          </div>
        </div>
        <div style="background:var(--pp-surface-alt,#181B23);border:1px solid var(--pp-border,#222631);border-radius:6px;padding:10px 12px;">
          <div style="display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--pp-text,#F1F5F9);margin-bottom:3px;">
            <span>VOCÊ: {human_hits}/{safe_rounds}</span>
            <strong style="color:#10B981;">{h_acc:.0f}%</strong>
          </div>
          <div style="width:100%;height:5px;background:var(--pp-border,#222631);border-radius:99px;overflow:hidden;margin-bottom:6px;">
            <div style="width:{min(100.0, h_acc):.1f}%;height:100%;background:#10B981;"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--pp-text,#F1F5F9);margin-bottom:3px;">
            <span>IA 3D: {cv_hits}/{safe_rounds}</span>
            <strong style="color:#38BDF8;">{cv_acc:.0f}%</strong>
          </div>
          <div style="width:100%;height:5px;background:var(--pp-border,#222631);border-radius:99px;overflow:hidden;margin-bottom:4px;">
            <div style="width:{min(100.0, cv_acc):.1f}%;height:100%;background:#38BDF8;"></div>
          </div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:9.5px;font-weight:800;color:{leader_col};text-align:right;">{leader_txt}</div>
        </div>
      </div>
    </div>
    """
    return _compact_html(raw_html)


def _predict_cv_from_angles(
    arm_slot: float,
    elbow_flex: float,
    rel_ht: float,
    trunk_tilt: float,
    base_profile: dict[str, float],
) -> str:
    slot_diff = arm_slot - float(base_profile.get("arm_slot_angle", 50.0))
    elbow_diff = elbow_flex - float(base_profile.get("elbow_flexion_angle", 140.0))
    if slot_diff < -3.0 or elbow_diff < -4.0:
        return "Breaking Ball"
    if slot_diff > 2.2:
        return "Offspeed"
    return "Fastball"


def _render_image_compat(image_data, caption: str) -> None:
    st.image(image_data, caption=caption, width="stretch")


# ---------------------------------------------------------------------------
# 3. Inicialização de Session State
# ---------------------------------------------------------------------------

if "minigame_step" not in st.session_state:
    st.session_state.minigame_step = "selection"

if "minigame_pitcher" not in st.session_state or st.session_state.minigame_pitcher not in MLB_PITCHERS:
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

MODE_OPTIONS = [
    "Modo 1: Arena Broadcast 3D (Você no Montinho)",
    "Modo 2: Desafio Olho Biônico (Adivinhe pela Pose 3D)",
]
if "minigame_mode" not in st.session_state or st.session_state.minigame_mode not in MODE_OPTIONS:
    st.session_state.minigame_mode = MODE_OPTIONS[0]

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

first_aim_key = next(iter(QUICK_AIM_MAP.keys()))
if "minigame_target_zone" not in st.session_state:
    st.session_state.minigame_target_zone = int(QUICK_AIM_MAP[first_aim_key]["zone"])

active_theme = resolve_ui_theme_mode()
active_pitcher_meta = MLB_PITCHERS.get(st.session_state.minigame_pitcher, MLB_PITCHERS["Paul Skenes"])
franchise_hex = active_pitcher_meta.get("team_primary_hex", "#F59E0B")

st.markdown(get_custom_css(theme_mode=active_theme, accent_hex=franchise_hex), unsafe_allow_html=True)
st.markdown(_get_console_broadcast_css(accent_hex=franchise_hex), unsafe_allow_html=True)

if st.session_state.minigame_audio and st.session_state.last_sound_trigger:
    st.markdown(
        _render_chiptune_audio(st.session_state.last_sound_trigger),
        unsafe_allow_html=True,
    )
    st.session_state.last_sound_trigger = None

game: ArcadeGameState = st.session_state.arcade_state

# ---------------------------------------------------------------------------
# 4. Painel Superior: Console Broadcast Scorebug HUD (Sempre Visível)
# ---------------------------------------------------------------------------

lives_bars = " ".join(["■" for _ in range(max(0, game.lives))] + ["□" for _ in range(max(0, 3 - game.lives))])
if game.lives <= 0:
    lives_bars = "0/3 [FIM]"

outs_display = " ".join(["●" if game.outs > i else "○" for i in range(2)])
combo_badge = f"x{game.combo_multiplier}" + (" [HOT]" if game.combo_multiplier >= 3 else "")

if game.timing_meter == 0:
    timing_html = """
    <div class="timing-meter-bar timing-cold">
        <span>TIMING DO REBATEDOR:</span>
        <span>[0/3] FRIO — Desprevenido</span>
    </div>
    """
elif game.timing_meter == 1:
    timing_html = """
    <div class="timing-meter-bar timing-alert">
        <span>TIMING DO REBATEDOR:</span>
        <span>[1/3] ALERTA — Antecipando repetição</span>
    </div>
    """
else:
    timing_html = f"""
    <div class="timing-meter-bar timing-hot">
        <span>TIMING DO REBATEDOR:</span>
        <span>[{game.timing_meter}/3] NA ESPERA! — Alto risco de contato extra-base</span>
    </div>
    """

hud_html = f"""
<div class="console-hud-header">
    <div class="console-hud-item">
        PONTUAÇÃO: <span class="hud-val-amber">{game.score:06d}</span>
    </div>
    <div class="console-hud-item">
        RECORDE: <span class="hud-val-sky">{game.high_score:06d}</span>
    </div>
    <div class="console-hud-item">
        VIDAS: <span class="hud-val-crimson">{lives_bars}</span>
    </div>
    <div class="console-hud-item">
        CONTAGEM: <span style="font-weight:800;color:var(--pp-text,#F1F5F9);font-size:13.5px;">{game.balls}B - {game.strikes}S</span>
    </div>
    <div class="console-hud-item">
        OUTS: <span class="hud-val-amber">{outs_display}</span>
    </div>
    <div class="console-hud-item">
        COMBO: <span class="hud-val-emerald">{combo_badge}</span>
    </div>
</div>
{timing_html}
<div class="arcade-hotkey-strip">
    <div class="hotkey-badge">
        <span class="hotkey-keys"><kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd></span>
        <span class="hotkey-label">ARSENAL TIER</span>
    </div>
    <div class="hotkey-sep">•</div>
    <div class="hotkey-badge">
        <span class="hotkey-keys"><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> / <kbd>↑</kbd><kbd>←</kbd><kbd>↓</kbd><kbd>→</kbd></span>
        <span class="hotkey-label">MIRA (13 ZONAS)</span>
    </div>
    <div class="hotkey-sep">•</div>
    <div class="hotkey-badge">
        <span class="hotkey-keys"><kbd>ESPAÇO</kbd> / <kbd>ENTER</kbd></span>
        <span class="hotkey-label">ARREMESSAR</span>
    </div>
</div>
"""
st.markdown(_compact_html(hud_html), unsafe_allow_html=True)

# Client-side Keyboard Shortcuts Controller (Keys 1-3: Tier, WASD/Arrows: Aim, Space/Enter: Throw)
components.html(
    """
    <script>
    (function() {
        const parentDoc = window.parent ? window.parent.document : document;
        if (!parentDoc || parentDoc.__ppArcadeKeysBound) return;
        parentDoc.__ppArcadeKeysBound = true;

        const NAV_MAP = {
            1:  { up: 11, down: 4, left: 11, right: 2 },
            2:  { up: 11, down: 5, left: 1,  right: 3 },
            3:  { up: 12, down: 6, left: 2,  right: 12 },
            4:  { up: 1,  down: 7, left: 11, right: 5 },
            5:  { up: 2,  down: 8, left: 4,  right: 6 },
            6:  { up: 3,  down: 9, left: 5,  right: 12 },
            7:  { up: 4,  down: 13, left: 13, right: 8 },
            8:  { up: 5,  down: 13, left: 7,  right: 9 },
            9:  { up: 6,  down: 14, left: 8,  right: 14 },
            11: { up: 11, down: 13, left: 11, right: 12 },
            12: { up: 12, down: 14, left: 11, right: 12 },
            13: { up: 11, down: 13, left: 13, right: 14 },
            14: { up: 12, down: 14, left: 13, right: 14 },
        };

        function findZoneElement(doc, zId) {
            const hits = Array.from(doc.querySelectorAll('.pp-zone-hit'));
            for (const h of hits) {
                const txt = (h.textContent || '').trim();
                if (txt === String(zId) || txt === ('Z' + zId)) return h;
            }
            return null;
        }

        function getActiveZoneFromIframes() {
            const iframes = Array.from(parentDoc.querySelectorAll('iframe'));
            for (const ifr of iframes) {
                try {
                    const d = ifr.contentDocument || ifr.contentWindow.document;
                    if (!d) continue;
                    const sel = d.querySelector('.pp-zone-hit[aria-selected="true"]');
                    if (sel) {
                        const t = (sel.textContent || '').trim().replace('Z', '');
                        const num = parseInt(t, 10);
                        if (!isNaN(num)) return num;
                    }
                } catch(err) {}
            }
            return 5;
        }

        function clickZoneInIframes(zId) {
            const iframes = Array.from(parentDoc.querySelectorAll('iframe'));
            for (const ifr of iframes) {
                try {
                    const d = ifr.contentDocument || ifr.contentWindow.document;
                    if (!d) continue;
                    const targetEl = findZoneElement(d, zId);
                    if (targetEl) {
                        targetEl.click();
                        return true;
                    }
                } catch(err) {}
            }
            return false;
        }

        parentDoc.addEventListener('keydown', function(e) {
            const active = parentDoc.activeElement;
            if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA')) return;

            const k = e.key;

            // Hotkeys 1, 2, 3: select pitch tier
            if (k === '1' || k === '2' || k === '3') {
                const idx = parseInt(k, 10) - 1;
                const labels = parentDoc.querySelectorAll('div[data-testid="stRadio"] label');
                if (labels && labels.length > idx) {
                    e.preventDefault();
                    labels[idx].click();
                }
                return;
            }

            // Space / Enter: trigger throw
            if (k === ' ' || k === 'Enter') {
                const btns = Array.from(parentDoc.querySelectorAll('button'));
                const throwBtn = btns.find(b => b.innerText && b.innerText.includes('ARREMESSAR CONTRA A IA'));
                if (throwBtn && !throwBtn.disabled) {
                    e.preventDefault();
                    throwBtn.click();
                }
                return;
            }

            // Arrow keys / WASD: strike zone navigation
            const lk = k.toLowerCase();
            let dir = null;
            if (lk === 'arrowup' || lk === 'w') dir = 'up';
            else if (lk === 'arrowdown' || lk === 's') dir = 'down';
            else if (lk === 'arrowleft' || lk === 'a') dir = 'left';
            else if (lk === 'arrowright' || lk === 'd') dir = 'right';

            if (dir) {
                const cur = getActiveZoneFromIframes();
                const nextZ = (NAV_MAP[cur] || NAV_MAP[5])[dir];
                if (nextZ) {
                    e.preventDefault();
                    clickZoneInIframes(nextZ);
                }
            }
        }, true);
    })();
    </script>
    """,
    height=0,
    width=0,
)

# ---------------------------------------------------------------------------
# 5. Tela de GAME OVER (Caso Vidas == 0)
# ---------------------------------------------------------------------------

if game.game_over:
    game_over_html = f"""
    <div class="console-gameover-banner">
        <h1 class="pp-display" style="font-size:28px;color:#EF4444;margin-bottom:8px;text-transform:uppercase;">FIM DE JOGO • DUELO ENCERRADO</h1>
        <p class="arcade-font" style="font-size:14px;color:#F59E0B;margin-bottom:8px;">
            PONTUAÇÃO FINAL: {game.score:06d} PTS
        </p>
        <p style="color:var(--pp-muted,#94A3B8);font-size:13.5px;margin-bottom:14px;">
            O modelo preditivo decodificou a sequência de arremessos. Registre suas iniciais no quadro oficial.
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
        if st.button("GRAVAR NO HALL DA FAMA", icon=":material/emoji_events:", width="stretch", type="primary"):
            st.session_state.arcade_leaderboard = register_high_score(
                initials=initials_input,
                score=game.score,
                leaderboard=st.session_state.arcade_leaderboard,
            )
            st.success("Iniciais registradas com sucesso!", icon=":material/check_circle:")

        if st.button("JOGAR NOVAMENTE", icon=":material/replay:", width="stretch"):
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

    st.markdown("### Hall da Fama Oficial (Top 10)")
    for rank, entry in enumerate(st.session_state.arcade_leaderboard, 1):
        st.markdown(f"**#{rank:02d}** `{entry['initials']}` — **{entry['score']:06d} pts**")
    st.stop()


# ---------------------------------------------------------------------------
# 6. PASSO 1: Seleção de Arremessador Estrela da MLB (5 Cards Verticais)
# ---------------------------------------------------------------------------

if st.session_state.minigame_step == "selection":
    st.markdown("## Passo 1: Escolha seu Arremessador Estrela")
    st.caption(
        "Cada astro da MLB possui identidade oficial da franquia, arsenal calibrado em Tiers (S, A, B), trajetória 3D e poses reais capturadas no montinho."
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
        p_meta = MLB_PITCHERS[p_name]
        p_hex = p_meta.get("team_primary_hex", "#F59E0B")
        tags_html = " ".join([f'<span class="quality-tag">{tag}</span>' for tag in p_info["tags"]])

        p_pitches = p_info["pitches"]
        pitch_s = p_pitches[0]
        pitch_a = p_pitches[1]
        pitch_b = p_pitches[2]

        photo_html = render_pitcher_photo_html(p_name)

        with cols[idx]:
            card_html = f"""
            <div class="console-pitcher-card" style="border-top:3px solid {p_hex};">
                <div style="margin-bottom:4px;min-height:42px;">
                    {tags_html}
                </div>
                <div class="pp-display" style="font-size:17px;font-weight:800;color:var(--pp-text,#F1F5F9);margin-bottom:1px;line-height:1.2;">
                    {p_info['name']}
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--pp-muted,#94A3B8);margin-bottom:4px;">
                    {p_meta.get('team_abbr', 'MLB')} {p_meta.get('jersey_number', '')} • <strong style="color:{p_hex};">{p_info['p_throws']}HP</strong>
                </div>

                {photo_html}

                <div style="background:var(--pp-surface-alt,#181B23);border:1px solid var(--pp-border,#222631);border-radius:6px;padding:6px 7px;font-size:10px;text-align:left;margin-bottom:6px;">
                    <div style="margin-bottom:5px;">
                        <span class="tier-badge-s">TIER S</span>
                        <strong style="color:var(--pp-text,#F1F5F9);font-size:10.5px;display:block;margin-top:2px;">{pitch_s['name']} ({pitch_s['speed']})</strong>
                        <span style="color:var(--pp-muted,#94A3B8);font-size:8.8px;">{pitch_s['desc']}</span>
                    </div>
                    <div style="margin-bottom:5px;">
                        <span class="tier-badge-a">TIER A</span>
                        <strong style="color:var(--pp-text,#F1F5F9);font-size:10.5px;display:block;margin-top:2px;">{pitch_a['name']} ({pitch_a['speed']})</strong>
                        <span style="color:var(--pp-muted,#94A3B8);font-size:8.8px;">{pitch_a['desc']}</span>
                    </div>
                    <div>
                        <span class="tier-badge-b">TIER B</span>
                        <strong style="color:var(--pp-text,#F1F5F9);font-size:10.5px;display:block;margin-top:2px;">{pitch_b['name']} ({pitch_b['speed']})</strong>
                        <span style="color:var(--pp-muted,#94A3B8);font-size:8.8px;">{pitch_b['desc']}</span>
                    </div>
                </div>
                <div style="font-size:9.5px;color:var(--pp-muted,#94A3B8);margin-bottom:4px;min-height:28px;line-height:1.25;">
                    {p_info['description']}
                </div>
            </div>
            """
            st.markdown(_compact_html(card_html), unsafe_allow_html=True)

            if st.button(
                f"Escolher {p_name}",
                key=f"btn_pick_{p_name.replace(' ', '_')}",
                icon=":material/sports_baseball:",
                type="primary",
                width="stretch",
            ):
                st.session_state.minigame_pitcher = p_name
                st.session_state.arcade_state.pitcher_name = p_name
                st.session_state.bionic_clip_idx = 0
                st.session_state.bionic_deck = None
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
team_abbr = full_profile.get("team_abbr", "MLB")
jersey_number = full_profile.get("jersey_number", "#00")

col_top_l, col_top_m, col_top_r = st.columns([1.3, 2.2, 0.8], vertical_alignment="center")
with col_top_l:
    st.markdown(
        f"**No Montinho:** <strong style='color:{franchise_hex};'>{pitcher_name}</strong> "
        f"`{team_abbr} {jersey_number}` ({p_throws}HP)",
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
    if st.button("Trocar (Bullpen)", icon=":material/swap_horiz:", width="stretch"):
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

    if (
        "bionic_deck" not in st.session_state
        or not isinstance(st.session_state.bionic_deck, list)
        or len(st.session_state.bionic_deck) != total_samples
    ):
        st.session_state.bionic_deck = _build_shuffled_bionic_deck(samples_df)

    deck_pos = int(st.session_state.bionic_clip_idx) % total_samples
    row_idx = int(st.session_state.bionic_deck[deck_pos])
    sample_row = samples_df.iloc[row_idx]

    cur_pid = int(sample_row["pitcher_id"])
    cur_pname = PID_TO_NAME.get(cur_pid, str(sample_row.get("pitcher_name", pitcher_name)))
    cur_p_info = get_pitcher_tier_info(cur_pname)
    cur_pitches = cur_p_info["pitches"]
    cur_full_profile = get_pitcher_profile(cur_pname)
    cur_p_throws = str(cur_p_info.get("p_throws", "R"))
    cur_hex = cur_full_profile.get("team_primary_hex", franchise_hex)

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

    b_res = st.session_state.bionic_last_result
    if b_res:
        st.markdown(
            _render_bionic_verdict_html(
                b_res=b_res,
                human_hits=int(st.session_state.bionic_human_hits),
                cv_hits=int(st.session_state.bionic_cv_hits),
                rounds=int(st.session_state.bionic_rounds),
            ),
            unsafe_allow_html=True,
        )

    current_round_num = int(st.session_state.bionic_rounds) + 1
    st.markdown(
        f"### Rodada #{current_round_num} • No Montinho: **{cur_pname}** ({cur_p_throws}HP — *{cur_p_info['team']}*)"
    )
    st.caption(
        f"Analise o **Print Real no Montinho**, o **Esqueleto 3D MediaPipe** e os **Desvios Angulares** de **{cur_pname}** no Frame #{r_frame} antes de a bola deixar sua mão."
    )

    bio_col1, bio_col2, bio_col3 = st.columns([1.15, 1.15, 1.1], gap="medium")

    with bio_col1:
        st.markdown(f"**01. Print Base de {cur_pname} (Frame #{r_frame})**")
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
        st.markdown("**02. Modelo da Pose 3D (Esqueleto Extraído)**")
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
        st.markdown("**03. Telemetria Biomecânica & Seu Palpite**")
        gauges_html = f"""
        <div class="console-cabinet-card" style="border-top-color:{cur_hex};padding:12px;">
            <div class="arcade-font" style="font-size:10px;font-weight:800;color:{cur_hex};margin-bottom:8px;">
                TELEMETRIA DE {cur_pname.upper()} (FRAME #{r_frame})
            </div>
            <div class="bio-gauge-row">
                <span>Arm Slot (Ângulo Braço):</span>
                <strong style="color:#38BDF8;">{arm_slot:.1f}° ({slot_diff:+.1f}°)</strong>
            </div>
            <div class="bio-gauge-row">
                <span>Flexão do Cotovelo:</span>
                <strong style="color:#F59E0B;">{elbow_flex:.1f}° ({elbow_diff:+.1f}°)</strong>
            </div>
            <div class="bio-gauge-row">
                <span>Altura de Soltura:</span>
                <strong style="color:#10B981;">{rel_ht:.2f} ({ht_diff:+.2f})</strong>
            </div>
            <div class="bio-gauge-row" style="border-bottom:none;">
                <span>Inclinação do Tronco:</span>
                <strong>{trunk_tilt:.1f}° ({tilt_diff:+.1f}°)</strong>
            </div>
            <div style="margin-top:8px;padding:7px;background:var(--pp-surface-alt,#181B23);border-radius:4px;font-size:10.5px;color:var(--pp-muted,#94A3B8);">
                <strong>Sinais Físicos:</strong> Queda de Arm Slot e cotovelo flexionado indicam <em>Breaking Ball</em>; Arm Slot elevado sugere <em>Offspeed</em>; alinhado à base indica <em>Fastball</em>.
            </div>
        </div>
        """
        st.markdown(_compact_html(gauges_html), unsafe_allow_html=True)

        guess_fb = st.button("PALPITE: FASTBALL [FF]", width="stretch", type="primary")
        guess_bb = st.button("PALPITE: BREAKING BALL [SL/ST]", width="stretch")
        guess_os = st.button("PALPITE: OFFSPEED [CH/FS]", width="stretch")
        next_clip = st.button("Sortear Outro Lance / Arremessador", icon=":material/shuffle:", width="stretch")

        if next_clip:
            next_pos = (deck_pos + 1) % total_samples
            if next_pos == 0:
                st.session_state.bionic_deck = _build_shuffled_bionic_deck(samples_df)
            st.session_state.bionic_clip_idx = next_pos
            next_row = samples_df.iloc[int(st.session_state.bionic_deck[next_pos])]
            next_pid = int(next_row["pitcher_id"])
            st.session_state.minigame_pitcher = PID_TO_NAME.get(next_pid, cur_pname)
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
            pts = 0
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

            p_label, p_speed = _resolve_pitch_label_and_speed(cur_pitches, real_cat, real_type)
            st.session_state.bionic_last_result = {
                "round_num": int(st.session_state.bionic_rounds),
                "pitcher_name": cur_pname,
                "pitcher_team": cur_p_info["team"],
                "p_throws": cur_p_throws,
                "frame_idx": r_frame,
                "user_guess": user_guess,
                "cv_guess": cv_ai_guess,
                "real_cat": real_cat,
                "real_type": real_type,
                "pitch_name": p_label,
                "speed": p_speed,
                "arm_slot": arm_slot,
                "slot_diff": slot_diff,
                "elbow_flex": elbow_flex,
                "human_ok": human_ok,
                "cv_ok": cv_ok,
                "points_earned": pts,
                "combo": int(game.combo_multiplier),
            }
            next_pos = (deck_pos + 1) % total_samples
            if next_pos == 0:
                st.session_state.bionic_deck = _build_shuffled_bionic_deck(samples_df)
            st.session_state.bionic_clip_idx = next_pos
            next_row = samples_df.iloc[int(st.session_state.bionic_deck[next_pos])]
            next_pid = int(next_row["pitcher_id"])
            st.session_state.minigame_pitcher = PID_TO_NAME.get(next_pid, cur_pname)
            st.rerun()


# ===========================================================================
# MODO 1 (PADRÃO): ARENA BROADCAST 3D — VOCÊ NO MONTINHO VS. IA
# ===========================================================================
else:
    col_batter, col_mound, col_feedback = st.columns([0.95, 1.55, 1.05], gap="medium")

    with col_batter:
        st.markdown("### No Bastão")

        batter_profile = next(
            (b for b in MLB_BATTERS if b["name"] == game.current_batter_name),
            MLB_BATTERS[0],
        )

        batter_card_html = f"""
        <div class="console-cabinet-card" style="padding:11px 13px;">
            <div class="arcade-font" style="font-size:9.5px;color:#F59E0B;font-weight:800;margin-bottom:3px;">
                REBATEDOR OFICIAL MLB
            </div>
            <div style="font-size:16.5px;font-weight:800;color:var(--pp-text,#F1F5F9);">
                {batter_profile['name']}
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:10.5px;color:#38BDF8;margin-bottom:6px;">
                {batter_profile['team']} • <strong>{batter_profile['stand']}HB</strong>
            </div>
            <div style="font-size:11.5px;color:var(--pp-muted,#94A3B8);margin-bottom:8px;line-height:1.35;">
                {batter_profile['description']}
            </div>
            <div style="background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.45);border-radius:4px;padding:4px 7px;font-size:10.5px;color:#F59E0B;">
                <strong>Traço Tático:</strong> {batter_profile['trait']}
            </div>
        </div>
        """
        st.markdown(_compact_html(batter_card_html), unsafe_allow_html=True)

        st.markdown(
            render_baseball_diamond_svg(game.on_1b, game.on_2b, game.on_3b, game.outs),
            unsafe_allow_html=True,
        )

    with col_mound:
        st.markdown("### Gabinete de Arremesso")
        st.caption("Escolha o arremesso por Tier, ajuste o alvo rápido ou clique diretamente nas 13 zonas da placa à direita:")

        tier_options = [
            f"[TIER S] {pitches[0]['name']} ({pitches[0]['speed']})",
            f"[TIER A] {pitches[1]['name']} ({pitches[1]['speed']})",
            f"[TIER B] {pitches[2]['name']} ({pitches[2]['speed']})",
        ]

        selected_tier_str = st.radio(
            "1. Selecione o Arremesso:",
            tier_options,
            index=0,
            label_visibility="collapsed",
            key="minigame_tier_selection_radio",
        )

        if "TIER S" in selected_tier_str:
            chosen_pitch = pitches[0]
        elif "TIER A" in selected_tier_str:
            chosen_pitch = pitches[1]
        else:
            chosen_pitch = pitches[2]

        def _on_quick_aim_change() -> None:
            choice = st.session_state.get("minigame_quick_aim_radio")
            if choice in QUICK_AIM_MAP:
                z = int(QUICK_AIM_MAP[choice]["zone"])
                st.session_state.minigame_target_zone = z
                st.session_state["minigame_ccv2_zone"] = {"zone": z}

        aim_choice = st.radio(
            "2. Ponto de Destino Rápido:",
            list(QUICK_AIM_MAP.keys()),
            format_func=lambda k: QUICK_AIM_MAP[k]["label"],
            horizontal=True,
            key="minigame_quick_aim_radio",
            on_change=_on_quick_aim_change,
        )

        target_category = chosen_pitch["category"]
        target_type = chosen_pitch["code"]
        target_tier = chosen_pitch["tier"]

    with col_feedback:
        st.markdown("### Resolução & Alvo CCv2")

        outcome = st.session_state.last_outcome
        if outcome:
            res = outcome.get("result", "")
            ai_guess = outcome.get("ai_prediction", "")
            user_p = outcome.get("user_category", "")
            tier = outcome.get("pitch_tier", "A")

            is_success = res in ("STRIKE", "WHIFF", "STRIKEOUT", "INNING_CLEARED", "OUT", "FOUL")
            box_border = "#10B981" if is_success else "#EF4444"
            badge_bg = (
                "rgba(16, 185, 129, 0.14)" if is_success else "rgba(239, 68, 68, 0.14)"
            )

            feedback_html = f"""
            <div class="console-cabinet-card" style="border-top-color:{box_border};margin-bottom:10px;">
                <div class="arcade-font" style="font-size:10px;color:{box_border};font-weight:800;margin-bottom:6px;">
                    RESULTADO DO DUELO
                </div>
                <div style="font-size:12.5px;color:var(--pp-text,#F1F5F9);margin-bottom:3px;">
                    Seu Arremesso: <strong style="color:#38BDF8;">Tier {tier} • {user_p} ({outcome.get('user_type')})</strong>
                </div>
                <div style="font-size:12.5px;color:var(--pp-text,#F1F5F9);margin-bottom:6px;">
                    Previsão da IA: <strong style="color:#F59E0B;">{ai_guess}</strong>
                </div>
                <div style="padding:6px 10px;background:{badge_bg};border:1px solid {box_border};border-radius:4px;font-weight:700;font-size:12px;color:var(--pp-text,#F1F5F9);">
                    {outcome.get('message', '')}
                </div>
            </div>
            """
            st.markdown(_compact_html(feedback_html), unsafe_allow_html=True)
        else:
            st.info(
                "Escolha o Tier, clique na Zona de Strike interativa abaixo e acione **ARREMESSAR CONTRA A IA**.",
                icon=":material/sports_baseball:",
            )

        clicked_zone = interactive_strike_zone_ccv2(
            selected_zone=int(st.session_state.minigame_target_zone),
            pitch_category=target_category,
            accent_hex=franchise_hex,
            theme_mode=active_theme,
            key="minigame_ccv2_zone",
        )
        st.session_state.minigame_target_zone = int(clicked_zone)

        st.markdown("#### Últimos Lances")
        if game.history:
            for idx, h in enumerate(reversed(game.history[-3:]), 1):
                st.markdown(
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:11px;color:var(--pp-muted,#94A3B8);margin-bottom:4px;'>"
                    f"<strong>#{len(game.history) - idx + 1}</strong> T-{h.get('pitch_tier', 'A')} {h.get('pitch_type')} (Z{h.get('pitch_zone', 5)}) "
                    f"/ IA: <code>{h.get('ai_prediction')}</code> / <strong style='color:var(--pp-text,#F1F5F9);'>{h.get('result')}</strong>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Nenhum arremesso realizado nesta partida.")

        st.session_state.minigame_audio = st.toggle(
            "Áudio Sintetizado 8-Bit",
            value=st.session_state.minigame_audio,
        )
        if st.button("Novo Jogo", icon=":material/restart_alt:", width="stretch"):
            st.session_state.arcade_state = ArcadeGameState(
                high_score=game.high_score,
                pitcher_name=pitcher_name,
            )
            st.session_state.last_outcome = None
            st.session_state.bionic_last_result = None
            st.session_state.last_sound_trigger = None
            st.session_state.minigame_step = "selection"
            st.rerun()

    # Renderiza a Arena 3D e o botão principal na coluna central já sincronizados com clicked_zone
    with col_mound:
        target_zone = int(st.session_state.minigame_target_zone)
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
            "ARREMESSAR CONTRA A IA!",
            icon=":material/bolt:",
            type="primary",
            width="stretch",
        )

        if pitch_btn:
            with st.spinner("Consultando modelo preditivo no Databricks..."):
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
