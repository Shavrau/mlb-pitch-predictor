"""
MLB Tactical Scenarios, Pitcher Archetypes, and Official At-Bat State Progression.

Provides:
- `PITCHER_ARCHETYPES`: Real MLB pitcher profiles with realistic repertoire priors.
- `TACTICAL_SCENARIOS`: Pre-configured MLB game situations for one-click simulation.
- `PITCH_TYPES_BY_CATEGORY`: Mapping of pitch categories to official Statcast pitch codes.
- `advance_at_bat_state`: Deterministic state machine advancing balls, strikes, outs, and
  previous pitch memory according to official MLB rules.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

PITCH_TYPES_BY_CATEGORY: dict[str, list[tuple[str, str]]] = {
    "First Pitch": [("None", "None — Abertura de Duelo (1º Arremesso)")],
    "Fastball": [
        ("FF", "FF — Four-Seam Fastball (Reta de 4 Costuras)"),
        ("SI", "SI — Sinker (Two-Seam / Bola Rápida de Afundamento)"),
        ("FC", "FC — Cutter (Bola Rápida Cortada)"),
    ],
    "Breaking Ball": [
        ("SL", "SL — Slider (Efeito Lateral Rápido)"),
        ("ST", "ST — Sweeper (Slider de Grande Varredura Horizontal)"),
        ("CU", "CU — Curveball (Curva Clássica 12-6)"),
        ("KC", "KC — Knuckle-Curve (Curva de Ponta de Dedo)"),
    ],
    "Offspeed": [
        ("CH", "CH — Changeup (Mudança de Velocidade)"),
        ("FS", "FS — Split-Finger (Splitter)"),
    ],
}

PITCHER_ARCHETYPES: dict[str, dict[str, Any]] = {
    "Média da Liga MLB (League Average)": {
        "pitcher_fb_prior": 0.5543,
        "pitcher_bb_prior": 0.3053,
        "pitcher_os_prior": 0.1404,
        "p_throws": "R",
        "signature_arsenal": "FF (95 mph) • SL (86 mph) • CH (87 mph)",
        "description": "Distribuição basal da MLB (~55% Fastball, ~31% Breaking, ~14% Offspeed).",
    },
    "Power Pitcher (Paul Skenes / Spencer Strider)": {
        "pitcher_fb_prior": 0.6400,
        "pitcher_bb_prior": 0.2800,
        "pitcher_os_prior": 0.0800,
        "p_throws": "R",
        "signature_arsenal": "FF (100+ mph) • SL/Splinker (90 mph) • CH (88 mph)",
        "description": "Dominância pela velocidade no topo da zona combinada com Slider de wipeout.",
    },
    "Breaking Specialist (Corbin Burnes / Sonny Gray)": {
        "pitcher_fb_prior": 0.4200,
        "pitcher_bb_prior": 0.4600,
        "pitcher_os_prior": 0.1200,
        "p_throws": "R",
        "signature_arsenal": "ST/CU (84 mph) • SL (87 mph) • FC (95 mph)",
        "description": "Arsenal ancorado em Sweepers, Sliders e Curvas com alta taxa de rotação (Spin Rate).",
    },
    "Offspeed Master (Logan Webb / Kevin Gausman)": {
        "pitcher_fb_prior": 0.4600,
        "pitcher_bb_prior": 0.2200,
        "pitcher_os_prior": 0.3200,
        "p_throws": "R",
        "signature_arsenal": "SI/FF (94 mph) • CH/FS (85 mph) • SL (83 mph)",
        "description": "Especialista em tunelamento com Changeup/Splitter na borda inferior (zonas 7-9 e 13-14).",
    },
    "Lefty Crafty Ace (Chris Sale / Blake Snell)": {
        "pitcher_fb_prior": 0.4800,
        "pitcher_bb_prior": 0.3600,
        "pitcher_os_prior": 0.1600,
        "p_throws": "L",
        "signature_arsenal": "FF (95 mph) • SL/CU (81 mph) • CH (86 mph)",
        "description": "Canhoto de ângulo cruzado letal contra rebatedores canhotos (LHP vs LHB) e mudança contra destros.",
    },
}

MLB_PITCHERS: dict[str, dict[str, Any]] = {
    "Paul Skenes": {
        "pitcher_id": 694973,
        "name": "Paul Skenes",
        "display_name": "Paul Skenes (#694973) — Pittsburgh Pirates",
        "team": "Pittsburgh Pirates",
        "archetype": "Power Pitcher (Paul Skenes / Spencer Strider)",
        "p_throws": "R",
        "pitcher_fb_prior": 0.6400,
        "pitcher_bb_prior": 0.2800,
        "pitcher_os_prior": 0.0800,
        "signature_arsenal": "FF (100+ mph) • Splinker (94 mph) • SL (88 mph)",
        "arm_slot_type": "High 3/4 (~50°)",
        "bio_profile": {
            "arm_slot_angle": 49.78,
            "elbow_flexion_angle": 146.11,
            "release_height": 0.710,
            "trunk_tilt": 11.67,
            "stride_extension": 1.45,
        },
        "description": "Fenômeno da MLB com bola rápida de 102 mph e arremesso híbrido splinker no topo da zona.",
    },
    "Corbin Burnes": {
        "pitcher_id": 669203,
        "name": "Corbin Burnes",
        "display_name": "Corbin Burnes (#669203) — Baltimore Orioles",
        "team": "Baltimore Orioles",
        "archetype": "Breaking Specialist (Corbin Burnes / Sonny Gray)",
        "p_throws": "R",
        "pitcher_fb_prior": 0.4200,
        "pitcher_bb_prior": 0.4600,
        "pitcher_os_prior": 0.1200,
        "signature_arsenal": "FC (95 mph) • ST/CU (84 mph) • CH (89 mph)",
        "arm_slot_type": "Standard 3/4 (~46°)",
        "bio_profile": {
            "arm_slot_angle": 46.15,
            "elbow_flexion_angle": 122.40,
            "release_height": 0.680,
            "trunk_tilt": 18.20,
            "stride_extension": 1.38,
        },
        "description": "Vencedor do Cy Young com altíssimo spin rate (2800+ RPM), Cutter de elite e curvas devastadoras.",
    },
    "Logan Webb": {
        "pitcher_id": 657277,
        "name": "Logan Webb",
        "display_name": "Logan Webb (#657277) — San Francisco Giants",
        "team": "San Francisco Giants",
        "archetype": "Offspeed Master (Logan Webb / Kevin Gausman)",
        "p_throws": "R",
        "pitcher_fb_prior": 0.4600,
        "pitcher_bb_prior": 0.2200,
        "pitcher_os_prior": 0.3200,
        "signature_arsenal": "SI (93 mph) • CH (87 mph) • SL (83 mph)",
        "arm_slot_type": "Low 3/4 (~38°)",
        "bio_profile": {
            "arm_slot_angle": 38.40,
            "elbow_flexion_angle": 112.50,
            "release_height": 0.640,
            "trunk_tilt": 15.30,
            "stride_extension": 1.40,
        },
        "description": "Líder em groundballs da MLB, mestre do tunelamento com Changeup que despenca na borda inferior.",
    },
    "Chris Sale": {
        "pitcher_id": 519242,
        "name": "Chris Sale",
        "display_name": "Chris Sale (#519242) — Atlanta Braves",
        "team": "Atlanta Braves",
        "archetype": "Lefty Crafty Ace (Chris Sale / Blake Snell)",
        "p_throws": "L",
        "pitcher_fb_prior": 0.4800,
        "pitcher_bb_prior": 0.3600,
        "pitcher_os_prior": 0.1600,
        "signature_arsenal": "FF (95 mph) • SL (81 mph) • CH (86 mph)",
        "arm_slot_type": "Low Sidearm / 3/4 Canhoto (~25°)",
        "bio_profile": {
            "arm_slot_angle": 25.10,
            "elbow_flexion_angle": 104.80,
            "release_height": 0.620,
            "trunk_tilt": 24.10,
            "stride_extension": 1.50,
        },
        "description": "Cy Young da NL 2024, canhoto lendário de ângulo cruzado extremo e slider com quebra horizontal fulminante.",
    },
    "Yoshinobu Yamamoto": {
        "pitcher_id": 808967,
        "name": "Yoshinobu Yamamoto",
        "display_name": "Yoshinobu Yamamoto (#808967) — Los Angeles Dodgers",
        "team": "Los Angeles Dodgers",
        "archetype": "Offspeed Master (Logan Webb / Kevin Gausman)",
        "p_throws": "R",
        "pitcher_fb_prior": 0.4800,
        "pitcher_bb_prior": 0.2600,
        "pitcher_os_prior": 0.2600,
        "signature_arsenal": "FF (96 mph) • Splitter (90 mph) • Rainbow Curve (78 mph) • Cutter (92 mph)",
        "arm_slot_type": "Flexible Mid 3/4 (~48°)",
        "bio_profile": {
            "arm_slot_angle": 47.80,
            "elbow_flexion_angle": 118.20,
            "release_height": 0.650,
            "trunk_tilt": 16.50,
            "stride_extension": 1.44,
        },
        "description": "Craque dos Dodgers e astro japonês, famoso pela mecânica elástica inspirada no dardo, splitter devastador e curva arco-íris.",
    },
}

TACTICAL_SCENARIOS: dict[str, dict[str, Any]] = {
    "⚾ Abertura de Duelo (0-0 First Pitch)": {
        "badge": "NEUTRAL • 0-0",
        "description": "Início do turno de rebatida na 1ª entrada sem corredores. O arremessador busca estabelecer o strike #1.",
        "inputs": {
            "prev_pitch_category": "First Pitch",
            "prev_pitch_type": "None",
            "prev_pitch_zone": 0,
            "prev_pitch_result": "First Pitch",
            "p_throws": "R",
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
            "pitcher_fb_prior": 0.5543,
            "pitcher_bb_prior": 0.3053,
            "pitcher_os_prior": 0.1404,
        },
    },
    "🎯 Pitcher Ahead (0-2 Waste / Chase Setup)": {
        "badge": "PITCHER AHEAD • 0-2",
        "description": "Arremessador em vantagem máxima (0-2) após bola rápida no topo da zona de strike. Momento clássico de Slider/Curva fora da zona (zona 14).",
        "inputs": {
            "prev_pitch_category": "Fastball",
            "prev_pitch_type": "FF",
            "prev_pitch_zone": 2,
            "prev_pitch_result": "S",
            "p_throws": "R",
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
            "pitcher_fb_prior": 0.4200,
            "pitcher_bb_prior": 0.4600,
            "pitcher_os_prior": 0.1200,
        },
    },
    "🔥 Contagem Cheia & Bases Lotadas (3-2 High Leverage)": {
        "badge": "FULL COUNT • BASES LOADED",
        "description": "Pressão máxima na 8ª entrada: 2 outs, contagem 3-2 e bases lotadas. O arremessador não pode ceder a base por bolas (Walk) e precisa atacar a zona.",
        "inputs": {
            "prev_pitch_category": "Breaking Ball",
            "prev_pitch_type": "SL",
            "prev_pitch_zone": 14,
            "prev_pitch_result": "B",
            "p_throws": "R",
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
            "pitcher_fb_prior": 0.6400,
            "pitcher_bb_prior": 0.2800,
            "pitcher_os_prior": 0.0800,
        },
    },
    "⚠️ Pitcher Behind com RISP (3-0 Green Light)": {
        "badge": "PITCHER BEHIND • 3-0",
        "description": "Contagem crítica 3-0 com corredor na 2ª base (posição de anotar corrida / RISP). Alta probabilidade de Bola Rápida para recuperar a zona de strike.",
        "inputs": {
            "prev_pitch_category": "Offspeed",
            "prev_pitch_type": "CH",
            "prev_pitch_zone": 13,
            "prev_pitch_result": "B",
            "p_throws": "R",
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
            "pitcher_fb_prior": 0.5543,
            "pitcher_bb_prior": 0.3053,
            "pitcher_os_prior": 0.1404,
        },
    },
    "🛑 Duelo de Mudança (1-1 Platoon Oposto & Corredor na 1B)": {
        "badge": "DOUBLE PLAY SETUP • 1-1",
        "description": "Corredor na 1ª base com 1 out e mão oposta (RHP vs LHB). Cenário clássico para Changeup/Splitter induzindo bola rasteira (groundball) para eliminação dupla (Double Play).",
        "inputs": {
            "prev_pitch_category": "Fastball",
            "prev_pitch_type": "SI",
            "prev_pitch_zone": 7,
            "prev_pitch_result": "B",
            "p_throws": "R",
            "stand": "L",
            "balls": 1,
            "strikes": 1,
            "outs_when_up": 1,
            "inning": 7,
            "inning_topbot": "Top",
            "score_diff": 0,
            "on_1b_flag": 1,
            "on_2b_flag": 0,
            "on_3b_flag": 0,
            "n_thruorder_pitcher": 3,
            "pitcher_fb_prior": 0.4600,
            "pitcher_bb_prior": 0.2200,
            "pitcher_os_prior": 0.3200,
        },
    },
}


def advance_at_bat_state(
    current_state: dict[str, Any],
    pitch_category: str,
    pitch_type: str,
    pitch_zone: int,
    outcome: str,
) -> dict[str, Any]:
    """
    Advance the tactical at-bat state according to official MLB rules.

    Parameters
    ----------
    current_state : dict
        Current tactical state dictionary (`balls`, `strikes`, `outs_when_up`, etc.).
    pitch_category : str
        Category of the pitch just thrown (`"Fastball"`, `"Breaking Ball"`, `"Offspeed"`).
    pitch_type : str
        Specific Statcast pitch code (`"FF"`, `"SL"`, `"CH"`, etc.).
    pitch_zone : int
        Statcast zone where the pitch crossed the plate (`1..9` or `11..14`).
    outcome : str
        Pitch result: `"S"` (Strike called/swinging), `"Foul"` (Foul ball),
        `"B"` (Ball), or `"X"` (Ball in play).

    Returns
    -------
    dict
        Updated tactical state with `at_bat_event` describing the transition.
    """
    next_state = deepcopy(current_state)
    balls = int(next_state.get("balls", 0))
    strikes = int(next_state.get("strikes", 0))
    outs = int(next_state.get("outs_when_up", 0))

    outcome_clean = outcome.strip()

    if outcome_clean == "S":
        if strikes < 2:
            next_state["strikes"] = strikes + 1
            next_state["prev_pitch_category"] = pitch_category
            next_state["prev_pitch_type"] = pitch_type
            next_state["prev_pitch_zone"] = pitch_zone
            next_state["prev_pitch_result"] = "S"
            next_state["at_bat_event"] = "Strike"
        else:
            # 3rd strike -> Strikeout (K), reset count for next batter
            next_state["balls"] = 0
            next_state["strikes"] = 0
            next_state["prev_pitch_category"] = "First Pitch"
            next_state["prev_pitch_type"] = "None"
            next_state["prev_pitch_zone"] = 0
            next_state["prev_pitch_result"] = "First Pitch"
            next_state["outs_when_up"] = (outs + 1) % 3
            next_state["at_bat_event"] = "Strikeout (K)"

    elif outcome_clean == "Foul":
        # Foul ball increments strikes up to 2, never causes a strikeout on 2 strikes
        next_state["strikes"] = min(2, strikes + 1)
        next_state["prev_pitch_category"] = pitch_category
        next_state["prev_pitch_type"] = pitch_type
        next_state["prev_pitch_zone"] = pitch_zone
        next_state["prev_pitch_result"] = "S"
        next_state["at_bat_event"] = "Foul Ball"

    elif outcome_clean == "B":
        if balls < 3:
            next_state["balls"] = balls + 1
            next_state["prev_pitch_category"] = pitch_category
            next_state["prev_pitch_type"] = pitch_type
            next_state["prev_pitch_zone"] = pitch_zone
            next_state["prev_pitch_result"] = "B"
            next_state["at_bat_event"] = "Ball"
        else:
            # 4th ball -> Walk (BB), reset count for next batter
            next_state["balls"] = 0
            next_state["strikes"] = 0
            next_state["prev_pitch_category"] = "First Pitch"
            next_state["prev_pitch_type"] = "None"
            next_state["prev_pitch_zone"] = 0
            next_state["prev_pitch_result"] = "First Pitch"
            next_state["at_bat_event"] = "Walk (BB)"

    elif outcome_clean == "X":
        # Ball in play -> ends at-bat, resets count for next batter
        next_state["balls"] = 0
        next_state["strikes"] = 0
        next_state["prev_pitch_category"] = "First Pitch"
        next_state["prev_pitch_type"] = "None"
        next_state["prev_pitch_zone"] = 0
        next_state["prev_pitch_result"] = "First Pitch"
        next_state["at_bat_event"] = "In Play (Em Jogo)"

    else:
        raise ValueError(f"Unsupported pitch outcome '{outcome}'. Use 'S', 'Foul', 'B', or 'X'.")

    return next_state
